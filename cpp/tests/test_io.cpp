// The MIDI writer (read back with a small parser) and the WAV reader. No weights needed.

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "common.hpp"
#include "io/midi.hpp"
#include "io/wav.hpp"

namespace
{
struct ParsedNote {
    int track, channel, pitch, velocity;
    long on_tick, off_tick;
};
struct ParsedTrack {
    std::string name;
    int program = -1;
};
struct ParsedMidi {
    int format = -1, ntracks = 0, division = 0;
    std::vector<ParsedTrack> tracks;
    std::vector<ParsedNote> notes;
};

ParsedMidi parse(const std::vector<unsigned char>& d)
{
    ParsedMidi m;
    auto be16 = [&](size_t o) { return (d.at(o) << 8) | d.at(o + 1); };
    auto be32 = [&](size_t o) { return (static_cast<uint32_t>(be16(o)) << 16) | static_cast<uint32_t>(be16(o + 2)); };
    REQUIRE(std::string(d.begin(), d.begin() + 4) == "MThd");
    REQUIRE(be32(4) == 6);
    m.format = be16(8);
    m.ntracks = be16(10);
    m.division = be16(12);
    size_t o = 14;
    for (int k = 0; k < m.ntracks; k++) {
        REQUIRE(std::string(d.begin() + static_cast<long>(o), d.begin() + static_cast<long>(o) + 4) == "MTrk");
        const size_t end = o + 8 + be32(o + 4);
        o += 8;
        ParsedTrack tr;
        std::map<std::pair<int, int>, std::pair<long, int>> open; // (channel, pitch) -> (tick, velocity)
        long tick = 0;
        int status = 0;
        while (o < end) {
            long delta = 0;
            unsigned char c;
            do {
                c = d.at(o++);
                delta = (delta << 7) | (c & 0x7f);
            } while (c & 0x80);
            tick += delta;
            if (d.at(o) & 0x80) {
                status = d.at(o++);
            }
            if (status == 0xff) {
                const int type = d.at(o++);
                long len = 0;
                do {
                    c = d.at(o++);
                    len = (len << 7) | (c & 0x7f);
                } while (c & 0x80);
                if (type == 0x03) {
                    tr.name.assign(d.begin() + static_cast<long>(o), d.begin() + static_cast<long>(o) + len);
                }
                o += static_cast<size_t>(len);
                continue;
            }
            const int kind = status & 0xf0, ch = status & 0x0f;
            if (kind == 0xc0 || kind == 0xd0) {
                if (kind == 0xc0) {
                    tr.program = d.at(o);
                }
                o += 1;
                continue;
            }
            const int a = d.at(o), b = d.at(o + 1);
            o += 2;
            if (kind == 0x90 && b > 0) {
                open[{ch, a}] = {tick, b};
            } else if (kind == 0x80 || (kind == 0x90 && b == 0)) {
                const auto it = open.find({ch, a});
                REQUIRE(it != open.end());
                m.notes.push_back({k, ch, a, it->second.second, it->second.first, tick});
                open.erase(it);
            }
        }
        CHECK(open.empty());
        m.tracks.push_back(tr);
        o = end;
    }
    CHECK(o == d.size());
    return m;
}

std::filesystem::path tempFile(const std::string& name)
{
    return std::filesystem::temp_directory_path() / ("dettato_test_" + name);
}

void put16(std::vector<unsigned char>& d, uint32_t v)
{
    d.push_back(static_cast<unsigned char>(v & 0xff));
    d.push_back(static_cast<unsigned char>((v >> 8) & 0xff));
}
void put32(std::vector<unsigned char>& d, uint32_t v)
{
    put16(d, v & 0xffff);
    put16(d, v >> 16);
}

/** A WAV file with `fmt` (1 = PCM, 3 = float), the given sample bytes. */
std::filesystem::path writeWav(const std::string& name, int fmt, int ch, int rate, int bits, const std::vector<unsigned char>& data)
{
    std::vector<unsigned char> d = {'R', 'I', 'F', 'F'};
    put32(d, static_cast<uint32_t>(36 + data.size()));
    for (char c : std::string("WAVEfmt ")) {
        d.push_back(static_cast<unsigned char>(c));
    }
    put32(d, 16);
    put16(d, static_cast<uint32_t>(fmt));
    put16(d, static_cast<uint32_t>(ch));
    put32(d, static_cast<uint32_t>(rate));
    put32(d, static_cast<uint32_t>(rate * ch * bits / 8));
    put16(d, static_cast<uint32_t>(ch * bits / 8));
    put16(d, static_cast<uint32_t>(bits));
    for (char c : std::string("data")) {
        d.push_back(static_cast<unsigned char>(c));
    }
    put32(d, static_cast<uint32_t>(data.size()));
    d.insert(d.end(), data.begin(), data.end());
    const auto p = tempFile(name);
    std::ofstream(p, std::ios::binary).write(reinterpret_cast<const char*>(d.data()), static_cast<std::streamsize>(d.size()));
    return p;
}
} // namespace

TEST_CASE("writeMidi: a type-1 file at 480 ticks per quarter, 120 BPM", "[pure][midi]")
{
    std::vector<dtt::MidiTrack> tracks(2);
    tracks[0].name = "piano";
    tracks[0].program = 0;
    tracks[0].notes = {{0.5, 1.0, 60, 100}, {1.0, 2.25, 64, 80}};
    tracks[1].name = "drums";
    tracks[1].is_drum = true;
    tracks[1].notes = {{0.0, 0.1, 36, 120}};
    const ParsedMidi m = parse(dtt::writeMidi(tracks));

    CHECK(m.format == 1);
    CHECK(m.division == 480);
    REQUIRE(m.ntracks == 3); // tempo track + 2
    CHECK(m.tracks[1].name == "piano");
    CHECK(m.tracks[1].program == 0);
    CHECK(m.tracks[2].name == "drums");
    REQUIRE(m.notes.size() == 3);
    for (const ParsedNote& n : m.notes) {
        if (n.track == 1) {
            CHECK(n.channel == 0);
            if (n.pitch == 60) {
                CHECK(n.on_tick == 480); // 0.5 s at 960 ticks per second
                CHECK(n.off_tick == 960);
                CHECK(n.velocity == 100);
            } else {
                CHECK(n.pitch == 64);
                CHECK(n.on_tick == 960);
                CHECK(n.off_tick == 2160);
            }
        } else {
            CHECK(n.track == 2);
            CHECK(n.channel == 9);
            CHECK(n.pitch == 36);
        }
    }
}

TEST_CASE("readWav: 16-bit stereo PCM is downmixed to mono", "[pure][wav]")
{
    std::vector<unsigned char> data;
    for (int i = 0; i < 100; i++) {
        put16(data, static_cast<uint16_t>(16384));          // L = 0.5
        put16(data, static_cast<uint16_t>(-8192 & 0xffff)); // R = -0.25
    }
    const auto p = writeWav("s16.wav", 1, 2, 22050, 16, data);
    const dtt::Audio a = dtt::readWav(p.string());
    std::filesystem::remove(p);
    CHECK(a.rate == 22050.0);
    REQUIRE(a.samples.size() == 100);
    for (float v : a.samples) {
        CHECK(v == 0.125f);
    }
}

TEST_CASE("readWav: 32-bit float and 24-bit PCM", "[pure][wav]")
{
    std::vector<unsigned char> f32;
    const float vals[3] = {0.25f, -1.0f, 0.75f};
    for (float v : vals) {
        uint32_t u;
        std::memcpy(&u, &v, 4);
        put32(f32, u);
    }
    const auto p = writeWav("f32.wav", 3, 1, 48000, 32, f32);
    const dtt::Audio a = dtt::readWav(p.string());
    std::filesystem::remove(p);
    CHECK(a.rate == 48000.0);
    REQUIRE(a.samples.size() == 3);
    CHECK(a.samples[0] == 0.25f);
    CHECK(a.samples[1] == -1.0f);
    CHECK(a.samples[2] == 0.75f);

    std::vector<unsigned char> s24 = {0x00, 0x00, 0x40, 0x00, 0x00, 0xc0}; // 0.5, -0.5
    const auto p24 = writeWav("s24.wav", 1, 1, 44100, 24, s24);
    const dtt::Audio b = dtt::readWav(p24.string());
    std::filesystem::remove(p24);
    REQUIRE(b.samples.size() == 2);
    CHECK(b.samples[0] == 0.5f);
    CHECK(b.samples[1] == -0.5f);
}

TEST_CASE("readWav rejects a file that is not a WAV", "[pure][wav]")
{
    const auto p = tempFile("not.wav");
    std::ofstream(p, std::ios::binary) << "ID3 this is an mp3, honestly";
    CHECK_THROWS_AS(dtt::readWav(p.string()), dtt::Failure);
    std::filesystem::remove(p);
}
