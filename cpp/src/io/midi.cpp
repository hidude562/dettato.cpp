#include "io/midi.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <tuple>

#include "common.hpp"

namespace dtt
{

namespace
{
    constexpr int TPQN = 480;
    constexpr double TICKS_PER_S = TPQN * 2.0; // 120 BPM

    void be32(std::vector<unsigned char>& o, uint32_t v)
    {
        for (int s = 24; s >= 0; s -= 8) {
            o.push_back(static_cast<unsigned char>(v >> s));
        }
    }

    void varlen(std::vector<unsigned char>& o, uint32_t v)
    {
        unsigned char buf[5];
        int n = 0;
        buf[n++] = v & 0x7f;
        while ((v >>= 7) != 0) {
            buf[n++] = static_cast<unsigned char>(0x80 | (v & 0x7f));
        }
        while (n > 0) {
            o.push_back(buf[--n]);
        }
    }

    void chunk(std::vector<unsigned char>& out, const char* id, const std::vector<unsigned char>& body)
    {
        out.insert(out.end(), id, id + 4);
        be32(out, static_cast<uint32_t>(body.size()));
        out.insert(out.end(), body.begin(), body.end());
    }

    void meta(std::vector<unsigned char>& o, unsigned char type, const std::string& data)
    {
        o.push_back(0x00);
        o.push_back(0xff);
        o.push_back(type);
        varlen(o, static_cast<uint32_t>(data.size()));
        for (char c : data) { // (vector::insert of a string trips a GCC 15 -Wstringop-overflow false positive)
            o.push_back(static_cast<unsigned char>(c));
        }
    }
} // namespace

std::vector<unsigned char> writeMidi(const std::vector<MidiTrack>& tracks)
{
    std::vector<unsigned char> out;
    std::vector<unsigned char> hdr;
    hdr.push_back(0x00);
    hdr.push_back(0x01); // format 1
    const uint16_t ntr = static_cast<uint16_t>(tracks.size() + 1);
    hdr.push_back(static_cast<unsigned char>(ntr >> 8));
    hdr.push_back(static_cast<unsigned char>(ntr & 0xff));
    hdr.push_back(TPQN >> 8);
    hdr.push_back(TPQN & 0xff);
    chunk(out, "MThd", hdr);

    std::vector<unsigned char> tempo;
    tempo.insert(tempo.end(), {0x00, 0xff, 0x51, 0x03, 0x07, 0xa1, 0x20}); // 500000 us per quarter
    tempo.insert(tempo.end(), {0x00, 0xff, 0x2f, 0x00});
    chunk(out, "MTrk", tempo);

    int next_ch = 0;
    for (const MidiTrack& t : tracks) {
        int ch = 9;
        if (!t.is_drum) {
            ch = next_ch;
            next_ch = (next_ch + 1) % 16;
            if (next_ch == 9) {
                next_ch = 10;
            }
        }
        std::vector<unsigned char> b;
        meta(b, 0x03, t.name);
        if (!t.is_drum) {
            b.insert(b.end(), {0x00, static_cast<unsigned char>(0xc0 | ch), static_cast<unsigned char>(std::clamp(t.program, 0, 127))});
        }
        // events: (tick, off-before-on at the same tick, pitch, velocity)
        struct Ev {
            int64_t tick;
            int on;
            int pitch, vel;
        };
        std::vector<Ev> ev;
        for (const MidiNote& n : t.notes) {
            const int64_t a = std::llround(n.onset * TICKS_PER_S);
            const int64_t e = std::max(a + 1, static_cast<int64_t>(std::llround(n.offset * TICKS_PER_S)));
            ev.push_back({a, 1, std::clamp(n.pitch, 0, 127), std::clamp(n.velocity, 1, 127)});
            ev.push_back({e, 0, std::clamp(n.pitch, 0, 127), 0});
        }
        std::stable_sort(ev.begin(), ev.end(), [](const Ev& x, const Ev& y) { return std::tie(x.tick, x.on) < std::tie(y.tick, y.on); });
        int64_t last = 0;
        for (const Ev& x : ev) {
            varlen(b, static_cast<uint32_t>(std::max<int64_t>(0, x.tick - last)));
            last = x.tick;
            b.push_back(static_cast<unsigned char>((x.on ? 0x90 : 0x80) | ch));
            b.push_back(static_cast<unsigned char>(x.pitch));
            b.push_back(static_cast<unsigned char>(x.on ? x.vel : 0x40));
        }
        b.insert(b.end(), {0x00, 0xff, 0x2f, 0x00});
        chunk(out, "MTrk", b);
    }
    return out;
}

void saveMidi(const std::string& path, const std::vector<MidiTrack>& tracks)
{
    const auto bytes = writeMidi(tracks);
    std::ofstream f(path, std::ios::binary);
    DTT_CHECK(f.good(), "cannot write %s", path.c_str());
    f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

} // namespace dtt
