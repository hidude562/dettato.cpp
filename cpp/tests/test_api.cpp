// The public API. The [model] tests need a dettato GGUF: set DETTATO_TEST_MODEL=/path/to/dettato-f16.gguf (they run on
// the CPU, about a minute in all); without it they are skipped.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <random>
#include <thread>
#include <vector>

#include "dettato/dettato.hpp"

namespace
{
const char* testModel()
{
    const char* p = std::getenv("DETTATO_TEST_MODEL");
    return p != nullptr && *p != '\0' ? p : nullptr;
}

dtt::Pipeline& cpuPipeline()
{
    static dtt::Pipeline p = [] {
        dtt::LoadOptions o;
        o.device = dtt::availableDevices().size() - 1; // the CPU
        o.n_threads = static_cast<int>(std::clamp(std::thread::hardware_concurrency(), 1u, 8u));
        auto r = dtt::Pipeline::load(testModel(), o);
        REQUIRE(r.has_value());
        return std::move(*r);
    }();
    return p;
}

/** Plucked harmonic tones (C4, E4, G4, then C5) with a few clicks, at 44.1 kHz. */
std::vector<float> testSignal(double rate)
{
    struct Tone {
        int pitch;
        double t0, t1;
    };
    const Tone tones[] = {{60, 0.5, 1.9}, {64, 2.0, 3.4}, {67, 3.5, 4.9}, {72, 5.0, 6.5}};
    std::vector<float> x(static_cast<size_t>(8.0 * rate), 0.0f);
    for (const Tone& t : tones) {
        const double f0 = 440.0 * std::pow(2.0, (t.pitch - 69) / 12.0);
        for (size_t i = static_cast<size_t>(t.t0 * rate); i < static_cast<size_t>(t.t1 * rate); i++) {
            const double s = static_cast<double>(i) / rate - t.t0;
            double v = 0;
            for (int h = 1; h <= 8; h++) {
                v += std::sin(2.0 * std::numbers::pi * f0 * h * s) / h * std::exp(-s * (1.5 + 0.6 * h));
            }
            x[i] += static_cast<float>(0.25 * v * std::min(1.0, s / 0.003));
        }
    }
    return x;
}

bool hasNote(const dtt::Transcription& r, int pitch, double onset)
{
    return std::any_of(r.notes.begin(), r.notes.end(), [&](const dtt::Note& n) {
        return !r.tracks[n.track].is_drum && n.pitch == pitch && std::abs(n.onset - onset) < 0.1;
    });
}
} // namespace

TEST_CASE("describe() names every error", "[pure][api]")
{
    using E = dtt::Error;
    for (E e : {E::FileNotFound, E::InvalidCheckpoint, E::UnsupportedCheckpointVersion, E::DeviceUnavailable, E::Cancelled,
                E::InvalidArgument, E::Internal}) {
        CHECK(std::string(dtt::describe(e)).size() > 3);
    }
}

TEST_CASE("the CPU is always the last device", "[pure][api]")
{
    const auto d = dtt::availableDevices();
    REQUIRE(!d.empty());
    CHECK(d.back().backend == "CPU");
}

TEST_CASE("load() reports a missing file and a file that is not a checkpoint", "[pure][api]")
{
    CHECK(dtt::Pipeline::load("this file does not exist.gguf").error() == dtt::Error::FileNotFound);

    const auto p = std::filesystem::temp_directory_path() / "dettato_test_not_a_model.gguf";
    std::ofstream(p, std::ios::binary) << "definitely not a GGUF file";
    dtt::LoadOptions o;
    o.device = dtt::availableDevices().size(); // one past the CPU
    CHECK(dtt::Pipeline::load(p, o).error() == dtt::Error::DeviceUnavailable);
    o.device = dtt::availableDevices().size() - 1;
    CHECK(dtt::Pipeline::load(p, o).error() == dtt::Error::InvalidCheckpoint);
    std::filesystem::remove(p);
}

TEST_CASE("a tone sequence comes out as its notes", "[model]")
{
    if (testModel() == nullptr) {
        SKIP("DETTATO_TEST_MODEL is not set");
    }
    dtt::Pipeline& p = cpuPipeline();
    CHECK(p.device().backend == "CPU");
    const auto x = testSignal(44100.0);

    std::vector<float> overall;
    dtt::TranscribeOptions o;
    o.on_progress = [&](const dtt::Progress& pr) { overall.push_back(pr.overall); };
    const auto r = p.transcribe(x, 44100.0, o);
    REQUIRE(r.has_value());
    CHECK(r->seconds == 8.0);
    CHECK(hasNote(*r, 60, 0.5));
    CHECK(hasNote(*r, 64, 2.0));
    CHECK(hasNote(*r, 67, 3.5));
    CHECK(hasNote(*r, 72, 5.0));
    CHECK(std::abs(r->tuning_cents) < 5.0);
    for (const dtt::Note& n : r->notes) {
        REQUIRE(n.track >= 0);
        REQUIRE(static_cast<size_t>(n.track) < r->tracks.size());
        CHECK(n.offset > n.onset);
        CHECK(n.velocity >= 1);
        CHECK(n.velocity <= 127);
    }
    CHECK(std::is_sorted(r->notes.begin(), r->notes.end(), [](const dtt::Note& a, const dtt::Note& b) { return a.onset < b.onset; }));
    REQUIRE(!overall.empty());
    CHECK(std::is_sorted(overall.begin(), overall.end()));
    CHECK(overall.back() <= 1.0f);

    // Fast mode at 16 kHz (no drums at that rate): the same notes
    dtt::TranscribeOptions fast;
    fast.quality = dtt::Quality::Fast;
    std::vector<float> x16 = testSignal(16000.0);
    const auto r16 = p.transcribe(x16, 16000.0, fast);
    REQUIRE(r16.has_value());
    CHECK(r16->drum_model.empty());
    CHECK(hasNote(*r16, 60, 0.5));
    CHECK(hasNote(*r16, 72, 5.0));
}

TEST_CASE("transcribe() checks its arguments and can be cancelled", "[model]")
{
    if (testModel() == nullptr) {
        SKIP("DETTATO_TEST_MODEL is not set");
    }
    dtt::Pipeline& p = cpuPipeline();
    const std::vector<float> x(8000 * 4, 0.0f);
    CHECK(p.transcribe(x, 8000.0).error() == dtt::Error::InvalidArgument);
    CHECK(!p.lastErrorMessage().empty());

    dtt::TranscribeOptions o;
    o.should_cancel = [] { return true; };
    CHECK(p.transcribe(testSignal(44100.0), 44100.0, o).error() == dtt::Error::Cancelled);

    const auto silent = p.transcribe(std::vector<float>(16000 * 3, 0.0f), 16000.0); // and it still works after a cancel
    REQUIRE(silent.has_value());
    CHECK(silent->notes.empty());
}
