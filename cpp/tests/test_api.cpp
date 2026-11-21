// The public API. The [model] tests need a dettato GGUF: set DETTATO_TEST_MODEL=/path/to/dettato-f16.gguf (they run on
// the CPU, about a minute in all); without it they are skipped.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <optional>
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

/** Plucked harmonic tones (C4, E4, G4, then C5) and a few noise bursts: 8 s, repeated reps times. */
std::vector<float> testSignal(double rate, int reps = 1)
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
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (double t0 = 6.75; t0 < 7.9; t0 += 0.25) { // short noise bursts: something for the drum model
        const size_t a = static_cast<size_t>(t0 * rate), n = static_cast<size_t>(0.08 * rate);
        for (size_t i = 0; i < n && a + i < x.size(); i++) {
            x[a + i] += 0.4f * u(rng) * static_cast<float>(std::exp(-static_cast<double>(i) / (0.012 * rate)));
        }
    }
    std::vector<float> y;
    y.reserve(x.size() * static_cast<size_t>(reps));
    for (int r = 0; r < reps; r++) {
        y.insert(y.end(), x.begin(), x.end());
    }
    return y;
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

    // at 16 kHz (no drums at that rate): the same notes
    std::vector<float> x16 = testSignal(16000.0);
    const auto r16 = p.transcribe(x16, 16000.0);
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

    // cancelled in the middle of the transcriber: on the CPU the network itself stops, not only the gap between windows
    std::optional<std::chrono::steady_clock::time_point> asked;
    o.on_progress = [&](const dtt::Progress& pr) {
        if (!asked && pr.stage == dtt::Stage::Transcribing && pr.stage_fraction > 0.0f) {
            asked = std::chrono::steady_clock::now();
        }
    };
    o.should_cancel = [&] { return asked.has_value(); };
    const auto x40 = testSignal(16000.0, 5);
    const auto mid = p.transcribe(x40, 16000.0, o);
    REQUIRE(asked.has_value());
    const double waited = std::chrono::duration<double>(std::chrono::steady_clock::now() - *asked).count();
    std::printf("cancel: returned %.2f s after should_cancel turned true\n", waited);
    CHECK(mid.error() == dtt::Error::Cancelled);
    CHECK(waited < 10.0);

    const auto silent = p.transcribe(std::vector<float>(16000 * 3, 0.0f), 16000.0); // and it still works after a cancel
    REQUIRE(silent.has_value());
    CHECK(silent->notes.empty());
}

namespace
{
std::vector<const dtt::Note*> pitched(const dtt::Transcription& r)
{
    std::vector<const dtt::Note*> v;
    for (const dtt::Note& n : r.notes) {
        if (!r.tracks[n.track].is_drum) {
            v.push_back(&n);
        }
    }
    return v;
}

/** Notes of a with a note of the same pitch in b starting within 20 ms. */
size_t matched(const dtt::Transcription& a, const dtt::Transcription& b)
{
    size_t m = 0;
    const auto pb = pitched(b);
    for (const dtt::Note* n : pitched(a)) {
        m += std::any_of(pb.begin(), pb.end(), [&](const dtt::Note* q) { return q->pitch == n->pitch && std::abs(q->onset - n->onset) < 0.02; });
    }
    return m;
}

size_t drumHits(const dtt::Transcription& r)
{
    return r.notes.size() - pitched(r).size();
}
} // namespace

TEST_CASE("the first GPU gives the CPU's notes", "[model][gpu]")
{
    if (testModel() == nullptr) {
        SKIP("DETTATO_TEST_MODEL is not set");
    }
    const auto devs = dtt::availableDevices();
    if (devs.size() < 2) {
        SKIP("no GPU device");
    }
    std::printf("GPU: %s (%s), %.1f GB\n", devs[0].name.c_str(), devs[0].backend.c_str(), devs[0].memory_total / 1e9);
    dtt::LoadOptions o;
    o.device = 0;
    std::string why;
    o.error_detail = &why;
    auto g = dtt::Pipeline::load(testModel(), o);
    INFO("load on " << devs[0].name << ": " << why);
    if (!g && devs[0].name.find("Paravirtual") != std::string::npos) {
        // GitHub's macOS runners: Apple's paravirtual GPU has no simdgroup operations, and ggml-metal runs no matmul
        // without them, so nothing can be tested on it
        SKIP("Apple paravirtual GPU (ggml-metal runs no matmul on it): " + why);
    }
    REQUIRE(g.has_value());

    const auto x = testSignal(44100.0, 5); // 40 s: the transcriber and the grouper run batches of several windows
    const auto t0 = std::chrono::steady_clock::now();
    const auto rg = g->transcribe(x, 44100.0);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    INFO(g->lastErrorMessage());
    REQUIRE(rg.has_value());
    const auto rc = cpuPipeline().transcribe(x, 44100.0);
    REQUIRE(rc.has_value());

    const size_t nc = pitched(*rc).size(), ng = pitched(*rg).size();
    std::printf("GPU: 40 s in %.2f s; pitched notes CPU %zu, GPU %zu, matched %zu / %zu; drum hits CPU %zu (%s), GPU %zu (%s); "
                "tracks CPU %zu, GPU %zu\n",
                secs, nc, ng, matched(*rc, *rg), matched(*rg, *rc), drumHits(*rc), rc->drum_model.c_str(), drumHits(*rg),
                rg->drum_model.c_str(), rc->tracks.size(), rg->tracks.size());
    CHECK(hasNote(*rg, 60, 0.5));
    CHECK(hasNote(*rg, 64, 2.0));
    CHECK(hasNote(*rg, 67, 3.5));
    CHECK(hasNote(*rg, 72, 5.0));
    CHECK(matched(*rc, *rg) >= nc * 95 / 100);
    CHECK(matched(*rg, *rc) >= ng * 95 / 100);
    CHECK(rg->drum_model == rc->drum_model);
    const size_t dc = drumHits(*rc), dg = drumHits(*rg);
    CHECK((dc > dg ? dc - dg : dg - dc) <= std::max<size_t>(1, dc / 20));
    CHECK(std::abs(rg->tuning_cents - rc->tuning_cents) < 1.0);
}

TEST_CASE("Auto loads on a GPU that runs everything, else on the CPU", "[model]")
{
    if (testModel() == nullptr) {
        SKIP("DETTATO_TEST_MODEL is not set");
    }
    dtt::LoadOptions o;
    o.n_threads = static_cast<int>(std::clamp(std::thread::hardware_concurrency(), 1u, 8u));
    std::string why;
    o.error_detail = &why;
    auto p = dtt::Pipeline::load(testModel(), o);
    INFO(why);
    REQUIRE(p.has_value());
    std::printf("Auto: %s (%s)%s%s\n", p->device().name.c_str(), p->device().backend.c_str(), why.empty() ? "" : "; ", why.c_str());
    const auto devs = dtt::availableDevices();
    if (p->device().backend == "CPU" && devs.size() > 1) {
        CHECK(!why.empty()); // it says why the GPU was passed over
    }
    for (const dtt::Device& d : devs) {
        CHECK(d.backend != "MTL"); // reported as "Metal"
    }
    const auto r = p->transcribe(testSignal(16000.0), 16000.0);
    REQUIRE(r.has_value());
    CHECK(hasNote(*r, 60, 0.5));
}
