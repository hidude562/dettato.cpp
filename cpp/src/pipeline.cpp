#include "dettato/dettato.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <future>
#include <mutex>
#include <string>
#include <tuple>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include "backend.hpp"
#include "common.hpp"
#include "drums/drums.hpp"
#include "dsp/resample.hpp"
#include "evidence/evidence.hpp"
#include "grouper/grouper.hpp"
#include "model_file.hpp"
#include "transcriber/transcriber.hpp"

namespace dtt
{

const char* describe(Error e)
{
    switch (e) {
        case Error::FileNotFound:
            return "model file not found";
        case Error::InvalidCheckpoint:
            return "not a dettato checkpoint";
        case Error::UnsupportedCheckpointVersion:
            return "checkpoint is for another version of dettato.cpp";
        case Error::DeviceUnavailable:
            return "the compute device cannot be used";
        case Error::Cancelled:
            return "cancelled";
        case Error::InvalidArgument:
            return "invalid argument";
        case Error::Internal:
            return "internal error";
    }
    return "unknown error";
}

namespace
{
    /** "Metal", "Vulkan", ... (ggml names its Metal backend "MTL"). */
    std::string backendName(ggml_backend_reg_t reg)
    {
        if (reg == nullptr) {
            return "GPU";
        }
        const std::string n = ggml_backend_reg_name(reg);
        return n == "MTL" ? "Metal" : n;
    }

    /**
     * Hands freed heap memory back to the system. glibc keeps what short-lived worker threads freed in per-thread arenas,
     * so without this the process grows with every call (measured: 0.8 -> 2.0 GB over 16 calls; flat with it). Other
     * allocators need nothing.
     */
    struct HeapTrim {
        ~HeapTrim()
        {
#if defined(__GLIBC__)
            malloc_trim(0);
#endif
        }
    };

    constexpr double MIN_END_S = 0.03;        // every pitched note lasts at least 30 ms before grouping
    constexpr double DRUM_MIN_RATE = 40000.0; // the drum model reads the 7.8-20 kHz band
    // typical share of the run time, for the overall progress
    constexpr float W_TRANSCRIBE = 0.74f, W_REPAIR = 0.02f, W_GROUP = 0.22f, W_DRUMS = 0.02f;
} // namespace

namespace
{
    /** ggml device indices of the GPUs, in availableDevices() order. */
    std::vector<size_t> gpuIndices()
    {
        std::vector<size_t> idx;
        for (size_t i = 0; i < Backend::deviceCount(); i++) {
            const auto t = ggml_backend_dev_type(ggml_backend_dev_get(i));
            if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                idx.push_back(i);
            }
        }
        return idx;
    }
} // namespace

std::vector<Device> availableDevices()
{
    std::vector<Device> gpus, out;
    for (size_t i : gpuIndices()) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const auto t = ggml_backend_dev_type(d);
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(d, &free, &total);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(d);
        gpus.push_back({ggml_backend_dev_description(d), backendName(reg), t == GGML_BACKEND_DEVICE_TYPE_IGPU, total});
    }
    out = gpus;
    out.push_back({"CPU", "CPU", false, 0});
    return out;
}

namespace
{
    void appendTracks(Transcription& out, const std::vector<GroupTrack>& tracks)
    {
        for (const GroupTrack& t : tracks) {
            const int ti = static_cast<int>(out.tracks.size());
            std::string name = t.instrument;
            std::replace(name.begin(), name.end(), '_', ' ');
            out.tracks.push_back({name, t.program, false});
            for (const auto& n : t.notes) {
                out.notes.push_back({n[0], n[1], static_cast<int>(n[2]), static_cast<int>(n[3]), ti});
            }
        }
    }
} // namespace

struct Pipeline::Impl {
    std::unique_ptr<Backend> be;
    std::unique_ptr<Backend> be_drums; // on a GPU the drum model runs alongside the transcriber, on its own backend instance
    std::unique_ptr<ModelFile> mf;
    std::unique_ptr<Transcriber> tr;
    std::unique_ptr<EvidenceRepairer> rep;
    std::unique_ptr<Grouper> gr;
    std::unique_ptr<DrumTranscriber> dr;
    Device device;
    std::string error;

    /**
     * The transcriber. transcribe() drops it after its last use and the next call builds it again: a network's compute
     * buffer is host memory on the CPU (~0.5 GB here), which the stages after it would otherwise carry.
     */
    Transcriber& transcriber()
    {
        if (!tr) {
            // windows per network call: every size gives the same notes. The host buffers grow with it (the compute
            // buffer on the CPU, the taps read back from a GPU); 2 is as fast as more on a GPU, 1 is the fastest on the CPU
            tr = std::make_unique<Transcriber>(*mf, *be, be->isCpu() ? 1 : 2, LaneNet::Taps{true, true, false});
        }
        return *tr;
    }
};

Pipeline::Pipeline(std::unique_ptr<Impl> impl) : mImpl(std::move(impl)) {}
Pipeline::Pipeline(Pipeline&&) noexcept = default;
Pipeline& Pipeline::operator=(Pipeline&&) noexcept = default;
Pipeline::~Pipeline() = default;

const Device& Pipeline::device() const
{
    return mImpl->device;
}

const std::string& Pipeline::lastErrorMessage() const
{
    return mImpl->error;
}

std::expected<Pipeline, Error> Pipeline::load(const std::filesystem::path& path, LoadOptions options)
{
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) {
        return std::unexpected(Error::FileNotFound);
    }
    const std::u8string p8 = path.u8string(); // ggml opens files by UTF-8 name (wide on Windows)
    const std::string path_utf8(p8.begin(), p8.end());
    const auto devices = availableDevices();
    std::string dev = "auto";
    if (options.device) {
        if (*options.device >= devices.size()) {
            if (options.error_detail) {
                *options.error_detail = format("device %zu does not exist (%zu devices)", *options.device, devices.size());
            }
            return std::unexpected(Error::DeviceUnavailable);
        }
        dev = *options.device + 1 == devices.size() ? "cpu" : std::to_string(gpuIndices()[*options.device]);
    }

    // everything on one device; the error and its reason when it fails
    std::string why;
    auto attempt = [&](const std::string& device) -> std::expected<std::unique_ptr<Impl>, Error> {
        auto impl = std::make_unique<Impl>();
        try {
            impl->be = std::make_unique<Backend>(device, options.n_threads);
            if (options.on_progress) {
                options.on_progress(0.0f);
            }
            impl->mf = std::make_unique<ModelFile>(path_utf8, impl->be->get());
            if (options.should_cancel && options.should_cancel()) {
                return std::unexpected(Error::Cancelled);
            }
            impl->transcriber();
            impl->rep = std::make_unique<EvidenceRepairer>(*impl->mf, *impl->be);
            impl->gr = std::make_unique<Grouper>(*impl->mf, *impl->be);
            // A ggml backend runs one graph at a time, so the drums (started in parallel on a GPU) get their own instance
            // on the same device; the weights live in device buffers both can read. On the CPU they stay in sequence:
            // ggml's CPU threads slow down sharply when two thread pools share the cores.
            if (!impl->be->isCpu()) {
                impl->be_drums = std::make_unique<Backend>(device, options.n_threads);
            }
            impl->dr = std::make_unique<DrumTranscriber>(*impl->mf, impl->be_drums ? *impl->be_drums : *impl->be);
            if (options.on_progress) {
                options.on_progress(1.0f);
            }
        } catch (const Failure& e) {
            why = e.what();
            if (why.find("cannot run") != std::string::npos || why.find("initialise") != std::string::npos) {
                return std::unexpected(Error::DeviceUnavailable);
            }
            if (why.find("version") != std::string::npos) {
                return std::unexpected(Error::UnsupportedCheckpointVersion);
            }
            return std::unexpected(Error::InvalidCheckpoint);
        }
        return impl;
    };
    auto r = attempt(dev);
    // Auto: a GPU that cannot run the networks (or cannot start) leaves the CPU, which always can
    if (!r && r.error() == Error::DeviceUnavailable && dev == "auto") {
        const std::string gpu_why = why;
        r = attempt("cpu");
        if (r) {
            why = "the GPU could not be used, so the CPU runs the networks: " + gpu_why;
        }
    }
    if (options.error_detail) {
        *options.error_detail = why;
    }
    if (!r) {
        return std::unexpected(r.error());
    }
    std::unique_ptr<Impl> impl = std::move(*r);
    impl->device = Device{"CPU", "CPU", false, 0};
    if (!impl->be->isCpu()) {
        ggml_backend_dev_t d = ggml_backend_get_device(impl->be->get());
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(d, &free, &total);
        impl->device = Device{ggml_backend_dev_description(d), backendName(ggml_backend_dev_backend_reg(d)),
                              ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_IGPU, total};
    }
    return Pipeline(std::move(impl));
}

std::expected<Transcription, Error> Pipeline::transcribe(std::span<const float> mono, double sample_rate, const TranscribeOptions& o)
{
    Impl& I = *mImpl;
    I.error.clear();
    const HeapTrim trim_on_return;
    // A NaN or an infinity (a damaged file) would spread through every transform and leave no notes at all: read
    // such samples as silence.
    std::vector<float> finite;
    if (std::any_of(mono.begin(), mono.end(), [](float v) { return !std::isfinite(v); })) {
        finite.assign(mono.begin(), mono.end());
        std::replace_if(finite.begin(), finite.end(), [](float v) { return !std::isfinite(v); }, 0.0f);
        mono = finite;
    }
    // should_cancel is polled between the stages, before every network call (on both backend instances: on a GPU the
    // drums run on a second thread) and, on the CPU, at every graph node. The calls are serialised and the answer
    // latches. Cleared from the backends on return, whatever it captured.
    std::mutex cancel_mutex;
    std::atomic<bool> cancelled{false};
    const std::function<bool()> poll_cancel = [&]() -> bool {
        if (!cancelled.load() && o.should_cancel) {
            const std::lock_guard<std::mutex> lock(cancel_mutex);
            if (!cancelled.load() && o.should_cancel()) {
                cancelled.store(true);
            }
        }
        return cancelled.load();
    };
    struct CancelScope {
        Impl& impl;
        ~CancelScope()
        {
            impl.be->setCancel({});
            if (impl.be_drums) {
                impl.be_drums->setCancel({});
            }
        }
    } cancel_scope{I};
    I.be->setCancel(poll_cancel);
    if (I.be_drums) {
        I.be_drums->setCancel(poll_cancel);
    }
    if (sample_rate < 16000.0 || !std::isfinite(sample_rate)) {
        I.error = "sample rate must be at least 16 kHz";
        return std::unexpected(Error::InvalidArgument);
    }
    auto report = [&](Stage s, float f) {
        float base = 0, w = 0;
        switch (s) {
            case Stage::Transcribing: base = 0; w = W_TRANSCRIBE; break;
            case Stage::Repairing: base = W_TRANSCRIBE; w = W_REPAIR; break;
            case Stage::Grouping: base = W_TRANSCRIBE + W_REPAIR; w = W_GROUP; break;
            case Stage::Drums: base = W_TRANSCRIBE + W_REPAIR + W_GROUP; w = W_DRUMS; break;
        }
        if (o.on_progress) {
            o.on_progress({s, f, base + w * f});
        }
        return !poll_cancel();
    };
    // DETTATO_TIMING=1: per-stage wall times on stderr
    const bool timing = std::getenv("DETTATO_TIMING") != nullptr;
    auto t_last = std::chrono::steady_clock::now();
    auto lap = [&](const char* what) {
        if (timing) {
            const auto now = std::chrono::steady_clock::now();
            std::fprintf(stderr, "  [dettato] %-28s %7.2f s\n", what, std::chrono::duration<double>(now - t_last).count());
            t_last = now;
        }
    };
    try {
        Transcription out;
        out.seconds = mono.size() / sample_rate;
        const std::vector<float> x16 = resample(mono.data(), mono.size(), sample_rate, 16000.0, I.be->threads());
        lap("resample to 16 kHz");

        // the drums only need the audio: on a GPU they start now, alongside the transcriber
        struct DrumResult {
            std::vector<DrumHit> hits;
            DrumInfo info;
            double seconds = 0;
        };
        const bool want_drums = o.drums && sample_rate >= DRUM_MIN_RATE;
        auto run_drums = [&I, mono, sample_rate] {
            const auto t0 = std::chrono::steady_clock::now();
            const Backend& be = I.be_drums ? *I.be_drums : *I.be; // its cancel check: a cancel waits for this thread
            DrumResult r;
            be.throwIfCancelled();
            const std::vector<float> x44 = resample(mono.data(), mono.size(), sample_rate, 44100.0, I.be->threads());
            be.throwIfCancelled();
            r.hits = I.dr->transcribe(x44.data(), x44.size(), &r.info);
            r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            return r;
        };
        std::future<DrumResult> drums_async;
        // the drums start once the transcriber pass is done (alongside the repairer and the grouper, so they do not slow
        // the transcriber down); on the CPU they run at the end
        auto start_drums = [&] {
            if (want_drums && I.be_drums && !drums_async.valid()) {
                drums_async = std::async(std::launch::async, run_drums);
            }
        };

        // 1. the tuning check: a recording clearly off A440 is resampled onto it (pitch and time together); every network
        //    then reads the retuned audio y, and the output times are mapped back by r
        if (!report(Stage::Transcribing, 0.0f)) {
            throw Cancelled();
        }
        const Grouper::Retune rt = I.gr->retune(x16.data(), x16.size());
        const std::vector<float>& y = rt.y;
        lap("tuning check (+ retune)");

        // 2. ONE transcriber pass on y, 8 s windows every 4 s. Two sinks read it as it streams: the evidence repairer's,
        //    and the grouper's store, which keeps what the grouper reads from it; maps2 is the same average the grouper's
        //    own pass would give.
        auto ev_sink = I.rep->makeSink();
        auto gr_sink = I.gr->makeStoreSink();
        FanOutSink fan({ev_sink.get(), gr_sink.get()});
        Maps maps2;
        Transcriber::RunOptions ro;
        ro.hop_frac = 0.5;
        ro.sink = &fan;
        ro.tap_mask = (1u << TapSink::HIDDEN) | (1u << TapSink::STEM);
        ro.sub_maps = &maps2;
        ro.sub_step = 64000;
        ro.progress = [&](float f) { return report(Stage::Transcribing, f); };
        Transcriber& tr = I.transcriber();
        const Maps maps = tr.run(y.data(), y.size(), ro);
        lap("transcriber (one pass)");
        start_drums();

        // 3. the evidence repairer (notes on y's time axis)
        const auto tnotes = tr.notes(maps);
        if (!report(Stage::Repairing, 0.0f)) {
            throw Cancelled();
        }
        RepairInfo ri;
        const auto repaired = I.rep->repair(maps, tnotes, *ev_sink, &ri);
        ev_sink.reset();
        lap("decode + evidence repairer");
        out.notes_transcribed = ri.transcribed;
        out.notes_dropped = ri.dropped;
        out.notes_inserted = ri.inserted;

        // 4. the grouper, from the store (it recomputes the few windows the store lacks for some note)
        if (!report(Stage::Grouping, 0.0f)) {
            throw Cancelled();
        }
        std::vector<std::array<double, 3>> notes3;
        notes3.reserve(repaired.size());
        for (const auto& r : repaired) {
            notes3.push_back({r.onset, std::max(r.offset, r.onset + MIN_END_S), static_cast<double>(r.pitch)});
        }
        std::vector<GroupTrack> tracks;
        if (!notes3.empty()) {
            // the grouper's two halves take about the same time: its features (recomputing the windows the store lacks)
            // and its network + decode
            const GroupFeatures f = I.gr->featuresFromStore(y.data(), y.size(), notes3, *gr_sink, maps2, tr,
                                                            [&](float g) { return report(Stage::Grouping, 0.5f * g); });
            gr_sink.reset();
            I.tr.reset(); // its last use
            HeapTrim{};   // and the buffers of the pass
            lap("grouper features (store)");
            const GroupPredictions P = I.gr->predict(y.data(), y.size(), f, [&](float g) { return report(Stage::Grouping, 0.5f + 0.5f * g); });
            tracks = I.gr->tracks(f, P, rt.r);
            I.gr->release();
            lap("grouper network + decode");
        }
        I.tr.reset();
        out.tuning_cents = rt.delta;
        appendTracks(out, tracks);

        // 5. drums (already running on a GPU)
        if (want_drums) {
            if (!report(Stage::Drums, 0.0f)) {
                throw Cancelled();
            }
            const DrumResult dres = drums_async.valid() ? drums_async.get() : run_drums();
            if (timing) {
                std::fprintf(stderr, "  [dettato] (drums ran %.2f s%s)\n", dres.seconds, I.be_drums ? ", in parallel" : "");
            }
            lap(I.be_drums ? "drums (waiting for)" : "drums");
            const auto& hits = dres.hits;
            out.drum_model = dres.info.model;
            if (!hits.empty()) {
                const int ti = static_cast<int>(out.tracks.size());
                out.tracks.push_back({"drums", 0, true});
                for (const auto& h : hits) {
                    const int vel = std::clamp(static_cast<int>(std::lround(h.velocity * 127.0)), 1, 127);
                    out.notes.push_back({h.onset, h.offset, h.pitch, vel, ti});
                }
            }
        }
        std::stable_sort(out.notes.begin(), out.notes.end(), [](const Note& a, const Note& b) {
            return std::tie(a.onset, a.track, a.pitch) < std::tie(b.onset, b.track, b.pitch);
        });
        report(Stage::Drums, 1.0f);
        return out;
    } catch (const Cancelled&) {
        return std::unexpected(Error::Cancelled);
    } catch (const std::exception& e) {
        I.error = e.what();
        return std::unexpected(Error::Internal);
    }
}

} // namespace dtt
