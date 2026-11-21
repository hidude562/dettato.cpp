#pragma once

// dettato.cpp: multi-instrument audio-to-MIDI transcription (KobiMusic's dettato) on ggml.
//
//     auto p = dtt::Pipeline::load("dettato-f16.gguf");
//     if (!p) return report(dtt::describe(p.error()));
//     auto r = p->transcribe(samples, 44100.0);           // mono float32 at the source's own rate
//     for (const dtt::Note& n : r->notes) use(r->tracks[n.track], n);
//
// The shape follows muscriptor.cpp (msl::Transcriber) so a host can wrap both the same way.

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace dtt
{

enum class Error {
    FileNotFound,
    InvalidCheckpoint,
    UnsupportedCheckpointVersion,
    DeviceUnavailable, // the requested device cannot be created or cannot run one of the networks
    Cancelled,
    InvalidArgument,
    Internal,
};

const char* describe(Error e);

/** A device the networks can run on. */
struct Device {
    std::string name;    // e.g. "NVIDIA GeForce RTX 5060", "Apple M1 Pro", "CPU"
    std::string backend; // "Metal", "Vulkan" or "CPU"
    bool integrated = false;
    size_t memory_total = 0; // bytes, 0 when unknown
};

/** GPUs in the backend's order, then the CPU (always last). Built once per process; the first call can be slow. */
std::vector<Device> availableDevices();

struct LoadOptions {
    /** Index into availableDevices(); empty = Auto (first discrete GPU, else integrated, else the CPU). */
    std::optional<size_t> device;
    /** CPU threads; 0 = all logical CPUs. */
    int n_threads = 0;
    /** Polled during loading; returning true makes load() return Error::Cancelled. */
    std::function<bool()> should_cancel;
    /** Fraction of the weights uploaded, 0..1. */
    std::function<void(float)> on_progress;
    /** When load() fails, set to the reason (e.g. the operation a device cannot run), for logs. */
    std::string* error_detail = nullptr;
};

enum class Stage : uint8_t {
    Transcribing, // the tuning check and the transcriber (one pass over the whole recording)
    Repairing,    // the evidence repairer
    Grouping,     // the grouper (instrument tracks)
    Drums,        // the drum model (on a GPU it runs alongside Repairing and Grouping; this stage waits for it)
};

struct Progress {
    Stage stage;
    float stage_fraction; // 0..1 within the stage
    float overall;        // 0..1 over the whole call (stages weighted by their typical run time)
};

struct TranscribeOptions {
    /** Run the drum model. It only runs when the source rate is at least 40 kHz (it reads the 7.8-20 kHz band). */
    bool drums = true;
    /** Polled between windows; returning true makes transcribe() return Error::Cancelled. */
    std::function<bool()> should_cancel;
    std::function<void(const Progress&)> on_progress;
};

/** One instrument track. Drums are their own track (is_drum, program unused). */
struct Track {
    std::string name; // the instrument, e.g. "piano", "alto sax", "drums"
    int program = 0;  // General MIDI program 0-127 the track is written with
    bool is_drum = false;
};

struct Note {
    double onset = 0;  // seconds from the start of the signal
    double offset = 0;
    int pitch = 0;     // MIDI note; the GM percussion key for drums
    int velocity = 0;  // 1-127
    int track = 0;     // index into Transcription::tracks
};

struct Transcription {
    std::vector<Track> tracks; // pitched tracks largest first, then the drum track if any
    std::vector<Note> notes;   // sorted by (onset, track, pitch)
    double seconds = 0;
    double tuning_cents = 0;     // the recording's offset from A440
    std::string drum_model;      // "drum-kit", "orchestral", or empty when the drums did not run
    int notes_transcribed = 0;   // before the evidence repairer
    int notes_dropped = 0;
    int notes_inserted = 0;
};

class Pipeline
{
public:
    /** Loads all four networks from one dettato GGUF onto the chosen device. */
    static std::expected<Pipeline, Error> load(const std::filesystem::path& path, LoadOptions options = {});

    /**
     * Transcribes a whole recording: mono float32 at its own sample rate (any rate >= 16 kHz; it is resampled
     * internally). Blocking: seconds to minutes. One call at a time per Pipeline.
     */
    std::expected<Transcription, Error> transcribe(std::span<const float> mono, double sample_rate,
                                                   const TranscribeOptions& options = {});

    /** The device the networks run on. */
    const Device& device() const;
    /** Why the last call failed, for logs; empty after a success. */
    const std::string& lastErrorMessage() const;

    Pipeline(Pipeline&&) noexcept;
    Pipeline& operator=(Pipeline&&) noexcept;
    ~Pipeline();

private:
    struct Impl;
    explicit Pipeline(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> mImpl;
};

} // namespace dtt
