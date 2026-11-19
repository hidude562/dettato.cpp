#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace dtt
{

class Backend;
class ModelFile;
class DrumFeatures;
class DrumNets;
struct DrumDecoder;
struct DrumTaps;
struct DrumTiming;

/** One drum hit, as dettato.drums.DrumTranscriber.transcribe returns it (onset / offset / velocity rounded the same way). */
struct DrumHit {
    double onset, offset; // seconds
    int pitch;            // GM drum key; timpani hits are on keys 41-53, the key being the drum's pitch
    float velocity;       // 0..1
};

/** DrumTranscriber.last_info: the router value and the model that wrote the hits. */
struct DrumInfo {
    float router = 0;  // NaN when the input was too short to transcribe (Python: None)
    std::string model; // "drum-kit" or "orchestral" ("" when nothing ran)
};

/**
 * The drum stage (dettato/drums): 44.1 kHz mono audio -> drum hits on GM drum keys. The drum-kit model (a DrumNet), three
 * orchestral experts and their router, all in the GGUF's "dr." tensors (py/convert_drums.py).
 *
 * Matches the reference in structure: the features (three STFTs, rounded through float16 as the reference does), every
 * network the reference runs, the router mean, peak picking and the timpani merge, transcribe()'s rounding. Two
 * deliberate differences, neither changing the hits: transcribe() skips the drum-kit network when the router picks the
 * orchestral model (its output would be discarded), and long songs run the experts' cq branch window by window
 * (bounded memory, the same values up to float rounding; see DrumNets::setCqChunk). The 44.1 -> 16 kHz resampler is dtt::resample, a stand-in for
 * soxr HQ (~0.3% rel L2 on the signal, ~1-2% of the hits moved; see cpp/tools/parity_drums.cpp).
 *
 * Memory: weights 13 MB (f16) / 22 MB (f32). Compute buffer (device memory on a GPU): the stems run in 80 s windows and
 * the experts' cq branch in windows beyond 2 min on a GPU (6 min on the CPU backend), so it stays near 0.26 GB on a GPU
 * (0.69 GB on the CPU for a 4-min song); what still grows with the song is the whole-song trunk / adapter graphs (the
 * attention's q, k, v and the GRU projections), ~0.4 MB per second of audio: 0.1 GB at 4 min, 0.5 GB at 20 min.
 * Threads: on a GPU backend the three experts run in parallel host threads that take turns on the backend, so one call
 * keeps the device and a few cores busy. A DrumTranscriber is not reentrant on one Backend: one transcribe() at a time.
 */
class DrumTranscriber
{
public:
    DrumTranscriber(const ModelFile& mf, const Backend& be);
    ~DrumTranscriber();
    DrumTranscriber(const DrumTranscriber&) = delete;
    DrumTranscriber& operator=(const DrumTranscriber&) = delete;

    /**
     * Mono audio at 44.1 kHz (the caller resamples; the source was recorded at >= 40 kHz) -> hits sorted by onset,
     * exactly as DrumTranscriber.transcribe returns them (rounded the same way). info may be null.
     */
    std::vector<DrumHit> transcribe(const float* audio44k, size_t n, DrumInfo* info) const;

    /**
     * The same, given also the 16 kHz version of the same audio that the pipeline already made for the transcriber (with
     * dtt::resample): the features then skip their own 44.1 -> 16 kHz resampling (~0.3 s of a 4-min song). The
     * reference resamples the level-normalised signal; resampling is linear, so audio16k times the level gain is the
     * same signal up to float rounding.
     */
    std::vector<DrumHit> transcribe(const float* audio44k, size_t n, const float* audio16k, size_t n16, DrumInfo* info) const;

    /**
     * The same with the parity tool's hooks: x16 replaces the internal 44.1 -> 16 kHz resampler (Python's soxr output),
     * taps collects intermediate tensors, timing the time per part; any may be null. all_models runs the drum-kit
     * network even when the router does not use it.
     */
    std::vector<DrumHit> transcribe(const float* audio44k, size_t n, DrumInfo* info, const std::vector<float>* x16,
                                    DrumTaps* taps, DrumTiming* timing, bool all_models) const;

    const DrumFeatures& features() const { return *mFeatures; }
    const DrumNets& nets() const { return *mNets; }
    DrumNets& nets() { return *mNets; }
    const DrumDecoder& decoder() const { return *mDecoder; }
    int sampleRate() const { return mSampleRate; }
    int minSourceRate() const { return mMinSourceRate; }

private:
    std::unique_ptr<DrumFeatures> mFeatures;
    std::unique_ptr<DrumNets> mNets;
    std::unique_ptr<DrumDecoder> mDecoder;
    int mSampleRate = 44100;
    int mMinSourceRate = 40000;
    int mThreads = 1;
};

} // namespace dtt
