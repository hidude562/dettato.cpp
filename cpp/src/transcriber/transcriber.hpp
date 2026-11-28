#pragma once

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "common.hpp"
#include "transcriber/decode.hpp"
#include "transcriber/hcqt.hpp"
#include "transcriber/lanenet.hpp"

namespace dtt
{

class Backend;
class ModelFile;

enum Head { ONSET = 0, FRAME, OFFSET, ONSET2, FRAME2, OFFSET2, N_HEADS };

/** The six probability maps of a song, each (T, 88) row-major (lane fastest), averaged over the windows. */
struct Maps {
    int T = 0;
    int L = 0;
    std::array<std::vector<float>, N_HEADS> m;
    const float* operator[](int h) const { return m[h].data(); }
    float at(int h, int t, int l) const { return m[h][static_cast<size_t>(t) * L + l]; }
};

/** A transcriber note with its voice: Transcriber.notes in the Python package. */
struct TranscribedNote {
    double onset;
    double offset;
    int pitch;
    float conf; // the voice's onset probability at the onset frame
    int slot;   // 1 or 2
};

/**
 * Receives each window's internal states as the windows complete, in time order, for the later stages to average or
 * gather. data is (C, lane, frame) channels-last for one window; frame f of the window is song frame g0 + f. Song frames
 * [a, b) are inside the song; w[f - (a - g0)] is the Hann weight of window frame f.
 */
class TapSink
{
public:
    virtual ~TapSink() = default;
    virtual void window(int tap, int g0, int a, int b, const float* w, const float* data, int C, int L, int Fw) = 0;
    /**
     * Song frames [t0, t1) are final: no later window touches them, and maps rows t < t1 hold their final (normalised)
     * values. Called after each batch, in order, ending with t1 = maps.T. wsum[t] (t < t1) is the Hann weight sum.
     */
    virtual void framesFinal(int /*t0*/, int /*t1*/, const Maps& /*maps*/, const std::vector<float>& /*wsum*/) {}
    static constexpr int HIDDEN = 0;
    static constexpr int STEM = 1;
};

/** Running the transcriber over a whole recording (dettato/transcriber/core.py Transcriber). */
class Transcriber
{
public:
    static constexpr int SR = 16000;
    static constexpr double WIN_S = 8.0;

    Transcriber(const ModelFile& mf, const Backend& be, int batch, LaneNet::Taps taps);
    ~Transcriber();

    double hop() const { return mHop; }
    const DecodeParams& params() const { return mParams; }
    float v2Threshold() const { return mV2Th; }
    int channels() const { return mNet->channels(); }
    LaneNet& net() { return *mNet; }
    const Hcqt& hcqt() const { return mHcqt; }

    struct Layout {
        long W = 0;
        int Fw = 0;
        int T = 0;
        std::vector<long> starts;
        long pad = 0;
        int g_pad = 0;
    };
    Layout layout(size_t n, double hop_frac) const;
    int frameOffset(long s0, int g_pad) const;
    /** The Hann taper over a window's frames (np.hanning(Fw + 2)[1:-1], float32). */
    std::vector<float> taper(int Fw) const;

    /**
     * Every window of the recording through the network -> the maps. The taps in tap_mask (bits 1 << TapSink::HIDDEN,
     * 1 << TapSink::STEM; each must have been enabled at construction) go to sink window by window.
     * progress(fraction) returning false cancels (the call then throws Cancelled).
     */
    Maps run(const float* audio, size_t n, double hop_frac, TapSink* sink, unsigned tap_mask,
             const std::function<bool(float)>& progress = {});
    struct RunOptions {
        double hop_frac = 0.25;
        TapSink* sink = nullptr;
        unsigned tap_mask = 0;
        /** Also average, into *sub_maps, only the windows whose start is a multiple of sub_step samples: with a hop of
         *  0.25 and sub_step 64000 these are exactly the windows a hop-0.5 pass would run. */
        Maps* sub_maps = nullptr;
        long sub_step = 0;
        /** Run only the windows i of layout(n, hop_frac) with (*only)[i] != 0 (the others contribute nothing). */
        const std::vector<char>* only = nullptr;
        std::function<bool(float)> progress;
    };
    Maps run(const float* audio, size_t n, const RunOptions& o);

    Maps run(const float* audio, size_t n, double hop_frac, TapSink* sink, const std::function<bool(float)>& progress = {})
    {
        return run(audio, n, hop_frac, sink, (mTaps.hidden ? 1u << TapSink::HIDDEN : 0u), progress);
    }

    /** Both voices decoded, sorted by (onset, offset, pitch). */
    std::vector<TranscribedNote> notes(const Maps& maps) const;

    /** HCQT of one window into a batch slot (exposed for the parity tests). */
    void windowHcqt(const float* x, float* out) const;

private:
    Hcqt mHcqt;
    std::unique_ptr<LaneNet> mNet;
    double mHop = 0.02;
    DecodeParams mParams;
    float mV2Th = -1.0f;
    LaneNet::Taps mTaps;
};

/** Forwards every window and every final-frames notice to several sinks, in order. */
class FanOutSink : public TapSink
{
public:
    explicit FanOutSink(std::vector<TapSink*> sinks) : mSinks(std::move(sinks)) {}
    void window(int tap, int g0, int a, int b, const float* w, const float* data, int C, int L, int Fw) override
    {
        for (TapSink* s : mSinks) {
            s->window(tap, g0, a, b, w, data, C, L, Fw);
        }
    }
    void framesFinal(int t0, int t1, const Maps& maps, const std::vector<float>& wsum) override
    {
        for (TapSink* s : mSinks) {
            s->framesFinal(t0, t1, maps, wsum);
        }
    }

private:
    std::vector<TapSink*> mSinks;
};

} // namespace dtt
