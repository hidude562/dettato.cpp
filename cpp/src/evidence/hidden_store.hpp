#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "transcriber/transcriber.hpp"

namespace dtt::ev
{

/** The transcriber's decode thresholds (tr.th.*), which bound where decoded notes can start and end. */
struct DecodeThresholds {
    float on_th = 0.25f;  // voice 1 onset threshold
    float v2_th = -1.0f;  // voice 2 onset threshold (< 0: voice 2 is not decoded)
    float fr_th = 0.4f;
    float off_th = 0.3f;
};

/**
 * The hidden-state tap of the transcriber's first pass, kept only where the evidence vectors read it.
 *
 * What the evidence needs (features.py transcribe_with_evidence + gather_hidden), per decoded note on lane L from frame A
 * to frame B (exclusive):
 *   at   = H[A, L]
 *   mean = (cs[B, L] - cs[A, L]) / (B - A),  cs[t] = float32(sum_{t' < t} H[t']) (torch's CPU cumsum: a float64 running
 *          sum, rounded to float32 per frame), all in float32 as torch computes it.
 * So the store keeps, per lane, H at the frames that can be an A, and cs at the frames that can be an A or a B. It needs
 * every frame of every lane once (for the running sums), but only while that frame is being averaged:
 *
 * 1. window(): each window's (C, lane, frame) block is added, times its Hann weight, into a ring of float32 frames that
 *    covers the frames not yet final (Transcriber.run's buf[a:b] += v * w, in float32 instead of float16).
 * 2. framesFinal(): a frame t whose map rows up to t + m + 3 are final is normalised (H = acc / max(wsum, 1e-6)), its
 *    candidate flags are evaluated from the maps, H / cs are stored where flagged, the per-lane float64 sums advance, and
 *    the ring slot is released.
 *
 * Which frames can be A or B (decode.py `decode` and core.py `notes`, for a note of voice v on lane l; th_v = on_th or
 * v2_th; a "peak" is decode's _peaks: p >= th, p > the previous frame, p >= the next (-1 beyond the song); all comparisons
 * in float32 as decode makes them):
 *
 *   t0 = the note's onset: an onset peak of voice v (NMS only removes peaks). u0 = t0 + refine, |refine| <= 0.5.
 *   onset = u0 * hop, A = clip(round(onset / hop), 0, T - 1). round() of a value within 0.5 (+ the float error of
 *   u0 * hop / hop) of t0 is t0 - 1, t0 or t0 + 1, so                                  (A)  A is in [t0 - 1, t0 + 1].
 *
 *   The end u1 is one of (decode.py):
 *    (1) e_off + refine(off): e_off an offset peak of voice v, e_off <= min(e_fr, nxt);
 *    (2) e_fr - 0.5: e_fr the first t >= t0 + 1 with run[t] (fr_v[t] < fr_th and, unless t = T - 1, fr_v[t + 1] < fr_th),
 *        or T when there is none; taken when e_fr <= nxt;
 *    (3) nxt + refine(on, nxt): nxt the next kept onset of the lane (an onset peak), taken when nxt < e_fr (so nxt < T).
 *   offset = max(u1 * hop, u0 * hop + min_dur) with min_dur = 1 frame, B = clip(round(offset / hop), A + 1, T). With the
 *   same rounding argument, u = B - 1 (the note's last frame) is:
 *    - from (1): B in {e_off - 1, e_off, e_off + 1}  -> an offset peak in [u, u + 2];
 *    - from (2): B in {e_fr - 1, e_fr}               -> run[e_fr] for an e_fr in [u + 1, u + 2], or (e_fr = T) u >= T - 2;
 *    - from (3): B in {nxt - 1, nxt, nxt + 1}        -> an onset peak in [u, u + 2];
 *    - from min_dur: round(u0 + 1) in {t0, t0 + 1, t0 + 2}, and from the clip B = A + 1 (u = A)
 *                                                    -> an onset peak in [u - 1, u + 1];
 *    - from the clip B = T                           -> u = T - 1.
 *   And u is a frame of the note, so (the frame argument) on_v >= th_v or fr_v >= fr_th at some frame in [u - 1, u + 1]:
 *
 *   The frame argument: every frame t in [A, B) has on_v >= th_v or fr_v >= fr_th within [t - 1, t + 1]. For t <= t0 + 1 it
 *   is the onset (A >= t0 - 1). For t >= t0 + 2: by the cases above B - 1 <= max(e_fr, t0 + 1) ((1): B - 1 <= e_off <= e_fr;
 *   (2): B - 1 <= e_fr - 1; (3): B - 1 <= nxt < e_fr; min_dur / clip: <= t0 + 1). If t < e_fr, run[t] is false: fr_v[t]
 *   >= fr_th or fr_v[t + 1] >= fr_th (t < T - 1), fr_v[t] >= fr_th (t = T - 1). If t = e_fr (only in case (1)), e_fr - 1 >=
 *   t0 + 1 has run false while fr_v[e_fr] < fr_th, so fr_v[e_fr - 1] >= fr_th.
 *
 * Hence, with every interval widened by a margin of m frames (default 0: the bounds above are exact; a miss would show in
 * the parity tool, which counts them and fails on any):
 *   candA(t)    = an onset peak of either voice in [t - 1 - m, t + 1 + m]                           -> keep H[t] and cs[t]
 *   candLast(u) = for some voice v: (on_v >= th_v or fr_v >= fr_th somewhere in [u - 1 - m, u + 1 + m]) and
 *                 (an offset peak in [u - m, u + 2 + m], or an onset peak in [u - 1 - m, u + 2 + m], or run[e] for an e in
 *                  [u + 1 - m, u + 2 + m], or u >= T - 2 - m)                                     -> keep cs[u + 1]
 * Frames outside [0, T) have no peaks, no run and nothing above threshold.
 *
 * Memory: the ring (the frames a batch of windows spans beyond the finalised ones, ~500 frames x 88 x 368 floats for batch
 * 2, independent of the song's length; freed after the pass) plus the kept vectors, which grow with the number of notes,
 * not with their lengths.
 */
class HiddenStore final : public TapSink
{
public:
    /** threads: for the accumulation (0: half the hardware threads, at most 8). */
    explicit HiddenStore(const DecodeThresholds& th, int margin = 0, int threads = 0);
    ~HiddenStore() override;

    void window(int tap, int g0, int a, int b, const float* w, const float* data, int C, int L, int Fw) override;
    void framesFinal(int t0, int t1, const Maps& maps, const std::vector<float>& wsum) override;

    /** True after the final framesFinal call (t1 = maps.T). */
    bool complete() const { return mDone; }
    int channels() const { return mC; }
    int frames() const { return mT; }

    /** H[t, l], normalised; nullptr where not kept. */
    const float* hidden(int t, int l) const;
    /** cs[t, l] = float32(sum_{t' < t} H[t', l]) as torch.cumsum gives it; nullptr where not kept. */
    const float* prefix(int t, int l) const;

    struct Stats {
        size_t ring_frames = 0; // peak ring capacity (frames) for the accumulation
        size_t ring_bytes = 0;
        size_t n_hidden = 0;    // kept H vectors
        size_t n_prefix = 0;    // kept cs vectors
        size_t store_bytes = 0; // the kept vectors + their indices
        size_t sums_bytes = 0;  // the per-lane float64 running sums and the per-frame flags
        double accum_s = 0;     // time spent in window()
        double final_s = 0;     // time spent in framesFinal()
    };
    Stats stats() const;

private:
    struct Lane {
        std::vector<int> h_t, p_t;          // frames, ascending
        std::vector<uint32_t> h_i, p_i;     // entries in the pool
    };
    void ensureRing(int upto);
    float* slot(int t) { return mRing.data() + static_cast<size_t>(t % mR) * mL * mC; }
    void computeFlags(int upto, const Maps& maps);
    void process(int t, const std::vector<float>& wsum);
    uint32_t put(const float* v);
    const float* find(const std::vector<int>& ts, const std::vector<uint32_t>& is, int t) const;
    bool candA(int t, int l) const;
    bool candLast(int u, int l) const;
    bool run(int t, int l, int v) const;
    uint8_t flag(int t, int l) const;

    DecodeThresholds mTh;
    int mM;
    int mThreads = 1;
    int mC = 0, mL = 0, mT = -1;
    // ring of not-yet-final frames [mBase, mTouched)
    std::vector<float> mRing;
    int mR = 0, mBase = 0, mTouched = 0;
    size_t mRingPeak = 0;
    // per frame and lane, for voice 1 / 2: onset peak, offset peak, on >= th or fr >= fr_th, fr < fr_th
    std::vector<uint8_t> mFlags;
    int mFlagged = 0;
    std::vector<double> mSum;   // (L, C) running sums
    std::vector<char> mPrevLast; // per lane: candLast(t - 1) of the frame being processed
    std::vector<Lane> mLanes;
    std::vector<std::unique_ptr<float[]>> mPool;
    size_t mPoolUsed = 0;
    std::vector<float> mH;      // one normalised frame (L, C)
    std::vector<float> mCs;     // one cs vector
    bool mDone = false;
    double mAccumS = 0, mFinalS = 0;
};

} // namespace dtt::ev
