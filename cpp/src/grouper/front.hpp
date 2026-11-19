#pragma once

// The grouper's own audio front end and per-note readers (dettato/grouper/front.py), on the CPU.
//
// Notes of a window are the model's (N, 12) float32 rows: onset, offset (window-relative s), pitch, slot, 8 map scalars.
// Every index computation follows PyTorch's float32 arithmetic (round = half to even, .long() = truncation).

#include <cstdint>
#include <vector>

#include "transcriber/hcqt.hpp"

namespace dtt
{

class ModelFile;

class GroupFront
{
public:
    static constexpr int SR = 16000, HOP = 160;
    static constexpr double PAD0 = 0.5;
    static constexpr int NOTE_COLS = 12;

    explicit GroupFront(const ModelFile& mf);

    int channels() const { return mHcqt.channels(); }
    int bins() const { return mHcqt.bins(); }
    int extBins() const { return mHcqt.extendedBins(); }
    int shiftLo() const { return mShiftLo; } // HDemix comb bins = 3 lane + shiftLo
    int hdHarmonics() const { return mH; }
    int hdFeatures() const;                  // H * (len(HD_K) + sus) + H

    /**
     * One window's audio (n samples) -> X (channels, T, bins): log HCQT -> PCEN -> standardise, and P (extBins, T): the
     * linear power on the extended log-frequency axis. T = n / HOP. P may be null.
     * PCEN is the exact causal convolution with the truncated exponential kernel (see pcen()).
     */
    void compute(const float* audio, size_t n, float* X, float* P) const;
    /** The first step of compute(): X = the log HCQT (no PCEN / standardisation yet), P. */
    void logHcqt(const float* audio, size_t n, float* X, float* P) const { mHcqt.compute(audio, n, X, P); }

    /**
     * front.pcen on log-power maps (channels, T, bins), in place. The reference computes the 200-tap causal convolution with
     * a float32 FFT, whose round-off swamps the convolution wherever a frequency channel is quiet next to loud parts (digital
     * silence, the window padding); this is the exact convolution (a recursion in float64 with the kernel s (1 - s)^j,
     * j < K), then the same float32 compression.
     */
    void pcen(float* X, int C, int T, int F) const;
    /** front.standardise, per channel over (T, bins), in place. */
    static void standardise(float* X, int C, int T, int F);

    /** HDemix.raw for one window: P (extBins, T), notes (N, 12), mask -> (N, hdFeatures()). */
    void hdemixRaw(const float* P, int T, const float* notes, const uint8_t* mask, int N, float* out) const;
    /** Prof's input for one window: X (channels, T, bins), notes -> (N, 4 channels). */
    void profRaw(const float* X, int T, const float* notes, int N, float* out) const;

    /**
     * front.rasters for one window: R channels-last (Tq, lanes, 4) (onsets, sustain, onset confidence, second-voice onsets),
     * plus each note's onset / end frame and lane (all N rows, padding included).
     */
    static void rasters(const float* notes, const uint8_t* mask, int N, int Tq, int lanes, float fps, float* R, int* on,
                        int* off, int* lane);

private:
    Hcqt mHcqt;
    int mShiftLo = 0;
    int mH = 10, mSus = 4;
    std::vector<int> mHdShift;
    std::vector<float> mPi; // softplus(prior)
    double mS = 0.025, mAlpha = 0.98, mDelta = 2.0, mR = 0.5, mEps = 1e-6;
    int mK = 200;
};

/** float -> float16 -> float, round to nearest even (numpy / torch .astype(float16)). */
float roundHalf(float x);

} // namespace dtt
