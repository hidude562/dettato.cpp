#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace dtt
{

class ModelFile;

/**
 * The drum model's input features (dettato/drums/features.py), on the CPU: 44.1 kHz mono -> (T, 280) log-power bands at
 * 100 fps, [lo 128 | hi 32 | cq 120], rounded through float16 as features() returns them.
 *
 *   level   x = x44 * gain, gain bringing the float64 RMS to -18 dBFS (numpy's pairwise sum, float32 product)
 *   x16     x resampled to 16 kHz (dtt::resample standing in for soxr HQ; the parity tool can inject soxr's output)
 *   lo      log(mel(|STFT 512 @16k, hop 160|^2) + 1e-5), 128 Slaney mel bands 20 Hz - 7.8 kHz
 *   hi      the same of x, STFT 1411 @44.1k hop 441, 32 mel bands 7.8 - 20 kHz
 *   cq      the same of x16, STFT 6144 hop 160, 120 log-frequency bands
 *   T       min(len(x16) // 160, frames of each block)
 *
 * The filterbanks and Hann windows come from the GGUF (built by the Python functions).
 */
class DrumFeatures
{
public:
    explicit DrumFeatures(const ModelFile& mf, const std::string& prefix = "dr");
    ~DrumFeatures();

    struct Debug {
        double gain = 0;
        std::vector<float> x16;        // the 16 kHz signal used
        std::vector<float> lo, hi, cq; // (T, bands), before the float16 rounding
        double resample_ms = 0, lo_ms = 0, hi_ms = 0, cq_ms = 0;
    };

    /**
     * @param x16 if non-null, used instead of resampling x44 * gain to 16 kHz.
     * @param T out: the number of frames.
     * @return (T, bands()) row-major, float16 values.
     */
    std::vector<float> compute(const float* x44, size_t n, int& T, const std::vector<float>* x16 = nullptr,
                               Debug* dbg = nullptr, int threads = 0) const;

    /** level_gain: 10^(-18/20) / (rms + 1e-9), rms in float64 with numpy's summation order. */
    static double levelGain(const float* x, size_t n);

    int bands() const { return mLo.bands + mHi.bands + mCq.bands; }
    int sampleRate() const { return mSr; }

    struct Spec {
        int n_fft = 0, hop = 0, bands = 0;
        std::vector<float> window;
        std::vector<int> start, len; // each band's nonzero bins
        std::vector<size_t> off;
        std::vector<float> w;
    };

private:
    void logspec(const float* x, size_t n, int frames, const Spec& s, float* out, int stride, int threads) const;

    int mSr = 44100, mSr16 = 16000;
    float mFloor = 1e-5f;
    float mLevelDbfs = -18.0f;
    Spec mLo, mHi, mCq;
};

} // namespace dtt
