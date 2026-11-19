#pragma once

#include <cstddef>
#include <vector>

namespace dtt
{

/**
 * Power spectrum frames, as torch.stft(x, n_fft, hop, window=win, center=True, pad_mode="constant") |.|^2 computes them:
 * frame t is centred on sample t * hop, with zeros outside the signal.
 */
class PowerStft
{
public:
    PowerStft(int n_fft, int hop, std::vector<float> window);

    int bins() const { return mNfft / 2 + 1; }
    int nfft() const { return mNfft; }

    /** out: n_frames rows of bins() floats. */
    void compute(const float* x, size_t n, int n_frames, float* out) const;

    /** Magnitude instead of power (|X|, as torch .abs()). */
    void computeMagnitude(const float* x, size_t n, int n_frames, float* out) const;

private:
    void transform(const float* x, size_t n, int n_frames, float* out, bool power) const;

    int mNfft;
    int mHop;
    std::vector<float> mWindow;
};

} // namespace dtt
