#include "dsp/stft.hpp"

#include <algorithm>
#include <cmath>
#include <complex>

#include "common.hpp"
#include "pocketfft_hdronly.h"

namespace dtt
{

PowerStft::PowerStft(int n_fft, int hop, std::vector<float> window)
    : mNfft(n_fft), mHop(hop), mWindow(std::move(window))
{
    DTT_CHECK(static_cast<int>(mWindow.size()) == n_fft, "window length %zu != n_fft %d", mWindow.size(), n_fft);
}

void PowerStft::compute(const float* x, size_t n, int n_frames, float* out) const
{
    transform(x, n, n_frames, out, true);
}

void PowerStft::computeMagnitude(const float* x, size_t n, int n_frames, float* out) const
{
    transform(x, n, n_frames, out, false);
}

void PowerStft::transform(const float* x, size_t n, int n_frames, float* out, bool power) const
{
    if (n_frames <= 0) {
        return;
    }
    const int N = mNfft, half = N / 2, nb = bins();
    // Frames in blocks, so the scratch stays small for long signals.
    constexpr int BLOCK = 256;
    std::vector<float> frames(static_cast<size_t>(BLOCK) * N);
    std::vector<std::complex<float>> spec(static_cast<size_t>(BLOCK) * nb);
    for (int f0 = 0; f0 < n_frames; f0 += BLOCK) {
        const int nf = std::min(BLOCK, n_frames - f0);
        for (int k = 0; k < nf; k++) {
            const long c = static_cast<long>(f0 + k) * mHop - half;
            float* fr = frames.data() + static_cast<size_t>(k) * N;
            for (int i = 0; i < N; i++) {
                const long j = c + i;
                fr[i] = (j >= 0 && j < static_cast<long>(n)) ? x[j] * mWindow[i] : 0.0f;
            }
        }
        pocketfft::shape_t shape = {static_cast<size_t>(nf), static_cast<size_t>(N)};
        pocketfft::stride_t sin = {static_cast<ptrdiff_t>(N * sizeof(float)), sizeof(float)};
        pocketfft::stride_t sout = {static_cast<ptrdiff_t>(nb * sizeof(std::complex<float>)), sizeof(std::complex<float>)};
        pocketfft::r2c(shape, sin, sout, {1}, pocketfft::FORWARD, frames.data(), spec.data(), 1.0f, 1);
        for (int k = 0; k < nf; k++) {
            const std::complex<float>* s = spec.data() + static_cast<size_t>(k) * nb;
            float* o = out + static_cast<size_t>(f0 + k) * nb;
            for (int b = 0; b < nb; b++) {
                const float p = s[b].real() * s[b].real() + s[b].imag() * s[b].imag();
                o[b] = power ? p : std::sqrt(p);
            }
        }
    }
}

} // namespace dtt
