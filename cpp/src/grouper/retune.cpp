#include "grouper/retune.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

#include "pocketfft_hdronly.h"

namespace dtt::retune
{

namespace
{
    constexpr int N_FFT = 8192, HOP = 1024, H = 6, BLOCK = 512;
    constexpr double FMIN = 65.0, FMAX = 1000.0, STEP_C = 5.0;
    constexpr float REL = 0.5f;

    /** numpy's pairwise summation of n contiguous values (loops_utils.h.src), accumulating in T. */
    template <typename T>
    T pairwise(const T* a, size_t n)
    {
        if (n < 8) {
            T res = 0;
            for (size_t i = 0; i < n; i++) {
                res = res + a[i];
            }
            return res;
        }
        if (n <= 128) {
            T r[8];
            for (int j = 0; j < 8; j++) {
                r[j] = a[j];
            }
            size_t i = 8;
            for (; i < n - (n % 8); i += 8) {
                for (int j = 0; j < 8; j++) {
                    r[j] = r[j] + a[i + j];
                }
            }
            T res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            for (; i < n; i++) {
                res = res + a[i];
            }
            return res;
        }
        size_t n2 = n / 2;
        n2 -= n2 % 8;
        return pairwise(a, n2) + pairwise(a + n2, n - n2);
    }

    /** numpy's complex pairwise sum (CDOUBLE_pairwise_sum): a holds n / 2 complex values as interleaved (re, im). */
    void pairwiseComplex(const double* a, size_t n, double& rr, double& ri)
    {
        if (n < 8) {
            rr = -0.0;
            ri = -0.0;
            for (size_t i = 0; i < n; i += 2) {
                rr += a[i];
                ri += a[i + 1];
            }
            return;
        }
        if (n <= 128) {
            double r[8];
            for (int j = 0; j < 8; j++) {
                r[j] = a[j];
            }
            size_t i = 8;
            for (; i < n - (n % 8); i += 8) {
                for (int j = 0; j < 8; j++) {
                    r[j] += a[i + j];
                }
            }
            rr = (r[0] + r[2]) + (r[4] + r[6]);
            ri = (r[1] + r[3]) + (r[5] + r[7]);
            for (; i < n; i += 2) {
                rr += a[i];
                ri += a[i + 1];
            }
            return;
        }
        size_t n2 = n / 2;
        n2 -= n2 % 8;
        double r1, i1, r2, i2;
        pairwiseComplex(a, n2, r1, i1);
        pairwiseComplex(a + n2, n - n2, r2, i2);
        rr = r1 + r2;
        ri = i1 + i2;
    }

    /** np.arange(start, stop, step) for float64: length ceil((stop - start) / step), values start + i * (a[1] - a[0]). */
    std::vector<double> npArange(double start, double stop, double step)
    {
        const double len = std::ceil((stop - start) / step);
        const size_t n = len > 0 ? static_cast<size_t>(len) : 0;
        std::vector<double> a(n);
        if (n == 0) {
            return a;
        }
        a[0] = start;
        if (n > 1) {
            a[1] = start + step;
            const double delta = a[1] - a[0];
            for (size_t i = 2; i < n; i++) {
                a[i] = start + static_cast<double>(i) * delta;
            }
        }
        return a;
    }

    /** np.remainder for float64 (the result takes the divisor's sign). */
    double npRemainder(double a, double b)
    {
        double mod = std::fmod(a, b);
        if (mod != 0.0) {
            if ((b < 0) != (mod < 0)) {
                mod += b;
            }
        } else {
            mod = std::copysign(0.0, b);
        }
        return mod;
    }
} // namespace

void tuning(const float* y, size_t n, double& cents_out, double& strength)
{
    cents_out = 0.0;
    strength = 0.0;
    if (n < static_cast<size_t>(N_FFT)) {
        return;
    }
    // np.hanning(n_fft).astype(float32)
    std::vector<float> win(N_FFT);
    for (int i = 0; i < N_FFT; i++) {
        win[i] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * i / (N_FFT - 1)));
    }
    const size_t nfr = 1 + (n - N_FFT) / HOP;
    const std::vector<double> cents = npArange(1200.0 * std::log2(FMIN / 440.0), 1200.0 * std::log2(FMAX / 440.0), STEP_C);
    const size_t nc = cents.size();
    std::vector<double> f0(nc);
    for (size_t i = 0; i < nc; i++) {
        f0[i] = 440.0 * std::pow(2.0, cents[i] / 1200.0);
    }
    const int nb = N_FFT / 2 + 1;
    // per harmonic: the bins (float64 positions), validity, float32 weights
    struct Harm {
        std::vector<char> ok;
        std::vector<int> i0;
        std::vector<float> w;
        float coef;
    };
    std::vector<Harm> harm(H);
    for (int h = 1; h <= H; h++) {
        Harm& hm = harm[h - 1];
        hm.ok.resize(nc);
        hm.i0.resize(nc);
        hm.w.resize(nc);
        hm.coef = static_cast<float>(std::pow(0.85, h - 1));
        for (size_t i = 0; i < nc; i++) {
            const double fb = h * f0[i] * N_FFT / SR;
            hm.ok[i] = fb < nb - 2;
            const double fl = std::floor(fb);
            hm.i0[i] = static_cast<int>(fl);
            hm.w[i] = static_cast<float>(fb - fl);
        }
    }

    std::vector<double> cs;
    std::vector<float> ws;
    std::vector<float> frames(static_cast<size_t>(BLOCK) * N_FFT);
    std::vector<std::complex<float>> spec(static_cast<size_t>(BLOCK) * nb);
    std::vector<float> S;   // (nb, frames) as numpy's S = |rfft|.T ** 0.5
    std::vector<float> sal; // (nc, frames)
    for (size_t b0 = 0; b0 < nfr; b0 += BLOCK) {
        const size_t nf = std::min<size_t>(BLOCK, nfr - b0);
        for (size_t k = 0; k < nf; k++) {
            const float* src = y + (b0 + k) * HOP;
            float* fr = frames.data() + k * N_FFT;
            for (int i = 0; i < N_FFT; i++) {
                fr[i] = src[i] * win[i];
            }
        }
        pocketfft::shape_t shape = {nf, static_cast<size_t>(N_FFT)};
        pocketfft::stride_t sin = {static_cast<ptrdiff_t>(N_FFT * sizeof(float)), sizeof(float)};
        pocketfft::stride_t sout = {static_cast<ptrdiff_t>(nb * sizeof(std::complex<float>)), sizeof(std::complex<float>)};
        pocketfft::r2c(shape, sin, sout, {1}, pocketfft::FORWARD, frames.data(), spec.data(), 1.0f, 1);
        S.assign(static_cast<size_t>(nb) * nf, 0.0f);
        for (size_t k = 0; k < nf; k++) {
            for (int b = 0; b < nb; b++) {
                const std::complex<float> z = spec[k * nb + b];
                S[static_cast<size_t>(b) * nf + k] = std::sqrt(std::hypot(z.real(), z.imag()));
            }
        }
        sal.assign(nc * nf, 0.0f);
        for (int h = 0; h < H; h++) {
            const Harm& hm = harm[h];
            for (size_t i = 0; i < nc; i++) {
                if (!hm.ok[i]) {
                    continue;
                }
                const float w = hm.w[i], w1 = 1.0f - w;
                const float* s0 = S.data() + static_cast<size_t>(hm.i0[i]) * nf;
                const float* s1 = s0 + nf;
                float* row = sal.data() + i * nf;
                for (size_t t = 0; t < nf; t++) {
                    row[t] = row[t] + hm.coef * (w1 * s0[t] + w * s1[t]);
                }
            }
        }
        std::vector<float> mx(nf, -INFINITY);
        for (size_t i = 0; i < nc; i++) {
            for (size_t t = 0; t < nf; t++) {
                mx[t] = std::max(mx[t], sal[i * nf + t]);
            }
        }
        for (size_t j = 0; j + 2 < nc; j++) {
            const float* a = sal.data() + j * nf;
            const float* b = a + nf;
            const float* c = b + nf;
            for (size_t t = 0; t < nf; t++) {
                if (b[t] > a[t] && b[t] >= c[t] && b[t] > REL * mx[t] && mx[t] > 0.0f) {
                    const float A = a[t], B = b[t], C = c[t];
                    const float d = (0.5f * (A - C)) / (((A - 2.0f * B) + C) - 1e-12f);
                    const float dc = std::clamp(d, -1.0f, 1.0f) * static_cast<float>(STEP_C);
                    cs.push_back(npRemainder(cents[j + 1] + static_cast<double>(dc) + 50.0, 100.0) - 50.0);
                    ws.push_back(B);
                }
            }
        }
    }
    const float wsum = 0.0f + pairwise(ws.data(), ws.size());
    if (!(wsum > 0.0f)) {
        return;
    }
    // (w * exp(2j pi c / 100)).sum() / w.sum(), as numpy evaluates it (complex arithmetic with real operands promoted)
    std::vector<double> z(2 * cs.size());
    const double twopi = 2.0 * std::numbers::pi, scl100 = 1.0 / 100.0;
    for (size_t i = 0; i < cs.size(); i++) {
        const double re = (0.0 * cs[i]) - (twopi * 0.0); // (0 + 2 pi j)(c + 0j)
        const double im = 0.0 * 0.0 + twopi * cs[i];
        const double er = re * scl100, ei = im * scl100;  // / 100 (Smith's division by 100 + 0j)
        const double ex = std::exp(er);
        const double cr = ex * std::cos(ei), ci = ex * std::sin(ei);
        const double w = ws[i];
        z[2 * i] = w * cr - 0.0 * ci;
        z[2 * i + 1] = w * ci + 0.0 * cr;
    }
    double zr, zi;
    pairwiseComplex(z.data(), z.size(), zr, zi);
    zr = -0.0 + zr; // np.add.reduce starts from the identity
    zi = -0.0 + zi;
    const double scl = 1.0 / static_cast<double>(wsum);
    const double rr = zr * scl, ri = zi * scl;
    cents_out = std::atan2(ri, rr) * 100.0 / (2.0 * std::numbers::pi);
    strength = std::hypot(rr, ri);
}

void ratio(const float* y, size_t n, double& delta, double& strength, double& shift, double& r)
{
    tuning(y, n, delta, strength);
    shift = (std::abs(delta) >= MIN_CENTS && strength >= MIN_STRENGTH) ? -delta : 0.0;
    r = std::pow(2.0, shift / 1200.0);
}

} // namespace dtt::retune
