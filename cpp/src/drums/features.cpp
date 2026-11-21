#include "drums/features.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <thread>

#include <ggml.h>

#include "common.hpp"
#include "dsp/resample.hpp"
#include "model_file.hpp"
#include "pocketfft_hdronly.h"

namespace dtt
{

namespace
{
    /** numpy's pairwise summation (DOUBLE_pairwise_sum), so the RMS is bit for bit numpy's. */
    double pairwiseSum(const double* a, size_t n)
    {
        if (n < 8) {
            double res = 0.0;
            for (size_t i = 0; i < n; i++) {
                res += a[i];
            }
            return res;
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
            double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            for (; i < n; i++) {
                res += a[i];
            }
            return res;
        }
        size_t n2 = n / 2;
        n2 -= n2 % 8;
        return pairwiseSum(a, n2) + pairwiseSum(a + n2, n - n2);
    }

    DrumFeatures::Spec makeSpec(const ModelFile& mf, const std::string& p, const std::string& fb, int n_fft, int hop)
    {
        DrumFeatures::Spec s;
        s.n_fft = n_fft;
        s.hop = hop;
        s.window = mf.hostF32(p + ".win." + std::to_string(n_fft));
        DTT_CHECK(static_cast<int>(s.window.size()) == n_fft, "window %d has %zu taps", n_fft, s.window.size());
        const auto dense = mf.hostF32(p + ".fb." + fb);
        const int nb = n_fft / 2 + 1;
        DTT_CHECK(dense.size() % nb == 0, "filterbank %s does not have %d bins", fb.c_str(), nb);
        s.bands = static_cast<int>(dense.size() / nb);
        for (int r = 0; r < s.bands; r++) {
            const float* row = dense.data() + static_cast<size_t>(r) * nb;
            int a = 0, b = nb;
            while (a < nb && row[a] == 0.0f) {
                a++;
            }
            while (b > a && row[b - 1] == 0.0f) {
                b--;
            }
            s.start.push_back(a);
            s.len.push_back(b - a);
            s.off.push_back(s.w.size());
            s.w.insert(s.w.end(), row + a, row + b);
        }
        return s;
    }

    template <class F>
    void parallelFor(int n, int threads, F&& f)
    {
        threads = std::max(1, std::min(threads, n));
        if (threads == 1) {
            f(0, n);
            return;
        }
        std::vector<std::thread> pool;
        const int chunk = (n + threads - 1) / threads;
        for (int i = 0; i < threads; i++) {
            const int a = i * chunk, b = std::min(n, a + chunk);
            if (a < b) {
                pool.emplace_back([&f, a, b] { f(a, b); });
            }
        }
        for (auto& t : pool) {
            t.join();
        }
    }
} // namespace

DrumFeatures::DrumFeatures(const ModelFile& mf, const std::string& p)
{
    mSr = static_cast<int>(mf.u32(p + ".sample_rate"));
    mFloor = mf.f32(p + ".log_floor");
    mLevelDbfs = mf.f32(p + ".level_dbfs");
    const int cq_nfft = static_cast<int>(mf.u32(p + ".cq_nfft"));
    mLo = makeSpec(mf, p, "lo", 512, mSr16 / 100);
    mHi = makeSpec(mf, p, "hi", 1411, mSr / 100);
    mCq = makeSpec(mf, p, "cq", cq_nfft, mSr16 / 100);
}

DrumFeatures::~DrumFeatures() = default;

double DrumFeatures::levelGain(const float* x, size_t n)
{
    std::vector<double> sq(n);
    for (size_t i = 0; i < n; i++) {
        sq[i] = static_cast<double>(x[i]) * static_cast<double>(x[i]);
    }
    const double mean = pairwiseSum(sq.data(), n) / static_cast<double>(n);
    const double rms = std::sqrt(mean) + 1e-9;
    return std::pow(10.0, -18.0 / 20.0) / rms;
}

void DrumFeatures::logspec(const float* x, size_t n, int frames, const Spec& s, float* out, int stride, int threads) const
{
    const int N = s.n_fft, half = N / 2, nb = N / 2 + 1;
    const float floor = mFloor;
    parallelFor(frames, threads, [&](int f0, int f1) {
        constexpr int BLOCK = 32;
        std::vector<float> buf(static_cast<size_t>(BLOCK) * N), pw(nb);
        std::vector<std::complex<float>> spec(static_cast<size_t>(BLOCK) * nb);
        for (int b0 = f0; b0 < f1; b0 += BLOCK) {
            const int nf = std::min(BLOCK, f1 - b0);
            for (int k = 0; k < nf; k++) {
                // torch.stft(center=True, pad_mode="constant"): frame t is centred on sample t * hop, zeros outside
                const long c = static_cast<long>(b0 + k) * s.hop - half;
                float* fr = buf.data() + static_cast<size_t>(k) * N;
                for (int i = 0; i < N; i++) {
                    const long j = c + i;
                    fr[i] = (j >= 0 && j < static_cast<long>(n)) ? x[j] * s.window[i] : 0.0f;
                }
            }
            pocketfft::shape_t shape = {static_cast<size_t>(nf), static_cast<size_t>(N)};
            pocketfft::stride_t sin = {static_cast<ptrdiff_t>(N * sizeof(float)), sizeof(float)};
            pocketfft::stride_t sout = {static_cast<ptrdiff_t>(nb * sizeof(std::complex<float>)), sizeof(std::complex<float>)};
            pocketfft::r2c(shape, sin, sout, {1}, pocketfft::FORWARD, buf.data(), spec.data(), 1.0f, 1);
            for (int k = 0; k < nf; k++) {
                const std::complex<float>* sp = spec.data() + static_cast<size_t>(k) * nb;
                for (int b = 0; b < nb; b++) {
                    const float m = std::sqrt(sp[b].real() * sp[b].real() + sp[b].imag() * sp[b].imag()); // s.abs()
                    pw[b] = m * m;                                                                             // ** 2
                }
                float* o = out + static_cast<size_t>(b0 + k) * stride;
                for (int r = 0; r < s.bands; r++) {
                    const float* w = s.w.data() + s.off[r];
                    const float* p = pw.data() + s.start[r];
                    double acc = 0.0;
                    for (int i = 0; i < s.len[r]; i++) {
                        acc += static_cast<double>(w[i]) * p[i];
                    }
                    o[r] = std::log(static_cast<float>(acc) + floor);
                }
            }
        }
    });
}

std::vector<float> DrumFeatures::compute(const float* x44, size_t n, int& T, const std::vector<float>* x16_in, Debug* dbg,
                                         int threads) const
{
    if (threads <= 0) {
        threads = defaultThreadCount();
    }
    const double gain = levelGain(x44, n);
    // x44 * gain with a Python float: numpy does it in float32 (the scalar cast to float32)
    const float g32 = static_cast<float>(gain);
    std::vector<float> x(n);
    for (size_t i = 0; i < n; i++) {
        x[i] = x44[i] * g32;
    }
    using Clock = std::chrono::steady_clock;
    auto ms = [](Clock::time_point a) { return std::chrono::duration<double, std::milli>(Clock::now() - a).count(); };
    auto t0 = Clock::now();
    std::vector<float> x16_own;
    const std::vector<float>* x16 = x16_in;
    if (x16 == nullptr) {
        x16_own = resample(x.data(), n, mSr, mSr16, threads);
        x16 = &x16_own;
    }
    const double t_rs = ms(t0);
    const size_t n16 = x16->size();
    const int f_lo = static_cast<int>(1 + n16 / mLo.hop), f_hi = static_cast<int>(1 + n / mHi.hop);
    const int f_cq = static_cast<int>(1 + n16 / mCq.hop);
    T = std::min({static_cast<int>(n16 / 160), f_lo, f_hi, f_cq});
    T = std::max(T, 0);
    const int C = bands();
    std::vector<float> feats(static_cast<size_t>(T) * C);
    t0 = Clock::now();
    logspec(x16->data(), n16, T, mLo, feats.data(), C, threads);
    const double t_lo = ms(t0);
    t0 = Clock::now();
    logspec(x.data(), n, T, mHi, feats.data() + mLo.bands, C, threads);
    const double t_hi = ms(t0);
    t0 = Clock::now();
    logspec(x16->data(), n16, T, mCq, feats.data() + mLo.bands + mHi.bands, C, threads);
    const double t_cq = ms(t0);
    if (dbg != nullptr) {
        dbg->resample_ms = t_rs;
        dbg->lo_ms = t_lo;
        dbg->hi_ms = t_hi;
        dbg->cq_ms = t_cq;
        dbg->gain = gain;
        dbg->x16 = *x16;
        auto block = [&](std::vector<float>& d, int off, int nb) {
            d.resize(static_cast<size_t>(T) * nb);
            for (int t = 0; t < T; t++) {
                std::copy_n(feats.data() + static_cast<size_t>(t) * C + off, nb, d.data() + static_cast<size_t>(t) * nb);
            }
        };
        block(dbg->lo, 0, mLo.bands);
        block(dbg->hi, mLo.bands, mHi.bands);
        block(dbg->cq, mLo.bands + mHi.bands, mCq.bands);
    }
    // features() returns float16; probabilities() casts back to float32
    std::vector<ggml_fp16_t> h(feats.size());
    ggml_fp32_to_fp16_row(feats.data(), h.data(), static_cast<int64_t>(feats.size()));
    ggml_fp16_to_fp32_row(h.data(), feats.data(), static_cast<int64_t>(feats.size()));
    return feats;
}

} // namespace dtt
