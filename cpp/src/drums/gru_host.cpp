#include "drums/gru_host.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>

#include <ggml.h>

#include "common.hpp"
#include "model_file.hpp"

// The step loops vectorise; on x86-64 Linux they are also built for AVX2 + FMA and picked at run time.
#if defined(__GNUC__) && defined(__x86_64__) && defined(__linux__) && !defined(__clang__)
#define DTT_MULTIVERSION __attribute__((target_clones("default", "arch=x86-64-v3")))
#else
#define DTT_MULTIVERSION
#endif

namespace dtt
{

DrumGru::DrumGru(const ModelFile& mf, const std::string& name)
{
    ggml_tensor* w = mf.tensor(name + ".hh.w"); // ne [H, 3H, 2]
    DTT_CHECK(w->ne[1] == 3 * w->ne[0] && w->ne[2] == 2, "%s.hh.w is not (2, 3H, H)", name.c_str());
    mH = static_cast<int>(w->ne[0]);
    const int H = mH, G = 3 * H;
    const auto wh = mf.hostF32(name + ".hh.w");
    const auto bh = mf.hostF32(name + ".hh.b");
    DTT_CHECK(static_cast<int>(bh.size()) == 2 * G, "%s.hh.b size", name.c_str());
    for (int d = 0; d < 2; d++) {
        mWt[d].resize(static_cast<size_t>(H) * G);
        for (int j = 0; j < G; j++) {
            for (int k = 0; k < H; k++) {
                mWt[d][static_cast<size_t>(k) * G + j] = wh[(static_cast<size_t>(d) * G + j) * H + k];
            }
        }
        mB[d].assign(bh.begin() + d * G, bh.begin() + (d + 1) * G);
    }
}

namespace
{
    /**
     * exp(x) for x in float32's range, branch free so the gate loops vectorise: x = n ln2 + r, |r| <= ln2 / 2, a degree-6
     * polynomial for exp(r) (Cephes expf, ~1 ulp), times 2^n built in the exponent bits.
     */
    inline float expApprox(float x)
    {
        x = std::min(std::max(x, -87.3f), 88.3f);
        const float n = std::floor(x * 1.44269504088896341f + 0.5f);
        float r = x - n * 0.693359375f;
        r = r - n * -2.12194440e-4f;
        float p = 1.9875691500e-4f;
        p = p * r + 1.3981999507e-3f;
        p = p * r + 8.3334519073e-3f;
        p = p * r + 4.1665795894e-2f;
        p = p * r + 1.6666665459e-1f;
        p = p * r + 5.0000001201e-1f;
        p = p * r * r + r + 1.0f;
        const int32_t e = (static_cast<int32_t>(n) + 127) << 23;
        float s;
        std::memcpy(&s, &e, sizeof(s));
        return p * s;
    }

    DTT_MULTIVERSION
    void gruSteps(int H, const float* wt, const float* bh, int dir, const float* gi, int gi_stride, int N, float* out,
                  int out_stride)
    {
        const int G = 3 * H;
        std::vector<float> h(H, 0.0f), acc(G), r(H), z(H), n(H);
        for (int s = 0; s < N; s++) {
            const int t = dir == 0 ? s : N - 1 - s;
            // W_hh h: rows of the transposed matrix, accumulated over k (vectorises over j)
            for (int j = 0; j < G; j++) {
                acc[j] = 0.0f;
            }
            for (int k = 0; k < H; k++) {
                const float hk = h[k];
                const float* w = wt + static_cast<size_t>(k) * G;
                for (int j = 0; j < G; j++) {
                    acc[j] += w[j] * hk;
                }
            }
            const float* g = gi + static_cast<size_t>(t) * gi_stride;
            float* o = out + static_cast<size_t>(t) * out_stride;
            // the gates, one vectorisable loop each: r, z = sigmoid; n = tanh = 2 sigmoid(2x) - 1
            for (int j = 0; j < H; j++) {
                r[j] = 1.0f / (1.0f + expApprox(-((acc[j] + bh[j]) + g[j])));
            }
            for (int j = 0; j < H; j++) {
                z[j] = 1.0f / (1.0f + expApprox(-((acc[H + j] + bh[H + j]) + g[H + j])));
            }
            for (int j = 0; j < H; j++) {
                const float x = g[2 * H + j] + (acc[2 * H + j] + bh[2 * H + j]) * r[j];
                n[j] = 2.0f / (1.0f + expApprox(-2.0f * x)) - 1.0f;
            }
            for (int j = 0; j < H; j++) {
                h[j] = (h[j] - n[j]) * z[j] + n[j];
                o[j] = h[j];
            }
        }
    }
} // namespace

DrumBandGru::DrumBandGru(const ModelFile& mf, const std::string& p)
{
    const char* dn[2] = {"f", "r"};
    for (int d = 0; d < 2; d++) {
        ggml_tensor* w = mf.tensor(p + ".bgru.hh" + dn[d] + ".w"); // ne [m, 3m]
        mM = static_cast<int>(w->ne[0]);
        const int m = mM, G = 3 * m;
        const auto wh = mf.hostF32(p + ".bgru.hh" + dn[d] + ".w");
        mWt[d].resize(static_cast<size_t>(m) * G);
        for (int j = 0; j < G; j++) {
            for (int k = 0; k < m; k++) {
                mWt[d][static_cast<size_t>(k) * G + j] = wh[static_cast<size_t>(j) * m + k];
            }
        }
        mBn[d] = mf.hostF32(p + ".bgru.hh" + dn[d] + ".bn");
    }
    const auto pw = mf.hostF32(p + ".bgru.proj.w"); // (W, 4m)
    mProjB = mf.hostF32(p + ".bgru.proj.b");
    mW = static_cast<int>(mProjB.size());
    const int I = 4 * mM;
    DTT_CHECK(static_cast<int>(pw.size()) == mW * I, "%s.bgru.proj.w is not (W, 4m)", p.c_str());
    mProjT.resize(static_cast<size_t>(I) * mW);
    for (int o = 0; o < mW; o++) {
        for (int i = 0; i < I; i++) {
            mProjT[static_cast<size_t>(i) * mW + o] = pw[static_cast<size_t>(o) * I + i];
        }
    }
}

namespace
{
    DTT_MULTIVERSION
    void bandFrames(int m, int F, int W, const std::vector<float>* wt, const std::vector<float>* bn, const float* projT,
                    const float* projB, const float* gi, int n0, int n1, float* out)
    {
        const int G = 3 * m;
        std::vector<float> h(m), acc(G), r(m), z(m), nn(m), feat(4 * m), y(W);
        for (int t = n0; t < n1; t++) {
            const float* g0 = gi + static_cast<size_t>(t) * F * 2 * G;
            for (int d = 0; d < 2; d++) {
                const float* w = wt[d].data();
                const float* b = bn[d].data();
                std::fill(h.begin(), h.end(), 0.0f);
                float* sum = feat.data() + 2 * m + d * m;
                std::fill(sum, sum + m, 0.0f);
                for (int s = 0; s < F; s++) {
                    const int f = d == 0 ? s : F - 1 - s;
                    const float* g = g0 + static_cast<size_t>(f) * 2 * G + d * G;
                    for (int j = 0; j < G; j++) {
                        acc[j] = 0.0f;
                    }
                    for (int k = 0; k < m; k++) {
                        const float hk = h[k];
                        const float* wr = w + static_cast<size_t>(k) * G;
                        for (int j = 0; j < G; j++) {
                            acc[j] += wr[j] * hk;
                        }
                    }
                    // the graph version's ops: r, z = sigmoid(gi + W h) (b_hr, b_hz in gi); n = tanh(gi_n + r (W_n h + b_hn))
                    for (int j = 0; j < m; j++) {
                        r[j] = 1.0f / (1.0f + expApprox(-(g[j] + acc[j])));
                    }
                    for (int j = 0; j < m; j++) {
                        z[j] = 1.0f / (1.0f + expApprox(-(g[m + j] + acc[m + j])));
                    }
                    for (int j = 0; j < m; j++) {
                        const float x = g[2 * m + j] + r[j] * (acc[2 * m + j] + b[j]);
                        nn[j] = 2.0f / (1.0f + expApprox(-2.0f * x)) - 1.0f;
                    }
                    for (int j = 0; j < m; j++) {
                        h[j] = nn[j] + z[j] * (h[j] - nn[j]);
                        sum[j] += h[j];
                    }
                }
                std::copy(h.begin(), h.end(), feat.begin() + d * m); // the final state: torch's hn[d]
            }
            const float inv = 1.0f / static_cast<float>(F);
            for (int j = 2 * m; j < 4 * m; j++) {
                feat[j] *= inv; // the mean over the bands
            }
            for (int o = 0; o < W; o++) {
                y[o] = 0.0f;
            }
            for (int i = 0; i < 4 * m; i++) {
                const float fi = feat[i];
                const float* pr = projT + static_cast<size_t>(i) * W;
                for (int o = 0; o < W; o++) {
                    y[o] += pr[o] * fi;
                }
            }
            float* o = out + static_cast<size_t>(t) * W;
            for (int k = 0; k < W; k++) {
                o[k] = y[k] + projB[k];
            }
        }
    }
} // namespace

void DrumBandGru::run(const float* gi, int N, int F, float* out, int threads) const
{
    threads = std::max(1, std::min(threads, N / 64 + 1));
    std::vector<std::thread> pool;
    const int chunk = (N + threads - 1) / threads;
    for (int i = 0; i < threads; i++) {
        const int a = i * chunk, b = std::min(N, a + chunk);
        if (a < b) {
            pool.emplace_back([this, gi, F, a, b, out] { bandFrames(mM, F, mW, mWt, mBn, mProjT.data(), mProjB.data(), gi, a, b, out); });
        }
    }
    for (auto& t : pool) {
        t.join();
    }
}

void DrumGru::run(int dir, const float* gi, int gi_stride, int N, float* out, int out_stride) const
{
    gruSteps(mH, mWt[dir].data(), mB[dir].data(), dir, gi, gi_stride, N, out, out_stride);
}

void DrumGru::runBoth(const float* gi, int N, float* out) const
{
    const int G = 3 * mH;
    std::thread rev([&] { run(1, gi + G, 2 * G, N, out + mH, 2 * mH); });
    run(0, gi, 2 * G, N, out, 2 * mH);
    rev.join();
}

} // namespace dtt
