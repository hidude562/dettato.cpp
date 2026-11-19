#include "grouper/front.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

#include "common.hpp"
#include "model_file.hpp"

namespace dtt
{

namespace
{
    constexpr int HD_K[] = {-2, 0, 2, 4, 8, 12, 16, 24, 32, 48}; // attack frame offsets (10 ms)
    constexpr int N_HDK = 10;
    constexpr float HD_SUS[] = {0.25f, 0.5f, 0.75f, 0.95f};       // sustain points (fractions of the note)

    /** torch float32 .round().long() (half to even). */
    inline long roundL(float v)
    {
        return static_cast<long>(std::nearbyint(v));
    }

    /** front.note_frames for one note row. */
    inline void noteFrames(const float* nr, int T, int& f_on, int& f_off, int& lane)
    {
        const long a = std::clamp(roundL((nr[0] + 0.5f) * 100.0f), 0L, static_cast<long>(T - 1));
        const long b = std::max(std::min(roundL((nr[1] + 0.5f) * 100.0f), static_cast<long>(T)), a + 1);
        f_on = static_cast<int>(a);
        f_off = static_cast<int>(b);
        lane = static_cast<int>(std::clamp(roundL(nr[2]) - 21, 0L, 87L));
    }
} // namespace

float roundHalf(float x)
{
    uint32_t u;
    std::memcpy(&u, &x, 4);
    const uint32_t sign = u & 0x80000000u;
    uint32_t a = u & 0x7fffffffu;
    if (a >= 0x7f800000u) {
        return x; // inf / nan
    }
    float r;
    if (a >= 0x477ff000u) { // >= 65520: overflows to inf
        a = 0x7f800000u;
        std::memcpy(&r, &a, 4);
    } else if (a < 0x38800000u) { // below the smallest normal half: multiples of 2^-24
        const float k = std::nearbyint(std::fabs(x) * 16777216.0f);
        r = k * (1.0f / 16777216.0f);
    } else {
        a += 0xfffu + ((a >> 13) & 1u);
        a &= ~0x1fffu;
        std::memcpy(&r, &a, 4);
    }
    uint32_t v;
    std::memcpy(&v, &r, 4);
    v |= sign;
    std::memcpy(&r, &v, 4);
    return r;
}

GroupFront::GroupFront(const ModelFile& mf) : mHcqt(mf, "gr")
{
    mShiftLo = static_cast<int>(mf.u32("gr.shift_lo"));
    mH = static_cast<int>(mf.u32("gr.hd"));
    mSus = static_cast<int>(mf.u32("gr.hd_sus"));
    DTT_CHECK(mSus == 4, "HDemix with %d sustain points is not ported (the model uses 4)", mSus);
    for (int h = 1; h <= mH; h++) {
        mHdShift.push_back(static_cast<int>(std::nearbyint(36.0 * std::log2(static_cast<double>(h)))));
    }
    mPi = mf.hostF32("gr.hd.pi");
    DTT_CHECK(static_cast<int>(mPi.size()) == mH, "gr.hd.pi has %zu values, expected %d", mPi.size(), mH);
    // the kernel s (1 - s)^j in float64 from the decimal constant (the metadata holds it as float32)
    mS = std::stod(format("%.7g", static_cast<double>(mf.f32("gr.pcen.s"))));
    mAlpha = mf.f32("gr.pcen.alpha");
    mDelta = mf.f32("gr.pcen.delta");
    mR = mf.f32("gr.pcen.r");
    mEps = mf.f32("gr.pcen.eps");
    mK = static_cast<int>(mf.u32("gr.pcen.K"));
    DTT_CHECK(mR == 0.5, "PCEN with r != 0.5 is not ported");
}

int GroupFront::hdFeatures() const
{
    return mH * (N_HDK + mSus) + mH;
}

void GroupFront::compute(const float* audio, size_t n, float* X, float* P) const
{
    const int T = static_cast<int>(n / HOP);
    mHcqt.compute(audio, n, X, P);
    pcen(X, channels(), T, bins());
    standardise(X, channels(), T, bins());
}

void GroupFront::pcen(float* X, int C, int T, int F) const
{
    // the model's constants as float32 (torch casts the Python floats to the tensor's dtype)
    const float eps = static_cast<float>(mEps), alpha = static_cast<float>(mAlpha), delta = static_cast<float>(mDelta);
    const float dr = static_cast<float>(std::sqrt(static_cast<double>(static_cast<float>(mDelta)))); // delta ** r, r = 0.5
    const double s = mS, a = 1.0 - mS;
    const double aK = s * std::pow(a, mK); // weight of the tap that leaves the window
    std::vector<float> E(T);
    std::vector<double> M(T);
    for (int c = 0; c < C; c++) {
        float* xc = X + static_cast<size_t>(c) * T * F;
        for (int f = 0; f < F; f++) {
            for (int t = 0; t < T; t++) {
                E[t] = std::exp(xc[static_cast<size_t>(t) * F + f]);
            }
            // M[t] = sum_{j < min(t + 1, K)} s a^j E[t - j]
            double m = 0.0;
            for (int t = 0; t < T; t++) {
                m = a * m + s * E[t];
                if (t >= mK) {
                    m -= aK * E[t - mK];
                }
                M[t] = m;
            }
            for (int t = 0; t < T; t++) {
                const float Mf = std::max(static_cast<float>(M[t]), 0.0f);
                const float g = std::pow(eps + Mf, alpha);
                const float o = std::sqrt(E[t] / g + delta) - dr;
                xc[static_cast<size_t>(t) * F + f] = std::log(o + 1e-4f);
            }
        }
    }
}

void GroupFront::standardise(float* X, int C, int T, int F)
{
    const size_t n = static_cast<size_t>(T) * F;
    for (int c = 0; c < C; c++) {
        float* x = X + static_cast<size_t>(c) * n;
        float mx = -INFINITY;
        for (size_t i = 0; i < n; i++) {
            mx = std::max(mx, x[i]);
        }
        const float lim = mx - 12.0f;
        double cnt = 0, s1 = 0;
        for (size_t i = 0; i < n; i++) {
            if (x[i] > lim) {
                cnt += 1;
                s1 += x[i];
            }
        }
        const float cntf = std::max(static_cast<float>(cnt), 1.0f);
        const float mu = static_cast<float>(s1) / cntf;
        double s2 = 0;
        for (size_t i = 0; i < n; i++) {
            if (x[i] > lim) {
                const float d = x[i] - mu;
                s2 += static_cast<double>(d * d);
            }
        }
        const float sd = std::max(std::sqrt(static_cast<float>(s2) / cntf), 0.5f);
        for (size_t i = 0; i < n; i++) {
            x[i] = (x[i] - mu) / sd;
        }
    }
}

void GroupFront::hdemixRaw(const float* P, int T, const float* notes, const uint8_t* mask, int N, float* out) const
{
    const int NE = extBins(), H = mH, Kt = N_HDK + mSus, T1 = T + 1;
    const int nf = hdFeatures();
    std::vector<int> f_on(N), f_off(N), bins(static_cast<size_t>(N) * H);
    std::vector<float> valid(static_cast<size_t>(N) * H), w(static_cast<size_t>(N) * H);
    for (int i = 0; i < N; i++) {
        int lane;
        noteFrames(notes + static_cast<size_t>(i) * NOTE_COLS, T, f_on[i], f_off[i], lane);
        for (int h = 0; h < H; h++) {
            const int b = 3 * lane + mShiftLo + mHdShift[h];
            valid[i * H + h] = b < NE ? 1.0f : 0.0f;
            bins[i * H + h] = std::clamp(b, 0, NE - 1);
            w[i * H + h] = mPi[h] * ((mask[i] ? 1.0f : 0.0f) * valid[i * H + h]);
        }
    }
    // occupancy: every note's claim scattered at its onset and removed at its end (in note order, as scatter_add), then a
    // cumulative sum over time (float64 accumulation, float32 values, as torch.cumsum on the CPU)
    std::vector<float> O(static_cast<size_t>(NE) * T1, 0.0f);
    for (int i = 0; i < N * H; i++) {
        O[static_cast<size_t>(bins[i]) * T1 + f_on[i / H]] += w[i];
    }
    for (int i = 0; i < N * H; i++) {
        O[static_cast<size_t>(bins[i]) * T1 + f_off[i / H]] += -w[i];
    }
    std::vector<char> used(NE, 0);
    for (int i = 0; i < N * H; i++) {
        used[bins[i]] = 1;
    }
    for (int b = 0; b < NE; b++) {
        if (!used[b]) {
            continue;
        }
        float* row = O.data() + static_cast<size_t>(b) * T1;
        double acc = 0;
        for (int t = 0; t < T1; t++) {
            acc += row[t];
            row[t] = static_cast<float>(acc);
        }
    }
    std::vector<int> tk(Kt);
    std::vector<float> act(Kt), lE(static_cast<size_t>(H) * Kt);
    for (int i = 0; i < N; i++) {
        const float dur = static_cast<float>(f_off[i] - f_on[i]);
        for (int k = 0; k < N_HDK; k++) {
            tk[k] = std::clamp(f_on[i] + HD_K[k], 0, T - 1);
        }
        for (int k = 0; k < mSus; k++) {
            tk[N_HDK + k] = std::clamp(f_on[i] + static_cast<int>(static_cast<long>(dur * HD_SUS[k])), 0, T - 1);
        }
        for (int k = 0; k < Kt; k++) {
            act[k] = (tk[k] >= f_on[i] && tk[k] < f_off[i]) ? 1.0f : 0.0f;
        }
        float mx = -INFINITY;
        for (int h = 0; h < H; h++) {
            const int b = bins[i * H + h];
            const float o4 = mPi[h];
            for (int k = 0; k < Kt; k++) {
                const float pg = P[static_cast<size_t>(b) * T + tk[k]];
                const float og = O[static_cast<size_t>(b) * T1 + tk[k]];
                const float share = o4 / std::max(og + o4 * (1.0f - act[k]), 1e-6f);
                const float v = std::log(pg * std::min(share, 1.0f) + 1e-7f) * valid[i * H + h];
                lE[h * Kt + k] = v;
                mx = std::max(mx, v);
            }
        }
        float* o = out + static_cast<size_t>(i) * nf;
        for (int j = 0; j < H * Kt; j++) {
            o[j] = std::max(lE[j] - mx, -20.0f) / 10.0f;
        }
        for (int h = 0; h < H; h++) {
            o[H * Kt + h] = valid[i * H + h];
        }
    }
}

void GroupFront::profRaw(const float* X, int T, const float* notes, int N, float* out) const
{
    const int C = channels(), F = bins();
    // cumulative sums over time of X and X^2 at the bins the notes read (float64 accumulation, float32 values)
    std::unordered_map<int, std::vector<float>> cs; // bin -> (C, 2, T + 1)
    auto column = [&](int fb) -> const std::vector<float>& {
        auto it = cs.find(fb);
        if (it != cs.end()) {
            return it->second;
        }
        std::vector<float> v(static_cast<size_t>(C) * 2 * (T + 1));
        for (int c = 0; c < C; c++) {
            const float* x = X + static_cast<size_t>(c) * T * F + fb;
            float* c1 = v.data() + static_cast<size_t>(c) * 2 * (T + 1);
            float* c2 = c1 + (T + 1);
            double a1 = 0, a2 = 0;
            c1[0] = c2[0] = 0.0f;
            for (int t = 0; t < T; t++) {
                const float xv = x[static_cast<size_t>(t) * F];
                a1 += xv;
                a2 += static_cast<double>(xv * xv);
                c1[t + 1] = static_cast<float>(a1);
                c2[t + 1] = static_cast<float>(a2);
            }
        }
        return cs.emplace(fb, std::move(v)).first->second;
    };
    for (int i = 0; i < N; i++) {
        int f_on, f_off, lane;
        noteFrames(notes + static_cast<size_t>(i) * NOTE_COLS, T, f_on, f_off, lane);
        const float Ln = static_cast<float>(std::max(f_off - f_on, 1));
        float* o = out + static_cast<size_t>(i) * 4 * C;
        for (int off = 0; off < 2; off++) {
            const int fbb = std::clamp(3 * lane + off, 0, F - 1);
            const std::vector<float>& v = column(fbb);
            for (int c = 0; c < C; c++) {
                const float* c1 = v.data() + static_cast<size_t>(c) * 2 * (T + 1);
                const float* c2 = c1 + (T + 1);
                const float m1 = (c1[f_off] - c1[f_on]) / Ln;
                const float m2 = (c2[f_off] - c2[f_on]) / Ln;
                o[(2 * off) * C + c] = m1;
                o[(2 * off + 1) * C + c] = std::sqrt(std::max(m2 - m1 * m1, 0.0f) + 1e-4f);
            }
        }
    }
}

void GroupFront::rasters(const float* notes, const uint8_t* mask, int N, int Tq, int P, float fps, float* R, int* on,
                         int* off, int* lane)
{
    std::vector<float> R4(static_cast<size_t>(4) * (Tq + 1) * P, 0.0f);
    const size_t plane = static_cast<size_t>(Tq + 1) * P;
    std::vector<int> io(N), ie(N);
    for (int i = 0; i < N; i++) {
        const float* nr = notes + static_cast<size_t>(i) * NOTE_COLS;
        const long a = std::clamp(static_cast<long>((nr[0] + 0.5f) * fps), 0L, static_cast<long>(Tq - 1));
        const long b = std::max(std::min(static_cast<long>((nr[1] + 0.5f) * fps), static_cast<long>(Tq)), a + 1);
        on[i] = static_cast<int>(a);
        off[i] = static_cast<int>(b);
        lane[i] = static_cast<int>(std::clamp(roundL(nr[2]) - 21, 0L, static_cast<long>(P - 1)));
        io[i] = on[i] * P + lane[i];
        ie[i] = off[i] * P + lane[i];
    }
    for (int i = 0; i < N; i++) {
        R4[io[i]] += mask[i] ? 1.0f : 0.0f;
    }
    for (int i = 0; i < N; i++) {
        R4[plane + io[i]] += mask[i] ? 1.0f : 0.0f;
    }
    for (int i = 0; i < N; i++) {
        R4[plane + ie[i]] += mask[i] ? -1.0f : -0.0f;
    }
    for (int i = 0; i < N; i++) {
        const float* nr = notes + static_cast<size_t>(i) * NOTE_COLS;
        const float conf = nr[3] > 0.5f ? nr[5] : nr[4];
        R4[2 * plane + io[i]] += (mask[i] ? 1.0f : 0.0f) * conf;
    }
    for (int i = 0; i < N; i++) {
        const float* nr = notes + static_cast<size_t>(i) * NOTE_COLS;
        R4[3 * plane + io[i]] += (mask[i] ? 1.0f : 0.0f) * (nr[3] > 0.5f ? 1.0f : 0.0f);
    }
    for (int l = 0; l < P; l++) { // sustain = cumulative sum over time
        double acc = 0;
        for (int t = 0; t <= Tq; t++) {
            float& v = R4[plane + static_cast<size_t>(t) * P + l];
            acc += v;
            v = static_cast<float>(acc);
        }
    }
    for (int t = 0; t < Tq; t++) {
        for (int l = 0; l < P; l++) {
            float* r = R + (static_cast<size_t>(t) * P + l) * 4;
            for (int c = 0; c < 4; c++) {
                r[c] = R4[c * plane + static_cast<size_t>(t) * P + l];
            }
        }
    }
}

} // namespace dtt
