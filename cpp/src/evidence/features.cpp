#include "evidence/features.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

#include "common.hpp"
#include "evidence/hidden_store.hpp"

#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

namespace dtt::ev
{

std::vector<NoteSpan> noteSpans(const std::vector<TranscribedNote>& notes, int T)
{
    std::vector<NoteSpan> sp(notes.size());
    for (size_t i = 0; i < notes.size(); i++) {
        const TranscribedNote& n = notes[i];
        // np.round (half to even) of float64 divisions, then the clips of transcribe_with_evidence
        const long a = static_cast<long>(std::nearbyint(n.onset / HOP));
        const int A = static_cast<int>(std::clamp(a, 0L, static_cast<long>(T - 1)));
        const long b = static_cast<long>(std::nearbyint(n.offset / HOP));
        const int B = static_cast<int>(std::clamp(b, static_cast<long>(A + 1), static_cast<long>(T)));
        sp[i] = {n.pitch - PITCH_LO, A, B, n.slot};
    }
    return sp;
}

std::vector<float> mapEvidence(const Maps& maps, const std::vector<NoteSpan>& sp)
{
    constexpr int NF = N_MAP - 2;
    const int T = maps.T;
    std::vector<float> out(sp.size() * NF, 0.0f);
    auto c = [T](int t) { return std::clamp(t, 0, T - 1); };
    for (size_t i = 0; i < sp.size(); i++) {
        const auto [L, A, B, S] = sp[i];
        DTT_CHECK(S == 1 || S == 2, "note voice %d", S);
        const Head on = S == 1 ? ONSET : ONSET2, fr = S == 1 ? FRAME : FRAME2, off = S == 1 ? OFFSET : OFFSET2;
        const Head o_on = S == 1 ? ONSET2 : ONSET, o_fr = S == 1 ? FRAME2 : FRAME;
        const int span = std::max(B - A, 1);
        float* f = out.data() + i * NF;
        int k = 0;
        for (int d = -2; d <= 2; d++) {
            f[k++] = maps.at(on, c(A + d), L);
        }
        for (int j = 0; j < 8; j++) {
            f[k++] = maps.at(fr, c(A + (j * span) / 8), L);
        }
        for (int d : {1, 3, 6, 12}) {
            f[k++] = maps.at(fr, c(B - 1 + d), L);
        }
        float mx = 0.0f;
        for (int d = 1; d <= 5; d++) {
            const float v = A - d >= 0 ? maps.at(fr, c(A - d), L) : 0.0f;
            mx = d == 1 ? v : std::max(mx, v);
        }
        f[k++] = mx;
        // the highest offset probability over [A + 1, B + 12) and its relative position (first maximum)
        float om = 0.0f, opos = 0.0f;
        const int s0 = A + 1, s1 = std::min(B + 12, T);
        if (s1 > s0) {
            int j = 0;
            float best = maps.at(off, s0, L);
            for (int t = s0 + 1; t < s1; t++) {
                const float v = maps.at(off, t, L);
                if (v > best) {
                    best = v;
                    j = t - s0;
                }
            }
            om = best;
            opos = static_cast<float>(std::min(static_cast<double>(j + 1) / span, 2.0));
        }
        f[k++] = om;
        f[k++] = opos;
        f[k++] = maps.at(off, c(B - 1), L);
        float m3 = maps.at(o_on, c(A - 1), L);
        m3 = std::max(m3, maps.at(o_on, c(A), L));
        m3 = std::max(m3, maps.at(o_on, c(A + 1), L));
        f[k++] = m3;
        f[k++] = maps.at(o_fr, c(A), L);
        float s = maps.at(o_fr, c(A), L); // np.stack(...).mean(0): ((a0 + a1) + a2) + a3, then / 4
        for (int j = 1; j < 4; j++) {
            s += maps.at(o_fr, c(A + (j * span) / 4), L);
        }
        f[k++] = static_cast<float>(static_cast<double>(s) / 4.0);
        f[k++] = S == 2 ? 1.0f : 0.0f;
        for (int dl : {-1, 1, -12, 12, 19, 24}) {
            const int l2 = L + dl;
            const bool ok = l2 >= 0 && l2 < N_KEYS;
            const int lc = std::clamp(l2, 0, N_KEYS - 1);
            float v = maps.at(ONSET, c(A - 1), lc);
            v = std::max(v, maps.at(ONSET, c(A), lc));
            v = std::max(v, maps.at(ONSET, c(A + 1), lc));
            f[k++] = ok ? v : 0.0f;
        }
        DTT_CHECK(k == NF, "map feature count %d", k);
    }
    return out;
}

float npSumF32(const float* a, size_t n)
{
    // numpy/_core/src/umath/loops_utils.h.src pairwise_sum (float accumulators)
    if (n < 8) {
        float res = 0.0f;
        for (size_t i = 0; i < n; i++) {
            res += a[i];
        }
        return res;
    }
    if (n <= 128) {
        float r[8];
        for (int j = 0; j < 8; j++) {
            r[j] = a[j];
        }
        size_t i = 8;
        for (; i < n - (n % 8); i += 8) {
            for (int j = 0; j < 8; j++) {
                r[j] += a[i + j];
            }
        }
        float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) {
            res += a[i];
        }
        return res;
    }
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return npSumF32(a, n2) + npSumF32(a + n2, n - n2);
}

float npLogF32(float xin)
{
    // numpy/_core/src/umath/loops_exponent_log.dispatch.c.src simd_log_FLOAT (AVX2 + FMA3 / AVX512F; coefficients from
    // npy_simd_data.h): mantissa in [sqrt(1/2), sqrt(2)), a 5/5 rational approximation of log(1 + t), + exponent * ln 2
    if (!(xin > 0.0f) || !std::isfinite(xin)) {
        return std::log(xin);
    }
    uint32_t b;
    float x = xin, scale = 0.0f;
    if (xin < 1.17549435e-38f) { // denormals: times 2^100
        x = xin * 1.2676506e30f;
        scale = 100.0f;
    }
    std::memcpy(&b, &x, 4);
    float e = static_cast<float>(static_cast<int32_t>(b >> 23) - 0x7E) - scale;
    const uint32_t mb = (b & 0x7fffffu) | (126u << 23);
    float m;
    std::memcpy(&m, &mb, 4);
    if (m <= 0.707106781186547524400844362104849039f) {
        m = m + m;
        e = e - 1.0f;
    }
    const float t = m - 1.0f;
    float num = std::fma(2.589979117907922693523e-02f, t, 3.808837741388407920751e-01f);
    num = std::fma(num, t, 1.480000633576506585156e+00f);
    num = std::fma(num, t, 2.112677543073053063722e+00f);
    num = std::fma(num, t, 9.999999999999998702752e-01f);
    num = std::fma(num, t, 0.0f);
    float den = std::fma(5.875095403124574342950e-03f, t, 1.546476374983906719538e-01f);
    den = std::fma(den, t, 9.864942958519418960339e-01f);
    den = std::fma(den, t, 2.453006071784736363091e+00f);
    den = std::fma(den, t, 2.612677543073109236779e+00f);
    den = std::fma(den, t, 1.0f);
    return std::fma(e, 0.693147180559945309417232121458176568f, num / den);
}

void stdQ(const float* x, int8_t* q, float* log_sd)
{
    constexpr size_t n = C_H;
    // x.mean(1): add.reduce (identity 0 + pairwise), then true_divide by an np.intp count: a float64 loop cast to float32
    const float mu = static_cast<float>(static_cast<double>(0.0f + npSumF32(x, n)) / static_cast<double>(n));
    float d[n];
    for (size_t i = 0; i < n; i++) {
        d[i] = x[i] - mu;
        d[i] = d[i] * d[i];
    }
    const float var = static_cast<float>(static_cast<double>(0.0f + npSumF32(d, n)) / static_cast<double>(n));
    const float sd = std::sqrt(var) + 1e-6f;
    *log_sd = npLogF32(sd); // np.log on float32
    for (size_t i = 0; i < n; i++) {
        const float z = std::clamp((x[i] - mu) / sd, -3.96f, 3.96f);
        q[i] = static_cast<int8_t>(std::nearbyint(z * Q));
    }
}

namespace
{
    uint16_t toHalf(float f)
    {
        uint32_t x;
        std::memcpy(&x, &f, 4);
        const uint32_t sign = (x >> 16) & 0x8000u, ax = x & 0x7fffffffu;
        if (ax >= 0x7f800000u) {
            return static_cast<uint16_t>(sign | 0x7c00u | (ax > 0x7f800000u ? 0x200u : 0u));
        }
        if (ax >= 0x477ff000u) { // >= 65520 rounds to infinity
            return static_cast<uint16_t>(sign | 0x7c00u);
        }
        if (ax < 0x38800000u) { // below 2^-14: a subnormal half, units of 2^-24
            float a;
            std::memcpy(&a, &ax, 4);
            return static_cast<uint16_t>(sign | static_cast<uint32_t>(std::nearbyint(a * 16777216.0f)));
        }
        const uint32_t mant = ax & 0x7fffffu;
        uint32_t h = ((((ax >> 23) - 127 + 15) & 0x1fu) << 10) | (mant >> 13);
        const uint32_t rem = mant & 0x1fffu;
        if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) {
            h++;
        }
        return static_cast<uint16_t>(sign | h);
    }

    float fromHalf(uint16_t h)
    {
        const uint32_t s = static_cast<uint32_t>(h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
        uint32_t f;
        if (e == 0) {
            if (m == 0) {
                f = s;
            } else {
                int ee = -1;
                uint32_t mm = m;
                do {
                    ee++;
                    mm <<= 1;
                } while ((mm & 0x400u) == 0);
                f = s | ((127 - 15 - ee) << 23) | ((mm & 0x3ffu) << 13);
            }
        } else if (e == 31) {
            f = s | 0x7f800000u | (m << 13);
        } else {
            f = s | ((e + 127 - 15) << 23) | (m << 13);
        }
        float r;
        std::memcpy(&r, &f, 4);
        return r;
    }
} // namespace

float roundF16(float x)
{
    return fromHalf(toHalf(x));
}

Gathered gatherHidden(const HiddenStore& hs, const std::vector<NoteSpan>& sp)
{
    Gathered g;
    const size_t N = sp.size();
    g.at.assign(N * C_H, 0.0f);
    g.mean.assign(N * C_H, 0.0f);
    if (N == 0) {
        return g;
    }
    DTT_CHECK(hs.channels() > 0, "the evidence sink received no hidden state: run the transcriber (built with Taps.hidden) with "
                                 "tap mask 1 << TapSink::HIDDEN");
    DTT_CHECK(hs.complete(), "the transcriber pass has not finished (no final framesFinal call)");
    DTT_CHECK(hs.channels() == C_H, "hidden tap has %d channels, the evidence expects %d", hs.channels(), C_H);
    for (size_t i = 0; i < N; i++) {
        const auto [L, A, B, S] = sp[i];
        const float* h = hs.hidden(A, L);
        const float* ca = hs.prefix(A, L);
        const float* cb = hs.prefix(B, L);
        if (h == nullptr || ca == nullptr || cb == nullptr) {
            g.misses++;
            continue;
        }
        std::memcpy(g.at.data() + i * C_H, h, C_H * sizeof(float));
        const float n = static_cast<float>(std::max(B - A, 1));
        float* m = g.mean.data() + i * C_H;
        for (int c = 0; c < C_H; c++) {
            m[c] = (cb[c] - ca[c]) / n;
        }
    }
    return g;
}

std::vector<float> assembleEvidence(const std::vector<float>& mapf, const Gathered& g, std::vector<int8_t>* qout)
{
    constexpr int NF = N_MAP - 2;
    const size_t N = mapf.size() / NF;
    std::vector<float> ev(N * N_EV);
    if (qout != nullptr) {
        qout->assign(N * 2 * C_H, 0);
    }
    int8_t qa[C_H], qm[C_H];
    for (size_t i = 0; i < N; i++) {
        float la, lm;
        stdQ(g.at.data() + i * C_H, qa, &la);
        stdQ(g.mean.data() + i * C_H, qm, &lm);
        float* e = ev.data() + i * N_EV;
        // the map numbers and the two log-spreads go through float16
        for (int j = 0; j < NF; j++) {
            e[j] = roundF16(mapf[i * NF + j]);
        }
        e[NF] = roundF16(la);
        e[NF + 1] = roundF16(lm);
        for (int c = 0; c < C_H; c++) {
            e[N_MAP + c] = static_cast<float>(qa[c]) / Q;
            e[N_MAP + C_H + c] = static_cast<float>(qm[c]) / Q;
        }
        if (qout != nullptr) {
            std::memcpy(qout->data() + i * 2 * C_H, qa, C_H);
            std::memcpy(qout->data() + i * 2 * C_H + C_H, qm, C_H);
        }
    }
    return ev;
}

double npRound4(double x)
{
    return std::nearbyint(x * 10000.0) / 10000.0;
}

double pyRound4(double x)
{
    // Python rounds the exact binary value to 4 decimals (ties to even), then returns the nearest double: decide the
    // rounding from the exact sign of x * 1e4 - (n + 0.5) (fma keeps the product exact), then n / 1e4 rounds correctly.
    if (!std::isfinite(x)) {
        return x;
    }
    const double y = x * 10000.0;
    double n = std::floor(y);
    // y may have rounded across an integer: settle n so that n <= x * 1e4 < n + 1 exactly
    if (std::fma(x, 10000.0, -n) < 0.0) {
        n -= 1.0;
    } else if (std::fma(x, 10000.0, -(n + 1.0)) >= 0.0) {
        n += 1.0;
    }
    const double r = std::fma(x, 10000.0, -(n + 0.5));
    if (r > 0.0 || (r == 0.0 && std::fmod(n, 2.0) != 0.0)) {
        n += 1.0;
    }
    return n / 10000.0;
}

void inputNotes(const std::vector<TranscribedNote>& notes, std::vector<InNote>& k, std::vector<int>& order)
{
    const size_t n = notes.size();
    std::vector<InNote> raw(n);
    for (size_t i = 0; i < n; i++) {
        const TranscribedNote& t = notes[i];
        const double on = npRound4(t.onset);
        const double off = npRound4(t.offset);
        // conf_vel: (conf - ONSET_TH) / (1 - ONSET_TH) in float64, np.round(1 + 126 x), clipped to 1..127
        const double x = (static_cast<double>(t.conf) - 0.25) / (1.0 - 0.25);
        const double v = std::clamp(std::nearbyint(1.0 + 126.0 * x), 1.0, 127.0);
        raw[i] = {t.pitch, static_cast<int>(v), static_cast<float>(on), static_cast<float>(std::max(off, on + 1e-3))};
    }
    // np.lexsort((pitch, inst, on)): by on, then pitch (inst is 0), stable
    order.resize(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        if (raw[a].on != raw[b].on) {
            return raw[a].on < raw[b].on;
        }
        return raw[a].pitch < raw[b].pitch;
    });
    k.resize(n);
    for (size_t i = 0; i < n; i++) {
        k[i] = raw[order[i]];
    }
}

} // namespace dtt::ev
