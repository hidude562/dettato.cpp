#include "evidence/repair.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>

#include "common.hpp"
#include "evidence/net.hpp"

#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

namespace dtt::ev
{

namespace
{
    constexpr float L_WIN = 8.0f;
    constexpr double DEDUP_S = 0.03;

    /** torch.sigmoid's scalar path (libm exp). */
    float sigmoidScalar(float x)
    {
        return 1.0f / (1.0f + std::exp(-x));
    }
    /** torch.sigmoid's vector path (SLEEF exp). */
    float sigmoidVec(float x)
    {
        return 1.0f / (1.0f + sleefExpF(-x));
    }

    int argmaxRow(const float* x, int n)
    {
        int b = 0;
        for (int i = 1; i < n; i++) {
            if (x[i] > x[b]) {
                b = i;
            }
        }
        return b;
    }
} // namespace

// sleef src/libm/sleefsimdsp.c xexpf with vmla = fused multiply-add (helperavx2.h); constants cast from the same literals.
// FMA(x, y, z) must be a correctly rounded fused multiply-add: a hardware instruction or std::fma give identical results.
#define DTT_SLEEF_EXPF_BODY(FMA)                                                                                              \
    const int q = static_cast<int>(                                                                                          \
        std::nearbyint(d * static_cast<float>(1.442695040888963407359924681001892137426645954152985934135449406931)));       \
    float s = FMA(static_cast<float>(q), -0.693145751953125f, d);                                                            \
    s = FMA(static_cast<float>(q), -1.428606765330187045e-06f, s);                                                           \
    float u = static_cast<float>(0.000198527617612853646278381);                                                             \
    u = FMA(u, s, static_cast<float>(0.00139304355252534151077271));                                                         \
    u = FMA(u, s, static_cast<float>(0.00833336077630519866943359));                                                         \
    u = FMA(u, s, static_cast<float>(0.0416664853692054748535156));                                                          \
    u = FMA(u, s, static_cast<float>(0.166666671633720397949219));                                                           \
    u = FMA(u, s, 0.5f);                                                                                                     \
    const float s2 = s * s;                                                                                                  \
    u = 1.0f + FMA(s2, u, s);                                                                                                \
    uint32_t b1 = static_cast<uint32_t>((q >> 1) + 0x7f) << 23, b2 = static_cast<uint32_t>(q - (q >> 1) + 0x7f) << 23;       \
    float p1, p2;                                                                                                            \
    std::memcpy(&p1, &b1, 4);                                                                                                \
    std::memcpy(&p2, &b2, 4);                                                                                                \
    u = u * p1 * p2;                                                                                                         \
    if (d < -104.0f) {                                                                                                       \
        u = 0.0f;                                                                                                            \
    }                                                                                                                        \
    if (100.0f < d) {                                                                                                        \
        u = std::numeric_limits<float>::infinity();                                                                          \
    }                                                                                                                        \
    return u;

namespace
{
    float sleefExpSoft(float d)
    {
        DTT_SLEEF_EXPF_BODY(std::fma)
    }
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__)) && !defined(__FMA__)
#define DTT_FMA_DISPATCH 1
    // the library is built for baseline x86-64, where std::fma is a slow libm call (~50 M of them per song): a hardware
    // FMA version, picked at run time
    __attribute__((target("fma"))) float sleefExpHw(float d)
    {
        DTT_SLEEF_EXPF_BODY(__builtin_fmaf)
    }
#endif
} // namespace

float sleefExpF(float d)
{
#if defined(DTT_FMA_DISPATCH)
    static const bool hw = __builtin_cpu_supports("fma");
    return hw ? sleefExpHw(d) : sleefExpSoft(d);
#else
    return sleefExpSoft(d);
#endif
}

void softmaxRow(const float* x, int n, float* out)
{
    float mx = x[0];
    for (int i = 1; i < n; i++) {
        mx = std::max(mx, x[i]);
    }
    for (int i = 0; i < n; i++) {
        out[i] = sleefExpF(x[i] - mx);
    }
    // vec::reduce_all with 8 lanes: lane sums, the partial tail into the first lanes, then (l0+l4 + l2+l6) + (l1+l5 + l3+l7)
    float s;
    if (n < 8) {
        s = out[0];
        for (int i = 1; i < n; i++) {
            s += out[i];
        }
    } else {
        float acc[8];
        for (int j = 0; j < 8; j++) {
            acc[j] = out[j];
        }
        int d = 8;
        for (; d < n - (n % 8); d += 8) {
            for (int j = 0; j < 8; j++) {
                acc[j] += out[d + j];
            }
        }
        for (int j = 0; j < n - d; j++) {
            acc[j] += out[d + j];
        }
        float b[4];
        for (int j = 0; j < 4; j++) {
            b[j] = acc[j] + acc[j + 4];
        }
        s = (b[0] + b[2]) + (b[1] + b[3]);
    }
    const float r = 1.0f / s;
    for (int i = 0; i < n; i++) {
        out[i] *= r;
    }
}

int boxArgmax(const float* p, int n, int h)
{
    int best = 0;
    float bv = 0.0f;
    for (int i = 0; i < n; i++) {
        float s = 0.0f;
        for (int j = std::max(0, i - h); j <= std::min(n - 1, i + h); j++) {
            s += p[j];
        }
        if (i == 0 || s > bv) {
            bv = s;
            best = i;
        }
    }
    return best;
}

void candidates(const float* logits, int n, const int* pitch, const float* on, const float* dur, const RepairerNet& net,
                std::vector<Cand>& out)
{
    const std::vector<float>&dur_cent = net.durCent(), &dur_W = net.durW(), &dr_exp = net.drExp();
    // torch.sigmoid of the contiguous (1, n) keep logits runs vectorised (SLEEF exp) over 16-element blocks and scalar (libm)
    // on the n % 16 left; the insertion probabilities are a strided view, so all scalar
    const int n_vec = n / 16 * 16;
    using N = RepairerNet;
    constexpr float DON_STEP = 0.005f, DON_MAX = 0.5f, INS_T_STEP = 0.01f, INS_T_MIN = -1.0f;
    constexpr int INS_P_MAX = 24;
    const int ND = N::N_DUR;
    out.assign(static_cast<size_t>(n) * (1 + N::N_SLOT), Cand{});
    std::vector<float> p(N::N_INS_T), wd(ND);
    std::vector<std::vector<int>> wcols(ND);
    for (int a = 0; a < ND; a++) {
        for (int j = 0; j < ND; j++) {
            DTT_CHECK(dur_W[static_cast<size_t>(a) * ND + j] == 0.0f || dur_W[static_cast<size_t>(a) * ND + j] == 1.0f, "dur_W not 0/1");
            if (dur_W[static_cast<size_t>(a) * ND + j] != 0.0f) {
                wcols[j].push_back(a);
            }
        }
    }
    for (int i = 0; i < n; i++) {
        const float* o = logits + static_cast<size_t>(i) * N::N_OUT;
        // the kept note: P(keep), the onset correction (box of +-50 ms), the duration ratio (box of +-3 bins)
        const float pk = i < n_vec ? sigmoidVec(o[0]) : sigmoidScalar(o[0]);
        softmaxRow(o + 1, N::N_DON, p.data());
        const int don_b = boxArgmax(p.data(), N::N_DON, 10);
        softmaxRow(o + 1 + N::N_DON, N::N_DR, p.data());
        const int rb = boxArgmax(p.data(), N::N_DR, 3);
        const float dur_k = std::max(dur[i], 1e-4f) * dr_exp[rb]; // torch.exp(rb * DR_STEP - DR_MAX)
        const float on_k = on[i] + (static_cast<float>(don_b) * DON_STEP - DON_MAX);
        out[i] = {pitch[i], on_k, on_k + dur_k, pk, 0};
        for (int s = 0; s < N::N_SLOT; s++) {
            const float* ins = o + 1 + N::N_DON + N::N_DR + s * N::N_INS;
            const float pe = sigmoidScalar(ins[0]);
            const int ip = argmaxRow(ins + 1, N::N_INS_P) - INS_P_MAX;
            softmaxRow(ins + 1 + N::N_INS_P, N::N_INS_T, p.data());
            const int it = boxArgmax(p.data(), N::N_INS_T, 5);
            // (softmax @ W).argmax: W[i][j] in {0, 1}, accumulated over i in order (sgemm's single-accumulator order)
            softmaxRow(ins + 1 + N::N_INS_P + N::N_INS_T, ND, p.data());
            for (int j = 0; j < ND; j++) {
                float acc = 0.0f;
                for (int a : wcols[j]) { // the rows with W[a][j] = 1, ascending: p[a] * 1 added in sgemm's order
                    acc += p[a];
                }
                wd[j] = acc;
            }
            const int idd = argmaxRow(wd.data(), ND);
            const float on_i = on[i] + (static_cast<float>(it) * INS_T_STEP + INS_T_MIN);
            const float dur_i = dur_cent[idd];
            out[static_cast<size_t>(n) + 2 * i + s] = {std::clamp(pitch[i] + ip, 0, 127), on_i, on_i + dur_i, pe, 1};
        }
    }
}

std::vector<WindowSpan> windowSpans(const std::vector<InNote>& k, int max_win)
{
    std::vector<WindowSpan> w;
    if (k.empty()) {
        return w;
    }
    double end = k[0].off;
    for (const InNote& x : k) {
        end = std::max(end, static_cast<double>(x.off));
    }
    // np.arange(0, end + L_WIN, L_WIN)
    const long nw = static_cast<long>(std::ceil((end + 8.0) / 8.0));
    auto lower = [&](double t) {
        return static_cast<int>(std::lower_bound(k.begin(), k.end(), t, [](const InNote& a, double v) {
                                    return static_cast<double>(a.on) < v;
                                }) -
                                k.begin());
    };
    for (long i = 0; i < nw; i++) {
        const double t0 = static_cast<double>(i) * 8.0;
        const int lo = lower(t0), hi = lower(t0 + 8.0);
        if (hi <= lo || hi - lo > max_win) {
            continue;
        }
        w.push_back({t0, lo, hi});
    }
    return w;
}

Edits::Edits(size_t n) : pk(n, std::nan("")), on_r(n, std::nan("")), off_r(n, std::nan("")) {}

void addEdits(const WindowSpan& w, const std::vector<Cand>& c, Edits& e)
{
    const int m = w.hi - w.lo;
    for (int j = 0; j < m; j++) {
        const Cand& x = c[j];
        const bool ok = x.on >= 0.0f && x.on < L_WIN; // a note whose corrected onset leaves the window is dropped
        e.pk[w.lo + j] = ok ? static_cast<double>(x.score) : -1.0;
        // float32 + np.float64 -> float64 (numpy 2)
        e.on_r[w.lo + j] = static_cast<double>(x.on) + w.t0;
        e.off_r[w.lo + j] = static_cast<double>(x.off) + w.t0;
    }
    for (size_t i = m; i < c.size(); i++) {
        const Cand& x = c[i];
        if (x.kind != 0 && x.on >= 0.0f && x.on < L_WIN) {
            e.ins.push_back({static_cast<double>(x.on) + w.t0, static_cast<double>(x.off) + w.t0, static_cast<double>(x.pitch),
                             static_cast<double>(x.score)});
        }
    }
}

namespace
{
    struct Row {
        int pitch;
        double on, off;
        float score;
        int kind;
    };

    /** _dedup_idx: same key within tol -> keep the higher score; kept indices, sorted. */
    std::vector<int> dedupIdx(const std::vector<Row>& c, const std::vector<int>& idx)
    {
        const size_t n = idx.size();
        std::vector<int> r(n);
        std::iota(r.begin(), r.end(), 0);
        if (n < 2) {
            return r;
        }
        // np.lexsort((on, key)): by key (= pitch, inst 0), then onset; stable
        std::vector<int> o(n);
        std::iota(o.begin(), o.end(), 0);
        std::stable_sort(o.begin(), o.end(), [&](int a, int b) {
            const Row &x = c[idx[a]], &y = c[idx[b]];
            if (x.pitch != y.pitch) {
                return x.pitch < y.pitch;
            }
            return x.on < y.on;
        });
        std::vector<char> keep(n, 1);
        for (size_t j = 0; j + 1 < n; j++) {
            const Row &a = c[idx[o[j]]], &b = c[idx[o[j + 1]]];
            if (!(a.pitch == b.pitch && b.on - a.on < DEDUP_S)) {
                continue;
            }
            if (!keep[j]) {
                continue;
            }
            if (a.score >= b.score) {
                keep[j + 1] = 0;
            } else {
                keep[j] = 0;
            }
        }
        std::vector<int> out;
        for (size_t j = 0; j < n; j++) {
            if (keep[j]) {
                out.push_back(o[j]);
            }
        }
        std::sort(out.begin(), out.end());
        return out;
    }
} // namespace

std::vector<RepairedNote> applyEdits(const std::vector<InNote>& k, const Edits& e, double keep_th, double insert_th, RepairInfo* info)
{
    std::vector<Row> c;
    for (size_t j = 0; j < k.size(); j++) {
        const InNote& x = k[j];
        if (std::isnan(e.pk[j])) {
            c.push_back({x.pitch, static_cast<double>(x.on), static_cast<double>(x.off), 1.0f, 0});
        } else if (e.pk[j] >= keep_th) {
            c.push_back({x.pitch, e.on_r[j], e.off_r[j], static_cast<float>(e.pk[j]), 0});
        }
    }
    for (const auto& r : e.ins) {
        if (r[3] >= insert_th) {
            c.push_back({static_cast<int>(r[2]), r[0], r[1], static_cast<float>(r[3]), 1});
        }
    }
    // _safe_dedup_idx: an insertion is dropped when a kept note of its pitch starts within tol (checking the neighbours
    // in the sorted onsets); the insertions among themselves go through _dedup_idx
    std::vector<int> sel;
    std::vector<int> ins_i;
    std::map<int, std::vector<double>> kp;
    for (size_t q = 0; q < c.size(); q++) {
        if (c[q].kind == 0) {
            kp[c[q].pitch].push_back(c[q].on);
        } else {
            ins_i.push_back(static_cast<int>(q));
        }
    }
    if (ins_i.empty()) {
        sel.resize(c.size());
        std::iota(sel.begin(), sel.end(), 0);
    } else {
        for (auto& [pch, v] : kp) {
            std::sort(v.begin(), v.end());
        }
        std::vector<int> ins_ok;
        for (int q : ins_i) {
            bool ok = true;
            const auto it = kp.find(c[q].pitch);
            if (it != kp.end()) {
                const auto& v = it->second;
                const long i = std::lower_bound(v.begin(), v.end(), c[q].on) - v.begin();
                for (long r : {i - 1, i}) {
                    if (r >= 0 && r < static_cast<long>(v.size()) && std::abs(v[r] - c[q].on) < DEDUP_S) {
                        ok = false;
                    }
                }
            }
            if (ok) {
                ins_ok.push_back(q);
            }
        }
        for (size_t q = 0; q < c.size(); q++) {
            if (c[q].kind == 0) {
                sel.push_back(static_cast<int>(q));
            }
        }
        for (int j : dedupIdx(c, ins_ok)) {
            sel.push_back(ins_ok[j]);
        }
        std::sort(sel.begin(), sel.end());
    }
    std::vector<RepairedNote> out;
    out.reserve(sel.size());
    for (int q : sel) {
        const Row& r = c[q];
        out.push_back({pyRound4(r.on), pyRound4(std::max(r.off, r.on + 0.01)), r.pitch, r.kind == 1, r.score});
    }
    if (info != nullptr) {
        info->transcribed = static_cast<int>(k.size());
        info->inserted = static_cast<int>(std::count_if(out.begin(), out.end(), [](const RepairedNote& x) { return x.inserted; }));
        info->kept = static_cast<int>(out.size()) - info->inserted;
        info->dropped = info->transcribed - info->kept;
    }
    return out;
}

} // namespace dtt::ev
