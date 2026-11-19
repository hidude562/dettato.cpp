#include "grouper/grouper.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_map>

#include <ggml.h>

#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
#include <immintrin.h>
#endif

#include "backend.hpp"
#include "common.hpp"
#include "dsp/resample.hpp"
#include "grouper/front.hpp"
#include "grouper/notenet.hpp"
#include "grouper/retune.hpp"
#include "model_file.hpp"
#include "transcriber/transcriber.hpp"

namespace dtt
{

namespace
{
    constexpr int SR = 16000;
    constexpr int PITCH_LO = 21;
    constexpr int SPAN_CAP = 50, STEM_SPAN = 10; // frames averaged for tap 1 (the note, <= 1 s) and tap 2 (the attack)
    constexpr double SLOT_TOL = 0.1;             // s: an input note takes the voice slot of the nearest same-pitch note
    constexpr double DEDUP = 0.03;               // s: same-pitch notes this close inside one track are one note
    constexpr double PAD0 = 0.5, PAD1 = 1.0;     // a window's audio starts PAD0 before it and ends PAD1 after it
    constexpr int NC = GroupFront::NOTE_COLS;
    constexpr int N_TAPS = 3;

    double seconds(std::chrono::steady_clock::time_point t0)
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    /** Runs f(i) for i in [0, n) on up to `threads` threads; rethrows the first exception. */
    template <class F>
    void parallelFor(int n, int threads, F&& f)
    {
        threads = std::max(1, std::min(threads, n));
        if (threads == 1) {
            for (int i = 0; i < n; i++) {
                f(i);
            }
            return;
        }
        std::atomic<int> next{0};
        std::exception_ptr err;
        std::mutex mu;
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; t++) {
            pool.emplace_back([&] {
                try {
                    for (int i; (i = next++) < n;) {
                        f(i);
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> lk(mu);
                    if (!err) {
                        err = std::current_exception();
                    }
                    next = n;
                }
            });
        }
        for (auto& th : pool) {
            th.join();
        }
        if (err) {
            std::rethrow_exception(err);
        }
    }

    /** Python's round(x, nd): the exact binary value rounded to nd decimals (half to even), as printf does it. */
    double pyRound(double x, int nd)
    {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.*f", nd, x);
        return std::strtod(buf, nullptr);
    }

    /** The frames a note's taps are averaged over (core.py gather), on the transcriber's time axis. */
    struct Span {
        int lane, t0, lim1, lim2; // H at t0; H over [t0, lim1); S over [t0, lim2)
    };

    /**
     * Pass 2's taps gathered per note while the windows stream.
     *
     * Default (f16 = true): exactly the reference's arithmetic. Transcriber.run accumulates the Hann-weighted taps per frame
     * in float16 (buf += (v w).half(), then buf /= wsum.half()), and gather averages those float16 values per note in
     * float32. Here every (frame, lane) a note reads gets a float16 accumulator when a window first touches it; when the
     * transcriber reports the frame final it is divided, handed to the notes that read it (in frame order, as gather's
     * loop adds them) and freed. Memory is bounded by the frames in flight, not by the song.
     *
     * f16 = false: float32 throughout, each window adding v w / wsum at the note's frames directly (no per-frame state);
     * kept to measure what the float16 rounding of the reference does.
     */
    class GatherSink : public TapSink
    {
    public:
        GatherSink(const std::vector<Span>& sp, int C, int T, int L, const std::vector<float>& ws, bool f16)
            : mSp(sp), mC(C), mT(T), mL(L), mWs(ws), mF16(f16)
        {
            const size_t n = sp.size();
            for (const Span& s : sp) {
                mT0.push_back(s.t0);
            }
            if (!f16) {
                mAcc.assign(n * N_TAPS * C, 0.0);
                return;
            }
            mX0.assign(n * C, 0.0f);
            mA1.assign(n * C, 0.0f);
            mA2.assign(n * C, 0.0f);
            mNeed.assign(static_cast<size_t>(T) * L, 0);
            for (const Span& s : sp) {
                for (int t = s.t0; t < s.lim1; t++) {
                    mNeed[static_cast<size_t>(t) * L + s.lane] |= 1;
                }
                for (int t = s.t0; t < s.lim2; t++) {
                    mNeed[static_cast<size_t>(t) * L + s.lane] |= 2;
                }
            }
        }

        void window(int tap, int g0, int a, int b, const float* w, const float* data, int C, int L, int Fw) override
        {
            DTT_CHECK(C == mC && L == mL, "tap shape %d x %d, expected %d x %d", C, L, mC, mL);
            if (tap != HIDDEN && tap != STEM) {
                return;
            }
            if (mF16) {
                const uint8_t bit = tap == HIDDEN ? 1 : 2;
                auto& bufs = tap == HIDDEN ? mBufH : mBufS;
                for (int t = a; t < b; t++) {
                    const float wt = w[t - a];
                    for (int l = 0; l < L; l++) {
                        if (!(mNeed[static_cast<size_t>(t) * L + l] & bit)) {
                            continue;
                        }
                        std::vector<float>& buf = bufs[static_cast<int64_t>(t) * L + l];
                        if (buf.empty()) {
                            buf.assign(C, 0.0f);
                        }
                        const float* src = data + (static_cast<size_t>(t - g0) * L + l) * C;
                        for (int c = 0; c < C; c++) {
                            buf[c] = roundHalf(buf[c] + roundHalf(src[c] * wt));
                        }
                    }
                }
                return;
            }
            // float32 path: notes whose frames [t0, t0 + SPAN_CAP) can meet [a, b): t0 in [a - SPAN_CAP, b)
            const auto lo = std::lower_bound(mT0.begin(), mT0.end(), a - SPAN_CAP) - mT0.begin();
            const auto hi = std::lower_bound(mT0.begin(), mT0.end(), b) - mT0.begin();
            for (auto i = lo; i < hi; i++) {
                const Span& s = mSp[i];
                auto add = [&](int slot, int t_from, int t_to) {
                    double* acc = mAcc.data() + (static_cast<size_t>(i) * N_TAPS + slot) * mC;
                    for (int t = std::max(t_from, a); t < std::min(t_to, b); t++) {
                        const double coef = static_cast<double>(w[t - a]) / static_cast<double>(std::max(mWs[t], 1e-6f));
                        const float* src = data + (static_cast<size_t>(t - g0) * L + s.lane) * C;
                        for (int c = 0; c < mC; c++) {
                            acc[c] += src[c] * coef;
                        }
                    }
                };
                if (tap == HIDDEN) {
                    add(0, s.t0, s.t0 + 1);
                    add(1, s.t0, s.lim1);
                } else {
                    add(2, s.t0, s.lim2);
                }
            }
        }

        void framesFinal(int t0, int t1, const Maps& maps, const std::vector<float>& wsum) override
        {
            if (!mF16) {
                return;
            }
            std::vector<float> zeros(mC, 0.0f);
            for (int t = t0; t < t1; t++) {
                // the hop-0.5 layout's own weight sum (the run's wsum when every window runs; with a window subset only
                // the frames whose covering windows all ran are read, and those need the full sum)
                const float wsh = roundHalf(std::max(mWs[t], 1e-6f));
                // the frame's averaged taps (float16 / float16), for the lanes notes read
                for (auto* bufs : {&mBufH, &mBufS}) {
                    for (int l = 0; l < mL; l++) {
                        auto it = bufs->find(static_cast<int64_t>(t) * mL + l);
                        if (it != bufs->end()) {
                            for (float& v : it->second) {
                                v = roundHalf(v / wsh);
                            }
                        }
                    }
                }
                auto get = [&](std::unordered_map<int64_t, std::vector<float>>& bufs, int l) -> const float* {
                    auto it = bufs.find(static_cast<int64_t>(t) * mL + l);
                    return it == bufs.end() ? zeros.data() : it->second.data();
                };
                // the notes reading frame t: t0 in (t - SPAN_CAP, t]
                const auto lo = std::lower_bound(mT0.begin(), mT0.end(), t - SPAN_CAP + 1) - mT0.begin();
                const auto hi = std::upper_bound(mT0.begin(), mT0.end(), t) - mT0.begin();
                for (auto i = lo; i < hi; i++) {
                    const Span& s = mSp[i];
                    if (t < s.lim1) {
                        const float* h = get(mBufH, s.lane);
                        float* a1 = mA1.data() + static_cast<size_t>(i) * mC;
                        for (int c = 0; c < mC; c++) {
                            a1[c] += h[c];
                        }
                        if (t == s.t0) {
                            std::copy(h, h + mC, mX0.data() + static_cast<size_t>(i) * mC);
                        }
                    }
                    if (t < s.lim2) {
                        const float* h = get(mBufS, s.lane);
                        float* a2 = mA2.data() + static_cast<size_t>(i) * mC;
                        for (int c = 0; c < mC; c++) {
                            a2[c] += h[c];
                        }
                    }
                }
                for (auto* bufs : {&mBufH, &mBufS}) {
                    for (int l = 0; l < mL; l++) {
                        bufs->erase(static_cast<int64_t>(t) * mL + l);
                    }
                }
            }
        }

        /** X (n, 3, C): H at the onset, H over the note, S over the attack. */
        std::vector<float> taps() const
        {
            const size_t n = mSp.size();
            std::vector<float> X(n * N_TAPS * mC);
            for (size_t i = 0; i < n; i++) {
                const int cnt1 = mSp[i].lim1 - mSp[i].t0, cnt2 = mSp[i].lim2 - mSp[i].t0;
                for (int c = 0; c < mC; c++) {
                    float* x = X.data() + i * N_TAPS * mC + c;
                    if (mF16) {
                        x[0] = mX0[i * mC + c];
                        x[mC] = mA1[i * mC + c] / static_cast<float>(cnt1);
                        x[2 * mC] = mA2[i * mC + c] / static_cast<float>(cnt2);
                    } else {
                        const double* acc = mAcc.data() + i * N_TAPS * mC + c;
                        x[0] = static_cast<float>(acc[0]);
                        x[mC] = static_cast<float>(acc[mC] / cnt1);
                        x[2 * mC] = static_cast<float>(acc[2 * mC] / cnt2);
                    }
                }
            }
            return X;
        }

    private:
        const std::vector<Span>& mSp;
        int mC, mT, mL;
        const std::vector<float>& mWs;
        bool mF16;
        std::vector<int> mT0;
        std::vector<double> mAcc;                 // float32 path
        std::vector<float> mX0, mA1, mA2;         // float16 path: per-note float32 accumulators
        std::vector<uint8_t> mNeed;               // (T, L): bit 1 = H read, bit 2 = S read
        std::unordered_map<int64_t, std::vector<float>> mBufH, mBufS; // frames in flight, float16 values
    };

    /** core.song_windows: [(t0, note indices)] over onsets `on` (sorted, float32; compared in float64 like numpy). */
    std::vector<std::pair<double, std::vector<int>>> songWindows(const std::vector<float>& on, double W, double hop, int nmax)
    {
        std::vector<std::pair<double, std::vector<int>>> out;
        const int n = static_cast<int>(on.size());
        if (n == 0) {
            return out;
        }
        auto ss = [&](double v) {
            return static_cast<int>(std::lower_bound(on.begin(), on.end(), v, [](float a, double x) { return static_cast<double>(a) < x; }) -
                                    on.begin());
        };
        const double end = on[n - 1];
        double t0 = 0.0;
        while (true) {
            const int a = ss(t0), b = ss(t0 + W);
            const int e = std::min(b, a + nmax);
            if (e - a >= 1) {
                std::vector<int> sel(e - a);
                std::iota(sel.begin(), sel.end(), a);
                out.emplace_back(t0, std::move(sel));
            }
            if (b >= n && a + nmax >= b) {
                break;
            }
            if (b - a > nmax) {
                t0 = std::max(t0 + 0.25 * hop, static_cast<double>(on[a + nmax / 2]));
            } else {
                t0 += hop;
            }
            if (t0 > end) {
                break;
            }
        }
        return out;
    }
} // namespace

Grouper::Grouper(const ModelFile& mf, const Backend& be) : mBe(be)
{
    mNet = std::make_unique<NoteNet>(mf, be);
    mFront = std::make_unique<GroupFront>(mf);
    std::stringstream names(mf.str("gr.instruments"));
    for (std::string s; std::getline(names, s, ',');) {
        mNames.push_back(s);
    }
    for (int32_t p : mf.i32s("gr.programs")) {
        mPrograms.push_back(p);
    }
    const int K = static_cast<int>(mf.u32("gr.n_inst"));
    DTT_CHECK(static_cast<int>(mNames.size()) == K && static_cast<int>(mPrograms.size()) == K && mNet->nInst() == K,
              "instrument table: %zu names, %zu programs, %d in the metadata, %d logits", mNames.size(), mPrograms.size(), K,
              mNet->nInst());
    const auto cr = mf.hostF32("gr.credit");
    DTT_CHECK(cr.size() == static_cast<size_t>(K) * K, "CREDIT matrix has %zu values, expected %d x %d", cr.size(), K, K);
    mCredit.assign(cr.begin(), cr.end());
    for (int i = 0; i < K; i++) {
        DTT_CHECK(mCredit[static_cast<size_t>(i) * K + i] == 1.0, "CREDIT diagonal");
    }
    const int nth = static_cast<int>(mf.u32("gr.th_layers"));
    for (int j = 0; j < nth; j++) {
        ggml_tensor* t = mf.tensor(format("gr.th.%d.w", j));
        TrackHeadLayer l;
        l.in = static_cast<int>(t->ne[0]);
        l.out = static_cast<int>(t->ne[1]);
        const auto w = mf.hostF32(format("gr.th.%d.w", j)), b = mf.hostF32(format("gr.th.%d.b", j));
        l.w.assign(w.begin(), w.end());
        l.b.assign(b.begin(), b.end());
        mTh.push_back(std::move(l));
    }
    DTT_CHECK(mTh.front().in == mNet->d() + 6 && mTh.back().out == K, "track head shape");
    mDec.tau = std::stod(mf.str("gr.dec.tau"));
    mDec.tau2 = std::stod(mf.str("gr.dec.tau2"));
    mDec.lam = std::stod(mf.str("gr.dec.lam"));
    mDec.alpha = std::stod(mf.str("gr.dec.alpha"));
    mDec.absorb = std::stod(mf.str("gr.dec.absorb"));
    mDec.absorb_frac = std::stod(mf.str("gr.dec.absorb_frac"));
    mDec.min_notes = static_cast<int>(mf.u32("gr.dec.min_notes"));
    mDec.refine = static_cast<int>(mf.u32("gr.dec.refine"));
    mDec.ec = mf.u32("gr.dec.ec") != 0;
    mDec.merge = mf.u32("gr.dec.merge") != 0;
    mWindowS = mf.f32("gr.window_s");
    mWindowHop = mf.f32("gr.window_hop");
    mMaxNotes = static_cast<int>(mf.u32("gr.max_notes"));
    const auto taps = mf.i32s("gr.s15_taps");
    DTT_CHECK(taps.size() == 3 && taps[0] == 0 && taps[1] == 1 && taps[2] == 2, "s15 taps other than (0, 1, 2) are not ported");
    // windows per network call: any batch gives the same per-note outputs; 4 bounds the GPU compute buffer
    mBatch = 4;
}

Grouper::~Grouper() = default;

// ------------------------------------------------------------------------------------------------------- features
namespace
{
    /** assign_slots' order and ends + the frames gather reads, for notes (N, 3) on the audio's time axis. */
    struct NotePlan {
        int N = 0, T = 0, L = 0, C = 0;
        double hop = 0;
        std::vector<int> o;                  // sorted position -> input index
        std::vector<double> on, off, pitch;  // sorted, end >= onset + 20 ms
        std::vector<Span> sp;
        std::vector<int> te;
        std::vector<float> wsum;             // the hop-0.5 layout's Hann weight sum (float32, window order)
    };

    NotePlan planNotes(const std::vector<std::array<double, 3>>& notes, size_t n, Transcriber& tr)
    {
        NotePlan g;
        const int N = static_cast<int>(notes.size());
        g.N = N;
        // assign_slots, first half: the order (onset, then pitch; stable) and the end >= onset + 20 ms
        g.o.resize(N);
        std::iota(g.o.begin(), g.o.end(), 0);
        std::stable_sort(g.o.begin(), g.o.end(), [&](int a, int b) {
            if (notes[a][0] != notes[b][0]) {
                return notes[a][0] < notes[b][0];
            }
            return notes[a][2] < notes[b][2];
        });
        g.on.resize(N);
        g.off.resize(N);
        g.pitch.resize(N);
        for (int i = 0; i < N; i++) {
            g.on[i] = notes[g.o[i]][0];
            g.off[i] = std::max(notes[g.o[i]][1], notes[g.o[i]][0] + 0.02);
            g.pitch[i] = notes[g.o[i]][2];
        }
        const Transcriber::Layout lo = tr.layout(n, 0.5);
        g.T = lo.T;
        g.L = tr.net().lanes();
        g.C = tr.channels();
        g.hop = tr.hop();
        // the Hann weight sum of every frame, accumulated in float32 in window order (Transcriber.run's wsum)
        g.wsum.assign(g.T, 0.0f);
        const auto w = tr.taper(lo.Fw);
        for (long s0 : lo.starts) {
            const int g0 = tr.frameOffset(s0, lo.g_pad);
            for (int t = std::max(0, g0); t < std::min(g.T, g0 + lo.Fw); t++) {
                g.wsum[t] += w[t - g0];
            }
        }
        g.sp.resize(N);
        g.te.resize(N);
        for (int i = 0; i < N; i++) {
            Span& s = g.sp[i];
            s.lane = static_cast<int>(std::clamp(static_cast<long>(std::nearbyint(g.pitch[i])) - PITCH_LO, 0L, static_cast<long>(g.L - 1)));
            s.t0 = static_cast<int>(std::clamp(static_cast<long>(std::nearbyint(g.on[i] / g.hop)), 0L, static_cast<long>(g.T - 1)));
            g.te[i] = static_cast<int>(std::min(std::max(static_cast<long>(std::ceil(g.off[i] / g.hop)), static_cast<long>(s.t0) + 1),
                                                static_cast<long>(g.T)));
            s.lim1 = std::min(g.te[i], s.t0 + SPAN_CAP);
            s.lim2 = std::min(g.te[i], s.t0 + STEM_SPAN);
        }
        return g;
    }

    /** assign_slots' second half + gather's 12 scalars, from pass 2's maps. */
    void metaFromMaps(GroupFeatures& out, const NotePlan& g, const std::vector<std::array<double, 3>>& notes, const Maps& maps,
                      const Transcriber& tr)
    {
        const int N = g.N, T = g.T, L = g.L;
        DTT_CHECK(maps.T == T && maps.L == L, "maps %d x %d, expected %d x %d", maps.T, maps.L, T, L);
        // the voice slot of the nearest same-pitch transcriber note within SLOT_TOL (in input order)
        const auto tn = tr.notes(maps);
        std::vector<double> slot_in(N, 0.0);
        std::map<double, std::vector<int>> byPitch;
        for (int i = 0; i < N; i++) {
            byPitch[notes[i][2]].push_back(i);
        }
        for (auto& [p, a] : byPitch) {
            std::vector<int> b;
            for (int j = 0; j < static_cast<int>(tn.size()); j++) {
                if (static_cast<double>(tn[j].pitch) == p) {
                    b.push_back(j);
                }
            }
            if (b.empty()) {
                continue;
            }
            for (int i : a) {
                double best = INFINITY;
                int bj = 0;
                for (int j : b) {
                    const double d = std::abs(notes[i][0] - tn[j].onset);
                    if (d < best) {
                        best = d;
                        bj = j;
                    }
                }
                slot_in[i] = best <= SLOT_TOL ? static_cast<double>(tn[bj].slot) - 1.0 : 0.0;
            }
        }
        // 12 scalars per note (core.gather; float32 arithmetic, torch's float64-accumulated cumsum)
        auto csum = [&](int h) {
            std::vector<float> cs(static_cast<size_t>(T + 1) * L, 0.0f);
            for (int l = 0; l < L; l++) {
                double acc = 0;
                for (int t = 0; t < T; t++) {
                    acc += maps.at(h, t, l);
                    cs[static_cast<size_t>(t + 1) * L + l] = static_cast<float>(acc);
                }
            }
            return cs;
        };
        const auto csF = csum(FRAME), csF2 = csum(FRAME2);
        auto at = [&](int h, int t, int l) { return maps.at(h, std::clamp(t, 0, T - 1), l); };
        auto mx3 = [&](int h, int t, int l) { return std::max(std::max(at(h, t - 1, l), at(h, t, l)), at(h, t + 1, l)); };
        auto spanMean = [&](const std::vector<float>& cs, int t0, int t1, int l) {
            return (cs[static_cast<size_t>(t1) * L + l] - cs[static_cast<size_t>(t0) * L + l]) / static_cast<float>(std::max(t1 - t0, 1));
        };
        out.meta.resize(static_cast<size_t>(N) * NC);
        for (int i = 0; i < N; i++) {
            float* m = out.meta.data() + static_cast<size_t>(i) * NC;
            const Span& s = g.sp[i];
            m[0] = static_cast<float>(g.on[i]);
            m[1] = static_cast<float>(g.off[i]);
            m[2] = static_cast<float>(g.pitch[i]);
            m[3] = static_cast<float>(slot_in[g.o[i]]);
            m[4] = mx3(ONSET, s.t0, s.lane);
            m[5] = mx3(ONSET2, s.t0, s.lane);
            m[6] = at(FRAME, s.t0, s.lane);
            m[7] = at(FRAME2, s.t0, s.lane);
            m[8] = spanMean(csF, s.t0, g.te[i], s.lane);
            m[9] = spanMean(csF2, s.t0, g.te[i], s.lane);
            m[10] = mx3(OFFSET, g.te[i] - 1, s.lane);
            m[11] = mx3(OFFSET2, g.te[i] - 1, s.lane);
        }
    }

    /** The taps out.X quantised: int8 per row with a float32 scale, stored as float16. */
    void quantiseTaps(GroupFeatures& out, int C)
    {
        const size_t rows = static_cast<size_t>(out.n) * N_TAPS;
        out.feat.resize(out.X.size());
        out.fsc.resize(rows);
        for (size_t r = 0; r < rows; r++) {
            const float* x = out.X.data() + r * C;
            float amax = 0.0f;
            for (int c = 0; c < C; c++) {
                amax = std::max(amax, std::abs(x[c]));
            }
            const float sc = std::max(amax, 1e-6f) / 127.0f;
            for (int c = 0; c < C; c++) {
                out.feat[r * C + c] = static_cast<int8_t>(std::clamp(std::nearbyint(x[c] / sc), -127.0f, 127.0f));
            }
            out.fsc[r] = roundHalf(sc);
        }
    }

    /** float16 bits of a value that is exactly representable in float16 (the store's values are). */
    inline uint16_t halfBits(float h)
    {
        return ggml_fp32_to_fp16(h);
    }
    inline float halfValue(uint16_t b)
    {
        return ggml_fp16_to_fp32(b);
    }

    // dst = f16(f32(dst) + f16(src * w)) and dst = f16(f32(dst) / d) over n values: the reference's float16 accumulation
    // and division (torch adds / divides float16 tensors through float32, rounding to nearest even). F16C when the CPU has
    // it (the same rounding), else scalar.
    void accumScalar(uint16_t* dst, const float* src, float w, size_t n)
    {
        for (size_t i = 0; i < n; i++) {
            dst[i] = halfBits(roundHalf(halfValue(dst[i]) + roundHalf(src[i] * w)));
        }
    }
    void divScalar(uint16_t* dst, const uint16_t* src, float d, size_t n)
    {
        for (size_t i = 0; i < n; i++) {
            dst[i] = halfBits(roundHalf(halfValue(src[i]) / d));
        }
    }
#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
    __attribute__((target("avx,f16c"))) void accumF16C(uint16_t* dst, const float* src, float w, size_t n)
    {
        const __m256 wv = _mm256_set1_ps(w);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            const __m256 x = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(_mm256_loadu_ps(src + i), wv), _MM_FROUND_TO_NEAREST_INT));
            const __m256 d = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(dst + i)));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), _mm256_cvtps_ph(_mm256_add_ps(d, x), _MM_FROUND_TO_NEAREST_INT));
        }
        accumScalar(dst + i, src + i, w, n - i);
    }
    __attribute__((target("avx,f16c"))) void divF16C(uint16_t* dst, const uint16_t* src, float dv, size_t n)
    {
        const __m256 d = _mm256_set1_ps(dv);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            const __m256 x = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i)));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), _mm256_cvtps_ph(_mm256_div_ps(x, d), _MM_FROUND_TO_NEAREST_INT));
        }
        divScalar(dst + i, src + i, dv, n - i);
    }
    const bool HAVE_F16C = __builtin_cpu_supports("avx") && __builtin_cpu_supports("f16c");
    inline void accumHalf(uint16_t* dst, const float* src, float w, size_t n)
    {
        HAVE_F16C ? accumF16C(dst, src, w, n) : accumScalar(dst, src, w, n);
    }
    inline void divHalf(uint16_t* dst, const uint16_t* src, float d, size_t n)
    {
        HAVE_F16C ? divF16C(dst, src, d, n) : divScalar(dst, src, d, n);
    }
#else
    inline void accumHalf(uint16_t* dst, const float* src, float w, size_t n)
    {
        accumScalar(dst, src, w, n);
    }
    inline void divHalf(uint16_t* dst, const uint16_t* src, float d, size_t n)
    {
        divScalar(dst, src, d, n);
    }
#endif

    /**
     * Pass 1's taps, kept for the grouper (Grouper::makeStoreSink).
     *
     * Of every window it sees, it uses only the ones a hop-0.5 pass runs (start a multiple of half a window: g0 a multiple of
     * Fw / 2), with its own weight sum over them, and accumulates per (frame, lane) exactly like GatherSink's float16 path
     * (all lanes, while the frame is in flight, as float16 bits). When pass 1 reports frames final, they are divided by the
     * float16 weight sum, and from the final maps a candidate set is kept: H at (t, lane) when an onset of either voice
     * (>= on_th) lies within [t - SPAN_CAP + 1 - delta, t + delta] at that lane and either an onset lies within the stem reach
     * or a frame probability >= fr_th lies within [t - delta, t + delta]; S at (t, lane) when an onset lies within
     * [t - STEM_SPAN + 1 - delta, t + delta]. Everything else is dropped, so the store grows with the note frames, not with
     * the song. (The decision for frame t waits until the maps are final up to t + delta.)
     */
    class StoreSink : public TapSink
    {
    public:
        struct Params {
            // measured on the 4-min song (4503 repaired notes): 0.02 / 0.02 / 3 keeps 228 MB and leaves 4-5 notes (8-9 of
            // 61 windows) to recompute; 0.05 / 0.05: 168 MB, 13-14 notes (18-20 windows); 0.01 / 0.01 / 5: 316 MB, 3 (6)
            float on_th = 0.02f, fr_th = 0.02f;
            int delta = 3;
            bool peaks = false; // events = local maxima of the onset evidence only (fewer pairs, more misses)
        };

        explicit StoreSink(Params p) : mP(p)
        {
        }

        void window(int tap, int g0, int a, int b, const float* w, const float* data, int C, int L, int Fw) override
        {
            if (tap != HIDDEN && tap != STEM) {
                return;
            }
            const int step = Fw / 2; // a hop-0.5 window starts every half window
            if (((g0 % step) + step) % step != 0) {
                return;
            }
            if (mC == 0) {
                mC = C;
                mL = L;
                mFw = Fw;
            }
            DTT_CHECK(C == mC && L == mL && Fw == mFw, "tap shape changed");
            (tap == HIDDEN ? mSawH : mSawS) = true;
            if (tap == HIDDEN) { // the window's weights, once per window
                if (static_cast<int>(mWs.size()) < b) {
                    mWs.resize(b, 0.0f);
                }
                for (int t = a; t < b; t++) {
                    mWs[t] += w[t - a];
                }
                mWindows++;
            }
            const size_t per = static_cast<size_t>(2) * L * C;
            std::vector<uint16_t*> dst(b - a);
            for (int t = a; t < b; t++) {
                std::vector<uint16_t>& blk = mLive[t];
                if (blk.empty()) {
                    blk.assign(per, halfBits(0.0f));
                    mLiveBytes += per * sizeof(uint16_t);
                }
                dst[t - a] = blk.data() + static_cast<size_t>(tap) * L * C;
            }
            parallelFor(b - a, mThreads, [&](int k) {
                accumHalf(dst[k], data + static_cast<size_t>(a + k - g0) * L * C, w[k], static_cast<size_t>(L) * C);
            });
            mPeak = std::max(mPeak, bytes());
        }

        void framesFinal(int t0, int t1, const Maps& maps, const std::vector<float>& wsum) override
        {
            if (mC == 0) {
                return; // no taps (the run had no sink taps)
            }
            mT = maps.T;
            const int L = mL;
            // the final maps' activity: onset and frame probability of either voice
            mOn.resize(static_cast<size_t>(t1) * L, 0.0f);
            mFr.resize(static_cast<size_t>(t1) * L, 0.0f);
            for (int t = t0; t < t1; t++) {
                for (int l = 0; l < L; l++) {
                    const size_t q = static_cast<size_t>(t) * L + l;
                    mOn[q] = std::max(maps.at(ONSET, t, l), maps.at(ONSET2, t, l));
                    mFr[q] = std::max(maps.at(FRAME, t, l), maps.at(FRAME2, t, l));
                }
            }
            // the frames' averaged taps (float16 / float16, as Transcriber.run divides them)
            for (int t = t0; t < t1; t++) {
                std::vector<uint16_t> blk;
                auto it = mLive.find(t);
                if (it != mLive.end()) {
                    blk = std::move(it->second);
                    mLive.erase(it);
                    mLiveBytes -= blk.size() * sizeof(uint16_t);
                } else {
                    blk.assign(static_cast<size_t>(2) * L * mC, halfBits(0.0f));
                }
                // divided (float16 / float16(wsum)) only for the lanes that are kept, at the decision below
                if (static_cast<int>(mWsh.size()) <= t) {
                    mWsh.resize(t + 1, 1.0f);
                }
                mWsh[t] = roundHalf(std::max(t < static_cast<int>(mWs.size()) ? mWs[t] : 0.0f, 1e-6f));
                mPendBytes += blk.size() * sizeof(uint16_t);
                mPend.emplace(t, std::move(blk));
            }
            mPeak = std::max(mPeak, bytes());
            // decide every pending frame whose maps neighbourhood is final
            const int upto = t1 >= mT ? t1 : t1 - 1 - mP.delta;
            if (mLast.empty()) {
                mLast.assign(L, -1000000);
            }
            for (int t = mDecided; t < upto; t++) {
                // the latest onset event at each lane up to t + delta
                for (; mScanned <= std::min(t + mP.delta, t1 - 1); mScanned++) {
                    const int u = mScanned;
                    for (int l = 0; l < L; l++) {
                        const float o = mOn[static_cast<size_t>(u) * L + l];
                        bool ev = o >= mP.on_th;
                        if (ev && mP.peaks) { // a local maximum of the onset evidence (the next frame is final: u < t1 - 1)
                            const float prev = u > 0 ? mOn[static_cast<size_t>(u - 1) * L + l] : -1.0f;
                            const float next = u + 1 < t1 ? mOn[static_cast<size_t>(u + 1) * L + l] : -1.0f;
                            ev = o >= prev && o >= next;
                        }
                        if (ev) {
                            mLast[l] = u;
                        }
                    }
                }
                auto it = mPend.find(t);
                if (it == mPend.end()) {
                    continue;
                }
                const std::vector<uint16_t>& blk = it->second;
                for (int l = 0; l < L; l++) {
                    const bool evH = mLast[l] >= t - SPAN_CAP + 1 - mP.delta;
                    const bool evS = mLast[l] >= t - STEM_SPAN + 1 - mP.delta;
                    bool active = false;
                    for (int v = std::max(0, t - mP.delta); v <= std::min(t1 - 1, t + mP.delta) && !active; v++) {
                        active = mFr[static_cast<size_t>(v) * L + l] >= mP.fr_th;
                    }
                    const int64_t key = static_cast<int64_t>(t) * L + l;
                    if (evH && (evS || active)) {
                        std::vector<uint16_t> v(mC);
                        divHalf(v.data(), blk.data() + static_cast<size_t>(l) * mC, mWsh[t], mC);
                        mH.emplace(key, std::move(v));
                        mKeptBytes += mC * sizeof(uint16_t);
                    }
                    if (evS) {
                        std::vector<uint16_t> v(mC);
                        divHalf(v.data(), blk.data() + (static_cast<size_t>(L) + l) * mC, mWsh[t], mC);
                        mS.emplace(key, std::move(v));
                        mKeptBytes += mC * sizeof(uint16_t);
                    }
                }
                mPendBytes -= blk.size() * sizeof(uint16_t);
                mPend.erase(it);
            }
            mDecided = std::max(mDecided, upto);
            mPeak = std::max(mPeak, bytes());
        }

        /** The stored float16 values of H / S at (t, lane), or null. */
        const uint16_t* hidden(int t, int l) const { return find(mH, t, l); }
        const uint16_t* stem(int t, int l) const { return find(mS, t, l); }
        bool complete() const { return mT > 0 && mDecided >= mT; }
        bool sawBothTaps() const { return mSawH && mSawS; }
        int frames() const { return mT; }
        int lanes() const { return mL; }
        int channels() const { return mC; }
        int windows() const { return mWindows; }
        size_t pairsH() const { return mH.size(); }
        size_t pairsS() const { return mS.size(); }
        size_t keptBytes() const { return mKeptBytes; }
        size_t peakBytes() const { return mPeak; }
        /** The store's weight sum, to check against the hop-0.5 layout's. */
        const std::vector<float>& wsum() const { return mWs; }

    private:
        static constexpr size_t ENTRY_OVERHEAD = 64; // hash node + vector header, roughly
        size_t bytes() const
        {
            return mLiveBytes + mPendBytes + mKeptBytes + (mH.size() + mS.size()) * ENTRY_OVERHEAD +
                   (mOn.size() + mFr.size() + mWs.size() + mWsh.size()) * sizeof(float);
        }
        const uint16_t* find(const std::unordered_map<int64_t, std::vector<uint16_t>>& m, int t, int l) const
        {
            auto it = m.find(static_cast<int64_t>(t) * mL + l);
            return it == m.end() ? nullptr : it->second.data();
        }

        Params mP;
        int mC = 0, mL = 0, mFw = 0, mT = 0, mDecided = 0, mScanned = 0, mWindows = 0;
        bool mSawH = false, mSawS = false;
        std::vector<float> mWs, mWsh, mOn, mFr;
        std::vector<int> mLast;
        int mThreads = std::max(1, std::min(4, static_cast<int>(std::thread::hardware_concurrency()) / 4));
        std::map<int, std::vector<uint16_t>> mLive, mPend;
        std::unordered_map<int64_t, std::vector<uint16_t>> mH, mS;
        size_t mLiveBytes = 0, mPendBytes = 0, mKeptBytes = 0, mPeak = 0;
    };
} // namespace

GroupFeatures Grouper::features(const float* y, size_t n, const std::vector<std::array<double, 3>>& notes, Transcriber& tr) const
{
    GroupFeatures out;
    out.n = static_cast<int>(notes.size());
    const NotePlan g = planNotes(notes, n, tr);
    GatherSink sink(g.sp, g.C, g.T, g.L, g.wsum, !(debug && debug->taps_f32));
    const Maps maps = g.N ? tr.run(y, n, 0.5, &sink, (1u << TapSink::HIDDEN) | (1u << TapSink::STEM)) : Maps{};
    if (g.N == 0) {
        return out;
    }
    metaFromMaps(out, g, notes, maps, tr);
    out.X = sink.taps();
    quantiseTaps(out, g.C);
    return out;
}

std::unique_ptr<TapSink> Grouper::makeStoreSink() const
{
    return std::make_unique<StoreSink>(StoreSink::Params{});
}

GroupFeatures Grouper::featuresFromStore(const float* y, size_t n, const std::vector<std::array<double, 3>>& notes_y, TapSink& store,
                                         const Maps& maps2, Transcriber& tr) const
{
    auto* st = dynamic_cast<StoreSink*>(&store);
    DTT_CHECK(st != nullptr, "featuresFromStore needs the sink made by makeStoreSink()");
    GroupFeatures out;
    out.n = static_cast<int>(notes_y.size());
    const NotePlan g = planNotes(notes_y, n, tr);
    if (g.N == 0) {
        return out;
    }
    DTT_CHECK(st->complete() && st->frames() == g.T, "the store has %d final frames, the audio %d (run pass 1 on the same audio)",
              st->frames(), g.T);
    DTT_CHECK(st->sawBothTaps(), "the store needs both taps of pass 1 (tap_mask HIDDEN | STEM)");
    DTT_CHECK(st->lanes() == g.L && st->channels() == g.C, "store taps %d x %d, expected %d x %d", st->lanes(), st->channels(),
              g.L, g.C);
    {
        // the store's windows must be exactly the hop-0.5 layout's (pass 1 at hop_frac 0.5 or 0.25 on the same audio)
        bool same = st->wsum().size() >= static_cast<size_t>(g.T);
        for (int t = 0; same && t < g.T; t++) {
            same = st->wsum()[t] == g.wsum[t];
        }
        DTT_CHECK(same, "the store's windows are not the hop-0.5 windows of this audio (run pass 1 at hop_frac 0.5 or 0.25 on y)");
    }
    const auto t_start = std::chrono::steady_clock::now();
    metaFromMaps(out, g, notes_y, maps2, tr);
    // the taps of every note the store holds, gathered as GatherSink's float16 path does (frame order, float32 sums)
    const int C = g.C;
    out.X.assign(static_cast<size_t>(g.N) * N_TAPS * C, 0.0f);
    std::vector<int> missing;
    std::vector<float> a1(C), a2(C);
    for (int i = 0; i < g.N; i++) {
        const Span& s = g.sp[i];
        bool ok = true;
        for (int t = s.t0; t < s.lim1 && ok; t++) {
            ok = st->hidden(t, s.lane) != nullptr;
        }
        for (int t = s.t0; t < s.lim2 && ok; t++) {
            ok = st->stem(t, s.lane) != nullptr;
        }
        if (!ok) {
            missing.push_back(i);
            continue;
        }
        std::fill(a1.begin(), a1.end(), 0.0f);
        std::fill(a2.begin(), a2.end(), 0.0f);
        float* x = out.X.data() + static_cast<size_t>(i) * N_TAPS * C;
        for (int t = s.t0; t < s.lim1; t++) {
            const uint16_t* h = st->hidden(t, s.lane);
            for (int c = 0; c < C; c++) {
                const float v = halfValue(h[c]);
                a1[c] += v;
                if (t == s.t0) {
                    x[c] = v;
                }
            }
        }
        for (int t = s.t0; t < s.lim2; t++) {
            const uint16_t* h = st->stem(t, s.lane);
            for (int c = 0; c < C; c++) {
                a2[c] += halfValue(h[c]);
            }
        }
        const float c1 = static_cast<float>(s.lim1 - s.t0), c2 = static_cast<float>(s.lim2 - s.t0);
        for (int c = 0; c < C; c++) {
            x[C + c] = a1[c] / c1;
            x[2 * C + c] = a2[c] / c2;
        }
    }
    const double t_gather = seconds(t_start);
    // the rest: exactly what features() does, over just the hop-0.5 windows that cover their frames
    const auto t_rec = std::chrono::steady_clock::now();
    int nwin = 0;
    const Transcriber::Layout lo = tr.layout(n, 0.5);
    if (!missing.empty()) {
        std::vector<char> only(lo.starts.size(), 0);
        std::vector<Span> sp;
        for (int i : missing) {
            sp.push_back(g.sp[i]);
        }
        for (size_t k = 0; k < lo.starts.size(); k++) {
            const int g0 = tr.frameOffset(lo.starts[k], lo.g_pad);
            for (const Span& s : sp) {
                if (s.t0 < g0 + lo.Fw && s.lim1 > g0) {
                    only[k] = 1;
                    break;
                }
            }
            nwin += only[k];
        }
        GatherSink sink(sp, C, g.T, g.L, g.wsum, true);
        Transcriber::RunOptions ro;
        ro.hop_frac = 0.5;
        ro.sink = &sink;
        ro.tap_mask = (1u << TapSink::HIDDEN) | (1u << TapSink::STEM);
        ro.only = &only;
        tr.run(y, n, ro);
        const auto X = sink.taps();
        for (size_t q = 0; q < missing.size(); q++) {
            std::copy(X.begin() + q * N_TAPS * C, X.begin() + (q + 1) * N_TAPS * C,
                      out.X.begin() + static_cast<size_t>(missing[q]) * N_TAPS * C);
        }
    }
    quantiseTaps(out, C);
    if (debug) {
        debug->store_notes_recomputed = static_cast<int>(missing.size());
        debug->store_windows_recomputed = nwin;
        debug->store_windows_total = static_cast<int>(lo.starts.size());
        debug->store_windows_seen = st->windows();
        debug->store_pairs_h = st->pairsH();
        debug->store_pairs_s = st->pairsS();
        debug->store_bytes = st->keptBytes();
        debug->store_peak_bytes = st->peakBytes();
        debug->t_store_gather = t_gather;
        debug->t_store_recompute = seconds(t_rec);
        debug->store_wsum_matches = true; // checked above
        std::set<int64_t> needH, needS;
        for (const Span& sp_ : g.sp) {
            for (int t = sp_.t0; t < sp_.lim1; t++) {
                needH.insert(static_cast<int64_t>(t) * g.L + sp_.lane);
            }
            for (int t = sp_.t0; t < sp_.lim2; t++) {
                needS.insert(static_cast<int64_t>(t) * g.L + sp_.lane);
            }
        }
        debug->store_need_h = needH.size();
        debug->store_need_s = needS.size();
    }
    return out;
}

// ------------------------------------------------------------------------------------------------------- retune
Grouper::Retune Grouper::retune(const float* x16, size_t n) const
{
    Retune rt;
    auto t = std::chrono::steady_clock::now();
    retune::ratio(x16, n, rt.delta, rt.strength, rt.shift, rt.r);
    if (debug) {
        debug->t_tuning = seconds(t);
    }
    t = std::chrono::steady_clock::now();
    if (debug && debug->inject_retuned) {
        rt.y = debug->retuned;
        rt.r = debug->r;
    } else if (rt.shift != 0.0 && !(debug && debug->exact_tuning_only)) {
        rt.y = resample(x16, n, SR, SR / rt.r, mBe.threads());
    } else {
        if (debug && debug->exact_tuning_only) {
            rt.r = 1.0;
        }
        rt.y.assign(x16, x16 + n);
    }
    if (debug) {
        debug->t_resample = seconds(t);
        debug->delta = rt.delta;
        debug->strength = rt.strength;
        debug->shift = rt.shift;
        debug->ratio = rt.r;
    }
    return rt;
}

// ------------------------------------------------------------------------------------------------------- predict
GroupPredictions Grouper::predict(const float* y, size_t n, const GroupFeatures& f) const
{
    const int N = f.n, K = mNet->nInst(), E = mNet->embDim(), D = mNet->d(), OD = mNet->outDim();
    const int TD = N_TAPS * mNet->tapDim();
    GroupPredictions P;
    P.n = N;
    P.K = K;
    P.E = E;
    P.D = D;
    P.W = mWindowS;
    if (N == 0) {
        return P;
    }
    // _Song: the audio as 16-bit samples
    std::vector<int16_t> pcm(n);
    for (size_t i = 0; i < n; i++) {
        pcm[i] = static_cast<int16_t>(std::clamp(std::nearbyint(y[i] * 32768.0f), -32768.0f, 32767.0f));
    }
    std::vector<float> on_all(N);
    for (int i = 0; i < N; i++) {
        on_all[i] = f.meta[static_cast<size_t>(i) * NC];
    }
    const double W = mWindowS;
    const auto jobs = songWindows(on_all, W, W * mWindowHop, mMaxNotes);
    const long L = std::lround((PAD0 + W + PAD1) * SR);
    const int Tf = static_cast<int>(L / GroupFront::HOP);
    const int C = mFront->channels(), F = mFront->bins(), NE = mFront->extBins();
    const size_t xs = static_cast<size_t>(C) * Tf * F;
    const int threads = std::max(1, mBe.threads());

    std::vector<double> acc(static_cast<size_t>(N) * K, 0.0), wsum(N, 0.0), best(N, -1.0), eacc(static_cast<size_t>(N) * E, 0.0);
    P.h.assign(static_cast<size_t>(N) * D, 0.0f);
    NoteNet::Inputs in;
    std::vector<float> heads, hidden;
    for (size_t bi = 0; bi < jobs.size(); bi += mBatch) {
        const int nb = static_cast<int>(std::min<size_t>(mBatch, jobs.size() - bi));
        mNet->resize(in, nb, mMaxNotes, Tf, *mFront);
        std::vector<std::vector<float>> Pw(nb, std::vector<float>(static_cast<size_t>(NE) * Tf));
        // the window's audio and the log HCQT, per window
        parallelFor(nb, threads, [&](int k) {
            const double t_a = jobs[bi + k].first - PAD0;
            const long s0 = static_cast<long>(std::nearbyint(t_a * SR));
            std::vector<float> x(L, 0.0f);
            const long lo = std::max(s0, 0L), hi = std::min(s0 + L, static_cast<long>(n));
            for (long i = lo; i < hi; i++) {
                x[i - s0] = static_cast<float>(pcm[i]) / 32768.0f;
            }
            mFront->logHcqt(x.data(), L, in.X.data() + k * xs, Pw[k].data());
        });
        // PCEN + standardisation, per window and channel
        parallelFor(nb * C, threads, [&](int q) {
            float* xc = in.X.data() + (q / C) * xs + static_cast<size_t>(q % C) * Tf * F;
            mFront->pcen(xc, 1, Tf, F);
            GroupFront::standardise(xc, 1, Tf, F);
        });
        // the notes and the readers' inputs
        parallelFor(nb, threads, [&](int k) {
            const double t0 = jobs[bi + k].first;
            const std::vector<int>& sel = jobs[bi + k].second;
            std::vector<int> rows = sel;
            if (rows.size() < 4) { // _Song.window returns None below 4 notes: the selection repeated
                rows.clear();
                for (int r = 0; r < 4; r++) {
                    rows.push_back(sel[r % sel.size()]);
                }
            }
            const int nr = static_cast<int>(rows.size());
            std::vector<float> nt(static_cast<size_t>(nr) * NC), ft(static_cast<size_t>(nr) * TD);
            const float t0f = static_cast<float>(t0), lim = static_cast<float>(W + PAD1);
            for (int j = 0; j < nr; j++) {
                const float* m = f.meta.data() + static_cast<size_t>(rows[j]) * NC;
                float* r = nt.data() + static_cast<size_t>(j) * NC;
                std::memcpy(r, m, sizeof(float) * NC);
                r[0] = m[0] - t0f;
                r[1] = std::min(m[1] - t0f, lim);
                for (int c = 0; c < TD; c++) {
                    const int tap = c / mNet->tapDim();
                    const float v = static_cast<float>(f.feat[static_cast<size_t>(rows[j]) * TD + c]) * f.fsc[static_cast<size_t>(rows[j]) * N_TAPS + tap];
                    ft[static_cast<size_t>(j) * TD + c] = roundHalf(v); // collate holds the taps as float16
                }
            }
            mNet->fillSlot(in, k, *mFront, nt.data(), nr, ft.data(), in.X.data() + k * xs, Pw[k].data());
        });
        mNet->run(in);
        mNet->heads(heads);
        mNet->hidden(hidden);
        // aggregate (Grouper.predict): centre-weighted means, h of the most central window
        for (int k = 0; k < nb; k++) {
            const double t0 = jobs[bi + k].first;
            const std::vector<int>& sel = jobs[bi + k].second;
            const int m = static_cast<int>(sel.size());
            GroupWindow gw;
            gw.t0 = t0;
            gw.sel = sel;
            gw.emb.resize(static_cast<size_t>(m) * E);
            gw.probs.resize(static_cast<size_t>(m) * K);
            const float t0f = static_cast<float>(t0), Wf = static_cast<float>(W);
            for (int j = 0; j < m; j++) {
                const float* o = heads.data() + (static_cast<size_t>(k) * mMaxNotes + j) * OD;
                // softmax of the instrument logits (float32)
                float mx = -INFINITY;
                for (int c = 0; c < K; c++) {
                    mx = std::max(mx, o[c]);
                }
                std::vector<float> p(K), e(E);
                double s = 0;
                for (int c = 0; c < K; c++) {
                    p[c] = std::exp(o[c] - mx);
                    s += p[c];
                }
                const float sf = static_cast<float>(s);
                for (int c = 0; c < K; c++) {
                    p[c] /= sf;
                }
                // F.normalize of the embedding
                double ss = 0;
                for (int c = 0; c < E; c++) {
                    ss += static_cast<double>(o[K + c]) * o[K + c];
                }
                const float nn = std::max(static_cast<float>(std::sqrt(ss)), 1e-12f);
                for (int c = 0; c < E; c++) {
                    e[c] = o[K + c] / nn;
                }
                const int i = sel[j];
                const float wv = std::clamp(1.0f - std::abs((on_all[i] - t0f) / Wf - 0.5f) * 2.0f, 0.05f, 1.0f);
                for (int c = 0; c < K; c++) {
                    acc[static_cast<size_t>(i) * K + c] += static_cast<double>(p[c] * wv);
                    gw.probs[static_cast<size_t>(j) * K + c] = roundHalf(p[c]);
                }
                wsum[i] += wv;
                for (int c = 0; c < E; c++) {
                    eacc[static_cast<size_t>(i) * E + c] += static_cast<double>(e[c] * wv);
                    gw.emb[static_cast<size_t>(j) * E + c] = roundHalf(e[c]);
                }
                if (wv > best[i]) {
                    const float* hh = hidden.data() + (static_cast<size_t>(k) * mMaxNotes + j) * D;
                    for (int c = 0; c < D; c++) {
                        P.h[static_cast<size_t>(i) * D + c] = roundHalf(hh[c]);
                    }
                    best[i] = wv;
                }
            }
            P.wins.push_back(std::move(gw));
        }
    }
    P.probs.resize(static_cast<size_t>(N) * K);
    P.emb.resize(static_cast<size_t>(N) * E);
    for (int i = 0; i < N; i++) {
        const double ws = std::max(wsum[i], 1e-9);
        for (int c = 0; c < K; c++) {
            P.probs[static_cast<size_t>(i) * K + c] = static_cast<float>(acc[static_cast<size_t>(i) * K + c] / ws);
        }
        // np.linalg.norm (pairwise sum of 32 squares)
        const double* e = eacc.data() + static_cast<size_t>(i) * E;
        std::vector<double> sq(E);
        for (int c = 0; c < E; c++) {
            sq[c] = e[c] * e[c];
        }
        double r8[8] = {0}, nrm2 = 0;
        if (E >= 8 && E <= 128) {
            for (int c = 0; c < 8; c++) {
                r8[c] = sq[c];
            }
            int c = 8;
            for (; c < E - (E % 8); c += 8) {
                for (int q = 0; q < 8; q++) {
                    r8[q] += sq[c + q];
                }
            }
            nrm2 = ((r8[0] + r8[1]) + (r8[2] + r8[3])) + ((r8[4] + r8[5]) + (r8[6] + r8[7]));
            for (; c < E; c++) {
                nrm2 += sq[c];
            }
        } else {
            for (int c = 0; c < E; c++) {
                nrm2 += sq[c];
            }
        }
        const double nn = std::max(std::sqrt(0.0 + nrm2), 1e-9);
        for (int c = 0; c < E; c++) {
            P.emb[static_cast<size_t>(i) * E + c] = static_cast<float>(e[c] / nn);
        }
    }
    return P;
}

// ------------------------------------------------------------------------------------------------------- tracks
std::vector<GroupTrack> Grouper::tracks(const GroupFeatures& f, const GroupPredictions& P, double r, GroupDecodeTrace* trace) const
{
    const int N = f.n;
    std::vector<double> on(N), off(N), pitch(N);
    std::vector<float> conf(N);
    for (int i = 0; i < N; i++) {
        const float* m = f.meta.data() + static_cast<size_t>(i) * NC;
        on[i] = m[0];
        off[i] = m[1];
        pitch[i] = std::nearbyint(m[2]);
        conf[i] = m[3] > 0.5f ? m[5] : m[4];
    }
    std::vector<std::pair<int, std::vector<int>>> groups;
    if (N) {
        groups = groupDecode(P, on, off, pitch, mTh, mDec, mCredit, trace);
    }
    if (debug) {
        debug->labels_inst.assign(N, -1);
        debug->labels_track.assign(N, -1);
        for (size_t g = 0; g < groups.size(); g++) {
            for (int i : groups[g].second) {
                debug->labels_inst[i] = groups[g].first;
                debug->labels_track[i] = static_cast<int64_t>(g);
            }
        }
    }
    std::stable_sort(groups.begin(), groups.end(), [](const auto& a, const auto& b) { return a.second.size() > b.second.size(); });
    std::vector<GroupTrack> out;
    for (const auto& [k, ids] : groups) {
        std::vector<double> onr(ids.size()), offr(ids.size());
        std::vector<int> o(ids.size());
        std::iota(o.begin(), o.end(), 0);
        for (size_t q = 0; q < ids.size(); q++) {
            onr[q] = on[ids[q]] * r;
            offr[q] = off[ids[q]] * r;
        }
        // same-pitch notes within DEDUP of the last kept one merge
        std::stable_sort(o.begin(), o.end(), [&](int a, int b) {
            if (pitch[ids[a]] != pitch[ids[b]]) {
                return pitch[ids[a]] < pitch[ids[b]];
            }
            return onr[a] < onr[b];
        });
        std::vector<int> keep;
        int last = -1;
        for (int q : o) {
            if (last >= 0 && pitch[ids[q]] == pitch[ids[last]] && onr[q] - onr[last] <= DEDUP) {
                continue;
            }
            keep.push_back(q);
            last = q;
        }
        std::stable_sort(keep.begin(), keep.end(), [&](int a, int b) { return onr[a] < onr[b]; });
        GroupTrack t;
        t.instrument = mNames[k];
        t.program = mPrograms[k];
        for (int q : keep) {
            const int i = ids[q];
            const float vel = std::clamp(20.0f + 107.0f * conf[i], 1.0f, 127.0f);
            t.notes.push_back({onr[q], std::max(offr[q], onr[q] + 0.02), pitch[i], static_cast<double>(static_cast<int>(vel))});
        }
        out.push_back(std::move(t));
    }
    return out;
}

// ------------------------------------------------------------------------------------------------------- group
std::vector<GroupTrack> Grouper::group(const float* audio16k, size_t n, const std::vector<std::array<double, 3>>& notes, Transcriber& tr,
                                  GroupInfo* info) const
{
    Retune rt = retune(audio16k, n);
    const double r = rt.r;
    const float* y = rt.y.data();
    const size_t ny = rt.y.size();
    if (debug) {
        debug->y = rt.y;
    }
    std::vector<std::array<double, 3>> nin = notes;
    for (auto& q : nin) { // onto the retuned time axis
        q[0] /= r;
        q[1] /= r;
    }
    auto t = std::chrono::steady_clock::now();
    GroupFeatures f = features(y, ny, nin, tr);
    if (debug) {
        debug->t_pass2 = seconds(t);
    }
    t = std::chrono::steady_clock::now();
    GroupPredictions P = predict(y, ny, f);
    if (debug) {
        debug->t_predict = seconds(t);
    }
    t = std::chrono::steady_clock::now();
    std::vector<GroupTrack> out = tracks(f, P, r, debug ? &debug->trace : nullptr);
    if (debug) {
        debug->t_decode = seconds(t);
        debug->features = std::move(f);
        debug->pred = std::move(P);
    }
    if (info != nullptr) {
        info->tuning_cents = pyRound(rt.delta, 2);
        info->tuning_strength = pyRound(rt.strength, 3);
        info->retune_cents = pyRound(rt.shift, 2);
        info->retune_ratio = r;
        info->notes_in = static_cast<int>(notes.size());
        int no = 0;
        for (const GroupTrack& tk : out) {
            no += static_cast<int>(tk.notes.size());
        }
        info->notes_out = no;
    }
    return out;
}

} // namespace dtt
