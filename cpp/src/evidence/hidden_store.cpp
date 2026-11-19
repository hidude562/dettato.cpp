#include "evidence/hidden_store.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>

#include "common.hpp"

namespace dtt::ev
{

namespace
{
    constexpr size_t POOL_CHUNK = 1024; // vectors per pool chunk
    // flags per frame and lane; voice 2 = voice 1 << 1
    enum : uint8_t { PON = 1, POFF = 4, HOT = 16, LOW = 64 };

    /** f(begin, end) over [0, n) split across up to nth threads (each item independent: results do not depend on nth). */
    template <class F>
    void parallelFor(int n, int nth, const F& f)
    {
        nth = std::max(1, std::min(nth, n));
        if (nth == 1) {
            f(0, n);
            return;
        }
        std::vector<std::thread> th;
        const int per = (n + nth - 1) / nth;
        for (int i = 1; i < nth; i++) {
            const int a = i * per, b = std::min(n, a + per);
            if (a < b) {
                th.emplace_back([&f, a, b] { f(a, b); });
            }
        }
        f(0, std::min(n, per));
        for (auto& t : th) {
            t.join();
        }
    }

    double seconds(std::chrono::steady_clock::time_point t0)
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
} // namespace

HiddenStore::HiddenStore(const DecodeThresholds& th, int margin, int threads) : mTh(th), mM(margin)
{
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    mThreads = threads > 0 ? threads : std::clamp(hw / 2, 1, 8);
}

HiddenStore::~HiddenStore() = default;

void HiddenStore::ensureRing(int upto)
{
    if (upto - mBase <= mR) {
        return;
    }
    const int R = upto - mBase + 16;
    const size_t fs = static_cast<size_t>(mL) * mC;
    std::vector<float> ring(static_cast<size_t>(R) * fs, 0.0f);
    for (int t = mBase; t < mTouched; t++) {
        std::memcpy(ring.data() + static_cast<size_t>(t % R) * fs, slot(t), fs * sizeof(float));
    }
    mRing.swap(ring);
    mR = R;
    mRingPeak = std::max(mRingPeak, static_cast<size_t>(mR));
}

void HiddenStore::window(int tap, int g0, int a, int b, const float* w, const float* data, int C, int L, int Fw)
{
    if (tap != HIDDEN || b <= a) {
        return;
    }
    const auto t_start = std::chrono::steady_clock::now();
    if (mC == 0) {
        mC = C;
        mL = L;
        mSum.assign(static_cast<size_t>(L) * C, 0.0);
        mPrevLast.assign(L, 0);
        mLanes.resize(L);
        mH.resize(static_cast<size_t>(L) * C);
        mCs.resize(C);
    }
    DTT_CHECK(C == mC && L == mL, "hidden tap shape changed");
    DTT_CHECK(a >= mBase, "window frames [%d, %d) before the finalised frame %d", a, b, mBase);
    ensureRing(b);
    mTouched = std::max(mTouched, b);
    const size_t fs = static_cast<size_t>(L) * C;
    parallelFor(b - a, mThreads, [&](int i0, int i1) {
        for (int t = a + i0; t < a + i1; t++) {
            // Transcriber.run: buf[a:b] += (v * wt) per window, in window order (here in float32)
            const float wt = w[t - a];
            const float* src = data + fs * static_cast<size_t>(t - g0);
            float* dst = slot(t);
            for (size_t i = 0; i < fs; i++) {
                dst[i] += src[i] * wt;
            }
        }
    });
    mAccumS += seconds(t_start);
}

uint8_t HiddenStore::flag(int t, int l) const
{
    // outside the song: no onset / frame / offset above threshold
    return (t < 0 || t >= mT) ? 0 : mFlags[static_cast<size_t>(t) * mL + l];
}

void HiddenStore::computeFlags(int upto, const Maps& maps)
{
    // a frame's peak flags need its next frame: flag frames t < upto, upto <= the first non-final row - 1 (or T)
    const bool v2 = mTh.v2_th >= 0.0f;
    auto peak = [&](Head h, int t, int l, float th) {
        // decode.py _peaks: p >= th & p > left & p >= right, with -1 beyond the song
        const float p = maps.at(h, t, l);
        const float left = t > 0 ? maps.at(h, t - 1, l) : -1.0f;
        const float right = t + 1 < mT ? maps.at(h, t + 1, l) : -1.0f;
        return p >= th && p > left && p >= right;
    };
    for (int t = mFlagged; t < upto; t++) {
        for (int l = 0; l < mL; l++) {
            uint8_t f = 0;
            for (int v = 0; v < 2; v++) {
                if (v == 1 && !v2) {
                    continue;
                }
                const Head on = v ? ONSET2 : ONSET, fr = v ? FRAME2 : FRAME, off = v ? OFFSET2 : OFFSET;
                const float th = v ? mTh.v2_th : mTh.on_th;
                f |= (peak(on, t, l, th) ? PON : 0) << v;
                f |= (peak(off, t, l, mTh.off_th) ? POFF : 0) << v;
                f |= ((maps.at(on, t, l) >= th || maps.at(fr, t, l) >= mTh.fr_th) ? HOT : 0) << v;
                f |= (maps.at(fr, t, l) < mTh.fr_th ? LOW : 0) << v;
            }
            mFlags[static_cast<size_t>(t) * mL + l] = f;
        }
    }
    mFlagged = std::max(mFlagged, upto);
}

bool HiddenStore::candA(int t, int l) const
{
    for (int s = t - 1 - mM; s <= t + 1 + mM; s++) {
        if (flag(s, l) & (PON | PON << 1)) {
            return true;
        }
    }
    return false;
}

bool HiddenStore::run(int t, int l, int v) const
{
    // decode.py: run[t] = fr < fr_th at t and t + 1 (only at t for the last frame)
    if (t < 0 || t >= mT) {
        return false;
    }
    const uint8_t low = LOW << v;
    return (flag(t, l) & low) && (t == mT - 1 || (flag(t + 1, l) & low));
}

bool HiddenStore::candLast(int u, int l) const
{
    if (u < 0) {
        return false;
    }
    for (int v = 0; v < 2; v++) {
        const uint8_t pon = PON << v, poff = POFF << v, hot = HOT << v;
        bool in_note = false;
        for (int s = u - 1 - mM; s <= u + 1 + mM && !in_note; s++) {
            in_note = (flag(s, l) & hot) != 0;
        }
        if (!in_note) {
            continue;
        }
        if (u >= mT - 2 - mM) {
            return true;
        }
        for (int s = u - 1 - mM; s <= u + 2 + mM; s++) {
            const uint8_t f = flag(s, l);
            if ((f & pon) || (s >= u - mM && (f & poff)) || (s >= u + 1 - mM && run(s, l, v))) {
                return true;
            }
        }
    }
    return false;
}

uint32_t HiddenStore::put(const float* v)
{
    const size_t i = mPoolUsed++;
    if (i / POOL_CHUNK >= mPool.size()) {
        mPool.emplace_back(new float[POOL_CHUNK * mC]);
    }
    std::memcpy(mPool[i / POOL_CHUNK].get() + (i % POOL_CHUNK) * mC, v, mC * sizeof(float));
    return static_cast<uint32_t>(i);
}

void HiddenStore::process(int t, const std::vector<float>& wsum)
{
    const size_t fs = static_cast<size_t>(mL) * mC;
    float* acc = slot(t);
    const float ws = std::max(wsum[t], 1e-6f);
    float* cs = mCs.data();
    for (int l = 0; l < mL; l++) {
        float* h = mH.data() + static_cast<size_t>(l) * mC;
        const float* a = acc + static_cast<size_t>(l) * mC;
        for (int c = 0; c < mC; c++) {
            h[c] = a[c] / ws; // buf /= ws
        }
        const bool ca = candA(t, l);
        const bool cl = candLast(t - 1, l);
        double* sum = mSum.data() + static_cast<size_t>(l) * mC;
        Lane& ln = mLanes[l];
        if (ca || cl) {
            for (int c = 0; c < mC; c++) {
                cs[c] = static_cast<float>(sum[c]);
            }
            ln.p_t.push_back(t);
            ln.p_i.push_back(put(cs));
        }
        if (ca) {
            ln.h_t.push_back(t);
            ln.h_i.push_back(put(h));
        }
        for (int c = 0; c < mC; c++) {
            sum[c] += h[c]; // torch cumsum: float64 accumulator
        }
    }
    std::memset(acc, 0, fs * sizeof(float));
}

void HiddenStore::framesFinal(int t0, int t1, const Maps& maps, const std::vector<float>& wsum)
{
    if (mC == 0) {
        return; // no hidden tap was delivered
    }
    const auto t_start = std::chrono::steady_clock::now();
    if (mT < 0) {
        mT = maps.T;
        mFlags.assign(static_cast<size_t>(mT) * mL, 0);
    }
    DTT_CHECK(maps.T == mT && maps.L == mL, "maps shape changed");
    const bool last = t1 >= mT;
    computeFlags(last ? mT : t1 - 1, maps);
    // frame t reads the flags of frames up to t + 2 + m (run: + 1), flagged up to t1 - 2
    const int upto = last ? mT : std::max(mBase, t1 - 4 - mM);
    for (int t = mBase; t < upto; t++) {
        process(t, wsum);
    }
    mBase = upto;
    if (last && !mDone) {
        // cs[T] (B = T) for lanes whose last frame can end a note
        std::vector<float> cs(mC);
        for (int l = 0; l < mL; l++) {
            if (candLast(mT - 1, l)) {
                const double* sum = mSum.data() + static_cast<size_t>(l) * mC;
                for (int c = 0; c < mC; c++) {
                    cs[c] = static_cast<float>(sum[c]);
                }
                mLanes[l].p_t.push_back(mT);
                mLanes[l].p_i.push_back(put(cs.data()));
            }
        }
        mDone = true;
        // the accumulation ring is no longer needed
        std::vector<float>().swap(mRing);
        mR = 0;
    }
    mFinalS += seconds(t_start);
}

const float* HiddenStore::find(const std::vector<int>& ts, const std::vector<uint32_t>& is, int t) const
{
    const auto it = std::lower_bound(ts.begin(), ts.end(), t);
    if (it == ts.end() || *it != t) {
        return nullptr;
    }
    const size_t i = is[it - ts.begin()];
    return mPool[i / POOL_CHUNK].get() + (i % POOL_CHUNK) * mC;
}

const float* HiddenStore::hidden(int t, int l) const
{
    return (l < 0 || l >= mL) ? nullptr : find(mLanes[l].h_t, mLanes[l].h_i, t);
}

const float* HiddenStore::prefix(int t, int l) const
{
    return (l < 0 || l >= mL) ? nullptr : find(mLanes[l].p_t, mLanes[l].p_i, t);
}

HiddenStore::Stats HiddenStore::stats() const
{
    Stats s;
    s.ring_frames = mRingPeak;
    s.ring_bytes = mRingPeak * static_cast<size_t>(mL) * mC * sizeof(float);
    for (const Lane& ln : mLanes) {
        s.n_hidden += ln.h_t.size();
        s.n_prefix += ln.p_t.size();
        s.store_bytes += (ln.h_t.capacity() + ln.p_t.capacity()) * sizeof(int) +
                         (ln.h_i.capacity() + ln.p_i.capacity()) * sizeof(uint32_t);
    }
    s.store_bytes += mPool.size() * POOL_CHUNK * mC * sizeof(float);
    s.sums_bytes = mSum.size() * sizeof(double) + mFlags.size() + mH.size() * sizeof(float);
    s.accum_s = mAccumS;
    s.final_s = mFinalS;
    return s;
}

} // namespace dtt::ev
