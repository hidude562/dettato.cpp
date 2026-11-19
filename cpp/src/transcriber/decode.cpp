#include "transcriber/decode.hpp"

#include <algorithm>
#include <numeric>
#include <tuple>

namespace dtt
{

namespace
{
    /** Sub-frame position of a peak at (t, l) from its two neighbours, clipped to +-0.5 frame (float32 arithmetic). */
    double refine(const float* p, int T, int L, int t, int l)
    {
        const float C = p[static_cast<size_t>(t) * L + l];
        const float A = t > 0 ? p[static_cast<size_t>(t - 1) * L + l] : C;
        const float Bv = t + 1 < T ? p[static_cast<size_t>(t + 1) * L + l] : C;
        const float den = 2.0f * (C - std::min(A, Bv));
        if (static_cast<double>(den) <= 1e-6) {
            return 0.0;
        }
        return static_cast<double>(std::clamp((Bv - A) / den, -0.5f, 0.5f));
    }

    /** Local maxima >= th along time (plateaus: first frame), per lane. */
    std::vector<char> peaks(const float* p, int T, int L, float th)
    {
        std::vector<char> m(static_cast<size_t>(T) * L, 0);
        for (int t = 0; t < T; t++) {
            for (int l = 0; l < L; l++) {
                const float v = p[static_cast<size_t>(t) * L + l];
                const float left = t > 0 ? p[static_cast<size_t>(t - 1) * L + l] : -1.0f;
                const float right = t + 1 < T ? p[static_cast<size_t>(t + 1) * L + l] : -1.0f;
                m[static_cast<size_t>(t) * L + l] = (v >= th) && (v > left) && (v >= right);
            }
        }
        return m;
    }
} // namespace

std::vector<DecodedNote> decodeVoice(const float* on, const float* fr, const float* off, int T, int L, double hop,
                                     const DecodeParams& p, int pitch_lo)
{
    const auto pk_on = peaks(on, T, L, p.on_th);
    const auto pk_off = peaks(off, T, L, p.off_th);
    // run[t] = fr below fr_th for fr_low_frames frames from t (the last frames keep fewer, as numpy's slicing does)
    std::vector<char> low(static_cast<size_t>(T) * L), run;
    for (size_t i = 0; i < low.size(); i++) {
        low[i] = fr[i] < p.fr_th;
    }
    run = low;
    for (int k = 1; k < p.fr_low_frames; k++) {
        for (int t = 0; t < T - k; t++) {
            for (int l = 0; l < L; l++) {
                run[static_cast<size_t>(t) * L + l] &= low[static_cast<size_t>(t + k) * L + l];
            }
        }
    }

    std::vector<DecodedNote> notes;
    std::vector<int> ons, offs, lows, order;
    std::vector<char> keep;
    for (int l = 0; l < L; l++) {
        ons.clear();
        for (int t = 0; t < T; t++) {
            if (pk_on[static_cast<size_t>(t) * L + l]) {
                ons.push_back(t);
            }
        }
        if (ons.empty()) {
            continue;
        }
        if (p.on_nms > 0 && ons.size() > 1) {
            keep.assign(ons.size(), 1);
            order.resize(ons.size());
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
                return on[static_cast<size_t>(ons[a]) * L + l] > on[static_cast<size_t>(ons[b]) * L + l];
            });
            for (int oi : order) {
                if (!keep[oi]) {
                    continue;
                }
                for (size_t j = 0; j < ons.size(); j++) {
                    if (static_cast<int>(j) != oi && std::abs(ons[j] - ons[oi]) <= p.on_nms) {
                        keep[j] = 0;
                    }
                }
            }
            std::vector<int> kept;
            for (size_t j = 0; j < ons.size(); j++) {
                if (keep[j]) {
                    kept.push_back(ons[j]);
                }
            }
            ons.swap(kept);
        }
        offs.clear();
        lows.clear();
        for (int t = 0; t < T; t++) {
            if (pk_off[static_cast<size_t>(t) * L + l]) {
                offs.push_back(t);
            }
            if (run[static_cast<size_t>(t) * L + l]) {
                lows.push_back(t);
            }
        }
        for (size_t i = 0; i < ons.size(); i++) {
            const int t0 = ons[i];
            const double u0 = t0 + refine(on, T, L, t0, l);
            const int nxt = i + 1 < ons.size() ? ons[i + 1] : T;
            auto j = std::lower_bound(lows.begin(), lows.end(), t0 + 1);
            const int e_fr = j != lows.end() ? *j : T;
            j = std::lower_bound(offs.begin(), offs.end(), t0 + p.off_min_frames);
            const int e_off = j != offs.end() ? *j : T;
            double u1;
            if (e_off < T && e_off <= std::min(e_fr, nxt)) {
                u1 = e_off + refine(off, T, L, e_off, l);
            } else if (e_fr <= nxt) {
                u1 = e_fr - 0.5;
            } else {
                u1 = nxt + (nxt < T ? refine(on, T, L, nxt, l) : 0.0);
            }
            const double on_s = u0 * hop;
            const double off_s = std::max(u1 * hop, u0 * hop + p.min_dur);
            notes.push_back({on_s, off_s, pitch_lo + l});
        }
    }
    std::sort(notes.begin(), notes.end(), [](const DecodedNote& a, const DecodedNote& b) {
        return std::tie(a.onset, a.offset, a.pitch) < std::tie(b.onset, b.offset, b.pitch);
    });
    return notes;
}

} // namespace dtt
