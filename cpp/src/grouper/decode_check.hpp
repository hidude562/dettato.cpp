#pragma once

// Checks of the grouper decode (grouper/decode.hpp) against scipy / numpy / the Python decode. Header-only; include it
// after "npy.hpp" (cpp/tools), which provides npy::load.
//
//   checkClusterCases(dir)       random cases written by py/dump_refs_grouper.py into testdata/refs_grouper/cases/:
//                                fcluster(linkage(y, "average"), t, "distance"), linear_sum_assignment, np.argsort, erf
//   checkDecodeRefs(fixture_dir) groupDecode on <fixture>/pred + <fixture>/decode/{th_*, credit, params}, every
//                                intermediate label array and the final groups compared exactly with the reference

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "grouper/decode.hpp"

namespace dtt::decode_check
{

inline std::vector<int64_t> i64(const npy::Array& a)
{
    return std::vector<int64_t>(a.data.begin(), a.data.end());
}

inline bool exists(const std::string& p)
{
    return std::filesystem::exists(p);
}

/** Exact comparison of two integer arrays; prints the first differences. Returns 1 on mismatch. */
inline int compareLabels(const char* name, const std::vector<int64_t>& ours, const std::vector<int64_t>& ref)
{
    size_t diff = 0, first = SIZE_MAX;
    const size_t n = std::min(ours.size(), ref.size());
    for (size_t i = 0; i < n; i++) {
        if (ours[i] != ref[i]) {
            diff++;
            if (first == SIZE_MAX) {
                first = i;
            }
        }
    }
    const bool ok = ours.size() == ref.size() && diff == 0;
    std::printf("  %-10s %7zu values  %s", name, ref.size(), ok ? "exact" : "MISMATCH");
    if (!ok) {
        std::printf(" (sizes ours %zu ref %zu, %zu differ", ours.size(), ref.size(), diff);
        if (first != SIZE_MAX) {
            std::printf(", first at %zu: ours %lld ref %lld", first, static_cast<long long>(ours[first]),
                        static_cast<long long>(ref[first]));
        }
        std::printf(")");
    }
    std::printf("\n");
    return ok ? 0 : 1;
}

inline int checkClusterCases(const std::string& dir)
{
    int fails = 0;
    std::printf("decode: random scipy / numpy cases (%s)\n", dir.c_str());
    {   // fcluster(linkage(y, "average"), t, "distance") - 1
        const auto m = i64(npy::load(dir + "/cluster_m.npy")), yoff = i64(npy::load(dir + "/cluster_yoff.npy")),
                   loff = i64(npy::load(dir + "/cluster_laboff.npy")), lab = i64(npy::load(dir + "/cluster_lab.npy"));
        const auto y = npy::load(dir + "/cluster_y.npy");
        const auto t = npy::load(dir + "/cluster_t.npy");
        const int nt = static_cast<int>(t.shape[1]);
        size_t bad = 0, total = 0;
        for (size_t c = 0; c < m.size(); c++) {
            const std::vector<double> yc(y.data.begin() + yoff[c], y.data.begin() + yoff[c + 1]);
            for (int q = 0; q < nt; q++) {
                const std::vector<int> ours = fclusterAverage(yc, static_cast<int>(m[c]), t.data[c * nt + q]);
                bool ok = static_cast<int64_t>(ours.size()) == m[c];
                for (int64_t i = 0; ok && i < m[c]; i++) {
                    ok = ours[i] == lab[loff[c] + q * m[c] + i];
                }
                total++;
                if (!ok) {
                    if (bad < 5) {
                        std::printf("  fcluster case %zu (m %lld, t %.17g) differs\n", c, static_cast<long long>(m[c]),
                                    t.data[c * nt + q]);
                    }
                    bad++;
                }
            }
        }
        std::printf("  fcluster(average): %zu / %zu cases exact\n", total - bad, total);
        fails += bad != 0;
    }
    {   // linear_sum_assignment
        const auto shp = i64(npy::load(dir + "/lsap_shape.npy")), coff = i64(npy::load(dir + "/lsap_coff.npy")),
                   roff = i64(npy::load(dir + "/lsap_roff.npy")), rr = i64(npy::load(dir + "/lsap_rows.npy")),
                   cc = i64(npy::load(dir + "/lsap_cols.npy"));
        const auto cost = npy::load(dir + "/lsap_cost.npy");
        size_t bad = 0;
        const size_t nc = shp.size() / 2;
        for (size_t c = 0; c < nc; c++) {
            const std::vector<double> cm(cost.data.begin() + coff[c], cost.data.begin() + coff[c + 1]);
            std::vector<int> r, k;
            linearSumAssignment(cm, static_cast<int>(shp[2 * c]), static_cast<int>(shp[2 * c + 1]), r, k);
            bool ok = static_cast<int64_t>(r.size()) == roff[c + 1] - roff[c];
            for (size_t i = 0; ok && i < r.size(); i++) {
                ok = r[i] == rr[roff[c] + i] && k[i] == cc[roff[c] + i];
            }
            if (!ok) {
                if (bad < 5) {
                    std::printf("  lsap case %zu (%lld x %lld) differs\n", c, static_cast<long long>(shp[2 * c]),
                                static_cast<long long>(shp[2 * c + 1]));
                }
                bad++;
            }
        }
        std::printf("  linear_sum_assignment: %zu / %zu cases exact\n", nc - bad, nc);
        fails += bad != 0;
    }
    {   // np.argsort (default kind)
        const auto off = i64(npy::load(dir + "/argsort_off.npy")), ref = i64(npy::load(dir + "/argsort_ref.npy"));
        const auto x = npy::load(dir + "/argsort_x.npy");
        size_t bad = 0;
        const size_t nc = off.size() - 1;
        for (size_t c = 0; c < nc; c++) {
            const std::vector<double> a(x.data.begin() + off[c], x.data.begin() + off[c + 1]);
            const std::vector<int64_t> o = npArgsort(a);
            bool ok = true;
            for (size_t i = 0; ok && i < o.size(); i++) {
                ok = o[i] == ref[off[c] + i];
            }
            if (!ok) {
                if (bad < 5) {
                    std::printf("  argsort case %zu (n %zu) differs\n", c, a.size());
                }
                bad++;
            }
        }
        std::printf("  np.argsort: %zu / %zu cases exact\n", nc - bad, nc);
        fails += bad != 0;
    }
    {   // scipy.special.erf
        const auto x = npy::load(dir + "/erf_x.npy"), ref = npy::load(dir + "/erf_ref.npy");
        size_t bad = 0;
        for (size_t i = 0; i < x.size(); i++) {
            const double v = scipyErf(x.data[i]);
            if (std::memcmp(&v, &ref.data[i], sizeof(double)) != 0) {
                bad++;
            }
        }
        std::printf("  scipy.special.erf: %zu / %zu values bit-exact\n", x.size() - bad, x.size());
        fails += bad != 0;
    }
    return fails;
}

/** Loads Grouper.predict's outputs as py/dump_refs_grouper.py writes them (<fixture>/pred). */
inline GroupPredictions loadPredictions(const std::string& pd, std::vector<double>& on, std::vector<double>& off,
                                        std::vector<double>& pitch)
{
    GroupPredictions P;
    const auto probs = npy::load(pd + "/probs.npy"), emb = npy::load(pd + "/emb.npy"), h = npy::load(pd + "/h.npy");
    P.n = static_cast<int>(probs.shape[0]);
    P.K = static_cast<int>(probs.shape[1]);
    P.E = static_cast<int>(emb.shape[1]);
    P.D = static_cast<int>(h.shape[1]);
    P.W = npy::load(pd + "/W.npy").data[0];
    P.probs = probs.f32();
    P.emb = emb.f32();
    P.h = h.f32();
    const auto t0 = npy::load(pd + "/win_t0.npy");
    const auto woff = i64(npy::load(pd + "/win_off.npy")), sel = i64(npy::load(pd + "/win_sel.npy"));
    const auto we = npy::load(pd + "/win_emb.npy"), wp = npy::load(pd + "/win_probs.npy");
    for (size_t w = 0; w + 1 < woff.size(); w++) {
        GroupWindow g;
        g.t0 = t0.data[w];
        for (int64_t i = woff[w]; i < woff[w + 1]; i++) {
            g.sel.push_back(static_cast<int>(sel[i]));
        }
        g.emb.assign(we.data.begin() + woff[w] * P.E, we.data.begin() + woff[w + 1] * P.E);
        g.probs.assign(wp.data.begin() + woff[w] * P.K, wp.data.begin() + woff[w + 1] * P.K);
        P.wins.push_back(std::move(g));
    }
    on = npy::load(pd + "/notes_on.npy").data;
    off = npy::load(pd + "/notes_off.npy").data;
    pitch = npy::load(pd + "/notes_pitch.npy").data;
    return P;
}

inline int checkDecodeRefs(const std::string& fixture_dir)
{
    const std::string pd = fixture_dir + "/pred", dd = fixture_dir + "/decode";
    std::vector<double> on, off, pitch;
    const GroupPredictions P = loadPredictions(pd, on, off, pitch);
    std::vector<TrackHeadLayer> th;
    for (int i = 0; exists(dd + "/th_w" + std::to_string(i) + ".npy"); i++) {
        const auto w = npy::load(dd + "/th_w" + std::to_string(i) + ".npy"), b = npy::load(dd + "/th_b" + std::to_string(i) + ".npy");
        TrackHeadLayer L;
        L.out = static_cast<int>(w.shape[0]);
        L.in = static_cast<int>(w.shape[1]);
        L.w = w.data;
        L.b = b.data;
        th.push_back(std::move(L));
    }
    const auto credit = npy::load(dd + "/credit.npy").data;
    const auto pv = npy::load(dd + "/params.npy").data;
    GroupDecodeParams prm;
    prm.tau = pv[0];
    prm.tau2 = pv[1];
    prm.lam = pv[2];
    prm.min_notes = static_cast<int>(pv[3]);
    prm.refine = static_cast<int>(pv[4]);
    prm.alpha = pv[5];
    prm.ec = pv[6] != 0;
    prm.merge = pv[7] != 0;
    prm.absorb = pv[8];
    prm.absorb_frac = pv[9];
    std::printf("decode on the reference predictions (%s): %d notes, %zu windows\n", fixture_dir.c_str(), P.n, P.wins.size());
    int fails = 0;
    if (exists(dd + "/dist.npy")) { // float32 distances: bit agreement with numpy (OpenBLAS SSYRK)
        const auto doff = i64(npy::load(dd + "/dist_off.npy"));
        const auto dist = npy::load(dd + "/dist.npy");
        std::vector<size_t> order(P.wins.size());
        for (size_t i = 0; i < order.size(); i++) {
            order[i] = i;
        }
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return P.wins[a].t0 < P.wins[b].t0; });
        size_t same = 0, total = 0;
        double maxrel = 0;
        for (size_t q = 0; q < order.size(); q++) {
            const GroupWindow& w = P.wins[order[q]];
            if (w.sel.size() < 2) {
                continue;
            }
            const std::vector<double> y = groupWindowDistances(w, P.E, P.K, prm.lam);
            for (size_t i = 0; i < y.size(); i++) {
                const double r = dist.data[doff[q] + i];
                same += y[i] == r;
                total++;
                if (r != 0) {
                    maxrel = std::max(maxrel, std::abs(y[i] - r) / std::abs(r));
                }
            }
        }
        std::printf("  float32 window distances: %zu / %zu bit-equal to numpy's (%.4f%%), max rel diff %.2e\n", same, total,
                    total ? 100.0 * same / total : 100.0, maxrel);
    }
    GroupDecodeTrace tr;
    const auto groups = groupDecode(P, on, off, pitch, th, prm, credit, &tr);
    fails += compareLabels("tracklets", tr.tracklets, i64(npy::load(dd + "/tracklets.npy")));
    fails += compareLabels("agglo", tr.agglo, i64(npy::load(dd + "/agglo.npy")));
    fails += compareLabels("idx", tr.idx, i64(npy::load(dd + "/idx.npy")));
    fails += compareLabels("refined", tr.refined, i64(npy::load(dd + "/refined.npy")));
    fails += compareLabels("folded", tr.folded, i64(npy::load(dd + "/folded.npy")));
    fails += compareLabels("absorbed", tr.absorbed, i64(npy::load(dd + "/absorbed.npy")));
    std::vector<int64_t> gk, goff{0}, gids;
    for (const auto& [k, ids] : groups) {
        gk.push_back(k);
        gids.insert(gids.end(), ids.begin(), ids.end());
        goff.push_back(static_cast<int64_t>(gids.size()));
    }
    fails += compareLabels("groups_k", gk, i64(npy::load(dd + "/groups_k.npy")));
    fails += compareLabels("groups_off", goff, i64(npy::load(dd + "/groups_off.npy")));
    fails += compareLabels("groups_ids", gids, i64(npy::load(dd + "/groups_ids.npy")));
    std::printf("  -> %zu tracks:", groups.size());
    for (const auto& [k, ids] : groups) {
        std::printf(" %d(%zu)", k, ids.size());
    }
    std::printf("  %s\n", fails ? "MISMATCH" : "identical to the reference");
    return fails;
}

} // namespace dtt::decode_check
