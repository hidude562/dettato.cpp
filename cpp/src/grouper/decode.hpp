#pragma once

// The grouper's decode (dettato/grouper/decode.py), in double precision on the CPU: per-window average-linkage clusters
// chained across windows (Hungarian matching), size-weighted agglomeration of the tracklets, k-means refines, small-track
// folding, absorb_small, track-head naming with expected credit, merging. scipy's linkage / fcluster /
// linear_sum_assignment are ported so ties break the same way.

#include <cstdint>
#include <utility>
#include <vector>

namespace dtt
{

/** One window's predictions as Grouper.predict keeps them in `wins` (values already rounded through float16). */
struct GroupWindow {
    double t0 = 0;
    std::vector<int> sel;      // note indices of the window
    std::vector<float> emb;    // (m, E) row-major
    std::vector<float> probs;  // (m, K) row-major
};

/** Grouper.predict's output. */
struct GroupPredictions {
    int n = 0, K = 0, E = 0, D = 0;
    double W = 0;              // window length, s
    std::vector<float> probs;  // (n, K) float32
    std::vector<float> emb;    // (n, E) float32 unit vectors
    std::vector<float> h;      // (n, D), values rounded through float16 (Python keeps them as float16)
    std::vector<GroupWindow> wins;
};

struct GroupDecodeParams {
    double tau = 0.1, tau2 = 0.4, lam = 0.5;
    int min_notes = 30, refine = 2;
    double alpha = 0.5;
    bool ec = true, merge = true;
    double absorb = 0.3, absorb_frac = 0.25;
};

/** One Linear layer of the track head, applied in float64: y = w x + b, w (out, in) row-major. */
struct TrackHeadLayer {
    int out = 0, in = 0;
    std::vector<double> w, b;
};

/** The decode's intermediate labels, for the parity tests (the Python variables of the same names). */
struct GroupDecodeTrace {
    std::vector<int64_t> tracklets; // (n,) group_chain
    std::vector<int64_t> agglo;     // (n,) group_tracklet
    std::vector<int64_t> idx;       // notes with a track
    std::vector<int64_t> refined;   // (len idx,) inv after the refine passes
    std::vector<int64_t> folded;    // inv after folding the small tracks
    std::vector<int64_t> absorbed;  // inv after absorb_small
};

/**
 * decode(P, (on, off, pitch), th, **params) -> [(instrument index, sorted note ids)] in the order Python's dict yields them.
 * credit: (K, K) row-major CREDIT matrix (predicted, true).
 */
std::vector<std::pair<int, std::vector<int>>> groupDecode(const GroupPredictions& P, const std::vector<double>& on,
                                                          const std::vector<double>& off, const std::vector<double>& pitch,
                                                          const std::vector<TrackHeadLayer>& th, const GroupDecodeParams& prm,
                                                          const std::vector<double>& credit, GroupDecodeTrace* trace = nullptr);

/** scipy fcluster(linkage(y, "average"), t, "distance") - 1. y: condensed distances of m points (m (m - 1) / 2). */
std::vector<int> fclusterAverage(const std::vector<double>& y, int m, double t);

/** scipy linear_sum_assignment(cost) (minimise), cost (nr, nc) row-major -> (row_ind, col_ind). */
void linearSumAssignment(const std::vector<double>& cost, int nr, int nc, std::vector<int>& rows, std::vector<int>& cols);

/**
 * The condensed distances decode._cluster hands to scipy for one window (float32 arithmetic as numpy does it, returned as
 * float64): 1 - e e^T + lam (1 - p p^T) with p row-normalised, clipped at 0.
 */
std::vector<double> groupWindowDistances(const GroupWindow& w, int E, int K, double lam);

/** np.argsort(a) as numpy 2.4 runs it on an AVX2 x86-64 CPU (Intel x86-simd-sort; not stable), NaN-free float64. */
std::vector<int64_t> npArgsort(const std::vector<double>& a);

/** scipy.special.erf (Cephes' erf, which is not the C library's). */
double scipyErf(double x);

} // namespace dtt
