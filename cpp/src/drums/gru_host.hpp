#pragma once

#include <string>
#include <vector>

namespace dtt
{

class ModelFile;

/**
 * The recurrent half of one bidirectional nn.GRU layer, run on the host over a whole song (thousands of steps: unrolling
 * them in a ggml graph would cost ~20 nodes per step). The input projections W_ih x + b_ih come precomputed (one matmul
 * in the graph); each step is PyTorch's GRUCell:
 *
 *     gh = W_hh h + b_hh
 *     r = sigmoid(gh_r + gi_r), z = sigmoid(gh_z + gi_z), n = tanh(gi_n + gh_n * r), h' = (h - n) * z + n
 */
class DrumGru
{
public:
    DrumGru() = default;
    /** From the converter's <name>.hh.w (2, 3H, H) and <name>.hh.b (2, 3H). */
    DrumGru(const ModelFile& mf, const std::string& name);

    int hidden() const { return mH; }

    /**
     * One direction over N steps. gi: frame t's 3H gate inputs at gi + t * gi_stride (+ dir * 3H for the converter's stacked
     * layout); out: frame t's H outputs at out + t * out_stride. dir 1 runs t = N-1 .. 0 (PyTorch's reverse direction).
     */
    void run(int dir, const float* gi, int gi_stride, int N, float* out, int out_stride) const;

    /** Both directions in two threads: gi (N, 6H) [forward 3H | reverse 3H], out (N, 2H) [forward H | reverse H]. */
    void runBoth(const float* gi, int N, float* out) const;

private:
    int mH = 0;
    std::vector<float> mWt[2]; // W_hh transposed, (H, 3H): the step is an axpy over contiguous rows
    std::vector<float> mB[2];  // b_hh (3H)
};

/**
 * DrumNet's BandGRU on the host (for the CPU backend, where its 20 unrolled steps would be ~700 graph nodes, each a
 * thread barrier): per frame, a bidirectional GRU (hidden m) across the F bands, then proj([h_fwd, h_rev, mean of the
 * outputs]) -> W values. Weights as the converter stores them for the graph version: <p>.bgru.hh{f,r}.w (3m, m),
 * .hh{f,r}.bn (m) (b_hr and b_hz are already in the input projections), .proj.w (W, 4m), .proj.b (W).
 */
class DrumBandGru
{
public:
    DrumBandGru() = default;
    DrumBandGru(const ModelFile& mf, const std::string& prefix);

    /** gi: (N, F, 6m) input projections [forward 3m | reverse 3m] per band; out: (N, W). */
    void run(const float* gi, int N, int F, float* out, int threads) const;

private:
    int mM = 0, mW = 0;
    std::vector<float> mWt[2], mBn[2]; // W_hh transposed (m, 3m); b_hn (m)
    std::vector<float> mProjT, mProjB; // proj transposed (4m, W); bias (W)
};

} // namespace dtt
