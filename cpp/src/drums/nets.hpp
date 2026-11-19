#pragma once

#include <map>
#include <memory>

struct ggml_context;
struct ggml_tensor;
#include <string>
#include <vector>

namespace dtt
{

class Backend;
class ModelFile;

/** Intermediate tensors for the parity tool, each in the reference's layout (numpy row-major, batch dimension dropped). */
struct DrumTaps {
    bool stem = false; // also the drum-kit stem output (56, T/2, 20): large
    std::map<std::string, std::vector<float>> t;
};

/**
 * Time per part of a transcribe() call, milliseconds, summed over the models. Graph parts count from the moment the graph
 * holds the backend; on a GPU the experts run concurrently, so the parts overlap and can sum to more than the total.
 */
struct DrumTiming {
    double features = 0, embed = 0, trunk = 0, cq = 0, adapter = 0, host_gru = 0, decode = 0, total = 0;
    size_t compute_bytes = 0; // the largest compute buffer (device memory besides the weights)
};

/** DrumTranscriber.probabilities: all maps at 50 fps, row-major (frame, key). */
struct DrumProbs {
    double router = 0;
    int N = 0;                          // frames of every map
    int K = 0, KT = 0;                  // drum keys (61), timpani keys (13)
    std::vector<float> kit, kit_vel;    // (N, K); empty when the drum-kit model was skipped
    std::vector<float> drum, timpani, vel; // (N, K), (N, KT), (N, K): the experts' average
    std::vector<double> routers;        // each expert's sigmoid(router logit)
};

/**
 * The drum networks (dettato/drums/model.py): the drum-kit DrumNet and the orchestral experts, on one ggml backend.
 *
 * Per DrumNet: the stem CNN and the band GRU (20 steps, unrolled) as one graph per input chunk (embed / embed_long, the
 * reference's chunking); the conv block, the GRU block's input projections, the song attention (flash attention) and the
 * heads' projections as graphs over the whole song; the GRU block's and the family towers' recurrences on the host.
 * Per expert, additionally: the cq branch (GroupNorm over the whole song) with the adapter input and the first GRU layer's
 * projections as one graph, the 2-layer GRU's recurrences on the host, the adapter attention + LayerNorm + heads as one
 * graph, the router on the host.
 */
class DrumNets
{
public:
    DrumNets(const ModelFile& mf, const Backend& be);
    ~DrumNets();

    int keys() const { return mK; }
    int timpaniKeys() const { return mKT; }
    float routerThreshold() const { return mThRouter; }

    /**
     * probabilities(feats): feats (T, 280) as float32 (the float16 values). all_models: false skips the drum-kit model when the
     * router reaches its threshold (P.kit / P.kit_vel stay empty); the experts always run (the router needs them).
     */
    DrumProbs run(const float* feats, int T, bool all_models, DrumTaps* taps, DrumTiming* tm) const;

    /**
     * The cq branch runs over the whole song in one graph up to 1.5 windows of this many output frames (50 fps); longer
     * songs run it window by window in four passes (bounded memory, 2-3x its convolutions). 0: always one graph.
     * Default: 4000 on a GPU, 12000 on the CPU backend.
     */
    void setCqChunk(int frames) { mCqChunk = frames; }
    int cqChunk() const { return mCqChunk; }

    struct Net;
    struct Expert;

private:
    class Exec;
    std::vector<float> embedChunk(Exec& ex, const Net& w, const float* feats, int stride, int a, int b, DrumTaps* taps,
                                  const std::string& p, DrumTiming* tm) const;
    std::vector<float> embed(Exec& ex, const Net& w, const float* feats, int stride, int T, bool long_path, DrumTaps* taps,
                             const std::string& p, DrumTiming* tm) const;
    /** trunk + heads: h (N, W), z (N, K), v (N, K). */
    void trunk(Exec& ex, const Net& w, const std::vector<float>& c, int N, std::vector<float>& h, std::vector<float>& z,
               std::vector<float>& v, DrumTaps* taps, const std::string& p, DrumTiming* tm) const;
    ggml_tensor* cqConv(struct ggml_context* ctx, const Expert& w, int i, struct ggml_tensor* x) const;
    ggml_tensor* cqProj(struct ggml_context* ctx, const Expert& w, struct ggml_tensor* z3) const;
    std::vector<float> cqChunked(Exec& ex, const Expert& w, const float* cqin, int T) const;
    /** One expert: (logits) drum (N, K), timpani (N, KT), velocity (N, K), base (N, K); returns the router logit. */
    float expert(Exec& ex, const Expert& w, const float* feats, int T, std::vector<float>& base, std::vector<float>& drum,
                 std::vector<float>& timp, std::vector<float>& vel, DrumTaps* taps, const std::string& p,
                 DrumTiming* tm) const;

    const Backend& mBe;
    int mK = 0, mKT = 0, mW = 0, mNMel = 160, mNCq = 120, mChunk = 8000, mMargin = 200, mKitLongAbove = 9000;
    int mAttnHeads = 2, mExHeads = 4, mTbPad = 7, mTbDil = 1;
    // GPU: 80 s windows above 2 min (a 4-min song: 3 windows, ~0.26 GB of compute buffer instead of ~0.69 GB). CPU (host
    // memory, and the windows cost 2-3x the branch's convolutions): 4-min windows above 6 min (up to ~1 GB).
    int mCqChunk = 4000;
    float mThRouter = 0.5f;
    std::unique_ptr<Net> mKit;
    std::vector<std::unique_ptr<Expert>> mExperts;
};

} // namespace dtt
