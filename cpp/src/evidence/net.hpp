#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ggml.h>

namespace dtt
{
class Backend;
class ModelFile;
} // namespace dtt

namespace dtt::ev
{

/**
 * The repairer network (dettato/evidence/model.py Repairer) over one 8 s window of notes.
 *
 * The note features (NoteEmbed's Fourier features, context_features, grid_features, the embedding lookups and the rotary
 * cos / sin of the continuous positions) are computed on the host in float32 with torch's operation order; the two MLPs, the
 * 9 transformer blocks and the heads run as one ggml graph. A window of n notes runs padded to a bucket of 64 tokens
 * (padded keys masked); queries are processed in chunks of up to 512 so the score matrices stay small for big windows. One
 * graph is alive at a time and rebuilt when the bucket changes (the caller runs windows grouped by bucket).
 */
class RepairerNet
{
public:
    static constexpr int N_CONT = 110;                 // the continuous features
    static constexpr int N_DON = 201, N_DR = 161, N_INS_P = 49, N_INS_T = 401, N_DUR = 128, N_SLOT = 2;
    static constexpr int N_INS = 1 + N_INS_P + N_INS_T + N_DUR; // 579 per slot
    static constexpr int N_OUT = 1 + N_DON + N_DR + N_SLOT * N_INS; // 1521: keep, don, dur, ins slot 0, ins slot 1
    static constexpr int BUCKET = 64;
    static constexpr int Q_CHUNK = 512;

    RepairerNet(const ModelFile& mf, const Backend& be);
    ~RepairerNet();
    RepairerNet(const RepairerNet&) = delete;
    RepairerNet& operator=(const RepairerNet&) = delete;

    static int bucket(int n) { return (n + BUCKET - 1) / BUCKET * BUCKET; }

    /** NoteEmbed's continuous input for n notes (window-relative onsets, durations), (n, 110) row-major. */
    void contFeatures(int n, const float* on, const float* dur, const int* pitch, float* out) const;

    /** One window -> (n, N_OUT) logits. ev: (n, 769). With debug on, the intermediate tensors are kept (debug()). */
    void run(int n, const int* pitch, const int* vel, const float* on, const float* dur, const float* ev, std::vector<float>& out);

    /** Queries per attention chunk (default Q_CHUNK; the parity tool sets small values to exercise the chunking). */
    void setQueryChunk(int q) { mQChunk = q; }

    /** Keep intermediate tensors (emb, block_0..8, bb = the final RMSNorm with its weight) for the parity tests. */
    void setDebug(bool on) { mDebug = on; }
    /** After run(): a debug tensor for the n real notes, (n, d) row-major. */
    void debug(const std::string& name, std::vector<float>& out) const;

    const std::vector<float>& durCent() const { return mDurCent; }
    const std::vector<float>& durW() const { return mDurW; }
    const std::vector<float>& drExp() const { return mDrExp; }
    int maxWindow() const { return mMaxWin; }
    int layers() const { return mLayers; }

    struct Stats {
        int builds = 0;
        double build_s = 0, features_s = 0, compute_s = 0;
        size_t compute_bytes_max = 0;
        int nodes_max = 0;
    };
    const Stats& stats() const { return mStats; }
    void resetStats() { mStats = Stats{}; }

private:
    struct Graph;
    void build(int Np);
    ggml_tensor* block(ggml_context* ctx, ggml_tensor* x, int i, ggml_tensor* C3, ggml_tensor* S3, ggml_tensor* mask, int Np);

    const Backend& mBe;
    int mD, mLayers, mHeads, mHd, mFfh, mNEv, mUsed, mMaxWin;
    std::vector<float> mRopeW, mFourOn, mFourLd, mGridP, mDurCent, mDurW, mDrExp;
    std::vector<float> mEmbInst, mEmbPitch, mEmbVel;
    struct W;
    std::unique_ptr<W> mW;
    std::unique_ptr<Graph> mG;
    bool mDebug = false;
    int mQChunk = Q_CHUNK;
    int mLastN = 0;
    Stats mStats;
    // host input buffers
    std::vector<float> mInEmb, mInCf, mInEv, mInC, mInS, mInMask;
};

} // namespace dtt::ev
