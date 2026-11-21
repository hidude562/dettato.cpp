#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <ggml-alloc.h>
#include <ggml.h>

namespace dtt
{

class Backend;
class ModelFile;

/**
 * The transcriber network (dettato/transcriber/model.py LaneNet) as one static ggml graph over a batch of 8 s windows.
 *
 * Layout: the trunk runs channels-last, ggml ne [C, lane, frame, window], so every pointwise conv is one matmul and the
 * depthwise convs read the same memory through a permuted view. Input: the HCQT of each window, (window, ch, T10, F).
 */
class LaneNet
{
public:
    struct Taps {
        bool hidden = false; // the heads' input, (C, lane, frame) per window
        bool stem = false;   // the stem output on lanes 0..87, (C, lane, frame) per window
        bool debug = false;  // intermediate tensors for the parity tests
        bool f16 = false;    // read the taps back as float16 (half the transfer; the Python package accumulates them in f16)
    };

    LaneNet(const ModelFile& mf, const Backend& be, int batch, int frames10, Taps taps);
    ~LaneNet();
    LaneNet(const LaneNet&) = delete;
    LaneNet& operator=(const LaneNet&) = delete;

    int batch() const { return mBatch; }
    int channels() const { return mC; }
    int lanes() const { return mLanes; }
    int framesOut() const { return mT20; }
    int heads() const { return mHeads; }
    int inChannels() const { return mInCh; }
    int bins() const { return mBins; }

    /** hcqt: batch windows of (inChannels, frames10, bins) floats. */
    void run(const float* hcqt);

    /** After run(): logits, ne [heads, lane, frame, window] (heads fastest). */
    void logits(std::vector<float>& out) const;
    /** After run(): ne [C, lane, frame, window]. */
    void hidden(std::vector<float>& out) const;
    void stem(std::vector<float>& out) const;
    bool tapsF16() const;
    void hiddenF16(std::vector<ggml_fp16_t>& out) const;
    void stemF16(std::vector<ggml_fp16_t>& out) const;

    std::vector<std::string> debugNames() const;
    /** Copies a debug tensor; ne gets its shape. */
    void debug(const std::string& name, std::vector<float>& out, std::vector<int64_t>& ne) const;

    /** Bytes of the compute buffer. */
    size_t computeBytes() const;
    int nodes() const;

private:
    struct Block;
    struct Attn;
    struct Gru;
    ggml_tensor* block(ggml_context* ctx, ggml_tensor* x, const Block& w, int dil);
    ggml_tensor* attention(ggml_context* ctx, ggml_tensor* x, const Attn& w, bool time_axis);
    ggml_tensor* gru(ggml_context* ctx, ggml_cgraph* gf, ggml_tensor* x, const Gru& w);
    void mark(ggml_tensor* t, const char* name);
    void setInputs(const float* hcqt);

    const Backend& mBe;
    int mBatch, mT10, mT20, mT40, mC, mLanes, mLanesStem, mBins, mInCh, mHeads, mNH, mGruH, mKt;
    ggml_tensor* mPmixS = nullptr; // [K + 1, P * P] lane-shift indicator, see py/convert.py pmix_shift
    bool mFlashAttn = true;        // fused attention (ggml_flash_attn_ext); DETTATO_FLASH_ATTN=0 for the explicit path
    bool mPmixBroadcast = true;    // PitchMix's matmul broadcasts the lane matrices over the batch (dim 3); where the
                                   // backend cannot (Backend::matmulBroadcastsDim3), they are repeated per window

    ggml_context* mCtx = nullptr;
    ggml_cgraph* mGraph = nullptr;
    ggml_gallocr_t mAlloc = nullptr;

    ggml_tensor* mIn = nullptr;
    ggml_tensor* mPos = nullptr;     // I32 [T40], rotary positions
    ggml_tensor* mRevT = nullptr;    // I32 [T40], T40-1 .. 0
    ggml_tensor* mRevTN = nullptr;   // I32 [T40, N], the same per sequence
    ggml_tensor* mRows = nullptr;    // I64 [T40], 0 .. T40-1
    ggml_tensor* mH0 = nullptr;      // F32 [H, N, 2], zeros
    ggml_tensor* mLogits = nullptr;
    ggml_tensor* mHidden = nullptr;
    ggml_tensor* mStem = nullptr;
    std::vector<std::pair<std::string, ggml_tensor*>> mDebug;
};

} // namespace dtt
