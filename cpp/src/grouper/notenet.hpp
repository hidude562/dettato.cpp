#pragma once

// The grouper network (dettato/grouper/model.py NoteNet) as a ggml graph over a batch of B windows of N notes (N padded,
// masked). The CPU prepares the front end and the per-note readers' inputs (front.hpp); the graph runs LaneFront (stem
// conv, occupancy conv, 6 LBlocks, 2 LanePitchAttn, the read at each note), the HDemix / Prof / feat / s15 MLPs, the
// pitch embedding, ln_in, 7 transformer blocks with rotary positions on onset time, ln_out and the note heads.
//
// Layout: LaneFront runs channels-last, ne [C, lane, frame, window]; note tensors are ne [d, note, window].

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <ggml.h>

namespace dtt
{

class Backend;
class GroupFront;
class ModelFile;
class StaticGraph;

class NoteNet
{
public:
    NoteNet(const ModelFile& mf, const Backend& be);
    ~NoteNet();
    NoteNet(const NoteNet&) = delete;
    NoteNet& operator=(const NoteNet&) = delete;

    int d() const { return mD; }
    int heads() const { return mHeads; }
    int headDim() const { return mD / mHeads; }
    int lanes() const { return mP; }
    int channels() const { return mC; }
    int frames(int frames10) const { return (frames10 - mTs) / mTs + 1; }
    int ts() const { return mTs; }
    int nInst() const { return mK; }
    int embDim() const { return mE; }
    int outDim() const { return mK + mE + 1; }
    int hdIn() const { return mHdIn; }
    int profIn() const { return mProfIn; }
    int nTaps() const { return mNTaps; }
    int tapDim() const { return mTapDim; }
    float posScale() const { return mPosScale; }
    const std::vector<float>& ropeFreqs() const { return mFreqs; }

    /** One batch's inputs, filled by the caller (sizes for B windows of N notes, Tf 10 ms frames). */
    struct Inputs {
        int B = 0, N = 0, Tf = 0;
        std::vector<float> X;        // B x (channels, Tf, bins): the normalised HCQT
        std::vector<float> R;        // B x (Tq, lanes, 4) rasters, channels-last
        std::vector<float> hd;       // B x N x hdIn: HDemix.raw
        std::vector<float> prof;     // B x N x profIn
        std::vector<float> scal;     // B x N x 11 note_scalars
        std::vector<float> s15;      // B x N x nTaps x tapDim, standardised over the window's notes
        std::vector<int32_t> pitch;  // B x N: pitch embedding row
        std::vector<int32_t> rowOn;  // B x N: row of (window, onset frame, lane) in LaneFront's output
        std::vector<int32_t> csOn, csOff; // B x N: rows of (window, frame, lane) in its time cumsum (Tq + 1 rows per lane)
        std::vector<float> len;      // B x N: max(end - onset, 1) frames
        std::vector<float> cosv, sinv; // B x N x headDim: rotary cos / sin (both halves)
        std::vector<float> mask;     // B x N (key j valid -> 0, padding -> -inf); expanded to (N, N) per window
    };

    /** Sizes every input for B windows of N notes and Tf 10 ms frames (zeroed). */
    void resize(Inputs& in, int B, int N, int Tf, const GroupFront& fr) const;
    /**
     * Fills window slot b from one window: its nr note rows (nr, 12) and dequantised taps (nr, nTaps, tapDim) (rows nr..N
     * become collate's padding: zeros, pitch 60, masked), its front end X (channels, Tf, bins) (copied unless it already is
     * slot b of in.X) and P (extBins, Tf). Computes HDemix.raw, Prof's input, the rasters and read rows, the note scalars,
     * the standardised s15 taps, the rotary tables and the key mask. Slots are independent (callable from threads).
     */
    void fillSlot(Inputs& in, int b, const GroupFront& fr, const float* notes, int nr, const float* feat, const float* X,
                  const float* P) const;

    /** Runs the batch. debug keeps the intermediate tensors readable through debug(). */
    void run(const Inputs& in, bool debug = false);
    /** After run(): ne [outDim, N, B] (inst logits, raw emb, keep) and ne [d, N, B] (the ln_out output "h"). */
    void heads(std::vector<float>& out) const;
    void hidden(std::vector<float>& out) const;

    std::vector<std::string> debugNames() const;
    void debug(const std::string& name, std::vector<float>& out, std::vector<int64_t>& ne) const;
    size_t computeBytes() const;
    int nodes() const;

private:
    struct Graph;
    struct LBlockW;
    struct LAttnW;
    struct BlockW;
    void build(int B, int N, int Tf, bool debug);
    ggml_tensor* lblock(ggml_context* ctx, ggml_tensor* x, const LBlockW& w, int dil);
    ggml_tensor* lattn(ggml_context* ctx, ggml_tensor* x, const LAttnW& w);
    ggml_tensor* block(ggml_context* ctx, Graph& g, ggml_tensor* x, const BlockW& w, int N, int B);

    const Backend& mBe;
    const ModelFile& mMf;
    int mD, mHeads, mFF, mE, mK, mC, mP, mTs, mDepth, mLaneDepth, mHdIn, mProfIn, mNTaps, mTapDim, mInCh, mBins;
    float mPosScale;
    std::vector<int32_t> mDil, mAttnAt;
    std::vector<float> mFreqs;
    std::unique_ptr<Graph> mG;
};

} // namespace dtt
