#include "grouper/notenet.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <ggml-backend.h>

#include "backend.hpp"
#include "common.hpp"
#include "ggml_compat.hpp"
#include "graph.hpp"
#include "grouper/front.hpp"
#include "model_file.hpp"

namespace dtt
{

// weights as py/convert_grouper.py folds them
struct NoteNet::LBlockW {
    ggml_tensor *dw_w, *pmix, *pm_bias, *pw1_w, *pw1_b, *pw2_w, *pw2_b;
};
struct NoteNet::LAttnW {
    ggml_tensor *qkv_w, *qkv_b, *out_w, *out_b, *mask;
};
struct NoteNet::BlockW {
    ggml_tensor *qkv_w, *qkv_b, *o_w, *o_b, *ff0_w, *ff0_b, *ff2_w, *ff2_b;
};

struct NoteNet::Graph {
    int B = 0, N = 0, Tf = 0;
    bool debug = false;
    std::unique_ptr<StaticGraph> g;
    ggml_tensor *X = nullptr, *R = nullptr, *hd = nullptr, *prof = nullptr, *scal = nullptr, *s15 = nullptr;
    ggml_tensor *pitch = nullptr, *rowOn = nullptr, *csOn = nullptr, *csOff = nullptr, *len = nullptr, *csZero = nullptr;
    ggml_tensor *cosv = nullptr, *sinv = nullptr, *mask = nullptr;
    ggml_tensor *heads = nullptr, *hidden = nullptr;
    std::vector<std::pair<std::string, ggml_tensor*>> dbg;
};

namespace
{
    constexpr int MAX_NODES = 4096;
    constexpr float LN_EPS = 1e-5f;    // nn.LayerNorm
    constexpr float LN2D_EPS = 1e-6f;  // model.py LayerNorm2d
    constexpr int N_SCAL = 11;

    ggml_tensor* mlp2(ggml_context* ctx, const ModelFile& mf, const std::string& p, ggml_tensor* x)
    {
        ggml_tensor* h = ggml_gelu_erf(ctx, linear(ctx, mf.tensor(p + ".0.w"), mf.tensor(p + ".0.b"), x));
        return linear(ctx, mf.tensor(p + ".2.w"), mf.tensor(p + ".2.b"), h);
    }
} // namespace

NoteNet::NoteNet(const ModelFile& mf, const Backend& be) : mBe(be), mMf(mf)
{
    mD = static_cast<int>(mf.u32("gr.d"));
    mHeads = static_cast<int>(mf.u32("gr.heads"));
    mFF = static_cast<int>(mf.u32("gr.ff"));
    mE = static_cast<int>(mf.u32("gr.emb"));
    mK = static_cast<int>(mf.u32("gr.n_inst"));
    mC = static_cast<int>(mf.u32("gr.lanes"));
    mP = static_cast<int>(mf.u32("gr.n_lanes"));
    mTs = static_cast<int>(mf.u32("gr.lane_ts"));
    mDepth = static_cast<int>(mf.u32("gr.depth"));
    mLaneDepth = static_cast<int>(mf.u32("gr.lane_depth"));
    mNTaps = static_cast<int>(mf.i32s("gr.s15_taps").size());
    mTapDim = static_cast<int>(mf.u32("gr.s15_dim"));
    mPosScale = mf.f32("gr.pos_scale");
    mDil = mf.i32s("gr.lane_dil");
    mAttnAt = mf.i32s("gr.lane_attn_at");
    mFreqs = mf.hostF32("gr.rope_freqs");
    mInCh = static_cast<int>(mf.i32s("gr.shifts").size() + mf.i32s("gr.extra_nffts").size());
    mBins = static_cast<int>(mf.u32("gr.n_bins"));
    mHdIn = static_cast<int>(mf.tensor("gr.hd.enc.0.w")->ne[0]);
    mProfIn = static_cast<int>(mf.tensor("gr.prof.0.w")->ne[0]);
    DTT_CHECK(mProfIn == 4 * mInCh, "Prof input %d != 4 x %d channels", mProfIn, mInCh);
    DTT_CHECK(mf.tensor("gr.heads.w")->ne[1] == mK + mE + 1, "head rows != instruments + emb + 1");
    DTT_CHECK(static_cast<int>(mFreqs.size()) * 2 == mD / mHeads, "rotary table size");
    DTT_CHECK(static_cast<int>(mDil.size()) == mLaneDepth, "lane dilations");
    DTT_CHECK(mf.tensor("gr.blk.0.qkv.w")->ne[1] == 5 * mD, "qkv rows: q, k, v, rot q, rot k expected");
}

NoteNet::~NoteNet() = default;

ggml_tensor* NoteNet::lblock(ggml_context* ctx, ggml_tensor* x, const LBlockW& w, int dil)
{
    const int64_t C = x->ne[0], P = x->ne[1], T = x->ne[2], B = x->ne[3];
    // depthwise time conv + PitchMix run channels-first, [P, T, C, B]
    ggml_tensor* y = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));
    y = ggml_conv_2d_dw_direct(ctx, w.dw_w, y, 1, 1, 0, 3 * dil, 1, dil); // kernel (7, 1), dilation dil, same padding
    // y + PitchMix(y): each channel's P x P banded matrix from its taps (centre 1), one batched matmul
    ggml_tensor* pm = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, mMf.tensor("gr.pmix_S"), w.pmix), P, P, C); // [Pj, Pi, C]
    if (B > 1 && !mBe.matmulBroadcastsDim3()) {
        pm = ggml_repeat_4d(ctx, pm, P, P, C, B); // [Pj, Pi, C, B] (Backend::matmulBroadcastsDim3)
    }
    y = ggml_mul_mat(ctx, pm, y);                                                                         // [Pi, T, C, B]
    accF32(y);
    y = ggml_cont(ctx, ggml_permute(ctx, y, 1, 2, 0, 3)); // [C, P, T, B]
    y = ggml_add(ctx, y, w.pm_bias);                      // the dw bias through (I + PitchMix): per (channel, lane)
    // LayerNorm2d over channels (affine folded into pw1) -> pw1 -> GELU -> pw2 (gamma folded), residual
    ggml_tensor* h = ggml_norm(ctx, ggml_reshape_2d(ctx, y, C, P * T * B), LN2D_EPS);
    h = ggml_gelu_erf(ctx, linear(ctx, w.pw1_w, w.pw1_b, h));
    h = linear(ctx, w.pw2_w, w.pw2_b, h);
    return ggml_add(ctx, x, ggml_reshape_4d(ctx, h, C, P, T, B));
}

ggml_tensor* NoteNet::lattn(ggml_context* ctx, ggml_tensor* x, const LAttnW& w)
{
    // at every frame the lanes attend to each other, with a learned bias per pitch distance (the mask)
    const int64_t C = x->ne[0], P = x->ne[1], T = x->ne[2], B = x->ne[3];
    const int64_t NH = w.mask->ne[2], D = C / NH, N = T * B;
    ggml_tensor* s = ggml_reshape_3d(ctx, x, C, P, N);
    ggml_tensor* qkv = linear(ctx, w.qkv_w, w.qkv_b, ggml_norm(ctx, s, LN_EPS)); // [3C, P, N]
    const size_t es = ggml_element_size(qkv);
    ggml_tensor* q = ggml_view_4d(ctx, qkv, D, NH, P, N, D * es, qkv->nb[1], qkv->nb[2], 0);
    ggml_tensor* k = ggml_view_4d(ctx, qkv, D, NH, P, N, D * es, qkv->nb[1], qkv->nb[2], C * es);
    ggml_tensor* v = ggml_view_4d(ctx, qkv, D, NH, P, N, D * es, qkv->nb[1], qkv->nb[2], 2 * C * es);
    q = ggml_permute(ctx, q, 0, 2, 1, 3); // [D, P, NH, N]
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    ggml_tensor* kq = ggml_mul_mat(ctx, k, q); // [Pk, Pq, NH, N]
    accF32(kq);
    kq = ggml_soft_max_ext(ctx, kq, w.mask, 1.0f / std::sqrt(static_cast<float>(D)), 0.0f);
    ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3)); // [P, D, NH, N]
    ggml_tensor* o = ggml_mul_mat(ctx, vt, kq);                         // [D, Pq, NH, N]
    accF32(o);
    o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3)); // [D, NH, P, N]
    o = linear(ctx, w.out_w, w.out_b, ggml_reshape_3d(ctx, o, C, P, N));
    return ggml_add(ctx, x, ggml_reshape_4d(ctx, o, C, P, T, B));
}

ggml_tensor* NoteNet::block(ggml_context* ctx, Graph& g, ggml_tensor* x, const BlockW& w, int N, int B)
{
    const int64_t d = mD, H = mHeads, D = d / H;
    // qkv rows: q, k, v, rot(q), rot(k) (rot(x) = [-x2, x1] per head, folded into the weights), ln1 folded
    ggml_tensor* qkv = linear(ctx, w.qkv_w, w.qkv_b, ggml_norm(ctx, x, LN_EPS)); // [5d, N*B]
    qkv = ggml_reshape_3d(ctx, qkv, 5 * d, N, B);
    const size_t es = ggml_element_size(qkv);
    auto view = [&](int64_t off) { return ggml_view_4d(ctx, qkv, D, H, N, B, D * es, qkv->nb[1], qkv->nb[2], off * es); };
    // rope_time on onset * pos_scale: x cos + rot(x) sin, cos / sin [D, 1, N, B] broadcast over the heads
    ggml_tensor* q = ggml_add(ctx, ggml_mul(ctx, view(0), g.cosv), ggml_mul(ctx, view(3 * d), g.sinv));
    ggml_tensor* k = ggml_add(ctx, ggml_mul(ctx, view(d), g.cosv), ggml_mul(ctx, view(4 * d), g.sinv));
    ggml_tensor* v = view(2 * d);
    q = ggml_permute(ctx, q, 0, 2, 1, 3); // [D, N, H, B]
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    ggml_tensor* kq = ggml_mul_mat(ctx, k, q); // [Nk, Nq, H, B]
    accF32(kq);
    kq = ggml_soft_max_ext(ctx, kq, g.mask, 1.0f / std::sqrt(static_cast<float>(D)), 0.0f);
    ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3)); // [N, D, H, B]
    ggml_tensor* o = ggml_mul_mat(ctx, vt, kq);                         // [D, Nq, H, B]
    accF32(o);
    o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3)); // [D, H, N, B]
    x = ggml_add(ctx, x, linear(ctx, w.o_w, w.o_b, ggml_reshape_2d(ctx, o, d, static_cast<int64_t>(N) * B)));
    ggml_tensor* h = ggml_gelu_erf(ctx, linear(ctx, w.ff0_w, w.ff0_b, ggml_norm(ctx, x, LN_EPS)));
    return ggml_add(ctx, x, linear(ctx, w.ff2_w, w.ff2_b, h));
}

void NoteNet::build(int B, int N, int Tf, bool debug)
{
    if (mG && mG->B == B && mG->N == N && mG->Tf == Tf && mG->debug == debug) {
        return;
    }
    mG.reset(); // free the previous compute buffer first
    auto G = std::make_unique<Graph>();
    G->B = B;
    G->N = N;
    G->Tf = Tf;
    G->debug = debug;
    G->g = std::make_unique<StaticGraph>(mBe, MAX_NODES);
    StaticGraph& sg = *G->g;
    ggml_context* ctx = sg.ctx();
    const ModelFile& mf = mMf;
    const int64_t C = mC, P = mP, Tq = frames(Tf), NB = static_cast<int64_t>(N) * B, d = mD;
    auto mark = [&](ggml_tensor* t, const char* name) {
        if (debug) {
            sg.output(t, name);
            G->dbg.emplace_back(name, t);
        }
    };

    G->X = sg.input(GGML_TYPE_F32, {mBins, Tf, mInCh, B}, "X");
    G->R = sg.input(GGML_TYPE_F32, {4, P * Tq * B}, "R");
    G->hd = sg.input(GGML_TYPE_F32, {mHdIn, NB}, "hd_raw");
    G->prof = sg.input(GGML_TYPE_F32, {mProfIn, NB}, "prof_raw");
    G->scal = sg.input(GGML_TYPE_F32, {N_SCAL, NB}, "scal");
    G->s15 = sg.input(GGML_TYPE_F32, {mTapDim, mNTaps * NB}, "s15");
    G->pitch = sg.input(GGML_TYPE_I32, {NB}, "pitch");
    G->rowOn = sg.input(GGML_TYPE_I32, {NB}, "row_on");
    G->csOn = sg.input(GGML_TYPE_I32, {NB}, "cs_on");
    G->csOff = sg.input(GGML_TYPE_I32, {NB}, "cs_off");
    G->len = sg.input(GGML_TYPE_F32, {1, NB}, "len");
    G->csZero = sg.input(GGML_TYPE_F32, {1, C, P, B}, "cs_zero");
    G->cosv = sg.input(GGML_TYPE_F32, {d / mHeads, 1, N, B}, "cos");
    G->sinv = sg.input(GGML_TYPE_F32, {d / mHeads, 1, N, B}, "sin");
    G->mask = sg.input(GGML_TYPE_F32, {N, N, 1, B}, "mask");

    // ---- LaneFront: stride-(ts, 3) stem conv on [F, T, ch, B] -> [P, Tq, C, B] -> channels-last
    ggml_tensor* y = ggml_conv_2d_direct(ctx, mf.tensor("gr.lf.stem.w"), G->X, 3, mTs, 1, 0, 1, 1);
    y = ggml_add(ctx, y, ggml_reshape_4d(ctx, mf.tensor("gr.lf.stem.b"), 1, 1, C, 1));
    y = ggml_cont(ctx, ggml_permute(ctx, y, 1, 2, 0, 3)); // [C, P, Tq, B]
    DTT_CHECK(y->ne[1] == P && y->ne[2] == Tq, "LaneFront stem output %lld x %lld, expected %lld x %lld",
              static_cast<long long>(y->ne[1]), static_cast<long long>(y->ne[2]), static_cast<long long>(P),
              static_cast<long long>(Tq));
    ggml_tensor* occ = linear(ctx, mf.tensor("gr.lf.occ.w"), mf.tensor("gr.lf.occ.b"), G->R);
    y = ggml_add(ctx, y, ggml_reshape_4d(ctx, occ, C, P, Tq, B));
    mark(y, "lf_y0");
    for (int i = 0; i < mLaneDepth; i++) {
        const std::string q = format("gr.lf.blk.%d", i);
        y = lblock(ctx, y,
                   LBlockW{mf.tensor(q + ".dw_w"), mf.tensor(q + ".pmix"), mf.tensor(q + ".pm_bias"), mf.tensor(q + ".pw1.w"),
                           mf.tensor(q + ".pw1.b"), mf.tensor(q + ".pw2.w"), mf.tensor(q + ".pw2.b")},
                   mDil[i]);
        mark(y, format("lf_blk%d", i).c_str());
        for (int a : mAttnAt) {
            if (a == i) {
                const std::string r = format("gr.lf.attn.%d", i);
                y = lattn(ctx, y,
                          LAttnW{mf.tensor(r + ".qkv.w"), mf.tensor(r + ".qkv.b"), mf.tensor(r + ".out.w"), mf.tensor(r + ".out.b"),
                                 mf.tensor(r + ".mask")});
                mark(y, format("lf_attn%d", i).c_str());
            }
        }
    }
    // read at each note: its lane at the onset frame, and averaged over its frames (a time cumsum with a zero first row)
    ggml_tensor* at_on = ggml_get_rows(ctx, ggml_reshape_2d(ctx, y, C, P * Tq * B), G->rowOn); // [C, NB]
    ggml_tensor* yt = ggml_cont(ctx, ggml_permute(ctx, y, 1, 2, 0, 3));                          // [Tq, C, P, B]
    // the zero first row is concatenated, not padded: ggml-metal before v1.0.0 cannot pad at the front
    ggml_tensor* cs = ggml_concat(ctx, G->csZero, ggml_cumsum(ctx, yt), 0);                      // [Tq + 1, C, P, B]
    cs = ggml_cont(ctx, ggml_permute(ctx, cs, 1, 0, 2, 3));                                      // [C, Tq + 1, P, B]
    ggml_tensor* cs2 = ggml_reshape_2d(ctx, cs, C, (Tq + 1) * P * B);
    ggml_tensor* mean = ggml_div(ctx, ggml_sub(ctx, ggml_get_rows(ctx, cs2, G->csOff), ggml_get_rows(ctx, cs2, G->csOn)), G->len);
    mark(at_on, "lf_at_on");
    mark(mean, "lf_mean");
    ggml_tensor* lanes = linear(ctx, mf.tensor("gr.lf.read.w"), mf.tensor("gr.lf.read.b"), ggml_concat(ctx, at_on, mean, 0));
    mark(lanes, "lanes");

    // ---- note tokens
    ggml_tensor* hd = mlp2(ctx, mf, "gr.hd.enc", G->hd);
    ggml_tensor* pr = mlp2(ctx, mf, "gr.prof", G->prof);
    mark(hd, "hd");
    mark(pr, "prof");
    ggml_tensor* xn = ggml_add(ctx, ggml_get_rows(ctx, mf.tensor("gr.pitch_emb"), G->pitch), mlp2(ctx, mf, "gr.feat", G->scal));
    mark(xn, "x_note");
    ggml_tensor* f = ggml_norm(ctx, G->s15, LN_EPS); // s15_ln per tap (affine folded into s15.0)
    f = mlp2(ctx, mf, "gr.s15", ggml_reshape_2d(ctx, f, static_cast<int64_t>(mTapDim) * mNTaps, NB));
    mark(f, "s15");
    ggml_tensor* x = ggml_add(ctx, ggml_add(ctx, xn, ggml_add(ctx, ggml_add(ctx, hd, pr), lanes)), f);
    mark(x, "tok");
    x = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, LN_EPS), mf.tensor("gr.ln_in.w")), mf.tensor("gr.ln_in.b"));
    mark(x, "ln_in");
    for (int i = 0; i < mDepth; i++) {
        const std::string q = format("gr.blk.%d", i);
        x = block(ctx, *G, x,
                  BlockW{mf.tensor(q + ".qkv.w"), mf.tensor(q + ".qkv.b"), mf.tensor(q + ".o.w"), mf.tensor(q + ".o.b"),
                         mf.tensor(q + ".ff0.w"), mf.tensor(q + ".ff0.b"), mf.tensor(q + ".ff2.w"), mf.tensor(q + ".ff2.b")},
                  N, B);
        mark(x, format("blk%d", i).c_str());
    }
    G->hidden = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, LN_EPS), mf.tensor("gr.ln_out.w")), mf.tensor("gr.ln_out.b"));
    G->heads = linear(ctx, mf.tensor("gr.heads.w"), mf.tensor("gr.heads.b"), G->hidden);
    sg.output(G->hidden, "h");
    sg.output(G->heads, "heads");
    sg.finalize("the grouper");
    mG = std::move(G);
}

void NoteNet::resize(Inputs& in, int B, int N, int Tf, const GroupFront& fr) const
{
    const size_t NB = static_cast<size_t>(N) * B;
    const int Tq = frames(Tf);
    in.B = B;
    in.N = N;
    in.Tf = Tf;
    in.X.assign(static_cast<size_t>(B) * fr.channels() * Tf * fr.bins(), 0.0f);
    in.R.assign(static_cast<size_t>(B) * Tq * mP * 4, 0.0f);
    in.hd.assign(NB * mHdIn, 0.0f);
    in.prof.assign(NB * mProfIn, 0.0f);
    in.scal.assign(NB * N_SCAL, 0.0f);
    in.s15.assign(NB * mNTaps * mTapDim, 0.0f);
    in.pitch.assign(NB, 0);
    in.rowOn.assign(NB, 0);
    in.csOn.assign(NB, 0);
    in.csOff.assign(NB, 0);
    in.len.assign(NB, 1.0f);
    in.cosv.assign(NB * headDim(), 0.0f);
    in.sinv.assign(NB * headDim(), 0.0f);
    in.mask.assign(NB, -INFINITY);
}

void NoteNet::fillSlot(Inputs& in, int b, const GroupFront& fr, const float* notes, int nr, const float* feat, const float* X,
                       const float* P) const
{
    constexpr int NC = GroupFront::NOTE_COLS;
    const int N = in.N, Tf = in.Tf, Tq = frames(Tf), D = headDim(), half = D / 2;
    DTT_CHECK(nr >= 1 && nr <= N, "window with %d notes, slot holds %d", nr, N);
    const size_t nb0 = static_cast<size_t>(b) * N;
    std::vector<float> nt(static_cast<size_t>(N) * NC, 0.0f);
    std::memcpy(nt.data(), notes, sizeof(float) * nr * NC);
    std::vector<uint8_t> mk(N, 0);
    for (int j = 0; j < N; j++) {
        if (j >= nr) {
            nt[static_cast<size_t>(j) * NC + 2] = 60.0f;
        }
        mk[j] = j < nr;
    }
    const size_t xs = static_cast<size_t>(fr.channels()) * Tf * fr.bins();
    float* xslot = in.X.data() + b * xs;
    if (X != xslot) {
        std::memcpy(xslot, X, xs * sizeof(float));
    }
    fr.hdemixRaw(P, Tf, nt.data(), mk.data(), N, in.hd.data() + nb0 * mHdIn);
    fr.profRaw(xslot, Tf, nt.data(), N, in.prof.data() + nb0 * mProfIn);
    std::vector<int> on(N), off(N), lane(N);
    GroupFront::rasters(nt.data(), mk.data(), N, Tq, mP, 100.0f / static_cast<float>(mTs),
                        in.R.data() + static_cast<size_t>(b) * Tq * mP * 4, on.data(), off.data(), lane.data());
    for (int j = 0; j < N; j++) {
        const float* r = nt.data() + static_cast<size_t>(j) * NC;
        const size_t q = nb0 + j;
        in.rowOn[q] = static_cast<int32_t>(lane[j] + static_cast<int64_t>(mP) * (on[j] + static_cast<int64_t>(Tq) * b));
        const int64_t base = static_cast<int64_t>(Tq + 1) * (lane[j] + static_cast<int64_t>(mP) * b);
        in.csOn[q] = static_cast<int32_t>(on[j] + base);
        in.csOff[q] = static_cast<int32_t>(off[j] + base);
        in.len[q] = static_cast<float>(std::max(off[j] - on[j], 1));
        // note_scalars: log duration, the 9 map scalars, register
        float* sc = in.scal.data() + q * N_SCAL;
        sc[0] = std::log(std::max(r[1] - r[0], 0.01f));
        for (int k = 0; k < 9; k++) {
            sc[1 + k] = r[3 + k];
        }
        sc[10] = r[2] / 64.0f - 1.0f;
        in.pitch[q] = static_cast<int32_t>(std::clamp(static_cast<long>(std::nearbyint(r[2])), 0L, 127L));
        // rope_time: angle = (onset * pos_scale) * freq, halves layout
        const float pos = r[0] * mPosScale;
        float* cv = in.cosv.data() + q * D;
        float* sv = in.sinv.data() + q * D;
        for (int i = 0; i < half; i++) {
            const float ang = pos * mFreqs[i];
            cv[i] = cv[i + half] = std::cos(ang);
            sv[i] = sv[i + half] = std::sin(ang);
        }
        in.mask[q] = mk[j] ? 0.0f : -INFINITY;
    }
    // the transcriber's taps, standardised per channel over the window's notes
    const int TD = mNTaps * mTapDim;
    float* s15 = in.s15.data() + nb0 * TD;
    const float cnt = static_cast<float>(nr);
    for (int c = 0; c < TD; c++) {
        double s1 = 0;
        for (int j = 0; j < nr; j++) {
            s1 += feat[static_cast<size_t>(j) * TD + c];
        }
        const float mu = static_cast<float>(s1) / cnt;
        double s2 = 0;
        for (int j = 0; j < nr; j++) {
            const float dv = feat[static_cast<size_t>(j) * TD + c] - mu;
            s2 += static_cast<double>(dv * dv);
        }
        const float sd = std::max(std::sqrt(static_cast<float>(s2) / cnt), 1e-3f);
        for (int j = 0; j < N; j++) {
            const float fv = j < nr ? feat[static_cast<size_t>(j) * TD + c] : 0.0f;
            s15[static_cast<size_t>(j) * TD + c] = (fv - mu) / sd;
        }
    }
}

void NoteNet::release()
{
    mG.reset();
}

void NoteNet::run(const Inputs& in, bool debug)
{
    build(in.B, in.N, in.Tf, debug);
    Graph& g = *mG;
    const size_t NB = static_cast<size_t>(in.N) * in.B;
    auto set = [](ggml_tensor* t, const void* data, size_t bytes) {
        DTT_CHECK(bytes == ggml_nbytes(t), "input %s: %zu bytes, expected %zu", t->name, bytes, ggml_nbytes(t));
        ggml_backend_tensor_set(t, data, 0, bytes);
    };
    set(g.X, in.X.data(), in.X.size() * sizeof(float));
    set(g.R, in.R.data(), in.R.size() * sizeof(float));
    set(g.hd, in.hd.data(), in.hd.size() * sizeof(float));
    set(g.prof, in.prof.data(), in.prof.size() * sizeof(float));
    set(g.scal, in.scal.data(), in.scal.size() * sizeof(float));
    set(g.s15, in.s15.data(), in.s15.size() * sizeof(float));
    set(g.pitch, in.pitch.data(), in.pitch.size() * sizeof(int32_t));
    set(g.rowOn, in.rowOn.data(), in.rowOn.size() * sizeof(int32_t));
    set(g.csOn, in.csOn.data(), in.csOn.size() * sizeof(int32_t));
    set(g.csOff, in.csOff.data(), in.csOff.size() * sizeof(int32_t));
    set(g.len, in.len.data(), in.len.size() * sizeof(float));
    const std::vector<float> zeros(static_cast<size_t>(mC) * mP * in.B, 0.0f);
    set(g.csZero, zeros.data(), zeros.size() * sizeof(float));
    set(g.cosv, in.cosv.data(), in.cosv.size() * sizeof(float));
    set(g.sinv, in.sinv.data(), in.sinv.size() * sizeof(float));
    // key-padding mask, the same for every query of a window
    std::vector<float> m(NB * in.N);
    for (int b = 0; b < in.B; b++) {
        const float* row = in.mask.data() + static_cast<size_t>(b) * in.N;
        for (int q = 0; q < in.N; q++) {
            std::memcpy(m.data() + (static_cast<size_t>(b) * in.N + q) * in.N, row, in.N * sizeof(float));
        }
    }
    set(g.mask, m.data(), m.size() * sizeof(float));
    g.g->compute();
}

void NoteNet::heads(std::vector<float>& out) const
{
    readTensor(mG->heads, out);
}

void NoteNet::hidden(std::vector<float>& out) const
{
    readTensor(mG->hidden, out);
}

std::vector<std::string> NoteNet::debugNames() const
{
    std::vector<std::string> n;
    for (auto& [name, t] : mG->dbg) {
        n.push_back(name);
    }
    return n;
}

void NoteNet::debug(const std::string& name, std::vector<float>& out, std::vector<int64_t>& ne) const
{
    for (auto& [n, t] : mG->dbg) {
        if (n == name) {
            readTensor(t, out);
            ne.assign(t->ne, t->ne + 4);
            return;
        }
    }
    throw Failure("no debug tensor " + name);
}

size_t NoteNet::computeBytes() const
{
    return mG ? mG->g->computeBytes() : 0;
}

int NoteNet::nodes() const
{
    return mG ? mG->g->nodes() : 0;
}

} // namespace dtt
