#include "transcriber/lanenet.hpp"

#include <cmath>
#include <cstdlib>
#include <numeric>

#include <ggml-backend.h>

#include "backend.hpp"
#include "common.hpp"
#include "model_file.hpp"

namespace dtt
{

// Weights as the converter folds them: no BatchNorm, LayerNorm affines, layer scales or depthwise biases left.
struct LaneNet::Block {
    ggml_tensor *dw_w, *pmix, *pw1_w, *pw1_b, *pw2_w, *pw2_b;
};
struct LaneNet::Attn {
    ggml_tensor *qkv_w, *qkv_b, *out_w, *out_b, *mask;
};
struct LaneNet::Gru {
    ggml_tensor *ih_w, *ih_b, *hh_w, *hh_bn, *out_wf, *out_wr, *out_b;
};

namespace
{
    constexpr int MAX_NODES = 32768;
    constexpr float LN_EPS = 1e-5f;

    /** Per-channel vector [C] as a broadcastable [1, 1, C, 1] for channels-first tensors. */
    ggml_tensor* chanFirst(ggml_context* ctx, ggml_tensor* v)
    {
        return ggml_reshape_4d(ctx, v, 1, 1, v->ne[0], 1);
    }

    /** [C, P, T, B] channels-last -> the same memory as a WHCN view ([P, T, C, B], channels contiguous). */
    ggml_tensor* asCwhn(ggml_context* ctx, ggml_tensor* x)
    {
        return ggml_permute(ctx, x, 2, 0, 1, 3);
    }

    /** A depthwise kernel stored channels-fastest, ne [C, 1, KW, KH] -> the [KW, KH, 1, C] view the CWHN conv reads. */
    ggml_tensor* dwKernel(ggml_context* ctx, ggml_tensor* k)
    {
        return ggml_permute(ctx, k, 3, 2, 0, 1);
    }

    /** The inverse of asCwhn: a CWHN result back to [C, P, T, B]. */
    ggml_tensor* fromCwhn(ggml_context* ctx, ggml_tensor* y)
    {
        return ggml_permute(ctx, y, 1, 2, 0, 3);
    }


    ggml_tensor* linear(ggml_context* ctx, ggml_tensor* w, ggml_tensor* b, ggml_tensor* x)
    {
        ggml_tensor* y = ggml_mul_mat(ctx, w, x);
        ggml_prec_set_acc(y, GGML_PREC_F32);
        return b != nullptr ? ggml_add(ctx, y, b) : y;
    }
} // namespace

void LaneNet::mark(ggml_tensor* t, const char* name)
{
    ggml_set_name(t, name);
    ggml_set_output(t);
    mDebug.emplace_back(name, t);
}

ggml_tensor* LaneNet::block(ggml_context* ctx, ggml_tensor* x, const Block& w, int dil)
{
    const int64_t C = x->ne[0], P = x->ne[1], T = x->ne[2], B = x->ne[3];
    // the two depthwise maps run channels-first, [P, T, C, B]
    ggml_tensor* y = ggml_cont(ctx, asCwhn(ctx, x));
    y = ggml_conv_2d_dw_direct(ctx, w.dw_w, y, 1, 1, 0, (mKt / 2) * dil, 1, dil); // time conv, kernel kt, dilation dil
    // y + PitchMix(y): each channel's P x P banded matrix (from its taps, centre 1 for the residual), one batched matmul
    ggml_tensor* pm = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, mPmixS, w.pmix), P, P, C); // [Pj, Pi, C]
    y = ggml_mul_mat(ctx, pm, y);                                                       // [Pi, T, C, B]
    ggml_prec_set_acc(y, GGML_PREC_F32);
    y = ggml_cont(ctx, fromCwhn(ctx, y));                                               // [C, P, T, B]
    // pw1 (BN and the depthwise bias folded: per-lane bias) -> GELU -> pw2 (layer scale folded), residual
    const int64_t n = T * B;
    ggml_tensor* h = ggml_mul_mat(ctx, w.pw1_w, ggml_reshape_2d(ctx, y, C, P * n)); // one 2D matmul
    ggml_prec_set_acc(h, GGML_PREC_F32);
    h = ggml_add(ctx, ggml_reshape_3d(ctx, h, h->ne[0], P, n), w.pw1_b);             // per-lane bias
    h = ggml_gelu_erf(ctx, ggml_reshape_2d(ctx, h, h->ne[0], P * n));
    h = linear(ctx, w.pw2_w, w.pw2_b, h);
    return ggml_add(ctx, x, ggml_reshape_4d(ctx, h, C, P, T, B));
}

ggml_tensor* LaneNet::attention(ggml_context* ctx, ggml_tensor* x, const Attn& w, bool time_axis)
{
    const int64_t C = x->ne[0], P = x->ne[1], T = x->ne[2], B = x->ne[3];
    const int64_t D = C / mNH;                       // the true head dim (46)
    const int64_t Dp = w.qkv_w->ne[1] / (3 * mNH);   // stored, zero-padded head dim (48: flash attention needs a multiple of 8)
    const int64_t HD = mNH * Dp;
    // sequences: pitch -> the lanes of each frame; time -> the frames of each lane
    ggml_tensor* s = time_axis ? ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3)) : x; // [C, L, ...]
    const int64_t L = time_axis ? T : P;
    const int64_t N = time_axis ? P * B : T * B;
    s = ggml_reshape_3d(ctx, s, C, L, N);
    ggml_tensor* h = ggml_norm(ctx, s, LN_EPS); // the LayerNorm's affine is folded into qkv
    // projections as 2D matmuls (a 3D operand makes ggml run a slower batched matmul)
    ggml_tensor* qkv = ggml_reshape_3d(ctx, linear(ctx, w.qkv_w, w.qkv_b, ggml_reshape_2d(ctx, h, C, L * N)), 3 * HD, L, N);
    const size_t es = ggml_element_size(qkv);
    ggml_tensor* v = ggml_view_4d(ctx, qkv, Dp, mNH, L, N, Dp * es, qkv->nb[1], qkv->nb[2], 2 * HD * es);
    ggml_tensor *q, *k;
    if (time_axis) {
        // rotary positions over the true head dims, first half against second half (NEOX), base 10000; the zero pad
        // dims stay zero. Rope runs on a 3D view [Dp, H, L*N] (positions repeat per sequence): every backend has it.
        auto rope = [&](size_t off) {
            ggml_tensor* t = ggml_view_3d(ctx, qkv, Dp, mNH, L * N, Dp * es, qkv->nb[1], off);
            t = ggml_rope_ext(ctx, t, mPos, nullptr, static_cast<int>(D), GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            return ggml_reshape_4d(ctx, t, Dp, mNH, L, N);
        };
        q = rope(0);
        k = rope(HD * es);
    } else {
        q = ggml_view_4d(ctx, qkv, Dp, mNH, L, N, Dp * es, qkv->nb[1], qkv->nb[2], 0);
        k = ggml_view_4d(ctx, qkv, Dp, mNH, L, N, Dp * es, qkv->nb[1], qkv->nb[2], HD * es);
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    ggml_tensor* mask = time_axis ? nullptr : w.mask; // f16 [Lk, Lq, H]: a learned bias per head and pitch distance
    q = ggml_permute(ctx, q, 0, 2, 1, 3); // [Dp, L, H, N]
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    ggml_tensor* o;
    if (mFlashAttn) {
        // fused softmax(q k^T * scale + mask) v; the result comes back permuted, [Dp, H, Lq, N]
        ggml_tensor* m16 = mask != nullptr ? ggml_cast(ctx, mask, GGML_TYPE_F16) : nullptr; // flash attention reads an f16 mask
        o = ggml_flash_attn_ext(ctx, q, k, ggml_permute(ctx, v, 0, 2, 1, 3), m16, scale, 0.0f, 0.0f);
        ggml_prec_set_acc(o, GGML_PREC_F32);
    } else {
        ggml_tensor* kq = ggml_mul_mat(ctx, k, q); // [Lk, Lq, H, N]
        ggml_prec_set_acc(kq, GGML_PREC_F32);
        kq = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);
        ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3)); // [L, Dp, H, N]
        o = ggml_mul_mat(ctx, vt, kq);                                      // [Dp, Lq, H, N]
        ggml_prec_set_acc(o, GGML_PREC_F32);
        o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));               // [Dp, H, L, N]
    }
    o = linear(ctx, w.out_w, w.out_b, ggml_reshape_2d(ctx, o, HD, L * N)); // zero columns for the pad dims; layer scale folded
    if (time_axis) {
        o = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, o, C, T, P, B), 0, 2, 1, 3));
    }
    return ggml_add(ctx, x, ggml_reshape_4d(ctx, o, C, P, T, B));
}

ggml_tensor* LaneNet::gru(ggml_context* ctx, ggml_cgraph* gf, ggml_tensor* x, const Gru& w)
{
    const int64_t C = x->ne[0], P = x->ne[1], T = x->ne[2], B = x->ne[3];
    const int64_t N = P * B, H = mGruH;
    // LayerNorm, then the sequences ordered (frame, sequence): [C, N, T] with N = lane + P * window
    ggml_tensor* s = ggml_norm(ctx, x, LN_EPS); // the LayerNorm's affine is folded into the input projections
    s = ggml_cont(ctx, ggml_permute(ctx, s, 0, 1, 3, 2)); // [C, P, B, T]
    ggml_tensor* s2 = ggml_reshape_2d(ctx, s, C * N, T);
    ggml_tensor* sr = ggml_get_rows(ctx, s2, mRevT); // time-reversed copy, for the reverse direction
    ggml_tensor* sc = ggml_concat(ctx, ggml_reshape_3d(ctx, s2, C, N * T, 1), ggml_reshape_3d(ctx, sr, C, N * T, 1), 2);
    // input projections of both directions in one batched matmul: [3H, N*T, 2], b_hr and b_hz pre-added
    ggml_tensor* gi = linear(ctx, w.ih_w, ggml_reshape_3d(ctx, w.ih_b, 3 * H, 1, 2), sc);
    gi = ggml_reshape_4d(ctx, gi, 3 * H, N, T, 2);
    ggml_tensor* bn = ggml_reshape_3d(ctx, w.hh_bn, H, 1, 2);

    // outputs, one row per step: [H, T, N*2], written by a chain of set_rows so every later read depends on all steps
    ggml_tensor* Y = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, T, N * 2);
    ggml_tensor* h = mH0;
    const size_t es = sizeof(float);
    for (int64_t t = 0; t < T; t++) {
        ggml_tensor* gt = ggml_view_3d(ctx, gi, 3 * H, N, 2, gi->nb[1], gi->nb[3], t * gi->nb[2]);
        ggml_tensor* gh = ggml_mul_mat(ctx, w.hh_w, h); // [3H, N, 2]
        ggml_prec_set_acc(gh, GGML_PREC_F32);
        ggml_tensor* rz = ggml_sigmoid(ctx, ggml_add(ctx, ggml_view_3d(ctx, gt, 2 * H, N, 2, gt->nb[1], gt->nb[2], 0),
                                                         ggml_view_3d(ctx, gh, 2 * H, N, 2, gh->nb[1], gh->nb[2], 0)));
        ggml_tensor* r = ggml_view_3d(ctx, rz, H, N, 2, rz->nb[1], rz->nb[2], 0);
        ggml_tensor* z = ggml_view_3d(ctx, rz, H, N, 2, rz->nb[1], rz->nb[2], H * es);
        ggml_tensor* ghn = ggml_add(ctx, ggml_view_3d(ctx, gh, H, N, 2, gh->nb[1], gh->nb[2], 2 * H * es), bn);
        ggml_tensor* n = ggml_tanh(ctx, ggml_add(ctx, ggml_view_3d(ctx, gt, H, N, 2, gt->nb[1], gt->nb[2], 2 * H * es),
                                                 ggml_mul(ctx, r, ghn)));
        h = ggml_add(ctx, n, ggml_mul(ctx, z, ggml_sub(ctx, h, n))); // (1 - z) n + z h
        Y = ggml_set_rows(ctx, Y, ggml_reshape_3d(ctx, h, H, 1, N * 2), ggml_view_1d(ctx, mRows, 1, t * sizeof(int64_t)));
    }
    ggml_build_forward_expand(gf, Y);
    ggml_tensor* yf = ggml_view_3d(ctx, Y, H, T, N, Y->nb[1], Y->nb[2], 0);
    ggml_tensor* yr = ggml_view_3d(ctx, Y, H, T, N, Y->nb[1], Y->nb[2], N * Y->nb[2]);
    yr = ggml_get_rows(ctx, yr, mRevTN); // back to forward time
    ggml_tensor* o = ggml_add(ctx, linear(ctx, w.out_wf, nullptr, ggml_reshape_2d(ctx, yf, H, T * N)),
                              linear(ctx, w.out_wr, w.out_b, ggml_reshape_2d(ctx, yr, H, T * N))); // [C, T*N]
    o = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, o, C, T, P, B), 0, 2, 1, 3));          // [C, P, T, B]
    return ggml_add(ctx, x, o);
}

LaneNet::LaneNet(const ModelFile& mf, const Backend& be, int batch, int frames10, Taps taps)
    : mBe(be), mBatch(batch), mT10(frames10)
{
    const std::string p = "tr";
    mC = static_cast<int>(mf.u32(p + ".c"));
    mLanes = static_cast<int>(mf.u32(p + ".n_lanes"));
    mBins = static_cast<int>(mf.u32(p + ".n_bins"));
    mInCh = static_cast<int>(mf.i32s(p + ".shifts").size() + mf.i32s(p + ".extra_nffts").size());
    mNH = static_cast<int>(mf.u32(p + ".attn_heads"));
    mGruH = static_cast<int>(mf.u32(p + ".gru_hidden"));
    mKt = static_cast<int>(mf.u32(p + ".kt"));
    mHeads = static_cast<int>(mf.tensor(p + ".head.b")->ne[0]);
        mPmixS = mf.tensor(p + ".pmix_S");
    mFlashAttn = !be.isCpu(); // on the CPU the explicit path is exact (flash attention reads an f16 mask) and as fast
    if (const char* e = std::getenv("DETTATO_FLASH_ATTN")) {
        mFlashAttn = e[0] != '0';
    }
    const int depth = static_cast<int>(mf.u32(p + ".depth"));
    const int refine = static_cast<int>(mf.u32(p + ".refine"));
    const auto dil = mf.i32s(p + ".dilations");
    const auto attn_at = mf.i32s(p + ".attn_at");
    const auto gru_at = mf.i32s(p + ".gru_at");
    mLanesStem = mBins / 3;
    mT20 = (mT10 + 2 - 3) / 2 + 1;
    mT40 = (mT20 + 2 - 3) / 2 + 1;
    DTT_CHECK(mT20 == 2 * mT40, "frames10 %d gives an odd frame count", mT10);

    auto blockW = [&](const std::string& q) {
        return Block{mf.tensor(q + ".dw_w"), mf.tensor(q + ".pmix"), mf.tensor(q + ".pw1.w"), mf.tensor(q + ".pw1.b"),
                     mf.tensor(q + ".pw2.w"), mf.tensor(q + ".pw2.b")};
    };
    auto attnW = [&](const std::string& q) {
        return Attn{mf.tensor(q + ".qkv.w"), mf.tensor(q + ".qkv.b"), mf.tensor(q + ".out.w"), mf.tensor(q + ".out.b"),
                    mf.findTensor(q + ".mask")};
    };
    auto gruW = [&](const std::string& q) {
        return Gru{mf.tensor(q + ".ih.w"), mf.tensor(q + ".ih.b"), mf.tensor(q + ".hh.w"), mf.tensor(q + ".hh.bn"),
                   mf.tensor(q + ".out.wf"), mf.tensor(q + ".out.wr"), mf.tensor(q + ".out.b")};
    };

    // The graph is built by this lambda so it can be rebuilt with the explicit attention path when the backend has no
    // flash attention for these shapes.
    auto build = [&] {
        ggml_init_params ip = {ggml_tensor_overhead() * MAX_NODES * 4 + ggml_graph_overhead_custom(MAX_NODES, false), nullptr, true};
        mCtx = ggml_init(ip);
        DTT_CHECK(mCtx != nullptr, "ggml_init failed");
        ggml_context* ctx = mCtx;
        mGraph = ggml_new_graph_custom(ctx, MAX_NODES, false);

        const int B = mBatch, C = mC, L = mLanes;
        const int64_t N = static_cast<int64_t>(L) * B;
        mIn = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, mBins, mT10, mInCh, B);
        ggml_set_name(mIn, "hcqt");
        ggml_set_input(mIn);
        mPos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, static_cast<int64_t>(mT40) * N); // frame index, per time sequence
        mRevT = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, mT40);
        mRevTN = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, mT40, N);
        mRows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, mT40);
        mH0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, mGruH, N, 2);
        for (ggml_tensor* t : {mPos, mRevT, mRevTN, mRows, mH0}) {
            ggml_set_input(t);
        }

        // stem, channels-first [F, T10, ch, B]: input BatchNorm, 3x3 conv, 3x3 conv, stride-(2, 3) conv (BN folded)
        ggml_tensor* x = ggml_add(ctx, ggml_mul(ctx, mIn, chanFirst(ctx, mf.tensor(p + ".stem.in_scale"))),
                                  chanFirst(ctx, mf.tensor(p + ".stem.in_shift")));
        x = ggml_conv_2d_direct(ctx, mf.tensor(p + ".stem.c1.w"), x, 1, 1, 1, 1, 1, 1);
        x = ggml_gelu_erf(ctx, ggml_add(ctx, x, chanFirst(ctx, mf.tensor(p + ".stem.c1.b"))));
        x = ggml_conv_2d_direct(ctx, mf.tensor(p + ".stem.c2.w"), x, 1, 1, 1, 1, 1, 1);
        x = ggml_gelu_erf(ctx, ggml_add(ctx, x, chanFirst(ctx, mf.tensor(p + ".stem.c2.b"))));
        x = ggml_conv_2d_direct(ctx, mf.tensor(p + ".stem.down.w"), x, 3, 2, 1, 1, 1, 1);
        x = ggml_add(ctx, x, chanFirst(ctx, mf.tensor(p + ".stem.down.b"))); // [96, T20, C, B]
        if (taps.debug) {
            mark(x, "stem");
        }
        // channels-last, lanes 0..87: the skip connection and the trunk input
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3)); // [C, 96, T20, B]
        ggml_tensor* skip = ggml_cont(ctx, ggml_view_4d(ctx, x, C, L, mT20, B, x->nb[1], x->nb[2], x->nb[3], 0));
        if (taps.stem) {
            mStem = taps.f16 ? ggml_cast(ctx, skip, GGML_TYPE_F16) : skip;
            ggml_set_name(mStem, "stem_tap");
            ggml_set_output(mStem);
        }

        // to 40 ms: depthwise (3, 1) conv, stride 2 in time
        ggml_tensor* h = ggml_conv_2d_dw_direct(ctx, dwKernel(ctx, mf.tensor(p + ".down.w")), asCwhn(ctx, skip), 1, 2, 0, 1, 1, 1);
        h = ggml_add(ctx, fromCwhn(ctx, h), mf.tensor(p + ".down.b")); // [C, L, T40, B]
        if (taps.debug) {
            mark(h, "down");
        }

        int layer = 0;
        for (int i = 0; i < depth; i++) {
            h = block(ctx, h, blockW(format("%s.blk.%d", p.c_str(), i)), dil[i % dil.size()]);
            if (taps.debug && i == 0) {
                mark(h, "layers_0");
            }
            layer++;
            for (size_t g = 0; g < attn_at.size(); g++) {
                if (attn_at[g] == i + 1) {
                    h = attention(ctx, h, attnW(format("%s.attn.%zu.pitch", p.c_str(), g)), false);
                    if (taps.debug && g == 0) {
                        mark(h, format("layers_%d", layer).c_str());
                    }
                    layer++;
                    h = attention(ctx, h, attnW(format("%s.attn.%zu.time", p.c_str(), g)), true);
                    if (taps.debug && g == 0) {
                        mark(h, format("layers_%d", layer).c_str());
                    }
                    layer++;
                }
            }
            for (size_t g = 0; g < gru_at.size(); g++) {
                if (gru_at[g] == i + 1) {
                    h = gru(ctx, mGraph, h, gruW(format("%s.gru.%zu", p.c_str(), g)));
                    if (taps.debug && g == 0) {
                        mark(h, format("layers_%d", layer).c_str());
                    }
                    layer++;
                }
            }
        }
        if (taps.debug) {
            mark(h, format("layers_%d", layer - 1).c_str());
        }

        // back to 20 ms: each frame twice, + the stem, BatchNorm
        h = ggml_repeat_4d(ctx, ggml_reshape_4d(ctx, h, static_cast<int64_t>(C) * L, 1, mT40, B), static_cast<int64_t>(C) * L, 2, mT40, B);
        h = ggml_add(ctx, ggml_reshape_4d(ctx, h, C, L, mT20, B), skip);
        h = ggml_add(ctx, ggml_mul(ctx, h, mf.tensor(p + ".up.scale")), mf.tensor(p + ".up.shift"));
        if (taps.debug) {
            mark(h, "up_norm");
        }
        for (int i = 0; i < refine; i++) {
            h = block(ctx, h, blockW(format("%s.ref.%d", p.c_str(), i)), 1);
        }
        if (taps.debug) {
            mark(h, format("refine_%d", refine - 1).c_str());
        }
        if (taps.hidden || taps.debug) {
            mHidden = taps.f16 && !taps.debug ? ggml_cast(ctx, h, GGML_TYPE_F16) : h;
            ggml_set_name(mHidden, "hidden");
            ggml_set_output(mHidden);
        }
        mLogits = linear(ctx, mf.tensor(p + ".head.w"), mf.tensor(p + ".head.b"), ggml_reshape_2d(ctx, h, C, static_cast<int64_t>(L) * mT20 * B));
        mLogits = ggml_reshape_4d(ctx, mLogits, mHeads, L, mT20, B);
        ggml_set_name(mLogits, "logits");
        ggml_set_output(mLogits);

        ggml_build_forward_expand(mGraph, mLogits);
        if (mHidden != nullptr) {
            ggml_build_forward_expand(mGraph, mHidden);
        }
        if (mStem != nullptr) {
            ggml_build_forward_expand(mGraph, mStem);
        }
        for (auto& [n, t] : mDebug) {
            ggml_build_forward_expand(mGraph, t);
        }

    };
    build();
    std::string bad = be.firstUnsupportedOp(mGraph);
    if (!bad.empty() && mFlashAttn && bad.find("FLASH_ATTN") != std::string::npos) {
        ggml_free(mCtx);
        mCtx = nullptr;
        mDebug.clear();
        mHidden = mStem = nullptr;
        mFlashAttn = false;
        build();
        bad = be.firstUnsupportedOp(mGraph);
    }
    DTT_CHECK(bad.empty(), "%s cannot run the transcriber: %s", be.name().c_str(), bad.c_str());
    mAlloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be.get()));
    DTT_CHECK(ggml_gallocr_alloc_graph(mAlloc, mGraph), "cannot allocate the transcriber's compute buffer");
}

LaneNet::~LaneNet()
{
    if (mAlloc != nullptr) {
        ggml_gallocr_free(mAlloc);
    }
    if (mCtx != nullptr) {
        ggml_free(mCtx);
    }
}

void LaneNet::setInputs(const float* hcqt)
{
    ggml_backend_tensor_set(mIn, hcqt, 0, ggml_nbytes(mIn));
    std::vector<int32_t> pos(static_cast<size_t>(ggml_nelements(mPos))), rev(mT40);
    std::vector<int64_t> rows(mT40);
    for (size_t i = 0; i < pos.size(); i++) {
        pos[i] = static_cast<int32_t>(i % mT40);
    }
    for (int t = 0; t < mT40; t++) {
        rev[t] = mT40 - 1 - t;
        rows[t] = t;
    }
    ggml_backend_tensor_set(mPos, pos.data(), 0, ggml_nbytes(mPos));
    ggml_backend_tensor_set(mRevT, rev.data(), 0, ggml_nbytes(mRevT));
    ggml_backend_tensor_set(mRows, rows.data(), 0, ggml_nbytes(mRows));
    std::vector<int32_t> revn(static_cast<size_t>(ggml_nelements(mRevTN)));
    for (size_t i = 0; i < revn.size(); i++) {
        revn[i] = rev[i % mT40];
    }
    ggml_backend_tensor_set(mRevTN, revn.data(), 0, ggml_nbytes(mRevTN));
    std::vector<float> zeros(static_cast<size_t>(ggml_nelements(mH0)), 0.0f);
    ggml_backend_tensor_set(mH0, zeros.data(), 0, ggml_nbytes(mH0));
}

void LaneNet::run(const float* hcqt)
{
    setInputs(hcqt);
    mBe.compute(mGraph);
}

namespace
{
    void get(ggml_tensor* t, std::vector<float>& out)
    {
        DTT_CHECK(t != nullptr, "tensor was not requested");
        out.resize(static_cast<size_t>(ggml_nelements(t)));
        DTT_CHECK(ggml_is_contiguous(t), "%s is not contiguous", t->name);
        ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
    }
} // namespace

void LaneNet::logits(std::vector<float>& out) const
{
    get(mLogits, out);
}

void LaneNet::hidden(std::vector<float>& out) const
{
    get(mHidden, out);
}

void LaneNet::stem(std::vector<float>& out) const
{
    get(mStem, out);
}

bool LaneNet::tapsF16() const
{
    return (mHidden != nullptr && mHidden->type == GGML_TYPE_F16) || (mStem != nullptr && mStem->type == GGML_TYPE_F16);
}

void LaneNet::hiddenF16(std::vector<ggml_fp16_t>& out) const
{
    DTT_CHECK(mHidden != nullptr && mHidden->type == GGML_TYPE_F16, "the hidden tap is not f16");
    out.resize(static_cast<size_t>(ggml_nelements(mHidden)));
    ggml_backend_tensor_get(mHidden, out.data(), 0, ggml_nbytes(mHidden));
}

void LaneNet::stemF16(std::vector<ggml_fp16_t>& out) const
{
    DTT_CHECK(mStem != nullptr && mStem->type == GGML_TYPE_F16, "the stem tap is not f16");
    out.resize(static_cast<size_t>(ggml_nelements(mStem)));
    ggml_backend_tensor_get(mStem, out.data(), 0, ggml_nbytes(mStem));
}

std::vector<std::string> LaneNet::debugNames() const
{
    std::vector<std::string> n;
    for (auto& [name, t] : mDebug) {
        n.push_back(name);
    }
    return n;
}

void LaneNet::debug(const std::string& name, std::vector<float>& out, std::vector<int64_t>& ne) const
{
    for (auto& [n, t] : mDebug) {
        if (n == name) {
            ggml_tensor* c = t;
            out.resize(static_cast<size_t>(ggml_nelements(c)));
            ne.assign(c->ne, c->ne + 4);
            if (ggml_is_contiguous(c)) {
                ggml_backend_tensor_get(c, out.data(), 0, ggml_nbytes(c));
            } else {
                // a permuted view: read the whole underlying storage it spans and gather with its strides
                std::vector<float> raw(static_cast<size_t>(ggml_nbytes(c) / sizeof(float) + 1));
                const size_t span = (ne[0] - 1) * c->nb[0] + (ne[1] - 1) * c->nb[1] + (ne[2] - 1) * c->nb[2] + (ne[3] - 1) * c->nb[3] + sizeof(float);
                raw.resize(span / sizeof(float));
                ggml_backend_tensor_get(c, raw.data(), 0, span);
                size_t i = 0;
                for (int64_t i3 = 0; i3 < ne[3]; i3++)
                    for (int64_t i2 = 0; i2 < ne[2]; i2++)
                        for (int64_t i1 = 0; i1 < ne[1]; i1++)
                            for (int64_t i0 = 0; i0 < ne[0]; i0++)
                                out[i++] = raw[(i0 * c->nb[0] + i1 * c->nb[1] + i2 * c->nb[2] + i3 * c->nb[3]) / sizeof(float)];
            }
            return;
        }
    }
    throw Failure("no debug tensor " + name);
}

size_t LaneNet::computeBytes() const
{
    return ggml_gallocr_get_buffer_size(mAlloc, 0);
}

int LaneNet::nodes() const
{
    return ggml_graph_n_nodes(mGraph);
}

} // namespace dtt
