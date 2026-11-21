#include "evidence/net.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>

#include "backend.hpp"
#include "common.hpp"
#include "ggml_compat.hpp"
#include "evidence/features.hpp"
#include "graph.hpp"
#include "model_file.hpp"

#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

namespace dtt::ev
{

namespace
{
    constexpr float RMS_EPS = 1e-6f;
    // `2 * math.pi * x` with x a float32 tensor: the Python product 2 * math.pi, cast to float32 by the tensor op
    constexpr float TWO_PI_F = static_cast<float>(2.0 * 3.141592653589793);
    constexpr int MAX_NODES = 8192;

    double since(std::chrono::steady_clock::time_point t0)
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    int ceilLog2(int64_t x)
    {
        if (x <= 2) {
            return 1;
        }
        int b = 0;
        uint64_t v = static_cast<uint64_t>(x) - 1;
        while (v) {
            b++;
            v >>= 1;
        }
        return b;
    }

    /**
     * torch's float32 sum over a strided dimension (ATen cascade_sum / multi_row_sum: 4 levels of partial sums, combined
     * every 16 elements), the order torch.sum(dim) uses on the CPU.
     */
    float torchSum(const float* x, int64_t stride, int64_t size)
    {
        constexpr int levels = 4;
        const int64_t power = std::max<int64_t>(4, ceilLog2(size) / levels);
        const int64_t step = int64_t{1} << power, mask = step - 1;
        float acc[levels] = {0.0f, 0.0f, 0.0f, 0.0f};
        int64_t i = 0;
        for (; i + step <= size;) {
            for (int64_t j = 0; j < step; ++j, ++i) {
                acc[0] += x[i * stride];
            }
            for (int j = 1; j < levels; ++j) {
                acc[j] += acc[j - 1];
                acc[j - 1] = 0.0f;
                if ((i & (mask << (j * power))) != 0) {
                    break;
                }
            }
        }
        for (; i < size; ++i) {
            acc[0] += x[i * stride];
        }
        for (int j = 1; j < levels; ++j) {
            acc[0] += acc[j];
        }
        return acc[0];
    }

    float signf(float x)
    {
        return static_cast<float>((0.0f < x) - (x < 0.0f));
    }
} // namespace

struct RepairerNet::W {
    ggml_tensor *cont0_w, *cont0_b, *cont2_w, *cont2_b, *evm0_w, *evm0_b, *evm2_w, *evm2_b, *emb_norm;
    struct Blk {
        ggml_tensor *qkv, *o, *w1, *w2;
    };
    std::vector<Blk> blk;
    ggml_tensor *head_w, *head_b;
};

struct RepairerNet::Graph {
    StaticGraph g;
    int Np = 0, Qc = 0;
    ggml_tensor *emb = nullptr, *cf = nullptr, *ev = nullptr, *ropeC = nullptr, *ropeS = nullptr, *mask = nullptr;
    ggml_tensor* out = nullptr;
    std::vector<std::pair<std::string, ggml_tensor*>> dbg;
    explicit Graph(const Backend& be) : g(be, MAX_NODES) {}
};

RepairerNet::RepairerNet(const ModelFile& mf, const Backend& be) : mBe(be), mW(std::make_unique<W>())
{
    const std::string p = "ev";
    mD = static_cast<int>(mf.u32(p + ".d"));
    mLayers = static_cast<int>(mf.u32(p + ".layers"));
    mHeads = static_cast<int>(mf.u32(p + ".heads"));
    mHd = mD / mHeads;
    mFfh = static_cast<int>(mf.u32(p + ".ff_hidden"));
    mNEv = static_cast<int>(mf.u32(p + ".n_ev"));
    mUsed = static_cast<int>(mf.u32(p + ".rope_used"));
    mMaxWin = static_cast<int>(mf.u32(p + ".max_win"));
    DTT_CHECK(mNEv == N_EV, "evidence width %d != %d", mNEv, N_EV);
    DTT_CHECK(static_cast<int>(mf.u32(p + ".n_don")) == N_DON && static_cast<int>(mf.u32(p + ".n_dr")) == N_DR &&
                  static_cast<int>(mf.u32(p + ".n_ins_p")) == N_INS_P && static_cast<int>(mf.u32(p + ".n_ins_t")) == N_INS_T &&
                  static_cast<int>(mf.u32(p + ".n_dur")) == N_DUR && static_cast<int>(mf.u32(p + ".n_slot")) == N_SLOT,
              "the repairer's output layout differs from this build");
    mRopeW = mf.hostF32(p + ".rope_w");
    DTT_CHECK(static_cast<int>(mRopeW.size()) * 2 == mUsed && mRopeW.size() == 14, "rope table");
    mFourOn = mf.hostF32(p + ".four_on");
    mFourLd = mf.hostF32(p + ".four_ld");
    mGridP = mf.hostF32(p + ".grid_P");
    mDurCent = mf.hostF32(p + ".dur_cent");
    mDurW = mf.hostF32(p + ".dur_W");
    mDrExp = mf.hostF32(p + ".dr_exp");
    DTT_CHECK(mDurCent.size() == N_DUR && mDurW.size() == N_DUR * N_DUR && mDrExp.size() == N_DR, "candidate tables");
    mEmbInst = mf.hostF32(p + ".emb.inst");
    mEmbPitch = mf.hostF32(p + ".emb.pitch");
    mEmbVel = mf.hostF32(p + ".emb.vel");
    W& w = *mW;
    w.cont0_w = mf.tensor(p + ".cont.0.w");
    w.cont0_b = mf.tensor(p + ".cont.0.b");
    w.cont2_w = mf.tensor(p + ".cont.2.w");
    w.cont2_b = mf.tensor(p + ".cont.2.b");
    w.evm0_w = mf.tensor(p + ".evm.0.w");
    w.evm0_b = mf.tensor(p + ".evm.0.b");
    w.evm2_w = mf.tensor(p + ".evm.2.w");
    w.evm2_b = mf.tensor(p + ".evm.2.b");
    w.emb_norm = mf.tensor(p + ".emb.norm");
    for (int i = 0; i < mLayers; i++) {
        const std::string b = format("%s.blk.%d", p.c_str(), i);
        w.blk.push_back({mf.tensor(b + ".qkv"), mf.tensor(b + ".o"), mf.tensor(b + ".w1"), mf.tensor(b + ".w2")});
    }
    w.head_w = mf.tensor(p + ".head.w");
    w.head_b = mf.tensor(p + ".head.b");
    DTT_CHECK(w.head_b->ne[0] == N_OUT, "head width %lld", static_cast<long long>(w.head_b->ne[0]));
}

RepairerNet::~RepairerNet() = default;

// ----------------------------------------------------------------------------------------------- host features
void RepairerNet::contFeatures(int n, const float* on, const float* dur, const int* pitch, float* out) const
{
    // fourier(x, 12, lo, hi): a = 2 pi x / lam -> [sin a, cos a]
    auto fourier = [](float x, const std::vector<float>& lam, float* f) {
        const float px = TWO_PI_F * x;
        for (size_t j = 0; j < lam.size(); j++) {
            const float a = px / lam[j];
            f[j] = std::sin(a);
            f[lam.size() + j] = std::cos(a);
        }
    };
    for (int i = 0; i < n; i++) {
        float* f = out + static_cast<size_t>(i) * N_CONT;
        const float ld = std::log(std::max(dur[i], 1e-4f));
        fourier(on[i], mFourOn, f);
        fourier(ld, mFourLd, f + 24);
        fourier(on[i] + dur[i], mFourOn, f + 48);
        f[72] = (ld + 3.0f) / 3.0f;
        f[73] = std::min(dur[i], 8.0f) / 2.0f;
    }

    // context_features: same-key neighbours, chord size, density (inst is 0 for every note: same_inst is always true)
    const float big = 10.0f;
    for (int i = 0; i < n; i++) {
        const float oni = on[i];
        const float offi = on[i] + dur[i];
        float prev = big, next = big;
        int prev_j = -1;
        int chord = 0, dens = 0, near = 0;
        for (int j = 0; j < n; j++) {
            const float dt = on[j] - oni;
            const float adt = std::abs(dt);
            chord += adt < 0.035f;
            dens += adt < 0.5f;
            if (j == i || pitch[j] != pitch[i]) {
                continue;
            }
            near += adt < 0.25f;
            if (dt <= 0.0f) {
                if (-dt < prev) { // torch.min: the first index of the minimum
                    prev = -dt;
                    prev_j = j;
                }
            } else if (dt < next) {
                next = dt;
            }
        }
        const float prev_off = prev_j >= 0 ? on[prev_j] + dur[prev_j] : 0.0f;
        const float prev_gap_off = prev < big ? oni - prev_off : big;
        const float next_overlap = next < big ? offi - (oni + next) : -big;
        auto sl = [](float x) { return signf(x) * std::log1p(std::abs(x) * 20.0f); };
        float* f = out + static_cast<size_t>(i) * N_CONT + 74;
        f[0] = sl(prev);
        f[1] = sl(prev_gap_off);
        f[2] = sl(next);
        f[3] = sl(next_overlap);
        f[4] = std::log1p(static_cast<float>(chord));
        f[5] = std::log1p(static_cast<float>(dens));
        f[6] = std::log1p(static_cast<float>(dens)); // dens_inst: one instrument
        f[7] = std::log1p(static_cast<float>(near));
    }

    // grid_features: a comb over 72 periods; the two strongest (3-period exclusion) give phase features
    const int K = static_cast<int>(mGridP.size());
    std::vector<float> cs(static_cast<size_t>(n) * K), sn(static_cast<size_t>(n) * K);
    for (int i = 0; i < n; i++) {
        const float px = TWO_PI_F * on[i];
        for (int k = 0; k < K; k++) {
            const float a = px / mGridP[k];
            cs[static_cast<size_t>(i) * K + k] = std::cos(a);
            sn[static_cast<size_t>(i) * K + k] = std::sin(a);
        }
    }
    std::vector<float> R(K), phase(K), Rm(K);
    const float nn = static_cast<float>(std::max(n, 1));
    for (int k = 0; k < K; k++) {
        const float zr = torchSum(cs.data() + k, K, n);
        const float zi = torchSum(sn.data() + k, K, n);
        R[k] = std::sqrt(zr * zr + zi * zi) / nn;
        phase[k] = std::atan2(zi, zr);
        Rm[k] = R[k];
    }
    static constexpr int SUB[6] = {1, 2, 3, 4, 6, 8};
    for (int pick = 0; pick < 2; pick++) {
        int kb = 0;
        for (int k = 1; k < K; k++) {
            if (Rm[k] > Rm[kb]) {
                kb = k;
            }
        }
        const float Pk = mGridP[kb], ph = phase[kb], rk = R[kb], lp = std::log(Pk);
        for (int i = 0; i < n; i++) {
            float* f = out + static_cast<size_t>(i) * N_CONT + 82 + pick * 14;
            const float base = (TWO_PI_F * on[i]) / Pk - ph;
            for (int m = 0; m < 6; m++) {
                const float a = static_cast<float>(SUB[m]) * base;
                f[2 * m] = std::sin(a);
                f[2 * m + 1] = std::cos(a);
            }
            f[12] = rk;
            f[13] = lp;
        }
        for (int k = 0; k < K; k++) {
            if (std::abs(k - kb) <= 3) {
                Rm[k] = -1.0f;
            }
        }
    }
}

// ----------------------------------------------------------------------------------------------- the graph
ggml_tensor* RepairerNet::block(ggml_context* ctx, ggml_tensor* x, int i, ggml_tensor* C3, ggml_tensor* S3, ggml_tensor* mask,
                                int Np)
{
    const W::Blk& w = mW->blk[i];
    const int64_t D = mD, H = mHeads, hd = mHd;
    // attention: rows [q, k, q', k', v]; RoPE as qk * C + qk' * S (see py/convert_evidence.py)
    ggml_tensor* h = ggml_rms_norm(ctx, x, RMS_EPS); // n1's weight is folded into qkv
    ggml_tensor* qkv = linear(ctx, w.qkv, nullptr, h); // [5D, Np]
    const size_t es = ggml_element_size(qkv);
    ggml_tensor* qk = ggml_view_3d(ctx, qkv, hd, 2 * H, Np, hd * es, qkv->nb[1], 0);
    ggml_tensor* qkp = ggml_view_3d(ctx, qkv, hd, 2 * H, Np, hd * es, qkv->nb[1], 2 * D * es);
    ggml_tensor* r = ggml_add(ctx, ggml_mul(ctx, qk, C3), ggml_mul(ctx, qkp, S3)); // [hd, 2H, Np]
    ggml_tensor* q = ggml_view_3d(ctx, r, hd, H, Np, r->nb[1], r->nb[2], 0);
    ggml_tensor* k = ggml_view_3d(ctx, r, hd, H, Np, r->nb[1], r->nb[2], H * hd * es);
    ggml_tensor* kp = ggml_permute(ctx, k, 0, 2, 1, 3); // [hd, Np, H]
    ggml_tensor* v = ggml_view_3d(ctx, qkv, hd, H, Np, hd * es, qkv->nb[1], 4 * D * es);
    ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3)); // [Np, hd, H]
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    ggml_tensor* att = nullptr;
    const int Qc = std::min(Np, mQChunk);
    for (int c0 = 0; c0 < Np; c0 += Qc) {
        const int nq = std::min(Qc, Np - c0);
        ggml_tensor* qc = ggml_view_3d(ctx, q, hd, H, nq, q->nb[1], q->nb[2], static_cast<size_t>(c0) * q->nb[2]);
        ggml_tensor* kq = ggml_mul_mat(ctx, kp, ggml_permute(ctx, qc, 0, 2, 1, 3)); // [Np, nq, H]
        accF32(kq);
        kq = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);
        ggml_tensor* o = ggml_mul_mat(ctx, vt, kq); // [hd, nq, H]
        accF32(o);
        o = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3)), D, nq);
        att = att == nullptr ? o : ggml_concat(ctx, att, o, 1);
    }
    x = ggml_add(ctx, x, linear(ctx, w.o, nullptr, att));
    // SwiGLU feed-forward (n2's weight folded into w1)
    ggml_tensor* g = linear(ctx, w.w1, nullptr, ggml_rms_norm(ctx, x, RMS_EPS)); // [2F, Np]
    g = ggml_swiglu(ctx, g);                                                     // silu(first half) * second half
    return ggml_add(ctx, x, linear(ctx, w.w2, nullptr, g));
}

void RepairerNet::build(int Np)
{
    const auto t0 = std::chrono::steady_clock::now();
    mG.reset();
    auto G = std::make_unique<Graph>(mBe);
    StaticGraph& sg = G->g;
    ggml_context* ctx = sg.ctx();
    const W& w = *mW;
    G->Np = Np;
    G->Qc = std::min(Np, mQChunk);
    G->emb = sg.input(GGML_TYPE_F32, {mD, Np}, "emb");
    G->cf = sg.input(GGML_TYPE_F32, {N_CONT, Np}, "cont");
    G->ev = sg.input(GGML_TYPE_F32, {mNEv, Np}, "ev");
    G->ropeC = sg.input(GGML_TYPE_F32, {mHd, Np}, "rope_c");
    G->ropeS = sg.input(GGML_TYPE_F32, {mHd, Np}, "rope_s");
    G->mask = sg.input(GGML_TYPE_F32, {Np, G->Qc}, "key_mask");
    auto mark = [&](ggml_tensor* t, const std::string& name) {
        if (mDebug) {
            sg.output(t, name.c_str());
            G->dbg.emplace_back(name, t);
        }
    };
    // NoteEmbed: (embeddings + cont MLP) + evidence MLP, RMSNorm
    ggml_tensor* x = ggml_add(ctx, G->emb, linear(ctx, w.cont2_w, w.cont2_b, ggml_silu(ctx, linear(ctx, w.cont0_w, w.cont0_b, G->cf))));
    x = ggml_add(ctx, x, linear(ctx, w.evm2_w, w.evm2_b, ggml_silu(ctx, linear(ctx, w.evm0_w, w.evm0_b, G->ev))));
    x = ggml_mul(ctx, ggml_rms_norm(ctx, x, RMS_EPS), w.emb_norm);
    mark(x, "emb");
    ggml_tensor* C3 = ggml_reshape_3d(ctx, G->ropeC, mHd, 1, Np);
    ggml_tensor* S3 = ggml_reshape_3d(ctx, G->ropeS, mHd, 1, Np);
    for (int i = 0; i < mLayers; i++) {
        x = block(ctx, x, i, C3, S3, G->mask, Np);
        mark(x, format("block_%d", i));
    }
    x = ggml_rms_norm(ctx, x, RMS_EPS); // bb.norm's weight is folded into the heads
    mark(x, "bb");
    G->out = linear(ctx, w.head_w, w.head_b, x); // [N_OUT, Np]
    sg.output(G->out, "out");
    sg.finalize("the evidence repairer");
    mStats.builds++;
    mStats.build_s += since(t0);
    mStats.compute_bytes_max = std::max(mStats.compute_bytes_max, sg.computeBytes());
    mStats.nodes_max = std::max(mStats.nodes_max, sg.nodes());
    mG = std::move(G);
}

void RepairerNet::run(int n, const int* pitch, const int* vel, const float* on, const float* dur, const float* ev,
                      std::vector<float>& out)
{
    DTT_CHECK(n >= 1, "empty window");
    mLastN = n;
    const int Np = bucket(n);
    if (!mG || mG->Np != Np || mG->Qc != std::min(Np, mQChunk) || mDebug == mG->dbg.empty()) {
        build(Np);
    }
    const auto t0 = std::chrono::steady_clock::now();
    const int D = mD, hd = mHd, Qc = mG->Qc;
    mInEmb.assign(static_cast<size_t>(D) * Np, 0.0f);
    mInCf.assign(static_cast<size_t>(N_CONT) * Np, 0.0f);
    mInEv.assign(static_cast<size_t>(mNEv) * Np, 0.0f);
    mInC.assign(static_cast<size_t>(hd) * Np, 1.0f);
    mInS.assign(static_cast<size_t>(hd) * Np, 0.0f);
    mInMask.assign(static_cast<size_t>(Np) * Qc, 0.0f);
    for (int i = 0; i < n; i++) {
        // inst(0) + pitch(p) + vel(v // 8), summed in that order
        DTT_CHECK(pitch[i] >= 0 && pitch[i] < 128 && vel[i] >= 0 && vel[i] < 128, "note %d: pitch %d vel %d", i, pitch[i], vel[i]);
        const float* ei = mEmbInst.data();
        const float* ep = mEmbPitch.data() + static_cast<size_t>(pitch[i]) * D;
        const float* evl = mEmbVel.data() + static_cast<size_t>(vel[i] / 8) * D;
        float* e = mInEmb.data() + static_cast<size_t>(i) * D;
        for (int c = 0; c < D; c++) {
            e[c] = (ei[c] + ep[c]) + evl[c];
        }
        // rotary angles: onset s x 10 frequencies, pitch x 4, each on an interleaved pair of every head's first 28 dims
        float* C = mInC.data() + static_cast<size_t>(i) * hd;
        float* S = mInS.data() + static_cast<size_t>(i) * hd;
        for (int j = 0; j < static_cast<int>(mRopeW.size()); j++) {
            const float pos = j < 10 ? on[i] : static_cast<float>(pitch[i]);
            const float a = pos * mRopeW[j];
            const float ca = std::cos(a), sa = std::sin(a);
            C[2 * j] = ca;
            C[2 * j + 1] = ca;
            S[2 * j] = -sa;
            S[2 * j + 1] = sa;
        }
    }
    std::memcpy(mInEv.data(), ev, static_cast<size_t>(n) * mNEv * sizeof(float));
    contFeatures(n, on, dur, pitch, mInCf.data());
    for (int r = 0; r < Qc; r++) {
        for (int j = n; j < Np; j++) {
            mInMask[static_cast<size_t>(r) * Np + j] = -std::numeric_limits<float>::infinity();
        }
    }
    mStats.features_s += since(t0);
    const auto t1 = std::chrono::steady_clock::now();
    ggml_backend_tensor_set(mG->emb, mInEmb.data(), 0, ggml_nbytes(mG->emb));
    ggml_backend_tensor_set(mG->cf, mInCf.data(), 0, ggml_nbytes(mG->cf));
    ggml_backend_tensor_set(mG->ev, mInEv.data(), 0, ggml_nbytes(mG->ev));
    ggml_backend_tensor_set(mG->ropeC, mInC.data(), 0, ggml_nbytes(mG->ropeC));
    ggml_backend_tensor_set(mG->ropeS, mInS.data(), 0, ggml_nbytes(mG->ropeS));
    ggml_backend_tensor_set(mG->mask, mInMask.data(), 0, ggml_nbytes(mG->mask));
    mG->g.compute();
    out.resize(static_cast<size_t>(n) * N_OUT);
    ggml_backend_tensor_get(mG->out, out.data(), 0, out.size() * sizeof(float));
    mStats.compute_s += since(t1);
}

void RepairerNet::debug(const std::string& name, std::vector<float>& out) const
{
    DTT_CHECK(mG, "no graph");
    for (const auto& [nm, t] : mG->dbg) {
        if (nm == name) {
            std::vector<float> all;
            readTensor(t, all);
            all.resize(static_cast<size_t>(mLastN) * t->ne[0]);
            out = std::move(all);
            return;
        }
    }
    throw Failure("no debug tensor " + name);
}

} // namespace dtt::ev
