#include "drums/nets.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include "backend.hpp"
#include "common.hpp"
#include "ggml_compat.hpp"
#include "drums/gru_host.hpp"
#include "model_file.hpp"

namespace dtt
{

namespace
{
    constexpr float LN_EPS = 1e-5f; // nn.LayerNorm default
    constexpr float GN_EPS = 1e-5f; // nn.GroupNorm default
    constexpr int MAX_NODES = 4096;

    using Clock = std::chrono::steady_clock;
    double msSince(Clock::time_point t0)
    {
        return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }

    ggml_tensor* lin(ggml_context* ctx, ggml_tensor* w, ggml_tensor* b, ggml_tensor* x)
    {
        ggml_tensor* y = ggml_mul_mat(ctx, w, x);
        accF32(y);
        return b != nullptr ? ggml_add(ctx, y, b) : y;
    }

    /** A per-channel vector [C] as [1, 1, C, 1], for channels-first (WHCN) tensors. */
    ggml_tensor* chan(ggml_context* ctx, ggml_tensor* v)
    {
        return ggml_reshape_4d(ctx, v, 1, 1, v->ne[0], 1);
    }

    float sigmoidf(float x)
    {
        return 1.0f / (1.0f + std::exp(-x)); // torch: (1 + exp(-x)).reciprocal()
    }

    /**
     * Self-attention over the whole sequence, content only: qkv [3 * da, N] (q | k | v, each heads x D) -> [da, N]. Flash
     * attention: the N x N scores are never stored.
     */
    ggml_tensor* songAttention(ggml_context* ctx, ggml_tensor* qkv, int heads)
    {
        const int64_t da = qkv->ne[0] / 3, N = qkv->ne[1], D = da / heads;
        const size_t es = ggml_element_size(qkv);
        auto part = [&](int i) {
            ggml_tensor* t = ggml_view_3d(ctx, qkv, D, heads, N, D * es, qkv->nb[1], i * da * es); // [D, H, N]
            return ggml_permute(ctx, t, 0, 2, 1, 3);                                              // [D, N, H]
        };
        // Metal's flash attention kernels start at a head size of 32: zero-pad smaller heads (the scores q.k are
        // unchanged and the padded output channels are zero) and cut the padding off the result.
        const int64_t Dk = std::max<int64_t>(D, 32);
        auto padded = [&](int i) {
            ggml_tensor* t = part(i);
            return Dk == D ? t : ggml_pad(ctx, ggml_cont(ctx, t), static_cast<int>(Dk - D), 0, 0, 0);
        };
        ggml_tensor* o = ggml_flash_attn_ext(ctx, padded(0), padded(1), padded(2), nullptr, 1.0f / std::sqrt(static_cast<float>(D)), 0.0f, 0.0f);
        accF32(o);
        if (Dk != D) {
            o = ggml_cont(ctx, ggml_view_3d(ctx, o, D, heads, N, o->nb[1], o->nb[2], 0)); // [Dk, H, N] -> [D, H, N]
        }
        return ggml_reshape_2d(ctx, o, da, N); // [D, H, N] -> head-major channels, as torch's transpose + reshape
    }

    /**
     * nn.GroupNorm(G, C) + affine on a contiguous [F, T, C, 1] tensor (torch (1, C, T, F)): statistics over each group's
     * channels and the whole (T, F) plane, two passes (the mean, then the mean square of x - mean).
     *
     * ggml_group_norm computes the same, but its kernels parallelise over the groups only (4 workgroups on Vulkan, 4
     * threads on the CPU) for groups of ~17M values on a 4-min song. Here the sums are row reductions that parallelise
     * over rows. On the CPU backend SUM_ROWS, SUB and SQR run as one task, so there the row sums are matrix-vector
     * products with a ones vector (ones: an input of >= max(F, T) ones) and x - mean, d^2 are ADD / MUL.
     */
    ggml_tensor* groupNorm(ggml_context* ctx, ggml_tensor* x, int G, float eps, ggml_tensor* gamma, ggml_tensor* beta,
                           ggml_tensor* ones)
    {
        const int64_t A = x->ne[0], B = x->ne[1], C = x->ne[2], k = C / G;
        const float inv_n = 1.0f / static_cast<float>(A * B * k);
        auto rowSums = [&](ggml_tensor* t, int64_t n, int64_t rows) { // [n, rows] -> [rows] (as [1, rows] or [rows, 1])
            t = ggml_reshape_2d(ctx, t, n, rows);
            if (ones == nullptr) {
                return ggml_sum_rows(ctx, t);
            }
            ggml_tensor* y = ggml_mul_mat(ctx, t, ggml_view_1d(ctx, ones, n, 0));
            accF32(y);
            return y;
        };
        auto groupSum = [&](ggml_tensor* t) { // [A, B, C] -> [1, G]
            ggml_tensor* s = rowSums(rowSums(t, A, B * C), B, C); // [C]
            return ggml_reshape_2d(ctx, ggml_sum_rows(ctx, ggml_reshape_2d(ctx, s, k, G)), 1, G);
        };
        auto perChannel = [&](ggml_tensor* g) { // [1, G] -> [1, 1, C, 1]
            return ggml_reshape_4d(ctx, ggml_repeat_4d(ctx, g, k, G, 1, 1), 1, 1, C, 1);
        };
        ggml_tensor* neg_mean = ggml_scale(ctx, groupSum(x), -inv_n);                    // [1, G]
        ggml_tensor* d = ggml_add(ctx, x, perChannel(neg_mean));                          // x - mean
        ggml_tensor* var = ggml_scale_bias(ctx, groupSum(ggml_mul(ctx, d, d)), inv_n, eps); // var + eps
        ggml_tensor* a = ggml_div(ctx, ggml_reshape_4d(ctx, gamma, 1, 1, C, 1), perChannel(ggml_sqrt(ctx, var))); // gamma / std
        return ggml_add(ctx, ggml_mul(ctx, d, a), ggml_reshape_4d(ctx, beta, 1, 1, C, 1));
    }

    template <class F>
    void parallelRun(std::vector<F>& jobs)
    {
        std::vector<std::thread> pool;
        for (size_t i = 1; i < jobs.size(); i++) {
            pool.emplace_back(jobs[i]);
        }
        if (!jobs.empty()) {
            jobs[0]();
        }
        for (auto& t : pool) {
            t.join();
        }
    }
} // namespace

// ----------------------------------------------------------------------------------------------------------- weights
struct DrumStemLayer {
    char kind = 0; // 'c' conv + SiLU, 'r' residual separable block, 'p' max pool
    ggml_tensor *w = nullptr, *b = nullptr, *dw_w = nullptr, *dw_wcl = nullptr, *dw_b = nullptr, *pw_w = nullptr, *pw_b = nullptr;
    int pt = 1, pf = 1;
};

struct DrumNets::Net {
    std::vector<DrumStemLayer> stem;
    DrumBandGru bgru; // the band GRU on the host (CPU backend)
    int m = 0;        // band GRU hidden size
    ggml_tensor *ih_w, *ih_b, *hhf_w, *hhf_bn, *hhr_w, *hhr_bn, *proj_w, *proj_b;
    ggml_tensor *tb_dw_w, *tb_dw_b, *tb_pw_w, *tb_pw_b;
    ggml_tensor *gru_ih_w, *gru_ih_b, *gru_out_w, *gru_out_b;
    DrumGru gru;
    ggml_tensor *qkv_w, *qkv_b, *o_w, *o_b;
    ggml_tensor *rest_w, *rest_b, *vel_w, *vel_b, *tw_ih_w, *tw_ih_b;
    struct Tower {
        DrumGru gru;
        std::vector<float> head_w, head_b; // (|g|, 2h + W), (|g|)
        std::vector<int> idx;              // the drum-key lanes it writes
    };
    std::vector<Tower> towers;
};

struct DrumNets::Expert {
    Net base;
    ggml_tensor *c_w[3], *c_b[3], *n_w[3], *n_b[3], *proj_w, *proj_b, *inp_w, *inp_b;
    std::vector<ggml_tensor*> rnn_ih_w, rnn_ih_b;
    std::vector<DrumGru> rnn;
    ggml_tensor *qkv_w, *qkv_b, *ao_w, *ao_b, *ln_w, *ln_b, *delta_w, *delta_b, *timp_w, *timp_b;
    std::vector<float> g0_w, g0_b, g2_w, g2_b;
};

namespace
{
    void loadNet(const ModelFile& mf, const std::string& p, const std::string& spec, const std::vector<int32_t>& keys,
                 const std::vector<int32_t>& tower_sizes, const std::vector<int32_t>& tower_keys, DrumNets::Net& n)
    {
        int j = 0;
        size_t pos = 0;
        while (pos <= spec.size()) {
            size_t e = spec.find(',', pos);
            if (e == std::string::npos) {
                e = spec.size();
            }
            std::string tok = spec.substr(pos, e - pos);
            tok.erase(0, tok.find_first_not_of(' '));
            tok.erase(tok.find_last_not_of(' ') + 1);
            const std::string q = format("%s.stem.%d", p.c_str(), j);
            DrumStemLayer l;
            l.kind = tok.at(0);
            if (l.kind == 'c') {
                l.w = mf.tensor(q + ".w");
                l.b = mf.tensor(q + ".b");
            } else if (l.kind == 'r') {
                l.dw_w = mf.tensor(q + ".dw.w");
                l.dw_wcl = mf.tensor(q + ".dw.wcl");
                l.dw_b = mf.tensor(q + ".dw.b");
                l.pw_w = mf.tensor(q + ".pw.w");
                l.pw_b = mf.tensor(q + ".pw.b");
            } else if (l.kind == 'p') {
                l.pt = tok.at(1) - '0';
                l.pf = std::stoi(tok.substr(2));
            } else {
                throw Failure("unknown stem layer " + tok);
            }
            n.stem.push_back(l);
            j++;
            pos = e + 1;
        }
        n.ih_w = mf.tensor(p + ".bgru.ih.w");
        n.ih_b = mf.tensor(p + ".bgru.ih.b");
        n.hhf_w = mf.tensor(p + ".bgru.hhf.w");
        n.hhf_bn = mf.tensor(p + ".bgru.hhf.bn");
        n.hhr_w = mf.tensor(p + ".bgru.hhr.w");
        n.hhr_bn = mf.tensor(p + ".bgru.hhr.bn");
        n.proj_w = mf.tensor(p + ".bgru.proj.w");
        n.proj_b = mf.tensor(p + ".bgru.proj.b");
        n.m = static_cast<int>(n.hhf_w->ne[0]);
        n.bgru = DrumBandGru(mf, p);
        n.tb_dw_w = mf.tensor(p + ".t0.dw.w");
        n.tb_dw_b = mf.tensor(p + ".t0.dw.b");
        n.tb_pw_w = mf.tensor(p + ".t0.pw.w");
        n.tb_pw_b = mf.tensor(p + ".t0.pw.b");
        n.gru_ih_w = mf.tensor(p + ".t1.rnn.ih.w");
        n.gru_ih_b = mf.tensor(p + ".t1.rnn.ih.b");
        n.gru_out_w = mf.tensor(p + ".t1.out.w");
        n.gru_out_b = mf.tensor(p + ".t1.out.b");
        n.gru = DrumGru(mf, p + ".t1.rnn");
        n.qkv_w = mf.tensor(p + ".t2.qkv.w");
        n.qkv_b = mf.tensor(p + ".t2.qkv.b");
        n.o_w = mf.tensor(p + ".t2.o.w");
        n.o_b = mf.tensor(p + ".t2.o.b");
        n.rest_w = mf.tensor(p + ".rest.w");
        n.rest_b = mf.tensor(p + ".rest.b");
        n.vel_w = mf.tensor(p + ".vel.w");
        n.vel_b = mf.tensor(p + ".vel.b");
        n.tw_ih_w = mf.tensor(p + ".tw.ih.w");
        n.tw_ih_b = mf.tensor(p + ".tw.ih.b");
        size_t off = 0;
        for (size_t g = 0; g < tower_sizes.size(); g++) {
            DrumNets::Net::Tower t;
            const std::string q = format("%s.tw%zu", p.c_str(), g);
            t.gru = DrumGru(mf, q);
            t.head_w = mf.hostF32(q + ".head.w");
            t.head_b = mf.hostF32(q + ".head.b");
            for (int k = 0; k < tower_sizes[g]; k++) {
                const auto it = std::find(keys.begin(), keys.end(), tower_keys[off + k]);
                DTT_CHECK(it != keys.end(), "tower key %d is not a drum key", tower_keys[off + k]);
                t.idx.push_back(static_cast<int>(it - keys.begin()));
            }
            off += tower_sizes[g];
            DTT_CHECK(t.head_b.size() == t.idx.size(), "%s: head size", q.c_str());
            n.towers.push_back(std::move(t));
        }
    }
} // namespace

// ----------------------------------------------------------------------------------------------------------- graphs
/**
 * Builds and runs one graph at a time on the backend, one compute buffer reused (grown when needed) across the graphs of
 * a call. A Session is the exclusive use of both for one graph (build, compute, read back): threads take turns.
 */
class DrumNets::Exec
{
public:
    explicit Exec(const Backend& be) : mBe(be)
    {
        mAlloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be.get()));
    }
    ~Exec()
    {
        reset();
        ggml_gallocr_free(mAlloc);
    }
    Exec(const Exec&) = delete;
    Exec& operator=(const Exec&) = delete;

    class Session
    {
    public:
        explicit Session(Exec& e) : mE(e), mLock(e.mMutex) { mCtx = mE.begin(); }
        ~Session() { mE.reset(); }
        Session(const Session&) = delete;
        Session& operator=(const Session&) = delete;
        ggml_context* ctx() const { return mCtx; }

    private:
        Exec& mE;
        std::unique_lock<std::mutex> mLock;
        ggml_context* mCtx = nullptr;
    };

    /** An input; data must stay valid until compute(). */
    ggml_tensor* input(std::vector<int64_t> ne, const float* data)
    {
        ggml_tensor* t = ggml_new_tensor(mCtx, GGML_TYPE_F32, static_cast<int>(ne.size()), ne.data());
        ggml_set_input(t);
        mInputs.emplace_back(t, data);
        return t;
    }
    void output(ggml_tensor* t)
    {
        ggml_set_output(t);
        ggml_build_forward_expand(mGraph, t);
    }
    void compute(const char* what)
    {
        const std::string bad = mBe.firstUnsupportedOp(mGraph);
        DTT_CHECK(bad.empty(), "%s cannot run %s: %s", mBe.name().c_str(), what, bad.c_str());
        if (!ggml_gallocr_alloc_graph(mAlloc, mGraph)) {
            // a failed reserve leaves the allocator with no buffer but the failed graph's assignments: a later graph that
            // fits them would be placed in a null buffer. Start over with a fresh allocator.
            ggml_gallocr_free(mAlloc);
            mAlloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(mBe.get()));
            throw Failure(format("cannot allocate the compute buffer of %s on %s (out of device memory?)", what, mBe.name().c_str()));
        }
        mPeak = std::max(mPeak, ggml_gallocr_get_buffer_size(mAlloc, 0));
        for (auto& [t, d] : mInputs) {
            DTT_CHECK(t->buffer != nullptr, "%s: an input is not used by the graph", what);
            ggml_backend_tensor_set(t, d, 0, ggml_nbytes(t));
        }
        mBe.compute(mGraph);
    }
    static std::vector<float> get(ggml_tensor* t)
    {
        DTT_CHECK(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t), "%s: not a contiguous f32 tensor", t->name);
        std::vector<float> v(static_cast<size_t>(ggml_nelements(t)));
        ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
        return v;
    }
    size_t peakBytes() const { return mPeak; }

    /** A weight tensor's values on the host (f32 or f16). */
    static std::vector<float> hostCopy(ggml_tensor* t)
    {
        std::vector<float> v(static_cast<size_t>(ggml_nelements(t)));
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
        } else {
            DTT_CHECK(t->type == GGML_TYPE_F16, "%s: unsupported type", t->name);
            std::vector<ggml_fp16_t> h(v.size());
            ggml_backend_tensor_get(t, h.data(), 0, ggml_nbytes(t));
            ggml_fp16_to_fp32_row(h.data(), v.data(), static_cast<int64_t>(v.size()));
        }
        return v;
    }

private:
    ggml_context* begin()
    {
        reset();
        ggml_init_params ip = {ggml_tensor_overhead() * MAX_NODES * 4 + ggml_graph_overhead_custom(MAX_NODES, false), nullptr, true};
        mCtx = ggml_init(ip);
        DTT_CHECK(mCtx != nullptr, "ggml_init failed");
        mGraph = ggml_new_graph_custom(mCtx, MAX_NODES, false);
        return mCtx;
    }
    void reset()
    {
        if (mCtx != nullptr) {
            ggml_free(mCtx);
        }
        mCtx = nullptr;
        mGraph = nullptr;
        mInputs.clear();
    }

    const Backend& mBe;
    std::mutex mMutex;
    ggml_gallocr_t mAlloc = nullptr;
    ggml_context* mCtx = nullptr;
    ggml_cgraph* mGraph = nullptr;
    std::vector<std::pair<ggml_tensor*, const float*>> mInputs;
    size_t mPeak = 0;
};

namespace
{
    /** Adds the time from construction to destruction to *ms (if ms is not null). */
    class Stopwatch
    {
    public:
        explicit Stopwatch(double* ms) : mMs(ms), mT0(Clock::now()) {}
        ~Stopwatch()
        {
            if (mMs != nullptr) {
                *mMs += msSince(mT0);
            }
        }

    private:
        double* mMs;
        Clock::time_point mT0;
    };
} // namespace

DrumNets::DrumNets(const ModelFile& mf, const Backend& be) : mBe(be)
{
    const std::string p = "dr";
    const auto keys = mf.i32s(p + ".keys");
    mK = static_cast<int>(keys.size());
    mKT = static_cast<int>(mf.i32s(p + ".timpani_keys").size());
    mW = static_cast<int>(mf.u32(p + ".w"));
    mNMel = static_cast<int>(mf.u32(p + ".n_mel"));
    mNCq = static_cast<int>(mf.u32(p + ".n_cq"));
    mChunk = static_cast<int>(mf.u32(p + ".embed_chunk"));
    mMargin = static_cast<int>(mf.u32(p + ".embed_margin"));
    mKitLongAbove = static_cast<int>(mf.u32(p + ".kit_long_above"));
    mAttnHeads = static_cast<int>(mf.u32(p + ".attn.heads"));
    mExHeads = static_cast<int>(mf.u32(p + ".ex.heads"));
    const int k = static_cast<int>(mf.u32(p + ".tb.k"));
    mTbDil = static_cast<int>(mf.u32(p + ".tb.d"));
    mTbPad = mTbDil * (k / 2);
    mThRouter = mf.f32(p + ".th.router");
    mCqChunk = be.isCpu() ? 12000 : 4000;
    const std::string spec = mf.str(p + ".stem");
    const auto tsz = mf.i32s(p + ".tower_sizes");
    const auto tkeys = mf.i32s(p + ".tower_keys");
    mKit = std::make_unique<Net>();
    loadNet(mf, p + ".kit", spec, keys, tsz, tkeys, *mKit);
    const int n_ex = static_cast<int>(mf.u32(p + ".n_experts"));
    const int layers = static_cast<int>(mf.u32(p + ".ex.layers"));
    for (int i = 0; i < n_ex; i++) {
        auto e = std::make_unique<Expert>();
        const std::string q = format("%s.ex%d", p.c_str(), i);
        loadNet(mf, q + ".base", spec, keys, tsz, tkeys, e->base);
        for (int c = 0; c < 3; c++) {
            e->c_w[c] = mf.tensor(format("%s.cq.c%d.w", q.c_str(), c + 1));
            e->c_b[c] = mf.tensor(format("%s.cq.c%d.b", q.c_str(), c + 1));
            e->n_w[c] = mf.tensor(format("%s.cq.n%d.w", q.c_str(), c + 1));
            e->n_b[c] = mf.tensor(format("%s.cq.n%d.b", q.c_str(), c + 1));
        }
        e->proj_w = mf.tensor(q + ".cq.proj.w");
        e->proj_b = mf.tensor(q + ".cq.proj.b");
        e->inp_w = mf.tensor(q + ".inp.w");
        e->inp_b = mf.tensor(q + ".inp.b");
        for (int l = 0; l < layers; l++) {
            e->rnn_ih_w.push_back(mf.tensor(format("%s.rnn%d.ih.w", q.c_str(), l)));
            e->rnn_ih_b.push_back(mf.tensor(format("%s.rnn%d.ih.b", q.c_str(), l)));
            e->rnn.emplace_back(mf, format("%s.rnn%d", q.c_str(), l));
        }
        e->qkv_w = mf.tensor(q + ".qkv.w");
        e->qkv_b = mf.tensor(q + ".qkv.b");
        e->ao_w = mf.tensor(q + ".ao.w");
        e->ao_b = mf.tensor(q + ".ao.b");
        e->ln_w = mf.tensor(q + ".ln.w");
        e->ln_b = mf.tensor(q + ".ln.b");
        e->delta_w = mf.tensor(q + ".delta.w");
        e->delta_b = mf.tensor(q + ".delta.b");
        e->timp_w = mf.tensor(q + ".timp.w");
        e->timp_b = mf.tensor(q + ".timp.b");
        e->g0_w = mf.hostF32(q + ".gate0.w");
        e->g0_b = mf.hostF32(q + ".gate0.b");
        e->g2_w = mf.hostF32(q + ".gate2.w");
        e->g2_b = mf.hostF32(q + ".gate2.b");
        mExperts.push_back(std::move(e));
    }
}

DrumNets::~DrumNets() = default;

std::vector<float> DrumNets::embedChunk(Exec& ex, const Net& w, const float* feats, int stride, int a, int b, DrumTaps* taps,
                                        const std::string& p, DrumTiming* tm) const
{
    const int L = b - a, F0 = mNMel;
    std::vector<float> xin(static_cast<size_t>(L) * F0);
    for (int t = 0; t < L; t++) {
        for (int f = 0; f < F0; f++) {
            xin[static_cast<size_t>(t) * F0 + f] = (feats[static_cast<size_t>(a + t) * stride + f] + 5.0f) / 3.0f;
        }
    }
    std::vector<float> zeros;
    Exec::Session session(ex);
    Stopwatch sw(tm != nullptr ? &tm->embed : nullptr); // after the lock: the time this graph holds the backend
    ggml_context* ctx = session.ctx();
    // stem. On a GPU everything runs channels-first, [F, T, C, 1] (torch (B, C, T, F)). On the CPU backend the residual
    // blocks run channels-last, [C, F, T, 1]: ggml's CPU depthwise conv is vectorised over channels only in that layout
    // (the channels-first kernel is a scalar loop), and the 1x1 conv becomes a plain matmul instead of im2col + GEMM +
    // permute. The full convs and the max pools read channels-first, so the layout is switched around them.
    const bool cl = mBe.isCpu();
    ggml_tensor* x = ex.input({F0, L, 1, 1}, xin.data());
    bool is_cl = false; // the layout of x
    auto toCf = [&](ggml_tensor* t) { return is_cl ? ggml_cont(ctx, ggml_permute(ctx, t, 2, 0, 1, 3)) : t; }; // [C,F,T] -> [F,T,C]
    auto toCl = [&](ggml_tensor* t) { return is_cl ? t : ggml_cont(ctx, ggml_permute(ctx, t, 1, 2, 0, 3)); }; // [F,T,C] -> [C,F,T]
    for (const DrumStemLayer& l : w.stem) {
        if (l.kind == 'c') {
            x = ggml_conv_2d_direct(ctx, l.w, toCf(x), 1, 1, 1, 1, 1, 1);
            is_cl = false;
            if (cl) {
                x = toCl(x);
                is_cl = true;
                x = ggml_silu(ctx, ggml_add(ctx, x, l.b));
            } else {
                x = ggml_silu(ctx, ggml_add(ctx, x, chan(ctx, l.b)));
            }
        } else if (l.kind == 'r') {
            const int64_t C = l.pw_w->ne[0];
            if (cl) {
                x = toCl(x);
                is_cl = true;
                ggml_tensor* d = ggml_conv_2d_dw_direct(ctx, ggml_permute(ctx, l.dw_wcl, 3, 2, 0, 1),
                                                        ggml_permute(ctx, x, 2, 0, 1, 3), 1, 1, 1, 1, 1, 1); // CWHN in and out
                d = ggml_silu(ctx, ggml_add(ctx, ggml_permute(ctx, d, 1, 2, 0, 3), l.dw_b));          // [C, F, T]
                ggml_tensor* y = lin(ctx, l.pw_w, l.pw_b, ggml_reshape_2d(ctx, d, C, d->ne[1] * d->ne[2]));
                x = ggml_add(ctx, x, ggml_reshape_4d(ctx, y, C, x->ne[1], x->ne[2], 1));
            } else {
                ggml_tensor* d = ggml_conv_2d_dw_direct(ctx, l.dw_w, x, 1, 1, 1, 1, 1, 1);
                d = ggml_silu(ctx, ggml_add(ctx, d, chan(ctx, l.dw_b)));
                d = ggml_conv_2d_direct(ctx, ggml_reshape_4d(ctx, l.pw_w, 1, 1, C, C), d, 1, 1, 0, 0, 1, 1);
                x = ggml_add(ctx, x, ggml_add(ctx, d, chan(ctx, l.pw_b)));
            }
        } else {
            x = ggml_pool_2d(ctx, toCf(x), GGML_OP_POOL_MAX, l.pf, l.pt, l.pf, l.pt, 0.0f, 0.0f);
            is_cl = false;
        }
    }
    // the band GRU reads channels-last [C, F, N]
    ggml_tensor* xcl = toCl(x);
    x = toCf(x);
    const int64_t F = x->ne[0], N = x->ne[1], C = x->ne[2];
    ggml_tensor* stem_out = nullptr;
    if (taps != nullptr && taps->stem && p == "kit") {
        stem_out = x;
        ex.output(stem_out);
    }
    // BandGRU: a bidirectional GRU across the F bands of every frame (N sequences of F steps). Input projections of
    // both directions in one matmul: [2 * 3m, F, N] (b_hr, b_hz pre-added)
    ggml_tensor* x2 = ggml_reshape_2d(ctx, xcl, C, F * N);
    const int64_t m = w.m;
    const size_t es = sizeof(float);
    ggml_tensor* gi = ggml_reshape_3d(ctx, lin(ctx, w.ih_w, w.ih_b, x2), 6 * m, F, N);
    if (cl) {
        // CPU backend: the recurrence on the host (20 unrolled steps would be ~700 nodes, each a thread barrier)
        ex.output(gi);
        ex.compute("the drum stem");
        if (stem_out != nullptr) {
            taps->t[p + "_stem"] = Exec::get(stem_out);
        }
        const auto g = Exec::get(gi);
        std::vector<float> y(static_cast<size_t>(N) * mW);
        w.bgru.run(g.data(), static_cast<int>(N), static_cast<int>(F), y.data(), mBe.threads());
        return y;
    }
    // GPU: unrolled in the graph
    zeros.assign(static_cast<size_t>(m * N), 0.0f);
    ggml_tensor* h0 = ex.input({m, N}, zeros.data());
    auto direction = [&](int d, ggml_tensor* hh_w, ggml_tensor* bn, ggml_tensor*& hn, ggml_tensor*& sum) {
        ggml_tensor* h = h0;
        sum = nullptr;
        for (int64_t s = 0; s < F; s++) {
            const int64_t f = d == 1 ? F - 1 - s : s;
            ggml_tensor* gt = ggml_view_2d(ctx, gi, 3 * m, N, gi->nb[2], f * gi->nb[1] + d * 3 * m * es);
            ggml_tensor* gh = ggml_mul_mat(ctx, hh_w, h); // [3m, N]
            accF32(gh);
            ggml_tensor* rz = ggml_sigmoid(ctx, ggml_add(ctx, ggml_view_2d(ctx, gt, 2 * m, N, gt->nb[1], 0),
                                                         ggml_view_2d(ctx, gh, 2 * m, N, gh->nb[1], 0)));
            ggml_tensor* r = ggml_view_2d(ctx, rz, m, N, rz->nb[1], 0);
            ggml_tensor* z = ggml_view_2d(ctx, rz, m, N, rz->nb[1], m * es);
            ggml_tensor* ghn = ggml_add(ctx, ggml_view_2d(ctx, gh, m, N, gh->nb[1], 2 * m * es), bn);
            ggml_tensor* n = ggml_tanh(ctx, ggml_add(ctx, ggml_view_2d(ctx, gt, m, N, gt->nb[1], 2 * m * es), ggml_mul(ctx, r, ghn)));
            h = ggml_add(ctx, n, ggml_mul(ctx, z, ggml_sub(ctx, h, n))); // (1 - z) n + z h
            sum = sum == nullptr ? h : ggml_add(ctx, sum, h);
        }
        hn = h;
    };
    ggml_tensor *hf, *sf, *hr, *sr;
    direction(0, w.hhf_w, w.hhf_bn, hf, sf);
    direction(1, w.hhr_w, w.hhr_bn, hr, sr);
    const float inv = 1.0f / static_cast<float>(F);
    ggml_tensor* zc = ggml_concat(ctx, ggml_concat(ctx, hf, hr, 0), ggml_concat(ctx, ggml_scale(ctx, sf, inv), ggml_scale(ctx, sr, inv), 0), 0);
    ggml_tensor* y = lin(ctx, w.proj_w, w.proj_b, zc); // [W, N]
    ex.output(y);
    ex.compute("the drum stem");
    if (stem_out != nullptr) {
        taps->t[p + "_stem"] = Exec::get(stem_out); // torch (C, T, F)
    }
    return Exec::get(y);
}

std::vector<float> DrumNets::embed(Exec& ex, const Net& w, const float* feats, int stride, int T, bool long_path,
                                   DrumTaps* taps, const std::string& p, DrumTiming* tm) const
{
    if (!long_path) {
        return embedChunk(ex, w, feats, stride, 0, T, taps, p, tm);
    }
    // DrumNet.embed_long: windows of chunk frames with margin frames of context each side
    std::vector<float> out;
    out.reserve(static_cast<size_t>(T / 2) * mW);
    for (int s = 0; s < T; s += mChunk) {
        const int a = std::max(0, s - mMargin), b = std::min(T, s + mChunk + mMargin);
        const auto h = embedChunk(ex, w, feats, stride, a, b, taps, p, tm);
        const int lo = (s - a) / 2, cnt = (std::min(s + mChunk, T) - s) / 2;
        DTT_CHECK(static_cast<size_t>(lo + cnt) * mW <= h.size(), "embed_long window out of range");
        out.insert(out.end(), h.begin() + static_cast<size_t>(lo) * mW, h.begin() + static_cast<size_t>(lo + cnt) * mW);
    }
    return out;
}

void DrumNets::trunk(Exec& ex, const Net& w, const std::vector<float>& c, int N, std::vector<float>& h, std::vector<float>& z,
                     std::vector<float>& v, DrumTaps* taps, const std::string& p, DrumTiming* tm) const
{
    const int W = mW;
    // conv block, then the GRU block's input projections (its LayerNorm's affine folded in)
    std::vector<float> x1, gi;
    {
        Exec::Session session(ex);
        Stopwatch sw(tm != nullptr ? &tm->trunk : nullptr);
        ggml_context* ctx = session.ctx();
        ggml_tensor* x = ex.input({W, N}, c.data());
        ggml_tensor* k = ggml_permute(ctx, w.tb_dw_w, 3, 2, 0, 1);                                // [K, 1, 1, W]
        ggml_tensor* xv = ggml_permute(ctx, ggml_reshape_4d(ctx, x, W, N, 1, 1), 2, 0, 1, 3);    // CWHN view [N, 1, W, 1]
        ggml_tensor* d = ggml_conv_2d_dw_direct(ctx, k, xv, 1, 1, mTbPad, 0, mTbDil, 1);
        d = ggml_add(ctx, ggml_permute(ctx, d, 1, 2, 0, 3), w.tb_dw_b);                          // [W, N, 1, 1]
        d = ggml_silu(ctx, d);
        ggml_tensor* y1 = ggml_add(ctx, x, lin(ctx, w.tb_pw_w, w.tb_pw_b, ggml_reshape_2d(ctx, d, W, N)));
        ggml_tensor* g = lin(ctx, w.gru_ih_w, w.gru_ih_b, ggml_norm(ctx, y1, LN_EPS));
        ex.output(y1);
        ex.output(g);
        ex.compute("the drum conv block");
        x1 = Exec::get(y1);
        gi = Exec::get(g);
    }
    const int H = w.gru.hidden();
    std::vector<float> y(static_cast<size_t>(N) * 2 * H);
    {
        Stopwatch sw(tm != nullptr ? &tm->host_gru : nullptr);
        w.gru.runBoth(gi.data(), N, y.data());
    }
    // GRU block output + residual, song attention, the heads' projections
    std::vector<float> x2, tg;
    {
        Exec::Session session(ex);
        Stopwatch sw(tm != nullptr ? &tm->trunk : nullptr);
        ggml_context* ctx = session.ctx();
        ggml_tensor* xa = ex.input({W, N}, x1.data());
        ggml_tensor* ya = ex.input({2 * H, N}, y.data());
        ggml_tensor* a = ggml_add(ctx, xa, lin(ctx, w.gru_out_w, w.gru_out_b, ya));
        ggml_tensor* qkv = lin(ctx, w.qkv_w, w.qkv_b, ggml_norm(ctx, a, LN_EPS)); // the norm's affine folded into qkv
        ggml_tensor* att = songAttention(ctx, qkv, mAttnHeads);
        ggml_tensor* hh = ggml_add(ctx, a, lin(ctx, w.o_w, w.o_b, att));
        ggml_tensor* zr = lin(ctx, w.rest_w, w.rest_b, hh);
        ggml_tensor* vv = lin(ctx, w.vel_w, w.vel_b, hh);
        ggml_tensor* t = lin(ctx, w.tw_ih_w, w.tw_ih_b, hh);
        for (ggml_tensor* o : {a, hh, zr, vv, t}) {
            ex.output(o);
        }
        ex.compute("the drum trunk");
        x2 = Exec::get(a);
        h = Exec::get(hh);
        z = Exec::get(zr);
        v = Exec::get(vv);
        tg = Exec::get(t);
    }
    // family towers: one bidirectional GRU each over the whole song, then its head writes its keys
    const int K = mK;
    const int nt = static_cast<int>(w.towers.size());
    const int th = nt > 0 ? w.towers[0].gru.hidden() : 0;
    const int tstride = static_cast<int>(tg.size() / N);
    DTT_CHECK(tstride == nt * 6 * th, "tower projections: %d columns for %d towers of %d", tstride, nt, th);
    std::vector<float> tout(static_cast<size_t>(N) * nt * 2 * th);
    const int ostride = nt * 2 * th;
    {
        Stopwatch sw(tm != nullptr ? &tm->host_gru : nullptr);
        std::vector<std::function<void()>> jobs;
        for (int g = 0; g < nt; g++) {
            for (int dir = 0; dir < 2; dir++) {
                jobs.emplace_back([&, g, dir] {
                    w.towers[g].gru.run(dir, tg.data() + g * 6 * th + dir * 3 * th, tstride, N, tout.data() + g * 2 * th + dir * th, ostride);
                });
            }
        }
        parallelRun(jobs);
    }
    {
        Stopwatch sw(tm != nullptr ? &tm->trunk : nullptr);
        for (int g = 0; g < nt; g++) {
            const auto& tw = w.towers[g];
            const int ng = static_cast<int>(tw.idx.size()), in = 2 * th + W;
            for (int t = 0; t < N; t++) {
                const float* to = tout.data() + static_cast<size_t>(t) * ostride + g * 2 * th;
                const float* ht = h.data() + static_cast<size_t>(t) * W;
                for (int j = 0; j < ng; j++) {
                    const float* wr = tw.head_w.data() + static_cast<size_t>(j) * in;
                    double acc = 0.0;
                    for (int k2 = 0; k2 < 2 * th; k2++) {
                        acc += static_cast<double>(wr[k2]) * to[k2];
                    }
                    for (int k2 = 0; k2 < W; k2++) {
                        acc += static_cast<double>(wr[2 * th + k2]) * ht[k2];
                    }
                    z[static_cast<size_t>(t) * K + tw.idx[j]] = static_cast<float>(acc) + tw.head_b[j];
                }
            }
        }
    }
    if (taps != nullptr) {
        taps->t[p + "_tb"] = x1;
        taps->t[p + "_gru"] = x2;
        taps->t[p + "_h"] = h;
        taps->t[p + "_z"] = z;
        taps->t[p + "_v"] = v;
    }
}

ggml_tensor* DrumNets::cqConv(ggml_context* ctx, const Expert& w, int i, ggml_tensor* x) const
{
    // (pad, stride) per conv along (F, T): c1 (5, 7) pad (2, 3); c2 (3, 5) stride (2, 1) pad (1, 2); c3 (3, 5) pad (1, 2)
    static const int s1[3] = {1, 2, 1}, pf[3] = {3, 2, 2}, pt[3] = {2, 1, 1};
    x = ggml_conv_2d_direct(ctx, w.c_w[i], x, 1, s1[i], pf[i], pt[i], 1, 1);
    return ggml_add(ctx, x, chan(ctx, w.c_b[i]));
}

ggml_tensor* DrumNets::cqProj(ggml_context* ctx, const Expert& w, ggml_tensor* z3) const
{
    // (B, C, T, F) -> (B, T, C * F): per frame, index c * F + f
    ggml_tensor* pz = ggml_cont(ctx, ggml_permute(ctx, z3, 0, 2, 1, 3)); // [F, C, N]
    return ggml_silu(ctx, lin(ctx, w.proj_w, w.proj_b, ggml_reshape_2d(ctx, pz, pz->ne[0] * pz->ne[1], pz->ne[2])));
}

std::vector<float> DrumNets::cqChunked(Exec& ex, const Expert& w, const float* cqin, int T) const
{
    // The cq branch in windows of mCqChunk output frames, so the compute buffer does not grow with the song. Its three
    // GroupNorms take statistics over the whole song: pass p (0..2) runs every window up to layer p's conv and sums
    // its output (per channel, over the window's own frames only), with the earlier layers normalised by the statistics
    // of the earlier passes; pass 3 runs the whole branch. Each window carries HALO feature frames of context each side
    // (more than the three convs' reach, 2 + 2 + 2 frames): only the windows' own frames are kept or counted.
    constexpr int HALO = 16;  // feature frames, even (the stride-2 conv keeps its global phase)
    constexpr int GROUPS = 4; // nn.GroupNorm(4, .)
    const int N = T / 2, Fq = mNCq, U = mCqChunk;
    const size_t plane = static_cast<size_t>(T) * Fq;
    std::vector<float> sc[3], sh[3]; // per channel: y * sc + sh = GroupNorm(y) with its affine
    std::vector<float> out, ones(static_cast<size_t>(Fq), 1.0f);
    for (int pass = 0; pass <= 3; pass++) {
        const int C = pass < 3 ? static_cast<int>(w.c_b[pass]->ne[0]) : 0;
        std::vector<double> s1(C, 0.0), s2(C, 0.0);
        for (int u0 = 0; u0 < N; u0 += U) {
            const int u1 = std::min(N, u0 + U);
            const int fa = std::max(0, 2 * u0 - HALO), fb = std::min(T, 2 * u1 + HALO), L = fb - fa;
            std::vector<float> xin(static_cast<size_t>(3) * L * Fq);
            for (int ch = 0; ch < 3; ch++) {
                std::copy_n(cqin + ch * plane + static_cast<size_t>(fa) * Fq, static_cast<size_t>(L) * Fq, xin.data() + static_cast<size_t>(ch) * L * Fq);
            }
            std::vector<std::vector<float>> scv, shv;
            Exec::Session session(ex);
            ggml_context* ctx = session.ctx();
            ggml_tensor* x = ex.input({Fq, L, 3, 1}, xin.data());
            ggml_tensor* onest = nullptr;
            ggml_tensor* z2 = nullptr;
            ggml_tensor* rs = nullptr;
            ggml_tensor* rs2 = nullptr;
            for (int i = 0; i <= std::min(pass, 2); i++) {
                x = cqConv(ctx, w, i, x);
                if (i == pass) {
                    // statistics: sums over the 120 bands of y and y^2, per (frame, channel)
                    auto bandSums = [&](ggml_tensor* t) {
                        t = ggml_reshape_2d(ctx, t, Fq, t->ne[1] * t->ne[2]);
                        if (!mBe.isCpu()) {
                            return ggml_sum_rows(ctx, t); // [1, frames * C], exact f32 on the GPU
                        }
                        if (onest == nullptr) {
                            onest = ex.input({Fq}, ones.data());
                        }
                        ggml_tensor* y = ggml_mul_mat(ctx, t, onest); // SUM_ROWS is one task on the CPU backend
                        accF32(y);
                        return y;
                    };
                    rs = bandSums(x);
                    rs2 = bandSums(ggml_mul(ctx, x, x));
                    break;
                }
                const int Ci = static_cast<int>(x->ne[2]);
                ggml_tensor* a = ex.input({1, 1, Ci, 1}, sc[i].data());
                ggml_tensor* b = ex.input({1, 1, Ci, 1}, sh[i].data());
                x = ggml_silu(ctx, ggml_add(ctx, ggml_mul(ctx, x, a), b));
                if (i == 1) {
                    z2 = x;
                }
            }
            if (pass < 3) {
                ex.output(rs);
                ex.output(rs2);
                ex.compute("the drum cq branch statistics");
                const auto r1 = Exec::get(rs), r2 = Exec::get(rs2);
                // the window's own frames: feature frames [2 u0, 2 u1) for layer 0, frames [u0, u1) at half rate after it
                const int frames = static_cast<int>(r1.size() / C);
                const int k0 = pass == 0 ? 2 * u0 - fa : u0 - fa / 2, k1 = pass == 0 ? 2 * u1 - fa : u1 - fa / 2;
                DTT_CHECK(k0 >= 0 && k1 <= frames, "cq window bookkeeping");
                for (int c = 0; c < C; c++) {
                    for (int k = k0; k < k1; k++) {
                        s1[c] += r1[static_cast<size_t>(c) * frames + k];
                        s2[c] += r2[static_cast<size_t>(c) * frames + k];
                    }
                }
            } else {
                x = ggml_add(ctx, x, z2);
                ggml_tensor* cqo = cqProj(ctx, w, x); // [D, window frames]
                ex.output(cqo);
                ex.compute("the drum cq branch");
                const auto o = Exec::get(cqo);
                const int D = static_cast<int>(cqo->ne[0]);
                if (out.empty()) {
                    out.resize(static_cast<size_t>(N) * D);
                }
                const int k0 = u0 - fa / 2;
                std::copy_n(o.data() + static_cast<size_t>(k0) * D, static_cast<size_t>(u1 - u0) * D, out.data() + static_cast<size_t>(u0) * D);
            }
        }
        if (pass < 3) {
            // GroupNorm(4, C): mean and variance per group over its channels and every (frame, band); then the affine
            const auto gamma = ex.hostCopy(w.n_w[pass]), beta = ex.hostCopy(w.n_b[pass]);
            const int k = C / GROUPS;
            const double n = static_cast<double>(pass == 0 ? T : N) * Fq * k;
            sc[pass].resize(C);
            sh[pass].resize(C);
            for (int g = 0; g < GROUPS; g++) {
                double m1 = 0, m2 = 0;
                for (int c = g * k; c < (g + 1) * k; c++) {
                    m1 += s1[c];
                    m2 += s2[c];
                }
                const double mean = m1 / n, var = std::max(0.0, m2 / n - mean * mean);
                const double rstd = 1.0 / std::sqrt(var + GN_EPS);
                for (int c = g * k; c < (g + 1) * k; c++) {
                    sc[pass][c] = static_cast<float>(gamma[c] * rstd);
                    sh[pass][c] = static_cast<float>(beta[c] - mean * gamma[c] * rstd);
                }
            }
        }
    }
    return out;
}

float DrumNets::expert(Exec& ex, const Expert& w, const float* feats, int T, std::vector<float>& base, std::vector<float>& drum,
                       std::vector<float>& timp, std::vector<float>& vel, DrumTaps* taps, const std::string& p,
                       DrumTiming* tm) const
{
    const int stride = mNMel + mNCq, N = T / 2, K = mK;
    const bool long_path = T > mChunk + 2 * mMargin;
    const auto c = embed(ex, w.base, feats, stride, T, long_path, taps, p, tm);
    if (taps != nullptr) {
        taps->t[p + "_embed"] = c;
    }
    std::vector<float> h0, z0, v0;
    trunk(ex, w.base, c, N, h0, z0, v0, taps, p, tm);

    // CQBranch input: level-free (minus the mean over the whole song, / 4), stacked with itself shifted up 14 and 24 bands
    const int Fq = mNCq;
    std::vector<float> cqin(static_cast<size_t>(3) * T * Fq), sz0(z0.size());
    {
        Stopwatch sw(tm != nullptr ? &tm->cq : nullptr);
        double sum = 0.0;
        for (int t = 0; t < T; t++) {
            const float* r = feats + static_cast<size_t>(t) * stride + mNMel;
            for (int f = 0; f < Fq; f++) {
                sum += r[f];
            }
        }
        const float mean = static_cast<float>(sum) / static_cast<float>(static_cast<int64_t>(T) * Fq);
        float lo = INFINITY;
        for (int t = 0; t < T; t++) {
            const float* r = feats + static_cast<size_t>(t) * stride + mNMel;
            float* o = cqin.data() + static_cast<size_t>(t) * Fq;
            for (int f = 0; f < Fq; f++) {
                o[f] = (r[f] - mean) / 4.0f;
                lo = std::min(lo, o[f]);
            }
        }
        const size_t plane = static_cast<size_t>(T) * Fq;
        for (int s = 1; s <= 2; s++) {
            const int k = s == 1 ? 14 : 24;
            for (int t = 0; t < T; t++) {
                const float* src = cqin.data() + static_cast<size_t>(t) * Fq;
                float* o = cqin.data() + s * plane + static_cast<size_t>(t) * Fq;
                for (int f = 0; f < Fq; f++) {
                    o[f] = f + k < Fq ? src[f + k] : lo;
                }
            }
        }
        for (size_t i = 0; i < z0.size(); i++) {
            sz0[i] = sigmoidf(z0[i]);
        }
    }
    if (taps != nullptr && taps->stem && p == "ex0") {
        taps->t[p + "_cqin"] = cqin;
    }
    std::vector<float> cq, a, gi1;
    const bool chunked = mCqChunk > 0 && N > mCqChunk + mCqChunk / 2;
    if (chunked) {
        Stopwatch sw(tm != nullptr ? &tm->cq : nullptr); // includes waiting for the backend between the passes
        cq = cqChunked(ex, w, cqin.data(), T);
    } else {
        std::vector<float> ones;
        Exec::Session session(ex);
        Stopwatch sw(tm != nullptr ? &tm->cq : nullptr);
        ggml_context* ctx = session.ctx();
        ggml_tensor* x = ex.input({Fq, T, 3, 1}, cqin.data());
        ggml_tensor* ones_t = nullptr;
        if (mBe.isCpu()) {
            ones.assign(static_cast<size_t>(std::max(T, Fq)), 1.0f);
            ones_t = ex.input({static_cast<int64_t>(ones.size())}, ones.data());
        }
        ggml_tensor* cq1 = nullptr;
        ggml_tensor* z2 = nullptr;
        for (int i = 0; i < 3; i++) {
            x = cqConv(ctx, w, i, x);
            x = ggml_silu(ctx, groupNorm(ctx, x, 4, GN_EPS, w.n_w[i], w.n_b[i], ones_t));
            if (i == 0) {
                cq1 = x;
            }
            if (i == 1) {
                z2 = x;
            }
        }
        x = ggml_add(ctx, x, z2); // [F, N, C, 1]
        DTT_CHECK(x->ne[1] == N, "cq branch frames %lld != %d", static_cast<long long>(x->ne[1]), N);
        ggml_tensor* cqo = cqProj(ctx, w, x);
        ex.output(cqo);
        const bool tap1 = taps != nullptr && taps->stem && p == "ex0";
        if (tap1) {
            ex.output(cq1);
        }
        ex.compute("the drum cq branch");
        cq = Exec::get(cqo);
        if (tap1) {
            taps->t[p + "_cq1"] = Exec::get(cq1);
        }
    }
    {
        // the adapter's input and the first GRU layer's input projections
        Exec::Session session(ex);
        Stopwatch sw(tm != nullptr ? &tm->cq : nullptr);
        ggml_context* ctx = session.ctx();
        ggml_tensor* cqo = ex.input({static_cast<int64_t>(cq.size() / N), N}, cq.data());
        ggml_tensor* h0t = ex.input({mW, N}, h0.data());
        ggml_tensor* sz = ex.input({K, N}, sz0.data());
        ggml_tensor* in = ggml_concat(ctx, ggml_concat(ctx, h0t, cqo, 0), sz, 0);
        ggml_tensor* at = ggml_silu(ctx, lin(ctx, w.inp_w, w.inp_b, in));
        ggml_tensor* g1 = lin(ctx, w.rnn_ih_w[0], w.rnn_ih_b[0], at);
        ex.output(at);
        ex.output(g1);
        ex.compute("the drum adapter input");
        a = Exec::get(at);
        gi1 = Exec::get(g1);
    }
    // the adapter's 2-layer bidirectional GRU: recurrences on the host, the second layer's projections in a graph
    const int Hr = w.rnn[0].hidden();
    std::vector<float> r1(static_cast<size_t>(N) * 2 * Hr);
    {
        Stopwatch sw(tm != nullptr ? &tm->host_gru : nullptr);
        w.rnn[0].runBoth(gi1.data(), N, r1.data());
    }
    std::vector<float> r = r1;
    for (size_t l = 1; l < w.rnn.size(); l++) {
        std::vector<float> gi;
        {
            Exec::Session session(ex);
            Stopwatch sw(tm != nullptr ? &tm->adapter : nullptr);
            ggml_context* ctx = session.ctx();
            ggml_tensor* x = ex.input({2 * Hr, N}, r.data());
            ggml_tensor* g = lin(ctx, w.rnn_ih_w[l], w.rnn_ih_b[l], x);
            ex.output(g);
            ex.compute("the drum adapter GRU projections");
            gi = Exec::get(g);
        }
        Stopwatch sw(tm != nullptr ? &tm->host_gru : nullptr);
        std::vector<float> out(static_cast<size_t>(N) * 2 * w.rnn[l].hidden());
        w.rnn[l].runBoth(gi.data(), N, out.data());
        r.swap(out);
    }
    // self-attention over the whole song, LayerNorm, heads
    std::vector<float> att, ln, delta;
    {
        Exec::Session session(ex);
        Stopwatch sw(tm != nullptr ? &tm->adapter : nullptr);
        ggml_context* ctx = session.ctx();
        ggml_tensor* x = ex.input({2 * Hr, N}, r.data());
        ggml_tensor* qkv = lin(ctx, w.qkv_w, w.qkv_b, x);
        ggml_tensor* o = songAttention(ctx, qkv, mExHeads);
        ggml_tensor* at2 = ggml_add(ctx, x, lin(ctx, w.ao_w, w.ao_b, o));
        ggml_tensor* l2 = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, at2, LN_EPS), w.ln_w), w.ln_b);
        ggml_tensor* dl = lin(ctx, w.delta_w, w.delta_b, l2);
        ggml_tensor* tp = lin(ctx, w.timp_w, w.timp_b, l2);
        for (ggml_tensor* t : {at2, l2, dl, tp}) {
            ex.output(t);
        }
        ex.compute("the drum adapter");
        att = Exec::get(at2);
        ln = Exec::get(l2);
        delta = Exec::get(dl);
        timp = Exec::get(tp);
    }
    Stopwatch sw(tm != nullptr ? &tm->adapter : nullptr);
    // router: the gate on the LayerNorm output's mean over the whole recording
    const int D = 2 * Hr;
    std::vector<double> mean_r(D, 0.0);
    for (int t = 0; t < N; t++) {
        for (int k = 0; k < D; k++) {
            mean_r[k] += ln[static_cast<size_t>(t) * D + k];
        }
    }
    std::vector<float> mr(D);
    for (int k = 0; k < D; k++) {
        mr[k] = static_cast<float>(mean_r[k]) / static_cast<float>(N);
    }
    const int G = static_cast<int>(w.g0_b.size());
    double logit = w.g2_b[0];
    for (int j = 0; j < G; j++) {
        double acc = 0.0;
        for (int k = 0; k < D; k++) {
            acc += static_cast<double>(w.g0_w[static_cast<size_t>(j) * D + k]) * mr[k];
        }
        const float u = static_cast<float>(acc) + w.g0_b[j];
        const float sl = u / (1.0f + std::exp(-u)); // SiLU
        logit += static_cast<double>(w.g2_w[j]) * sl;
    }
    const float router = static_cast<float>(logit);
    base = z0;
    vel = v0;
    drum.resize(z0.size());
    for (size_t i = 0; i < z0.size(); i++) {
        drum[i] = z0[i] + delta[i];
    }
    if (taps != nullptr) {
        taps->t[p + "_cq"] = cq;
        taps->t[p + "_a"] = a;
        taps->t[p + "_r1"] = r1;
        taps->t[p + "_rnn"] = r;
        taps->t[p + "_att"] = att;
        taps->t[p + "_r"] = ln;
        taps->t[p + "_router"] = {router};
        taps->t[p + "_drum"] = drum;
        taps->t[p + "_timpani"] = timp;
    }
    return router;
}

DrumProbs DrumNets::run(const float* feats, int T0, bool all_models, DrumTaps* taps, DrumTiming* tm) const
{
    const int T = T0 / 2 * 2, N = T / 2, K = mK, KT = mKT;
    const int stride = mNMel + mNCq;
    Exec ex(mBe);
    DrumProbs P;
    P.N = N;
    P.K = K;
    P.KT = KT;
    // the experts (always: the router needs them). On a GPU they run in parallel threads that take turns on the
    // backend, so one expert's host GRUs overlap the others' graphs. On the CPU backend the graphs use every core and
    // ggml's spinning workers degrade badly next to busy threads: one after the other.
    const size_t n_ex = mExperts.size();
    struct Out {
        std::vector<float> base, drum, timp, vel;
        float router = 0;
        DrumTaps taps;
        DrumTiming tm;
        std::exception_ptr err;
    };
    std::vector<Out> outs(n_ex);
    auto one = [&](size_t i) {
        try {
            Out& o = outs[i];
            o.taps.stem = taps != nullptr && taps->stem;
            o.router = expert(ex, *mExperts[i], feats, T, o.base, o.drum, o.timp, o.vel, taps != nullptr ? &o.taps : nullptr,
                              format("ex%zu", i), tm != nullptr ? &o.tm : nullptr);
        } catch (...) {
            outs[i].err = std::current_exception();
        }
    };
    if (!mBe.isCpu() && n_ex > 1) {
        std::vector<std::thread> pool;
        for (size_t i = 0; i < n_ex; i++) {
            pool.emplace_back(one, i);
        }
        for (auto& t : pool) {
            t.join();
        }
    } else {
        for (size_t i = 0; i < n_ex; i++) {
            one(i);
        }
    }
    for (auto& o : outs) {
        if (o.err) {
            std::rethrow_exception(o.err);
        }
    }
    // probabilities(): sigmoid(base + (drum - base)), as during training; the mean over the experts accumulates in
    // float32 in order and divides in float64 (numpy's mean of a float32 stack by an intp count)
    const size_t nk = static_cast<size_t>(N) * K, nt = static_cast<size_t>(N) * KT;
    std::vector<float> sd(nk, 0.0f), st(nt, 0.0f), sv(nk, 0.0f);
    for (auto& o : outs) {
        P.routers.push_back(static_cast<double>(sigmoidf(o.router)));
        for (size_t j = 0; j < nk; j++) {
            sd[j] += sigmoidf(o.base[j] + (o.drum[j] - o.base[j]));
            sv[j] += sigmoidf(o.vel[j]);
        }
        for (size_t j = 0; j < nt; j++) {
            st[j] += sigmoidf(o.timp[j]);
        }
        if (taps != nullptr) {
            taps->t.merge(o.taps.t);
        }
        if (tm != nullptr) {
            tm->embed += o.tm.embed;
            tm->trunk += o.tm.trunk;
            tm->cq += o.tm.cq;
            tm->adapter += o.tm.adapter;
            tm->host_gru += o.tm.host_gru;
        }
    }
    const double ne = static_cast<double>(n_ex);
    double rs = 0.0;
    for (double r : P.routers) {
        rs += r;
    }
    P.router = rs / ne;
    auto div = [&](std::vector<float>& v) {
        for (float& x : v) {
            x = static_cast<float>(static_cast<double>(x) / ne);
        }
    };
    div(sd);
    div(st);
    div(sv);
    P.drum.swap(sd);
    P.timpani.swap(st);
    P.vel.swap(sv);
    // the drum-kit model
    if (all_models || P.router < static_cast<double>(mThRouter)) {
        const auto c = embed(ex, *mKit, feats, stride, T, T > mKitLongAbove, taps, "kit", tm);
        if (taps != nullptr) {
            taps->t["kit_embed"] = c;
        }
        std::vector<float> h, z, v;
        trunk(ex, *mKit, c, N, h, z, v, taps, "kit", tm);
        P.kit.resize(nk);
        P.kit_vel.resize(nk);
        for (size_t j = 0; j < nk; j++) {
            P.kit[j] = sigmoidf(z[j]);
            P.kit_vel[j] = sigmoidf(v[j]);
        }
    }
    if (tm != nullptr) {
        tm->compute_bytes = std::max(tm->compute_bytes, ex.peakBytes());
    }
    return P;
}

} // namespace dtt
