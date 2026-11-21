#include "graph.hpp"

#include "backend.hpp"
#include "common.hpp"
#include "ggml_compat.hpp"

namespace dtt
{

StaticGraph::StaticGraph(const Backend& be, int max_nodes) : mBe(be)
{
    ggml_init_params ip = {ggml_tensor_overhead() * static_cast<size_t>(max_nodes) * 4 + ggml_graph_overhead_custom(max_nodes, false),
                           nullptr, true};
    mCtx = ggml_init(ip);
    DTT_CHECK(mCtx != nullptr, "ggml_init failed");
    mGraph = ggml_new_graph_custom(mCtx, max_nodes, false);
}

StaticGraph::~StaticGraph()
{
    if (mAlloc != nullptr) {
        ggml_gallocr_free(mAlloc);
    }
    if (mCtx != nullptr) {
        ggml_free(mCtx);
    }
}

ggml_tensor* StaticGraph::input(ggml_type type, std::vector<int64_t> ne, const char* name)
{
    DTT_CHECK(!ne.empty() && ne.size() <= 4, "input needs 1-4 dims");
    ggml_tensor* t = ggml_new_tensor(mCtx, type, static_cast<int>(ne.size()), ne.data());
    if (name != nullptr) {
        ggml_set_name(t, name);
    }
    ggml_set_input(t);
    return t;
}

void StaticGraph::output(ggml_tensor* t, const char* name)
{
    if (name != nullptr) {
        ggml_set_name(t, name);
    }
    ggml_set_output(t);
    ggml_build_forward_expand(mGraph, t);
}

void StaticGraph::expand(ggml_tensor* t)
{
    ggml_build_forward_expand(mGraph, t);
}

void StaticGraph::finalize(const char* what)
{
    const std::string bad = mBe.firstUnsupportedOp(mGraph);
    DTT_CHECK(bad.empty(), "%s cannot run %s: %s", mBe.name().c_str(), what, bad.c_str());
    mAlloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(mBe.get()));
    DTT_CHECK(ggml_gallocr_alloc_graph(mAlloc, mGraph), "cannot allocate the compute buffer of %s", what);
}

void StaticGraph::compute() const
{
    mBe.compute(mGraph);
}

size_t StaticGraph::computeBytes() const
{
    return mAlloc != nullptr ? ggml_gallocr_get_buffer_size(mAlloc, 0) : 0;
}

int StaticGraph::nodes() const
{
    return ggml_graph_n_nodes(mGraph);
}

ggml_tensor* linear(ggml_context* ctx, ggml_tensor* w, ggml_tensor* b, ggml_tensor* x)
{
    ggml_tensor* y = ggml_mul_mat(ctx, w, x);
    accF32(y);
    return b != nullptr ? ggml_add(ctx, y, b) : y;
}

void readTensor(ggml_tensor* t, std::vector<float>& out)
{
    DTT_CHECK(t != nullptr && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t), "readTensor needs a contiguous f32 tensor");
    out.resize(static_cast<size_t>(ggml_nelements(t)));
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
}

} // namespace dtt
