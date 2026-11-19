#pragma once

#include <string>
#include <vector>

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

namespace dtt
{

class Backend;

/**
 * A ggml graph built once and computed many times: one context for the node metadata, one compute buffer from the
 * graph allocator. Typical use:
 *
 *     StaticGraph g(be, 8192);
 *     ggml_tensor* x = g.input(GGML_TYPE_F32, {D, N});
 *     ggml_tensor* y = linear(g.ctx(), w, b, x);
 *     g.output(y, "y");
 *     g.finalize("the evidence repairer");      // op-support check + allocation; throws with the first bad op
 *     ... ggml_backend_tensor_set(x, ...); g.compute(); ggml_backend_tensor_get(y, ...);
 *
 * Inputs are re-set before every compute (the allocator may reuse their memory once the graph has consumed them).
 */
class StaticGraph
{
public:
    StaticGraph(const Backend& be, int max_nodes);
    ~StaticGraph();
    StaticGraph(const StaticGraph&) = delete;
    StaticGraph& operator=(const StaticGraph&) = delete;

    ggml_context* ctx() const { return mCtx; }
    ggml_cgraph* graph() const { return mGraph; }

    /** A new input tensor, ne fastest first (1-4 dims). */
    ggml_tensor* input(ggml_type type, std::vector<int64_t> ne, const char* name = nullptr);
    /** Marks a result to keep and read back; adds it (and what it depends on) to the graph. */
    void output(ggml_tensor* t, const char* name = nullptr);
    /** Adds a tensor to the graph without marking it as an output (e.g. a set_rows chain). */
    void expand(ggml_tensor* t);
    /** Checks every op against the backend, then allocates. what names the network in the error. */
    void finalize(const char* what);
    void compute() const;
    size_t computeBytes() const;
    int nodes() const;

private:
    const Backend& mBe;
    ggml_context* mCtx = nullptr;
    ggml_cgraph* mGraph = nullptr;
    ggml_gallocr_t mAlloc = nullptr;
};

/** y = w x + b with f32 accumulation (b may be null). w ne [K, M], x ne [K, N, ...]. */
ggml_tensor* linear(ggml_context* ctx, ggml_tensor* w, ggml_tensor* b, ggml_tensor* x);

/** Reads a contiguous tensor into a float vector. */
void readTensor(ggml_tensor* t, std::vector<float>& out);

} // namespace dtt
