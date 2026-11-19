#pragma once

#include <string>

#include <ggml-backend.h>

namespace dtt
{

/** The one ggml backend every network of a pipeline runs on. */
class Backend
{
public:
    /**
     * @param device "cpu", "gpu" (first GPU, else an error), "auto" (first GPU, else the CPU) or a device index into
     *        ggml's device list.
     * @param n_threads CPU threads; 0 picks the hardware concurrency.
     */
    Backend(const std::string& device, int n_threads);
    ~Backend();
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    ggml_backend_t get() const { return mBackend; }
    bool isCpu() const { return mIsCpu; }
    int threads() const { return mThreads; }
    const std::string& name() const { return mName; }

    /** Compute a graph; throws on failure. */
    void compute(struct ggml_cgraph* graph) const;

    /** @return The first op of the graph the backend cannot run, as "OP (types)", or empty. */
    std::string firstUnsupportedOp(struct ggml_cgraph* graph) const;

    /** Lists ggml's devices, one per line, with their indices. */
    static std::string listDevices();

private:
    ggml_backend_t mBackend = nullptr;
    bool mIsCpu = true;
    int mThreads = 1;
    std::string mName;
};

} // namespace dtt
