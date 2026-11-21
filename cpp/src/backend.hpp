#pragma once

#include <functional>
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
     * @param n_threads CPU threads; 0 picks defaultThreadCount().
     */
    Backend(const std::string& device, int n_threads);
    ~Backend();
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    ggml_backend_t get() const { return mBackend; }
    bool isCpu() const { return mIsCpu; }
    int threads() const { return mThreads; }
    const std::string& name() const { return mName; }

    /** Compute a graph; throws on failure, and Cancelled when the check of setCancel() stopped it. */
    void compute(struct ggml_cgraph* graph) const;

    /**
     * A check the CPU backend polls at every graph node (a window of the transcriber takes seconds there); empty to
     * clear it. GPU graphs are short, so their callers poll between them.
     */
    void setCancel(std::function<bool()> should_cancel) { mCancel = std::move(should_cancel); }

    /** @return The first op of the graph the backend cannot run, as "OP (types)", or empty. */
    std::string firstUnsupportedOp(struct ggml_cgraph* graph) const;
    /**
     * Whether a matmul may broadcast its first operand over dim 3 (a [K, M, C, 1] x b [K, N, C, B]). ggml-vulkan before
     * v1.0.0 cannot (the v0.23 fork muscriptor.cpp pins); the PitchMix blocks then repeat their matrices per window.
     */
    bool matmulBroadcastsDim3() const;

    /**
     * The number of devices in ggml's registry (every backend ggml was built with); index them with
     * ggml_backend_dev_get. 0 where the registry must not be touched: creating it calls into every backend, and on a
     * Windows machine without a Vulkan loader a ggml with the Vulkan backend then faults, because hosts delay-load
     * vulkan-1.dll so that they still start there (muscriptor.cpp does; so does this project's own ggml). The CPU is
     * not in question: it is never taken from the registry.
     */
    static size_t deviceCount();

    /** Lists ggml's devices, one per line, with their indices. */
    static std::string listDevices();

private:
    ggml_backend_t mBackend = nullptr;
    mutable int mBcast3 = -1; // matmulBroadcastsDim3(), probed once
    std::function<bool()> mCancel;
    bool mIsCpu = true;
    int mThreads = 1;
    std::string mName;
};

} // namespace dtt
