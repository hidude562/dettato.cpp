#include "backend.hpp"

#include <algorithm>
#include <cctype>
#include <thread>

#include <ggml-cpu.h>
#include <ggml.h>

#include "common.hpp"

namespace dtt
{

namespace
{
    bool isGpu(ggml_backend_dev_t dev)
    {
        const auto t = ggml_backend_dev_type(dev);
        return t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU;
    }
} // namespace

Backend::Backend(const std::string& device, int n_threads)
{
    mThreads = n_threads > 0 ? n_threads : std::max(1u, std::thread::hardware_concurrency());
    ggml_backend_dev_t dev = nullptr;
    const bool is_index = !device.empty() && std::all_of(device.begin(), device.end(), [](char c) { return std::isdigit(static_cast<unsigned char>(c)); });

    if (is_index) {
        const size_t i = std::stoul(device);
        DTT_CHECK(i < ggml_backend_dev_count(), "device index %zu out of range (%zu devices)", i, ggml_backend_dev_count());
        dev = ggml_backend_dev_get(i);
    } else if (device == "gpu" || device == "auto") {
        // Discrete before integrated.
        for (auto want : {GGML_BACKEND_DEVICE_TYPE_GPU, GGML_BACKEND_DEVICE_TYPE_IGPU}) {
            for (size_t i = 0; i < ggml_backend_dev_count() && dev == nullptr; i++) {
                if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == want) {
                    dev = ggml_backend_dev_get(i);
                }
            }
        }
        DTT_CHECK(dev != nullptr || device == "auto", "no GPU device");
    } else {
        DTT_CHECK(device == "cpu", "unknown device '%s' (cpu, gpu, auto or an index)", device.c_str());
    }

    if (dev != nullptr && isGpu(dev)) {
        mBackend = ggml_backend_dev_init(dev, nullptr);
        DTT_CHECK(mBackend != nullptr, "could not initialise %s", ggml_backend_dev_description(dev));
        mIsCpu = false;
        mName = std::string(ggml_backend_dev_name(dev)) + " (" + ggml_backend_dev_description(dev) + ")";
    } else {
        mBackend = ggml_backend_cpu_init();
        DTT_CHECK(mBackend != nullptr, "could not initialise the CPU backend");
        ggml_backend_cpu_set_n_threads(mBackend, mThreads);
        mIsCpu = true;
        mName = format("CPU (%d threads)", mThreads);
    }
}

Backend::~Backend()
{
    if (mBackend != nullptr) {
        ggml_backend_free(mBackend);
    }
}

void Backend::compute(ggml_cgraph* graph) const
{
    const ggml_status st = ggml_backend_graph_compute(mBackend, graph);
    DTT_CHECK(st == GGML_STATUS_SUCCESS, "graph compute failed (status %d)", static_cast<int>(st));
}

std::string Backend::firstUnsupportedOp(ggml_cgraph* graph) const
{
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        ggml_tensor* t = ggml_graph_node(graph, i);
        if (!ggml_backend_supports_op(mBackend, t)) {
            std::string s = std::string(ggml_op_desc(t)) + " (" + ggml_type_name(t->type);
            for (int j = 0; j < GGML_MAX_SRC && t->src[j] != nullptr; j++) {
                s += std::string(", ") + ggml_type_name(t->src[j]->type);
            }
            return s + ") " + t->name;
        }
    }
    return {};
}

std::string Backend::listDevices()
{
    std::string s;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(d, &free, &total);
        s += format("%2zu  %-10s %-40s %6.2f GB\n", i, ggml_backend_dev_name(d), ggml_backend_dev_description(d), total / 1e9);
    }
    return s;
}

} // namespace dtt
