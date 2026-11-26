#include "backend.hpp"

#include <algorithm>
#include <cctype>
#include <thread>

#include <ggml-cpu.h>
#include <ggml.h>

#include "common.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

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

size_t Backend::deviceCount()
{
    static const size_t count = []() -> size_t {
#if defined(_WIN32) && defined(DETTATO_HAS_VULKAN)
        if (LoadLibraryW(L"vulkan-1.dll") == nullptr) {
            return 0; // no Vulkan loader: the first Vulkan call would fault (see the header)
        }
        // Implicit Vulkan layers, which screen recorders and overlays install system-wide, can crash the driver when
        // ggml creates its device (found by muscriptor.cpp): keep them out of the instance created here. The loader
        // reads the variable in vkCreateInstance only; a value that is already set is left alone.
        const wchar_t* layers = L"VK_LOADER_LAYERS_DISABLE";
        const bool set = GetEnvironmentVariableW(layers, nullptr, 0) == 0 && SetEnvironmentVariableW(layers, L"~implicit~") != 0;
#endif
        size_t n = 0;
        try {
            n = ggml_backend_dev_count();
        } catch (...) {
            n = 0; // a backend that throws while it lists its devices (a broken driver) leaves the CPU
        }
#if defined(_WIN32) && defined(DETTATO_HAS_VULKAN)
        if (set) {
            SetEnvironmentVariableW(layers, nullptr);
        }
#endif
        return n;
    }();
    return count;
}

Backend::Backend(const std::string& device, int n_threads)
{
    mThreads = n_threads > 0 ? n_threads : defaultThreadCount();
    ggml_backend_dev_t dev = nullptr;
    const bool is_index = !device.empty() && std::all_of(device.begin(), device.end(), [](char c) { return std::isdigit(static_cast<unsigned char>(c)); });

    if (is_index) {
        const size_t i = std::stoul(device);
        DTT_CHECK(i < deviceCount(), "device index %zu out of range (%zu devices)", i, deviceCount());
        dev = ggml_backend_dev_get(i);
    } else if (device == "gpu" || device == "auto") {
        // Discrete before integrated.
        for (auto want : {GGML_BACKEND_DEVICE_TYPE_GPU, GGML_BACKEND_DEVICE_TYPE_IGPU}) {
            for (size_t i = 0; i < deviceCount() && dev == nullptr; i++) {
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
        ggml_backend_cpu_set_abort_callback(
            mBackend, [](void* self) { const auto& f = static_cast<Backend*>(self)->mCancel; return f && f(); }, this);
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
    throwIfCancelled();
    const ggml_status st = ggml_backend_graph_compute(mBackend, graph);
    if (st == GGML_STATUS_ABORTED) {
        throw Cancelled();
    }
    DTT_CHECK(st == GGML_STATUS_SUCCESS, "graph compute failed (status %d)", static_cast<int>(st));
}

std::string Backend::firstUnsupportedOp(ggml_cgraph* graph) const
{
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        ggml_tensor* t = ggml_graph_node(graph, i);
        if (!ggml_backend_supports_op(mBackend, t)) {
            // the op, then each tensor's type and shape (ne, fastest first), marked when it is not contiguous
            auto desc = [](const ggml_tensor* x) {
                return format("%s [%lld, %lld, %lld, %lld]%s", ggml_type_name(x->type), static_cast<long long>(x->ne[0]),
                              static_cast<long long>(x->ne[1]), static_cast<long long>(x->ne[2]),
                              static_cast<long long>(x->ne[3]), ggml_is_contiguous(x) ? "" : " non-contiguous");
            };
            std::string s = std::string(ggml_op_desc(t)) + " " + t->name + ": " + desc(t);
            for (int j = 0; j < GGML_MAX_SRC && t->src[j] != nullptr; j++) {
                s += std::string(j == 0 ? " <- " : ", ") + desc(t->src[j]);
            }
            return s;
        }
    }
    return {};
}

bool Backend::matmulBroadcastsDim3() const
{
    if (mBcast3 < 0) {
        ggml_init_params p{4 * ggml_tensor_overhead(), nullptr, true};
        ggml_context* ctx = ggml_init(p);
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 8, 2, 1);
        ggml_tensor* b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 16, 2, 4);
        mBcast3 = ggml_backend_supports_op(mBackend, ggml_mul_mat(ctx, a, b)) ? 1 : 0;
        ggml_free(ctx);
    }
    return mBcast3 == 1;
}

std::string Backend::listDevices()
{
    std::string s;
    for (size_t i = 0; i < deviceCount(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(d, &free, &total);
        s += format("%2zu  %-10s %-40s %6.2f GB\n", i, ggml_backend_dev_name(d), ggml_backend_dev_description(d), total / 1e9);
    }
    return s;
}

} // namespace dtt
