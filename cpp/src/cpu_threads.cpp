// defaultThreadCount(): one thread per physical performance core this process may use.
//
// ggml's CPU threads meet at a barrier on every graph node, so an efficiency core, a hyper-thread sibling or a core
// shared with other work holds every node back; a host (a plugin inside a DAW) also needs cores left for itself.

#include "common.hpp"

#include <algorithm>
#include <thread>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <fstream>
#include <set>
#include <string>

#include <sched.h>
#elif defined(_WIN32)
#include <vector>

#include <windows.h>
#endif

namespace dtt
{

int defaultThreadCount()
{
    const int logical = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    int n = 0;
#if defined(__APPLE__)
    // perflevel0 is the performance tier; a Mac with uniform cores only has hw.physicalcpu
    size_t size = sizeof(n);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &n, &size, nullptr, 0) != 0 || n <= 0) {
        size = sizeof(n);
        if (sysctlbyname("hw.physicalcpu", &n, &size, nullptr, 0) != 0) {
            n = 0;
        }
    }
#elif defined(__linux__)
    auto parseList = [](const std::string& path) { // "0-7,16" -> {0..7, 16}; empty when the file is missing
        std::set<int> out;
        std::ifstream f(path);
        for (int a, b; f >> a;) {
            b = a;
            if (f.peek() == '-') {
                f.ignore();
                f >> b;
            }
            for (int i = a; i <= b; i++) {
                out.insert(i);
            }
            if (f.peek() == ',') {
                f.ignore();
            }
        }
        return out;
    };
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    const bool have_mask = sched_getaffinity(0, sizeof(allowed), &allowed) == 0;
    const std::set<int> pcores = parseList("/sys/devices/cpu_core/cpus"); // hybrid Intel: the performance cores
    std::set<int> cores;                                                   // one entry per physical core
    for (int c = 0; c < CPU_SETSIZE; c++) {
        if ((have_mask && !CPU_ISSET(c, &allowed)) || (!pcores.empty() && !pcores.contains(c))) {
            continue;
        }
        const auto sib = parseList("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list");
        if (!sib.empty()) {
            cores.insert(*sib.begin());
        }
    }
    n = static_cast<int>(cores.size());
    // a container's CPU quota (cgroup v2), which the affinity mask does not show
    std::ifstream quota("/sys/fs/cgroup/cpu.max");
    double q = 0, period = 0;
    if (n > 0 && quota >> q >> period && period > 0) {
        n = std::clamp(static_cast<int>(q / period + 0.5), 1, n);
    }
#elif defined(_WIN32)
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
    std::vector<char> buf(bytes);
    if (bytes > 0 && GetLogicalProcessorInformationEx(RelationProcessorCore,
                                                      reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()), &bytes)) {
        int best = -1; // the highest efficiency class is the performance tier
        for (DWORD o = 0; o < bytes;) {
            const auto* info = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + o);
            const int cls = info->Processor.EfficiencyClass;
            n = cls > best ? 1 : n + (cls == best ? 1 : 0);
            best = std::max(best, cls);
            o += info->Size;
        }
    }
#endif
    return n > 0 ? std::min(n, logical) : logical;
}

} // namespace dtt
