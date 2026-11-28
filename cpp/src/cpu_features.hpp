#pragma once

// Run-time x86 feature checks for the host loops that have an FMA / AVX2 / F16C version picked at run time.
//
// GCC and Clang answer them with __builtin_cpu_supports, whose CPU table lives in libgcc or compiler-rt. clang-cl (Clang
// with the MSVC ABI, which is how NeuralNote builds this library) links neither, so there the same bits come from cpuid
// and xgetbv. MSVC's cl compiles none of the dispatched loops.

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace dtt::cpu
{
#if defined(_MSC_VER)
    namespace detail
    {
        struct Features {
            bool avx = false, avx2 = false, fma = false, f16c = false;
        };

        inline Features detect()
        {
            Features f;
            int r[4];
            __cpuid(r, 0);
            const int max_leaf = r[0];
            __cpuid(r, 1);
            const unsigned ecx = static_cast<unsigned>(r[2]);
            // AVX needs the OS to save the YMM registers: OSXSAVE, then XCR0 bits 1 (SSE) and 2 (AVX)
            if ((ecx & (1u << 27)) == 0) {
                return f;
            }
            unsigned xcr0 = 0, xcr0_hi = 0;
            __asm__ volatile("xgetbv" : "=a"(xcr0), "=d"(xcr0_hi) : "c"(0));
            if ((xcr0 & 6u) != 6u) {
                return f;
            }
            f.avx = (ecx & (1u << 28)) != 0;
            f.fma = f.avx && (ecx & (1u << 12)) != 0;
            f.f16c = f.avx && (ecx & (1u << 29)) != 0;
            if (max_leaf >= 7) {
                __cpuidex(r, 7, 0);
                f.avx2 = f.avx && (static_cast<unsigned>(r[1]) & (1u << 5)) != 0;
            }
            return f;
        }

        inline const Features& features()
        {
            static const Features f = detect();
            return f;
        }
    } // namespace detail

    inline bool hasAvx() { return detail::features().avx; }
    inline bool hasAvx2() { return detail::features().avx2; }
    inline bool hasFma() { return detail::features().fma; }
    inline bool hasF16c() { return detail::features().f16c; }
#else
    inline bool hasAvx() { return __builtin_cpu_supports("avx"); }
    inline bool hasAvx2() { return __builtin_cpu_supports("avx2"); }
    inline bool hasFma() { return __builtin_cpu_supports("fma"); }
    inline bool hasF16c() { return __builtin_cpu_supports("f16c"); }
#endif
} // namespace dtt::cpu
