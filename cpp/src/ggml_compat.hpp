#pragma once

// The ggml versions this library builds against: v1.0.0, and the v0.23 fork muscriptor.cpp pins (DamRsn/ggml b01bbb8), so a
// host can build both libraries on one ggml. The only API difference used here: v1.0.0 sets a result's accumulation
// precision with ggml_prec_set_acc; before it, one function per op did (ggml_mul_mat_set_prec,
// ggml_flash_attn_ext_set_prec). accF32() calls whichever this ggml has, picked at compile time (the calls depend on the
// template parameter, so the one that does not exist is never looked up).

#include <ggml.h>

namespace dtt
{
namespace detail
{
    template <typename T>
    auto accF32(T* t, int) -> decltype(ggml_prec_set_acc(t, GGML_PREC_F32), void())
    {
        ggml_prec_set_acc(t, GGML_PREC_F32);
    }

    template <typename T>
    void accF32(T* t, long)
    {
        if (t->op == GGML_OP_MUL_MAT) {
            ggml_mul_mat_set_prec(t, GGML_PREC_F32);
        } else if (t->op == GGML_OP_FLASH_ATTN_EXT) {
            ggml_flash_attn_ext_set_prec(t, GGML_PREC_F32);
        }
    }
} // namespace detail

/** Accumulate this matmul or flash-attention result in float32. */
inline void accF32(ggml_tensor* t)
{
    detail::accF32(t, 0);
}

} // namespace dtt
