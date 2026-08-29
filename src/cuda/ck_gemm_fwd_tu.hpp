// Internal seam between the CK GEMM translation units and the router
// (ck_gemm_fwd.cu). Deliberately free of any CK include, for the same
// reason ck_conv_fwd_tu.hpp is: the router and src/ck_gemm_torch.cpp
// compile against this in about a second, while only the eight
// instantiating units pay CK's build time.
#pragma once

#include "ck_gemm_fwd.hpp"

namespace amd_tuned_torch_ck_gemm {

// Matches detail::Action; kept as an int across this seam so the router
// needs none of CK's headers to name it.
enum Action { kActSupported = 0, kActWorkspace = 1, kActRun = 2 };

// One unit per (dtype, epilogue). Splitting this finely is what keeps the
// tier's wall clock to roughly one unit rather than eight -- see
// ck_gemm_fwd_impl.hpp's build-cost note.
#define AMD_TUNED_TORCH_CK_GEMM_TU(NAME)                                                    \
    int ck_gemm_count_##NAME();                                                             \
    bool ck_gemm_entry_##NAME(int idx, int action, const GemmProblem& p, const void* p_a,    \
                              const void* p_b, const void* p_bias, void* p_e, void* p_ws,   \
                              size_t* out_ws, hipStream_t stream);

AMD_TUNED_TORCH_CK_GEMM_TU(f16)
AMD_TUNED_TORCH_CK_GEMM_TU(bf16)
AMD_TUNED_TORCH_CK_GEMM_TU(add_f16)
AMD_TUNED_TORCH_CK_GEMM_TU(add_bf16)
AMD_TUNED_TORCH_CK_GEMM_TU(gelu_f16)
AMD_TUNED_TORCH_CK_GEMM_TU(gelu_bf16)
AMD_TUNED_TORCH_CK_GEMM_TU(silu_f16)
AMD_TUNED_TORCH_CK_GEMM_TU(silu_bf16)

#undef AMD_TUNED_TORCH_CK_GEMM_TU

}  // namespace amd_tuned_torch_ck_gemm
