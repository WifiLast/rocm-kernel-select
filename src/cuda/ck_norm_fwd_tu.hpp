// Internal seam between the CK normalization translation units and the
// router (ck_norm_fwd.cu). Deliberately free of any CK include, for the
// same reason ck_conv_fwd_tu.hpp is: the router and src/ck_norm_torch.cpp
// compile against this in about a second, while only the three
// instantiating units pay CK's build time.
#pragma once

#include "ck_norm_fwd.hpp"

namespace amd_tuned_torch_ck_norm {

// Matches detail::Action; kept as an int across this seam so the router
// needs none of CK's headers to name it.
enum Action { kActSupported = 0, kActWorkspace = 1, kActRun = 2 };

// Both Y-elementwise ops live in one translation unit per dtype. They
// share every template argument but the epilogue functor, so splitting
// them further would duplicate the instantiation cost without buying any
// more compile parallelism than the three dtypes already give.
#define AMD_TUNED_TORCH_CK_NORM_TU(NAME)                                                    \
    int ck_norm_count_##NAME(int yop);                                                      \
    bool ck_norm_entry_##NAME(int yop, int idx, int action, const NormProblem& p,            \
                              const void* p_x, const void* p_gamma, const void* p_beta,     \
                              void* p_y, void* p_ws, size_t* out_ws, hipStream_t stream);

AMD_TUNED_TORCH_CK_NORM_TU(f16)
AMD_TUNED_TORCH_CK_NORM_TU(bf16)
AMD_TUNED_TORCH_CK_NORM_TU(f32)

#undef AMD_TUNED_TORCH_CK_NORM_TU

}  // namespace amd_tuned_torch_ck_norm
