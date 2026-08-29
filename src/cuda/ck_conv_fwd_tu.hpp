// Internal seam between the CK translation units and the router
// (ck_conv_fwd.cu). Deliberately free of any CK include: the router and
// src/main_rocm.cpp compile against this in about a second, while only
// the four instantiating units below pay CK's minutes-long build.
#pragma once

#include "ck_conv_fwd.hpp"

namespace amd_tuned_torch_ck {

// Matches detail::Action; kept as an int across this seam so the router
// needs none of CK's headers to name it.
enum Action { kActSupported = 0, kActWorkspace = 1, kActRun = 2 };

#define AMD_TUNED_TORCH_CK_TU(NAME)                                                    \
    int ck_count_##NAME();                                                             \
    bool ck_entry_##NAME(int idx, int action, const ConvProblem& p, const void* p_in,   \
                         const void* p_wei, const void* p_bias, void* p_out,           \
                         void* p_ws, size_t* out_ws, hipStream_t stream);

AMD_TUNED_TORCH_CK_TU(2d_f16)
AMD_TUNED_TORCH_CK_TU(2d_bf16)
AMD_TUNED_TORCH_CK_TU(3d_f16)
AMD_TUNED_TORCH_CK_TU(3d_bf16)

#undef AMD_TUNED_TORCH_CK_TU

}  // namespace amd_tuned_torch_ck
