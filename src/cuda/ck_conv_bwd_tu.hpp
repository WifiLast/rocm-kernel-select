// Internal seam between the CK backward translation units and the router
// (ck_conv_bwd.cu). Deliberately free of any CK include, same reasoning
// as ck_conv_fwd_tu.hpp: the router and the torch-facing .cpp compile
// against this in about a second, while only the four instantiating
// units below pay CK's minutes-long build.
#pragma once

#include "ck_conv_bwd.hpp"

namespace amd_tuned_torch_ck {

// Matches detail::Action (ck_conv_bwd_impl.hpp, reused from
// ck_conv_fwd_impl.hpp); kept as an int across this seam so the router
// needs none of CK's headers to name it.
enum ActionBwd { kActBwdSupported = 0, kActBwdWorkspace = 1, kActBwdRun = 2 };

#define AMD_TUNED_TORCH_CK_BWD_DATA_TU(NAME)                                            \
    int ck_count_bwd_data_##NAME();                                                     \
    bool ck_entry_bwd_data_##NAME(int idx, int action, const ConvProblem& p,            \
                                  const void* p_grad_out, const void* p_wei,            \
                                  void* p_grad_in, void* p_ws, size_t* out_ws,          \
                                  hipStream_t stream);

#define AMD_TUNED_TORCH_CK_BWD_WEIGHT_TU(NAME)                                          \
    int ck_count_bwd_weight_##NAME();                                                   \
    bool ck_entry_bwd_weight_##NAME(int idx, int action, const ConvProblem& p,          \
                                    const void* p_in, const void* p_grad_out,           \
                                    void* p_grad_wei, void* p_ws, size_t* out_ws,       \
                                    hipStream_t stream);

AMD_TUNED_TORCH_CK_BWD_DATA_TU(f16)
AMD_TUNED_TORCH_CK_BWD_DATA_TU(bf16)
AMD_TUNED_TORCH_CK_BWD_WEIGHT_TU(f16)
AMD_TUNED_TORCH_CK_BWD_WEIGHT_TU(bf16)

#undef AMD_TUNED_TORCH_CK_BWD_DATA_TU
#undef AMD_TUNED_TORCH_CK_BWD_WEIGHT_TU

}  // namespace amd_tuned_torch_ck
