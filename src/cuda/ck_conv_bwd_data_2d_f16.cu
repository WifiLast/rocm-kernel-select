// CK WMMA grouped-conv backward-data instantiation: 2D, ck::half_t.
// One dtype per translation unit so hipcc builds them in parallel -- see
// ck_conv_bwd_impl.hpp / ck_conv_fwd_impl.hpp for why that matters.
#include "ck_conv_bwd_impl.hpp"
#include "ck_conv_bwd_tu.hpp"

namespace amd_tuned_torch_ck {

int ck_count_bwd_data_f16() { return detail::count_bwd_data<ck::half_t>(); }

bool ck_entry_bwd_data_f16(int idx, int action, const ConvProblem& p, const void* p_grad_out,
                           const void* p_wei, void* p_grad_in, void* p_ws, size_t* out_ws,
                           hipStream_t stream) {
    return detail::entry_bwd_data<ck::half_t>(idx, static_cast<detail::Action>(action), p,
                                              p_grad_out, p_wei, p_grad_in, p_ws, out_ws, stream);
}

}  // namespace amd_tuned_torch_ck
