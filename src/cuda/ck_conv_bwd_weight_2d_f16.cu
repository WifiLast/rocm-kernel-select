// CK WMMA grouped-conv backward-weight instantiation: 2D, ck::half_t.
// One dtype per translation unit so hipcc builds them in parallel -- see
// ck_conv_bwd_impl.hpp / ck_conv_fwd_impl.hpp for why that matters.
#include "ck_conv_bwd_impl.hpp"
#include "ck_conv_bwd_tu.hpp"

namespace amd_tuned_torch_ck {

int ck_count_bwd_weight_f16() { return detail::count_bwd_weight<ck::half_t>(); }

bool ck_entry_bwd_weight_f16(int idx, int action, const ConvProblem& p, const void* p_in,
                             const void* p_grad_out, void* p_grad_wei, void* p_ws, size_t* out_ws,
                             hipStream_t stream) {
    return detail::entry_bwd_weight<ck::half_t>(idx, static_cast<detail::Action>(action), p, p_in,
                                                p_grad_out, p_grad_wei, p_ws, out_ws, stream);
}

}  // namespace amd_tuned_torch_ck
