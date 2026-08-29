// CK WMMA grouped-conv-forward instantiation: 2D, ck::bhalf_t.
// One (rank, dtype) per translation unit so hipcc builds them in
// parallel -- see ck_conv_fwd_impl.hpp for why that matters.
#include "ck_conv_fwd_impl.hpp"
#include "ck_conv_fwd_tu.hpp"

namespace amd_tuned_torch_ck {

int ck_count_2d_bf16() { return detail::count<2, ck::bhalf_t>(); }

bool ck_entry_2d_bf16(int idx, int action, const ConvProblem& p, const void* p_in,
                      const void* p_wei, const void* p_bias, void* p_out,
                      void* p_ws, size_t* out_ws, hipStream_t stream) {
    return detail::entry<2, ck::bhalf_t>(idx, static_cast<detail::Action>(action), p,
                                        p_in, p_wei, p_bias, p_out, p_ws, out_ws, stream);
}

}  // namespace amd_tuned_torch_ck
