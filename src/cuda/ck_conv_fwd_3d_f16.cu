// CK WMMA grouped-conv-forward instantiation: 3D, ck::half_t.
// One (rank, dtype) per translation unit so hipcc builds them in
// parallel -- see ck_conv_fwd_impl.hpp for why that matters.
#include "ck_conv_fwd_impl.hpp"
#include "ck_conv_fwd_tu.hpp"

namespace amd_tuned_torch_ck {

int ck_count_3d_f16() { return detail::count<3, ck::half_t>(); }

bool ck_entry_3d_f16(int idx, int action, const ConvProblem& p, const void* p_in,
                      const void* p_wei, const void* p_bias, void* p_out,
                      void* p_ws, size_t* out_ws, hipStream_t stream) {
    return detail::entry<3, ck::half_t>(idx, static_cast<detail::Action>(action), p,
                                        p_in, p_wei, p_bias, p_out, p_ws, out_ws, stream);
}

}  // namespace amd_tuned_torch_ck
