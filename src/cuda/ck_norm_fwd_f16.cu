// CK normalization-forward instantiation: ck::half_t, both epilogues.
// One dtype per translation unit so hipcc builds them in parallel -- see
// ck_norm_fwd_impl.hpp.
#include "ck_norm_fwd_impl.hpp"
#include "ck_norm_fwd_tu.hpp"

namespace amd_tuned_torch_ck_norm {

int ck_norm_count_f16(int yop) {
    return yop == kSwish ? detail::count<ck::half_t, detail::Swish>()
                         : detail::count<ck::half_t, detail::PassThrough>();
}

bool ck_norm_entry_f16(int yop, int idx, int action, const NormProblem& p, const void* p_x,
                           const void* p_gamma, const void* p_beta, void* p_y, void* p_ws,
                           size_t* out_ws, hipStream_t stream) {
    const auto act = static_cast<detail::Action>(action);
    if (yop == kSwish)
        return detail::entry<ck::half_t, detail::Swish>(idx, act, p, p_x, p_gamma, p_beta, p_y,
                                                      p_ws, out_ws, stream);
    return detail::entry<ck::half_t, detail::PassThrough>(idx, act, p, p_x, p_gamma, p_beta, p_y,
                                                        p_ws, out_ws, stream);
}

}  // namespace amd_tuned_torch_ck_norm
