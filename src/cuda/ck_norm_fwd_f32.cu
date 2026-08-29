// CK normalization-forward instantiation: float, both epilogues.
// One dtype per translation unit so hipcc builds them in parallel -- see
// ck_norm_fwd_impl.hpp.
#include "ck_norm_fwd_impl.hpp"
#include "ck_norm_fwd_tu.hpp"

namespace amd_tuned_torch_ck_norm {

int ck_norm_count_f32(int yop) {
    return yop == kSwish ? detail::count<float, detail::Swish>()
                         : detail::count<float, detail::PassThrough>();
}

bool ck_norm_entry_f32(int yop, int idx, int action, const NormProblem& p, const void* p_x,
                           const void* p_gamma, const void* p_beta, void* p_y, void* p_ws,
                           size_t* out_ws, hipStream_t stream) {
    const auto act = static_cast<detail::Action>(action);
    if (yop == kSwish)
        return detail::entry<float, detail::Swish>(idx, act, p, p_x, p_gamma, p_beta, p_y,
                                                      p_ws, out_ws, stream);
    return detail::entry<float, detail::PassThrough>(idx, act, p, p_x, p_gamma, p_beta, p_y,
                                                        p_ws, out_ws, stream);
}

}  // namespace amd_tuned_torch_ck_norm
