// CK WMMA GEMM instantiation: plain (no bias), ck::half_t.
// One (dtype, epilogue) per translation unit so hipcc builds them in
// parallel -- see ck_gemm_fwd_impl.hpp for why that matters here.
#include "ck_gemm_fwd_impl.hpp"
#include "ck_gemm_fwd_tu.hpp"

namespace amd_tuned_torch_ck_gemm {

int ck_gemm_count_f16() { return detail::count_plain<ck::half_t>(); }

bool ck_gemm_entry_f16(int idx, int action, const GemmProblem& p, const void* p_a,
                          const void* p_b, const void* p_bias, void* p_e, void* p_ws,
                          size_t* out_ws, hipStream_t stream) {
    if (p_bias != nullptr) return false;  // this unit has no D tensor
    return detail::entry_plain<ck::half_t>(idx, static_cast<detail::Action>(action), p, p_a, p_b,
                                        p_e, p_ws, out_ws, stream);
}

}  // namespace amd_tuned_torch_ck_gemm
