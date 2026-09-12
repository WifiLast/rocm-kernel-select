#include "rocsparse_spmm.hpp"

#ifdef AMD_TUNED_TORCH_HAS_ROCSPARSE

#include <c10/hip/HIPStream.h>

// One spelling covers both places rocSPARSE can come from (see setup.py's
// detection comment): a system ROCm install lays the public header at
// $ROCM_PATH/include/rocsparse/rocsparse.h (the -dev package's own install
// layout); a from-source build of the vendored third_party/rocsparse tree
// (its own install.sh, NOT header-only -- rocSPARSE's compiled device
// kernels live in the .so, unlike Composable Kernel's header-only
// instantiation) reproduces that same rocsparse/ subdirectory under
// install.sh's own install prefix. Only which directory setup.py points -I
// at differs; the include line itself never needs to change.
#include <rocsparse/rocsparse.h>

#include <mutex>

namespace {

// One handle for the process, exactly like hipblaslt_gemm.cpp's lt_handle()
// -- rocsparse_create_handle is not free and the handle is documented as
// reusable across calls on one thread's stream.
rocsparse_handle rs_handle() {
    static rocsparse_handle h = nullptr;
    static std::once_flag once;
    std::call_once(once, [] {
        if (rocsparse_create_handle(&h) != rocsparse_status_success) h = nullptr;
    });
    return h;
}

// -1 (via the invalid cast below) for anything rocSPARSE has no uniform/
// mixed-precision SpMM row for, which is how fp64 and the integer dtypes
// decline without a separate check at each call site -- same convention as
// hipblaslt_gemm.cpp's hip_dtype().
rocsparse_datatype rs_dtype(torch::ScalarType t) {
    switch (t) {
        case torch::kHalf: return rocsparse_datatype_f16_r;
        case torch::kBFloat16: return rocsparse_datatype_bf16_r;
        case torch::kFloat: return rocsparse_datatype_f32_r;
        default: return static_cast<rocsparse_datatype>(-1);
    }
}

// f16_r/bf16_r data computes in f32_r (rocsparse_spmm.h's own "Mixed
// precisions" table); f32_r data also allows an f32_r compute type (its
// "Uniform Precisions" table) -- so f32_r compute covers every dtype this
// tier accepts, and alpha/beta below are always plain floats regardless of
// which of the three `input`/`other` actually are.
constexpr rocsparse_datatype kComputeType = rocsparse_datatype_f32_r;

}  // namespace

c10::optional<torch::Tensor> rocsparse_spmm_torch(torch::Tensor input, torch::Tensor other) {
    if (rs_handle() == nullptr) return c10::nullopt;
    if (!input.is_cuda() || !other.is_cuda()) return c10::nullopt;
    if (input.layout() != c10::kSparseCsr) return c10::nullopt;
    if (input.dim() != 2 || other.dim() != 2) return c10::nullopt;
    if (input.scalar_type() != other.scalar_type()) return c10::nullopt;
    const auto dt = rs_dtype(input.scalar_type());
    if (static_cast<int>(dt) < 0) return c10::nullopt;
    if (input.size(1) != other.size(0)) return c10::nullopt;

    // CSR's three backing arrays are already separate contiguous 1D
    // tensors by construction on the torch side (crow_indices/col_indices/
    // values) -- .contiguous() below is a no-op in the overwhelmingly
    // common case and only actually copies for an input that was built or
    // sliced unusually.
    const auto crow = input.crow_indices().contiguous();
    const auto col = input.col_indices().contiguous();
    const auto val = input.values().contiguous();
    if (crow.scalar_type() != col.scalar_type()) return c10::nullopt;

    rocsparse_indextype idx_type;
    if (crow.scalar_type() == torch::kInt32) {
        idx_type = rocsparse_indextype_i32;
    } else if (crow.scalar_type() == torch::kInt64) {
        idx_type = rocsparse_indextype_i64;
    } else {
        return c10::nullopt;
    }

    const int64_t M = input.size(0), K = input.size(1), nnz = val.size(0);
    const auto other_c = other.contiguous();
    const int64_t N = other_c.size(1);
    if (M == 0 || K == 0 || N == 0) return c10::nullopt;

    auto out = torch::zeros({M, N}, other_c.options());

    // rocsparse_spmm takes non-const descriptor handles for mat_A/mat_B
    // even though it only reads through them (mat_C is the sole in/out
    // operand) -- create_csr_descr/create_dnmat_descr have no "const"
    // variant that also accepts a mutable data pointer, so these are built
    // the same way for every operand regardless of which one is actually
    // written.
    rocsparse_spmat_descr mat_a = nullptr;
    if (rocsparse_create_csr_descr(&mat_a, M, K, nnz, crow.data_ptr(), col.data_ptr(),
                                   val.data_ptr(), idx_type, idx_type,
                                   rocsparse_index_base_zero, dt) != rocsparse_status_success) {
        return c10::nullopt;
    }

    rocsparse_dnmat_descr mat_b = nullptr;
    // Row order: torch's row-major [K, N] `other` IS rocSPARSE's row-order
    // dense descriptor with ld == N, no reinterpretation needed (unlike
    // hipblaslt_gemm.cpp's column-major library, rocSPARSE natively
    // supports row order for B/C -- see rocsparse_spmm.h's own docs
    // recommending it for best performance).
    if (rocsparse_create_dnmat_descr(&mat_b, K, N, N, other_c.data_ptr(), dt,
                                     rocsparse_order_row) != rocsparse_status_success) {
        rocsparse_destroy_spmat_descr(mat_a);
        return c10::nullopt;
    }

    rocsparse_dnmat_descr mat_c = nullptr;
    if (rocsparse_create_dnmat_descr(&mat_c, M, N, N, out.data_ptr(), dt, rocsparse_order_row) !=
        rocsparse_status_success) {
        rocsparse_destroy_dnmat_descr(mat_b);
        rocsparse_destroy_spmat_descr(mat_a);
        return c10::nullopt;
    }

    const float alpha = 1.0f, beta = 0.0f;
    const auto stream = c10::hip::getCurrentHIPStream().stream();
    rocsparse_set_stream(rs_handle(), stream);

    // Three-stage contract from rocsparse_spmm.h's own docstring:
    // buffer_size (writes the required byte count, performs no computation)
    // -> allocate -> preprocess (analyzes mat_A, blocking) -> compute. The
    // first two stages only ever need to run once per distinct mat_A, but
    // this tier is not repeated-call-cached the way flex_gemm's
    // NeighborCache is -- see rocsparse_ops.py's module docstring for why
    // that's left for a future caller building on this rather than done
    // here.
    size_t buffer_size = 0;
    auto status = rocsparse_spmm(rs_handle(), rocsparse_operation_none, rocsparse_operation_none,
                                 &alpha, mat_a, mat_b, &beta, mat_c, kComputeType,
                                 rocsparse_spmm_alg_default, rocsparse_spmm_stage_buffer_size,
                                 &buffer_size, nullptr);

    torch::Tensor workspace;
    void* workspace_ptr = nullptr;
    if (status == rocsparse_status_success && buffer_size > 0) {
        workspace = torch::empty({static_cast<int64_t>(buffer_size)},
                                 other_c.options().dtype(torch::kUInt8));
        workspace_ptr = workspace.data_ptr();
    }

    if (status == rocsparse_status_success) {
        status = rocsparse_spmm(rs_handle(), rocsparse_operation_none, rocsparse_operation_none,
                                &alpha, mat_a, mat_b, &beta, mat_c, kComputeType,
                                rocsparse_spmm_alg_default, rocsparse_spmm_stage_preprocess,
                                &buffer_size, workspace_ptr);
    }
    if (status == rocsparse_status_success) {
        status = rocsparse_spmm(rs_handle(), rocsparse_operation_none, rocsparse_operation_none,
                                &alpha, mat_a, mat_b, &beta, mat_c, kComputeType,
                                rocsparse_spmm_alg_default, rocsparse_spmm_stage_compute,
                                &buffer_size, workspace_ptr);
    }

    rocsparse_destroy_dnmat_descr(mat_c);
    rocsparse_destroy_dnmat_descr(mat_b);
    rocsparse_destroy_spmat_descr(mat_a);

    if (status != rocsparse_status_success) return c10::nullopt;
    return out;
}

#endif  // AMD_TUNED_TORCH_HAS_ROCSPARSE
