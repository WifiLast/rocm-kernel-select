// Independent extension for Composable Kernel (CK) bindings: conv, GroupNorm,
// GEMM. Built as amd_tuned_torch._native_ck (see setup.py) -- split out of
// main_rocm.cpp so that touching/rebuilding CK code (~94% of a full build's
// CPU time, see setup.py's BUILD COST WARNING) never relinks the core
// amd_tuned_torch._native extension, and vice versa.
//
// Always compiled, exactly like src/hipblaslt_gemm.cpp already was before
// this split: with every AMD_TUNED_TORCH_HAS_CK_* define absent (CK not
// available, or every tier declined), this file still builds and exposes a
// working has_ck() -> false, so amd_tuned_torch/ck_ops.py's available()
// check has a real module to call into rather than an import failure. The
// three tiers are still gated separately in their own #ifdef (see setup.py's
// AMD_TUNED_TORCH_CK_{CONV,NORM,GEMM}) -- has_ck() means "at least one tier
// is in", matching what amd_tuned_torch/ck_ops.py's available() has always
// meant; each Python wrapper additionally checks that its OWN entry point
// exists (hasattr), so switching one tier off degrades that tier alone.
#include <torch/extension.h>

#ifdef AMD_TUNED_TORCH_HAS_CK_CONV
#include "ck_conv_torch.hpp"
#include "ck_conv_bwd_torch.hpp"
#endif
#ifdef AMD_TUNED_TORCH_HAS_CK_GEMM
#include "ck_gemm_torch.hpp"
#endif
#ifdef AMD_TUNED_TORCH_HAS_CK_NORM
#include "ck_norm_torch.hpp"
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef AMD_TUNED_TORCH_HAS_CK_CONV
    // Composable Kernel WMMA conv tier -- fp16/bf16, conv2d and conv3d.
    // Returns None when no CK instance supports the problem so the caller
    // can fall back (see amd_tuned_torch/ck_ops.py).
    m.def("ck_conv", &ck_conv_forward,
          "Composable Kernel WMMA grouped-conv forward (fp16/bf16, 2D/3D, groups=1); "
          "None if unsupported");
    // Backward halves of the tier above -- conv2d only (see
    // src/cuda/ck_conv_bwd.hpp), so training can eventually use this path
    // too instead of always falling back once any input requires_grad.
    m.def("ck_conv2d_backward_data", &ck_conv2d_backward_data,
          "Composable Kernel WMMA grouped-conv2d backward data (fp16/bf16, groups=1); "
          "None if unsupported");
    m.def("ck_conv2d_backward_weight", &ck_conv2d_backward_weight,
          "Composable Kernel WMMA grouped-conv2d backward weight (fp16/bf16, groups=1); "
          "None if unsupported");
#endif
#ifdef AMD_TUNED_TORCH_HAS_CK_NORM
    // CK normalization tier -- GroupNorm with SiLU optionally fused into
    // the same kernel (src/cuda/ck_norm_fwd.hpp explains why the fused
    // form is the point and plain GroupNorm is the side effect).
    m.def("ck_group_norm", &ck_group_norm_forward,
          "Composable Kernel GroupNorm, optionally with SiLU fused into the "
          "epilogue (fp16/bf16/fp32, rank 4/5, affine only); None if unsupported",
          py::arg("input"), py::arg("num_groups"), py::arg("weight"), py::arg("bias"),
          py::arg("eps"), py::arg("fuse_silu") = false);
#endif
#ifdef AMD_TUNED_TORCH_HAS_CK_GEMM
    // CK WMMA GEMM tier -- a third F.linear candidate, with bias and
    // activation fused into the epilogue (src/cuda/ck_gemm_fwd.hpp).
    m.def("ck_gemm_linear", &ck_gemm_linear,
          "Composable Kernel WMMA Y = X @ W^T (+ bias) with an optional fused "
          "GELU/SiLU epilogue (fp16/bf16); None if unsupported",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("epilogue") = 0);
#endif
#ifdef AMD_TUNED_TORCH_HAS_CK
    m.def("has_ck", []() { return true; },
          "Whether any Composable Kernel tier was compiled in");
#else
    m.def("has_ck", []() { return false; },
          "Whether any Composable Kernel tier was compiled in");
#endif
}
