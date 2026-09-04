// Independent extension for the hipBLASLt GEMM tier. Built as
// amd_tuned_torch._native_hipblaslt (see setup.py) -- split out of
// main_rocm.cpp so that touching/rebuilding hipBLASLt code never relinks the
// core amd_tuned_torch._native extension (or the CK extension), and vice
// versa. See src/hipblaslt_gemm.hpp for what this tier is and why it exists.
//
// Always compiled: src/hipblaslt_gemm.cpp was already "compiled either way"
// before this split (internally #ifdef'd on AMD_TUNED_TORCH_HAS_HIPBLASLT),
// and this file follows the same convention -- with the define absent, it
// still builds and exposes a working has_hipblaslt() -> false, so
// amd_tuned_torch/hipblaslt_ops.py's available() check has a real module to
// call into rather than an import failure.
#include <torch/extension.h>

#ifdef AMD_TUNED_TORCH_HAS_HIPBLASLT
#include "hipblaslt_gemm.hpp"
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef AMD_TUNED_TORCH_HAS_HIPBLASLT
    // Both return None when the installed navi31 logic has no kernel for
    // the problem, same opportunistic-kernel convention as ck_conv.
    m.def("hipblaslt_linear", &hipblaslt_linear,
          "hipBLASLt Y = X @ W^T (+ bias) with an optional fused GELU/SiLU/ReLU "
          "epilogue (fp16/bf16/fp32); None if unsupported",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("epilogue") = 0);
    m.def("hipblaslt_bmm", &hipblaslt_bmm,
          "hipBLASLt batched C[i] = A[i] @ B[i] (fp16/bf16/fp32); None if unsupported");
    m.def("has_hipblaslt", []() { return true; },
          "Whether the hipBLASLt GEMM tier was compiled in");
#else
    m.def("has_hipblaslt", []() { return false; },
          "Whether the hipBLASLt GEMM tier was compiled in");
#endif
}
