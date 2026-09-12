// Independent extension for the rocSPARSE SpMM tier. Built as
// amd_tuned_torch._native_rocsparse (see setup.py) -- same "own extension
// per optional tier" split as src/hipblaslt_native.cpp, so touching/
// rebuilding this tier never relinks the core amd_tuned_torch._native
// extension (or any other tier's), and vice versa. See
// src/rocsparse_spmm.hpp for what this tier is and why it exists.
//
// Always compiled: src/rocsparse_spmm.cpp is internally #ifdef'd on
// AMD_TUNED_TORCH_HAS_ROCSPARSE, and this file follows the same convention
// -- with the define absent, it still builds and exposes a working
// has_rocsparse() -> false, so amd_tuned_torch/rocsparse_ops.py's
// available() check has a real module to call into rather than an import
// failure.
#include <torch/extension.h>

#ifdef AMD_TUNED_TORCH_HAS_ROCSPARSE
#include "rocsparse_spmm.hpp"
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef AMD_TUNED_TORCH_HAS_ROCSPARSE
    // None when rocSPARSE declines the problem (an unsupported dtype/
    // shape combination) or any call in the three-stage sequence fails --
    // same opportunistic-kernel convention as hipblaslt_linear/
    // hipblaslt_bmm.
    m.def("rocsparse_spmm", &rocsparse_spmm_torch,
          "rocSPARSE sparse (CSR) @ dense -> dense SpMM (fp16/bf16/fp32); "
          "None if unsupported",
          py::arg("input"), py::arg("other"));
    m.def("has_rocsparse", []() { return true; },
          "Whether the rocSPARSE SpMM tier was compiled in");
#else
    m.def("has_rocsparse", []() { return false; },
          "Whether the rocSPARSE SpMM tier was compiled in");
#endif
}
