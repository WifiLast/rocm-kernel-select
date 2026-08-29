// Torch-facing entry point for the CK GEMM tier (src/cuda/ck_gemm_fwd.hpp).
// Declared separately from main_rocm.cpp for the same reasons
// ck_conv_torch.hpp is: that file gains only an include and one m.def, and
// the whole tier compiles out where Composable Kernel isn't available.
#pragma once

#include <torch/extension.h>

// Y = X @ W^T (+ bias) with `epilogue` applied in-kernel -- F.linear's
// semantics, weight [N, K] passed through as-is.
//
// `epilogue` uses the SAME vocabulary as hipblaslt_linear's
// (AmdTunedTorchEpilogue in src/hipblaslt_gemm.hpp), deliberately: the two
// tiers are candidates in one contest, so a caller picks a fused form once
// and offers it to both rather than translating between two enums. CK has
// no ReLU epilogue compiled in here, so that value declines -- which is the
// ordinary way a candidate drops out of a contest, not an error.
//
// Returns nullopt when no compiled instance supports the problem, matching
// ck_conv_forward and hipblaslt_linear.
//
// fp16/bf16 only (CK's WMMA GEMM instances cover no other dtype on
// gfx1100), input [..., K] with the leading dims flattened, weight [N, K].
c10::optional<torch::Tensor> ck_gemm_linear(torch::Tensor input,
                                            torch::Tensor weight,
                                            c10::optional<torch::Tensor> bias,
                                            int64_t epilogue);
