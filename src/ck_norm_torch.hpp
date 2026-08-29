// Torch-facing entry point for the CK normalization tier
// (src/cuda/ck_norm_fwd.hpp). Declared separately from main_rocm.cpp for
// the same reasons ck_conv_torch.hpp is: that file gains only an include
// and one m.def, and the whole tier compiles out where Composable Kernel
// isn't available (see setup.py and AMD_TUNED_TORCH_HAS_CK).
#pragma once

#include <torch/extension.h>

// GroupNorm over `input` [N, C, spatial...], optionally with SiLU fused
// into the same kernel (`fuse_silu`) so the activation is never written to
// memory between the two.
//
// Returns nullopt -- "no CK instance supports this problem, use your
// fallback" -- rather than throwing, matching ck_conv_forward. Reasons this
// is a routine outcome rather than a failure: rank other than 4 or 5,
// missing affine parameters (CK's kernels read gamma and beta
// unconditionally), or a channels-per-group that no compiled instance's
// vector width divides.
//
// fp16/bf16/fp32. Accepts any memory format and returns the output in the
// SAME format it was handed, converting internally where CK's
// channels-last view requires it.
c10::optional<torch::Tensor> ck_group_norm_forward(torch::Tensor input,
                                                   int64_t num_groups,
                                                   c10::optional<torch::Tensor> weight,
                                                   c10::optional<torch::Tensor> bias,
                                                   double eps,
                                                   bool fuse_silu);
