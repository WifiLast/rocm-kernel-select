// Torch-facing entry points for the CK conv2d backward tier
// (src/cuda/ck_conv_bwd.hpp). Declared separately from ck_native.cpp so
// that file only gains an include and two m.def lines, and so this whole
// tier compiles out together with the rest of the CK conv tier where
// Composable Kernel isn't available (see setup.py's CK detection and
// AMD_TUNED_TORCH_HAS_CK_CONV).
//
// conv2d only (see ck_conv_bwd.hpp), fp16/bf16 only, groups=1 only --
// exactly forward's dtype/groups constraints. Both return nullopt --
// "no CK instance supports this problem, use your fallback" -- rather
// than throwing, matching ck_conv_forward's and the rest of this
// project's opportunistic-kernel convention.
#pragma once

#include <torch/extension.h>
#include <vector>

// grad_output: dL/dY, shape [N,K,Ho,Wo] (any input memory format
// accepted). weight: [K,C,Kh,Kw]. input_size: the ORIGINAL conv2d
// input's [N,C,Hi,Wi] shape (needed because backward-data can't
// otherwise recover Hi/Wi from grad_output/weight/stride/padding alone
// when they're ambiguous, e.g. stride>1). Returns dL/dX in the SAME
// memory format grad_output was handed, or nullopt if unsupported.
c10::optional<torch::Tensor> ck_conv2d_backward_data(torch::Tensor grad_output,
                                                     torch::Tensor weight,
                                                     std::vector<int64_t> input_size,
                                                     std::vector<int64_t> stride,
                                                     std::vector<int64_t> padding,
                                                     std::vector<int64_t> dilation);

// input: the original conv2d input [N,C,Hi,Wi]. grad_output: dL/dY
// [N,K,Ho,Wo]. Returns dL/dWeight with shape [K,C,Kh,Kw] (kernel_size
// taken from weight_size), or nullopt if unsupported.
c10::optional<torch::Tensor> ck_conv2d_backward_weight(torch::Tensor input,
                                                       torch::Tensor grad_output,
                                                       std::vector<int64_t> weight_size,
                                                       std::vector<int64_t> stride,
                                                       std::vector<int64_t> padding,
                                                       std::vector<int64_t> dilation);
