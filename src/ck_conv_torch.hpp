// Torch-facing entry point for the CK conv tier (src/cuda/ck_conv_fwd.hpp).
// Declared separately from main_rocm.cpp so that file only gains an
// include and one m.def, and so this whole tier can be compiled out where
// Composable Kernel isn't available (see setup.py's CK detection and
// AMD_TUNED_TORCH_HAS_CK).
#pragma once

#include <torch/extension.h>
#include <vector>

// Returns nullopt -- "no CK instance supports this problem, use your
// fallback" -- rather than throwing, matching the opportunistic-kernel
// convention the rest of this project uses (see src/trt/*.cpp and
// launch_conv3d_fp32_winograd's guard).
//
// fp16/bf16 only, groups=1, 4D (conv2d) or 5D (conv3d). Accepts any
// memory format and returns the output in the SAME format it was handed,
// converting internally where CK's channels-last-only WMMA instances
// require it.
c10::optional<torch::Tensor> ck_conv_forward(torch::Tensor input,
                                             torch::Tensor weight,
                                             c10::optional<torch::Tensor> bias,
                                             std::vector<int64_t> stride,
                                             std::vector<int64_t> padding,
                                             std::vector<int64_t> dilation);
