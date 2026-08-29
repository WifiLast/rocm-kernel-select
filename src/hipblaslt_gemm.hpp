// Torch-facing entry points for the hipBLASLt GEMM tier.
//
// Declared separately from main_rocm.cpp for the same two reasons
// ck_conv_torch.hpp is: that file gains only an include and an m.def, and
// the whole tier compiles out where hipBLASLt's headers aren't present
// (see setup.py's detection and AMD_TUNED_TORCH_HAS_HIPBLASLT).
//
// WHY THIS TIER EXISTS. Until now linear/matmul/bmm had no non-stock
// candidate on gfx1100 at all: aiter's Triton GEMM raises
// KeyError('gfx1100') (no RDNA3 tuning config shipped), so
// amd_tuned_torch/kernel_select.py explicitly excluded those three ops
// from its contest -- there was nothing to hold a contest against. There
// is now, and it is already installed: PyTorch's default ROCm BLAS
// backend is rocBLAS, whose gfx1100 Tensile libraries are all named
// TensileLibrary_*_fallback_gfx1100 (untuned fallback logic), while
// hipBLASLt ships a dedicated navi31 tuned-logic set. Measured here with
// F.linear (torch 2.15/ROCm 7.2, RX 7900 XTX, 200 iters, cuda events):
//
//     4096x4096x4096    bf16   rocBLAS 10.78 ms   hipBLASLt  8.85 ms  1.22x
//     8192x4096x11008   bf16   rocBLAS 11.71 ms   hipBLASLt  9.65 ms  1.21x
//     4096x4096x1024    bf16   rocBLAS  4.02 ms   hipBLASLt  2.00 ms  2.01x
//     4096x4096x1024    fp16   rocBLAS  5.95 ms   hipBLASLt  4.01 ms  1.48x
//     1024x1024x1024    bf16   rocBLAS  0.27 ms   hipBLASLt  0.30 ms  0.88x
//     4096x1280x1280    fp16   rocBLAS  1.69 ms   hipBLASLt  1.99 ms  0.85x
//
// It wins large and loses small, which is why this is a kernel_select
// candidate rather than a replacement -- exactly the shape of decision
// that module already makes for conv2d/conv3d.
//
// Reaching hipBLASLt through torch (torch.backends.cuda.
// preferred_blas_library("hipblaslt")) would get those rows and nothing
// else. Going through the library directly additionally gets the FUSED
// EPILOGUE the navi31 logic ships kernels for -- bias, GELU, SiLU/Swish
// and ReLU applied inside the GEMM's own epilogue rather than as a
// separate elementwise pass over D. That is the part torch cannot express:
// F.linear followed by F.gelu is two kernels and two full round-trips of
// the [M, N] activation through HBM no matter which BLAS backend runs the
// first one.
#pragma once

#include <torch/extension.h>

// Epilogue selector for hipblaslt_linear. Values are this project's own,
// not hipBLASLt's -- mapped to hipblasLtEpilogue_t in the .cpp so the
// Python side never has to include a ROCm header to name one.
//
// GELU here is hipBLASLt's tanh approximation, which is F.gelu's
// approximate="tanh" and NOT its default exact erf form. te_ops.py draws
// the same line for the same reason: silently swapping the default would
// change numerics.
enum AmdTunedTorchEpilogue : int64_t {
    AMD_TUNED_TORCH_EPI_NONE = 0,
    AMD_TUNED_TORCH_EPI_BIAS = 1,
    AMD_TUNED_TORCH_EPI_GELU = 2,   // tanh approximation
    AMD_TUNED_TORCH_EPI_SILU = 3,   // Swish(x, 1)
    AMD_TUNED_TORCH_EPI_RELU = 4,
};

// Y = X @ W^T (+ bias) with `epilogue` applied in-kernel -- F.linear's
// semantics, weight [N, K] passed through as-is with no pre-transpose.
//
// Returns nullopt -- "hipBLASLt has no kernel for this problem, use your
// fallback" -- rather than throwing, matching the opportunistic-kernel
// convention ck_conv_forward and launch_conv3d_fp32_winograd already use.
// That is a routine outcome, not a failure: the heuristic query legitimately
// returns zero algorithms for dtype/epilogue combinations the installed
// navi31 logic doesn't cover.
//
// fp16/bf16/fp32, input [..., K] (leading dims flattened), weight [N, K].
c10::optional<torch::Tensor> hipblaslt_linear(torch::Tensor input,
                                              torch::Tensor weight,
                                              c10::optional<torch::Tensor> bias,
                                              int64_t epilogue);

// C[i] = A[i] @ B[i] -- torch.bmm's semantics, so B is [batch, K, N] and is
// NOT transposed on the way in (unlike hipblaslt_linear's weight). Same
// nullopt convention. No epilogue: bmm has no bias to fuse and no
// activation follows it in the attention graphs this exists for.
c10::optional<torch::Tensor> hipblaslt_bmm(torch::Tensor a, torch::Tensor b);
