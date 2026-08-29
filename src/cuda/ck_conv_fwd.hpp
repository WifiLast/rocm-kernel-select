// Composable Kernel (CK) WMMA grouped-conv-forward wrappers -- the
// channels-last conv2d/conv3d tier for gfx1100.
//
// WHY THIS EXISTS, and why CK is back after being removed. This project
// dropped its CK dependency once already (an older revision linked CK's
// DeviceGemm instances for the GEMM ops; aiter's Triton GEMM replaced
// that -- see README). This is not that: it is conv, not GEMM, and it is
// here because of a measured hole in stock ROCm rather than a preference
// for CK. Profiling F.conv2d on gfx1100 (see tools/bench_ck.py) shows
// MIOpen dispatches 3x3 fp16/fp32 convs to a hand-written assembly
// Winograd solver (miopenSp3AsmConvFury_*_f2x3_*) that runs at ~90% of
// this card's peak -- nothing in this repo will beat it, and nothing
// should try. But MIOpen has NO bf16 Winograd solver on gfx11: bf16
// falls back to an explicit im2col into memory plus a Tensile GEMM, and
// lands at ~14% of peak. CK's WMMA implicit-GEMM conv covers bf16
// natively and measured 4.7x faster than that fallback.
//
// Unlike the hand-written kernels in this directory, nothing here is a
// kernel we maintain: these are CK's own pre-tuned instances, selected
// per shape at runtime the same way run_conv2d_fp16 already selects among
// its own tile-shape variants (src/main_rocm.cpp).
//
// LAYOUT (the one real constraint). CK ships WMMA conv-fwd instances only
// for channels-last layouts -- NHWGC/GKYXC/NHWGK and
// NDHWGC/GKZYXC/NDHWGK. Its NGCHW (== NCHW at G=1) instances are xdl/CDNA
// only, so there is no contiguous-NCHW path on this card. Every pointer
// passed in here must therefore already be channels-last; the caller owns
// that conversion and the decision about whether it is worth paying (see
// amd_tuned_torch/ck_ops.py, which skips the copy when the tensor is
// already channels_last, as it is throughout an inference graph that set
// PYTORCH_MIOPEN_SUGGEST_NHWC=1).
//
// BIAS is fused into CK's epilogue as a broadcast D tensor (zero strides
// on N and the spatial dims, OutElementOp=Add), which is why there is no
// separate bias-add pass and no unbiased instantiation: a null bias would
// have doubled the instantiation count -- and therefore the build time,
// which is the dominant cost of using CK at all -- to save nothing.
// Callers with no bias pass a zeroed K-element buffer.
#pragma once

#include <cstddef>
#include <hip/hip_runtime.h>

namespace amd_tuned_torch_ck {

// Spatial dims are stored D,H,W-major with the leading entries unused for
// NDim==2 (i.e. 2D fills [0]=H, [1]=W).
struct ConvProblem {
    int G, N, C, K;          // G must be 1 (grouped conv is not routed here)
    int in_spatial[3];
    int filt[3];
    int out_spatial[3];
    int stride[3];
    int dilation[3];
    int lpad[3];
    int rpad[3];
};

enum DType { kF16 = 0, kBF16 = 1 };

// How many CK instances are compiled in for this (ndim, dtype). 0 if the
// combination isn't built.
int num_instances(int ndim, int dtype);

// True if instance `idx` accepts this problem (CK's own
// IsSupportedArgument -- vector-load alignment, padding, tile divisibility).
bool supported(int ndim, int dtype, int idx, const ConvProblem& p);

// Bytes of scratch instance `idx` needs for this problem, 0 if none.
size_t workspace_bytes(int ndim, int dtype, int idx, const ConvProblem& p);

// Runs instance `idx`. All pointers are channels-last device memory;
// p_bias is a K-element vector and must be non-null (zeros if the op has
// no bias). Returns false if the instance doesn't support the problem.
bool run(int ndim, int dtype, int idx, const ConvProblem& p,
         const void* p_in, const void* p_wei, const void* p_bias, void* p_out,
         void* p_workspace, hipStream_t stream);

}  // namespace amd_tuned_torch_ck
