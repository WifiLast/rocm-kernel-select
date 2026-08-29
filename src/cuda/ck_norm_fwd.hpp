// Composable Kernel normalization-forward wrappers -- the fused
// GroupNorm(+SiLU) tier for gfx1100.
//
// WHY THIS EXISTS. `F.group_norm` already has a hand-written kernel here
// (src/cuda/group_norm.cu) and it is a wash against stock: ~1.10x raw,
// which is about what the Python dispatch wrapper costs back. Beating
// stock at GroupNorm alone is not the opportunity, because GroupNorm alone
// is not what a diffusion U-Net runs. Every ResBlock runs
// GroupNorm -> SiLU, and both are memory-bound elementwise-ish passes over
// the same [N, C, H, W] activation, so as two kernels they read and write
// that tensor twice. CK's normalization instances take a Y-elementwise
// operation applied inside the kernel, on values already in registers, so
// the SiLU costs one exp per element and no memory traffic at all. The
// second round-trip simply stops happening.
//
// Unlike ck_conv_fwd.hpp's tier this one is not about matrix cores -- the
// underlying kernels are blockwise reductions with no WMMA in them, so
// nothing here depends on CK having tuned RDNA3 instances. What it depends
// on is CK exposing the fused form at all, which neither aiter nor
// TransformerEngine do: TE covers SiLU as a standalone op and neither
// covers GroupNorm, a diffusion-specific op, in any form.
//
// LAYOUT. CK indexes this as a rank-5 tensor [N, S1, S2, G, C] reducing
// over {S1, S2, C} -- spatial and the within-group channels, exactly
// GroupNorm's reduction. Note C here is channels PER GROUP, and G is the
// outer half of the channel dimension, which is the same split
// torch.nn.functional.group_norm makes. For a channels-last tensor that
// makes the CK view a pure reshape of contiguous memory, free; for
// contiguous NCHW it is a permute in and out, the same trade the CK conv
// tier documents. The two tiers agree on the layout that costs nothing,
// so a channels-last graph pays it in neither.
//
// Everything spatial is flattened into S1 x S2 by the caller, which is why
// one rank-5 instantiation serves 3D, 4D and 5D inputs (conv1d/2d/3d
// U-Nets) instead of one per rank. The reduction is over all of it either
// way; the split into two dims only exists because CK's shipped groupnorm
// instances are rank 5 and there is no reason to instantiate more.
#pragma once

#include <cstddef>
#include <hip/hip_runtime.h>

namespace amd_tuned_torch_ck_norm {

// [N, S1, S2, G, C], channels-last: C contiguous, then G, then S2, S1, N.
// gamma/beta are [G, C] broadcast over N and both spatial dims.
struct NormProblem {
    int N, S1, S2, G, C;
    float epsilon;
};

enum DType { kF16 = 0, kBF16 = 1, kF32 = 2 };

// Which Y-elementwise op is fused into the epilogue. kPassThrough is plain
// GroupNorm; kSwish is GroupNorm followed by SiLU (Swish with beta=1,
// which is exactly F.silu) with no intervening memory traffic.
enum YOp { kPassThrough = 0, kSwish = 1 };

// How many CK instances are compiled in for this (dtype, yop). 0 if the
// combination isn't built.
int num_instances(int dtype, int yop);

// True if instance `idx` accepts this problem (CK's own
// IsSupportedArgument -- vector-load alignment and tile divisibility, both
// of which bite here: a U-Net's C-per-group is often something like 10 or
// 20, so the widely-vectorised instances decline and the scalar ones win).
bool supported(int dtype, int yop, int idx, const NormProblem& p);

// Bytes of scratch instance `idx` needs for this problem, 0 if none. Only
// the split-K instances want any.
size_t workspace_bytes(int dtype, int yop, int idx, const NormProblem& p);

// Runs instance `idx`. All pointers are device memory in the layout above
// and all of them must be non-null: CK's kernels read gamma and beta
// unconditionally, with no affine=False path, so the caller is responsible
// for declining (or materialising ones/zeros) when the module has no affine
// parameters. Returns false if the instance doesn't support the problem.
bool run(int dtype, int yop, int idx, const NormProblem& p, const void* p_x, const void* p_gamma,
         const void* p_beta, void* p_y, void* p_workspace, hipStream_t stream);

}  // namespace amd_tuned_torch_ck_norm
