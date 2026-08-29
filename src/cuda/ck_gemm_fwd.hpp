// Composable Kernel WMMA GEMM wrappers -- a third candidate for
// F.linear, with the bias and activation fused into the GEMM epilogue.
//
// WHY CK IS A GEMM DEPENDENCY AGAIN. This project linked CK's DeviceGemm
// instances once, dropped them for aiter's Triton WMMA GEMM, and then
// discovered aiter raises KeyError('gfx1100') because it ships no RDNA3
// tuning config -- leaving linear/matmul/bmm with no non-stock candidate
// at all. src/hipblaslt_gemm.hpp added one. This is a second, and it earns
// its place by winning exactly where the first one loses.
//
// AN EARLIER VERSION OF THIS COMMENT WAS WRONG and the correction is worth
// keeping. It claimed both rocBLAS and hipBLASLt collapse to ~9 TF/s at a
// square 4096 fp16 GEMM while reaching ~60 TF/s at a larger awkward one,
// and offered that six-fold hole as the reason for a third candidate.
// There is no such hole: those numbers were measured while the machine was
// compiling this extension, and what they recorded was a CPU-starved
// launch loop rather than the GPU. Re-measured on an idle machine, stock
// reaches 64.4 TF/s at 4096^3. The lesson is in analyse/README.md's own
// warning -- measure interleaved, on a quiet machine, against a
// calibration shape whose expected time you already know.
//
// The real case, measured on RX 7900 XTX (torch 2.15 / ROCm 7.2, stock =
// rocBLAS = 1.00x), is that the two candidates fail on opposite shapes:
//
//                              hipBLASLt tier    CK GEMM tier
//     4096^3            fp16        1.29x            1.20x
//     8192x4096x11008   fp16        1.09x            1.11x
//     4096x4096x1024    fp16        1.49x            1.47x
//     4096x1280x1280    fp16        0.96x            1.14x
//     1024^3            fp16        1.02x            1.21x
//     1x4096x4096       fp16        0.60x            0.90x
//
// hipBLASLt owns the large end and falls off a cliff on the small one --
// 0.60x on a single-token decode GEMM, where its heuristic picks a kernel
// sized for throughput it will never get to use. CK is flat by comparison:
// slightly behind at 4096^3, ahead everywhere hipBLASLt is behind, and
// 1.21x at 1024^3 where hipBLASLt is a wash. Neither dominates, which is
// precisely the situation kernel_select's per-shape contest exists for --
// and is why this tier is a candidate rather than a replacement.
//
// FUSED EPILOGUES are the other half, and measurement made them the more
// valuable half. CK applies bias and the activation to values still in
// registers, so `F.gelu(F.linear(x, w, b))` is one kernel and one write of
// the [M, N] activation instead of two of each. Against the two-kernel
// stock form:
//
//     4096x4096x11008 + GELU   fp16   1.22x
//     4096x1280x5120  + GELU   fp16   1.15x
//     1024x1024x4096  + GELU   fp16   2.62x
//     1024x1024x4096  + SiLU   fp16   2.40x
//
// The small-shape rows are the point: once the GEMM is short enough, a
// second full pass over [M, N] is most of the work, and removing it is
// worth more than any tile-selection difference. hipBLASLt's epilogues do
// this too and measure within a few percent -- which is the other reason
// the fused forms have to exist in both candidates, since otherwise the
// contest compares a fused kernel against an unfused one and attributes
// the difference to the GEMM.
//
// LAYOUT costs nothing here, unlike the conv tier. F.linear is
// Y[M,N] = X[M,K] @ W[N,K]^T, and CK's Row/Col/Row layout triple means
// exactly that: A row-major [M,K], B column-major [K,N] -- which is the
// same bytes as a row-major [N,K] weight -- and E row-major [M,N]. Torch's
// own tensors already satisfy it, so no operand is transposed or copied.
// Bias rides along as a single D tensor with a zero row stride, so one
// N-element vector broadcasts down the M dimension.
//
// GEMM is where CK's build cost actually bites (the conv tier's warning in
// setup.py applies here too, and for the same reason), so what is compiled
// in is deliberately narrow: see ck_gemm_fwd_impl.hpp for which
// configurations and why.
#pragma once

#include <cstddef>
#include <hip/hip_runtime.h>

namespace amd_tuned_torch_ck_gemm {

// Row-major throughout: A is [M, K] with row stride K, the weight is
// [N, K] with row stride K (CK sees it as a column-major [K, N]), E is
// [M, N] with row stride N. Strides are carried explicitly rather than
// derived so a caller with a padded row can still use the tier.
struct GemmProblem {
    int M, N, K;
    int stride_a, stride_b, stride_e;
};

enum DType { kF16 = 0, kBF16 = 1 };

// What the epilogue does with the accumulator. kNone takes no bias tensor
// at all (a different CK device op, not a zero bias); the other three read
// an N-element bias broadcast over M.
//
// kAddFastGelu is CK's FastGelu, the tanh approximation -- that is
// F.gelu(approximate="tanh") and NOT its exact erf default. te_ops.py and
// hipblaslt_ops.py draw the same line for the same reason: a fused kernel
// that quietly changed the default's numerics would be a correctness
// surprise rather than an optimisation.
enum Epilogue { kNone = 0, kAdd = 1, kAddFastGelu = 2, kAddSilu = 3 };

// How many CK instances are compiled in for this (dtype, epilogue). 0 if
// the combination isn't built.
int num_instances(int dtype, int epilogue);

// True if instance `idx` accepts this problem (CK's own
// IsSupportedArgument -- vector-load alignment and tile divisibility).
bool supported(int dtype, int epilogue, int idx, const GemmProblem& p);

// Bytes of scratch instance `idx` needs for this problem, 0 if none.
size_t workspace_bytes(int dtype, int epilogue, int idx, const GemmProblem& p);

// Runs instance `idx`. p_bias must be non-null for every epilogue except
// kNone, and null for kNone. Returns false if the instance doesn't support
// the problem.
bool run(int dtype, int epilogue, int idx, const GemmProblem& p, const void* p_a,
         const void* p_b, const void* p_bias, void* p_e, void* p_workspace, hipStream_t stream);

}  // namespace amd_tuned_torch_ck_gemm
