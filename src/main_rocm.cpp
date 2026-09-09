// amd_tuned_torch native extension -- ROCm/RX 7900 XTX (gfx1100) build.
//
// group_norm, conv2d, and conv3d live here -- linear/matmul/bmm used to be
// here too (backed by Composable Kernel's DeviceGemm instances,
// src/cuda/ck_gemm.cu), but that was replaced with aiter's Triton WMMA GEMM
// (amd_tuned_torch/aiter_ops.py), which needs no separate C++/HIP
// compilation at all. group_norm/conv2d/conv3d stay real compiled HIP
// kernels: group_norm (src/cuda/group_norm.cu) because neither aiter nor
// TransformerEngine cover it (a diffusion-U-Net-specific op); conv2d/conv3d
// as hand-rolled alternatives to aiter's Triton conv2d (fp16/bf16 only, and
// benchmarked slower than stock for the fp16 case on RX 7900 XTX -- see
// amd_tuned_torch/__init__.py's enable()) and to stock conv3d (never
// covered by aiter or TE at all). fp32 conv2d/conv3d are hand-written .cu
// files (src/cuda/conv{2,3}d_fp32.cu); fp16 conv2d/conv3d are WMMA kernels
// codegen'd from src/cuda/templates/*.cu.tmpl into
// src/cuda/generated/conv{2,3}d_fp16_*.cu, one file per tile-shape variant
// in tools/kernelgen/variants.py -- see run_conv2d_fp16/run_conv3d_fp16
// below for the runtime dispatch across those variants. fp16/fp32 only for
// both -- bf16 conv2d still routes through aiter (see _patched_conv2d).
// fp32 conv3d additionally has a second kernel, conv3d_fp32_winograd.cu, for
// its narrow-scope case (batch=1, 3x3x3/stride1/pad1/dilation1) --
// run_conv3d_fp32 below benchmarks it against the direct kernel once per
// distinct shape and caches whichever wins, since Winograd isn't always
// faster just because it's applicable (see that file's own header).
//
// attention/rms_norm/gelu/silu are NOT here either -- they're routed
// straight to TransformerEngine's PyTorch bindings from
// amd_tuned_torch/te_ops.py, since TE already ships tuned CK/AOTriton-backed fused
// kernels for those on ROCm.
#include <torch/extension.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Optional.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <array>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <tuple>

#include "dispatch_common.h"
#include "cuda/iu4_gemm_fwd.hpp"

// Composable Kernel and hipBLASLt bindings used to live in this file's own
// PYBIND11_MODULE, gated by #ifdef -- but that meant all three tiers (this
// core one, CK, hipBLASLt) linked into ONE .so, so touching CK code (~94%
// of a full build's CPU time) forced a relink of this file's own
// group_norm/conv2d/conv3d bindings too, and vice versa. They now live in
// their own independent extensions -- see src/ck_native.cpp and
// src/hipblaslt_native.cpp, each with its own PYBIND11_MODULE, built as
// amd_tuned_torch._native_ck / amd_tuned_torch._native_hipblaslt (see
// setup.py). This file no longer includes or binds either tier at all.

// ------------------------------------------------------------------
// Launchers (src/cuda/group_norm.cu, conv2d_fp32.cu, conv3d_fp{16,32}.cu,
// src/cuda/generated/conv{2,3}d_fp16_*.cu -- see tools/kernelgen/)
// ------------------------------------------------------------------

void launch_group_norm_fp16(void* output, const void* input, const void* gamma,
                             const void* beta, int N, int C, int HxW, int groups, float eps,
                             hipStream_t stream);
void launch_group_norm_bf16(void* output, const void* input, const void* gamma,
                             const void* beta, int N, int C, int HxW, int groups, float eps,
                             hipStream_t stream);
void launch_group_norm_fp32(float* output, const float* input, const float* gamma,
                             const float* beta, int N, int C, int HxW, int groups, float eps,
                             hipStream_t stream);

// src/cuda/generated/conv2d_fp16_*.cu -- one launcher per tile-shape
// variant in tools/kernelgen/variants.py; run_conv2d_fp16 below (in the
// anonymous namespace) picks which one to call per input shape, via a
// real live benchmark-and-cache dispatch (unlike run_conv3d_fp16, which
// still only has one variant to pick from).
void launch_conv2d_fp16_bm256_bn128_bk32_s2(
    const void* input, const void* weight, const void* bias, void* output,
    int B, int C_in, int H_in, int W_in,
    int C_out, int K_H, int K_W,
    int H_out, int W_out,
    int stride_h, int stride_w, int pad_h, int pad_w, int dil_h, int dil_w,
    hipStream_t stream);
void launch_conv2d_fp16_bm256_bn64_bk32_s2(
    const void* input, const void* weight, const void* bias, void* output,
    int B, int C_in, int H_in, int W_in,
    int C_out, int K_H, int K_W,
    int H_out, int W_out,
    int stride_h, int stride_w, int pad_h, int pad_w, int dil_h, int dil_w,
    hipStream_t stream);
void launch_conv2d_fp16_bm128_bn128_bk32_s3(
    const void* input, const void* weight, const void* bias, void* output,
    int B, int C_in, int H_in, int W_in,
    int C_out, int K_H, int K_W,
    int H_out, int W_out,
    int stride_h, int stride_w, int pad_h, int pad_w, int dil_h, int dil_w,
    hipStream_t stream);

// Per-variant diagnostics (static VGPR count, static/dynamic LDS bytes,
// achievable occupancy) for the three conv2d_fp16 launchers above -- see
// each generated file's conv2d_fp16_diagnostics_<suffix> for what these
// actually query (hipFuncGetAttributes / hipOccupancyMaxActiveBlocksPer-
// Multiprocessor) and why this is diagnostic-only, not wired into
// run_conv2d_fp16's dispatch. Exposed to Python via
// conv2d_fp16_variant_diagnostics further down, for
// tools/kernelgen/autotune.py to report alongside timing.
void conv2d_fp16_diagnostics_bm256_bn128_bk32_s2(
    int* num_regs, int* static_lds_bytes, int* dynamic_lds_bytes, int* max_active_blocks_per_cu);
void conv2d_fp16_diagnostics_bm256_bn64_bk32_s2(
    int* num_regs, int* static_lds_bytes, int* dynamic_lds_bytes, int* max_active_blocks_per_cu);
void conv2d_fp16_diagnostics_bm128_bn128_bk32_s3(
    int* num_regs, int* static_lds_bytes, int* dynamic_lds_bytes, int* max_active_blocks_per_cu);

void launch_conv2d_fp32(
    const float* input, const float* weight, const float* bias, float* output,
    int B, int C_in, int H_in, int W_in,
    int C_out, int K_H, int K_W,
    int H_out, int W_out,
    int s_h, int s_w, int p_h, int p_w, int d_h, int d_w,
    hipStream_t stream);

// src/cuda/generated/conv3d_fp16_*.cu -- one launcher per tile-shape
// variant in tools/kernelgen/variants.py; run_conv3d_fp16 below (in the
// anonymous namespace) picks which one to call per input shape.
void launch_conv3d_fp16_bm256_bn128_bk32_s2(
    const void* input, const void* weight, const void* bias, void* output,
    int B, int C_in, int D_in, int H_in, int W_in,
    int C_out, int K_D, int K_H, int K_W,
    int D_out, int H_out, int W_out,
    int s_d, int s_h, int s_w,
    int p_d, int p_h, int p_w,
    int d_d, int d_h, int d_w,
    hipStream_t stream);

void launch_conv3d_fp32(
    const float* input, const float* weight, const float* bias, float* output,
    int B, int C_in, int D_in, int H_in, int W_in,
    int C_out, int K_D, int K_H, int K_W,
    int D_out, int H_out, int W_out,
    int s_d, int s_h, int s_w,
    int p_d, int p_h, int p_w,
    int d_d, int d_h, int d_w,
    hipStream_t stream);

// src/cuda/conv3d_fp32_winograd.cu -- narrower-scope fp32 conv3d (batch=1,
// 3x3x3/stride1/pad1/dilation1 only). Returns false, output untouched, for
// any shape outside that scope -- see run_conv3d_fp32 below for the
// benchmark-and-cache dispatch between this and launch_conv3d_fp32.
extern "C" bool launch_conv3d_fp32_winograd(
    const float* input, const float* weight, const float* bias, float* output,
    int B, int C_in, int D_in, int H_in, int W_in,
    int C_out, int K_D, int K_H, int K_W,
    int D_out, int H_out, int W_out,
    int s_d, int s_h, int s_w,
    int p_d, int p_h, int p_w,
    int d_d, int d_h, int d_w,
    hipStream_t stream);

// src/cuda/generated/conv3d_fp16_winograd_bt8_bc8.cu (codegen'd from
// src/cuda/templates/conv3d_fp16_winograd.cu.tmpl, see tools/kernelgen/)
// -- narrower-scope fp16 conv3d, same scope restrictions as
// launch_conv3d_fp32_winograd above. Returns false, output untouched, for
// any shape outside that scope. UNLIKE the fp32 Winograd kernel, this one
// is NOT wired into run_conv3d_fp16's dispatch below (it's numerically
// unvalidated on any hardware -- see that template's header) -- it's only
// reachable via conv3d_fp16_winograd_bt8_bc8_forward further down, the
// pybind entry point amd_tuned_torch.enable_conv3d_winograd_fp16() (an
// explicit, off-by-default opt-in in amd_tuned_torch/__init__.py) calls.
extern "C" bool launch_conv3d_fp16_winograd_bt8_bc8(
    const void* input, const void* weight, const void* bias, void* output,
    int B, int C_in, int D_in, int H_in, int W_in,
    int C_out, int K_D, int K_H, int K_W,
    int D_out, int H_out, int W_out,
    int s_d, int s_h, int s_w,
    int p_d, int p_h, int p_w,
    int d_d, int d_h, int d_w,
    hipStream_t stream);

namespace {
hipStream_t current_stream() {
    return at::cuda::getCurrentCUDAStream().stream();
}

int64_t conv_out_size(int64_t in, int64_t kernel, int64_t stride, int64_t pad, int64_t dil) {
    return (in + 2 * pad - dil * (kernel - 1) - 1) / stride + 1;
}

// ------------------------------------------------------------------
// Conv3d fp32: dispatch between conv3d_fp32.cu's direct kernel and
// conv3d_fp32_winograd.cu's Winograd kernel. Winograd only ever applies to
// a narrow shape (batch=1, 3x3x3/stride1/pad1/dilation1, even D/H/W,
// tile-count/C_out divisible by 8) -- see that file's launch guard -- and
// isn't assumed faster than the direct kernel just because it's
// applicable (the file's own header documents this as measured, shape-
// dependent). So: the first time a given shape is seen, both kernels run
// and are timed with HIP events; the loser is never run again for that
// exact shape (cached by every dimension/stride/padding/dilation value
// that affects timing or eligibility). Every call after that just runs
// the cached winner directly, at normal single-kernel cost.
//
// Conv3dShapeKey/Conv3dShapeKeyHash live in dispatch_common.h -- shared
// with Conv2dShapeKey/run_conv2d_fp16 and run_conv3d_fp16 below, which
// generalize this same shape-keyed-cache idea to picking among N
// codegen'd WMMA tile-shape variants instead of a binary Winograd choice.
std::mutex g_conv3d_winograd_mutex;
std::unordered_map<Conv3dShapeKey, bool, Conv3dShapeKeyHash> g_conv3d_prefer_winograd;

void run_conv3d_fp32(
    const float* input, const float* weight, const float* bias, float* output,
    int64_t B, int64_t C_in, int64_t D_in, int64_t H_in, int64_t W_in,
    int64_t C_out, int64_t K_D, int64_t K_H, int64_t K_W,
    int64_t D_out, int64_t H_out, int64_t W_out,
    int64_t s_d, int64_t s_h, int64_t s_w,
    int64_t p_d, int64_t p_h, int64_t p_w,
    int64_t d_d, int64_t d_h, int64_t d_w,
    const torch::TensorOptions& scratch_options,
    hipStream_t stream)
{
    auto call_direct = [&]() {
        launch_conv3d_fp32(input, weight, bias, output,
                            (int)B, (int)C_in, (int)D_in, (int)H_in, (int)W_in,
                            (int)C_out, (int)K_D, (int)K_H, (int)K_W,
                            (int)D_out, (int)H_out, (int)W_out,
                            (int)s_d, (int)s_h, (int)s_w,
                            (int)p_d, (int)p_h, (int)p_w,
                            (int)d_d, (int)d_h, (int)d_w, stream);
    };
    auto call_winograd = [&](float* out_ptr) {
        return launch_conv3d_fp32_winograd(input, weight, bias, out_ptr,
                                            (int)B, (int)C_in, (int)D_in, (int)H_in, (int)W_in,
                                            (int)C_out, (int)K_D, (int)K_H, (int)K_W,
                                            (int)D_out, (int)H_out, (int)W_out,
                                            (int)s_d, (int)s_h, (int)s_w,
                                            (int)p_d, (int)p_h, (int)p_w,
                                            (int)d_d, (int)d_h, (int)d_w, stream);
    };

    Conv3dShapeKey key{B, C_in, D_in, H_in, W_in, C_out, K_D, K_H, K_W,
                       D_out, H_out, W_out, s_d, s_h, s_w, p_d, p_h, p_w, d_d, d_h, d_w};

    {
        bool cached, prefer_winograd = false;
        {
            std::lock_guard<std::mutex> lock(g_conv3d_winograd_mutex);
            auto it = g_conv3d_prefer_winograd.find(key);
            cached = it != g_conv3d_prefer_winograd.end();
            if (cached) prefer_winograd = it->second;
        }
        if (cached) {
            // Winograd's eligibility guard is a pure function of shape, so
            // a shape cached as "Winograd wins" should always take that
            // path again -- but if it unexpectedly declines (e.g. a
            // transient hipMalloc failure under memory pressure), fall
            // back to the direct kernel rather than leave `output` unwritten.
            if (!prefer_winograd || !call_winograd(output)) {
                call_direct();
            }
            return;
        }
    }

    // First time this exact shape has been seen: try Winograd into the
    // real output first (if it's out of scope it just returns false, no
    // allocation/compute happened, and we fall back to the direct kernel
    // -- cheap, so no need to benchmark something that never runs).
    // Every hipEvent_* call in this file (here and in run_conv2d_fp16's
    // variant sweep below) is (void)-cast: ROCm 7 marks hipError_t
    // [[nodiscard]], and dropping the status is deliberate. These events
    // only feed the which-kernel-is-faster caches; every hipEventElapsedTime
    // out-param is zero-initialized first, so a failed query degrades to a
    // 0.0ms reading -- i.e. a deterministic pick of the first candidate,
    // never an uninitialized one -- and correctness never depends on it,
    // since `output` is written by a real kernel launch regardless of which
    // candidate the timings crown.
    hipEvent_t t_start, t_mid, t_end;
    (void)hipEventCreate(&t_start);
    (void)hipEventCreate(&t_mid);
    (void)hipEventCreate(&t_end);

    (void)hipEventRecord(t_start, stream);
    bool winograd_applicable = call_winograd(output);
    (void)hipEventRecord(t_mid, stream);

    if (!winograd_applicable) {
        (void)hipEventDestroy(t_start);
        (void)hipEventDestroy(t_mid);
        (void)hipEventDestroy(t_end);
        std::lock_guard<std::mutex> lock(g_conv3d_winograd_mutex);
        g_conv3d_prefer_winograd[key] = false;
        call_direct();
        return;
    }

    // Winograd is in-scope and `output` already holds its (correct) result
    // for this call. Time the direct kernel too, into a scratch buffer
    // purely to decide which kernel to prefer for this shape from now on
    // -- the scratch result itself is discarded either way.
    torch::Tensor scratch = torch::empty({B, C_out, D_out, H_out, W_out}, scratch_options);
    launch_conv3d_fp32(input, weight, bias, scratch.data_ptr<float>(),
                        (int)B, (int)C_in, (int)D_in, (int)H_in, (int)W_in,
                        (int)C_out, (int)K_D, (int)K_H, (int)K_W,
                        (int)D_out, (int)H_out, (int)W_out,
                        (int)s_d, (int)s_h, (int)s_w,
                        (int)p_d, (int)p_h, (int)p_w,
                        (int)d_d, (int)d_h, (int)d_w, stream);
    (void)hipEventRecord(t_end, stream);
    (void)hipEventSynchronize(t_end);

    float winograd_ms = 0.0f, direct_ms = 0.0f;
    (void)hipEventElapsedTime(&winograd_ms, t_start, t_mid);
    (void)hipEventElapsedTime(&direct_ms, t_mid, t_end);
    (void)hipEventDestroy(t_start);
    (void)hipEventDestroy(t_mid);
    (void)hipEventDestroy(t_end);

    bool prefer_winograd = winograd_ms < direct_ms;
    {
        std::lock_guard<std::mutex> lock(g_conv3d_winograd_mutex);
        g_conv3d_prefer_winograd[key] = prefer_winograd;
    }
    // `output` already has the right answer from the Winograd call above
    // regardless of which kernel wins the benchmark -- both compute the
    // same convolution, so there's nothing left to copy.
}

// ------------------------------------------------------------------
// Conv2d fp16 / Conv3d fp16: dispatch across codegen'd WMMA tile-shape
// variants (src/cuda/generated/conv{2,3}d_fp16_*.cu, one per entry in
// tools/kernelgen/variants.py). This generalizes run_conv3d_fp32 above
// from a binary (Winograd-or-not) choice to an N-way one, using the same
// shape-keyed-cache shape (lock, look up, insert-if-missing, call the
// resolved kernel).
//
// conv2d_fp16 now has 3 variants (tools/kernelgen/variants.py) -- see
// run_conv2d_fp16 below for the real live benchmark-and-cache sweep this
// enables. conv3d_fp16 still ships exactly one variant (today's original
// BM=256/BN=128/BK=32/STAGES=2 tile shape, reproduced byte-for-byte
// through the codegen pipeline), so run_conv3d_fp16's "first time this
// shape is seen" branch still always resolves to index 0 with no
// benchmarking -- the cache/lookup machinery there already supports N
// variants unchanged, same as conv2d_fp16's did before this.
using Conv2dFp16Launcher = void (*)(
    const void*, const void*, const void*, void*,
    int, int, int, int,
    int, int, int,
    int, int,
    int, int, int, int, int, int,
    hipStream_t);

using Conv3dFp16Launcher = void (*)(
    const void*, const void*, const void*, void*,
    int, int, int, int, int,
    int, int, int, int,
    int, int, int,
    int, int, int,
    int, int, int,
    int, int, int,
    hipStream_t);

constexpr Conv2dFp16Launcher kConv2dFp16Variants[] = {
    launch_conv2d_fp16_bm256_bn128_bk32_s2,
    launch_conv2d_fp16_bm256_bn64_bk32_s2,
    launch_conv2d_fp16_bm128_bn128_bk32_s3,
};

// Order MUST match kConv2dFp16Variants above (and, in turn,
// tools/kernelgen/variants.py's conv2d_fp16 entries) -- variant_idx means
// the same thing across the launcher array, this one, and the live
// dispatch cache below.
using Conv2dFp16Diagnostics = void (*)(int*, int*, int*, int*);

constexpr Conv2dFp16Diagnostics kConv2dFp16DiagnosticsFns[] = {
    conv2d_fp16_diagnostics_bm256_bn128_bk32_s2,
    conv2d_fp16_diagnostics_bm256_bn64_bk32_s2,
    conv2d_fp16_diagnostics_bm128_bn128_bk32_s3,
};

constexpr Conv3dFp16Launcher kConv3dFp16Variants[] = {
    launch_conv3d_fp16_bm256_bn128_bk32_s2,
};

std::mutex g_conv2d_fp16_mutex;
std::unordered_map<Conv2dShapeKey, int, Conv2dShapeKeyHash> g_conv2d_fp16_variant;

std::mutex g_conv3d_fp16_mutex;
std::unordered_map<Conv3dShapeKey, int, Conv3dShapeKeyHash> g_conv3d_fp16_variant;

void run_conv2d_fp16(
    const void* input, const void* weight, const void* bias, void* output,
    int64_t B, int64_t C_in, int64_t H_in, int64_t W_in,
    int64_t C_out, int64_t K_H, int64_t K_W,
    int64_t H_out, int64_t W_out,
    int64_t s_h, int64_t s_w, int64_t p_h, int64_t p_w, int64_t d_h, int64_t d_w,
    const torch::TensorOptions& scratch_options,
    hipStream_t stream)
{
    constexpr int kNumVariants = sizeof(kConv2dFp16Variants) / sizeof(kConv2dFp16Variants[0]);

    auto call_variant = [&](int idx, void* out_ptr) {
        kConv2dFp16Variants[idx](
            input, weight, bias, out_ptr,
            (int)B, (int)C_in, (int)H_in, (int)W_in,
            (int)C_out, (int)K_H, (int)K_W,
            (int)H_out, (int)W_out,
            (int)s_h, (int)s_w, (int)p_h, (int)p_w, (int)d_h, (int)d_w,
            stream);
    };

    Conv2dShapeKey key{B, C_in, H_in, W_in, C_out, K_H, K_W, H_out, W_out,
                       s_h, s_w, p_h, p_w, d_h, d_w};

    {
        bool cached = false;
        int variant_idx = 0;
        {
            std::lock_guard<std::mutex> lock(g_conv2d_fp16_mutex);
            auto it = g_conv2d_fp16_variant.find(key);
            cached = it != g_conv2d_fp16_variant.end();
            if (cached) variant_idx = it->second;
        }
        if (cached) {
            call_variant(variant_idx, output);
            return;
        }
    }

    if (kNumVariants == 1) {
        // Nothing to benchmark -- resolve to the only candidate and cache it.
        call_variant(0, output);
        std::lock_guard<std::mutex> lock(g_conv2d_fp16_mutex);
        g_conv2d_fp16_variant[key] = 0;
        return;
    }

    // First time this exact shape has been seen: every candidate is
    // unconditionally applicable (no eligibility narrowing the way
    // Winograd has -- see conv3d_fp16_winograd_bt8_bc8_forward), so with
    // kNumVariants candidates a single cold timed sample is more likely
    // to crown the wrong winner from GPU clock/thermal noise alone as
    // that count grows -- 2 warmup iterations + 1 timed per candidate,
    // not run_conv3d_fp32's single-shot (that's a much lower-risk binary,
    // eligibility-gated choice). Every candidate (warmup AND timed) runs
    // into one shared scratch buffer -- never the real `output`, including
    // during the eventual winner's own sweep runs -- then the confirmed
    // winner is launched exactly once more into `output`. All timed
    // launches are queued back-to-back on the stream (one hipEventRecord
    // pair per candidate), with a single hipEventSynchronize after the
    // last one so an intermediate host sync doesn't serialize what's
    // meant to measure steady-state back-to-back behavior. Total
    // first-touch cost: kNumVariants*(2 warmup + 1 timed) + 1 extra
    // launches -- paid once per distinct shape; every later call for the
    // same shape hits the cache above at normal single-kernel cost.
    torch::Tensor scratch = torch::empty({B, C_out, H_out, W_out}, scratch_options);
    void* scratch_ptr = scratch.data_ptr();

    hipEvent_t starts[kNumVariants];
    hipEvent_t ends[kNumVariants];
    for (int i = 0; i < kNumVariants; i++) {
        (void)hipEventCreate(&starts[i]);
        (void)hipEventCreate(&ends[i]);
    }

    for (int i = 0; i < kNumVariants; i++) {
        for (int w = 0; w < 2; w++) {
            call_variant(i, scratch_ptr);
        }
        (void)hipEventRecord(starts[i], stream);
        call_variant(i, scratch_ptr);
        (void)hipEventRecord(ends[i], stream);
    }
    (void)hipEventSynchronize(ends[kNumVariants - 1]);

    int best = 0;
    float best_ms = 0.0f;
    for (int i = 0; i < kNumVariants; i++) {
        float ms = 0.0f;
        (void)hipEventElapsedTime(&ms, starts[i], ends[i]);
        if (i == 0 || ms < best_ms) {
            best = i;
            best_ms = ms;
        }
        (void)hipEventDestroy(starts[i]);
        (void)hipEventDestroy(ends[i]);
    }

    {
        std::lock_guard<std::mutex> lock(g_conv2d_fp16_mutex);
        g_conv2d_fp16_variant[key] = best;
    }

    call_variant(best, output);
}

void run_conv3d_fp16(
    const void* input, const void* weight, const void* bias, void* output,
    int64_t B, int64_t C_in, int64_t D_in, int64_t H_in, int64_t W_in,
    int64_t C_out, int64_t K_D, int64_t K_H, int64_t K_W,
    int64_t D_out, int64_t H_out, int64_t W_out,
    int64_t s_d, int64_t s_h, int64_t s_w,
    int64_t p_d, int64_t p_h, int64_t p_w,
    int64_t d_d, int64_t d_h, int64_t d_w,
    hipStream_t stream)
{
    Conv3dShapeKey key{B, C_in, D_in, H_in, W_in, C_out, K_D, K_H, K_W,
                       D_out, H_out, W_out, s_d, s_h, s_w, p_d, p_h, p_w, d_d, d_h, d_w};

    int variant_idx;
    {
        std::lock_guard<std::mutex> lock(g_conv3d_fp16_mutex);
        auto it = g_conv3d_fp16_variant.find(key);
        if (it != g_conv3d_fp16_variant.end()) {
            variant_idx = it->second;
        } else {
            variant_idx = 0;  // Phase 1: only one variant exists.
            g_conv3d_fp16_variant[key] = variant_idx;
        }
    }

    kConv3dFp16Variants[variant_idx](
        input, weight, bias, output,
        (int)B, (int)C_in, (int)D_in, (int)H_in, (int)W_in,
        (int)C_out, (int)K_D, (int)K_H, (int)K_W,
        (int)D_out, (int)H_out, (int)W_out,
        (int)s_d, (int)s_h, (int)s_w,
        (int)p_d, (int)p_h, (int)p_w,
        (int)d_d, (int)d_h, (int)d_w,
        stream);
}

}  // namespace

// ------------------------------------------------------------------
// GroupNorm
// ------------------------------------------------------------------

torch::Tensor custom_group_norm_forward(torch::Tensor input, int64_t num_groups,
                                         c10::optional<torch::Tensor> weight,
                                         c10::optional<torch::Tensor> bias, double eps) {
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() >= 3, "Input dim must be >= 3 (N, C, ...)");
    int N = input.size(0);
    int C = input.size(1);
    int64_t HxW = 1;
    for (int i = 2; i < input.dim(); ++i) HxW *= input.size(i);
    TORCH_CHECK(C % num_groups == 0, "Channels must be divisible by groups");

    const void* weight_ptr = nullptr;
    const void* bias_ptr = nullptr;
    if (weight.has_value() && weight->defined()) {
        TORCH_CHECK(weight->size(0) == C && weight->dtype() == input.dtype(),
                    "Weight shape/dtype mismatch");
        weight_ptr = weight->contiguous().data_ptr();
    }
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->size(0) == C && bias->dtype() == input.dtype(),
                    "Bias shape/dtype mismatch");
        bias_ptr = bias->contiguous().data_ptr();
    }

    auto output = torch::empty_like(input);
    hipStream_t stream = current_stream();

    if (input.dtype() == torch::kFloat16) {
        launch_group_norm_fp16(output.data_ptr(), input.data_ptr(), weight_ptr, bias_ptr, N, C,
                                (int)HxW, (int)num_groups, (float)eps, stream);
    } else if (input.dtype() == torch::kBFloat16) {
        launch_group_norm_bf16(output.data_ptr(), input.data_ptr(), weight_ptr, bias_ptr, N, C,
                                (int)HxW, (int)num_groups, (float)eps, stream);
    } else if (input.dtype() == torch::kFloat32) {
        launch_group_norm_fp32(output.data_ptr<float>(), input.data_ptr<float>(),
                                static_cast<const float*>(weight_ptr),
                                static_cast<const float*>(bias_ptr), N, C, (int)HxW,
                                (int)num_groups, (float)eps, stream);
    } else {
        TORCH_CHECK(false, "Unsupported dtype for group_norm");
    }

    return output;
}

// ------------------------------------------------------------------
// Conv2d (groups=1 only -- see amd_tuned_torch/__init__.py's
// _patched_conv2d for the groups!=1 fallback, same convention as aiter's)
// ------------------------------------------------------------------

torch::Tensor custom_conv2d_forward(torch::Tensor input, torch::Tensor weight,
                                     c10::optional<torch::Tensor> bias,
                                     std::vector<int64_t> stride,
                                     std::vector<int64_t> padding,
                                     std::vector<int64_t> dilation) {
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "Input/weight must be contiguous");
    TORCH_CHECK(input.dim() == 4 && weight.dim() == 4, "Conv2d expects 4D input/weight (NCHW/[O,I,kH,kW])");
    TORCH_CHECK(input.dtype() == weight.dtype(), "Input/weight dtype mismatch");
    TORCH_CHECK(input.dtype() == torch::kFloat16 || input.dtype() == torch::kFloat32,
                "Native conv2d only supports fp16/fp32 (bf16 routes through aiter)");

    int64_t B = input.size(0), C_in = input.size(1), H_in = input.size(2), W_in = input.size(3);
    int64_t C_out = weight.size(0), K_H = weight.size(2), K_W = weight.size(3);
    TORCH_CHECK(weight.size(1) == C_in, "weight.size(1) must equal input channels (groups=1 only)");

    int64_t H_out = conv_out_size(H_in, K_H, stride[0], padding[0], dilation[0]);
    int64_t W_out = conv_out_size(W_in, K_W, stride[1], padding[1], dilation[1]);
    TORCH_CHECK(H_out > 0 && W_out > 0, "Conv2d output size must be positive");

    auto output = torch::empty({B, C_out, H_out, W_out}, input.options());
    hipStream_t stream = current_stream();

    const void* bias_ptr = nullptr;
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->size(0) == C_out && bias->dtype() == input.dtype(), "Bias shape/dtype mismatch");
        bias_ptr = bias->contiguous().data_ptr();
    }

    if (input.dtype() == torch::kFloat16) {
        // Dispatches across codegen'd WMMA tile-shape variants -- see
        // run_conv2d_fp16 and tools/kernelgen/.
        run_conv2d_fp16(input.data_ptr(), weight.data_ptr(), bias_ptr, output.data_ptr(),
                         B, C_in, H_in, W_in, C_out, K_H, K_W,
                         H_out, W_out, stride[0], stride[1],
                         padding[0], padding[1], dilation[0], dilation[1],
                         input.options(), stream);
    } else {
        launch_conv2d_fp32(input.data_ptr<float>(), weight.data_ptr<float>(),
                            static_cast<const float*>(bias_ptr), output.data_ptr<float>(),
                            (int)B, (int)C_in, (int)H_in, (int)W_in, (int)C_out, (int)K_H, (int)K_W,
                            (int)H_out, (int)W_out, (int)stride[0], (int)stride[1],
                            (int)padding[0], (int)padding[1], (int)dilation[0], (int)dilation[1],
                            stream);
    }

    return output;
}

// ------------------------------------------------------------------
// Conv2d fp16 cached-variant lookup -- testing/tooling-only entry point.
// Peeks at whatever variant_idx is currently cached in
// g_conv2d_fp16_variant for this exact shape, WITHOUT triggering a
// benchmark if it isn't cached yet (returns None instead). Lets
// tools/kernelgen/autotune.py find out which variant run_conv2d_fp16's
// live dispatch actually settled on for a shape it already ran through
// amd_tuned_torch.ops.conv2d(...), so it can print THAT variant's
// diagnostics (conv2d_fp16_variant_diagnostics) next to the timing
// result -- connecting "this candidate won" to "here's its occupancy
// profile" (or flagging it as a surprise worth a second look, if the
// winner's occupancy isn't obviously better than a loser's).
// ------------------------------------------------------------------

torch::Tensor conv2d_fp16_run_variant(
    int64_t variant_idx, torch::Tensor input, torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation) {
    // Debug/bisection tool only -- calls ONE specific tile-shape variant
    // directly, bypassing run_conv2d_fp16's benchmark-and-cache dispatch
    // entirely (no timing, no winner caching, no correctness check). Exists
    // so a numerical mismatch caught by amd_tuned_torch.kernel_select's
    // native-vs-stock verification (which only ever sees "native" as a
    // whole, i.e. whichever variant run_conv2d_fp16 happened to pick as
    // fastest for that shape) can be bisected to a specific variant --
    // compare this function's output against F.conv2d for each variant_idx
    // in turn to find out whether the bug is in code shared by every
    // variant (the LOAD_A/LOAD_B macros, the epilogue -- all defined once
    // in src/cuda/templates/conv2d_fp16.cu.tmpl) or specific to one
    // variant's BM/BN/BK/STAGES combination.
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "Input/weight must be contiguous");
    TORCH_CHECK(input.dim() == 4 && weight.dim() == 4, "Conv2d expects 4D input/weight (NCHW/[O,I,kH,kW])");
    TORCH_CHECK(input.dtype() == torch::kFloat16 && weight.dtype() == torch::kFloat16,
                "conv2d_fp16_run_variant is fp16-only (the WMMA kernel; conv2d's fp32 path has "
                "only one implementation, nothing to bisect)");
    constexpr int kNumVariants = sizeof(kConv2dFp16Variants) / sizeof(kConv2dFp16Variants[0]);
    TORCH_CHECK(variant_idx >= 0 && variant_idx < kNumVariants,
                "variant_idx out of range (0..", kNumVariants - 1, ")");

    int64_t B = input.size(0), C_in = input.size(1), H_in = input.size(2), W_in = input.size(3);
    int64_t C_out = weight.size(0), K_H = weight.size(2), K_W = weight.size(3);
    TORCH_CHECK(weight.size(1) == C_in, "weight.size(1) must equal input channels (groups=1 only)");

    int64_t H_out = conv_out_size(H_in, K_H, stride[0], padding[0], dilation[0]);
    int64_t W_out = conv_out_size(W_in, K_W, stride[1], padding[1], dilation[1]);
    TORCH_CHECK(H_out > 0 && W_out > 0, "Conv2d output size must be positive");

    const void* bias_ptr = nullptr;
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->size(0) == C_out && bias->dtype() == input.dtype(), "Bias shape/dtype mismatch");
        bias_ptr = bias->contiguous().data_ptr();
    }

    auto output = torch::empty({B, C_out, H_out, W_out}, input.options());
    kConv2dFp16Variants[variant_idx](
        input.data_ptr(), weight.data_ptr(), bias_ptr, output.data_ptr(),
        (int)B, (int)C_in, (int)H_in, (int)W_in,
        (int)C_out, (int)K_H, (int)K_W, (int)H_out, (int)W_out,
        (int)stride[0], (int)stride[1], (int)padding[0], (int)padding[1],
        (int)dilation[0], (int)dilation[1],
        current_stream());
    return output;
}

c10::optional<int64_t> conv2d_fp16_cached_variant(
        int64_t B, int64_t C_in, int64_t H_in, int64_t W_in,
        int64_t C_out, int64_t K_H, int64_t K_W,
        int64_t H_out, int64_t W_out,
        int64_t s_h, int64_t s_w, int64_t p_h, int64_t p_w, int64_t d_h, int64_t d_w) {
    Conv2dShapeKey key{B, C_in, H_in, W_in, C_out, K_H, K_W, H_out, W_out,
                       s_h, s_w, p_h, p_w, d_h, d_w};
    std::lock_guard<std::mutex> lock(g_conv2d_fp16_mutex);
    auto it = g_conv2d_fp16_variant.find(key);
    if (it == g_conv2d_fp16_variant.end()) return c10::nullopt;
    return (int64_t)it->second;
}

// ------------------------------------------------------------------
// Conv3d (groups=1 only)
// ------------------------------------------------------------------

torch::Tensor custom_conv3d_forward(torch::Tensor input, torch::Tensor weight,
                                     c10::optional<torch::Tensor> bias,
                                     std::vector<int64_t> stride,
                                     std::vector<int64_t> padding,
                                     std::vector<int64_t> dilation) {
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "Input/weight must be contiguous");
    TORCH_CHECK(input.dim() == 5 && weight.dim() == 5,
                "Conv3d expects 5D input/weight (NCDHW/[O,I,kD,kH,kW])");
    TORCH_CHECK(input.dtype() == weight.dtype(), "Input/weight dtype mismatch");
    TORCH_CHECK(input.dtype() == torch::kFloat16 || input.dtype() == torch::kFloat32,
                "Native conv3d only supports fp16/fp32");

    int64_t B = input.size(0), C_in = input.size(1);
    int64_t D_in = input.size(2), H_in = input.size(3), W_in = input.size(4);
    int64_t C_out = weight.size(0), K_D = weight.size(2), K_H = weight.size(3), K_W = weight.size(4);
    TORCH_CHECK(weight.size(1) == C_in, "weight.size(1) must equal input channels (groups=1 only)");

    int64_t D_out = conv_out_size(D_in, K_D, stride[0], padding[0], dilation[0]);
    int64_t H_out = conv_out_size(H_in, K_H, stride[1], padding[1], dilation[1]);
    int64_t W_out = conv_out_size(W_in, K_W, stride[2], padding[2], dilation[2]);
    TORCH_CHECK(D_out > 0 && H_out > 0 && W_out > 0, "Conv3d output size must be positive");

    auto output = torch::empty({B, C_out, D_out, H_out, W_out}, input.options());
    hipStream_t stream = current_stream();

    const void* bias_ptr = nullptr;
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->size(0) == C_out && bias->dtype() == input.dtype(), "Bias shape/dtype mismatch");
        bias_ptr = bias->contiguous().data_ptr();
    }

    if (input.dtype() == torch::kFloat16) {
        // Dispatches across codegen'd WMMA tile-shape variants -- see
        // run_conv3d_fp16 and tools/kernelgen/.
        run_conv3d_fp16(input.data_ptr(), weight.data_ptr(), bias_ptr, output.data_ptr(),
                         B, C_in, D_in, H_in, W_in,
                         C_out, K_D, K_H, K_W, D_out, H_out, W_out,
                         stride[0], stride[1], stride[2],
                         padding[0], padding[1], padding[2],
                         dilation[0], dilation[1], dilation[2], stream);
    } else {
        // fp32: try conv3d_fp32_winograd.cu first (benchmarked once per
        // distinct shape against this direct kernel, cached thereafter --
        // see run_conv3d_fp32). Falls straight through to the direct
        // kernel for any shape outside Winograd's scope.
        run_conv3d_fp32(input.data_ptr<float>(), weight.data_ptr<float>(),
                         static_cast<const float*>(bias_ptr), output.data_ptr<float>(),
                         B, C_in, D_in, H_in, W_in,
                         C_out, K_D, K_H, K_W, D_out, H_out, W_out,
                         stride[0], stride[1], stride[2],
                         padding[0], padding[1], padding[2],
                         dilation[0], dilation[1], dilation[2],
                         input.options(), stream);
    }

    return output;
}

// ------------------------------------------------------------------
// Conv3d fp16 Winograd -- testing/opt-in-only entry point (groups=1).
// Deliberately bypasses run_conv3d_fp16's dispatch entirely (that
// function doesn't know this kernel exists -- see the forward
// declaration above for why) and returns c10::nullopt rather than a
// Tensor for any shape outside src/cuda/templates/conv3d_fp16_winograd.cu.tmpl's
// scope, mirroring that kernel's own bool-return/no-side-effect-on-
// decline convention instead of custom_conv3d_forward's always-a-Tensor
// one. The only production caller is amd_tuned_torch.compile_ops.
// conv3d_fp16_winograd_bt8_bc8 / amd_tuned_torch.enable_conv3d_winograd_fp16()
// (Python, off by default); tools/kernelgen/autotune.py also calls this
// directly to benchmark it against stock and the native WMMA kernel.
// ------------------------------------------------------------------

c10::optional<torch::Tensor> conv3d_fp16_winograd_bt8_bc8_forward(
        torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias,
        std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation) {
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "Input/weight must be contiguous");
    TORCH_CHECK(input.dim() == 5 && weight.dim() == 5,
                "Conv3d expects 5D input/weight (NCDHW/[O,I,kD,kH,kW])");
    TORCH_CHECK(input.dtype() == torch::kFloat16 && weight.dtype() == torch::kFloat16,
                "conv3d_fp16_winograd_bt8_bc8 only supports fp16");

    int64_t B = input.size(0), C_in = input.size(1);
    int64_t D_in = input.size(2), H_in = input.size(3), W_in = input.size(4);
    int64_t C_out = weight.size(0), K_D = weight.size(2), K_H = weight.size(3), K_W = weight.size(4);
    TORCH_CHECK(weight.size(1) == C_in, "weight.size(1) must equal input channels (groups=1 only)");

    int64_t D_out = conv_out_size(D_in, K_D, stride[0], padding[0], dilation[0]);
    int64_t H_out = conv_out_size(H_in, K_H, stride[1], padding[1], dilation[1]);
    int64_t W_out = conv_out_size(W_in, K_W, stride[2], padding[2], dilation[2]);
    TORCH_CHECK(D_out > 0 && H_out > 0 && W_out > 0, "Conv3d output size must be positive");

    auto output = torch::empty({B, C_out, D_out, H_out, W_out}, input.options());
    hipStream_t stream = current_stream();

    const void* bias_ptr = nullptr;
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->size(0) == C_out && bias->dtype() == input.dtype(), "Bias shape/dtype mismatch");
        bias_ptr = bias->contiguous().data_ptr();
    }

    bool ok = launch_conv3d_fp16_winograd_bt8_bc8(
        input.data_ptr(), weight.data_ptr(), bias_ptr, output.data_ptr(),
        (int)B, (int)C_in, (int)D_in, (int)H_in, (int)W_in,
        (int)C_out, (int)K_D, (int)K_H, (int)K_W, (int)D_out, (int)H_out, (int)W_out,
        (int)stride[0], (int)stride[1], (int)stride[2],
        (int)padding[0], (int)padding[1], (int)padding[2],
        (int)dilation[0], (int)dilation[1], (int)dilation[2], stream);

    if (!ok) return c10::nullopt;
    return output;
}

// ------------------------------------------------------------------
// Conv2d fp16 variant diagnostics -- testing/tooling-only entry point.
// Wraps conv2d_fp16_diagnostics_<suffix> (see each generated file, and
// kConv2dFp16DiagnosticsFns above) behind a single Python-callable
// function indexed the same way the live dispatch cache is
// (variant_idx into kConv2dFp16Variants / kConv2dFp16DiagnosticsFns,
// matching tools/kernelgen/variants.py's conv2d_fp16 entry order). Purely
// introspective -- doesn't launch the kernel, just reads back its
// compiled resource usage and computes achievable occupancy for it. The
// only caller is tools/kernelgen/autotune.py.
// ------------------------------------------------------------------

std::tuple<int, int, int, int> conv2d_fp16_variant_diagnostics(int64_t variant_idx) {
    constexpr int kNumVariants =
        sizeof(kConv2dFp16DiagnosticsFns) / sizeof(kConv2dFp16DiagnosticsFns[0]);
    TORCH_CHECK(variant_idx >= 0 && variant_idx < kNumVariants,
                "conv2d_fp16_variant_diagnostics: variant_idx out of range [0, ",
                kNumVariants, ")");

    int num_regs = 0, static_lds_bytes = 0, dynamic_lds_bytes = 0, max_active_blocks_per_cu = 0;
    kConv2dFp16DiagnosticsFns[variant_idx](
        &num_regs, &static_lds_bytes, &dynamic_lds_bytes, &max_active_blocks_per_cu);
    return std::make_tuple(num_regs, static_lds_bytes, dynamic_lds_bytes, max_active_blocks_per_cu);
}

// ------------------------------------------------------------------
// Experimental gfx1100 WMMA integer GEMM (src/cuda/iu4_gemm_fwd.cu) --
// see amd_tuned_torch/_vendor/gfx1100_iu4_gemm/NOTICE.md for where this
// came from and amd_tuned_torch/iu4_gemm_ops.py for why it is opt-in only
// (never entered into kernel_select's stock-vs-candidate contest). Raw
// packed-integer matmul, int32 accumulate, no dequant/bias epilogue --
// that happens in Python around these.
// ------------------------------------------------------------------

bool iu4_gemm_supported() {
    // Matches the upstream probe's own restriction: the WMMA
    // iu4/iu8 instructions this kernel emits are gfx11-specific, and this
    // extension can be rebuilt for other targets via AMD_TUNED_TORCH_GPU_ARCH
    // (see setup.py), so this has to be a runtime device check, not a
    // compile-time assumption.
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, 0) != hipSuccess) {
        return false;
    }
    return std::strncmp(properties.gcnArchName, "gfx1100", 7) == 0;
}

namespace {
void check_iu_gemm_inputs(const torch::Tensor& a, const torch::Tensor& b, int a_cols, int b_cols) {
    TORCH_CHECK(iu4_gemm_supported(), "iu4/iu8 GEMM requires gfx1100 (RDNA3 WMMA int4/int8)");
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "iu_gemm inputs must be on the GPU");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "iu_gemm inputs must be contiguous");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "iu_gemm expects 2D [M,K]/[N,K] inputs");
    TORCH_CHECK(a.size(1) == a_cols && b.size(1) == b_cols,
                "iu_gemm: A/B row byte-width mismatch with the requested K");
}
}  // namespace

// a_packed/b_packed: [M, row_bytes]/[N, row_bytes] uint8, two signed
// 4-bit nibbles per byte (see amd_tuned_torch.iu4_gemm_ops.pack_int4_rows).
// k is the true (unpacked) reduction dimension -- row_bytes is
// ceil(k,16)/2, allocated by the caller, not derived here, so the tail
// path's bounds check always sees the real k.
torch::Tensor iu4_gemm(torch::Tensor a_packed, torch::Tensor b_packed, int64_t k) {
    const int row_bytes = (int)a_packed.size(1);
    check_iu_gemm_inputs(a_packed, b_packed, row_bytes, row_bytes);
    TORCH_CHECK(a_packed.dtype() == torch::kUInt8 && b_packed.dtype() == torch::kUInt8,
                "iu4_gemm expects packed uint8 inputs");
    const int64_t m = a_packed.size(0);
    const int64_t n = b_packed.size(0);
    auto output = torch::empty({m, n}, a_packed.options().dtype(torch::kInt32));
    launch_iu4_gemm(a_packed.data_ptr<uint8_t>(), b_packed.data_ptr<uint8_t>(),
                     output.data_ptr<int32_t>(), (int)m, (int)n, (int)k, row_bytes, row_bytes,
                     current_stream());
    return output;
}

torch::Tensor iu8_gemm(torch::Tensor a, torch::Tensor b) {
    const int k = (int)a.size(1);
    check_iu_gemm_inputs(a, b, k, k);
    TORCH_CHECK(a.dtype() == torch::kInt8 && b.dtype() == torch::kInt8,
                "iu8_gemm expects int8 inputs");
    const int64_t m = a.size(0);
    const int64_t n = b.size(0);
    auto output = torch::empty({m, n}, a.options().dtype(torch::kInt32));
    launch_iu8_gemm(a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), output.data_ptr<int32_t>(),
                     (int)m, (int)n, k, k, k, current_stream());
    return output;
}

torch::Tensor dot4_i8_gemm(torch::Tensor a, torch::Tensor b) {
    const int k = (int)a.size(1);
    check_iu_gemm_inputs(a, b, k, k);
    TORCH_CHECK(a.dtype() == torch::kInt8 && b.dtype() == torch::kInt8,
                "dot4_i8_gemm expects int8 inputs");
    const int64_t m = a.size(0);
    const int64_t n = b.size(0);
    auto output = torch::empty({m, n}, a.options().dtype(torch::kInt32));
    launch_dot4_i8_gemm(a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), output.data_ptr<int32_t>(),
                         (int)m, (int)n, k, k, k, current_stream());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("group_norm", &custom_group_norm_forward, "Hand-written HIP GroupNorm");
    m.def("conv2d", &custom_conv2d_forward, "Hand-written HIP Conv2d (fp16/fp32, groups=1)");
    m.def("conv3d", &custom_conv3d_forward, "Hand-written HIP Conv3d (fp16/fp32, groups=1)");
    // CK (has_ck/ck_conv/ck_group_norm/ck_gemm_linear) and hipBLASLt
    // (has_hipblaslt/hipblaslt_linear/hipblaslt_bmm) used to be bound here
    // too -- they now live in their own PYBIND11_MODULEs, see
    // src/ck_native.cpp and src/hipblaslt_native.cpp.
    m.def("conv3d_fp16_winograd_bt8_bc8", &conv3d_fp16_winograd_bt8_bc8_forward,
          "Testing/opt-in-only: direct call to the codegen'd Winograd fp16 conv3d "
          "kernel (groups=1). Returns None (not a Tensor) if the shape is outside "
          "that kernel's scope -- see amd_tuned_torch.enable_conv3d_winograd_fp16().");
    m.def("conv2d_fp16_variant_diagnostics", &conv2d_fp16_variant_diagnostics,
          "Testing/tooling-only: (num_regs, static_lds_bytes, dynamic_lds_bytes, "
          "max_active_blocks_per_cu) for the conv2d_fp16 tile-shape variant at "
          "this index (see tools/kernelgen/variants.py's conv2d_fp16 entry order). "
          "Purely introspective (hipFuncGetAttributes / "
          "hipOccupancyMaxActiveBlocksPerMultiprocessor) -- doesn't launch anything.");
    m.def("conv2d_fp16_run_variant", &conv2d_fp16_run_variant,
          "Debug/bisection tool: run ONE specific conv2d_fp16 tile-shape variant "
          "(see tools/kernelgen/variants.py's conv2d_fp16 entry order) directly, "
          "bypassing run_conv2d_fp16's benchmark-and-cache dispatch. Compare against "
          "F.conv2d per variant_idx to bisect a numerical mismatch to a specific "
          "variant vs. code shared by all of them.");
    m.def("conv2d_fp16_cached_variant", &conv2d_fp16_cached_variant,
          "Testing/tooling-only: the variant_idx run_conv2d_fp16's live dispatch "
          "cached for this exact shape, or None if that shape hasn't been run "
          "through amd_tuned_torch.ops.conv2d(...) yet. Never triggers a benchmark "
          "itself -- pure lookup.");
    m.def("iu4_gemm_supported", &iu4_gemm_supported,
          "True if the current device is gfx1100 (required by the iu4/iu8/dot4_i8 "
          "WMMA integer GEMM kernels below -- see "
          "amd_tuned_torch/_vendor/gfx1100_iu4_gemm/NOTICE.md).");
    m.def("iu4_gemm", &iu4_gemm,
          "EXPERIMENTAL, opt-in only (see amd_tuned_torch.iu4_gemm_ops): raw "
          "packed-int4 x packed-int4 -> int32 GEMM via gfx1100 WMMA. No dequant/"
          "bias epilogue -- caller supplies packed nibble rows and the true K.");
    m.def("iu8_gemm", &iu8_gemm,
          "EXPERIMENTAL, opt-in only (see amd_tuned_torch.iu4_gemm_ops): raw "
          "int8 x int8 -> int32 GEMM via gfx1100 WMMA. No dequant/bias epilogue.");
    m.def("dot4_i8_gemm", &dot4_i8_gemm,
          "EXPERIMENTAL, opt-in only (see amd_tuned_torch.iu4_gemm_ops): raw "
          "int8 x int8 -> int32 GEMM via v_dot4_i32_i8 -- small/tail fallback "
          "control for iu8_gemm, not expected to win at WMMA-tile sizes.");
}
