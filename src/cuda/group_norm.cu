// Hand-written GroupNorm forward for amd_tuned_torch on ROCm/RDNA3 (RX 7900 XTX,
// gfx1100). Neither aiter nor TransformerEngine cover GroupNorm -- it's a
// diffusion-U-Net-specific op (per-spatial-group norm), not a
// transformer/LLM one those two target. So this stays a genuine hand-rolled
// kernel, unlike linear/matmul/bmm (aiter Triton GEMM, amd_tuned_torch/aiter_ops.py)
// and attention/layernorm/rmsnorm/gelu/silu (TE, amd_tuned_torch/te_ops.py).
//
// Uses only native HIP types (__half, hip_bfloat16) -- no Composable Kernel
// dependency. This project used to link against CK's DeviceGemm instances
// for the GEMM ops (src/cuda/ck_gemm.cu, since deleted in favor of aiter's
// Triton GEMM) and pulled in CK's header-only half_t/bhalf_t/type_convert
// utilities here too; with the GEMM path gone, there was no reason left for
// this file to depend on CK either.
//
// One block per (n, group): block-wide two-pass reduction (sum/sumsq via
// warp shuffle + shared mem, matching HIP's `warpSize` -- 32 or 64 depending
// on wavefront mode, never hardcoded) then a normalize+affine pass. This is
// the standard batch/group-norm block-reduction pattern, not the
// one-thread-per-output anti-pattern the original CMP-Turing project's
// unverified conv3d/interpolate kernels used.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

namespace {

__device__ __forceinline__ float to_float(const __half& v) { return __half2float(v); }
__device__ __forceinline__ float to_float(const hip_bfloat16& v) { return static_cast<float>(v); }
__device__ __forceinline__ float to_float(const float& v) { return v; }

template <typename T>
__device__ __forceinline__ T from_float(float v);
template <>
__device__ __forceinline__ __half from_float<__half>(float v) { return __float2half(v); }
template <>
__device__ __forceinline__ hip_bfloat16 from_float<hip_bfloat16>(float v) { return hip_bfloat16(v); }
template <>
__device__ __forceinline__ float from_float<float>(float v) { return v; }

template <typename T, typename AccT>
__global__ void group_norm_kernel(T* __restrict__ out, const T* __restrict__ in,
                                   const T* __restrict__ gamma, const T* __restrict__ beta,
                                   int C, int HxW, int groups, float eps) {
    const int n = blockIdx.x / groups;
    const int g = blockIdx.x % groups;
    const int channels_per_group = C / groups;
    const long long group_size = (long long)channels_per_group * HxW;
    const int c_start = g * channels_per_group;

    const T* in_group = in + ((long long)n * C + c_start) * (long long)HxW;
    T* out_group = out + ((long long)n * C + c_start) * (long long)HxW;

    AccT sum = 0;
    AccT sumsq = 0;
    for (long long i = threadIdx.x; i < group_size; i += blockDim.x) {
        AccT v = to_float(in_group[i]);
        sum += v;
        sumsq += v * v;
    }

    __shared__ AccT s_sum[32];
    __shared__ AccT s_sumsq[32];

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
        sumsq += __shfl_down(sumsq, offset);
    }

    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) {
        s_sum[warp_id] = sum;
        s_sumsq[warp_id] = sumsq;
    }
    __syncthreads();

    const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (warp_id == 0) {
        sum = (lane < num_warps) ? s_sum[lane] : AccT(0);
        sumsq = (lane < num_warps) ? s_sumsq[lane] : AccT(0);
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            sum += __shfl_down(sum, offset);
            sumsq += __shfl_down(sumsq, offset);
        }
        if (lane == 0) {
            s_sum[0] = sum;
            s_sumsq[0] = sumsq;
        }
    }
    __syncthreads();

    const AccT mean = s_sum[0] / (AccT)group_size;
    const AccT var = s_sumsq[0] / (AccT)group_size - mean * mean;
    const AccT rstd = AccT(1) / sqrtf((float)var + eps);

    for (long long i = threadIdx.x; i < group_size; i += blockDim.x) {
        const int c = c_start + (int)(i / HxW);
        AccT v = to_float(in_group[i]);
        AccT norm = (v - mean) * rstd;
        if (gamma != nullptr) norm *= to_float(gamma[c]);
        if (beta != nullptr) norm += to_float(beta[c]);
        out_group[i] = from_float<T>(norm);
    }
}

constexpr int kThreadsPerBlock = 256;

}  // namespace

void launch_group_norm_fp16(void* output, const void* input, const void* gamma, const void* beta,
                             int N, int C, int HxW, int groups, float eps, hipStream_t stream) {
    dim3 grid(N * groups);
    hipLaunchKernelGGL((group_norm_kernel<__half, float>), grid, dim3(kThreadsPerBlock), 0,
                        stream, reinterpret_cast<__half*>(output),
                        reinterpret_cast<const __half*>(input),
                        reinterpret_cast<const __half*>(gamma),
                        reinterpret_cast<const __half*>(beta), C, HxW, groups, eps);
}

void launch_group_norm_bf16(void* output, const void* input, const void* gamma, const void* beta,
                             int N, int C, int HxW, int groups, float eps, hipStream_t stream) {
    dim3 grid(N * groups);
    hipLaunchKernelGGL((group_norm_kernel<hip_bfloat16, float>), grid, dim3(kThreadsPerBlock), 0,
                        stream, reinterpret_cast<hip_bfloat16*>(output),
                        reinterpret_cast<const hip_bfloat16*>(input),
                        reinterpret_cast<const hip_bfloat16*>(gamma),
                        reinterpret_cast<const hip_bfloat16*>(beta), C, HxW, groups, eps);
}

void launch_group_norm_fp32(float* output, const float* input, const float* gamma,
                             const float* beta, int N, int C, int HxW, int groups, float eps,
                             hipStream_t stream) {
    dim3 grid(N * groups);
    hipLaunchKernelGGL((group_norm_kernel<float, float>), grid, dim3(kThreadsPerBlock), 0, stream,
                        output, input, gamma, beta, C, HxW, groups, eps);
}
