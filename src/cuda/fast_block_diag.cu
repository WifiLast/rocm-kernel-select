// "Fast block diagonal" assembly/disassembly kernel for BOFT (Butterfly
// Orthogonal Fine-Tuning, source/peft's peft.tuners.boft) -- ported from
// that package's own optional CUDA extension
// (source/peft/src/peft/tuners/boft/fbd/fbd_cuda_kernel.cu, Yao Feng,
// 2023/08) into this project's own always-built native extension. See
// amd_tuned_torch/boft_ops.py's module docstring for why this is worth
// having as a real compiled tier here rather than relying on BOFT's own
// runtime torch.utils.cpp_extension.load() JIT build, and for the bf16 gap
// this port fixes along the way (upstream's AT_DISPATCH_FLOATING_TYPES_AND_
// HALF never covered bf16 at all -- a real crash for a bf16 BOFT adapter,
// not a ROCm-specific issue).
//
// ALGORITHM. Forward takes a batch of small (b, b) blocks, input[z, N, b, b],
// and scatters each one into its own diagonal position of a bigger
// (N*b, N*b) matrix per batch element z, output[z, N*b, N*b] -- i.e. builds
// block_diag(input[z, 0], input[z, 1], ..., input[z, N-1]) for every z at
// once. Backward is the exact inverse: gather each block back out of the
// (now upstream-gradient) big matrix's diagonal position. Pure data
// movement, no arithmetic at all -- unlike group_norm.cu's reduction kernel,
// there is nothing dtype-specific here beyond the element type itself, so
// one template covers every dtype with no to_float/from_float conversion
// helpers needed (a plain `output[...] = input[...]` copy is exact for
// half/bfloat16 too).
//
// PORTABILITY. Flat linear-index elementwise kernel, no shared memory, no
// warp-level ops, no tensor-core intrinsics -- about as portable a CUDA
// kernel as exists, hence the straight syntax translation (CUDA launch ->
// hipLaunchKernelGGL, matching every other kernel in this directory) with
// no change to the addressing arithmetic itself.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#define DIV_CEIL(a, b) (((a) + (b) - 1) / (b))

namespace {

template <typename T>
__global__ void fast_block_diag_fwd_kernel(
    const T* __restrict__ input,   // [z, N, b, b]
    T* __restrict__ output,        // [z, N*b, N*b], pre-zeroed by the caller
    int z, int N, int b) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= z * N * b * b) return;

    const int zi = i / (N * b * b);
    const int Ni = (i % (N * b * b)) / (b * b);
    const int x = ((i % (N * b * b)) % (b * b)) / b;
    const int y = ((i % (N * b * b)) % (b * b)) % b;

    output[zi * N * b * N * b + (Ni * b + x) * N * b + Ni * b + y] =
        input[zi * N * b * b + Ni * b * b + x * b + y];
}

template <typename T>
__global__ void fast_block_diag_bwd_kernel(
    const T* __restrict__ grad_output,  // [z, N*b, N*b]
    T* __restrict__ grad_input,         // [z, N, b, b]
    int z, int N, int b) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= z * N * b * b) return;

    const int zi = i / (N * b * b);
    const int Ni = (i % (N * b * b)) / (b * b);
    const int x = ((i % (N * b * b)) % (b * b)) / b;
    const int y = ((i % (N * b * b)) % (b * b)) % b;

    grad_input[zi * N * b * b + Ni * b * b + x * b + y] =
        grad_output[zi * N * b * N * b + (Ni * b + x) * N * b + Ni * b + y];
}

constexpr int kThreads = 512;

}  // namespace

// One typed launcher per dtype this tier supports, same convention as
// group_norm.cu/conv2d_fp32.cu's launch_* functions declared in
// main_rocm.cpp -- fp16/bf16/fp32/fp64 covers upstream's original
// fp16/fp32/fp64 (AT_DISPATCH_FLOATING_TYPES_AND_HALF) plus the bf16 case
// it never had.
#define DEFINE_FAST_BLOCK_DIAG_LAUNCHERS(SUFFIX, T)                                     \
    void launch_fast_block_diag_fwd_##SUFFIX(const void* input, void* output, int z,     \
                                             int N, int b, hipStream_t stream) {          \
        const int n = z * N * b * b;                                                     \
        const dim3 blocks(DIV_CEIL(n, kThreads));                                        \
        hipLaunchKernelGGL((fast_block_diag_fwd_kernel<T>), blocks, dim3(kThreads), 0,    \
                           stream, static_cast<const T*>(input),                         \
                           static_cast<T*>(output), z, N, b);                            \
    }                                                                                    \
    void launch_fast_block_diag_bwd_##SUFFIX(const void* grad_output, void* grad_input,  \
                                             int z, int N, int b, hipStream_t stream) {   \
        const int n = z * N * b * b;                                                     \
        const dim3 blocks(DIV_CEIL(n, kThreads));                                        \
        hipLaunchKernelGGL((fast_block_diag_bwd_kernel<T>), blocks, dim3(kThreads), 0,    \
                           stream, static_cast<const T*>(grad_output),                    \
                           static_cast<T*>(grad_input), z, N, b);                        \
    }

DEFINE_FAST_BLOCK_DIAG_LAUNCHERS(fp16, __half)
DEFINE_FAST_BLOCK_DIAG_LAUNCHERS(bf16, hip_bfloat16)
DEFINE_FAST_BLOCK_DIAG_LAUNCHERS(fp32, float)
DEFINE_FAST_BLOCK_DIAG_LAUNCHERS(fp64, double)

#undef DEFINE_FAST_BLOCK_DIAG_LAUNCHERS
