// Launcher declarations for src/cuda/iu4_gemm_fwd.cu -- see that file's
// header for what these kernels are and amd_tuned_torch/_vendor/
// gfx1100_iu4_gemm/NOTICE.md for where they came from and why this tier is
// not wired into kernel_select the way every other GEMM candidate is.
#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

// A_packed: [M, ceil(K,16)/2] bytes, two signed 4-bit nibbles per byte
// (low nibble = even k, high nibble = odd k -- see pack_i4 in
// amd_tuned_torch/iu4_gemm_ops.py). B_packed: same layout, [N, ceil(K,16)/2].
// C: [M, N] int32, exact (int4 x int4 -> int32 has no rounding).
// a_stride_bytes/b_stride_bytes are the actual per-row byte strides of the
// packed buffers (>= ceil(K,16)/2; the tail path reads nibbles directly so
// any stride at least that large works, matching the upstream probe).
void launch_iu4_gemm(const uint8_t* a_packed, const uint8_t* b_packed, int32_t* c,
                      int m, int n, int k, int a_stride_bytes, int b_stride_bytes,
                      hipStream_t stream);

// A/B: [M, K]/[N, K] signed int8, row-major, contiguous per row.
// C: [M, N] int32, exact.
void launch_iu8_gemm(const int8_t* a, const int8_t* b, int32_t* c,
                      int m, int n, int k, int a_stride_bytes, int b_stride_bytes,
                      hipStream_t stream);

// Same contract as launch_iu8_gemm, using v_dot4_i32_i8 instead of WMMA --
// the small/tail fallback control, not expected to win at any size that
// fills a WMMA tile.
void launch_dot4_i8_gemm(const int8_t* a, const int8_t* b, int32_t* c,
                          int m, int n, int k, int a_stride_bytes, int b_stride_bytes,
                          hipStream_t stream);
