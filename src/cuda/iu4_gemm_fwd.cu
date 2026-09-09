// Native gfx1100 WMMA integer GEMM -- packed-INT4 (iu4_gemm_candidate),
// INT8 WMMA (iu8_gemm_control), and a DOT4 INT8 fallback (dot4_i8_control).
// Ported from an experimental correctness/roofline probe, NOT a production
// kernel -- see amd_tuned_torch/_vendor/gfx1100_iu4_gemm/NOTICE.md for
// exactly what was and wasn't carried over, and iu4_gemm_ops.py for why
// this tier is opt-in only and never enters kernel_select's contest.
//
// Each kernel computes C[M,N] = A[M,K] * B[N,K]^T with exact INT32
// accumulation (no dequant, no bias, no epilogue -- that's
// iu4_gemm_ops.py's job on the Python side, same division of labor as the
// upstream probe's separate quantize/pack step). One 16x16 output tile per
// 32-lane wavefront-quarter (wave32 WMMA): `lane & 15` selects the A row
// and B/output column, each lane contributes all 16 K values per WMMA
// call, and accumulator register `i` maps to output row `2*i + (lane>>4)`.
//
// direct global-memory fragment loads, no LDS staging -- see this file's
// own NOTICE.md entry for what that costs relative to a tuned production
// GEMM.
#include "iu4_gemm_fwd.hpp"

#include <cstdint>

using v2i32 = int __attribute__((ext_vector_type(2)));
using v4i32 = int __attribute__((ext_vector_type(4)));
using v8i32 = int __attribute__((ext_vector_type(8)));

static constexpr int kTile = 16;

static __device__ __forceinline__ v8i32 wmma_i4_signed(v2i32 a, v2i32 b, v8i32 c) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c, false);
}

static __device__ __forceinline__ v8i32 wmma_i8_signed(v4i32 a, v4i32 b, v8i32 c) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
}

static __device__ __forceinline__ int dot4_i8_signed(int a, int b, int c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

static __device__ __forceinline__ int load_i4_nibble(const uint8_t* row, int k) {
    return (row[k >> 1] >> (4 * (k & 1))) & 0x0f;
}

static __device__ __forceinline__ int pack_i4_tail_word(
        const uint8_t* row, int k0, int valid_k) {
    uint32_t word = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int k = k0 + i;
        const uint32_t nibble = k < valid_k ? uint32_t(load_i4_nibble(row, k)) : 0u;
        word |= nibble << (4 * i);
    }
    return int(word);
}

static __device__ __forceinline__ int pack_i8_tail_word(
        const int8_t* row, int k0, int valid_k) {
    uint32_t word = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int k = k0 + i;
        const uint32_t byte = k < valid_k ? uint32_t(uint8_t(row[k])) : 0u;
        word |= byte << (8 * i);
    }
    return int(word);
}

extern "C" __global__ __launch_bounds__(32) void iu4_gemm_candidate(
        const uint8_t* __restrict__ a,
        const uint8_t* __restrict__ b,
        int32_t* __restrict__ c,
        int m,
        int n,
        int k,
        int a_stride_bytes,
        int b_stride_bytes) {
    const int lane = int(threadIdx.x);
    const int lane_lo = lane & 15;
    const int lane_hi = lane >> 4;
    const int tile_m = int(blockIdx.y) * kTile;
    const int tile_n = int(blockIdx.x) * kTile;
    const int a_row = tile_m + lane_lo;
    const int b_row = tile_n + lane_lo;

    v8i32 acc = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int k0 = 0; k0 < k; k0 += kTile) {
        v2i32 a_frag = {0, 0};
        v2i32 b_frag = {0, 0};
        if (a_row < m) {
            const uint8_t* row = a + size_t(a_row) * size_t(a_stride_bytes);
            if (k0 + kTile <= k) {
                const uint32_t* words = reinterpret_cast<const uint32_t*>(row + (k0 >> 1));
                a_frag[0] = int(words[0]);
                a_frag[1] = int(words[1]);
            } else {
                a_frag[0] = pack_i4_tail_word(row, k0, k);
                a_frag[1] = pack_i4_tail_word(row, k0 + 8, k);
            }
        }
        if (b_row < n) {
            const uint8_t* row = b + size_t(b_row) * size_t(b_stride_bytes);
            if (k0 + kTile <= k) {
                const uint32_t* words = reinterpret_cast<const uint32_t*>(row + (k0 >> 1));
                b_frag[0] = int(words[0]);
                b_frag[1] = int(words[1]);
            } else {
                b_frag[0] = pack_i4_tail_word(row, k0, k);
                b_frag[1] = pack_i4_tail_word(row, k0 + 8, k);
            }
        }
        acc = wmma_i4_signed(a_frag, b_frag, acc);
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = tile_m + 2 * i + lane_hi;
        const int col = tile_n + lane_lo;
        if (row < m && col < n) {
            c[size_t(row) * size_t(n) + size_t(col)] = acc[i];
        }
    }
}

extern "C" __global__ __launch_bounds__(32) void iu8_gemm_control(
        const int8_t* __restrict__ a,
        const int8_t* __restrict__ b,
        int32_t* __restrict__ c,
        int m,
        int n,
        int k,
        int a_stride_bytes,
        int b_stride_bytes) {
    const int lane = int(threadIdx.x);
    const int lane_lo = lane & 15;
    const int lane_hi = lane >> 4;
    const int tile_m = int(blockIdx.y) * kTile;
    const int tile_n = int(blockIdx.x) * kTile;
    const int a_row = tile_m + lane_lo;
    const int b_row = tile_n + lane_lo;

    v8i32 acc = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int k0 = 0; k0 < k; k0 += kTile) {
        v4i32 a_frag = {0, 0, 0, 0};
        v4i32 b_frag = {0, 0, 0, 0};
        if (a_row < m) {
            const int8_t* row = a + size_t(a_row) * size_t(a_stride_bytes);
#pragma unroll
            for (int word = 0; word < 4; ++word) {
                if (k0 + kTile <= k) {
                    a_frag[word] = reinterpret_cast<const int32_t*>(row + k0)[word];
                } else {
                    a_frag[word] = pack_i8_tail_word(row, k0 + 4 * word, k);
                }
            }
        }
        if (b_row < n) {
            const int8_t* row = b + size_t(b_row) * size_t(b_stride_bytes);
#pragma unroll
            for (int word = 0; word < 4; ++word) {
                if (k0 + kTile <= k) {
                    b_frag[word] = reinterpret_cast<const int32_t*>(row + k0)[word];
                } else {
                    b_frag[word] = pack_i8_tail_word(row, k0 + 4 * word, k);
                }
            }
        }
        acc = wmma_i8_signed(a_frag, b_frag, acc);
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = tile_m + 2 * i + lane_hi;
        const int col = tile_n + lane_lo;
        if (row < m && col < n) {
            c[size_t(row) * size_t(n) + size_t(col)] = acc[i];
        }
    }
}

extern "C" __global__ void dot4_i8_control(
        const int8_t* __restrict__ a,
        const int8_t* __restrict__ b,
        int32_t* __restrict__ c,
        int m,
        int n,
        int k,
        int a_stride_bytes,
        int b_stride_bytes) {
    const int col = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    const int row = int(blockIdx.y) * int(blockDim.y) + int(threadIdx.y);
    if (row >= m || col >= n) {
        return;
    }
    const int8_t* a_row = a + size_t(row) * size_t(a_stride_bytes);
    const int8_t* b_row = b + size_t(col) * size_t(b_stride_bytes);
    int acc = 0;
    for (int k0 = 0; k0 < k; k0 += 4) {
        const int a_word = k0 + 4 <= k
            ? *reinterpret_cast<const int32_t*>(a_row + k0)
            : pack_i8_tail_word(a_row, k0, k);
        const int b_word = k0 + 4 <= k
            ? *reinterpret_cast<const int32_t*>(b_row + k0)
            : pack_i8_tail_word(b_row, k0, k);
        acc = dot4_i8_signed(a_word, b_word, acc);
    }
    c[size_t(row) * size_t(n) + size_t(col)] = acc;
}

static int round_up(int value, int multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

static dim3 grid_for(int m, int n) {
    return dim3(unsigned(round_up(n, kTile) / kTile), unsigned(round_up(m, kTile) / kTile), 1);
}

void launch_iu4_gemm(const uint8_t* a_packed, const uint8_t* b_packed, int32_t* c,
                      int m, int n, int k, int a_stride_bytes, int b_stride_bytes,
                      hipStream_t stream) {
    hipLaunchKernelGGL(iu4_gemm_candidate, grid_for(m, n), dim3(32), 0, stream,
                        a_packed, b_packed, c, m, n, k, a_stride_bytes, b_stride_bytes);
}

void launch_iu8_gemm(const int8_t* a, const int8_t* b, int32_t* c,
                      int m, int n, int k, int a_stride_bytes, int b_stride_bytes,
                      hipStream_t stream) {
    hipLaunchKernelGGL(iu8_gemm_control, grid_for(m, n), dim3(32), 0, stream,
                        a, b, c, m, n, k, a_stride_bytes, b_stride_bytes);
}

void launch_dot4_i8_gemm(const int8_t* a, const int8_t* b, int32_t* c,
                          int m, int n, int k, int a_stride_bytes, int b_stride_bytes,
                          hipStream_t stream) {
    const dim3 block(16, 16, 1);
    const dim3 grid(unsigned(round_up(n, int(block.x)) / int(block.x)),
                     unsigned(round_up(m, int(block.y)) / int(block.y)), 1);
    hipLaunchKernelGGL(dot4_i8_control, grid, block, 0, stream,
                        a, b, c, m, n, k, a_stride_bytes, b_stride_bytes);
}
