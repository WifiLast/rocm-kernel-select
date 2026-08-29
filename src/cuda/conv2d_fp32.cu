// Hand-written direct Conv2d (register-blocked, FP32) for amd_tuned_torch on
// ROCm/RDNA3 (RX 7900 XTX, gfx1100). Ported from the original CMP-Turing
// project's CUDA kernel (src/cuda/kernel_example/fp32_conv.cu) -- plain
// scalar FMA, no shared-memory tiling and no tensor cores, so the port is a
// straight syntax translation (CUDA launch -> hipLaunchKernelGGL) with no
// change to the compute itself.
//
// Thread coarsening: each thread computes 2 W-adjacent output pixels x CTILE
// output channels (accumulators sum0/sum1, CTILE registers each). Pointer
// hoisting: CTILE weight pointers are advanced with ++ once per (kh, kw)
// step instead of recomputed via multiplication, so each loaded weight
// value is reused across both pixels and the address arithmetic leaves the
// hot loop.
//
// Overridable via -DCONV2D_FP32_CTILE=.. at build time (see ../../setup.py).
#include <hip/hip_runtime.h>

#define DIV_CEIL(a, b) (((a) + (b) - 1) / (b))

#ifndef CONV2D_FP32_CTILE
#define CONV2D_FP32_CTILE 8
#endif
#define CTILE CONV2D_FP32_CTILE

namespace {

__global__ void __launch_bounds__(256) conv2d_fp32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C_in, int H_in, int W_in,
    int C_out, int K_H, int K_W,
    int H_out, int W_out,
    int s_h, int s_w, int p_h, int p_w, int d_h, int d_w
) {
    int tid_x = threadIdx.x;  // 0..15
    int tid_y = threadIdx.y;  // 0..15

    // Each block covers 16 rows x 32 cols of output (16 threads * 2 pixels/thread).
    int blocks_in_row = DIV_CEIL(W_out, 32);
    int blk_w = blockIdx.x % blocks_in_row;
    int blk_h = blockIdx.x / blocks_in_row;

    // Thread t handles w = 2t, 2t+1 (horizontally adjacent, so the weight
    // value loaded per (kh, kw, c_out) step is reused across both).
    int w_out_0 = blk_w * 32 + tid_x * 2;
    int w_out_1 = w_out_0 + 1;
    int h_out = blk_h * 16 + tid_y;
    int c_out_base = blockIdx.y * CTILE;
    int b_idx = blockIdx.z;

    if (h_out >= H_out) return;

    bool valid_w0 = (w_out_0 < W_out);
    bool valid_w1 = (w_out_1 < W_out);
    if (!valid_w0 && !valid_w1) return;

    float sum0[CTILE] = {0.0f};
    float sum1[CTILE] = {0.0f};

    int h_in_base = h_out * s_h - p_h;
    int w_in_base_0 = w_out_0 * s_w - p_w;
    int w_in_base_1 = w_out_1 * s_w - p_w;

    long long input_batch_offset = (long long)b_idx * C_in * H_in * W_in;
    const float* input_base_ptr = input + input_batch_offset;

    // Precompute CTILE output-channel weight start pointers; out-of-range
    // channels point at weight[0] to avoid an illegal address (the
    // write-back stage filters them out via current_c_out).
    const float* w_ptrs[CTILE];
    int weight_stride_oc = C_in * K_H * K_W;

    #pragma unroll
    for (int k = 0; k < CTILE; ++k) {
        int c = c_out_base + k;
        w_ptrs[k] = (c < C_out) ? (weight + (long long)c * weight_stride_oc) : weight;
    }

    for (int c = 0; c < C_in; ++c) {
        const float* current_in_channel = input_base_ptr + (long long)c * H_in * W_in;

        for (int i = 0; i < K_H; ++i) {
            int in_row = h_in_base + i * d_h;
            bool row_valid = (in_row >= 0 && in_row < H_in);
            long long row_offset = row_valid ? (long long)in_row * W_in : 0;

            for (int j = 0; j < K_W; ++j) {
                float in_val0 = 0.0f;
                float in_val1 = 0.0f;

                if (row_valid) {
                    int in_col0 = w_in_base_0 + j * d_w;
                    int in_col1 = w_in_base_1 + j * d_w;

                    if (valid_w0 && in_col0 >= 0 && in_col0 < W_in) {
                        in_val0 = current_in_channel[row_offset + in_col0];
                    }
                    if (valid_w1 && in_col1 >= 0 && in_col1 < W_in) {
                        in_val1 = current_in_channel[row_offset + in_col1];
                    }
                }

                // Weight pointers must advance exactly once per (kh, kw)
                // step regardless of row_valid, to stay in sync with the
                // flattened weight layout.
                #pragma unroll
                for (int k = 0; k < CTILE; ++k) {
                    float w_val = *w_ptrs[k];
                    w_ptrs[k]++;

                    sum0[k] += in_val0 * w_val;
                    sum1[k] += in_val1 * w_val;
                }
            }
        }
    }

    long long out_batch_offset = (long long)b_idx * C_out * H_out * W_out;
    long long total_pixels = (long long)H_out * W_out;

    auto write_back = [&](int w_curr, float* acc, bool w_valid) {
        if (!w_valid) return;
        long long out_spatial = (long long)h_out * W_out + w_curr;

        #pragma unroll
        for (int k = 0; k < CTILE; ++k) {
            int current_c_out = c_out_base + k;
            if (current_c_out < C_out) {
                float val = acc[k];
                if (bias) val += bias[current_c_out];
                long long out_addr = out_batch_offset + (long long)current_c_out * total_pixels + out_spatial;
                output[out_addr] = val;
            }
        }
    };

    write_back(w_out_0, sum0, valid_w0);
    write_back(w_out_1, sum1, valid_w1);
}

}  // namespace

void launch_conv2d_fp32(
    const float* input, const float* weight, const float* bias, float* output,
    int B, int C_in, int H_in, int W_in,
    int C_out, int K_H, int K_W,
    int H_out, int W_out,
    int s_h, int s_w, int p_h, int p_w, int d_h, int d_w,
    hipStream_t stream
) {
    dim3 threads_per_block(16, 16);

    int blocks_w = DIV_CEIL(W_out, 32);
    int blocks_h = DIV_CEIL(H_out, 16);
    int grid_x = blocks_w * blocks_h;
    int grid_y = DIV_CEIL(C_out, CTILE);
    int grid_z = B;

    dim3 blocks(grid_x, grid_y, grid_z);

    hipLaunchKernelGGL((conv2d_fp32_kernel), blocks, threads_per_block, 0, stream,
                        input, weight, bias, output,
                        B, C_in, H_in, W_in, C_out, K_H, K_W, H_out, W_out,
                        s_h, s_w, p_h, p_w, d_h, d_w);
}
