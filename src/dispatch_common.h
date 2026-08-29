// Shared shape-key types for runtime kernel-variant dispatch, used by
// main_rocm.cpp to pick which compiled tile-shape variant of conv2d_fp16/
// conv3d_fp16 (src/cuda/generated/conv{2,3}d_fp16_*.cu, see
// tools/kernelgen/) to run for a given input shape, and to cache that
// choice per distinct shape for the process lifetime.
//
// ShapeKeyHash is the same FNV-offset-basis + golden-ratio-mix hash this
// file's Conv3dShapeKey used before it lived here (originally in
// main_rocm.cpp, only ever used for conv3d_fp32's direct-vs-Winograd
// choice) -- generalized to any std::array<int64_t, N> so conv2d and
// conv3d share one hash implementation instead of two copies.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

template <std::size_t N>
struct ShapeKeyHash {
    std::size_t operator()(const std::array<int64_t, N>& k) const {
        std::size_t h = 1469598103934665603ull;  // FNV-1a 64-bit offset basis
        for (int64_t v : k) {
            h ^= (std::size_t)v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        }
        return h;
    }
};

// B, C_in, H_in, W_in, C_out, K_H, K_W, H_out, W_out, s_h, s_w, p_h, p_w, d_h, d_w
using Conv2dShapeKey     = std::array<int64_t, 15>;
using Conv2dShapeKeyHash = ShapeKeyHash<15>;

// B, C_in, D_in, H_in, W_in, C_out, K_D, K_H, K_W, D_out, H_out, W_out,
// s_d, s_h, s_w, p_d, p_h, p_w, d_d, d_h, d_w
using Conv3dShapeKey     = std::array<int64_t, 21>;
using Conv3dShapeKeyHash = ShapeKeyHash<21>;
