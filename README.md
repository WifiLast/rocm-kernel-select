# rocm-kernel-select

![Logo](Logo.jpeg)

A PyTorch extension for AMD ROCm GPUs that selects the fastest available kernel for a given operation, instead of assuming one backend is always best.

## Overview

Several kernel sources can implement the same PyTorch operation on ROCm: stock ROCm (rocBLAS/hipBLASLt/MIOpen), [aiter](https://github.com/ROCm/aiter), [Composable Kernel](https://github.com/ROCm/composable_kernel), [TransformerEngine's ROCm fork](https://github.com/ROCm/TransformerEngine), and this project's own hand-written HIP kernels. No single source wins for every operation, dtype, and shape.

`amd_tuned_torch` monkeypatches a small set of `torch` / `torch.nn.functional` ops. On the first call for a given shape, it benchmarks the available candidates for that op — including stock — and caches the winner. Later calls for the same shape dispatch straight to the cached choice.

## Patched operations

- `F.linear`, `torch.matmul`, `torch.bmm`
- `F.conv2d`, `F.conv3d`
- `F.group_norm`
- `F.scaled_dot_product_attention`, `F.rms_norm`, `F.gelu`, `F.silu` (optional, TransformerEngine-backed)

Operations not listed above are left on stock PyTorch.

## Benchmarks

Measured with `tools/bench_fast.py` on an AMD Radeon RX 7900 XTX (ROCm 7.2.53211, torch 2.15.0.dev20260820+rocm7.2). Each op is timed with the same call site, alternating `amd_tuned_torch.disable()` / `.enable()`, taking the min over 3 rounds. Full raw results in [`bench.json`](bench.json).

| Op | Shape | Dtype | Stock (ms) | Patched (ms) | Speedup |
|---|---|---|---|---|---|
| linear | 2048x2048 @ 2048x2048.T | float16 | 0.679 | 0.277 | 2.45x |
| linear | 2048x2048 @ 2048x2048.T | bfloat16 | 0.665 | 0.276 | 2.41x |
| linear | 2048x2048 @ 2048x2048.T | float32 | 0.690 | 0.673 | 1.03x |
| matmul | 2048x2048 @ 2048x2048 | float16 | 0.188 | 0.190 | 0.99x |
| matmul | 2048x2048 @ 2048x2048 | float32 | 0.688 | 0.687 | 1.00x |
| bmm | batch=8, 512x512 @ 512x512 | float16 | 0.025 | 0.031 | 0.78x |
| bmm | batch=8, 512x512 @ 512x512 | float32 | 0.024 | 0.026 | 0.93x |
| conv2d | N=16, C=64->64, 64x64, k=3 | float16 | 0.089 | 0.090 | 0.99x |
| conv2d | N=16, C=64->64, 64x64, k=3 | float32 | 0.123 | 0.122 | 1.01x |
| conv3d | N=1, C=64->64, 16x32x32, k=3 | float16 | 0.443 | 0.094 | 4.72x |
| conv3d | N=1, C=64->64, 16x32x32, k=3 | float32 | 0.717 | 0.289 | 2.48x |
| conv1d (depthwise) | N=8, C=256, L=2048, k=3, groups=256 | float16 | 0.863 | 0.051 | 16.81x |
| conv1d (depthwise) | N=8, C=256, L=2048, k=3, groups=256 | float32 | 0.466 | 0.050 | 9.24x |
| group_norm | N=16, C=128, 64x64, groups=32 | float16 | 0.059 | 0.058 | 1.02x |
| group_norm | N=16, C=128, 64x64, groups=32 | float32 | 0.062 | 0.065 | 0.95x |
| sdpa (causal) | B=2, H=16, S=1024, D=64 | float16 | 0.128 | 0.124 | 1.03x |

Speedups vary by shape and available backends (`aiter` and TransformerEngine were not installed for this run — see `backends_available` in `bench.json`). When no candidate beats stock, the dispatcher falls back to stock so patched performance is never worse by more than benchmarking noise.

### conv3d parameter sweep

`tools/bench_conv3d_sweep.py` sweeps `F.conv3d` across kernel size, spatial extent, channel count, batch size, anisotropic shapes, stride/padding, dtype, and asymmetric channels (38 cases). Full per-case detail (including which backend won each shape) is in [`bench_dim.json`](bench_dim.json).

| | |
|---|---|
| Cases timed | 38 (0 failed) |
| Winner breakdown | ck: 30, fftconv: 3, native: 3, stock: 2 |
| Faster than stock | 27 |
| Slower than stock | 6 |
| Best speedup | 11.65x — `large_kernel`, N=1, C=32→32, 32x32x32, k=15, float16 |
| Worst speedup | 0.52x — `dtype`, N=1, C=64→64, 16x16x16, k=3, float32 |

The `ck` (Composable Kernel) backend wins most cases, especially large kernels and channel-heavy shapes. `stock` and plain `native` occasionally win on small/float32 cases where dispatch overhead outweighs the kernel gain — the selector caches per-shape so this only costs one extra benchmark call the first time a shape is seen.

<details>
<summary>All 38 cases (from <code>bench_dim.json</code>)</summary>

| Group | Shape | Dtype | Winner | Stock (ms) | Patched (ms) | Speedup |
|---|---|---|---|---|---|---|
| kernel_size | N=1, C=16→16, 16x16x16, k=1, s=1, p=0 | float16 | ck | 0.062 | 0.065 | 0.96x |
| kernel_size | N=1, C=16→16, 16x16x16, k=1, s=1, p=0 | float32 | native | 0.179 | 0.215 | 0.83x |
| kernel_size | N=1, C=16→16, 16x16x16, k=3, s=1, p=1 | float16 | ck | 0.154 | 0.154 | 1.00x |
| kernel_size | N=1, C=16→16, 16x16x16, k=3, s=1, p=1 | float32 | native | 0.161 | 0.139 | 1.16x |
| kernel_size | N=1, C=16→16, 16x16x16, k=5, s=1, p=2 | float16 | ck | 0.159 | 0.145 | 1.09x |
| kernel_size | N=1, C=16→16, 16x16x16, k=5, s=1, p=2 | float32 | stock | 0.217 | 0.265 | 0.82x |
| kernel_size | N=1, C=16→16, 16x16x16, k=7, s=1, p=3 | float16 | ck | 0.377 | 0.075 | 5.06x |
| kernel_size | N=1, C=16→16, 16x16x16, k=7, s=1, p=3 | float32 | stock | 0.441 | 0.446 | 0.99x |
| kernel_size | N=1, C=16→16, 16x16x16, k=9, s=1, p=4 | float16 | ck | 0.727 | 0.147 | 4.95x |
| kernel_size | N=1, C=16→16, 16x16x16, k=9, s=1, p=4 | float32 | fftconv | 2.912 | 0.926 | 3.15x |
| kernel_size | N=1, C=16→16, 16x16x16, k=15, s=1, p=7 | float16 | ck | 3.198 | 0.542 | 5.90x |
| kernel_size | N=1, C=16→16, 16x16x16, k=15, s=1, p=7 | float32 | fftconv | 8.606 | 0.866 | 9.94x |
| spatial | N=1, C=64→64, 8x8x8, k=3, s=1, p=1 | float16 | ck | 0.172 | 0.128 | 1.34x |
| spatial | N=1, C=64→64, 16x16x16, k=3, s=1, p=1 | float16 | ck | 0.198 | 0.149 | 1.33x |
| spatial | N=1, C=64→64, 32x32x32, k=3, s=1, p=1 | float16 | ck | 0.908 | 0.190 | 4.77x |
| spatial | N=1, C=64→64, 48x48x48, k=3, s=1, p=1 | float16 | ck | 3.201 | 0.548 | 5.84x |
| channels | N=1, C=32→32, 16x16x16, k=3, s=1, p=1 | float16 | ck | 0.072 | 0.071 | 1.01x |
| channels | N=1, C=64→64, 16x16x16, k=3, s=1, p=1 | float16 | ck | 0.125 | 0.073 | 1.72x |
| channels | N=1, C=128→128, 16x16x16, k=3, s=1, p=1 | float16 | ck | 0.251 | 0.087 | 2.88x |
| channels | N=1, C=256→256, 16x16x16, k=3, s=1, p=1 | float16 | ck | 0.620 | 0.252 | 2.46x |
| batch | N=1, C=64→64, 16x16x16, k=3, s=1, p=1 | float16 | ck | 0.125 | 0.074 | 1.68x |
| batch | N=2, C=64→64, 16x16x16, k=3, s=1, p=1 | float16 | ck | 0.267 | 0.141 | 1.89x |
| batch | N=4, C=64→64, 16x16x16, k=3, s=1, p=1 | float16 | ck | 0.509 | 0.093 | 5.49x |
| batch | N=8, C=64→64, 16x16x16, k=3, s=1, p=1 | float16 | ck | 0.989 | 0.156 | 6.35x |
| anisotropic | N=2, C=32→32, 8x64x64, k=3, s=1, p=1 | float16 | ck | 0.891 | 0.129 | 6.93x |
| anisotropic | N=1, C=128→128, 4x16x16, k=3, s=1, p=1 | float16 | ck | 0.152 | 0.163 | 0.93x |
| anisotropic | N=1, C=64→64, 32x32x8, k=3, s=1, p=1 | float16 | ck | 0.247 | 0.146 | 1.70x |
| stride_padding | N=1, C=64→64, 16x16x16, k=3, s=2, p=1 | float16 | ck | 0.158 | 0.186 | 0.85x |
| stride_padding | N=1, C=64→64, 16x16x16, k=3, s=1, p=0 | float16 | ck | 0.185 | 0.144 | 1.28x |
| stride_padding | N=1, C=64→64, 16x16x16, k=1, s=1, p=0 | float16 | ck | 0.173 | 0.183 | 0.95x |
| stride_padding | N=1, C=64→64, 32x32x32, k=3, s=2, p=1 | float16 | ck | 0.176 | 0.159 | 1.11x |
| dtype | N=1, C=64→64, 16x16x16, k=3, s=1, p=1 | float16 | ck | 0.131 | 0.076 | 1.73x |
| dtype | N=1, C=64→64, 16x16x16, k=3, s=1, p=1 | bfloat16 | ck | 0.200 | 0.146 | 1.37x |
| dtype | N=1, C=64→64, 16x16x16, k=3, s=1, p=1 | float32 | native | 0.225 | 0.437 | 0.52x |
| channels_asym | N=1, C=32→128, 16x16x16, k=3, s=1, p=1 | float16 | ck | 0.075 | 0.072 | 1.04x |
| channels_asym | N=1, C=128→32, 16x16x16, k=3, s=1, p=1 | float16 | ck | 0.235 | 0.072 | 3.27x |
| large_kernel | N=1, C=32→32, 32x32x32, k=15, s=1, p=7 | float16 | ck | 57.797 | 4.962 | 11.65x |
| large_kernel | N=1, C=32→32, 32x32x32, k=15, s=1, p=7 | float32 | fftconv | 82.064 | 15.281 | 5.37x |

</details>

## Installation

Linux only (ROCm is Linux-first). Requires a ROCm PyTorch build for your target GPU.

```bash
pip install -e . --no-build-isolation
```

See `Build prerequisites` in the source docs for optional dependencies (aiter, Composable Kernel, TransformerEngine).

## Usage

```python
import amd_tuned_torch   # patches torch / torch.nn.functional on import

amd_tuned_torch.disable()
amd_tuned_torch.enable()
```

Set `AMD_TUNED_TORCH_AUTOPATCH=0` to import without patching.

## Testing

```bash
python -m pytest tests
```

Tests cover dispatch logic only (eligibility checks, fallback behavior) with the native extension, aiter, and TransformerEngine mocked out — no GPU required.

## License

MIT
