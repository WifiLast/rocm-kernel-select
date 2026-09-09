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
