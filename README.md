# rocm-kernel-select

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
