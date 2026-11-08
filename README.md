# rocm-kernel-select

![Logo](Logo.jpeg)

A PyTorch extension for AMD ROCm GPUs that picks the fastest kernel per op and shape instead of assuming one backend always wins.

`import amd_tuned_torch` monkeypatches a set of `torch` ops. The first call for a new shape benchmarks every available candidate (always including stock) and caches the winner; later calls dispatch straight to it. If nothing beats stock, stock is used.

Candidate sources: stock ROCm (rocBLAS / hipBLASLt / MIOpen), [aiter](https://github.com/ROCm/aiter), [Composable Kernel](https://github.com/ROCm/composable_kernel), [TransformerEngine (ROCm)](https://github.com/ROCm/TransformerEngine), FFT-conv, comfy-kitchen HIP kernels, and this project's own HIP kernels. Missing backends are skipped.

Developed and tuned on an RX 7900 XTX (gfx1100).

## Patched operations

| Op | Backends contested |
|---|---|
| `F.linear`, `torch.matmul`, `torch.bmm`, `torch.einsum` | aiter, hipBLASLt, CK GEMM, stock |
| `F.conv2d`, `F.conv3d` | native HIP, CK, FFT-conv (large kernels), aiter (bf16), stock |
| `F.conv1d` (depthwise) | HIP port of FlashFFTConv conv1d, stock |
| `F.group_norm` | native HIP, stock |
| `F.layer_norm`, `F.rms_norm` | TransformerEngine / comfy-kitchen (tall training inputs), stock |
| `F.scaled_dot_product_attention` | TransformerEngine (large masked training calls), stock |
| `F.dropout`, Adam/AdamW foreach step, grad-norm | TransformerEngine, stock |
| `nn.Sequential` conv blocks | Triton conv + epilogue fusion |

Opt-in extras: `F.gelu`/`F.silu` via TE (`AMD_TUNED_TORCH_TE_ACTIVATIONS=1`), rocWMMA flash attention, BOFT, ZLUDA cuDNN, block-scaled int8 ops, VRAM management helpers. `fastvideo_kernel_amd/` is a ROCm port of fastvideo-kernel.

TransformerEngine routes are off by default; set `AMD_TUNED_TORCH_ENABLE_TE=1` once `import transformer_engine.pytorch` works on its own (a mismatched TE build can segfault on import). Anything not listed stays on stock PyTorch.

## Benchmarks

RX 7900 XTX, ROCm 7.2, `tools/bench_fast.py`, min of 3 rounds. Raw data: [`bench.json`](bench.json).

| Op | Shape | Dtype | Stock (ms) | Patched (ms) | Speedup |
|---|---|---|---|---|---|
| linear | 2048x2048 @ 2048x2048.T | float16 | 0.679 | 0.277 | 2.45x |
| linear | 2048x2048 @ 2048x2048.T | bfloat16 | 0.665 | 0.276 | 2.41x |
| conv3d | N=1, C=64→64, 16x32x32, k=3 | float16 | 0.443 | 0.094 | 4.72x |
| conv3d | N=1, C=64→64, 16x32x32, k=3 | float32 | 0.717 | 0.289 | 2.48x |
| conv1d (depthwise) | N=8, C=256, L=2048, k=3 | float16 | 0.863 | 0.051 | 16.81x |
| conv1d (depthwise) | N=8, C=256, L=2048, k=3 | float32 | 0.466 | 0.050 | 9.24x |
| matmul / bmm / conv2d / group_norm / sdpa | various | various | — | — | 0.78x–1.03x |

The conv3d sweep (`tools/bench_conv3d_sweep.py`, 38 cases, [`bench_dim.json`](bench_dim.json)) beat stock in 27 cases, with CK winning most. Best: 11.65x (k=15, fp16); worst: 0.52x (small fp32).

More benchmark logs are in `analyse/benchmarks/`.

## Installation

Linux + a ROCm build of PyTorch.

```bash
pip install -e . --no-build-isolation
```

Or use `install_prebuilt.sh`. aiter, Composable Kernel and TransformerEngine are optional.

## Usage

```python
import amd_tuned_torch   # patches torch on import

amd_tuned_torch.disable()
amd_tuned_torch.enable()
```

Useful environment variables:

| Variable | Effect |
|---|---|
| `AMD_TUNED_TORCH_AUTOPATCH=0` | import without patching |
| `AMD_TUNED_TORCH_ENABLE_TE=1` | enable the TransformerEngine routes |
| `AMD_TUNED_TORCH_DISPATCH_DEBUG=1` | log which backend each call takes |
| `AMD_TUNED_TORCH_CONV_FUSION=0` | turn off conv-block fusion |
| `AMD_TUNED_TORCH_FFTCONV2D=0` / `..._FFTCONV3D=0` | turn off FFT-conv |

## Testing

```bash
python -m pytest tests            # dispatch logic, backends mocked, no GPU needed
python -m pytest tests_hardware   # real kernels, needs a ROCm GPU
```

## License

MIT
