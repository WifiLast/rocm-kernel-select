# Provenance

`conv1d_bhl.cu`, `conv1d_blh.cu`, `conv1d_bwd_cuda_bhl.cu`,
`conv1d_bwd_cuda_blh.cu`, `shared.h`, `conv1d.h`, and `LICENSE` in this
directory are vendored from:

- Upstream: https://github.com/HazyResearch/flash-fft-conv
- Path: `csrc/flashfftconv/conv1d/`
- License: Apache License 2.0 (see `LICENSE` in this directory)

## What was changed from upstream

Apart from a short attribution header comment added at the top of each
file pointing back here, two small edits were needed to make hipify's
output actually compile — both are no-ops on CUDA:

- `shared.h`: added explicit `#include <cuda_fp16.h>` and
  `#include <cuda_bf16.h>`. Upstream gets these implicitly on CUDA; hipify
  rewrites them to `hip/hip_fp16.h` and `hip/hip_bf16.h`, and without the
  latter `hip_runtime.h` supplies only the legacy `hip_bfloat16` struct, so
  every `__hip_bfloat16`/`__hip_bfloat162` use in the `DISPATCH_*` macro and
  the `set_value()` overloads failed with "unknown type name".
- `conv1d_blh.cu`: upstream's `__float2bfloat162_rn(0.0f)` (CUDA's
  scalar-broadcast overload) has no HIP equivalent and no hipify mapping;
  replaced with `__float22bfloat162_rn(make_float2(0.0f, 0.0f))`, which
  exists identically in `cuda_bf16.h` and `hip_bf16.h`.
- `conv1d_blh.cu`: `_conv1d_k_3()`'s return type changed from `T` to `void`.
  Upstream declares it `T` but never returns a value (it writes through its
  `out` argument, and the caller discards the result). nvcc tolerates that;
  amdclang warns `-Wreturn-type` and then, at `-O3`, treats falling off the
  end of a non-void function as unreachable and deletes the entire body --
  so the `k == 3` BLH forward path silently returned all zeros. Only the
  BLH file is affected; `conv1d_bhl.cu`'s same-named helper does return.

## Why this is vendored separately from `third_party/FlashFFTConv/`

The rest of FlashFFTConv's `csrc/flashfftconv/` (its `monarch_cuda/` and
`butterfly/` kernels — the actual Monarch-matrix FFT-conv algorithm) is
implemented entirely with `nvcuda::wmma` tensor-core intrinsics and
required a hand rewrite against rocWMMA to build for ROCm at all (see
`third_party/FlashFFTConv/csrc/flashfftconv/setup.py` and
`amd_tuned_torch/flashfftconv_ops.py`'s docstring). This `conv1d/`
subdirectory is a different, much smaller piece of that same upstream
repo: a plain depthwise (`groups == in_channels == out_channels`) 1D
convolution with dilation/stride fixed at 1, arbitrary odd kernel width,
symmetric padding, and independently-choosable input/weight dtypes
(fp16/bf16/fp32 in any combination) — implemented in ordinary CUDA with
**zero** WMMA/tensor-core/`cub::`/cooperative-groups/inline-PTX usage
(confirmed by grep before vendoring). It builds via PyTorch's ordinary
hipify step exactly like this package's own `src/cuda/*.cu` sources, with
no dependency on FlashFFTConv's `monarch_cuda` extension, its rocWMMA port,
or anything else under `third_party/FlashFFTConv/` — hence vendoring it
here as its own independent unit rather than pulling in the whole
FlashFFTConv package for what is, numerically, a small standalone kernel.

## What this gives amd_tuned_torch that it didn't already have

`amd_tuned_torch.miopen_fallback` already has a depthwise-conv1d fast path
(`causal_conv1d`, from `source/cmp_ext_turing/src/causal-conv1d-amd`) but
it only covers the narrow *causal* idiom: kernel width 2–4, and
`padding == width - 1` specifically (the "pad left, trim right" pattern
Mamba/Hyena/TCN blocks use). This kernel covers the general case instead:
any odd kernel width, any symmetric padding value, output length matching
stock `F.conv1d`'s formula exactly (no truncation caveat the way
`causal_conv1d_fn`'s output has) — so unlike `causal_conv1d`, which is
wired in via a structural eligibility match, this one produces the
*same numerical answer* as stock depthwise conv1d and can be entered
directly into `kernel_select`'s measure-and-cache contest, the same way
`fftconv_ops.fftconv1d_candidate` already is. See
`amd_tuned_torch/depthwise_conv1d_ops.py` and
`amd_tuned_torch/miopen_fallback.py`'s "DEPTHWISE CONV1D FAST PATH" section
for the wiring.
