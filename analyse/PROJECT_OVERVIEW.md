# amd_tuned_torch - PyTorch/ComfyUI ROCm Extension (RX 7900 XTX / gfx1100)

A PyTorch/ComfyUI extension that monkeypatches a small set of
`torch`/`torch.nn.functional` ops with the most optimal kernels available on
RDNA3 (RX 7900 XTX, gfx1100), by dispatching into [aiter](../aiter) and
[TransformerEngine's ROCm fork](../TransformerEngine) instead of stock
PyTorch/ROCm. Composable Kernel is **not** a GEMM dependency -- an earlier
version linked directly against CK's `DeviceGemm` instances for the GEMM
ops; that's been replaced with aiter's Triton WMMA GEMM, which needs no
separate C++ library build/link step (see git history if you're looking for
the CK-based version). CK *is* used for **conv2d/conv3d in fp16/bf16**
(`src/cuda/ck_conv_fwd*.cu`), which is a different thing and was added for
a measured reason rather than a preference -- see "Composable Kernel conv
tier" below. That tier is header-only (CK's device ops are instantiated
here; nothing is built or linked from CK) and **optional**: without a CK
checkout the extension builds exactly as before.

## This is a different project than the Turing/CMP version

This package started as a fork of [eastmoe/cmp_ext](https://github.com/eastmoe/cmp_ext)
adapted for CMP mining cards (Turing/TU10x), where the driver enforces an
FFMA/Tensor-Core instruction-pattern throttle -- even trivial ops were slow
on those cards unless you dodged the throttle with alternate instruction
sequences written by hand in raw CUDA. **RX 7900 XTX has no such throttle.**
There is nothing to bypass. The kernels in this ROCm build exist purely for
ordinary performance reasons: pick the best available kernel per op instead
of whatever PyTorch's default ROCm backend uses, for the handful of ops that
dominate diffusion/LLM inference time. If you're looking for the original
Turing-targeted CUDA kernels, they're in this repo's git history (everything
under the old `src/cuda/` and `src/cuda-base/` was deleted when this build
was retargeted at ROCm).

## What's patched, and by what

| Op | Backed by | Why |
|---|---|---|
| `F.linear` | aiter Triton WMMA GEMM (`gemm_a16w16_`) | `amd_tuned_torch/aiter_ops.py`. Weight `[N, K]` is passed straight through -- aiter's kernel transposes it internally and computes `Y = X @ W^T`, exactly `F.linear`'s semantics, no pre-transpose/copy needed here. |
| `torch.matmul`, `torch.bmm` | aiter Triton batched GEMM (`batched_gemm_bf16`) | `amd_tuned_torch/aiter_ops.py`. That kernel computes `Y[i] = X[i] @ W[i]^T`; `mat2.transpose(-2, -1)` is passed as `W` so the internal transpose cancels out, giving plain `X[i] @ mat2[i]` (torch.bmm's actual semantics). `torch.matmul` also covers >=4D batched inputs -- e.g. attention's `(batch, heads, seq, head_dim)` Q@K^T and attn@V matmuls, which are 4D and previously fell through to stock -- by flattening the leading batch dims into one before the aiter call and reshaping back. Only when both operands share the exact same batch shape (no broadcast needed); broadcasting inputs still fall back to stock. |
| `F.conv2d` (groups=1) | fp16/bf16: Composable Kernel WMMA conv; then fp16/fp32: hand-written HIP kernel; then bf16: aiter Triton conv2d; 1x1/stride1/pad0/dilation1: stock | fp16/fp32 -- `src/cuda/conv2d_fp{16,32}.cu`, ported from the original CMP-Turing project's register-blocked CUDA kernels (`src/cuda/kernel_example/`). bf16 -- `amd_tuned_torch/aiter_ops.py` (`conv2d_fp16`, despite the name it's fp16/bf16), the second tier since the native kernel doesn't cover bf16. aiter's Triton conv2d alone benchmarked slower than stock MIOpen for fp16 (0.93x, see `benchmark.json`), which is why it's the fallback tier and not the primary one. Pointwise (1x1) convs skip both tiers entirely -- `miopen_amd_log.txt` shows MIOpen's own rocBLAS GEMM solver beating its best Winograd kernel by 3.4x and naive direct by 450x for exactly this case (see `_is_pointwise_conv2d` in `amd_tuned_torch/__init__.py`), since it's a pure channel-mixing GEMM neither of our tiers can realistically win. **CK tier first** (`amd_tuned_torch/ck_ops.py`, `src/cuda/ck_conv_fwd*.cu`): it beats the hand-written kernel at every shape measured, and for **bf16 it beats stock by ~3x** (stock has no bf16 Winograd solver on gfx11 and falls back to im2col+GEMM). For **fp16, stock still wins outright** (MIOpen dispatches 3x3 fp16 to a hand-written assembly Winograd solver at ~90% of peak) -- the tier order does not currently encode that; see `tools/bench_ck.py`. |
| `F.conv3d` (groups=1, fp16/bf16/fp32) | Composable Kernel WMMA conv, then hand-written HIP kernel(s) | `src/cuda/conv3d_fp{16,32}.cu`, also ported from the CMP-Turing project. Neither aiter nor TransformerEngine cover conv3d at all, which is exactly the gap the CK tier fills. fp32 additionally has a second, narrower-scope kernel, `src/cuda/conv3d_fp32_winograd.cu` (batch=1, kernel=3x3x3, stride=1, padding=1, dilation=1 only) -- `src/main_rocm.cpp`'s `run_conv3d_fp32` benchmarks it against the direct fp32 kernel the first time a given shape is seen and caches whichever wins, since Winograd isn't always faster just because it's applicable (see that file's own header for the reasoning and the register-blocking that makes it competitive at all). **CK tier first**, and this is the clearest win in the extension: ~2x faster than stock in both fp16 and bf16 (stock's 3D path only reaches ~30% of peak in every dtype), and it gives conv3d a bf16 path where there was none. The hand-written kernels remain the fp32 path and the fallback. |
| `F.group_norm` | Hand-written HIP kernel | `src/cuda/group_norm.cu`. Neither aiter nor TransformerEngine cover GroupNorm -- it's a diffusion-U-Net-specific op (per-spatial-group norm), not a transformer/LLM one those two target. Block-per-`(n, group)`, warp-shuffle + shared-mem reduction -- not the one-thread-per-output pattern that made the old Turing project's *unverified* conv3d/interpolate kernels slow. |
| `F.scaled_dot_product_attention` | TransformerEngine `DotProductAttention` | `amd_tuned_torch/te_ops.py`. TE-ROCm auto-selects between its CK and AOTriton fused-attention backends; no reason to reimplement either. Fires for `is_causal=True` *or* an explicit boolean `attn_mask` tensor that structurally matches TE's fused "causal_bottom_right" pattern (`te_ops.is_bottom_right_causal_mask`) -- HF model code and dflash build an explicit mask for KV-cache decoding rather than setting the flag, so without this amd_tuned_torch's SDPA patch would never fire for them. Any other mask (padding, sliding-window, genuinely arbitrary) still falls back to stock: TE's fused backends only accept named mask types, and forcing `attn_mask_type="arbitrary"` disables both the CK and AOTriton fused paths. |
| `F.rms_norm` | TransformerEngine `transformer_engine_torch.rmsnorm_fwd` (+ backward) | `amd_tuned_torch/te_ops.py`. |
| `F.gelu` (only `approximate="tanh"`), `F.silu` (fp16/bf16 only) | TransformerEngine `transformer_engine_torch.gelu`/`silu` (+ backward) | `amd_tuned_torch/te_ops.py`. `approximate="none"` (the default, exact erf-based GELU) is deliberately left on stock -- TE's `gelu` is the tanh approximation only, and silently swapping the default would change numerics. `silu`'s fp32 case is also left on stock -- see benchmark note below. |

## Fastest kernel by default (`amd_tuned_torch/kernel_select.py`)

This package does not assume its own kernels are faster than stock. For
every op where there is a real choice, the first call on each distinct
shape runs a **contest** -- stock ROCm included as an ordinary candidate --
and the winner is cached for that shape. Subsequent calls dispatch straight
to it (~2us of wrapper).

That is a deliberate reversal of the project's original premise. It started
as a fork targeting Turing/CMP mining cards, where the driver throttled
stock kernels and ours were faster essentially by construction. **gfx1100
throttles nothing**, MIOpen is frequently excellent, and measuring showed
`F.conv2d` had been a 4.6x (fp16) and 6.7x (fp32) *regression* against stock
while presenting itself as an optimisation.

What a default `import torch` now gets, measured end-to-end through the
patch against unpatched stock (`analyse/benchmarks/default_dispatch.txt`,
regenerate with the script beside it):

| Op | dtype | stock | patched | vs stock | picked |
|---|---|---|---|---|---|
| `F.conv2d` | fp16 | 1.428 ms | 1.462 ms | 0.98x | stock |
| `F.conv2d` | **bf16** | 8.845 ms | **3.843 ms** | **2.30x** | ck |
| `F.conv2d` | fp32 | 3.392 ms | 3.403 ms | 1.00x | stock |
| `F.conv3d` | **fp16** | 3.025 ms | **1.713 ms** | **1.77x** | ck |
| `F.conv3d` | **bf16** | 3.076 ms | **1.737 ms** | **1.77x** | ck |
| `F.conv3d` | fp32 | 6.453 ms | 6.294 ms | 1.03x | stock |
| `F.group_norm` | fp16 | 0.1665 ms | 0.1709 ms | 0.97x | native |
| `F.group_norm` | bf16 | 0.1722 ms | 0.1661 ms | 1.04x | native |
| `F.group_norm` | fp32 | 0.2112 ms | 0.2052 ms | 1.03x | native |
| `F.linear`, `torch.matmul`, `torch.bmm` | any | — | — | 1.00x (stock) | see below |

Reading it honestly: **conv3d and conv2d-bf16 are large real wins**; every
other row is parity within a few percent, where the residual is the ~5us
Python dispatch wrapper, and `group_norm` is a wash on this hardware
(its kernel beats stock by ~1.10x raw, which is about what the wrapper
costs back).

**linear/matmul/bmm have no contest because there is no second candidate on
this card.** aiter's Triton GEMM -- the intended tier -- raises
`KeyError('gfx1100')`: the installed build ships no tuning config for RDNA3.
Separately, `amd_tuned_torch/aiter_ops.py` imports symbols
(`aiter.ops.triton.gemm.basic.gemm_a16w16`, ...) that do not exist in the
installed `amd-aiter`, so `aiter_ops.available()` is `False` and the ops
quietly stay on stock. Repairing those import paths would not help until
aiter supports gfx1100; the current behaviour (stock) is already the fastest
available.

Controls: `AMD_TUNED_TORCH_MEASURE_KERNELS=0` restores the old
always-prefer-our-kernels ordering (`AMD_TUNED_TORCH_CONV_MEASURE` is still
honoured). `amd_tuned_torch.kernel_select.debug_winners()` reports what won
where -- otherwise invisible, since every candidate returns a numerically
equivalent tensor. Reproduce any of it with `tools/bench_conv2d_fp16.py`,
`tools/bench_ck.py`, and the harnesses in `analyse/`.

## Composable Kernel conv tier

`F.conv2d`/`F.conv3d` in fp16/bf16 route first through Composable Kernel's
WMMA grouped-conv-forward instances (`src/cuda/ck_conv_fwd*.cu`,
`amd_tuned_torch/ck_ops.py`). This is the only place CK is used, it is
**optional**, and it was added because of a specific measured hole rather
than a general preference -- reproduce all of it with
`python tools/bench_ck.py`, which also prints the MIOpen solver stock chose.

Measured on RX 7900 XTX at `tools/bench.py`'s shapes (kernel time; the CK
column is channels-last input, see the layout note below):

| Op | dtype | Stock | This project's HIP kernel | CK | Winner |
|---|---|---|---|---|---|
| conv2d | fp16 | **1.41 ms** | 6.51 ms | 1.85 ms | stock |
| conv2d | bf16 | 8.72 ms | *(none)* | **1.87 ms** | **CK, 4.7x** |
| conv3d | fp16 | 3.02 ms | 3.79 ms | **1.16 ms** | **CK, 2.6x** |
| conv3d | bf16 | 3.17 ms | *(none)* | **1.18 ms** | **CK, 2.7x** |

The asymmetry is algorithmic, not a matter of tuning quality. MIOpen
dispatches 3x3 fp16/fp32 conv2d to a hand-written gfx11 **assembly
Winograd** solver (`miopenSp3AsmConvFury_v2_4_1_gfx11_..._f2x3_...`) that
runs at ~90% of this card's peak -- Winograd does fewer multiplies than
direct convolution, so no implicit-GEMM kernel (CK's or ours) can catch it,
and none should try. But MIOpen has **no bf16 Winograd solver on gfx11**:
bf16 falls back to `Im2d2Col_v2` writing a full im2col buffer to memory plus
a Tensile GEMM, at ~14% of peak. And MIOpen's 3D path reaches only ~30% of
peak in every dtype. Those two gaps are what CK fills.

**Layout is the one real cost.** CK ships WMMA conv instances for
channels-last layouts only (`NHWGC`/`GKYXC`/`NHWGK` and the 3D equivalents);
its `NGCHW` -- i.e. plain NCHW at `groups=1` -- instances are xdl/CDNA-only,
so there is no contiguous-NCHW path on RDNA3. An NCHW caller therefore pays
a permute in and out (conv2d bf16: 4.33 ms end-to-end instead of 2.70 ms,
still 2x better than stock); a channels-last caller pays nothing, which is
the normal case inside an inference graph running with
`PYTORCH_MIOPEN_SUGGEST_NHWC=1`. The wrapper returns the output in whatever
format it was handed. Bias is fused into CK's epilogue as a broadcast `D`
tensor, so there is no separate bias pass.

**Build cost.** CK is header-only here -- its device ops are instantiated
directly, so there is no CK cmake build and nothing to link -- but those
instantiations dominate build time (minutes per rank/dtype, which is why
there is one translation unit per combination so they compile in parallel).
Point `AMD_TUNED_TORCH_CK_ROOT` at a CK checkout to enable the tier; it
defaults to the sibling `composable_kernel` checkout, and if CK isn't found
the tier compiles out and the extension builds exactly as it did before
(`amd_tuned_torch.ck_ops.available()` reports which). Which CK instances are
compiled in is itself a measured tradeoff -- re-derive it with
`tools/ck_instance_sweep.hip`, and see `src/cuda/ck_conv_fwd_impl.hpp`.

**Stock is a candidate, not a fallback.** For conv2d fp16/fp32 stock beats
both the CK tier and this project's own kernel, and for bf16 it loses by
3x, so no fixed tier order is right for all dtypes. Rather than hardcode
that split, `amd_tuned_torch/kernel_select.py` runs a contest on the first
call for each distinct shape -- stock included as an ordinary candidate --
and caches the winner, the same benchmark-once-then-cache policy
`run_conv2d_fp16` already uses to choose among its own tile shapes. Result
at the bench shapes:

| Op | dtype | vs stock before | vs stock now | picked |
|---|---|---|---|---|
| conv2d | fp16 | 0.22x (4.6x slower) | **1.0x** (parity) | stock |
| conv2d | fp32 | 0.15x (6.7x slower) | **1.0x** (parity) | stock |
| conv2d | bf16 | n/a | **2.1x faster** | ck |
| conv3d | fp16 | 0.80x | **1.7x faster** | ck |
| conv3d | bf16 | n/a | **1.8x faster** | ck |
| conv3d | fp32 | 0.95x | **1.0x** (parity) | stock |

The wrapper costs ~2us per call once a shape is decided, and parity rows
are parity within run-to-run noise (measured interleaved in one process;
measuring stock and patched in separate phases misattributes allocator and
clock state as overhead). `AMD_TUNED_TORCH_MEASURE_KERNELS=0` restores the old
always-prefer-our-kernels ordering, and
`amd_tuned_torch.kernel_select.debug_winners()` shows what won where -- which
is otherwise invisible, since every candidate returns a numerically
equivalent tensor.

Everything else (`embedding`, `softmax`, `interpolate`, grouped `conv2d`/
`conv3d` (`groups != 1`), etc.) is intentionally **not** patched. Unlike on
Turing, stock rocBLAS/hipBLASLt/MIOpen on ROCm isn't throttled, so there's
no correctness or performance reason to replace those until they're
actually measured as a bottleneck for a real workload.

**`F.layer_norm` has a working patch that `enable()` doesn't install**, per
`benchmark.json`: TE's `layernorm_fwd` benchmarked slower than stock for
both fp16 (0.71x) and fp32 (0.61x). `_patched_layer_norm` remains in the
source for manual use if a future build benchmarks favorably -- it's just
not wired into `enable()`/`disable()`. Similarly, `F.silu`'s patch only
engages for fp16/bf16 inputs; fp32 benchmarked slower (0.80x, vs. fp16's
1.15x) and stays on stock. (`benchmark.json`'s conv2d/conv3d numbers predate
the native kernels above and measured aiter's Triton conv2d alone --
re-benchmark the native kernels once built before trusting they hold.)

## `torch.compile` compatibility (`amd_tuned_torch/compile_ops.py`)

`F.linear`/`torch.matmul`/`torch.bmm`/`F.conv2d`(bf16)/`F.group_norm` and
the native `conv2d`/`conv3d` kernels are called through
`amd_tuned_torch/compile_ops.py`, not `aiter_ops.py`/the native extension
directly. Each is registered as a `torch.library.custom_op` with a
`register_fake` shape-only stand-in -- a pattern ported from NVIDIA
TransformerEngine's own `torch.compile` glue
(`transformer_engine/pytorch/attention/custom_ops.py`, whose docstring says
exactly why it exists: *"Attention kernels wrapped as custom ops, so they
don't graph-break under torch.compile"*). Without this, Dynamo has no
special-case recognition for a monkeypatched `F.linear` etc. (that fast-path
match is by identity against the *original* torch function, lost the moment
it's rebound), so it falls back to inlining each as a plain Python function
-- which succeeds through the eligibility-check logic but then hits a call
into an aiter Triton kernel launcher or amd_tuned_torch's own pybind11 `group_norm`
extension, neither a registered PyTorch op, so Dynamo can't trace through
that boundary. Result without `compile_ops.py`: a graph break at every one
of these calls under plain `torch.compile(model)` (still correct, just
fragments the graph and loses fusion across it), or a hard
`torch._dynamo.exc.Unsupported` under `fullgraph=True`.

Unlike NVIDIA's version, these aren't registered with `device_types="cuda"`:
`amd_tuned_torch._usable()` already checks `tensor.is_cuda` (true for ROCm HIP
tensors too, which PyTorch reports under the `"cuda"` dispatch key for
compatibility) before any of these ever get called, so a device
restriction at the op-registration level would be redundant -- and would
also make them uncallable with plain CPU tensors, unlike every other piece
of dispatch logic this project's test suite exercises (aiter/TE are always
mocked out and driven with CPU tensors). Registering for all device types
keeps that same testability without weakening real eligibility gating,
which still happens exactly where it always has.

Falls back to plain function calls (today's behavior, numerically
identical, no `torch.compile` graph-node benefit) on PyTorch builds without
`torch.library.custom_op` (added in PyTorch 2.4). Nothing about autograd
changes: these six ops still have no backward pass, same as before this
existed -- `amd_tuned_torch._grad_safe()` already guarantees they're only ever
called where no backward pass is needed.

fp16 and bf16 are both first-class (RDNA3's WMMA units have native bf16
matrix throughput, unlike Turing -- no bf16->fp32 conversion at any
boundary). `F.linear`/`torch.matmul`/`torch.bmm` only intercept fp16/bf16
(aiter's Triton GEMM kernels are fp16/bf16 only); `group_norm` and every
TE-backed op handle fp32 natively too. `conv2d`/`conv3d`'s native HIP
kernels are fp16/fp32 only (no bf16) -- `conv2d`'s bf16 case falls to
aiter's Triton conv2d instead, `conv3d`'s bf16 case falls straight to
stock.

`F.linear`/`torch.matmul`/`torch.bmm` are only patched if aiter is
installed (`amd_tuned_torch.aiter_ops.available()`); left unpatched (stock
rocBLAS/hipBLASLt) otherwise -- **which is what actually happens on
gfx1100**: aiter's Triton GEMM has no tuning config for this GPU and raises
`KeyError('gfx1100')` when called, so these three ops run on stock. See
"Fastest kernel by default" above. Unlike TransformerEngine (see below), aiter's
import is not gated behind an extra opt-in env var -- it's treated as a
required-tier dependency, same as the compiled `_native` extension.

## Optional: Hugging Face Hub kernels (opt-in, network dependency)

[`kernels`](../kernel/kernels) (Hugging Face's Hub kernel loader/builder,
`source/kernel/kernels`) downloads precompiled kernels from the Hub at
runtime instead of requiring a local build -- a third kernel source
alongside aiter and TransformerEngine, both local builds from source in
this repo.

```python
import amd_tuned_torch
# AMD_TUNED_TORCH_ENABLE_HUB_KERNELS=1 in the environment before this import
```

Disabled by default -- set `AMD_TUNED_TORCH_ENABLE_HUB_KERNELS=1`, checked once at
import time (same posture as `AMD_TUNED_TORCH_ENABLE_TE`). Unlike aiter/TE (local
Python imports, fast even when they fail), fetching a Hub kernel is a
**network call** the first time a given repo/version is used, cached
locally afterward. Nothing in `amd_tuned_torch/hub_ops.py` is ever attempted at
`amd_tuned_torch` import time even with the flag on -- every fetch is lazy,
triggered only the first time the corresponding op actually runs, and
`get_hub_kernel()` catches every exception broadly (network failure,
missing Hub variant for the installed torch/ROCm build, auth error for a
gated repo -- these surface as different exception types depending on
`huggingface_hub`/`requests` internals), returning `None` instead of
raising, so a slow or absent connection can never crash an inference call
-- it just falls back the same way a TE failure would.

**Not every Hub kernel repo actually has a ROCm build**, despite
kernel-builder's build system supporting ROCm variants in the abstract.
[`kernels-community/activation`](https://huggingface.co/kernels-community/activation)
(the obvious first candidate for an `F.gelu`/`F.silu` backend, and what an
earlier version of this integration wired up) was checked against its
published file tree and dropped: every single build variant is
`torch2xx-cxx11-cuXXX-*` or `torch2xx-metal-*` -- zero `rocm` variants.
It would have silently never engaged on RDNA3, just fallen through to
stock every time.

[`kernels-community/aiter-kernels`](https://huggingface.co/kernels-community/aiter-kernels)
was the other candidate checked: a real ROCm-native repackaging (confirmed
`torch-rocm` build variant, not CUDA-only) of the *same* aiter project
this package already depends on locally (`source/aiter`). Its
`activation.fused_silu_mul` turned out to be the exact same function as
`aiter.ops.triton.activation.fused_silu_mul`, already importable locally
with zero network dependency -- so it's exposed as
**`amd_tuned_torch.aiter_ops.fused_silu_mul`** instead (see the INT8 section
below and `amd_tuned_torch/aiter_ops.py`), not fetched from the Hub at all:

```python
import amd_tuned_torch
x = amd_tuned_torch.aiter_ops.fused_silu_mul(gate_up_projection_output)
```

No reason to pay a network round-trip for code already sitting in this
repo's own local aiter dependency. `hub_ops.py` itself is left as a
generic, tested utility (`available()`/`get_hub_kernel()`) with nothing
currently wired up as a concrete backend -- call
`amd_tuned_torch.hub_ops.get_hub_kernel(repo_id, version=...)` yourself for a
future Hub-only kernel that doesn't already have a local equivalent, the
same way you'd call `kernels.get_kernel()` directly, just with amd_tuned_torch's
caching and broad failure handling added on top.

## Optional: INT8 quantized linear (aiter, opt-in, changes numerics)

`F.linear` can additionally be routed through aiter's W8A8 (int8 activation,
int8 weight) quantized GEMM (`amd_tuned_torch/aiter_ops.py: linear_int8`) -- on
gfx11/RDNA3, `aiter.gemm_a8w8` auto-routes to a pure-Triton kernel (its
CK/ASM int8 paths are CDNA-only and gated off on RDNA3, so this never
attempts a code path that wouldn't run here). Weight is quantized once
per-output-channel and cached (keyed on the weight tensor itself, so it's
dropped automatically if the weight is freed/replaced); activation is
quantized fresh per-token on every call.

**Unlike everything else in this README, this is not numerically
transparent** -- int8 introduces real accuracy loss, so it's a separate,
explicit opt-in, never installed by `enable()`/`disable()`:

```python
import amd_tuned_torch
amd_tuned_torch.enable_int8_linear()   # composes with enable()/disable() in any order
...
amd_tuned_torch.disable_int8_linear()
```

Falls back to whatever `F.linear` was routed to before (the aiter Triton
fp16/bf16 path if `enable()` had already run, stock otherwise) on
ineligible input/weight or a RuntimeError from aiter. No backward pass, so
it also falls back under autograd, same as linear/matmul/bmm/group_norm
above. Only `F.linear` is covered -- not `torch.matmul`/`torch.bmm`, which
have no static "weight" operand worth quantizing once and caching.

### Optional: SmoothQuant calibration (further opt-in on top of INT8 linear)

Plain per-token/per-channel int8 (above) has no calibration step -- it just
quantizes whatever's in front of it, every call. On top of that,
`amd_tuned_torch.calibrate_smoothquant()` adapts NVIDIA Model-Optimizer's
SmoothQuant calibration algorithm (`modelopt/torch/quantization/model_calib.py`
-- pure tensor math, nothing TensorRT/CUDA-specific about the algorithm
itself) to migrate per-input-channel dynamic range from activations (which
have LLM-typical outlier channels) into weights (which don't) before
quantizing, using aiter's own fused `smoothquant_quantize` Triton kernel
(`aiter/ops/triton/moe/quant_moe.py`) for the activation side:

```python
import amd_tuned_torch
amd_tuned_torch.enable_int8_linear()
amd_tuned_torch.calibrate_smoothquant(lambda: run_a_few_representative_batches(model))
```

Collects per-input-channel activation amax by temporarily patching
`torch.nn.Linear.forward` directly -- a module-level patch, not another
functional one -- since that's the one place a weight tensor and the exact
input about to be multiplied against it reliably meet, regardless of
whether the surrounding model code calls `F.linear` (already covered by
`amd_tuned_torch.enable()`) or holds some other reference to it. `forward()` is
restored afterward even if the calibration batches raise. Only base
`torch.nn.Linear` instances are covered -- a subclass overriding `forward()`
bypasses this the same way it'd bypass the `F.linear` patch too.

Once calibrated, `linear_int8` automatically uses the smoothed scale for
every weight seen; weights never calibrated keep using plain per-token
quantization, so calibrating a subset of layers (or none) is safe.
`amd_tuned_torch.compute_smoothquant_scale(weight)` / `amd_tuned_torch.set_smooth_scale(weight,
scale)` are exposed separately if you want to compute or supply scales
without running calibration's forward hook.

## `amd_tuned_torch.magcache` -- generic diffusion step-skipping cache

A different kind of thing from everything above: not a kernel swap, not
RDNA3/ROCm-specific, no aiter/TE dependency, and not installed by
`enable()`/`disable()`. It's a generic port of
[MagCache](../MagCache)'s (arXiv:2506.09045, NeurIPS 2025) skip/cache
decision engine for diffusion transformer inference -- reusing a cached
residual instead of recomputing a denoising step's transformer-block-stack
pass whenever that step's output magnitude is predictable from a
calibration table.

Upstream ships this as several hundred near-identical lines *per model*
(`MagCache4FLUX/magcache_flux.py`, `MagCache4Wan2.1/magcache_generate.py`,
`MagCache4HunyuanVideo/magcache_sample_video.py`, ...), each hand-copying
the same `cnt`/`accumulated_ratio`/`accumulated_err`/`accumulated_steps`
state machine into that model's own `forward()`. What actually differs per
model is only *where* to call the block stack and what counts as its
input/output residual -- the skip/cache decision itself never touches a
model class, block type, or tensor shape beyond an elementwise subtraction.
`amd_tuned_torch/magcache.py` extracts exactly that decision engine so it's shared
instead of re-copied:

```python
import amd_tuned_torch

magcache = amd_tuned_torch.magcache.MagCache(num_steps=28, mag_ratios=my_calibrated_table)

for step in range(num_inference_steps):
    if magcache.should_skip():
        hidden_states = hidden_states + magcache.cached_residual
    else:
        ori_hidden_states = hidden_states
        hidden_states = run_transformer_blocks(hidden_states, ...)  # your model's own block loop
        magcache.record(hidden_states - ori_hidden_states)
    magcache.advance()
```

`mag_ratios` is a per-step calibration table specific to a model and step
count -- there's no universal one. Either use one of upstream's published
tables for a supported model (e.g. FLUX's is hardcoded in
`MagCache4FLUX/magcache_flux.py`), or calibrate your own once with
`MagCacheCalibrator`:

```python
calib = amd_tuned_torch.magcache.MagCacheCalibrator(num_steps=28)
for step in range(num_inference_steps):
    ori_hidden_states = hidden_states
    hidden_states = run_transformer_blocks(hidden_states, ...)
    calib.record(hidden_states - ori_hidden_states)
mag_ratios = calib.finalize()  # hardcode this for later MagCache(...) use, same as upstream does
```

Running at a different step count than a table was calibrated at needs a
resample first: `amd_tuned_torch.magcache.nearest_interp(mag_ratios, target_length)`
(ported verbatim from upstream's own `nearest_interp`).

## `amd_tuned_torch.teacache` -- the same idea, a different skip signal

Same family as `amd_tuned_torch.magcache` above (reuse a cached residual instead of
recomputing a step), ported from a different paper --
[TeaCache](../TeaCache) (arXiv:2411.19108). The difference is what decides
whether to skip: MagCache looks up a precomputed, content-independent
per-step ratio; TeaCache computes a live signal every step from the
model's own timestep-embedding modulation -- the relative L1 distance
between this step's and the previous step's "modulated input" (e.g.
`transformer_blocks[0].norm1(hidden_states, emb=temb)` in a
diffusers-style DiT), rescaled through a per-model calibrated polynomial.

```python
import amd_tuned_torch

teacache = amd_tuned_torch.teacache.TeaCache(
    num_steps=28, rel_l1_thresh=0.6, coefficients=my_calibrated_coefficients,
)

for step in range(num_inference_steps):
    modulated_input, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        transformer_blocks[0].norm1(hidden_states, emb=temb)  # your model's own first-block modulation
    )
    if teacache.should_skip(modulated_input):
        hidden_states = hidden_states + teacache.cached_residual
    else:
        ori_hidden_states = hidden_states
        hidden_states = run_transformer_blocks(hidden_states, ...)  # your model's own block loop
        teacache.record(hidden_states - ori_hidden_states)
    teacache.advance()
```

`coefficients` is a per-model calibrated polynomial -- there's no universal
one; use one of upstream's published per-model values (e.g. FLUX's is
hardcoded in `TeaCache4FLUX/teacache_flux.py`) or leave it `None` to use
the raw relative-L1 distance unscaled. Unlike MagCache, there's no
separate calibrator here: TeaCache's raw signal is directly observable
every inference step (no dedicated calibration pass needed to produce
it) -- only the optional polynomial rescale is model-specific, and
upstream fits that offline once per model, the same way MagCache's
`mag_ratios` table is fit offline.

## `amd_tuned_torch.cache` -- generic similarity-gated cache (opt-in, env-flagged)

Generalizes `amd_tuned_torch.teacache`'s mechanism beyond diffusion transformers
entirely: no step counter, no calibration table, no residual structure
required. `SimilarityCache` (and its decorator sugar `similarity_cached`)
wrap *any* callable, skipping it and reusing its last output whenever the
current call's primary tensor argument is close enough (relative L1
distance) to the previous call's:

```python
import amd_tuned_torch

@amd_tuned_torch.cache.similarity_cached(thresh=0.1)
def run_expensive_step(hidden_states, *aux_args):
    return transformer_blocks(hidden_states, *aux_args)

# or, for explicit control instead of a decorator:
cache = amd_tuned_torch.cache.SimilarityCache(thresh=0.1)
output = cache.call(run_expensive_step, hidden_states, *aux_args)
```

**This is the least safe-by-default piece of amd_tuned_torch.** "Close enough" as
a proxy for "safe to skip" only holds for workloads where consecutive
calls are expected to drift smoothly (diffusion timesteps, video frames,
iterative refinement) -- it is *not* safe for e.g. independent batch items
or unrelated calls sharing a call site, where similar input doesn't imply
reusing the last output is a reasonable approximation. That's why, on top
of never being auto-installed by `enable()`/`disable()`, it also needs its
own environment flag: `AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE` (default off,
`"0"`). Every `SimilarityCache` constructed without an explicit `enabled=`
argument reads this once, at construction time -- left unset, it's a pure
passthrough (`fn` always runs, nothing is ever skipped, output is
identical to calling `fn` directly) at the cost of a cheap L1-distance
check per call. Pass `enabled=True`/`enabled=False` to override the
environment flag for a specific instance.

Only the *first* positional argument drives the skip decision; further
positional/keyword arguments are forwarded to `fn` unchanged on every call
and assumed not to affect whether skipping is safe. Cache hits return the
*same* output object each time, not a copy -- don't mutate a returned
value in place if you're relying on the cache (the same aliasing caveat
`amd_tuned_torch.magcache`/`teacache`'s `cached_residual` already carries). Call
`.reset()` between unrelated runs so the first call of the next one is
never mistaken for similar to the last call of the previous one.

### `KeyedCache` -- exact, identity-keyed memoization

Not every reuse pattern is "close enough to the previous call." `SimilarityCache`
answers a statistical question -- is this input close enough to approximate
the last output -- correct only when consecutive calls are *expected* to
drift smoothly. `KeyedCache` answers a different, exact one: has this
*exact logical query* already been computed, ever -- correct whenever you
can name the query's identity, with no approximation and no dependence on
how similar two queries' tensor contents look.

This distinction matters concretely for a call site like a geometry
decoder invoked once per spatial chunk of a dense query grid: consecutive
chunks are adjacent regions, not independently-random batches, so
`SimilarityCache`'s coarse aggregate-difference metric can coincidentally
read two *different* chunks as "close enough" -- silently returning the
wrong chunk's decoded output. No `thresh` value fixes this; tightening it
only shrinks the odds of a coincidental match, it can't eliminate them,
because the failure is in what's being measured (aggregate similarity),
not the tolerance on it. `KeyedCache` sidesteps the whole question --
there's no similarity metric to fool:

```python
cache = amd_tuned_torch.cache.KeyedCache()
key = (mesh_id, resolution, chunk_index)
occupancy = cache.call(geo_decoder, key, queries=chunk_queries, latents=latents)
```

`key` must be hashable and must NOT be derived from tensor contents --
computing it from a GPU tensor's values (even via a GPU-side hash) would
reintroduce the same sync-stall problem this is meant to avoid, for no
benefit over comparing values directly. Derive it from whatever already
identifies the query logically -- an index, a resolution, a request ID --
values you have on hand before the tensor is built.

**Whether this helps at all depends entirely on whether your workload ever
presents the same key twice.** A single sweep that visits each key exactly
once -- e.g. one pass over `generate_dense_grid_points()`'s chunks, where
every chunk is queried once and the correct output differs by `latents`
per generation (so even the same spatial chunk index decodes differently
for a different mesh) -- gets zero hits and pure overhead from being
wrapped in a `KeyedCache`. Only wrap a call site with confirmed or
expected repeated identical queries (e.g. retries of the same generation),
not a single forward pass.

Like `SimilarityCache`, explicit-usage only -- there's no module-level
auto-cache equivalent, since that mechanism's entire value is not needing
domain knowledge, and a key IS domain knowledge only the caller has.
`max_entries` caps retained keys (default unbounded); once reached, new
misses are computed and returned but not stored. `enabled=` defaults to
reading `AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE`, same flag `SimilarityCache`
reads, even though `KeyedCache` carries none of its approximation risk --
what they share is the more basic risk any cache carries: silently
serving a stale value if the underlying computation's *other* inputs
(e.g. `latents` here, if not included in the key) change without the key
changing. `reset()` clears everything -- call it when that happens (e.g.
between meshes, if you keyed by chunk/resolution but not mesh identity).

### `AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE=1` -- global, automatic, process-wide

Unlike every other flag in this project, setting
`AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE` doesn't just change a default -- it
patches `torch.nn.Module.__call__` process-wide the moment `amd_tuned_torch` is
imported, giving *every* `nn.Module` instance in the process its own
private similarity cache (keyed by that instance's identity, so different
layers/modules never compare against each other's inputs). No code changes
needed: this is the literal "use it in torch by default" behavior the flag
promises.

This is meaningfully riskier than the explicit decorator usage above,
because amd_tuned_torch has no way to know whether a given module is actually
called in a safe "smoothly drifting consecutive inputs" pattern (e.g. a
top-level diffusion transformer invoked once per external sampling-loop
timestep) or called many times per single forward pass with logically
*unrelated* inputs each time (e.g. a block called once per token, once per
expert, once per recurrent step). In the unsafe case, this can silently
reuse a completely wrong previous output instead of merely approximating
-- **not a quality/speed tradeoff, a correctness bug.** Three safety
valves:

- **Grad-safety**: a module is only ever cached under the same condition
  that gates amd_tuned_torch's other non-autograd kernel swaps (`no_grad`/
  `inference_mode`, or none of its call args `requires_grad`) -- training
  runs are effectively exempt regardless of this flag, since reusing a
  stale output tensor from a different forward pass would otherwise
  corrupt autograd.
- **Per-module opt-out**: set `module._amd_tuned_torch_no_cache = True` on any
  specific instance (or its class) to exclude it even while the global
  flag is on.
- **`amd_tuned_torch.cache.disable_module_cache()`** restores stock
  `torch.nn.Module.__call__` and forgets every per-module cache
  immediately.

`AMD_TUNED_TORCH_SIMILARITY_CACHE_THRESH` (default `"0.1"`) sets the relative-L1
threshold used for every auto-created per-module cache; call
`amd_tuned_torch.cache.enable_module_cache(thresh=...)` yourself instead of
relying on the import-time auto-install if you want a different value
without setting another environment variable.

`AMD_TUNED_TORCH_SIMILARITY_CACHE_MAX_BYTES` (default `"67108864"`, 64 MiB) caps
how large a module's primary-argument tensor (`numel() * element_size()`)
can be before that module's cache stops caching it entirely for that call
-- the real call still runs, but neither the input nor the output is
retained. Set it to `"0"` for no limit. This one defaults to *on* even
though every other amd_tuned_torch knob defaults to off, because module-level
caching keeps `previous_input`/`previous_output` alive for as long as the
module instance exists (i.e. for the life of the process, in a long-running
server) -- for a module that only ever sees small tensors that's a bounded
cost, but for one that sees large image/activation tensors (a VAE decoder,
say) it's a standing VRAM cost on top of whatever the model itself needs,
and was enough on its own to turn a routine allocation into an OOM in
practice. `SimilarityCache`'s own `max_bytes=` (explicit decorator/
direct-construction usage) still defaults to `None` -- no limit -- since
you're opting into that usage yourself.

`AMD_TUNED_TORCH_SIMILARITY_CACHE_MIN_BYTES` (default `"4096"`, 4 KiB) is
`MAX_BYTES`'s mirror image: caps how *small* a module's primary-argument
tensor can be before its cache stops caching it, same bypass semantics
(real call still runs, nothing retained). Defaults to a real floor here for
a different reason than `MAX_BYTES`'s memory-safety one: below a few KiB,
the fixed per-call overhead of the distance check itself (an elementwise
subtract, two reductions, and a `.item()` device sync) -- plus, with
`compile=True`, torch.compile's own guard/dispatch overhead -- stops being
negligible next to just running the tiny op directly, so caching a
genuinely small tensor (a timestep-embedding lookup, a small gating
scalar) can be a net slowdown. 4 KiB is conservative: well below a typical
hidden-state tensor (a single-token 4096-dim fp16 hidden state is already
8 KiB), so it shouldn't disable caching for anything that would actually
benefit. Set to `"0"` for no floor. `SimilarityCache`'s own `min_bytes=`
(explicit decorator/direct-construction usage) still defaults to `None` --
no floor -- same asymmetry as `max_bytes=` above.

`AMD_TUNED_TORCH_SIMILARITY_CACHE_MAX_MISSES` (default `"5"`) is a circuit breaker:
once a module's auto-created cache sees this many misses in a row, it
auto-disables itself -- every later call skips the distance check
(including its `.item()` sync) entirely until `reset()`. This catches a
case size bounds can't: a module invoked repeatedly with inputs that are
simply never similar to each other, regardless of size -- e.g. a geometry
decoder called once per spatial chunk while decoding a 3D volume, where
each chunk queries a different, unrelated region. Measured on exactly that
workload (Hunyuan3D-2's volume decoding): every one of the decoder's ~16
internal submodule calls paid the mandatory GPU->CPU sync on every one of
~2000+ chunks for a skip that essentially never happened, roughly halving
throughput. A hit resets the miss streak (it's evidence the check is worth
its cost); `reset()` re-arms a tripped breaker for a fresh run. Set to
`"0"` to never auto-disable. `SimilarityCache`'s own
`max_consecutive_misses=` (explicit decorator/direct-construction usage)
still defaults to `None` -- never auto-disable -- same asymmetry as
`max_bytes=`/`min_bytes=` above.

`AMD_TUNED_TORCH_SIMILARITY_CACHE_MIN_HIT_RATE` (default `"0.2"`, i.e. 20%) is a
second, independent circuit breaker -- rate-based instead of streak-based
-- for a gap `MAX_MISSES` leaves open: a "never similar" workload measured
by a strict consecutive-miss streak can be defeated by rare *coincidental*
hits. That's exactly what happens decoding a dense 3D query grid: chunks
come from `generate_dense_grid_points()`, so consecutive chunks are
spatially *adjacent* regions, not independently-random batches, and
`_is_close_enough`'s similarity signal is one coarse number (mean absolute
difference over the *entire* tensor) -- for a large batch of
quasi-uniformly-distributed coordinates, that aggregate statistic can land
within `thresh` of the previous chunk's purely by coincidence, even though
every individual query point differs. Each such coincidence resets
`MAX_MISSES`'s streak counter (indefinitely postponing that breaker) *and*
is itself a correctness bug -- silently returning a different chunk's
decoded output for this chunk's actual query points.

This breaker tracks hit/miss outcomes over a rolling wall-clock **time**
window instead of requiring an unbroken streak, so scattered coincidental
hits can no longer indefinitely block a trip -- the decision is "what
fraction of calls in the last `hit_rate_window_seconds` (default 2.0,
requiring at least `hit_rate_min_samples`=20 samples) actually hit",
not "how many misses in a row". It bounds the *cost* of the coincidental-
hit problem, it doesn't fix it -- a low-enough rate to trip still means
every hit along the way served wrong output for that call. For a module
where even occasional wrong hits are unacceptable, the reliable fix
remains the per-module opt-out (`module._amd_tuned_torch_no_cache = True`), not
either circuit breaker. Set to `"0"` to disable; `SimilarityCache`'s own
`min_hit_rate=` (explicit decorator/direct-construction usage) still
defaults to `None` -- disabled -- same asymmetry as the other module-cache
defaults above. `hit_rate_window_seconds`/`hit_rate_min_samples` aren't
exposed as their own environment variables -- pass them to
`enable_module_cache()` directly if the defaults don't suit your call
frequency.

`AMD_TUNED_TORCH_SIMILARITY_CACHE_AUTO_EXCLUDE` (default off, `"0"`): once a
per-module cache's breaker trips (either one) and there's no `compile=True`
speedup worth preserving, permanently exclude that module from the wrapper
entirely -- setting `_amd_tuned_torch_no_cache = True` on it, the same attribute
the manual opt-out uses -- instead of continuing to route every future
call through `SimilarityCache.call()` just to hit its already-cheap
`_auto_disabled` fast-path. Real savings: even that fast-path still pays a
weakref dict lookup, the grad-safety check, and building a fresh closure,
every call, forever. Measured on the volume-decoding workload the other
breakers were tuned against: once both breakers were tripping within the
first ~12 of 2122 chunks, throughput barely improved further without
this -- the remaining gap was this fixed per-call dispatch cost, paid
~34,000 times for calls that were already going to pass straight through.
Off by default because it changes what `reset()` means for that module:
`reset()` only clears a `SimilarityCache`'s own state, it can't undo an
attribute set on the module -- once excluded, a module stays excluded for
the rest of the process (across `reset()`, and even across
`disable_module_cache()` + re-enable) until you clear
`module._amd_tuned_torch_no_cache` yourself. Fine -- often exactly what you want
-- for a workload you've already confirmed never benefits from caching.

`AMD_TUNED_TORCH_SIMILARITY_CACHE_CLASS_COOLDOWN` (default unset, disabled), a
number of seconds: when any one instance's breaker trips, put every OTHER
instance of that same *class* on a temporary cooldown too -- they skip
straight through, no check, no bookkeeping, whether or not they have their
own `SimilarityCache` yet. Complements `AUTO_EXCLUDE` rather than
replacing it: that one waits for a specific instance to prove itself,
permanently; this generalizes one instance's trip to its whole class
immediately, temporarily. Useful for a deep stack of many instances of the
same class (e.g. every `RMSNorm` in a decoder) all seeing their own slice
of the same never-similar per-chunk activations -- whichever trips first
puts the rest on cooldown right away instead of each independently
re-discovering the same thing over its own next several calls.
Self-correcting if the trip doesn't generalize: the cooldown expires and
per-instance tracking resumes (untouched, not reset, during the cooldown).
No correctness cost beyond any breaker's usual one -- cooldown'd calls
still run `fn` for real, they just skip checking first. Set to a positive
number of seconds (e.g. `"20"`) to enable.

`AMD_TUNED_TORCH_SIMILARITY_CACHE_COMPILE` (default off, `"0"`) makes every
auto-created per-module cache run its module's forward through
`amd_tuned_torch.torch_compile.compile_fn` the first time that module's cache
actually calls it (a miss), reusing the compiled callable for every later
miss. Defaults to off, unlike `MAX_BYTES`: this flag wraps
`torch.nn.Module.__call__` process-wide, so turning it on means compiling
every distinct module individually -- potentially hundreds of leaf modules
in a real model, each paying its own first-call Inductor autotuning cost,
which is both slow to warm up and a known torch.compile anti-pattern
(compiling many tiny leaf forwards individually captures far less fusion
opportunity than compiling one larger containing module once). Prefer
calling `amd_tuned_torch.torch_compile.compile_module()` yourself on a handful of
top-level submodules (a UNet, a VAE) instead of reaching for this flag.

See `amd_tuned_torch/cache.py`'s module docstring section "Pairing with
amd_tuned_torch.torch_compile" for why skipping (fewer real calls) and compiling
(cheaper real calls) stack rather than compete -- a `SimilarityCache` hit
requires an exact shape match, so the population of calls that ever reach
`fn` is already clustered around a small number of distinct shapes, which
is exactly what `torch_compile`'s static-shape (`dynamic=False`) default
wants.

### Debugging

Every `SimilarityCache` (explicit or module-level auto-cache) tracks plain
`hits`/`misses`/`size_bypassed`/`breaker_bypassed` counters (plus
`trip_reason`, saying which breaker fired and why) regardless of any debug
flag. For the module-level auto-cache, call `amd_tuned_torch.cache.debug_summary()`
to print a per-module table of them, sorted by call volume -- run it right
after the pass you're investigating. A nonzero `hits` count next to a
hit-rate `trip_reason` is the signature of the coincidental-hit problem
`MIN_HIT_RATE` exists for: real hits happened, just not often enough to be
worth the check.

```python
import amd_tuned_torch
# ... run the pass you care about ...
amd_tuned_torch.cache.debug_summary()
```

`AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG` (default off, `"0"`) turns on additional
lifecycle logging (cache creation, circuit breaker tripping, resets) --
**setting the env var alone is enough to see output**, even if the host
application never calls `logging.basicConfig()` itself: a `StreamHandler`
is attached directly to the `"amd_tuned_torch.cache"` logger (not the root logger,
so it never changes what any other logger in the process does).

`AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG_FILE` (default unset), set to a path,
writes that same lifecycle logging to a file instead of (or in addition
to, if `AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG` is also set) stderr -- append
mode, so nothing is lost across restarts. Setting this alone is enough to
turn debug logging on too. Useful for a long-running server (Gradio,
etc.) where stderr scrollback is easy to lose track of:

```bash
export AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE=1
export AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG_FILE=/tmp/amd_tuned_torch-cache-debug.log
python3 gradio_app.py ...
# in another shell:
tail -f /tmp/amd_tuned_torch-cache-debug.log
```

If a fix here doesn't seem to be taking effect in a running process, check
`amd_tuned_torch.cache.__file__` before assuming the logic is wrong -- a stale or
separately-installed copy of `amd_tuned_torch` (e.g. `pip install`ed into a conda
env rather than the editable `-e` install pointing at this checkout) is a
more common cause than it looks.

## Autograd

`linear`/`matmul`/`bmm` (aiter Triton GEMM), `conv2d`/`conv3d` (native HIP
kernels, plus aiter for `conv2d`'s bf16 tier), and `group_norm` (native HIP
kernel) have **no backward pass** -- every patched call for these six
falls back to stock whenever autograd is live for the tensors involved
(grad mode enabled and any input actually `requires_grad`).

`attention`/`rms_norm`/`gelu`/`silu` are different: each is wrapped in a real
`torch.autograd.Function` backed by TransformerEngine's own forward/backward
bindings (`amd_tuned_torch/te_ops.py`), so these stay active for training, not
just `torch.no_grad()`/`torch.inference_mode()` inference. `layer_norm` has
the same autograd support but isn't installed by `enable()` -- see the
benchmark note above.

The opt-in INT8 linear (`aiter_ops.linear_int8`) also has no backward pass
-- it's an inference-only optimization, same fallback-under-autograd
behavior as the six above.

## Build prerequisites

This is Linux-only (ROCm is Linux-first; native Windows HIP SDK builds are
not supported here).

1. **ROCm PyTorch**, built/installed for gfx1100.
2. **aiter installed** (`source/aiter`, see that project's own build docs).
   Not a build-time dependency of `amd_tuned_torch` itself (nothing here links
   against it), but required at *import* time for linear/matmul/bmm to
   route through amd_tuned_torch, and for `conv2d`'s bf16 tier specifically (fp16/
   fp32 conv2d/conv3d use the native HIP kernels below and don't need
   aiter at all) -- `amd_tuned_torch.aiter_ops.available()` is `False` and those
   stay on stock rocBLAS/hipBLASLt/MIOpen if it's missing.
3. **rocWMMA** (the `rocwmma-dev` package, or any install that puts
   `<rocwmma/rocwmma.hpp>` on the include path -- hipcc searches
   `$ROCM_PATH/include` by default, which is where `rocwmma-dev` installs
   it). Build-time only, needed by the codegen'd fp16 conv2d/conv3d kernels
   (`src/cuda/generated/conv{2,3}d_fp16_*.cu`, see `tools/kernelgen/`),
   which use it to issue gfx1100's WMMA matrix-core instructions. fp32
   conv2d/conv3d and the winograd kernel don't need it.
4. **TransformerEngine's ROCm fork**, installed from `source/TransformerEngine`
   (see that project's own build docs). Also import-time only, not linked
   against by this extension, required for attention/rms_norm/gelu/silu to
   route through amd_tuned_torch (`layer_norm` also needs it if you call
   `_patched_layer_norm` manually, but `enable()` never installs it).

   **TE is disabled by default** -- `amd_tuned_torch/te_ops.py` never even attempts
   `import transformer_engine` unless `AMD_TUNED_TORCH_ENABLE_TE=1` is set in the
   environment before `amd_tuned_torch` is imported. This isn't just "not
   installed, so we skip it": a TE build that's ABI-mismatched against your
   installed PyTorch/ROCm (very easy to hit, since TE has to be rebuilt
   from source every time either changes) can **segfault the whole process
   on import**, not raise a catchable exception -- no amount of
   `try/except` in this package can protect against that. So it's opt-in.
   Verify it imports cleanly on its own first:
   ```
   python -c "import transformer_engine.pytorch"
   ```
   Only once that works, set `AMD_TUNED_TORCH_ENABLE_TE=1`. Left unset (or if that
   check fails), `amd_tuned_torch.te_ops.available()` is `False` and all five
   TE-backed ops silently stay on stock PyTorch.

This extension itself only compiles `group_norm`'s hand-written HIP kernel
(`src/cuda/group_norm.cu`) plus pybind glue (`src/main_rocm.cpp`) --
everything else is pure-Python dispatch into aiter/TransformerEngine:

```
pip install -e . --no-build-isolation
```

Override the target arch with `AMD_TUNED_TORCH_GPU_ARCH` if you're building for a
different RDNA3/3.5 part (gfx1101/gfx1102/gfx1150/gfx1151) -- defaults to
`gfx1100`.

## Usage

```python
import amd_tuned_torch   # patches torch/torch.nn.functional on import

# or control it explicitly:
import amd_tuned_torch
amd_tuned_torch.disable()
amd_tuned_torch.enable()
```

Set `AMD_TUNED_TORCH_AUTOPATCH=0` in the environment to import without patching.

Set `AMD_TUNED_TORCH_ENABLE_TE=1` in the environment to opt in to the four
TransformerEngine-backed patches `enable()` installs (attention/rms_norm/
gelu/silu; disabled by default -- see "Build prerequisites" above for why).

### Global auto-load (no per-script `import amd_tuned_torch` needed)

`tools/sitecustomize.py` hooks `builtins.__import__` so that *any* Python
process in this environment gets `amd_tuned_torch` auto-imported the moment it
imports `torch` -- no edits to whatever app/script you're running. Install
by copying it into your environment's site-packages root:

```bash
cp tools/sitecustomize.py \
    $(python -c "import site; print(site.getsitepackages()[0])")/sitecustomize.py
```

(If a `sitecustomize.py` already exists there from another package, append
this file's contents instead of overwriting it.) This affects *every*
Python program run in this environment, not just one app -- see the file's
own docstring for the full tradeoff, the try/except safety net around the
auto-import, and how to verify/uninstall it.

Raw, unpatched access regardless of `enable()`/`disable()` state:

```python
amd_tuned_torch.ops.group_norm(...)
amd_tuned_torch.aiter_ops.linear_fp16(...)  # / .bmm_fp16(...) / .linear_int8(...) (opt-in W8A8, see above)
amd_tuned_torch.te_ops.layer_norm(...)      # / .rms_norm / .gelu / .silu / .scaled_dot_product_attention
```

## Testing

```
python -m pytest source/cmp_ext_turing/tests
```

These test the *Python dispatch logic* only (eligibility gating, fallback
behavior) with `amd_tuned_torch._native`, `amd_tuned_torch.aiter_ops`, and `amd_tuned_torch.te_ops`
mocked out -- no GPU or build required. See `tests_hardware/README.md` for
what real-hardware validation is still needed (nothing has been authored
there yet -- it needs an actual RX 7900 XTX to write against).

## License

MIT
