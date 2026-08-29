"""
amd_tuned_torch -- monkeypatches PyTorch's ROCm op dispatch on RX 7900 XTX
(gfx1100/RDNA3) with the most optimal kernels available on this platform:

  - F.linear, torch.matmul, torch.bmm are backed by aiter's Triton WMMA
    GEMM kernels (amd_tuned_torch/aiter_ops.py: linear_fp16/bmm_fp16) -- gluon
    backend on gfx1250, Triton (WMMA-targeted) everywhere else including
    gfx1100/RDNA3, per aiter's own kernel docstrings. torch.matmul also
    covers >=4D batched inputs (e.g. attention's (batch, heads, seq,
    head_dim) Q@K^T / attn@V) by flattening the leading batch dims into one
    before calling aiter and reshaping back -- only when both operands
    share the exact same batch shape (no broadcasting; aiter's batched GEMM
    has none of its own).
  - F.conv2d (groups=1, fp16/fp32) is backed by a hand-written HIP kernel
    (src/cuda/conv2d_fp{16,32}.cu, via compile_ops.conv2d_native) -- a port
    of the original CMP-Turing project's register-blocked CUDA kernels (see
    each .cu file's header comment). aiter's Triton conv2d (fp16/bf16 only)
    benchmarked slower than stock MIOpen on RX 7900 XTX for fp16 (0.93x, see
    benchmark.json), so it's now the SECOND tier, used only for bf16 (which
    the native kernel doesn't support) -- see _patched_conv2d for the exact
    tier order. Non-contiguous inputs/weights and groups!=1 fall back to
    stock, same as before. 1x1/stride1/pad0/dilation1 convs ("pointwise")
    are a special case that skips BOTH tiers and goes straight to stock:
    they're a pure channel-mixing GEMM with no spatial reduction, and
    miopen_amd_log.txt shows MIOpen's own rocBLAS GEMM solver winning that
    case by 3.4x over its best Winograd kernel and 450x over naive direct --
    see _is_pointwise_conv2d's docstring for the full numbers and why
    neither of this module's tiers can realistically beat that.
  - F.conv3d (groups=1, fp16/fp32) is backed by a hand-written HIP kernel
    (src/cuda/conv3d_fp{16,32}.cu, via compile_ops.conv3d_native) -- also
    ported from the original CMP-Turing project. Neither aiter nor
    TransformerEngine cover conv3d at all, so there's no second tier before
    stock (see _patched_conv3d).
  - F.group_norm is a hand-written HIP kernel (src/cuda/group_norm.cu, via
    the native HIP/C++ extension in src/main_rocm.cpp) -- neither aiter nor
    TransformerEngine cover GroupNorm (a diffusion-U-Net-specific op, not a
    transformer/LLM one), so this one stays genuinely hand-rolled.
  - F.scaled_dot_product_attention, F.rms_norm, F.gelu
    (approximate="tanh" only), F.silu (fp16/bf16 only) are routed to
    TransformerEngine's ROCm-native bindings (amd_tuned_torch/te_ops.py) -- TE
    already ships tuned CK/AOTriton fused attention and fused
    RMSNorm/GELU/SiLU kernels for ROCm, so there's no reason to hand-roll
    those. F.layer_norm is NOT auto-patched: TE's layernorm_fwd benchmarked
    slower than stock for both fp16 (0.71x) and fp32 (0.61x), see
    benchmark.json -- _patched_layer_norm remains available for manual use.
    F.silu's fp32 case also benchmarked slower (0.80x, vs. fp16's 1.15x) and
    is excluded from the patch, falling back to stock per-call. Attention
    also fires for an explicit causal `attn_mask` tensor, not just
    `is_causal=True` -- HF model code and dflash both build one for
    KV-cache decoding instead of setting the flag -- but only when it
    structurally matches TE's fused "causal_bottom_right" pattern (see
    te_ops.is_bottom_right_causal_mask); every other mask shape (padding,
    sliding-window, arbitrary) still falls back to stock.

amd_tuned_torch.aiter_ops.fused_silu_mul is a manually-callable helper (not
auto-patched -- no stock F.* op matches its SwiGLU-style gate*up
semantics), backed by aiter.ops.triton.activation.fused_silu_mul directly,
same local-import dependency as every other aiter_ops function, no
network involved. (kernels-community/aiter-kernels, a Hugging Face Hub
repackaging of this exact aiter source, was investigated first via
amd_tuned_torch/hub_ops.py -- turned out redundant with what's already local, so
importing it directly here removes a network round-trip that would have
fetched the identical code. See amd_tuned_torch/hub_ops.py for that history, and
for why kernels-community/activation -- the *other* Hub kernel tried --
was dropped instead of adapted: zero published ROCm build variants, so it
could never actually engage on RDNA3.)

Composable Kernel is NOT a dependency of this package (an earlier version
routed linear/matmul/bmm through CK's DeviceGemm instances directly --
src/cuda/ck_gemm.cu, since deleted -- before switching to aiter's Triton
GEMM, which needs no separate C++ library build/link step).

This is a DIFFERENT premise from the original CMP-Turing version of this
project (see git history / README for that version). CMP mining cards
(Turing TU10x) had a driver-enforced FFMA/Tensor-Core instruction-pattern
throttle that made even trivial ops slow unless you dodged it with
alternate instruction sequences -- that doesn't exist on a consumer RDNA3
card. The reason to intercept these particular ops on RX 7900 XTX is
ordinary kernel-selection: pick the best available ROCm kernel per op
instead of whatever `torch.nn.functional` currently uses by default, for
the small set of ops that dominate diffusion/LLM inference time. Every op
NOT patched here (conv3d, embedding, softmax, upsample, grouped conv2d,
etc.) is intentionally left on stock PyTorch/ROCm (rocBLAS/hipBLASLt/MIOpen)
-- unlike on Turing, that stock path is not throttled, so there's no
correctness or performance reason to replace it until it's actually
measured as a bottleneck.

Native fp16 AND bf16 support (RDNA3's WMMA units have native bf16 matrix
throughput, unlike Turing -- no bf16->fp32 conversion at any boundary here).
fp32 inputs fall back to stock ops for linear/matmul/bmm/conv2d (aiter's
Triton kernels are fp16/bf16 only); group_norm and every TE-backed op
handle fp32 natively.

linear_fp16/bmm_fp16/conv2d_fp16/group_norm/linear_int8 are called through
amd_tuned_torch/compile_ops.py, not aiter_ops.py/the native extension directly --
each is registered as a torch.library.custom_op with a register_fake
shape-only stand-in (ported from NVIDIA TransformerEngine's own
torch.compile glue, transformer_engine/pytorch/attention/custom_ops.py),
so torch.compile can place a call to one of these as a real graph node
instead of graph-breaking trying to inline through a raw Triton kernel
launcher or pybind11 extension call (linear_int8 additionally avoids
Dynamo tracing into its own per-weight quantization cache lookup -- see
compile_ops.py). Numerically and behaviorally identical either way -- this
only affects whether torch.compile can trace through the call, not what
it computes.

amd_tuned_torch.magcache is a different kind of thing from everything above: not a
kernel swap, and not RDNA3/ROCm-specific at all. It's a generic port of
MagCache's (source/MagCache, arXiv:2506.09045) skip/cache decision engine
for diffusion transformer inference -- reusing a cached residual instead of
recomputing a denoising step's block stack when its output magnitude is
predictable from a calibration table. Model-agnostic by construction (it
never touches a model class or block type, just the residual tensor a
caller's own block loop already produces), so it's not installed by
enable()/disable() and has no ROCm/aiter/TE dependency -- see
amd_tuned_torch/magcache.py for usage.

amd_tuned_torch.teacache is the same idea from a different paper (TeaCache,
source/TeaCache, arXiv:2411.19108) -- reuses a cached residual the same
way, but decides whether to skip using a live, content-dependent signal
(relative L1 distance between consecutive steps' timestep-embedding
modulation, rescaled through a per-model calibrated polynomial) instead of
magcache's precomputed per-step lookup table. Same properties: not
installed by enable()/disable(), no ROCm/aiter/TE dependency -- see
amd_tuned_torch/teacache.py for usage.

amd_tuned_torch.cache generalizes teacache's mechanism beyond diffusion transformers
entirely: SimilarityCache/similarity_cached wrap *any* callable, skipping
it and reusing its last output whenever the current call's primary tensor
argument is close enough (relative L1) to the previous call's -- no step
counter, no calibration table, no residual structure required. Gated by
its own AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE flag (default off) on top of never
being auto-installed -- this is the least safe-by-default piece of
amd_tuned_torch, since "close enough" as a proxy for "safe to skip" only holds for
workloads with smoothly-drifting consecutive calls; see amd_tuned_torch/cache.py's
module docstring before reaching for it. amd_tuned_torch.cache.KeyedCache is the
sibling for the opposite case -- exact, identity-keyed memoization for
workloads where reuse depends on precise input identity rather than
tolerating drift (e.g. a decoder called once per spatial chunk of a grid,
where forcing SimilarityCache's aggregate-similarity metric to work via a
tighter thresh can't eliminate coincidental false-positive matches, only
shrink their odds).

amd_tuned_torch.miopen_fallback is a different kind of thing again: not a plain
kernel swap and not a skip/reuse decision, but a crash rescue -- now with
a real kernel swap bolted onto its front. MIOpen's conv algorithm search
can fail for a specific (shape, dtype, current allocator state)
combination that has a perfectly good solution on a fresh/idle GPU --
observed directly on a Conv1d call in a long-running multi-model pipeline
(Wan2GP's ACE-Step Oobleck audio VAE), where the exact failing shape
succeeds cleanly in isolation, most likely because MIOpen's workspace
needs one contiguous free block and a session's accumulated allocator
fragmentation can starve that even with plenty of total free VRAM.
enable_conv1d_fallback() now first checks whether the call is the
depthwise-causal Conv1d pattern used by Mamba/SSM-style blocks (groups ==
channels, kernel width 2-4, padding == width - 1) and, if so, routes it
through source/kernel/causal-conv1d-amd's HIP kernel directly -- faster
than MIOpen and immune to the fragmentation failure mode, so it never even
attempts MIOpen for that shape family. Everything else still goes through
MIOpen first, retries once after torch.cuda.empty_cache() (cheap, often
enough to defragment and succeed on GPU) if that raises a miopenStatus
error, then falls back to a CPU float32 round-trip as a last resort -- see
amd_tuned_torch/miopen_fallback.py's module docstring for the full diagnosis, the
causal-conv1d fast path's exact eligibility rule, and its output-shape
caveat. Installed by default at import time
(AMD_TUNED_TORCH_MIOPEN_CONV1D_FALLBACK=0 to opt out) but still not bundled into
enable()/disable() -- it patches torch.nn.functional.conv1d directly and
is toggled independently, same as amd_tuned_torch.cache.

amd_tuned_torch.torch_compile is a different kind of optimization from everything
above: not a kernel swap or a skip/reuse decision, but whole-graph capture
via torch.compile (Dynamo + Inductor) -- the runtime analogue of what a
tool like onnxslim does ahead-of-time to a static ONNX graph (constant
folding, operator fusion), except captured live on first call and
Triton-autotuned per shape instead of rewritten once on a serialized graph.
Built on top of compile_ops.py's custom-op registrations (which just stop
amd_tuned_torch's own patched ops from graph-breaking Dynamo -- they don't compile
anything by themselves). Like similarity caching, this is never installed
by enable()/disable(): compile_module()/compile_fn() are explicit,
per-callable opt-ins, since first-call compile latency and per-shape
recompilation cost are workload-specific judgment calls. See
amd_tuned_torch/torch_compile.py's module docstring for the aggressive defaults
(max-autotune, Inductor freezing, static shapes) and ROCm-specific caveats.

Import this package to enable the patch:

    import amd_tuned_torch   # patches torch/torch.nn.functional on import

Or control it explicitly:

    import amd_tuned_torch
    amd_tuned_torch.disable()
    amd_tuned_torch.enable()

Set AMD_TUNED_TORCH_AUTOPATCH=0 in the environment to import without patching.

TransformerEngine is DISABLED BY DEFAULT (a separate gate from
AMD_TUNED_TORCH_AUTOPATCH): set AMD_TUNED_TORCH_ENABLE_TE=1 to opt in. See te_ops.py's
module docstring for why -- a TE build that's ABI-mismatched against the
installed PyTorch/ROCm can segfault the whole process just from `import
transformer_engine.pytorch`, which no amount of try/except in this package
can catch. With TE disabled (the default), `import amd_tuned_torch` can never
crash for that reason; enable()/te_ops.available() just skip the four
TE-backed patches (attention/rms_norm/gelu/silu) and those ops stay on
stock PyTorch. Verify
`python -c "import transformer_engine.pytorch"` works cleanly on its own
before setting AMD_TUNED_TORCH_ENABLE_TE=1.

SAFETY
------
linear/matmul/bmm (aiter Triton kernels), conv2d/conv3d (native HIP
kernels), and group_norm (native HIP kernel) have no backward pass --
every patched call for these six falls back to the stock op whenever
autograd is live for the tensors involved, exactly like the original
Turing version's fallback behavior.

attention/rms_norm/gelu/silu are DIFFERENT: te_ops.py wraps each one in a
real torch.autograd.Function backed by TransformerEngine's own
forward/backward bindings, so these stay active under autograd -- you get
the speedup during training too, not just torch.no_grad() inference.
layer_norm shares this autograd support but isn't auto-installed regardless
(see above).

Every wrapper falls back to the stock op on RuntimeError or TypeError from
the native/aiter/TE call (unsupported shape, wrong arg type from a
pybind11/TE binding, etc.). Every aiter-backed wrapper (linear/matmul/bmm,
plus conv2d's aiter tier and the opt-in int8 linear) additionally catches
AssertionError, since that's how aiter's own kernels signal an unsupported
case (e.g. groups != 1) -- a bare Python assert, not a raised exception
type the other two would already cover.
"""
from __future__ import annotations

import os
from typing import Any, Callable

import torch
import torch.nn.functional as F

# The compiled extension is resolved per-PyTorch rather than by ordinary
# submodule import: one editable source tree can serve several environments,
# but a .so links one specific libtorch. See amd_tuned_torch/_native_loader.py
# -- it also keeps working for the single-environment case, where the .so
# simply sits beside this file.
from . import _native_loader

_C = _native_loader.load(__name__, os.path.dirname(os.path.abspath(__file__)))
_native = _C  # so `from . import _native` elsewhere in the package resolves


def native_build_info() -> str:
    """Which _native.so is loaded and why it was chosen."""
    return _native_loader.describe()

from . import te_ops
from . import aiter_ops
from . import ck_ops
from . import ck_gemm_ops
from . import ck_norm_ops
from . import hipblaslt_ops
from . import kernel_select
from . import compile_ops
from . import magcache
from . import teacache
from . import cache
from . import hub_ops
from . import torch_compile
from . import miopen_fallback

# Raw, unpatched access to the native HIP group_norm kernel for manual use
# regardless of what enable() patches. (linear/matmul/bmm live in
# aiter_ops, not here -- see amd_tuned_torch.aiter_ops.linear_fp16/bmm_fp16.)
ops = _C

_GEMM_DTYPES = (torch.float16, torch.bfloat16)              # linear/matmul/bmm: aiter Triton fp16/bf16 only
_GROUPNORM_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_TE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_CONV_NATIVE_DTYPES = (torch.float16, torch.float32)  # conv2d/conv3d native HIP kernels: no bf16 support
# Composable Kernel's WMMA conv instances: fp16/bf16 (no fp32 -- gfx1100's
# WMMA units have no fp32 mode). bf16 is the important half of that: it is
# the one conv case where stock ROCm has no good solver at all (see
# ck_ops.py's measured table).
_CK_CONV_DTYPES = (torch.float16, torch.bfloat16)
# silu excludes fp32: benchmarked slower than stock (0.80x, see
# benchmark.json) even though fp16 is a real win (1.15x) -- fp32 stays on
# the TE path's fp16/bf16-only sibling dtypes instead of falling back per-call.
_SILU_DTYPES = (torch.float16, torch.bfloat16)
_INT8_LINEAR_DTYPES = (torch.float16, torch.bfloat16, torch.float32)  # aiter quantizes from any of these

_ORIGINALS: dict[tuple[Any, str], Callable] = {}
_ENABLED = False

# Resolved once at import time, not on every call -- _grad_safe runs on
# every single patched op invocation (it's the first check in the hottest
# of hot paths: F.linear/matmul/bmm/conv2d/conv3d/group_norm), so re-doing
# a hasattr() lookup on the torch module there for the life of the process
# is a pure waste.
_HAS_INFERENCE_MODE = hasattr(torch, "is_inference_mode_enabled")


def _is_compiling() -> bool:
    """True while Dynamo is tracing. Guarded because the API moved across
    PyTorch versions and this must never raise on the hot path."""
    try:
        return bool(torch.compiler.is_compiling())
    except AttributeError:
        try:
            return bool(torch._dynamo.is_compiling())
        except Exception:
            return False


def _grad_safe(*tensors: Any) -> bool:
    """True if swapping in a kernel with no autograd support is safe. Only
    relevant to linear/matmul/bmm/conv2d (aiter Triton kernels) and
    group_norm (native HIP kernel) -- the TE-backed ops all have real
    backward passes and don't need this check."""
    if _HAS_INFERENCE_MODE and torch.is_inference_mode_enabled():
        return True
    if not torch.is_grad_enabled():
        return True
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.requires_grad:
            return False
    return True


def _usable(*tensors: Any, dtypes: tuple = _GEMM_DTYPES) -> bool:
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            continue
        if not t.is_cuda or t.dtype not in dtypes:
            return False
    return True


def _install(target: Any, name: str, wrapper: Callable) -> None:
    key = (target, name)
    if key not in _ORIGINALS:
        _ORIGINALS[key] = getattr(target, name)
    setattr(target, name, wrapper)


def _restore(target: Any, name: str) -> None:
    original = _ORIGINALS.pop((target, name), None)
    if original is not None:
        setattr(target, name, original)


# ---------------------------------------------------------------------------
# linear/matmul/bmm/conv2d: aiter Triton kernels (no autograd -- grad-safety
# checked up front, plus a RuntimeError/AssertionError fallback for
# unsupported shapes).
# ---------------------------------------------------------------------------

def _patched_linear(input, weight, bias=None):
    """F.linear over a per-shape contest: stock, hipBLASLt, CK GEMM.

    This op had no contest until hipblaslt_ops and ck_gemm_ops existed,
    because aiter's Triton GEMM -- the only other candidate -- raises
    KeyError('gfx1100'), leaving nothing to compete against stock. It has
    two now, and measurement says neither of them wins outright. Against
    stock (rocBLAS = 1.00x) on RX 7900 XTX, fp16:

                                 hipBLASLt   CK GEMM
        4096^3                     1.29x      1.20x
        8192x4096x11008            1.09x      1.11x
        4096x4096x1024             1.49x      1.47x
        4096x1280x1280             0.96x      1.14x
        1024^3                     1.02x      1.21x
        1x4096x4096 (decode)       0.60x      0.90x

    hipBLASLt owns the large end and collapses on single-token decode --
    0.60x, where its heuristic picks a throughput kernel that never gets to
    use its throughput. CK is flatter and ahead everywhere hipBLASLt is
    behind. Two rows are outright regressions, which is exactly why this
    goes through kernel_select rather than a fixed tier order: stock is an
    ordinary candidate and wins the shapes it deserves to win.
    """
    orig = _ORIGINALS[(F, "linear")]
    if not (_grad_safe(input, weight, bias) and _usable(input, weight)):
        return orig(input, weight, bias)
    if input.dim() < 2 or weight.dim() != 2:
        return orig(input, weight, bias)

    if kernel_select.enabled():
        # Keyed on the GEMM's actual identity (M, K, N and dtype), not on
        # the caller's leading dims: [4, 128, K] and [512, K] are one
        # problem to every candidate here, and keying on the raw shape
        # would re-run the contest for each of them.
        key = (input.dtype, int(input.numel() // input.size(-1)),
               tuple(weight.shape), bias is not None)
        won = kernel_select.cached_key("linear", key)
        if won == "stock":
            return orig(input, weight, bias)
        if won is None:
            out = kernel_select.pick_key("linear", key, [
                ("hipblaslt", lambda: hipblaslt_ops.linear(
                    input, weight, bias,
                    hipblaslt_ops.EPILOGUE_BIAS if bias is not None
                    else hipblaslt_ops.EPILOGUE_NONE)),
                ("ck_gemm", lambda: ck_gemm_ops.linear(
                    input, weight, bias,
                    ck_gemm_ops.EPILOGUE_BIAS if bias is not None
                    else ck_gemm_ops.EPILOGUE_NONE)),
                ("aiter", lambda: _linear_aiter(input, weight, bias)),
                ("stock", lambda: orig(input, weight, bias)),
            ])
            if out is not None:
                return out
        elif won == "hipblaslt":
            out = hipblaslt_ops.linear(input, weight, bias,
                                       hipblaslt_ops.EPILOGUE_BIAS if bias is not None
                                       else hipblaslt_ops.EPILOGUE_NONE)
            if out is not None:
                return out
        elif won == "ck_gemm":
            out = ck_gemm_ops.linear(input, weight, bias,
                                     ck_gemm_ops.EPILOGUE_BIAS if bias is not None
                                     else ck_gemm_ops.EPILOGUE_NONE)
            if out is not None:
                return out

    out = _linear_aiter(input, weight, bias)
    return orig(input, weight, bias) if out is None else out


def _linear_aiter(input, weight, bias):
    """The pre-existing aiter path, as a candidate that may decline.

    Returns None rather than raising so it loses the contest instead of
    breaking the call -- which on gfx1100 is what always happens, since
    aiter ships no RDNA3 tuning config.
    """
    try:
        return compile_ops.linear_fp16(input, weight, bias)
    except (RuntimeError, TypeError, AssertionError, KeyError):
        return None


def _patched_matmul(input, other, *, out=None):
    orig = _ORIGINALS[(torch, "matmul")]
    if out is not None or not (_grad_safe(input, other) and _usable(input, other)):
        return orig(input, other) if out is None else orig(input, other, out=out)
    if input.dtype == other.dtype:
        try:
            if input.dim() == 2 and other.dim() == 2:
                return compile_ops.bmm_fp16(input.unsqueeze(0), other.unsqueeze(0)).squeeze(0)
            if input.dim() == 3 and other.dim() == 3:
                return compile_ops.bmm_fp16(input, other)
            # Attention's Q@K^T / attn@V matmuls are (batch, heads, seq,
            # head_dim) -- 4D, not 3D -- so they fell through to stock above
            # until now. aiter's batched_gemm_bf16 has no broadcasting
            # semantics of its own, so only take this path when both
            # operands share the exact same batch shape (no broadcast
            # needed): flatten every leading dim but the last two into one,
            # call aiter, then reshape back.
            if input.dim() == other.dim() >= 4 and input.shape[:-2] == other.shape[:-2]:
                batch_shape = input.shape[:-2]
                flat_out = compile_ops.bmm_fp16(
                    input.reshape(-1, *input.shape[-2:]),
                    other.reshape(-1, *other.shape[-2:]),
                )
                return flat_out.view(*batch_shape, *flat_out.shape[-2:])
        except (RuntimeError, TypeError, AssertionError):
            pass
    return orig(input, other)


def _patched_bmm(input, mat2, *, out=None):
    """torch.bmm over a per-shape contest: hipBLASLt, aiter, stock.

    Same reasoning as _patched_linear -- hipBLASLt is a real candidate on
    this card where aiter is not -- but with one candidate fewer: the CK
    GEMM tier compiles no batched instances. That was a build-cost decision
    (see src/cuda/ck_gemm_fwd.hpp), taken because hipBLASLt already covers
    batched shapes well; the attention-shaped GEMM it exists for measures
    1.49x fp16 / 1.73x bf16 against stock.
    """
    orig = _ORIGINALS[(torch, "bmm")]
    if out is not None or not (_grad_safe(input, mat2) and _usable(input, mat2)):
        return orig(input, mat2) if out is None else orig(input, mat2, out=out)
    if input.dtype != mat2.dtype:
        return orig(input, mat2)

    if kernel_select.enabled():
        key = (input.dtype, tuple(input.shape), tuple(mat2.shape))
        won = kernel_select.cached_key("bmm", key)
        if won == "stock":
            return orig(input, mat2)
        if won == "hipblaslt":
            got = hipblaslt_ops.bmm(input, mat2)
            if got is not None:
                return got
        elif won is None:
            got = kernel_select.pick_key("bmm", key, [
                ("hipblaslt", lambda: hipblaslt_ops.bmm(input, mat2)),
                ("aiter", lambda: _bmm_aiter(input, mat2)),
                ("stock", lambda: orig(input, mat2)),
            ])
            if got is not None:
                return got

    got = _bmm_aiter(input, mat2)
    return orig(input, mat2) if got is None else got


def _bmm_aiter(input, mat2):
    """The pre-existing aiter path, as a candidate that may decline."""
    try:
        return compile_ops.bmm_fp16(input, mat2)
    except (RuntimeError, TypeError, AssertionError, KeyError):
        return None


def _is_pointwise_conv2d(weight, stride, padding, dilation) -> bool:
    """True for a 1x1/stride1/pad0/dilation1 conv2d -- a pure per-pixel
    channel-mixing GEMM (Y = W @ X over channels) with no spatial reduction
    at all, unlike every other conv2d shape this module handles.

    Measured directly on RX 7900 XTX (see miopen_amd_log.txt): for a
    512->256ch, 1024x512 fp16 1x1 conv, MIOpen's own rocBLAS strided-batched
    GEMM solver (GemmFwd1x1_0_1) took 1.69ms, beating its own best Winograd/
    asm-direct kernel (5.74ms, 3.4x slower) and its naive direct kernel
    (759.9ms, 450x slower). rocBLAS wins this by such a wide margin because
    its GEMM primitive consumes the input's existing NCHW strides directly
    (channel-major, spatially-contiguous is already a valid (C_out,C_in) @
    (C_in, H*W) matmul via lda/ldb) with zero data movement -- neither of
    this module's own tiers can match that for K=1: the native im2col-tiled
    kernel below has no spatial window to exploit, and aiter's Triton WMMA
    GEMM is a channels-last kernel that would need an actual NCHW->NHWC
    transpose copy first. So this case is deliberately left on stock rather
    than routed through either tier."""
    return (tuple(weight.shape[2:]) == (1, 1)
            and tuple(aiter_ops._pair(stride)) == (1, 1)
            and tuple(aiter_ops._pair(padding)) == (0, 0)
            and tuple(aiter_ops._pair(dilation)) == (1, 1))


def _patched_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # Three tiers: native HIP kernel (fp16/fp32, this project's own
    # src/cuda/conv2d_fp{16,32}.cu) first, then aiter's Triton conv2d
    # (fp16/bf16 only -- covers the bf16 case the native kernel doesn't),
    # then stock. The native kernel is always available (hard-required at
    # import time, same as group_norm); aiter is optional.
    orig = _ORIGINALS[(F, "conv2d")]
    if groups != 1 or not _grad_safe(input, weight, bias):
        return orig(input, weight, bias, stride, padding, dilation, groups)
    if _is_pointwise_conv2d(weight, stride, padding, dilation):
        return orig(input, weight, bias, stride, padding, dilation, groups)
    # Which kernel wins is measured per shape, stock included as a
    # candidate -- see kernel_select.py. This is deliberately NOT a fixed
    # ordering: on gfx1100 stock beats our kernels for fp16/fp32 conv2d
    # (MIOpen has an assembly Winograd solver) and loses badly for bf16
    # (it has none), so any static order is wrong for some dtype.
    if kernel_select.enabled():
        # Fast path: once this shape has been decided, go straight to the
        # winner. Building the candidate thunks below costs ~0.09ms per
        # call -- real money on the many small convs a U-Net is made of,
        # and pointless when the answer is already known.
        _won = kernel_select.cached("conv2d", input, weight, stride, padding, dilation)
        if _won == "stock":
            return orig(input, weight, bias, stride, padding, dilation, groups)
        if _won == "ck":
            _out = ck_ops.conv2d(input, weight, bias, stride, padding, dilation)
            if _out is not None:
                return _out
        elif _won == "native":
            try:
                return compile_ops.conv2d_native(input, weight, bias, stride, padding, dilation)
            except (RuntimeError, TypeError):
                pass

        candidates = []
        if ck_ops.available() and _usable(input, weight, dtypes=_CK_CONV_DTYPES):
            candidates.append(
                ("ck", lambda: ck_ops.conv2d(input, weight, bias, stride, padding, dilation)))
        if (input.is_contiguous() and weight.is_contiguous()
                and _usable(input, weight, dtypes=_CONV_NATIVE_DTYPES)):
            candidates.append(
                ("native", lambda: compile_ops.conv2d_native(
                    input, weight, bias, stride, padding, dilation)))
        candidates.append(
            ("stock", lambda: orig(input, weight, bias, stride, padding, dilation, groups)))
        out = kernel_select.pick("conv2d", input, weight, stride, padding, dilation, candidates)
        if out is not None:
            return out

    # conv_select disabled (AMD_TUNED_TORCH_MEASURE_KERNELS=0): fall back to the
    # old fixed ordering, our kernels first.
    if ck_ops.available() and _usable(input, weight, dtypes=_CK_CONV_DTYPES):
        try:
            _ck = ck_ops.conv2d(input, weight, bias, stride, padding, dilation)
            if _ck is not None:
                return _ck
        except (RuntimeError, TypeError):
            pass
    if (input.is_contiguous() and weight.is_contiguous()
            and _usable(input, weight, dtypes=_CONV_NATIVE_DTYPES)):
        try:
            return compile_ops.conv2d_native(input, weight, bias, stride, padding, dilation)
        except (RuntimeError, TypeError):
            pass
    if aiter_ops.available() and _usable(input, weight):
        try:
            return compile_ops.conv2d_fp16(input, weight, bias, stride, padding, dilation)
        except (RuntimeError, TypeError, AssertionError):
            pass
    return orig(input, weight, bias, stride, padding, dilation, groups)


def _patched_conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # Two tiers now: Composable Kernel's WMMA conv (fp16/bf16) then this
    # project's hand-written HIP kernel (fp16/fp32,
    # src/cuda/conv3d_fp{16,32}.cu). Neither aiter nor TransformerEngine
    # cover conv3d at all, so CK is what fills that gap -- and it is also
    # the first conv3d bf16 path here that isn't stock.
    orig = _ORIGINALS[(F, "conv3d")]
    if groups != 1 or not _grad_safe(input, weight, bias):
        return orig(input, weight, bias, stride, padding, dilation, groups)
    # Measured per shape with stock as a candidate, exactly as in
    # _patched_conv2d. conv3d is where our kernels look best (CK 1.16ms vs
    # stock 3.02ms at fp16) but fp32 is within noise of stock (0.95x), so
    # the same contest applies rather than a fixed order.
    if kernel_select.enabled():
        # Fast path: once this shape has been decided, go straight to the
        # winner. Building the candidate thunks below costs ~0.09ms per
        # call -- real money on the many small convs a U-Net is made of,
        # and pointless when the answer is already known.
        _won = kernel_select.cached("conv3d", input, weight, stride, padding, dilation)
        if _won == "stock":
            return orig(input, weight, bias, stride, padding, dilation, groups)
        if _won == "ck":
            _out = ck_ops.conv3d(input, weight, bias, stride, padding, dilation)
            if _out is not None:
                return _out
        elif _won == "native":
            try:
                return compile_ops.conv3d_native(input, weight, bias, stride, padding, dilation)
            except (RuntimeError, TypeError):
                pass

        candidates = []
        if ck_ops.available() and _usable(input, weight, dtypes=_CK_CONV_DTYPES):
            candidates.append(
                ("ck", lambda: ck_ops.conv3d(input, weight, bias, stride, padding, dilation)))
        if (input.is_contiguous() and weight.is_contiguous()
                and _usable(input, weight, dtypes=_CONV_NATIVE_DTYPES)):
            candidates.append(
                ("native", lambda: compile_ops.conv3d_native(
                    input, weight, bias, stride, padding, dilation)))
        candidates.append(
            ("stock", lambda: orig(input, weight, bias, stride, padding, dilation, groups)))
        out = kernel_select.pick("conv3d", input, weight, stride, padding, dilation, candidates)
        if out is not None:
            return out

    # conv_select disabled: old fixed ordering, our kernels first.
    if ck_ops.available() and _usable(input, weight, dtypes=_CK_CONV_DTYPES):
        try:
            _ck = ck_ops.conv3d(input, weight, bias, stride, padding, dilation)
            if _ck is not None:
                return _ck
        except (RuntimeError, TypeError):
            pass
    # groups/grad-safety already checked above.
    if (not (input.is_contiguous() and weight.is_contiguous())
            or not _usable(input, weight, dtypes=_CONV_NATIVE_DTYPES)):
        return orig(input, weight, bias, stride, padding, dilation, groups)
    try:
        return compile_ops.conv3d_native(input, weight, bias, stride, padding, dilation)
    except (RuntimeError, TypeError):
        return orig(input, weight, bias, stride, padding, dilation, groups)


# ---------------------------------------------------------------------------
# group_norm: native HIP kernel (no autograd, same fallback pattern as above).
# ---------------------------------------------------------------------------

def _group_norm_native(input, num_groups, weight, bias, eps):
    """The native group_norm kernel, routed to avoid paying for
    torch.compile safety in eager.

    compile_ops.group_norm wraps the kernel in a torch.library custom op so
    it doesn't graph-break under torch.compile (see compile_ops.py). That
    wrapper costs ~15us of dispatch, which is nothing for a conv but is
    most of this kernel's margin: raw it beats stock 1.10x (0.150ms vs
    0.166ms at the bench shape), through the custom op it loses. Since the
    graph-break protection only matters while Dynamo is actually tracing,
    take the custom op then and the direct call otherwise -- identical
    numerics either way.
    """
    if _is_compiling():
        return compile_ops.group_norm(input, num_groups, weight, bias, eps)
    return ops.group_norm(input, num_groups, weight, bias, eps)


def _patched_group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    """F.group_norm over a per-shape contest: native HIP, CK, stock.

    The CK candidate only enters for CHANNELS-LAST input, and that is a
    measured restriction rather than a conservative one. CK's normalization
    instances index the tensor as [N, S1, S2, G, C], which is a free reshape
    of channels-last memory and a permute-in-and-out of contiguous NCHW.
    Measured against stock on plain GroupNorm:

        2x320x64x64  fp16    NHWC 2.09x    NCHW 0.85x
        2x640x32x32  fp16    NHWC 1.37x    NCHW 0.48x
        2x1280x16x16 fp32    NHWC 1.26x    NCHW 0.40x

    So it is offered where it wins and withheld where it loses. NCHW keeps
    the native kernel, which is what it was already getting.

    The larger win from this tier is not reachable here at all: fusing the
    SiLU that follows GroupNorm in every diffusion ResBlock is 1.6-2.1x, and
    F.group_norm cannot know an activation follows it. See
    ck_norm_ops.group_norm_silu for the call a model can make directly.
    """
    orig = _ORIGINALS[(F, "group_norm")]
    channels_last = ck_norm_ops._channels_last(input)
    if not ((input.is_contiguous() or channels_last) and _grad_safe(input, weight, bias)
            and _usable(input, dtypes=_GROUPNORM_DTYPES)):
        return orig(input, num_groups, weight, bias, eps)

    # Measured per shape like conv, rather than assuming the native kernel
    # wins. It usually does, but only by 1.03-1.23x depending on dtype --
    # small enough that "assume ours is faster" is not a safe default, and
    # the contest costs nothing after the first call for a shape.
    if kernel_select.enabled():
        # Layout belongs in the key: it decides which candidates are even
        # eligible, not merely how fast they are.
        key = (input.dtype, tuple(input.shape), int(num_groups), channels_last)
        won = kernel_select.cached_key("group_norm", key)
        if won == "stock":
            return orig(input, num_groups, weight, bias, eps)
        if won == "ck":
            out = ck_norm_ops.group_norm(input, num_groups, weight, bias, eps)
            if out is not None:
                return out
        elif won != "native":
            candidates = []
            if channels_last:
                candidates.append(
                    ("ck", lambda: ck_norm_ops.group_norm(input, num_groups, weight, bias, eps)))
            if input.is_contiguous():
                candidates.append(
                    ("native", lambda: _group_norm_native(input, num_groups, weight, bias, eps)))
            candidates.append(("stock", lambda: orig(input, num_groups, weight, bias, eps)))
            out = kernel_select.pick_key("group_norm", key, candidates)
            if out is not None:
                return out

    if input.is_contiguous():
        try:
            return _group_norm_native(input, num_groups, weight, bias, eps)
        except (RuntimeError, TypeError):
            pass
    return orig(input, num_groups, weight, bias, eps)


# ---------------------------------------------------------------------------
# Wrappers backed by TransformerEngine (real autograd -- no grad-safety
# check needed, active for training as well as inference).
# ---------------------------------------------------------------------------

def _patched_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    orig = _ORIGINALS[(F, "layer_norm")]
    if weight is None or bias is None or not _usable(input, weight, bias, dtypes=_TE_DTYPES):
        return orig(input, normalized_shape, weight, bias, eps)
    try:
        return te_ops.layer_norm(input, normalized_shape, weight, bias, eps)
    except (RuntimeError, TypeError):
        return orig(input, normalized_shape, weight, bias, eps)


def _patched_rms_norm(input, normalized_shape, weight=None, eps=None):
    orig = _ORIGINALS[(F, "rms_norm")]
    if weight is None or not _usable(input, weight, dtypes=_TE_DTYPES):
        return orig(input, normalized_shape, weight, eps)
    try:
        return te_ops.rms_norm(input, normalized_shape, weight, eps)
    except (RuntimeError, TypeError):
        return orig(input, normalized_shape, weight, eps)


def _patched_gelu(input, approximate="none"):
    orig = _ORIGINALS[(F, "gelu")]
    # tex.gelu computes the tanh approximation only -- patching the default
    # (exact, erf-based) approximate="none" path would silently change
    # numerics for anything expecting exact GELU.
    if approximate != "tanh" or not _usable(input, dtypes=_TE_DTYPES):
        return orig(input, approximate=approximate)
    try:
        return te_ops.gelu(input)
    except (RuntimeError, TypeError):
        return orig(input, approximate=approximate)


def _patched_silu(input, inplace=False):
    orig = _ORIGINALS[(F, "silu")]
    if inplace or not _usable(input, dtypes=_SILU_DTYPES):
        return orig(input, inplace=inplace)
    try:
        return te_ops.silu(input)
    except (RuntimeError, TypeError):
        return orig(input, inplace=inplace)


def _patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                   scale=None, **kwargs):
    orig = _ORIGINALS[(F, "scaled_dot_product_attention")]
    # Cheap guards first: dropout_p/dim/dtype are plain Python attribute
    # reads. te_ops.is_bottom_right_causal_mask (below) is NOT cheap -- it
    # ends in torch.equal(), which forces a GPU sync to pull the boolean
    # result back to host. Checking it before these guards would mean
    # paying for that sync on every training-mode call (dropout_p != 0)
    # only to immediately discard the result and fall back to stock anyway.
    if (dropout_p != 0.0 or query.dim() != 4
            or not _usable(query, key, value, dtypes=_TE_DTYPES)):
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                     is_causal=is_causal, scale=scale, **kwargs)
    # attn_mask is allowed through only when it structurally matches the
    # bottom-right causal pattern (see te_ops.is_bottom_right_causal_mask) --
    # e.g. dflash and HF's sdpa_attention_forward build an explicit boolean
    # mask for causal/KV-cache decoding instead of using is_causal=True.
    # Anything else (padding, sliding-window, genuinely arbitrary masks)
    # still falls back to stock: TE's fused backends don't accept arbitrary
    # tensors, only named attn_mask_type strings.
    mask_ok = attn_mask is None or (
        not is_causal and te_ops.is_bottom_right_causal_mask(attn_mask)
    )
    if not mask_ok:
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                     is_causal=is_causal, scale=scale, **kwargs)
    try:
        return te_ops.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, scale=scale, is_causal=is_causal
        )
    except (RuntimeError, TypeError):
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                     is_causal=is_causal, scale=scale, **kwargs)


def enable() -> None:
    """Patch torch/torch.nn.functional to route eligible calls through amd_tuned_torch."""
    global _ENABLED
    if _ENABLED:
        return

    # linear/matmul/bmm. Installed when ANY of the three GEMM candidates is
    # available -- aiter's Triton kernels, hipBLASLt, or the CK WMMA GEMM
    # tier -- since each of the three patches runs a contest that includes
    # stock and will simply pick stock if nothing else can serve the shape.
    #
    # This used to be gated on aiter alone, which was right when aiter was
    # the only candidate and wrong the moment it stopped being. On gfx1100
    # aiter_ops.available() is False (it ships no RDNA3 tuning config), so
    # that gate silently withheld the patch entirely -- and with it
    # hipBLASLt's 1.29x and CK's 1.21x -- long after there was something to
    # gain. A capability check has to name the capability it is checking
    # for, not one dependency that used to imply it.
    if aiter_ops.available() or hipblaslt_ops.available() or ck_gemm_ops.available():
        _install(F, "linear", _patched_linear)
        _install(torch, "matmul", _patched_matmul)
        _install(torch, "bmm", _patched_bmm)

    # Native HIP kernels (see src/main_rocm.cpp). Always installed -- the
    # native extension is hard-required at import time (see the ImportError
    # guard above), unlike the aiter-gated and TE-gated patches. conv2d/conv3d
    # try the native kernel first (fp16/fp32; see _patched_conv2d/
    # _patched_conv3d for the tier order and fallbacks).
    _install(F, "group_norm", _patched_group_norm)
    _install(F, "conv2d", _patched_conv2d)
    if hasattr(F, "conv3d"):
        _install(F, "conv3d", _patched_conv3d)

    # TransformerEngine-backed (see amd_tuned_torch/te_ops.py). Skipped entirely if
    # TE isn't installed -- these ops just stay on stock PyTorch/ROCm.
    #
    # F.layer_norm intentionally NOT installed: TE's layernorm_fwd
    # benchmarked slower than stock on RX 7900 XTX for both fp16 (0.71x)
    # and fp32 (0.61x), see benchmark.json -- _patched_layer_norm stays
    # available for manual use.
    if te_ops.available():
        if hasattr(F, "rms_norm"):
            _install(F, "rms_norm", _patched_rms_norm)
        _install(F, "gelu", _patched_gelu)
        _install(F, "silu", _patched_silu)
        if hasattr(F, "scaled_dot_product_attention"):
            _install(F, "scaled_dot_product_attention", _patched_sdpa)

    _ENABLED = True


def disable() -> None:
    """Restore every patched function to the stock PyTorch implementation."""
    global _ENABLED
    for target, name in list(_ORIGINALS.keys()):
        _restore(target, name)
    _ENABLED = False


def is_enabled() -> bool:
    return _ENABLED


# ---------------------------------------------------------------------------
# TorchScript compatibility.
#
# THE PROBLEM. torch.jit.script compiles the Python SOURCE of whatever it
# finds behind a name, so `torch.bmm(a, b)` inside a scripted function
# compiles whatever `torch.bmm` is bound to at that moment. Once enable()
# has run that is a wrapper of ours, and none of these wrappers are
# scriptable -- nor should they be. TorchScript rejects each for a
# different reason:
#
#     torch.bmm, torch.matmul   keyword-only argument with a default (out=None)
#     F.linear, F.group_norm    lambdas (the kernel_select candidate thunks)
#     F.conv2d                  try blocks
#
# and the ones it does not reject it would compile wrongly, since the body
# reaches into a dict keyed by module objects and calls into a pybind
# extension. This is not a defect in any single wrapper; it is what
# monkeypatching costs, and no rewriting of the wrappers fixes it.
#
# It is also not new. F.conv2d has been unscriptable since it was first
# patched -- it simply went unnoticed, because a library has to script code
# that calls a patched op before anything breaks. kornia scripts
# kornia.geometry.boxes at import time and that calls torch.bmm, which
# stayed harmless only for as long as enable() declined to install the
# torch.bmm patch (it was gated on aiter, absent on gfx1100). Fixing that
# gate turned a latent incompatibility into an import-time crash for every
# stable-diffusion-webui launch.
#
# THE FIX. Compile against stock. torch.jit.script and torch.jit.trace are
# wrapped so the patches are lifted for the duration of the call and
# restored after, which makes the compiler bind to the real aten ops. The
# cost is that code inside a scripted or traced region runs stock kernels
# rather than ours -- correct, and the only honest option, since a
# TorchScript graph cannot call a Python dispatch wrapper anyway.
#
# Tracing needs this at least as much as scripting, and more quietly:
# torch.jit.trace records the aten ops a function executes, and these
# wrappers execute pybind calls that are not traceable ops. A trace taken
# with the patches live would bake in a constant instead of a computation
# and be silently wrong, where scripting at least fails loudly.
#
# The guard is installed at import time, not by enable(), because the
# breakage is caused by the patches existing rather than by anyone opting
# in -- and because this module is auto-loaded before user code (see the
# sitecustomize hook), so libraries that script at import time are already
# covered.
_ORIG_JIT_SCRIPT = torch.jit.script
_ORIG_JIT_TRACE = torch.jit.trace


def _script_with_stock_ops(obj, *args, **kwargs):
    """torch.jit.script, compiled against stock ops.

    _frames_up IS NOT OPTIONAL BOOKKEEPING. When the object being scripted
    is a class, torch's _script_impl builds its name-resolution callback
    with createResolutionCallbackFromFrame(_frames_up + 1) -- it walks a
    fixed number of stack frames up to find the caller's module globals.
    This wrapper is one extra frame, so without compensating, every name a
    scripted class refers to is looked up in amd_tuned_torch's namespace
    instead of the caller's. That surfaces as `undefined value <helper>` on
    a perfectly ordinary module-level function, which is how kornia's
    Boxes3D failed the first time this guard was written. Functions resolve
    through createResolutionCallbackFromClosure and are unaffected, which
    is exactly why the bug hides until something scripts a class.

    The adjustment is unconditional: this frame exists whether or not the
    patches were active.
    """
    if len(args) >= 2:
        # positional: script(obj, optimize, _frames_up, ...)
        args = args[:1] + (args[1] + 1,) + args[2:]
    else:
        kwargs["_frames_up"] = kwargs.get("_frames_up", 0) + 1

    if not _ENABLED:
        return _ORIG_JIT_SCRIPT(obj, *args, **kwargs)
    disable()
    try:
        return _ORIG_JIT_SCRIPT(obj, *args, **kwargs)
    finally:
        enable()


def _trace_with_stock_ops(*args, **kwargs):
    """torch.jit.trace, recorded against stock ops.

    No frame compensation needed here: tracing executes the function and
    records the aten ops it runs, rather than resolving names out of the
    caller's namespace.
    """
    if not _ENABLED:
        return _ORIG_JIT_TRACE(*args, **kwargs)
    disable()
    try:
        return _ORIG_JIT_TRACE(*args, **kwargs)
    finally:
        enable()


_script_with_stock_ops.__doc__ = _ORIG_JIT_SCRIPT.__doc__
_trace_with_stock_ops.__doc__ = _ORIG_JIT_TRACE.__doc__
torch.jit.script = _script_with_stock_ops
torch.jit.trace = _trace_with_stock_ops


# ---------------------------------------------------------------------------
# INT8 (W8A8) quantized linear -- aiter-backed, opt-in only (see
# amd_tuned_torch/aiter_ops.py). Unlike everything enable() installs, this changes
# numerics, so it's a separate toggle, never installed automatically and
# never bundled into enable()/disable().
#
# This composes with enable()/disable() regardless of call order: it
# captures whatever F.linear currently *is* (stock, or already
# _patched_linear if enable() ran first) as its own fallback, rather than
# going through the shared _ORIGINALS bookkeeping enable()/disable() use --
# so toggling this on/off never disturbs, and is never disturbed by, the
# main patch set.
# ---------------------------------------------------------------------------

_INT8_LINEAR_ENABLED = False
_int8_linear_fallback: Callable | None = None


def _patched_linear_int8(input, weight, bias=None):
    if not (_grad_safe(input, weight, bias) and _usable(input, weight, dtypes=_INT8_LINEAR_DTYPES)):
        return _int8_linear_fallback(input, weight, bias)
    if input.dim() < 2 or weight.dim() != 2:
        return _int8_linear_fallback(input, weight, bias)
    try:
        return compile_ops.linear_int8(input, weight, bias)
    except (RuntimeError, TypeError, AssertionError):
        return _int8_linear_fallback(input, weight, bias)


def enable_int8_linear() -> None:
    """Opt-in: route F.linear through aiter's W8A8 (int8 activation, int8
    weight) quantized GEMM, which auto-routes to a pure-Triton kernel on
    gfx11/RDNA3 (see amd_tuned_torch/aiter_ops.py). This trades accuracy for speed
    -- unlike every other patch amd_tuned_torch installs, this is NOT numerically
    transparent. No-ops with a warning if aiter isn't installed."""
    global _INT8_LINEAR_ENABLED, _int8_linear_fallback
    if _INT8_LINEAR_ENABLED:
        return
    if not aiter_ops.available():
        import warnings
        warnings.warn("amd_tuned_torch.enable_int8_linear(): aiter not installed, F.linear left as-is")
        return
    _int8_linear_fallback = F.linear
    F.linear = _patched_linear_int8
    _INT8_LINEAR_ENABLED = True


def disable_int8_linear() -> None:
    """Restore whatever F.linear was before enable_int8_linear() -- stock,
    or _patched_linear if the main enable() had already run."""
    global _INT8_LINEAR_ENABLED, _int8_linear_fallback
    if not _INT8_LINEAR_ENABLED:
        return
    F.linear = _int8_linear_fallback
    _int8_linear_fallback = None
    _INT8_LINEAR_ENABLED = False


# ---------------------------------------------------------------------------
# SmoothQuant calibration -- opt-in on top of enable_int8_linear(), never run
# automatically. See amd_tuned_torch/aiter_ops.py's module docstring and the
# SmoothQuant section there for the full derivation; these are thin
# re-exports so `amd_tuned_torch.calibrate_smoothquant(...)` reads the same as
# `amd_tuned_torch.enable_int8_linear()` rather than requiring
# `amd_tuned_torch.aiter_ops.calibrate_smoothquant(...)`.
# ---------------------------------------------------------------------------

def calibrate_smoothquant(forward_fn: Callable[[], Any], alpha: float = 0.5) -> None:
    """Run `forward_fn()` with torch.nn.Linear.forward temporarily patched
    to collect per-input-channel activation statistics, then bake a
    SmoothQuant scale into every weight seen so subsequent linear_int8
    calls (enable_int8_linear() must already be on) route activation
    quantization through aiter's fused smoothquant_quantize kernel instead
    of plain per-token quantization. See amd_tuned_torch/aiter_ops.py for details."""
    aiter_ops.calibrate_smoothquant(forward_fn, alpha=alpha)


# ---------------------------------------------------------------------------
# Conv3d fp16 Winograd -- opt-in only, same "separate toggle, never bundled
# into enable()" shape as enable_int8_linear() above, for the same reason:
# this changes which kernel actually runs, and unlike every tier enable()
# installs, it has never been validated. src/cuda/templates/
# conv3d_fp16_winograd.cu.tmpl (tools/kernelgen/) has not been run,
# correctness-checked, or benchmarked on any hardware -- see that
# template's header. Composes with enable()/disable() regardless of call
# order (captures whatever F.conv3d currently *is* as its own fallback),
# same mechanism as enable_int8_linear().
# ---------------------------------------------------------------------------

_CONV3D_WINOGRAD_FP16_ENABLED = False
_conv3d_winograd_fp16_fallback: Callable | None = None


def _triple(v):
    return (v, v, v) if isinstance(v, int) else tuple(v)


def _is_winograd_eligible_conv3d(input, weight, stride, padding, dilation) -> bool:
    """True only for the exact shape family
    src/cuda/templates/conv3d_fp16_winograd.cu.tmpl's F(2x2x2,3x3x3) kernel
    applies to -- mirrors that template's launcher guard exactly (kernel=
    3x3x3, stride=1, padding=1, dilation=1, batch=1, even D_in/H_in/W_in,
    and a resulting tile-count/C_out both divisible by that kernel's
    BT_TILES/BT_COUT=8 register-blocking factor) so this predicate never
    calls into the kernel for a shape it would decline anyway.

    UNLIKE _is_pointwise_conv2d above, this is NOT informed by a real
    measurement on this hardware -- satisfying this predicate is necessary
    but not sufficient for the Winograd tier to run; see
    enable_conv3d_winograd_fp16's docstring for what to verify first."""
    if input.dim() != 5 or weight.dim() != 5:
        return False
    if input.shape[0] != 1:
        return False
    if tuple(weight.shape[2:]) != (3, 3, 3):
        return False
    if (_triple(stride) != (1, 1, 1) or _triple(padding) != (1, 1, 1)
            or _triple(dilation) != (1, 1, 1)):
        return False
    _, _, D_in, H_in, W_in = input.shape
    if D_in % 2 or H_in % 2 or W_in % 2:
        return False
    # stride=1/padding=1/kernel=3 -> output spatial size equals input's.
    total_tiles = (D_in // 2) * (H_in // 2) * (W_in // 2)
    C_out = weight.shape[0]
    return total_tiles % 8 == 0 and C_out % 8 == 0


def _patched_conv3d_winograd_fp16(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if (groups != 1
            or not _grad_safe(input, weight, bias)
            or not (input.is_contiguous() and weight.is_contiguous())
            or not _usable(input, weight, dtypes=(torch.float16,))
            or not _is_winograd_eligible_conv3d(input, weight, stride, padding, dilation)):
        return _conv3d_winograd_fp16_fallback(input, weight, bias, stride, padding, dilation, groups)
    try:
        out = compile_ops.conv3d_fp16_winograd_bt8_bc8(input, weight, bias, stride, padding, dilation)
        if out is not None:
            return out
    except (RuntimeError, TypeError):
        pass
    return _conv3d_winograd_fp16_fallback(input, weight, bias, stride, padding, dilation, groups)


def enable_conv3d_winograd_fp16() -> None:
    """Opt-in: try the codegen'd F(2x2x2,3x3x3) Winograd fp16 conv3d kernel
    first for eligible shapes (see _is_winograd_eligible_conv3d), falling
    back to whatever F.conv3d currently is -- stock, or _patched_conv3d if
    the main enable() already ran -- for any other shape, or if the kernel
    declines/raises.

    DO NOT call this without first, on your actual RX 7900 XTX:
      1. Running tests_hardware/test_conv_kernels.py (or an equivalent
         direct comparison against F.conv3d, ideally cross-checked in
         fp64) to confirm correctness at a tolerance you're comfortable
         with. Winograd's error profile (few large-magnitude-swing terms
         per output vs. this project's other conv kernels' many small
         ordered accumulations) means their tolerances should NOT be
         assumed to transfer -- determine one empirically.
      2. Running tools/kernelgen/autotune.py to confirm this is actually
         faster than both stock and the native WMMA conv3d kernel for
         your shapes -- nothing here benchmarks it before turning it on.
    This kernel has never been run on any hardware; both checks are
    unverified until you perform them.

    Disable ordering, if you also use the main enable()/disable(): undo
    these LIFO -- call disable_conv3d_winograd_fp16() BEFORE disable(),
    not after. disable() restores F.conv3d from its own bookkeeping
    (_ORIGINALS) unconditionally and clears that bookkeeping; doing so
    while this is still enabled leaves _conv3d_winograd_fp16_fallback
    pointing at an now-orphaned _patched_conv3d, and a later
    disable_conv3d_winograd_fp16() call would re-install that orphaned
    reference into F.conv3d -- which then raises KeyError on its own
    _ORIGINALS lookup the next time anything calls it. (Confirmed by
    triggering it: see TestConv3dWinogradFp16Dispatch.
    test_composes_on_top_of_main_patched_conv3d in
    tests/test_amd_tuned_torch_monkeypatch.py.)"""
    global _CONV3D_WINOGRAD_FP16_ENABLED, _conv3d_winograd_fp16_fallback
    if _CONV3D_WINOGRAD_FP16_ENABLED:
        return
    _conv3d_winograd_fp16_fallback = F.conv3d
    F.conv3d = _patched_conv3d_winograd_fp16
    _CONV3D_WINOGRAD_FP16_ENABLED = True


def disable_conv3d_winograd_fp16() -> None:
    """Restore whatever F.conv3d was before enable_conv3d_winograd_fp16()."""
    global _CONV3D_WINOGRAD_FP16_ENABLED, _conv3d_winograd_fp16_fallback
    if not _CONV3D_WINOGRAD_FP16_ENABLED:
        return
    F.conv3d = _conv3d_winograd_fp16_fallback
    _conv3d_winograd_fp16_fallback = None
    _CONV3D_WINOGRAD_FP16_ENABLED = False


def compute_smoothquant_scale(weight: torch.Tensor, alpha: float = 0.5):
    """The per-input-channel SmoothQuant scale for `weight`, or None if
    calibrate_smoothquant() hasn't collected activation data for it yet."""
    return aiter_ops.compute_smoothquant_scale(weight, alpha=alpha)


def set_smooth_scale(weight: torch.Tensor, smooth_scale: torch.Tensor) -> None:
    """Register a precomputed SmoothQuant scale for `weight` directly,
    bypassing calibrate_smoothquant()'s forward-hook data collection."""
    aiter_ops.set_smooth_scale(weight, smooth_scale)


def is_int8_linear_enabled() -> bool:
    return _INT8_LINEAR_ENABLED


if os.environ.get("AMD_TUNED_TORCH_AUTOPATCH", "1") != "0":
    enable()