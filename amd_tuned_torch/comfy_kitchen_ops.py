"""comfy-kitchen's HIP kernels, wrapped as decline-or-deliver candidates for
the always-on contests in _dispatch.py.

comfy-kitchen (third_party/comfy-kitchen) is Comfy Org's diffusion kernel
library. Its `hip` backend is the reason it is worth wiring in here: every
matmul in it is a WMMA kernel compiled from backends/hip/*.h, and it neither
links nor calls hipBLAS/hipBLASLt/CK. So each candidate below is an
ALGORITHMICALLY DIFFERENT implementation from anything this package already
contests -- which is the only kind of candidate kernel_select can do anything
useful with. gfx1100 is in that backend's `wmma_gfx11` manifest group, so the
GEMM-bearing kernels register here, not just the elementwise ones.

Four entry points, one per gap in the existing tier set:

  layer_norm / rms_norm   The one place this adds a tier where there is
                          currently NOTHING. _te_norm_worth_it() declines
                          every forward-only norm by design (TE's win lives in
                          the backward), so at inference F.layer_norm and
                          F.rms_norm are pure pass-through to stock today.
                          comfy-kitchen's adaln/rms_adaln are inference-only
                          kernels that are algebraically a superset of affine
                          normalization:

                              adaln(x, s, b)     = layernorm(x) * (1 + s) + b
                              rms_adaln(x, s, b) = rmsnorm(x)   * (1 + s) + b

                          so F.layer_norm(x, ns, w, b) is adaln(x, w - 1, b)
                          and F.rms_norm(x, ns, w) is rms_adaln(x, w - 1, 0).

  conv3d                  A WMMA implicit-GEMM conv3d with bias fused into the
                          epilogue -- a fourth algorithm beside the native HIP,
                          CK and FFT-conv tiers.

  linear                  A WMMA fp16 GEMM with an fp32 accumulator. Deep
                          reductions only -- the kernel declines K <= 4096.

  sdpa_decode             Exact flash decode attention for q_len == 1, the
                          shape PyTorch's own backends handle worst and the one
                          _patched_sdpa sends to stock unconditionally today.

WHY EVERY FUNCTION HERE RETURNS None RATHER THAN RAISING. Same contract as
ck_ops/ck_norm_ops: a candidate that cannot serve a call must lose the contest,
not break it. comfy-kitchen's own public API does the opposite -- its
`fp16_linear`, `fp16_conv3d` and `group_norm_silu_pad3d` fall back to
torch/eager internally when a shape is outside the kernel's domain -- which is
the right default for a standalone library and exactly wrong here, for two
reasons:

  1. Its eager fallback is SLOWER than the stock op _dispatch.py would
     otherwise reach, so silently taking it would make the tier a pure loss
     that kernel_select would still record as a "win" over nothing.
  2. Its torch fallback calls torch.nn.functional.linear / .conv3d -- which,
     inside this package, IS the monkeypatched wrapper that called us.
     Unguarded, that is infinite recursion, not a slow path.

So this module calls the HIP backend's own functions directly (and, for conv3d,
its private _wmma_fp16_conv3d, which is the decline-returning form with no
torch fallback at all), re-checks their internal support predicates here before
calling, and additionally holds a re-entrancy guard (_guard) so that a fallback
reached anyway -- via a shape predicate that drifts upstream, or the `served`
flag a kernel can still return false at launch time -- degrades to one wasted
attempt and a correct stock result rather than a blown stack.

BUILD. The HIP extension is optional and NOT built by installing this package:

    COMFY_HIP_ARCHS=gfx1100 pip install third_party/comfy-kitchen

(CMake >= 3.26 and Ninja required.) Without it, available() is False, every
function here returns None on its first line, and dispatch behaves exactly as
it did before this module existed. AMD_TUNED_TORCH_COMFY_KITCHEN=0 forces that
same state with the extension present.
"""
from __future__ import annotations

import os
import threading
import weakref
from typing import Any, Optional

import torch

_ENV_ENABLED = os.environ.get("AMD_TUNED_TORCH_COMFY_KITCHEN", "1") not in (
    "0", "", "false", "False")

# AMD_TUNED_TORCH_COMFY_GEMM=0 removes the two WMMA GEMM tiers (linear,
# conv3d) while leaving the norms and decode attention in place -- one switch
# per group of kernels, so a bad result from one does not cost the others.
_GEMM_ENABLED = os.environ.get("AMD_TUNED_TORCH_COMFY_GEMM", "1") not in (
    "0", "", "false", "False")

# launch_fp16_gemm_kernel's own envelope (ops/gemm_fp16.hip), re-stated here so
# a call outside it declines without first allocating an output tensor for a
# kernel that will refuse it. The kernel's `served` flag stays the authority;
# this only avoids paying for the obvious no.
_GEMM_MIN_K = 4096   # exclusive: K must be > this
_GEMM_MAX_M = 8192

_NORM_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_ck: Any = None    # the comfy_kitchen package
_hip: Any = None   # comfy_kitchen.backends.hip
_AVAILABLE: Optional[bool] = None
_UNAVAILABLE_REASON = "not probed yet"

# Re-entrancy: see the module docstring. Thread-local because the guard is
# about THIS thread's call stack -- another thread legitimately dispatching
# into comfy-kitchen at the same time must not be declined.
_reentry = threading.local()


def available() -> bool:
    """True if comfy-kitchen's HIP backend registered on this device.

    Probed once and cached, including the negative: the extension is loaded at
    comfy_kitchen import time and the visible device set is fixed for the life
    of the process, so a "no" here can never become a "yes" later. False is
    normal, not an error -- the extension is optional at build time, same as
    the CK tier.
    """
    global _AVAILABLE, _ck, _hip, _UNAVAILABLE_REASON
    if _AVAILABLE is not None:
        return _AVAILABLE
    _AVAILABLE = False
    if not _ENV_ENABLED:
        _UNAVAILABLE_REASON = "disabled by AMD_TUNED_TORCH_COMFY_KITCHEN=0"
        return False
    if not getattr(torch.version, "hip", None):
        # comfy_kitchen imports its HIP backend only under a ROCm torch, and
        # its CUDA/triton/eager backends are of no use to this package.
        _UNAVAILABLE_REASON = "not a ROCm PyTorch build"
        return False
    try:
        import comfy_kitchen as _pkg
        from comfy_kitchen.backends import hip as _hip_backend
        from comfy_kitchen.registry import registry
    except Exception as exc:  # ImportError, or a broken extension at load
        _UNAVAILABLE_REASON = "import failed: %r" % (exc,)
        return False
    if not registry.is_available("hip"):
        try:
            _UNAVAILABLE_REASON = (
                registry.list_backends().get("hip", {}).get("unavailable_reason")
                or "HIP backend did not register")
        except Exception:
            _UNAVAILABLE_REASON = "HIP backend did not register"
        return False
    _ck, _hip = _pkg, _hip_backend
    _AVAILABLE = True
    _UNAVAILABLE_REASON = ""
    return True


def unavailable_reason() -> str:
    """Why available() is False, for the one-line backend report enable()
    logs. Empty string when the tier IS available."""
    available()
    return _UNAVAILABLE_REASON


def has_wmma() -> bool:
    """True if every visible device has matrix cores (RDNA3/3.5/4).

    The GEMM-bearing kernels -- fp16_linear, fp16_conv3d, and the bf16 decode
    attention -- need this; adaln/rms_adaln do not. comfy-kitchen takes the
    INTERSECTION over visible devices, which is the right answer for this box:
    the NVIDIA card is invisible to ROCm, but a second AMD card without matrix
    cores would have to disqualify the GEMMs, since kernels launch on the
    tensor's own device.
    """
    if not available():
        return False
    try:
        return bool(_hip.has_wmma())
    except Exception:
        return False


def _gemm_enabled() -> bool:
    """Whether the two WMMA GEMM tiers may be offered.

    NOT gated on torch.backends.cuda.matmul.allow_fp16_accumulation, and that
    is a deliberate correction rather than an omission. comfy-kitchen's public
    fp16_linear/fp16_conv3d document themselves as carrying that flag's
    numerics, which is true of the CUTLASS path they were written for; the HIP
    kernels this package actually reaches say the opposite in their own
    sources -- "the accumulators are fp32, so this is at least as accurate as
    the fp16-accumulate contract fp16_linear promises" (ops/gemm_fp16.hip) and
    "Accumulation is fp32" (ops/conv3d_fp16.hip). They are fp32-accumulate
    WMMA kernels, numerically on a par with stock, so withholding them until
    the caller asks for reduced precision would withhold them from everyone
    while buying nothing.

    What enforces the contract instead is stronger than an env flag:
    kernel_select verifies every candidate against stock's own output on the
    first call for each shape, at the per-dtype bar in kernel_select._TOLERANCES
    (1e-2 for fp16). If a future comfy-kitchen build ever did ship an
    fp16-accumulate HIP GEMM, its ~sqrt(K) * 2^-11 error -- about 3% at
    K=4096 -- would fail that bar, the candidate would be blacklisted for the
    shape, and dispatch would go on using stock. The check is per shape, on
    real data, and it cannot be forgotten at upgrade time.
    """
    return _GEMM_ENABLED


class _guard:
    """Marks this thread as being inside a comfy-kitchen call.

    Every entry point tests _busy() first, so a comfy-kitchen internal fallback
    that lands back on a monkeypatched F.linear/F.conv3d finds the candidate
    declining and gets the stock op -- one wasted attempt, a correct result,
    and no recursion. See the module docstring.
    """

    def __enter__(self):
        _reentry.depth = getattr(_reentry, "depth", 0) + 1
        return self

    def __exit__(self, *exc):
        _reentry.depth -= 1
        return False


def _busy() -> bool:
    return getattr(_reentry, "depth", 0) > 0


def _triple(v) -> tuple:
    """(a, b, c) from an int or any 3-sequence. Local copy rather than
    _dispatch._triple: _dispatch imports THIS module, so importing back would
    be a cycle."""
    return (v, v, v) if isinstance(v, int) else tuple(v)


# ---------------------------------------------------------------------------
# Normalization: F.layer_norm / F.rms_norm via adaln / rms_adaln.
# ---------------------------------------------------------------------------

# weight -> (weakref, version, weight - 1 in fp32). See _scale_from_weight.
_SCALE_CACHE: dict = {}
_SCALE_CACHE_MAX = 512


def _scale_from_weight(weight: torch.Tensor) -> torch.Tensor:
    """`weight - 1` as fp32, cached per weight tensor.

    FP32 IS NOT OPTIONAL HERE, it is the whole correctness argument for routing
    an affine norm through a modulation kernel. The kernel recomputes
    weight as (1 + scale), so any error in `weight - 1` lands directly on the
    weight. Computed in fp16 that subtraction is exact only while weight >= 1
    (Sterbenz), and catastrophically lossy below it: a weight of 0.001 gives
    -0.999, whose fp16 neighbours are 0.00049 apart, so the reconstructed
    weight carries up to 24% relative error. In fp32 the round trip is exact
    to within fp32 epsilon for every fp16/bf16 weight. The kernel takes an
    independent dtype code per operand (see dlpack_bindings.cpp's adaln_impl),
    so an fp32 scale against an fp16 x is a supported combination, not a
    coincidence.

    Cached because it is a per-weight constant recomputed on every call
    otherwise, and a norm at these sizes is only tens of microseconds. The
    weakref, not id() alone, is what makes the cache safe: a freed weight's
    address can be handed straight back to another tensor by the caching
    allocator, and `ref() is weight` is the only test that cannot be fooled by
    that. `_version` catches an in-place mutation (a merged LoRA) of a weight
    that is still alive.
    """
    key = id(weight)
    entry = _SCALE_CACHE.get(key)
    if entry is not None:
        ref, version, scale = entry
        if ref() is weight and version == weight._version:
            return scale
    scale = weight.detach().float() - 1.0
    if len(_SCALE_CACHE) >= _SCALE_CACHE_MAX:
        # Plain clear rather than LRU eviction: this is a cache of tiny [D]
        # vectors that only fills up in a process cycling through hundreds of
        # distinct norm weights, and rebuilding it costs one elementwise op per
        # weight.
        _SCALE_CACHE.clear()
    _SCALE_CACHE[key] = (weakref.ref(weight), weight._version, scale)
    return scale


_ZERO_CACHE: dict = {}


def _zeros(width: int, device: torch.device) -> torch.Tensor:
    """fp32 zeros [width] -- rms_norm's missing `shift`, cached per (width,
    device) since F.rms_norm has no bias at all and every call needs one."""
    key = (width, device.type, device.index)
    z = _ZERO_CACHE.get(key)
    if z is None:
        if len(_ZERO_CACHE) >= 64:
            _ZERO_CACHE.clear()
        z = torch.zeros(width, dtype=torch.float32, device=device)
        _ZERO_CACHE[key] = z
    return z


def _norm_eligible(input: torch.Tensor, normalized_shape, weight: torch.Tensor,
                   bias: Optional[torch.Tensor]) -> bool:
    """Whether adaln/rms_adaln can stand in for this exact norm call.

    The binding restriction that matters: adaln normalizes over the LAST
    dimension only, while F.layer_norm's `normalized_shape` may span several
    trailing dims. A multi-dim normalized_shape is a different reduction, not a
    reshape of this one, so it declines rather than being flattened.
    """
    if input.dim() < 1 or input.numel() == 0:
        return False
    width = input.shape[-1]
    if isinstance(normalized_shape, int):
        if normalized_shape != width:
            return False
    else:
        ns = tuple(normalized_shape)
        if len(ns) != 1 or ns[0] != width:
            return False
    if weight.dim() != 1 or weight.numel() != width or weight.device != input.device:
        return False
    if bias is not None and (bias.dim() != 1 or bias.numel() != width
                             or bias.device != input.device
                             or bias.dtype not in _NORM_DTYPES):
        return False
    return True


def layer_norm(input: torch.Tensor, normalized_shape, weight: Optional[torch.Tensor],
               bias: Optional[torch.Tensor], eps: float) -> Optional[torch.Tensor]:
    """F.layer_norm through comfy-kitchen's adaln. None if out of domain.

    INFERENCE ONLY -- the kernel has no backward. The caller checks that; see
    _dispatch._comfy_norm.
    """
    if not available() or _busy():
        return None
    if weight is None or bias is None:
        return None
    if input.dtype not in _NORM_DTYPES or not input.is_cuda:
        return None
    if not _norm_eligible(input, normalized_shape, weight, bias):
        return None
    try:
        with _guard():
            return _hip.adaln(input, _scale_from_weight(weight), bias.detach(), float(eps))
    except (RuntimeError, TypeError, ValueError):
        return None


def rms_norm(input: torch.Tensor, normalized_shape, weight: Optional[torch.Tensor],
             eps: Optional[float]) -> Optional[torch.Tensor]:
    """F.rms_norm through comfy-kitchen's rms_adaln. None if out of domain.

    `eps=None` is resolved to torch's own default (finfo(dtype).eps) rather
    than to the kernel's 1e-6 default: the point of this tier is to compute
    what F.rms_norm would have computed, and at fp16 those two epsilons differ
    by three orders of magnitude.
    """
    if not available() or _busy():
        return None
    if weight is None:
        return None
    if input.dtype not in _NORM_DTYPES or not input.is_cuda:
        return None
    if not _norm_eligible(input, normalized_shape, weight, None):
        return None
    if eps is None:
        eps = torch.finfo(input.dtype).eps
    try:
        with _guard():
            return _hip.rms_adaln(input, _scale_from_weight(weight),
                                  _zeros(input.shape[-1], input.device), float(eps))
    except (RuntimeError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# conv3d and linear: the fp16-accumulate WMMA tier.
# ---------------------------------------------------------------------------

def conv3d(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
           stride, padding, dilation) -> Optional[torch.Tensor]:
    """F.conv3d through comfy-kitchen's WMMA implicit-GEMM conv3d.

    fp16 only, dilation 1, groups 1 (the caller has already excluded grouped
    conv). The kernel itself takes zero padding only, so a padded conv is
    materialized here with an explicit F.pad -- which is what the kernel would
    have done internally anyway, and is inside the thunk kernel_select times,
    so the contest sees its real cost.

    THE OUTPUT LAYOUT IS CONVERTED BACK, and that is not a detail. The kernel
    is channels-last-native: it takes channels_last_3d in and hands
    channels_last_3d out. Returning that from a patched F.conv3d would leak a
    layout change into the rest of the graph, and on gfx1100 a channels-last
    tensor arriving at a following conv is not a small regression -- measured
    elsewhere in this project at 239-753x. kernel_select would happily cache
    this candidate as the winner of ITS shape while the op after it paid for
    the decision, which is exactly the locally-optimal, globally-catastrophic
    outcome a per-op contest cannot see. So the conversion is done here, and
    timed here.
    """
    if not available() or not has_wmma() or _busy():
        return None
    if not _gemm_enabled():
        return None
    if input.dtype is not torch.float16 or weight.dtype is not torch.float16:
        return None
    if input.dim() != 5 or weight.dim() != 5:
        return None
    if weight.device != input.device or not input.is_cuda:
        return None
    if bias is not None and (bias.dtype is not torch.float16 or bias.dim() != 1
                             or bias.numel() != weight.shape[0]):
        return None
    if isinstance(padding, str):
        return None  # 'same'/'valid': resolved by torch, not expressible here
    try:
        s, p, d = _triple(stride), _triple(padding), _triple(dilation)
    except TypeError:
        return None
    if any(x != 1 for x in d) or any(x < 1 for x in s) or any(x < 0 for x in p):
        return None
    channels_last_in = input.is_contiguous(memory_format=torch.channels_last_3d)
    x = input
    if any(p):
        # F.pad's 5D order is (w_l, w_r, h_l, h_r, d_l, d_r); conv3d's padding
        # is (d, h, w) applied symmetrically. F.pad is not patched by this
        # package, so this cannot re-enter dispatch.
        x = torch.nn.functional.pad(x, (p[2], p[2], p[1], p[1], p[0], p[0]))
    try:
        with _guard():
            # _wmma_fp16_conv3d, NOT the public fp16_conv3d: the private form
            # returns None for a shape it cannot serve, where the public one
            # silently runs torch's conv3d -- which here is our own patched
            # wrapper. See the module docstring.
            out = _hip._wmma_fp16_conv3d(x, weight, bias, None, list(s))
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return None
    if out is None:
        return None
    return out if channels_last_in else out.contiguous()


def linear(input: torch.Tensor, weight: torch.Tensor,
           bias: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """F.linear through comfy-kitchen's fp32-accumulate WMMA GEMM.

    THIS DRIVES _C.fp16_gemm DIRECTLY RATHER THAN THE BACKEND'S fp16_linear,
    and that is the single most important thing about this function. The kernel
    refuses a shape by returning a `served` flag of false, and fp16_linear
    answers that by quietly running torch.nn.functional.linear and returning
    the result -- indistinguishable, to its caller, from a kernel that ran.
    Inside this package that is wrong twice over: the torch call re-enters our
    own patched F.linear, and, worse, the candidate would then NEVER decline,
    so kernel_select would time a wrapper around stock, find it a hair slower
    than stock, and record a verdict about a kernel that never executed.
    Reading `served` is what makes this candidate honest. (conv3d needs no
    equivalent: _wmma_fp16_conv3d already returns None on a false `served`,
    which is why it, and not the public fp16_conv3d, is what conv3d() calls.)

    The declines below are the kernel's own envelope (ops/gemm_fp16.hip): deep
    reductions only -- K > 4096 and a multiple of 8, N a multiple of 8,
    M <= 8192. That is a narrow door and worth knowing before going looking for
    this tier in a trace: a diffusion U-Net's K is typically 320-2560 and never
    reaches it. Where it does apply is an LLM's MLP projections (K = 11008 and
    up), which is where _patched_linear's existing candidates are weakest.
    """
    if not available() or not has_wmma() or _busy():
        return None
    if not _gemm_enabled():
        return None
    if input.dtype is not torch.float16 or weight.dtype is not torch.float16:
        return None
    if input.dim() < 2 or weight.dim() != 2:
        return None
    if weight.device != input.device or not input.is_cuda:
        return None
    if bias is not None and (bias.dtype is not torch.float16
                             or bias.device != input.device):
        return None
    n, k = weight.shape
    if input.shape[-1] != k:
        return None
    m = input.numel() // k
    if k <= _GEMM_MIN_K or k % 8 or n % 8 or m > _GEMM_MAX_M or m == 0 or n == 0:
        return None
    x2 = (input if input.dim() == 2 and input.is_contiguous()
          else input.reshape(-1, k).contiguous())
    # The tile stager issues 16-byte row loads, so a misaligned slice is
    # exactly the case fp16_linear would have sent to its torch fallback.
    if x2.data_ptr() % 16:
        return None
    w = weight if weight.is_contiguous() else weight.contiguous()
    if w.data_ptr() % 16:
        return None
    try:
        with _guard():
            out = torch.empty((m, n), dtype=torch.float16, device=x2.device)
            bias_arg = (None if bias is None
                        else _hip._vector_operand(bias, x2.device, torch.float16))
            served = _hip._C.fp16_gemm(
                _hip._dl(x2), _hip._dl(w), _hip._dl(out),
                None if bias_arg is None else _hip._dl(bias_arg),
                None, None, m, n, k, _hip._stream(x2))
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return None
    if not served:
        return None
    return out.reshape(*input.shape[:-1], n)


# ---------------------------------------------------------------------------
# Decode attention.
# ---------------------------------------------------------------------------

# 1/sqrt(128) is compiled into the kernel (kSoftmaxScaleLog2 in
# ops/flash_decode.hip), so a caller-supplied scale can only be honoured by
# being the same number.
_DECODE_HEAD_DIM = 128
_DECODE_SCALE = _DECODE_HEAD_DIM ** -0.5


def sdpa_decode(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Any, is_causal: bool, scale: Optional[float],
                enable_gqa: bool) -> Optional[torch.Tensor]:
    """Exact flash decode attention for a single query position. None otherwise.

    Fills the hole _patched_sdpa leaves at inference: PyTorch's own backends
    are at their weakest at q_len == 1 (one query row cannot occupy the card,
    whichever backend takes it), and every forward-only SDPA call currently
    goes to stock because TE only wins in the backward.

    NO KV COPY, which is what makes this worth doing at all. SDPA hands q/k/v
    as (B, H, S, D) and this kernel wants (B, S, H, D), but it addresses k and
    v through their own strides (the binding requires only that head_dim be
    contiguous), so the transposed view is read where it lies. Copying the KV
    cache to satisfy a layout would cost the whole win, every token.

    is_causal declines rather than being honoured: with q_len == 1 PyTorch
    aligns the causal mask TOP-LEFT, so is_causal=True there means "attend key
    0 only" -- not "attend the whole cache", which is what this computes and
    what a decoding caller means. Silently computing the useful thing instead
    of the specified thing is the one failure mode a transparent dispatch layer
    must not have.
    """
    if not available() or _busy():
        return None
    if attn_mask is not None or is_causal:
        return None
    if scale is not None and scale != _DECODE_SCALE:
        return None
    if not (query.dtype is torch.bfloat16 and key.dtype is torch.bfloat16
            and value.dtype is torch.bfloat16):
        return None
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        return None
    if not query.is_cuda or key.device != query.device or value.device != query.device:
        return None
    batch, q_heads, q_len, head_dim = query.shape
    if q_len != 1 or head_dim != _DECODE_HEAD_DIM:
        return None
    if key.shape != value.shape or key.shape[0] != batch or key.shape[3] != head_dim:
        return None
    kv_heads, kv_len = key.shape[1], key.shape[2]
    if kv_heads == 0 or q_heads % kv_heads:
        return None
    if q_heads != kv_heads and not enable_gqa:
        # Plain SDPA requires matching head counts; only enable_gqa asks for
        # the broadcast this kernel would apply.
        return None
    try:
        if not _hip.flash_attention_decode_is_available():
            return None
    except Exception:
        return None
    kv_lengths = torch.full((batch,), kv_len, dtype=torch.int32, device=query.device)
    try:
        with _guard():
            out = _ck.flash_attention_decode(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                kv_lengths)
    except (RuntimeError, TypeError, ValueError):
        return None
    return out.transpose(1, 2)
