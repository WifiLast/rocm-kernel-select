"""
aiter-backed GEMM and conv2d kernels for amd_tuned_torch.

Four tiers, with very different default-on status:

1. **linear_fp16 / bmm_fp16 / conv2d_fp16** -- the DEFAULT fp16/bf16 backend
   for F.linear/torch.matmul/torch.bmm/F.conv2d, installed automatically by
   amd_tuned_torch.enable() (via amd_tuned_torch/__init__.py, gated on `available()`).
   Numerically transparent (same precision-preserving swap as everything
   else enable() installs), backed by aiter's Triton kernels
   (aiter.ops.triton.gemm.basic.gemm_a16w16, .gemm.batched.batched_gemm_bf16,
   .conv.conv2d) -- the GEMM kernels use the gluon backend on gfx1250 and
   Triton (WMMA-targeted) everywhere else including gfx1100/RDNA3, per their
   own docstrings; conv2d is Triton-only with dedicated gfx1100-tuned
   configs (aiter/ops/triton/configs/conv/gfx1100-CONV-*.json). This
   replaced an earlier Composable-Kernel-backed GEMM implementation
   (src/cuda/ck_gemm.cu, since deleted) -- CK is no longer a dependency of
   this package at all.

2. **linear_int8** -- opt-in only, NEVER installed by enable(). Unlike (1),
   int8 quantization changes numerics -- real accuracy loss, not just a
   kernel swap. Opt in explicitly:

       import amd_tuned_torch
       amd_tuned_torch.enable_int8_linear()

   Backed by aiter.gemm_a8w8. On gfx11 (RDNA3 -- RX 7900 XTX / Pro W7900),
   it auto-routes to a pure-Triton kernel
   (aiter/ops/triton/gemm/basic/gemm_a8w8.py): aiter's CK/ASM int8 GEMM
   paths are CDNA-only (gfx9), and gemm_a8w8 itself checks
   `aiter.jit.utils.chip_info.get_gfx_runtime().startswith("gfx9")` before
   ever trying that path -- so calling it unconditionally here is safe, it
   never attempts a CDNA-only code path on RDNA3.

   Only F.linear is covered by linear_int8 -- not torch.matmul/torch.bmm.
   a8w8 GEMM assumes one operand (the weight) is static across calls, worth
   quantizing once and caching; matmul/bmm (e.g. attention's Q @ K.T) have
   no such static operand in the shapes amd_tuned_torch would otherwise intercept,
   so there's nothing to cache and dynamically quantizing both operands
   every call would likely cost more than the int8 GEMM saves.

   Quantization scheme (symmetric, matching aiter's own convention -- see
   aiter.ops.triton.gemm.basic.gemm_a8w8's docstring: "Y = (X @ W^T) *
   (x_scale * w_scale)", scales applied to the int32 output, not the
   inputs):
     - Activation: per-token (per-row) scale, shape (M, 1). Recomputed
       every call -- activations change every forward pass, nothing to
       cache.
     - Weight: per-output-channel scale, shape (1, N). Computed once per
       weight tensor and cached in a WeakKeyDictionary keyed on the weight
       tensor itself, so a freed/replaced weight silently drops out of the
       cache -- no manual invalidation needed for that case. A weight
       updated *in place* (e.g. `layer.weight.data.copy_(new_weights)` to
       hot-swap a LoRA adapter without reallocating the tensor -- object
       identity, and therefore the cache key, doesn't change) is a
       different case that identity-keying alone can't catch: the cache
       also stores the weight's `._version` (PyTorch's own in-place-
       mutation counter, already relied on internally by autograd for this
       same kind of staleness check) and recomputes whenever it no longer
       matches, so a swapped-in-place weight is requantized on its next
       call instead of silently reusing the old quantized bytes forever.

3. **calibrate_smoothquant** -- opt-in on top of (2), never run
   automatically. Adapted from NVIDIA Model-Optimizer's smoothquant()
   calibration algorithm (pure tensor math, no TensorRT/CUDA dependency) --
   migrates per-input-channel dynamic range from activations (which have
   LLM-typical outlier channels) into weights (which don't) before
   quantizing, reducing int8 error for models that need it:

       import amd_tuned_torch
       amd_tuned_torch.enable_int8_linear()
       amd_tuned_torch.calibrate_smoothquant(lambda: run_a_few_batches(model))

   Collects per-input-channel activation amax by temporarily patching
   torch.nn.Linear.forward directly (a module-level patch, not another
   functional one) -- the one place a weight and its corresponding input
   reliably meet regardless of how the surrounding model code invokes it.
   Once calibrated, linear_int8 automatically routes activation
   quantization through aiter's fused smoothquant_quantize kernel instead
   of the plain per-token path. See the SmoothQuant section further down
   for the full derivation.

4. **fused_silu_mul** -- manually-callable helper, NEVER installed by
   enable(). SwiGLU-style gate*up (chunks the last dim, computes
   silu(gate) * up -- the activation LLaMA/Mistral-style SwiGLU MLPs use)
   has no stock F.* op to intercept, unlike linear/matmul/bmm/conv2d
   above, so there's nothing for enable() to auto-patch. Call it directly:

       import amd_tuned_torch
       x = amd_tuned_torch.aiter_ops.fused_silu_mul(gate_up_projection_output)

   Backed by aiter.ops.triton.activation.fused_silu_mul directly -- a
   local import, same as every other function in this file, no network
   dependency. (An earlier version of this integration fetched the same
   function from kernels-community/aiter-kernels on the Hugging Face Hub
   at runtime via amd_tuned_torch/hub_ops.py; that Hub repo turned out to be a
   repackaging of this exact local aiter source, so importing it directly
   here removes the network round-trip entirely -- see hub_ops.py's
   module docstring for that history.)
"""
from __future__ import annotations

import weakref
from typing import Optional

import torch

try:
    import aiter
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16_
    from aiter.ops.triton.gemm.batched.batched_gemm_bf16 import batched_gemm_bf16
    from aiter.ops.triton.conv.conv2d import conv2d as _aiter_conv2d
    from aiter.ops.triton.moe.quant_moe import smoothquant_quantize as _smoothquant_quantize
    from aiter.ops.triton.activation import fused_silu_mul as _fused_silu_mul

    _AITER_AVAILABLE = True
except ImportError:
    aiter = None
    gemm_a16w16_ = None
    batched_gemm_bf16 = None
    _aiter_conv2d = None
    _smoothquant_quantize = None
    _fused_silu_mul = None
    _AITER_AVAILABLE = False


def available() -> bool:
    return _AITER_AVAILABLE


# ---------------------------------------------------------------------------
# fp16/bf16 GEMM -- default GEMM backend, installed by enable().
# ---------------------------------------------------------------------------

def linear_fp16(input_: torch.Tensor, weight: torch.Tensor,
                 bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """fp16/bf16 replacement for F.linear. weight is (N, K) -- aiter's
    gemm_a16w16_ transposes it internally and computes Y = X @ W^T, i.e.
    exactly F.linear's semantics, no pre-transpose needed here."""
    input_dims = input_.shape
    K = input_dims[-1]
    x = input_.contiguous().view(-1, K)
    out = gemm_a16w16_(x, weight, bias=bias, dtype=input_.dtype)
    return out.view(*input_dims[:-1], weight.size(0))


def bmm_fp16(input_: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """fp16/bf16 replacement for torch.bmm/torch.matmul's 3D case. aiter's
    batched_gemm_bf16 computes Y[i] = X[i] @ W[i]^T for a (B, N, K)-shaped
    W -- pass mat2.transpose(-2, -1) as W so the internal transpose cancels
    out, giving plain Y[i] = X[i] @ mat2[i] (torch.bmm's actual semantics,
    no transpose)."""
    x = input_.contiguous()
    w = mat2.transpose(-2, -1).contiguous()
    return batched_gemm_bf16(x, w, dtype=input_.dtype)


def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


def conv2d_fp16(input_: torch.Tensor, weight: torch.Tensor,
                 bias: Optional[torch.Tensor] = None, stride=1, padding=0,
                 dilation=1) -> torch.Tensor:
    """fp16/bf16 replacement for F.conv2d (groups=1 only -- see
    _patched_conv2d in amd_tuned_torch/__init__.py for the groups!=1 fallback).
    aiter's conv2d takes the same NCHW input / [O, I, kH, kW] weight layout
    as F.conv2d, with a shape-driven router (1x1 / 3x3 direct / 3x3
    Winograd / general R x S) picking the kernel per call, and dedicated
    gfx1100-tuned autotune configs
    (aiter/ops/triton/configs/conv/gfx1100-CONV-*.json)."""
    return _aiter_conv2d(input_.contiguous(), weight.contiguous(), bias=bias,
                          stride=_pair(stride), padding=_pair(padding),
                          dilation=_pair(dilation))


# ---------------------------------------------------------------------------
# fused_silu_mul -- manually-callable helper, not installed by enable().
# SwiGLU-style gate*up has no stock F.* op to intercept (F.silu is plain
# elementwise x * sigmoid(x), no chunking, no second operand), so unlike
# linear_fp16/bmm_fp16/conv2d_fp16 above there's nothing for this to
# transparently replace -- call it directly in a SwiGLU-style MLP forward.
# ---------------------------------------------------------------------------

def fused_silu_mul(x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Fused SiLU-and-mul along the last dimension: `x` (even size(-1) =
    2*d) is split into a gate half and an up half, computing
    silu(gate) * up -- the activation LLaMA/Mistral-style SwiGLU MLPs use.
    Writes into `out` if given, otherwise allocates and returns a new
    tensor (aiter.ops.triton.activation.fused_silu_mul's own convention,
    passed straight through unchanged here)."""
    return _fused_silu_mul(x, out=out)


# ---------------------------------------------------------------------------
# INT8 (W8A8) GEMM -- opt-in only, never installed by enable(). See module
# docstring: this changes numerics, unlike linear_fp16/bmm_fp16 above.
# ---------------------------------------------------------------------------

_INT8_MAX = 127.0


class _WeakTensorKeyDict:
    """A weakref.WeakKeyDictionary substitute that's actually safe for
    torch.Tensor keys.

    WeakKeyDictionary keys its internal storage by weakref.ref(key), and
    CPython's weakref.ref equality falls back to `referent1 == referent2`
    whenever two refs land in the same hash bucket -- but torch.Tensor's
    `==` returns an elementwise tensor, not a bool, for anything with more
    than one element. That can't be coerced to a plain bool, so a hash
    *collision alone* between two arbitrary distinct multi-element tensors
    (not two equal ones -- any two landing in the same bucket) crashes with
    "RuntimeError: Boolean value of Tensor with more than one value is
    ambiguous" the moment both are keys in the same WeakKeyDictionary.
    Tensor hashes are identity-based (id()) so this is rare, but real: it's
    exactly what this project's own test suite hit once enough distinct
    weight tensors passed through the same cache in one process, and a real
    model has far more distinct weights than that.

    This keys the underlying plain dict by id(tensor) (a plain int -- no
    custom __eq__ to trip over) and holds a weakref with a finalizer
    callback per entry to evict it when the tensor is garbage collected,
    matching WeakKeyDictionary's auto-eviction behavior without ever
    calling `==` on a tensor for dict bookkeeping.
    """

    def __init__(self):
        self._data: dict[int, tuple[weakref.ref, object]] = {}

    def get(self, key, default=None):
        entry = self._data.get(id(key))
        return entry[1] if entry is not None else default

    def __getitem__(self, key):
        entry = self._data.get(id(key))
        if entry is None:
            raise KeyError(key)
        return entry[1]

    def __contains__(self, key) -> bool:
        return id(key) in self._data

    def __setitem__(self, key, value) -> None:
        key_id = id(key)

        def _on_collected(_ref, data=self._data, key_id=key_id):
            data.pop(key_id, None)

        self._data[key_id] = (weakref.ref(key, _on_collected), value)

    def pop(self, key, default=None):
        entry = self._data.pop(id(key), None)
        return entry[1] if entry is not None else default

    def keys(self):
        """Live tensors currently held, dereferenced from their weakrefs."""
        for ref, _value in list(self._data.values()):
            tensor = ref()
            if tensor is not None:
                yield tensor


_weight_cache = _WeakTensorKeyDict()


def _quantize_activation(x: torch.Tensor):
    """Per-token (per-row) symmetric int8 quantization. x: (M, K) ->
    (int8 (M, K), fp32 scale (M, 1))."""
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / _INT8_MAX
    q = (x.float() / scale).round().clamp(-_INT8_MAX, _INT8_MAX).to(torch.int8)
    return q, scale


def _quantize_weight(weight: torch.Tensor):
    """Per-output-channel symmetric int8 quantization, cached. weight:
    (N, K) -> (int8 (N, K), fp32 scale (1, N)) -- transposed relative to
    the activation scale's (M, 1), per aiter.gemm_a8w8's expected shape.

    If a SmoothQuant scale has been registered for this weight (see
    calibrate_smoothquant() below), the weight is divided by it (per input
    channel, i.e. along dim=-1) before quantizing -- this is the weight side
    of the smoothing swap; the activation side is applied in linear_int8.

    Cache entries carry weight._version alongside the quantized result and
    are recomputed whenever it no longer matches -- see the module
    docstring's weight-caching note for why identity-keying alone (a freed/
    replaced weight tensor) doesn't cover a weight mutated in place (e.g. a
    LoRA swap via `.copy_()`, same object, new values).
    """
    cached = _weight_cache.get(weight)
    if cached is not None and cached[0] == weight._version:
        return cached[1], cached[2]
    w = weight.float()
    smooth_scale = _smooth_scale_cache.get(weight)
    if smooth_scale is not None:
        w = w / smooth_scale.to(device=w.device, dtype=w.dtype)
    amax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)  # (N, 1)
    row_scale = amax / _INT8_MAX  # (N, 1)
    q = (w / row_scale).round().clamp(-_INT8_MAX, _INT8_MAX).to(torch.int8)
    scale = row_scale.t().contiguous()  # (1, N)
    _weight_cache[weight] = (weight._version, q, scale)
    return q, scale


def linear_int8(input_: torch.Tensor, weight: torch.Tensor,
                 bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """W8A8 replacement for F.linear. See module docstring for the
    quantization scheme and why only F.linear (not matmul/bmm) is covered.

    If calibrate_smoothquant() has registered a scale for this weight,
    activations are quantized through aiter's fused smoothquant_quantize
    kernel (single Triton launch, vs. _quantize_activation's four separate
    elementwise ops) instead of the plain per-token quantization -- see the
    SmoothQuant section below for why smoothing exists and how the
    activation/weight sides cancel out exactly before quantization error is
    introduced.
    """
    input_dims = input_.shape
    K = input_dims[-1]
    x = input_.contiguous().view(-1, K)

    smooth_scale = _smooth_scale_cache.get(weight)
    if smooth_scale is not None and _smoothquant_quantize is not None:
        xq, x_scale = _smoothquant_quantize(x, smooth_scale.to(device=x.device, dtype=torch.float32))
        x_scale = x_scale.view(-1, 1)
    else:
        xq, x_scale = _quantize_activation(x)
    wq, w_scale = _quantize_weight(weight)

    out = aiter.gemm_a8w8(xq, wq, x_scale, w_scale, bias=bias, dtype=input_.dtype)

    return out.view(*input_dims[:-1], weight.size(0))


# ---------------------------------------------------------------------------
# SmoothQuant -- opt-in calibration on top of linear_int8, adapted from
# NVIDIA Model-Optimizer's smoothquant() (modelopt/torch/quantization/
# model_calib.py) and the alpha-interpolated scale formula it uses:
#
#     smooth_scale = weight_amax^(1 - alpha) / act_amax^alpha   (per input channel)
#
# The idea (Xiao et al., SmoothQuant): LLM activations have per-channel
# outliers that dominate int8 quantization error; weights don't. Migrating
# some of that dynamic range from activations into weights before
# quantizing -- x_smooth = x * smooth_scale, w_smooth = w / smooth_scale --
# leaves the matmul's *unquantized* result exactly unchanged
# (x_smooth @ w_smooth^T == x @ w^T, verify: (x*s) @ (w/s)^T = x @ w^T) but
# shifts which operand absorbs the rounding error, reducing it overall for
# outlier-heavy activations. This is pure tensor arithmetic, not a
# TensorRT/CUDA-specific technique -- and aiter already ships the consumer
# kernel for the activation side (aiter.ops.triton.moe.quant_moe.
# smoothquant_quantize computes exactly "x * smooth_scale, then per-row
# int8 quantize" in one Triton launch, per its own docstring).
#
# What's NOT ported from Model-Optimizer: its calibration machinery wraps
# every quantized nn.Linear in a TensorQuantizer module and hooks forward()
# there -- doesn't fit amd_tuned_torch, which patches functional ops, not modules.
# calibrate_smoothquant() below is the amd_tuned_torch-appropriate equivalent: it
# temporarily patches torch.nn.Linear.forward directly (the one place a
# weight tensor and its corresponding input reliably meet up, independent
# of whether the surrounding model code calls F.linear -- already covered
# by amd_tuned_torch.enable() -- or holds some other reference to it) to collect
# per-input-channel activation amax keyed by weight identity, the same
# WeakKeyDictionary pattern _weight_cache already uses.
# ---------------------------------------------------------------------------

# _WeakTensorKeyDict, not weakref.WeakKeyDictionary -- see that class's
# docstring above _weight_cache for why a plain WeakKeyDictionary keyed on
# tensors is unsafe.
_calib_act_amax = _WeakTensorKeyDict()
_smooth_scale_cache = _WeakTensorKeyDict()


def _update_calib_amax(weight: torch.Tensor, input_: torch.Tensor) -> None:
    x = input_.reshape(-1, input_.shape[-1]).float()
    amax = x.abs().amax(dim=0)  # per input channel, (K,)
    prev = _calib_act_amax.get(weight)
    _calib_act_amax[weight] = amax if prev is None else torch.maximum(prev, amax.to(prev.device))


def compute_smoothquant_scale(weight: torch.Tensor, alpha: float = 0.5) -> Optional[torch.Tensor]:
    """SmoothQuant's per-input-channel migration scale for `weight`, using
    activation amax collected by calibrate_smoothquant(). Returns None if no
    calibration data has been collected for this weight yet.

    `alpha` trades off how much dynamic range moves from activations (higher
    alpha) to weights (lower alpha); 0.5 is SmoothQuant's own default and
    matches Model-Optimizer's.
    """
    act_amax = _calib_act_amax.get(weight)
    if act_amax is None:
        return None
    weight_amax = weight.detach().float().abs().amax(dim=0).clamp(min=1e-8)  # per input channel, (K,)
    act_amax = act_amax.to(device=weight.device).clamp(min=1e-8)
    scale = weight_amax.pow(1 - alpha) / act_amax.pow(alpha)
    return scale.clamp(min=1e-4, max=1e4)


def set_smooth_scale(weight: torch.Tensor, smooth_scale: torch.Tensor) -> None:
    """Register `smooth_scale` (per input channel, shape (K,)) for `weight`.
    Evicts any already-cached (unsmoothed) quantized weight for it, so the
    next linear_int8 call recomputes _quantize_weight with smoothing
    applied instead of silently serving a stale cache entry."""
    _smooth_scale_cache[weight] = smooth_scale
    _weight_cache.pop(weight, None)


def calibrate_smoothquant(forward_fn, alpha: float = 0.5) -> None:
    """Run `forward_fn()` (e.g. a handful of representative batches through
    the model) with torch.nn.Linear.forward temporarily patched to collect
    per-input-channel activation amax for every nn.Linear encountered, then
    bake a SmoothQuant scale into every weight seen (see set_smooth_scale)
    so subsequent linear_int8 calls use it automatically.

    This patches the torch.nn.Linear *class*, not a global function -- it's
    the one place guaranteed to see both a weight tensor and the exact
    input it's about to be multiplied by, regardless of whether the model
    calls F.linear directly (already covered by amd_tuned_torch.enable()) or a
    custom module wraps it some other way. Only base torch.nn.Linear
    instances are covered; a subclass overriding forward() bypasses this
    (and amd_tuned_torch.enable()'s F.linear patch too, for that matter) same as
    any other monkeypatch-based approach would.

    Restores torch.nn.Linear.forward afterward even if forward_fn() raises.
    """
    if _smoothquant_quantize is None:
        import warnings
        warnings.warn(
            "amd_tuned_torch.calibrate_smoothquant(): aiter's smoothquant_quantize "
            "kernel isn't available -- calibration will still collect "
            "amax and set scales, but linear_int8 will have nothing to "
            "route them through."
        )

    orig_forward = torch.nn.Linear.forward

    def _calibrating_forward(self, input):
        _update_calib_amax(self.weight, input)
        return orig_forward(self, input)

    torch.nn.Linear.forward = _calibrating_forward
    try:
        forward_fn()
    finally:
        torch.nn.Linear.forward = orig_forward

    for weight in list(_calib_act_amax.keys()):
        scale = compute_smoothquant_scale(weight, alpha=alpha)
        if scale is not None:
            set_smooth_scale(weight, scale)
