"""
TransformerEngine-backed replacements for attention/layer_norm/rms_norm/
gelu/silu, used by amd_tuned_torch's monkeypatch layer on ROCm.

Unlike linear/matmul/bmm/group_norm (native HIP/CK, see src/main_rocm.cpp),
these ops are NOT reimplemented here -- TransformerEngine's ROCm fork already
ships tuned CK- and AOTriton-backed fused kernels for exactly these five ops
(fused attention with automatic CK/AOTriton backend selection; fused
LayerNorm/RMSNorm/GELU/SiLU), so this module is a thin, autograd-capable
adapter from PyTorch's stateless `F.foo(tensor, ...)` calling convention onto
TE's lowest-level stateless bindings (`transformer_engine_torch`, imported as
`tex`) for the norms/activations, and onto TE's public `DotProductAttention`
module (cached per shape) for attention.

This is a real improvement over the original CMP-Turing kernels, which had no
backward pass at all (see amd_tuned_torch/__init__.py's SAFETY note) and always fell
back to stock ops under autograd. Every wrapper here defines a proper
torch.autograd.Function, so these stay active for training, not just
inference.

Attention also accepts an explicit `attn_mask` tensor, not just `is_causal`,
but only routes it to TE's fused path when it structurally matches the
bottom-right-aligned causal pattern (see is_bottom_right_causal_mask() below)
-- HF model code and dflash's model.py both express plain/KV-cache causal
masking as an explicit boolean tensor rather than `is_causal=True`, so without
this, amd_tuned_torch's SDPA patch would never fire for them. Anything else
(padding, sliding-window, genuinely arbitrary masks) still falls back to
stock: TE's fused backends only accept named attn_mask_type strings, and
forwarding an arbitrary tensor would force attn_mask_type="arbitrary", which
disables TE's fused backend entirely -- often slower than just leaving the
call on stock PyTorch.

Everything in this module degrades to "unavailable" (never raises at import
time) if `transformer_engine` isn't installed -- amd_tuned_torch/__init__.py checks
`te_ops.available()` before installing any of these patches and falls back to
stock PyTorch ops otherwise.

TE DISABLED BY DEFAULT
-----------------------
`import transformer_engine.pytorch` is not merely allowed to fail cleanly --
a TE build that's ABI-mismatched against the installed PyTorch/ROCm (a
depressingly common state: TE has to be rebuilt from source every time
PyTorch or ROCm changes, see source/TransformerEngine/README.md) can
**segfault the whole process on import**, not raise a catchable
ImportError. A `try/except ImportError` is powerless against that -- the
crash happens inside the native extension's module-init code, before
Python's exception machinery ever gets a chance to run.

So the import below only runs at all if AMD_TUNED_TORCH_ENABLE_TE=1 is set in the
environment BEFORE amd_tuned_torch is imported. Left unset (the default),
`transformer_engine` is never touched, `available()` returns False, and
`amd_tuned_torch.enable()` silently skips the five TE-backed patches -- `import
amd_tuned_torch` itself can never crash because of a broken TE install. Verify TE
imports cleanly on its own first (`python -c "import
transformer_engine.pytorch"`) before opting in.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

_TE_ENABLED_ENV = os.environ.get("AMD_TUNED_TORCH_ENABLE_TE", "0") not in ("0", "", "false", "False")

if _TE_ENABLED_ENV:
    # TE JIT-compiles its "tuned" norm/softmax kernels through hipRTC, feeding
    # the compile $ROCM_PATH/include while stamping it with the HIP version of
    # the *runtime* that's loaded. Those two only agree if TE was built against
    # the same ROCm the loaded HIP runtime comes from; here they don't (system
    # ROCm 7.14 headers vs torch's HIP 7.2 runtime), and every RTC compile dies
    # with HIPRTC_ERROR_COMPILATION inside hip_bf16.h -- which surfaces as a
    # RuntimeError from the first layer_norm/rms_norm/attention call, i.e. TE
    # would be "available" but fall back to stock on every single call.
    # NVTE_DISABLE_NVRTC (see common/util/rtc_hip.cpp) keeps TE on its
    # precompiled kernels, which work. setdefault, not a plain assignment:
    # NVTE_DISABLE_NVRTC=0 re-enables the JIT path on a machine whose ROCm
    # headers do match its runtime.
    os.environ.setdefault("NVTE_DISABLE_NVRTC", "1")

    try:
        # `transformer_engine.pytorch` MUST come first. TE no longer installs a
        # top-level `transformer_engine_torch` module: the binding .so lives
        # inside the transformer_engine package directory, and
        # transformer_engine.common.load_framework_extension("torch") is what
        # loads it and registers it under that name in sys.modules. Importing
        # it first raises ModuleNotFoundError, which this except clause would
        # swallow into a permanent available() == False.
        import transformer_engine.pytorch as te
        import transformer_engine_torch as tex
        from transformer_engine.pytorch.constants import TE_DType

        _TE_AVAILABLE = True
    except ImportError:
        tex = None
        te = None
        TE_DType = None
        _TE_AVAILABLE = False
else:
    tex = None
    te = None
    TE_DType = None
    _TE_AVAILABLE = False


def available() -> bool:
    return _TE_AVAILABLE


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------

class _LayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias, eps):
        input_dims = input_.shape
        inner_dim = weight.numel()
        dtype = weight.dtype
        x = input_.contiguous().view(-1, inner_dim)
        w = weight.contiguous().view(inner_dim)
        b = bias.contiguous().view(inner_dim)

        y, means, rstdevs = tex.layernorm_fwd(
            x, w, b, eps, None, None, TE_DType[dtype], 0, False,
        )

        ctx.save_for_backward(x, means, rstdevs, w)
        ctx.input_dims = input_dims
        return y.view(input_dims)

    @staticmethod
    def backward(ctx, grad_output):
        x, means, rstdevs, w = ctx.saved_tensors
        dy = grad_output.contiguous().view(x.size())
        dx, dw, db = tex.layernorm_bwd(dy, x, means, rstdevs, w, 0, False)
        return dx.view(ctx.input_dims), dw, db, None


def layer_norm(input_, normalized_shape, weight, bias, eps=1e-5):
    return _LayerNormFn.apply(input_, weight, bias, eps)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class _RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, eps):
        input_dims = input_.shape
        inner_dim = weight.numel()
        dtype = weight.dtype
        x = input_.contiguous().view(-1, inner_dim)
        w = weight.contiguous().view(inner_dim)

        y, _, rstdevs = tex.rmsnorm_fwd(
            x, w, eps, None, None, TE_DType[dtype], 0, False,
        )

        ctx.save_for_backward(x, rstdevs, w)
        ctx.input_dims = input_dims
        return y.view(input_dims)

    @staticmethod
    def backward(ctx, grad_output):
        x, rstdevs, w = ctx.saved_tensors
        dy = grad_output.contiguous().view(x.size())
        dx, dw = tex.rmsnorm_bwd(dy, x, rstdevs, w, 0, False)
        return dx.view(ctx.input_dims), dw, None


# TE's experimental ROCm Triton RMSNorm (what NVTE_USE_RMSNORM_TRITON=1
# switches on inside TE's own modules; that env var never reaches the direct
# tex.* calls above, so it is wired in here explicitly). Same signatures as
# tex.rmsnorm_fwd/bwd. Measured fwd+bwd vs stock on the 7900 XTX
# (tools/bench_te_triton_norms.py, analyse/benchmarks/te_triton_norms_bench.txt):
# 1.21-1.82x at >= 16384 rows x 4096-8192 hidden, ahead of the hipified kernel
# there; behind it at hidden <= 3072, and 0.17-0.33x on small inputs (Python
# launch overhead in TE's wrapper). TE's Triton LayerNorm was mixed against
# the hipified one, so it is not used. _dispatch decides which shapes get it.
_triton_rms = None  # (fwd, bwd) once imported; False if the import failed


def _triton_rms_fns():
    global _triton_rms
    if _triton_rms is None:
        try:
            from transformer_engine.pytorch.triton_kernels.norms_common import (
                te_rmsnorm_fwd_triton, te_rmsnorm_bwd_triton)
            _triton_rms = (te_rmsnorm_fwd_triton, te_rmsnorm_bwd_triton)
        except Exception:  # noqa: BLE001 -- no triton / old TE: stay on HIP
            _triton_rms = False
    return _triton_rms


def triton_rms_norm_available() -> bool:
    return _TE_AVAILABLE and bool(_triton_rms_fns())


class _RMSNormTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, eps):
        fwd, _ = _triton_rms_fns()
        input_dims = input_.shape
        inner_dim = weight.numel()
        x = input_.contiguous().view(-1, inner_dim)
        w = weight.contiguous().view(inner_dim)
        y, _, rstdevs = fwd(x, w, eps, None, None, TE_DType[weight.dtype], 0, False)
        ctx.save_for_backward(x, rstdevs, w)
        ctx.input_dims = input_dims
        return y.view(input_dims)

    @staticmethod
    def backward(ctx, grad_output):
        _, bwd = _triton_rms_fns()
        x, rstdevs, w = ctx.saved_tensors
        dy = grad_output.contiguous().view(x.size())
        dx, dw = bwd(dy, x, rstdevs, w, 0, False)
        return dx.view(ctx.input_dims), dw, None


def rms_norm(input_, normalized_shape, weight, eps=None, backend: str = "hip"):
    """backend="hip" (hipified TE kernel) or "triton" (TE's ROCm Triton
    kernel; falls back to "hip" when it cannot be imported)."""
    if eps is None:
        eps = torch.finfo(input_.dtype).eps
    if backend == "triton" and triton_rms_norm_available():
        return _RMSNormTritonFn.apply(input_, weight, eps)
    return _RMSNormFn.apply(input_, weight, eps)


# ---------------------------------------------------------------------------
# GELU / SiLU
# ---------------------------------------------------------------------------

class _GeluFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        x = input_.contiguous()
        y = tex.gelu(x, None)
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        dx = tex.dgelu(grad_output.contiguous(), x, None)
        return dx


class _SiluFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        x = input_.contiguous()
        y = tex.silu(x, None)
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        dx = tex.dsilu(grad_output.contiguous(), x, None)
        return dx


def gelu(input_):
    return _GeluFn.apply(input_)


def silu(input_):
    return _SiluFn.apply(input_)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
#
# TE's DotProductAttention has no learnable parameters -- its module state is
# just static config (head count/dim, mask type, qkv layout, softmax scale),
# so caching instances per config and reusing them across forward() calls is
# the intended usage pattern (same as reusing a stateless nn.Module).
#
# DotProductAttention requires "bshd"/"sbhd"/"thd" layouts, not the (batch,
# heads, seq, head_dim) layout F.scaled_dot_product_attention uses -- hence
# the transpose(1, 2) below. attn_mask_type is overridable per forward() call
# (see dot_product_attention.py's forward signature), so one cached module
# per (num_heads, head_dim, softmax_scale) covers both causal and non-causal
# calls.

_attention_cache: dict = {}

# ---------------------------------------------------------------------------
# Bottom-right-aligned causal mask detection.
#
# TE's fused (CK/AOTriton) attention backends only accept named
# attn_mask_type strings ("no_mask", "causal", "causal_bottom_right",
# "padding", ...) -- forwarding an arbitrary tensor forces
# attn_mask_type="arbitrary", which TE's own docs say disables both
# FlashAttention and its fused backend, falling back to
# UnfusedDotProductAttention. Routing every non-None attn_mask to TE that way
# would often make things *slower*, defeating the point of amd_tuned_torch -- so only
# this one well-defined, structurally-verifiable pattern is recognized:
# causal, optionally with q_len < kv_len (the shape KV-cache decoding
# produces, e.g. dflash's model.py:_attention_mask or HF's
# sdpa_attention_forward). Anything else (padding masks, sliding-window,
# genuinely arbitrary block-diffusion masks) falls back to stock PyTorch,
# same as before this existed.
# ---------------------------------------------------------------------------

_causal_reference_cache: dict = {}


def _bottom_right_causal_reference(q_len: int, kv_len: int, device) -> torch.Tensor:
    """The exact boolean pattern KV-cache-aware causal masking builds: query
    position i is offset by (kv_len - q_len) so the causal diagonal lines up
    with the *last* q_len keys rather than the first. Cached per
    (q_len, kv_len, device) since the pattern is pure position arithmetic,
    independent of any tensor's content."""
    key = (q_len, kv_len, device)
    ref = _causal_reference_cache.get(key)
    if ref is None:
        query_position = kv_len - q_len + torch.arange(q_len, device=device)[:, None]
        key_position = torch.arange(kv_len, device=device)[None, :]
        ref = key_position <= query_position
        _causal_reference_cache[key] = ref
    return ref


def is_bottom_right_causal_mask(attn_mask: torch.Tensor) -> bool:
    """True only if `attn_mask` is a boolean tensor exactly matching the
    bottom-right-aligned causal pattern (broadcast over any batch/head dims).
    Used by amd_tuned_torch.__init__._patched_sdpa to decide whether an explicit
    attn_mask tensor (as opposed to the `is_causal=True` flag) can still be
    routed to TE's fused "causal_bottom_right" path. See the module comment
    above for why this check is narrow rather than a blanket passthrough."""
    if attn_mask.dtype != torch.bool or attn_mask.dim() < 2:
        return False
    q_len, kv_len = attn_mask.shape[-2], attn_mask.shape[-1]
    reference = _bottom_right_causal_reference(q_len, kv_len, attn_mask.device)
    try:
        return torch.equal(attn_mask, reference.expand_as(attn_mask))
    except RuntimeError:
        # attn_mask's leading dims aren't broadcastable from (q_len, kv_len)
        # (e.g. it varies per batch/head) -- definitely not this pattern.
        return False


def _get_attention_module(num_heads: int, head_dim: int, softmax_scale: Optional[float]):
    key = (num_heads, head_dim, softmax_scale)
    module = _attention_cache.get(key)
    if module is None:
        module = te.DotProductAttention(
            num_attention_heads=num_heads,
            kv_channels=head_dim,
            attention_dropout=0.0,
            qkv_format="bshd",
            attn_mask_type="no_mask",
            softmax_scale=softmax_scale,
        )
        _attention_cache[key] = module
    return module


def scaled_dot_product_attention(query, key, value, attn_mask: Optional[torch.Tensor] = None,
                                  scale: Optional[float] = None, is_causal: bool = False):
    """query/key/value: (batch, heads, seq, head_dim), matching
    F.scaled_dot_product_attention's layout -- NOT TE's native bshd/sbhd.

    `attn_mask`, if given, must already be verified by the caller (see
    amd_tuned_torch.__init__._patched_sdpa) via is_bottom_right_causal_mask() -- this
    function does not re-check it and never forwards the tensor itself to
    TE. Recognizing it just selects the named "causal_bottom_right"
    attn_mask_type; TE computes that mask internally from q_len/kv_len, the
    same way it does for plain "causal"."""
    num_heads = query.size(1)
    head_dim = query.size(3)
    module = _get_attention_module(num_heads, head_dim, scale)

    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()

    if attn_mask is not None:
        attn_mask_type = "causal_bottom_right"
    elif is_causal:
        attn_mask_type = "causal"
    else:
        attn_mask_type = "no_mask"
    out = module(q, k, v, attn_mask_type=attn_mask_type)
    # TE returns (batch, seq, heads * head_dim); reshape back to bshd then
    # transpose to the caller's bhsd convention.
    out = out.view(out.size(0), out.size(1), num_heads, head_dim)
    return out.transpose(1, 2)
