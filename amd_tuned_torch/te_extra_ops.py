"""TransformerEngine kernels exposed as an explicit API, not as monkeypatches.

This is the second half of this package's TransformerEngine surface. The split
against te_ops.py is by *how the op is reached*, not by backend:

  - te_ops.py         the five ops amd_tuned_torch patches over stock PyTorch
                      (attention, layer_norm, rms_norm, gelu, silu). Its
                      contract -- "these five, installed by enable(), checked
                      via te_ops.available()" -- is referenced from
                      __init__.py, _dispatch.py and _opt_in_tiers.py, so it is
                      left alone.
  - te_extra_ops.py   TE kernels with no stock-PyTorch equivalent to patch, or
                      whose equivalent takes a different calling convention
                      than torch's. Nothing here is installed by enable();
                      callers reach for these by name.

WHY THESE FOUR FAMILIES
-----------------------
TE's two headline features are unreachable on this project's target GPU:

  - Fused attention. Both ROCm backends are CDNA-only and this tree's own
    setup.py (see its `ck_fused_attn_archs`/`aotriton_image_archs` gating)
    compiles them OUT on gfx1100 -- CK's kernels are MFMA, AOTriton publishes
    no gfx1100 kernel image. With neither define compiled in,
    nvte_get_fused_attn_backend() returns NVTE_No_Backend.
  - FP8/MXFP8/NVFP4. TE gates FP8 on gfx arch (9,4)/(9,5)/(12,5)
    (transformer_engine/pytorch/quantization.py::_compute_fp8_support).
    gfx1100 reports (11,0). That removes the quantize/cast-transpose/recipe/
    swizzle half of the library.

What is left is TE's arch-agnostic kernel set -- everything in the ROCm build
that is neither MFMA-based nor FP8-conditional. These four families are the
part of it this package does not already cover by another backend:

  1. multi-tensor optimizers    no optimizer kernels anywhere in this package
  2. fused softmax              no softmax kernel anywhere in this package
  3. fused dropout              8-bit-RNG mask, cheaper than torch's byte mask
  4. sequence-layout utilities  thd<->bshd, row padding, KV-cache scatter

Deliberately NOT wrapped here, because this package already has a better or
equal path: norms (ck_norm_ops/fused_norm_ops/triton_kernels_ops), RoPE
(rope_ops, which has a real backward), GEMM (hipblaslt_ops reaches the same
hipBLASLt that TE's rocm_gemm.cu does), and attention (flash_attn_rocwmma_ops
has a native gfx1100 WMMA kernel; TE's would fall back to its *unfused*
PyTorch path here, which is slower than stock).

WAVE SIZE
---------
TE hardcodes THREADS_PER_WARP = 32 (transformer_engine/common/utils.cuh:83).
On CDNA (wave64) TE papers over that with explicit `width` arguments to
__shfl_xor; on gfx1100 wave32 is native, so the softmax and multi-tensor
kernels are a closer fit to this hardware than to the hardware they were
tuned for. This is the one place where RDNA3 is the easier target.

AVAILABILITY
------------
available() defers entirely to te_ops.available(), so the
AMD_TUNED_TORCH_ENABLE_TE=1 opt-in gates both modules and `transformer_engine`
is imported at most once, through one gate. See te_ops.py's "TE DISABLED BY
DEFAULT" section for why that opt-in exists at all (a TE built against a
different PyTorch/ROCm can segfault on import, which no try/except can catch).
Every entry point here returns/raises nothing at import time and checks
available() before touching `tex`.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

from . import te_ops


def available() -> bool:
    """True when TransformerEngine's torch extension is importable and opted
    into. Same gate as te_ops -- never a separate TE import."""
    return te_ops.available()


def _tex():
    """The `transformer_engine_torch` extension module, or raise a clear error.

    Called at the top of every public entry point rather than captured at
    import time, so that a module-level `from . import te_extra_ops` stays
    free of TE even when TE is disabled.
    """
    if not te_ops.available():
        raise RuntimeError(
            "te_extra_ops requires TransformerEngine, which is not available. "
            "Set AMD_TUNED_TORCH_ENABLE_TE=1 before importing amd_tuned_torch "
            "and verify `python -c 'import transformer_engine.pytorch'` works."
        )
    return te_ops.tex


# ---------------------------------------------------------------------------
# 1. Multi-tensor optimizer kernels
# ---------------------------------------------------------------------------
#
# One kernel launch per *chunk of the whole parameter list*, instead of one
# launch per parameter tensor. For a model with a few hundred parameters this
# is the difference between a few hundred tiny elementwise launches per
# optimizer step and a handful of saturating ones -- the same win apex's
# multi_tensor_applier gives, which is where TE's kernels come from.
#
# CHUNK SIZE. 2048 * 32 matches TE's own MultiTensorApply instance
# (transformer_engine/pytorch/optimizers/multi_tensor_apply.py) -- the kernels
# are tuned around it, so it is mirrored rather than re-derived.
#
# NOOP FLAG. Every one of these kernels takes an int32 "skip" buffer as its
# second argument: nonzero means "an inf/nan was found upstream, do not apply
# this update". Callers doing AMP-style gradient scaling copy their
# found_inf into it; callers that are not just want a zero buffer, which is
# what _noop_flag() hands out (cached per device -- allocating a one-element
# tensor per optimizer step would undo part of what this saves).

_CHUNK_SIZE = 2048 * 32

_noop_flag_cache: dict = {}


def _noop_flag(device: torch.device) -> torch.Tensor:
    """A cached zero int32 buffer, the "no overflow found" value."""
    key = (device.type, device.index)
    flag = _noop_flag_cache.get(key)
    if flag is None:
        flag = torch.zeros(1, dtype=torch.int32, device=device)
        _noop_flag_cache[key] = flag
    return flag


def _align_tensor_lists(tensor_lists: Sequence[Sequence[torch.Tensor]]):
    """Drop empty-numel slots and check the lists line up.

    Mirrors MultiTensorApply.__call__: the kernels index every list with the
    same chunk index, so the lists must be the same length, and a zero-element
    parameter would produce a chunk with nothing in it. Returns None when
    nothing is left to do (an all-empty parameter group), which every caller
    below treats as a no-op.
    """
    lists = [list(t) for t in tensor_lists]
    if not lists or not lists[0]:
        return None
    if any(len(t) != len(lists[0]) for t in lists):
        raise RuntimeError(
            "Expected aligned multi-tensor lists; got lengths "
            f"{[len(t) for t in lists]}."
        )
    keep = [t.numel() > 0 for t in lists[0]]
    lists = [[t for t, k in zip(tensors, keep) if k] for tensors in lists]
    if not lists[0]:
        return None
    return lists


def multi_tensor_adam(
    grads: Sequence[torch.Tensor],
    params: Sequence[torch.Tensor],
    exp_avgs: Sequence[torch.Tensor],
    exp_avg_sqs: Sequence[torch.Tensor],
    *,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    step: int,
    adam_w_mode: bool = True,
    bias_correction: bool = True,
    weight_decay: float = 0.0,
    master_params: Optional[Sequence[torch.Tensor]] = None,
    noop_flag: Optional[torch.Tensor] = None,
) -> None:
    """Fused Adam/AdamW step applied in-place across whole parameter lists.

    The four (or five) lists are positional in TE's kernel and must be in this
    order -- grads, params, exp_avg, exp_avg_sq[, master_params] -- which is
    what FusedAdam.step() builds (see TE's fused_adam.py tensor_lists).

    `adam_w_mode=True` is decoupled weight decay (AdamW, TE's mode 1);
    False applies L2 regularization into the gradient (Adam, mode 0).

    `step` is 1-based and is used for bias correction, so pass the optimizer's
    own step counter *after* incrementing it.

    `master_params`, when given, is an fp32 copy updated alongside lower
    precision `params` -- the mixed-precision training shape. Omit it for a
    plain fp32 or pure-bf16 step.

    In-place: nothing is returned. No autograd -- an optimizer step runs under
    torch.no_grad() by construction.
    """
    tex = _tex()
    lists: List[Sequence[torch.Tensor]] = [list(grads), list(params), list(exp_avgs),
                                           list(exp_avg_sqs)]
    if master_params is not None:
        lists.append(list(master_params))
    aligned = _align_tensor_lists(lists)
    if aligned is None:
        return
    flag = _noop_flag(aligned[0][0].device) if noop_flag is None else noop_flag
    tex.multi_tensor_adam(
        _CHUNK_SIZE,
        flag,
        aligned,
        lr,
        beta1,
        beta2,
        eps,
        step,
        1 if adam_w_mode else 0,
        1 if bias_correction else 0,
        weight_decay,
    )


def multi_tensor_sgd(
    grads: Sequence[torch.Tensor],
    params: Sequence[torch.Tensor],
    momentum_buffers: Sequence[torch.Tensor],
    *,
    lr: float,
    momentum: float = 0.0,
    dampening: float = 0.0,
    weight_decay: float = 0.0,
    nesterov: bool = False,
    first_run: bool = False,
    wd_after_momentum: bool = False,
    scale: float = 1.0,
    noop_flag: Optional[torch.Tensor] = None,
) -> None:
    """Fused SGD step applied in-place across whole parameter lists.

    List order is grads, params, momentum_buffers (TE's fused_sgd.py
    launch_set). `first_run` tells the kernel the momentum buffers hold
    nothing yet, so the first update seeds them from the gradient rather than
    blending into uninitialized memory -- pass True on the step where a
    parameter's buffer was just allocated.

    `scale` is a gradient un-scaling factor applied inside the kernel (AMP's
    1/loss_scale); leave at 1.0 when not loss-scaling.
    """
    tex = _tex()
    aligned = _align_tensor_lists([list(grads), list(params), list(momentum_buffers)])
    if aligned is None:
        return
    flag = _noop_flag(aligned[0][0].device) if noop_flag is None else noop_flag
    tex.multi_tensor_sgd(
        _CHUNK_SIZE,
        flag,
        aligned,
        weight_decay,
        momentum,
        dampening,
        lr,
        nesterov,
        first_run,
        wd_after_momentum,
        scale,
    )


def multi_tensor_l2norm(
    tensors: Sequence[torch.Tensor],
    *,
    per_tensor: bool = False,
    inv_scale: Optional[torch.Tensor] = None,
    noop_flag: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Global L2 norm over a whole list of tensors, in one kernel.

    This is the gradient-clipping primitive: torch.nn.utils.clip_grad_norm_
    computes a per-tensor norm then a norm of those, which is two launches per
    parameter. Here it is one launch per chunk.

    Returns (total_norm, per_tensor_norms). per_tensor_norms is None unless
    `per_tensor=True`.

    `inv_scale`, when given, un-scales the values before the norm is taken
    WITHOUT writing the un-scaled values back (TE's
    multi_tensor_unscale_l2norm) -- the AMP case where you want the true
    gradient norm while the stored gradients are still loss-scaled.
    """
    tex = _tex()
    aligned = _align_tensor_lists([list(tensors)])
    if aligned is None:
        empty = torch.zeros(1, device="cuda" if torch.cuda.is_available() else "cpu")
        return empty, None
    flag = _noop_flag(aligned[0][0].device) if noop_flag is None else noop_flag
    if inv_scale is None:
        norm, per_tensor_norm = tex.multi_tensor_l2norm(_CHUNK_SIZE, flag, aligned, per_tensor)
    else:
        norm, per_tensor_norm = tex.multi_tensor_unscale_l2norm(
            _CHUNK_SIZE, flag, aligned, inv_scale, per_tensor
        )
    return norm, (per_tensor_norm if per_tensor else None)


def multi_tensor_scale(
    inputs: Sequence[torch.Tensor],
    outputs: Sequence[torch.Tensor],
    scale: float,
    *,
    noop_flag: Optional[torch.Tensor] = None,
) -> None:
    """out = in * scale across a whole list, with a fused overflow check.

    Writes into `outputs` (which may be the same tensors as `inputs`). The
    overflow check is the point: the kernel sets `noop_flag` nonzero if any
    scaled value is inf/nan, which is how an AMP scaler learns to back off
    without a separate reduction pass. Pass your own `noop_flag` to read it.
    """
    tex = _tex()
    aligned = _align_tensor_lists([list(inputs), list(outputs)])
    if aligned is None:
        return
    flag = _noop_flag(aligned[0][0].device) if noop_flag is None else noop_flag
    tex.multi_tensor_scale(_CHUNK_SIZE, flag, aligned, scale)


# ---------------------------------------------------------------------------
# 2. Fused scaled softmax
# ---------------------------------------------------------------------------
#
# softmax(scale * x), with the scale, the mask and the softmax in one pass
# instead of three materialized intermediates. The backward is equally fused.
#
# These kernels validate their inputs with AT_ASSERTM, which aborts into a
# C++ exception rather than returning a status -- so kernel_available() below
# pre-checks every constraint the bindings assert
# (transformer_engine/pytorch/csrc/extensions/softmax.cpp) plus the shape
# rules TE's own dispatcher applies. Call it before scaled_softmax(), or use
# scaled_softmax_or_torch() which does it for you.

THREADS_PER_WARP = 32
THREADS_PER_BLOCK = 128

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


def batch_per_block(key_seq_len: int) -> int:
    """Rows each thread block handles, for the divisibility rule below.

    Pure arithmetic on the kernel's launch geometry, reproduced from TE's
    FusedScaleMaskSoftmax.get_batch_per_block rather than called through it
    (that is a method on a module we do not construct).
    """
    pow2 = 1 << (key_seq_len - 1).bit_length()
    warp_size = pow2 if pow2 < THREADS_PER_WARP else THREADS_PER_WARP
    batches_per_warp = 2 if pow2 <= 128 else 1
    warps_per_block = THREADS_PER_BLOCK // warp_size
    return warps_per_block * batches_per_warp


def kernel_available(
    scores: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    attn_mask_type: str = "no_mask",
) -> bool:
    """True when the fused kernel can handle this exact shape and dtype.

    `scores` is the pre-softmax attention score tensor, (b, np, sq, sk).
    Mirrors TE's FusedScaleMaskSoftmax.is_kernel_available, which is the only
    authority on what these kernels accept; anything it rejects has to go to
    torch's softmax instead. Never raises -- a False here is a routing
    decision, not an error.
    """
    if scores.dim() != 4:
        return False
    if scores.dtype not in _SUPPORTED_DTYPES:
        return False  # kernels are fp16/bf16 only
    b, np_, sq, sk = scores.size()
    attn_batches = b * np_

    if not 16 < sk < 16384:
        return False
    if sk % 8 != 0:
        return False
    if sq == 1:
        return False
    if attn_mask_type == "causal" and sq != sk:
        return False  # top-left causal needs a square score matrix

    if sq % 4 != 0 or attn_batches % 4 != 0:
        return False
    if sq % batch_per_block(int(sk)) != 0:
        return False

    if "padding" in attn_mask_type or attn_mask_type == "arbitrary":
        # ScaledMaskedSoftmax reads the mask directly and asserts its shape.
        return (
            mask is not None
            and mask.dim() == 4
            and mask.shape[0] in (1, b)
            and tuple(mask.shape[1:]) == (1, sq, sk)
        )
    return True


class _ScaledSoftmaxFn(torch.autograd.Function):
    """Unmasked softmax(scale * x) over a 4D score tensor."""

    @staticmethod
    def forward(ctx, scores, scale):
        tex = _tex()
        out = tex.scaled_softmax_forward(scores, scale)
        ctx.save_for_backward(out)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        tex = _tex()
        # The kernel needs the softmax *output*, not the input -- softmax's
        # Jacobian is expressible entirely in terms of its own result.
        return tex.scaled_softmax_backward(grad_output, out, ctx.scale), None


class _ScaledMaskedSoftmaxFn(torch.autograd.Function):
    """softmax(scale * x) with an additive/boolean mask broadcast over heads."""

    @staticmethod
    def forward(ctx, scores, mask, scale):
        tex = _tex()
        out = tex.scaled_masked_softmax_forward(scores, mask, scale)
        ctx.save_for_backward(out)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        tex = _tex()
        # The mask is not needed in backward: masked positions are already
        # exactly zero in `out`, so their gradient falls out as zero too.
        return tex.scaled_masked_softmax_backward(grad_output, out, ctx.scale), None, None


class _ScaledUpperTriangMaskedSoftmaxFn(torch.autograd.Function):
    """Top-left-aligned causal softmax over a 3D (attn_batches, s, s) tensor.

    Note the rank: this kernel takes the heads folded into the batch dim and
    asserts size(1) == size(2), unlike every other variant here, which is 4D.
    """

    @staticmethod
    def forward(ctx, scores, scale):
        tex = _tex()
        out = tex.scaled_upper_triang_masked_softmax_forward(scores, scale)
        ctx.save_for_backward(out)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        tex = _tex()
        return (
            tex.scaled_upper_triang_masked_softmax_backward(grad_output, out, ctx.scale),
            None,
        )


class _ScaledAlignedCausalMaskedSoftmaxFn(torch.autograd.Function):
    """Bottom-right-aligned causal softmax, the KV-cache-shaped mask.

    KNOWN UPSTREAM BUG -- see scaled_softmax()'s docstring before using this.
    """

    @staticmethod
    def forward(ctx, scores, scale):
        tex = _tex()
        out = tex.scaled_aligned_causal_masked_softmax_forward(scores, scale)
        ctx.save_for_backward(out)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        tex = _tex()
        return (
            tex.scaled_aligned_causal_masked_softmax_backward(grad_output, out, ctx.scale),
            None,
        )


def scaled_softmax(
    scores: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    attn_mask_type: str = "no_mask",
    *,
    use_aligned_causal_kernel: bool = False,
) -> torch.Tensor:
    """Fused softmax(scale * scores), autograd-capable.

    `scores` is (b, np, sq, sk); `attn_mask_type` uses TE's vocabulary
    ("no_mask", "causal", "causal_bottom_right", "padding",
    "padding_causal", "arbitrary", ...).

    Dispatch mirrors TE's own forward_fused_softmax: a mask plus any mask type
    other than "no_mask" goes to ScaledMaskedSoftmax, everything else to
    ScaledSoftmax. In particular a causal type with no explicit mask does NOT
    reach the aligned-causal kernel by default, because TE itself disabled
    that route -- see the commented-out dispatch and the note
    "Disable for now until unalignment bug is fixed" in
    transformer_engine/pytorch/attention/dot_product_attention/softmax.py.
    That kernel is still reachable via use_aligned_causal_kernel=True for
    anyone wanting to test it, and via aligned_causal_softmax() directly, but
    it is not something to turn on without checking its output first.

    Raises if the shape is ineligible -- call kernel_available() first, or use
    scaled_softmax_or_torch().
    """
    _tex()
    if use_aligned_causal_kernel and attn_mask_type in ("causal", "causal_bottom_right"):
        return _ScaledAlignedCausalMaskedSoftmaxFn.apply(scores, scale)
    if mask is not None and attn_mask_type != "no_mask":
        return _ScaledMaskedSoftmaxFn.apply(scores, mask, scale)
    return _ScaledSoftmaxFn.apply(scores, scale)


def upper_triang_causal_softmax(scores: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Top-left causal fused softmax over a 3D (b * np, s, s) tensor.

    Kept as its own entry point rather than folded into scaled_softmax's
    dispatch because of the rank difference -- callers have to reshape for it,
    and silently doing that reshape inside a 4D-looking API would hide a real
    cost (a contiguous view is free, a copy is not).
    """
    _tex()
    return _ScaledUpperTriangMaskedSoftmaxFn.apply(scores, scale)


def aligned_causal_softmax(scores: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Bottom-right-aligned causal fused softmax. See the bug note in
    scaled_softmax() before reaching for this."""
    _tex()
    return _ScaledAlignedCausalMaskedSoftmaxFn.apply(scores, scale)


def scaled_softmax_or_torch(
    scores: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    attn_mask_type: str = "no_mask",
) -> torch.Tensor:
    """scaled_softmax() when the kernel accepts this shape, torch otherwise.

    The routing wrapper, shaped like the rest of this package: never raises
    for an ineligible shape, just costs one eligibility check. The fallback
    computes in fp32 and casts back, matching TE's forward_torch_softmax.
    """
    if available() and kernel_available(scores, mask, attn_mask_type):
        return scaled_softmax(scores, mask, scale, attn_mask_type)

    out = scores.float() * scale
    if mask is not None and attn_mask_type != "no_mask":
        out = out.masked_fill(mask, float("-inf"))
    out = torch.nn.functional.softmax(out, dim=-1)
    return out.to(scores.dtype)


# ---------------------------------------------------------------------------
# 3. Fused dropout
# ---------------------------------------------------------------------------
#
# TE's dropout keeps its mask as one BIT per element (an 8-bit RNG state
# packed 8 elements to the byte), where torch.nn.functional.dropout saves a
# full-width tensor. For an activation of N elements that is N/8 bytes of
# saved-for-backward instead of 2N (fp16) -- which for a long training step is
# real memory, not a rounding error.
#
# The mask is NOT interchangeable with torch's: it is TE's own packed layout,
# only meaningful to tex.dropout_bwd. Do not try to inspect or reuse it.

class _DropoutFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, p):
        tex = _tex()
        out, mask = tex.dropout_fwd(input_.contiguous(), p)
        ctx.save_for_backward(mask)
        ctx.p = p
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        tex = _tex()
        return tex.dropout_bwd(grad_output.contiguous(), mask, ctx.p), None


def dropout(input_: torch.Tensor, p: float = 0.5, training: bool = True) -> torch.Tensor:
    """Fused dropout with a bit-packed mask, autograd-capable.

    Same semantics as F.dropout: scales surviving elements by 1/(1-p), and is
    the identity when `training` is False or p == 0.
    """
    if not training or p == 0.0:
        return input_
    if not 0.0 <= p < 1.0:
        raise ValueError(f"dropout probability must be in [0, 1), got {p}")
    return _DropoutFn.apply(input_, p)


# ---------------------------------------------------------------------------
# 4. Sequence-layout utilities
# ---------------------------------------------------------------------------
#
# These come from fused_attn/kv_cache.cu and fused_attn/context_parallel.cu,
# which -- unlike fused_attn_f16_arbitrary_seqlen.cu and fused_attn_fp8.cu --
# are NOT in the common CMakeLists' cuda_only list. So they survive on ROCm
# even with both fused-attention backends compiled out, which is the state
# this package's target GPU builds in. They are layout plumbing, not
# attention math.
#
# "thd" is the ragged layout: all sequences of a batch concatenated along one
# token axis, with cu_seqlens[i] the start offset of sequence i (so cu_seqlens
# has batch_size + 1 entries and cu_seqlens[-1] is the total token count).
# "bshd" is the padded-rectangle layout torch code usually holds.

class _ConvertTHDtoBSHDFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, thd_tensor, cu_seqlens, max_seqlen):
        tex = _tex()
        batch_size = cu_seqlens.shape[0] - 1
        thd_tensor = thd_tensor.contiguous()
        out = tex.convert_thd_to_bshd(thd_tensor, cu_seqlens, batch_size, max_seqlen)
        ctx.save_for_backward(cu_seqlens)
        ctx.num_tokens = thd_tensor.shape[0]
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (cu_seqlens,) = ctx.saved_tensors
        tex = _tex()
        # The inverse conversion IS the backward: padding positions in the
        # bshd gradient correspond to no input token and are simply dropped.
        return (
            tex.convert_bshd_to_thd(grad_output.contiguous(), cu_seqlens, ctx.num_tokens),
            None,
            None,
        )


class _ConvertBSHDtoTHDFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bshd_tensor, cu_seqlens):
        tex = _tex()
        num_tokens = int(cu_seqlens[-1])
        bshd_tensor = bshd_tensor.contiguous()
        out = tex.convert_bshd_to_thd(bshd_tensor, cu_seqlens, num_tokens)
        ctx.save_for_backward(cu_seqlens)
        ctx.max_seqlen = bshd_tensor.shape[1]
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (cu_seqlens,) = ctx.saved_tensors
        tex = _tex()
        batch_size = cu_seqlens.shape[0] - 1
        return (
            tex.convert_thd_to_bshd(
                grad_output.contiguous(), cu_seqlens, batch_size, ctx.max_seqlen
            ),
            None,
        )


def thd_to_bshd(tensor: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
    """Ragged (total_tokens, h, d) -> padded (b, max_seqlen, h, d)."""
    _tex()
    return _ConvertTHDtoBSHDFn.apply(tensor, cu_seqlens, max_seqlen)


def bshd_to_thd(tensor: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Padded (b, s, h, d) -> ragged (total_tokens, h, d)."""
    _tex()
    return _ConvertBSHDtoTHDFn.apply(tensor, cu_seqlens)


def pad_rows(
    input_: torch.Tensor,
    output: torch.Tensor,
    input_row_list: Sequence[int],
    padded_row_list: Sequence[int],
) -> None:
    """Pad each of several row-groups up to an alignment, in one kernel.

    The grouped-GEMM shape: several variable-height matrices stacked in one
    buffer, each of which a GEMM wants rounded up to a tile multiple. Writes
    into `output`, which the caller allocates with sum(padded_row_list) rows.
    """
    tex = _tex()
    tex.fused_multi_row_padding(
        input_, output, list(input_row_list), list(padded_row_list)
    )


def unpad_rows(
    input_: torch.Tensor,
    output: torch.Tensor,
    input_row_list: Sequence[int],
    unpadded_row_list: Sequence[int],
) -> None:
    """The inverse of pad_rows: strip each group's padding rows back off."""
    tex = _tex()
    tex.fused_multi_row_unpadding(
        input_, output, list(input_row_list), list(unpadded_row_list)
    )


def swap_first_dims(tensor: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Swap dims 0 and 1 with an actual copy, not a stride change.

    torch's .transpose(0, 1) is free but leaves a non-contiguous tensor that
    the next kernel may have to materialize anyway, often less efficiently
    than this does. Use it when the consumer needs contiguity.
    """
    tex = _tex()
    return tex.swap_first_dims(tensor, out)


def copy_to_kv_cache(
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cu_new_lens: torch.Tensor,
    cu_cached_lens: torch.Tensor,
    qkv_format: str,
    batch_size: int,
    max_ctx_len: int,
    max_seqlen: int,
    max_pages_per_seq: int = 1,
    is_non_paged: bool = False,
) -> None:
    """Scatter this step's K/V into a KV cache, in one kernel.

    A thin passthrough, deliberately: the arguments are exactly TE's, because
    reimplementing a cache manager on top of them would be a second source of
    truth for page-table layout. TE's own two managers call it like this
    (transformer_engine/pytorch/attention/inference.py):

      non-paged: page_table=batch_indices, max_pages_per_seq=1,
                 is_non_paged=<whether the cache needs reindexing this step>
      paged:     page_table=self.page_table, max_pages_per_seq=<pages per seq>,
                 is_non_paged=False

    `qkv_format` is "bshd", "sbhd" or "thd" and is translated here to TE's
    NVTE_QKV_Format enum. cu_new_lens/cu_cached_lens are cumulative-length
    tensors over the batch, as in thd_to_bshd above.

    In-place into k_cache/v_cache; nothing is returned, and there is no
    autograd (a KV cache write is inference-only by construction).
    """
    tex = _tex()
    if qkv_format not in ("bshd", "sbhd", "thd"):
        raise ValueError(f"qkv_format must be one of bshd/sbhd/thd, got {qkv_format!r}")

    # Imported here, not at module scope: this is the only entry point that
    # needs it, and it lives in a TE submodule that pulls in attention code.
    # Checked after the format above so a plain typo does not depend on TE's
    # attention submodule importing successfully to be reported.
    from transformer_engine.pytorch.cpp_extensions.fused_attn import QKVFormat

    tex.copy_to_kv_cache(
        new_k,
        new_v,
        k_cache,
        v_cache,
        page_table,
        cu_new_lens,
        cu_cached_lens,
        QKVFormat[qkv_format],
        batch_size,
        max_ctx_len,
        max_seqlen,
        max_pages_per_seq,
        is_non_paged,
    )


__all__ = [
    "available",
    # multi-tensor optimizers
    "multi_tensor_adam",
    "multi_tensor_sgd",
    "multi_tensor_l2norm",
    "multi_tensor_scale",
    # fused softmax
    "batch_per_block",
    "kernel_available",
    "scaled_softmax",
    "scaled_softmax_or_torch",
    "upper_triang_causal_softmax",
    "aligned_causal_softmax",
    # dropout
    "dropout",
    # sequence layout
    "thd_to_bshd",
    "bshd_to_thd",
    "pad_rows",
    "unpad_rows",
    "swap_first_dims",
    "copy_to_kv_cache",
]
