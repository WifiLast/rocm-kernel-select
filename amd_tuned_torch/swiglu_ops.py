"""Bias-fused SwiGLU (SiLU-Gated Linear Unit), with a real backward pass --
adapted from NVIDIA Megatron-LM (see
amd_tuned_torch/_vendor/megatron_swiglu/NOTICE.md for exactly what
changed from upstream and why).

THE FUSION BOUNDARY THIS COVERS, that nothing else in this package does:
aiter_ops.fused_silu_mul (this package's existing SwiGLU-shaped kernel)
and TE's silu op both take an already-bias-added tensor -- neither
absorbs a preceding bias-add. If a GEMM path in your model produces
`(output, bias)` as two separate tensors (not every hipBLASLt/CK/aiter
dispatch path fuses bias into the GEMM epilogue), `output + bias` is an
extra full elementwise pass over the whole FFN-intermediate-sized tensor
before SwiGLU ever runs. This module fuses that add into the same custom
autograd.Function as the SwiGLU forward/backward, saving one tensor
(`input + bias`, not `input` and `bias` separately) for backward instead
of two, and skipping the extra elementwise launch.

Unlike every other Triton kernel in this package, this one has NO Triton
dependency at all -- it's plain PyTorch (chunk/silu/sigmoid/mul/cat), so
available() always returns True. The "fusion" here is entirely in the
custom autograd.Function (fewer saved tensors, fewer op dispatches for
autograd to unwind), not in launching fewer GPU kernels the way a Triton
port would provide -- see the module NOTICE for why upstream's own
torch.jit.script/torch.compile wrapping was deliberately not carried over
into this port.

Not a monkeypatch target (there is no F.* op for "bias-add then SwiGLU"
to intercept), so call this directly from an FFN/MoE block's forward in
place of:

    y = hidden @ up_proj.weight.T + up_proj.bias
    out = F.silu(y[..., :d]) * y[..., d:]

with:

    out = amd_tuned_torch.bias_swiglu(hidden @ up_proj.weight.T, up_proj.bias)

UNVALIDATED: adapted from Megatron-LM (which itself makes no ROCm support
claim, unlike Liger-Kernel or Conch), not run on any hardware this
project has access to. Before relying on this:
  1. Compare its output AND gradients against the plain unfused
     computation (bias-add, chunk, F.silu, multiply) numerically, for
     your actual shapes/dtype -- especially with clamp_value set, since
     the clamped backward's boundary-mask terms (`y <= clamp_value`,
     `-clamp_value <= y <= clamp_value`) are exactly the kind of
     off-by-one-prone code a from-scratch numerical check catches that
     reading the source doesn't.
  2. Benchmark against the unfused version for your actual shapes -- the
     saved-tensor-count reduction only pays off during training (a
     forward-only/inference call gets no benefit from a smaller backward
     graph).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def available() -> bool:
    """Always True: this module is plain PyTorch with no Triton/CK/
    hipBLASLt dependency, so it works wherever stock PyTorch does."""
    return True


def _swiglu_fwd(y: torch.Tensor) -> torch.Tensor:
    y1, y2 = torch.chunk(y, 2, dim=-1)
    return F.silu(y1) * y2


def _swiglu_bwd(g: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y1, y2 = torch.chunk(y, 2, dim=-1)
    sig = torch.sigmoid(y1)
    dy1 = g * sig * (1 + y1 * (1 - sig)) * y2
    dy2 = g * F.silu(y1)
    return torch.cat((dy1, dy2), dim=-1)


def _clamped_swiglu_fwd(y: torch.Tensor, clamp_value: float) -> torch.Tensor:
    dtype = y.dtype
    y1, y2 = torch.chunk(y.float(), 2, dim=-1)
    y1c = y1.clamp(max=clamp_value)
    y2c = y2.clamp(min=-clamp_value, max=clamp_value)
    return (F.silu(y1c) * y2c).to(dtype)


def _clamped_swiglu_bwd(g: torch.Tensor, y: torch.Tensor, clamp_value: float) -> torch.Tensor:
    dtype = y.dtype
    y1, y2 = torch.chunk(y.float(), 2, dim=-1)
    y1c = y1.clamp(max=clamp_value)
    y2c = y2.clamp(min=-clamp_value, max=clamp_value)
    sig = torch.sigmoid(y1c)
    dy1 = g * sig * (1 + y1c * (1 - sig)) * y2c * (y1 <= clamp_value).to(g.dtype)
    dy2 = g * F.silu(y1c) * ((y2 >= -clamp_value) & (y2 <= clamp_value)).to(g.dtype)
    return torch.cat((dy1, dy2), dim=-1).to(dtype)


def _reduce_bias_grad(grad: torch.Tensor, bias_shape: torch.Size) -> torch.Tensor:
    """Standard trailing-dim-broadcast bias-gradient reduction -- see
    module NOTICE.md for why this port doesn't reuse upstream's
    unreduced-gradient shortcut."""
    extra_dims = grad.dim() - len(bias_shape)
    if extra_dims <= 0:
        return grad
    return grad.sum(dim=tuple(range(extra_dims)))


class _BiasSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bias, clamp_value):
        y = input + bias if bias is not None else input
        ctx.save_for_backward(y)
        ctx.clamp_value = clamp_value
        ctx.bias_shape = bias.shape if bias is not None else None
        if clamp_value is not None:
            return _clamped_swiglu_fwd(y, clamp_value)
        return _swiglu_fwd(y)

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        if ctx.clamp_value is not None:
            grad = _clamped_swiglu_bwd(grad_output, y, ctx.clamp_value)
        else:
            grad = _swiglu_bwd(grad_output, y)
        grad_bias = _reduce_bias_grad(grad, ctx.bias_shape) if ctx.bias_shape is not None else None
        return grad, grad_bias, None


def bias_swiglu(input: torch.Tensor, bias: Optional[torch.Tensor] = None,
                clamp_value: Optional[float] = None) -> torch.Tensor:
    """`SiLU(y1) * y2` where `y = input + bias` (bias optional) is split
    in half along the last dimension: `y1, y2 = y.chunk(2, dim=-1)`.

    input: (..., 2*ffn_hidden). bias: (2*ffn_hidden,) or any shape that's
        a trailing suffix of input's shape (standard broadcasting), or
        None.
    clamp_value: if set, hard-clamps y1 to `&lt;= clamp_value` and y2 to
        `[-clamp_value, clamp_value]` before the SwiGLU math (a stability
        technique some models use, e.g. Grok/OLMoE-family) -- clamping
        is done in fp32 regardless of input dtype.

    Returns (..., ffn_hidden), with a real backward through both input
    and bias."""
    assert input.shape[-1] % 2 == 0, "bias_swiglu's last dim must be even (split in half)"
    if bias is not None:
        assert input.shape[-1] == bias.shape[-1], "bias's last dim must match input's last dim"
    return _BiasSwiGLUFunction.apply(input, bias, clamp_value)
