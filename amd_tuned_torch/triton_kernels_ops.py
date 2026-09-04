"""
triton-kernels-backed RMSNorm and SwiGLU helpers for amd_tuned_torch.

source/triton-kernels (bassrehab/triton-kernels, vendored as a sibling
project under source/, not this package's own _vendor/ tree -- it's a
plain pip-installable package with no build step, `pip install -e
../triton-kernels` from this package's own venv makes `import
triton_kernels` succeed, same optional-dependency shape as aiter/
TransformerEngine) exposes RMSNorm, SwiGLU, and quantized-GEMM Triton
kernels for LLM inference. Of those, only two have a use here:

  - rms_norm() -- a manually-selectable ALTERNATIVE backend for
    F.rms_norm, opt-in via enable_triton_kernels_rmsnorm() (see
    amd_tuned_torch/__init__.py). NOT installed by the main enable(): TE's
    rms_norm (te_ops.py) already covers this op with a real backward pass
    (a torch.autograd.Function backed by TE's fused CK/AOTriton kernel),
    while triton_kernels.rmsnorm.rmsnorm has no backward defined at all --
    swapping it in under autograd would silently break gradients. This
    module's eligibility guard (see _is_eligible in __init__.py) rejects
    any call where autograd is live, the same _grad_safe() check
    linear_fp16/bmm_fp16/conv2d/group_norm use elsewhere in this package,
    for exactly that reason. Nothing here has been benchmarked against TE's
    rms_norm or stock F.rms_norm on RX 7900 XTX -- see
    enable_triton_kernels_rmsnorm's docstring before turning it on.

  - swiglu_fused() -- a manually-callable helper, never auto-patched,
    same shape as aiter_ops.fused_silu_mul: silu(gate)*up has no stock
    F.* op to intercept (F.silu is plain elementwise, one input, no second
    operand to multiply against), so there's nothing to transparently
    replace. Semantically identical to aiter_ops.fused_silu_mul, just
    taking gate/up as two separate tensors instead of one pre-concatenated
    one (aiter's convention splits a single (..., 2*d) tensor in half
    internally; triton_kernels' takes the two halves already split) -- use
    whichever matches how your model already produces gate/up.

Everything else triton-kernels ships (INT8/W4A16 quantized GEMM, MoE
routing/dispatch) is out of scope for this module: linear_int8
(aiter_ops.py) already covers W8A8 quantized linear for this project, and
this project doesn't run MoE models, so there's no call site for the MoE
kernels to plug into.
"""
from __future__ import annotations

from typing import Optional

import torch

try:
    from triton_kernels.rmsnorm import rmsnorm as _rmsnorm
    from triton_kernels.swiglu import swiglu_fused as _swiglu_fused

    _TRITON_KERNELS_AVAILABLE = True
except ImportError:
    _rmsnorm = None
    _swiglu_fused = None
    _TRITON_KERNELS_AVAILABLE = False


def available() -> bool:
    return _TRITON_KERNELS_AVAILABLE


def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: Optional[float] = None) -> torch.Tensor:
    """F.rms_norm-compatible wrapper around triton_kernels.rmsnorm.rmsnorm.
    normalized_shape isn't needed here -- triton_kernels' kernel always
    normalizes over the last dimension, same as F.rms_norm's own default
    (and only supported, for a 1D weight) usage."""
    return _rmsnorm(input, weight, eps=1e-6 if eps is None else eps)


def swiglu_fused(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """silu(gate) * up, fused. See module docstring for how this differs
    from aiter_ops.fused_silu_mul (two tensors instead of one split in
    half)."""
    return _swiglu_fused(gate, up)
