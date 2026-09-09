"""Composable Kernel WMMA GEMM tier for F.linear (fp16/bf16).

Wraps the `ck_gemm_linear` entry point built from src/cuda/ck_gemm_fwd*.cu
-- read src/cuda/ck_gemm_fwd.hpp first; it documents why CK is a GEMM
dependency again after having been dropped as one, and carries the
measurements that motivated it. Everything here is thin: instance
selection, workspace allocation and the bias broadcast all happen on the
C++ side, cached per shape.

WHAT THIS IS FOR. It is a *third* candidate for `F.linear`, alongside
stock and hipblaslt_ops, not a replacement for either. It earns its place
by winning where hipBLASLt loses. Measured on RX 7900 XTX against stock
(rocBLAS = 1.00x), fp16:

                              hipblaslt_ops    this tier
    4096^3                        1.29x          1.20x
    8192x4096x11008               1.09x          1.11x
    4096x4096x1024                1.49x          1.47x
    4096x1280x1280                0.96x          1.14x
    1024^3                        1.02x          1.21x
    1x4096x4096 (decode)          0.60x          0.90x

hipBLASLt owns the large end and collapses on the small one; CK is flatter,
slightly behind at 4096^3 and ahead everywhere hipBLASLt is behind. Neither
dominates, which is exactly what kernel_select's per-shape contest is for.

(An earlier version of this docstring justified the tier by a supposed 6x
efficiency hole at square-4096 shapes. That was measured while the machine
was compiling this extension and recorded CPU starvation, not the GPU; on
an idle machine stock reaches 64 TF/s there. The tier is still worth having
-- for the reason in the table, not that one.)

FUSED EPILOGUES use the SAME integer vocabulary as hipblaslt_ops
(EPILOGUE_NONE/BIAS/GELU/SILU/RELU), on purpose: a caller decides once that
it wants `linear + bias + GELU` and offers that to every candidate, rather
than translating between two enums. CK has no ReLU instances compiled in
here and no unbiased-activation instances, so those requests return None --
which is how a candidate drops out of a contest, not an error.

NUMERICS: `linear_gelu` is CK's FastGelu, the *tanh* approximation, which
matches `F.gelu(approximate="tanh")` and not the exact erf default. Same
line hipblaslt_ops.py and te_ops.py draw, for the same reason.

fp16/bf16 only. CK ships no fp32 WMMA GEMM instances -- fp32 has no WMMA
path on gfx1100 at all -- so fp32 linear stays with stock, where it was.
"""
from __future__ import annotations

from typing import Optional

import torch

from . import _native_ck as _C
# The epilogue vocabulary is shared with the hipBLASLt tier; it is defined
# there because that tier is the one whose C++ header owns the enum.
from .hipblaslt_ops import (  # noqa: F401  (re-exported for callers)
    EPILOGUE_NONE,
    EPILOGUE_BIAS,
    EPILOGUE_GELU,
    EPILOGUE_SILU,
    EPILOGUE_RELU,
)

_DTYPES = (torch.float16, torch.bfloat16)


def available() -> bool:
    """True if the extension was built with the CK GEMM tier present.

    False is normal, not an error: the tier is optional at build time (see
    setup.py's AMD_TUNED_TORCH_CK_GEMM detection), so callers must treat
    this as a capability check exactly like ck_ops.available().

    As in ck_ops.available(), has_ck() only says some CK tier was built --
    this tier is the most expensive half of the CK build cost and is the
    one most often switched off, so its own entry point is what gets
    checked.
    """
    try:
        return bool(_C.has_ck()) and hasattr(_C, "ck_gemm_linear")
    except AttributeError:
        return False


def is_eligible(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> bool:
    """Every shape/dtype/device check `linear` applies before ever calling
    the native extension -- pulled out into its own function (rather than
    inlined in `linear` the way it used to be) so
    amd_tuned_torch.compile_ops's register_fake for this op can replicate
    the exact same eligibility decision at trace time instead of
    duplicating it by hand. Every check here inspects only static tensor
    metadata (dtype/dim/is_cuda/shape) -- none of it is data-dependent --
    so it is exactly as valid to call on a FakeTensor under
    torch.compile/FakeTensorMode as on a real one; `available()` is a
    plain build-time capability flag, equally valid either way."""
    if not available():
        return False
    if not isinstance(input, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return False
    if not input.is_cuda or not weight.is_cuda:
        return False
    if input.dtype not in _DTYPES or weight.dtype != input.dtype:
        return False
    if input.dim() < 2 or weight.dim() != 2:
        return False
    if bias is not None:
        if not bias.is_cuda or bias.dtype != input.dtype:
            return False
        if bias.dim() != 1 or bias.size(0) != weight.size(0):
            return False
    return True


def linear(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
           epilogue: int = EPILOGUE_NONE) -> Optional[torch.Tensor]:
    """Y = X @ W^T (+ bias), `epilogue` fused. None if unavailable/unsupported.

    Weight is [N, K] and is passed through untransposed: CK's Row/Col/Row
    layout triple is F.linear's own layout, so no operand is copied.
    """
    if not is_eligible(input, weight, bias):
        return None
    try:
        return _C.ck_gemm_linear(input, weight, bias, int(epilogue))
    except (RuntimeError, TypeError):
        return None


def linear_gelu(input: torch.Tensor, weight: torch.Tensor,
                bias: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
    """linear(...) with tanh-approximate GELU fused into the epilogue.

    Equivalent to F.gelu(F.linear(x, w, b), approximate="tanh") -- NOT to
    F.gelu's exact erf default. Requires a bias; see this module's docstring.
    """
    return linear(input, weight, bias, EPILOGUE_GELU)


def linear_silu(input: torch.Tensor, weight: torch.Tensor,
                bias: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
    """linear(...) with SiLU fused into the epilogue. Requires a bias."""
    return linear(input, weight, bias, EPILOGUE_SILU)
