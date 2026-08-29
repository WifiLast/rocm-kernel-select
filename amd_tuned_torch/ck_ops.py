"""Composable Kernel WMMA conv2d/conv3d tier (fp16/bf16, groups=1).

Wraps the `ck_conv` entry point built from src/cuda/ck_conv_fwd*.cu (read
src/cuda/ck_conv_fwd.hpp first -- it documents why CK is a dependency again
after having been dropped as a GEMM backend, and what the layout constraint
costs). Everything here is thin: instance selection, workspace allocation
and the channels-last conversion all happen on the C++ side, per shape.

WHERE THIS TIER IS THE RIGHT ANSWER, measured on gfx1100 at tools/bench.py's
shapes (kernel time, channels-last input so no conversion is charged):

    conv2d fp16   stock 1.41 ms   CK 1.85 ms   -> stock wins (MIOpen
                  dispatches 3x3 fp16 to a hand-written assembly Winograd
                  solver at ~90% of peak; CK is implicit-GEMM and cannot
                  catch a cheaper algorithm)
    conv2d bf16   stock 8.72 ms   CK 1.87 ms   -> CK 4.7x  (MIOpen has NO
                  bf16 Winograd on gfx11 and falls back to im2col+GEMM)
    conv3d fp16   stock 3.02 ms   CK 1.16 ms   -> CK 2.6x
    conv3d bf16   stock 3.31 ms   CK 1.18 ms   -> CK 2.8x

So CK is the best kernel available on this card for three of those four
cases, and for conv2d fp16 it is merely much better than this project's own
hand-written kernel (6.5 ms) while still losing to stock. The tier order in
amd_tuned_torch/__init__.py reflects that; it does not attempt to decide
whether stock should be preferred over this project's kernels in general
(see the conv2d fp16 row -- that policy question is unresolved).

LAYOUT: CK's WMMA conv instances are channels-last only. A channels-last
caller pays nothing; an NCHW caller pays a permute in and a permute out,
which at the conv2d bf16 shape above is roughly 0.9 ms of the 8.7 ms it
saves -- still worth it, but it is why `conv2d`/`conv3d` here return the
output in whatever format they were handed rather than silently changing it.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from . import _native as _C

_CK_DTYPES = (torch.float16, torch.bfloat16)


def available() -> bool:
    """True if the extension was built with Composable Kernel present.

    False is normal, not an error: the tier is optional at build time (see
    setup.py's AMD_TUNED_TORCH_CK_ROOT detection), so callers must treat
    this as a capability check exactly like aiter_ops.available().
    """
    try:
        return bool(_C.has_ck())
    except AttributeError:
        return False


def _as_list(v, n: int) -> list:
    if isinstance(v, int):
        return [v] * n
    return list(v)


def _conv(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
          stride, padding, dilation, ndim: int) -> Optional[torch.Tensor]:
    if not available():
        return None
    if input.dim() != ndim + 2 or weight.dim() != ndim + 2:
        return None
    if input.dtype not in _CK_DTYPES or weight.dtype != input.dtype:
        return None
    # Returns None (not an exception) when no compiled CK instance supports
    # the problem -- CK rejects on vector-load alignment and tile
    # divisibility, so this is a routine outcome for odd channel counts,
    # not a failure.
    return _C.ck_conv(input, weight, bias,
                      _as_list(stride, ndim), _as_list(padding, ndim), _as_list(dilation, ndim))


def conv2d(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
           stride=1, padding=0, dilation=1) -> Optional[torch.Tensor]:
    """CK WMMA conv2d. None if unavailable or unsupported for this problem."""
    return _conv(input, weight, bias, stride, padding, dilation, ndim=2)


def conv3d(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
           stride=1, padding=0, dilation=1) -> Optional[torch.Tensor]:
    """CK WMMA conv3d. None if unavailable or unsupported for this problem."""
    return _conv(input, weight, bias, stride, padding, dilation, ndim=3)
