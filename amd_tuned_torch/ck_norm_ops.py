"""Composable Kernel GroupNorm tier, with SiLU optionally fused (fp16/bf16/fp32).

Wraps the `ck_group_norm` entry point built from src/cuda/ck_norm_fwd*.cu --
read src/cuda/ck_norm_fwd.hpp first; it explains why the *fused* form is the
point and plain GroupNorm is the side effect. Everything here is thin:
instance selection, workspace allocation and the channels-last view all
happen on the C++ side, cached per shape.

LAYOUT DECIDES EVERYTHING HERE, more starkly than for the CK conv tier.
Measured on RX 7900 XTX against stock (torch 2.15 / ROCm 7.2), on the
GroupNorm->SiLU pair a diffusion U-Net ResBlock actually runs:

    shape                      NHWC (free)   NCHW (permutes)
    2x320x64x64   fp16            1.64x           0.76x
    2x640x32x32   fp16            1.88x           0.70x
    2x1280x16x16  fp16            2.10x           0.74x
    2x640x32x32   fp32            2.12x           0.67x

Channels-last is a 1.6-2.1x win; contiguous NCHW is a LOSS, because the
permute in and out costs more than the fused activation saves. So the
wrappers here decline NCHW input for the fused path rather than quietly
taking it -- an automatic tier that made a graph slower would be worse than
no tier. A caller that knows it wants this can force it with
`allow_permute=True`.

Plain GroupNorm (no fusion) is also faster than stock in channels-last
(1.26-2.09x) and slower in NCHW, and is offered on the same terms.

AFFINE ONLY: CK's kernels read gamma and beta unconditionally, with no
affine=False path, so weight=None or bias=None declines.
"""
from __future__ import annotations

from typing import Optional

import torch

from . import _native_ck as _C

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def available() -> bool:
    """True if the extension was built with Composable Kernel present.

    False is normal, not an error -- the tier is optional at build time.
    """
    try:
        return bool(_C.has_ck()) and hasattr(_C, "ck_group_norm")
    except AttributeError:
        return False


def _channels_last(x: torch.Tensor) -> bool:
    if x.dim() == 4:
        return x.is_contiguous(memory_format=torch.channels_last)
    if x.dim() == 5:
        return x.is_contiguous(memory_format=torch.channels_last_3d)
    return False


def group_norm(input: torch.Tensor, num_groups: int, weight: Optional[torch.Tensor],
               bias: Optional[torch.Tensor], eps: float = 1e-5, *,
               fuse_silu: bool = False,
               allow_permute: bool = False) -> Optional[torch.Tensor]:
    """CK GroupNorm, optionally with SiLU fused. None if unsupported.

    Declines contiguous-NCHW input unless `allow_permute=True`: see this
    module's docstring for the measurements: that layout is a loss, and a
    tier that silently made the caller slower would be worse than nothing.
    """
    if not available():
        return None
    if not isinstance(input, torch.Tensor) or not input.is_cuda:
        return None
    if input.dtype not in _DTYPES:
        return None
    if input.dim() not in (4, 5):
        return None
    if weight is None or bias is None:
        return None
    if weight.dtype != input.dtype or bias.dtype != input.dtype:
        return None
    if not allow_permute and not _channels_last(input):
        return None
    try:
        return _C.ck_group_norm(input, int(num_groups), weight, bias, float(eps),
                                bool(fuse_silu))
    except (RuntimeError, TypeError):
        return None


def group_norm_silu(input: torch.Tensor, num_groups: int, weight: Optional[torch.Tensor],
                    bias: Optional[torch.Tensor], eps: float = 1e-5, *,
                    allow_permute: bool = False) -> Optional[torch.Tensor]:
    """F.silu(F.group_norm(...)) as ONE kernel -- the reason this tier exists.

    The activation is applied to values still in registers, so the [N, C, *]
    tensor is written once instead of twice. 1.6-2.1x over the two-kernel
    form in channels-last; see this module's docstring.
    """
    return group_norm(input, num_groups, weight, bias, eps, fuse_silu=True,
                      allow_permute=allow_permute)
