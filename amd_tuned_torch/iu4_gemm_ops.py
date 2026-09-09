"""EXPERIMENTAL gfx1100 WMMA INT4/INT8 GEMM -- explicit opt-in only, NOT a
kernel_select candidate for F.linear.

WHERE THIS CAME FROM. The three raw kernels this wraps
(src/cuda/iu4_gemm_fwd.cu) are ported from an isolated correctness/roofline
probe in trellis2-convrot-rocm (a TRELLIS.2-for-ComfyUI patch kit), MIT
licensed -- see amd_tuned_torch/_vendor/gfx1100_iu4_gemm/NOTICE.md for the
exact provenance and, importantly, for why the *other* kernel in that
project (the actually-tuned, TRELLIS-shape-specific per-row W8A8 Triton
GEMM behind its ConvRotLinear) was deliberately NOT vendored: it only
exists as a patch against an AGPL-3.0 repository, and copying it here would
pull AGPL-3.0 code into this project.

WHY THIS IS NOT WIRED INTO kernel_select. Every other GEMM candidate in
this package (ck_gemm_ops, hipblaslt_ops, splitk_gemm_ops) computes the
SAME answer as stock F.linear, just faster or not depending on shape --
kernel_select's correctness gate (torch.allclose against an fp16/bf16
reference, see kernel_select.py) exists to catch a candidate that
accidentally computes something else. An INT4/INT8-quantized GEMM
*deliberately* computes a different, lossy answer in exchange for integer
throughput -- entering it into that same contest would fail verification
on every real shape (quantization error routinely exceeds
kernel_select's dtype tolerances) and get it permanently blacklisted for
doing exactly what it was built to do. So this module is a plain library
function a caller opts into for specific layers it has decided are
quantization-tolerant (mirroring how upstream's ConvRotLinear replaces
specific Linear modules rather than monkeypatching F.linear globally),
not something that competes for every F.linear call automatically.

WHAT "UNVALIDATED" MEANS HERE, MORE THAN ELSEWHERE IN THIS PACKAGE. The
upstream probe never claimed to beat anything -- its own README calls it
"a correctness/roofline probe, not yet an LDS-staged production GEMM" with
"direct global-memory fragment loads", and its own dispatcher intent
explicitly says "Promotion requires beating the existing tuned W8A8
implementation". Nothing in cmp_ext_turing has a tuned W8A8 implementation
to beat, so there is no baseline this can even be measured against yet --
benchmark against `iu8_linear`'s own dot4_i8 fallback and against stock
F.linear (dequantized to the same nominal precision) for your actual shapes
before trusting a speed claim. Correctness (packed-integer accumulation is
exact; only the quantization step is lossy) has PC-side unit coverage in
tests/test_iu4_gemm_ops.py and hardware coverage in
tests_hardware/test_iu4_gemm_ops.py, but the *packing/dispatch glue* here
is new code with no upstream Python counterpart to compare against -- the
upstream probe packs/unpacks on the CPU purely to feed its own benchmark
harness, not to serve a real linear layer.

QUANTIZATION. Per-row (activations) / per-row (weight, i.e. per-output-
channel) symmetric quantization -- same shape of scheme as
triton_int8_linear_per_row's activation side, just derived independently
here rather than copied (see the module docstring above on why that
kernel's code itself wasn't vendored). `iu8_linear` quantizes to the full
int8 range [-127, 127]; `iu4_linear` quantizes to int4's signed range
[-7, 7] (one code point given up to keep the range symmetric, avoiding a
-8 with no positive counterpart) and packs two values per byte for the
native kernel's nibble layout.
"""
from __future__ import annotations

from typing import Optional

import torch

from . import _native as _C

_INT8_MAX = 127
_INT4_MAX = 7


def available() -> bool:
    """True if the extension exposes these kernels AND the current device
    is gfx1100 -- see src/main_rocm.cpp's iu4_gemm_supported()."""
    try:
        return hasattr(_C, "iu4_gemm_supported") and bool(_C.iu4_gemm_supported())
    except (AttributeError, RuntimeError):
        return False


def _quantize_rowwise(x: torch.Tensor, qmax: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric quantization to int8, values clamped to
    [-qmax, qmax]. Returns (x_int8 [rows, K], scale [rows] float32) with
    x ~= x_int8.float() * scale[:, None]."""
    x_f32 = x.reshape(-1, x.shape[-1]).float()
    row_max = x_f32.abs().amax(dim=-1).clamp_min(1e-12)
    scale = row_max / qmax
    x_int8 = (x_f32 / scale.unsqueeze(-1)).round().clamp(-qmax, qmax).to(torch.int8)
    return x_int8, scale


def pack_int4_rows(x_int8: torch.Tensor) -> torch.Tensor:
    """[rows, K] int8 (values in [-7, 7]) -> [rows, ceil(K,16)/2] uint8,
    two signed nibbles per byte (low nibble = even k, high nibble = odd k
    -- matches src/cuda/iu4_gemm_fwd.cu's load_i4_nibble). Padded to a
    16-column (8-byte) multiple so the native kernel's full-tile load path
    never reads past an odd allocation boundary, same margin the upstream
    probe's i4_stride uses."""
    rows, k = x_int8.shape
    k_padded = (k + 15) // 16 * 16
    if k_padded != k:
        pad = torch.zeros((rows, k_padded - k), dtype=torch.int8, device=x_int8.device)
        x_int8 = torch.cat([x_int8, pad], dim=-1)
    nibbles = (x_int8.to(torch.uint8) & 0x0F)
    lo = nibbles[:, 0::2]
    hi = nibbles[:, 1::2]
    return (lo | (hi << 4)).contiguous()


def iu8_linear(input: torch.Tensor, weight: torch.Tensor,
               bias: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
    """Y ~= X @ W^T (+ bias) via per-row INT8 quantization and a native
    gfx1100 WMMA integer GEMM. Lossy (int8 quantization on both operands) --
    an explicit choice by the caller, never a kernel_select candidate (see
    module docstring). None if unavailable or the shape/dtype isn't
    supported; never raises for an ordinary ineligible call, same
    drop-out convention as every other tier in this package."""
    if not available():
        return None
    if not isinstance(input, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return None
    if not input.is_cuda or not weight.is_cuda or weight.dim() != 2:
        return None
    if input.shape[-1] != weight.shape[-1]:
        return None
    try:
        orig_shape = input.shape
        x_int8, x_scale = _quantize_rowwise(input, _INT8_MAX)
        w_int8, w_scale = _quantize_rowwise(weight, _INT8_MAX)
        acc = _C.iu8_gemm(x_int8.contiguous(), w_int8.contiguous())
        out = acc.float() * x_scale.unsqueeze(-1) * w_scale.unsqueeze(0)
        if bias is not None:
            out = out + bias.float()
        out = out.to(input.dtype)
        return out.reshape(*orig_shape[:-1], weight.shape[0])
    except (RuntimeError, TypeError):
        return None


def iu4_linear(input: torch.Tensor, weight: torch.Tensor,
               bias: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
    """Same contract as iu8_linear, at INT4 precision (both operands
    quantized to [-7, 7] and nibble-packed) -- materially lossier than
    iu8_linear; only worth it if the WMMA IU4 throughput advantage matters
    more than the extra quantization error for your layer. See module
    docstring for what "unvalidated" means for this specific kernel."""
    if not available():
        return None
    if not isinstance(input, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return None
    if not input.is_cuda or not weight.is_cuda or weight.dim() != 2:
        return None
    if input.shape[-1] != weight.shape[-1]:
        return None
    try:
        orig_shape = input.shape
        k = input.shape[-1]
        x_int8, x_scale = _quantize_rowwise(input, _INT4_MAX)
        w_int8, w_scale = _quantize_rowwise(weight, _INT4_MAX)
        x_packed = pack_int4_rows(x_int8)
        w_packed = pack_int4_rows(w_int8)
        acc = _C.iu4_gemm(x_packed, w_packed, k)
        out = acc.float() * x_scale.unsqueeze(-1) * w_scale.unsqueeze(0)
        if bias is not None:
            out = out + bias.float()
        out = out.to(input.dtype)
        return out.reshape(*orig_shape[:-1], weight.shape[0])
    except (RuntimeError, TypeError):
        return None
