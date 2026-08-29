"""hipBLASLt GEMM tier for linear/matmul/bmm (fp16/bf16/fp32).

Wraps the `hipblaslt_linear` / `hipblaslt_bmm` entry points built from
src/hipblaslt_gemm.cpp -- read src/hipblaslt_gemm.hpp first; it documents
why this tier exists and carries the measurements. Everything here is thin:
the column-major reinterpretation, algorithm selection and workspace
allocation all happen on the C++ side, cached per problem.

THE SHORT VERSION. `F.linear`/`torch.matmul`/`torch.bmm` had no non-stock
candidate on gfx1100 -- aiter's Triton GEMM raises KeyError('gfx1100')
because it ships no RDNA3 tuning config -- so kernel_select.py excluded
them from its contest. Meanwhile PyTorch's default ROCm BLAS backend is
rocBLAS, whose gfx1100 Tensile libraries are all named
`TensileLibrary_*_fallback_gfx1100` (untuned fallback), and hipBLASLt --
already installed, same ROCm -- ships a tuned `navi31` logic set. Measured
with F.linear on this card:

    4096x4096x4096    bf16   rocBLAS 10.78 ms   hipBLASLt  8.85 ms  1.22x
    8192x4096x11008   bf16   rocBLAS 11.71 ms   hipBLASLt  9.65 ms  1.21x
    4096x4096x1024    bf16   rocBLAS  4.02 ms   hipBLASLt  2.00 ms  2.01x
    4096x4096x1024    fp16   rocBLAS  5.95 ms   hipBLASLt  4.01 ms  1.48x
    1024x1024x1024    bf16   rocBLAS  0.27 ms   hipBLASLt  0.30 ms  0.88x
    4096x1280x1280    fp16   rocBLAS  1.69 ms   hipBLASLt  1.99 ms  0.85x

Large and skinny shapes win, small ones lose. That is a per-shape decision,
which is precisely what kernel_select.py is for -- this tier is registered
as a candidate there, not as a replacement.

FUSED EPILOGUES are the part that is not reachable any other way. Setting
`torch.backends.cuda.preferred_blas_library("hipblaslt")` would get the
table above and stop there. Calling the library directly also gets bias,
GELU, SiLU and ReLU applied inside the GEMM epilogue, saving a full
round-trip of the [M, N] activation through HBM per fused op. Use
`linear_gelu` / `linear_silu` where the model really does apply that
activation to a linear's output -- for an MLP's up-projection that is one
of the two largest tensors in the block.

NUMERICS: `linear_gelu` is the *tanh* approximation of GELU, matching
`F.gelu(approximate="tanh")` and not the exact erf default. te_ops.py draws
the same line for the same reason -- a fused kernel that silently changed
the default's numerics would be a correctness surprise, not an
optimisation. Callers are responsible for only fusing what the model asked
for; nothing here inspects the graph.
"""
from __future__ import annotations

from typing import Optional

import torch

from . import _native as _C

# Mirrors AmdTunedTorchEpilogue in src/hipblaslt_gemm.hpp. Kept as plain
# ints so this module never has to import a ROCm header's enum.
EPILOGUE_NONE = 0
EPILOGUE_BIAS = 1
EPILOGUE_GELU = 2
EPILOGUE_SILU = 3
EPILOGUE_RELU = 4

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def available() -> bool:
    """True if the extension was built with hipBLASLt present.

    False is normal, not an error: the tier is optional at build time (see
    setup.py's detection and AMD_TUNED_TORCH_HIPBLASLT), so callers must
    treat this as a capability check exactly like ck_ops.available() and
    aiter_ops.available().
    """
    try:
        return bool(_C.has_hipblaslt())
    except AttributeError:
        return False


def _eligible(*tensors: Optional[torch.Tensor]) -> bool:
    ref = None
    for t in tensors:
        if t is None:
            continue
        if not isinstance(t, torch.Tensor) or not t.is_cuda:
            return False
        if t.dtype not in _DTYPES:
            return False
        if ref is None:
            ref = t.dtype
        elif t.dtype != ref:
            return False
    return ref is not None


def linear(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
           epilogue: int = EPILOGUE_NONE) -> Optional[torch.Tensor]:
    """Y = X @ W^T (+ bias), `epilogue` fused. None if unavailable/unsupported.

    Weight is [N, K] and is passed through untransposed -- F.linear's own
    layout -- so there is no pre-transpose copy here or on the C++ side.

    Returning None rather than raising is deliberate and routine: the
    installed navi31 logic legitimately has no kernel for some
    dtype/epilogue/shape combinations, and the caller is expected to have a
    stock fallback (see kernel_select.py's contest).
    """
    if not available() or not _eligible(input, weight, bias):
        return None
    if input.dim() < 2 or weight.dim() != 2:
        return None
    if bias is not None and (bias.dim() != 1 or bias.size(0) != weight.size(0)):
        return None
    try:
        return _C.hipblaslt_linear(input, weight, bias, int(epilogue))
    except (RuntimeError, TypeError) as _:
        return None


def linear_gelu(input: torch.Tensor, weight: torch.Tensor,
                bias: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
    """linear(...) with tanh-approximate GELU fused into the epilogue.

    Equivalent to F.gelu(F.linear(x, w, b), approximate="tanh") -- NOT to
    F.gelu's exact erf default. See this module's docstring.
    """
    return linear(input, weight, bias, EPILOGUE_GELU)


def linear_silu(input: torch.Tensor, weight: torch.Tensor,
                bias: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
    """linear(...) with SiLU/Swish fused into the epilogue.

    Equivalent to F.silu(F.linear(x, w, b)); hipBLASLt's SWISH epilogue is
    Swish(x, 1), which is exactly SiLU.
    """
    return linear(input, weight, bias, EPILOGUE_SILU)


def linear_relu(input: torch.Tensor, weight: torch.Tensor,
                bias: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
    """linear(...) with ReLU fused into the epilogue."""
    return linear(input, weight, bias, EPILOGUE_RELU)


def bmm(input: torch.Tensor, mat2: torch.Tensor) -> Optional[torch.Tensor]:
    """C[i] = A[i] @ B[i] for 3D inputs -- torch.bmm's semantics, so mat2 is
    [batch, K, N] and is not transposed on the way in. None if
    unavailable/unsupported."""
    if not available() or not _eligible(input, mat2):
        return None
    if input.dim() != 3 or mat2.dim() != 3:
        return None
    try:
        return _C.hipblaslt_bmm(input, mat2)
    except (RuntimeError, TypeError):
        return None


def matmul(input: torch.Tensor, other: torch.Tensor) -> Optional[torch.Tensor]:
    """torch.matmul for the batched cases this tier covers: 2D @ 2D, 3D @ 3D,
    and >=4D @ >=4D where both operands share the exact same batch shape.

    Broadcasting operands fall through (None): hipBLASLt's strided-batch
    mode has no broadcast semantics of its own, and faking one by expanding
    would materialise a copy that costs more than the GEMM saves. This is
    the same restriction, for the same reason, that aiter_ops' 4D path
    documents in __init__._patched_matmul.
    """
    if not available() or not _eligible(input, other):
        return None
    if input.dim() == 2 and other.dim() == 2:
        out = bmm(input.unsqueeze(0), other.unsqueeze(0))
        return None if out is None else out.squeeze(0)
    if input.dim() == 3 and other.dim() == 3:
        return bmm(input, other)
    if input.dim() == other.dim() >= 4 and input.shape[:-2] == other.shape[:-2]:
        batch_shape = input.shape[:-2]
        flat = bmm(input.reshape(-1, *input.shape[-2:]),
                   other.reshape(-1, *other.shape[-2:]))
        return None if flat is None else flat.view(*batch_shape, *flat.shape[-2:])
    return None
