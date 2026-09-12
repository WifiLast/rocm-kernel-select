"""rocSPARSE-backed sparse-tensor matmul (SpMM) for amd_tuned_torch.

WHERE THIS COMES FROM. rocSPARSE is vendored in full at
third_party/rocsparse -- this package's own copy, not a sibling checkout,
same "everything needed lives under source/cmp_ext_turing" posture as
third_party/composable_kernel/third_party/rocm-headers. Unlike CK, rocSPARSE
is NOT header-only: its device kernels live in a compiled librocsparse.so,
so the vendored tree is source for a full build (its own CMake/install.sh),
not a drop-in header fallback the way third_party/rocm-headers is for
hipBLASLt. See setup.py's rocSPARSE tier for exactly what it looks for and
in which order (a system ROCm's rocsparse-dev package first -- the common
case, since ROCm ships this alongside hipBLASLt -- then a from-source build
of the vendored copy at a conventional install prefix).

WHY THIS EXISTS. torch.matmul already accepts a torch.sparse_csr/csc/coo
`input` and produces a correct result through ATen's own generic sparse
dispatch on ROCm -- this tier is not filling a correctness gap the way
flexgemm_ops's sparse-voxel conv tiers are (F.conv3d has no sparse
equivalent at all). It exists for the same reason the hipBLASLt/CK GEMM
tiers exist for the DENSE case: ATen's generic path is not guaranteed to be
the fastest one available on this card, and rocSPARSE's own generic SpMM
(rocsparse_spmm, see src/rocsparse_spmm.cpp) is a real vendor-tuned kernel
where ATen's dispatch may fall back to something more generic.

MEASURED, NOT ASSUMED. Being a real vendor library is not by itself
evidence rocSPARSE beats ATen's own sparse dispatch on this card -- the same
"prior belief about our own kernel" trap kernel_select.py's own module
docstring documents rocBLAS/MIOpen already disproving for conv2d/linear/bmm.
So `amd_tuned_torch._dispatch._patched_matmul` does not call `spmm`/
`maybe_spmm` unconditionally the way flexgemm_ops's occupancy-gated conv
fast paths do -- it contests `maybe_spmm` against stock through
`kernel_select.cached_key`/`pick_key` (kind "matmul_sparse", keyed on
dtype + both operands' layout + both shapes, since a sparse `input`'s
layout changes which candidates even apply), the same benchmark-once-
cache-the-winner policy conv2d/conv3d/linear/bmm/fftconv1d already use.
`maybe_spmm` itself still declines cleanly (returns None) when ineligible
or disabled, which is exactly what a kernel_select candidate thunk is
required to do.

TEST FOR SPARSE, NOT OCCUPANCY. Unlike flexgemm_ops.maybe_sparse_conv{2,3}d
-- which scan a DENSE tensor's actual content to guess whether it's mostly
empty -- `is_spmm_eligible` below tests `input`'s LAYOUT (torch.sparse_csr/
csc/coo vs. the ordinary torch.strided), which is metadata, not data: a
sparse-layout tensor already says how it is stored, and a caller hands this
package one BECAUSE it is sparse, not because a heuristic guessed so. No
reduction over `input` is needed the way flexgemm_ops's occupancy estimate
needs one -- see amd_tuned_torch._dispatch._patched_matmul's own comment at
its call site for the same distinction.

WHAT'S SUPPORTED. `other` (the dense operand) must be an ordinary 2D
torch.strided CUDA tensor and `input` a 2D sparse CUDA tensor, same
fp16/bf16/fp32 dtype as `other`. rocSPARSE's generic SpMM also supports
batched CSR/COO and BSR/blocked-ELL formats (see rocsparse_spmm.h's own
docs) -- none of that is wired up here: `_patched_matmul` only ever reaches
this tier for the shape `is_spmm_eligible` checks, an unbatched
sparse-matrix-times-dense-matrix problem (a GNN adjacency/aggregation
matmul, a sparse embedding table, ...), the common case rather than the
rarer batched one. A COO or CSC `input` is converted to CSR via
`Tensor.to_sparse_csr()` before reaching the native call (one cheap format
conversion torch already implements correctly) so the C++ side only ever
has to build one kind of sparse-matrix descriptor.

VALIDATION STATUS -- READ BEFORE TRUSTING THIS ON HARDWARE. Written in a dev
environment with no ROCm/HIP toolchain and no GPU (same posture as
flash_mm_kernel.py's own VALIDATION STATUS section -- see that module's
docstring for why this project says so plainly rather than staying silent).
The CSR descriptor setup and the three-stage rocsparse_spmm call
(buffer_size/preprocess/compute) in src/rocsparse_spmm.cpp follow the
vendored header's documented contract, but the call has never been compiled
or run against real device memory. `available()` reports False until the
extension is actually built (setup.py's rocSPARSE tier), so until then this
whole module is a documented no-op that always declines --
`_patched_matmul`'s fast path simply never fires and every sparse matmul
call keeps going through stock, exactly as it did before this module
existed. Before relying on `spmm`/`maybe_spmm` for anything real, build the
extension against a real ROCm GPU and compare its output against
`torch.sparse.mm`/`input.to_dense() @ other` for both CSR and COO inputs;
treat any mismatch as this module's bug, not rocSPARSE's.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

# amd_tuned_torch/__init__.py always sets this attribute (to None when the
# tier wasn't built, required=False -- see its own _native_rocsparse loader
# call), same split-extension convention as ._native_ck/._native_hipblaslt,
# so this import always succeeds; `_C` being None is what available() below
# checks for, exactly like hipblaslt_ops.py's own `_C`.
from . import _native_rocsparse as _C

# torch.sparse_csc has no direct .to_sparse_csr() in every torch version
# this package supports, so it goes through .to_sparse_csr() the same as
# COO -- both are one conversion call away from the CSR the native side
# actually implements. torch.sparse_bsr/bsc/coo-with-batch-dims are
# deliberately excluded: rocsparse_spmm.h supports batched formats, but
# `is_spmm_eligible` below only ever offers this tier the unbatched 2D
# case (see this module's WHAT'S SUPPORTED section), and converting a
# blocked format to CSR would silently discard the block structure it was
# chosen for.
_CONVERTIBLE_SPARSE_LAYOUTS = (torch.sparse_coo, torch.sparse_csc)
_SPARSE_LAYOUTS = (torch.sparse_csr,) + _CONVERTIBLE_SPARSE_LAYOUTS
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _env_flag(name: str, default: str = "1") -> bool:
    """Same on/off-string parsing as flexgemm_ops._env_flag -- not imported
    from there to avoid a cross-module dependency for one four-line
    helper."""
    return os.environ.get(name, default).strip().lower() not in ("0", "", "false", "no", "off")


_SPMM_ENABLED = _env_flag("AMD_TUNED_TORCH_ROCSPARSE_SPMM", default="1")


def available() -> bool:
    """True if the extension was built with rocSPARSE present (see
    setup.py's rocSPARSE tier). False is normal, not an error: the tier is
    optional at build time exactly like hipblaslt_ops.available() and
    ck_ops.available(), so callers must treat this as a capability check,
    not assume it."""
    try:
        return bool(_C.has_rocsparse())
    except AttributeError:
        return False


def spmm_enabled() -> bool:
    """True unless AMD_TUNED_TORCH_ROCSPARSE_SPMM=0. Read once at import
    time, same posture as every other env-var gate in this package. Does
    not by itself mean maybe_spmm will actually route to rocSPARSE for any
    given call -- see is_spmm_eligible for the per-call layout/shape test
    that decides that."""
    return _SPMM_ENABLED


def is_spmm_eligible(input: torch.Tensor, other: torch.Tensor) -> bool:
    """True when `input`/`other` are exactly the shape `spmm` supports: a 2D
    sparse CSR/CSC/COO CUDA `input` times a 2D dense (torch.strided) CUDA
    `other`, same dtype, both in {fp16, bf16, fp32} (the uniform/mixed-
    precision rows of rocsparse_spmm's own dtype table -- see
    rocsparse_spmm.h). Batched sparse matmul, a sparse `other`, or a dtype
    mismatch all decline -- see this module's docstring for why only the
    unbatched case is wired up."""
    if not isinstance(input, torch.Tensor) or not isinstance(other, torch.Tensor):
        return False
    if input.layout not in _SPARSE_LAYOUTS or other.layout != torch.strided:
        return False
    if not input.is_cuda or not other.is_cuda:
        return False
    if input.dim() != 2 or other.dim() != 2:
        return False
    if input.dtype != other.dtype or input.dtype not in _DTYPES:
        return False
    return input.shape[1] == other.shape[0]


def spmm(input: torch.Tensor, other: torch.Tensor) -> Optional[torch.Tensor]:
    """input @ other via rocSPARSE's generic SpMM, `input` sparse CSR/CSC/
    COO, `other` dense -- [M,K] @ [K,N] -> dense [M,N]. None if
    unavailable/ineligible/the call fails for any reason, matching every
    other opportunistic-kernel candidate in this package (see
    hipblaslt_ops.linear's docstring: returning None rather than raising is
    routine, not a failure)."""
    if not available() or not is_spmm_eligible(input, other):
        return None
    if input.layout in _CONVERTIBLE_SPARSE_LAYOUTS:
        # One-time format conversion, not a per-element scan -- the native
        # side only ever builds a CSR descriptor (see this module's
        # docstring), so a COO/CSC caller pays this instead of every future
        # caller needing its own CSC/COO native path.
        try:
            input = input.to_sparse_csr()
        except (RuntimeError, TypeError):
            return None
    try:
        return _C.rocsparse_spmm(input, other)
    except (RuntimeError, TypeError):
        return None


def maybe_spmm(input: torch.Tensor, other: torch.Tensor) -> Optional[torch.Tensor]:
    """spmm(...) gated by spmm_enabled() -- the one entry point
    amd_tuned_torch._dispatch._patched_matmul actually calls. None whenever
    AMD_TUNED_TORCH_ROCSPARSE_SPMM=0, on top of every reason spmm() itself
    can decline."""
    if not spmm_enabled():
        return None
    return spmm(input, other)
