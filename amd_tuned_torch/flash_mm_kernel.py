"""
"Flash" (memory-/IO-aware) butterfly matrix multiplication.

`triton_kernel.py`'s `butterfly_mm_triton` launches one Triton kernel PER
butterfly stage (e = log2(L) launches). Each launch reads the *entire*
(B*F, L) tensor from GPU HBM and writes it back before the next stage's
kernel can start -- e-1 redundant HBM round trips per row that carry no
useful work, purely because each stage is a separate kernel. The one-shot
"fused" kernel already in this repo (`optimization/fused_kernel.py`) fixes
this for exactly L=8 by hand-unrolling both stages -- this module
generalizes that idea to any power-of-two L in one kernel body, following
the same "IO-awareness" framing FlashAttention popularized (Dao et al.):
minimize HBM traffic by keeping the working set resident in fast on-chip
memory (registers here, since a whole row fits) across every stage of a
fused computation, touching global memory exactly once on the way in and
once on the way out.

ALGORITHM. Per stage with stride s = 2^i, the classic butterfly step pairs
element j with element j+s inside each contiguous block of 2s elements:
    [y_j, y_{j+s}] = [[a0, b0], [a1, b1]] . [x_j, x_{j+s}]
(same coefficient-layout convention as `triton_kernel.py`/`reference_impl.py`:
w_stage[j, :] = [a0, a1] for the low ("j0") half of a pair, w_stage[j+s, :]
= [b0, b1] for the high ("j1") half.) Within a block of 2s elements the
partner of element c is exactly `c ^ s` -- s is a power of two and the
blocks are aligned, so the "is this the low or the high half" bit IS bit s
of the index -- which collapses both halves of every pair into a single
expression evaluated for all L columns at once:

    k   = 1 if (c & s) else 0        # which coefficient slot column c reads
    y_c = w[i, c, k] * x_c + w[i, c^s, k] * x_{c^s}

The kernel below implements one stage as exactly that: two small coefficient
loads and one `tl.gather` along the L axis for the partner shuffle, with the
(BLOCK_BF, L) tile living in registers from the single HBM read to the single
HBM write, so consecutive stages hand off through registers and never touch
global memory in between. See the block comment on the kernel for why this
is written as an XOR shuffle rather than the more obvious
reshape-to-(num_blocks, 2, s) regrouping (short version: the reshape makes
`s` a shape, shapes must be compile-time ints, and under Triton 3.7 a
`tl.static_range` induction variable is not one).

VALIDATION STATUS. The stage index arithmetic is mirrored in pure PyTorch as
`flash_butterfly_mm_torch` and checked bit-exact against
`reference_impl.butterfly_mm_ref` for L in {2,4,...,256} and both stage
orders in `unittests/test_flash_mm_kernel.py`; that mirror runs anywhere
torch does and needs no GPU. The `@triton.jit` kernel itself has now been
compiled and run on real hardware (ROCm 7.2 / Triton 3.7.1, Radeon RX 7900
XTX) and agrees with the torch mirror to <= 2.8e-7 max relative error in
fp32 across L in {8,64,512} x both stage orders. It has NOT been exercised
on an NVIDIA backend, and `tl.gather` is a comparatively recent Triton
primitive -- if you move this to a different Triton version or vendor,
re-run that comparison before trusting it rather than assuming it ports.

PERFORMANCE -- MEASURE BEFORE ASSUMING ANY PATH HERE IS THE FAST ONE.
Benchmarked on the card above, fp32, interleaved medians (an earlier revision
of this paragraph quoted single-shot numbers taken with an untuned BLOCK_BF
and was wrong by up to 6x -- if you re-measure, interleave the candidates and
tune BLOCK_BF before believing a ratio).

Plain fused butterfly vs a dense `X @ M` GEMM against the (L, L) matrix the
factors imply, BF=65536, BLOCK_BF tuned per L:

    L=8   0.085ms vs 0.072ms dense   dense   1.19x
    L=64  0.084ms vs 0.214ms dense   BUTTERFLY 2.56x
    L=128 0.114ms vs 0.191ms dense   BUTTERFLY 1.67x
    L=256 0.502ms vs 0.306ms dense   dense   1.64x
    L=1024 15.5ms vs 4.73ms dense    dense   3.28x

So the fused kernel's window is roughly L=32..128. Past that the whole
(BLOCK_BF, L) tile stops fitting and it spills: effective bandwidth falls to
37 GB/s at L=1024 against ~960 GB/s peak, and the O(L log L) advantage never
materialises.

`flash_butterfly_mm_monarch` covers the large-L end instead, by regrouping the
same arithmetic into block-diagonal Monarch factors and working in (L, BF)
layout (see the block comment on the kernels for why the layout, not the FLOP
count, is what mattered). At BF=65536, as a drop-in taking and returning the
natural (BF, L) layout:

    L=256  0.508ms vs 0.330ms dense   dense 1.54x
    L=512  1.092ms vs 1.251ms dense   MONARCH 1.15x
    L=1024 2.524ms vs 4.863ms dense   MONARCH 1.93x   (5.7x over the butterfly kernel)

running at 701 GB/s -- 73% of peak, i.e. bandwidth-bound, which is the ceiling
for an op with this little arithmetic. A caller that can keep activations in
(L, BF) across layers skips the output transpose and gets 3.24x at L=1024
instead of 1.93x, so the layout is worth pushing outward rather than
transposing per call.

The crossover depends on batch too, not just L (at L=512: 1.32x over dense at
BF=8192 but 0.48x at BF=2048), which is why none of this is hardcoded beyond
the `_MONARCH_MIN_L` pre-filter -- the kernel_select contest in
`flash_butterfly_mm` picks per shape. Do not hard-prefer any of these three
on the strength of one being compiled.

SCOPE. Forward-only, matching every other kernel in this repository (see
`README.md`'s note that autograd requires falling back to
`reference_impl.py`) -- no backward pass is implemented here either. `L`
must fit in one program's registers/on-chip memory as a live (BLOCK_BF, L)
tile across all e stages simultaneously (stricter than the per-stage
kernel, which only ever holds one stage's data live) -- this repo's own
tuning notes found BLOCK_BF=64 / L<=2048 already the practical ceiling for
the *per-stage* kernel, so treat that L as this module's practical ceiling
too until benchmarked otherwise; very large L will need genuine tiling
across row-chunks of L, not attempted here.

MONARCH MATRICES (below, `MonarchLinear`) -- A DIFFERENT, UNRELATED
STRUCTURED-MATRIX CLASS, added to this file on request rather than as its
own module. Everything above this point is butterfly matrix
multiplication (power-of-2 sizes only, forward-only, restricted to
factors already given as butterfly parameters). Monarch matrices (Dao,
Chen et al., "Monarch: Expressive Structured Matrices for Efficient and
Accurate Training", arXiv:2204.00595) solve the exact limitation the
"can I just pad it?" investigation above ran into: butterfly's recursive
stride-doubling structure only partitions cleanly when L is a power of 2
-- for a non-power-of-2 size, zero-padding contaminates the kept output
starting at the SECOND stage (stride=2), not just some final stage, so
there is no cheap padding trick. Monarch sidesteps this by factoring
in_features/out_features as a product of ARBITRARY integers you choose
(any valid factorization -- (3,5), (2,3,5), whatever divides evenly, not
restricted to powers of two at all), applying one batched-matmul step per
factor with an implicit permutation between steps. See `MonarchLinear`'s
own docstring for the exact algorithm and shape conventions.

`matmul()` (above) dispatches to Monarch automatically: pass a list/tuple
of Monarch weight tensors as `other` (instead of butterfly factors) and
it is detected via `_is_monarch_factors` and routed to `monarch_matmul`
-- one stateless entry point for both factorizations, no separate flag
needed. `monarch_matmul` is also the function `MonarchLinear.forward`
itself calls (with its own stored `self.weights`/`self.bias`), so the
choreography exists in exactly one place either way.

Unlike the butterfly kernel above, MonarchLinear is plain PyTorch
(`torch.bmm`/`permute`/`reshape`/`swapaxes_`, no Triton, no custom
CUDA/HIP) -- it already runs correctly on ROCm with zero porting work,
and gets a REAL backward pass for free via ordinary autograd (verified:
`torch.autograd.gradcheck` passes in double precision), unlike the
forward-only butterfly kernel. The tradeoff going the other way is FLOPs:
Monarch costs O(N * sum(factor_dims)) versus butterfly's O(N log N) for a
power-of-2 N -- still far sub-quadratic, but not as cheap as butterfly
when butterfly's stricter preconditions actually apply.

VALIDATION STATUS. Unlike everything above, this part needed no Triton/
GPU to validate: it is checked, on CPU, against TWO independent
references -- a hand-derived einsum formulation of the classic 2-factor
case (exact match, max error 0.0) and a probe-built dense-matrix
self-consistency check (feed the standard basis through the layer to
recover its actual (in_features, out_features) matrix, then confirm
`layer(x) == x @ that_matrix` for fresh inputs) across three
non-power-of-2 factorizations including a rectangular one -- plus a real
`gradcheck` in double precision, and `checkpoint=True` confirmed to
produce bit-identical forward output and gradients (weights, bias, and
input) against `checkpoint=False` on the same initial weights. See
tests/test_flash_mm_kernel.py for the exact cases.

PROVENANCE. Ported from a loose `source/monarch.py` file with no
attached LICENSE or attribution header. The implementation style
(`in_dims`/`out_dims` grid generalization, the `roll`-based permute
choreography) is consistent with a known community implementation of
Monarch matrices rather than something written from scratch for this
project, but exact upstream attribution could not be confirmed (web
search was unavailable when checked) -- flag this if it matters for this
project's own licensing posture before this ships anywhere external to
this dev checkout.
"""
from __future__ import annotations

import math
import os
from functools import partial
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

from . import kernel_select

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

# What `flash_butterfly_mm`'s `_try_triton` treats as "this kernel declined,
# use the torch mirror" rather than letting it escape. RuntimeError/TypeError
# alone is NOT enough: a kernel that fails to compile for this Triton version
# raises triton.compiler.errors.CompilationError, which descends from neither,
# so a compile failure used to propagate straight through the fallback that
# exists precisely to absorb it (this module shipped for a while with a kernel
# that could not compile at all -- see the XOR-vs-reshape note on the kernel
# below -- and `matmul` raised on every GPU input as a result).
try:
    from triton.compiler.errors import CompilationError as _TritonCompilationError

    _TRITON_FALLBACK_ERRORS = (RuntimeError, TypeError, _TritonCompilationError)
except ImportError:
    _TRITON_FALLBACK_ERRORS = (RuntimeError, TypeError)


def available() -> bool:
    """True if Triton is importable and a CUDA or ROCm device is visible.
    `flash_butterfly_mm_torch` works regardless -- this only gates the
    compiled-kernel path."""
    return _TRITON_AVAILABLE and torch.cuda.is_available()


def _num_stages(L: int) -> int:
    e = int(math.log2(L))
    if (1 << e) != L:
        raise ValueError(f"L={L} must be a power of 2")
    return e


# --------------------------------------------------------------------------
# Triton kernel (all stages fused into one launch). Only defined when Triton
# is importable, since `@triton.jit` decorates eagerly at import time.
#
# STAGE PAIRING IS DONE BY XOR, NOT BY RESHAPE. The obvious way to express a
# butterfly stage is to reshape the row to (num_blocks, 2, s) and peel the
# size-2 axis apart -- but that makes the stage stride `s` a *shape*, so it
# has to be a compile-time int inside the stage loop, and under Triton 3.7
# it cannot be: the induction variable of `tl.static_range` is a tensor, so
# every such shape argument is rejected (`constexpr[tensor]` where
# `constexpr[int]` is required), and annotating `s: tl.constexpr = 1 << i`
# to force the issue is rejected in turn on the loop's second unrolled trip
# ("constexpr cannot be reassigned"). An earlier revision of this kernel was
# written that way and could not be compiled at all.
#
# Within a 2s-block the partner of column c is exactly `c ^ s` (s is a power
# of two and the blocks are aligned), which makes the low and high half of
# each pair one expression instead of two:
#
#     k   = 1 if (c & s) else 0
#     y_c = w[i, c, k] * x_c + w[i, c^s, k] * x_{c^s}
#
# (low half, c = j0, k = 0: y_j0 = a0*v0 + b0*v1; high half, c = j1, k = 1:
# y_j1 = a1*v0 + b1*v1 -- the same coefficient convention as before, and the
# same arithmetic.) `s` is now only ever an integer *operand*, never a shape,
# so it can be an ordinary runtime value and the stage loop needs no
# unrolling. The partner shuffle is one `tl.gather` along the L axis, which
# stays in registers, so the one-HBM-read/one-HBM-write property this whole
# module is built around is unchanged.
# --------------------------------------------------------------------------
if _TRITON_AVAILABLE:

    @triton.jit
    def flash_butterfly_kernel(
        x_ptr,              # (BF, L) row-major input, one row per (batch,feature)
        w_ptr,               # (E, L, 2) row-major butterfly coefficients
        out_ptr,              # (BF, L) row-major output
        BF,
        L: tl.constexpr,
        E: tl.constexpr,             # number of stages = log2(L)
        REVERSE_STAGES: tl.constexpr,  # False -> stage order 0..E-1 (rightmost=True)
        BLOCK_BF: tl.constexpr,
    ):
        pid_row = tl.program_id(axis=0)
        row0 = pid_row * BLOCK_BF + tl.arange(0, BLOCK_BF)
        mask = row0 < BF

        col = tl.arange(0, L)
        # Single HBM read for the whole multi-stage computation.
        x = tl.load(x_ptr + row0[:, None] * L + col[None, :], mask=mask[:, None], other=0.0)

        for stage in range(E):
            i = (E - 1 - stage) if REVERSE_STAGES else stage
            s = 1 << i

            # Partner column within each 2s-block, and which of the two
            # coefficient slots this column reads: k=0 for the low half of a
            # pair, k=1 for the high half. Same w[i, j0, :] = [a0, a1] /
            # w[i, j1, :] = [b0, b1] layout as triton_kernel.py and
            # reference_impl.py.
            partner = col ^ s
            k = tl.where((col & s) != 0, 1, 0)

            w_base = w_ptr + i * (L * 2)
            w_self = tl.load(w_base + col * 2 + k)      # w[i, c,   k]
            w_part = tl.load(w_base + partner * 2 + k)  # w[i, c^s, k]

            xp = tl.gather(x, tl.broadcast_to(partner[None, :], (BLOCK_BF, L)), axis=1)
            x = w_self[None, :] * x + w_part[None, :] * xp

        tl.store(out_ptr + row0[:, None] * L + col[None, :], x, mask=mask[:, None])


    # ----------------------------------------------------------------------
    # MONARCH-GROUPED BUTTERFLY. Same linear map as the kernel above, but
    # instead of e = log2(L) elementwise stages it runs a handful of
    # BLOCK-DIAGONAL steps: butterfly stages [c, c+k) only ever pair indices
    # differing in bits [c, c+k), so composing k of them gives an operator
    # that is block-diagonal with R = 2^k dense blocks over
    #     idx = a*(R*B) + r*B + b        (r < R, B = 2^c)
    # i.e. exactly one Monarch factor -- one `tl.dot` per block instead of k
    # gather/multiply passes. That is the Monarch <-> butterfly relationship
    # (Dao/Chen et al., arXiv:2204.00595): Monarch is block-generalised
    # butterfly, so this is a regrouping of the SAME arithmetic, not an
    # approximation (measured max relative error <= 5.3e-7 vs
    # flash_butterfly_mm_torch).
    #
    # THE LAYOUT IS THE WHOLE POINT -- DO NOT "SIMPLIFY" IT BACK TO (BF, L).
    # In the natural (BF, L) row-major layout a group's R inputs sit B apart,
    # so every program does a scattered read; measured, that runs at 61 GB/s
    # and is SLOWER than both the plain butterfly kernel and a dense GEMM
    # (two separate attempts, one via torch.bmm and one via tl.dot, both
    # lost). In (L, BF) layout those same R inputs are R contiguous runs of
    # BLOCK_BF floats, the tile is only R*BLOCK_BF elements (no full-row
    # register pressure), and the same kernel runs at 701 GB/s -- 73% of a
    # 7900 XTX's peak, i.e. genuinely bandwidth-bound, which is the ceiling
    # for an op this FLOP-light. Everything below exists to work in that
    # layout and to get in and out of it cheaply.
    # ----------------------------------------------------------------------

    @triton.jit
    def _monarch_group_kernel(x_ptr, m_ptr, out_ptr, BF,
                              R: tl.constexpr, B: tl.constexpr, BLOCK_BF: tl.constexpr):
        """One Monarch factor, (L, BF) -> (L, BF)."""
        pid_ab = tl.program_id(0)      # = a*B + b, matching the factor layout
        pid_col = tl.program_id(1)
        a = pid_ab // B
        b = pid_ab % B

        cols = pid_col * BLOCK_BF + tl.arange(0, BLOCK_BF)
        cmask = cols < BF
        r = tl.arange(0, R)

        rowidx = a * (R * B) + r * B + b
        x = tl.load(x_ptr + rowidx[:, None] * BF + cols[None, :],
                    mask=cmask[None, :], other=0.0)          # (R, BLOCK_BF), coalesced
        s = tl.arange(0, R)
        m = tl.load(m_ptr + pid_ab * (R * R) + s[:, None] * R + r[None, :])
        y = tl.dot(m, x)
        outidx = a * (R * B) + s * B + b
        tl.store(out_ptr + outidx[:, None] * BF + cols[None, :], y, mask=cmask[None, :])

    @triton.jit
    def _monarch_group_in_kernel(x_ptr, m_ptr, out_ptr, BF, L,
                                 R: tl.constexpr, B: tl.constexpr, BLOCK_BF: tl.constexpr):
        """First Monarch factor, reading the caller's natural (BF, L) layout
        and writing (L, BF) -- the input transpose folded into a pass the
        kernel was making anyway. Cheap specifically because the first group
        has B=1, so its R inputs are contiguous within each row."""
        pid_ab = tl.program_id(0)
        pid_col = tl.program_id(1)
        a = pid_ab // B
        b = pid_ab % B
        cols = pid_col * BLOCK_BF + tl.arange(0, BLOCK_BF)   # rows of the (BF, L) input
        cmask = cols < BF
        r = tl.arange(0, R)
        rowidx = a * (R * B) + r * B + b
        xr = tl.load(x_ptr + cols[:, None] * L + rowidx[None, :],
                     mask=cmask[:, None], other=0.0)         # (BLOCK_BF, R)
        x = tl.trans(xr)
        s = tl.arange(0, R)
        m = tl.load(m_ptr + pid_ab * (R * R) + s[:, None] * R + r[None, :])
        y = tl.dot(m, x)
        outidx = a * (R * B) + s * B + b
        tl.store(out_ptr + outidx[:, None] * BF + cols[None, :], y, mask=cmask[None, :])

    @triton.jit
    def _transpose_kernel(src, dst, M, N, BM: tl.constexpr, BN: tl.constexpr):
        """Tiled transpose. Exists because torch's own `.t().contiguous()`
        manages ~10 GB/s on a (65536, 1024) fp32 tensor here (53.8 ms) --
        enough to erase the entire win on its own. This does it in 1.8 ms."""
        pm, pn = tl.program_id(0), tl.program_id(1)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        a = tl.load(src + rm[:, None] * N + rn[None, :],
                    mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)
        tl.store(dst + rn[:, None] * M + rm[None, :], tl.trans(a),
                 mask=(rn[:, None] < N) & (rm[None, :] < M))


def flash_butterfly_mm_triton(X: torch.Tensor, W_par, rightmost: bool = True,
                              BLOCK_BF: int | None = None) -> torch.Tensor:
    """One-kernel-launch fused butterfly matmul. Requires Triton + a CUDA/
    ROCm device (see `available()`); raises otherwise -- callers that need
    a CPU/no-Triton fallback should call `flash_butterfly_mm_torch`
    directly, or use `flash_butterfly_mm` which picks automatically.

    Validated on ROCm/Triton 3.7.1 against `flash_butterfly_mm_torch`, and
    measurably slower than a dense GEMM for every L benchmarked so far --
    see VALIDATION STATUS and PERFORMANCE in the module docstring before
    reaching for this directly instead of `flash_butterfly_mm`.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("triton is not installed; use flash_butterfly_mm_torch instead")
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA/ROCm device visible; use flash_butterfly_mm_torch instead")

    B, F, L = X.shape
    e = _num_stages(L)
    if isinstance(W_par, (list, tuple)):
        W_par = torch.stack(list(W_par), dim=0)
    if W_par.shape != (e, L, 2):
        raise ValueError(f"W_par has wrong shape: expected ({e}, {L}, 2), got {tuple(W_par.shape)}")

    BF = B * F
    x2d = X.reshape(BF, L).contiguous()
    w = W_par.contiguous()
    out = torch.empty_like(x2d)

    if BLOCK_BF is None:
        # Measured on gfx1100: a flat 64 is only right for small L. The whole
        # (BLOCK_BF, L) tile stays live across all e stages, so past L=128 the
        # tile stops fitting and the kernel spills -- 16 beat 64 at every L
        # from 128 up (at L=512, ~2x). Below that the larger tile amortises
        # the launch better.
        BLOCK_BF = 16 if L >= 128 else 64
    block_bf = min(BLOCK_BF, triton.next_power_of_2(BF))
    grid = (triton.cdiv(BF, block_bf),)
    flash_butterfly_kernel[grid](
        x2d, w, out,
        BF, L=L, E=e, REVERSE_STAGES=not rightmost,
        BLOCK_BF=block_bf,
    )
    return out.reshape(B, F, L)


# --------------------------------------------------------------------------
# Pure-PyTorch mirror of the exact same reshape/transpose/pack algorithm.
# This IS validated (bit-exact vs. reference_impl.butterfly_mm_ref, see
# unittests/test_flash_mm_kernel.py) and runs anywhere torch does -- CPU
# included, no Triton/GPU required. Use this when `available()` is False,
# or as the ground truth when testing `flash_butterfly_mm_triton`.
# --------------------------------------------------------------------------
def flash_butterfly_mm_torch(X: torch.Tensor, W_par, rightmost: bool = True) -> torch.Tensor:
    B, F, L = X.shape
    e = _num_stages(L)
    if isinstance(W_par, (list, tuple)):
        W_par = torch.stack(list(W_par), dim=0)
    if W_par.shape != (e, L, 2):
        raise ValueError(f"W_par has wrong shape: expected ({e}, {L}, 2), got {tuple(W_par.shape)}")

    x = X.reshape(B * F, L)
    stage_order = range(e) if rightmost else reversed(range(e))
    for i in stage_order:
        x = _flash_stage_torch(x, W_par[i], 1 << i)
    return x.reshape(B, F, L)


def _flash_stage_torch(x: torch.Tensor, w_stage: torch.Tensor, s: int) -> torch.Tensor:
    BF, L = x.shape
    num_blocks = L // (2 * s)
    x4 = x.reshape(BF, num_blocks, 2, s)
    v0 = x4[:, :, 0, :]
    v1 = x4[:, :, 1, :]

    blk = torch.arange(num_blocks, device=x.device)
    within = torch.arange(s, device=x.device)
    j0 = blk[:, None] * (2 * s) + within[None, :]  # (num_blocks, s)
    j1 = j0 + s

    a0 = w_stage[j0, 0]
    a1 = w_stage[j0, 1]
    b0 = w_stage[j1, 0]
    b1 = w_stage[j1, 1]

    y0 = a0 * v0 + b0 * v1
    y1 = a1 * v0 + b1 * v1

    y4 = torch.stack([y0, y1], dim=2)  # (BF, num_blocks, 2, s)
    return y4.reshape(BF, L)


# ---------------------------------------------------------------------------
# Monarch-grouped path: factor construction (one-time weight prep) + forward.
# ---------------------------------------------------------------------------
# Below this L the grouping cannot even be formed usefully: `tl.dot` wants
# blocks of at least 16, so two groups need e = log2(L) >= 8. That lines up
# with the measurements -- at L <= 128 the grouped path never won at any batch
# size, so declining outright costs nothing and saves a kernel_select contest.
# Above it the contest decides, because the win is real but not universal
# (measured vs a dense GEMM at L=512: 1.32x at BF=8192 but 0.48x at BF=2048).
_MONARCH_MIN_L = int(os.environ.get("AMD_TUNED_TORCH_FLASH_MM_MONARCH_MIN_L", "256"))

# Building the factors costs ~e dense (L, L) matmuls, which is far more than
# one forward -- so it is cached, and this path is only worth taking when the
# butterfly parameters are FIXED (inference, or a frozen layer). The cache key
# includes `_version` so a mutated weight tensor rebuilds instead of silently
# serving stale factors; that also means a training loop updating W every step
# rebuilds every step and should not use this path.
_MONARCH_FACTOR_CACHE: "dict" = {}
_MONARCH_FACTOR_CACHE_MAX = 32


def _split_stages(e: int) -> "list[int]":
    """Partition e butterfly stages into Monarch groups. Two groups for the
    sizes that matter (e<=12), each at least 4 stages so the blocks are at
    least 16x16 (tl.dot's floor); extras go to the LATER group because the
    measured [4,5] beat [5,4] at L=512."""
    n = max(2, math.ceil(e / 6))
    while n > 1 and e // n < 4:
        n -= 1
    if n < 2:
        return [e]
    base, rem = divmod(e, n)
    return [base + (1 if i >= n - rem else 0) for i in range(n)]


def _butterfly_stage_matrix(w_i: torch.Tensor, L: int, s: int) -> torch.Tensor:
    c = torch.arange(L, device=w_i.device)
    p = c ^ s
    slot = ((c & s) != 0).long()
    S = torch.zeros(L, L, dtype=w_i.dtype, device=w_i.device)
    S[c, c] = w_i[c, slot]
    S[c, p] = w_i[p, slot]
    return S


def _monarch_factors(W_par: torch.Tensor, rightmost: bool):
    """[(blocks (A*B, R, R), A, R, B)] in application order, cached."""
    e, L, _ = W_par.shape
    key = (W_par.data_ptr(), W_par._version, tuple(W_par.shape),
           W_par.dtype, str(W_par.device), bool(rightmost))
    hit = _MONARCH_FACTOR_CACHE.get(key)
    if hit is not None:
        return hit

    groups = _split_stages(e)
    order = list(range(e)) if rightmost else list(range(e))[::-1]
    facs, pos = [], 0
    for k in groups:
        idxs = order[pos:pos + k]
        G = torch.eye(L, dtype=W_par.dtype, device=W_par.device)
        for i in idxs:
            G = _butterfly_stage_matrix(W_par[i], L, 1 << i) @ G
        B_, R = 1 << min(idxs), 1 << k
        A = L // (R * B_)
        idx = (torch.arange(A, device=W_par.device).view(A, 1, 1) * (R * B_)
               + torch.arange(R, device=W_par.device).view(1, R, 1) * B_
               + torch.arange(B_, device=W_par.device).view(1, 1, B_))
        rows = idx.permute(0, 2, 1).reshape(A * B_, R)
        facs.append((G[rows.unsqueeze(2), rows.unsqueeze(1)].contiguous(), A, R, B_))
        pos += k

    if len(_MONARCH_FACTOR_CACHE) >= _MONARCH_FACTOR_CACHE_MAX:
        _MONARCH_FACTOR_CACHE.pop(next(iter(_MONARCH_FACTOR_CACHE)))
    _MONARCH_FACTOR_CACHE[key] = facs
    return facs


def _fast_transpose(x: torch.Tensor, BM: int = 64, BN: int = 128) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty((N, M), dtype=x.dtype, device=x.device)
    _transpose_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](x, out, M, N, BM=BM, BN=BN)
    return out


def monarch_eligible(X: torch.Tensor, L: int) -> bool:
    """Cheap pre-filter for the grouped path -- see `_MONARCH_MIN_L`."""
    return (_TRITON_AVAILABLE and X.is_cuda and L >= _MONARCH_MIN_L
            and _num_stages(L) >= 8 and X.dtype in (torch.float32, torch.float64))


def flash_butterfly_mm_monarch(X: torch.Tensor, W_par, rightmost: bool = True,
                               BLOCK_BF: int = 64) -> torch.Tensor:
    """Butterfly via Monarch-grouped block factors, working in (L, BF) layout.

    Same result as `flash_butterfly_mm_torch` (measured <= 5.3e-7 max relative
    error), computed as a few block-diagonal `tl.dot`s instead of e elementwise
    stages. Measured against a dense GEMM at BF=65536 on gfx1100: 1.93x at
    L=1024, 1.15x at L=512, and a loss at L=256 -- see PERFORMANCE in the
    module docstring. Forward-only and non-differentiable, like every other
    Triton path here."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("triton is not installed; use flash_butterfly_mm_torch instead")
    B, F, L = X.shape
    e = _num_stages(L)
    if isinstance(W_par, (list, tuple)):
        W_par = torch.stack(list(W_par), dim=0)
    if W_par.shape != (e, L, 2):
        raise ValueError(f"W_par has wrong shape: expected ({e}, {L}, 2), got {tuple(W_par.shape)}")

    BF = B * F
    x2d = X.reshape(BF, L).contiguous()
    facs = _monarch_factors(W_par.contiguous(), rightmost)

    # First factor reads (BF, L) and writes (L, BF), absorbing the input
    # transpose; the rest stay in (L, BF).
    blocks, A, R, B_ = facs[0]
    x = torch.empty((L, BF), dtype=x2d.dtype, device=x2d.device)
    _monarch_group_in_kernel[(A * B_, triton.cdiv(BF, BLOCK_BF))](
        x2d, blocks, x, BF, L, R=R, B=B_, BLOCK_BF=BLOCK_BF)
    for blocks, A, R, B_ in facs[1:]:
        nxt = torch.empty_like(x)
        _monarch_group_kernel[(A * B_, triton.cdiv(BF, BLOCK_BF))](
            x, blocks, nxt, BF, R=R, B=B_, BLOCK_BF=BLOCK_BF)
        x = nxt
    return _fast_transpose(x).reshape(B, F, L)


def _flash_triton_eligible(X: torch.Tensor) -> bool:
    """True if the compiled kernel could apply at all: Triton importable, a
    CUDA/ROCm device visible, and X actually resident on it. A separate,
    trivially mockable function (rather than an inline `available() and
    X.is_cuda` check in `flash_butterfly_mm`) specifically so
    unittests/test_flash_mm_kernel.py can monkeypatch this ONE thing to
    exercise the kernel_select contest's SELECTION policy on plain CPU
    tensors, without a real GPU to actually run the kernel on -- the same
    "test the dispatch logic anywhere, independent of real hardware"
    convention this package's own tests/conftest.py establishes for every
    other contest (force_eligible, the `aiter`/`te` fixtures, etc.)."""
    return available() and X.is_cuda


def flash_butterfly_mm(X: torch.Tensor, W_par, rightmost: bool = True) -> torch.Tensor:
    """Dispatches to whichever of the compiled Triton kernel or the
    validated pure-PyTorch mirror actually measures faster for this
    (dtype, shape) combination -- via kernel_select's per-shape
    measure-once-cache-the-winner contest (amd_tuned_torch/kernel_select.py,
    the same mechanism linear/matmul/bmm/conv2d/conv3d/group_norm/attention
    already use), NOT an unconditional preference for the compiled kernel
    just because it's compiled.

    THIS MATTERS SPECIFICALLY HERE. kernel_select's whole reason to exist
    (see its own module docstring) is that "our kernel must be faster than
    stock" is a belief, not a fact, until measured on the actual card --
    and for this kernel the measurement came back saying the compiled path
    is NOT the fast one: it loses to a dense GEMM at every L benchmarked so
    far (see PERFORMANCE in this module's docstring). Blindly preferring it
    the moment `available()` is True would be exactly the failure mode
    kernel_select exists to prevent -- note the contest here is
    triton-vs-torch-mirror, so a shape where a dense GEMM would beat both
    is still a win for whichever of the two the contest picks, not a reason
    to believe this module is the right tool for that shape at all.
    kernel_select's own numerical verification (torch.allclose against the
    torch-fallback candidate, which is this contest's reference -- see
    kernel_select.py's CORRECTNESS VERIFICATION section) is also load-bearing
    here in a way it usually isn't: a Triton kernel that measures fast but
    computes something wrong for some shape gets caught and permanently
    excluded for that shape (with a warning) rather than silently returned,
    which is the realistic failure mode for a kernel that's never touched
    real hardware.

    Falls back to the plain preference (Triton if available()+CUDA, else
    torch) when kernel_select is disabled (AMD_TUNED_TORCH_MEASURE_KERNELS=0)
    or Triton/CUDA aren't available at all -- same escape hatch every other
    contest in this package offers, and the only behavior possible without
    a device to measure on.
    """
    if not _flash_triton_eligible(X):
        return flash_butterfly_mm_torch(X, W_par, rightmost=rightmost)

    if isinstance(W_par, (list, tuple)):
        W_par = torch.stack(list(W_par), dim=0)

    # Every Triton path here is a raw kernel with no autograd.Function, so its
    # output carries no grad_fn. `flash_butterfly_mm_torch` is built from
    # differentiable primitives and DOES backprop, so silently handing back a
    # graph-detached tensor when a gradient is wanted would break backward
    # rather than merely being forward-only. The module is declared
    # forward-only (see SCOPE), but that is a statement about the kernels, not
    # a licence to detach a caller's graph without saying so.
    if torch.is_grad_enabled() and (X.requires_grad or W_par.requires_grad):
        return flash_butterfly_mm_torch(X, W_par, rightmost=rightmost)

    def _fallback():
        return flash_butterfly_mm_torch(X, W_par, rightmost=rightmost)

    def _try_triton():
        try:
            return flash_butterfly_mm_triton(X, W_par, rightmost=rightmost)
        except _TRITON_FALLBACK_ERRORS:
            return None

    def _try_monarch():
        if not monarch_eligible(X, X.shape[-1]):
            return None
        try:
            return flash_butterfly_mm_monarch(X, W_par, rightmost=rightmost)
        except _TRITON_FALLBACK_ERRORS:
            return None

    if not kernel_select.enabled():
        out = _try_triton()
        return _fallback() if out is None else out

    # W_par's shape (not just X's) belongs in the key: the same X shape
    # with a different L/stage-count factorization is a different problem,
    # same reasoning _patched_sdpa_flash_attn_rocwmma's contest_key uses
    # for query/key/value shapes.
    key = (X.dtype, tuple(X.shape), tuple(W_par.shape), bool(rightmost))
    won = kernel_select.cached_key("flash_butterfly_mm", key)
    if won == "torch":
        return _fallback()
    if won is None:
        candidates = [("triton", _try_triton), ("torch", _fallback)]
        if monarch_eligible(X, X.shape[-1]):
            # Best-guess-first: at every shape where it is eligible AND won a
            # measurement, it won by a wide margin (1.9x over dense at
            # L=1024); where it loses it declines cheaply.
            candidates.insert(0, ("monarch", _try_monarch))
        out = kernel_select.pick_key("flash_butterfly_mm", key, candidates)
        if out is not None:
            return out
    elif won == "monarch":
        out = _try_monarch()
        if out is not None:
            return out
    elif won == "triton":
        out = _try_triton()
        if out is not None:
            return out
    return _fallback()


# --------------------------------------------------------------------------
# torch.matmul-style entry point.
#
# flash_butterfly_mm/_torch/_triton above all require X pre-shaped exactly
# (B, F, L) -- the caller has to know to split its own batch dims into that
# specific (batch, feature, length) triple. `matmul` removes that
# requirement: like torch.matmul, it accepts X of ANY shape (*, L) --
# a bare 1D length-L vector, a 2D (M, L) matrix, or an arbitrarily-batched
# (..., L) tensor -- by flattening every leading dim into one batch axis
# before calling the kernel and restoring the original leading shape
# afterwards. This is the same normalize-then-restore-shape preprocessing
# amd_tuned_torch.__init__._patched_matmul already does for torch.matmul's
# own aiter/hipBLASLt dispatch (reshape to a shape the underlying kernel
# wants, run it, reshape the result back) -- applied here to this kernel's
# own (B, F, L) requirement instead of a GEMM's (batch, M, K) one.
# --------------------------------------------------------------------------
def is_eligible(X: torch.Tensor, W_par) -> bool:
    """Non-raising precondition check for `matmul`: True if `W_par` is
    EITHER valid butterfly factors for X (X has at least one dimension,
    its last dimension L is a power of 2, and W_par -- a
    list/tuple/ParameterList of (L, 2) tensors, or an already-stacked
    (E, L, 2) tensor -- has exactly the E = log2(L) stages L requires) OR
    valid Monarch factors for X (see `_is_monarch_factors` -- a
    list/tuple of >=2 weight tensors whose in_dims product matches X's
    last dimension; no power-of-2 restriction at all in this case).

    Use this to decide whether to call `matmul` at all -- e.g. as a
    kernel_select-style candidate elsewhere that should decline rather than
    raise for shapes this module doesn't cover -- without needing a
    try/except around the raising checks `matmul`/`flash_butterfly_mm_torch`
    perform themselves."""
    if not isinstance(X, torch.Tensor) or X.dim() < 1:
        return False
    L = X.shape[-1]

    if _is_monarch_factors(W_par):
        # Not just "do the in_dims multiply to L": each factor's BATCH dim is
        # fixed by the running intermediate size (see MonarchLinear's
        # construction -- weight i is (current_numel // in_dim, in_dim,
        # out_dim), and current_numel evolves as the factors are applied).
        # Checking only the product accepts factor lists that then blow up
        # inside monarch_matmul's own torch.bmm with a shape RuntimeError,
        # which matters now that torch.matmul routes list operands here:
        # declining cleanly gives the caller back torch's own TypeError
        # instead of an error from four frames down.
        current = 1
        for w in W_par:
            current *= w.shape[1]
        if current != L:
            return False
        current = L
        for w in W_par:
            batch, in_dim, out_dim = w.shape
            if in_dim == 0 or current % in_dim != 0 or batch != current // in_dim:
                return False
            current = current // in_dim * out_dim
        return True

    try:
        e = _num_stages(L)
    except ValueError:
        return False
    if not isinstance(W_par, torch.Tensor):
        try:
            W_par = torch.stack(list(W_par), dim=0)
        except (TypeError, RuntimeError):
            return False
    return tuple(W_par.shape) == (e, L, 2)


def matmul(input: torch.Tensor, other, *, rightmost: bool = True,
           out: torch.Tensor | None = None) -> torch.Tensor:
    """Y = input @ W, W the matrix implied by `other`'s structured
    factors -- called the same way as `torch.matmul(input, other)`, for
    the two restricted classes of "matrix" this module supports.

    `other` may be EITHER of two, structurally different, factor
    conventions -- detected automatically via `_is_monarch_factors`, no
    separate flag needed:

    1. BUTTERFLY factors (a list/ParameterList of (L, 2) stage tensors, or
       a pre-stacked (E, L, 2) tensor) -- requires input's last dim L to be
       a power of 2 (see is_eligible/_num_stages). PREPROCESSING: `input`
       may have any shape `(*, L)`, including a bare 1D vector `(L,)` --
       every leading dim is flattened into a single batch axis, passed
       through `flash_butterfly_mm`, then reshaped back. On CUDA, that
       call runs the compiled-vs-torch-fallback kernel_select contest
       described in the module docstring -- this function does not itself
       prefer the compiled kernel, it inherits whichever measures faster.
       `rightmost` controls butterfly stage order (ignored for Monarch
       factors below -- Monarch has no analogous per-factor order choice).

    2. MONARCH factors (a list/tuple of >=2 weight tensors, each shape
       (batch_i, in_dim_i, out_dim_i) -- see MonarchLinear/monarch_matmul)
       -- input's last dim must equal the product of each factor's
       in_dim, with NO power-of-2 restriction at all; this is the path
       that handles the sizes butterfly factors structurally can't (see
       the module docstring's MONARCH MATRICES section for exactly why
       padding doesn't substitute for this). Dispatches to
       `monarch_matmul`, which handles `input`'s arbitrary leading shape
       itself -- no separate reshape wrapper needed here the way the
       butterfly path requires.

    THE TENSOR(S) THIS SUPPORTS -- read before reaching for this as a
    general `torch.matmul` replacement: `other` must ALREADY be given as
    butterfly OR Monarch factors. There is no preprocessing step here that
    accepts an arbitrary dense (L, L) weight matrix and derives EITHER
    factorization for it automatically -- exactly recovering (or even
    well-approximating) an arbitrary matrix's butterfly factorization is a
    real, nontrivial numerical-algorithm problem in its own right (see
    Li/Yang et al.'s "butterfly factorization" literature); Monarch
    factors, similarly, must already be given -- this module has no
    from-scratch Monarch-factorization-of-an-arbitrary-matrix routine
    either. Call `is_eligible(input, other)` first if you need a
    non-raising check instead of a caught ValueError.

    This is still forward-only for the BUTTERFLY path, like every function
    in this module before the Monarch section -- see the module
    docstring's SCOPE section. The MONARCH path, by contrast, is fully
    differentiable (ordinary autograd through `torch.bmm`/`permute`/
    `reshape`), same as MonarchLinear itself.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"input must be a torch.Tensor, got {type(input)}")
    if input.dim() < 1:
        raise ValueError("input must have at least 1 dimension (the length-L axis)")

    if _is_monarch_factors(other):
        result = monarch_matmul(input, other)
        if out is not None:
            out.copy_(result)
            return out
        return result

    if not isinstance(other, torch.Tensor):
        other = torch.stack(list(other), dim=0)

    L = input.shape[-1]
    e = _num_stages(L)
    if other.shape != (e, L, 2):
        raise ValueError(
            f"other has wrong shape for input's last dim L={L}: expected "
            f"({e}, {L}, 2) butterfly factors, got {tuple(other.shape)} -- "
            f"matmul() does not accept an arbitrary dense (L, L) weight "
            f"matrix, see this function's docstring"
        )

    leading_shape = input.shape[:-1]
    x3 = input.reshape(1, -1, L)
    y3 = flash_butterfly_mm(x3, other, rightmost=rightmost)
    result = y3.reshape(*leading_shape, L)

    if out is not None:
        out.copy_(result)
        return out
    return result


# --------------------------------------------------------------------------
# Monarch matrices. See the module docstring's MONARCH MATRICES section for
# what this is, how it relates to the butterfly kernel above, and its
# validation/provenance status.
# --------------------------------------------------------------------------
def _is_monarch_factors(other) -> bool:
    """True if `other` looks like a list/tuple of Monarch weight tensors
    (each 3D: (batch_i, in_dim_i, out_dim_i), see monarch_matmul) rather
    than butterfly factors (a single (E, L, 2) tensor, or a list of 2D
    (L, 2) stage tensors -- butterfly stage tensors are never 3D anywhere
    in this module's own convention, so this check is unambiguous)."""
    if isinstance(other, torch.Tensor):
        return False  # a bare tensor is always butterfly's stacked (E, L, 2) form
    if not isinstance(other, (list, tuple)) or len(other) < 2:
        return False
    return all(isinstance(w, torch.Tensor) and w.dim() == 3 for w in other)


def monarch_matmul(input: torch.Tensor, weights: Sequence[torch.Tensor],
                    bias: torch.Tensor | None = None) -> torch.Tensor:
    """Functional, stateless form of MonarchLinear.forward: applies a
    Monarch factorization given directly as a list of weight tensors
    (shape (batch_i, in_dim_i, out_dim_i) each, one per factor) instead of
    an nn.Module's own stored parameters. `in_dims`/`out_dims` are read
    directly off each weight tensor's own shape (in_dims[i] =
    weights[i].shape[1], out_dims[i] = weights[i].shape[2]) rather than
    passed separately -- the same convention `_is_monarch_factors`/
    `matmul` rely on to detect and dispatch to this function.

    MonarchLinear itself is defined in terms of this function (not the
    other way around) so the actual bmm/permute choreography exists in
    exactly one place; see MonarchLinear's own docstring for the full
    algorithm description, shape convention, and worked examples of
    `in_dims`/`out_dims`. No `checkpoint`/gradient-checkpointing option
    here -- that is specifically an nn.Module-level concern (it needs
    somewhere to hold the `self.checkpoint` flag across calls), and
    checkpointing this function yourself is one `torch.utils.checkpoint.
    checkpoint(monarch_matmul, input, weights, bias)` call away for a
    caller who wants it without going through MonarchLinear."""
    in_dims = tuple(w.shape[1] for w in weights)
    input_shape = input.shape
    expected = math.prod(in_dims)
    if input_shape[-1] != expected:
        raise ValueError(
            f"monarch_matmul: input's last dim ({input_shape[-1]}) must equal "
            f"the product of the Monarch factors' in_dims {in_dims} (={expected})"
        )

    tensor = input.reshape(-1, *in_dims)
    # shape: [flat_batch_size, in_dim[0], ..., in_dim[N-1]]
    tensor = tensor.permute(*np.roll(range(len(in_dims) + 1), -2))
    # new shape: [in_dim[1], ..., in_dim[N-1], flat_batch_size, in_dim[0]]

    for i, w in enumerate(weights):
        # loop maintains tensor in shape: [*all_dims_except_i, batch, dim[i]]
        tensor = torch.bmm(tensor.flatten(0, -3), w).view(*tensor.shape[:-1], -1)
        # bmm output: [*other_dims, batch, out_dim[i]]
        tensor = tensor.swapaxes_(-1, i)
        # safe in-place: `tensor` is always bmm's own fresh output by this
        # point, never a view aliasing `input` or any caller-held tensor.

    # after loop: [out_dim[0], ..., out_dim[N-1], batch]
    tensor = tensor.flatten(0, -2).swapaxes_(0, 1)
    tensor = tensor.reshape(*input_shape[:-1], -1)
    if bias is not None:
        tensor = tensor + bias
    return tensor


class MonarchLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 in_dims: Sequence[int], out_dims: Sequence[int],
                 bias: bool = True, checkpoint: bool = False,
                 ):
        """
        Monarch linear layer, a generalization of https://arxiv.org/abs/2204.00595
        Ths implementation interprets Monarch as a product over an M by M grid (in_features=M ^ 2).
        The first product applies over all rows of the grid, the second runs over columns.
        In general, the grid may have uneven size or more than 2 dimensions.
        In the 2d case, the two products use [M x M x M] weight tensors. In the general case,
        it uses grid_dim weight tensors of shape [grid_numel / in_dims[i], in_dims[i], out_dims[i]].
        :param in_features: input dimension, same as in nn.Linear
        :param out_features: output dimension, same as in nn.Linear
        :param in_dims: a tuple of numbers that multiply to in_features, see example below
        :param out_dims: a tuple of numbers that multiply to out_features, see example below
        :param bias: whether or not to use a bias term, same as in nn.Linear
        :param checkpoint: if True, apply gradient checkpointing over this entire layer.
           This adds ~30% compute overhead for forward+backward, but reduces the memory overhead;
           otherwise, monarch must to store ndim - 1 additional tensors for intermediate activations.
        :example:
        >>> # classic monarch:
        >>> MonarchLinear(in_features=1024, in_dims=(32, 32), out_features=1024, out_dims=(32, 32))
        >>> # generalization to rectangular matrices
        >>> MonarchLinear(in_features=1024, in_dims=(32, 32), out_features=4096, out_dims=(64, 64))
        >>> MonarchLinear(in_features=1024, in_dims=(32, 32), out_features=1536, out_dims=(32, 48))
        >>> # generalization to higher dimension
        >>> MonarchLinear(in_features=4096, in_dims=(16, 16, 16), out_features=4096, out_dims=(16, 16, 16))
        >>> MonarchLinear(in_features=4096, in_dims=(16, 16, 16), out_features=1536, out_dims=(8, 12, 16))
        """
        super().__init__()
        assert len(in_dims) == len(out_dims) and len(in_dims) > 1
        assert np.prod(in_dims) == in_features
        assert np.prod(out_dims) == out_features
        self.in_features, self.out_features = in_features, out_features
        self.in_dims, self.out_dims = in_dims, out_dims
        self.checkpoint = checkpoint

        # construct weight tensors by keeping track of intermediate tensor dimension at each step
        self.weights = nn.ParameterList()
        current_numel = np.prod(in_dims)
        assert current_numel == in_features
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            self.weights.append(nn.Parameter(torch.empty(current_numel // in_dim, in_dim, out_dim)))
            current_numel = current_numel // in_dim * out_dim
        assert current_numel == out_features
        self.register_parameter('bias', nn.Parameter(torch.empty(out_features)) if bias else None)
        self.reset_parameters()

    def reset_parameters(self, gain: float = 1.0):
        # initialize, re-scale to account for the number of multiplied tensors
        init_std = (gain / np.sqrt(self.in_features)) ** (1 / len(self.in_dims))
        for weight in self.weights:
            nn.init.normal_(weight, std=init_std)
        if self.bias is not None:
            bound = 1 / np.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor, _inside_checkpoint: bool = False):
        if self.checkpoint and not _inside_checkpoint and torch.is_grad_enabled():
            return _torch_checkpoint(partial(self.forward, _inside_checkpoint=True),
                              input if input.requires_grad else input.detach().requires_grad_(True),
                              preserve_rng_state=False)
        # The actual bmm/permute choreography lives in monarch_matmul (one
        # copy, shared with matmul()'s dispatch for Monarch-shaped `other`
        # below) -- this just supplies this instance's own parameters.
        return monarch_matmul(input, list(self.weights), self.bias)
