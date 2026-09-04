"""Split-K GEMM for decode-shaped (small-M, "tall-skinny") F.linear calls
-- a from-scratch kernel, not a port, inspired by the same problem
NVIDIA's CUTLASS solves with its Stream-K algorithm (see
amd_tuned_torch/_vendor/ if you're looking for a NOTICE.md here: there
isn't one, because no CUTLASS code was copied -- only the underlying
idea. CUTLASS itself is CUDA-only C++ templates with no ROCm path at
all, see the investigation that led to this file for the full
comparison).

THE PROBLEM (already measured, not speculative -- see
kernel_select.py's own module docstring):

    1x4096x4096 (decode)       hipBLASLt 0.60x   CK GEMM 0.90x

Both existing GEMM tiers LOSE to stock at single-token decode. A plain
tiled GEMM assigns each thread block one (M_tile, N_tile) output tile;
at M=1 there are only ceil(N/N_tile) tiles total, usually far fewer than
this card's CU count, so most CUs sit idle for the whole call no matter
how fast each tile computes. Splitting the K-reduction dimension across
MORE thread blocks than output tiles exist -- so several blocks
cooperate on one output tile, each reducing a different slice of K --
fills those otherwise-idle CUs. This is the core idea CUTLASS's Stream-K
generalizes (with dynamic, fractional-tile work assignment for uneven
loads); what's implemented here is the simpler, static "Split-K" version
of that idea: K is split into a fixed number of equal-ish slices chosen
once per call from the shape, not dynamically rebalanced across running
blocks. Simpler to reason about and verify without hardware access to
tune against, at the cost of being less adaptive than true Stream-K to
shapes where the K-split itself is uneven.

NO ATOMICS: each of the SPLIT_K slices writes its own partial sum into a
distinct (split, M, N) scratch buffer position -- never contended, no
tl.atomic_add -- and the SPLIT_K partials are combined with a plain
`.sum(dim=0)` afterward. This trades a small amount of extra memory
(SPLIT_K * M * N floats, negligible at decode-shaped M) for not needing
to reason about atomic-add dtype/backend support or accumulation-order
nondeterminism at all.

THIS IS A CONTEST CANDIDATE, not a forced replacement. It's wired into
_patched_linear's kernel_select contest (amd_tuned_torch/__init__.py)
alongside hipblaslt/ck_gemm/aiter/stock -- kernel_select measures it
against stock on first use for a given shape and PERMANENTLY EXCLUDES it
for that shape if its output disagrees beyond tolerance (see
kernel_select.py's CORRECTNESS VERIFICATION). That safety net is the
reason this is a reasonable thing to ship despite being new, untested-on-
hardware code in the primary F.linear path every model calls: a
correctness bug here degrades to "loses the contest, stock wins instead"
automatically, not silent wrong output.

Eligible ONLY for small M (decode-shaped calls) -- see _MAX_M. For larger
M, ordinary tiling already produces enough output tiles to occupy the
GPU, and splitting K on top adds scratch-buffer memory and a reduction
pass for no benefit, so this correctly declines (returns None) outside
its target shape range rather than attempting to compete everywhere.

available() gates on `triton` importability, same reasoning as every
other Triton module in this package.

UNVALIDATED -- more so than anything else in this package: every other
kernel here is either a port of tested upstream code or (fused_norm_ops/
rope_ops's backward passes) a well-established closed-form formula
applied to new code. This is original kernel design with no reference
implementation to check against beyond the numerical tolerance check
kernel_select already performs automatically, and no hardware access to
tune SPLIT_K/BLOCK_N/BLOCK_K against. Before trusting the speedup claim
specifically (correctness is self-checking via kernel_select, but speed
is not): benchmark against stock/hipBLASLt/CK for your actual decode
shapes -- there is a real chance the SPLIT_K heuristic below is simply
wrong for this card and this candidate always loses its own contest,
which is a safe (if disappointing) failure mode, not a dangerous one.
"""
from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Decode-shaped calls only -- see module docstring for why larger M
# doesn't benefit and shouldn't pay this kernel's overhead.
_MAX_M = 16
# Below this, K is too short for splitting it further to make sense --
# each split needs at least a full BLOCK_K's worth of real work.
_MIN_K = 256

_BLOCK_N = 128
_BLOCK_K = 128
_MAX_SPLIT_K = 32
# Aim for this many times the CU count in total thread blocks
# (num_n_blocks * split_k * M) -- enough oversubscription to hide memory
# latency without so many blocks that per-block overhead dominates.
_TARGET_OCCUPANCY_MULTIPLIER = 4
_DEFAULT_CU_COUNT = 96  # RX 7900 XTX; used only if device-property query fails


def available() -> bool:
    return _TRITON_AVAILABLE


def _cu_count(device) -> int:
    try:
        return torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:
        return _DEFAULT_CU_COUNT


def _ceildiv(a: int, b: int) -> int:
    return -(-a // b)


def _choose_split_k(M: int, N: int, K: int, device) -> int:
    """Pure host-side integer math -- deliberately not using triton.cdiv
    here (unlike the kernel's own internal k_per_split computation, which
    must run on-device) so this helper is testable without triton being
    importable at all."""
    num_n_blocks = _ceildiv(N, _BLOCK_N)
    target_blocks = _TARGET_OCCUPANCY_MULTIPLIER * _cu_count(device)
    denom = max(1, num_n_blocks * M)
    split_k = max(1, target_blocks // denom)
    split_k = min(split_k, max(1, K // _BLOCK_K))
    return min(split_k, _MAX_SPLIT_K)


if _TRITON_AVAILABLE:

    @triton.jit
    def _splitk_gemm_kernel(
        x_ptr, w_ptr, out_ptr,
        N, K,
        x_stride_m, x_stride_k,
        w_stride_n, w_stride_k,
        out_stride_split, out_stride_m, out_stride_n,
        SPLIT_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # Grid: (n_blocks, SPLIT_K, M). Each program reduces ONE slice of
        # K for ONE block of N columns of ONE row of M, writing its own
        # dedicated (split, m, n_block) scratch slot -- no two programs
        # ever write the same output element, so no atomics are needed;
        # out_ptr[k_split] holds that split's partial sum, summed across
        # the split axis by the caller afterward.
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        pid_m = tl.program_id(2)

        n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        k_per_split = tl.cdiv(K, SPLIT_K)
        k_start = pid_k * k_per_split
        k_end = tl.minimum(k_start + k_per_split, K)

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in range(k_start, k_end, BLOCK_K):
            k_offsets = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < k_end
            x_vals = tl.load(
                x_ptr + pid_m * x_stride_m + k_offsets * x_stride_k, mask=k_mask, other=0.0
            ).to(tl.float32)
            w_vals = tl.load(
                w_ptr + n_offsets[:, None] * w_stride_n + k_offsets[None, :] * w_stride_k,
                mask=n_mask[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float32)
            acc += tl.sum(x_vals[None, :] * w_vals, axis=1)

        tl.store(
            out_ptr + pid_k * out_stride_split + pid_m * out_stride_m + n_offsets * out_stride_n,
            acc, mask=n_mask,
        )


def linear(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
    """Y = X @ W^T (+ bias), split-K accumulated -- None if unavailable,
    ineligible (see _MAX_M/_MIN_K), or the launch itself fails. Weight is
    [N, K], F.linear's own layout, passed through untransposed."""
    if not available():
        return None
    if not isinstance(input, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return None
    if not input.is_cuda or not weight.is_cuda:
        return None
    if input.dtype not in _DTYPES or weight.dtype != input.dtype:
        return None
    if bias is not None and (not bias.is_cuda or bias.dtype != input.dtype):
        return None
    if input.dim() < 1 or weight.dim() != 2:
        return None

    K = input.shape[-1]
    if weight.shape[1] != K:
        return None
    orig_shape = input.shape
    x2d = input.reshape(-1, K)
    M = x2d.shape[0]
    N = weight.shape[0]
    if M > _MAX_M or K < _MIN_K:
        return None

    try:
        x2d = x2d.contiguous()
        w = weight.contiguous()
        split_k = _choose_split_k(M, N, K, x2d.device)
        num_n_blocks = _ceildiv(N, _BLOCK_N)

        partial = torch.empty((split_k, M, N), dtype=torch.float32, device=x2d.device)
        _splitk_gemm_kernel[(num_n_blocks, split_k, M)](
            x2d, w, partial,
            N, K,
            x2d.stride(0), x2d.stride(1),
            w.stride(0), w.stride(1),
            partial.stride(0), partial.stride(1), partial.stride(2),
            SPLIT_K=split_k, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
        )
        out = partial.sum(dim=0)
        if bias is not None:
            out = out + bias.float()
        out = out.to(input.dtype)
        return out.view(*orig_shape[:-1], N)
    except (RuntimeError, TypeError):
        return None
