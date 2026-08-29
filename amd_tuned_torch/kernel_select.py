"""Per-shape "whichever is actually fastest" kernel selection, with stock
ROCm included as an ordinary candidate.

WHY THIS EXISTS. Every other tier in this project is ordered by a prior
belief that its own kernels beat stock, which was true on the Turing/CMP
hardware this started as (the driver throttled stock there -- see the
README) and is NOT reliably true on gfx1100, where nothing is throttled and
MIOpen is often excellent. Measured on this card, `F.conv2d` was a
REGRESSION against stock for two of three dtypes:

    conv2d fp16   stock 1.41 ms   ours 6.51 ms   -> 4.6x SLOWER
    conv2d fp32   stock 3.38 ms   ours 22.7 ms   -> 6.7x SLOWER
    conv2d bf16   stock 8.72 ms   ours 2.70 ms   -> 3.2x faster

The fp16 gap is not a tuning failure and cannot be closed by tuning: MIOpen
dispatches 3x3 fp16/fp32 conv to a hand-written gfx11 assembly *Winograd*
solver (miopenSp3AsmConvFury_*_f2x3_*), and Winograd computes fewer
multiplies than direct convolution. An implicit-GEMM kernel is playing a
strictly more expensive game, so it loses on arithmetic, not on quality.
bf16 is the mirror image: MIOpen has no bf16 Winograd solver on gfx11 at
all and falls back to im2col-to-memory plus a GEMM, which is where our CK
tier wins by 3x.

Rather than hardcode that per-dtype split -- which would be another prior
belief, just a better-informed one, and would be wrong the moment a shape
or a ROCm release moves -- this measures both on the first call for each
distinct shape and caches the winner. That is the same
benchmark-once-then-cache policy `run_conv2d_fp16` already uses to choose
among its own tile-shape variants and `run_conv3d_fp32` uses to choose
between direct and Winograd; this simply admits stock to the same contest.

COST: a few extra kernel launches on the first call per distinct shape,
paid once. Note candidates are run before being timed, which matters here
beyond ordinary warmup: the native and CK tiers do their OWN first-touch
instance selection inside C++, so timing a cold candidate would measure
that search rather than the kernel.

Used by conv2d/conv3d, group_norm, linear and bmm.

linear and bmm were excluded until recently, and the reason they no longer
are is worth recording. aiter's Triton GEMM was the only non-stock candidate
this project had for them, and it raises KeyError('gfx1100') because it
ships no RDNA3 tuning config -- so there was genuinely nothing to hold a
contest against. There are candidates now: hipblaslt_ops and ck_gemm_ops,
and measurement says neither beats stock everywhere. Against stock, fp16:

                             hipBLASLt   CK GEMM
    4096^3                     1.29x      1.20x
    4096x4096x1024             1.49x      1.47x
    4096x1280x1280             0.96x      1.14x
    1x4096x4096 (decode)       0.60x      0.90x

Two regressions and two large wins in the same column is the textbook case
for this module: hipBLASLt collapses to 0.60x on single-token decode while
winning 1.49x on an attention-shaped GEMM, and CK is flatter but rarely the
best at the top end. A fixed tier order would have to be wrong somewhere.

Set AMD_TUNED_TORCH_MEASURE_KERNELS=0 to disable and restore the old
always-prefer-our-kernels ordering (AMD_TUNED_TORCH_CONV_MEASURE is still
honoured as the previous name).
"""
from __future__ import annotations

import os
import threading
from typing import Callable, Optional

import torch

_ENABLED = (os.environ.get("AMD_TUNED_TORCH_MEASURE_KERNELS",
                           os.environ.get("AMD_TUNED_TORCH_CONV_MEASURE", "1")) != "0")

# shape key -> name of the winning candidate
_winners: dict = {}
_lock = threading.Lock()

# Timed runs per candidate. One is enough: this picks between kernels that
# differ by integer factors, not by percentages, and every extra run is
# latency the first call to a new shape pays.
_WARMUP = 2
_ITERS = 3


def enabled() -> bool:
    return _ENABLED


def _key(kind, input, weight, stride, padding, dilation):
    # Memory format belongs in the key: it changes which candidate wins
    # (the CK tier is channels-last native and pays a conversion otherwise),
    # not merely how fast the winner is.
    return (kind, input.dtype, tuple(input.shape), tuple(weight.shape),
            _norm(stride), _norm(padding), _norm(dilation),
            input.is_contiguous(memory_format=torch.channels_last)
            or input.is_contiguous(memory_format=torch.channels_last_3d))


def _norm(v):
    if isinstance(v, int):
        return (v,)
    return tuple(v)


def _time(fn) -> Optional[float]:
    """Median-free single measurement of `fn`, or None if it can't run.

    Returns None rather than raising so an unsupported candidate simply
    loses the contest instead of breaking the call.
    """
    try:
        for _ in range(_WARMUP):
            if fn() is None:
                return None
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(_ITERS):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / _ITERS
    except (RuntimeError, TypeError, AssertionError):
        return None


def cached(kind: str, input, weight, stride, padding, dilation) -> Optional[str]:
    """Name of the already-decided winner for this shape, or None if the
    contest hasn't run yet.

    Exists so the caller can dispatch straight to the winner without first
    building the candidate thunks. That matters more than it looks: the
    thunks are closures constructed per call, and once the decision is
    cached they are pure overhead on every conv in the model -- measured at
    ~0.09ms, which is 6% of a 1.4ms conv2d and proportionally far worse for
    the many small convs a U-Net is made of.
    """
    if not _ENABLED:
        return None
    with _lock:
        return _winners.get(_key(kind, input, weight, stride, padding, dilation))


def cached_key(kind: str, key) -> Optional[str]:
    """Winner for an already-built key, or None. See `cached`."""
    if not _ENABLED:
        return None
    with _lock:
        return _winners.get((kind,) + tuple(key))


def pick_key(kind: str, key, candidates: "list[tuple[str, Callable]]"):
    """Contest over a caller-supplied key, for ops whose identity isn't
    (input, weight, stride, padding, dilation) -- group_norm keys on the
    group count instead. Same policy as `pick`, including honouring the
    kill switch here rather than relying on every caller to check first."""
    if not _ENABLED:
        return None
    return _contest((kind,) + tuple(key), candidates)


def pick(kind: str, input, weight, stride, padding, dilation,
         candidates: "list[tuple[str, Callable]]"):
    """Returns the output of whichever candidate is fastest for this shape.

    `candidates` is ordered best-guess-first and each entry is
    (name, thunk); a thunk returns None to decline (the CK tier does this
    for problems no compiled instance supports). The last candidate is
    assumed to always work -- it is stock -- so there is always a winner.
    """
    if not _ENABLED:
        return None
    return _contest(_key(kind, input, weight, stride, padding, dilation), candidates)


def _contest(key, candidates: "list[tuple[str, Callable]]"):
    with _lock:
        winner = _winners.get(key)

    if winner is not None:
        for name, thunk in candidates:
            if name == winner:
                out = thunk()
                if out is not None:
                    return out
                break  # cached winner declined; re-measure below
        with _lock:
            _winners.pop(key, None)

    ranked = []
    for name, thunk in candidates:
        ms = _time(thunk)
        if ms is not None:
            ranked.append((ms, name, thunk))
    ranked.sort(key=lambda r: r[0])

    # Walk in measured order rather than just running the winner: a
    # candidate can decline at run time even though it measured fine (CK
    # allocates a workspace per call, so it can fail under memory pressure
    # having succeeded moments earlier). Falling through to the next-fastest
    # keeps that a performance event instead of a wrong answer -- returning
    # None here would make the caller think no kernel applied.
    for ms, name, thunk in ranked:
        out = thunk()
        if out is not None:
            with _lock:
                _winners[key] = name
            return out
    return None


def debug_winners() -> dict:
    """Snapshot of what won where -- {shape key: candidate name}.

    Exposed because "did my conv actually go to stock?" is otherwise
    invisible: every candidate returns a numerically equivalent tensor, so
    there is no way to tell from the output which kernel produced it.
    """
    with _lock:
        return dict(_winners)


def reset() -> None:
    """Drops all cached winners, forcing re-measurement. For benchmarking
    and tests -- selection is otherwise sticky for the process lifetime."""
    with _lock:
        _winners.clear()
