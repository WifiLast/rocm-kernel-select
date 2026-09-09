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

Used by conv2d/conv3d, group_norm, linear, bmm, and (when
enable_flash_attn_rocwmma() has installed it) attention -- see
amd_tuned_torch/__init__.py's _patched_sdpa_flash_attn_rocwmma, contesting
the vendored rocWMMA FlashAttention-2 kernel against whatever
F.scaled_dot_product_attention would otherwise have been (TE if enabled,
stock otherwise) instead of always preferring it unconditionally whenever
eligible.

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

CORRECTNESS VERIFICATION. Timing alone answers "is this candidate fast",
never "is this candidate right" -- a kernel that measures fastest but
silently computes the wrong answer for some shape would win the contest and
stay cached forever, with nothing in the timing loop able to notice. Every
`candidates` list passed into `pick`/`pick_key` ends with a reference
candidate that's "assumed to always work" (stock, in every call site this
module has today) -- so before crowning a NON-reference candidate the
winner, `_contest` now also numerically verifies its output against that
reference (torch.allclose, dtype-appropriate tolerance -- see
`_TOLERANCES`). A candidate that measures fastest but fails verification is
never returned: it's permanently excluded for that exact shape (in memory
AND on disk, same file the winner cache lives in) with a warning, and the
next-fastest candidate is verified in its turn, all the way down to the
reference itself if nothing else passes -- which, being assumed correct by
convention, needs no verification against itself.

This costs one extra real call (computing the reference's actual output,
not just its timing) on the first occurrence of a shape where a
non-reference candidate wins -- paid once per shape, exactly like the
timing sweep itself, never on the fast cached-winner path afterward. Set
AMD_TUNED_TORCH_KERNEL_SELECT_VERIFY=0 to skip this and go back to
trusting whichever candidate measured fastest, same as before this existed.

DISK PERSISTENCE. Winners were originally process-lifetime only -- a new
process re-paid every distinct shape's contest from scratch, which is fine
for a long-running server but wasteful for a script that starts fresh every
invocation (a benchmark harness, a CLI image-gen tool, anything re-run often
against the same handful of shapes). Winners are now additionally loaded
from, and saved to, a JSON file on first use / on every new decision, keyed
by GPU name + torch version + Python version
(amd_tuned_torch/_kernel_select_cache/<key>.json) -- the same
build-key-per-environment idea amd_tuned_torch/_native_loader.py already
uses for compiled extensions, for the same reason: a winner measured on one
GPU model or against one torch/ROCm build is not evidence about another.

Set AMD_TUNED_TORCH_KERNEL_SELECT_CACHE=0 to disable disk persistence and
go back to the original in-memory-only, re-measure-every-process behaviour
(AMD_TUNED_TORCH_MEASURE_KERNELS=0 still disables the contest itself
entirely, on top of this). AMD_TUNED_TORCH_KERNEL_SELECT_CACHE_DIR
relocates the cache directory -- point it outside the source tree to share
one cache across several checkouts of this project on the same machine, or
delete it to force every shape to be re-measured once. reset() (below)
clears the in-memory cache AND deletes that GPU's on-disk file, so a
benchmarking script that wants a clean slate doesn't need to know the path.

A cached winner that later declines at runtime (see `_contest`'s docstring
for why that can happen even after measuring fine) is evicted from both the
in-memory dict and the disk file, not just re-measured for that call -- a
stale entry must not keep surviving process restarts once it starts
declining.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
import warnings
from typing import Callable, Optional

import torch

_ENABLED = (os.environ.get("AMD_TUNED_TORCH_MEASURE_KERNELS",
                           os.environ.get("AMD_TUNED_TORCH_CONV_MEASURE", "1")) != "0")

_VERIFY_ENABLED = os.environ.get("AMD_TUNED_TORCH_KERNEL_SELECT_VERIFY", "1") != "0"

# shape key -> name of the winning candidate
_winners: dict = {}
# shape key -> set of candidate names that measured fastest at some point
# but failed numerical verification against the reference -- excluded from
# all future contests for that exact key, not just the one that caught them.
_bad_candidates: dict = {}
_lock = threading.Lock()

# GEMM/conv/attention kernels accumulate in different orders (different
# tile sizes, different reduction trees) across implementations, so
# bit-exact equality is not a realistic bar even between two CORRECT
# kernels -- these follow the same order-of-magnitude
# torch.testing.assert_close uses by default per dtype.
_TOLERANCES = {
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (2e-2, 2e-2),
    torch.float32: (1e-4, 1e-5),
}
_DEFAULT_TOLERANCE = (2e-2, 2e-2)


# Fraction of the reference output's own RMS magnitude that _verify's
# absolute tolerance is allowed to grow to. See _verify's docstring for
# the three measured cases that made a purely absolute atol reject
# correct kernels; `rtol` is reused as that fraction so the bar stays
# "the same relative slack, measured against the tensor's scale instead
# of one element's".
_VERIFY_SCALE_SAMPLE = 1 << 20


def _reference_scale(reference_out) -> "Optional[float]":
    """RMS magnitude of `reference_out`, or None if it can't be measured.

    Sampled with a stride rather than reduced whole: a reference output
    here can be hundreds of MB (a VAE decoder's 1x128x2048x1024 fp16
    activation is 512MB), and upcasting all of it to float32 to compute
    one scalar would cost more than the convolution being verified. A
    strided sample spans the whole tensor, unlike a prefix, so a padded
    border or an all-zero leading region can't stand in for the whole."""
    try:
        flat = reference_out.flatten()
        step = max(1, flat.numel() // _VERIFY_SCALE_SAMPLE)
        sample = flat[::step].float()
        scale = float(sample.pow(2).mean().sqrt())
    except (RuntimeError, TypeError, ValueError, AttributeError, ZeroDivisionError):
        return None
    return scale if math.isfinite(scale) and scale > 0.0 else None


def _verify(candidate_out, reference_out, tolerance: "Optional[tuple[float, float]]" = None) -> bool:
    """Best-effort numerical agreement check. A comparison that itself
    can't run (shape mismatch, an exotic dtype, a non-tensor return) is
    treated as a verification FAILURE, not a pass -- unlike a thunk
    returning None (a candidate declining a problem it recognizes it can't
    handle), "couldn't even compare" must never be silently treated as
    "fine".

    THE BAR IS RELATIVE TO THE OUTPUT'S MAGNITUDE, not absolute. `atol`
    from the table below (or from `tolerance`) is a floor; the bar
    actually used is `max(atol, rtol * RMS(reference_out))`. A fixed atol
    asks the near-zero elements of a large accumulation to agree to a
    precision they never had, and since a rejection here is a PERMANENT
    per-shape blacklist persisted to disk, that permanently loses correct,
    faster kernels. All three cases measured on gfx1100 were of exactly
    that shape -- a handful of near-zero elements out of millions, on
    candidates whose worst disagreement was a rounding artifact:

      * CK conv3d fp16, 1x512x8x32x32 k3: 11 of 4.19M elements outside
        (1e-2, 1e-2), median |ref| among them 0.06, while CK's error
        against an fp32 reference was IDENTICAL to stock's (max 0.417,
        mean 0.0207). Cost: 1.8x, CK 1.70ms vs the elected 3.0ms.
      * FFT conv3d fp32, 1x32x32x64x64 k15: 158 of 4.19M outside
        (1e-3, 1e-4), median |ref| 0.034, worst disagreement 7.3e-4 on an
        output whose RMS is 291 -- 2.5e-6 relative. Cost: 157x
        (6845ms vs 43.6ms).
      * CK conv2d fp16, 2x640x128x64->640 k3 stride2 (a real SDXL-style
        downsampling conv, see miopen.logs): worst disagreement 0.375 on
        an output RMS of 75.4 (0.5%). Cost: 2.24x, CK 0.564ms vs stock's
        1.262ms on the one shape MIOpen serves with im2col+GEMM rather
        than Winograd.

    With the scaled bar those three pass with 4x, 400x and 2x of headroom
    respectively, while a genuinely wrong candidate is off by order-RMS
    (100x the bar) and even a subtly wrong one -- a dropped channel, say,
    which costs RMS/sqrt(C) -- stays an order of magnitude above it.

    `tolerance`, when given, overrides the `_TOLERANCES` dtype lookup for
    the (rtol, atol) pair -- for a contest between algorithmically
    DIFFERENT implementations (not just different tile/instance choices of
    the same algorithm), a dtype-keyed default calibrated for
    direct-convolution/GEMM-style candidates can be the wrong bar. E.g.
    FFT-based convolution (amd_tuned_torch.fftconv_ops) accumulates
    rounding error differently from direct convolution -- correct, but
    with a different rtol than fp32's default (1e-4); see
    fftconv_ops.fftconv_tolerance for the looser pair that contest passes
    instead of this module's shared default."""
    try:
        if tolerance is not None:
            rtol, atol = tolerance
        else:
            rtol, atol = _TOLERANCES.get(candidate_out.dtype, _DEFAULT_TOLERANCE)
        scale = _reference_scale(reference_out)
        if scale is not None:
            atol = max(atol, rtol * scale)
        return bool(torch.allclose(candidate_out.float(), reference_out.float(),
                                    rtol=rtol, atol=atol, equal_nan=True))
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return False

# Bumped whenever _verify's bar changes. A blacklist entry is a RECORD OF
# A JUDGEMENT, not a fact about the kernel: an entry written when the bar
# was purely absolute (see _verify's docstring for the three correct
# kernels that bar rejected) is not evidence under the current one, and
# without this it would outlive the fix forever -- the entries are
# persisted, and a cached winner short-circuits the contest that would
# otherwise re-measure. On a version mismatch the blacklist is dropped
# AND the winners for exactly those shapes are dropped with it, so each
# affected shape is re-contested once and every other cached decision
# survives untouched.
_VERIFY_BAR_VERSION = 2
_DISK_CACHE_ENABLED = os.environ.get("AMD_TUNED_TORCH_KERNEL_SELECT_CACHE", "1") != "0"
_CACHE_DIR = os.environ.get(
    "AMD_TUNED_TORCH_KERNEL_SELECT_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "_kernel_select_cache"),
)


def _device_key() -> str:
    """GPU model + torch + Python version, sanitized for a filename -- a
    winner measured on one of these is not evidence about a different one,
    same reasoning as amd_tuned_torch/_native_loader.py's build_key()."""
    try:
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        name = "unknown"
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    torch_ver = torch.__version__.replace("+", "_").replace("/", "_")
    return f"{safe_name}-torch-{torch_ver}-cp{sys.version_info.major}{sys.version_info.minor}"


def _cache_path() -> str:
    return os.path.join(_CACHE_DIR, f"{_device_key()}.json")


def _encode(v):
    """torch.dtype and tuple aren't JSON types -- tag them so _decode can
    invert this exactly, rather than guessing from plain lists/strings."""
    if isinstance(v, torch.dtype):
        return {"__dtype__": str(v)}
    if isinstance(v, tuple):
        return {"__tuple__": [_encode(x) for x in v]}
    return v


def _decode(v):
    if isinstance(v, dict):
        if "__dtype__" in v:
            return getattr(torch, v["__dtype__"].rsplit(".", 1)[-1])
        if "__tuple__" in v:
            return tuple(_decode(x) for x in v["__tuple__"])
    return v


def _decode_key(encoded_key_str: str):
    return tuple(_decode(x) for x in json.loads(encoded_key_str))


def _load_disk_cache() -> None:
    """Best-effort: a missing, corrupt, or unreadable file just means an
    empty starting cache (identical to today's in-memory-only behaviour),
    never an import-time failure.

    Handles both the current format ({"winners": {...}, "bad": {...}}) and
    the older flat format (a bare {encoded key: winner} dict, from before
    _bad_candidates existed) -- a file written by an older amd_tuned_torch
    must not be silently discarded just because this one added a section."""
    if not _DISK_CACHE_ENABLED:
        return
    try:
        with open(_cache_path(), "r") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return
    if "winners" in raw or "bad" in raw:
        winners_raw = raw.get("winners", {})
        bad_raw = raw.get("bad", {})
    else:
        winners_raw, bad_raw = raw, {}  # old flat-format file
    # Blacklists recorded under a superseded verification bar are discarded,
    # along with the winners of those shapes so they get re-contested once
    # -- see _VERIFY_BAR_VERSION.
    stale_judgements = raw.get("verify_bar") != _VERIFY_BAR_VERSION
    with _lock:
        for encoded_key_str, winner in winners_raw.items():
            if stale_judgements and encoded_key_str in bad_raw:
                continue
            try:
                _winners.setdefault(_decode_key(encoded_key_str), winner)
            except (ValueError, TypeError):
                continue
        if not stale_judgements:
            for encoded_key_str, names in bad_raw.items():
                try:
                    _bad_candidates.setdefault(_decode_key(encoded_key_str), set()).update(names)
                except (ValueError, TypeError):
                    continue


def _save_disk_cache() -> None:
    """Rewrites the whole file from the current in-memory _winners and
    _bad_candidates -- only called after a new decision, eviction, or
    verification failure (once per distinct shape event, not once per
    call), so this is nowhere near the hot path. Written via a temp file +
    os.replace so a crash mid-write can't leave a truncated/corrupt JSON
    file for the next process to trip over."""
    if not _DISK_CACHE_ENABLED:
        return
    path = _cache_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _lock:
            raw = {
                "verify_bar": _VERIFY_BAR_VERSION,
                "winners": {json.dumps([_encode(x) for x in key]): winner
                            for key, winner in _winners.items()},
                "bad": {json.dumps([_encode(x) for x in key]): sorted(names)
                        for key, names in _bad_candidates.items() if names},
            }
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(raw, f)
        os.replace(tmp, path)
    except OSError:
        pass

# Timed runs per candidate, each timed separately so _time can take the
# fastest rather than the mean (see its docstring: a single delayed call
# was flipping decisions). Kept small either way -- this picks between
# kernels that differ by integer factors, not by percentages, and every
# extra run is latency the first call to a new shape pays.
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
    """Fastest of `_ITERS` timed runs of `fn`, or None if it can't run.

    Returns None rather than raising so an unsupported candidate simply
    loses the contest instead of breaking the call.

    WHY THE MINIMUM AND NOT THE MEAN. Per-call times here are not
    symmetrically noisy -- they have a long right tail and no left tail,
    because a call can be delayed (driver work, another process on the
    GPU, a kernel launch queued behind something else) but cannot run
    faster than the kernel. With `_ITERS` this small, one delayed call
    moves a mean far enough to flip a decision: measured call-by-call on
    the CK tier for 2x640x128x64->640 k3 stride2 (a real downsampling
    conv from miopen.logs), 1.47 1.31 6.29 1.35 1.32 7.49 1.32 ms -- a
    steady 1.3ms with occasional ~6ms outliers. Averaged over 3 runs
    that reads as ~3ms and loses to stock's 1.6ms; the minimum reads
    1.31ms and wins, which is the truth (CK is 2.24x faster than stock
    on that shape at 20 iterations). Since the contest's whole job is
    ranking candidates against each other, the least-contaminated
    estimate of each one is what it should compare, and the outliers
    belong to the machine, not the kernel.

    Costs nothing extra: the same `_ITERS` calls, timed individually with
    one event pair each instead of one pair around the whole loop, and
    still a single synchronize at the end."""
    try:
        for _ in range(_WARMUP):
            if fn() is None:
                return None
        torch.cuda.synchronize()
        events = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(_ITERS)]
        for start, end in events:
            start.record()
            fn()
            end.record()
        torch.cuda.synchronize()
        return min(start.elapsed_time(end) for start, end in events)
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


def pick_key(kind: str, key, candidates: "list[tuple[str, Callable]]",
             tolerance: "Optional[tuple[float, float]]" = None):
    """Contest over a caller-supplied key, for ops whose identity isn't
    (input, weight, stride, padding, dilation) -- group_norm keys on the
    group count instead. Same policy as `pick`, including honouring the
    kill switch here rather than relying on every caller to check first.

    `tolerance`, when given, overrides this contest's numerical-verification
    bar -- see `_verify`'s docstring for why (contesting algorithmically
    different implementations, not just different instances of the same
    one, can need a looser bar than `_TOLERANCES`' dtype defaults)."""
    if not _ENABLED:
        return None
    return _contest((kind,) + tuple(key), candidates, tolerance=tolerance)


def pick(kind: str, input, weight, stride, padding, dilation,
         candidates: "list[tuple[str, Callable]]",
         tolerance: "Optional[tuple[float, float]]" = None):
    """Returns the output of whichever candidate is fastest for this shape.

    `candidates` is ordered best-guess-first and each entry is
    (name, thunk); a thunk returns None to decline (the CK tier does this
    for problems no compiled instance supports). The last candidate is
    assumed to always work -- it is stock -- so there is always a winner.

    `tolerance`, when given, overrides this contest's numerical-verification
    bar -- see `_verify`'s docstring."""
    if not _ENABLED:
        return None
    return _contest(_key(kind, input, weight, stride, padding, dilation), candidates,
                     tolerance=tolerance)


def _contest(key, candidates: "list[tuple[str, Callable]]",
             tolerance: "Optional[tuple[float, float]]" = None):
    reference_name = candidates[-1][0] if candidates else None

    with _lock:
        winner = _winners.get(key)
        bad = set(_bad_candidates.get(key, ()))

    if winner is not None and winner not in bad:
        for name, thunk in candidates:
            if name == winner:
                out = thunk()
                if out is not None:
                    return out
                break  # cached winner declined; re-measure below
        with _lock:
            _winners.pop(key, None)
        _save_disk_cache()

    ranked = []
    for name, thunk in candidates:
        if name in bad:
            continue  # verified wrong for this exact shape before; never time it again
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
    #
    # A candidate that DOES produce output but disagrees with the reference
    # (see module docstring's CORRECTNESS VERIFICATION section) is a
    # different failure mode from declining, and handled the same way:
    # excluded, and the next-fastest one gets its turn -- fast-and-wrong
    # must never win just because nothing timed it out.
    _UNSET = object()
    reference_out = _UNSET
    for ms, name, thunk in ranked:
        out = thunk()
        if out is None:
            continue
        if name == reference_name or not _VERIFY_ENABLED:
            with _lock:
                _winners[key] = name
            _save_disk_cache()
            return out
        if reference_out is _UNSET:
            reference_out = None
            for ref_name, ref_thunk in candidates:
                if ref_name == reference_name:
                    reference_out = ref_thunk()
                    break
            if reference_out is None:
                # The reference itself -- "assumed to always work" by every
                # call site's own convention -- declined this round. Nothing
                # to verify against, so accept the fastest candidate rather
                # than wrongly blacklist it for a problem that isn't its own.
                with _lock:
                    _winners[key] = name
                _save_disk_cache()
                return out
        if _verify(out, reference_out, tolerance=tolerance):
            with _lock:
                _winners[key] = name
            _save_disk_cache()
            return out
        warnings.warn(
            f"amd_tuned_torch.kernel_select: candidate '{name}' for {key} measured "
            f"fastest but its output didn't match '{reference_name}' within tolerance "
            "-- excluding it for this exact shape from now on and using the "
            "next-fastest candidate instead. This usually means a real numerical "
            "correctness bug in that kernel for this shape, not benchmark noise.",
            UserWarning,
        )
        with _lock:
            _bad_candidates.setdefault(key, set()).add(name)
        _save_disk_cache()

    if reference_out not in (_UNSET, None):
        # Every non-reference candidate that measured fine either failed
        # verification or isn't reached here at all (already returned) --
        # the reference's own output, already computed above, is correct
        # by convention and doesn't need re-running.
        with _lock:
            _winners[key] = reference_name
        _save_disk_cache()
        return reference_out
    return None


def debug_winners() -> dict:
    """Snapshot of what won where -- {shape key: candidate name}.

    Exposed because "did my conv actually go to stock?" is otherwise
    invisible: every candidate returns a numerically equivalent tensor, so
    there is no way to tell from the output which kernel produced it.
    """
    with _lock:
        return dict(_winners)


def debug_bad_candidates() -> dict:
    """Snapshot of every candidate excluded for failing correctness
    verification -- {shape key: {candidate names}}. Exposed for the same
    reason debug_winners() is: a permanently-excluded candidate is
    otherwise invisible after its one warning at the moment it failed."""
    with _lock:
        return {key: set(names) for key, names in _bad_candidates.items()}


def reset() -> None:
    """Drops all cached winners AND all correctness exclusions (in memory
    AND this GPU's on-disk file), forcing re-measurement and
    re-verification. For benchmarking and tests -- selection is otherwise
    sticky for the process lifetime, and now across process restarts too
    (see the module docstring's DISK PERSISTENCE section)."""
    with _lock:
        _winners.clear()
        _bad_candidates.clear()
    if _DISK_CACHE_ENABLED:
        try:
            os.remove(_cache_path())
        except OSError:
            pass


if _ENABLED:
    _load_disk_cache()
