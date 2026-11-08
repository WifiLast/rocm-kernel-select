"""amd_tuned_torch._dispatch -- the always-on monkeypatch tier enable()/
disable() installs: linear/matmul/bmm (aiter/hipBLASLt/CK GEMM contest),
conv2d/conv3d (native HIP/CK/FFT-conv contest), group_norm (native HIP/CK
contest), and the TransformerEngine-backed layer_norm/rms_norm/gelu/silu/
scaled_dot_product_attention wrappers -- plus the shared eligibility/
install-and-restore machinery every one of those wrappers is built from,
enable()/disable()/is_enabled() themselves, and the TorchScript
(torch.jit.script/trace) compatibility guard enable()'s patches require
(see that section's own comment block below for why).

Split out of amd_tuned_torch/__init__.py (previously one ~2300-line file)
purely for readability -- no behavior changed by this split. Every name
defined here is re-exported at the package's top level
(`from ._dispatch import *` in __init__.py, with this module's __all__
below listing every name including the underscore-prefixed ones tests
reach into directly, e.g. amd_tuned_torch._grad_safe/_usable/_ORIGINALS/
_GROUPNORM_DTYPES), so `amd_tuned_torch.X` resolves exactly as it did
before the split, for every X defined here.

See amd_tuned_torch/_opt_in_tiers.py for everything NOT installed by
enable()/disable(): opt-in-only kernel tiers (INT8 linear, conv3d
Winograd, flash_attn_rocwmma, triton_kernels rmsnorm) and the standalone
never-monkeypatched ops (mla_decode, fused_add_rms_norm, rotary_embedding,
fused_linear_cross_entropy, bias_swiglu, SmoothQuant calibration).
"""
from __future__ import annotations

import atexit
import logging
import math
import os
from typing import Any, Callable

import torch
import torch.nn.functional as F

from . import _C
from . import aiter_ops
from . import hipblaslt_ops
from . import ck_ops
from . import ck_gemm_ops
from . import ck_norm_ops
from . import te_ops
from . import te_extra_ops
from . import compile_ops
from . import kernel_select
from . import conv_fusion_ops
from . import flexgemm_ops
from . import rocsparse_ops
from . import fftconv_ops
from . import flash_mm_kernel
from . import fftconv_calibration
from . import splitk_gemm_ops
from . import comfy_kitchen_ops
from . import einsum_ops

# ---------------------------------------------------------------------------
# Dispatch debugging.
#
# Every patched op in this file ends in one of two places -- an accelerated
# tier, or the stock op it shadows -- and from outside the process the two
# are indistinguishable: the call returns the right numbers either way.
# That is the whole design (see this module's docstring), and it is also
# what makes "why is nothing getting faster?" unanswerable without
# instrumentation. The answer is almost never "the kernel is slow"; it is a
# ROUTING decision made here -- a dtype outside the tier's list, a Tensor
# subclass, groups != 1, a gradient in flight, a candidate declining by
# returning None, a kernel raising into one of the swallow-and-fall-back
# except clauses below, or kernel_select having measured stock as the
# genuine winner for that shape. This block makes every one of those
# decisions observable.
#
# AMD_TUNED_TORCH_DISPATCH_DEBUG=1 turns it on (logger
# "amd_tuned_torch.dispatch", one line per routing decision at DEBUG, plus
# a route summary at process exit at INFO).
# AMD_TUNED_TORCH_DISPATCH_DEBUG_FILE=<path> sends that same output to a
# file, and is enough on its own to turn debugging on -- same two-env-var
# contract cache.py already uses, including attaching a handler directly to
# this logger rather than to the root (so it works in a host app that never
# calls logging.basicConfig(), and changes nothing for any other logger in
# the process).
#
# COST WHEN OFF. Every call site is written `if _DEBUG: _debug(...)`, never
# a bare `_debug(...)`. That is deliberate and not merely stylistic: these
# are the hottest paths in the package (F.linear/matmul/bmm/conv2d run
# thousands of times per diffusion step), and a bare call would still build
# an argument tuple, a kwargs dict, and any shape tuple inside it before
# the function could early-return. With the guard, a non-debugging run pays
# one module-global bool lookup per decision and nothing else.
#
# COST WHEN ON. One dict update per decision always, plus string formatting
# only if a DEBUG-level handler is actually listening. The route counters
# are therefore usable on their own (AMD_TUNED_TORCH_DISPATCH_DEBUG=1 with
# the per-call log left unconsumed at WARNING) when the per-call trace
# would be too much output -- which, for a U-Net, it generally is.
_dbg_log = logging.getLogger("amd_tuned_torch.dispatch")

_DEBUG_ENV_SET = os.environ.get("AMD_TUNED_TORCH_DISPATCH_DEBUG", "0") not in ("0", "", "false", "False")
_DEBUG_FILE = os.environ.get("AMD_TUNED_TORCH_DISPATCH_DEBUG_FILE", "")
_DEBUG = _DEBUG_ENV_SET or bool(_DEBUG_FILE)

if _DEBUG:
    _dbg_log.setLevel(logging.DEBUG)
    _dbg_formatter = logging.Formatter("[amd_tuned_torch.dispatch] %(asctime)s %(levelname)s %(message)s")
    # Same per-handler-type duplicate guard as cache.py: a stderr handler
    # and a file handler are independent opt-ins that can both be present.
    if _DEBUG_ENV_SET and not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in _dbg_log.handlers
    ):
        _dbg_stream_handler = logging.StreamHandler()
        _dbg_stream_handler.setFormatter(_dbg_formatter)
        _dbg_log.addHandler(_dbg_stream_handler)
    if _DEBUG_FILE and not any(isinstance(h, logging.FileHandler) for h in _dbg_log.handlers):
        _dbg_file_handler = logging.FileHandler(_DEBUG_FILE)  # mode="a": append, never truncate
        _dbg_file_handler.setFormatter(_dbg_formatter)
        _dbg_log.addHandler(_dbg_file_handler)

# (op, route) -> call count, e.g. ("conv2d", "stock") -> 812. Plain dict
# updates with no lock: every mutation is a single get/set pair under the
# GIL, so a count can be lost under thread contention but the dict can
# never be corrupted, and an approximate count is entirely adequate for
# what this is -- a diagnostic tally, never anything a dispatch decision
# reads back.
_ROUTE_COUNTS: dict = {}


def _dbg_t(t: Any) -> str:
    """One tensor as a compact `fp16(2, 320, 64, 64)` string, with the
    details that actually decide routing here appended: a non-strided
    layout (sparse matmul), and requires_grad (the single most common
    reason an accelerated tier declines -- see the TRAINING-TIME GAP note
    below)."""
    if not isinstance(t, torch.Tensor):
        return repr(t)
    s = "%s%s" % (str(t.dtype).replace("torch.", ""), tuple(t.shape))
    if t.layout != torch.strided:
        s += ":" + str(t.layout).replace("torch.", "")
    if t.requires_grad:
        s += "+grad"
    if type(t) not in _PLAIN_TENSOR_TYPES:
        s += ":" + type(t).__name__  # a subclass -- see _PLAIN_TENSOR_TYPES
    return s


def _debug(op: str, route: str, why: str = "", **fields: Any) -> None:
    """Record one routing decision: `op` is the patched op (linear, conv2d,
    ...), `route` is where the call actually went -- a tier name
    (hipblaslt/ck/ck_gemm/native/fftconv/splitk/aiter/rocsparse/sparse/te),
    "stock", "contest" for the one call per shape that runs the
    kernel_select measurement, or "autograd" for a call handed to one of
    the torch.autograd.Function wrappers. `why` is the reason, which for a
    "stock" route is the part worth reading.

    Only ever called from behind an `if _DEBUG:` guard -- see the COST WHEN
    OFF note above."""
    _ROUTE_COUNTS[(op, route)] = _ROUTE_COUNTS.get((op, route), 0) + 1
    if not _dbg_log.isEnabledFor(logging.DEBUG):
        return
    parts = ["%-10s -> %-10s" % (op, route)]
    if why:
        parts.append(why)
    for k, v in fields.items():
        parts.append("%s=%s" % (k, _dbg_t(v) if isinstance(v, torch.Tensor) else v))
    _dbg_log.debug("  ".join(parts))


def dispatch_debug_enabled() -> bool:
    """True when AMD_TUNED_TORCH_DISPATCH_DEBUG/_FILE turned routing
    debug on for this process. Read once at import time -- flipping the env
    var later does nothing, same as everywhere else in this package."""
    return _DEBUG


def dispatch_debug_counts() -> dict:
    """A copy of the (op, route) -> count tally. Empty unless
    dispatch_debug_enabled()."""
    return dict(_ROUTE_COUNTS)


def dispatch_debug_reset() -> None:
    """Drop the tally -- e.g. between a warm-up pass (where every shape
    pays its one-off "contest" route) and the steady-state run you actually
    want the numbers for."""
    _ROUTE_COUNTS.clear()


def dispatch_debug_report() -> str:
    """The tally as a table, ops in call-count order, routes within an op
    likewise. The line to look for is a large `stock` count: that is the
    work this package did NOT accelerate, and the per-call log (DEBUG
    level) carries the reason for each one."""
    if not _ROUTE_COUNTS:
        return ("amd_tuned_torch dispatch: no routing decisions recorded"
                + ("" if _DEBUG else " (AMD_TUNED_TORCH_DISPATCH_DEBUG is not set)"))
    per_op: dict = {}
    for (op, route), n in _ROUTE_COUNTS.items():
        per_op.setdefault(op, {})[route] = n
    total = sum(_ROUTE_COUNTS.values())
    lines = ["amd_tuned_torch dispatch routes (%d decisions):" % total]
    for op in sorted(per_op, key=lambda o: -sum(per_op[o].values())):
        op_total = sum(per_op[op].values())
        lines.append("  %-12s %d" % (op, op_total))
        for route in sorted(per_op[op], key=lambda r: -per_op[op][r]):
            n = per_op[op][route]
            lines.append("      %-12s %8d  %5.1f%%" % (route, n, 100.0 * n / op_total))
    return "\n".join(lines)


def _dbg_atexit_report() -> None:
    """Emit the summary once at interpreter exit. Guarded on the tally
    being non-empty so a process that imported the package but never ran a
    patched op stays silent."""
    if _ROUTE_COUNTS:
        _dbg_log.info("%s", dispatch_debug_report())


if _DEBUG:
    atexit.register(_dbg_atexit_report)


# Raw, unpatched access to the native HIP group_norm kernel for manual use
# regardless of what enable() patches. (linear/matmul/bmm live in
# aiter_ops, not here -- see amd_tuned_torch.aiter_ops.linear_fp16/bmm_fp16.)
ops = _C

_GEMM_DTYPES = (torch.float16, torch.bfloat16)              # linear/matmul/bmm: aiter Triton fp16/bf16 only
_GROUPNORM_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_TE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_CONV_NATIVE_DTYPES = (torch.float16, torch.float32)  # conv2d/conv3d native HIP kernels: no bf16 support
# Composable Kernel's WMMA conv instances: fp16/bf16 (no fp32 -- gfx1100's
# WMMA units have no fp32 mode). bf16 is the important half of that: it is
# the one conv case where stock ROCm has no good solver at all (see
# ck_ops.py's measured table).
_CK_CONV_DTYPES = (torch.float16, torch.bfloat16)
# FFT-conv (amd_tuned_torch.fftconv_ops) as a conv2d/conv3d tier: fp32
# included, unlike the CK/native tiers above. This is not a WMMA kernel --
# the transform runs in float32 for every input dtype anyway (see
# fft_conv's MIXED PRECISION docstring section), so fp32 input costs it
# nothing extra, and fp16/bf16 buy less here than elsewhere.
_FFTCONV_CONV_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# Kernel width at/above which the FFT tier is worth CONTESTING (not
# preferring -- kernel_select still measures it against stock/CK/native).
# See _fftconv_conv_candidate's docstring for the measured crossovers
# these defaults come from, and why 2D's sits an order of magnitude
# higher in kernel width than 3D's.
_FFTCONV_CONV2D_ENABLED = os.environ.get("AMD_TUNED_TORCH_FFTCONV2D", "1") != "0"
_FFTCONV_CONV3D_ENABLED = os.environ.get("AMD_TUNED_TORCH_FFTCONV3D", "1") != "0"
# _FFTCONV_CONV{2,3}D_MIN_KERNEL are defined further down, next to
# _FFTCONV_CONV3D_MIN_POSITIONS -- they need _calibrated_fftconv_default,
# which needs the calibration file loaded first.
# Measured calibration (amd_tuned_torch.fftconv_calibration, written by
# tools/benchmark_fftconv3d_min_positions.py) for the min_positions guess
# below -- same "loaded once, before the constant it feeds is defined"
# shape as flexgemm_ops.py's own _sparse_conv_calibration/_calibrated_default.
# Precedence per env var: explicit env var (checked at the os.environ.get
# call site below) > this measured calibration > the hardcoded guess as a
# last resort for a GPU/build that hasn't been benchmarked yet.
_fftconv_calibration_data = fftconv_calibration.load()


def _calibrated_fftconv_default(dim_key: str, field: str, hardcoded_default: str) -> str:
    """Same contract as flexgemm_ops._calibrated_default: the env-var
    default for one (dim_key, field) pair, as a string (feeds straight into
    os.environ.get(name, default)) -- a measured value if
    fftconv_calibration loaded one, else `hardcoded_default`. Only ever
    decides what the *default* is, never overrides an explicit env var."""
    value = _fftconv_calibration_data.get(dim_key, {}).get(field)
    return str(value) if value is not None else hardcoded_default


# conv3d only, not conv2d: FFT-conv3d's padded transform is the most
# memory-hungry tier in this file (measured ~1.5GB peak on a B=8 C=512
# L=8192 conv1d whose inputs are only 134MB -- see _try_fftconv1d_fastpath's
# OOM note), and for a small input volume there's no reasonable kernel width
# for which paying that fixed transform/allocation overhead beats a cheap
# direct conv outright, wide kernel or not. min(weight.shape[2:]) above only
# gates on KERNEL size; this gates on `input`'s own spatial size (batch *
# every dim after channel, same quantity flexgemm_ops._n_spatial_positions
# computes for its sparse-conv gate, not imported from there to avoid a
# cross-module dependency for one line). "2048" is a hardcoded fallback
# guess, used only until tools/benchmark_fftconv3d_min_positions.py has
# measured this GPU's actual crossover (see _calibrated_fftconv_default
# above) -- override per-process via AMD_TUNED_TORCH_FFTCONV3D_MIN_POSITIONS.
_FFTCONV_CONV3D_MIN_POSITIONS = int(
    os.environ.get("AMD_TUNED_TORCH_FFTCONV3D_MIN_POSITIONS",
                    _calibrated_fftconv_default("conv3d", "min_positions", "2048")))

# Kernel width at/above which the FFT tier is worth CONTESTING (not
# preferring -- kernel_select still measures it against stock/CK/native).
# Same precedence as min_positions above: explicit env var > measured
# calibration > this default. These are no longer the original guesses (32
# and 7): both are now MEASURED on gfx1100 in fp32 by
# tools/benchmark_fftconv_min_kernel.py. 2D's still sits far above 3D's,
# because a 2D transform's fixed cost is amortized over far fewer output
# positions per unit of kernel work -- that part of the original reasoning
# held up.
#
# The two errors are NOT symmetric. Too LOW costs one wasted contest per
# distinct shape, paid once and cached by kernel_select, which then measures
# FFT against stock/CK/native and rejects it if it actually loses. Too HIGH
# means the FFT candidate is never built at all, so the loss is silent and
# repeats on every call forever. The cheap error is the low one, which is
# why tools/benchmark_fftconv_min_kernel.py writes the SMALLEST per-shape
# crossover it measured -- the width below which no shape won, i.e. the
# point below which this pre-filter can be confident -- rather than the
# largest or an average.
#
# The crossover moves a lot with input size, which is what makes a single
# number a compromise at all. Per-shape crossovers measured on gfx1100,
# fp32, over repeated runs:
#
#   conv2d  8x32x128x128   k=25  (25, 25, 31 across three runs)
#   conv2d  4x64x256x256   k=41  (41 every run)
#   conv2d  16x16x64x64    never wins within the swept range (k <= 63)
#   conv3d  2x16x32x32x32  k=5   (5 every run, winning 2.1-2.3x there)
#   conv3d  1x32x48x48x48  k=9
#
# Hence 25 and 5. conv3d's 5 is solid -- three runs agreed exactly, with
# control columns at 0.99-1.03x. conv2d's is the weaker of the two: one of
# three runs read 31 rather than 25, and that run's FFT timings were
# uniformly ~2x slower than the other two (the GPU was shared with another
# sweep), so 25 is taken as the clean measurement and 31 as the
# contaminated one. If conv2d's tier ever looks like it is contesting
# shapes it cannot win, re-run the sweep on an idle GPU before assuming
# this number is wrong.
#
# NOT a claim about other dtypes: the sweep is fp32 only. fp16/bf16 inputs
# still transform in float32 (see fft_conv's MIXED PRECISION section), so
# the FFT side barely moves, but stock convNd gets materially faster in
# half precision -- if anything these thresholds are too LOW there, which
# is the cheap direction.
_FFTCONV_CONV2D_MIN_KERNEL = int(
    os.environ.get("AMD_TUNED_TORCH_FFTCONV2D_MIN_KERNEL",
                    _calibrated_fftconv_default("conv2d", "min_kernel", "27")))
_FFTCONV_CONV3D_MIN_KERNEL = int(
    os.environ.get("AMD_TUNED_TORCH_FFTCONV3D_MIN_KERNEL",
                    _calibrated_fftconv_default("conv3d", "min_kernel", "6")))
# silu excludes fp32: benchmarked slower than stock (0.80x, see
# benchmark.json) even though fp16 is a real win (1.15x) -- fp32 stays on
# the TE path's fp16/bf16-only sibling dtypes instead of falling back per-call.
_SILU_DTYPES = (torch.float16, torch.bfloat16)

# ---------------------------------------------------------------------------
# Which TE-backed ops are worth installing, and when.
#
# Measured on RX 7900 XTX (gfx1100) by tools/bench_te_ops.py; raw numbers in
# analyse/benchmarks/te_ops_bench.txt. Two things about that measurement
# shape every constant below:
#
#   1. It times forward AND forward+backward separately. The older
#      forward-only figures in benchmark.json disagree with the fwd+bwd ones
#      for nearly every op here, and training spends most of its time in
#      backward, so the fwd+bwd numbers are the ones these defaults follow.
#   2. Each data point comes from its OWN process. Sweeping shapes in a
#      single process gave badly wrong answers -- an in-process sweep
#      reported TE attention at 0.66x for a config that measures 1.58x in
#      isolation, and produced a non-monotonic curve (S=512 "slower" than
#      S=768) that invented a crossover which does not exist. Re-measure
#      one shape per process.
#
# Both patched norms and patched attention lose on small inputs for the same
# structural reason: TE's kernels carry a fixed setup cost per call (~1.15ms
# for the unfused attention path), so stock wins until there is enough work
# to amortize it.
#
#   * NORMS (layer_norm, rms_norm) need enough ROWS (tokens) -- row count
#     matters independently of total size, because TE's norm kernels
#     parallelize over rows: at a fixed 16.8M elements, 65536 rows measured
#     1.79x/2.55x (LN/RMS) while 1024 rows measured 0.60x/0.68x. Every
#     shape at >= 16384 rows with hidden size < 8192 won (1.03x-2.55x);
#     rows <= 4096 was a loss or a wash (0.60x-1.09x) whatever the size.
#   * ...but a WIDE hidden size loses even with plenty of rows:
#     16384x8192 measured 1.04x LN and 0.79x RMS, while 16384x5120 gets
#     1.09x/1.25x and 16384x4096 gets 1.16x/1.25x. Hence the column cap as
#     well as the row floor.
#   * NORMS in inference lose almost everywhere (forward alone measured
#     0.45x-1.29x), so the gate also requires a live backward pass. This
#     has to be a runtime check rather than an install-time decision: one
#     process trains at 16k+ tokens and decodes at 1 token through the same
#     patched F.rms_norm.
#
#   * ACTIVATIONS (gelu, silu) never win: 0.87x-0.99x fwd+bwd and as low as
#     0.68x forward, across fp16 and bf16. They are pure bandwidth-bound
#     elementwise ops, so there is nothing for TE to fuse that stock ROCm
#     is not already doing. Not installed by default.
#
#   * ATTENTION only wins where STOCK collapses, never by being fast: TE's
#     fused CK/AOTriton attention is compiled out for gfx1100 (setup.py
#     gates it to gfx942/gfx950) and the unfused path runs instead. With
#     is_causal=True or no mask, PyTorch keeps its flash backend and wins
#     (0.20x-1.07x). With an explicit boolean mask,
#     torch.backends.cuda.can_use_flash_attention() returns False, PyTorch
#     drops to the mem-efficient backend, and TE wins -- but only once the
#     work (batch * heads * q_len * kv_len) clears TE's fixed overhead:
#     >= 16.8M measured 1.37x-1.87x, 8.4M measured 0.90x-0.95x, and it
#     degrades smoothly to 0.18x on tiny shapes. The threshold is work, NOT
#     sequence length: B=1,H=32,S=512 (8.4M) loses at 0.90x while
#     B=4,H=16,S=512 (16.8M) wins at 1.60x with the same S.
#     Forward alone never wins at any size.
#
# Each threshold is overridable per-process, same convention as the
# FFTCONV_* knobs above; re-measure with tools/bench_te_ops.py after
# changing GPU, ROCm, or TE build.
_TE_NORM_MIN_ROWS = int(os.environ.get("AMD_TUNED_TORCH_TE_NORM_MIN_ROWS", "16384"))
# Hidden sizes at or above this lose even on tall inputs (see 16384x8192
# above). Exclusive bound: 5120 is fine, 8192 is not.
_TE_NORM_MAX_COLS = int(os.environ.get("AMD_TUNED_TORCH_TE_NORM_MAX_COLS", "8192"))
# Route norms to TE for forward-only calls too. Default off: forward-only
# measured 0.45x-1.29x (only one shape above 1.0).
_TE_NORM_INFERENCE = os.environ.get("AMD_TUNED_TORCH_TE_NORM_INFERENCE", "0") == "1"
# RMSNorm via TE's ROCm Triton kernel instead of the hipified one, for
# training calls with >= _TE_NORM_MIN_ROWS rows and a hidden size in
# [_TE_RMS_TRITON_MIN_COLS, _TE_RMS_TRITON_MAX_COLS] (inclusive). Measured
# fwd+bwd vs stock (tools/bench_te_triton_norms.py, 16384 rows unless noted):
#   hidden 2048: hip 1.38-1.40x, triton 0.70-0.72x   -> hip
#   hidden 3072: hip 1.28-1.30x, triton 1.12-1.29x   -> hip
#   hidden 4096: hip 0.87-1.20x, triton 1.21-1.33x   -> triton
#   hidden 5120: hip 1.17-1.22x, triton 1.48-1.51x   -> triton
#   hidden 8192: hip 0.71-1.18x, triton 1.48-1.49x   -> triton (hip was
#                excluded by _TE_NORM_MAX_COLS, so this was stock before)
#   32768x4096:  hip 1.08-1.53x, triton 1.82x
# Small/inference inputs lose badly on Triton (0.17-0.33x: Python launch
# overhead in TE's wrapper). TE's Triton LayerNorm was mixed, so unused.
_TE_RMS_TRITON = os.environ.get("AMD_TUNED_TORCH_TE_RMS_TRITON", "1") != "0"
_TE_RMS_TRITON_MIN_COLS = int(os.environ.get("AMD_TUNED_TORCH_TE_RMS_TRITON_MIN_COLS", "3072"))
_TE_RMS_TRITON_MAX_COLS = int(os.environ.get("AMD_TUNED_TORCH_TE_RMS_TRITON_MAX_COLS", "8192"))
# Install the gelu/silu patches anyway (e.g. to re-measure them after a TE
# rebuild). Default off -- see ACTIVATIONS above.
_TE_ACTIVATIONS = os.environ.get("AMD_TUNED_TORCH_TE_ACTIVATIONS", "0") == "1"
# batch * heads * q_len * kv_len, i.e. the size of the attention score
# matrix TE's unfused path materializes. 2**24 = 16777216.
_TE_ATTN_MIN_WORK = int(os.environ.get("AMD_TUNED_TORCH_TE_ATTN_MIN_WORK", "16777216"))
# Route is_causal=True / unmasked attention to TE as well. Default off:
# that is precisely where PyTorch's own flash backend is available and
# faster. Only the explicit-boolean-mask path (where flash is disqualified)
# is a win.
_TE_ATTN_UNMASKED = os.environ.get("AMD_TUNED_TORCH_TE_ATTN_UNMASKED", "0") == "1"
# Re-swept 2026-11-06 on torch 2.15.0.dev20260920+rocm7.14
# (tools/sweep_te_thresholds.py -> analyse/benchmarks/sweep_attn.csv): with the explicit
# bottom-right mask, TE wins only at head_dim 128 (1.23-1.71x fwd+bwd, every
# dtype, work >= 16.8M); at head_dim 64 it lost almost everywhere
# (0.47-0.81x, one 1.0-1.13x pocket) -- the old work-only rule routed those.
_TE_ATTN_MIN_HEAD_DIM = int(os.environ.get("AMD_TUNED_TORCH_TE_ATTN_MIN_HEAD_DIM", "128"))
# fp32 with NO mask (and not is_causal): stock has no flash path in fp32 and
# TE wins 1.53-2.05x at head_dim 128 / work >= 16.8M. fp16/bf16 unmasked is a
# tie (0.89-1.10x) and stays stock. Set to 0 to keep fp32 unmasked on stock.
_TE_ATTN_FP32_UNMASKED = os.environ.get("AMD_TUNED_TORCH_TE_ATTN_FP32_UNMASKED", "1") != "0"

# te_extra_ops kernels installed over stock by default, measured by
# tools/bench_te_extra_ops.py (analyse/benchmarks/te_extra_ops_bench.txt):
#   Adam/AdamW step   foreach default path only (fused=True stays stock)
#   grad-norm (clip)  1.2-7.5x vs get_total_norm, and fp32-accurate for bf16 grads
#   dropout           1.24-1.32x fwd+bwd, 8x smaller saved mask -- exact-p only
# NOT installed: softmax. Its 1.5-3x win was over softmax(x*scale)+masked_fill,
# i.e. fusion an F.softmax patch never sees; against plain F.softmax it measured
# 0.56-0.88x outside s=1024-2048 (1.18-1.23x inside). SGD (0.87-0.90x) and
# multi_tensor_scale (0.94-1.00x) lose outright. Set any of these to 0 to turn
# that patch off.
_TE_ADAM = os.environ.get("AMD_TUNED_TORCH_TE_ADAM", "1") != "0"
_TE_GRAD_NORM = os.environ.get("AMD_TUNED_TORCH_TE_GRAD_NORM", "1") != "0"
_TE_DROPOUT = os.environ.get("AMD_TUNED_TORCH_TE_DROPOUT", "1") != "0"

# ---------------------------------------------------------------------------
# comfy-kitchen (third_party/comfy-kitchen), via comfy_kitchen_ops.py.
#
# Read that module's docstring first: it explains what the tier is, why every
# entry point declines by returning None instead of falling back the way the
# upstream library does, and how to build the HIP extension. What belongs HERE
# is only the two things the contests in this file need to know about it.
#
# 1. THE CONTEST CACHE HAS TO BE PARTITIONED. kernel_select stores one winner
#    name per shape key and the key does NOT include the candidate set (see
#    kernel_select._contest). So adding a candidate to linear/conv3d does
#    nothing at all on a machine with a warm _kernel_select_cache: every
#    already-measured shape goes straight to its cached winner and the new
#    candidate is never timed. The tier would look dead and the cache would be
#    the reason, which is a miserable thing to debug.
#
#    Putting the tier's presence in the `kind` -- the first component of every
#    kernel_select key -- fixes that in the direction that costs nothing: with
#    the extension present, linear/conv3d shapes are re-contested once under
#    the new kind; without it, the kind is unchanged and every existing cached
#    decision is used exactly as before. Availability is fixed for the life of
#    the process (the extension loads at import), so this is resolved once.
#
# 2. THE CANDIDATES ARE JUDGED AT THE ORDINARY BAR. comfy-kitchen's public API
#    documents fp16_linear/fp16_conv3d as carrying
#    torch.backends.cuda.matmul.allow_fp16_accumulation's numerics, which would
#    have needed a widened verification tolerance to be usable at all. It is
#    true of the CUTLASS path those functions were written for and NOT of the
#    HIP kernels reached here, which say so in their own sources ("the
#    accumulators are fp32", ops/gemm_fp16.hip; "Accumulation is fp32",
#    ops/conv3d_fp16.hip). They are fp32-accumulate WMMA kernels on a par with
#    stock, so they are contested at kernel_select's per-dtype default like
#    every other candidate -- which also means verification, not a comment,
#    is what enforces this if a later build ever changes it.
_COMFY = comfy_kitchen_ops.available()
# Appended to the kernel_select `kind` for the two ops whose candidate list
# this tier changes -- see (1) above.
_COMFY_KIND = ".comfy" if _COMFY else ""
_LINEAR_KIND = "linear" + _COMFY_KIND
_CONV3D_KIND = "conv3d" + _COMFY_KIND

_ORIGINALS: dict[tuple[Any, str], Callable] = {}
_ENABLED = False

# Resolved once at import time, not on every call -- _grad_safe runs on
# every single patched op invocation (it's the first check in the hottest
# of hot paths: F.linear/matmul/bmm/conv2d/conv3d/group_norm), so re-doing
# a hasattr() lookup on the torch module there for the life of the process
# is a pure waste.
_HAS_INFERENCE_MODE = hasattr(torch, "is_inference_mode_enabled")


def _is_compiling() -> bool:
    """True while Dynamo is tracing. Guarded because the API moved across
    PyTorch versions and this must never raise on the hot path."""
    try:
        return bool(torch.compiler.is_compiling())
    except AttributeError:
        try:
            return bool(torch._dynamo.is_compiling())
        except Exception:
            return False


def _grad_safe(*tensors: Any) -> bool:
    """True if a kernel with no autograd support of its own can be called
    directly -- i.e. nothing will ever need to backprop through it.

    This used to decide whether an accelerated tier ran AT ALL. It no
    longer does for any always-on tier: linear/matmul/bmm (_LinearFn/
    _BmmFn), conv2d (_Conv2dFn), conv3d (_Conv3dFn) and group_norm
    (_GroupNormFn) all have autograd wrappers now, so False just routes
    the same accelerated forward through the wrapper instead of calling
    it directly. Returning False is a slightly slower path (one
    autograd.Function of overhead, plus whatever the backward costs), not
    a decline.

    Where it IS still a decline: the opt-in tiers that have no wrapper --
    linear_int8 and conv3d_winograd_fp16 in _opt_in_tiers.py, the
    triton_kernels rmsnorm tier, and miopen_fallback's sparse conv1d fast
    path. The TE-backed ops never needed the check at all (real backward
    passes of their own)."""
    if _HAS_INFERENCE_MODE and torch.is_inference_mode_enabled():
        return True
    if not torch.is_grad_enabled():
        return True
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.requires_grad:
            return False
    return True


def _te_grad_live(*tensors: Any) -> bool:
    """True when this call will actually build a backward pass.

    The exact inverse of _grad_safe, and deliberately expressed as such so
    the two can never drift apart on the inference_mode/no_grad edge cases.
    The TE-backed ops need it for the opposite reason the others need
    _grad_safe: they all HAVE a backward, but their measured win over stock
    lives almost entirely IN that backward (see the _TE_* block above), so
    the question here is not "is dropping autograd safe" but "is there a
    backward to win in at all".
    """
    return not _grad_safe(*tensors)


def _te_norm_worth_it(input_: Any, weight: Any) -> bool:
    """Back-compat: True when a LayerNorm call would route to TE."""
    return _te_norm_route("layer", input_, weight) is not None


def _te_norm_route(kind: str, input_: Any, weight: Any):
    """Which TE norm backend wins for this call: "triton", "hip", or None
    (stock). Re-derived 2026-11-06 on torch rocm7.14 from a rows x hidden x
    dtype sweep (tools/sweep_te_thresholds.py -> analyse/benchmarks/sweep_norm.csv),
    fwd+bwd vs stock. R = _TE_NORM_MIN_ROWS (16384):

      RMSNorm   rows >= R,   3072 <= hidden <= 8192    triton 1.36-1.90x
                rows >= R/2, 5120 <= hidden <= 8192    triton 1.43-1.53x (16-bit)
                rows >= R,   hidden < 3072             hip    1.19-1.98x
                rows >= R/2, 1024 <= hidden < 5120     hip    1.01-1.58x
      LayerNorm rows >= 2R,  hidden <= 8192            hip    1.16-2.04x
                rows >= R,   hidden < 3072             hip    1.12-1.71x
                fp32, rows >= R/2, hidden >= 2048      hip    1.16-1.29x
                (16-bit at R rows, hidden 3072-8192: 0.94-1.17x -> stock)

    Forward-only calls stay on stock unless _TE_NORM_INFERENCE (hip forward
    measured 0.86-0.95x at most large shapes). Hidden sizes above
    _TE_NORM_MAX_COLS (8192, now inclusive) were not measured -> stock.
    """
    if not (_TE_NORM_INFERENCE or _te_grad_live(input_, weight)):
        return None
    cols = weight.numel()
    if cols == 0 or cols > _TE_NORM_MAX_COLS:
        return None
    rows = input_.numel() // cols
    R = _TE_NORM_MIN_ROWS
    if kind == "rms":
        if (_TE_RMS_TRITON and cols <= _TE_RMS_TRITON_MAX_COLS
                and ((rows >= R and cols >= _TE_RMS_TRITON_MIN_COLS)
                     or (rows >= R // 2 and cols >= 5120))
                and te_ops.triton_rms_norm_available()):
            return "triton"
        if rows >= R and cols < _TE_NORM_MAX_COLS:
            return "hip"
        if rows >= R // 2 and 1024 <= cols < 5120:
            return "hip"
        return None
    if rows >= 2 * R:
        return "hip"
    if rows >= R and cols < 3072:
        return "hip"
    if input_.dtype == torch.float32 and rows >= R // 2 and cols >= 2048:
        return "hip"
    return None


# Types this package's kernels can actually be handed. Every accelerated
# path below eventually reaches a HIP kernel or a torch.library custom op
# that reads a dense tensor's storage directly, so the argument has to BE a
# dense tensor -- not something that merely quacks like one.
#
# torch.Tensor SUBCLASSES do not qualify, and `isinstance` cannot tell the
# difference. A quantized weight (optimum.quanto's QBytesTensor, as reached
# through mmgp's quant_router) is a Tensor subclass that reports
# `.is_cuda == True` and a `.dtype` of float16 -- the DEQUANTIZED dtype --
# so it passed the old isinstance/dtype gate untouched and was handed
# straight to compile_ops.ck_gemm_linear. What happens then is not a clean
# decline: the custom op dispatches into quanto's __torch_dispatch__, which
# does not recognize it, falls back to dequantizing every argument, and the
# whole thing ends in an illegal memory access rather than an exception any
# of the fallback paths here could have caught.
#
# Declining sends the call to the stock op, which is exactly where such a
# tensor wants to go: quanto's own __torch_function__ intercepts F.linear
# and runs its proper quantized path. The same reasoning covers every other
# subclass with custom dispatch semantics -- DTensor, FakeTensor, functorch
# wrappers -- so the test is an exact type match rather than a blocklist of
# the ones seen so far.
_PLAIN_TENSOR_TYPES = (torch.Tensor, torch.nn.Parameter)


def _is_plain_tensor(t: Any) -> bool:
    """True only for a dense tensor this package's kernels can read."""
    return type(t) in _PLAIN_TENSOR_TYPES


def _usable(*tensors: Any, dtypes: tuple = _GEMM_DTYPES) -> bool:
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            continue
        if not _is_plain_tensor(t):
            return False  # a Tensor subclass -- see _PLAIN_TENSOR_TYPES above
        if not t.is_cuda or t.dtype not in dtypes:
            return False
    return True


def _install(target: Any, name: str, wrapper: Callable) -> None:
    key = (target, name)
    if key not in _ORIGINALS:
        _ORIGINALS[key] = getattr(target, name)
    setattr(target, name, wrapper)


def _restore(target: Any, name: str) -> None:
    original = _ORIGINALS.pop((target, name), None)
    if original is not None:
        setattr(target, name, original)


# Small standalone helper, used by _patched_conv3d below (sparse-voxel
# fast path's stride/padding/dilation normalization) and by the conv3d
# Winograd opt-in tier further down this package -- kept here, with the
# other small dispatch helpers, rather than moved with the Winograd code
# it was originally adjacent to, since a name only defined in the opt-in
# module would leave this file's own _patched_conv3d importing from a
# module that in turn imports helpers back from this one.
def _triple(v):
    return (v, v, v) if isinstance(v, int) else tuple(v)


# ---------------------------------------------------------------------------
# linear/matmul/bmm/conv2d: aiter Triton kernels (no autograd -- grad-safety
# checked up front, plus a RuntimeError/AssertionError fallback for
# unsupported shapes).
# ---------------------------------------------------------------------------
#
# TRAINING-TIME GAP. _grad_safe() falling back to stock the instant any
# input requires_grad was, for a long time, assumed to only matter for
# occasional no_grad-adjacent calls. Measured on a real SDXL LoRA training
# run (MIOpen verbose log, gfx1100, character-mode LoKr on to_q/to_k/to_v),
# it is not occasional: once ANY adapter anywhere in the model needs a
# gradient, autograd must be able to backprop through every frozen layer
# between it and the loss, so nearly every activation in the whole forward
# pass ends up requires_grad=True. _grad_safe() then declines for
# essentially every linear/matmul/bmm/conv2d call in the network, not a few
# of them -- this tier was, in practice, inference-only despite nothing
# about it being conceptually limited to inference.
#
# _LinearFn/_BmmFn below close that gap for linear/matmul/bmm specifically
# (matmul reuses _BmmFn -- _patched_matmul already reshapes every shape
# family it accepts down to a bmm problem before the contest even runs, see
# its own docstring, so its backward is exactly _BmmFn's too, composed with
# the reshape/unsqueeze ops' own already-correct stock backward) -- conv2d's
# backward is not reducible to calling conv2d again the same simple way,
# see _patched_conv2d's own comment -- by wrapping the
# existing accelerated forward candidates in a torch.autograd.Function
# whose backward is the textbook GEMM derivative (dA = dY @ B^T, dB =
# A^T @ dY), computed with plain `@`. That's it -- no new kernel, no new
# numerically-unverified code path: autograd disables grad-tracking during
# a Function's own backward() (nothing here calls torch.enable_grad()), so
# _grad_safe() itself reports those backward matmuls as safe too, meaning
# this module's own linear/matmul/bmm patches transparently accelerate the
# backward pass as well, through the exact same kernel_select contest as
# the forward pass. Forward is the only place a kernel *choice* is made;
# backward is just correct math using operators this project didn't write.
#
# THE GAP IS NOW CLOSED FOR EVERY ALWAYS-ON TIER. conv2d followed
# (_Conv2dFn, routing backward-data/backward-weight through CK's own
# device ops), then the last two: _Conv3dFn (accelerated forward, stock
# aten.convolution_backward for the backward -- there is no CK conv3d
# backward to contest) and _GroupNormFn (accelerated forward, GroupNorm's
# textbook derivative in plain torch ops, same playbook as _LinearFn).
# group_norm was the one that mattered most and the one this note used to
# call a dead end: SDXL's UNet and VAE are almost entirely
# GroupNorm-normalised ResNet blocks, and it was declining on every
# single call during training. What is still forward-only is the opt-in
# tiers -- see _grad_safe's docstring for that list.


class _LinearFn(torch.autograd.Function):
    """Makes _patched_linear's accelerated forward usable when a gradient
    is needed -- see the TRAINING-TIME GAP note above for the full
    rationale. fwd_fn is _patched_linear's own accelerated-candidate
    selection, called unchanged; only the backward is new here."""

    @staticmethod
    def forward(ctx, input, weight, bias, fwd_fn):
        ctx.save_for_backward(input, weight)
        ctx.has_bias = bias is not None
        return fwd_fn(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        # F.linear itself accepts any input.dim() >= 1 (batched over every
        # leading dim) but the GEMM math is 2D -- flatten every leading dim
        # into one, matching how _patched_linear's own kernel_select key
        # already treats "input.numel() // input.size(-1)" as the M dim.
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        input_2d = input.reshape(-1, input.shape[-1])
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = (grad_output_2d @ weight).reshape(input.shape)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output_2d.t() @ input_2d
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output_2d.sum(0)
        return grad_input, grad_weight, grad_bias, None


class _BmmFn(torch.autograd.Function):
    """Makes _patched_bmm's (and, via _patched_matmul, its reshaped-to-3D
    reuse of the same contest) accelerated forward usable when a gradient
    is needed. Y = A @ B, both always exactly 3D (batch, M, K) @ (batch, K,
    N) by the time this is called -- _patched_matmul reshapes every shape
    family it accepts (2D, 3D, >=4D) down to this via ordinary
    unsqueeze/reshape before ever reaching here. Those are themselves
    differentiable stock ops, so calling this on their output composes
    correctly with no extra work: autograd chains this Function's backward
    straight into unsqueeze/reshape's own (already correct) backward to
    produce a gradient in the original, un-reshaped shape."""

    @staticmethod
    def forward(ctx, a, b, fwd_fn):
        ctx.save_for_backward(a, b)
        return fwd_fn(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_a = grad_output @ b.transpose(-2, -1)
        if ctx.needs_input_grad[1]:
            grad_b = a.transpose(-2, -1) @ grad_output
        return grad_a, grad_b, None


def _patched_linear(input, weight, bias=None):
    """F.linear over a per-shape contest: stock, hipBLASLt, CK GEMM.

    This op had no contest until hipblaslt_ops and ck_gemm_ops existed,
    because aiter's Triton GEMM -- the only other candidate -- raises
    KeyError('gfx1100'), leaving nothing to compete against stock. It has
    two now, and measurement says neither of them wins outright. Against
    stock (rocBLAS = 1.00x) on RX 7900 XTX, fp16:

                                 hipBLASLt   CK GEMM
        4096^3                     1.29x      1.20x
        8192x4096x11008            1.09x      1.11x
        4096x4096x1024             1.49x      1.47x
        4096x1280x1280             0.96x      1.14x
        1024^3                     1.02x      1.21x
        1x4096x4096 (decode)       0.60x      0.90x

    hipBLASLt owns the large end and collapses on single-token decode --
    0.60x, where its heuristic picks a throughput kernel that never gets to
    use its throughput. CK is flatter and ahead everywhere hipBLASLt is
    behind. Two rows are outright regressions, which is exactly why this
    goes through kernel_select rather than a fixed tier order: stock is an
    ordinary candidate and wins the shapes it deserves to win.

    A fourth candidate, splitk_gemm_ops (see that module's docstring),
    targets exactly the row both existing tiers regress on: single-token/
    small-batch decode, where a plain tiled GEMM has too few output tiles
    to occupy this card's CUs regardless of per-tile throughput. It splits
    the K-reduction across more thread blocks instead, and -- like every
    candidate here -- only ever wins if kernel_select's own numerical
    verification against stock actually agrees; unmeasured on real
    hardware, so it may simply always lose its own contest here, which is
    the safe failure mode this whole module exists to guarantee.
    """
    orig = _ORIGINALS[(F, "linear")]
    if not _usable(input, weight):
        if _DEBUG:
            _debug("linear", "stock", "not usable: dtype/device/tensor-subclass",
                   input=input, weight=weight)
        return orig(input, weight, bias)
    if input.dim() < 2 or weight.dim() != 2:
        if _DEBUG:
            _debug("linear", "stock", "not a 2D-weight GEMM", input=input, weight=weight)
        return orig(input, weight, bias)

    def _accelerated(input, weight, bias):
        if kernel_select.enabled():
            # Keyed on the GEMM's actual identity (M, K, N and dtype), not
            # on the caller's leading dims: [4, 128, K] and [512, K] are one
            # problem to every candidate here, and keying on the raw shape
            # would re-run the contest for each of them.
            key = (input.dtype, int(input.numel() // input.size(-1)),
                   tuple(weight.shape), bias is not None)
            won = kernel_select.cached_key(_LINEAR_KIND, key)
            if won == "stock":
                if _DEBUG:
                    _debug("linear", "stock", "measured winner for this shape", key=key)
                return orig(input, weight, bias)
            if won is None:
                if _DEBUG:
                    _debug("linear", "contest", "first call for this shape", key=key)
                # comfy-kitchen's WMMA GEMM joins the list only where it
                # could possibly serve; it takes fp16 with K > 4096 and
                # declines everything else, and a candidate that always
                # declines is pure contest overhead. Judged at the ordinary
                # per-dtype bar: the HIP kernel accumulates in fp32 (see
                # comfy_kitchen_ops._gemm_enabled), so it has no claim on any
                # extra slack -- and that verification is what would catch it
                # if a later build stopped being fp32-accumulate.
                _comfy_linear = _COMFY and input.dtype is torch.float16
                _cands = [
                    ("hipblaslt", lambda: compile_ops.hipblaslt_linear(
                        input, weight, bias,
                        hipblaslt_ops.EPILOGUE_BIAS if bias is not None
                        else hipblaslt_ops.EPILOGUE_NONE)),
                    ("ck_gemm", lambda: compile_ops.ck_gemm_linear(
                        input, weight, bias,
                        ck_gemm_ops.EPILOGUE_BIAS if bias is not None
                        else ck_gemm_ops.EPILOGUE_NONE)),
                    # Decode-shaped (small-M) candidate only -- see
                    # splitk_gemm_ops.py's module docstring for why
                    # hipBLASLt/CK both regress at M=1 and what this does
                    # about it. Declines (returns None) for any M outside
                    # its target range, same as every other candidate here.
                    ("splitk", lambda: splitk_gemm_ops.linear(input, weight, bias)),
                    ("aiter", lambda: _linear_aiter(input, weight, bias)),
                ]
                if _comfy_linear:
                    _cands.append(
                        ("comfy", lambda: comfy_kitchen_ops.linear(input, weight, bias)))
                _cands.append(("stock", lambda: orig(input, weight, bias)))
                out = kernel_select.pick_key(_LINEAR_KIND, key, _cands)
                if out is not None:
                    if _DEBUG:
                        # Re-read the cache rather than have pick_key report
                        # the name: this runs once per shape, so the extra
                        # lookup is free, and it reports what was actually
                        # CACHED (what every later call will take) rather
                        # than merely what produced this one result.
                        _debug("linear", kernel_select.cached_key(_LINEAR_KIND, key) or "unknown",
                               "contest decided", key=key)
                    return out
            elif won == "hipblaslt":
                # compile_ops.hipblaslt_linear/ck_gemm_linear (not
                # hipblaslt_ops.linear/ck_gemm_ops.linear directly): this is
                # the hot path a cached contest winner takes on every
                # subsequent call, i.e. the actual monkeypatched-F.linear
                # call torch.compile would trace through -- see
                # compile_ops.py's own docstring for why it wraps these two
                # as torch.library custom ops.
                out = compile_ops.hipblaslt_linear(input, weight, bias,
                                           hipblaslt_ops.EPILOGUE_BIAS if bias is not None
                                           else hipblaslt_ops.EPILOGUE_NONE)
                if out is not None:
                    if _DEBUG:
                        _debug("linear", "hipblaslt", "cached winner", key=key)
                    return out
                if _DEBUG:
                    # A cached winner that declines on a later call is worth
                    # a line of its own: the contest is not re-run, so this
                    # silently costs the win on every call from here on.
                    _debug("linear", "declined", "cached winner hipblaslt returned None", key=key)
            elif won == "ck_gemm":
                out = compile_ops.ck_gemm_linear(input, weight, bias,
                                         ck_gemm_ops.EPILOGUE_BIAS if bias is not None
                                         else ck_gemm_ops.EPILOGUE_NONE)
                if out is not None:
                    if _DEBUG:
                        _debug("linear", "ck_gemm", "cached winner", key=key)
                    return out
                if _DEBUG:
                    _debug("linear", "declined", "cached winner ck_gemm returned None", key=key)
            elif won == "splitk":
                out = splitk_gemm_ops.linear(input, weight, bias)
                if out is not None:
                    if _DEBUG:
                        _debug("linear", "splitk", "cached winner", key=key)
                    return out
                if _DEBUG:
                    _debug("linear", "declined", "cached winner splitk returned None", key=key)
            elif won == "comfy":
                out = comfy_kitchen_ops.linear(input, weight, bias)
                if out is not None:
                    if _DEBUG:
                        _debug("linear", "comfy", "cached winner", key=key)
                    return out
                if _DEBUG:
                    # Expected the moment a host app clears
                    # torch.backends.cuda.matmul.allow_fp16_accumulation: this
                    # candidate reads that flag on every call, by design, so
                    # turning fp16 accumulation off takes effect immediately
                    # rather than at the next contest.
                    _debug("linear", "declined", "cached winner comfy returned None", key=key)

        out = _linear_aiter(input, weight, bias)
        if _DEBUG:
            _debug("linear", "stock" if out is None else "aiter",
                   "no contest (kernel_select off, or every candidate declined)",
                   input=input, weight=weight)
        return orig(input, weight, bias) if out is None else out

    if _grad_safe(input, weight, bias):
        return _accelerated(input, weight, bias)
    # A gradient is needed -- route through _LinearFn so the accelerated
    # forward above still runs, with a real (textbook, stock-matmul-based)
    # backward instead of skipping acceleration entirely. See the
    # TRAINING-TIME GAP note above _LinearFn's definition.
    if _DEBUG:
        _debug("linear", "autograd", "gradient needed: accelerated fwd via _LinearFn",
               input=input, weight=weight)
    return _LinearFn.apply(input, weight, bias, _accelerated)


def _linear_aiter(input, weight, bias):
    """The pre-existing aiter path, as a candidate that may decline.

    Returns None rather than raising so it loses the contest instead of
    breaking the call -- which on gfx1100 is what always happens, since
    aiter ships no RDNA3 tuning config.
    """
    try:
        return compile_ops.linear_fp16(input, weight, bias)
    except (RuntimeError, TypeError, AssertionError, KeyError):
        return None


def _patched_matmul(input, other, *, out=None):
    """torch.matmul over the SAME hipBLASLt/aiter/stock contest _patched_bmm
    already uses, for the batched cases matmul reduces to a bmm problem: 2D
    @ 2D, 3D @ 3D, and >=4D @ >=4D where both operands share the exact same
    batch shape (no broadcasting -- neither hipBLASLt's nor aiter's batched
    kernels have broadcast semantics of their own). Every candidate,
    including stock, operates on the reshaped-to-3D view; the result is
    reshaped back to matmul's own shape convention once, after the contest
    has already picked a winner, so the contest itself only ever has to
    reason about one shape family (matching _patched_bmm's own key).

    This used to call aiter directly with no fallback to hipBLASLt/CK at
    all -- the same gap _patched_linear/_patched_bmm closed once hipBLASLt
    and CK GEMM existed as real candidates, just never closed here too.

    STRUCTURED-FACTOR OPERANDS. `other` may also be butterfly or Monarch
    FACTORS (a list/tuple/ParameterList of weight tensors), in which case the
    call is routed to flash_mm_kernel -- see the block comment on that branch
    below for why only the list form is intercepted."""
    orig = _ORIGINALS[(torch, "matmul")]

    # ---- structured factors (butterfly / Monarch) -------------------------
    # flash_mm_kernel.matmul is deliberately spelled like torch.matmul, but
    # nothing routed to it until now: it had to be called by name. This makes
    # `torch.matmul(x, factors)` reach it.
    #
    # ONLY THE LIST/TUPLE FORM IS INTERCEPTED, AND THAT IS NOT AN OVERSIGHT.
    # torch.matmul rejects a list outright today ("argument 'other' must be
    # Tensor, not list"), so treating one as factors cannot change the result
    # of any call that works at present -- it only gives meaning to a call
    # that currently raises. A PRE-STACKED (E, L, 2) factor tensor is left
    # alone on purpose: that is a perfectly valid 3D operand for an ordinary
    # batched matmul, and hijacking it on shape alone would silently change
    # the meaning of real dense calls. Callers holding stacked factors should
    # call flash_mm_kernel.matmul directly.
    #
    # A DENSE `other` CANNOT BE ROUTED HERE AT ALL. Going from a dense (L, L)
    # matrix to butterfly factors is a factorization problem, not a reshape,
    # and most matrices admit no exact butterfly factorization -- so there is
    # no version of this that accelerates ordinary dense torch.matmul calls.
    # The structured path is only ever reachable when the caller already
    # holds its weights factored.
    #
    # is_eligible() is the non-raising precondition check (right factor
    # count, power-of-2 L for butterfly, matching in_dims product for
    # Monarch); anything it declines falls through to `orig`, which raises
    # the same TypeError it always did.
    if isinstance(other, (list, tuple, torch.nn.ParameterList)):
        if isinstance(input, torch.Tensor) and flash_mm_kernel.is_eligible(input, other):
            return flash_mm_kernel.matmul(input, other, out=out)
        return orig(input, other, out=out) if out is not None else orig(input, other)

    # Sparse-input fast path: tests `input`'s LAYOUT (torch.sparse_csr/csc/
    # coo vs. the ordinary torch.strided every other branch below assumes),
    # not its content -- unlike flexgemm_ops's occupancy-gated conv2d/conv3d
    # fast paths, there is no reduction to pay here, a sparse tensor's own
    # layout already says so. See rocsparse_ops.py's module docstring for
    # what's supported (2D CSR/CSC/COO sparse @ 2D dense only) and why.
    # Checked FIRST, before _usable/the reshape-to-3D dance below: those
    # assume a plain strided tensor (.unsqueeze/.reshape on a sparse CSR
    # tensor is not the batching operation they need it to be), so a sparse
    # `input`/`other` must never reach them either way, whether or not
    # rocSPARSE ends up handling the call. Same _grad_safe gate
    # flexgemm_ops's sparse conv fast paths need for the same reason:
    # rocsparse_spmm has no backward, so this only fires when no gradient is
    # needed -- otherwise stock's own (correct, autograd-aware) sparse
    # matmul handles it instead, same as if rocSPARSE were unavailable.
    #
    # rocSPARSE vs. stock is MEASURED, not assumed, exactly like every other
    # tier in this file (conv2d/conv3d's kernel_select contest, fftconv's
    # -- see that section's own comment for why this project doesn't trust
    # "our kernel is a real vendor library" as a substitute for a benchmark
    # here either: ATen's own generic sparse dispatch on ROCm may or may not
    # be the naive fallback rocSPARSE would beat). Keyed on (dtype, both
    # operands' layout, both shapes) rather than shared with the dense
    # (a3.dtype, a3.shape, b3.shape) key the contest below uses -- a sparse
    # `input`'s layout changes which candidates even apply, so it needs its
    # own kernel_select "kind" and key shape rather than colliding with the
    # dense matmul contest's cache entries.
    if out is None and isinstance(input, torch.Tensor) and isinstance(other, torch.Tensor) and (
            input.layout != torch.strided or other.layout != torch.strided):
        if _grad_safe(input, other):
            if kernel_select.enabled():
                key = (input.dtype, str(input.layout), str(other.layout),
                       tuple(input.shape), tuple(other.shape))
                won = kernel_select.cached_key("matmul_sparse", key)
                if won is None:
                    if _DEBUG:
                        _debug("matmul_sparse", "contest", "first call for this shape", key=key)
                    _out = kernel_select.pick_key("matmul_sparse", key, [
                        ("rocsparse", lambda: rocsparse_ops.maybe_spmm(input, other)),
                        ("stock", lambda: orig(input, other)),
                    ])
                    if _out is not None:
                        if _DEBUG:
                            _debug("matmul_sparse",
                                   kernel_select.cached_key("matmul_sparse", key) or "unknown",
                                   "contest decided", key=key)
                        return _out
                elif won == "rocsparse":
                    _out = rocsparse_ops.maybe_spmm(input, other)
                    if _out is not None:
                        if _DEBUG:
                            _debug("matmul_sparse", "rocsparse", "cached winner", key=key)
                        return _out
                # won == "stock", or a cached "rocsparse" winner declined this
                # call (e.g. AMD_TUNED_TORCH_ROCSPARSE_SPMM=0 toggled off since
                # it was cached) -- either way, stock below is always correct.
            else:
                _sparse_out = rocsparse_ops.maybe_spmm(input, other)
                if _sparse_out is not None:
                    if _DEBUG:
                        _debug("matmul_sparse", "rocsparse", "no contest (kernel_select off)",
                               input=input, other=other)
                    return _sparse_out
        if _DEBUG:
            # Covers all three ways a sparse operand ends up on stock: no
            # grad-safety, rocSPARSE declining the layout/dtype, and a
            # contest that measured stock as the winner.
            _debug("matmul_sparse", "stock", "rocSPARSE declined or lost",
                   input=input, other=other)
        return orig(input, other)

    if out is not None or not _usable(input, other):
        if _DEBUG:
            _debug("matmul", "stock",
                   "out= given" if out is not None else "not usable: dtype/device/tensor-subclass",
                   input=input, other=other)
        return orig(input, other) if out is None else orig(input, other, out=out)
    if input.dtype != other.dtype:
        if _DEBUG:
            _debug("matmul", "stock", "mixed dtypes", input=input, other=other)
        return orig(input, other)

    if input.dim() == 2 and other.dim() == 2:
        a3, b3 = input.unsqueeze(0), other.unsqueeze(0)
        def _reshape_back(o3): return o3.squeeze(0)
    elif input.dim() == 3 and other.dim() == 3:
        a3, b3 = input, other
        def _reshape_back(o3): return o3
    elif input.dim() == other.dim() >= 4 and input.shape[:-2] == other.shape[:-2]:
        batch_shape = input.shape[:-2]
        a3 = input.reshape(-1, *input.shape[-2:])
        b3 = other.reshape(-1, *other.shape[-2:])
        def _reshape_back(o3): return o3.view(*batch_shape, *o3.shape[-2:])
    else:
        if _DEBUG:
            # Broadcast batch dims, or a 1D operand: no candidate here has
            # broadcast semantics, so this shape family never enters the
            # contest at all -- see this function's docstring.
            _debug("matmul", "stock", "shape does not reduce to a bmm",
                   input=input, other=other)
        return orig(input, other)

    # matmul on the reshaped 3D tensors is bmm's own semantics exactly (no
    # broadcasting possible once both operands are forced to the same
    # leading batch dim), so `orig` applied here is a safe, cheap stock
    # reference -- no need to reach for a separate "true" matmul call on
    # the un-reshaped tensors.
    def _stock3(a3, b3): return orig(a3, b3)

    def _accelerated(a3, b3):
        if kernel_select.enabled():
            key = (a3.dtype, tuple(a3.shape), tuple(b3.shape))
            won = kernel_select.cached_key("matmul", key)
            if won == "stock":
                if _DEBUG:
                    _debug("matmul", "stock", "measured winner for this shape", key=key)
                return _stock3(a3, b3)
            if won == "hipblaslt":
                # compile_ops.hipblaslt_bmm, not hipblaslt_ops.bmm directly
                # -- same reasoning as _patched_linear's cached-winner
                # branches above.
                got = compile_ops.hipblaslt_bmm(a3, b3)
                if got is not None:
                    if _DEBUG:
                        _debug("matmul", "hipblaslt", "cached winner", key=key)
                    return got
                if _DEBUG:
                    _debug("matmul", "declined", "cached winner hipblaslt returned None", key=key)
            elif won is None:
                if _DEBUG:
                    _debug("matmul", "contest", "first call for this shape", key=key)
                got = kernel_select.pick_key("matmul", key, [
                    ("hipblaslt", lambda: compile_ops.hipblaslt_bmm(a3, b3)),
                    ("aiter", lambda: _bmm_aiter(a3, b3)),
                    ("stock", lambda: _stock3(a3, b3)),
                ])
                if got is not None:
                    if _DEBUG:
                        _debug("matmul", kernel_select.cached_key("matmul", key) or "unknown",
                               "contest decided", key=key)
                    return got

        got = _bmm_aiter(a3, b3)
        if _DEBUG:
            _debug("matmul", "stock" if got is None else "aiter",
                   "no contest (kernel_select off, or every candidate declined)",
                   input=a3, other=b3)
        return _stock3(a3, b3) if got is None else got

    if _grad_safe(input, other):
        return _reshape_back(_accelerated(a3, b3))
    # See _LinearFn/_patched_linear's identical reasoning. a3/b3 are
    # themselves ordinary autograd-tracked views of input/other (unsqueeze
    # or reshape), so _BmmFn's backward composes automatically with their
    # own (stock, already-correct) backward -- nothing extra needed for the
    # 2D/>=4D reshape cases here.
    if _DEBUG:
        _debug("matmul", "autograd", "gradient needed: accelerated fwd via _BmmFn",
               input=input, other=other)
    return _reshape_back(_BmmFn.apply(a3, b3, _accelerated))


def _patched_bmm(input, mat2, *, out=None):
    """torch.bmm over a per-shape contest: hipBLASLt, aiter, stock.

    Same reasoning as _patched_linear -- hipBLASLt is a real candidate on
    this card where aiter is not -- but with one candidate fewer: the CK
    GEMM tier compiles no batched instances. That was a build-cost decision
    (see src/cuda/ck_gemm_fwd.hpp), taken because hipBLASLt already covers
    batched shapes well; the attention-shaped GEMM it exists for measures
    1.49x fp16 / 1.73x bf16 against stock.
    """
    orig = _ORIGINALS[(torch, "bmm")]
    if out is not None or not _usable(input, mat2):
        if _DEBUG:
            _debug("bmm", "stock",
                   "out= given" if out is not None else "not usable: dtype/device/tensor-subclass",
                   input=input, mat2=mat2)
        return orig(input, mat2) if out is None else orig(input, mat2, out=out)
    if input.dtype != mat2.dtype:
        if _DEBUG:
            _debug("bmm", "stock", "mixed dtypes", input=input, mat2=mat2)
        return orig(input, mat2)

    def _accelerated(input, mat2):
        if kernel_select.enabled():
            key = (input.dtype, tuple(input.shape), tuple(mat2.shape))
            won = kernel_select.cached_key("bmm", key)
            if won == "stock":
                if _DEBUG:
                    _debug("bmm", "stock", "measured winner for this shape", key=key)
                return orig(input, mat2)
            if won == "hipblaslt":
                # compile_ops.hipblaslt_bmm, not hipblaslt_ops.bmm directly
                # -- same reasoning as _patched_linear's cached-winner
                # branches above.
                got = compile_ops.hipblaslt_bmm(input, mat2)
                if got is not None:
                    if _DEBUG:
                        _debug("bmm", "hipblaslt", "cached winner", key=key)
                    return got
                if _DEBUG:
                    _debug("bmm", "declined", "cached winner hipblaslt returned None", key=key)
            elif won is None:
                if _DEBUG:
                    _debug("bmm", "contest", "first call for this shape", key=key)
                got = kernel_select.pick_key("bmm", key, [
                    ("hipblaslt", lambda: compile_ops.hipblaslt_bmm(input, mat2)),
                    ("aiter", lambda: _bmm_aiter(input, mat2)),
                    ("stock", lambda: orig(input, mat2)),
                ])
                if got is not None:
                    if _DEBUG:
                        _debug("bmm", kernel_select.cached_key("bmm", key) or "unknown",
                               "contest decided", key=key)
                    return got

        got = _bmm_aiter(input, mat2)
        if _DEBUG:
            _debug("bmm", "stock" if got is None else "aiter",
                   "no contest (kernel_select off, or every candidate declined)",
                   input=input, mat2=mat2)
        return orig(input, mat2) if got is None else got

    if _grad_safe(input, mat2):
        return _accelerated(input, mat2)
    # See _LinearFn/_patched_linear's identical reasoning -- _BmmFn's
    # backward is the textbook bmm derivative, computed with plain `@`.
    if _DEBUG:
        _debug("bmm", "autograd", "gradient needed: accelerated fwd via _BmmFn",
               input=input, mat2=mat2)
    return _BmmFn.apply(input, mat2, _accelerated)


def _bmm_aiter(input, mat2):
    """The pre-existing aiter path, as a candidate that may decline."""
    try:
        return compile_ops.bmm_fp16(input, mat2)
    except (RuntimeError, TypeError, AssertionError, KeyError):
        return None


# einsum: fp32 too -- stock einsum's own planning loses even where torch.bmm
# itself stays stock (measured 1.32x fp32 64x256x256 bmm-shaped equation).
_EINSUM_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# (equation, dtype, *shapes) -> "stock" | "bmm_plan"; only signatures whose
# decision is final (planner declined, or kernel_select decided). Cleared by
# einsum_memo_clear(), e.g. after kernel_select.reset().
_EINSUM_MEMO: dict = {}


def einsum_memo_clear() -> None:
    _EINSUM_MEMO.clear()


def _patched_einsum(*args, **kwargs):
    """torch.einsum planned as batched matmuls (einsum_ops.py), contested
    against stock einsum once per (equation, shapes, dtype).

    Stock einsum lowers to rocBLAS bmm inside C++, where _patched_bmm's
    hipBLASLt/aiter/stock contest never sees it; the rewrite calls the
    patched torch.bmm, so the per-shape bmm winner runs. Measured on the
    7900 XTX (CUDA events, rewrite vs stock): bhqk,bhkd->bhqd fp16
    2x10x1024x1024 1.70x, bij,bjk->bik fp32 64x256x256 1.32x, ij,jk->ik bf16
    2048^2 1.17x, the rest within 1 %. kernel_select keeps stock wherever the
    rewrite loses or disagrees numerically.

    Falls through to stock for: sublist format / keyword arguments, mixed
    dtypes, non-GPU or unsupported dtypes, a live autograd graph (the
    rewrite is differentiable, but contesting under autograd would build
    two graphs), torch.compile tracing, and equations the planner declines
    (ellipsis, diagonals -- einsum_ops.supported)."""
    orig = _ORIGINALS[(torch, "einsum")]
    if kwargs or len(args) < 2 or args[0].__class__ is not str:
        return orig(*args, **kwargs)
    eq = args[0]
    ops = args[1:]
    if len(ops) == 1 and isinstance(ops[0], (list, tuple)):
        ops = tuple(ops[0])
    try:
        sig = (eq, ops[0].dtype) + tuple(t.shape for t in ops)
    except (AttributeError, IndexError):
        return orig(*args)
    # Hot path: decision already made for this exact signature. Eligibility
    # that depends on the call, not the signature (device, grad, compile)
    # is still checked; the per-signature work (parse, key building,
    # kernel_select lookup) is not repeated -- it cost ~20-30 us per call on
    # this host, as much as a small einsum itself.
    won = _EINSUM_MEMO.get(sig)
    if won is not None:
        if (won == "stock" or not ops[0].is_cuda or not _grad_safe(*ops) or _is_compiling()):
            return orig(*args)
        return einsum_ops.einsum(eq, ops)
    if (not all(isinstance(t, torch.Tensor) for t in ops)
            or not _usable(*ops, dtypes=_EINSUM_DTYPES) or len({t.dtype for t in ops}) != 1
            or not _grad_safe(*ops) or _is_compiling() or not kernel_select.enabled()):
        return orig(*args)
    why = einsum_ops.supported(eq, ops)
    if why is not None:
        if _DEBUG:
            _debug("einsum", "stock", "planner declined: " + why)
        _EINSUM_MEMO[sig] = "stock"
        return orig(*args)
    key = (ops[0].dtype, eq.replace(" ", ""), tuple(tuple(t.shape) for t in ops))
    won = kernel_select.cached_key("einsum", key)
    if won in ("stock", "bmm_plan"):
        _EINSUM_MEMO[sig] = won
        return orig(*args) if won == "stock" else einsum_ops.einsum(eq, ops)
    out = kernel_select.pick_key("einsum", key, [
        ("bmm_plan", lambda: einsum_ops.einsum(eq, ops)),
        ("stock", lambda: orig(*args)),
    ])
    won = kernel_select.cached_key("einsum", key)
    if won in ("stock", "bmm_plan"):
        _EINSUM_MEMO[sig] = won
    if _DEBUG:
        _debug("einsum", won or "unknown", "contest decided", key=key)
    return orig(*args) if out is None else out

def _is_pointwise_conv2d(weight, stride, padding, dilation) -> bool:
    """True for a 1x1/stride1/pad0/dilation1 conv2d -- a pure per-pixel
    channel-mixing GEMM (Y = W @ X over channels) with no spatial reduction
    at all, unlike every other conv2d shape this module handles.

    Measured directly on RX 7900 XTX (see miopen_amd_log.txt): for a
    512->256ch, 1024x512 fp16 1x1 conv, MIOpen's own rocBLAS strided-batched
    GEMM solver (GemmFwd1x1_0_1) took 1.69ms, beating its own best Winograd/
    asm-direct kernel (5.74ms, 3.4x slower) and its naive direct kernel
    (759.9ms, 450x slower). rocBLAS wins this by such a wide margin because
    its GEMM primitive consumes the input's existing NCHW strides directly
    (channel-major, spatially-contiguous is already a valid (C_out,C_in) @
    (C_in, H*W) matmul via lda/ldb) with zero data movement -- neither of
    this module's own tiers can match that for K=1: the native im2col-tiled
    kernel below has no spatial window to exploit, and aiter's Triton WMMA
    GEMM is a channels-last kernel that would need an actual NCHW->NHWC
    transpose copy first. So this case is deliberately left on stock rather
    than routed through either tier."""
    return (tuple(weight.shape[2:]) == (1, 1)
            and tuple(aiter_ops._pair(stride)) == (1, 1)
            and tuple(aiter_ops._pair(padding)) == (0, 0)
            and tuple(aiter_ops._pair(dilation)) == (1, 1))


def _fftconv_conv_candidate(input, weight, bias, stride, padding, dilation, groups,
                             *, ndim: int):
    """The FFT-conv tier (amd_tuned_torch.fftconv_ops) as a (name, thunk)
    kernel_select candidate for conv2d/conv3d, or None when this call
    isn't worth contesting it for at all.

    WHY A PRE-FILTER AND NOT JUST ANOTHER CANDIDATE. FFT-conv's
    O(N log N)-per-spatial-dim win only arrives once the kernel is wide
    enough to pay for the transform (see fftconv_ops' own module
    docstring). Measured on gfx1100 against stock, fp32:

      conv2d 4x32x256x256   7x7  1.31ms -> 18.79ms  (14.3x LOSS)
                           15x15  2.50ms -> 16.93ms  ( 6.8x loss)
                           31x31 10.17ms -> 19.59ms  ( 1.9x loss)
                           63x63 38.65ms -> 38.85ms  (even)
      conv3d 1x32x32x64x64   3^3  2.71ms -> 13.26ms  ( 4.9x loss)
                             7^3 33.76ms -> 30.24ms  ( 1.1x win)
                            15^3 6907ms  -> 38.77ms  ( 178x WIN)

    Direct convolution costs K^ndim multiplies per output element while
    the transform costs the same regardless, so 3D crosses over an order
    of magnitude lower in kernel width than 2D -- hence two separate
    thresholds rather than one shared "large kernel" number. Both sit far
    above the 3x3-style kernels this package's other conv tiers exist
    for, and offering the candidate down there would buy nothing while
    costing every distinct small shape one contest measurement plus a
    full padded-transform allocation. Same posture as
    fftconv_ops._FFTCONV1D_MIN_KERNEL on the conv1d path: a cheap
    pre-filter, NOT the win/lose decision, which stays measured per shape.

    None when the tier is switched off (AMD_TUNED_TORCH_FFTCONV2D=0 /
    AMD_TUNED_TORCH_FFTCONV3D=0), `input`/`weight` aren't `ndim`-spatial
    conv tensors, any spatial kernel extent is below the threshold,
    `input` is a small volume (conv3d only -- see
    AMD_TUNED_TORCH_FFTCONV3D_MIN_POSITIONS above this function), `padding`
    is one of F.convNd's string modes ("same"/"valid" -- the contest's
    shape key and fft_conv's own padding handling both want numbers), or
    the dtype/device isn't one this tier covers. The thunk itself returns
    None (declining, per kernel_select.pick's contract) rather than raising
    if the call fails -- including on OOM, which is a
    realistic outcome here specifically: the padded transform is by far
    the most memory-hungry tier in this file (measured ~1.5GB peak on a
    B=8 C=512 L=8192 conv1d whose inputs are 134MB), and a tier that
    can't fit should lose the contest, not kill the process."""
    if ndim == 2:
        if not _FFTCONV_CONV2D_ENABLED:
            return None
        min_kernel, fn = _FFTCONV_CONV2D_MIN_KERNEL, fftconv_ops.fft_conv2d
    else:
        if not _FFTCONV_CONV3D_ENABLED:
            return None
        min_kernel, fn = _FFTCONV_CONV3D_MIN_KERNEL, fftconv_ops.fft_conv3d
    if input.dim() != ndim + 2 or weight.dim() != ndim + 2:
        return None
    if ndim == 3 and input.shape[0] * math.prod(input.shape[2:]) < _FFTCONV_CONV3D_MIN_POSITIONS:
        return None
    if isinstance(padding, str):
        return None
    if min(weight.shape[2:]) < min_kernel:
        return None
    if not _usable(input, weight, dtypes=_FFTCONV_CONV_DTYPES):
        return None

    def _thunk():
        try:
            return fn(input, weight, bias=bias, padding=padding, stride=stride,
                      dilation=dilation, groups=groups)
        except (RuntimeError, ValueError, TypeError):
            return None

    return ("fftconv", _thunk)


def _fftconv_contest_tolerance(fft_candidate, input):
    """kernel_select verification tolerance for a conv contest that has the
    FFT tier in it, or None to keep kernel_select's per-dtype default for
    a contest that doesn't.

    Only the RELATIVE part differs here. FFT-conv doesn't merely round
    differently from direct convolution, it rounds BETTER -- measured
    against a float64 reference on a 1x8x1024 fp32 conv1d, at K=255
    direct is off by 2.6e-4 and FFT by 7.9e-5; at K=1023, 8.6e-4 vs
    1.5e-4 -- but it is a different algorithm, so its disagreement with
    direct convolution is larger than fp32's default rtol of 1e-4 admits
    (fftconv_ops.fftconv_tolerance carries the measured pairs and the
    reasoning). The ABSOLUTE part is no longer this function's problem:
    kernel_select._verify now scales its atol floor by the reference
    output's own RMS, from the reference tensor it already holds, which
    is both more accurate than estimating the output magnitude from the
    operands here and fixes the same defect for every other tier at once
    (see _verify's docstring for the three kernels a purely absolute atol
    was rejecting)."""
    if fft_candidate is None:
        return None
    return fftconv_ops.fftconv_tolerance(input.dtype)


def _conv2d_backward_data_stock(grad_output, weight, input_size, stride, padding, dilation, groups):
    """Reference (always-correct) conv2d backward-data via the same ATen op
    autograd itself would have used had this module never patched
    F.conv2d. `input` is passed as an uninitialized same-shape/dtype/device
    tensor -- grad_input is mathematically independent of the original
    input's VALUES (only grad_output/weight/stride/padding/dilation/groups
    matter), so nothing ever reads it; output_mask requests only this one
    output, matching the convention set by the CK candidate it's contested
    against (see _conv2d_backward_data)."""
    dummy_input = grad_output.new_empty(input_size)
    grad_input, _, _ = torch.ops.aten.convolution_backward(
        grad_output, dummy_input, weight, None,
        list(aiter_ops._pair(stride)), list(aiter_ops._pair(padding)),
        list(aiter_ops._pair(dilation)), False, [0, 0], groups, [True, False, False])
    return grad_input


def _conv2d_backward_weight_stock(input, grad_output, weight_size, stride, padding, dilation, groups):
    """Reference conv2d backward-weight -- see _conv2d_backward_data_stock,
    same reasoning mirrored for the weight-gradient output (dummy_weight's
    values are never read; grad_weight depends only on input/grad_output)."""
    dummy_weight = input.new_empty(weight_size)
    _, grad_weight, _ = torch.ops.aten.convolution_backward(
        grad_output, input, dummy_weight, None,
        list(aiter_ops._pair(stride)), list(aiter_ops._pair(padding)),
        list(aiter_ops._pair(dilation)), False, [0, 0], groups, [False, True, False])
    return grad_weight


def _conv2d_backward_data(grad_output, weight, input_size, stride, padding, dilation, groups):
    """dL/dInput for _Conv2dFn.backward: CK's WMMA backward-data device op
    (ck_ops.conv2d_backward_data) contested against stock via
    kernel_select, same "never assume our kernel wins" policy the forward
    tier already uses (see kernel_select.py's module docstring) -- CK's
    forward WMMA conv already loses to stock's Winograd solver for
    symmetric-channel/stride-1 fp16 shapes and only opens a real gap on the
    asymmetric-channel/stride-2 shapes MIOpen backward falls back to
    im2col+GEMM for (see the TRAINING-TIME GAP comment above
    _patched_conv2d), so a fixed preference either way would be wrong for
    some shape -- exactly what kernel_select measures instead of guessing."""
    stock = lambda: _conv2d_backward_data_stock(
        grad_output, weight, input_size, stride, padding, dilation, groups)
    if groups != 1 or not (ck_ops.available() and _usable(grad_output, weight, dtypes=_CK_CONV_DTYPES)):
        if _DEBUG:
            _debug("conv2d_bwd_data", "stock",
                   "grouped conv" if groups != 1 else "CK unavailable, or dtype outside its tier",
                   grad_output=grad_output, weight=weight)
        return stock()
    ck = lambda: ck_ops.conv2d_backward_data(grad_output, weight, input_size, stride, padding, dilation)
    if not kernel_select.enabled():
        out = ck()
        if _DEBUG:
            _debug("conv2d_bwd_data", "ck" if out is not None else "stock",
                   "no contest (kernel_select off)", grad_output=grad_output, weight=weight)
        return out if out is not None else stock()
    key = ("conv2d_bwd_data", grad_output.dtype, tuple(grad_output.shape), tuple(weight.shape),
           tuple(aiter_ops._pair(stride)), tuple(aiter_ops._pair(padding)),
           tuple(aiter_ops._pair(dilation)))
    out = kernel_select.pick_key("conv2d_bwd_data", key, [("ck", ck), ("stock", stock)])
    if _DEBUG:
        # pick_key both runs the first-call contest and serves the cached
        # winner afterwards, so the cache is the only place that knows
        # which tier actually ran -- read it back rather than guess.
        _debug("conv2d_bwd_data",
               (kernel_select.cached_key("conv2d_bwd_data", key) or "unknown")
               if out is not None else "stock",
               "contest winner" if out is not None else "every candidate declined",
               grad_output=grad_output, weight=weight)
    return out if out is not None else stock()


def _conv2d_backward_weight(input, grad_output, weight_size, stride, padding, dilation, groups):
    """dL/dWeight for _Conv2dFn.backward -- see _conv2d_backward_data,
    same contest shape, CK's backward-weight device op vs stock."""
    stock = lambda: _conv2d_backward_weight_stock(
        input, grad_output, weight_size, stride, padding, dilation, groups)
    if groups != 1 or not (ck_ops.available() and _usable(input, grad_output, dtypes=_CK_CONV_DTYPES)):
        if _DEBUG:
            _debug("conv2d_bwd_weight", "stock",
                   "grouped conv" if groups != 1 else "CK unavailable, or dtype outside its tier",
                   input=input, grad_output=grad_output)
        return stock()
    ck = lambda: ck_ops.conv2d_backward_weight(input, grad_output, weight_size, stride, padding, dilation)
    if not kernel_select.enabled():
        out = ck()
        if _DEBUG:
            _debug("conv2d_bwd_weight", "ck" if out is not None else "stock",
                   "no contest (kernel_select off)", input=input, grad_output=grad_output)
        return out if out is not None else stock()
    key = ("conv2d_bwd_weight", input.dtype, tuple(input.shape), tuple(grad_output.shape),
           tuple(weight_size), tuple(aiter_ops._pair(stride)), tuple(aiter_ops._pair(padding)),
           tuple(aiter_ops._pair(dilation)))
    out = kernel_select.pick_key("conv2d_bwd_weight", key, [("ck", ck), ("stock", stock)])
    if _DEBUG:
        _debug("conv2d_bwd_weight",
               (kernel_select.cached_key("conv2d_bwd_weight", key) or "unknown")
               if out is not None else "stock",
               "contest winner" if out is not None else "every candidate declined",
               input=input, grad_output=grad_output)
    return out if out is not None else stock()


class _Conv2dFn(torch.autograd.Function):
    """Makes _patched_conv2d's accelerated forward usable when a gradient
    is needed, with a REAL backward -- unlike _LinearFn/_BmmFn, conv2d's
    backward is not reducible to calling conv2d again the same simple way
    (see _patched_conv2d's TRAINING-TIME GAP comment: backward-data/
    backward-weight are algorithmically distinct convolution variants, not
    GEMMs), so this routes through CK's WMMA backward-data/backward-weight
    device ops (source/cmp_ext_turing/src/cuda/ck_conv_bwd*.{hpp,cu} via
    ck_ops.conv2d_backward_data/_weight) instead, each contested against
    PyTorch's own stock backward via kernel_select
    (_conv2d_backward_data/_weight above). Bias gradient is a plain
    reduction, not a kernel choice, computed the same way regardless of
    which candidate wins the data/weight contests.

    fwd_fn is _patched_conv2d's own accelerated-candidate selection,
    called unchanged -- only the backward is new here. Scoped to conv2d
    only: conv3d has no corresponding CK backward tier (see
    _patched_conv3d), so it keeps falling back to stock whenever grad is
    needed, same as before this class existed."""

    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups, fwd_fn):
        ctx.save_for_backward(input, weight)
        ctx.has_bias = bias is not None
        ctx.stride, ctx.padding, ctx.dilation, ctx.groups = stride, padding, dilation, groups
        return fwd_fn(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        stride, padding, dilation, groups = ctx.stride, ctx.padding, ctx.dilation, ctx.groups
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = _conv2d_backward_data(
                grad_output, weight, tuple(input.shape), stride, padding, dilation, groups)
        if ctx.needs_input_grad[1]:
            grad_weight = _conv2d_backward_weight(
                input, grad_output, tuple(weight.shape), stride, padding, dilation, groups)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=(0, 2, 3))
        return grad_input, grad_weight, grad_bias, None, None, None, None, None


def _patched_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # Three tiers: native HIP kernel (fp16/fp32, this project's own
    # src/cuda/conv2d_fp{16,32}.cu) first, then aiter's Triton conv2d
    # (fp16/bf16 only -- covers the bf16 case the native kernel doesn't),
    # then stock. The native kernel is always available (hard-required at
    # import time, same as group_norm); aiter is optional.
    orig = _ORIGINALS[(F, "conv2d")]
    if groups != 1:
        if _DEBUG:
            _debug("conv2d", "stock", "grouped conv: no tier here supports groups != 1",
                   input=input, weight=weight, groups=groups)
        return orig(input, weight, bias, stride, padding, dilation, groups)
    if _is_pointwise_conv2d(weight, stride, padding, dilation):
        if _DEBUG:
            _debug("conv2d", "stock", "1x1 pointwise conv", input=input, weight=weight)
        return orig(input, weight, bias, stride, padding, dilation, groups)

    def _accelerated(input, weight, bias, stride, padding, dilation, groups):
        return _conv2d_accelerated_forward(input, weight, bias, stride, padding, dilation, groups, orig)

    if _grad_safe(input, weight, bias):
        return _accelerated(input, weight, bias, stride, padding, dilation, groups)
    # A gradient is needed. UNTIL RECENTLY this always meant falling
    # straight back to stock -- CONFIRMED VIA A REAL SDXL LoRA TRAINING RUN
    # (MIOpen verbose log, gfx1100, character-mode LoKr targeting
    # to_q/to_k/to_v): this branch is not a rare edge case during training,
    # it is EVERY conv2d call. Frozen conv weights don't matter here --
    # _grad_safe() also declines whenever the *input activation*
    # requires_grad, and once any upstream param anywhere in the graph
    # needs a gradient (any LoRA adapter, anywhere in the UNet), autograd
    # must be able to backprop THROUGH every frozen conv layer between it
    # and the loss, so its activations requires_grad=True too. The logged
    # run showed zero native/CK/kernel_select decisions and 100% stock
    # MIOpen calls for every single conv2d, forward and backward.
    #
    # That same log is also where stock itself doesn't have a fast path:
    # every logged backward-data call with symmetric in==out channels and
    # stride 1 got MIOpen's Winograd assembly kernel
    # (miopenSp3AsmConv_v30_3_1_gfx11_fp16_dot2_f2x3_stride1) -- but every
    # call with asymmetric channels (concatenated-skip-connection ResNet
    # convs: 1280->640, 1920->640, 1920->1280, 2560->1280) or stride 2
    # (downsampling convs: 640->640 H64xW37, 320->320 H128xW73) fell back
    # to the slower Col2Im2dU (im2col+GEMM) kernel instead. Those specific
    # shapes -- not the symmetric stride-1 case, where stock's Winograd
    # already wins by 4.6x per ck_ops.py's own fp16 benchmark -- are where
    # CK's real backward-data/backward-weight device ops (_Conv2dFn,
    # _conv2d_backward_data/_weight above) have an actual opening: they're
    # contested against stock per shape via kernel_select rather than
    # assumed to win, exactly like the forward tier already is, so the
    # symmetric-stride-1 case simply keeps picking stock instead of
    # regressing. _grad_safe()'s decline is now the entry into that
    # contest instead of a dead end.
    if _DEBUG:
        _debug("conv2d", "autograd", "gradient needed: accelerated fwd via _Conv2dFn",
               input=input, weight=weight)
    return _Conv2dFn.apply(input, weight, bias, stride, padding, dilation, groups, _accelerated)


def _conv2d_accelerated_forward(input, weight, bias, stride, padding, dilation, groups, orig):
    # Sparse-pixel fast path, same occupancy-gated design as _patched_conv3d's
    # equivalent check above it (see flexgemm_ops.maybe_sparse_conv2d's
    # docstring) -- content-dependent, so re-checked every call rather than
    # folded into kernel_select's shape cache below. The check itself is
    # cheap on the dense activations a U-Net is made of: 0.0035ms against a
    # 0.326ms conv on 1x320x64x64, i.e. 1.1%, and it declines immediately.
    # AMD_TUNED_TORCH_SPARSE_CONV2D=0 disables it. It also declines unless
    # flex_gemm's native extension is built: this used to run the
    # pure-Python gather/scatter instead, which measured 20-60x SLOWER than
    # stock on exactly the sparse inputs it accepted (numbers in
    # maybe_sparse_conv2d's docstring).
    # _is_plain_tensor, not just dim(): this branch runs BEFORE any _usable
    # check in this function, and FlexGEMM's kernels read dense storage the
    # same way the GEMM tiers do. A Tensor subclass (a quantized weight, say)
    # reaching them fails the way it did in F.linear -- an illegal memory
    # access from inside the subclass's own dispatch fallback, not a catchable
    # exception. See _PLAIN_TENSOR_TYPES.
    if (flexgemm_ops.sparse_conv2d_enabled() and input.dim() == 4 and weight.dim() == 4
            and _is_plain_tensor(input) and _is_plain_tensor(weight)):
        _sparse_out = flexgemm_ops.maybe_sparse_conv2d(
            input, weight, bias, stride=aiter_ops._pair(stride),
            padding=aiter_ops._pair(padding), dilation=aiter_ops._pair(dilation))
        if _sparse_out is not None:
            if _DEBUG:
                _debug("conv2d", "sparse", "flex_gemm sparse_conv2d took it",
                       input=input, weight=weight)
            return _sparse_out

    # Which kernel wins is measured per shape, stock included as a
    # candidate -- see kernel_select.py. This is deliberately NOT a fixed
    # ordering: on gfx1100 stock beats our kernels for fp16/fp32 conv2d
    # (MIOpen has an assembly Winograd solver) and loses badly for bf16
    # (it has none), so any static order is wrong for some dtype.
    if kernel_select.enabled():
        # Fast path: once this shape has been decided, go straight to the
        # winner. Building the candidate thunks below costs ~0.09ms per
        # call -- real money on the many small convs a U-Net is made of,
        # and pointless when the answer is already known.
        _won = kernel_select.cached("conv2d", input, weight, stride, padding, dilation)
        if _won == "stock":
            if _DEBUG:
                _debug("conv2d", "stock", "measured winner for this shape",
                       input=input, weight=weight)
            return orig(input, weight, bias, stride, padding, dilation, groups)
        if _won == "ck":
            _out = ck_ops.conv2d(input, weight, bias, stride, padding, dilation)
            if _out is not None:
                if _DEBUG:
                    _debug("conv2d", "ck", "cached winner", input=input, weight=weight)
                return _out
            if _DEBUG:
                _debug("conv2d", "declined", "cached winner ck returned None",
                       input=input, weight=weight)
        elif _won == "native":
            try:
                if _DEBUG:
                    _debug("conv2d", "native", "cached winner", input=input, weight=weight)
                return compile_ops.conv2d_native(input, weight, bias, stride, padding, dilation)
            except (RuntimeError, TypeError) as exc:
                # Swallowed on purpose (stock below is always correct), but
                # silently swallowing the reason is what makes a kernel that
                # raises on every call indistinguishable from one that was
                # never picked.
                if _DEBUG:
                    _debug("conv2d", "raised", "cached winner native raised: %r" % (exc,),
                           input=input, weight=weight)
        elif _won == "fftconv":
            _fft_won = _fftconv_conv_candidate(input, weight, bias, stride, padding,
                                                dilation, groups, ndim=2)
            if _fft_won is not None:
                _out = _fft_won[1]()
                if _out is not None:
                    if _DEBUG:
                        _debug("conv2d", "fftconv", "cached winner", input=input, weight=weight)
                    return _out
            if _DEBUG:
                _debug("conv2d", "declined", "cached winner fftconv declined this call",
                       input=input, weight=weight)

        candidates = []
        if ck_ops.available() and _usable(input, weight, dtypes=_CK_CONV_DTYPES):
            candidates.append(
                ("ck", lambda: ck_ops.conv2d(input, weight, bias, stride, padding, dilation)))
        if (input.is_contiguous() and weight.is_contiguous()
                and _usable(input, weight, dtypes=_CONV_NATIVE_DTYPES)):
            candidates.append(
                ("native", lambda: compile_ops.conv2d_native(
                    input, weight, bias, stride, padding, dilation)))
        # Large-kernel tier: only contested at all above a measured kernel
        # width (see _fftconv_conv_candidate), so a U-Net's 3x3 convs never
        # pay for it.
        _fft = _fftconv_conv_candidate(input, weight, bias, stride, padding,
                                        dilation, groups, ndim=2)
        if _fft is not None:
            candidates.append(_fft)
        candidates.append(
            ("stock", lambda: orig(input, weight, bias, stride, padding, dilation, groups)))
        if _DEBUG:
            _debug("conv2d", "contest", "first call for this shape",
                   input=input, weight=weight,
                   candidates=",".join(name for name, _ in candidates))
        out = kernel_select.pick("conv2d", input, weight, stride, padding, dilation, candidates,
                                  tolerance=_fftconv_contest_tolerance(_fft, input))
        if out is not None:
            if _DEBUG:
                _debug("conv2d",
                       kernel_select.cached("conv2d", input, weight, stride, padding, dilation)
                       or "unknown",
                       "contest decided", input=input, weight=weight)
            return out

    # conv_select disabled (AMD_TUNED_TORCH_MEASURE_KERNELS=0): fall back to the
    # old fixed ordering, our kernels first. The FFT tier goes first of all
    # for a kernel wide enough to have passed its pre-filter: with no contest
    # to measure anything, that pre-filter is all the evidence there is, and
    # it is the same static-heuristic posture fftconv_ops.maybe_fft_conv1d
    # keeps for callers who don't want to pay a contest either.
    _fft = _fftconv_conv_candidate(input, weight, bias, stride, padding,
                                    dilation, groups, ndim=2)
    if _fft is not None:
        _out = _fft[1]()
        if _out is not None:
            if _DEBUG:
                _debug("conv2d", "fftconv", "fixed order (kernel_select off)",
                       input=input, weight=weight)
            return _out
    if ck_ops.available() and _usable(input, weight, dtypes=_CK_CONV_DTYPES):
        try:
            _ck = ck_ops.conv2d(input, weight, bias, stride, padding, dilation)
            if _ck is not None:
                if _DEBUG:
                    _debug("conv2d", "ck", "fixed order (kernel_select off)",
                           input=input, weight=weight)
                return _ck
        except (RuntimeError, TypeError) as exc:
            if _DEBUG:
                _debug("conv2d", "raised", "ck raised: %r" % (exc,), input=input, weight=weight)
    if (input.is_contiguous() and weight.is_contiguous()
            and _usable(input, weight, dtypes=_CONV_NATIVE_DTYPES)):
        try:
            if _DEBUG:
                _debug("conv2d", "native", "fixed order (kernel_select off)",
                       input=input, weight=weight)
            return compile_ops.conv2d_native(input, weight, bias, stride, padding, dilation)
        except (RuntimeError, TypeError) as exc:
            if _DEBUG:
                _debug("conv2d", "raised", "native raised: %r" % (exc,),
                       input=input, weight=weight)
    if aiter_ops.available() and _usable(input, weight):
        try:
            if _DEBUG:
                _debug("conv2d", "aiter", "fixed order (kernel_select off)",
                       input=input, weight=weight)
            return compile_ops.conv2d_fp16(input, weight, bias, stride, padding, dilation)
        except (RuntimeError, TypeError, AssertionError) as exc:
            if _DEBUG:
                _debug("conv2d", "raised", "aiter raised: %r" % (exc,),
                       input=input, weight=weight)
    if _DEBUG:
        _debug("conv2d", "stock", "every tier declined or is unavailable",
               input=input, weight=weight)
    return orig(input, weight, bias, stride, padding, dilation, groups)


def _conv3d_accelerated_forward(input, weight, bias, stride, padding, dilation, groups):
    """F.conv3d's accelerated forward: sparse-voxel fast path, then a
    per-shape CK/native/stock contest.

    Forward only, and called either directly (no gradient needed) or as
    _Conv3dFn's forward (gradient needed) -- see _patched_conv3d."""
    # Two tiers now: Composable Kernel's WMMA conv (fp16/bf16) then this
    # project's hand-written HIP kernel (fp16/fp32,
    # src/cuda/conv3d_fp{16,32}.cu). Neither aiter nor TransformerEngine
    # cover conv3d at all, so CK is what fills that gap -- and it is also
    # the first conv3d bf16 path here that isn't stock.
    orig = _ORIGINALS[(F, "conv3d")]
    if groups != 1:
        if _DEBUG:
            _debug("conv3d", "stock", "grouped conv: no tier here supports groups != 1",
                   input=input, weight=weight, groups=groups)
        return orig(input, weight, bias, stride, padding, dilation, groups)

    # Sparse-voxel fast path: routes to flex_gemm's sparse_conv3d instead of
    # a dense kernel when `input` is mostly empty (e.g. a voxelized surface
    # or point cloud, as opposed to a diffusion U-Net's normally-dense
    # activations). This is a CONTENT-dependent decision -- occupancy, not
    # shape -- so unlike the kernel_select contest just below, it can't be
    # cached by (dtype, shape, stride, padding, dilation): two calls with an
    # identical shape key can have very different occupancy. It is instead
    # re-checked on every eligible call via a single cheap reduction over
    # `input` (see flexgemm_ops.maybe_sparse_conv3d's docstring for that
    # cost and the unvalidated default occupancy threshold). Declines
    # (returns None) instantly back to the dense contest below whenever
    # flex_gemm isn't installed, AMD_TUNED_TORCH_SPARSE_CONV3D=0 has
    # disabled it, or occupancy is at/above the threshold -- so a normal
    # dense conv3d call pays only the one reduction pass, never a wasted
    # sparse-kernel attempt.
    # Same subclass guard as the conv2d sparse path above.
    if (flexgemm_ops.sparse_conv3d_enabled() and input.dim() == 5 and weight.dim() == 5
            and _is_plain_tensor(input) and _is_plain_tensor(weight)):
        _sparse_out = flexgemm_ops.maybe_sparse_conv3d(
            input, weight, bias, stride=_triple(stride), padding=_triple(padding),
            dilation=_triple(dilation))
        if _sparse_out is not None:
            if _DEBUG:
                _debug("conv3d", "sparse", "flex_gemm sparse_conv3d took it",
                       input=input, weight=weight)
            return _sparse_out

    # Measured per shape with stock as a candidate, exactly as in
    # _patched_conv2d. conv3d is where our kernels look best (CK 1.16ms vs
    # stock 3.02ms at fp16) but fp32 is within noise of stock (0.95x), so
    # the same contest applies rather than a fixed order.
    if kernel_select.enabled():
        # Fast path: once this shape has been decided, go straight to the
        # winner. Building the candidate thunks below costs ~0.09ms per
        # call -- real money on the many small convs a U-Net is made of,
        # and pointless when the answer is already known.
        _won = kernel_select.cached(_CONV3D_KIND, input, weight, stride, padding, dilation)
        if _won == "stock":
            if _DEBUG:
                _debug("conv3d", "stock", "measured winner for this shape",
                       input=input, weight=weight)
            return orig(input, weight, bias, stride, padding, dilation, groups)
        if _won == "ck":
            _out = ck_ops.conv3d(input, weight, bias, stride, padding, dilation)
            if _out is not None:
                if _DEBUG:
                    _debug("conv3d", "ck", "cached winner", input=input, weight=weight)
                return _out
            if _DEBUG:
                _debug("conv3d", "declined", "cached winner ck returned None",
                       input=input, weight=weight)
        elif _won == "native":
            try:
                if _DEBUG:
                    _debug("conv3d", "native", "cached winner", input=input, weight=weight)
                return compile_ops.conv3d_native(input, weight, bias, stride, padding, dilation)
            except (RuntimeError, TypeError) as exc:
                if _DEBUG:
                    _debug("conv3d", "raised", "cached winner native raised: %r" % (exc,),
                           input=input, weight=weight)
        elif _won == "comfy":
            _out = comfy_kitchen_ops.conv3d(input, weight, bias, stride, padding, dilation)
            if _out is not None:
                if _DEBUG:
                    _debug("conv3d", "comfy", "cached winner", input=input, weight=weight)
                return _out
            if _DEBUG:
                # Reachable without a shape change: this candidate also gates
                # on torch.backends.cuda.matmul.allow_fp16_accumulation, which
                # a host app can switch off between calls.
                _debug("conv3d", "declined", "cached winner comfy returned None",
                       input=input, weight=weight)
        elif _won == "fftconv":
            _fft_won = _fftconv_conv_candidate(input, weight, bias, stride, padding,
                                                dilation, groups, ndim=3)
            if _fft_won is not None:
                _out = _fft_won[1]()
                if _out is not None:
                    if _DEBUG:
                        _debug("conv3d", "fftconv", "cached winner", input=input, weight=weight)
                    return _out
            if _DEBUG:
                _debug("conv3d", "declined", "cached winner fftconv declined this call",
                       input=input, weight=weight)

        candidates = []
        if ck_ops.available() and _usable(input, weight, dtypes=_CK_CONV_DTYPES):
            candidates.append(
                ("ck", lambda: ck_ops.conv3d(input, weight, bias, stride, padding, dilation)))
        if (input.is_contiguous() and weight.is_contiguous()
                and _usable(input, weight, dtypes=_CONV_NATIVE_DTYPES)):
            candidates.append(
                ("native", lambda: compile_ops.conv3d_native(
                    input, weight, bias, stride, padding, dilation)))
        # Large-kernel tier, and the one place in this file where it wins
        # by orders of magnitude rather than percent: direct conv3d pays
        # K^3 multiplies per output element (measured 15^3 fp32: stock
        # 6907ms vs 38.8ms here). See _fftconv_conv_candidate.
        _fft = _fftconv_conv_candidate(input, weight, bias, stride, padding,
                                        dilation, groups, ndim=3)
        if _fft is not None:
            candidates.append(_fft)
        # comfy-kitchen's WMMA implicit-GEMM conv3d -- a fourth ALGORITHM here,
        # not a fourth instance of the same one, which is the only sort of
        # candidate worth the cost of a contest. fp16-accumulate, so it is
        # offered only where the caller asked for that; see the _COMFY block
        # above and comfy_kitchen_ops.conv3d.
        _comfy_conv3d = _COMFY and _usable(input, weight, dtypes=(torch.float16,))
        if _comfy_conv3d:
            candidates.append(
                ("comfy", lambda: comfy_kitchen_ops.conv3d(
                    input, weight, bias, stride, padding, dilation)))
        candidates.append(
            ("stock", lambda: orig(input, weight, bias, stride, padding, dilation, groups)))
        if _DEBUG:
            _debug("conv3d", "contest", "first call for this shape",
                   input=input, weight=weight,
                   candidates=",".join(name for name, _ in candidates))
        out = kernel_select.pick(_CONV3D_KIND, input, weight, stride, padding, dilation,
                                  candidates,
                                  tolerance=_fftconv_contest_tolerance(_fft, input))
        if out is not None:
            if _DEBUG:
                _debug("conv3d",
                       kernel_select.cached(_CONV3D_KIND, input, weight, stride, padding,
                                            dilation)
                       or "unknown",
                       "contest decided", input=input, weight=weight)
            return out

    # conv_select disabled: old fixed ordering, our kernels first -- with
    # the FFT tier ahead of them for a wide-enough kernel, same reasoning
    # as _patched_conv2d's equivalent branch.
    _fft = _fftconv_conv_candidate(input, weight, bias, stride, padding,
                                    dilation, groups, ndim=3)
    if _fft is not None:
        _out = _fft[1]()
        if _out is not None:
            if _DEBUG:
                _debug("conv3d", "fftconv", "fixed order (kernel_select off)",
                       input=input, weight=weight)
            return _out
    if ck_ops.available() and _usable(input, weight, dtypes=_CK_CONV_DTYPES):
        try:
            _ck = ck_ops.conv3d(input, weight, bias, stride, padding, dilation)
            if _ck is not None:
                if _DEBUG:
                    _debug("conv3d", "ck", "fixed order (kernel_select off)",
                           input=input, weight=weight)
                return _ck
        except (RuntimeError, TypeError) as exc:
            if _DEBUG:
                _debug("conv3d", "raised", "ck raised: %r" % (exc,), input=input, weight=weight)
    # groups/grad-safety already checked above.
    if (not (input.is_contiguous() and weight.is_contiguous())
            or not _usable(input, weight, dtypes=_CONV_NATIVE_DTYPES)):
        if _DEBUG:
            _debug("conv3d", "stock", "non-contiguous, or dtype outside the native tier",
                   input=input, weight=weight)
        return orig(input, weight, bias, stride, padding, dilation, groups)
    try:
        if _DEBUG:
            _debug("conv3d", "native", "fixed order (kernel_select off)",
                   input=input, weight=weight)
        return compile_ops.conv3d_native(input, weight, bias, stride, padding, dilation)
    except (RuntimeError, TypeError) as exc:
        if _DEBUG:
            _debug("conv3d", "raised", "native raised: %r" % (exc,), input=input, weight=weight)
        return orig(input, weight, bias, stride, padding, dilation, groups)


class _Conv3dFn(torch.autograd.Function):
    """Makes _conv3d_accelerated_forward usable when a gradient is needed.

    Before this existed, a live gradient was a hard decline for conv3d --
    the one op here where _patched_conv3d's own debug message said so out
    loud ("gradient needed (no conv3d backward tier)"). The forward is
    where this package's conv3d tiers actually win (CK 1.16ms vs stock
    3.02ms at fp16 on the bench shape), and losing it for the whole of
    training because the BACKWARD had no tier was the same
    inference-only-by-accident trap the TRAINING-TIME GAP note above
    _LinearFn describes for linear/matmul/bmm/conv2d.

    HOW THIS DIFFERS FROM _Conv2dFn. conv2d routes its backward through
    CK's WMMA backward-data/backward-weight device ops, each contested
    against stock. There is no CK conv3d backward (ck_ops has no
    conv3d_backward_* at all), so there is nothing to contest: the
    backward here is a single torch.ops.aten.convolution_backward call --
    exactly the op autograd itself would have used had this module never
    patched F.conv3d, so it is neither faster nor slower than not
    patching, and it is correct by construction rather than by
    validation. Forward accelerated, backward at stock speed, which is
    strictly better than the old behaviour of forward AND backward at
    stock speed.

    If a CK (or native) conv3d backward tier is ever added, this is the
    one place to contest it -- mirror _conv2d_backward_data/_weight.
    """

    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups):
        ctx.save_for_backward(input, weight)
        ctx.bias_sizes = [bias.shape[0]] if bias is not None else None
        ctx.stride, ctx.padding, ctx.dilation, ctx.groups = stride, padding, dilation, groups
        return _conv3d_accelerated_forward(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        needs = [bool(ctx.needs_input_grad[0]), bool(ctx.needs_input_grad[1]),
                 ctx.bias_sizes is not None and bool(ctx.needs_input_grad[2])]
        if not any(needs):
            return None, None, None, None, None, None, None
        grad_input, grad_weight, grad_bias = torch.ops.aten.convolution_backward(
            grad_output.contiguous(), input, weight, ctx.bias_sizes,
            list(_triple(ctx.stride)), list(_triple(ctx.padding)),
            list(_triple(ctx.dilation)), False, [0, 0, 0], ctx.groups, needs)
        return grad_input, grad_weight, grad_bias, None, None, None, None


def _patched_conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """F.conv3d: the accelerated forward either way, with autograd support
    when a gradient is live (see _Conv3dFn)."""
    if _grad_safe(input, weight, bias):
        return _conv3d_accelerated_forward(input, weight, bias, stride, padding, dilation, groups)
    if groups != 1:
        # No tier here supports grouped conv in the first place, so there
        # is nothing to wrap -- let stock handle forward and backward.
        if _DEBUG:
            _debug("conv3d", "stock", "grouped conv: no tier here supports groups != 1",
                   input=input, weight=weight, groups=groups)
        return _ORIGINALS[(F, "conv3d")](input, weight, bias, stride, padding, dilation, groups)
    if _DEBUG:
        _debug("conv3d", "autograd", "gradient needed: accelerated forward + stock backward",
               input=input, weight=weight)
    return _Conv3dFn.apply(input, weight, bias, stride, padding, dilation, groups)


# ---------------------------------------------------------------------------
# group_norm: native HIP kernel, with autograd support (see _GroupNormFn).
# ---------------------------------------------------------------------------

def _group_norm_native(input, num_groups, weight, bias, eps):
    """The native group_norm kernel, routed to avoid paying for
    torch.compile safety in eager.

    compile_ops.group_norm wraps the kernel in a torch.library custom op so
    it doesn't graph-break under torch.compile (see compile_ops.py). That
    wrapper costs ~15us of dispatch, which is nothing for a conv but is
    most of this kernel's margin: raw it beats stock 1.10x (0.150ms vs
    0.166ms at the bench shape), through the custom op it loses. Since the
    graph-break protection only matters while Dynamo is actually tracing,
    take the custom op then and the direct call otherwise -- identical
    numerics either way.
    """
    if _is_compiling():
        return compile_ops.group_norm(input, num_groups, weight, bias, eps)
    return ops.group_norm(input, num_groups, weight, bias, eps)


def _group_norm_accelerated_forward(input, num_groups, weight, bias, eps):
    """F.group_norm's accelerated forward over a per-shape contest: native
    HIP, CK, stock.

    Forward only, and called either directly (no gradient needed) or as
    _GroupNormFn's forward (gradient needed) -- see _patched_group_norm.

    The CK candidate only enters for CHANNELS-LAST input, and that is a
    measured restriction rather than a conservative one. CK's normalization
    instances index the tensor as [N, S1, S2, G, C], which is a free reshape
    of channels-last memory and a permute-in-and-out of contiguous NCHW.
    Measured against stock on plain GroupNorm:

        2x320x64x64  fp16    NHWC 2.09x    NCHW 0.85x
        2x640x32x32  fp16    NHWC 1.37x    NCHW 0.48x
        2x1280x16x16 fp32    NHWC 1.26x    NCHW 0.40x

    So it is offered where it wins and withheld where it loses. NCHW keeps
    the native kernel, which is what it was already getting.

    The larger win from this tier is not reachable here at all: fusing the
    SiLU that follows GroupNorm in every diffusion ResBlock is 1.6-2.1x, and
    F.group_norm cannot know an activation follows it. See
    ck_norm_ops.group_norm_silu for the call a model can make directly.
    """
    orig = _ORIGINALS[(F, "group_norm")]
    channels_last = ck_norm_ops._channels_last(input)
    # No _grad_safe() check here any more: a live gradient used to be a
    # hard decline and is now just a different entry point (_GroupNormFn,
    # via _patched_group_norm). Layout and dtype are still real limits.
    if not ((input.is_contiguous() or channels_last)
            and _usable(input, dtypes=_GROUPNORM_DTYPES)):
        if _DEBUG:
            _debug("group_norm", "stock", "layout/dtype outside this tier",
                   input=input, channels_last=channels_last)
        return orig(input, num_groups, weight, bias, eps)

    # Measured per shape like conv, rather than assuming the native kernel
    # wins. It usually does, but only by 1.03-1.23x depending on dtype --
    # small enough that "assume ours is faster" is not a safe default, and
    # the contest costs nothing after the first call for a shape.
    if kernel_select.enabled():
        # Layout belongs in the key: it decides which candidates are even
        # eligible, not merely how fast they are.
        key = (input.dtype, tuple(input.shape), int(num_groups), channels_last)
        won = kernel_select.cached_key("group_norm", key)
        if won == "stock":
            if _DEBUG:
                _debug("group_norm", "stock", "measured winner for this shape", key=key)
            return orig(input, num_groups, weight, bias, eps)
        if won == "ck":
            out = ck_norm_ops.group_norm(input, num_groups, weight, bias, eps)
            if out is not None:
                if _DEBUG:
                    _debug("group_norm", "ck", "cached winner", key=key)
                return out
            if _DEBUG:
                _debug("group_norm", "declined", "cached winner ck returned None", key=key)
        elif won != "native":
            candidates = []
            if channels_last:
                candidates.append(
                    ("ck", lambda: ck_norm_ops.group_norm(input, num_groups, weight, bias, eps)))
            if input.is_contiguous():
                candidates.append(
                    ("native", lambda: _group_norm_native(input, num_groups, weight, bias, eps)))
            candidates.append(("stock", lambda: orig(input, num_groups, weight, bias, eps)))
            if _DEBUG:
                _debug("group_norm", "contest", "first call for this shape", key=key,
                       candidates=",".join(name for name, _ in candidates))
            out = kernel_select.pick_key("group_norm", key, candidates)
            if out is not None:
                if _DEBUG:
                    _debug("group_norm",
                           kernel_select.cached_key("group_norm", key) or "unknown",
                           "contest decided", key=key)
                return out

    if input.is_contiguous():
        try:
            if _DEBUG:
                # Also the cached won == "native" path: that branch has no
                # body of its own and deliberately falls through to here.
                _debug("group_norm", "native", "cached winner, or kernel_select off",
                       input=input)
            return _group_norm_native(input, num_groups, weight, bias, eps)
        except (RuntimeError, TypeError) as exc:
            if _DEBUG:
                _debug("group_norm", "raised", "native raised: %r" % (exc,), input=input)
    if _DEBUG:
        _debug("group_norm", "stock", "every tier declined", input=input)
    return orig(input, num_groups, weight, bias, eps)


def _group_norm_backward(grad_output, input, num_groups, weight, eps, needs,
                         bias_dtype=None):
    """GroupNorm's own derivative, in plain torch ops -- see _GroupNormFn.

    `needs` is autograd's own (input, weight, bias) needs_input_grad
    triple; anything not needed is skipped and returned as None.
    `weight` may be None (non-affine GroupNorm), which drops the weight
    term of grad_input and forces grad_weight to None.

    Each gradient is returned in ITS OWN input's dtype, which is what
    autograd requires of them: grad_input in input's, grad_weight in
    weight's, and grad_bias in `bias_dtype` (bias itself is not needed
    here -- the derivative doesn't read its values -- but an fp32 bias
    under fp16 activations is normal, so its dtype cannot be assumed to
    match either of the others). bias_dtype defaults to the input's.

    Shapes: everything is viewed as (N, G, M, S) -- batch, groups,
    channels per group, spatial -- so the per-group reductions are over
    dims (2, 3) and the per-channel ones over (0, 3). That holds for any
    spatial rank (1d/2d/3d GroupNorm) with no special cases.

    DTYPE. Reductions accumulate via the `dtype=` argument, which
    accumulates in a wider type without materialising a wider copy of the
    activation; the per-element arithmetic then runs in the input's own
    dtype with the statistics cast back down, the same split stock
    GroupNorm uses. The promotion is not optional for the half types: one
    SDXL group is up to 163,840 elements, and an fp16 sum over that many
    unit-variance squares overflows fp16's 65504 ceiling outright -- the
    exact trap the Liger GroupNorm kernel fell into (see
    liger_group_norm_ops' VALIDATION STATUS, bug 2).

    It promotes ONLY the half types, and that `only` is load-bearing: an
    unconditional dtype=torch.float32 here silently DOWNCAST fp64 (and
    left fp32 no better off), which showed up as ~1.7e-7 disagreement
    with autograd's own gradients on float64 inputs -- small, but a
    precision regression this function was supposed to prevent, not
    cause. fp32 and fp64 already accumulate in themselves.
    """
    N, C = input.shape[0], input.shape[1]
    M = C // num_groups
    dtype = input.dtype
    acc = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    # reshape indexes LOGICALLY (NCHW order) whatever the memory layout,
    # so this is correct for channels-last input too -- it just copies
    # there instead of viewing, since NHWC strides can't be viewed as
    # (N, G, M, S). The grad_input built from it is then put back into the
    # input's own memory format at the end: the accelerated forward's CK
    # candidate is channels-last-only, so on that path every tensor
    # around this one is NHWC and handing autograd an NCHW gradient would
    # push a layout conversion into the next op instead.
    x = input.reshape(N, num_groups, M, -1)
    dy = grad_output.reshape(N, num_groups, M, -1)

    needs_input, needs_weight, needs_bias = needs
    grad_input = grad_weight = grad_bias = None

    if needs_bias:
        grad_bias = dy.sum(dim=(0, 3), dtype=acc).reshape(C).to(
            bias_dtype if bias_dtype is not None else dtype)

    if not (needs_input or needs_weight):
        return grad_input, grad_weight, grad_bias

    mean = x.mean(dim=(2, 3), keepdim=True, dtype=acc)
    x_centered = x - mean.to(dtype)
    # Mean of squared deviations rather than E[x^2] - E[x]^2: same cost
    # here, no catastrophic cancellation.
    var = x_centered.square().mean(dim=(2, 3), keepdim=True, dtype=acc)
    rstd = (var + eps).rsqrt().to(dtype)
    x_hat = x_centered * rstd

    if needs_weight and weight is not None:
        grad_weight = (dy * x_hat).sum(
            dim=(0, 3), dtype=acc).reshape(C).to(weight.dtype)

    if needs_input:
        wdy = dy if weight is None else dy * weight.reshape(1, num_groups, M, 1)
        # The two per-group correction terms of the normalisation
        # derivative: dx = rstd * (wdy - mean(wdy) - x_hat * mean(wdy * x_hat)).
        c1 = wdy.mean(dim=(2, 3), keepdim=True, dtype=acc).to(dtype)
        c2 = (wdy * x_hat).mean(dim=(2, 3), keepdim=True, dtype=acc).to(dtype)
        grad_input = ((wdy - c1 - x_hat * c2) * rstd).reshape(input.shape)
        if ck_norm_ops._channels_last(input):
            # _channels_last() covers BOTH ranks it can apply to -- 4D
            # channels_last and 5D channels_last_3d -- and asking for the
            # 4D format on a 5D tensor raises, so the rank picks the enum.
            grad_input = grad_input.contiguous(
                memory_format=torch.channels_last if input.dim() == 4
                else torch.channels_last_3d)

    return grad_input, grad_weight, grad_bias


class _GroupNormFn(torch.autograd.Function):
    """Makes _group_norm_accelerated_forward usable when a gradient is
    needed, with a REAL backward.

    THE LAST OF THE TRAINING-TIME GAPS (see that note above
    _LinearFn). group_norm was the one hot op where _grad_safe() stayed a
    dead end: SDXL's UNet and VAE are built almost entirely out of
    GroupNorm-normalised ResNet blocks, and during LoRA/LyCORIS training
    essentially every activation requires_grad, so this tier declined on
    essentially every call and GroupNorm got no acceleration at all --
    which is the entire reason liger_group_norm_ops exists as a separate
    opt-in tier with its own Triton kernel.

    Same playbook as _LinearFn/_BmmFn rather than
    _Conv2dFn's: no new kernel and no new numerically-unverified code
    path, just GroupNorm's textbook derivative in plain torch ops
    (_group_norm_backward). Autograd disables grad tracking inside
    a Function's backward, so the reductions and elementwise ops that
    derivative is made of are themselves eligible for this module's own
    patches -- the backward pass rides the same tiers as everything else
    without this class knowing anything about them.

    What this does NOT do is accelerate the backward with a dedicated
    GroupNorm-backward kernel the way _Conv2dFn routes through CK's
    backward-data/backward-weight ops: there is no such kernel here (the
    native HIP group_norm is forward-only, hence this whole problem).
    Forward gets the contest winner; backward gets correct math at
    stock-ish speed. That is still strictly better than the old
    behaviour, where a live gradient meant the FORWARD fell back to stock
    too.
    """

    @staticmethod
    def forward(ctx, input, weight, bias, num_groups, eps):
        ctx.save_for_backward(input, weight)
        ctx.num_groups = num_groups
        ctx.eps = eps
        ctx.has_bias = bias is not None
        # bias isn't saved (the derivative doesn't need its values) but its
        # dtype is: autograd requires each gradient to come back in its own
        # input's dtype, and an fp32 bias under fp16 activations is normal.
        ctx.bias_dtype = bias.dtype if bias is not None else None
        return _group_norm_accelerated_forward(input, num_groups, weight, bias, eps)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        needs = (
            ctx.needs_input_grad[0],
            weight is not None and ctx.needs_input_grad[1],
            ctx.has_bias and ctx.needs_input_grad[2],
        )
        return _group_norm_backward(
            grad_output.contiguous(), input, ctx.num_groups, weight, ctx.eps, needs,
            bias_dtype=ctx.bias_dtype) + (None, None)


def _patched_group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    """F.group_norm: the accelerated contest either way, with autograd
    support when a gradient is live (see _GroupNormFn)."""
    if _grad_safe(input, weight, bias):
        return _group_norm_accelerated_forward(input, num_groups, weight, bias, eps)
    if _DEBUG:
        _debug("group_norm", "autograd", "gradient needed: accelerated forward + real backward",
               input=input)
    return _GroupNormFn.apply(input, weight, bias, num_groups, eps)


# ---------------------------------------------------------------------------
# Wrappers backed by TransformerEngine (real autograd -- no grad-safety
# check needed, active for training as well as inference).
# ---------------------------------------------------------------------------

def _comfy_norm(kind, input, width, has_bias, orig_thunk, comfy_thunk):
    """Per-shape contest between comfy-kitchen's modulation kernel and stock,
    for a norm TE has already declined. Returns a result, or None to mean
    "keep going, stock is still the answer".

    THIS IS THE ONLY TIER IN THIS FILE THAT ADDS A CANDIDATE WHERE THERE WAS
    NONE. Everywhere else an accelerated path declines, the op still had
    something better than stock to fall back to; here _te_norm_worth_it()
    declines every forward-only norm by design -- TE's measured win lives
    entirely in the backward (see the _TE_NORM_* block above) -- so at
    inference F.layer_norm and F.rms_norm have been straight pass-throughs to
    stock. comfy-kitchen's adaln/rms_adaln are inference-only kernels, which is
    exactly the half of the problem TE does not cover.

    IT REQUIRES kernel_select AND DOES NOT GUESS. Every other tier here comes
    with its own measurements on this card and a documented shape gate built
    from them; this one has none yet, and inventing a row threshold to match
    would be fiction. So the contest IS the gate: with kernel_select off, this
    returns None and dispatch behaves exactly as it did before the tier
    existed. Nothing is routed anywhere on the strength of an assumption.

    The key is (dtype, rows, cols, has_bias) rather than the raw shape, for the
    same reason _patched_linear keys on (M, K, N): the kernel flattens to
    (numel // width, width), so [2, 512, 1280] and [1024, 1280] are one problem
    to every candidate and keying on the shape would re-contest each of them.
    """
    if not _COMFY or not kernel_select.enabled():
        return None
    key = (input.dtype, int(input.numel() // width), int(width), has_bias)
    won = kernel_select.cached_key(kind, key)
    if won == "stock":
        if _DEBUG:
            _debug(kind, "stock", "measured winner for this shape", key=key)
        return None
    if won == "comfy":
        out = comfy_thunk()
        if out is not None:
            if _DEBUG:
                _debug(kind, "comfy", "cached winner", key=key)
            return out
        if _DEBUG:
            _debug(kind, "declined", "cached winner comfy returned None", key=key)
        return None
    if won is None:
        if _DEBUG:
            _debug(kind, "contest", "first call for this shape", key=key,
                   candidates="comfy,stock")
        out = kernel_select.pick_key(kind, key, [
            ("comfy", comfy_thunk),
            ("stock", orig_thunk),
        ])
        if out is not None:
            if _DEBUG:
                _debug(kind, kernel_select.cached_key(kind, key) or "unknown",
                       "contest decided", key=key)
            # Either candidate's result is the same numbers; returning stock's
            # here just saves recomputing it in the caller.
            return out
    return None


def _patched_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    orig = _ORIGINALS[(F, "layer_norm")]
    if weight is None or bias is None or not _usable(input, weight, bias, dtypes=_TE_DTYPES):
        if _DEBUG:
            _debug("layer_norm", "stock",
                   "no affine weight/bias" if (weight is None or bias is None)
                   else "not usable: dtype/device/tensor-subclass",
                   input=input)
        return orig(input, normalized_shape, weight, bias, eps)
    if not (te_ops.available() and _te_norm_route("layer", input, weight)):
        # The usual answer for a diffusion model: TE's norm win lives in the
        # backward, so an inference call or a short input is declined by
        # design -- see the _TE_NORM_* block above. That is precisely the case
        # comfy-kitchen's adaln covers, so it gets the shape before stock does.
        # Grad-safety is a hard requirement, not a preference: the kernel has
        # no backward at all, and unlike conv/linear/group_norm there is no
        # autograd.Function wrapper here to give it one.
        if _grad_safe(input, weight, bias):
            _out = _comfy_norm(
                "layer_norm", input, weight.numel(), True,
                lambda: orig(input, normalized_shape, weight, bias, eps),
                lambda: comfy_kitchen_ops.layer_norm(input, normalized_shape,
                                                     weight, bias, eps))
            if _out is not None:
                return _out
        if _DEBUG:
            _debug("layer_norm", "stock", "not worth TE (forward-only, too few rows, or too wide)",
                   input=input, rows=input.numel() // max(weight.numel(), 1), cols=weight.numel())
        return orig(input, normalized_shape, weight, bias, eps)
    try:
        if _DEBUG:
            _debug("layer_norm", "te", "", input=input)
        return te_ops.layer_norm(input, normalized_shape, weight, bias, eps)
    except (RuntimeError, TypeError) as exc:
        if _DEBUG:
            _debug("layer_norm", "raised", "TE raised: %r" % (exc,), input=input)
        return orig(input, normalized_shape, weight, bias, eps)


def _patched_rms_norm(input, normalized_shape, weight=None, eps=None):
    orig = _ORIGINALS[(F, "rms_norm")]
    if weight is None or not _usable(input, weight, dtypes=_TE_DTYPES):
        if _DEBUG:
            _debug("rms_norm", "stock",
                   "no affine weight" if weight is None
                   else "not usable: dtype/device/tensor-subclass",
                   input=input)
        return orig(input, normalized_shape, weight, eps)
    _route = _te_norm_route("rms", input, weight) if te_ops.available() else None
    if _route == "triton" and not _is_compiling():
        try:
            if _DEBUG:
                _debug("rms_norm", "te_triton", "", input=input)
            return te_ops.rms_norm(input, normalized_shape, weight, eps, backend="triton")
        except Exception as exc:  # noqa: BLE001 -- Triton raises its own types
            # Disable for the process (e.g. Triton's 2-driver confusion on
            # this AMD+NVIDIA box) rather than paying a failed launch per call.
            global _TE_RMS_TRITON
            _TE_RMS_TRITON = False
            _route = "hip" if input.numel() // max(weight.numel(), 1) >= _TE_NORM_MIN_ROWS else None
            if _DEBUG:
                _debug("rms_norm", "raised", "TE Triton raised, disabled: %r" % (exc,),
                       input=input)
    if _route is None:
        # Same handoff as layer_norm above, through rms_adaln with a zero
        # shift; see _comfy_norm.
        if _grad_safe(input, weight):
            _out = _comfy_norm(
                "rms_norm", input, weight.numel(), False,
                lambda: orig(input, normalized_shape, weight, eps),
                lambda: comfy_kitchen_ops.rms_norm(input, normalized_shape, weight, eps))
            if _out is not None:
                return _out
        if _DEBUG:
            _debug("rms_norm", "stock", "not worth TE (forward-only, too few rows, or too wide)",
                   input=input, rows=input.numel() // max(weight.numel(), 1), cols=weight.numel())
        return orig(input, normalized_shape, weight, eps)
    try:
        if _DEBUG:
            _debug("rms_norm", "te", "", input=input)
        return te_ops.rms_norm(input, normalized_shape, weight, eps)
    except (RuntimeError, TypeError) as exc:
        if _DEBUG:
            _debug("rms_norm", "raised", "TE raised: %r" % (exc,), input=input)
        return orig(input, normalized_shape, weight, eps)


def _patched_gelu(input, approximate="none"):
    orig = _ORIGINALS[(F, "gelu")]
    # tex.gelu computes the tanh approximation only -- patching the default
    # (exact, erf-based) approximate="none" path would silently change
    # numerics for anything expecting exact GELU.
    if approximate != "tanh" or not _usable(input, dtypes=_TE_DTYPES):
        if _DEBUG:
            _debug("gelu", "stock",
                   "approximate=%r (TE computes tanh only)" % (approximate,)
                   if approximate != "tanh" else "not usable: dtype/device/tensor-subclass",
                   input=input)
        return orig(input, approximate=approximate)
    try:
        if _DEBUG:
            _debug("gelu", "te", "", input=input)
        return te_ops.gelu(input)
    except (RuntimeError, TypeError) as exc:
        if _DEBUG:
            _debug("gelu", "raised", "TE raised: %r" % (exc,), input=input)
        return orig(input, approximate=approximate)


def _patched_silu(input, inplace=False):
    orig = _ORIGINALS[(F, "silu")]
    if inplace or not _usable(input, dtypes=_SILU_DTYPES):
        if _DEBUG:
            _debug("silu", "stock",
                   "inplace=True" if inplace else "not usable: dtype/device/tensor-subclass",
                   input=input)
        return orig(input, inplace=inplace)
    try:
        if _DEBUG:
            _debug("silu", "te", "", input=input)
        return te_ops.silu(input)
    except (RuntimeError, TypeError) as exc:
        if _DEBUG:
            _debug("silu", "raised", "TE raised: %r" % (exc,), input=input)
        return orig(input, inplace=inplace)


def _patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                   scale=None, **kwargs):
    orig = _ORIGINALS[(F, "scaled_dot_product_attention")]
    # Cheap guards first: dropout_p/dim/dtype are plain Python attribute
    # reads. te_ops.is_bottom_right_causal_mask (below) is NOT cheap -- it
    # ends in torch.equal(), which forces a GPU sync to pull the boolean
    # result back to host. Checking it before these guards would mean
    # paying for that sync on every training-mode call (dropout_p != 0)
    # only to immediately discard the result and fall back to stock anyway.
    if (dropout_p != 0.0 or query.dim() != 4
            or not _usable(query, key, value, dtypes=_TE_DTYPES)):
        if _DEBUG:
            _debug("sdpa", "stock",
                   "dropout_p != 0" if dropout_p != 0.0
                   else ("query is not 4D" if query.dim() != 4
                         else "not usable: dtype/device/tensor-subclass"),
                   query=query, key=key)
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                     is_causal=is_causal, scale=scale, **kwargs)
    # comfy-kitchen's exact flash decode, before any of TE's gates: this is a
    # DIFFERENT case from the one they describe, and the one they leave on the
    # floor. Everything below routes a TRAINING call; a forward-only call never
    # wins with TE at any sequence length and so has always gone to stock --
    # including single-token decode, which is where PyTorch's own backends are
    # weakest, because one query row cannot occupy this card whichever backend
    # takes it. The guard is cheap (rank, one shape read, dtype) and the tier
    # declines on its first line when the extension is absent.
    if _COMFY and _grad_safe(query, key, value):
        _decoded = comfy_kitchen_ops.sdpa_decode(
            query, key, value, attn_mask, is_causal, scale,
            bool(kwargs.get("enable_gqa", False)))
        if _decoded is not None:
            if _DEBUG:
                _debug("sdpa", "comfy", "single-token decode", query=query, key=key)
            return _decoded
    # Everything below narrows this patch to the ONE case TE measured faster
    # than stock on gfx1100 (see the _TE_ATTN_* block above): a training
    # call, long enough sequence, with an explicit boolean causal mask that
    # disqualifies PyTorch's own flash backend. Each guard is ordered by
    # cost, cheapest first -- is_bottom_right_causal_mask() ends in
    # torch.equal(), which forces a GPU sync, so it must stay last.
    # te_ops.available() is part of this gate, not assumed by it: this wrapper
    # is now installed for comfy-kitchen's sake as well (see enable()), so on a
    # machine with no TE build everything from here down has no backend behind
    # it, and te_ops.is_bottom_right_causal_mask() two guards below would raise
    # outside any try.
    if not (te_ops.available()
            and (_te_grad_live(query, key, value) or _TE_ATTN_UNMASKED)):
        # Forward-only attention never won at any sequence length measured.
        if _DEBUG:
            _debug("sdpa", "stock",
                   "no TransformerEngine build" if not te_ops.available()
                   else "forward-only (no gradient to win in)",
                   query=query, key=key)
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                     is_causal=is_causal, scale=scale, **kwargs)
    _attn_work = query.size(0) * query.size(1) * query.size(2) * key.size(2)
    if _attn_work < _TE_ATTN_MIN_WORK:
        # Too little work to amortize TE's fixed per-call cost on the
        # unfused path (~1.15ms): 0.90x at 8.4M, down to 0.18x on tiny
        # shapes. Sized off the score matrix (batch * heads * q_len *
        # kv_len) rather than sequence length -- same S wins or loses
        # depending on batch and head count.
        if _DEBUG:
            _debug("sdpa", "stock", "too little work to amortize TE's fixed cost",
                   work=_attn_work, min_work=_TE_ATTN_MIN_WORK)
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                     is_causal=is_causal, scale=scale, **kwargs)
    if query.size(-1) < _TE_ATTN_MIN_HEAD_DIM or key.size(1) != query.size(1) \
            or kwargs.get("enable_gqa", False):
        # head_dim 64 lost 0.47-0.81x even with the explicit mask (see
        # _TE_ATTN_MIN_HEAD_DIM); GQA (fewer KV heads) was never measured.
        if _DEBUG:
            _debug("sdpa", "stock", "head_dim below %d or GQA" % _TE_ATTN_MIN_HEAD_DIM,
                   query=query, key=key)
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                     is_causal=is_causal, scale=scale, **kwargs)
    _fp32_unmasked = (_TE_ATTN_FP32_UNMASKED and attn_mask is None and not is_causal
                      and query.dtype == torch.float32)
    if attn_mask is None and not (_TE_ATTN_UNMASKED or _fp32_unmasked):
        # is_causal=True / no mask in fp16/bf16: PyTorch keeps flash here and
        # TE ties or loses (0.89-1.10x unmasked, 0.49-0.72x is_causal).
        if _DEBUG:
            _debug("sdpa", "stock", "no explicit mask: PyTorch keeps flash here",
                   is_causal=is_causal)
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                     is_causal=is_causal, scale=scale, **kwargs)
    # attn_mask is allowed through only when it structurally matches the
    # bottom-right causal pattern (see te_ops.is_bottom_right_causal_mask) --
    # e.g. dflash and HF's sdpa_attention_forward build an explicit boolean
    # mask for causal/KV-cache decoding instead of using is_causal=True.
    # Anything else (padding, sliding-window, genuinely arbitrary masks)
    # still falls back to stock: TE's fused backends don't accept arbitrary
    # tensors, only named attn_mask_type strings.
    mask_ok = attn_mask is None or (
        not is_causal and te_ops.is_bottom_right_causal_mask(attn_mask)
    )
    if not mask_ok:
        if _DEBUG:
            _debug("sdpa", "stock", "attn_mask is not bottom-right causal",
                   mask=attn_mask, is_causal=is_causal)
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                     is_causal=is_causal, scale=scale, **kwargs)
    try:
        if _DEBUG:
            _debug("sdpa", "te", "", query=query, key=key, work=_attn_work)
        return te_ops.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, scale=scale, is_causal=is_causal
        )
    except (RuntimeError, TypeError) as exc:
        if _DEBUG:
            _debug("sdpa", "raised", "TE raised: %r" % (exc,), query=query, key=key)
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                     is_causal=is_causal, scale=scale, **kwargs)


# ---------------------------------------------------------------------------
# Automatic conv-block fusion (nn.Sequential.forward).
#
# WHY A MODULE METHOD AND NOT A FUNCTION. Every other patch in this file
# replaces a torch FUNCTION, and none of them could do this one: the win
# is fusing GroupNorm+SiLU into the convolution's prologue, and by the
# time F.conv2d is called those two have already run and written their
# output to memory. Fusion has to see more than one op, so the patch has
# to sit somewhere that does. nn.Sequential.forward is that place, and it
# is a torch method like any other -- _install/_restore handle it
# unchanged, so disable() unwinds it with everything else.
#
# WHY THIS EXACT PATTERN. Stable Diffusion's ResBlock builds
# nn.Sequential(GroupNorm, SiLU, Conv2d) for both its in_layers and its
# out_layers (sgm/modules/diffusionmodules/openaimodel.py), so matching
# that triple structurally catches every ResBlock convolution in an SDXL
# UNet without knowing anything about the repository that built it, and
# without the caller having to run conv_fusion_ops.replace_conv_blocks on
# a model by hand. Structural match, not a class-name match: a fork that
# renames ResBlock still gets fused, and a Sequential that merely looks
# similar (BatchNorm, or ReLU, or a Conv1d) does not.
#
# COST ON EVERY OTHER Sequential. This runs for EVERY nn.Sequential in the
# process, so the decline path has to be nearly free. It reads
# self._modules (a plain dict) directly rather than going through
# Sequential.__getitem__, and the length check rejects almost everything
# before any isinstance runs. Deliberately NOT cached on the instance: a
# cached decision goes stale if someone swaps a child afterwards, and the
# check is a few hundred nanoseconds against a convolution measured in
# milliseconds.
#
# SAFETY. group_norm_silu runs kernel_select's per-shape contest, so a
# shape where the fused kernel is slower than stock (they exist -- see
# conv_fusion_ops' docstring) selects stock and this patch costs only the
# pattern check. It also declines whenever a gradient is live, so training
# is untouched.
# Bound once at install time rather than looked up per call. Reading
# _ORIGINALS[(nn.Sequential, "forward")] on the decline path builds a tuple
# and hashes a class on EVERY Sequential in the model; measured, that was
# most of this patch's cost on the ops it does not fuse.
_SEQ_FORWARD_ORIG = None


def _patched_sequential_forward(self, input):
    mods = self._modules
    if len(mods) == 3:
        norm, act, conv = mods.values()
        if (type(act) is torch.nn.SiLU
                and isinstance(norm, torch.nn.GroupNorm) and norm.affine
                and isinstance(conv, torch.nn.Conv2d)):
            if _DEBUG:
                _debug("sequential", "conv_fusion", "GroupNorm+SiLU fused",
                       input=input)
            h = conv_fusion_ops.group_norm_silu(
                input, norm.num_groups, norm.weight, norm.bias, norm.eps)
            return conv(h)
    return _SEQ_FORWARD_ORIG(self, input)


# ---------------------------------------------------------------------------
# te_extra_ops patches: Adam/AdamW, gradient-norm, dropout. See the _TE_ADAM
# knob block for the measurements. Each wrapper declines to the stock function
# for anything outside what it was validated on -- the decline list is the
# contract, not an optimization.
# ---------------------------------------------------------------------------

import importlib as _importlib

# `torch.optim.adam` the attribute can be shadowed by the `adam` function in
# some torch versions; the module is what adam() reads _fused_adam /
# _multi_tensor_adam from at call time, so patch it via sys.modules.
_torch_optim_adam = _importlib.import_module("torch.optim.adam")
_torch_clip_grad = _importlib.import_module("torch.nn.utils.clip_grad")

# TE's Adam moments must be fp32 or bf16 (adam.hip dtype checks), and torch
# keeps moments in the param dtype -- so fp16 params (fp16 moments) decline.
_TE_ADAM_DTYPES = (torch.float32, torch.bfloat16)


def _te_adam_tensors_ok(params, grads, device, dtype) -> bool:
    """One pass over params+grads (0.24 ms for 600 tensors, vs ~1 ms for
    all four lists). exp_avg/exp_avg_sq need no check: torch creates them
    with zeros_like(p, memory_format=preserve_format) and load_state_dict
    casts them to the param's device/dtype, so a contiguous param implies
    matching moments -- and TE's own NVTE_CHECKs reject a dtype mismatch
    before launching (caught below as a clean fallback)."""
    for p, g in zip(params, grads):
        if ((type(p) is not torch.Tensor and type(p) is not torch.nn.Parameter)
                or p.device != device or p.dtype is not dtype
                or not p.is_contiguous() or not g.is_contiguous()):
            return False
    return True


_CPU_ONE = torch.tensor(1.0)


def _patched_multi_tensor_adam(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
                               state_steps, grad_scale, found_inf, *, amsgrad,
                               has_complex, beta1, beta2, lr, weight_decay, eps, maximize,
                               capturable, differentiable, decoupled_weight_decay, **extra):
    """torch.optim.adam._multi_tensor_adam -- the foreach path Adam and AdamW
    take by default on CUDA. Only this path is patched: fused=True keeps
    torch's C++ kernel, which measured as fast as TE once the wrapper's
    bookkeeping is counted (and its step counters live on the GPU, so
    reading them here would force a sync every step)."""
    orig = _ORIGINALS[(_torch_optim_adam, "_multi_tensor_adam")]

    def stock(why):
        if _DEBUG:
            _debug("adam", "stock", why, n=len(params))
        return orig(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps,
                    grad_scale, found_inf, amsgrad=amsgrad, has_complex=has_complex,
                    beta1=beta1, beta2=beta2, lr=lr, weight_decay=weight_decay, eps=eps,
                    maximize=maximize, capturable=capturable,
                    differentiable=differentiable,
                    decoupled_weight_decay=decoupled_weight_decay, **extra)

    if not params:
        return stock("empty")
    if extra:
        return stock("unknown kwargs %s" % sorted(extra))
    if amsgrad or maximize or capturable or differentiable or has_complex:
        return stock("amsgrad/maximize/capturable/differentiable/complex")
    if grad_scale is not None or found_inf is not None:
        return stock("grad_scale/found_inf")
    if any(isinstance(v, torch.Tensor) for v in (lr, beta1, beta2, eps, weight_decay)):
        return stock("tensor hyperparameter")
    if _is_compiling():
        return stock("compiling")
    device, dtype = params[0].device, params[0].dtype
    if device.type != "cuda" or dtype not in _TE_ADAM_DTYPES:
        return stock("device/dtype %s %s" % (device, dtype))
    if not state_steps[0].is_cpu:
        return stock("state_steps on GPU (reading them would sync)")
    if not _te_adam_tensors_ok(params, grads, device, dtype):
        return stock("mixed device/dtype, subclass or non-contiguous")

    # torch increments each step THEN uses it; TE takes the post-increment
    # value as a plain int. Compute it without mutating, launch, and only
    # then increment -- so a TE failure falls back with the state untouched.
    groups: dict = {}
    for i, sv in enumerate([s.item() for s in state_steps]):
        groups.setdefault(int(sv) + 1, []).append(i)
    try:
        for step, idx in groups.items():
            if len(idx) == len(params):
                lists = (grads, params, exp_avgs, exp_avg_sqs)
            else:
                lists = tuple([lst[i] for i in idx]
                              for lst in (grads, params, exp_avgs, exp_avg_sqs))
            te_extra_ops.multi_tensor_adam(
                *lists, lr=float(lr), beta1=float(beta1), beta2=float(beta2),
                eps=float(eps), step=step, adam_w_mode=bool(decoupled_weight_decay),
                bias_correction=True, weight_decay=float(weight_decay))
    except (RuntimeError, TypeError) as exc:
        if len(groups) == 1:
            return stock("TE raised: %r" % (exc,))  # nothing applied yet
        raise
    # Same form torch uses for CPU steps (0.3 ms vs 1.2 ms for a scalar 1).
    torch._foreach_add_(state_steps, _CPU_ONE, alpha=1.0)
    if _DEBUG:
        _debug("adam", "te", "", n=len(params), dtype=dtype, steps=len(groups))


_TE_NORM_GRAD_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _patched_get_total_norm(tensors, norm_type=2.0, error_if_nonfinite=False, foreach=None):
    orig = _ORIGINALS[(_torch_clip_grad, "_get_total_norm")]
    ts = [tensors] if isinstance(tensors, torch.Tensor) else list(tensors)
    if not ts or float(norm_type) != 2.0 or foreach is False or _is_compiling():
        return orig(ts, norm_type, error_if_nonfinite, foreach)
    device, dtype = ts[0].device, ts[0].dtype
    if device.type != "cuda" or dtype not in _TE_NORM_GRAD_DTYPES or not all(
            _is_plain_tensor(t) and t.device == device and t.dtype == dtype
            and t.layout == torch.strided and t.is_contiguous() for t in ts):
        if _DEBUG:
            _debug("grad_norm", "stock", "mixed/unsupported tensors", n=len(ts))
        return orig(ts, norm_type, error_if_nonfinite, foreach)
    try:
        with torch.no_grad():
            norm, _ = te_extra_ops.multi_tensor_l2norm(ts)
    except (RuntimeError, TypeError) as exc:
        if _DEBUG:
            _debug("grad_norm", "raised", "TE raised: %r" % (exc,), n=len(ts))
        return orig(ts, norm_type, error_if_nonfinite, foreach)
    # Same shape (0-d) and dtype stock returns; TE accumulates in fp32, so
    # for bf16/fp16 grads this is the more accurate number, then rounded.
    total = norm.reshape(()).to(dtype)
    if error_if_nonfinite and torch.logical_or(total.isnan(), total.isinf()):
        raise RuntimeError(
            f"The total norm of order {float(norm_type)} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`")
    if _DEBUG:
        _debug("grad_norm", "te", "", n=len(ts), dtype=dtype)
    return total


_TE_DROPOUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _patched_dropout(input, p=0.5, training=True, inplace=False):
    orig = _ORIGINALS[(F, "dropout")]
    # TE thresholds an 8-bit random byte at floor(p*256) but rescales by
    # 1/(1-p): unless p*256 is an integer the expected output is biased
    # (p=0.1 -> keeps 90.23%, +0.26%). Only exact p (0.5, 0.25, 0.125, ...)
    # is routed, so the result is distributionally identical to stock.
    if (not training or inplace or not isinstance(p, float) or not 0.0 < p < 1.0
            or p * 256 != math.floor(p * 256)):
        if _DEBUG:
            _debug("dropout", "stock", "eval/inplace/p not a multiple of 1/256", p=p)
        return orig(input, p, training, inplace)
    if (not _usable(input, dtypes=_TE_DROPOUT_DTYPES) or input.numel() == 0
            or _is_compiling() or input.device.index != torch.cuda.current_device()):
        # TE draws from the CURRENT device's default generator.
        if _DEBUG:
            _debug("dropout", "stock", "not usable / compiling / not current device", input=input)
        return orig(input, p, training, inplace)
    try:
        if _DEBUG:
            _debug("dropout", "te", "", input=input, p=p)
        return te_extra_ops.dropout(input, p, training=True)
    except (RuntimeError, TypeError) as exc:
        if _DEBUG:
            _debug("dropout", "raised", "TE raised: %r" % (exc,), input=input)
        return orig(input, p, training, inplace)


def enable() -> None:
    """Patch torch/torch.nn.functional to route eligible calls through amd_tuned_torch."""
    global _ENABLED
    if _ENABLED:
        if _DEBUG:
            _debug("enable", "noop", "already enabled")
        return
    if _DEBUG:
        # Which BACKENDS are present decides which patches get installed at
        # all, and an op that was never patched produces no routing lines
        # below -- so an empty trace for, say, F.linear means this line,
        # not a per-call decline. Logged once, at enable() time.
        _debug("enable", "backends", "",
               aiter=aiter_ops.available(), hipblaslt=hipblaslt_ops.available(),
               ck_gemm=ck_gemm_ops.available(), ck=ck_ops.available(),
               te=te_ops.available(), kernel_select=kernel_select.enabled(),
               # Reports WHY when absent, not merely that it is: the HIP
               # extension is a separate build step and "not built" reads very
               # differently from "this GPU is not in the target manifest".
               comfy_kitchen=_COMFY or comfy_kitchen_ops.unavailable_reason())

    # linear/matmul/bmm. Installed when ANY of the three GEMM candidates is
    # available -- aiter's Triton kernels, hipBLASLt, or the CK WMMA GEMM
    # tier -- since each of the three patches runs a contest that includes
    # stock and will simply pick stock if nothing else can serve the shape.
    #
    # This used to be gated on aiter alone, which was right when aiter was
    # the only candidate and wrong the moment it stopped being. On gfx1100
    # aiter_ops.available() is False (it ships no RDNA3 tuning config), so
    # that gate silently withheld the patch entirely -- and with it
    # hipBLASLt's 1.29x and CK's 1.21x -- long after there was something to
    # gain. A capability check has to name the capability it is checking
    # for, not one dependency that used to imply it.
    if aiter_ops.available() or hipblaslt_ops.available() or ck_gemm_ops.available():
        _install(F, "linear", _patched_linear)
        _install(torch, "matmul", _patched_matmul)
        _install(torch, "bmm", _patched_bmm)
        _install(torch, "einsum", _patched_einsum)

    # Native HIP kernels (see src/main_rocm.cpp). Always installed -- the
    # native extension is hard-required at import time (see the ImportError
    # guard above), unlike the aiter-gated and TE-gated patches. conv2d/conv3d
    # try the native kernel first (fp16/fp32; see _patched_conv2d/
    # _patched_conv3d for the tier order and fallbacks).
    _install(F, "group_norm", _patched_group_norm)
    _install(F, "conv2d", _patched_conv2d)

    # Auto-fuse Stable Diffusion's GroupNorm->SiLU->Conv2d blocks. Gated on
    # conv_fusion_ops.enabled() (Triton + a GPU, and
    # AMD_TUNED_TORCH_CONV_FUSION != 0) so a machine without Triton never
    # pays the pattern check, and so this is one switch to turn off.
    if conv_fusion_ops.enabled():
        global _SEQ_FORWARD_ORIG
        # Take the original from _ORIGINALS, which _install populates only
        # the FIRST time a target is patched, rather than reading
        # nn.Sequential.forward here. Reading it directly captures the
        # PATCHED function if this runs while the patch is already
        # installed, and _patched_sequential_forward then calls itself:
        # infinite recursion on every Sequential in the process. enable()
        # guards against re-entry with _ENABLED, but that guard is one
        # early-return away from being bypassed and this failure mode is
        # far too loud to leave resting on it.
        _install(torch.nn.Sequential, "forward", _patched_sequential_forward)
        _SEQ_FORWARD_ORIG = _ORIGINALS[(torch.nn.Sequential, "forward")]
    if hasattr(F, "conv3d"):
        _install(F, "conv3d", _patched_conv3d)

    # TransformerEngine-backed (see amd_tuned_torch/te_ops.py). Skipped entirely if
    # TE isn't installed -- these ops just stay on stock PyTorch/ROCm.
    #
    # Which of the five are installed, and the per-call shape/mode guards
    # inside each wrapper, both follow tools/bench_te_ops.py's fwd+bwd
    # measurements -- see the _TE_* constants block near the top of this
    # file for the numbers and the reasoning.
    #
    # F.layer_norm IS now installed (it was not before). The old exclusion
    # cited benchmark.json's 0.71x fp16 / 0.61x fp32, but those are
    # FORWARD-ONLY figures; on the fwd+bwd path layer_norm is the same kind
    # of win as rms_norm on tall inputs (1.06x-2.07x at >= 16384 rows), and
    # _te_norm_worth_it() now declines the shapes where the old number was
    # right instead of declining the op outright.
    #
    # The norm and attention wrappers are installed when EITHER backend is
    # present, because they now cover two disjoint cases: TE takes the
    # training-time shapes it measured wins on, and comfy-kitchen's
    # inference-only kernels take the forward-only ones TE declines by design
    # (see _comfy_norm and the decode branch in _patched_sdpa). Gating them on
    # TE alone would have withheld the second from a machine with no TE build
    # -- the same mistake the aiter-only gate above made for the GEMM patches,
    # for the same reason: the check has to name the capability, not one
    # backend that used to be the only way to have it.
    if te_ops.available() or _COMFY:
        _install(F, "layer_norm", _patched_layer_norm)
        if hasattr(F, "rms_norm"):
            _install(F, "rms_norm", _patched_rms_norm)
        if hasattr(F, "scaled_dot_product_attention"):
            _install(F, "scaled_dot_product_attention", _patched_sdpa)
    if te_ops.available():
        # gelu/silu: never faster than stock (0.87x-0.99x fwd+bwd, 0.68x
        # forward). A tier that can only lose does not belong on by default
        # -- same standard already applied to flash_attn_rocwmma in
        # __init__.py. Set AMD_TUNED_TORCH_TE_ACTIVATIONS=1 to install them
        # anyway, e.g. to re-measure after a TE rebuild. comfy-kitchen has no
        # activation kernels, so this stays TE-only.
        if _TE_ACTIVATIONS:
            _install(F, "gelu", _patched_gelu)
            _install(F, "silu", _patched_silu)
        # te_extra_ops kernels -- see the _TE_ADAM knob block for numbers.
        if _TE_ADAM and hasattr(_torch_optim_adam, "_multi_tensor_adam"):
            _install(_torch_optim_adam, "_multi_tensor_adam", _patched_multi_tensor_adam)
        if _TE_GRAD_NORM and hasattr(_torch_clip_grad, "_get_total_norm"):
            _install(_torch_clip_grad, "_get_total_norm", _patched_get_total_norm)
            # torch.nn.utils.get_total_norm is the same function re-exported.
            if getattr(torch.nn.utils, "get_total_norm", None) is _ORIGINALS[
                    (_torch_clip_grad, "_get_total_norm")]:
                _install(torch.nn.utils, "get_total_norm", _patched_get_total_norm)
        if _TE_DROPOUT:
            _install(F, "dropout", _patched_dropout)

    _ENABLED = True
    if _DEBUG:
        _debug("enable", "installed", "",
               patched=",".join(sorted(name for _, name in _ORIGINALS)))


def disable() -> None:
    """Restore every patched function to the stock PyTorch implementation."""
    global _ENABLED
    if _DEBUG:
        _debug("disable", "restored", "",
               patched=",".join(sorted(name for _, name in _ORIGINALS)))
    for target, name in list(_ORIGINALS.keys()):
        _restore(target, name)
    _ENABLED = False


def is_enabled() -> bool:
    return _ENABLED


# ---------------------------------------------------------------------------
# TorchScript compatibility.
#
# THE PROBLEM. torch.jit.script compiles the Python SOURCE of whatever it
# finds behind a name, so `torch.bmm(a, b)` inside a scripted function
# compiles whatever `torch.bmm` is bound to at that moment. Once enable()
# has run that is a wrapper of ours, and none of these wrappers are
# scriptable -- nor should they be. TorchScript rejects each for a
# different reason:
#
#     torch.bmm, torch.matmul   keyword-only argument with a default (out=None)
#     F.linear, F.group_norm    lambdas (the kernel_select candidate thunks)
#     F.conv2d                  try blocks
#
# and the ones it does not reject it would compile wrongly, since the body
# reaches into a dict keyed by module objects and calls into a pybind
# extension. This is not a defect in any single wrapper; it is what
# monkeypatching costs, and no rewriting of the wrappers fixes it.
#
# It is also not new. F.conv2d has been unscriptable since it was first
# patched -- it simply went unnoticed, because a library has to script code
# that calls a patched op before anything breaks. kornia scripts
# kornia.geometry.boxes at import time and that calls torch.bmm, which
# stayed harmless only for as long as enable() declined to install the
# torch.bmm patch (it was gated on aiter, absent on gfx1100). Fixing that
# gate turned a latent incompatibility into an import-time crash for every
# stable-diffusion-webui launch.
#
# THE FIX. Compile against stock. torch.jit.script and torch.jit.trace are
# wrapped so the patches are lifted for the duration of the call and
# restored after, which makes the compiler bind to the real aten ops. The
# cost is that code inside a scripted or traced region runs stock kernels
# rather than ours -- correct, and the only honest option, since a
# TorchScript graph cannot call a Python dispatch wrapper anyway.
#
# Tracing needs this at least as much as scripting, and more quietly:
# torch.jit.trace records the aten ops a function executes, and these
# wrappers execute pybind calls that are not traceable ops. A trace taken
# with the patches live would bake in a constant instead of a computation
# and be silently wrong, where scripting at least fails loudly.
#
# The guard is installed at import time, not by enable(), because the
# breakage is caused by the patches existing rather than by anyone opting
# in -- and because this module is auto-loaded before user code (see the
# sitecustomize hook), so libraries that script at import time are already
# covered.
_ORIG_JIT_SCRIPT = torch.jit.script
_ORIG_JIT_TRACE = torch.jit.trace


def _script_with_stock_ops(obj, *args, **kwargs):
    """torch.jit.script, compiled against stock ops.

    _frames_up IS NOT OPTIONAL BOOKKEEPING. When the object being scripted
    is a class, torch's _script_impl builds its name-resolution callback
    with createResolutionCallbackFromFrame(_frames_up + 1) -- it walks a
    fixed number of stack frames up to find the caller's module globals.
    This wrapper is one extra frame, so without compensating, every name a
    scripted class refers to is looked up in amd_tuned_torch's namespace
    instead of the caller's. That surfaces as `undefined value <helper>` on
    a perfectly ordinary module-level function, which is how kornia's
    Boxes3D failed the first time this guard was written. Functions resolve
    through createResolutionCallbackFromClosure and are unaffected, which
    is exactly why the bug hides until something scripts a class.

    The adjustment is unconditional: this frame exists whether or not the
    patches were active.
    """
    if len(args) >= 2:
        # positional: script(obj, optimize, _frames_up, ...)
        args = args[:1] + (args[1] + 1,) + args[2:]
    else:
        kwargs["_frames_up"] = kwargs.get("_frames_up", 0) + 1

    if not _ENABLED:
        return _ORIG_JIT_SCRIPT(obj, *args, **kwargs)
    disable()
    try:
        return _ORIG_JIT_SCRIPT(obj, *args, **kwargs)
    finally:
        enable()


def _trace_with_stock_ops(*args, **kwargs):
    """torch.jit.trace, recorded against stock ops.

    No frame compensation needed here: tracing executes the function and
    records the aten ops it runs, rather than resolving names out of the
    caller's namespace.
    """
    if not _ENABLED:
        return _ORIG_JIT_TRACE(*args, **kwargs)
    disable()
    try:
        return _ORIG_JIT_TRACE(*args, **kwargs)
    finally:
        enable()


_script_with_stock_ops.__doc__ = _ORIG_JIT_SCRIPT.__doc__
_trace_with_stock_ops.__doc__ = _ORIG_JIT_TRACE.__doc__
torch.jit.script = _script_with_stock_ops
torch.jit.trace = _trace_with_stock_ops


# Re-exported at the package top level via `from ._dispatch import *` in
# __init__.py -- includes underscore-prefixed names because tests and other
# submodules already reach into them directly (amd_tuned_torch._grad_safe,
# amd_tuned_torch._usable, amd_tuned_torch._ORIGINALS, etc.), and this split
# must not change what amd_tuned_torch.X resolves to for any X below.
__all__ = [
    "ops",
    "_dbg_log", "_DEBUG", "_DEBUG_ENV_SET", "_DEBUG_FILE", "_ROUTE_COUNTS",
    "_debug", "_dbg_t",
    "dispatch_debug_enabled", "dispatch_debug_counts",
    "dispatch_debug_reset", "dispatch_debug_report",
    "_GEMM_DTYPES", "_GROUPNORM_DTYPES", "_TE_DTYPES", "_CONV_NATIVE_DTYPES",
    "_CK_CONV_DTYPES", "_FFTCONV_CONV_DTYPES",
    "_FFTCONV_CONV2D_ENABLED", "_FFTCONV_CONV3D_ENABLED",
    "_FFTCONV_CONV2D_MIN_KERNEL", "_FFTCONV_CONV3D_MIN_KERNEL",
    "_fftconv_calibration_data", "_calibrated_fftconv_default",
    "_FFTCONV_CONV3D_MIN_POSITIONS", "_SILU_DTYPES",
    "_TE_NORM_MIN_ROWS", "_TE_NORM_MAX_COLS", "_TE_NORM_INFERENCE",
    "_TE_ACTIVATIONS", "_TE_ATTN_MIN_WORK", "_TE_ATTN_UNMASKED",
    "_COMFY", "_COMFY_KIND", "_LINEAR_KIND", "_CONV3D_KIND",
    "_comfy_norm",
    "_te_grad_live", "_te_norm_worth_it",
    "_ORIGINALS", "_ENABLED", "_HAS_INFERENCE_MODE",
    "_is_compiling", "_grad_safe", "_usable", "_install", "_restore", "_triple",
    "_LinearFn", "_BmmFn", "_patched_linear", "_linear_aiter",
    "_patched_matmul", "_patched_bmm", "_bmm_aiter", "_patched_einsum", "_EINSUM_DTYPES", "_EINSUM_MEMO", "einsum_memo_clear",
    "_is_pointwise_conv2d", "_fftconv_conv_candidate", "_fftconv_contest_tolerance",
    "_Conv2dFn", "_conv2d_accelerated_forward",
    "_conv2d_backward_data", "_conv2d_backward_weight",
    "_conv2d_backward_data_stock", "_conv2d_backward_weight_stock",
    "_patched_conv2d",
    "_Conv3dFn", "_conv3d_accelerated_forward", "_patched_conv3d",
    "_group_norm_native", "_GroupNormFn", "_group_norm_backward",
    "_group_norm_accelerated_forward", "_patched_group_norm",
    "_patched_layer_norm", "_patched_rms_norm", "_patched_gelu", "_patched_silu",
    "_patched_sdpa",
    "enable", "disable", "is_enabled",
    "_ORIG_JIT_SCRIPT", "_ORIG_JIT_TRACE",
    "_script_with_stock_ops", "_trace_with_stock_ops",
]
