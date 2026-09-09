"""Finds the crossover for amd_tuned_torch's FFT-conv3d min-positions gate
and saves it as this GPU's calibration (amd_tuned_torch/fftconv_calibration.py)
-- replacing a hardcoded, explicitly-documented-as-unvalidated guess
(`__init__.py`'s _FFTCONV_CONV3D_MIN_POSITIONS, default 2048) with a number
actually measured on this GPU.

WHY THIS EXISTS. amd_tuned_torch._fftconv_conv_candidate declines a conv3d
call outright when `input`'s spatial size (batch * D*H*W) is below
_FFTCONV_CONV3D_MIN_POSITIONS, before ever even considering kernel width --
the reasoning (see that function's own docstring) is that FFT-conv3d's
padded-transform overhead (allocating and transforming a signal roughly
double the input's own size, in complex64) can't be beaten by a cheap
direct conv3d below some volume size, no matter how favorable the kernel.
That threshold was never measured, only guessed, the same gap
tools/benchmark_sparse_conv.py already closed for flexgemm_ops's
sparse/dense switch (see that script's own docstring) -- this is the same
idea applied to the FFT-conv3d gate instead.

METHOD. Fixes kernel width at one deliberately FAVORABLE value (default
15 -- see --kernel-width; the measured table in
amd_tuned_torch._fftconv_conv_candidate's own docstring shows 15^3 winning
FFT-conv3d by 178x at a 1x32x32x64x64 input, the strongest measured signal
available, versus 7^3's marginal 1.1x), then sweeps INPUT SIZE (spatial
positions, ~cube-shaped) to find the smallest size where FFT-conv3d still
wins against stock F.conv3d, and keeps winning for every larger size
measured (a single lucky win with a loss right above it is noise, not a
trend -- same discipline benchmark_sparse_conv.py's own size sweep uses).
Times both with CUDA events (same warmup/median-of-N shape as
amd_tuned_torch.kernel_select's own _time).

WHY A SEPARATE SCRIPT, NOT A tools/benchmark_sparse_conv.py FLAG. Different
subject entirely (FFT-conv3d vs. flexgemm's sparse conv3d), different
calibration file (fftconv_calibration.py, not sparse_conv_calibration.py),
and no shared code path between the two switches in flexgemm_ops.py/
__init__.py -- folding this in would just be two unrelated sweeps under one
argument parser. Kept as its own small script instead, same posture
tools/probe_fftconv_max_size.py already has for a different FFT-conv
question (its own memory ceiling, not this crossover).

Run with:

    python tools/benchmark_fftconv3d_min_positions.py
    python tools/benchmark_fftconv3d_min_positions.py --kernel-width 7
    python tools/benchmark_fftconv3d_min_positions.py --sizes 512 1000 2048 4096 8192
    python tools/benchmark_fftconv3d_min_positions.py --reset

This does NOT run automatically as part of any build step -- needs a real
GPU and real wall-clock time, same posture as tools/benchmark_sparse_conv.py
(see setup.py's own comment above BenchmarkSparseConv for why).
"""
from __future__ import annotations

import os

# Must precede `import torch` -- MIOpen reads these once, at
# initialization (which a ROCm torch build can trigger as a side effect of
# `import torch` itself), not per-call -- see
# amd_tuned_torch.benchmark_report.MIOPEN_LOGGING_ENV's own comment for why
# this has to be inlined here rather than imported from that module.
# Values match tools/bench_conv2d_fp16.py's own (see analyse/README.md).
# The dense F.conv3d candidate this script measures against fft_conv3d
# reaches MIOpen the normal way, so this makes its solver choice visible
# too, same as it already does for tools/benchmark_sparse_conv.py.
os.environ.setdefault("MIOPEN_ENABLE_LOGGING", "1")
os.environ.setdefault("MIOPEN_ENABLE_LOGGING_CMD", "1")
os.environ.setdefault("MIOPEN_LOG_LEVEL", "6")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import List, Optional, Tuple  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

try:
    from amd_tuned_torch import benchmark_report, fftconv_calibration, fftconv_ops
except ImportError:
    # None of these three modules has a compiled-extension dependency at all
    # (fftconv_ops: pure torch.fft/pad/kron, see its own module docstring;
    # fftconv_calibration/benchmark_report: plain JSON I/O) -- if the REST of
    # amd_tuned_torch can't import here (no native extension built for this
    # checkout/platform), load all three files directly rather than require a
    # full package build just to run this benchmark, same fallback
    # tools/probe_fftconv_max_size.py already uses for the same reason.
    def _load(name: str):
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).resolve().parent.parent / "amd_tuned_torch" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    fftconv_ops = _load("fftconv_ops")
    fftconv_calibration = _load("fftconv_calibration")
    benchmark_report = _load("benchmark_report")

_WARMUP = 3
_ITERS = 5
_C_IN, _C_OUT = 4, 8

# Spatial-position sizes to sweep, as powers of 2 -- from a small patch up
# to comfortably past where FFT-conv3d is expected to win at a favorable
# kernel width (see amd_tuned_torch._fftconv_conv_candidate's own
# measurements, taken at a 1x32x32x64x64 = 131,072-position input).
_DEFAULT_SIZES = [2 ** p for p in range(6, 19)]  # 64 .. 262,144
_DEFAULT_KERNEL_WIDTH = 15


def _cube_shape(n_positions: int) -> Tuple[int, int, int]:
    """n_positions spatial positions, shaped as close to a cube as integer
    rounding allows -- the exact aspect ratio doesn't matter for this
    measurement, only the total position count does (same reasoning
    tools/benchmark_sparse_conv.py's own _make_shape uses)."""
    side = max(1, round(n_positions ** (1 / 3)))
    return (side, side, side)


def _time_ms(fn, device: torch.device) -> Optional[float]:
    """None if `fn` raises -- same "can't measure, so it can't win"
    convention as kernel_select._time / benchmark_sparse_conv._time_ms."""
    is_cuda = device.type == "cuda"
    try:
        for _ in range(_WARMUP):
            fn()
        if is_cuda:
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(_ITERS):
                fn()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / _ITERS
        import time
        t0 = time.perf_counter()
        for _ in range(_ITERS):
            fn()
        return (time.perf_counter() - t0) * 1000 / _ITERS
    except (RuntimeError, TypeError, ValueError):
        return None


def _sweep(sizes: List[int], kernel_width: int, device: torch.device, dtype: torch.dtype) -> dict:
    results = []
    weight = torch.randn(_C_OUT, _C_IN, kernel_width, kernel_width, kernel_width,
                          device=device, dtype=dtype)
    padding = kernel_width // 2

    for n_positions in sizes:
        d, h, w = _cube_shape(n_positions)
        x = torch.randn(1, _C_IN, d, h, w, device=device, dtype=dtype)
        actual_n = d * h * w

        dense_ms = _time_ms(lambda: F.conv3d(x, weight, padding=padding), device)
        fft_ms = _time_ms(lambda: fftconv_ops.fft_conv3d(x, weight, padding=padding), device)
        won = fft_ms is not None and dense_ms is not None and fft_ms < dense_ms
        results.append({
            "n_positions": actual_n, "shape": [d, h, w],
            "dense_ms": dense_ms, "fftconv_ms": fft_ms, "fftconv_won": won,
        })
        dense_s = f"{dense_ms:.3f}" if dense_ms is not None else "None"
        fft_s = f"{fft_ms:.3f}" if fft_ms is not None else "None"
        winner = "fftconv" if won else "dense"
        print(f"  n={actual_n:>10,}  shape={d}x{h}x{w:<4}  "
              f"dense={dense_s:>10}  fftconv={fft_s:>10}  winner={winner}")

    # Crossover: smallest size where fftconv wins AND stays winning for
    # every larger size measured -- scan from the end so a late, permanent
    # win is found even if smaller sizes are noisy in both directions (same
    # discipline benchmark_sparse_conv.py's _sweep_size uses).
    crossover = None
    for i in range(len(results) - 1, -1, -1):
        if not results[i]["fftconv_won"]:
            break
        crossover = results[i]["n_positions"]
    return {"min_positions": crossover, "sweep": results}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kernel-width", type=int, default=_DEFAULT_KERNEL_WIDTH,
                         help=f"Fixed, favorable cubic kernel width the sweep measures at "
                              f"(default {_DEFAULT_KERNEL_WIDTH}).")
    parser.add_argument("--sizes", type=int, nargs="+", default=None,
                         help="Override the default power-of-2 spatial-position sweep.")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp32",
                         help="Default fp32 -- matches the measured table in "
                              "_fftconv_conv_candidate's own docstring.")
    parser.add_argument("--reset", action="store_true",
                         help="Delete this GPU's saved calibration and exit without measuring.")
    args = parser.parse_args()

    if args.reset:
        fftconv_calibration.reset()
        print(f"Reset calibration for {fftconv_calibration.device_key()} "
              f"-- amd_tuned_torch will use its hardcoded default until this is run again.")
        # Re-export so the combined report stops carrying this now-deleted
        # domain's stale numbers.
        benchmark_report.save()
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA/ROCm device visible -- timing on CPU. This measures relative "
              "Python/FFT-library overhead, not the GPU kernel crossover this script exists to "
              "find; treat any resulting calibration as provisional and re-run on the real "
              "target GPU before trusting it.", file=sys.stderr)

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    sizes = args.sizes or _DEFAULT_SIZES
    print(f"Device: {fftconv_calibration.device_key()}")
    print(f"Kernel width: {args.kernel_width}, dtype: {args.dtype}, sizes: {sizes}")
    print()

    print("=== conv3d ===")
    # MIOpen's own solver-selection log for the whole sweep below -- the
    # dense F.conv3d candidate this measures against fft_conv3d reaches
    # MIOpen the normal way. Only produces anything if MIOPEN_ENABLE_LOGGING
    # etc. actually took effect (a ROCm build with MIOpen; harmless empty
    # capture otherwise) -- see
    # amd_tuned_torch.benchmark_report.CaptureMiopenLog's own docstring.
    with benchmark_report.CaptureMiopenLog() as miopen_cap:
        outcome = _sweep(sizes, args.kernel_width, device, dtype)
    if miopen_cap.text:
        log_path = benchmark_report.save_miopen_log("fftconv", miopen_cap.text)
        print(f"MIOpen debug log for this run saved to {log_path}")
    if outcome["min_positions"] is not None:
        print(f"  -> min_positions crossover: {outcome['min_positions']:,} spatial positions")
        fftconv_calibration.save(
            {"conv3d": {"min_positions": outcome["min_positions"]}},
            extra_metadata={"conv3d": {
                "min_positions_sweep": {
                    "kernel_width": args.kernel_width, "dtype": args.dtype,
                    "sweep": outcome["sweep"],
                }
            }},
        )
        print(f"Saved to {fftconv_calibration.calibration_path()}")
    else:
        print("  -> FFT-conv3d never won a stable trend at this kernel width; "
              "leaving min_positions uncalibrated. Nothing was written; existing "
              "calibration (if any) is untouched. Try a larger --kernel-width or "
              "larger --sizes.")

    # Re-export the combined report (this calibration file plus whatever
    # sparse_conv_calibration/others have already written) regardless of
    # whether this run calibrated anything new -- see
    # amd_tuned_torch/benchmark_report.py's own docstring for why this is a
    # full re-export every time, not a partial merge.
    report_path = benchmark_report.save()
    print(f"Combined benchmark report (all calibration domains + system info) "
          f"saved to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
