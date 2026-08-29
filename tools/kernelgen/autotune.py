"""Benchmarks stock PyTorch conv2d/conv3d against amd_tuned_torch's native
kernels -- the codegen'd WMMA implicit-GEMM kernels
(src/cuda/generated/conv{2,3}d_fp16_*.cu), and for eligible conv3d shapes
the opt-in Winograd kernel (conv3d_fp16_winograd_bt8_bc8.cu) -- across
tools/kernelgen/shapes.py's corpus of real/representative input
dimensions (some sourced from a real MIOpen trace, miopen_amd_log.txt --
see shapes.py's own header), prints a report, and writes
tools/kernelgen/tuned_shapes.json.

Also reports each conv2d_fp16 variant's static resource usage (VGPR
count, LDS) and achievable occupancy (hipFuncGetAttributes /
hipOccupancyMaxActiveBlocksPerMultiprocessor, queried via
conv2d_fp16_variant_diagnostics -- see src/cuda/templates/
conv2d_fp16.cu.tmpl's conv2d_fp16_diagnostics_<suffix> for what that
actually reads off the compiled kernel), and which variant
run_conv2d_fp16's live dispatch actually picked for each shape
(conv2d_fp16_cached_variant) -- so the printed report connects "this
candidate won" to its occupancy profile, not just a bare timing number.
This is diagnostic, not a second selection criterion: run_conv2d_fp16
already picks on measured wall-clock time, which already reflects
whatever penalty low occupancy causes -- these numbers explain *why* a
winner won (or flag a surprise worth a second look), and inform which
new BM/BN/BK/STAGES combinations are worth generating and benchmarking
in the first place.

This does NOT automatically wire results into
amd_tuned_torch/__init__.py's dispatch. That file's conv2d/conv3d tiers
are static, pre-benchmarked rules -- see _is_pointwise_conv2d there for
the existing precedent this follows: real numbers folded into a docstring
plus a shape-class predicate, not a runtime lookup table read on every
forward call. Read this script's printed report (and tuned_shapes.json)
and update amd_tuned_torch/__init__.py by hand -- exactly the workflow
that produced _is_pointwise_conv2d in the first place. conv3d fp16
Winograd specifically stays behind its own explicit opt-in
(amd_tuned_torch.enable_conv3d_winograd_fp16()) regardless of what this
script reports, until tests_hardware/test_conv_kernels.py (or an
equivalent check) has also confirmed correctness -- speed alone doesn't
establish that a numerically-unvalidated kernel is safe to make a default.

Requires a real CUDA/ROCm device and a built amd_tuned_torch._native
extension. Run with:

    python tools/kernelgen/autotune.py
"""
import dataclasses
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shapes import CONV2D_SHAPES, CONV3D_SHAPES  # noqa: E402
from variants import variants_for  # noqa: E402

import amd_tuned_torch  # noqa: E402
from amd_tuned_torch import compile_ops  # noqa: E402

# amd_tuned_torch._is_winograd_eligible_conv3d is private (leading
# underscore) but deliberately reused here rather than duplicated: it's
# the single source of truth for "is this shape in Winograd's scope",
# shared with amd_tuned_torch.enable_conv3d_winograd_fp16's dispatch --
# re-implementing the same guard here would risk the two drifting apart.

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = Path(__file__).resolve().parent / "tuned_shapes.json"

device = torch.device("cuda")
DTYPE = torch.float16

N_WARMUP = 10
N_ITER = 50


def _time_ms(fn, n_warmup=N_WARMUP, n_iter=N_ITER):
    """Same timing idiom as tools/bench.py's benchmark_op: warm up, then
    time n_iter back-to-back launches with torch.cuda.Event. Returns
    (None, reason) if fn raises during warmup -- expected/normal for the
    Winograd candidate on an out-of-scope shape, not just an error case."""
    try:
        for _ in range(n_warmup):
            fn()
    except (RuntimeError, TypeError) as exc:
        return None, str(exc)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter, None


def _conv_out_size(in_size, kernel, stride, padding, dilation):
    """Mirrors src/main_rocm.cpp's conv_out_size() exactly -- needed here
    (not just read off an output tensor's .shape) so
    conv2d_fp16_cached_variant can be queried with the same shape key
    run_conv2d_fp16 used to cache its dispatch decision, without an extra
    kernel launch just to inspect a real output tensor's shape."""
    return (in_size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def report_conv2d_fp16_variant_diagnostics():
    """Static per-variant diagnostics (VGPR count, LDS, achievable
    occupancy) -- properties of the compiled kernel, not the input shape,
    so printed once up front rather than repeated per shape below. See
    conv2d_fp16_variant_diagnostics's own docstring for what's queried;
    purely introspective, doesn't launch anything."""
    variants = variants_for("conv2d_fp16")
    print("=" * 70)
    print("conv2d_fp16 variant diagnostics (static, shape-independent)")
    print("=" * 70)
    diagnostics = []
    for idx, variant in enumerate(variants):
        num_regs, static_lds, dynamic_lds, max_active_blocks = (
            amd_tuned_torch.ops.conv2d_fp16_variant_diagnostics(idx)
        )
        print(f"[{idx}] {variant.suffix}  (params={variant.params})")
        print(f"  VGPRs                : {num_regs}")
        print(f"  static LDS           : {static_lds} B")
        print(f"  dynamic LDS          : {dynamic_lds} B ({dynamic_lds / 1024:.1f} KB)")
        print(f"  max active blocks/CU : {max_active_blocks}")
        if max_active_blocks == 0:
            print("  WARNING: 0 active blocks/CU -- this variant cannot launch at all "
                  "(exceeds VGPR/LDS budget for even one resident block on this GPU).")
        print()
        diagnostics.append({
            "variant_idx": idx,
            "suffix": variant.suffix,
            "params": variant.params,
            "num_regs": num_regs,
            "static_lds_bytes": static_lds,
            "dynamic_lds_bytes": dynamic_lds,
            "max_active_blocks_per_cu": max_active_blocks,
        })
    return diagnostics


def bench_conv2d(shape):
    x = torch.randn(shape.B, shape.C_in, shape.H_in, shape.W_in, device=device, dtype=DTYPE)
    w = torch.randn(shape.C_out, shape.C_in, shape.K, shape.K, device=device, dtype=DTYPE)
    b = torch.randn(shape.C_out, device=device, dtype=DTYPE)
    stride, padding, dilation = [shape.stride] * 2, [shape.padding] * 2, [shape.dilation] * 2

    stock_ms, _ = _time_ms(lambda: F.conv2d(x, w, b, stride=stride, padding=padding, dilation=dilation))
    native_ms, native_err = _time_ms(lambda: amd_tuned_torch.ops.conv2d(x, w, b, stride, padding, dilation))

    # After the timed run above, run_conv2d_fp16's live dispatch has
    # already cached a winner for this exact shape -- peek at which one
    # (pure lookup, no extra benchmark) so the report can show that
    # variant's occupancy profile next to the timing, not just the bare
    # "native" number.
    variant_idx = None
    variant_diagnostics = None
    if native_err is None:
        H_out = _conv_out_size(shape.H_in, shape.K, shape.stride, shape.padding, shape.dilation)
        W_out = _conv_out_size(shape.W_in, shape.K, shape.stride, shape.padding, shape.dilation)
        variant_idx = amd_tuned_torch.ops.conv2d_fp16_cached_variant(
            shape.B, shape.C_in, shape.H_in, shape.W_in,
            shape.C_out, shape.K, shape.K, H_out, W_out,
            shape.stride, shape.stride, shape.padding, shape.padding,
            shape.dilation, shape.dilation,
        )
        if variant_idx is not None:
            num_regs, static_lds, dynamic_lds, max_active_blocks = (
                amd_tuned_torch.ops.conv2d_fp16_variant_diagnostics(variant_idx)
            )
            variant_diagnostics = {
                "num_regs": num_regs,
                "static_lds_bytes": static_lds,
                "dynamic_lds_bytes": dynamic_lds,
                "max_active_blocks_per_cu": max_active_blocks,
            }

    del x, w, b
    torch.cuda.empty_cache()
    return {
        "op": "conv2d_fp16",
        "label": shape.label,
        "source": shape.source,
        "shape": dataclasses.asdict(shape),
        "stock_ms": stock_ms,
        "native_ms": native_ms,
        "native_error": native_err,
        "native_variant_idx": variant_idx,
        "native_variant_diagnostics": variant_diagnostics,
    }


def bench_conv3d(shape):
    # n_iter=20, not the default 50 -- matches tools/bench.py's own
    # precedent for Conv3d (heavier than conv2d per call), and the
    # Winograd candidate specifically does 2 hipMalloc + 3 kernel launches
    # + 2 hipFree per call, not just one launch.
    n_iter = 20

    x = torch.randn(shape.B, shape.C_in, shape.D_in, shape.H_in, shape.W_in, device=device, dtype=DTYPE)
    w = torch.randn(shape.C_out, shape.C_in, shape.K, shape.K, shape.K, device=device, dtype=DTYPE)
    b = torch.randn(shape.C_out, device=device, dtype=DTYPE)
    stride, padding, dilation = [shape.stride] * 3, [shape.padding] * 3, [shape.dilation] * 3

    stock_ms, _ = _time_ms(lambda: F.conv3d(x, w, b, stride=stride, padding=padding, dilation=dilation), n_iter=n_iter)
    native_ms, native_err = _time_ms(
        lambda: amd_tuned_torch.ops.conv3d(x, w, b, stride, padding, dilation), n_iter=n_iter
    )

    winograd_ms, winograd_note = None, "not eligible (see _is_winograd_eligible_conv3d)"
    if amd_tuned_torch._is_winograd_eligible_conv3d(x, w, stride, padding, dilation):
        def _winograd_call():
            out = compile_ops.conv3d_fp16_winograd_bt8_bc8(x, w, b, stride, padding, dilation)
            if out is None:
                raise RuntimeError("kernel declined despite passing the eligibility predicate")
        winograd_ms, winograd_note = _time_ms(_winograd_call, n_iter=n_iter)

    del x, w, b
    torch.cuda.empty_cache()
    return {
        "op": "conv3d_fp16",
        "label": shape.label,
        "source": shape.source,
        "shape": dataclasses.asdict(shape),
        "stock_ms": stock_ms,
        "native_ms": native_ms,
        "native_error": native_err,
        "winograd_ms": winograd_ms,
        "winograd_note": winograd_note,
    }


def _fmt(ms):
    return f"{ms:8.3f} ms" if ms is not None else "     N/A   "


def report(result):
    print(f"[{result['op']}] {result['label']}  ({result['source']})")
    print(f"  stock    : {_fmt(result['stock_ms'])}")
    err = f"   ({result['native_error']})" if result.get("native_error") else ""
    print(f"  native   : {_fmt(result['native_ms'])}{err}")
    variant_idx = result.get("native_variant_idx")
    if variant_idx is not None:
        variants = variants_for(result["op"])
        suffix = variants[variant_idx].suffix if variant_idx < len(variants) else "?"
        print(f"    -> run_conv2d_fp16 picked variant [{variant_idx}] {suffix}")
        diag = result.get("native_variant_diagnostics")
        if diag:
            print(f"       VGPRs={diag['num_regs']}  "
                  f"dynamic LDS={diag['dynamic_lds_bytes']}B  "
                  f"max active blocks/CU={diag['max_active_blocks_per_cu']}")
            if diag["max_active_blocks_per_cu"] == 0:
                print("       WARNING: the winning variant has 0 active blocks/CU -- "
                      "it shouldn't have been able to launch. Investigate before trusting "
                      "this result (possible hipOccupancyMaxActiveBlocksPerMultiprocessor "
                      "vs. actual launch discrepancy, or a driver/runtime issue).")
    times = {"stock": result["stock_ms"], "native": result["native_ms"]}
    if "winograd_ms" in result:
        note = f"   ({result['winograd_note']})" if result["winograd_note"] else ""
        print(f"  winograd : {_fmt(result['winograd_ms'])}{note}")
        times["winograd"] = result["winograd_ms"]
    times = {k: v for k, v in times.items() if v is not None}
    if times:
        winner = min(times, key=times.get)
        print(f"  winner   : {winner} ({times[winner]:.3f} ms)")
    print()


def main():
    was_enabled = amd_tuned_torch.is_enabled()
    if was_enabled:
        # Same reason as tools/bench.py: disable the monkeypatch first so
        # F.conv2d/F.conv3d above actually hit stock, not amd_tuned_torch itself.
        amd_tuned_torch.disable()

    # Static, shape-independent -- print once up front so the per-shape
    # results below can reference "[idx] suffix" without repeating the
    # full VGPR/LDS/occupancy breakdown every time.
    conv2d_fp16_diagnostics = report_conv2d_fp16_variant_diagnostics()

    results = []
    try:
        print("=" * 70)
        print("conv2d_fp16")
        print("=" * 70)
        for shape in CONV2D_SHAPES:
            result = bench_conv2d(shape)
            report(result)
            results.append(result)

        print("=" * 70)
        print("conv3d_fp16")
        print("=" * 70)
        for shape in CONV3D_SHAPES:
            result = bench_conv3d(shape)
            report(result)
            results.append(result)
    finally:
        if was_enabled:
            amd_tuned_torch.enable()

    output = {
        "conv2d_fp16_variant_diagnostics": conv2d_fp16_diagnostics,
        "results": results,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
