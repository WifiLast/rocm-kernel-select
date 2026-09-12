"""conv3d shape sweep: stock PyTorch vs amd_tuned_torch, and which tier won.

tools/bench_fast.py answers "is amd_tuned_torch faster on this machine?" with
one shape per op. conv3d is the op where that single number hides the most,
because _patched_conv3d has four candidates with genuinely different shape
regimes -- Composable Kernel's WMMA conv (fp16/bf16), the hand-written HIP
kernel (fp16/fp32, src/cuda/conv3d_fp{16,32}.cu), the FFT tier (which only
pays off once K^3 gets large), and stock -- and kernel_select re-runs its
contest per (dtype, shape, stride, padding, dilation). One shape tells you
about one cell of that space.

So this sweeps the axes separately -- kernel width, spatial extent, channel
count, batch, anisotropy, stride/padding, dtype -- and reports the contest
winner alongside each timing, read back from kernel_select.cached(). A row
where the winner is "stock" is the contest working, not a gap.

Timing harness is shared with bench_fast.py (same probe-then-alternating-
rounds-keep-the-minimum method) so numbers from the two files are directly
comparable.

Reading the small rows: anything under ~0.2ms per call carries roughly a 2x
run-to-run spread on this machine, because the patched path's fixed costs
(the sparse-occupancy probe most of all -- see the `config` block in the
results file) are the same order as the conv itself there. `rounds_ms`
records every round so within-run spread is visible too. Trust the ordering
of the large rows; treat a sub-0.2ms row as "about the same" unless it moves
consistently across runs.

Shapes are deliberately small: this is meant to run after a build, not
overnight. A few cases are sized right at the edge of what stock can do --
at fp32/K=15 MIOpen asks for a 14 GB workspace and OOMs, and so does the
patched path, which is why each phase's failure is recorded per phase
instead of discarding the whole row.

Run with:

    python tools/bench_conv3d_sweep.py
    python tools/bench_conv3d_sweep.py --out /tmp/conv3d.json
    python tools/bench_conv3d_sweep.py --group kernel_size,dtype
    python tools/bench_conv3d_sweep.py --budget-ms 20      # quicker/noisier
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# bench_fast pins sys.path to this checkout above its own `import torch`
# (see its comment on why that has to happen before torch loads), so it must
# be imported before torch is touched here too.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_fast  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from amd_tuned_torch import flexgemm_ops, kernel_select  # noqa: E402

DEVICE = torch.device("cuda")
DEFAULT_BUDGET_MS = 60.0


class Conv3dCase:
    """One conv3d shape. Cout defaults to Cin -- the sweep varies width and
    depth independently elsewhere, and keeping them equal here means a row
    differs from its neighbours in exactly one axis."""

    def __init__(self, group: str, n: int, c_in: int, d: int, h: int, w: int,
                 k: int, dtype: torch.dtype, stride: int = 1,
                 padding: Optional[int] = None, c_out: Optional[int] = None) -> None:
        self.group = group
        self.n, self.c_in, self.c_out = n, c_in, c_out if c_out is not None else c_in
        self.d, self.h, self.w = d, h, w
        self.k, self.stride = k, stride
        self.padding = k // 2 if padding is None else padding
        self.dtype = dtype
        self.dtype_name = str(dtype).rsplit(".", 1)[-1]

    @property
    def key(self) -> str:
        return (f"{self.group}/N{self.n}_C{self.c_in}to{self.c_out}"
                f"_{self.d}x{self.h}x{self.w}_k{self.k}"
                f"_s{self.stride}p{self.padding}_{self.dtype_name}")

    @property
    def shape_desc(self) -> str:
        return (f"N={self.n}, C={self.c_in}->{self.c_out}, "
                f"{self.d}x{self.h}x{self.w}, k={self.k}, "
                f"stride={self.stride}, pad={self.padding}")

    def gflops(self) -> float:
        d_out = (self.d + 2 * self.padding - self.k) // self.stride + 1
        h_out = (self.h + 2 * self.padding - self.k) // self.stride + 1
        w_out = (self.w + 2 * self.padding - self.k) // self.stride + 1
        macs = (self.n * self.c_out * d_out * h_out * w_out
                * self.c_in * self.k ** 3)
        return 2.0 * macs / 1e9

    def make_args(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(self.n, self.c_in, self.d, self.h, self.w,
                        device=DEVICE, dtype=self.dtype)
        weight = torch.randn(self.c_out, self.c_in, self.k, self.k, self.k,
                             device=DEVICE, dtype=self.dtype)
        bias = torch.randn(self.c_out, device=DEVICE, dtype=self.dtype)
        return x, weight, bias

    def invoke(self, x, weight, bias):
        return F.conv3d(x, weight, bias, stride=self.stride, padding=self.padding)


def build_cases() -> List[Conv3dCase]:
    cases: List[Conv3dCase] = []

    # Kernel width, the axis the FFT tier exists for. C and the spatial
    # extent are held small here because stock's workspace demand grows with
    # K^3: at C=32/32^3 a K=15 fp32 conv3d already asks MIOpen for 14 GB.
    for k in (1, 3, 5, 7, 9, 15):
        for dtype in (torch.float16, torch.float32):
            cases.append(Conv3dCase("kernel_size", 1, 16, 16, 16, 16, k, dtype))

    # Spatial extent at fixed K=3 -- the diffusion/U-Net regime.
    for s in (8, 16, 32, 48):
        cases.append(Conv3dCase("spatial", 1, 64, s, s, s, 3, torch.float16))

    # Channel count: where the WMMA tiers should pull ahead.
    for c in (32, 64, 128, 256):
        cases.append(Conv3dCase("channels", 1, c, 16, 16, 16, 3, torch.float16))

    # Batch.
    for n in (1, 2, 4, 8):
        cases.append(Conv3dCase("batch", n, 64, 16, 16, 16, 3, torch.float16))

    # Anisotropic volumes -- video/medical-shaped rather than cubic, which is
    # what a kernel tuned only on cubes tends to be worst at.
    cases.append(Conv3dCase("anisotropic", 2, 32, 8, 64, 64, 3, torch.float16))
    cases.append(Conv3dCase("anisotropic", 1, 128, 4, 16, 16, 3, torch.float16))
    cases.append(Conv3dCase("anisotropic", 1, 64, 32, 32, 8, 3, torch.float16))

    # stride/padding: strided and unpadded convs take different code paths in
    # every one of the candidates, and 1x1x1 is the pointwise special case.
    cases.append(Conv3dCase("stride_padding", 1, 64, 16, 16, 16, 3, torch.float16, stride=2))
    cases.append(Conv3dCase("stride_padding", 1, 64, 16, 16, 16, 3, torch.float16, padding=0))
    cases.append(Conv3dCase("stride_padding", 1, 64, 16, 16, 16, 1, torch.float16, padding=0))
    cases.append(Conv3dCase("stride_padding", 1, 64, 32, 32, 32, 3, torch.float16, stride=2))

    # dtype at one fixed shape, so the three are comparable to each other.
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        cases.append(Conv3dCase("dtype", 1, 64, 16, 16, 16, 3, dtype))

    # Asymmetric channel counts (Cin != Cout), as every real network has.
    cases.append(Conv3dCase("channels_asym", 1, 32, 16, 16, 16, 3, torch.float16, c_out=128))
    cases.append(Conv3dCase("channels_asym", 1, 128, 16, 16, 16, 3, torch.float16, c_out=32))

    # The large-kernel headline, at the largest size stock survives at fp16.
    # The fp32 twin is included knowing both sides OOM -- a documented "no
    # implementation on this GPU" is a more useful row than a missing one.
    cases.append(Conv3dCase("large_kernel", 1, 32, 32, 32, 32, 15, torch.float16))
    cases.append(Conv3dCase("large_kernel", 1, 32, 32, 32, 32, 15, torch.float32))

    return cases


def run_case(case: Conv3dCase, state: "bench_fast.PatchState",
             budget_ms: float) -> Dict[str, Any]:
    """Time both phases, tolerating the failure of either.

    Unlike bench_fast's equivalent, a phase that raises (OOM, no kernel for
    the shape) is recorded as that phase's error and the other phase still
    reports a number -- at the edges of what this GPU can do, "stock cannot
    run this shape at all" is the result.
    """
    args = case.make_args()
    row: Dict[str, Any] = {
        "group": case.group, "shape": case.shape_desc,
        "dtype": case.dtype_name, "gflops": round(case.gflops(), 3),
    }

    def call():
        return case.invoke(*args)

    outputs: Dict[str, Any] = {}
    iters: Dict[str, int] = {}
    for phase, patched in (("stock", False), ("patched", True)):
        state.set(patched)
        try:
            iters[phase] = bench_fast._probe_iters(call, budget_ms)
            outputs[phase] = call()
            torch.cuda.synchronize()
        except Exception as exc:
            row[f"{phase}_error"] = f"{type(exc).__name__}: {exc}".split("\n")[0]
            torch.cuda.empty_cache()

    # Read the winner only after the patched warmup, i.e. after the contest
    # has actually run for this shape -- before that it is always None.
    if "patched" in outputs:
        state.set(True)
        row["winner"] = kernel_select.cached(
            "conv3d", args[0], args[1], case.stride, case.padding, 1)

    # Every round is kept, not just the minimum. Below ~0.15ms per call the
    # spread between rounds is a sizeable fraction of the number itself, and
    # a reader comparing two runs of this file needs to see that rather than
    # infer a real change from it.
    per_round: Dict[str, List[float]] = {"stock": [], "patched": []}
    for _ in range(bench_fast.ROUNDS):
        for phase, patched in (("stock", False), ("patched", True)):
            if phase not in iters:
                continue
            state.set(patched)
            try:
                per_round[phase].append(bench_fast._time_ms(call, iters[phase]))
            except Exception as exc:
                row.setdefault(f"{phase}_error", f"{type(exc).__name__}: {exc}".split("\n")[0])

    row["stock_ms"] = min(per_round["stock"]) if per_round["stock"] else None
    row["patched_ms"] = min(per_round["patched"]) if per_round["patched"] else None
    row["rounds_ms"] = {k: [round(v, 4) for v in vs] for k, vs in per_round.items() if vs}
    if row["stock_ms"] and row["patched_ms"]:
        row["speedup"] = row["stock_ms"] / row["patched_ms"]
        row["patched_tflops"] = case.gflops() / row["patched_ms"]  # GFLOP/ms == TFLOP/s
        row["max_abs_err"], row["max_rel_err"] = bench_fast._err(
            outputs["patched"], outputs["stock"])
    row["iters"] = iters

    del args, outputs
    torch.cuda.empty_cache()
    return row


def print_table(rows: Dict[str, Dict[str, Any]]) -> None:
    head = (f"{'shape':<48}{'dtype':<10}{'stock ms':>10}{'patch ms':>10}"
            f"{'speedup':>9}{'TFLOP/s':>9}{'winner':>9}{'rel err':>10}")
    last_group = None
    for key, r in rows.items():
        group = r["group"]
        if group != last_group:
            print()
            print(f"--- {group} " + "-" * (len(head) - len(group) - 5))
            print(head)
            last_group = group
        if r.get("stock_ms") is None or r.get("patched_ms") is None:
            def _phase(name: str) -> str:
                ms = r.get(f"{name}_ms")
                if ms is not None:
                    return f"{ms:.2f}ms"
                return r.get(f"{name}_error", "?").split(":")[0]
            print(f"{r['shape']:<48}{r['dtype']:<10}"
                  f"  stock={_phase('stock')}  patched={_phase('patched')}")
            continue
        rel = r.get("max_rel_err")
        rel_s = "n/a" if rel is None else f"{rel:.1e}"
        print(f"{r['shape']:<48}{r['dtype']:<10}{r['stock_ms']:>10.3f}"
              f"{r['patched_ms']:>10.3f}{r['speedup']:>8.2f}x"
              f"{r['patched_tflops']:>9.1f}{str(r.get('winner')):>9}{rel_s:>10}")


def summarize(rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    timed = {k: r for k, r in rows.items() if r.get("speedup")}
    winners: Dict[str, int] = {}
    for r in timed.values():
        winners[str(r.get("winner"))] = winners.get(str(r.get("winner")), 0) + 1
    speedups = sorted((r["speedup"], k) for k, r in timed.items())
    return {
        "cases_timed": len(timed),
        "cases_failed": len(rows) - len(timed),
        "winners": winners,
        "faster": sum(1 for r in timed.values() if r["speedup"] > 1.05),
        "slower": sum(1 for r in timed.values() if r["speedup"] < 0.95),
        "best": {"case": speedups[-1][1], "speedup": speedups[-1][0]} if speedups else None,
        "worst": {"case": speedups[0][1], "speedup": speedups[0][0]} if speedups else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None,
                        help="Results file (default: bench_conv3d_sweep.json beside setup.py).")
    parser.add_argument("--budget-ms", type=float, default=DEFAULT_BUDGET_MS)
    parser.add_argument("--group", default=None,
                        help="Comma-separated sweep groups to run (default: all).")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No GPU visible to torch -- nothing to compare.", file=sys.stderr)
        return 1

    out_path = args.out or os.path.join(bench_fast.REPO_ROOT, "bench_conv3d_sweep.json")
    wanted = {g.strip() for g in args.group.split(",")} if args.group else None
    cases = [c for c in build_cases() if wanted is None or c.group in wanted]
    if not cases:
        print(f"No cases matched --group {args.group!r}", file=sys.stderr)
        return 1

    state = bench_fast.PatchState()
    rows: Dict[str, Dict[str, Any]] = {}
    t_start = time.perf_counter()
    try:
        for case in cases:
            rows[case.key] = run_case(case, state, args.budget_ms)
    finally:
        state.restore()
    elapsed = time.perf_counter() - t_start

    print_table(rows)
    summary = summarize(rows)
    print()
    print(f"timed {summary['cases_timed']} shapes "
          f"({summary['cases_failed']} unrunnable): "
          f"{summary['faster']} faster, {summary['slower']} slower, "
          f"rest within noise")
    print(f"contest winners: {summary['winners']}")
    print(f"total wall time: {elapsed:.1f}s")

    system = bench_fast._system()
    if not system.get("checkout_matches", False):
        print(f"WARNING: measured amd_tuned_torch at {system['amd_tuned_torch']}, "
              f"not the checkout at {bench_fast.REPO_ROOT}.", file=sys.stderr)

    report = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "harness": "tools/bench_conv3d_sweep.py",
        "method": ("F.conv3d called with amd_tuned_torch.disable() vs .enable(); "
                   "min of alternating rounds; CUDA-event timed; winner read "
                   "from kernel_select.cached()"),
        "rounds": bench_fast.ROUNDS,
        "budget_ms": args.budget_ms,
        "wall_time_s": round(elapsed, 3),
        "system": system,
        "backends_available": bench_fast._backends(),
        # Both of these change what the patched column means, so they belong
        # next to the numbers. The sparse-conv3d occupancy probe in
        # particular runs a reduction over `input` on EVERY eligible dense
        # call, which is a fixed cost that small conv3ds cannot amortize --
        # measured here at ~0.05-0.13ms, i.e. larger than an 8^3/16^3 conv.
        "config": {
            "sparse_conv3d_enabled": bool(flexgemm_ops.sparse_conv3d_enabled()),
            "kernel_select_enabled": bool(kernel_select.enabled()),
            "AMD_TUNED_TORCH_SPARSE_CONV3D": os.environ.get(
                "AMD_TUNED_TORCH_SPARSE_CONV3D"),
        },
        "summary": summary,
        "cases": rows,
    }
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
