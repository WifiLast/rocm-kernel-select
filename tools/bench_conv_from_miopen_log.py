"""Benchmark (or pre-warm) amd_tuned_torch's conv tiers on the shapes a
REAL workload ran, taken from a MIOpen log instead of guessed.

WHY. Every conv benchmark in this repo picks its own shapes, so it can
only answer "is this kernel good on the shape I chose". A MIOpen verbose
log answers a different and more useful question -- which convolutions a
particular pipeline actually issues, how often, and which solver stock
picked for each -- and that inventory is what should drive tuning
decisions. The worked example this was built against was such a log from an
SD-style UNet + VAE decode at 2048x1024 (captured as described below,
not tracked in this repo): 84 forward convs over 38 distinct shapes, 82 of them on stock's Winograd/rocBLAS fast paths and exactly one
falling to im2col+GEMM, which turned out to be the one shape where the CK
tier is 2.2x faster. That is not a conclusion any synthetic sweep in this
repo would have reached.

HOW TO PRODUCE A LOG. Run the workload with MIOpen's own logging on --
these must be set before torch is imported, since MIOpen reads them at
init (tools/bench_conv2d_fp16.py's docstring has the fd-level capture
details if you need to collect it from inside Python):

    MIOPEN_ENABLE_LOGGING=1 MIOPEN_ENABLE_LOGGING_CMD=1 \
    MIOPEN_LOG_LEVEL=6 python your_app.py 2> miopen.log

USAGE

    python tools/bench_conv_from_miopen_log.py miopen.logs
    python tools/bench_conv_from_miopen_log.py miopen.logs --warm
    python tools/bench_conv_from_miopen_log.py miopen.logs --top 10 --json out.json

`--warm` skips the per-tier timing and just calls the patched op once per
shape, which runs kernel_select's contest and persists the winners (see
kernel_select.py's disk-cache section). That moves the one-time
measurement cost -- 70-431ms per shape, measured on the shapes in
miopen.logs -- out of the user's first real run and into this script.

Only forward convolutions are covered: that is what a `-F 1` log line is,
and the tiers this contests have no backward pass anyway (see
_patched_conv2d's own comment on training).
"""
import argparse
import collections
import json
import re
import sys

import torch
import torch.nn.functional as F

import amd_tuned_torch
from amd_tuned_torch import ck_ops, compile_ops, kernel_select

# MIOpen's own driver-command line, e.g.
#   convfp16 -n 2 -c 640 -H 128 -W 64 -k 640 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 ... -F 1
_CMD = re.compile(
    r"conv(?P<dt>fp16|bfp16|fp32)\s+-n (?P<n>\d+) -c (?P<c>\d+) -H (?P<H>\d+) -W (?P<W>\d+) "
    r"-k (?P<k>\d+) -y (?P<y>\d+) -x (?P<x>\d+) -p (?P<p>\d+) -q (?P<q>\d+) "
    r"-u (?P<u>\d+) -v (?P<v>\d+)")
# "[run] kernel_name = ..." tells us which solver stock actually used, which
# is how a shape on a slow path (Im2d2Col_v2) is spotted without timing it.
_KERNEL = re.compile(r"kernel_name = ([A-Za-z0-9_]+)")
_DTYPES = {"fp16": torch.float16, "bfp16": torch.bfloat16, "fp32": torch.float32}


def parse(path):
    """[(shape tuple, call count, {stock kernel names})], most-called first.

    Shapes are keyed on everything that changes the convolution; the
    kernel names are attributed to the most recent conv command, which is
    how MIOpen's log is ordered (command, then the launches it caused)."""
    counts = collections.Counter()
    kernels = collections.defaultdict(set)
    current = None
    for line in open(path, errors="replace"):
        # [LogCmdConvolution] is an actual convolution; [LogCmdFindConvolution]
        # is MIOpen's algorithm SEARCH for one, carrying the same driver
        # command. Counting both inflates every searched shape's call count
        # by one, so match the exact tag rather than just the -F 1 flag.
        m = _CMD.search(line) if "[LogCmdConvolution]" in line else None
        if m and "-F 1" in line:
            g = m.groupdict()
            current = (_DTYPES[g["dt"]], int(g["n"]), int(g["c"]), int(g["H"]), int(g["W"]),
                       int(g["k"]), int(g["y"]), int(g["p"]), int(g["u"]))
            counts[current] += 1
            continue
        if current is not None:
            k = _KERNEL.search(line)
            if k:
                kernels[current].add(k.group(1))
    return [(shape, n, kernels[shape]) for shape, n in counts.most_common()]


def timed(fn, n=20, warm=5):
    try:
        for _ in range(warm):
            if fn() is None:
                return float("nan")
        torch.cuda.synchronize()
    except (RuntimeError, TypeError):
        return float("nan")
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log")
    ap.add_argument("--top", type=int, default=0, help="only the N most-called shapes")
    ap.add_argument("--warm", action="store_true",
                    help="run the contest per shape and persist winners, don't benchmark")
    ap.add_argument("--json", help="write the table here")
    args = ap.parse_args()

    shapes = parse(args.log)
    if not shapes:
        sys.exit(f"no forward convolutions found in {args.log} -- was "
                 "MIOPEN_ENABLE_LOGGING_CMD=1 set when it was captured?")
    total = sum(n for _, n, _ in shapes)
    print(f"{args.log}: {total} forward convs over {len(shapes)} distinct shapes")
    if args.top:
        shapes = shapes[:args.top]

    rows = []
    if not args.warm:
        print(f"\n  {'shape':<34s} {'calls':>5s} {'stock':>8s} {'ck':>8s} {'native':>8s} "
              f"{'winner':<8s} {'gain':>6s}  stock solver")
    for (dtype, N, C, H, W, K, k, p, u), calls, kernels in shapes:
        x = torch.randn(N, C, H, W, device="cuda", dtype=dtype)
        w = torch.randn(K, C, k, k, device="cuda", dtype=dtype)
        b = torch.randn(K, device="cuda", dtype=dtype)
        name = f"{N}x{C}x{H}x{W}->{K} k{k}s{u}"
        if args.warm:
            amd_tuned_torch.enable()
            F.conv2d(x, w, b, u, p)
            amd_tuned_torch.disable()
            won = kernel_select.cached("conv2d", x, w, u, p, 1)
            print(f"  warmed {name:<34s} {str(dtype).split('.')[-1]:>8s} -> {won}")
            rows.append({"shape": name, "dtype": str(dtype), "calls": calls, "winner": won})
        else:
            amd_tuned_torch.disable()
            st = timed(lambda: F.conv2d(x, w, b, u, p))
            ck = timed(lambda: ck_ops.conv2d(x, w, b, u, p, 1))
            nv = timed(lambda: compile_ops.conv2d_native(x, w, b, u, p, 1))
            amd_tuned_torch.enable()
            F.conv2d(x, w, b, u, p)
            won = kernel_select.cached("conv2d", x, w, u, p, 1)
            amd_tuned_torch.disable()
            best = min(v for v in (st, ck, nv) if v == v)
            print(f"  {name:<34s} {calls:>5d} {st:8.3f} {ck:8.3f} {nv:8.3f} "
                  f"{str(won):<8s} {st / best:5.2f}x  {','.join(sorted(kernels))[:34]}")
            rows.append({"shape": name, "dtype": str(dtype), "calls": calls,
                         "stock_ms": st, "ck_ms": ck, "native_ms": nv,
                         "winner": won, "stock_solvers": sorted(kernels)})
        del x, w, b
        torch.cuda.empty_cache()

    if not args.warm:
        # What the workload would save if every shape used its best tier,
        # weighted by how often the log actually called it.
        got = sum(r["calls"] * r["stock_ms"] for r in rows if r["stock_ms"] == r["stock_ms"])
        best = sum(r["calls"] * min(v for v in (r["stock_ms"], r["ck_ms"], r["native_ms"])
                                    if v == v) for r in rows)
        print(f"\n  call-weighted conv time: stock {got:.1f} ms, best-tier {best:.1f} ms "
              f"({got / best:.2f}x available)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
