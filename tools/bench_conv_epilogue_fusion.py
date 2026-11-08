"""Measure what the POST-CONV EPILOGUE costs on the shapes a real workload
ran, i.e. how much a fused conv+bias+add would actually save.

WHY THIS AND NOT tools/analyse_miopen_log.py's heuristic. That script can
say "a 1x1 sits next to a 3x3 here, so this looks like a ResBlock". It
cannot say whether fusing the epilogue is worth writing a kernel for. This
runs the real shapes on the real card and measures the gap.

WHAT THE EPILOGUE ACTUALLY IS, for SDXL's UNet. Read sgm/modules/
diffusionmodules/openaimodel.py's ResBlock: the layer order is
GroupNorm -> SiLU -> Conv, so the ACTIVATION PRECEDES the convolution and
there is no conv+bias+activation to fuse. What follows a conv is:

    in_layers conv  ->  + bias  ->  + emb_out      (broadcast over C)
    out_layers conv ->  + bias  ->  + skip(x)      (full elementwise add)

So the fusable pattern in this model is conv+bias+add, and that is what
this measures. Anyone reaching for MIOpen's
MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_BIAS_ACTIV solvers for SDXL is fusing a
pattern the model does not contain.

METHOD. Per shape, four timings with CUDA events, after warmup:
    conv            torch.conv2d with bias=None
    conv+bias       conv with a real bias tensor (still one ATen call --
                    MIOpen/ATen already applies bias inside)
    conv+bias+add   the above plus a separate residual add, which is what
                    eager mode really does
    epilogue_only   the residual add alone
The fusion headroom is (conv+bias+add) - (conv+bias): the part a fused
kernel could remove by writing the sum once instead of writing the conv
output and reading it back.

STOCK ONLY. amd_tuned_torch patches F.conv2d at import (this venv's
torch/__init__.py calls enable()), so this calls torch.conv2d -- the ATen
builtin, which is not patched -- and additionally calls disable() when the
package is present, so nothing here measures a tuned tier by accident.

USAGE
    python tools/bench_conv_epilogue_fusion.py LOG
    python tools/bench_conv_epilogue_fusion.py LOG --top 10 --iters 30
    python tools/bench_conv_epilogue_fusion.py LOG --json epilogue.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse_miopen_log import parse  # noqa: E402


def _time(fn, iters, warmup=20, repeats=5):
    """Min of `repeats` timed batches.

    A single batch is not trustworthy here: the first run of a shape pays
    MIOpen's algorithm search, and this card's clocks ramp over the first
    few hundred ms, so an early variant is measured slow and a later one
    fast. That produced a conv+bias+add timing FASTER than conv+bias alone
    -- impossible, since it is strictly more work -- in the first version
    of this script. Min-of-N rejects both effects: the fastest batch is the
    one where nothing interfered."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeats):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / iters)
    return best  # ms


def bench_shape(conv, iters, device="cuda"):
    dtype = {"fp16": torch.float16, "bfp16": torch.bfloat16,
             "fp32": torch.float32}.get(conv.dt, torch.float16)
    x = torch.randn(conv.n, conv.c, conv.H, conv.W, device=device, dtype=dtype)
    w = torch.randn(conv.k, conv.c // max(conv.g, 1), conv.y, conv.x,
                    device=device, dtype=dtype)
    b = torch.randn(conv.k, device=device, dtype=dtype)
    stride, pad, dil, groups = (conv.u, conv.v), (conv.p, conv.q), (conv.l, conv.j), conv.g

    def conv_only():
        return torch.conv2d(x, w, None, stride, pad, dil, groups)

    def conv_bias():
        return torch.conv2d(x, w, b, stride, pad, dil, groups)

    y = conv_bias()
    resid = torch.randn_like(y)

    def conv_bias_add():
        return torch.conv2d(x, w, b, stride, pad, dil, groups) + resid

    def add_only():
        return y + resid

    # Warm ALL variants first: each one's own algorithm search and any clock
    # ramp then happens before the first measurement, not between them.
    for fn in (conv_only, conv_bias, conv_bias_add, add_only):
        for _ in range(10):
            fn()
    torch.cuda.synchronize()

    out = {
        "conv": _time(conv_only, iters),
        "conv_bias": _time(conv_bias, iters),
        "conv_bias_add": _time(conv_bias_add, iters),
        "add_only": _time(add_only, iters),
    }
    # Adding work cannot be faster. If it measures faster the run is noise-
    # dominated and the number must not be reported as a saving.
    out["suspect"] = out["conv_bias_add"] < out["conv_bias"]
    out["headroom_ms"] = out["conv_bias_add"] - out["conv_bias"]
    out["headroom_pct"] = 100.0 * out["headroom_ms"] / out["conv_bias_add"]
    out["bias_ms"] = out["conv_bias"] - out["conv"]
    # Bytes the separate add must move: read conv output, read residual,
    # write result. A fused kernel writes the sum directly and moves none.
    elems = y.numel()
    out["add_bytes"] = 3 * elems * y.element_size()
    out["add_gbs"] = out["add_bytes"] / (out["add_only"] / 1e3) / 1e9
    del x, w, b, y, resid
    torch.cuda.empty_cache()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log")
    ap.add_argument("--top", type=int, default=8,
                    help="shapes to measure, ranked by total FLOP (default 8)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        ap.error("no GPU -- this measures real kernels, there is no CPU path")

    try:
        import amd_tuned_torch
        amd_tuned_torch.disable()
        print("amd_tuned_torch.disable() -- measuring stock ATen/MIOpen\n")
    except Exception:
        pass

    data = parse(args.log)
    # Rank by total FLOP, not call count: the shape worth optimising is the
    # one that owns the most work, which is not always the most frequent.
    ranked = sorted(
        ((data["conv_objs"][k], n, data["conv_objs"][k].flops() * n)
         for k, n in data["convs"].items() if data["conv_objs"][k].F == 1),
        key=lambda t: -t[2])[:args.top]

    total_flops = sum(data["conv_objs"][k].flops() * n
                      for k, n in data["convs"].items())
    print(f"{os.path.basename(args.log)}: {sum(data['convs'].values()):,} convs, "
          f"{total_flops / 1e12:.1f} TFLOP total")
    print(f"measuring the top {len(ranked)} shapes by FLOP share, "
          f"{args.iters} iters each\n")

    hdr = (f"{'shape':<34}{'calls':>7}{'conv':>9}{'+bias':>8}{'+add':>8}"
           f"{'fuse win':>10}{'add GB/s':>10}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    run_conv_ms = 0.0
    run_headroom_ms = 0.0
    for conv, calls, fl in ranked:
        r = bench_shape(conv, args.iters)
        oh, ow = conv.out_hw()
        label = f"n{conv.n} c{conv.c} {conv.H}x{conv.W} k{conv.k} {conv.y}x{conv.x}"
        flag = " noisy" if r["suspect"] else ""
        print(f"{label:<34}{calls:>7,}{r['conv']:>8.3f}m{r['conv_bias']:>7.3f}m"
              f"{r['conv_bias_add']:>7.3f}m{r['headroom_pct']:>9.1f}%"
              f"{r['add_gbs']:>9.0f}{flag}")
        run_conv_ms += r["conv_bias"] * calls
        # Never let a negative (noise) headroom subtract from the total.
        run_headroom_ms += max(r["headroom_ms"], 0.0) * calls
        rows.append({"shape": label, "calls": calls, "tflop_total": fl / 1e12, **r})

    covered = sum(f for _, _, f in ranked) / total_flops * 100
    print()
    print(f"Over these {len(ranked)} shapes ({covered:.0f}% of all conv FLOPs), "
          f"across the {sum(c for _, c, _ in ranked):,} calls they account for:")
    print(f"  conv+bias time      : {run_conv_ms / 1e3:.2f} s")
    print(f"  separate-add time   : {run_headroom_ms / 1e3:.2f} s")
    print(f"  fusing the add saves: {100 * run_headroom_ms / (run_conv_ms + run_headroom_ms):.1f}%"
          f" of that conv time")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"log": os.path.abspath(args.log),
                       "device": torch.cuda.get_device_name(0),
                       "iters": args.iters, "rows": rows,
                       "conv_s": run_conv_ms / 1e3,
                       "headroom_s": run_headroom_ms / 1e3}, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
