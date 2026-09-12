"""Benchmark amd_tuned_torch.boft_ops.fast_block_diag against the stock
PyTorch construction it replaces, on RX 7900 XTX (gfx1100/RDNA3), ROCm.

WHAT IS BEING COMPARED. BOFT's block-diagonal assembly has no F.* stock
equivalent -- the thing this kernel actually competes with is PEFT's own
portable fallback in peft/tuners/boft/layer.py, i.e.

    torch.block_diag(*torch.unbind(orth_rotate_butterfly.squeeze(0))).unsqueeze(0)

one Python-level call per block plus a materialized (N*b, N*b) zero tensor.
Both are timed here, directly (not through the monkeypatch), so the numbers
isolate the kernel from any dispatch overhead -- same posture as
tools/bench.py.

NOTE THAT PEFT'S FALLBACK ONLY HANDLES z == 1. That `squeeze(0)` above is
not incidental: upstream forces boft_n_butterfly_factor to 1 whenever no
compiled fbd extension is available (layer.py:286), because its pure-PyTorch
path cannot build a batch of block-diagonals at all. So for z > 1 the
"stock" column below is a GENERIC loop over z that upstream does not
actually ship -- for those rows the honest reading is not "the kernel is
Nx faster" but "multi-factor BOFT is only reachable at all with a compiled
kernel in place". Both variants are reported separately for z == 1 so the
literal upstream line is never conflated with the generic one.

SHAPES are real BOFT parameter shapes: boft_R is
(boft_n_butterfly_factor + 1, in_features / boft_block_size,
boft_block_size, boft_block_size) -- see peft/tuners/boft/layer.py:364.

WHERE THE TIME ACTUALLY GOES (the ROOFLINE section below). The scatter
itself writes only z*N*b*b elements, but the output it scatters into is
z*(N*b)^2 -- for a realistic BOFT shape like [1, 256, 32, 32] that is 262144
elements written into a 67M-element tensor, i.e. the torch::zeros() that
makes everything off the block diagonal zero moves ~256x more memory than
the kernel does. So this op is pinned to the cost of materializing its own
zeroed output and cannot go faster than a memset of that size, in any dtype.
The ROOFLINE section measures that floor directly so the kernel is judged
against what is physically achievable rather than only against the Python
fallback. Do not "optimize" the scatter on the strength of the FORWARD table
alone -- it is already single-digit microseconds.

A CAVEAT ON THE STOCK COLUMN. The fallback is CPU-launch-bound (one
Python-level torch.block_diag call per block), so its timings carry real
host-side variance even at min-of-N -- treat them as an order of magnitude,
not a precise figure. The kernel column and the ROOFLINE section are
GPU-bound and reproduce tightly.

Run with:

    python tools/bench_boft.py
    python tools/bench_boft.py --json bench_boft.json
"""
import argparse
import json

import torch

import amd_tuned_torch
from amd_tuned_torch import boft_ops

device = torch.device("cuda")

# (in_features, boft_block_size, boft_n_butterfly_factor) -> (z, N, b).
# Block sizes and feature counts taken from BOFTConfig's own defaults and
# the ranges the BOFT paper sweeps; z = n_butterfly_factor + 1.
CONFIGS = [
    (1024, 8, 0),
    (1024, 32, 0),
    (4096, 8, 0),
    (4096, 32, 0),
    (4096, 64, 0),
    (8192, 32, 0),
    (4096, 32, 1),   # z=2 -- unreachable without a compiled fbd kernel
    (4096, 32, 3),   # z=4
]

DTYPES = [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]


def peft_fallback(x):
    """peft.tuners.boft.layer's literal fbd_cuda_available=False path.
    z == 1 only -- squeeze(0)/unsqueeze(0) is upstream's own code."""
    return torch.block_diag(*torch.unbind(x.squeeze(0))).unsqueeze(0)


def generic_fallback(x):
    """What a z > 1 fallback would have to do (upstream ships no such path)."""
    return torch.stack([torch.block_diag(*torch.unbind(t, dim=0)) for t in x])


def time_ms(fn, arg, n_warmup=10, n_iter=50, n_rounds=5):
    """Minimum over n_rounds timed rounds, not the mean of one.

    The stock path here is CPU-launch-bound (one Python-level
    torch.block_diag call per block), so its wall time is dominated by host
    scheduling noise -- a single round gave the same dtype-independent loop
    anywhere from 0.7 to 4.9 ms run to run. The minimum is the stable
    statistic for "how fast can this go"; the mean here mostly measures
    whatever else the machine was doing.

    The caller interleaves candidates round-robin (see _compare) so clock
    and thermal drift hits both sides equally rather than whichever ran
    second."""
    for _ in range(n_warmup):
        fn(arg)
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(n_rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iter):
            fn(arg)
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / n_iter)
    return best


def _compare(stock, kernel, x, n_iter, n_rounds=5):
    """One round of each, alternating, n_rounds times -- keeps a clock ramp
    or a background process from being charged to just one candidate."""
    t_stock = t_kern = float("inf")
    for _ in range(n_rounds):
        t_stock = min(t_stock, time_ms(stock, x, n_warmup=3, n_iter=n_iter, n_rounds=1))
        t_kern = min(t_kern, time_ms(kernel, x, n_warmup=3, n_iter=n_iter, n_rounds=1))
    return t_stock, t_kern


def peak_mib(fn, arg):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    out = fn(arg)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    del out
    return peak / 1024 ** 2


def bwd_of(fn):
    """Time the gradient, not just the assembly: PEFT calls this inside a
    trainable adapter, so the backward gather is on the hot path too."""
    def run(x):
        out = fn(x)
        out.backward(torch.ones_like(out))
        x.grad = None
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write results to this file")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    if not boft_ops.available():
        raise SystemExit("fast_block_diag not in this build of amd_tuned_torch._native")

    print(f"device: {torch.cuda.get_device_name(0)}   torch {torch.__version__}")
    print(f"hip: {torch.version.hip}\n")

    rows = []
    header = (f"{'in_feat':>8} {'blk':>4} {'z':>3} {'shape':>18} {'dtype':>6} "
              f"{'stock ms':>9} {'kernel ms':>10} {'speedup':>8} "
              f"{'stock MiB':>10} {'kern MiB':>9} {'exact':>6}")
    print("FORWARD");  print(header);  print("-" * len(header))

    for in_features, blk, nbf in CONFIGS:
        z, N, b = nbf + 1, in_features // blk, blk
        stock = peft_fallback if z == 1 else generic_fallback
        stock_name = "peft-fallback" if z == 1 else "generic-loop"

        for dname, dtype in DTYPES:
            x = torch.randn(z, N, b, b, device=device, dtype=torch.float32).to(dtype)

            exact = torch.equal(boft_ops.fast_block_diag(x), stock(x))
            t_stock, t_kern = _compare(stock, boft_ops.fast_block_diag, x, args.iters)
            m_stock, m_kern = peak_mib(stock, x), peak_mib(boft_ops.fast_block_diag, x)

            print(f"{in_features:>8} {blk:>4} {z:>3} {f'[{z},{N},{b},{b}]':>18} {dname:>6} "
                  f"{t_stock:>9.3f} {t_kern:>10.3f} {t_stock / t_kern:>7.1f}x "
                  f"{m_stock:>10.1f} {m_kern:>9.1f} {str(exact):>6}")
            rows.append(dict(in_features=in_features, block_size=blk, z=z, N=N, b=b,
                             dtype=dname, pass_="forward", stock=stock_name,
                             stock_ms=t_stock, kernel_ms=t_kern,
                             speedup=t_stock / t_kern, stock_mib=m_stock,
                             kernel_mib=m_kern, bitwise_exact=exact))
        print()

    print("FORWARD + BACKWARD");  print(header);  print("-" * len(header))
    for in_features, blk, nbf in CONFIGS:
        z, N, b = nbf + 1, in_features // blk, blk
        stock = peft_fallback if z == 1 else generic_fallback
        stock_name = "peft-fallback" if z == 1 else "generic-loop"

        for dname, dtype in DTYPES:
            x = (torch.randn(z, N, b, b, device=device, dtype=torch.float32)
                 .to(dtype).requires_grad_(True))
            t_stock, t_kern = _compare(bwd_of(stock), bwd_of(boft_ops.fast_block_diag),
                                       x, args.iters)
            print(f"{in_features:>8} {blk:>4} {z:>3} {f'[{z},{N},{b},{b}]':>18} {dname:>6} "
                  f"{t_stock:>9.3f} {t_kern:>10.3f} {t_stock / t_kern:>7.1f}x "
                  f"{'':>10} {'':>9} {'':>6}")
            rows.append(dict(in_features=in_features, block_size=blk, z=z, N=N, b=b,
                             dtype=dname, pass_="fwd+bwd", stock=stock_name,
                             stock_ms=t_stock, kernel_ms=t_kern,
                             speedup=t_stock / t_kern))
        print()

    # ------------------------------------------------------------------
    # Roofline: how close is the kernel to the cost of merely allocating
    # its own zeroed output? See this module's docstring.
    # ------------------------------------------------------------------
    rhead = (f"{'shape':>18} {'dtype':>6} {'out MiB':>8} {'zeros() ms':>11} "
             f"{'kernel ms':>10} {'scatter ms':>11} {'GB/s':>7} {'of floor':>9}")
    print("ROOFLINE -- kernel vs. the torch.zeros() floor for the same output")
    print(rhead);  print("-" * len(rhead))
    for in_features, blk, nbf in CONFIGS:
        z, N, b = nbf + 1, in_features // blk, blk
        M = N * b
        for dname, dtype in DTYPES:
            x = torch.randn(z, N, b, b, device=device, dtype=torch.float32).to(dtype)
            out_bytes = z * M * M * x.element_size()
            t_zeros = time_ms(lambda _: torch.zeros(z, M, M, device=device, dtype=dtype),
                              x, n_iter=args.iters)
            t_kern = time_ms(boft_ops.fast_block_diag, x, n_iter=args.iters)
            gbs = out_bytes / (t_kern * 1e-3) / 1e9
            print(f"{f'[{z},{N},{b},{b}]':>18} {dname:>6} {out_bytes / 2 ** 20:>8.1f} "
                  f"{t_zeros:>11.4f} {t_kern:>10.4f} {t_kern - t_zeros:>11.4f} "
                  f"{gbs:>7.0f} {t_zeros / t_kern * 100:>8.0f}%")
            rows.append(dict(in_features=in_features, block_size=blk, z=z, N=N, b=b,
                             dtype=dname, pass_="roofline", out_mib=out_bytes / 2 ** 20,
                             zeros_ms=t_zeros, kernel_ms=t_kern,
                             scatter_ms=t_kern - t_zeros, gb_per_s=gbs,
                             frac_of_memset_floor=t_zeros / t_kern))
        print()

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"device": torch.cuda.get_device_name(0),
                       "torch": torch.__version__, "hip": torch.version.hip,
                       "results": rows}, fh, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
