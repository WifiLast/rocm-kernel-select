"""Benchmark amd_tuned_torch.te_ops against stock PyTorch/ROCm, on RX 7900 XTX
(gfx1100/RDNA3).

tools/bench.py already times the five TE-backed ops, but only forward, only
one shape each, and without checking that TE returns the same numbers. This
script exists for the thing te_ops.py is actually claiming to buy: a fused
kernel *with a backward pass* (see te_ops.py's module docstring -- the
original CMP-Turing kernels had none and fell back to stock under autograd).
A forward-only number says nothing about whether that claim pays off, since
training spends most of its time in backward.

So every op is measured three ways:

  fwd       -- inference path, no autograd graph (torch.no_grad)
  fwd+bwd   -- the training path: build the graph, then .backward()
  numerics  -- max|TE - stock| on the output and on every input gradient,
               reported next to stock's own error against an fp32 reference
               so a "big" absolute diff in fp16 can be read as either
               "TE is wrong" or "this is just fp16"

Ops are called directly (amd_tuned_torch.te_ops.foo), not through the F.foo
monkeypatch, so the numbers isolate the kernel from dispatch overhead -- same
convention as tools/bench.py.

Requires AMD_TUNED_TORCH_ENABLE_TE=1 in the environment BEFORE import (te_ops
is inert otherwise -- see its "TE DISABLED BY DEFAULT" docstring section).

Run with:

    AMD_TUNED_TORCH_ENABLE_TE=1 python tools/bench_te_ops.py
    AMD_TUNED_TORCH_ENABLE_TE=1 python tools/bench_te_ops.py --dtype bf16 --quick

Note on gfx1100: TE's setup.py gates the CK/AOTriton fused-attention backends
to gfx942/gfx950, so attention here runs TE's *unfused* path. That is expected
to lose to PyTorch's own flash/mem-efficient SDPA -- this script is how you
confirm that rather than assume it.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import statistics
import sys

# site-packages holds a physical copy of amd_tuned_torch that shadows the
# editable install unless the source tree comes first on sys.path -- running
# this as `python tools/bench_te_ops.py` puts tools/ at sys.path[0], not the
# repo root, so without this the script silently benchmarks whatever was last
# copied into site-packages. The banner below prints te_ops.__file__ so the
# result is auditable either way.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# TE's JIT path is dead on this box (ROCm-header/runtime version skew) and
# te_ops.py setdefault's this anyway; set it before the import so it also
# covers the case where this script is run against an older te_ops.
os.environ.setdefault("NVTE_DISABLE_NVRTC", "1")
# tools/bench.py inherits whatever TORCH_LOGS the shell exports; a benchmark
# run under +dynamo,+inductor buries its own output in compile logs and the
# logging itself perturbs the timings.
os.environ.pop("TORCH_LOGS", None)
os.environ.pop("TORCH_COMPILE_DEBUG", None)

import torch
import torch.nn.functional as F

import amd_tuned_torch
import amd_tuned_torch.te_ops as te_ops

device = torch.device("cuda")


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _time_ms(fn, n_iter, n_warmup=5, n_rounds=5):
    """Median-of-rounds mean-per-iter, in ms, via CUDA events.

    Median across rounds rather than min: min flatters whichever kernel gets
    lucky with clocks, and on a power-limited consumer card (the 7900 XTX
    clocks down under sustained load) the first round is systematically the
    fastest. Median is the number you would actually see in a training loop.

    5 rounds, not 3: at 3 a single transient round (something else touching
    the GPU, a clock excursion) can land on the median and move a reported
    speedup by 4-5x -- observed on RMSNorm 4096x4096, which read 0.27x once
    and 1.26-1.31x on every repeat.
    """
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(n_rounds):
        start.record()
        for _ in range(n_iter):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / n_iter)
    return statistics.median(times)


def _fwd_only(fn, args):
    def run():
        with torch.no_grad():
            fn(*args)
    return run


def _fwd_bwd(fn, args, grad_out, leaves):
    """fwd+bwd closure. .grad is cleared each iteration so gradient
    accumulation doesn't grow the work (and so a leaked graph shows up as an
    error rather than as a quietly rising number)."""
    def run():
        for t in leaves:
            t.grad = None
        out = fn(*args)
        out.backward(grad_out)
    return run


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------

def _max_err(a, b):
    if a is None or b is None:
        return float("nan")
    return (a.float() - b.float()).abs().max().item()


def _grads_of(fn, args, grad_out, leaves):
    for t in leaves:
        t.grad = None
    out = fn(*args)
    out.backward(grad_out)
    return out.detach().clone(), [None if t.grad is None else t.grad.detach().clone()
                                  for t in leaves]


def _numerics(te_fn, ref_fn, args, grad_out, leaves, ref32_fn=None, args32=None,
              leaves32=None, grad_out32=None):
    """max|TE - stock| for output and each grad, plus stock's own max error
    against the same computation in fp32 -- the yardstick that says whether
    the TE-vs-stock gap is a bug or just low-precision noise."""
    te_out, te_grads = _grads_of(te_fn, args, grad_out, leaves)
    ref_out, ref_grads = _grads_of(ref_fn, args, grad_out, leaves)

    out_err = _max_err(te_out, ref_out)
    grad_errs = [_max_err(g, r) for g, r in zip(te_grads, ref_grads)]

    baseline = float("nan")
    if ref32_fn is not None:
        out32, grads32 = _grads_of(ref32_fn, args32, grad_out32, leaves32)
        baseline = max([_max_err(ref_out, out32)]
                       + [_max_err(r, g32) for r, g32 in zip(ref_grads, grads32)])
    return out_err, grad_errs, baseline


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_rows = []


def report(op, shape, t_ref_f, t_te_f, t_ref_b, t_te_b, out_err, grad_errs, baseline):
    su_f = t_ref_f / t_te_f
    su_b = t_ref_b / t_te_b
    verdict = "FASTER" if su_b > 1.0 else "slower"
    print(f"\n{op}  [{shape}]")
    print(f"  fwd      stock {t_ref_f:8.3f} ms   te_ops {t_te_f:8.3f} ms   "
          f"speedup {su_f:6.2f}x")
    print(f"  fwd+bwd  stock {t_ref_b:8.3f} ms   te_ops {t_te_b:8.3f} ms   "
          f"speedup {su_b:6.2f}x   [{verdict}]")
    grad_str = ", ".join(f"{e:.2e}" for e in grad_errs)
    print(f"  numerics max|TE-stock| out {out_err:.2e}  grads [{grad_str}]"
          f"   (stock's own fp32 error: {baseline:.2e})")
    _rows.append((op, shape, t_ref_f, t_te_f, su_f, t_ref_b, t_te_b, su_b,
                  max([out_err] + list(grad_errs)), baseline))


def summary():
    if not _rows:
        return
    print(f"\n{'=' * 78}")
    print("SUMMARY  (speedup > 1.00 means te_ops beats stock PyTorch)")
    print(f"{'=' * 78}")
    print(f"{'op':<26} {'shape':<22} {'fwd':>8} {'fwd+bwd':>9} {'max err':>10}")
    for op, shape, _, _, su_f, _, _, su_b, err, _ in _rows:
        print(f"{op:<26} {shape:<22} {su_f:7.2f}x {su_b:8.2f}x {err:10.2e}")
    fwd = statistics.median(r[4] for r in _rows)
    bwd = statistics.median(r[7] for r in _rows)
    print(f"\nmedian speedup: fwd {fwd:.2f}x, fwd+bwd {bwd:.2f}x")


# ---------------------------------------------------------------------------
# Op benchmarks
# ---------------------------------------------------------------------------

def bench_norms(dtype, iters, shapes):
    for rows, cols in shapes:
        shape = f"{rows}x{cols}"

        # --- LayerNorm ---
        x = torch.randn(rows, cols, device=device, dtype=dtype, requires_grad=True)
        w = torch.randn(cols, device=device, dtype=dtype, requires_grad=True)
        b = torch.randn(cols, device=device, dtype=dtype, requires_grad=True)
        go = torch.randn(rows, cols, device=device, dtype=dtype)
        leaves = [x, w, b]
        x32, w32, b32 = (t.detach().float().requires_grad_(True) for t in leaves)

        ref = lambda x, w, b: F.layer_norm(x, [cols], w, b, eps=1e-5)
        te = lambda x, w, b: te_ops.layer_norm(x, [cols], w, b, 1e-5)

        t_ref_f = _time_ms(_fwd_only(ref, (x, w, b)), iters)
        t_te_f = _time_ms(_fwd_only(te, (x, w, b)), iters)
        t_ref_b = _time_ms(_fwd_bwd(ref, (x, w, b), go, leaves), iters)
        t_te_b = _time_ms(_fwd_bwd(te, (x, w, b), go, leaves), iters)
        errs = _numerics(te, ref, (x, w, b), go, leaves,
                         ref32_fn=ref, args32=(x32, w32, b32),
                         leaves32=[x32, w32, b32], grad_out32=go.float())
        report("LayerNorm", shape, t_ref_f, t_te_f, t_ref_b, t_te_b, *errs)
        del x, w, b, go, x32, w32, b32
        torch.cuda.empty_cache()

        # --- RMSNorm ---
        if not hasattr(F, "rms_norm"):
            print("[Skip] RMSNorm: torch.nn.functional.rms_norm unavailable")
            continue
        x = torch.randn(rows, cols, device=device, dtype=dtype, requires_grad=True)
        w = torch.randn(cols, device=device, dtype=dtype, requires_grad=True)
        go = torch.randn(rows, cols, device=device, dtype=dtype)
        leaves = [x, w]
        x32, w32 = (t.detach().float().requires_grad_(True) for t in leaves)

        # eps=None on both sides: te_ops.rms_norm and F.rms_norm both fall
        # back to torch.finfo(dtype).eps, so this compares like with like.
        ref = lambda x, w: F.rms_norm(x, [cols], w)
        te = lambda x, w: te_ops.rms_norm(x, [cols], w)

        t_ref_f = _time_ms(_fwd_only(ref, (x, w)), iters)
        t_te_f = _time_ms(_fwd_only(te, (x, w)), iters)
        t_ref_b = _time_ms(_fwd_bwd(ref, (x, w), go, leaves), iters)
        t_te_b = _time_ms(_fwd_bwd(te, (x, w), go, leaves), iters)
        errs = _numerics(te, ref, (x, w), go, leaves,
                         ref32_fn=ref, args32=(x32, w32),
                         leaves32=[x32, w32], grad_out32=go.float())
        report("RMSNorm", shape, t_ref_f, t_te_f, t_ref_b, t_te_b, *errs)
        del x, w, go, x32, w32
        torch.cuda.empty_cache()


def bench_activations(dtype, iters, shapes):
    for rows, cols in shapes:
        shape = f"{rows}x{cols}"
        x = torch.randn(rows, cols, device=device, dtype=dtype, requires_grad=True)
        go = torch.randn(rows, cols, device=device, dtype=dtype)
        x32 = x.detach().float().requires_grad_(True)

        # te_ops.gelu is TE's tanh-approximate GELU, so the stock comparison
        # has to be approximate="tanh" too -- against the default exact erf
        # GELU the numerics column would show a ~1e-3 "error" that is just a
        # different function.
        for name, ref, te in (
            ("GELU (tanh)", lambda x: F.gelu(x, approximate="tanh"), te_ops.gelu),
            ("SiLU", F.silu, te_ops.silu),
        ):
            t_ref_f = _time_ms(_fwd_only(ref, (x,)), iters)
            t_te_f = _time_ms(_fwd_only(te, (x,)), iters)
            t_ref_b = _time_ms(_fwd_bwd(ref, (x,), go, [x]), iters)
            t_te_b = _time_ms(_fwd_bwd(te, (x,), go, [x]), iters)
            errs = _numerics(te, ref, (x,), go, [x],
                             ref32_fn=ref, args32=(x32,), leaves32=[x32],
                             grad_out32=go.float())
            report(name, shape, t_ref_f, t_te_f, t_ref_b, t_te_b, *errs)
        del x, go, x32
        torch.cuda.empty_cache()


def bench_attention(dtype, iters, shapes):
    for B, H, S, D in shapes:
        shape = f"B{B} H{H} S{S} D{D}"
        q, k, v = (torch.randn(B, H, S, D, device=device, dtype=dtype,
                               requires_grad=True) for _ in range(3))
        go = torch.randn(B, H, S, D, device=device, dtype=dtype)
        leaves = [q, k, v]
        q32, k32, v32 = (t.detach().float().requires_grad_(True) for t in leaves)

        # Three call shapes, because te_ops routes them to three different
        # attn_mask_type strings and only the first two exist in bench.py:
        #   is_causal=True   -> "causal"
        #   no mask          -> "no_mask"
        #   bottom-right mask-> "causal_bottom_right" (the KV-cache/HF path
        #                       that is_bottom_right_causal_mask() unlocks)
        br_mask = te_ops._bottom_right_causal_reference(S, S, device)

        variants = [
            ("Attention causal",
             lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True),
             lambda q, k, v: te_ops.scaled_dot_product_attention(q, k, v, is_causal=True)),
            ("Attention no_mask",
             lambda q, k, v: F.scaled_dot_product_attention(q, k, v),
             lambda q, k, v: te_ops.scaled_dot_product_attention(q, k, v)),
            ("Attention br-causal mask",
             lambda q, k, v: F.scaled_dot_product_attention(q, k, v, attn_mask=br_mask),
             lambda q, k, v: te_ops.scaled_dot_product_attention(q, k, v, attn_mask=br_mask)),
        ]

        for name, ref, te in variants:
            try:
                t_ref_f = _time_ms(_fwd_only(ref, (q, k, v)), iters)
                t_te_f = _time_ms(_fwd_only(te, (q, k, v)), iters)
                t_ref_b = _time_ms(_fwd_bwd(ref, (q, k, v), go, leaves), iters)
                t_te_b = _time_ms(_fwd_bwd(te, (q, k, v), go, leaves), iters)
                errs = _numerics(te, ref, (q, k, v), go, leaves,
                                 ref32_fn=ref, args32=(q32, k32, v32),
                                 leaves32=[q32, k32, v32], grad_out32=go.float())
            except (RuntimeError, torch.OutOfMemoryError) as exc:
                # TE's unfused attention materializes the full S x S score
                # matrix; at large S it OOMs where PyTorch's flash path does
                # not. That is a result, not a crash -- record and move on.
                print(f"\n{name}  [{shape}]\n  FAILED: {type(exc).__name__}: "
                      f"{str(exc).splitlines()[0][:140]}")
                torch.cuda.empty_cache()
                continue
            report(name, shape, t_ref_f, t_te_f, t_ref_b, t_te_b, *errs)
        del q, k, v, go, q32, k32, v32
        torch.cuda.empty_cache()


DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dtype", default="fp16", choices=list(DTYPES) + ["all"])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--quick", action="store_true",
                    help="one small shape per op (keeps the run well under a minute)")
    ap.add_argument("--skip-attention", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no GPU visible")
    if not te_ops.available():
        sys.exit("te_ops unavailable -- run with AMD_TUNED_TORCH_ENABLE_TE=1 set "
                 "in the environment (and check `python -c \"import "
                 "transformer_engine.pytorch\"` works on its own first)")
    # Confirm which copy of te_ops is actually being measured (see the
    # sys.path.insert at the top): benchmarking the site-packages copy
    # silently reports whatever was last installed there, not the source tree.
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    measured = pathlib.Path(te_ops.__file__).resolve()
    print(f"te_ops      : {measured}")
    if repo_root not in measured.parents:
        print(f"[WARNING] that is NOT the source tree at {repo_root} -- these "
              f"numbers describe the installed copy, which may be stale.")
    print(f"torch       : {torch.__version__}  (hip {torch.version.hip})")
    print(f"device      : {torch.cuda.get_device_name(0)}")
    print(f"NVTE_DISABLE_NVRTC={os.environ.get('NVTE_DISABLE_NVRTC')}")

    if args.quick:
        norm_shapes = [(4096, 4096)]
        act_shapes = [(4096, 4096)]
        attn_shapes = [(2, 16, 1024, 64)]
    else:
        norm_shapes = [(4096, 4096), (16384, 2048)]
        act_shapes = [(8192, 4096)]
        attn_shapes = [(2, 16, 1024, 64), (2, 32, 2048, 128)]

    # Measure the raw ops against stock F.foo: with the monkeypatch installed,
    # the "stock" side of every comparison would be te_ops as well.
    was_enabled = amd_tuned_torch.is_enabled()
    if was_enabled:
        amd_tuned_torch.disable()
    try:
        for name in (list(DTYPES) if args.dtype == "all" else [args.dtype]):
            dtype = DTYPES[name]
            print(f"\n{'=' * 78}\n{name.upper()}\n{'=' * 78}")
            bench_norms(dtype, args.iters, norm_shapes)
            bench_activations(dtype, args.iters, act_shapes)
            if not args.skip_attention:
                bench_attention(dtype, args.iters, attn_shapes)
        summary()
    finally:
        if was_enabled:
            amd_tuned_torch.enable()


if __name__ == "__main__":
    main()
