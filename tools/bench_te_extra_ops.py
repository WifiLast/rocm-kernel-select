"""Benchmark amd_tuned_torch.te_extra_ops against stock PyTorch/ROCm, on RX
7900 XTX (gfx1100/RDNA3).

te_extra_ops.py wraps four TransformerEngine kernel families that have no
monkeypatch (callers reach for them by name), and until this script none of
them had been measured -- unlike te_ops, whose gelu/silu turned out to LOSE to
stock (analyse/benchmarks/te_ops_bench.txt). This answers the same question
for the extra ops: does calling them by name buy anything on this card?

Each family is compared with the best stock equivalent a caller would
otherwise use, not a strawman:

  optimizers  multi_tensor_adam  vs torch.optim.AdamW(fused=True) and (foreach=True)
              multi_tensor_sgd   vs torch.optim.SGD(fused=True) and (foreach=True)
              multi_tensor_l2norm vs torch.nn.utils.get_total_norm (foreach)
              multi_tensor_scale vs torch._foreach_mul_
  softmax     scaled_softmax / upper_triang_causal_softmax vs torch.softmax
              (+ masked_fill for the masked/causal variants)
  dropout     te dropout vs F.dropout, plus saved-for-backward memory
  layout      thd<->bshd, pad_rows, swap_first_dims vs index/copy equivalents
              (the stock side gets its index tensors precomputed -- generous
              to stock, since a real caller would cache them too)

Numerics are reported as max|TE - stock| with stock's own error against an
fp32 reference where that is meaningful. Dropout cannot be compared
elementwise (different RNG and mask layout), so it is checked structurally:
keep rate, survivor scaling, and that the gradient uses the forward's mask.

Run with (the shell exports AUTOPATCH=1 and a user-site torch otherwise wins):

    AMD_TUNED_TORCH_ENABLE_TE=1 AMD_TUNED_TORCH_AUTOPATCH=0 PYTHONNOUSERSITE=1 \\
        python tools/bench_te_extra_ops.py [--quick] [--only optim,softmax,...]
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(1, str(pathlib.Path(__file__).resolve().parent))

os.environ.setdefault("NVTE_DISABLE_NVRTC", "1")
os.environ.pop("TORCH_LOGS", None)
os.environ.pop("TORCH_COMPILE_DEBUG", None)

import torch
import torch.nn.functional as F

import amd_tuned_torch
import amd_tuned_torch.te_extra_ops as tx
from bench_te_ops import _time_ms, _max_err  # same timing method as te_ops_bench

device = torch.device("cuda")
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

_rows = []


def report(op, shape, timings, err="", note=""):
    """timings: list of (label, t_stock_ms, t_te_ms)."""
    print(f"\n{op}  [{shape}]")
    for label, t_ref, t_te in timings:
        su = t_ref / t_te
        print(f"  {label:<22} stock {t_ref:8.3f} ms   te {t_te:8.3f} ms   "
              f"speedup {su:6.2f}x   [{'FASTER' if su > 1.0 else 'slower'}]")
        _rows.append((op, shape, label, su))
    if err:
        print(f"  numerics {err}")
    if note:
        print(f"  note     {note}")


def summary():
    if not _rows:
        return
    print(f"\n{'=' * 84}\nSUMMARY  (speedup > 1.00 means te_extra_ops beats stock)\n{'=' * 84}")
    print(f"{'op':<30} {'shape':<26} {'vs':<22} {'speedup':>8}")
    for op, shape, label, su in _rows:
        print(f"{op:<30} {shape:<26} {label:<22} {su:7.2f}x")


# ---------------------------------------------------------------------------
# 1. multi-tensor optimizers
# ---------------------------------------------------------------------------

def _param_set(kind, dtype):
    """A realistic parameter list. "llm": a small transformer's worth of big
    matrices plus norms/biases; "unet": many small conv/norm tensors -- the
    launch-bound case multi-tensor kernels exist for."""
    g = torch.Generator(device="cpu").manual_seed(0)
    shapes = []
    if kind == "llm":           # ~12 layers, hidden 1024, ffn 4096 (~150M params)
        for _ in range(12):
            shapes += [(3072, 1024), (3072,), (1024, 1024), (1024,), (1024,), (1024,),
                       (4096, 1024), (4096,), (1024, 4096), (1024,), (1024,), (1024,)]
        shapes += [(32000, 1024)]
    else:                       # ~600 tensors, mostly small
        for i in range(100):
            c = (64, 128, 256, 512)[i % 4]
            shapes += [(c, c, 3, 3), (c,), (c,), (c,), (c, c, 1, 1), (c,)]
    return [torch.randn(s, generator=g).to(device=device, dtype=dtype) for s in shapes]


def _clone_all(ts):
    return [t.clone() for t in ts]


def bench_optim(iters, quick):
    kinds = ["unet"] if quick else ["unet", "llm"]
    for kind in kinds:
        for pdtype in (torch.float32, torch.bfloat16):
            base = _param_set(kind, pdtype)
            grads = [torch.randn_like(p) * 1e-2 for p in base]
            n = sum(p.numel() for p in base)
            shape = f"{kind} {len(base)}t {n/1e6:.0f}M {str(pdtype)[6:]}"
            hp = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

            # --- Adam ---
            def torch_opt(**kw):
                ps = [torch.nn.Parameter(p.clone()) for p in base]
                for p, g in zip(ps, grads):
                    p.grad = g.clone()
                return ps, torch.optim.AdamW(ps, **hp, **kw)

            ps_f, opt_f = torch_opt(fused=True)
            ps_e, opt_e = torch_opt(foreach=True)
            te_p = _clone_all(base)
            te_m = [torch.zeros_like(p) for p in base]
            te_v = [torch.zeros_like(p) for p in base]
            te_g = _clone_all(grads)

            def te_step(step=1):
                tx.multi_tensor_adam(te_g, te_p, te_m, te_v, lr=hp["lr"], beta1=0.9,
                                     beta2=0.999, eps=1e-8, step=step,
                                     adam_w_mode=True, weight_decay=0.01)

            # numerics: one step from identical state
            try:
                with torch.no_grad():
                    opt_f.step()
                    te_step(1)
                err = max(_max_err(a, b) for a, b in zip(te_p, ps_f))
                err32 = ""
                if pdtype != torch.float32:
                    ref = [p.float().clone() for p in base]
                    ps32 = [torch.nn.Parameter(r) for r in ref]
                    for p, g in zip(ps32, grads):
                        p.grad = g.float()
                    torch.optim.AdamW(ps32, **hp, foreach=True).step()
                    err32 = f"   (stock's own err vs fp32: " \
                            f"{max(_max_err(a, b) for a, b in zip(ps_f, ps32)):.2e})"
                t_te = _time_ms(lambda: te_step(5), iters)
                t_f = _time_ms(opt_f.step, iters)
                t_e = _time_ms(opt_e.step, iters)
                report("multi_tensor_adam (AdamW)", shape,
                       [("vs AdamW fused=True", t_f, t_te),
                        ("vs AdamW foreach=True", t_e, t_te)],
                       err=f"params after 1 step max|TE-fused| {err:.2e}{err32}")
            except Exception as e:  # noqa: BLE001 -- report and keep going
                print(f"\nmulti_tensor_adam [{shape}]  FAILED: {type(e).__name__}: {e}")

            # --- SGD (momentum) ---
            def torch_sgd(**kw):
                ps = [torch.nn.Parameter(p.clone()) for p in base]
                for p, g in zip(ps, grads):
                    p.grad = g.clone()
                return ps, torch.optim.SGD(ps, lr=1e-2, momentum=0.9, weight_decay=0.01, **kw)

            try:
                ps_s, opt_s = torch_sgd(fused=True)
                _, opt_se = torch_sgd(foreach=True)
                sp = _clone_all(base)
                sb = [torch.zeros_like(p) for p in base]
                sg = _clone_all(grads)

                def te_sgd(first):
                    tx.multi_tensor_sgd(sg, sp, sb, lr=1e-2, momentum=0.9,
                                        weight_decay=0.01, first_run=first)
                with torch.no_grad():
                    opt_s.step()
                    te_sgd(True)
                    opt_s.step()
                    te_sgd(False)
                err = max(_max_err(a, b) for a, b in zip(sp, ps_s))
                t_te = _time_ms(lambda: te_sgd(False), iters)
                t_f = _time_ms(opt_s.step, iters)
                t_e = _time_ms(opt_se.step, iters)
                report("multi_tensor_sgd (momentum)", shape,
                       [("vs SGD fused=True", t_f, t_te),
                        ("vs SGD foreach=True", t_e, t_te)],
                       err=f"params after 2 steps max|TE-fused| {err:.2e}")
            except Exception as e:  # noqa: BLE001
                print(f"\nmulti_tensor_sgd [{shape}]  FAILED: {type(e).__name__}: {e}")

            # --- grad-norm + scale (grad clipping / AMP unscale) ---
            try:
                norm_te, _ = tx.multi_tensor_l2norm(grads)
                norm_ref = torch.nn.utils.get_total_norm(grads, 2.0, foreach=True)
                norm64 = torch.sqrt(sum((g.double() ** 2).sum() for g in grads))
                t_te = _time_ms(lambda: tx.multi_tensor_l2norm(grads), iters)
                t_ref = _time_ms(lambda: torch.nn.utils.get_total_norm(grads, 2.0,
                                                                        foreach=True), iters)
                rel = lambda a: abs(a.double().item() - norm64.item()) / norm64.item()
                report("multi_tensor_l2norm", shape, [("vs get_total_norm", t_ref, t_te)],
                       err=f"rel err vs fp64: te {rel(norm_te):.2e}  stock {rel(norm_ref):.2e}")

                outs = [torch.empty_like(g) for g in grads]
                tx.multi_tensor_scale(grads, outs, 0.5)
                err = max(_max_err(o, g * 0.5) for o, g in zip(outs, grads))
                work = _clone_all(grads)
                t_te = _time_ms(lambda: tx.multi_tensor_scale(work, work, 1.0), iters)
                t_ref = _time_ms(lambda: torch._foreach_mul_(work, 1.0), iters)
                report("multi_tensor_scale", shape, [("vs _foreach_mul_", t_ref, t_te)],
                       err=f"max|TE-stock| {err:.2e}",
                       note="TE also does the inf/nan check in the same pass; stock "
                            "would need an extra _amp_foreach_non_finite_check")
            except Exception as e:  # noqa: BLE001
                print(f"\nl2norm/scale [{shape}]  FAILED: {type(e).__name__}: {e}")

            del base, grads, ps_f, opt_f, ps_e, opt_e, te_p, te_m, te_v, te_g
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 2. fused softmax
# ---------------------------------------------------------------------------

def _fwd_bwd_time(fn, x, grad, iters):
    def run():
        x.grad = None
        fn(x).backward(grad)
    return _time_ms(run, iters)


def _fwd_time(fn, x, iters):
    def run():
        with torch.no_grad():
            fn(x)
    return _time_ms(run, iters)


def _softmax_case(name, shape, te_fn, ref_fn, ref32_fn, dtype, iters):
    x = (torch.randn(shape, device=device, dtype=dtype) * 3).requires_grad_()
    grad = torch.randn(shape, device=device, dtype=dtype)

    def out_and_grad(fn, inp):
        inp.grad = None
        y = fn(inp)
        y.backward(grad.to(inp.dtype))
        return y.detach().float(), inp.grad.detach().float()

    y_te, g_te = out_and_grad(te_fn, x)
    y_ref, g_ref = out_and_grad(ref_fn, x)
    x32 = x.detach().float().requires_grad_()
    y32, g32 = out_and_grad(ref32_fn, x32)
    t = [("fwd", _fwd_time(ref_fn, x, iters), _fwd_time(te_fn, x, iters)),
         ("fwd+bwd", _fwd_bwd_time(ref_fn, x, grad, iters),
          _fwd_bwd_time(te_fn, x, grad, iters))]
    report(name, "x".join(map(str, shape)) + f" {str(dtype)[6:]}", t,
           err=f"max|TE-stock| out {_max_err(y_te, y_ref):.2e} grad {_max_err(g_te, g_ref):.2e}"
               f"   (stock's own fp32 err: out {_max_err(y_ref, y32):.2e} "
               f"grad {_max_err(g_ref, g32):.2e})")
    del x, grad, x32
    torch.cuda.empty_cache()


def bench_softmax(dtype, iters, quick):
    shapes = [(4, 16, 1024, 1024)] if quick else [(4, 16, 1024, 1024), (2, 32, 2048, 2048),
                                                 (1, 16, 4096, 4096)]
    scale = 0.125
    for b, h, sq, sk in shapes:
        probe = torch.empty((b, h, sq, sk), device=device, dtype=dtype)
        if not tx.kernel_available(probe, None, "no_mask"):
            print(f"\nscaled_softmax [{b}x{h}x{sq}x{sk}]  kernel rejects this shape, skipped")
            continue
        del probe

        _softmax_case("scaled_softmax no_mask", (b, h, sq, sk),
                      lambda x: tx.scaled_softmax(x, None, scale),
                      lambda x: torch.softmax(x * scale, -1),
                      lambda x: torch.softmax(x * scale, -1), dtype, iters)

        # padding-style mask, broadcast over heads (True = masked out, TE's convention)
        lens = torch.randint(sk // 2, sk + 1, (b,), device=device)
        mask = (torch.arange(sk, device=device)[None, :] >= lens[:, None])
        mask = mask[:, None, None, :].expand(b, 1, sq, sk).contiguous()
        if tx.kernel_available(torch.empty((b, h, sq, sk), device=device, dtype=dtype),
                               mask, "padding"):
            _softmax_case("scaled_masked_softmax padding", (b, h, sq, sk),
                          lambda x: tx.scaled_softmax(x, mask, scale, "padding"),
                          lambda x: torch.softmax((x * scale).masked_fill(mask, float("-inf")), -1),
                          lambda x: torch.softmax((x * scale).masked_fill(mask, float("-inf")), -1),
                          dtype, iters)

        if sq == sk:
            causal = torch.ones(sq, sk, device=device, dtype=torch.bool).triu(1)
            _softmax_case("upper_triang_causal_softmax", (b * h, sq, sk),
                          lambda x: tx.upper_triang_causal_softmax(x, scale),
                          lambda x: torch.softmax((x * scale).masked_fill(causal, float("-inf")), -1),
                          lambda x: torch.softmax((x * scale).masked_fill(causal, float("-inf")), -1),
                          dtype, iters)


# ---------------------------------------------------------------------------
# 3. fused dropout
# ---------------------------------------------------------------------------

def _saved_bytes(fn, x):
    """Bytes kept alive by the autograd graph after forward (output excluded)."""
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    y = fn(x)
    torch.cuda.synchronize()
    held = torch.cuda.memory_allocated() - before - y.numel() * y.element_size()
    del y
    return held


def bench_dropout(dtype, iters, quick):
    shapes = [(8192, 4096)] if quick else [(8192, 4096), (4, 4096, 5120)]
    p = 0.1
    for shape in shapes:
        x = torch.randn(shape, device=device, dtype=dtype).requires_grad_()
        grad = torch.randn(shape, device=device, dtype=dtype)
        te_fn = lambda t: tx.dropout(t, p)
        ref_fn = lambda t: F.dropout(t, p, training=True)

        x.grad = None
        y = te_fn(x)
        y.backward(torch.ones_like(y))
        # backward(ones) gives exactly mask * 1/(1-p): use it as the mask
        # (dividing y by x breaks on inputs that are 0 or subnormal in fp16).
        gx = x.grad.float()
        keep = gx != 0
        keep_rate = keep.float().mean().item()
        grad_vals_ok = bool(((gx[keep] - 1 / (1 - p)).abs() < 1e-2).all())
        expect = (x.detach().float() * gx).to(dtype).float()
        fwd_bwd_err = (y.detach().float() - expect).abs().max().item()
        del y, gx, expect

        t = [("fwd", _fwd_time(ref_fn, x, iters), _fwd_time(te_fn, x, iters)),
             ("fwd+bwd", _fwd_bwd_time(ref_fn, x, grad, iters),
              _fwd_bwd_time(te_fn, x, grad, iters))]
        mem_te, mem_ref = _saved_bytes(te_fn, x), _saved_bytes(ref_fn, x)
        report(f"dropout p={p}", "x".join(map(str, shape)) + f" {str(dtype)[6:]}", t,
               err=f"keep rate {keep_rate:.4f} (expect {1-p:.4f}), grad = mask/(1-p): "
                   f"{grad_vals_ok}, max|y - x*grad| {fwd_bwd_err:.2e} (0 = same mask)",
               note=f"saved for backward: te {mem_te/2**20:.1f} MiB vs stock "
                    f"{mem_ref/2**20:.1f} MiB")
        del x, grad
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 4. layout utilities
# ---------------------------------------------------------------------------

def bench_layout(dtype, iters, quick):
    g = torch.Generator(device="cpu").manual_seed(1)
    b, max_s, h, d = (8, 2048, 16, 128) if not quick else (8, 1024, 16, 64)
    lens = torch.randint(max_s // 4, max_s + 1, (b,), generator=g)
    lens[0] = max_s
    cu = torch.zeros(b + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(lens, 0)
    total = int(cu[-1])
    cu_d = cu.to(device)
    # flat padded-row index for every packed token (precomputed: generous to stock)
    idx = torch.cat([i * max_s + torch.arange(int(l)) for i, l in enumerate(lens)]).to(device)

    thd = torch.randn(total, h, d, device=device, dtype=dtype)
    bshd = torch.randn(b, max_s, h, d, device=device, dtype=dtype)

    def ref_thd_to_bshd(t):
        out = t.new_zeros(b * max_s, h, d)
        out.index_copy_(0, idx, t)
        return out.view(b, max_s, h, d)

    def ref_bshd_to_thd(t):
        return t.reshape(b * max_s, h, d).index_select(0, idx)

    shape = f"b{b} s{max_s} h{h} d{d} tok{total} {str(dtype)[6:]}"
    try:
        e1 = _max_err(tx.thd_to_bshd(thd, cu_d, max_s), ref_thd_to_bshd(thd))
        e2 = _max_err(tx.bshd_to_thd(bshd, cu_d), ref_bshd_to_thd(bshd))
        with torch.no_grad():
            report("thd_to_bshd", shape,
                   [("vs index_copy_", _time_ms(lambda: ref_thd_to_bshd(thd), iters),
                     _time_ms(lambda: tx.thd_to_bshd(thd, cu_d, max_s), iters))],
                   err=f"max|TE-stock| {e1:.2e}")
            report("bshd_to_thd", shape,
                   [("vs index_select", _time_ms(lambda: ref_bshd_to_thd(bshd), iters),
                     _time_ms(lambda: tx.bshd_to_thd(bshd, cu_d), iters))],
                   err=f"max|TE-stock| {e2:.2e}")
    except Exception as e:  # noqa: BLE001
        print(f"\nthd/bshd [{shape}]  FAILED: {type(e).__name__}: {e}")

    # pad_rows: grouped-GEMM row padding to a multiple of 16
    try:
        rows = [int(r) for r in torch.randint(1, 3000, (32,), generator=g)]
        padded = [math.ceil(r / 16) * 16 for r in rows]
        cols = 4096
        inp = torch.randn(sum(rows), cols, device=device, dtype=dtype)
        out = torch.empty(sum(padded), cols, device=device, dtype=dtype)

        def ref_pad():
            return torch.cat([F.pad(c, (0, 0, 0, pr - r))
                              for c, r, pr in zip(inp.split(rows), rows, padded)])
        tx.pad_rows(inp, out, rows, padded)
        err = _max_err(out, ref_pad())
        report("pad_rows", f"32 groups x {cols} cols {str(dtype)[6:]}",
               [("vs split+F.pad+cat", _time_ms(ref_pad, iters),
                 _time_ms(lambda: tx.pad_rows(inp, out, rows, padded), iters))],
               err=f"max|TE-stock| {err:.2e}")
        del inp, out
    except Exception as e:  # noqa: BLE001
        print(f"\npad_rows  FAILED: {type(e).__name__}: {e}")

    # swap_first_dims: (s, b, h*d) -> (b, s, h*d) materialized
    try:
        x = torch.randn(max_s, b, h * d, device=device, dtype=dtype)
        err = _max_err(tx.swap_first_dims(x), x.transpose(0, 1).contiguous())
        report("swap_first_dims", "x".join(map(str, x.shape)) + f" {str(dtype)[6:]}",
               [("vs transpose.contiguous", _time_ms(lambda: x.transpose(0, 1).contiguous(), iters),
                 _time_ms(lambda: tx.swap_first_dims(x), iters))],
               err=f"max|TE-stock| {err:.2e}")
    except Exception as e:  # noqa: BLE001
        print(f"\nswap_first_dims  FAILED: {type(e).__name__}: {e}")
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dtype", default="fp16", choices=list(DTYPES) + ["all"],
                    help="activation dtype for softmax/dropout/layout (optimizers "
                         "always run fp32 and bf16 params)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--quick", action="store_true", help="one small shape per op")
    ap.add_argument("--only", default="optim,softmax,dropout,layout")
    args = ap.parse_args()
    only = set(args.only.split(","))

    if not torch.cuda.is_available():
        sys.exit("no GPU visible")
    if not tx.available():
        sys.exit("te_extra_ops unavailable -- run with AMD_TUNED_TORCH_ENABLE_TE=1")
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    measured = pathlib.Path(tx.__file__).resolve()
    print(f"te_extra_ops: {measured}")
    if repo_root not in measured.parents:
        print(f"[WARNING] NOT the source tree at {repo_root} -- may be a stale copy.")
    print(f"torch       : {torch.__version__}  (hip {torch.version.hip})")
    print(f"device      : {torch.cuda.get_device_name(0)}")

    was_enabled = amd_tuned_torch.is_enabled()
    if was_enabled:
        amd_tuned_torch.disable()
    try:
        if "optim" in only:
            print(f"\n{'=' * 84}\nOPTIMIZERS\n{'=' * 84}", flush=True)
            bench_optim(args.iters, args.quick)
        for name in (list(DTYPES) if args.dtype == "all" else [args.dtype]):
            dtype = DTYPES[name]
            print(f"\n{'=' * 84}\n{name.upper()}\n{'=' * 84}", flush=True)
            if "softmax" in only:
                bench_softmax(dtype, args.iters, args.quick)
            if "dropout" in only:
                bench_dropout(dtype, args.iters, args.quick)
            if "layout" in only:
                bench_layout(dtype, args.iters, args.quick)
        summary()
    finally:
        if was_enabled:
            amd_tuned_torch.enable()


if __name__ == "__main__":
    main()
