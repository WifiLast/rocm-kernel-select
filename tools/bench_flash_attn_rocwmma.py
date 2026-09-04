"""Benchmark (and first-ever validate) the vendored rocWMMA FlashAttention-2
kernel behind amd_tuned_torch.flash_attn_rocwmma_ops.

That module's docstring says the kernel is UNVALIDATED: upstream's own
numbers are Windows+ZLUDA, and nothing in this project had run its
Linux/ROCm path. So this script does not start with timings. It starts
with numerics, because a fast wrong kernel is worth nothing and this is
the first opportunity anyone has had to find out which one this is.

Four phases, printed as they finish so a later crash still leaves the
earlier results on screen (a bad kernel here faults the GPU, which takes
the process with it -- no exception to catch):

  1. BUILD.    Time the lazy JIT compile, which is the first-use cost a
               caller of enable_flash_attn_rocwmma() actually pays.
  2. NUMERICS. Compare against an fp32 reference, with stock's own
               fp16/bf16 SDPA error against that same reference as the
               yardstick. "Differs from fp32" is not a verdict -- every
               half-precision attention does. Being materially worse than
               stock at the same shape is.
  3. SPEED.    Wall clock vs F.scaled_dot_product_attention on the same
               tensors, as TFLOP/s and as a ratio.
  4. BACKWARD. Same two questions for the backward kernel.

The monkeypatch is never installed here: amd_tuned_torch.enable() is not
called, so "stock" in the tables below is genuinely torch's own kernel.

Run with:

    python tools/bench_flash_attn_rocwmma.py
    python tools/bench_flash_attn_rocwmma.py --no-backward
"""
import os

# Must precede `import torch`. The environment this repo is developed in
# exports TORCH_LOGS=+dynamo,+inductor and TORCH_COMPILE_DEBUG=1, which
# bury a benchmark table under thousands of lru_cache_stats lines.
#
# POP, do not set to "". torch._logging._init_logs on this build parses a
# TORCH_LOGS that is present-but-empty into a plain dict and then calls
# get_log_level_pairs() on it, so `TORCH_LOGS=""` is an AttributeError at
# `import torch` -- noisier than the logging it was meant to silence.
os.environ.pop("TORCH_LOGS", None)
os.environ.pop("TORCH_COMPILE_DEBUG", None)

# THE TRAP THIS SCRIPT EXISTS TO AVOID. amd_tuned_torch/__init__.py
# auto-patches at import (AMD_TUNED_TORCH_AUTOPATCH defaults to 1) and,
# nested under it, calls enable_flash_attn_rocwmma() by default too. So a
# plain `import amd_tuned_torch` REPLACES F.scaled_dot_product_attention
# with the very kernel this script is measuring -- the "stock" column
# would silently be the rocwmma kernel racing itself, reporting a tidy
# 1.00x, and both the numerics and the speedup would be meaningless.
#
# Turning AUTOPATCH off also keeps the JIT compile out of import, which
# is what lets phase 1 below time a genuinely cold build instead of a
# cache hit.
os.environ["AMD_TUNED_TORCH_AUTOPATCH"] = "0"

import argparse  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# Captured BEFORE amd_tuned_torch is imported at all, so that even if some
# other import path installs the monkeypatch, the baseline in every table
# below is torch's own kernel. sys.path/PYTHONPATH resolution is not
# something this script gets to be optimistic about: site-packages holds a
# stale non-editable copy of amd_tuned_torch that shadows the editable
# install and is missing _vendor/rocwmma_fattn's sources entirely (setup.py
# declares no package_data), so run this with PYTHONPATH=<repo root>.
_STOCK_SDPA = F.scaled_dot_product_attention

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from amd_tuned_torch import flash_attn_rocwmma_ops as fa  # noqa: E402

# RX 7900 XTX fp16/bf16 WMMA peak, the same figure
# tools/bench_conv2d_fp16.py measures against.
PEAK_TFLOPS = 122.9

# (batch, heads, seq_q, seq_kv, head_dim, causal, label)
SHAPES = [
    (2, 16, 1024, 1024, 64, False, "SD-XL attn (1k ctx)"),
    (1, 24, 4096, 4096, 64, False, "SD 1.5 attn (4k ctx)"),
    (1, 32, 2048, 2048, 128, True, "LLM 7B-ish, 2k causal"),
    (1, 32, 4096, 4096, 128, True, "LLM 7B-ish, 4k causal"),
    (4, 8, 512, 512, 64, False, "small/batched"),
    (1, 16, 1024, 1024, 40, False, "head_dim 40 (padded)"),
]


def _mk(b, h, n, nkv, d, dtype, requires_grad=False):
    g = torch.Generator(device="cuda").manual_seed(1234)

    def t(seq):
        x = torch.randn(b, h, seq, d, device="cuda", dtype=torch.float32, generator=g)
        return x.to(dtype).requires_grad_(requires_grad)

    return t(n), t(nkv), t(nkv)


def _ref_fp32(q, k, v, causal):
    """fp32 math-backend reference. Both candidates are measured against
    this, never against each other -- neither is the ground truth."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.MATH):
        return _STOCK_SDPA(q.float(), k.float(), v.float(), is_causal=causal)


def _err(x, ref):
    d = (x.float() - ref).abs()
    return d.max().item(), d.mean().item()


def _time(fn, warmup=5, iters=20):
    """Median of `iters` CUDA-event-timed runs. Median, not mean: the first
    few post-warmup runs still catch allocator growth, and one outlier
    should not set the number."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def _fwd_flops(b, h, n, nkv, d, causal):
    f = 4.0 * b * h * n * nkv * d          # two matmuls, 2*MNK each
    return f * 0.5 if causal else f


def _dtname(dt):
    return {torch.float16: "fp16", torch.bfloat16: "bf16"}[dt]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-backward", action="store_true")
    ap.add_argument("--dtypes", default="fp16,bf16")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no ROCm device visible")
    # Cheap proof the baseline is real, printed rather than assumed.
    patched = F.scaled_dot_product_attention is not _STOCK_SDPA
    print(f"device : {torch.cuda.get_device_name(0)}")
    print(f"module : {fa.__file__}")
    print(f"stock  : F.scaled_dot_product_attention "
          f"{'IS PATCHED (baseline uses the captured original)' if patched else 'unpatched'}")
    print(f"torch  : {torch.__version__}  (HIP {torch.version.hip})")

    # ---- 1. BUILD -----------------------------------------------------
    print("\n=== 1. JIT BUILD ===", flush=True)
    t0 = time.time()
    ok = fa.available()
    dt = time.time() - t0
    if not ok:
        print(f"FAILED after {dt:.1f}s: {fa.load_error()}")
        sys.exit(1)
    print(f"built/loaded in {dt:.1f}s "
          f"({'cold compile' if dt > 20 else 'cached'})", flush=True)

    dtypes = [{"fp16": torch.float16, "bf16": torch.bfloat16}[x]
              for x in args.dtypes.split(",")]

    # ---- 2. NUMERICS --------------------------------------------------
    print("\n=== 2. NUMERICS (vs fp32 ref; stock at same dtype is the yardstick) ===")
    hdr = (f"{'shape':<24} {'dt':>4} {'rocwmma max':>12} {'mean':>10} "
           f"{'stock max':>12} {'mean':>10}  verdict")
    print(hdr)
    print("-" * len(hdr), flush=True)
    suspect = []
    for (b, h, n, nkv, d, causal, label) in SHAPES:
        for dt_ in dtypes:
            q, k, v = _mk(b, h, n, nkv, d, dt_)
            ref = _ref_fp32(q, k, v, causal)
            try:
                o = fa.scaled_dot_product_attention(q, k, v, is_causal=causal)
                mx, mn = _err(o, ref)
            except Exception as exc:
                print(f"{label:<24} {_dtname(dt_):>4}  RAISED: {exc}", flush=True)
                suspect.append((label, _dtname(dt_), f"raised: {exc}"))
                continue
            so = _STOCK_SDPA(q, k, v, is_causal=causal)
            smx, smn = _err(so, ref)
            # 8x stock's own max error is the line: half-precision attention
            # varies with summation order, but not by an order of magnitude.
            bad = mx > max(8 * smx, 1e-2)
            if bad:
                suspect.append((label, _dtname(dt_),
                                f"fwd max {mx:.3e} vs stock {smx:.3e}"))
            print(f"{label:<24} {_dtname(dt_):>4} {mx:>12.3e} {mn:>10.3e} "
                  f"{smx:>12.3e} {smn:>10.3e}  {'SUSPECT' if bad else 'ok'}",
                  flush=True)
            del q, k, v, ref, o, so
            torch.cuda.empty_cache()

    # ---- 3. FORWARD SPEED ---------------------------------------------
    print("\n=== 3. FORWARD SPEED ===")
    hdr = (f"{'shape':<24} {'dt':>4} {'rocwmma ms':>11} {'TF/s':>7} {'%pk':>5} "
           f"{'stock ms':>9} {'TF/s':>7}  {'speedup':>8}")
    print(hdr)
    print("-" * len(hdr), flush=True)
    for (b, h, n, nkv, d, causal, label) in SHAPES:
        for dt_ in dtypes:
            q, k, v = _mk(b, h, n, nkv, d, dt_)
            fl = _fwd_flops(b, h, n, nkv, d, causal)
            try:
                ours = _time(lambda: fa.scaled_dot_product_attention(
                    q, k, v, is_causal=causal))
            except Exception as exc:
                print(f"{label:<24} {_dtname(dt_):>4}  RAISED: {exc}", flush=True)
                continue
            stock = _time(lambda: _STOCK_SDPA(q, k, v, is_causal=causal))
            print(f"{label:<24} {_dtname(dt_):>4} "
                  f"{ours:>11.3f} {fl/ours/1e9:>7.1f} "
                  f"{100*fl/ours/1e9/PEAK_TFLOPS:>4.0f}% "
                  f"{stock:>9.3f} {fl/stock/1e9:>7.1f}  {stock/ours:>7.2f}x",
                  flush=True)
            del q, k, v
            torch.cuda.empty_cache()

    # ---- 4. BACKWARD ---------------------------------------------------
    if not args.no_backward:
        print("\n=== 4. FORWARD+BACKWARD (dQ/dK/dV vs autograd through fp32) ===")
        hdr = (f"{'shape':<24} {'dt':>4} {'grad max err':>13} {'stock':>11} "
               f"{'rocwmma ms':>11} {'stock ms':>9} {'speedup':>8}")
        print(hdr)
        print("-" * len(hdr), flush=True)
        for (b, h, n, nkv, d, causal, label) in SHAPES:
            for dt_ in dtypes:
                q, k, v = _mk(b, h, n, nkv, d, dt_, requires_grad=True)
                qr, kr, vr = (x.detach().float().requires_grad_(True)
                              for x in (q, k, v))
                try:
                    o = fa.scaled_dot_product_attention(q, k, v, is_causal=causal)
                    go = torch.randn_like(o)
                    gq, gk, gv = torch.autograd.grad(o, (q, k, v), go)
                except Exception as exc:
                    print(f"{label:<24} {_dtname(dt_):>4}  RAISED: {exc}", flush=True)
                    suspect.append((label, _dtname(dt_), f"bwd raised: {exc}"))
                    continue
                ro = _ref_fp32(qr, kr, vr, causal)
                rq, rk, rv = torch.autograd.grad(ro, (qr, kr, vr), go.float())
                ours_err = max(_err(gq, rq)[0], _err(gk, rk)[0], _err(gv, rv)[0])

                so = _STOCK_SDPA(q, k, v, is_causal=causal)
                sq, sk, sv = torch.autograd.grad(so, (q, k, v), go)
                stock_err = max(_err(sq, rq)[0], _err(sk, rk)[0], _err(sv, rv)[0])
                bad = ours_err > max(8 * stock_err, 1e-2)
                if bad:
                    suspect.append((label, _dtname(dt_),
                                    f"bwd max {ours_err:.3e} vs stock {stock_err:.3e}"))

                def _ours():
                    oo = fa.scaled_dot_product_attention(q, k, v, is_causal=causal)
                    torch.autograd.grad(oo, (q, k, v), go)

                def _stock():
                    oo = _STOCK_SDPA(q, k, v, is_causal=causal)
                    torch.autograd.grad(oo, (q, k, v), go)

                ours = _time(_ours, warmup=3, iters=10)
                stock = _time(_stock, warmup=3, iters=10)
                print(f"{label:<24} {_dtname(dt_):>4} {ours_err:>13.3e} "
                      f"{stock_err:>11.3e} {ours:>11.3f} {stock:>9.3f} "
                      f"{stock/ours:>7.2f}x{'  SUSPECT' if bad else ''}",
                      flush=True)
                del q, k, v, qr, kr, vr, o, ro, so, gq, gk, gv, rq, rk, rv, sq, sk, sv
                torch.cuda.empty_cache()

    print("\n=== SUMMARY ===")
    if suspect:
        print("NUMERICALLY SUSPECT -- do not enable this backend as-is:")
        for s in suspect:
            print(f"  {s[0]} / {s[1]}: {s[2]}")
    else:
        print("numerics: no shape exceeded 8x stock's own error vs the fp32 reference.")


if __name__ == "__main__":
    main()
