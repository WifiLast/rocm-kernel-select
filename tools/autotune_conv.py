"""Runs the conv2d/conv3d kernel contest across a corpus of real shapes and
reports who wins, by how much, and what that implies for tuning effort.

The point is to aim optimisation work rather than assume where it pays.
`tools/bench.py` and `tools/bench_conv2d_fp16.py` measure one or two shapes;
a model runs dozens, and the winner is strongly shape-dependent -- the CK
tier alone picks different tuned instances for 2D vs 3D, and small convs
behave nothing like the big ones the benchmarks use.

Corpus is tools/kernelgen/shapes.py (some shapes transcribed from a real
MIOpen trace on this card), extended here with small U-Net-ish convs, since
a diffusion model spends many of its calls on those and they are exactly
where fixed per-call overheads stop being negligible.

Run with:

    python tools/autotune_conv.py                 # fp16 + bf16 + fp32
    python tools/autotune_conv.py fp16            # one dtype
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "kernelgen"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import amd_tuned_torch  # noqa: E402
from amd_tuned_torch import ck_ops, compile_ops  # noqa: E402

from shapes import CONV2D_SHAPES, CONV3D_SHAPES  # noqa: E402

ANALYSE = Path(__file__).resolve().parent.parent / "analyse"

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

# Small convs the corpus doesn't cover. A U-Net issues far more of these
# than it does of the big VAE-decoder shapes, and per-call overhead that is
# invisible at 1.4ms is 20% at 0.05ms.
EXTRA_2D = [
    ("unet_small_320ch_32x32", 1, 320, 32, 32, 320, 3, 1, 1),
    ("unet_small_640ch_16x16", 1, 640, 16, 16, 640, 3, 1, 1),
    ("unet_tiny_1280ch_8x8", 1, 1280, 8, 8, 1280, 3, 1, 1),
    ("sd_vae_128ch_512x512", 1, 128, 512, 512, 128, 3, 1, 1),
]


def timed(fn, n, warmup):
    try:
        for _ in range(warmup):
            if fn() is None:
                return None
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(n):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / n
    except (RuntimeError, TypeError, AssertionError):
        return None


def bench_case(label, ndim, N, C, spatial, K, k, stride, pad, dtype):
    dev = "cuda"
    try:
        x = torch.randn(N, C, *spatial, device=dev, dtype=dtype)
        w = torch.randn(K, C, *([k] * ndim), device=dev, dtype=dtype)
        b = torch.randn(K, device=dev, dtype=dtype)
    except RuntimeError:
        return None  # OOM

    conv = F.conv2d if ndim == 2 else F.conv3d
    nat = compile_ops.conv2d_native if ndim == 2 else compile_ops.conv3d_native
    ckf = ck_ops.conv2d if ndim == 2 else ck_ops.conv3d
    st = [stride] * ndim
    pd = [pad] * ndim
    dl = [1] * ndim
    fmt = torch.channels_last if ndim == 2 else torch.channels_last_3d

    out_sp = [(s + 2 * pad - k) // stride + 1 for s in spatial]
    flops = 2 * N * C * K * (k ** ndim)
    for o in out_sp:
        flops *= o
    # keep each case around ~0.2s of GPU time
    n = max(5, min(200, int(2e11 / max(flops, 1))))

    res = {"stock": timed(lambda: conv(x, w, b, stride=stride, padding=pad), n, 5)}
    res["native"] = timed(lambda: nat(x, w, b, st, pd, dl), n, 5)
    res["ck"] = timed(lambda: ckf(x, w, b, stride, pad, 1), n, 5) if ck_ops.available() else None
    xc, wc = x.contiguous(memory_format=fmt), w.contiguous(memory_format=fmt)
    res["ck_cl"] = timed(lambda: ckf(xc, wc, b, stride, pad, 1), n, 5) if ck_ops.available() else None

    del x, w, b, xc, wc
    torch.cuda.empty_cache()
    return res, flops


def main():
    wanted = sys.argv[1:] or list(DTYPES)
    amd_tuned_torch.disable()  # candidates are called directly, not via F.*

    cases = []
    for s in CONV2D_SHAPES:
        cases.append((s.label, 2, s.B, s.C_in, (s.H_in, s.W_in), s.C_out, s.K, s.stride, s.padding))
    for lbl, N, C, H, W, K, k, st, pd in EXTRA_2D:
        cases.append((lbl, 2, N, C, (H, W), K, k, st, pd))
    for s in CONV3D_SHAPES:
        cases.append((s.label, 3, s.B, s.C_in, (s.D_in, s.H_in, s.W_in), s.C_out, s.K,
                      s.stride, s.padding))

    lines = []
    def emit(t=""):
        print(t)
        lines.append(t)

    emit(f"{'shape':34s} {'dt':5s} {'stock':>8s} {'native':>8s} {'ck':>8s} {'ck_cl':>8s} "
         f"{'winner':>8s} {'gain':>7s}")
    tally = {}
    for (label, ndim, N, C, spatial, K, k, stride, pad) in cases:
        for name in wanted:
            dtype = DTYPES[name]
            out = bench_case(label, ndim, N, C, spatial, K, k, stride, pad, dtype)
            if out is None:
                emit(f"{label[:34]:34s} {name:5s} {'OOM':>8s}")
                continue
            res, _ = out
            if res["stock"] is None:
                continue
            cand = {kk: v for kk, v in res.items() if v is not None}
            best = min(cand, key=cand.get)
            gain = res["stock"] / cand[best]
            tally[best] = tally.get(best, 0) + 1
            fmt_ = lambda v: f"{v:8.3f}" if v is not None else f"{'-':>8s}"
            emit(f"{label[:34]:34s} {name:5s} {fmt_(res['stock'])} {fmt_(res['native'])} "
                 f"{fmt_(res['ck'])} {fmt_(res['ck_cl'])} {best:>8s} {gain:6.2f}x")

    emit()
    emit("winners: " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items(), key=lambda r: -r[1])))

    ANALYSE.joinpath("benchmarks").mkdir(parents=True, exist_ok=True)
    ANALYSE.joinpath("benchmarks", "autotune_conv_corpus.txt").write_text("\n".join(lines) + "\n")
    emit(f"\nwritten -> {ANALYSE / 'benchmarks' / 'autotune_conv_corpus.txt'}")


if __name__ == "__main__":
    main()
