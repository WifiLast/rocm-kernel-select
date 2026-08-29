"""Benchmarks the Composable Kernel conv tier against stock ROCm and this
project's own hand-written kernels, and reports which MIOpen solver stock
actually dispatched.

That last part is the point. The CK tier exists because of a specific,
measurable hole in stock ROCm rather than a general belief that CK is
faster, and this script is what establishes that: it profiles F.conv2d /
F.conv3d and prints the kernel name MIOpen chose. On gfx1100 you should
see 3x3 fp16/fp32 land on a hand-written assembly Winograd solver
(miopenSp3AsmConvFury_*_f2x3_*, ~90% of peak -- do not try to beat it),
while bf16 has no Winograd path at all and falls back to Im2d2Col_v2 plus
a Tensile GEMM at a fraction of that. CK covers bf16 natively, which is
where the win comes from.

Both CK columns matter because CK's WMMA conv instances are
channels-last only (see src/cuda/ck_conv_fwd.hpp): an NCHW caller pays a
permute in and out, a channels-last caller pays nothing. Real inference
graphs running with PYTORCH_MIOPEN_SUGGEST_NHWC=1 are usually the latter.

Run with:

    python tools/bench_ck.py
"""
import torch
import torch.nn.functional as F

import amd_tuned_torch
from amd_tuned_torch import ck_ops

# Compare raw kernels, not the F.foo monkeypatch, so the numbers isolate
# the kernels from dispatch overhead -- same rationale as tools/bench.py.
amd_tuned_torch.disable()

device = torch.device("cuda")

# RX 7900 XTX: fp16/bf16 WMMA vs fp32 vector.
PEAK = {torch.float16: 122.9, torch.bfloat16: 122.9, torch.float32: 61.4}

SHAPES = [
    # (label, ndim, N, C_in, spatial, C_out, k)
    ("conv2d", 2, 64, 128, (128, 128), 64, 3),
    ("conv3d", 3, 1, 512, (8, 32, 32), 512, 3),
]


def timed(fn, n, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(n):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n


def stock_solver_name(conv, x, w, b):
    """The kernel MIOpen actually dispatched, via the torch profiler.

    MIOpen's own MIOPEN_ENABLE_LOGGING/MIOPEN_LOG_LEVEL env vars produce
    nothing on this ROCm build, so the profiler is the reliable route.
    """
    from torch.profiler import ProfilerActivity, profile

    for _ in range(5):
        conv(x, w, b, stride=1, padding=1)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        conv(x, w, b, stride=1, padding=1)
        torch.cuda.synchronize()
    hot = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    if not hot:
        return "<no device kernel recorded>"
    return max(hot, key=lambda e: e.self_device_time_total).key


def main():
    if not ck_ops.available():
        print("[Skip] extension was built without Composable Kernel "
              "(see setup.py's AMD_TUNED_TORCH_CK_ROOT)")
        return

    for label, ndim, N, C_in, spatial, C_out, k in SHAPES:
        conv = F.conv2d if ndim == 2 else F.conv3d
        ck_conv = ck_ops.conv2d if ndim == 2 else ck_ops.conv3d
        native = amd_tuned_torch.ops.conv2d if ndim == 2 else amd_tuned_torch.ops.conv3d
        fmt = torch.channels_last if ndim == 2 else torch.channels_last_3d
        ones = [1] * ndim
        n_iter = 30 if ndim == 2 else 20

        out_spatial = spatial  # stride 1, padding 1, k=3 preserves size
        flops = 2 * N * C_in * C_out * (k ** ndim)
        for s in out_spatial:
            flops *= s

        print(f"\n=== {label}: N={N}, C_in={C_in}, C_out={C_out}, "
              f"{'x'.join(str(s) for s in spatial)}, k={k} "
              f"({flops / 1e9:.1f} GFLOP) ===")

        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            x = torch.randn(N, C_in, *spatial, device=device, dtype=dtype)
            w = torch.randn(C_out, C_in, *([k] * ndim), device=device, dtype=dtype)
            b = torch.randn(C_out, device=device, dtype=dtype)

            t_stock = timed(lambda: conv(x, w, b, stride=1, padding=1), n_iter)
            solver = stock_solver_name(conv, x, w, b)

            def fmt_row(name, ms):
                if ms != ms:  # NaN
                    return f"  {name:<22s}      n/a"
                return (f"  {name:<22s} {ms:8.3f} ms  "
                        f"{flops / (ms * 1e-3) / 1e12:6.1f} TFLOP/s  "
                        f"({flops / (ms * 1e-3) / 1e12 / PEAK[dtype] * 100:3.0f}% of peak)")

            print(f" {str(dtype).replace('torch.', '')}:")
            print(fmt_row("stock", t_stock))
            print(f"    via {solver[:96]}")

            # The hand-written kernels are fp16/fp32 only.
            t_native = float("nan")
            if dtype in (torch.float16, torch.float32):
                try:
                    t_native = timed(lambda: native(x, w, b, ones, ones, ones), n_iter)
                except RuntimeError:
                    pass
            print(fmt_row("native HIP", t_native))

            # CK is fp16/bf16 only (gfx1100 WMMA has no fp32 mode).
            t_nchw = t_cl = float("nan")
            if dtype in (torch.float16, torch.bfloat16):
                if ck_conv(x, w, b, 1, 1, 1) is not None:
                    t_nchw = timed(lambda: ck_conv(x, w, b, 1, 1, 1), n_iter)
                    x_cl = x.contiguous(memory_format=fmt)
                    w_cl = w.contiguous(memory_format=fmt)
                    t_cl = timed(lambda: ck_conv(x_cl, w_cl, b, 1, 1, 1), n_iter)
                    del x_cl, w_cl
            print(fmt_row("CK (NCHW in/out)", t_nchw))
            print(fmt_row("CK (channels-last)", t_cl))

            best = min((v, n) for n, v in
                       (("stock", t_stock), ("native HIP", t_native),
                        ("CK", min(t_nchw, t_cl))) if v == v)
            print(f"  -> fastest: {best[1]} ({best[0]:.3f} ms)")

            del x, w, b
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
