"""Measures how much of fft_conv's wall-clock time is 'glue' (kron dilation
expand, F.pad x2, dtype casts, imag negate, crop+contiguous, bias add) vs the
'transform core' (rfftn x2, complex_matmul, irfftn), to check whether fusing
the glue into one/two Triton kernels (the flash_mm_kernel.py-style memory
trick) would actually move the needle for amd_tuned_torch's fft_conv.

Reimplements fft_conv's own steps (same code path fftconv_ops.fft_conv
runs) with CUDA-event timing bracketing each phase, for a few shapes
representative of what the module's docstring says this targets: long-
sequence 1D (Hyena-style), plus 2D/3D at sizes from the module's own
measured examples.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"C:\Users\wiffzack\Documents\GIT\amd_tools\source\cmp
_ext_turing")))

import torch
import torch.nn.functional as F
from amd_tuned_torch import fftconv_ops as fc

assert torch.cuda.is_available()
dev = torch.device("cuda")
WARMUP, ITERS = 5, 20


def time_ms(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_tim
ing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


def instrumented_fft_conv(signal, kernel, bias=None, padding=0, padding_mode="cons
tant",
                            stride=1, dilation=1, groups=1):
    """Same steps as fftconv_ops.fft_conv, bucketed into 'glue' vs
    'transform' phase totals (ms) returned alongside the normal output."""
    from fftconv_ops import to_ntuple  # noqa
    n = signal.ndim - 2
    stride_ = fc.to_ntuple(stride, n=n)
    dilation_ = fc.to_ntuple(dilation, n=n)
    padding_ = fc.to_ntuple(padding, n=n)
    out_dtype = signal.dtype

    t = {"glue": 0.0, "transform": 0.0}

    def timed(bucket, fn):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timi
ng=True)
        s.record()
        r = fn()
        e.record()
        torch.cuda.synchronize()
        t[bucket] += s.elapsed_time(e)
        return r

    offset = torch.zeros(1, 1, *dilation_, device=signal.device, dtype=signal.dtyp
e)
    offset[(slice(None), slice(None), *((0,) * n))] = 1.0
    cutoff = tuple(slice(None, -d + 1 if d != 1 else None) for d in dilation_)
    kernel = timed("glue", lambda: torch.kron(kernel, offset)[(slice(None), slice(
None)) + cutoff])

    signal_padding = [r(p) for p in padding_[::-1] for r in (__import__("math").fl
oor, __import__("math").ceil)]
    signal = timed("glue", lambda: F.pad(signal, signal_padding, mode=padding_mode
))

    signal_size = signal.size()
    if signal.size(-1) % 2 != 0:
        signal = timed("glue", lambda: F.pad(signal, [0, 1]))

    kernel_padding = [pad for i in reversed(range(2, signal.ndim))
                       for pad in [0, signal.size(i) - kernel.size(i)]]
    padded_kernel = timed("glue", lambda: F.pad(kernel, kernel_padding))

    signal_f = timed("glue", lambda: signal.float())
    kernel_f = timed("glue", lambda: padded_kernel.float())

    signal_fr = timed("transform", lambda: fc.rfftn(signal_f, dim=tuple(range(2, s
ignal.ndim))))
    kernel_fr = timed("transform", lambda: fc.rfftn(kernel_f, dim=tuple(range(2, s
ignal.ndim))))

    def neg():
        kernel_fr.imag *= -1
        return kernel_fr
    kernel_fr = timed("glue", neg)

    output_fr = timed("transform", lambda: fc.complex_matmul(signal_fr, kernel_fr,
 groups=groups))
    output = timed("transform", lambda: fc.irfftn(output_fr, dim=tuple(range(2, si
gnal.ndim))))

    output = timed("glue", lambda: output.to(out_dtype))

    crop_slices = (slice(None), slice(None)) + tuple(
        slice(0, (signal_size[i] - kernel.size(i) + 1), stride_[i - 2])
        for i in range(2, signal.ndim))
    output = timed("glue", lambda: output[crop_slices].contiguous())

    if bias is not None:
        bias_shape = tuple([1, -1] + (signal.ndim - 2) * [1])
        output = timed("glue", lambda: output + bias.to(out_dtype).view(bias_shape
))

    return output, t


def run_case(name, signal, kernel, **kw):
    total_ms = time_ms(lambda: fc.fft_conv(signal, kernel, **kw))
    _, t = instrumented_fft_conv(signal, kernel, **kw)
    glue, xform = t["glue"], t["transform"]
    frac = glue / (glue + xform) * 100
    print(f"{name:40s} total={total_ms:8.3f}ms  glue={glue:8.3f}ms  transform={xfo
rm:8.3f}ms  "
          f"glue_frac={frac:5.1f}%")


print(f"GPU: {torch.cuda.get_device_name(0)}  dtype=fp32\n")

# 1D, Hyena-style long sequence, moderate channel count.
for L, K, C in [(8192, 64, 8), (65536, 64, 8), (65536, 256, 8), (16384, 4096, 8)]:
    sig = torch.randn(2, C, L, device=dev)
    ker = torch.randn(C, C, K, device=dev)
    run_case(f"1D L={L} K={K} C={C}", sig, ker, padding=K // 2)

# 2D, module docstring's own measured example scale.
sig2 = torch.randn(4, 8, 96, 96, device=dev)
ker2 = torch.randn(8, 8, 15, 15, device=dev)
run_case("2D 96x96 K=15 C=8", sig2, ker2, padding=7)


# 1D, Hyena-style long sequence, moderate channel count.
for L, K, C in [(8192, 64, 8), (65536, 64, 8), (65536, 256, 8), (16384, 4096, 8)]:
    sig = torch.randn(2, C, L, device=dev)
    ker = torch.randn(C, C, K, device=dev)
    run_case(f"1D L={L} K={K} C={C}", sig, ker, padding=K // 2)

# 2D, module docstring's own measured example scale.
sig2 = torch.randn(4, 8, 96, 96, device=dev)
ker2 = torch.randn(8, 8, 15, 15, device=dev)
run_case("2D 96x96 K=15 C=8", sig2, ker2, padding=7)

# 3D, module docstring's own measured example scale.
sig3 = torch.randn(1, 4, 64, 64, 64, device=dev)
ker3 = torch.randn(4, 4, 15, 15, 15, device=dev)
run_case("3D 64^3 K=15 C=4", sig3, ker3, padding=7)