#!/usr/bin/env python
"""VkFFT (pyvkfft/OpenCL) vs rocFFT, on the same GPU.

Reports TWO separate things, because they answer different questions and
conflating them is how this comparison usually goes wrong:

  KERNEL-ONLY -- each library transforming a buffer it already owns, with
    no host transfer on either side. This is the honest "whose FFT is
    faster" number.

  END-TO-END FROM TORCH -- what a caller holding a torch tensor actually
    pays. rocFFT reads the tensor's own device memory; VkFFT cannot (its
    OpenCL backend takes a cl_mem handle, not a device address -- see
    amd_tuned_torch/vkfft_ops.py), so it copies device->host->OpenCL and
    back. That round trip is inherent, and this column exists to size it.

MEASUREMENT. This GPU drops to a 96 MHz memory clock when idle and an FFT
is memory-bound, so short bursts measure the clock ramp rather than the
kernel. Mitigations: a sustained warm-up before timing, interleaved
round-robin across candidates, medians rather than means, and a CONTROL
column that times the reference a second time under a different name. The
control should read 1.00x; however far it strays is this run's noise
floor, and any ratio inside that band means nothing. For tight numbers,
pin the clocks first:  sudo rocm-smi --setperflevel high

ENVIRONMENT. Needs the AMD OpenCL ICD visible:
    export OCL_ICD_VENDORS=$CONDA_PREFIX/etc/OpenCL/vendors
Without it OpenCL sees only the NVIDIA platform on this box and the
comparison silently runs on the wrong card (vkfft_ops._pick_device prefers
an AMD device, and this script refuses to run if it didn't get one).

Usage:  python tools/benchmark_vkfft_vs_rocfft.py [--rounds N] [--json OUT]
"""
import argparse
import json
import statistics
import time

import numpy as np
import torch

from amd_tuned_torch import rocfft_ops, vkfft_ops

# (shape, ndim_transformed) -- a spread of power-of-2, smooth, prime and
# multi-dimensional extents, since VkFFT's and rocFFT's radix coverage is
# exactly where they are expected to differ.
SHAPES = [
    ((8, 4096), 1),
    ((8, 16384), 1),
    ((8, 65536), 1),
    ((8, 15625), 1),      # 5^6 -- smooth but no power of 2
    ((8, 16381), 1),      # prime
    ((8, 20011), 1),      # prime
    ((32, 1024, 1024), 2),
    ((64, 256, 256), 2),
    ((2, 128, 128, 128), 3),
]


def heat(seconds=4.0):
    """Sustained load so DPM ramps the clocks up before anything is timed."""
    a = torch.randn(4096, 4096, device="cuda")
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a = a @ a.T
        a = a / a.norm()
    torch.cuda.synchronize()


def time_torch(fn, it=20):
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    st.record()
    for _ in range(it):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / it


def time_cl(fn, queue, it=20):
    queue.finish()
    t0 = time.perf_counter()
    for _ in range(it):
        fn()
    queue.finish()
    return (time.perf_counter() - t0) / it * 1e3


def run(variants, rounds):
    """Interleaved round-robin; median per variant. `variants` is
    [(name, callable_returning_ms)]."""
    for _, f in variants:      # warm up every candidate (plan/kernel build)
        f()
    acc = {n: [] for n, _ in variants}
    for _ in range(rounds):
        for n, f in variants:
            acc[n].append(f())
    return {n: statistics.median(v) for n, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    if not vkfft_ops.available():
        raise SystemExit(f"vkfft_ops unavailable: {vkfft_ops.load_error()}\n"
                         "Hint: export OCL_ICD_VENDORS=$CONDA_PREFIX/etc/OpenCL/vendors")
    dev = vkfft_ops.device_name()
    if "gfx" not in dev.lower():
        raise SystemExit(f"OpenCL picked '{dev}', not the AMD GPU -- refusing to "
                         "compare rocFFT against VkFFT on different cards. "
                         "Set AMD_TUNED_TORCH_VKFFT_DEVICE or fix OCL_ICD_VENDORS.")

    import pyopencl.array as cla
    queue = vkfft_ops._queue

    print(f"VkFFT: {vkfft_ops.version_string()}  on OpenCL '{dev}'")
    print(f"rocFFT: {rocfft_ops.version_string()}  available={rocfft_ops.available()}")
    print(f"torch {torch.__version__}  on {torch.cuda.get_device_name(0)}")
    print("\nAll times ms/transform, complex64, median of "
          f"{args.rounds} interleaved rounds.")
    print("ctrl = torch.fft timed twice under different names; it should read "
          "1.00x, and\n       its deviation is this run's noise floor.\n")

    hdr = (f"{'shape':>22} {'nd':>2} | {'rocFFT':>8} {'ctrl':>6} | "
           f"{'VkFFT':>8} {'vs roc':>7} | {'VkFFT+copy':>10} {'vs roc':>7}")
    print(hdr)
    print("-" * len(hdr))

    results = []
    heat()
    for shape, nd in SHAPES:
        dim = tuple(range(len(shape) - nd, len(shape)))
        x = torch.randn(*shape, dtype=torch.complex64, device="cuda")
        a = np.ascontiguousarray(x.cpu().numpy())
        # OUT-OF-PLACE on both sides. An in-place VkFFT would re-transform
        # its own output every iteration, so the buffer's magnitude grows by
        # N per pass and overflows to inf within a few rounds -- timing
        # inf/NaN arithmetic against rocFFT's constant, finite input is not a
        # comparison of anything. torch.fft.fftn is out-of-place already.
        ga = cla.to_device(queue, a.copy())
        gb = cla.empty_like(ga)
        app = vkfft_ops._get_app(ga.shape, ga.dtype, nd, r2c=False, inplace=False)

        # VERIFY BEFORE TIMING. A candidate that is fast because it did not
        # do the work is the one failure mode a benchmark must never report
        # as a win, so each VkFFT plan is checked against torch.fft once,
        # for this exact shape, before any of its numbers are believed.
        app.fft(ga, gb)
        queue.finish()
        ref = torch.fft.fftn(x, dim=dim).cpu().numpy()
        err = np.abs(gb.get() - ref).max() / np.abs(ref).max()
        if not err < 1e-4:
            raise SystemExit(f"VkFFT disagrees with torch.fft on {shape} "
                             f"(rel err {err:.2e}) -- refusing to report timings.")

        variants = [
            ("roc",  lambda x=x, dim=dim: time_torch(lambda: torch.fft.fftn(x, dim=dim))),
            ("vk",   lambda ga=ga, gb=gb, app=app: time_cl(lambda: app.fft(ga, gb), queue)),
            ("vkcp", lambda x=x, dim=dim: time_torch(lambda: vkfft_ops.fftn(x, dim=dim))),
            ("ctrl", lambda x=x, dim=dim: time_torch(lambda: torch.fft.fftn(x, dim=dim))),
        ]
        r = run(variants, args.rounds)
        roc = r["roc"]
        print(f"{str(shape):>22} {nd:>2} | {roc:>8.3f} {roc/r['ctrl']:>5.2f}x | "
              f"{r['vk']:>8.3f} {roc/r['vk']:>6.2f}x | "
              f"{r['vkcp']:>10.3f} {roc/r['vkcp']:>6.2f}x")
        results.append({"shape": list(shape), "ndim": nd, "rocfft_ms": roc,
                        "control_ratio": roc / r["ctrl"], "vkfft_ms": r["vk"],
                        "vkfft_speedup": roc / r["vk"],
                        "vkfft_with_copy_ms": r["vkcp"],
                        "vkfft_with_copy_speedup": roc / r["vkcp"]})

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"device": dev, "vkfft": vkfft_ops.version_string(),
                       "results": results}, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
