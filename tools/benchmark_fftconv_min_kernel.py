"""Finds the kernel-width crossover for amd_tuned_torch's FFT-conv2d/conv3d
tiers and saves it as this GPU's calibration (amd_tuned_torch/
fftconv_calibration.py), replacing two hardcoded guesses --
_dispatch.py's _FFTCONV_CONV2D_MIN_KERNEL (32) and
_FFTCONV_CONV3D_MIN_KERNEL (7).

WHY THIS EXISTS. `_fftconv_conv_candidate` skips building/timing the FFT
candidate at all when `min(weight.shape[2:])` is below those thresholds --
a cheap pre-filter so a U-Net's 3x3 convs never pay for a contest they
cannot win. The min_positions gate next to them was measured
(tools/benchmark_fftconv3d_min_positions.py); these two never were.
fftconv_calibration's schema has accepted a `min_kernel` field since it was
written, with no script producing one. This is that script.

WHAT "CROSSOVER" MEANS HERE. The smallest kernel width at which FFT-conv
beats stock F.convNd AND keeps beating it at every larger width measured.
A single win with a loss above it is noise, not a trend -- the same
discipline benchmark_fftconv3d_min_positions.py applies to its size sweep.
AGGREGATION ACROSS SHAPES: THE MINIMUM, NOT THE MAXIMUM. The crossover
moves with input size, so a sweep over several shapes yields several
crossovers and only one number can be written. The two errors are not
symmetric. This threshold only decides whether to CONTEST; kernel_select
then measures FFT against stock/CK/native for that exact shape and rejects
it if it loses. So setting it too LOW costs one wasted contest per distinct
shape, paid once and cached. Setting it too HIGH means the FFT candidate is
never even built, so the loss is silent and repeats on every call forever.
The cheap error is the low one, which makes the right value the SMALLEST
crossover observed -- the width below which no measured shape won, i.e. the
point below which the pre-filter can be confident. (An earlier version of
this script wrote the largest instead, which contradicted this same
paragraph.)

STOCK IS CAPTURED AFTER AN EXPLICIT disable(). Capturing it "before
importing amd_tuned_torch" is NOT sufficient and was the first version of
this script's bug: a .pth in this environment auto-imports and auto-enables
the package at interpreter startup, so `F.conv2d` is already
`_patched_conv2d` before the first line of any script runs. Timing the FFT
tier against a reference that may itself dispatch to the FFT tier reports a
meaningless ~1.00x -- which is exactly what the broken run showed at every
width above the threshold. `amd_tuned_torch.disable()` restores the real
`F.conv2d`, and the assertion below refuses to run if it didn't.

MEASUREMENT. This GPU drops to a 96 MHz memory clock when idle and these
transforms are memory-bound, so short bursts measure the clock ramp, not
the kernel. Mitigations: a sustained warm-up, interleaved round-robin,
medians, and a CONTROL column (stock timed twice under different names)
that should read 1.00x -- its deviation is the run's noise floor, and any
ratio inside that band is not evidence. For tight numbers, first run:
    sudo rocm-smi --setperflevel high

Run with:
    python tools/benchmark_fftconv_min_kernel.py            # sweep + save
    python tools/benchmark_fftconv_min_kernel.py --dry-run  # sweep only
"""
import torch.nn.functional as _F

import amd_tuned_torch as _att

# Uninstall this package's conv patches BEFORE capturing the reference --
# see the docstring's "STOCK IS CAPTURED AFTER AN EXPLICIT disable()".
_att.disable()
_STOCK_CONV2D = _F.conv2d
_STOCK_CONV3D = _F.conv3d
for _name, _fn in (("conv2d", _STOCK_CONV2D), ("conv3d", _STOCK_CONV3D)):
    if _fn.__name__.startswith("_patched"):
        raise SystemExit(
            f"F.{_name} is still {_fn.__name__} after amd_tuned_torch.disable() -- "
            "the reference would be this package's own FFT tier, making every "
            "ratio below meaningless. Refusing to calibrate.")

import argparse  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

from amd_tuned_torch import fftconv_calibration, fftconv_ops  # noqa: E402

# (batch, channels, spatial...) -- a few shapes rather than one, because the
# crossover moves with input size and a single shape would hide that.
_CONV2D_SHAPES = [(8, 32, 128, 128), (4, 64, 256, 256), (16, 16, 64, 64)]
_CONV3D_SHAPES = [(2, 16, 32, 32, 32), (1, 32, 48, 48, 48)]

_CONV2D_WIDTHS = [3, 5, 7, 9, 11, 13, 15, 17, 21, 25, 31, 33, 41, 49, 63]
_CONV3D_WIDTHS = [3, 5, 7, 9, 11, 13, 15, 17, 21]

_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def heat(seconds=4.0):
    """Sustained load so DPM ramps clocks before anything is timed."""
    a = torch.randn(4096, 4096, device="cuda")
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a = a @ a.T
        a = a / a.norm()
    torch.cuda.synchronize()


def time_ms(fn, it=10):
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    st.record()
    for _ in range(it):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / it


def ab(variants, rounds=5):
    """Interleaved round-robin, median per variant."""
    for _, f in variants:
        try:
            f()
        except RuntimeError:
            return None
    acc = {n: [] for n, _ in variants}
    for _ in range(rounds):
        for n, f in variants:
            acc[n].append(time_ms(f))
    return {n: statistics.median(v) for n, v in acc.items()}


def sweep(ndim, shape, widths, dtype, rounds):
    """Per-width (stock_ms, fft_ms, speedup, control) for one input shape.
    Returns a dict width -> row, skipping widths that don't fit the input."""
    dev = "cuda"
    stock = _STOCK_CONV2D if ndim == 2 else _STOCK_CONV3D
    fft = fftconv_ops.fft_conv2d if ndim == 2 else fftconv_ops.fft_conv3d
    cin = shape[1]
    x = torch.randn(*shape, device=dev, dtype=dtype)
    rows = {}
    for k in widths:
        if any(k > s for s in shape[2:]):
            continue
        w = torch.randn(cin, cin, *([k] * ndim), device=dev, dtype=dtype)
        # Correctness gate: a fast wrong answer must never set a threshold.
        try:
            a = fft(x, w)
            b = stock(x, w)
        except RuntimeError as exc:           # OOM at a wide kernel
            rows[k] = None
            continue
        rel = (a - b).abs().max().item() / max(b.abs().max().item(), 1e-30)
        if not rel < 2e-2:
            raise SystemExit(f"FFT-conv{ndim}d disagrees with stock at k={k} "
                             f"(rel {rel:.2e}) -- refusing to calibrate.")
        r = ab([("stock", lambda x=x, w=w: stock(x, w)),
                ("fft", lambda x=x, w=w: fft(x, w)),
                ("ctrl", lambda x=x, w=w: stock(x, w))], rounds)
        if r is None:
            rows[k] = None
            continue
        rows[k] = (r["stock"], r["fft"], r["stock"] / r["fft"],
                   r["stock"] / r["ctrl"])
    return rows


def crossover(rows, margin=1.15):
    """Smallest width where FFT wins by at least `margin` AND does not lose
    at any larger width.

    The margin is the point. A bare `speedup > 1.0` test reads a 1.05x
    result as a crossover even when the control column for that same row is
    1.03x off -- i.e. the "win" is smaller than the measurement's own error,
    which on this GPU (idle clocks, memory-bound transforms) is routinely
    5-10%. Requiring the crossover row to clear the noise floor, while
    letting the rows above it merely not regress (>= 1.0), keeps a genuine
    trend while rejecting a single lucky sample."""
    widths = [k for k in sorted(rows) if rows[k] is not None]
    for i, k in enumerate(widths):
        if rows[k][2] >= margin and all(rows[j][2] >= 1.0 for j in widths[i:]):
            return k
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dtype", choices=list(_DTYPES), default="fp32")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--margin", type=float, default=1.15,
                    help="speedup a width must clear to count as the crossover, "
                         "so a win inside the noise floor is not mistaken for one")
    ap.add_argument("--dry-run", action="store_true",
                    help="sweep and print, write no calibration")
    ap.add_argument("--ndim", type=int, choices=[2, 3], nargs="+", default=[2, 3])
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU")
    dtype = _DTYPES[args.dtype]
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | {args.dtype}")
    print("ctrl should read 1.00x; its deviation is the noise floor.\n")

    heat()
    measurements = {}
    for ndim in args.ndim:
        shapes = _CONV2D_SHAPES if ndim == 2 else _CONV3D_SHAPES
        widths = _CONV2D_WIDTHS if ndim == 2 else _CONV3D_WIDTHS
        per_shape = {}
        for shape in shapes:
            rows = sweep(ndim, shape, widths, dtype, args.rounds)
            x = crossover(rows, args.margin)
            per_shape[shape] = x
            print(f"conv{ndim}d  input {shape}")
            print(f"{'k':>5} {'stock ms':>9} {'fft ms':>9} {'fft vs stock':>13} {'ctrl':>6}")
            for k in sorted(rows):
                if rows[k] is None:
                    print(f"{k:>5} {'--':>9} {'--':>9} {'(failed/OOM)':>13}")
                    continue
                s, f, sp, c = rows[k]
                mark = "  <- crossover" if k == x else ""
                print(f"{k:>5} {s:>9.3f} {f:>9.3f} {sp:>12.2f}x {c:>5.2f}x{mark}")
            print(f"  crossover: {x if x is not None else 'none (FFT never wins)'}\n")

        found = [v for v in per_shape.values() if v is not None]
        if found:
            # The SMALLEST crossover seen -- see the docstring's AGGREGATION
            # section for why under-gating is the cheap error here.
            measurements[f"conv{ndim}d"] = {"min_kernel": min(found)}
            print(f"conv{ndim}d min_kernel -> {min(found)} "
                  f"(min over per-shape crossovers "
                  f"{[v for v in per_shape.values()]})\n")
        else:
            print(f"conv{ndim}d: FFT never won at any measured width; "
                  f"leaving the existing threshold alone.\n")

    if args.dry_run:
        print("--dry-run: nothing written")
        return 0
    if not measurements:
        print("nothing to write")
        return 0
    fftconv_calibration.save(measurements)
    print(f"saved to {fftconv_calibration.calibration_path()}")
    print("Takes effect on the next `import amd_tuned_torch`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
