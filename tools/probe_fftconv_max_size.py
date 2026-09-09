"""Finds the largest conv1d signal length amd_tuned_torch.fftconv_ops.fft_conv1d
can actually run on THIS machine before it runs out of memory, by doubling
the signal length until a trial OOMs, then binary-searching the boundary.

WHY THIS EXISTS. fftconv_ops.fft_conv1d's memory footprint is not simply
proportional to the input tensor's own size the way a direct/im2col
conv1d's is: it zero-pads the signal to roughly double length (to avoid
circular-convolution wraparound), keeps BOTH the padded time-domain signal
and its full complex-valued frequency-domain transform (`rfftn`'s output,
`signal_length/2 + 1` complex64 values -- 8 bytes each, vs. the original
real value's 4 (fp32) or 2 (fp16/bf16) bytes) live at once, and does the
same for the kernel padded out to the signal's own length -- exactly the
"kernel is comparably large to the signal" regime this module targets
(global/long convolution, Hyena-style; see fftconv_ops.py's own module
docstring). None of that shows up in a naive "how big is my input tensor"
estimate, so the actual ceiling has to be measured, not guessed -- the same
reasoning tools/benchmark_sparse_conv.py measures flexgemm_ops's crossovers
instead of trusting a hardcoded guess.

This is a CAPABILITY PROBE, not a correctness test -- the answer is
machine-specific (depends on available VRAM/RAM at the moment it runs) and
changes across GPUs/processes, so unlike tests/test_fftconv_ops.py this
lives under tools/ and is run by hand, the same posture
tools/benchmark_sparse_conv.py and tools/autotune_conv.py already have for
"measure this machine, don't assert a fixed number."

METHOD. Build a synthetic (batch, channels, length) signal and a kernel
whose width scales with `length` (see --kernel-mode; default "full" --
kernel_width == length, the worst-case global-convolution memory profile,
so the reported ceiling is a conservative lower bound for narrower-kernel
calls too) at the requested dtype and device, and call fft_conv1d on it.
Success or an out-of-memory error (checked by message, same discipline
amd_tuned_torch.miopen_fallback uses for "is this actually an OOM or some
other RuntimeError" -- an unrecognized RuntimeError is re-raised, not
silently treated as a size limit) decides whether to keep doubling. Once a
trial fails, binary-search between the last success and first failure to
tighten the boundary. Peak memory is reported via
torch.cuda.max_memory_allocated() on a CUDA/ROCm device, or (best-effort,
via psutil if installed) this process's RSS growth on CPU -- CPU has no
analogue of cudaMemGetInfo, so that number is an approximation, not an
exact allocator figure.

Run with:

    python tools/probe_fftconv_max_size.py                       # auto device, fp16, worst-case kernel
    python tools/probe_fftconv_max_size.py --device cpu
    python tools/probe_fftconv_max_size.py --dtype fp32 --batch 2 --channels 64
    python tools/probe_fftconv_max_size.py --kernel-mode fixed --kernel-width 4096
    python tools/probe_fftconv_max_size.py --start 4096 --max-elements 268435456
    python tools/probe_fftconv_max_size.py --no-refine             # doubling phase only, faster/coarser
    python tools/probe_fftconv_max_size.py --max-trial-seconds 120  # allow slower trials before giving up

Stops on whichever comes first: an actual out-of-memory error, or a single
trial taking at least --max-trial-seconds (default 30s) -- CPU FFT cost
grows steeply with size well before memory runs out, so a pure memory-cap
search can take an impractically long time to reach its own ceiling. The
final report says clearly which of the two stopped the search.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch

try:
    from amd_tuned_torch import fftconv_ops
except ImportError:
    # fftconv_ops itself has no compiled-extension dependency at all (see
    # its own module docstring: pure torch.fft/pad/kron) -- if the REST of
    # amd_tuned_torch can't import here (no native extension built for this
    # checkout/platform), load the file directly rather than require a full
    # package build just to probe FFT memory scaling.
    _spec = importlib.util.spec_from_file_location(
        "fftconv_ops",
        Path(__file__).resolve().parent.parent / "amd_tuned_torch" / "fftconv_ops.py",
    )
    fftconv_ops = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(fftconv_ops)

try:
    import psutil
except ImportError:
    psutil = None

_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

# Substrings that mark a RuntimeError as an out-of-memory failure rather
# than a real bug -- CUDA/ROCm ("CUDA out of memory"/"HIP out of memory")
# and CPU (torch's DefaultCPUAllocator raises "... not enough memory: you
# tried to allocate ... bytes.") phrase it differently, so match both
# rather than one hardcoded string. Anything else propagates -- same
# "don't swallow a real bug as if it were a capacity limit" discipline
# amd_tuned_torch.miopen_fallback applies to its own "is this miopenStatus
# or something else" check.
_OOM_MARKERS = ("out of memory", "not enough memory", "cuda error", "hip error")


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc).lower()
    return any(marker in msg for marker in _OOM_MARKERS)


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _default_max_elements() -> int:
    """B*C*L ceiling for the doubling phase's search space, before any
    trial is actually attempted -- a safety backstop so a machine with very
    little free memory doesn't get driven into heavy swapping (which can
    hang far longer than a clean allocator failure) chasing a length the
    doubling phase would never reach anyway. Derived from psutil's
    available-memory reading when installed (this process's own budget,
    not the whole system's), a fixed conservative guess otherwise."""
    if psutil is not None:
        try:
            available = psutil.virtual_memory().available
            # fft_conv keeps several buffers alive at once at fp32 (padded
            # signal/kernel, both their rfftn outputs at ~2x element count
            # each in complex64, complex_matmul's broadcasted intermediate,
            # the irfftn output) -- budgeting 24 bytes/element (6x fp32) is
            # a deliberately loose (safe) multiplier, not a tight estimate;
            # the binary search below finds the real boundary regardless,
            # this only bounds how far the doubling phase is allowed to
            # reach before ever trying an allocation.
            return max(2 ** 16, int(available // 24))
        except Exception:
            pass
    return 2 ** 24  # 16M elements -- conservative when psutil isn't available


def _make_inputs(batch: int, channels: int, groups: int, length: int, kernel_width: int,
                  dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    signal = torch.randn(batch, channels, length, dtype=dtype, device=device)
    weight = torch.randn(channels, channels // groups, kernel_width, dtype=dtype, device=device)
    return signal, weight


def _kernel_width(length: int, mode: str, fixed_width: Optional[int]) -> int:
    if mode == "full":
        return length
    if mode == "half":
        return max(1, length // 2)
    if mode == "fixed":
        if fixed_width is None:
            raise ValueError("--kernel-mode fixed requires --kernel-width")
        return min(fixed_width, length)
    raise ValueError(mode)


def _reset_peak_tracking(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_bytes(device: torch.device, rss_before: Optional[int]) -> Optional[int]:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device)
    if psutil is not None and rss_before is not None:
        return max(0, psutil.Process().memory_info().rss - rss_before)
    return None


def _try_length(length: int, batch: int, channels: int, groups: int, kernel_mode: str,
                 fixed_kernel_width: Optional[int], dtype: torch.dtype, device: torch.device
                 ) -> Tuple[bool, Optional[int], float]:
    """Runs one fft_conv1d trial at `length`. Returns (succeeded, peak_bytes,
    elapsed_seconds) -- peak_bytes is None when it can't be measured (CPU
    without psutil)."""
    kernel_width = _kernel_width(length, kernel_mode, fixed_kernel_width)
    rss_before = psutil.Process().memory_info().rss if (psutil is not None and device.type == "cpu") else None
    _reset_peak_tracking(device)
    signal = weight = out = None
    start_time = time.perf_counter()
    try:
        signal, weight = _make_inputs(batch, channels, groups, length, kernel_width, dtype, device)
        out = fftconv_ops.fft_conv1d(signal, weight, padding=kernel_width // 2, groups=groups)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_time
        peak = _peak_bytes(device, rss_before)
        return True, peak, elapsed
    except (RuntimeError, MemoryError) as exc:
        if not _is_oom(exc):
            raise
        return False, None, time.perf_counter() - start_time
    finally:
        del signal, weight, out
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _format_bytes(n: Optional[int]) -> str:
    if n is None:
        return "unmeasured"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def probe(device: torch.device, dtype: torch.dtype, batch: int, channels: int, groups: int,
          kernel_mode: str, fixed_kernel_width: Optional[int], start: int, max_elements: int,
          refine: bool, max_trial_seconds: float) -> None:
    per_length_elements = batch * channels  # elements per unit of `length`, ignoring kernel
    max_length = max(start, max_elements // max(1, per_length_elements))

    print(f"Probing fftconv_ops.fft_conv1d on device={device}, dtype={dtype}, "
          f"batch={batch}, channels={channels}, groups={groups}, "
          f"kernel_mode={kernel_mode}"
          + (f" (width={fixed_kernel_width})" if kernel_mode == "fixed" else ""), flush=True)
    print(f"Search range: start={start}, cap={max_length} "
          f"(cap derived from {'--max-elements' if max_elements != _default_max_elements() else 'available memory'}), "
          f"per-trial time budget={max_trial_seconds:.0f}s", flush=True)

    length = start
    last_ok: Optional[int] = None
    last_ok_peak: Optional[int] = None
    first_fail: Optional[int] = None
    stopped_on_time = False

    while True:
        ok, peak, elapsed = _try_length(length, batch, channels, groups, kernel_mode,
                                         fixed_kernel_width, dtype, device)
        status = "OK" if ok else "OOM"
        print(f"  L={length:>12,}  {status}  ({elapsed:.2f}s)"
              + (f"  peak={_format_bytes(peak)}" if ok else ""), flush=True)
        if ok:
            last_ok, last_ok_peak = length, peak
            if elapsed >= max_trial_seconds:
                print(f"Trial at L={length:,} took {elapsed:.1f}s, at or above the "
                      f"{max_trial_seconds:.0f}s per-trial budget (--max-trial-seconds) -- "
                      "stopping here because it's TOO SLOW to be practical, not because it "
                      "ran out of memory. This machine may fit larger lengths in memory; "
                      "raise --max-trial-seconds (or accept a slower run) to find out.",
                      flush=True)
                stopped_on_time = True
                break
            if length >= max_length:
                print(f"Reached the search cap ({max_length:,}) without hitting an OOM -- "
                      "this machine comfortably fits every length probed. Re-run with a "
                      "larger --max-elements to push further.", flush=True)
                return
            length *= 2
        else:
            first_fail = length
            break

    if last_ok is None:
        print(f"Even the starting length ({start:,}) OOMs on this machine/config -- "
              "try a smaller --start, --batch, or --channels.", flush=True)
        return

    if refine and not stopped_on_time:
        print(f"Binary-searching between {last_ok:,} (OK) and {first_fail:,} (OOM)...", flush=True)
        lo, hi = last_ok, first_fail
        # Stop once the bracket is tight enough that halving it again
        # wouldn't meaningfully change the answer -- a fixed fraction of
        # the starting length, not an absolute constant, so this scales
        # sensibly whether `start` is 1,024 or 1,048,576.
        tolerance = max(1, start // 16)
        while hi - lo > tolerance:
            mid = lo + (hi - lo) // 2
            ok, peak, elapsed = _try_length(mid, batch, channels, groups, kernel_mode,
                                             fixed_kernel_width, dtype, device)
            status = "OK" if ok else "OOM"
            print(f"  L={mid:>12,}  {status}  ({elapsed:.2f}s)"
                  + (f"  peak={_format_bytes(peak)}" if ok else ""), flush=True)
            if ok:
                lo, last_ok, last_ok_peak = mid, mid, peak
            else:
                hi = mid

    print(flush=True)
    print("=" * 60, flush=True)
    if stopped_on_time:
        print(f"Stopped by the per-trial TIME budget, not memory -- last length actually "
              f"measured OK: {last_ok:,}"
              + (f"  (peak {_format_bytes(last_ok_peak)})" if last_ok_peak is not None else ""),
              flush=True)
        print("This is a lower bound on the memory ceiling, not necessarily the true one -- "
              "raise --max-trial-seconds to keep searching.", flush=True)
    else:
        print(f"Max working conv1d length on this machine: {last_ok:,}"
              + (f"  (peak {_format_bytes(last_ok_peak)})" if last_ok_peak is not None else ""),
              flush=True)
    print(f"Shape at that length: signal=({batch}, {channels}, {last_ok:,}), "
          f"kernel=({channels}, {channels // groups}, "
          f"{_kernel_width(last_ok, kernel_mode, fixed_kernel_width):,}), dtype={dtype}", flush=True)
    print("=" * 60, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=list(_DTYPES), default="fp16")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--kernel-mode", choices=["full", "half", "fixed"], default="full",
                         help="full: kernel width == signal length (worst-case memory, "
                              "Hyena-style global conv -- the default). half: kernel width "
                              "== length/2. fixed: use --kernel-width regardless of length.")
    parser.add_argument("--kernel-width", type=int, default=None,
                         help="Required when --kernel-mode fixed.")
    parser.add_argument("--start", type=int, default=1024,
                         help="Starting signal length for the doubling phase.")
    parser.add_argument("--max-elements", type=int, default=None,
                         help="B*C*L search cap. Defaults to a budget derived from available "
                              "memory (psutil) or a conservative fixed guess otherwise.")
    parser.add_argument("--no-refine", dest="refine", action="store_false",
                         help="Skip the binary-search refinement; report only the last power "
                              "of two that worked (faster, coarser).")
    parser.add_argument("--max-trial-seconds", type=float, default=30.0,
                         help="Stop the doubling phase once a single trial takes at least this "
                              "long, even if it succeeded -- CPU FFT cost grows steeply with "
                              "size, so the memory ceiling alone can be impractically slow to "
                              "reach. Default 30s; raise it (or set to a large number) to keep "
                              "searching purely by memory.")
    args = parser.parse_args()

    device_str = _default_device() if args.device == "auto" else args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("--device cuda requested but no CUDA/ROCm device is available.", file=sys.stderr)
        sys.exit(1)
    device = torch.device(device_str)
    dtype = _DTYPES[args.dtype]
    max_elements = args.max_elements if args.max_elements is not None else _default_max_elements()

    probe(device, dtype, args.batch, args.channels, args.groups,
          args.kernel_mode, args.kernel_width, args.start, max_elements, args.refine,
          args.max_trial_seconds)


if __name__ == "__main__":
    main()
