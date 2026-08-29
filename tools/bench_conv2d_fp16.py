"""Mini benchmark for conv2d fp16 on gfx1100 that answers one question:
why is this project's conv2d slower than stock, and is that fixable?

It runs the same shape through every kernel available here and, crucially,
turns on MIOpen's own logging so stock's internal solver search is visible.
That search is the answer: MIOpen evaluates several algorithm classes for
the shape and reports its own timing for each, which lets this project's
kernel be compared against the solver *in its own algorithm class* rather
than only against the winner. Those are very different comparisons, and
only the first one says anything about implementation quality.

The debug flags (MIOPEN_ENABLE_LOGGING, MIOPEN_ENABLE_LOGGING_CMD,
MIOPEN_LOG_LEVEL) are set below BEFORE torch is imported, because MIOpen
reads them when it initialises. Two traps worth knowing if you run this by
hand:

  1. The monkeypatch must be OFF while profiling stock. With it on,
     F.conv2d never reaches MIOpen and the log stays empty -- which looks
     exactly like "the logging flags don't work on this build".
  2. MIOpen logs from C++ to fd 2, so Python-level stderr redirection does
     not capture it. This uses an fd-level dup2.

Run with:

    python tools/bench_conv2d_fp16.py            # default bench shape
    python tools/bench_conv2d_fp16.py 8 1024 64 64 512 3
"""
import os

# Must precede `import torch` -- see the module docstring.
os.environ.setdefault("MIOPEN_ENABLE_LOGGING", "1")
os.environ.setdefault("MIOPEN_ENABLE_LOGGING_CMD", "1")
os.environ.setdefault("MIOPEN_LOG_LEVEL", "6")
# The contest would pick a winner and hide the per-kernel numbers this
# script exists to show.
os.environ.setdefault("AMD_TUNED_TORCH_CONV_MEASURE", "0")

import re  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import amd_tuned_torch  # noqa: E402
from amd_tuned_torch import ck_ops  # noqa: E402

from pathlib import Path  # noqa: E402

# Every artifact this repo's investigations produce lands under analyse/ --
# see analyse/README.md for the index.
ANALYSE = Path(__file__).resolve().parent.parent / "analyse"

# RX 7900 XTX fp16 WMMA peak. Note a Winograd solver can legitimately
# exceed 100% of this: it computes fewer multiplies than the direct
# algorithm this FLOP count assumes.
PEAK_TFLOPS = 122.9


class CaptureFd2:
    """Captures fd 2, which is where MIOpen's C++ logging goes."""

    def __enter__(self):
        self._tmp = tempfile.TemporaryFile(mode="w+b")
        self._saved = os.dup(2)
        sys.stderr.flush()
        os.dup2(self._tmp.fileno(), 2)
        return self

    def __exit__(self, *exc):
        sys.stderr.flush()
        os.dup2(self._saved, 2)
        os.close(self._saved)
        self._tmp.seek(0)
        self.text = self._tmp.read().decode("utf-8", "replace")
        self._tmp.close()
        return False


def timed(fn, n=30, warmup=10):
    for _ in range(warmup):
        if fn() is None:
            return float("nan")
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(n):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n


def main():
    a = sys.argv[1:]
    N, C, H, W, K, k = ([int(v) for v in a] if len(a) == 6 else [64, 128, 128, 128, 64, 3])
    pad = k // 2
    dev = torch.device("cuda")
    torch.manual_seed(0)

    x = torch.randn(N, C, H, W, device=dev, dtype=torch.float16)
    w = torch.randn(K, C, k, k, device=dev, dtype=torch.float16)
    b = torch.randn(K, device=dev, dtype=torch.float16)

    H_out = H + 2 * pad - k + 1
    W_out = W + 2 * pad - k + 1
    flops = 2 * N * C * K * H_out * W_out * k * k

    print(f"conv2d fp16  N={N} C_in={C} {H}x{W} -> C_out={K}, k={k}, stride=1, pad={pad}")
    print(f"{flops / 1e9:.1f} GFLOP by the direct algorithm\n")

    amd_tuned_torch.disable()  # trap 1 in the docstring

    # --- run everything with fd 2 captured -----------------------------
    # The capture has to span the whole run, not just the first call:
    # MIOpen logs on every convolution, so the timed iterations below would
    # otherwise dump tens of thousands of lines to the terminal. Results
    # are collected here and printed afterwards.
    rows = []
    with CaptureFd2() as cap:
        F.conv2d(x, w, b, stride=1, padding=pad)
        torch.cuda.synchronize()
        rows.append(("stock (MIOpen)",
                     timed(lambda: F.conv2d(x, w, b, stride=1, padding=pad))))
        rows.append(("this project's HIP kernel",
                     timed(lambda: amd_tuned_torch.ops.conv2d(
                         x, w, b, [1, 1], [pad, pad], [1, 1]))))
        if ck_ops.available():
            rows.append(("CK WMMA (NCHW in/out)",
                         timed(lambda: ck_ops.conv2d(x, w, b, 1, pad, 1))))
            x_cl = x.contiguous(memory_format=torch.channels_last)
            w_cl = w.contiguous(memory_format=torch.channels_last)
            rows.append(("CK WMMA (channels-last)",
                         timed(lambda: ck_ops.conv2d(x_cl, w_cl, b, 1, pad, 1))))
    log = cap.text

    log_path = ANALYSE / "logs" / f"miopen_conv2d_fp16_{N}x{C}x{H}x{W}_k{K}x{k}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(log)

    cmd = re.search(r"\[LogCmdConvolution\]\s*(.+)", log)
    if cmd:
        print("MIOpen sees this as:\n  " + cmd.group(1).strip() + "\n")

    # "Selected: <solver>: <kernel>: <ms>, workspace_sz = <bytes>"
    sel = re.findall(
        r"\[EvaluateInvokers\] Selected: ([^:]+): ([^:]*): ([0-9.]+), workspace_sz = (\d+)", log)
    if sel:
        print("MIOpen evaluated these algorithm classes (its own timings):")
        print(f"  {'solver':<28s} {'ms':>9s} {'workspace':>11s}  kernel")
        for solver, kernel, ms, ws in sorted(sel, key=lambda r: float(r[2])):
            ws_mb = int(ws) / 1e6
            print(f"  {solver.strip():<28s} {float(ms):9.3f} {ws_mb:9.1f} MB  {kernel.strip()[:60]}")
        print()
    else:
        print("(no solver evaluation in the log -- MIOpen likely served this "
              "shape from its tuning DB; delete ~/.config/miopen to force a search)\n")

    print(f"Full MIOpen log ({len(log.splitlines())} lines) -> {log_path}\n")
    print("Measured here:")
    print(f"  {'kernel':<28s} {'ms':>9s} {'TFLOP/s':>9s} {'%peak':>7s}")
    for name, ms in rows:
        if ms != ms:
            print(f"  {name:<28s} {'n/a':>9s}")
            continue
        tf = flops / (ms * 1e-3) / 1e12
        print(f"  {name:<28s} {ms:9.3f} {tf:9.1f} {tf / PEAK_TFLOPS * 100:6.0f}%")

    # --- the actual diagnosis -----------------------------------------
    gemm = next((float(m) for s, _, m, _ in sel if "Gemm" in s), None)
    wino = next((float(m) for s, _, m, _ in sel if "Wino" in s), None)
    ours = rows[1][1]
    print()
    if gemm and wino and ours == ours:
        print("Diagnosis (the gap to stock splits into two independent parts):")
        print()
        print(f"  1. ALGORITHM. Stock's winner is Winograd ({wino:.2f} ms), which computes")
        print(f"     fewer multiplies than the direct algorithm. The best GEMM-class")
        print(f"     solver -- MIOpen's own -- manages {gemm:.2f} ms, i.e. Winograd is")
        print(f"     {gemm / wino:.1f}x ahead of the entire GEMM approach on this shape. Our")
        print(f"     kernel is GEMM-class, so no amount of tuning reaches stock; that")
        print(f"     would require implementing Winograd.")
        print()
        if ours <= gemm:
            print(f"  2. IMPLEMENTATION. None: ours is {ours:.2f} ms against MIOpen's own")
            print(f"     GEMM-class {gemm:.2f} ms, so our kernel is already at or ahead of")
            print(f"     stock's implementation of the same algorithm. The whole gap is (1).")
        else:
            print(f"  2. IMPLEMENTATION. Real here: ours is {ours:.2f} ms against MIOpen's")
            print(f"     own GEMM-class {gemm:.2f} ms -- {ours / gemm:.1f}x slower at the SAME")
            print(f"     algorithm, so {ours - gemm:.2f} ms of the gap is ours to close and is")
            print(f"     not explained by Winograd. This part is worth optimising; the")
            print(f"     rest is not.")
        ck = next((ms for n, ms in rows if "channels-last" in n), None)
        if ck and ck == ck:
            print()
            print(f"  For reference the CK tier, also GEMM-class, does {ck:.2f} ms here --")
            print(f"  {'ahead of' if ck < gemm else 'behind'} MIOpen's own GEMM solver, which bounds what")
            print(f"  GEMM-class tuning can buy on this shape.")


if __name__ == "__main__":
    main()
