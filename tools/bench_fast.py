"""Fast stock-PyTorch vs amd_tuned_torch comparison, through the real patch surface.

Complements tools/bench.py rather than replacing it. That one calls the raw
replacement ops directly (amd_tuned_torch.ops.*, .aiter_ops.*, .te_ops.*) at
large shapes to isolate a kernel from dispatch overhead, and takes minutes.
This one flips amd_tuned_torch.enable()/disable() around calls to the SAME
torch entry point (F.linear, F.conv2d, ...) at small shapes, so what it
reports is the end-to-end difference a user actually sees -- kernel choice,
kernel_select contest bookkeeping, eligibility checks and all -- and it
finishes in well under a minute so it can be run after every build.

Because it goes through the patch surface, an op amd_tuned_torch does not
patch on this machine (no aiter/TE installed, a shape no native kernel
covers, ...) shows up as ~1.00x rather than being silently skipped. That is
the point: 1.00x is a real answer about this machine.

Each op is timed in alternating stock/patched rounds and the MINIMUM per
implementation is kept, which is what you want when the noise is one-sided
(clock/thermal drift, a stray kernel from another process). Iteration counts
are chosen per-op from a quick probe to hit a fixed time budget, so a
cheap pointwise op and a GEMM both get a meaningful number of samples.

Correctness is checked alongside speed: a patched op that picked a different
algorithm is expected to differ slightly from stock, so every row also
carries max abs error against the stock output. A large speedup next to a
large error is a bug report, not a win.

Run with:

    python tools/bench_fast.py
    python tools/bench_fast.py --out /tmp/bench_fast.json   # results elsewhere
    python tools/bench_fast.py --budget-ms 50               # quicker/noisier
    python tools/bench_fast.py --only linear,conv2d         # subset
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

# Measure the amd_tuned_torch in THIS checkout, not whichever copy is
# installed in site-packages. Two things conspire against that: running
# `python tools/bench_fast.py` puts tools/ on sys.path rather than the repo
# root, and `import torch` AUTOLOADS amd_tuned_torch (via its torch plugin
# entry point) before any later sys.path edit could matter -- so on a machine
# where an installed copy also exists, the numbers would silently describe
# that copy. Hence the pin here, above `import torch`, and the
# checkout_matches flag in the results file to prove which one ran.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402  (must follow the sys.path pin above)
import torch.nn.functional as F  # noqa: E402

import amd_tuned_torch  # noqa: E402

DEVICE = torch.device("cuda")

# Time budget per (op, implementation, round). Iteration counts are derived
# from this, so it is the one knob that trades total runtime for stability.
DEFAULT_BUDGET_MS = 150.0
ROUNDS = 3
MIN_ITERS, MAX_ITERS = 5, 500


# -----------------------------------------------------------
# Patch-state control
# -----------------------------------------------------------
# amd_tuned_torch enables itself at import, so "stock" here means
# explicitly disabled, not "before we touched anything". The opt-in patches
# (conv1d fallback, rocWMMA flash attention) are toggled per-section by the
# cases that need them -- enable() alone does not install those.

class PatchState:
    """Flips the default patch surface plus whichever opt-in patches a case
    asks for, and always puts the process back the way it was found."""

    def __init__(self) -> None:
        self.was_enabled = amd_tuned_torch.is_enabled()
        self._conv1d = getattr(amd_tuned_torch, "miopen_fallback", None)
        self._was_conv1d = bool(
            self._conv1d is not None and self._conv1d.is_conv1d_fallback_enabled()
        )

    def set(self, patched: bool, extras: Tuple[str, ...] = ()) -> None:
        if patched:
            amd_tuned_torch.enable()
        else:
            amd_tuned_torch.disable()
        if "conv1d" in extras and self._conv1d is not None:
            if patched:
                self._conv1d.enable_conv1d_fallback()
            else:
                self._conv1d.disable_conv1d_fallback()
        if "flash_attn" in extras:
            if patched:
                amd_tuned_torch.enable_flash_attn_rocwmma()
            else:
                amd_tuned_torch.disable_flash_attn_rocwmma()

    def restore(self) -> None:
        if self.was_enabled:
            amd_tuned_torch.enable()
        else:
            amd_tuned_torch.disable()
        if self._conv1d is not None:
            if self._was_conv1d:
                self._conv1d.enable_conv1d_fallback()
            else:
                self._conv1d.disable_conv1d_fallback()


# -----------------------------------------------------------
# Timing
# -----------------------------------------------------------

def _probe_iters(fn: Callable[[], Any], budget_ms: float) -> int:
    """One timed call (after a warmup) decides how many iterations fit the
    budget. Deliberately crude -- it only has to land in the right order of
    magnitude, and being wrong costs a little time, never correctness."""
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    per_call_ms = (time.perf_counter() - t0) * 1e3
    if per_call_ms <= 0:
        return MAX_ITERS
    return max(MIN_ITERS, min(MAX_ITERS, int(budget_ms / per_call_ms)))


def _time_ms(fn: Callable[[], Any], iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _err(patched: Any, stock: Any) -> Tuple[Optional[float], Optional[float]]:
    """Max abs error and that error relative to the stock output's scale.

    The absolute number alone is unreadable across ops: 0.1 is alarming on a
    normalization output and unremarkable on a conv3d that sums 1728 terms.
    The relative one is what tells you whether a row is fp arithmetic noise
    or a wrong kernel.
    """
    if not isinstance(patched, torch.Tensor) or not isinstance(stock, torch.Tensor):
        return None, None
    if patched.shape != stock.shape:
        return float("inf"), float("inf")
    stock_f = stock.float()
    abs_err = (patched.float() - stock_f).abs().max().item()
    scale = stock_f.abs().max().item()
    return abs_err, (abs_err / scale if scale > 0 else None)


def run_case(case: "Case", state: PatchState, budget_ms: float) -> Dict[str, Any]:
    """Build the inputs once, then measure the same call under both states.

    The warmup inside each phase matters more than usual: the first patched
    call for a given shape is where kernel_select runs its measure-and-cache
    contest, and timing that would report the contest, not the winner.
    """
    args = case.make_args()

    def call() -> Any:
        # Resolved from the module on every call, so the timed path is the
        # one a user's code takes -- not a callable captured before the flip.
        return case.invoke(*args)

    results: Dict[str, Any] = {"stock_ms": None, "patched_ms": None}
    outputs: Dict[str, Any] = {}
    per_phase_iters: Dict[str, int] = {}

    for phase, patched in (("stock", False), ("patched", True)):
        state.set(patched, case.extras)
        per_phase_iters[phase] = _probe_iters(call, budget_ms)
        outputs[phase] = call()

    best: Dict[str, float] = {}
    for _ in range(ROUNDS):
        for phase, patched in (("stock", False), ("patched", True)):
            state.set(patched, case.extras)
            ms = _time_ms(call, per_phase_iters[phase])
            best[phase] = min(best.get(phase, float("inf")), ms)

    results["stock_ms"] = best["stock"]
    results["patched_ms"] = best["patched"]
    results["speedup"] = best["stock"] / best["patched"] if best["patched"] else None
    results["max_abs_err"], results["max_rel_err"] = _err(
        outputs["patched"], outputs["stock"])
    results["iters"] = per_phase_iters
    results["shape"] = case.shape_desc
    results["dtype"] = case.dtype_name
    del args, outputs
    torch.cuda.empty_cache()
    return results


# -----------------------------------------------------------
# Cases
# -----------------------------------------------------------

class Case:
    def __init__(self, name: str, dtype: torch.dtype, shape_desc: str,
                 make_args: Callable[[], Tuple[Any, ...]],
                 invoke: Callable[..., Any],
                 extras: Tuple[str, ...] = ()) -> None:
        self.name = name
        self.dtype = dtype
        self.dtype_name = str(dtype).rsplit(".", 1)[-1]
        self.shape_desc = shape_desc
        self.make_args = make_args
        self.invoke = invoke
        self.extras = extras

    @property
    def key(self) -> str:
        return f"{self.name}_{self.dtype_name}"


def _rand(*shape: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(*shape, device=DEVICE, dtype=dtype)


def build_cases() -> List[Case]:
    """Small shapes on purpose -- big enough that the kernel dominates launch
    overhead, small enough that the whole file runs in seconds. Every shape
    below is one amd_tuned_torch has a candidate for on gfx1100, so a 1.00x
    row means the contest picked stock, not that the shape was out of scope."""
    cases: List[Case] = []

    # --- GEMM family: F.linear / torch.matmul / torch.bmm (hipBLASLt and CK
    # WMMA tiers, contested against stock through kernel_select) ---
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        M = K = N = 2048
        cases.append(Case(
            "linear", dtype, f"{M}x{K} @ {N}x{K}.T",
            lambda dtype=dtype: (_rand(M, K, dtype=dtype), _rand(N, K, dtype=dtype),
                                 _rand(N, dtype=dtype)),
            lambda x, w, b: F.linear(x, w, b),
        ))
    for dtype in (torch.float16, torch.float32):
        M = K = N = 2048
        cases.append(Case(
            "matmul", dtype, f"{M}x{K} @ {K}x{N}",
            lambda dtype=dtype: (_rand(M, K, dtype=dtype), _rand(K, N, dtype=dtype)),
            lambda a, b: torch.matmul(a, b),
        ))
        B, Mb, Kb, Nb = 8, 512, 512, 512
        cases.append(Case(
            "bmm", dtype, f"batch={B}, {Mb}x{Kb} @ {Kb}x{Nb}",
            lambda dtype=dtype: (_rand(B, Mb, Kb, dtype=dtype), _rand(B, Kb, Nb, dtype=dtype)),
            lambda a, b: torch.bmm(a, b),
        ))

    # --- Convolutions: native HIP kernels, fp16/fp32 only (src/cuda/conv*.cu) ---
    for dtype in (torch.float16, torch.float32):
        Nb, Cin, HW, Cout, k = 16, 64, 64, 64, 3
        cases.append(Case(
            "conv2d", dtype, f"N={Nb}, C={Cin}->{Cout}, {HW}x{HW}, k={k}",
            lambda dtype=dtype: (_rand(Nb, Cin, HW, HW, dtype=dtype),
                                 _rand(Cout, Cin, k, k, dtype=dtype),
                                 _rand(Cout, dtype=dtype)),
            lambda x, w, b: F.conv2d(x, w, b, stride=1, padding=1),
        ))
        Nb3, Cin3, D, HW3, Cout3 = 1, 64, 16, 32, 64
        cases.append(Case(
            "conv3d", dtype, f"N={Nb3}, C={Cin3}->{Cout3}, {D}x{HW3}x{HW3}, k={k}",
            lambda dtype=dtype: (_rand(Nb3, Cin3, D, HW3, HW3, dtype=dtype),
                                 _rand(Cout3, Cin3, k, k, k, dtype=dtype),
                                 _rand(Cout3, dtype=dtype)),
            lambda x, w, b: F.conv3d(x, w, b, stride=1, padding=1),
        ))
        # Depthwise conv1d -- the vendored FlashFFTConv kernel and the fftconv
        # candidate, both reached only via the opt-in conv1d fallback patch.
        Nb1, C1, L1, k1 = 8, 256, 2048, 3
        cases.append(Case(
            "conv1d_depthwise", dtype, f"N={Nb1}, C={C1}, L={L1}, k={k1}, groups={C1}",
            lambda dtype=dtype: (_rand(Nb1, C1, L1, dtype=dtype),
                                 _rand(C1, 1, k1, dtype=dtype),
                                 _rand(C1, dtype=dtype)),
            lambda x, w, b: F.conv1d(x, w, b, stride=1, padding=k1 // 2, groups=C1),
            extras=("conv1d",),
        ))

    # --- GroupNorm: native HIP kernel ---
    for dtype in (torch.float16, torch.float32):
        Nb, C, HW, groups = 16, 128, 64, 32
        cases.append(Case(
            "group_norm", dtype, f"N={Nb}, C={C}, {HW}x{HW}, groups={groups}",
            lambda dtype=dtype: (_rand(Nb, C, HW, HW, dtype=dtype),
                                 _rand(C, dtype=dtype), _rand(C, dtype=dtype)),
            lambda x, w, b: F.group_norm(x, groups, w, b, eps=1e-5),
        ))

    # --- Attention: rocWMMA flash-attention kernel, opt-in patch ---
    for dtype in (torch.float16,):
        B, H, S, Dh = 2, 16, 1024, 64
        cases.append(Case(
            "sdpa_causal", dtype, f"B={B}, H={H}, S={S}, D={Dh}",
            lambda dtype=dtype: (_rand(B, H, S, Dh, dtype=dtype),
                                 _rand(B, H, S, Dh, dtype=dtype),
                                 _rand(B, H, S, Dh, dtype=dtype)),
            lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True),
            extras=("flash_attn",),
        ))

    return cases


# -----------------------------------------------------------
# Reporting
# -----------------------------------------------------------

def _backends() -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    for mod_name in ("aiter_ops", "hipblaslt_ops", "ck_gemm_ops", "te_ops",
                     "flash_attn_rocwmma_ops", "depthwise_conv1d_ops", "fftconv_ops"):
        mod = getattr(amd_tuned_torch, mod_name, None)
        available = getattr(mod, "available", None) if mod is not None else None
        out[mod_name] = bool(available()) if callable(available) else False
    return out


def _system() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        from amd_tuned_torch import benchmark_report
        info.update(benchmark_report.system_metadata())
        info["device_key"] = benchmark_report.device_key()
    except Exception:
        # Metadata is nice-to-have; a missing helper must not lose results.
        info["device_name"] = torch.cuda.get_device_name(0)
        info["torch_version"] = torch.__version__
    pkg_dir = os.path.dirname(os.path.abspath(amd_tuned_torch.__file__))
    info["amd_tuned_torch"] = pkg_dir
    info["checkout"] = REPO_ROOT
    info["checkout_matches"] = (pkg_dir == os.path.join(REPO_ROOT, "amd_tuned_torch"))
    return info


def print_table(rows: Dict[str, Dict[str, Any]]) -> None:
    head = (f"{'op':<26}{'dtype':<10}{'stock ms':>10}{'patched ms':>12}"
            f"{'speedup':>10}{'max abs err':>13}{'rel err':>11}")
    print(head)
    print("-" * len(head))
    for key, r in rows.items():
        abs_err, rel_err = r.get("max_abs_err"), r.get("max_rel_err")
        abs_s = "n/a" if abs_err is None else f"{abs_err:.3g}"
        rel_s = "n/a" if rel_err is None else f"{rel_err:.2e}"
        print(f"{key.rsplit('_', 1)[0]:<26}{r['dtype']:<10}"
              f"{r['stock_ms']:>10.3f}{r['patched_ms']:>12.3f}"
              f"{r['speedup']:>9.2f}x{abs_s:>13}{rel_s:>11}")
    print("-" * len(head))
    wins = [k for k, r in rows.items() if r["speedup"] and r["speedup"] > 1.05]
    losses = [k for k, r in rows.items() if r["speedup"] and r["speedup"] < 0.95]
    print(f"faster than stock: {len(wins)}/{len(rows)}   slower: {len(losses)}/{len(rows)}"
          f"   within noise: {len(rows) - len(wins) - len(losses)}/{len(rows)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None,
                        help="Where to write the results file "
                             "(default: bench_fast.json beside setup.py).")
    parser.add_argument("--budget-ms", type=float, default=DEFAULT_BUDGET_MS,
                        help=f"Time budget per op/impl/round (default {DEFAULT_BUDGET_MS:g}).")
    parser.add_argument("--only", default=None,
                        help="Comma-separated op names to run (default: all).")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No GPU visible to torch -- nothing to compare.", file=sys.stderr)
        return 1

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench_fast.json")
    wanted = {n.strip() for n in args.only.split(",")} if args.only else None

    cases = [c for c in build_cases() if wanted is None or c.name in wanted]
    if not cases:
        print(f"No cases matched --only {args.only!r}", file=sys.stderr)
        return 1

    state = PatchState()
    rows: Dict[str, Dict[str, Any]] = {}
    t_start = time.perf_counter()
    try:
        for case in cases:
            try:
                rows[case.key] = run_case(case, state, args.budget_ms)
            except Exception as exc:  # one unsupported shape must not end the run
                rows[case.key] = {"error": f"{type(exc).__name__}: {exc}",
                                  "shape": case.shape_desc, "dtype": case.dtype_name}
                print(f"[skip] {case.key}: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        state.restore()
    elapsed = time.perf_counter() - t_start

    ok = {k: r for k, r in rows.items() if "error" not in r}
    print_table(ok)
    system = _system()
    if not system.get("checkout_matches", False):
        print(f"WARNING: measured amd_tuned_torch at {system['amd_tuned_torch']}, "
              f"not the checkout at {REPO_ROOT} -- results describe the installed "
              f"copy.", file=sys.stderr)
    print(f"total wall time: {elapsed:.1f}s")

    report = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "harness": "tools/bench_fast.py",
        "method": ("same torch entry point called with amd_tuned_torch.disable() "
                   "vs .enable(); min of alternating rounds; CUDA-event timed"),
        "rounds": ROUNDS,
        "budget_ms": args.budget_ms,
        "wall_time_s": round(elapsed, 3),
        "system": system,
        "backends_available": _backends(),
        "ops": rows,
    }
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
