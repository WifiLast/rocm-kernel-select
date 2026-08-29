# analyse/ — measurements behind the conv kernel decisions

Every log, harness and result referenced by comments in `src/` and by the
README's performance claims. Nothing here is built or imported by the
package; it exists so the numbers can be re-derived rather than trusted.

Reproduce the headline result with:

    python tools/bench_conv2d_fp16.py

---

## The finding: why this project's conv2d fp16 is slower than stock

**Two independent causes, and their weights change with the shape.** Run the
tool on your own shape rather than assuming either one dominates.

Turning on MIOpen's own logging (`MIOPEN_ENABLE_LOGGING=1`,
`MIOPEN_ENABLE_LOGGING_CMD=1`, `MIOPEN_LOG_LEVEL=6`) makes stock's internal
solver search visible. For `N=64 C_in=128 128x128 -> C_out=64, k=3`, MIOpen
evaluates three algorithm classes and reports its **own** timings
(`logs/miopen_conv2d_fp16_*.log`, search for `EvaluateInvokers`):

| MIOpen solver | Algorithm class | MIOpen's own timing | Workspace |
|---|---|---|---|
| `ConvWinoFuryRxS<2-3>` | **Winograd** (hand-written gfx11 assembly) | **1.03 ms** | 2 KB |
| `GemmFwdRest` | **im2col + GEMM** | **7.06 ms** | 37.7 MB |
| `ConvDirectNaiveConvFwd` | naive direct | 519.51 ms | 0 |

Our kernel is implicit-GEMM/im2col class, so the honest comparison is
against MIOpen's solver *of the same class*, not against the winner:

| shape | stock winner (Winograd) | MIOpen's own GEMM | ours | our gap that is ALGORITHM | that is IMPLEMENTATION |
|---|---|---|---|---|---|
| `64x128x128x128 -> 64, k3` | 1.03 ms | 7.16 ms | 6.53 ms | all of it | **none** — we beat MIOpen's GEMM |
| `8x1024x64x64 -> 512, k3` | 2.10 ms | 4.79 ms | 9.36 ms | 2.3x | **2.0x — 4.6 ms is ours to close** |

On the first shape our kernel is already ahead of stock's own GEMM
implementation, so the entire gap is algorithmic and tuning cannot help. On
the channel-heavy shape it is 2x *behind* MIOpen's GEMM solver, so roughly
half that gap is a real implementation deficit worth attacking. An earlier
version of this note claimed the gap was never about implementation
quality; that generalised from a single shape and was wrong.

The algorithmic half is not closable by tuning: Winograd computes fewer
multiplies than direct convolution — which is why
stock reads as ~100 TFLOP/s against a 122.9 TFLOP/s peak computed with
direct-algorithm FLOPs, and why other shapes read *above* 100% of "peak".
No GEMM-class kernel can close that by being better written. Matching stock
means implementing Winograd; beating *our* kernel only needs better
GEMM-class tuning, which is what the CK tier does (2.41 ms channels-last).

Practical consequence, already implemented in `amd_tuned_torch/kernel_select.py`:
stock is a candidate in the per-shape contest rather than a fallback. It
wins fp16/fp32 conv2d, and loses bf16 by 3x because MIOpen has **no bf16
Winograd solver on gfx11** at all.

## Where our conv2d fp16 kernel spends its time

`ablation/` — a standalone harness (no torch) that compiles the generated
kernel with parts switched out. `bash analyse/ablation/run2.sh`, results in
`ablation/results.txt`:

| Configuration | Time | Implies |
|---|---|---|
| baseline | 6.55 ms | |
| no global input read (address math kept) | 2.72 ms | **im2col gather = 3.84 ms (59%)** |
| no `LOAD_A` body at all | 2.69 ms | address math ≈ 0 (the carry rewrite worked) |
| no `mma_sync` | 5.19 ms | WMMA = 1.36 ms |
| no output store | 6.43 ms | store = 0.12 ms |
| no gather + no WMMA | 1.04 ms | LDS/barrier/epilogue skeleton = 1.04 ms |
| revert epilogue coalescing | 7.15 ms | that fix is worth 0.60 ms |

So the kernel is dominated by the im2col gather: 1.2e9 two-byte loads over a
2.4 GB expansion of a 268 MB input. Even with the gather free it would sit
at 2.7 ms — still above stock's 1.53 ms — so this design cannot win here.

Two further gather fixes were tried and **rejected on measurement**:
pixel-major lane mapping (neutral) and splitting address computation from
the loads (worse — register pressure).

## CK instance sweep

`ck_sweep/` — `sweep.hip` times every tuned CK WMMA conv instance for a
shape; `results_*.txt` are the runs that decided which instance tuples
`src/cuda/ck_conv_fwd_impl.hpp` compiles in. The winner is at index 3 for
both 2D dtypes and index 11 for both 3D dtypes — i.e. inside `part1` for 2D
but only inside `part2` for 3D, which is why both are compiled.

| | best CK | vs stock |
|---|---|---|
| conv2d fp16 | 1.85 ms | stock wins (1.03 ms Winograd) |
| conv2d bf16 | 1.81 ms | **CK ~4.7x** (stock has no bf16 Winograd) |
| conv3d fp16 | 1.16 ms | **CK ~2.6x** |
| conv3d bf16 | 1.19 ms | **CK ~2.8x** |

`ck_sweep/ck_headers_used.txt` is the exact set of CK headers pulled in (139;
all header-only, zero CK `.cpp` compiled or linked).

## Contents

- `logs/build_0*.log` — build logs in chronological order, including the
  43-`[[nodiscard]]`-warning clean build and the serial-vs-ninja pair
  (`build_08` distutils, `build_09` ninja).
- `logs/miopen_*.log` — MIOpen solver-search logs.
- `benchmarks/conv2d_fp16_bench.txt`, `..._1024x512.txt` — output of
  `tools/bench_conv2d_fp16.py` on two shapes; the pair is the point, since
  the algorithm/implementation split differs between them.
- `benchmarks/bench_py_baseline.txt` — `tools/bench.py` before any changes.
- `benchmarks/*.py` — the ad-hoc scripts used along the way (shape sweeps,
  CK correctness across 72 cases, stock-vs-patched comparisons).
- `ablation/`, `ck_sweep/` — as above.

## Two traps if you re-run any of this by hand

1. **Disable the monkeypatch before profiling stock.** `import torch` in this
   environment auto-imports `amd_tuned_torch` (a line was added to
   `torch/__init__.py`), so `F.conv2d` is patched and never reaches MIOpen —
   the log comes back empty, which looks exactly like the logging flags not
   working on this build. `amd_tuned_torch.disable()` first.
2. **MIOpen logs from C++ to fd 2**, so Python-level stderr redirection does
   not capture it; `tools/bench_conv2d_fp16.py` uses an fd-level `dup2`.
