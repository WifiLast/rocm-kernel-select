"""Finds two crossovers for flexgemm_ops's sparse conv1d/2d/3d switch and
saves them as this GPU's calibration (amd_tuned_torch/sparse_conv_calibration.py)
-- replacing the two hardcoded guesses flexgemm_ops.py's own comments
already flag as unvalidated:

  - min_positions (AMD_TUNED_TORCH_SPARSE_CONV{1,2,3}D_MIN_POSITIONS, default
    1024): the spatial size below which the sparse path can't win even in
    its best (most favorable/sparse) case.
  - max_occupancy (AMD_TUNED_TORCH_SPARSE_CONV{1,2,3}D_MAX_OCCUPANCY, default
    0.1): the occupancy fraction above which the sparse path is doing
    nearly as much work as dense while still paying its own overhead, at a
    size comfortably above the min_positions crossover.

WHY TWO SEPARATE SWEEPS, NOT ONE 2D SWEEP. Each answers a different
question and is used as a different kind of gate in flexgemm_ops.py
(min_positions is checked BEFORE occupancy is even estimated; max_occupancy
is checked after). Sweeping both together (a full size x occupancy grid)
would answer a more complete question -- the true 2D win/lose boundary --
but at a cost that scales with the product of both sweep lengths rather
than the sum, and flexgemm_ops.py's own gating structure only ever needs
one 1D slice of that boundary along each axis, not the full surface. So:

  - The SIZE sweep fixes occupancy at one deliberately favorable value
    (default 5% -- see --occupancy) and finds the smallest size where
    sparse still wins, and keeps winning for every larger size measured
    (a single lucky win with a loss right above it is noise, not a trend).
  - The OCCUPANCY sweep fixes size at one deliberately large value (default
    65536 spatial positions -- see --occupancy-sweep-size, comfortably
    above where min_positions typically lands, so this measurement isn't
    confounded by the small-size effect the other sweep is about) and
    finds the LARGEST occupancy where sparse still wins, and kept winning
    for every SMALLER occupancy measured (occupancy sweeps ascending, so
    the stability check runs in the opposite direction from the size
    sweep's).

Run either or both via --sweep (default: both).

METHOD (both sweeps). Build a synthetic input at the candidate size/
occupancy, time real F.convNd and flexgemm_ops.sparse_convNd_from_dense
with CUDA events (same warmup/median-of-N-runs shape as
amd_tuned_torch.kernel_select's own _time -- see that module for why one
iteration count works: this compares kernels that differ by integer
factors, not by percentages), and record which won.

THIRD CONTENDER FOR conv3d: torchsparse. When third_party/torchsparse is
installed, every conv3d measurement ALSO times
torchsparse_ops.sparse_conv3d_from_dense alongside dense and flex_gemm,
printing a three-way (or more) comparison and recording all of them in the
saved metadata. This is purely informational, though -- torchsparse is a
separate, more feature-complete sparse-conv library
(amd_tuned_torch/torchsparse_ops.py) that is NOT wired into any runtime
dispatch switch (unlike flex_gemm's sparse_conv3d, which
flexgemm_ops.maybe_sparse_conv3d actually routes F.conv3d calls through),
so only flex_gemm's own win/lose record against dense decides the
min_positions/max_occupancy crossover this script calibrates. There is
also no 1D/2D torchsparse conv to compare against for those dimensions --
this three-way comparison only ever appears for --dims conv3d.

SYNTHETIC DATA'S SPATIAL PATTERN. Occupied positions are placed in a
handful of spatially-clustered blobs by default (--pattern clustered),
not scattered uniformly at random (--pattern uniform, the only pattern
this script originally used). Real sparse data this feature targets --
voxelized surfaces, point clouds, masked/inpainted latent regions -- is
spatially coherent, not uniform noise; a crossover measured against
uniform noise may not transfer to what this is actually meant to help
with. Both patterns are offered since it's not yet established which one
predicts real workloads better -- run both and compare if in doubt.

Run with:

    python tools/benchmark_sparse_conv.py                       # all three dims, both sweeps
    python tools/benchmark_sparse_conv.py --dims conv2d          # just one dim
    python tools/benchmark_sparse_conv.py --sweep size            # only the min_positions sweep
    python tools/benchmark_sparse_conv.py --sweep occupancy       # only the max_occupancy sweep
    python tools/benchmark_sparse_conv.py --pattern uniform       # old uniform-random occupancy
    python tools/benchmark_sparse_conv.py --occupancy 0.05        # size sweep's fixed occupancy
    python tools/benchmark_sparse_conv.py --occupancy-sweep-size 65536  # occupancy sweep's fixed size
    python tools/benchmark_sparse_conv.py --reset                 # wipe calibration, don't re-measure

Also wired into setup.py as a proper command (BenchmarkSparseConv there),
same options via setuptools' own --long-option parsing:

    python setup.py benchmark_sparse_conv
    python setup.py benchmark_sparse_conv --dims conv2d,conv3d
    python setup.py benchmark_sparse_conv --sweep size
    python setup.py benchmark_sparse_conv --occupancy 0.1
    python setup.py benchmark_sparse_conv --reset

This does NOT run automatically as part of build_ext/install -- see
setup.py's own comment above BenchmarkSparseConv for why (needs a real GPU
and real wall-clock time, neither of which belongs on an ordinary build's
critical path).
"""
from __future__ import annotations

import os

# Must precede `import torch` -- MIOpen reads these once, at
# initialization (which a ROCm torch build can trigger as a side effect of
# `import torch` itself), not per-call -- see
# amd_tuned_torch.benchmark_report.MIOPEN_LOGGING_ENV's own comment for why
# this has to be inlined here rather than imported from that module.
# Values match tools/bench_conv2d_fp16.py's own (see analyse/README.md).
os.environ.setdefault("MIOPEN_ENABLE_LOGGING", "1")
os.environ.setdefault("MIOPEN_ENABLE_LOGGING_CMD", "1")
os.environ.setdefault("MIOPEN_LOG_LEVEL", "6")

import argparse  # noqa: E402
import sys  # noqa: E402
from typing import Callable, List, Optional, Tuple  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import amd_tuned_torch  # noqa: E402,F401  (registers amd_tuned_torch.flexgemm_ops via package import)
from amd_tuned_torch import benchmark_report, flexgemm_ops, sparse_conv_calibration, torchsparse_ops  # noqa: E402

_WARMUP = 3
_ITERS = 5

# Spatial-position sizes to sweep, as powers of 2 -- covers "a few pixels"
# up to "a full SDXL-scale latent" without an unreasonably long run.
_DEFAULT_SIZES = [2 ** p for p in range(6, 21)]  # 64 .. 1,048,576
_DEFAULT_OCCUPANCY = 0.05
# Ascending so the occupancy sweep's "stays winning below this" scan (see
# _sweep_occupancy_one) can walk the list in the order it's given.
_DEFAULT_OCCUPANCIES = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
_DEFAULT_OCCUPANCY_SWEEP_SIZE = 65536
_C_IN, _C_OUT, _KERNEL = 4, 8, 3


def _make_shape(dim: str, n_positions: int, batch: int = 1) -> Tuple[int, ...]:
    """n_positions spatial positions (batch already divided out), shaped as
    close to an isotropic (square/cube) extent as integer rounding allows
    -- the exact aspect ratio doesn't matter for this measurement, only the
    total position count does."""
    per_batch = max(1, n_positions // batch)
    if dim == "conv1d":
        return (per_batch,)
    if dim == "conv2d":
        side = max(1, round(per_batch ** 0.5))
        return (side, side)
    if dim == "conv3d":
        side = max(1, round(per_batch ** (1 / 3)))
        return (side, side, side)
    raise ValueError(dim)


def _uniform_occupied_indices(n_total: int, n_occupied: int, device: torch.device) -> torch.Tensor:
    """n_occupied flat indices chosen uniformly at random out of n_total --
    the original (pre-clustering) generator, kept as --pattern uniform."""
    return torch.randperm(n_total, device=device)[:n_occupied]


def _clustered_occupied_coords(spatial: Tuple[int, ...], n_occupied: int,
                                device: torch.device, cluster_radius: int = 2) -> torch.Tensor:
    """n_occupied (approx) coordinates in `spatial`, arranged as a handful
    of spatially-compact blobs rather than uniform noise: pick a small
    number of random cluster centers, then scatter points within
    `cluster_radius` of each center (clipped to bounds), dedup, and trim to
    the target count. Fully vectorized (no per-point Python loop) so this
    stays cheap even for the largest sizes in the default sweep.

    Number of clusters is chosen so each one gets roughly 16 points --
    enough clusters that a large occupied set doesn't collapse into one
    single misshapen blob, few enough that a small one isn't split into
    single-point "clusters" indistinguishable from --pattern uniform."""
    ndim = len(spatial)
    n_clusters = max(1, min(64, -(-n_occupied // 16)))
    per_cluster = -(-n_occupied // n_clusters)  # ceil

    centers = torch.stack(
        [torch.randint(0, s, (n_clusters,), device=device) for s in spatial], dim=1)  # [n_clusters, ndim]
    offsets = torch.randint(-cluster_radius, cluster_radius + 1,
                             (n_clusters, per_cluster, ndim), device=device)
    coords = (centers.unsqueeze(1) + offsets).reshape(-1, ndim)  # [n_clusters*per_cluster, ndim]
    for d, s in enumerate(spatial):
        coords[:, d] = coords[:, d].clamp(0, s - 1)
    coords = torch.unique(coords, dim=0)
    if coords.shape[0] > n_occupied:
        perm = torch.randperm(coords.shape[0], device=device)[:n_occupied]
        coords = coords[perm]
    return coords


def _ravel(coords: torch.Tensor, spatial: Tuple[int, ...]) -> torch.Tensor:
    """[N,ndim] coordinates -> [N] flat indices into a row-major
    prod(spatial)-length array, matching torch's own .view(-1) layout."""
    strides = [1] * len(spatial)
    for i in range(len(spatial) - 2, -1, -1):
        strides[i] = strides[i + 1] * spatial[i + 1]
    strides_t = torch.tensor(strides, device=coords.device, dtype=coords.dtype)
    return (coords * strides_t).sum(dim=1)


def _make_input(dim: str, spatial: Tuple[int, ...], occupancy: float,
                 device: torch.device, pattern: str = "clustered") -> torch.Tensor:
    shape = (1, _C_IN, *spatial)
    x = torch.zeros(shape, device=device)
    n_total = 1
    for s in spatial:
        n_total *= s
    n_occupied = max(1, int(round(n_total * occupancy)))
    flat = x.view(1, _C_IN, -1)

    if pattern == "clustered":
        coords = _clustered_occupied_coords(spatial, n_occupied, device)
        idx = _ravel(coords, spatial)
        if idx.numel() < n_occupied:
            # Clusters didn't reach the target count (typical for a small
            # spatial extent where cluster_radius already covers most of
            # it) -- top up the remainder uniformly rather than silently
            # under-shooting the requested occupancy.
            remaining = n_occupied - idx.numel()
            extra = _uniform_occupied_indices(n_total, remaining, device)
            idx = torch.unique(torch.cat([idx, extra]))[:n_occupied]
    elif pattern == "uniform":
        idx = _uniform_occupied_indices(n_total, n_occupied, device)
    else:
        raise ValueError(f"unknown pattern {pattern!r}")

    flat[0, :, idx] = torch.randn(_C_IN, idx.numel(), device=device)
    return x


def _make_weight(dim: str, device: torch.device) -> torch.Tensor:
    ndim = {"conv1d": 1, "conv2d": 2, "conv3d": 3}[dim]
    return torch.randn(_C_OUT, _C_IN, *([_KERNEL] * ndim), device=device)


def _dense_fn(dim: str) -> Callable:
    return {"conv1d": F.conv1d, "conv2d": F.conv2d, "conv3d": F.conv3d}[dim]


def _sparse_from_dense_fn(dim: str) -> Callable:
    return {
        "conv1d": flexgemm_ops.sparse_conv1d_from_dense,
        "conv2d": flexgemm_ops.sparse_conv2d_from_dense,
        "conv3d": flexgemm_ops.sparse_conv3d_from_dense,
    }[dim]


def _sparse_contenders(dim: str) -> List[Tuple[str, Callable]]:
    """Named (backend, from_dense_fn) contenders for `dim`, ALWAYS with
    flex_gemm first. Only flex_gemm's win/lose record decides the
    min_positions/max_occupancy crossover this script calibrates (see
    _sweep_size/_sweep_occupancy) -- it's the only backend actually wired
    into amd_tuned_torch's runtime dispatch
    (flexgemm_ops.maybe_sparse_conv{1,2,3}d). torchsparse is added as a
    SECOND, purely-informational contender for conv3d specifically (it has
    no 1D/2D conv in its own API, and third_party/torchsparse must be
    installed) -- its numbers are printed and recorded in the saved
    metadata's `sweep` entries, but never used to decide what gets
    calibrated, since nothing in amd_tuned_torch currently dispatches to it."""
    contenders = [("flex_gemm", _sparse_from_dense_fn(dim))]
    if dim == "conv3d" and torchsparse_ops.available():
        contenders.append(("torchsparse", torchsparse_ops.sparse_conv3d_from_dense))
    return contenders


def _pad(dim: str) -> Tuple[int, ...]:
    ndim = {"conv1d": 1, "conv2d": 2, "conv3d": 3}[dim]
    return tuple([_KERNEL // 2] * ndim)


def _time_ms(fn: Callable[[], Optional[torch.Tensor]], device: torch.device) -> Optional[float]:
    """None if `fn` ever returns None (declines) or raises -- same "can't
    measure, so it can't win" convention as kernel_select._time."""
    is_cuda = device.type == "cuda"
    try:
        for _ in range(_WARMUP):
            if fn() is None:
                return None
        if is_cuda:
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(_ITERS):
                fn()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / _ITERS
        import time
        t0 = time.perf_counter()
        for _ in range(_ITERS):
            fn()
        return (time.perf_counter() - t0) * 1000 / _ITERS
    except (RuntimeError, TypeError, ValueError):
        return None


def _time_all(dim: str, contenders: List[Tuple[str, Callable]], x: torch.Tensor, weight: torch.Tensor,
               padding: Tuple[int, ...], device: torch.device) -> Tuple[Optional[float], "dict[str, Optional[float]]"]:
    """Times dense once and every sparse contender once each. Returns
    (dense_ms, {contender_name: ms_or_None})."""
    dense_ms = _time_ms(lambda: _dense_fn(dim)(x, weight, padding=padding), device)
    contender_ms = {
        name: _time_ms(lambda fn=fn: fn(x, weight, padding=padding), device)
        for name, fn in contenders
    }
    return dense_ms, contender_ms


def _format_contenders(dense_ms: Optional[float], contender_ms: "dict[str, Optional[float]]") -> str:
    """'dense=1.23  flex_gemm=0.98  torchsparse=1.05  winner=flex_gemm' --
    the winner is whichever of dense/every contender measured fastest
    (None entries can't win), purely for the printed report; it is NOT
    what decides the saved min_positions/max_occupancy crossover (that's
    flex_gemm's own win/lose against dense specifically, computed
    separately -- see _sweep_size/_sweep_occupancy)."""
    parts = [f"dense={dense_ms!s:>10}"]
    candidates = {"dense": dense_ms}
    for name, ms in contender_ms.items():
        parts.append(f"{name}={ms!s:>10}")
        candidates[name] = ms
    winner = min((name for name, ms in candidates.items() if ms is not None),
                 key=lambda name: candidates[name], default="none")
    return "  ".join(parts) + f"  winner={winner}"


def _sweep_size(dim: str, sizes: List[int], occupancy: float, pattern: str,
                 device: torch.device) -> dict:
    weight = _make_weight(dim, device)
    padding = _pad(dim)
    contenders = _sparse_contenders(dim)
    results = []
    for n_positions in sizes:
        spatial = _make_shape(dim, n_positions)
        x = _make_input(dim, spatial, occupancy, device, pattern=pattern)
        dense_ms, contender_ms = _time_all(dim, contenders, x, weight, padding, device)

        actual_n = x.shape[0] * (x.numel() // x.shape[0] // x.shape[1])
        flex_gemm_ms = contender_ms["flex_gemm"]
        won = flex_gemm_ms is not None and dense_ms is not None and flex_gemm_ms < dense_ms
        results.append({
            "n_positions": actual_n, "dense_ms": dense_ms,
            **{f"{name}_ms": ms for name, ms in contender_ms.items()}, "sparse_won": won,
        })
        print(f"  {dim} size n={actual_n:>10,}  {_format_contenders(dense_ms, contender_ms)}")

    # Crossover: smallest size where flex_gemm wins AND stays winning for
    # every larger size measured -- a single win with a loss right above it
    # is noise, not a real trend. Scan from the end so a late, permanent win
    # is found even if smaller sizes are noisy in both directions.
    crossover = None
    for i in range(len(results) - 1, -1, -1):
        if not results[i]["sparse_won"]:
            break
        crossover = results[i]["n_positions"]
    return {"min_positions": crossover, "sweep": results}


def _sweep_occupancy(dim: str, occupancies: List[float], size: int, pattern: str,
                      device: torch.device) -> dict:
    weight = _make_weight(dim, device)
    padding = _pad(dim)
    spatial = _make_shape(dim, size)
    contenders = _sparse_contenders(dim)
    results = []
    for occ in occupancies:
        x = _make_input(dim, spatial, occ, device, pattern=pattern)
        dense_ms, contender_ms = _time_all(dim, contenders, x, weight, padding, device)

        flex_gemm_ms = contender_ms["flex_gemm"]
        won = flex_gemm_ms is not None and dense_ms is not None and flex_gemm_ms < dense_ms
        results.append({
            "occupancy": occ, "dense_ms": dense_ms,
            **{f"{name}_ms": ms for name, ms in contender_ms.items()}, "sparse_won": won,
        })
        print(f"  {dim} occupancy={occ:<6}  {_format_contenders(dense_ms, contender_ms)}")

    # Crossover: LARGEST occupancy where flex_gemm wins AND stays winning
    # for every SMALLER occupancy measured (occupancies ascends, so this
    # walks forward rather than backward -- the inverse direction from the
    # size sweep's, since higher occupancy favors dense the way smaller
    # size does) -- a single win at high occupancy with a loss right below
    # it is noise, not a real trend.
    crossover = None
    for r in results:
        if not r["sparse_won"]:
            break
        crossover = r["occupancy"]
    return {"max_occupancy": crossover, "sweep": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dims", nargs="+", choices=["conv1d", "conv2d", "conv3d"],
                         default=["conv1d", "conv2d", "conv3d"])
    parser.add_argument("--sweep", choices=["both", "size", "occupancy"], default="both",
                         help="Which crossover(s) to measure (default: both).")
    parser.add_argument("--pattern", choices=["clustered", "uniform"], default="clustered",
                         help="Spatial pattern for synthetic occupied positions (default: clustered).")
    parser.add_argument("--occupancy", type=float, default=_DEFAULT_OCCUPANCY,
                         help="Fixed, favorable occupancy the SIZE sweep sweeps size at (default 0.05).")
    parser.add_argument("--sizes", type=int, nargs="+", default=None,
                         help="Override the default power-of-2 size sweep.")
    parser.add_argument("--occupancy-sweep-size", type=int, default=_DEFAULT_OCCUPANCY_SWEEP_SIZE,
                         help="Fixed spatial size the OCCUPANCY sweep sweeps occupancy at "
                              f"(default {_DEFAULT_OCCUPANCY_SWEEP_SIZE:,}).")
    parser.add_argument("--occupancies", type=float, nargs="+", default=None,
                         help="Override the default occupancy sweep (ascending order expected).")
    parser.add_argument("--reset", action="store_true",
                         help="Delete this GPU's saved calibration and exit without measuring.")
    args = parser.parse_args()

    if args.reset:
        sparse_conv_calibration.reset()
        print(f"Reset calibration for {sparse_conv_calibration.device_key()} "
              f"-- flexgemm_ops will use its hardcoded defaults until this is run again.")
        # Re-export so the combined report stops carrying this now-deleted
        # domain's stale numbers.
        benchmark_report.save()
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA/ROCm device visible -- timing on CPU. This measures relative "
              "Python/indexing overhead, not the GPU kernel crossover this script exists to "
              "find; treat any resulting calibration as provisional and re-run on the real "
              "target GPU before trusting it.", file=sys.stderr)

    sizes = args.sizes or _DEFAULT_SIZES
    occupancies = args.occupancies or _DEFAULT_OCCUPANCIES
    print(f"Device: {sparse_conv_calibration.device_key()}")
    print(f"Pattern: {args.pattern}, sweep: {args.sweep}")
    if args.sweep in ("both", "size"):
        print(f"Size sweep occupancy: {args.occupancy}, sizes: {sizes}")
    if args.sweep in ("both", "occupancy"):
        print(f"Occupancy sweep size: {args.occupancy_sweep_size:,}, occupancies: {occupancies}")
    print()

    measurements = {}
    metadata = {}
    # MIOpen's own solver-selection log for the whole sweep below (every
    # dense F.convNd call this makes reaches MIOpen the normal way -- the
    # amd_tuned_torch monkeypatch tiers aren't installed by this tool, only
    # imported for flexgemm_ops) -- see
    # amd_tuned_torch.benchmark_report.CaptureMiopenLog's own docstring.
    # Only produces anything if MIOPEN_ENABLE_LOGGING etc. actually took
    # effect (a ROCm build with MIOpen; harmless empty capture otherwise).
    with benchmark_report.CaptureMiopenLog() as miopen_cap:
        for dim in args.dims:
            print(f"=== {dim} ===")
            dim_values = {}
            dim_metadata = {}

            if args.sweep in ("both", "size"):
                size_outcome = _sweep_size(dim, sizes, args.occupancy, args.pattern, device)
                if size_outcome["min_positions"] is not None:
                    dim_values["min_positions"] = size_outcome["min_positions"]
                    print(f"  -> min_positions crossover: {size_outcome['min_positions']:,} spatial positions")
                else:
                    print("  -> size sweep: sparse never won a stable trend; leaving min_positions uncalibrated")
                dim_metadata["size_sweep"] = {"occupancy": args.occupancy, "pattern": args.pattern,
                                              "sweep": size_outcome["sweep"]}

            if args.sweep in ("both", "occupancy"):
                occ_outcome = _sweep_occupancy(dim, occupancies, args.occupancy_sweep_size,
                                                args.pattern, device)
                if occ_outcome["max_occupancy"] is not None:
                    dim_values["max_occupancy"] = occ_outcome["max_occupancy"]
                    print(f"  -> max_occupancy crossover: {occ_outcome['max_occupancy']}")
                else:
                    print("  -> occupancy sweep: sparse never won a stable trend; leaving max_occupancy uncalibrated")
                dim_metadata["occupancy_sweep"] = {"size": args.occupancy_sweep_size, "pattern": args.pattern,
                                                    "sweep": occ_outcome["sweep"]}

            print()
            if dim_values:
                measurements[dim] = dim_values
            metadata[dim] = dim_metadata

    if measurements:
        sparse_conv_calibration.save(measurements, extra_metadata=metadata)
        print(f"Saved to {sparse_conv_calibration.calibration_path()}")
    else:
        print("Nothing calibrated -- no dimension found a stable sparse-wins crossover "
              "in this sweep. Nothing was written; existing calibration (if any) is untouched.")

    if miopen_cap.text:
        log_path = benchmark_report.save_miopen_log("sparse_conv", miopen_cap.text)
        print(f"MIOpen debug log for this run saved to {log_path}")

    # Re-export the combined report (this calibration file plus whatever
    # fftconv_calibration/others have already written) regardless of
    # whether this run calibrated anything new -- see
    # amd_tuned_torch/benchmark_report.py's own docstring for why this is a
    # full re-export every time, not a partial merge.
    report_path = benchmark_report.save()
    print(f"Combined benchmark report (all calibration domains + system info) "
          f"saved to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
