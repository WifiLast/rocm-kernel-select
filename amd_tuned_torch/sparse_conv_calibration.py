"""Persisted, measured crossover points for flexgemm_ops's sparse/dense
conv1d/2d/3d switch -- replaces TWO hardcoded, explicitly-documented-as-
unvalidated guesses (AMD_TUNED_TORCH_SPARSE_CONV{1,2,3}D_MIN_POSITIONS'
default of 1024, and _MAX_OCCUPANCY's default of 0.1) with numbers actually
measured on this GPU, once tools/benchmark_sparse_conv.py has been run.

WHY THIS EXISTS. flexgemm_ops.maybe_sparse_conv{1,2,3}d makes two separate
guesses about when the sparse path is worth trying:

  - min_positions (the "Minimum-size gate" section in flexgemm_ops.py):
    skip the sparse path outright below this many spatial positions, on
    the reasoning that gather-scatter/index-grid bookkeeping overhead
    can't be beaten by a small enough dense conv regardless of occupancy.
  - max_occupancy: above this fraction of occupied positions, the sparse
    path is doing nearly as much work as dense while paying its own
    overhead on top, so it's not worth attempting.

Neither was a measurement. tools/benchmark_sparse_conv.py closes both gaps
with two separate sweeps: one over spatial size at a fixed favorable
(sparse) occupancy to find min_positions, one over occupancy at a fixed
large size to find max_occupancy -- see that script's own docstring for
why they have to be separate sweeps rather than one combined one.

DISK PERSISTENCE, same pattern as amd_tuned_torch/kernel_select.py's own
disk cache (_device_key/_cache_path/temp-file+os.replace) -- a value
measured on one GPU model or torch/ROCm build is not evidence about
another, so this is keyed by GPU name + torch version + Python version,
one JSON file per key under _sparse_conv_calibration/.

Set AMD_TUNED_TORCH_SPARSE_CONV_CALIBRATION=0 to ignore any persisted file and
always fall back to the hardcoded env-var-configurable defaults in
flexgemm_ops.py (same on/off gate shape as AMD_TUNED_TORCH_KERNEL_SELECT_CACHE).

RESET. Set AMD_TUNED_TORCH_SPARSE_CONV_CALIBRATION_RESET=1 to have `load()`
ignore (and delete) whatever is currently on disk for this device -- the
next flexgemm_ops import falls back to the hardcoded defaults until
tools/benchmark_sparse_conv.py is run again to write a fresh file. This is
the "load it until a reset flag is set" behavior: normal imports load the
persisted calibration silently and cheaply (a single JSON read), and stay
on it indefinitely -- nothing here re-benchmarks automatically or expires
an entry by itself. Re-measuring only ever happens when a human explicitly
runs the benchmark script again (e.g. after a ROCm/ PyTorch upgrade, or a
different GPU).

Usage:

    # Run once (or after a ROCm/PyTorch/GPU change) to (re)calibrate:
    python tools/benchmark_sparse_conv.py

    # Reset back to the hardcoded defaults without benchmarking again:
    AMD_TUNED_TORCH_SPARSE_CONV_CALIBRATION_RESET=1 python -c "import amd_tuned_torch"
    # or, programmatically:
    from amd_tuned_torch import sparse_conv_calibration
    sparse_conv_calibration.reset()
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, Optional

import torch

_ENABLED = os.environ.get("AMD_TUNED_TORCH_SPARSE_CONV_CALIBRATION", "1") != "0"
_RESET = os.environ.get("AMD_TUNED_TORCH_SPARSE_CONV_CALIBRATION_RESET", "0") != "0"
_CALIBRATION_DIR = os.environ.get(
    "AMD_TUNED_TORCH_SPARSE_CONV_CALIBRATION_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sparse_conv_calibration"),
)

# The three top-level keys a calibration file may carry -- one entry per
# conv dimensionality, matching flexgemm_ops's own naming.
_DIMS = ("conv1d", "conv2d", "conv3d")

# The measurement fields within each dimensionality's entry, and the
# type(s) each must be to be trusted -- either can be present without the
# other (tools/benchmark_sparse_conv.py's --sweep flag can run just one
# sweep), so load()/save() treat them independently rather than requiring
# both. min_positions is always a whole count; max_occupancy is a fraction
# but JSON round-trips a value like 1.0 as either int or float depending
# on how it was written, so both are accepted for it.
_MEASUREMENT_TYPES = {"min_positions": (int,), "max_occupancy": (int, float)}


def enabled() -> bool:
    """True unless AMD_TUNED_TORCH_SPARSE_CONV_CALIBRATION=0. Read once at
    import time, same posture as every other env-var gate in this package."""
    return _ENABLED


def device_key() -> str:
    """GPU model + torch + Python version, sanitized for a filename -- a
    crossover size measured on one of these is not evidence about a
    different one. Identical construction to
    amd_tuned_torch.kernel_select._device_key(), duplicated rather than
    imported to keep this module independently droppable (same posture as
    every *_ops module in this package not importing its siblings)."""
    try:
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        name = "unknown"
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    torch_ver = torch.__version__.replace("+", "_").replace("/", "_")
    return f"{safe_name}-torch-{torch_ver}-cp{sys.version_info.major}{sys.version_info.minor}"


def calibration_path(key: Optional[str] = None) -> str:
    return os.path.join(_CALIBRATION_DIR, f"{key or device_key()}.json")


def load() -> Dict[str, Dict[str, float]]:
    """{"conv1d": {"min_positions": ..., "max_occupancy": ...}, ...} for
    whichever dimensionalities have at least one measured field for this
    device -- a dimensionality with only one of the two fields measured
    (e.g. tools/benchmark_sparse_conv.py was run with --sweep size only)
    returns just that one key, letting the caller fall back to its own
    hardcoded default for the other. Dimensionalities missing from the
    file entirely (or the whole file, if none exists yet) are simply
    absent from the returned dict.

    Returns {} (never raises) when disabled
    (AMD_TUNED_TORCH_SPARSE_CONV_CALIBRATION=0), reset
    (AMD_TUNED_TORCH_SPARSE_CONV_CALIBRATION_RESET=1 -- and deletes the
    stale file in that case, see reset()), or the file is missing/corrupt/
    unreadable -- calibration is a pure optimization input, never something
    an import should be able to fail on."""
    if not _ENABLED:
        return {}
    if _RESET:
        reset()
        return {}
    try:
        with open(calibration_path(), "r") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    result: Dict[str, Dict[str, float]] = {}
    for dim in _DIMS:
        entry = raw.get(dim)
        if not isinstance(entry, dict):
            continue
        values = {}
        for field, types in _MEASUREMENT_TYPES.items():
            value = entry.get(field)
            if isinstance(value, types) and not isinstance(value, bool):
                values[field] = value
        if values:
            result[dim] = values
    return result


def save(measurements: Dict[str, Dict[str, float]],
         extra_metadata: Optional[Dict[str, Dict]] = None) -> None:
    """Persist {"conv1d": {"min_positions": ..., "max_occupancy": ...},
    ...} (any subset of _DIMS, and any subset of the two measurement
    fields within each) for this device, merging onto whatever is already
    on disk for it -- both across dimensionalities AND within one
    dimensionality's own fields, so calibrating min_positions today and
    max_occupancy tomorrow (or one dimensionality at a time) never
    clobbers a measurement this call didn't re-measure.

    `extra_metadata`, if given, is a {dim: {...}} dict of additional
    (non-authoritative) fields to store alongside each dimensionality's
    measurements -- tools/benchmark_sparse_conv.py uses this for the raw
    sweep data (timings per size/occupancy) so the calibration file is
    also a record of how the numbers were derived, not just the numbers
    themselves.

    Written via a temp file + os.replace, same crash-safety as
    amd_tuned_torch.kernel_select._save_disk_cache -- a process killed
    mid-write can never leave a truncated/corrupt file for the next run.
    Silently does nothing on any OSError (e.g. an unwritable directory) --
    calibration is best-effort, never something a benchmark run should
    crash over after doing the actual (expensive) measurement work."""
    path = calibration_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "r") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                raw = {}
        except (OSError, ValueError):
            raw = {}
        for dim, values in measurements.items():
            if dim not in _DIMS:
                continue
            entry = raw.get(dim)
            entry = dict(entry) if isinstance(entry, dict) else {}
            for field, types in _MEASUREMENT_TYPES.items():
                if field in values:
                    caster = int if field == "min_positions" else float
                    entry[field] = caster(values[field])
            if extra_metadata and dim in extra_metadata:
                entry.update(extra_metadata[dim])
            raw[dim] = entry
        raw["_device_key"] = device_key()
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(raw, f, indent=2)
        os.replace(tmp, path)
    except OSError:
        pass


def reset() -> None:
    """Deletes this device's calibration file, if any -- the next load()
    (this process or a later one) returns {} until
    tools/benchmark_sparse_conv.py is run again. Missing-file is not an
    error (nothing to reset is a no-op, not a failure)."""
    try:
        os.remove(calibration_path())
    except FileNotFoundError:
        pass
    except OSError:
        pass
