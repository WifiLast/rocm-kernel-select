"""Persisted, measured crossover points for fftconv_ops's FFT-vs-dense
conv2d/conv3d tiers -- same shape as amd_tuned_torch/sparse_conv_calibration.py
(flexgemm_ops's own calibration file), applied to a different pair of
hardcoded, explicitly-documented-as-unvalidated guesses:
_FFTCONV_CONV3D_MIN_POSITIONS' default of 4096 (`__init__.py`'s
_fftconv_conv_candidate) with a number actually measured on this GPU, once
tools/benchmark_fftconv3d_min_positions.py has been run.

WHY THIS EXISTS. `_fftconv_conv_candidate` declines a conv3d call outright
when `input`'s spatial size (batch * D*H*W) is below
_FFTCONV_CONV3D_MIN_POSITIONS, on the reasoning that FFT-conv3d's padded-
transform overhead can't be beaten even at a favorable (large) kernel width
below some volume size -- see that function's own docstring. That
threshold was never measured, only guessed; this module + its benchmark
script close that gap the same way sparse_conv_calibration.py/
benchmark_sparse_conv.py already closed it for flexgemm_ops's sparse/dense
switch.

SCHEMA. Deliberately shaped like sparse_conv_calibration.py's (dim -> field
-> value) even though only `{"conv3d": {"min_positions": ...}}` is written
by anything today -- `min_kernel` and `conv2d` are accepted fields/dims too
(fftconv_ops/`__init__.py`'s own _FFTCONV_CONV{2,3}D_MIN_KERNEL are
hardcoded guesses of exactly the same kind, just not yet the subject of a
benchmark script), so a future calibration script for those doesn't need a
new file format, only a new sweep.

DISK PERSISTENCE, identical pattern to sparse_conv_calibration.py: keyed by
GPU name + torch version + Python version (a value measured on one GPU
model or torch/ROCm build is not evidence about another), one JSON file per
key under _fftconv_calibration/, temp-file + os.replace for crash safety.

Set AMD_TUNED_TORCH_FFTCONV_CALIBRATION=0 to ignore any persisted file and
always fall back to the hardcoded env-var-configurable defaults in
`__init__.py` (same on/off gate shape as
AMD_TUNED_TORCH_SPARSE_CONV_CALIBRATION).

RESET. Set AMD_TUNED_TORCH_FFTCONV_CALIBRATION_RESET=1 to have `load()`
ignore (and delete) whatever is currently on disk for this device -- the
next `amd_tuned_torch` import falls back to the hardcoded defaults until
tools/benchmark_fftconv3d_min_positions.py is run again. Same "load it
until a reset flag is set" behavior as sparse_conv_calibration.py: normal
imports load the persisted calibration silently and cheaply, and stay on
it indefinitely -- nothing here re-benchmarks automatically.

Usage:

    # Run once (or after a ROCm/PyTorch/GPU change) to (re)calibrate:
    python tools/benchmark_fftconv3d_min_positions.py

    # Reset back to the hardcoded default without benchmarking again:
    AMD_TUNED_TORCH_FFTCONV_CALIBRATION_RESET=1 python -c "import amd_tuned_torch"
    # or, programmatically:
    from amd_tuned_torch import fftconv_calibration
    fftconv_calibration.reset()
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, Optional

import torch

_ENABLED = os.environ.get("AMD_TUNED_TORCH_FFTCONV_CALIBRATION", "1") != "0"
_RESET = os.environ.get("AMD_TUNED_TORCH_FFTCONV_CALIBRATION_RESET", "0") != "0"
_CALIBRATION_DIR = os.environ.get(
    "AMD_TUNED_TORCH_FFTCONV_CALIBRATION_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fftconv_calibration"),
)

# The dimensionalities a calibration file may carry -- conv1d isn't here:
# its FFT tier lives entirely in fftconv_ops.py with its own
# _FFTCONV1D_MIN_KERNEL, a separate concern from __init__.py's
# _fftconv_conv_candidate this module calibrates.
_DIMS = ("conv2d", "conv3d")

# The measurement fields within each dimensionality's entry -- both whole
# counts (a kernel width or a spatial-position count), unlike
# sparse_conv_calibration.py's max_occupancy fraction. Either can be present
# without the other, same independent-fields handling as that module.
_MEASUREMENT_TYPES = {"min_kernel": (int,), "min_positions": (int,)}


def enabled() -> bool:
    """True unless AMD_TUNED_TORCH_FFTCONV_CALIBRATION=0. Read once at import
    time, same posture as every other env-var gate in this package."""
    return _ENABLED


def device_key() -> str:
    """GPU model + torch + Python version, sanitized for a filename --
    identical construction to sparse_conv_calibration.device_key()/
    amd_tuned_torch.kernel_select._device_key(), duplicated rather than
    imported to keep this module independently droppable (same posture as
    every *_ops/*_calibration module in this package not importing its
    siblings)."""
    try:
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        name = "unknown"
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    torch_ver = torch.__version__.replace("+", "_").replace("/", "_")
    return f"{safe_name}-torch-{torch_ver}-cp{sys.version_info.major}{sys.version_info.minor}"


def calibration_path(key: Optional[str] = None) -> str:
    return os.path.join(_CALIBRATION_DIR, f"{key or device_key()}.json")


def load() -> Dict[str, Dict[str, int]]:
    """{"conv2d": {"min_kernel": ..., "min_positions": ...}, "conv3d": {...}}
    for whichever dimensionalities have at least one measured field for this
    device. Dimensionalities missing from the file entirely (or the whole
    file, if none exists yet) are simply absent from the returned dict.

    Returns {} (never raises) when disabled
    (AMD_TUNED_TORCH_FFTCONV_CALIBRATION=0), reset
    (AMD_TUNED_TORCH_FFTCONV_CALIBRATION_RESET=1 -- and deletes the stale
    file in that case, see reset()), or the file is missing/corrupt/
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
    result: Dict[str, Dict[str, int]] = {}
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


def save(measurements: Dict[str, Dict[str, int]],
         extra_metadata: Optional[Dict[str, Dict]] = None) -> None:
    """Persist {"conv3d": {"min_positions": ...}, ...} (any subset of
    _DIMS, any subset of the two measurement fields within each) for this
    device, merging onto whatever is already on disk for it -- both across
    dimensionalities AND within one dimensionality's own fields, so
    calibrating min_positions today and min_kernel tomorrow never clobbers
    a measurement this call didn't re-measure.

    `extra_metadata`, if given, is a {dim: {...}} dict of additional
    (non-authoritative) fields to store alongside each dimensionality's
    measurements -- tools/benchmark_fftconv3d_min_positions.py uses this for
    the raw sweep data (timings per size) so the calibration file is also a
    record of how the number was derived, not just the number itself.

    Written via a temp file + os.replace, same crash-safety as
    sparse_conv_calibration.save/amd_tuned_torch.kernel_select._save_disk_cache.
    Silently does nothing on any OSError -- calibration is best-effort,
    never something a benchmark run should crash over after doing the
    actual (expensive) measurement work."""
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
                    entry[field] = int(values[field])
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
    tools/benchmark_fftconv3d_min_positions.py is run again. Missing-file
    is not an error (nothing to reset is a no-op, not a failure)."""
    try:
        os.remove(calibration_path())
    except FileNotFoundError:
        pass
    except OSError:
        pass
