"""Combines every calibration file this package's benchmark tools have
written on this machine (amd_tuned_torch/sparse_conv_calibration.py's,
amd_tuned_torch/fftconv_calibration.py's, and any future *_calibration.py
module of the same shape) into ONE JSON report, with the system metadata
(GPU model, ROCm/HIP version, CUDA version, PyTorch version, Python
version, OS) needed to make sense of numbers measured on someone else's
machine -- meant to be handed off (uploaded, attached to an issue, etc.) to
build a cross-machine picture of where these tuning constants actually
land, not just this one machine's own runtime use of them.

WHY THIS EXISTS SEPARATELY FROM THE CALIBRATION MODULES THEMSELVES. Each
`*_calibration.py` module (sparse_conv_calibration.py, fftconv_calibration.py)
owns its own on-disk format and is the thing amd_tuned_torch/__init__.py and
flexgemm_ops.py actually read from at import time via their own
`_calibrated_default`-style precedence -- that contract (device_key()-named
JSON files under their own `_*_calibration/` directory) stays exactly as it
is; this module doesn't change how any of them load, save, or get consulted
at runtime. It only READS whatever they've already written (their raw
on-disk JSON, extra_metadata sweep details included, not just the filtered
view their own `load()` returns to a runtime caller) and re-packages it
alongside system metadata into one combined file -- an export step, not a
replacement for the per-tool calibration mechanism.

WHAT'S IN THE COMBINED FILE. See `collect()`'s own docstring for the exact
shape; roughly:

    {
      "schema_version": 1,
      "device_key": "...",           # same key format every *_calibration
                                      # module already uses for its own
                                      # per-device filename
      "generated_at": "<ISO 8601 UTC>",
      "system": {
        "gpu_name": "...", "rocm_version": "...", "cuda_version": "...",
        "torch_version": "...", "python_version": "...", "platform": "...",
        "pcie": {"link_speed": "...", "link_width": "..."}
      },
      "sparse_conv": { ... sparse_conv_calibration's raw file content ... },
      "fftconv": { ... fftconv_calibration's raw file content ... },
      "miopen_logs": {"sparse_conv": "...", "fftconv": "..."}
    }

A calibration domain absent from the combined file (or present but with
only some dims/fields populated) simply means that benchmark hasn't been
run yet on this machine, or was only run for some dims -- `collect()` never
raises over a missing calibration file, same "calibration is a pure
optimization input" posture the calibration modules themselves take.

MIOPEN DEBUG LOGGING. `MIOPEN_LOGGING_ENV` and `CaptureMiopenLog` capture
MIOpen's own solver-selection log (same technique and same three env vars
tools/bench_conv2d_fp16.py already uses -- see analyse/README.md) for
whichever tier a benchmark run actually reaches MIOpen through, so a
combined report says not just "which candidate won" but "what MIOpen
itself considered and timed" for that shape. `save_miopen_log(domain, text)`
persists one domain's captured text the same way a *_calibration module
persists its measurements -- as its OWN file
(`<device_key>.<domain>.miopen.log`), so a later `save()` (even from a
different tool run) still finds and re-embeds it, the same reasoning
calibration values persist to their own files rather than living only in
one process's memory. See `MIOPEN_LOGGING_ENV`'s own docstring for why
setting those env vars is NOT a function this module can safely expose to
be *called* -- every benchmark tool must set them itself, inlined, before
its own `import torch`.

GPU PCIE LINK. `system_metadata()`'s `"pcie"` field (link speed + width)
matters because it can materially change relative timings for anything
with a real host<->device transfer in it, and a report meant to be compared
across different people's machines has no other way to know two "same GPU
model" results were measured at, say, PCIe Gen4 x16 vs. Gen3 x4 (a
common outcome of the GPU being in the wrong physical slot, or a
narrower-lane riser/enclosure). Best-effort (Linux sysfs, no ROCm tool
required) -- see `gpu_pcie_info()`'s own docstring for its one real
limitation (multi-GPU disambiguation).

Usage:

    python tools/export_benchmark_report.py

    # or programmatically, e.g. from another tool's own main() right after
    # it writes its own calibration file:
    from amd_tuned_torch import benchmark_report
    path = benchmark_report.save()

Both tools/benchmark_sparse_conv.py and
tools/benchmark_fftconv3d_min_positions.py call `save()` automatically at
the end of a successful run, so the combined file stays current without a
separate manual step -- tools/export_benchmark_report.py exists for
re-combining on demand (e.g. after editing a calibration file by hand, or
simply to regenerate the combined file without re-running any benchmark).
"""
from __future__ import annotations

import datetime
import glob
import importlib
import importlib.util
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Identical values to tools/bench_conv2d_fp16.py's own top-of-file
# os.environ.setdefault calls (see analyse/README.md) -- the single source
# of truth for what "the MIOpen debug flags" means across every benchmark
# tool in this project.
#
# THIS IS NOT A FUNCTION TO CALL, DELIBERATELY. MIOpen reads these once,
# when it initializes, which on a ROCm torch build can happen as a side
# effect of `import torch` itself -- tools/bench_conv2d_fp16.py's own
# module docstring already documents this ("set BELOW BEFORE torch is
# imported, because MIOpen reads them when it initialises"). Every
# benchmark tool must therefore set them ITSELF, inlined, as literally the
# first lines of its own file, before its own `import torch` --
# `from amd_tuned_torch import benchmark_report` (or even
# `from amd_tuned_torch.benchmark_report import MIOPEN_LOGGING_ENV`)
# cannot substitute for that: Python always imports a dotted submodule's
# parent package first, and amd_tuned_torch/__init__.py imports torch
# immediately, before control would ever reach anything in THIS module.
# `MIOPEN_LOGGING_ENV` exists so every tool sets the exact same values from
# one place to copy, not so one can be imported and called too late to
# matter.
MIOPEN_LOGGING_ENV = {
    "MIOPEN_ENABLE_LOGGING": "1",
    "MIOPEN_ENABLE_LOGGING_CMD": "1",
    "MIOPEN_LOG_LEVEL": "6",
}

_REPORT_DIR = os.environ.get(
    "AMD_TUNED_TORCH_BENCHMARK_REPORT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "_benchmark_report"),
)

_SCHEMA_VERSION = 1

# Inline cap for one domain's captured MIOpen log in the combined JSON --
# MIOPEN_LOG_LEVEL=6 over a real multi-size sweep can run to several MB,
# and this file is meant to be small enough to comfortably upload/attach.
# The FULL text is never lost regardless: save_miopen_log always writes it
# untruncated to its own file (miopen_log_path), this constant only caps
# what collect()/save() re-embeds inline.
_MAX_INLINE_LOG_BYTES = 500_000


class CaptureMiopenLog:
    """Captures MIOpen's own solver-selection log for the duration of a
    `with` block. Identical technique to
    tools/bench_conv2d_fp16.py's own CaptureFd2 (duplicated rather than
    imported -- same independently-droppable reasoning every module in
    this package already follows for a small, self-contained helper):
    MIOpen logs from native C++ straight to file descriptor 2, so
    Python-level stderr reassignment (`sys.stderr = ...`) never sees it --
    this dup2's the real OS file descriptor instead, then restores it.

    Only produces anything useful if MIOPEN_LOGGING_ENV was already set
    (via plain os.environ.setdefault calls, BEFORE `import torch`) at the
    calling tool's own top -- see MIOPEN_LOGGING_ENV's own comment for why
    this class does not, and structurally cannot, set that up itself.

    Usage:

        with CaptureMiopenLog() as cap:
            ...run whatever conv/gemm calls should reach MIOpen...
        amd_tuned_torch_benchmark_report.save_miopen_log("sparse_conv", cap.text)
    """

    def __init__(self) -> None:
        self.text = ""

    def __enter__(self) -> "CaptureMiopenLog":
        self._tmp = tempfile.TemporaryFile(mode="w+b")
        self._saved = os.dup(2)
        sys.stderr.flush()
        os.dup2(self._tmp.fileno(), 2)
        return self

    def __exit__(self, *exc: Any) -> bool:
        sys.stderr.flush()
        os.dup2(self._saved, 2)
        os.close(self._saved)
        self._tmp.seek(0)
        self.text = self._tmp.read().decode("utf-8", "replace")
        self._tmp.close()
        return False


def miopen_log_path(domain: str, key: Optional[str] = None) -> str:
    return os.path.join(_REPORT_DIR, f"{key or device_key()}.{domain}.miopen.log")


def save_miopen_log(domain: str, text: str) -> str:
    """Persists one calibration domain's captured MIOpen log (see
    CaptureMiopenLog) to its own file under `miopen_log_path(domain)`, the
    same "its own persisted file, not just an in-memory value" reasoning
    every *_calibration module already applies to its measurements -- so a
    later save() (even one triggered by a DIFFERENT tool's run) still finds
    and re-embeds it instead of losing it the moment this process exits.
    `domain` can be any short identifier ("sparse_conv", "fftconv",
    "conv2d_fp16_investigation", ...) -- collect() discovers whichever
    `<device_key>.*.miopen.log` files exist rather than requiring `domain`
    to be a name it already knows about. Overwrites this domain's own
    prior log, if any; does not touch any other domain's."""
    os.makedirs(_REPORT_DIR, exist_ok=True)
    path = miopen_log_path(domain)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)
    return path


def _read_miopen_logs() -> Dict[str, str]:
    """{"sparse_conv": "<log text, possibly truncated>", ...} for every
    `<device_key>.<domain>.miopen.log` file save_miopen_log has written on
    this machine -- {} if none exist yet. A log past
    _MAX_INLINE_LOG_BYTES is truncated with a clear marker for the combined
    JSON; the file on disk (see miopen_log_path) always keeps the full
    text regardless."""
    key = device_key()
    prefix = os.path.join(_REPORT_DIR, f"{key}.")
    suffix = ".miopen.log"
    logs: Dict[str, str] = {}
    for path in glob.glob(f"{prefix}*{suffix}"):
        domain = os.path.basename(path)[len(f"{key}."):-len(suffix)]
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        if len(text) > _MAX_INLINE_LOG_BYTES:
            text = (text[:_MAX_INLINE_LOG_BYTES]
                    + f"\n... [truncated, {len(text):,} bytes total -- "
                      f"see {path} for the full log]")
        logs[domain] = text
    return logs


def gpu_pcie_info() -> Dict[str, Optional[str]]:
    """Best-effort PCIe link speed/width for the active GPU, read straight
    from Linux's sysfs (no ROCm tool required, works even without
    rocm-smi installed) -- e.g. {"link_speed": "16.0 GT/s PCIe",
    "link_width": "16"}. Both None (not a missing dict) on Windows, a
    CPU-only build, or if sysfs simply doesn't expose it -- this is
    genuinely best-effort, informational metadata, never something a
    benchmark should fail or block over.

    LIMITATION: matches whichever /sys/class/drm/card*/device happens to
    be the first one exposing current_link_speed/current_link_width --
    correct for the single-discrete-GPU machine this project targets, but
    not guaranteed to pick the right card on a genuine multi-GPU system
    (an iGPU + dGPU, or more than one dGPU)."""
    if not sys.platform.startswith("linux"):
        return {"link_speed": None, "link_width": None}
    for device_dir in sorted(glob.glob("/sys/class/drm/card*/device")):
        speed_path = os.path.join(device_dir, "current_link_speed")
        width_path = os.path.join(device_dir, "current_link_width")
        if not (os.path.isfile(speed_path) and os.path.isfile(width_path)):
            continue
        try:
            with open(speed_path) as f:
                speed = f.read().strip()
            with open(width_path) as f:
                width = f.read().strip()
            return {"link_speed": speed or None, "link_width": width or None}
        except OSError:
            continue
    return {"link_speed": None, "link_width": None}

# report key -> calibration module's basename (both the package-relative
# module name and its filename under this same directory). Each module must
# expose calibration_path() and device_key() with the same contract
# sparse_conv_calibration.py/fftconv_calibration.py already do -- adding a
# future *_calibration.py module to this package only needs one new entry
# here, nothing else in this file changes.
_CALIBRATION_MODULES = {
    "sparse_conv": "sparse_conv_calibration",
    "fftconv": "fftconv_calibration",
}


def device_key() -> str:
    """Identical construction to every *_calibration module's own
    device_key() (GPU model + torch + Python version, sanitized for a
    filename) -- duplicated rather than imported for the same
    independently-droppable reason those modules duplicate it from each
    other rather than from amd_tuned_torch.kernel_select."""
    import re

    try:
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        name = "unknown"
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    torch_ver = torch.__version__.replace("+", "_").replace("/", "_")
    return f"{safe_name}-torch-{torch_ver}-cp{sys.version_info.major}{sys.version_info.minor}"


def system_metadata() -> Dict[str, Optional[str]]:
    """GPU model, ROCm/HIP version, CUDA version, PyTorch version, Python
    version, and OS platform string -- everything needed to make sense of
    a calibration number measured on a DIFFERENT machine than the one
    reading this report. `rocm_version`/`cuda_version` are None on a CPU-only
    build (torch.version.hip/.cuda are always present as attributes, just
    None, on every torch build -- no try/except needed for those two)."""
    try:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        gpu_name = None
    return {
        "gpu_name": gpu_name,
        "rocm_version": torch.version.hip,
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "pcie": gpu_pcie_info(),
    }


def _import_calibration_module(basename: str):
    """`amd_tuned_torch.<basename>` if the full package is importable here,
    else the file loaded directly by path -- same fallback
    tools/benchmark_fftconv3d_min_positions.py/probe_fftconv_max_size.py use
    and for the same reason: a *_calibration module has no compiled-
    extension dependency of its own (plain json/os/torch.version reads), so
    it souldn't need a full native build importable just to read the plain
    JSON file it already wrote. Required here specifically because
    `importlib.import_module("amd_tuned_torch.X")` has to run
    amd_tuned_torch/__init__.py first (Python always imports a dotted
    submodule's parent package first) -- exactly the import that fails on a
    checkout with no compiled extension, which this module must tolerate
    since it's meant to be run from tools/export_benchmark_report.py in
    that same situation."""
    try:
        return importlib.import_module(f"amd_tuned_torch.{basename}")
    except ImportError:
        pass
    spec = importlib.util.spec_from_file_location(
        basename, Path(__file__).resolve().parent / f"{basename}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_raw_calibration(basename: str) -> Optional[Dict[str, Any]]:
    """The full on-disk JSON a *_calibration module has written for THIS
    device (extra_metadata sweep details included), or None if that
    module's file doesn't exist yet on this machine (never benchmarked) or
    is corrupt -- either just means that calibration domain is absent from
    the combined report, not an error."""
    module = _import_calibration_module(basename)
    try:
        with open(module.calibration_path(), "r") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


def collect() -> Dict[str, Any]:
    """Builds the combined report dict (see this module's own docstring
    for the exact top-level shape) without writing anything to disk --
    `save()` is this plus a write. Every entry in _CALIBRATION_MODULES that
    has no on-disk file yet for this device is simply omitted from the
    result, not present-with-nulls."""
    report: Dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "device_key": device_key(),
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
                         .isoformat(timespec="seconds"),
        "system": system_metadata(),
    }
    for report_key, basename in _CALIBRATION_MODULES.items():
        raw = _read_raw_calibration(basename)
        if raw is not None:
            report[report_key] = raw
    miopen_logs = _read_miopen_logs()
    if miopen_logs:
        report["miopen_logs"] = miopen_logs
    return report


def report_path(key: Optional[str] = None) -> str:
    return os.path.join(_REPORT_DIR, f"{key or device_key()}.json")


def save() -> str:
    """Collects everything (see collect()) and writes it to report_path(),
    via a temp file + os.replace so a process killed mid-write never
    leaves a truncated/corrupt file for the next reader -- same
    crash-safety convention as every *_calibration module's own save()
    and amd_tuned_torch.kernel_select's disk cache. Returns the path
    written to. Re-running this after a calibration module gets a new
    measurement simply overwrites the combined file with the current
    on-disk state of everything -- there is no partial-merge here the way
    a calibration module's own save() merges onto its prior content,
    because this file is a full re-export every time, never hand-edited."""
    report = collect()
    path = report_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, path)
    return path
