"""Combines every calibration file this package's benchmark tools have
written on this machine into one JSON report, ready to hand off (upload,
attach to an issue, etc.) -- see amd_tuned_torch/benchmark_report.py's own
module docstring for exactly what goes into it and why it's a separate
export step rather than a change to how any individual calibration module
loads/saves.

Both tools/benchmark_sparse_conv.py and
tools/benchmark_fftconv3d_min_positions.py already call
benchmark_report.save() automatically at the end of a successful run, so
you only need to run THIS script by hand to re-combine on demand -- e.g.
after hand-editing a calibration file, after copying calibration files from
another machine into this one's calibration directories (which would be
unusual and not really what those per-device files are for, but nothing
stops it), or simply to regenerate the combined file without re-running
any benchmark.

Run with:

    python tools/export_benchmark_report.py
    python tools/export_benchmark_report.py --print          # also print the JSON to stdout
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def _load_benchmark_report():
    try:
        from amd_tuned_torch import benchmark_report
        return benchmark_report
    except ImportError:
        # No compiled-extension dependency here either (plain json/platform/
        # torch.version.* reads) -- same fallback
        # tools/probe_fftconv_max_size.py/benchmark_fftconv3d_min_positions.py
        # use so this doesn't need a full native build just to combine
        # already-written calibration files.
        spec = importlib.util.spec_from_file_location(
            "benchmark_report",
            Path(__file__).resolve().parent.parent / "amd_tuned_torch" / "benchmark_report.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--print", dest="print_json", action="store_true",
                         help="Also print the combined report to stdout.")
    args = parser.parse_args()

    benchmark_report = _load_benchmark_report()
    report = benchmark_report.collect()
    non_domain_keys = ("schema_version", "device_key", "generated_at", "system", "miopen_logs")
    found = [k for k in report if k not in non_domain_keys]

    path = benchmark_report.save()
    print(f"Device: {report['device_key']}")
    print(f"System: {report['system']}")
    if found:
        print(f"Calibration domains found: {', '.join(found)}")
    else:
        print("No calibration files found yet on this machine -- run "
              "tools/benchmark_sparse_conv.py and/or "
              "tools/benchmark_fftconv3d_min_positions.py first. "
              "Writing an (empty-of-results) report anyway, with system metadata only.")
    if "miopen_logs" in report:
        print(f"MIOpen debug logs found for: {', '.join(report['miopen_logs'])}")
    print(f"Saved combined report to {path}")

    if args.print_json:
        print()
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
