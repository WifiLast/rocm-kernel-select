"""Warns about ROCm/HIP/MIOpen/rocBLAS/PyTorch environment variables that,
individually often harmless (usually left on from debugging, or copied from
a config that made sense for a different problem), COMPOUND into a real
performance regression once combined with each other or with how this
package actually uses the GPU.

This is advisory only -- it never changes torch/ROCm behavior itself
(unlike every other module amd_tuned_torch/__init__.py installs), just
reads os.environ and this package's own tier-availability state and prints
a warning when a known-bad combination is present. Nothing here is
benchmarked on RX 7900 XTX specifically -- these are documented ROCm/HIP/
MIOpen/rocBLAS/PyTorch behaviors (see each rule's docstring for what it's
based on), not measurements from this project's own benchmark.json. Treat
the mechanism explanations as reliable and the "how much slower" framing as
qualitative, not a number to cite.

Runs once at import time (unconditionally -- unlike the patches
enable()/miopen_fallback.py/etc. install, this never touches torch or ROCm
state, so there is no AMD_TUNED_TORCH_AUTOPATCH-shaped reason to gate it).
Set AMD_TUNED_TORCH_ROCM_ENV_CHECK=0 to skip it entirely (e.g. if you've
already read the warnings once and don't want them repeated on every
process start of a script you run often).

Add a new rule by appending a `_Rule` to `_RULES` -- each is a pure
predicate over (env, tiers) so this stays testable without any real ROCm
install or GPU (see tests/test_rocm_env_check.py).
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional


class Finding(NamedTuple):
    name: str
    severity: str  # "critical" | "warning"
    message: str


def _truthy(value: Optional[str]) -> bool:
    """Same convention this package uses everywhere else (cache.py,
    te_ops.py, hub_ops.py): unset or "0"/""/"false"/"False" is off, anything
    else is on. ROCm/MIOpen/rocBLAS themselves are less consistent about
    this (some treat any non-"0" digit as a mode selector, not a bool), so
    individual rules re-check the exact value where that distinction
    matters instead of assuming this helper's notion of "on"."""
    return (value or "0") not in ("0", "", "false", "False")


@dataclass(frozen=True)
class _Env:
    """A thin, mockable wrapper over os.environ so rules and tests share
    one lookup path instead of each rule calling os.environ.get directly."""
    data: dict

    def get(self, name: str) -> Optional[str]:
        return self.data.get(name)

    def truthy(self, name: str) -> bool:
        return _truthy(self.get(name))


@dataclass(frozen=True)
class _Tiers:
    """This package's own tier-availability state -- lazily computed (see
    `_tiers()` below) so importing this module never forces aiter_ops/
    ck_gemm_ops/hipblaslt_ops to be imported before amd_tuned_torch/
    __init__.py has actually gotten around to importing them itself."""
    aiter: bool
    ck_gemm: bool
    hipblaslt: bool
    autopatch: bool


def _tiers() -> _Tiers:
    from . import aiter_ops, ck_gemm_ops, hipblaslt_ops

    return _Tiers(
        aiter=aiter_ops.available(),
        ck_gemm=ck_gemm_ops.available(),
        hipblaslt=hipblaslt_ops.available(),
        autopatch=_truthy(os.environ.get("AMD_TUNED_TORCH_AUTOPATCH", "1")),
    )


class _Rule(NamedTuple):
    name: str
    severity: str
    predicate: Callable[[_Env, _Tiers], bool]
    # A plain str for every rule whose advice doesn't depend on a runtime
    # value; a callable(env, tiers) -> str for the rare rule (see
    # hsa_override_gfx_version below) that needs to report an actual
    # detected value rather than talk about the possibility in the
    # abstract -- resolved in `check()`.
    message: str | Callable[[_Env, _Tiers], str]


def _is_writable_dir(path: str) -> bool:
    """Best-effort: a path that doesn't exist yet is treated as writable if
    its parent is (MIOpen creates the file itself on first write) -- only a
    path that unambiguously can't be written to should trigger the rule,
    never a false positive from a directory MIOpen just hasn't created yet."""
    try:
        target = path if os.path.isdir(path) else os.path.dirname(path) or "."
        return os.access(target, os.W_OK)
    except OSError:
        return False


def _bad_miopen_db_paths(env: "_Env") -> "list[tuple[str, str]]":
    """(env var name, path) for every MIOPEN_*_DB_PATH that's set but not
    writable -- one place computing this so the predicate and the message
    can never disagree about which variable is actually the problem."""
    bad = []
    for name in ("MIOPEN_USER_DB_PATH", "MIOPEN_SYSTEM_DB_PATH"):
        path = env.get(name)
        if path and not _is_writable_dir(path):
            bad.append((name, path))
    return bad


def _detected_gcn_arch() -> Optional[str]:
    """Best-effort torch-reported gfxNNNN for device 0, or None if no GPU
    is visible right now -- never raises, since a CPU-only environment (a
    dev machine, this test suite, WSL without a GPU passed through) must
    not turn a missing GPU into an import-time failure."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:
        return None


_RULES: "list[_Rule]" = [
    _Rule(
        name="hip_launch_blocking",
        severity="critical",
        predicate=lambda env, tiers: env.truthy("HIP_LAUNCH_BLOCKING") and tiers.autopatch,
        message=(
            "HIP_LAUNCH_BLOCKING=1 forces every HIP kernel launch to be "
            "synchronous (HIP's equivalent of CUDA_LAUNCH_BLOCKING) -- it "
            "serializes the launch queue that amd_tuned_torch's patched "
            "F.linear/matmul/bmm/conv2d/conv3d/attention calls all rely on "
            "overlapping. Combined with amd_tuned_torch actively patching "
            "those hot-path ops (AMD_TUNED_TORCH_AUTOPATCH is on), every "
            "one of them now pays a host-device round trip it wouldn't "
            "otherwise need. This is almost always a debugging flag left "
            "on by accident (e.g. to get a clean stack trace at the exact "
            "failing kernel) -- unset it for anything performance-sensitive."
        ),
    ),
    _Rule(
        name="miopen_find_enforce_search",
        severity="critical",
        predicate=lambda env, tiers: (
            (env.get("MIOPEN_FIND_ENFORCE") or "") in ("2", "3", "4", "SEARCH", "SEARCH_DB_UPDATE")
            and _truthy(os.environ.get("AMD_TUNED_TORCH_MEASURE_KERNELS",
                                        os.environ.get("AMD_TUNED_TORCH_CONV_MEASURE", "1")))
        ),
        message=(
            "MIOPEN_FIND_ENFORCE=SEARCH{,_DB_UPDATE} makes MIOpen bypass "
            "its find-db and run a live algorithm search on every find() "
            "call, REGARDLESS of whether MIOPEN_DEBUG_DISABLE_FIND_DB is "
            "also set or whether a cached winner already exists -- SEARCH "
            "mode ignores the cache for the purpose of deciding, not just "
            "for persisting. In this project specifically that's worse "
            "than it sounds: amd_tuned_torch's own kernel_select contest "
            "(enabled by default) calls the 'stock' candidate for "
            "conv2d/conv3d/group_norm several times per shape (warmup + "
            "timed iterations) to measure it against the native/CK "
            "tiers -- with MIOpen re-searching on every one of those "
            "calls, 'stock' measures as artificially, catastrophically "
            "slow, which can make kernel_select wrongly avoid stock even "
            "on shapes where MIOpen's un-search-forced kernel would "
            "actually win (see this project's own conv2d fp16 case: MIOpen "
            "beats every one of amd_tuned_torch's own kernels there via a "
            "hand-written Winograd solver). Also present, and additionally "
            "irreversible, if MIOPEN_DEBUG_DISABLE_FIND_DB=1 is set "
            "alongside this: nothing from any of those searches is ever "
            "persisted either, so every future process pays full search "
            "cost again too. Unset MIOPEN_FIND_ENFORCE (or set it to NONE) "
            "unless you are actively re-tuning MIOpen's find-db on purpose."
        ),
    ),
    _Rule(
        name="rocblas_logging_with_kernel_select",
        severity="warning",
        predicate=lambda env, tiers: (
            (env.get("ROCBLAS_LAYER") or "0") not in ("0", "")
            and _truthy(os.environ.get("AMD_TUNED_TORCH_MEASURE_KERNELS",
                                        os.environ.get("AMD_TUNED_TORCH_CONV_MEASURE", "1")))
        ),
        message=(
            "ROCBLAS_LAYER is set (rocBLAS call logging/tracing/profiling) "
            "while amd_tuned_torch's kernel_select contest is enabled "
            "(default). kernel_select times stock rocBLAS as one of the "
            "candidates for F.linear/torch.bmm on its first call per shape "
            "-- rocBLAS logging adds per-call overhead to every logged "
            "call, including the ones kernel_select is trying to measure, "
            "which can bias the contest toward hipBLASLt/CK GEMM/aiter "
            "even on shapes where plain rocBLAS would actually win once "
            "logging is off. It also just makes every rocBLAS call in the "
            "process slower and noisier, independent of this package. "
            "Unset it unless you're actively debugging rocBLAS dispatch."
        ),
    ),
    _Rule(
        name="verbose_runtime_logging_with_autopatch",
        severity="warning",
        predicate=lambda env, tiers: (
            (env.get("AMD_LOG_LEVEL") or "0") not in ("0", "", "1")
            and tiers.autopatch
        ),
        message=(
            "AMD_LOG_LEVEL is set to a verbose level (2=warning and up "
            "print/log on every HIP runtime call) while amd_tuned_torch is "
            "actively patching F.linear/matmul/bmm/conv2d/conv3d/"
            "attention -- a model forward pass through this package makes "
            "many more HIP calls per step than an unpatched one in some "
            "cases (e.g. kernel_select's own per-candidate timing calls on "
            "a shape's first occurrence), each one now paying a logging "
            "write. Fine for a one-off diagnosis; leave it unset for "
            "anything you're timing or running at scale."
        ),
    ),
    _Rule(
        name="kernel_select_disabled_wastes_compiled_tiers",
        severity="warning",
        predicate=lambda env, tiers: (
            not _truthy(os.environ.get("AMD_TUNED_TORCH_MEASURE_KERNELS",
                                        os.environ.get("AMD_TUNED_TORCH_CONV_MEASURE", "1")))
            and (tiers.ck_gemm or tiers.hipblaslt)
        ),
        message=(
            "AMD_TUNED_TORCH_MEASURE_KERNELS=0 (or AMD_TUNED_TORCH_CONV_MEASURE=0) is "
            "set, but this build has the CK GEMM and/or hipBLASLt tier(s) "
            "compiled in (see amd_tuned_torch.native_build_info()). "
            "_patched_linear/_patched_bmm only route through hipBLASLt/CK "
            "GEMM via the kernel_select contest -- with it disabled they "
            "skip straight to aiter-or-stock, so a real, already-built, "
            "measured-faster-on-many-shapes GEMM tier sits unused for no "
            "reason. Only disable kernel_select if you specifically need "
            "amd_tuned_torch's old fixed-tier-order behavior (e.g. to "
            "reproduce a specific baseline for A/B comparison)."
        ),
    ),
    _Rule(
        name="tunableop_numerical_check",
        severity="critical",
        predicate=lambda env, tiers: (
            env.truthy("PYTORCH_TUNABLEOP_ENABLED")
            and env.truthy("PYTORCH_TUNABLEOP_NUMERICAL_CHECK")
        ),
        message=(
            "PYTORCH_TUNABLEOP_ENABLED=1 together with "
            "PYTORCH_TUNABLEOP_NUMERICAL_CHECK=1 makes PyTorch's TunableOp "
            "validate every tuned GEMM's result against a reference "
            "implementation on every call it tunes, not just once -- "
            "documented PyTorch behavior, and a large, ongoing overhead "
            "rather than a one-time tuning cost. NUMERICAL_CHECK exists "
            "for debugging a suspected TunableOp correctness issue; leave "
            "it off (the default) for anything performance-sensitive, "
            "including this package's own kernel_select contest running "
            "underneath it."
        ),
    ),
    _Rule(
        name="sdma_disabled_with_similarity_cache",
        severity="warning",
        predicate=lambda env, tiers: (
            (env.get("HSA_ENABLE_SDMA") or "1") == "0"
            and _truthy(os.environ.get("AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE"))
        ),
        message=(
            "HSA_ENABLE_SDMA=0 disables the dedicated DMA copy engine, "
            "routing every host<->device transfer through the compute "
            "queue instead -- normally a workaround for a specific "
            "multi-GPU peer-access bug, not a general-purpose setting. "
            "amd_tuned_torch.cache's SimilarityCache (AMD_TUNED_TORCH_ENABLE_"
            "SIMILARITY_CACHE=1) reads back a small comparison result from "
            "device to host on every wrapped call to decide whether to "
            "skip -- with SDMA off, each of those reads now contends with "
            "whatever compute kernel is also using that queue instead of "
            "overlapping for free on the copy engine. Only disable SDMA if "
            "you have the specific bug it works around; otherwise leave it "
            "on."
        ),
    ),
    _Rule(
        name="hsa_override_gfx_version",
        severity="warning",
        predicate=lambda env, tiers: env.get("HSA_OVERRIDE_GFX_VERSION") is not None,
        message=lambda env, tiers: (
            f"HSA_OVERRIDE_GFX_VERSION={env.get('HSA_OVERRIDE_GFX_VERSION')} is set -- "
            "a common workaround to get ROCm to accept an unsupported/consumer card, "
            "but every rocBLAS/MIOpen/hipBLASLt call now believes it's running on "
            "whatever architecture that value maps to, which may not be the "
            f"gfx1100 (RDNA3) this package's kernels were compiled for"
            + (f" (torch currently reports the real device as "
               f"{_detected_gcn_arch()})" if _detected_gcn_arch() else "")
            + ". A mismatch doesn't necessarily crash -- it can silently fall back "
            "to generic/reference kernels instead of the tuned navi31 logic "
            "hipBLASLt/CK's WMMA instances exist specifically to use, which is "
            "exactly the kind of regression that's invisible in the output (every "
            "candidate returns a numerically equivalent tensor) and only shows up "
            "as unexplained slowness. Verify this override is still actually "
            "needed for your card and, if so, that it maps to gfx1100 -- remove it "
            "entirely if it's leftover from testing a different GPU."
        ),
    ),
    _Rule(
        name="hsa_enable_interrupt_disabled",
        severity="warning",
        predicate=lambda env, tiers: (env.get("HSA_ENABLE_INTERRUPT") or "1") == "0" and tiers.autopatch,
        message=(
            "HSA_ENABLE_INTERRUPT=0 forces busy-wait polling for kernel "
            "completion instead of interrupt-driven signaling -- trades "
            "CPU cycles (a full core spinning) for potentially lower "
            "latency on some workloads, but combined with amd_tuned_torch "
            "actively patching many hot-path ops (AUTOPATCH on), the "
            "extra CPU contention from that spin can start competing with "
            "whatever else the process is doing on the host side (data "
            "loading, kernel_select's own per-shape timing loop, Python "
            "dispatch overhead) rather than being free. Only set this if "
            "you've specifically measured a latency win for your workload; "
            "it's not a general-purpose speedup."
        ),
    ),
    _Rule(
        name="amd_serialize_kernel",
        severity="critical",
        predicate=lambda env, tiers: (env.get("AMD_SERIALIZE_KERNEL") or "0") != "0" and tiers.autopatch,
        message=(
            "AMD_SERIALIZE_KERNEL is set to a nonzero value -- a lesser-known "
            "sibling of HIP_LAUNCH_BLOCKING that also forces every kernel "
            "launch to be synchronous (some values additionally print "
            "before/after each one). Same compounding effect as "
            "HIP_LAUNCH_BLOCKING=1 with amd_tuned_torch actively patching "
            "F.linear/matmul/bmm/conv2d/conv3d/attention: every one of "
            "those now pays a full host-device sync it wouldn't otherwise "
            "need. Almost always a leftover from debugging a specific "
            "kernel crash (getting a clean stack trace at the exact "
            "failing launch) -- unset it for anything performance-sensitive."
        ),
    ),
    _Rule(
        name="gpu_max_hw_queues_serialized",
        severity="warning",
        predicate=lambda env, tiers: (env.get("GPU_MAX_HW_QUEUES") or "") == "1" and tiers.autopatch,
        message=(
            "GPU_MAX_HW_QUEUES=1 limits the GPU to a single hardware queue, "
            "so concurrent streams serialize onto it instead of running "
            "side by side. amd_tuned_torch's kernel_select contest "
            "deliberately runs each candidate's warmup+timed iterations "
            "back to back on the current stream to measure them (see "
            "kernel_select.py) -- with only one HW queue available, any "
            "overlap those launches could otherwise get from the GPU's "
            "scheduler is unavailable, and any part of your own pipeline "
            "that relies on multiple streams overlapping loses that "
            "benefit too. Only set this if you have a specific reason "
            "(some multi-process GPU-sharing setups need it); it's not a "
            "default worth copying between machines."
        ),
    ),
    _Rule(
        name="hip_memory_caching_disabled",
        severity="critical",
        predicate=lambda env, tiers: (
            env.truthy("PYTORCH_NO_HIP_MEMORY_CACHING")
            and (_truthy(os.environ.get("AMD_TUNED_TORCH_MEASURE_KERNELS",
                                         os.environ.get("AMD_TUNED_TORCH_CONV_MEASURE", "1")))
                 or _truthy(os.environ.get("AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE")))
        ),
        message=(
            "PYTORCH_NO_HIP_MEMORY_CACHING=1 disables PyTorch's caching "
            "allocator, so every tensor allocation round-trips the ROCm "
            "driver instead of reusing a pooled block. amd_tuned_torch's "
            "kernel_select contest allocates fresh scratch/output tensors "
            "for every candidate's warmup+timed iterations on a shape's "
            "first occurrence, and/or SimilarityCache allocates small "
            "comparison tensors on every wrapped call -- both patterns are "
            "exactly what the caching allocator exists to make cheap. "
            "This flag is normally a memory-debugging tool (finding a "
            "leak, isolating a corruption), not something to leave on "
            "during normal operation."
        ),
    ),
    _Rule(
        name="miopen_db_path_not_writable",
        severity="critical",
        predicate=lambda env, tiers: bool(_bad_miopen_db_paths(env)),
        message=lambda env, tiers: (
            ", ".join(f"{name}={path}" for name, path in _bad_miopen_db_paths(env))
            + " points at a location this process can't write to. "
            "This has the SAME effect as MIOPEN_DEBUG_DISABLE_FIND_DB=1 "
            "even though that flag isn't set: MIOpen can still search for "
            "the best algorithm, but can never persist the result, so "
            "every process -- and every distinct conv shape within a "
            "process, if MIOPEN_FIND_ENFORCE also forces search -- pays "
            "full search cost again. Common cause: a container overlay "
            "filesystem, a read-only bind mount, or a tmpfs that gets "
            "wiped between runs. Point it at a real, persistent, "
            "writable directory."
        ),
    ),
    _Rule(
        name="rocm_profiler_attached",
        severity="warning",
        predicate=lambda env, tiers: (
            any(env.get(name) for name in
                ("HSA_TOOLS_LIB", "ROCPROFILER_METRICS_PATH", "HIP_TRACE_API"))
            and tiers.autopatch
        ),
        message=(
            "A ROCm profiling/tracing hook (HSA_TOOLS_LIB, "
            "ROCPROFILER_METRICS_PATH, or HIP_TRACE_API) is set while "
            "amd_tuned_torch is actively patching hot-path ops. These "
            "instrument every HIP call for collection, which is the point "
            "during a dedicated profiling run but a per-call tax left on "
            "by accident during normal use -- same compounding effect as "
            "the AMD_LOG_LEVEL/ROCBLAS_LAYER rules above, just from "
            "tooling rather than a logging verbosity setting. Unset these "
            "once the profiling session is done."
        ),
    ),
]


def check(warn: bool = True) -> "list[Finding]":
    """Evaluate every rule against the current environment and this
    package's own tier state. Returns every triggered Finding regardless of
    `warn`; when `warn` is True (the default) each is also emitted via
    warnings.warn so it surfaces the same way every other advisory in this
    package does (see e.g. enable_flash_attn_rocwmma's unavailable-tier
    warning)."""
    env = _Env(dict(os.environ))
    tiers = _tiers()
    findings = []
    for rule in _RULES:
        try:
            triggered = rule.predicate(env, tiers)
        except Exception:
            # A rule that can't evaluate (e.g. a future one that reaches
            # into a tier module not yet imported) must never break
            # `import amd_tuned_torch` -- same policy as every other
            # optional-dependency probe in this package.
            continue
        if triggered:
            msg = rule.message(env, tiers) if callable(rule.message) else rule.message
            finding = Finding(rule.name, rule.severity, msg)
            findings.append(finding)
            if warn:
                warnings.warn(
                    f"amd_tuned_torch.rocm_env_check [{finding.severity}] "
                    f"{finding.name}: {finding.message}",
                    UserWarning,
                    stacklevel=2,
                )
    return findings


if os.environ.get("AMD_TUNED_TORCH_ROCM_ENV_CHECK", "1") != "0":
    check()
