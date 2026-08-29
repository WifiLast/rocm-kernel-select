"""Optional Hugging Face Hub kernel backend for amd_tuned_torch, using the
`kernels` package (source/kernel/kernels) to download precompiled kernels
from the Hub at runtime -- a third kernel source alongside aiter
(source/aiter) and TransformerEngine's ROCm fork (source/TransformerEngine),
both of which are local builds from source in this repo.

Disabled by default: set AMD_TUNED_TORCH_ENABLE_HUB_KERNELS=1 to opt in, checked
once at import time (same posture as AMD_TUNED_TORCH_ENABLE_TE).

Unlike aiter and TE (both local Python imports -- fast even when they
fail), fetching a Hub kernel is a NETWORK CALL the first time a given
repo/version is used (cached locally afterward via the same
huggingface_hub cache model downloads use). So nothing here ever runs at
amd_tuned_torch import time, even when the env var is on -- every fetch is lazy,
triggered only the first time the corresponding op is actually invoked,
and get_hub_kernel() catches every exception broadly (not just
RuntimeError/ImportError -- a network failure, a missing Hub variant for
the installed torch/ROCm build, an auth error for a gated repo, and a
generic huggingface_hub error all surface as different exception types),
returning None instead of raising. A slow or absent connection must never
crash an inference call.

Not every Hub kernel repo actually has a ROCm build, despite kernel-
builder's build system supporting ROCm variants in the abstract --
`kernels-community/activation` (the obvious first candidate, and what an
earlier version of this module wired up as an F.gelu backend) was tried
here first and dropped after checking its published file tree: every
single variant is `torch2xx-cxx11-cuXXX-*` or `torch2xx-metal-*` -- zero
`rocm` variants exist for it. It would have silently never engaged on
RDNA3, just fallen through to stock every time.

kernels-community/aiter-kernels was the other candidate checked here: a
real ROCm-native repackaging (confirmed `torch-rocm` build variant) of the
*same* aiter project this amd_tuned_torch build already depends on locally
(source/aiter) -- its activation.fused_silu_mul turned out to be the
exact same function as aiter.ops.triton.activation.fused_silu_mul,
already importable locally with zero network dependency. So it's exposed
as amd_tuned_torch.aiter_ops.fused_silu_mul instead (see that module), not fetched
from the Hub at all -- no reason to pay a network round-trip for
something already sitting in this repo's own local aiter dependency.

Nothing is currently wired up as a concrete backend in this module as a
result: it's left as a generic, tested utility (available()/
get_hub_kernel()) for a future Hub-only kernel that doesn't already have
a local equivalent, rather than force a redundant example just to have
one.
"""
from __future__ import annotations

import os
import threading
from typing import Optional

try:
    import kernels as _hub_kernels

    _KERNELS_PACKAGE_AVAILABLE = True
except ImportError:
    _hub_kernels = None
    _KERNELS_PACKAGE_AVAILABLE = False


def _enabled_env() -> bool:
    return os.environ.get("AMD_TUNED_TORCH_ENABLE_HUB_KERNELS", "0") not in ("0", "", "false", "False")


_ENABLED = _enabled_env()

_kernel_cache: dict = {}
_failed_repos: set = set()
_cache_lock = threading.Lock()


def available() -> bool:
    """True if the `kernels` package is importable AND
    AMD_TUNED_TORCH_ENABLE_HUB_KERNELS is set. Does not mean any particular fetch
    will succeed -- that's a per-repo network call, attempted lazily and
    cached (success or failure) by get_hub_kernel()."""
    return _KERNELS_PACKAGE_AVAILABLE and _ENABLED


def get_hub_kernel(
    repo_id: str, version: Optional[int] = None, revision: Optional[str] = None
):
    """Fetch (and cache) a kernel module from the Hugging Face Hub via
    kernels.get_kernel(). Returns None on ANY failure -- network, missing
    variant for the installed torch/ROCm build, auth, etc. -- instead of
    raising; callers should treat None as "not available, fall back."

    A successful fetch is cached in-process for (repo_id, version,
    revision) so only the first call pays the network cost. A failed
    fetch is also cached (as a failure) so a repeatedly-called op with no
    Hub kernel available doesn't retry the network on every single call --
    restart the process (or clear amd_tuned_torch.hub_ops._failed_repos yourself)
    to retry.
    """
    if not available():
        return None
    key = (repo_id, version, revision)
    with _cache_lock:
        if key in _kernel_cache:
            return _kernel_cache[key]
        if key in _failed_repos:
            return None
    try:
        module = _hub_kernels.get_kernel(repo_id, version=version, revision=revision)
    except Exception:
        # Intentionally broad: huggingface_hub/requests raise many
        # different exception types for "no internet", "repo not found",
        # "no variant for this torch/ROCm build", "gated repo, no auth",
        # etc. -- none of them may crash a caller that would otherwise
        # just fall back to TE/stock.
        with _cache_lock:
            _failed_repos.add(key)
        return None
    with _cache_lock:
        _kernel_cache[key] = module
    return module
