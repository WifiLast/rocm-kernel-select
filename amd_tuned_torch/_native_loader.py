"""Locate the compiled extension for the PyTorch that is actually running.

WHY THIS EXISTS. An editable install is the right way to develop this
package -- edit amd_tuned_torch/*.py and every environment that installed it
sees the change immediately, with no reinstall and no second copy to drift.
But an editable install points every environment at ONE directory, and a
compiled extension is not shareable the way a .py file is: `_native.so`
links a specific libtorch, and loading it into a different PyTorch is an
ABI mismatch that usually ends in a segfault rather than an ImportError.

This machine has exactly that: two environments on one source tree.

    py310_amd   torch 2.15.0.dev+rocm7.2
    SD WebUI    torch 2.13.0+rocm7.1

Whichever one built last would leave its `.so` beside __init__.py and the
other would load it. So the Python stays shared and editable, and the
binary gets a directory per environment:

    amd_tuned_torch/_native_builds/torch-2.13.0_rocm7.1-cp310/_native...so
    amd_tuned_torch/_native_builds/torch-2.15.0.dev20260820_rocm7.2-cp310/...

setup.py deposits each build into the directory for the torch it was built
against (see CachedBuildExtension), and this module picks the matching one
at import.

FALLBACK. A plain `_native*.so` sitting next to __init__.py is still used
when no keyed build matches -- that is what a single-environment checkout
looks like after `pip install -e .`, and it must keep working untouched.
The cost is that the fallback cannot be verified: nothing here can tell
which libtorch an arbitrary .so was linked against without loading it, and
loading it is the dangerous act. So the fallback is used only when the
keyed lookup finds nothing, and _describe() reports which path was taken
so a confusing crash has somewhere to start.

MULTIPLE EXTENSIONS. amd_tuned_torch._native is no longer the only compiled
extension -- Composable Kernel and hipBLASLt each got split into their own
(amd_tuned_torch._native_ck / ._native_hipblaslt, see setup.py), so every
function here takes a `module_name` parameter instead of a hardcoded
constant. `_find` matches the exact stem (`_native_ck.cpython-....so`'s stem
is `_native_ck`), not a prefix -- a naive `startswith("_native")` would also
match `_native_ck...` and `_native_hipblaslt...` when looking for the core
`_native` module, since they share that prefix.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Optional

_BUILDS_DIRNAME = "_native_builds"
_DEFAULT_MODULE_NAME = "_native"


def build_key(torch_version: Optional[str] = None) -> str:
    """Directory name identifying an ABI-compatible build.

    torch.__version__ carries the ROCm suffix ('2.15.0.dev+rocm7.2'), which
    is what makes this a usable ABI key: a different ROCm build of the same
    torch release is still a different libtorch. The Python tag is included
    because the extension filename already encodes it and a mismatch there
    is a different failure worth keeping distinct.
    """
    if torch_version is None:
        import torch
        torch_version = torch.__version__
    safe = torch_version.replace("+", "_").replace("/", "_").replace(os.sep, "_")
    return f"torch-{safe}-cp{sys.version_info.major}{sys.version_info.minor}"


def _find(directory: str, module_name: str) -> Optional[str]:
    if not os.path.isdir(directory):
        return None
    for name in sorted(os.listdir(directory)):
        # Exact stem match, not startswith: "_native_ck...so" and
        # "_native_hipblaslt...so" both start with "_native" too, and would
        # otherwise be mistaken for the core module (see module docstring).
        stem = name.split(".", 1)[0]
        if stem == module_name and name.endswith((".so", ".pyd")):
            return os.path.join(directory, name)
    return None


def locate(package_dir: str, module_name: str = _DEFAULT_MODULE_NAME) -> tuple[Optional[str], str]:
    """(path to the extension, how it was chosen)."""
    keyed = os.path.join(package_dir, _BUILDS_DIRNAME, build_key())
    found = _find(keyed, module_name)
    if found:
        return found, f"matched this torch ({build_key()})"
    found = _find(package_dir, module_name)
    if found:
        return found, "fallback: unkeyed build beside __init__.py"
    return None, "no build found"


def load(package_name: str, package_dir: str, module_name: str = _DEFAULT_MODULE_NAME,
         required: bool = True):
    """Import the extension as `<package>.<module_name>` from wherever it lives.

    Registered in sys.modules under the canonical name so the rest of the
    package's `from . import _native as _C` (or `_native_ck`/
    `_native_hipblaslt`) keeps working unchanged.

    `required=False` is for the optional CK/hipBLASLt extensions: unlike the
    core `_native` extension (hard-required at import time, same as always),
    a build predating this module's split, or a partial rebuild that never
    ran for these two, should degrade to "tier unavailable" -- the same way
    aiter/TransformerEngine being uninstalled does -- rather than making
    `import amd_tuned_torch` itself fail. Returns None instead of raising in
    that case.
    """
    qualified = f"{package_name}.{module_name}"
    if qualified in sys.modules:
        return sys.modules[qualified]

    path, how = locate(package_dir, module_name)
    if path is None:
        if not required:
            return None
        import torch
        raise ImportError(
            f"amd_tuned_torch native extension '{module_name}' not built for torch "
            f"{torch.__version__}.\n"
            f"  Looked for: {os.path.join(package_dir, _BUILDS_DIRNAME, build_key())}/\n"
            f"          and: {package_dir}/{module_name}*.so\n"
            f"  Build it with: pip install -e . --no-build-isolation"
        )

    spec = importlib.util.spec_from_file_location(qualified, path)
    if spec is None or spec.loader is None:
        if not required:
            return None
        raise ImportError(f"could not load the native extension from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # A failed load must not leave a half-initialised module behind for
        # the next importer to find and believe.
        sys.modules.pop(qualified, None)
        if not required:
            return None
        raise
    module.__amd_tuned_torch_origin__ = (path, how)
    return module


def describe(module_name: str = _DEFAULT_MODULE_NAME) -> str:
    """Which extension is loaded and why -- for bug reports and confusion."""
    mod = sys.modules.get(f"amd_tuned_torch.{module_name}")
    origin = getattr(mod, "__amd_tuned_torch_origin__", None)
    if origin is None:
        return f"amd_tuned_torch.{module_name}: not loaded"
    path, how = origin
    return f"amd_tuned_torch.{module_name}: {path}\n  ({how})"
