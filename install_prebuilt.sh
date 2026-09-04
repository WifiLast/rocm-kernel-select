#!/bin/bash
# Registers amd_tuned_torch as importable in the CURRENT active Python
# environment using an already-compiled build from
# amd_tuned_torch/_native_builds/<torch-version-key>/ -- no C++/HIP
# compilation, no setup.py, no pip build_ext invocation at all.
#
# WHY THIS EXISTS. `pip install -e .` always drives setuptools' build_ext,
# which always attempts to compile the ext_modules from source -- it has no
# awareness of _native_builds/ and will not skip compiling just because a
# matching .so is already deposited there (see setup.py's BUILD CACHE
# section and amd_tuned_torch/_native_loader.py's module docstring for the
# full mechanism). Since 2026-09-04's editable-wheel investigation, a plain
# `pip install -e .` also can't reuse setup.py's own pinned build cache
# either (it hands build_ext a fresh random temp directory every
# invocation) -- so re-running it when a matching prebuilt set already
# exists is pure waste: minutes to tens of minutes recompiling Composable
# Kernel for no reason.
#
# What actually needs to happen to USE an existing build is much smaller
# than a real install: amd_tuned_torch just needs to be importable (on
# sys.path), and amd_tuned_torch/_native_loader.py already finds and loads
# whichever _native*.so under _native_builds/ matches the running
# torch.__version__ + Python version, entirely at import time -- see
# amd_tuned_torch/__init__.py's `_C = _native_loader.load(...)` calls. So
# this script does the "on sys.path" part the cheap way: a .pth file in the
# active environment's site-packages pointing at this directory, the same
# mechanism `pip install -e .` used before PEP 660 (and still uses
# internally in "compat" mode) -- no compiler ever runs.
#
# Usage:
#   ./install_prebuilt.sh              # use whichever `python` is on PATH
#   ./install_prebuilt.sh /path/to/venv/bin/python
#
# Exits non-zero with a clear message (and does NOT fall back to building)
# if no build matching the active torch/Python exists yet -- run
# `AMD_TUNED_TORCH_GPU_ARCH=gfx1100 python setup.py build_ext --inplace`
# once first in that case; see README.md.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${1:-python}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "install_prebuilt.sh: '$PYTHON' not found on PATH" >&2
    exit 1
fi

# Mirrors amd_tuned_torch/_native_loader.py's build_key() exactly -- this
# script and that module MUST compute the same key from the same
# torch.__version__/Python version, or this script would register an
# environment for a build the loader would never actually pick.
read -r BUILD_KEY SITE_PACKAGES < <("$PYTHON" - <<'PYEOF'
import sys
import sysconfig
import torch

safe = torch.__version__.replace("+", "_").replace("/", "_")
key = f"torch-{safe}-cp{sys.version_info.major}{sys.version_info.minor}"
print(key, sysconfig.get_path("purelib"))
PYEOF
)

BUILD_DIR="$HERE/amd_tuned_torch/_native_builds/$BUILD_KEY"

echo "install_prebuilt.sh: looking for a prebuilt set matching this Python's torch"
echo "  build key:       $BUILD_KEY"
echo "  looked in:       $BUILD_DIR"

if [ ! -d "$BUILD_DIR" ]; then
    echo "install_prebuilt.sh: NO PREBUILT SET for $BUILD_KEY -- refusing to silently" >&2
    echo "  fall back to a real build (that's a multi-minute Composable Kernel compile," >&2
    echo "  not something to trigger by accident). Build it first:" >&2
    echo "      cd \"$HERE\" && AMD_TUNED_TORCH_GPU_ARCH=gfx1100 \"$PYTHON\" setup.py build_ext --inplace" >&2
    exit 1
fi

MISSING=0
for mod in _native _native_ck _native_hipblaslt; do
    if ! compgen -G "$BUILD_DIR/$mod.*" >/dev/null; then
        echo "install_prebuilt.sh: WARNING -- $mod.*.so not found in $BUILD_DIR" >&2
        MISSING=1
    fi
done
if [ "$MISSING" = "1" ]; then
    echo "install_prebuilt.sh: prebuilt set is incomplete (see warnings above) -- CK and/or" >&2
    echo "  hipBLASLt will report available()=False until a real build fills them in." >&2
fi

PTH_FILE="$SITE_PACKAGES/amd_tuned_torch.pth"
echo "$HERE" > "$PTH_FILE"
echo "install_prebuilt.sh: wrote $PTH_FILE -> $HERE"

# Verify end to end -- import for real, through the exact loader path a
# caller's own `import amd_tuned_torch` will take, rather than just
# trusting the .pth file was written correctly.
"$PYTHON" - <<PYEOF
import amd_tuned_torch
print(amd_tuned_torch.native_build_info())
print("ck available:", amd_tuned_torch.ck_ops.available())
print("hipblaslt available:", amd_tuned_torch.hipblaslt_ops.available())
PYEOF

echo "install_prebuilt.sh: done -- 'import amd_tuned_torch' now works in this environment, no compilation ran."
