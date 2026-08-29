"""Global auto-loader: patches every `import torch` in this Python
environment to also `import amd_tuned_torch` right after, with no per-script edits.

Python's `site` module auto-imports a module named `sitecustomize` at
interpreter startup if one is importable from `sys.path` -- this file only
does anything once it's actually placed somewhere Python will find it (see
"Installation" below). Sitting in this repo, it's inert.

This hooks `builtins.__import__` rather than eagerly importing amd_tuned_torch (or
torch) itself, so torch/amd_tuned_torch only load the moment something in the
process actually does `import torch` -- scripts that never touch torch pay
zero cost, unlike a naive `.pth` file with a bare `import amd_tuned_torch` line
(which would load torch, and therefore amd_tuned_torch, at every single Python
startup in this environment regardless of whether that particular script
ever uses it).

SAFETY: amd_tuned_torch's auto-import is wrapped in try/except -- if it's not built
yet, aiter/TE aren't installed, or anything else goes wrong, this prints a
one-line warning to stderr and leaves torch unpatched. A global hook that
could itself break unrelated Python programs in this environment (pytest,
pip, other tools) would be far worse than a silent no-op.

Installation
------------
Copy (or symlink) this file directly into your environment's site-packages
root -- NOT into a subpackage, `site` only looks for a top-level
`sitecustomize` module:

    cp tools/sitecustomize.py \
        $(python -c "import site; print(site.getsitepackages()[0])")/sitecustomize.py

If a `sitecustomize.py` already exists there (from another package), don't
overwrite it -- append this file's contents (everything below the
docstring) to the existing one instead, or the other package's
customization will silently stop running.

To confirm it's active:

    python -c "import torch; import amd_tuned_torch; print(amd_tuned_torch.is_enabled())"

should print `True` without you having imported amd_tuned_torch explicitly.

Uninstall: delete (or revert) that copied sitecustomize.py, or set
AMD_TUNED_TORCH_AUTOPATCH=0 in the environment to keep the auto-import but stop it
from actually patching anything.
"""
import builtins
import sys

_real_import = builtins.__import__
_amd_tuned_torch_loading = False


def _import_hook(name, *args, **kwargs):
    global _amd_tuned_torch_loading
    module = _real_import(name, *args, **kwargs)
    if not _amd_tuned_torch_loading and name.split(".")[0] == "torch":
        _amd_tuned_torch_loading = True
        try:
            import amd_tuned_torch  # noqa: F401 -- imported for its patching side effect
        except Exception as exc:  # noqa: BLE001 -- must never break the host program
            print(f"[amd_tuned_torch sitecustomize] auto-import failed, torch left "
                  f"unpatched: {exc}", file=sys.stderr)
        finally:
            _amd_tuned_torch_loading = False
    return module


builtins.__import__ = _import_hook
