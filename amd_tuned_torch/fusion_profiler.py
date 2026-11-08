"""amd_tuned_torch.fusion_profiler -- find fusable GroupNorm+activation
sites by WATCHING a model run, then fuse them on the next run.

WHY WATCHING BEATS PATTERN-MATCHING HERE. conv_fusion_ops.replace_conv_blocks
recognises nn.Sequential(GroupNorm, SiLU, Conv2d) structurally, which is the
shape SDXL's UNet ResBlock builds. It finds nothing in the VAE, because the
VAE's ResnetBlock (sgm/modules/diffusionmodules/model.py) writes the same
computation inline:

    h = self.norm1(h)
    h = nonlinearity(h)        # x * torch.sigmoid(x)
    h = self.conv1(h)

There is no Sequential to match, and `nonlinearity` is a hand-written
sigmoid-multiply rather than F.silu or nn.SiLU, so a patch on either of
those sees nothing either. The VAE decode is where this package's captured
SDXL workload spends its largest-tensor convolutions (c320 at 256x128), so
that blind spot is not a corner case.

Watching the model run finds those sites without knowing anything about
the repository that wrote them.

WHY THE SECOND RUN, AND NOT THIS ONE. A fused GroupNorm+SiLU has to produce
the activated output IN PLACE OF the plain one -- but by the time
F.group_norm is called, the code that will apply the activation has not run
yet, and once it has, the fusion opportunity is gone. You cannot fuse
retroactively within a call. Two ways out:

  1. SPECULATE. Return an already-activated tensor, tag it, and patch
     torch.sigmoid to return a broadcast 1.0 for tagged inputs so the
     caller's `h * sigmoid(h)` collapses to `h`. Cheap and it works -- until
     a call site that the profile said was always followed by SiLU is not,
     on some branch or resolution the profile never saw, and the output is
     silently wrong. This module does NOT do that. Everything else in this
     package exists to keep fast-and-wrong from winning; a rewrite whose
     correctness rests on "we saw it behave this way earlier" is exactly
     that bargain.

  2. COMPILE. Hand the modules that own a fusable site to torch.compile,
     which captures the real graph -- inline sigmoid-multiply and all --
     and fuses it with the compiler responsible for correctness. That is
     what apply() does.

MEASURED, on the VAE ResnetBlock shape this finds and replace_conv_blocks
does not, fp16, RX 7900 XTX, interleaved A/B:

    VAE c320 256x128   3.189ms -> 2.605ms   1.22x
    VAE c640 128x64    2.643ms -> 2.324ms   1.14x
    VAE c512 64x64     0.946ms -> 0.813ms   1.16x

with a null control (the identical model on both sides of the same
harness) at 0.98x, so those are outside the noise for that session. Note
the caveat in conv_fusion_ops' docstring about this card's clocks: pin
them with `rocm-smi --setperflevel high` before trusting a number, and
re-run the null control alongside anything you measure.

OBSERVATION IS READ-ONLY. The profiling pass records tensor identities and
nothing else -- no output is altered, no op is replaced with a different
implementation. A model produces bit-identical results whether or not it is
being profiled, which is checked in tests/test_fusion_profiler.py rather
than asserted here.

USAGE

    from amd_tuned_torch import fusion_profiler

    with fusion_profiler.observing(model):      # run 1: one real forward
        model(x)

    print(fusion_profiler.sites(model))         # what it found
    fusion_profiler.apply(model)                # run 2 onwards: fused

`save()`/`load()` persist the findings by module path, so run 1 and run 2
can be separate processes -- which is the usual case for a webui that
loads a model, generates, and exits.
"""
from __future__ import annotations

import contextlib
import json
import os
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn

_CACHE_ENV = "AMD_TUNED_TORCH_FUSION_PROFILE"
_ENABLE_ENV = "AMD_TUNED_TORCH_FUSION_PROFILE_APPLY"

# id(tensor) -> the GroupNorm module that produced it, for the current
# observation window only. Ids are reused once a tensor is freed, so this is
# cleared when observation stops and is never consulted outside the window.
_pending: Dict[int, nn.Module] = {}
_consumed: Set[int] = set()          # id() of GroupNorm modules followed by an activation
_observing = False


def _gn_hook(module, inputs, output):
    if _observing and isinstance(output, torch.Tensor):
        _pending[id(output)] = module


def _note_consumer(x):
    """Called from the watched activation ops. Read-only."""
    if _observing and isinstance(x, torch.Tensor):
        owner = _pending.get(id(x))
        if owner is not None:
            _consumed.add(id(owner))


@contextlib.contextmanager
def observing(model: nn.Module):
    """Watch `model` for one or more forward passes and record which
    GroupNorm outputs are consumed by an activation.

    Catches BOTH spellings that matter: F.silu / nn.SiLU (the UNet) and a
    bare torch.sigmoid feeding a multiply (the VAE's `nonlinearity`). The
    sigmoid watcher is why the VAE is visible at all -- nothing else in
    this package sees it."""
    global _observing
    import torch.nn.functional as F

    # Observation must see the model's OWN code, not a partially-fused
    # version of it. _dispatch's nn.Sequential patch routes a matching
    # block straight to conv_fusion_ops.group_norm_silu, which calls
    # torch.group_norm directly -- so nn.GroupNorm.forward never runs, the
    # hook below never fires, and the profiler silently under-reports
    # exactly the blocks that are already handled. Worse, what it reports
    # would then depend on whether enable() happened to have run, which
    # makes the profile non-reproducible. Stand that patch down for the
    # window.
    seq_restore = None
    try:
        from . import _dispatch
        if nn.Sequential.forward is _dispatch._patched_sequential_forward:
            seq_restore = _dispatch._ORIGINALS.get((nn.Sequential, "forward"))
            if seq_restore is not None:
                nn.Sequential.forward = seq_restore
    except Exception:
        seq_restore = None

    handles = [m.register_forward_hook(_gn_hook)
               for m in model.modules() if isinstance(m, nn.GroupNorm)]
    orig_sigmoid, orig_silu = torch.sigmoid, F.silu

    def watched_sigmoid(x, *a, **k):
        _note_consumer(x)
        return orig_sigmoid(x, *a, **k)

    def watched_silu(x, *a, **k):
        _note_consumer(x)
        return orig_silu(x, *a, **k)

    torch.sigmoid, F.silu = watched_sigmoid, watched_silu
    _observing = True
    try:
        yield
    finally:
        _observing = False
        torch.sigmoid, F.silu = orig_sigmoid, orig_silu
        for h in handles:
            h.remove()
        _pending.clear()
        if seq_restore is not None:
            from . import _dispatch
            nn.Sequential.forward = _dispatch._patched_sequential_forward


def sites(model: nn.Module) -> List[str]:
    """Qualified names of the GroupNorm modules observed to be followed by
    an activation, sorted. Empty before any observation."""
    return sorted(name for name, mod in model.named_modules()
                  if id(mod) in _consumed)


def owners(model: nn.Module) -> List[str]:
    """Qualified names of the nearest enclosing modules that OWN a fusable
    site -- these are what apply() compiles, since a GroupNorm on its own
    is not a graph worth capturing. The parent of `dec.block.norm1` is
    `dec.block`."""
    out = []
    for name in sites(model):
        parent = name.rsplit(".", 1)[0] if "." in name else ""
        if parent and parent not in out:
            out.append(parent)
    return out


def reset() -> None:
    _consumed.clear()
    _pending.clear()


# ---------------------------------------------------------------------------
# Persistence -- run 1 and run 2 are usually different processes
# ---------------------------------------------------------------------------

def _cache_path() -> str:
    override = os.environ.get(_CACHE_ENV)
    if override:
        return override
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "_fusion_profile", "sites.json")


def save(model: nn.Module, path: Optional[str] = None) -> str:
    """Persist the observed sites by module path."""
    path = path or _cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"sites": sites(model), "owners": owners(model)}, fh, indent=2)
    return path


def load(path: Optional[str] = None) -> List[str]:
    """Owner module paths recorded by a previous run, or [] if none."""
    path = path or _cache_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path) as fh:
            return list(json.load(fh).get("owners", ()))
    except (OSError, ValueError):
        return []


# ---------------------------------------------------------------------------
# Applying it
# ---------------------------------------------------------------------------

def apply(model: nn.Module, owner_names: Optional[List[str]] = None,
          **compile_kwargs) -> int:
    """Wrap each module that owns a fusable site in torch.compile, in
    place. Returns how many were wrapped.

    `owner_names` defaults to what was observed in this process, falling
    back to what load() finds on disk -- so a second PROCESS can fuse using
    a first process's findings.

    torch.compile does the fusion and owns its correctness. This module
    only decides WHERE to point it, which is the part a pattern matcher
    cannot work out for inline code. Compiling the whole model instead
    would also work and is a perfectly reasonable thing to do; the point of
    narrowing it is that a diffusion model has plenty of graph-break-prone
    code (samplers, conditioning, dynamic shapes) that compiling wholesale
    drags in, and these blocks do not."""
    names = owner_names if owner_names is not None else (owners(model) or load())
    if not names:
        return 0
    wrapped = 0
    for name in names:
        parent, _, leaf = name.rpartition(".")
        try:
            container = model.get_submodule(parent) if parent else model
            target = getattr(container, leaf)
        except AttributeError:
            continue  # a path from a previous run that this model does not have
        if isinstance(target, torch._dynamo.eval_frame.OptimizedModule) \
                if hasattr(torch, "_dynamo") else False:
            continue  # already compiled
        setattr(container, leaf, torch.compile(target, **compile_kwargs))
        wrapped += 1
    return wrapped


def describe(model: Optional[nn.Module] = None) -> str:
    found = sites(model) if model is not None else []
    if not found:
        cached = load()
        return (f"fusion_profiler: nothing observed yet"
                + (f"; {len(cached)} owner(s) on disk" if cached else ""))
    return (f"fusion_profiler: {len(found)} fusable GroupNorm site(s) in "
            f"{len(owners(model))} owning module(s)")
