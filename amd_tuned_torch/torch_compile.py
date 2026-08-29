"""amd_tuned_torch.torch_compile -- applies torch.compile (Dynamo capture + Inductor
codegen) to a whole module or function, tuned for the most aggressive
optimization Inductor offers. This is the runtime analogue of what a tool
like onnxslim does ahead-of-time to a static ONNX graph: onnxslim rewrites a
serialized node list (constant-fold, fuse Conv+BN, drop identity/dropout
nodes); this captures a live FX graph on first call and hands it to Inductor
for fusion, constant folding, and per-shape Triton-template autotuning,
caching the compiled artifact for the shapes actually seen.

amd_tuned_torch/compile_ops.py is the prerequisite groundwork this builds on:
linear_fp16/bmm_fp16/conv2d_fp16/group_norm are registered there as real
torch.library custom ops (with register_fake shape-only stand-ins) precisely
so Dynamo can place a call to one of them as an opaque graph node instead of
graph-breaking trying to inline through an aiter Triton kernel launcher or
amd_tuned_torch's pybind11 group_norm extension. This module is what actually
invokes torch.compile on top of that -- compile_ops.py alone doesn't compile
anything, it just stops amd_tuned_torch's own patched ops from being the thing that
breaks the graph once you do.

Usage:

    import amd_tuned_torch

    unet = amd_tuned_torch.torch_compile.compile_module(unet)
    # or, for a bare function/callable instead of an nn.Module:
    step_fn = amd_tuned_torch.torch_compile.compile_fn(step_fn)

Nothing here is installed automatically by amd_tuned_torch.enable()/disable() --
unlike amd_tuned_torch's op patches (numerically transparent, cheap to try),
compiling a specific module/function costs real first-call latency (whole
graph capture, then Inductor autotuning benchmarks multiple Triton template
configs per shape) and is worth it only for a callable that's actually hot
and called repeatedly at a small number of distinct shapes -- something only
the caller can judge, so this is always opt-in and explicit per-callable,
same posture as enable_int8_linear()/similarity_cached().

AGGRESSIVE BY DEFAULT
----------------------
mode="max-autotune"/"max-autotune-no-cudagraphs" (see no_cudagraphs below):
    Inductor benchmarks multiple Triton GEMM/conv/reduction template
    configs for the shapes actually seen and keeps whichever is fastest,
    instead of a single fixed heuristic lowering. Slower to compile --
    every new shape re-benchmarks -- fastest steady-state runtime. Also
    turns on coordinate-descent tuning and epilogue fusion where this
    PyTorch build exposes those knobs (older builds silently skip ones
    that don't exist -- see _configure_inductor).
freeze=True (default): sets torch._inductor.config.freezing, which bakes a
    compiled callable's parameters/buffers into the graph as constants --
    unlocking constant folding and fusion across parameter reads, the
    direct runtime analogue of onnxslim folding initializers into constant
    nodes on a static ONNX graph. Because baking in a parameter snapshot
    would silently go stale the moment training updates it, the wrapper
    this produces checks grad-safety on every call (same pattern
    amd_tuned_torch.__init__._grad_safe/amd_tuned_torch.cache._grad_safe already use) and
    falls back to the *uncompiled* original whenever autograd is actually
    live for this call's inputs -- so freeze=True is safe to leave on for
    a callable that's sometimes used for inference and sometimes for
    training, not just inference-only ones.
dynamic=False (the default here -- opposite of torch.compile's own default
    of guessing dynamic after a couple of shape changes): specializes the
    compiled graph to the exact shapes of the first call, recompiling
    (not silently falling back to a shape-generic, less-fused kernel) on
    every new shape. Right call for the diffusion-pipeline workloads this
    project targets, where resolution is typically fixed per run -- pass
    dynamic=True yourself if your callable actually sees many distinct
    shapes back-to-back (recompiling for each would cost more than it
    saves).
torch._dynamo.config.cache_size_limit is bumped (never lowered) so a
    handful of distinct shapes don't silently exhaust Dynamo's per-callable
    compile cache and fall back to eager mid-run without warning.
fx_graph_cache/autotune_local_cache are turned on (where this PyTorch
    build exposes them) so Inductor persists the lowered FX graph and
    max-autotune's own benchmark results to disk (TORCHINDUCTOR_CACHE_DIR,
    a per-user temp dir by default) and reuses them across process
    restarts, not just across calls within one process. max-autotune's
    whole cost is benchmarking multiple Triton template configs per shape
    on first compile; without this, a dev loop that restarts the process
    between iterations (e.g. iterating on a Gradio app) re-pays that
    benchmarking cost on every single restart even though the shapes and
    kernels compiled are identical run to run. Purely additive -- a cache
    hit only ever skips redundant work, it can't serve a wrong answer
    (unlike similarity caching's approximation risk), so unlike
    mode="max-autotune" or freeze=True there's no correctness tradeoff
    here to opt into; both flags already default to True on recent PyTorch
    builds, this just makes that explicit rather than depending on a
    version-dependent default. Set TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
    yourself (a torch.compile-native env var, not one of amd_tuned_torch's) if
    you're actively debugging a compile issue and need to rule out a stale
    cache entry -- it overrides both of these regardless.

Known ROCm caveats (this project targets RX 7900 XTX / gfx1100 -- see
amd_tuned_torch/__init__.py's module docstring):
  - CUDA graphs (captured as part of "reduce-overhead" and, unless
    disabled, "max-autotune") record and replay a fixed sequence of kernel
    launches. HIP graphs work on ROCm but are less battle-tested than on
    CUDA and interact badly with anything that changes tensor
    addresses/control flow between calls. no_cudagraphs=True (the default
    here) uses mode="max-autotune-no-cudagraphs" instead -- same
    autotuning, no graph capture/replay. Pass no_cudagraphs=False yourself
    only after verifying HIP graphs behave correctly for your specific
    model/ROCm version.
  - Inductor's Triton template autotuning only ever benchmarks *its own*
    generated Triton kernels for ops it lowers itself -- it has no
    visibility into aiter's kernels (already RDNA3-tuned, see
    amd_tuned_torch/aiter_ops.py) once linear_fp16/bmm_fp16/conv2d_fp16 are
    wrapped as opaque custom ops specifically so Dynamo won't try to
    inline and re-lower them. So enabling max-autotune here doesn't
    re-optimize (or risk regressing) the ops amd_tuned_torch already patches --
    aiter's kernels run exactly as before -- it searches everything else
    in the graph around them instead (elementwise fusion, softmax, grouped
    conv2d (groups != 1, left un-patched by amd_tuned_torch.enable()), attention
    epilogues, reductions).
"""
from __future__ import annotations

import os
import warnings
from typing import Any, Callable, Optional, TypeVar

import torch

T = TypeVar("T")


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in ("0", "", "false", "False")


def available() -> bool:
    return hasattr(torch, "compile")


def _grad_safe() -> bool:
    """Same check amd_tuned_torch.__init__/amd_tuned_torch.cache use to gate their own
    non-autograd kernel swaps -- duplicated here (rather than imported) to
    keep this module import-order-independent, same rationale as
    amd_tuned_torch.cache's own copy. Only tells you whether autograd is live
    *process-wide*; combine with _any_requires_grad(args, kwargs) for a
    specific call's actual tensors."""
    if hasattr(torch, "is_inference_mode_enabled") and torch.is_inference_mode_enabled():
        return True
    return not torch.is_grad_enabled()


def _any_requires_grad(args: tuple, kwargs: dict) -> bool:
    for v in (*args, *kwargs.values()):
        if isinstance(v, torch.Tensor) and v.requires_grad:
            return True
    return False


def _configure_inductor(no_cudagraphs: bool, freeze: bool) -> str:
    """Applies the aggressive Inductor/Dynamo config knobs process-wide (a
    global config, same as torch.compile's own mode= does under the hood --
    there's no per-callable scoping for these) and returns the torch.compile
    `mode` string to use. Every attribute is guarded with hasattr() since
    Inductor's config surface has grown across PyTorch versions -- an older
    build simply skips whichever knobs it doesn't expose yet rather than
    raising."""
    try:
        import torch._inductor.config as inductor_config
    except ImportError:
        inductor_config = None

    if inductor_config is not None:
        if hasattr(inductor_config, "max_autotune"):
            inductor_config.max_autotune = True
        if hasattr(inductor_config, "max_autotune_gemm"):
            inductor_config.max_autotune_gemm = True
        if hasattr(inductor_config, "max_autotune_conv"):
            inductor_config.max_autotune_conv = True
        if hasattr(inductor_config, "epilogue_fusion"):
            inductor_config.epilogue_fusion = True
        if hasattr(inductor_config, "coordinate_descent_tuning"):
            inductor_config.coordinate_descent_tuning = True
        if freeze and hasattr(inductor_config, "freezing"):
            inductor_config.freezing = True
        # Persist the lowered FX graph and max-autotune's benchmark results
        # to disk (see the module docstring's "fx_graph_cache/
        # autotune_local_cache" entry) so a process restart doesn't re-pay
        # max-autotune's benchmarking cost for shapes/kernels already
        # compiled in a previous run. Doesn't touch force_disable_caches --
        # if the caller has set TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 to
        # debug a stale-cache issue, that already overrides both of these,
        # nothing here needs to check for it separately.
        if hasattr(inductor_config, "fx_graph_cache"):
            inductor_config.fx_graph_cache = True
        if hasattr(inductor_config, "autotune_local_cache"):
            inductor_config.autotune_local_cache = True

    try:
        import torch._dynamo.config as dynamo_config
        if hasattr(dynamo_config, "cache_size_limit"):
            dynamo_config.cache_size_limit = max(dynamo_config.cache_size_limit, 64)
    except ImportError:
        pass

    return "max-autotune-no-cudagraphs" if no_cudagraphs else "max-autotune"


def _resolve_mode(mode: Optional[str], no_cudagraphs: bool, freeze: bool,
                   compile_kwargs: dict) -> Optional[str]:
    # mode is an Inductor-only concept -- if the caller explicitly picked a
    # different backend (e.g. "eager", for CPU-only testing without a real
    # Triton/Inductor stack -- see tests/test_torch_compile.py), leave mode
    # unset rather than forcing an Inductor-specific string torch.compile
    # would reject for that backend.
    if mode is not None:
        return mode
    if "backend" in compile_kwargs and compile_kwargs["backend"] != "inductor":
        return None
    return _configure_inductor(no_cudagraphs=no_cudagraphs, freeze=freeze)


def _wrap(orig: Callable[..., T], compiled: Callable[..., T]) -> Callable[..., T]:
    def _dispatch(*args: Any, **kwargs: Any) -> T:
        if not _grad_safe() and _any_requires_grad(args, kwargs):
            return orig(*args, **kwargs)
        return compiled(*args, **kwargs)

    _dispatch.__wrapped__ = orig
    _dispatch.amd_tuned_torch_compiled = compiled
    return _dispatch


def compile_fn(fn: Callable[..., T], *, mode: Optional[str] = None,
                fullgraph: Optional[bool] = None, dynamic: bool = False,
                freeze: bool = True, no_cudagraphs: bool = True,
                **compile_kwargs: Any) -> Callable[..., T]:
    """Wrap `fn` with torch.compile using amd_tuned_torch's aggressive defaults --
    see the module docstring for what each default actually does and the
    ROCm-specific caveats.

    fullgraph: None (the default) reads AMD_TUNED_TORCH_COMPILE_FULLGRAPH (default
        off) -- fullgraph=True makes Dynamo raise instead of silently
        fragmenting the graph on anything it can't trace through, useful to
        confirm a callable compiles cleanly but likely to hard-fail on
        arbitrary model code with real Python control flow.

    The returned callable is not simply the compiled one: every call first
    checks grad-safety (autograd live + an actual input requires_grad) and
    falls back to calling the original, uncompiled `fn` in that case -- so
    freeze=True's parameter-baking can never silently serve a stale
    snapshot to a training step, even if this same compiled callable is
    also used for inference elsewhere.
    """
    if not available():
        warnings.warn(
            "amd_tuned_torch.torch_compile.compile_fn(): torch.compile not available "
            "on this PyTorch build -- returning fn uncompiled."
        )
        return fn
    if fullgraph is None:
        fullgraph = _env_flag("AMD_TUNED_TORCH_COMPILE_FULLGRAPH")
    resolved_mode = _resolve_mode(mode, no_cudagraphs, freeze, compile_kwargs)
    compiled = torch.compile(fn, mode=resolved_mode, fullgraph=fullgraph,
                              dynamic=dynamic, **compile_kwargs)
    return _wrap(fn, compiled)


def compile_module(module: "torch.nn.Module", *, mode: Optional[str] = None,
                    fullgraph: Optional[bool] = None, dynamic: bool = False,
                    freeze: bool = True, no_cudagraphs: bool = True,
                    **compile_kwargs: Any) -> "torch.nn.Module":
    """Like compile_fn, but replaces `module.forward` in place and returns
    `module` itself (so `m = amd_tuned_torch.torch_compile.compile_module(m)` and
    bare `amd_tuned_torch.torch_compile.compile_module(m)` are equally fine)."""
    if not available():
        warnings.warn(
            "amd_tuned_torch.torch_compile.compile_module(): torch.compile not "
            "available on this PyTorch build -- returning module uncompiled."
        )
        return module
    module.forward = compile_fn(
        module.forward, mode=mode, fullgraph=fullgraph, dynamic=dynamic,
        freeze=freeze, no_cudagraphs=no_cudagraphs, **compile_kwargs,
    )
    return module
