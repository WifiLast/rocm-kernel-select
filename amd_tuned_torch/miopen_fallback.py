"""amd_tuned_torch.miopen_fallback -- CPU rescue for MIOpen convolution calls that
have no working GPU algorithm for a given shape/context on this hardware
(RX 7900 XTX / gfx1100), instead of the process crashing outright.

This is a DIFFERENT kind of thing from every op amd_tuned_torch.enable() patches:
those swap a working stock op for a faster kernel; this exists because a
stock op sometimes doesn't work at all. MIOpen's conv algorithm search can
raise `RuntimeError: miopenStatusUnknownError ... No suitable algorithm was
found to execute the required convolution` for a specific (shape, dtype,
stride/padding/dilation, *and current allocator state*) combination that
has a perfectly good solution on a fresh/idle GPU -- observed directly:
the exact (1, 1024, 20000) x (1024, 1024, 7) Conv1d call that crashed
Wan2GP's ACE-Step (models/TTS/ace_step15/models/autoencoder_oobleck.py, an
Oobleck audio VAE decoder built from plain weight_norm(nn.Conv1d(...))
layers, no per-class hook to patch individually) succeeds cleanly in a
fresh Python process with the GPU otherwise idle, and *still* succeeds
under a synthetic 1.5GB-free-VRAM ballast allocation -- so it isn't simply
"this shape is unsupported" or "not enough total free bytes" either.

The likely mechanism: MIOpen's workspace requirement (its own error log
reports "IsEnoughWorkspace ... workspace required: 286720000, provided
ptr: 0 size: 0" for this exact failure) needs one *contiguous* block; a
long-running multi-model pipeline (video DiT + VAE + audio VAE + text
encoders, churning many different tensor shapes over a session) can
fragment PyTorch's caching allocator's free space into many blocks with no
single one large enough, even while total free bytes look fine. That's a
transient, context-dependent condition -- not a hard capability wall -- so
the fix tries a defragmenting retry before paying for a CPU round-trip:

  1. Run the real op. If it raises a miopenStatus RuntimeError, continue.
  2. torch.cuda.empty_cache() (returns cached-but-unused blocks to the
     driver, coalescing fragmented free space) and retry the SAME call on
     GPU once. If that succeeds, return it -- no CPU involved, no
     slowdown, the common case if the fragmentation theory holds.
  3. Only if the retry ALSO raises a miopenStatus error, fall back to
     running the op on CPU in float32 (MIOpen's algorithm search doesn't
     apply there) and cast the result back to the original device/dtype.
     This is the same shape of fallback models/wan/modules/vae.py's
     CausalConv3d already hand-rolls for Conv3d in Wan2GP -- this module
     generalizes it to any conv dimensionality instead of re-copying that
     pattern per project/per op.

Every branch prints which path it took (plus free/total VRAM for the
retry paths) specifically so a live failure produces the diagnostic data
needed to confirm or refute the fragmentation theory, rather than staying
a guess.

Not installed by amd_tuned_torch.enable()/disable() -- like amd_tuned_torch.cache, this is
a standalone patch on torch.nn.functional.conv1d, installed via
enable_conv1d_fallback() rather than bundled with the kernel-dispatch
patches. It changes error-handling behavior (a call that would have
crashed now succeeds, possibly on CPU and slowly) rather than swapping one
correct-either-way kernel for a faster one, so it lives in its own
opt-out toggle instead of silently riding along with enable() --
AMD_TUNED_TORCH_MIOPEN_CONV1D_FALLBACK, default ON (see below), read once at
import time the same way AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE is read for
amd_tuned_torch.cache.

CAUSAL_CONV1D FAST PATH
------------------------
As of this integration, `wrapped()` tries one more thing before ever
calling MIOpen: source/kernel/causal-conv1d-amd (a HIP port of Tri Dao's
causal_conv1d, the depthwise causal Conv1d used by Mamba/SSM-style blocks
-- short kernel width 2-4, one input channel per group, no cross-channel
mixing) ships a hand-written HIP kernel for exactly this op shape. When a
conv1d call structurally matches that pattern (see
_is_causal_conv1d_eligible: groups == in_channels == out_channels,
weight.shape == (dim, 1, width), width in [2, 4], stride=1, dilation=1,
padding == width - 1 -- the standard "pad left, trim right" causal-conv
idiom used by Mamba/Hyena/TCN-style blocks), it's routed to
causal_conv1d_cuda's causal_conv1d_fn directly, skipping MIOpen's
algorithm search entirely rather than waiting for it to maybe crash. This
is strictly a fast path, not just a crash rescue: it fires unconditionally
whenever eligible, even when MIOpen would have succeeded, because the
custom kernel is both faster and immune to the workspace-fragmentation
failure mode the rest of this module exists to route around. It also has
a real backward pass (CausalConv1dFn is a torch.autograd.Function backed
by the HIP fwd/bwd kernels), so unlike aiter's conv2d/linear swaps this
one stays active under autograd too.

Output-shape caveat: causal_conv1d_fn returns output already truncated to
`seqlen` (the causal slice), whereas plain
`F.conv1d(x, w, padding=width-1, groups=dim)` returns the full
`seqlen + width - 1` symmetric-padded length. Substituting the truncated
result is only safe because the padding=width-1/depthwise combination is,
in practice, always the causal idiom -- callers either expect the causal
length directly, or (matching causal_conv1d_ref's own reference
implementation, see causal_conv1d_interface.py) immediately slice
`[..., :seqlen]` themselves, which becomes a no-op against an
already-seqlen-length tensor. A depthwise conv1d call using
padding=width-1 that genuinely wants the untrimmed symmetric output
(unusual -- padding=width//2 is the normal choice for that) would get a
shorter tensor back; no such call site exists in this project's model
code. Requires source/kernel/causal-conv1d-amd's `causal_conv1d_cuda`
extension to be built and importable
(`pip install -e source/kernel/causal-conv1d-amd`) -- silently skipped
(falls through to the MIOpen path below) if it isn't.

Usage:

    import amd_tuned_torch
    amd_tuned_torch.miopen_fallback.enable_conv1d_fallback()  # also runs at import by default

Only Conv1d is wired up right now (the actual failure this was built for);
the retry-then-CPU machinery itself (_miopen_safe_conv) is dimensionality-
agnostic and could back a Conv2d/Conv3d variant the same way if a similar
failure ever shows up there too -- F.conv2d already goes through aiter's
Triton kernel under amd_tuned_torch.enable() instead of MIOpen, so it hasn't needed
this; F.conv3d isn't patched by amd_tuned_torch at all (Wan2GP's own CausalConv3d
already covers its one call site).
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional

import torch

try:
    from causal_conv1d import causal_conv1d_fn as _causal_conv1d_fn

    _CAUSAL_CONV1D_AVAILABLE = True
except ImportError:
    _causal_conv1d_fn = None
    _CAUSAL_CONV1D_AVAILABLE = False


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "", "false", "no", "off")


def causal_conv1d_available() -> bool:
    """True if source/kernel/causal-conv1d-amd's compiled extension is
    importable. When False, _try_causal_conv1d_fastpath always returns None
    and every conv1d call falls straight through to the MIOpen path below."""
    return _CAUSAL_CONV1D_AVAILABLE


def _scalar(v: Any) -> Any:
    return v[0] if isinstance(v, (tuple, list)) else v


def _is_causal_conv1d_eligible(
    input: torch.Tensor, weight: torch.Tensor, stride: Any, padding: Any, dilation: Any, groups: int,
) -> bool:
    """True if this is exactly the depthwise causal Conv1d pattern
    causal_conv1d_cuda implements -- see the module docstring's
    CAUSAL_CONV1D FAST PATH section for the full reasoning and the
    output-shape caveat this relies on."""
    if not (input.is_cuda and input.dim() == 3 and weight.dim() == 3):
        return False
    dim = input.shape[1]
    out_channels, in_channels_per_group, width = weight.shape
    if groups != dim or out_channels != dim or in_channels_per_group != 1:
        return False
    if not (2 <= width <= 4):
        return False
    if _scalar(stride) != 1 or _scalar(dilation) != 1:
        return False
    pad = _scalar(padding)
    return isinstance(pad, int) and pad == width - 1


def _try_causal_conv1d_fastpath(
    input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
    stride: Any, padding: Any, dilation: Any, groups: int,
) -> Optional[torch.Tensor]:
    """The fast HIP-native result for an eligible depthwise-causal conv1d
    call, or None if not eligible/available -- callers fall through to the
    MIOpen path on None, same as any other unsupported-shape signal in this
    package (e.g. conv2d's AssertionError fallback in amd_tuned_torch/__init__.py)."""
    if not _CAUSAL_CONV1D_AVAILABLE or not _is_causal_conv1d_eligible(
        input, weight, stride, padding, dilation, groups
    ):
        return None
    try:
        return _causal_conv1d_fn(input, weight.squeeze(1), bias)
    except RuntimeError:
        return None


def _miopen_safe_conv(op_name: str, orig_fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Build a MIOpen-failure-tolerant wrapper around a torch.nn.functional
    convN op. See the module docstring for the retry-then-CPU-fallback
    sequence; op_name is only used to label the printed diagnostics."""

    def wrapped(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                stride: Any = 1, padding: Any = 0, dilation: Any = 1, groups: int = 1) -> torch.Tensor:
        fast = _try_causal_conv1d_fastpath(input, weight, bias, stride, padding, dilation, groups)
        if fast is not None:
            return fast
        try:
            return orig_fn(input, weight, bias, stride, padding, dilation, groups)
        except RuntimeError as e:
            if "miopenStatus" not in str(e):
                raise
            if input.is_cuda:
                free_before, total = torch.cuda.mem_get_info()
                torch.cuda.empty_cache()
                try:
                    out = orig_fn(input, weight, bias, stride, padding, dilation, groups)
                    free_after, _ = torch.cuda.mem_get_info()
                    print(f"⚠️ MIOpen {op_name} failed once, but succeeded on GPU after "
                          f"empty_cache() (free VRAM {free_before/1e9:.2f}GB -> "
                          f"{free_after/1e9:.2f}GB / {total/1e9:.2f}GB). "
                          f"x shape: {tuple(input.shape)}, weight shape: {tuple(weight.shape)}")
                    return out
                except RuntimeError as e2:
                    if "miopenStatus" not in str(e2):
                        raise
                    free_now, _ = torch.cuda.mem_get_info()
                    print(f"⚠️ MIOpen fallback ({op_name}): CPU used for this convolution "
                          f"(slow), empty_cache() retry did not help "
                          f"(free VRAM {free_now/1e9:.2f}GB / {total/1e9:.2f}GB). "
                          f"x shape: {tuple(input.shape)}, weight shape: {tuple(weight.shape)}")
            else:
                print(f"⚠️ MIOpen fallback ({op_name}): CPU used for this convolution (slow). "
                      f"x shape: {tuple(input.shape)}, weight shape: {tuple(weight.shape)}")
            device, dtype = input.device, input.dtype
            out = orig_fn(
                input.float().cpu(), weight.float().cpu(),
                bias.float().cpu() if bias is not None else None,
                stride, padding, dilation, groups,
            )
            return out.to(device=device, dtype=dtype)

    return wrapped


_original_conv1d: Optional[Callable[..., torch.Tensor]] = None
_CONV1D_FALLBACK_ENABLED = False


def enable_conv1d_fallback() -> None:
    """Patch torch.nn.functional.conv1d with the MIOpen retry/CPU-fallback
    wrapper (see module docstring). Idempotent -- calling this again while
    already enabled is a no-op, same as amd_tuned_torch.cache.enable_module_cache()."""
    global _original_conv1d, _CONV1D_FALLBACK_ENABLED
    if _CONV1D_FALLBACK_ENABLED:
        return
    _original_conv1d = torch.nn.functional.conv1d
    torch.nn.functional.conv1d = _miopen_safe_conv("conv1d", _original_conv1d)
    _CONV1D_FALLBACK_ENABLED = True


def disable_conv1d_fallback() -> None:
    """Restore stock torch.nn.functional.conv1d."""
    global _original_conv1d, _CONV1D_FALLBACK_ENABLED
    if not _CONV1D_FALLBACK_ENABLED or _original_conv1d is None:
        return
    torch.nn.functional.conv1d = _original_conv1d
    _original_conv1d = None
    _CONV1D_FALLBACK_ENABLED = False


def is_conv1d_fallback_enabled() -> bool:
    return _CONV1D_FALLBACK_ENABLED


if _env_flag("AMD_TUNED_TORCH_MIOPEN_CONV1D_FALLBACK", default="1"):
    enable_conv1d_fallback()
