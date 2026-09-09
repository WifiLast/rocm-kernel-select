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

FFTCONV1D FAST PATH
--------------------
Tried next, for any conv1d call the causal fast path above didn't already
claim: fftconv_ops.fftconv1d_candidate (amd_tuned_torch/fftconv_ops.py) is
contested against stock conv1d through amd_tuned_torch.kernel_select.pick --
the same measure-once-then-cache-the-winner policy already used for
conv2d/conv3d/linear/bmm/attention (see kernel_select.py's own module
docstring), rather than trusting fftconv_ops's own hardcoded kernel-width
guess outright. This decides, per (dtype, shape, stride, padding,
dilation), whether FFT-conv's O(N log N) cost actually beats direct
convolution's O(N*K) on THIS card for THIS shape -- the long-kernel/
global-convolution regime this project's native/CK conv tiers (tuned for
3x3-style small kernels) and MIOpen's own Winograd solvers don't cover,
the same use case (Hyena/long-conv blocks) `source/flash-fft-conv` targets
with a tensor-core-only implementation this project couldn't port (see
fftconv_ops.py's module docstring). The contest passes a looser-than-
kernel_select's-default verification tolerance (fftconv_ops.fftconv_tolerance)
since FFT-based and direct convolution accumulate rounding differently --
see that function's docstring. Unlike the sparse fast path just below, no
_grad_safe check is needed here: fft_conv's entire computation is ordinary
differentiable PyTorch with a real backward pass, so this stays active
during training too. On by default; AMD_TUNED_TORCH_FFTCONV1D=0 disables it
independently of this module's own AMD_TUNED_TORCH_MIOPEN_CONV1D_FALLBACK
gate (and of AMD_TUNED_TORCH_MEASURE_KERNELS, kernel_select's own master
switch, which independently disables the underlying contest mechanism for
every kernel_select-gated tier in this package, not just this one).

DEPTHWISE CONV1D FAST PATH
--------------------------
Folded into the SAME contest as the FFTCONV1D fast path above (see
_try_fftconv1d_fastpath's own docstring for exactly how), not a separate
sequential try: depthwise_conv1d_ops.depthwise_conv1d_candidate
(amd_tuned_torch/depthwise_conv1d_ops.py, backed by a HIP kernel vendored
from FlashFFTConv independently of third_party/FlashFFTConv/ itself -- see
that module's docstring) is entered into kernel_select's contest whenever
the call is depthwise (groups == in_channels == out_channels) with an odd
kernel width and stride == dilation == 1
(_is_depthwise_conv1d_eligible). Broader than the CAUSAL_CONV1D fast path
above -- any odd width and any symmetric padding, not just width in [2, 4]
with padding == width - 1 -- and, unlike that fast path, its output matches
stock conv1d's formula exactly rather than needing a truncation caveat, so
it's contested for real instead of pattern-matched in unconditionally. Has
a real backward pass, so stays active during training too.

SPARSE CONV1D FAST PATH
------------------------
Tried next, for any groups=1, grad-safe conv1d call neither fast path
above already claimed: flexgemm_ops.maybe_sparse_conv1d
(amd_tuned_torch/flexgemm_ops.py) estimates `input`'s occupancy and, only
when it's mostly empty, routes through flex_gemm's sparse convolution (a
real ROCm/HIP kernel when third_party/FlexGEMM is installed, via the same
"1D conv is a 3D conv with two spatial axes of size 1" lift
sparse_conv2d_native already uses to reuse the same kernel -- see that
module's docstring; a pure-PyTorch fallback otherwise, so this always
produces a result on CPU too). Content-dependent, so it's checked on
every eligible call rather than cached by shape -- see
maybe_sparse_conv1d's own docstring. On by default;
AMD_TUNED_TORCH_SPARSE_CONV1D=0 disables it independently of this module's own
AMD_TUNED_TORCH_MIOPEN_CONV1D_FALLBACK gate.

GRAD-SAFETY. flexgemm_ops's sparse conv path has no backward pass, so
_try_sparse_conv1d_fastpath checks _grad_safe(input, weight, bias) (same
semantics as amd_tuned_torch.__init__._grad_safe, duplicated here to avoid
a circular import) before ever calling it -- during training, this fast
path is skipped entirely and the call falls through to the causal-conv1d/
MIOpen path below, same as every other non-autograd tier elsewhere in this
package. This check was ABSENT when the fast path was first wired in: a
grad-tracked call would have silently returned a tensor with a real
gradient for this layer's weight/bias but a permanently zero gradient for
everything upstream, since sparse_conv1d_from_dense extracts `input`
via `.detach()` internally. Fixed by adding the check above -- see
_grad_safe's own docstring for the full failure mode this closes.

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

from . import depthwise_conv1d_ops
from . import fftconv_ops
from . import flexgemm_ops
from . import kernel_select

try:
    from causal_conv1d import causal_conv1d_fn as _causal_conv1d_fn

    _CAUSAL_CONV1D_AVAILABLE = True
except ImportError:
    _causal_conv1d_fn = None
    _CAUSAL_CONV1D_AVAILABLE = False


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "", "false", "no", "off")


# Resolved once, same reasoning as amd_tuned_torch.__init__._HAS_INFERENCE_MODE:
# _grad_safe runs on every eligible conv1d call, so a hasattr() lookup on
# the torch module there for the life of the process is pure waste.
_HAS_INFERENCE_MODE = hasattr(torch, "is_inference_mode_enabled")


def _grad_safe(*tensors: Any) -> bool:
    """Same semantics as amd_tuned_torch.__init__._grad_safe -- duplicated
    rather than imported (this module is imported BY
    amd_tuned_torch/__init__.py, not the other way around, so importing it
    back would be circular; also keeps this module independently droppable,
    same posture as flexgemm_ops._env_flag not importing this module's).

    WHY THIS EXISTS -- A REAL BUG THIS CLOSES. _try_sparse_conv1d_fastpath
    below routes to flexgemm_ops.maybe_sparse_conv1d, which has NO backward
    pass (see flexgemm_ops.sparse_conv1d_from_dense's own docstring: "No
    autograd support -- caller must ensure grad-safety first"). Without
    this check, calling it during training would not raise or fall back --
    sparse_conv1d_from_dense extracts its working copy of `input` via
    `input.detach()` internally, but leaves `weight`/`bias` un-detached in
    its own matmul/bias-add, so the tensor it returns has requires_grad
    reflecting `weight`/`bias` but NOT `input`. backward() would then run
    to completion, produce a real (correct) gradient for this layer's own
    weight/bias, and silently propagate a ZERO gradient to every layer
    upstream of this conv1d call -- a partially-broken backward pass with
    no visible error. This is exactly the failure mode _grad_safe already
    prevents for every other non-autograd tier in this package (aiter's
    conv2d/matmul, the native HIP kernels) -- conv1d's sparse fast path
    was simply missing the same guard when it was wired in."""
    if _HAS_INFERENCE_MODE and torch.is_inference_mode_enabled():
        return True
    if not torch.is_grad_enabled():
        return True
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.requires_grad:
            return False
    return True


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


def _is_depthwise_conv1d_eligible(
    input: torch.Tensor, weight: torch.Tensor, stride: Any, dilation: Any, groups: int,
) -> bool:
    """True if this is a depthwise Conv1d (groups == in_channels ==
    out_channels, one input channel per group) with an odd kernel width
    and stride == dilation == 1 -- the exact shape
    depthwise_conv1d_ops.depthwise_conv1d_candidate's vendored kernel
    supports (see conv1d.h's `TORCH_CHECK(k % 2 == 1, ...)` and the fact
    that its forward signature has no stride/dilation parameters at all).
    Broader than _is_causal_conv1d_eligible (any odd width and any
    symmetric padding value, not just width in [2, 4] with
    padding == width - 1) -- see depthwise_conv1d_ops's module docstring
    for how the two fast paths' scopes relate."""
    if not (input.is_cuda and input.dim() == 3 and weight.dim() == 3):
        return False
    dim = input.shape[1]
    out_channels, in_channels_per_group, width = weight.shape
    if groups != dim or out_channels != dim or in_channels_per_group != 1:
        return False
    if width % 2 != 1:
        return False
    return _scalar(stride) == 1 and _scalar(dilation) == 1


def _try_fftconv1d_fastpath(
    input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
    stride: Any, padding: Any, dilation: Any, groups: int,
    orig_fn: Callable[..., torch.Tensor],
) -> Optional[torch.Tensor]:
    """Large-kernel fast path, tried before MIOpen (and before the sparse
    fast path below, so a large AND sparse kernel prefers whichever of
    fftconv/stock wins the contest here -- sparse_conv1d's per-kernel-
    position gather loop scales with kernel width, exactly the regime this
    path exists to avoid): contests fftconv_ops.fftconv1d_candidate against
    `orig_fn` (stock conv1d) through amd_tuned_torch.kernel_select.pick,
    the same measure-once-then-cache-the-winner policy conv2d/conv3d/
    linear/bmm/attention already use (see kernel_select.py's own module
    docstring) -- rather than trusting fftconv_ops's own hardcoded
    kernel-width guess (`maybe_fft_conv1d`) outright. `tolerance` overrides
    kernel_select's default per-dtype verification bar with one calibrated
    for an FFT-vs-direct-conv contest specifically (see
    fftconv_ops.fftconv_tolerance's docstring for why the shared default is
    too strict here). Unlike the sparse fast path, no _grad_safe check is
    needed -- fft_conv has a real, correct backward pass (see
    fftconv_ops.fft_conv's own docstring), so this stays active under
    training too. `stock` is listed last (kernel_select's convention for
    "the reference, assumed to always work") so a shape kernel_select can't
    time at all (AMD_TUNED_TORCH_MEASURE_KERNELS=0) or where fftconv declines
    still gets a real conv1d result rather than None. AMD_TUNED_TORCH_FFTCONV1D=0
    disables the fftconv candidate entirely (contest never built, `orig_fn`
    called directly by the caller as before) independently of this
    module's own AMD_TUNED_TORCH_MIOPEN_CONV1D_FALLBACK gate.

    When the call is ALSO depthwise-eligible (see
    _is_depthwise_conv1d_eligible) and depthwise_conv1d_ops is available, a
    third candidate ("depthwise") is folded into the SAME contest --
    deliberately one kernel_select.pick call with up to three candidates
    rather than two sequential 2-way contests, so a depthwise+large-kernel
    shape picks the genuine fastest of {fftconv, depthwise-direct, stock}
    instead of whichever fast path happened to run first permanently
    winning the shape. depthwise-direct's own output matches stock's
    formula exactly (unlike fftconv's), so reusing fftconv's looser
    tolerance for it is safe -- a looser bar only ever accepts a *correct*
    candidate more readily, never incorrectly accepts a wrong one, since
    kernel_select's tolerance is strictly a floor for acceptance, not
    per-candidate. If fftconv is disabled but depthwise is eligible, the
    contest still runs with just {depthwise, stock} -- this function is
    the entry point for both, not just fftconv, despite the name (kept
    to avoid a wider rename touching every existing test)."""
    depthwise_eligible = (
        depthwise_conv1d_ops.available()
        and _is_depthwise_conv1d_eligible(input, weight, stride, dilation, groups)
    )
    if not fftconv_ops.fft_conv1d_enabled() and not depthwise_eligible:
        return None
    s, p, d = _scalar(stride), _scalar(padding), _scalar(dilation)
    candidates = []
    if fftconv_ops.fft_conv1d_enabled():
        candidates.append(
            ("fftconv", lambda: fftconv_ops.fftconv1d_candidate(input, weight, bias, s, p, d, groups)))
    if depthwise_eligible:
        candidates.append(
            ("depthwise", lambda: depthwise_conv1d_ops.depthwise_conv1d_candidate(
                input, weight, bias, stride, padding, dilation, groups)))
    candidates.append(("stock", lambda: orig_fn(input, weight, bias, stride, padding, dilation, groups)))
    return kernel_select.pick("conv1d", input, weight, stride, padding, dilation, candidates,
                               tolerance=fftconv_ops.fftconv_tolerance(input.dtype))


def _try_sparse_conv1d_fastpath(
    input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
    stride: Any, padding: Any, dilation: Any, groups: int,
) -> Optional[torch.Tensor]:
    """Sparse-pixel fast path, tried before MIOpen for any groups=1,
    grad-safe conv1d call -- flexgemm_ops.maybe_sparse_conv1d only
    actually engages (rather than declining back to None) when `input` is
    mostly empty, per its own occupancy-threshold design
    (AMD_TUNED_TORCH_SPARSE_CONV1D=0 to disable entirely; see
    flexgemm_ops.py's module docstring for the full design shared with the
    conv2d/conv3d versions). groups!=1 is out of scope -- flexgemm_ops.
    sparse_conv1d has no notion of grouped convolution, unlike the
    depthwise-only causal fast path above (the one grouped case this
    module handles). Works on CPU as well as ROCm (the pure-Python
    fallback inside maybe_sparse_conv1d has no device requirement), same
    as this module's other checks not gating on input.is_cuda before the
    MIOpen-specific retry logic further down.

    The _grad_safe check is REQUIRED, not defensive: flexgemm_ops's sparse
    conv path has no backward pass at all -- see _grad_safe's own
    docstring above for exactly what silently breaks without this check."""
    if groups != 1 or not _grad_safe(input, weight, bias):
        return None
    return flexgemm_ops.maybe_sparse_conv1d(
        input, weight, bias, stride=(_scalar(stride),), padding=(_scalar(padding),),
        dilation=(_scalar(dilation),))


def _miopen_safe_conv(op_name: str, orig_fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Build a MIOpen-failure-tolerant wrapper around a torch.nn.functional
    convN op. See the module docstring for the retry-then-CPU-fallback
    sequence; op_name is only used to label the printed diagnostics."""

    def wrapped(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                stride: Any = 1, padding: Any = 0, dilation: Any = 1, groups: int = 1) -> torch.Tensor:
        fast = _try_causal_conv1d_fastpath(input, weight, bias, stride, padding, dilation, groups)
        if fast is not None:
            return fast
        fftconv = _try_fftconv1d_fastpath(input, weight, bias, stride, padding, dilation, groups, orig_fn)
        if fftconv is not None:
            return fftconv
        sparse = _try_sparse_conv1d_fastpath(input, weight, bias, stride, padding, dilation, groups)
        if sparse is not None:
            return sparse
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
