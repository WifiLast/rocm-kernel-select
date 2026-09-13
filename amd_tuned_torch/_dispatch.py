"""amd_tuned_torch._dispatch -- the always-on monkeypatch tier enable()/
disable() installs: linear/matmul/bmm (aiter/hipBLASLt/CK GEMM contest),
conv2d/conv3d (native HIP/CK/FFT-conv contest), group_norm (native HIP/CK
contest), and the TransformerEngine-backed layer_norm/rms_norm/gelu/silu/
scaled_dot_product_attention wrappers -- plus the shared eligibility/
install-and-restore machinery every one of those wrappers is built from,
enable()/disable()/is_enabled() themselves, and the TorchScript
(torch.jit.script/trace) compatibility guard enable()'s patches require
(see that section's own comment block below for why).

Split out of amd_tuned_torch/__init__.py (previously one ~2300-line file)
purely for readability -- no behavior changed by this split. Every name
defined here is re-exported at the package's top level
(`from ._dispatch import *` in __init__.py, with this module's __all__
below listing every name including the underscore-prefixed ones tests
reach into directly, e.g. amd_tuned_torch._grad_safe/_usable/_ORIGINALS/
_GROUPNORM_DTYPES), so `amd_tuned_torch.X` resolves exactly as it did
before the split, for every X defined here.

See amd_tuned_torch/_opt_in_tiers.py for everything NOT installed by
enable()/disable(): opt-in-only kernel tiers (INT8 linear, conv3d
Winograd, flash_attn_rocwmma, triton_kernels rmsnorm) and the standalone
never-monkeypatched ops (mla_decode, fused_add_rms_norm, rotary_embedding,
fused_linear_cross_entropy, bias_swiglu, SmoothQuant calibration).
"""
from __future__ import annotations

import math
import os
from typing import Any, Callable

import torch
import torch.nn.functional as F

from . import _C
from . import aiter_ops
from . import hipblaslt_ops
from . import ck_ops
from . import ck_gemm_ops
from . import ck_norm_ops
from . import te_ops
from . import compile_ops
from . import kernel_select
from . import flexgemm_ops
from . import rocsparse_ops
from . import fftconv_ops
from . import flash_mm_kernel
from . import fftconv_calibration
from . import splitk_gemm_ops

# Raw, unpatched access to the native HIP group_norm kernel for manual use
# regardless of what enable() patches. (linear/matmul/bmm live in
# aiter_ops, not here -- see amd_tuned_torch.aiter_ops.linear_fp16/bmm_fp16.)
ops = _C

_GEMM_DTYPES = (torch.float16, torch.bfloat16)              # linear/matmul/bmm: aiter Triton fp16/bf16 only
_GROUPNORM_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_TE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_CONV_NATIVE_DTYPES = (torch.float16, torch.float32)  # conv2d/conv3d native HIP kernels: no bf16 support
# Composable Kernel's WMMA conv instances: fp16/bf16 (no fp32 -- gfx1100's
# WMMA units have no fp32 mode). bf16 is the important half of that: it is
# the one conv case where stock ROCm has no good solver at all (see
# ck_ops.py's measured table).
_CK_CONV_DTYPES = (torch.float16, torch.bfloat16)
# FFT-conv (amd_tuned_torch.fftconv_ops) as a conv2d/conv3d tier: fp32
# included, unlike the CK/native tiers above. This is not a WMMA kernel --
# the transform runs in float32 for every input dtype anyway (see
# fft_conv's MIXED PRECISION docstring section), so fp32 input costs it
# nothing extra, and fp16/bf16 buy less here than elsewhere.
_FFTCONV_CONV_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# Kernel width at/above which the FFT tier is worth CONTESTING (not
# preferring -- kernel_select still measures it against stock/CK/native).
# See _fftconv_conv_candidate's docstring for the measured crossovers
# these defaults come from, and why 2D's sits an order of magnitude
# higher in kernel width than 3D's.
_FFTCONV_CONV2D_ENABLED = os.environ.get("AMD_TUNED_TORCH_FFTCONV2D", "1") != "0"
_FFTCONV_CONV3D_ENABLED = os.environ.get("AMD_TUNED_TORCH_FFTCONV3D", "1") != "0"
_FFTCONV_CONV2D_MIN_KERNEL = int(os.environ.get("AMD_TUNED_TORCH_FFTCONV2D_MIN_KERNEL", "32"))
_FFTCONV_CONV3D_MIN_KERNEL = int(os.environ.get("AMD_TUNED_TORCH_FFTCONV3D_MIN_KERNEL", "7"))
# Measured calibration (amd_tuned_torch.fftconv_calibration, written by
# tools/benchmark_fftconv3d_min_positions.py) for the min_positions guess
# below -- same "loaded once, before the constant it feeds is defined"
# shape as flexgemm_ops.py's own _sparse_conv_calibration/_calibrated_default.
# Precedence per env var: explicit env var (checked at the os.environ.get
# call site below) > this measured calibration > the hardcoded guess as a
# last resort for a GPU/build that hasn't been benchmarked yet.
_fftconv_calibration_data = fftconv_calibration.load()


def _calibrated_fftconv_default(dim_key: str, field: str, hardcoded_default: str) -> str:
    """Same contract as flexgemm_ops._calibrated_default: the env-var
    default for one (dim_key, field) pair, as a string (feeds straight into
    os.environ.get(name, default)) -- a measured value if
    fftconv_calibration loaded one, else `hardcoded_default`. Only ever
    decides what the *default* is, never overrides an explicit env var."""
    value = _fftconv_calibration_data.get(dim_key, {}).get(field)
    return str(value) if value is not None else hardcoded_default


# conv3d only, not conv2d: FFT-conv3d's padded transform is the most
# memory-hungry tier in this file (measured ~1.5GB peak on a B=8 C=512
# L=8192 conv1d whose inputs are only 134MB -- see _try_fftconv1d_fastpath's
# OOM note), and for a small input volume there's no reasonable kernel width
# for which paying that fixed transform/allocation overhead beats a cheap
# direct conv outright, wide kernel or not. min(weight.shape[2:]) above only
# gates on KERNEL size; this gates on `input`'s own spatial size (batch *
# every dim after channel, same quantity flexgemm_ops._n_spatial_positions
# computes for its sparse-conv gate, not imported from there to avoid a
# cross-module dependency for one line). "2048" is a hardcoded fallback
# guess, used only until tools/benchmark_fftconv3d_min_positions.py has
# measured this GPU's actual crossover (see _calibrated_fftconv_default
# above) -- override per-process via AMD_TUNED_TORCH_FFTCONV3D_MIN_POSITIONS.
_FFTCONV_CONV3D_MIN_POSITIONS = int(
    os.environ.get("AMD_TUNED_TORCH_FFTCONV3D_MIN_POSITIONS",
                    _calibrated_fftconv_default("conv3d", "min_positions", "2048")))
# silu excludes fp32: benchmarked slower than stock (0.80x, see
# benchmark.json) even though fp16 is a real win (1.15x) -- fp32 stays on
# the TE path's fp16/bf16-only sibling dtypes instead of falling back per-call.
_SILU_DTYPES = (torch.float16, torch.bfloat16)

_ORIGINALS: dict[tuple[Any, str], Callable] = {}
_ENABLED = False

# Resolved once at import time, not on every call -- _grad_safe runs on
# every single patched op invocation (it's the first check in the hottest
# of hot paths: F.linear/matmul/bmm/conv2d/conv3d/group_norm), so re-doing
# a hasattr() lookup on the torch module there for the life of the process
# is a pure waste.
_HAS_INFERENCE_MODE = hasattr(torch, "is_inference_mode_enabled")


def _is_compiling() -> bool:
    """True while Dynamo is tracing. Guarded because the API moved across
    PyTorch versions and this must never raise on the hot path."""
    try:
        return bool(torch.compiler.is_compiling())
    except AttributeError:
        try:
            return bool(torch._dynamo.is_compiling())
        except Exception:
            return False


def _grad_safe(*tensors: Any) -> bool:
    """True if swapping in a kernel with no autograd support is safe. Only
    relevant to linear/matmul/bmm/conv2d (aiter Triton kernels) and
    group_norm (native HIP kernel) -- the TE-backed ops all have real
    backward passes and don't need this check."""
    if _HAS_INFERENCE_MODE and torch.is_inference_mode_enabled():
        return True
    if not torch.is_grad_enabled():
        return True
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.requires_grad:
            return False
    return True


# Types this package's kernels can actually be handed. Every accelerated
# path below eventually reaches a HIP kernel or a torch.library custom op
# that reads a dense tensor's storage directly, so the argument has to BE a
# dense tensor -- not something that merely quacks like one.
#
# torch.Tensor SUBCLASSES do not qualify, and `isinstance` cannot tell the
# difference. A quantized weight (optimum.quanto's QBytesTensor, as reached
# through mmgp's quant_router) is a Tensor subclass that reports
# `.is_cuda == True` and a `.dtype` of float16 -- the DEQUANTIZED dtype --
# so it passed the old isinstance/dtype gate untouched and was handed
# straight to compile_ops.ck_gemm_linear. What happens then is not a clean
# decline: the custom op dispatches into quanto's __torch_dispatch__, which
# does not recognize it, falls back to dequantizing every argument, and the
# whole thing ends in an illegal memory access rather than an exception any
# of the fallback paths here could have caught.
#
# Declining sends the call to the stock op, which is exactly where such a
# tensor wants to go: quanto's own __torch_function__ intercepts F.linear
# and runs its proper quantized path. The same reasoning covers every other
# subclass with custom dispatch semantics -- DTensor, FakeTensor, functorch
# wrappers -- so the test is an exact type match rather than a blocklist of
# the ones seen so far.
_PLAIN_TENSOR_TYPES = (torch.Tensor, torch.nn.Parameter)


def _is_plain_tensor(t: Any) -> bool:
    """True only for a dense tensor this package's kernels can read."""
    return type(t) in _PLAIN_TENSOR_TYPES


def _usable(*tensors: Any, dtypes: tuple = _GEMM_DTYPES) -> bool:
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            continue
        if not _is_plain_tensor(t):
            return False  # a Tensor subclass -- see _PLAIN_TENSOR_TYPES above
        if not t.is_cuda or t.dtype not in dtypes:
            return False
    return True


def _install(target: Any, name: str, wrapper: Callable) -> None:
    key = (target, name)
    if key not in _ORIGINALS:
        _ORIGINALS[key] = getattr(target, name)
    setattr(target, name, wrapper)


def _restore(target: Any, name: str) -> None:
    original = _ORIGINALS.pop((target, name), None)
    if original is not None:
        setattr(target, name, original)


# Small standalone helper, used by _patched_conv3d below (sparse-voxel
# fast path's stride/padding/dilation normalization) and by the conv3d
# Winograd opt-in tier further down this package -- kept here, with the
# other small dispatch helpers, rather than moved with the Winograd code
# it was originally adjacent to, since a name only defined in the opt-in
# module would leave this file's own _patched_conv3d importing from a
# module that in turn imports helpers back from this one.
def _triple(v):
    return (v, v, v) if isinstance(v, int) else tuple(v)


# ---------------------------------------------------------------------------
# linear/matmul/bmm/conv2d: aiter Triton kernels (no autograd -- grad-safety
# checked up front, plus a RuntimeError/AssertionError fallback for
# unsupported shapes).
# ---------------------------------------------------------------------------
#
# TRAINING-TIME GAP. _grad_safe() falling back to stock the instant any
# input requires_grad was, for a long time, assumed to only matter for
# occasional no_grad-adjacent calls. Measured on a real SDXL LoRA training
# run (MIOpen verbose log, gfx1100, character-mode LoKr on to_q/to_k/to_v),
# it is not occasional: once ANY adapter anywhere in the model needs a
# gradient, autograd must be able to backprop through every frozen layer
# between it and the loss, so nearly every activation in the whole forward
# pass ends up requires_grad=True. _grad_safe() then declines for
# essentially every linear/matmul/bmm/conv2d call in the network, not a few
# of them -- this tier was, in practice, inference-only despite nothing
# about it being conceptually limited to inference.
#
# _LinearFn/_BmmFn below close that gap for linear/matmul/bmm specifically
# (matmul reuses _BmmFn -- _patched_matmul already reshapes every shape
# family it accepts down to a bmm problem before the contest even runs, see
# its own docstring, so its backward is exactly _BmmFn's too, composed with
# the reshape/unsqueeze ops' own already-correct stock backward) -- conv2d's
# backward is not reducible to calling conv2d again the same simple way,
# see _patched_conv2d's own comment -- by wrapping the
# existing accelerated forward candidates in a torch.autograd.Function
# whose backward is the textbook GEMM derivative (dA = dY @ B^T, dB =
# A^T @ dY), computed with plain `@`. That's it -- no new kernel, no new
# numerically-unverified code path: autograd disables grad-tracking during
# a Function's own backward() (nothing here calls torch.enable_grad()), so
# _grad_safe() itself reports those backward matmuls as safe too, meaning
# this module's own linear/matmul/bmm patches transparently accelerate the
# backward pass as well, through the exact same kernel_select contest as
# the forward pass. Forward is the only place a kernel *choice* is made;
# backward is just correct math using operators this project didn't write.


class _LinearFn(torch.autograd.Function):
    """Makes _patched_linear's accelerated forward usable when a gradient
    is needed -- see the TRAINING-TIME GAP note above for the full
    rationale. fwd_fn is _patched_linear's own accelerated-candidate
    selection, called unchanged; only the backward is new here."""

    @staticmethod
    def forward(ctx, input, weight, bias, fwd_fn):
        ctx.save_for_backward(input, weight)
        ctx.has_bias = bias is not None
        return fwd_fn(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        # F.linear itself accepts any input.dim() >= 1 (batched over every
        # leading dim) but the GEMM math is 2D -- flatten every leading dim
        # into one, matching how _patched_linear's own kernel_select key
        # already treats "input.numel() // input.size(-1)" as the M dim.
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        input_2d = input.reshape(-1, input.shape[-1])
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = (grad_output_2d @ weight).reshape(input.shape)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output_2d.t() @ input_2d
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output_2d.sum(0)
        return grad_input, grad_weight, grad_bias, None


class _BmmFn(torch.autograd.Function):
    """Makes _patched_bmm's (and, via _patched_matmul, its reshaped-to-3D
    reuse of the same contest) accelerated forward usable when a gradient
    is needed. Y = A @ B, both always exactly 3D (batch, M, K) @ (batch, K,
    N) by the time this is called -- _patched_matmul reshapes every shape
    family it accepts (2D, 3D, >=4D) down to this via ordinary
    unsqueeze/reshape before ever reaching here. Those are themselves
    differentiable stock ops, so calling this on their output composes
    correctly with no extra work: autograd chains this Function's backward
    straight into unsqueeze/reshape's own (already correct) backward to
    produce a gradient in the original, un-reshaped shape."""

    @staticmethod
    def forward(ctx, a, b, fwd_fn):
        ctx.save_for_backward(a, b)
        return fwd_fn(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_a = grad_output @ b.transpose(-2, -1)
        if ctx.needs_input_grad[1]:
            grad_b = a.transpose(-2, -1) @ grad_output
        return grad_a, grad_b, None


def _patched_linear(input, weight, bias=None):
    """F.linear over a per-shape contest: stock, hipBLASLt, CK GEMM.

    This op had no contest until hipblaslt_ops and ck_gemm_ops existed,
    because aiter's Triton GEMM -- the only other candidate -- raises
    KeyError('gfx1100'), leaving nothing to compete against stock. It has
    two now, and measurement says neither of them wins outright. Against
    stock (rocBLAS = 1.00x) on RX 7900 XTX, fp16:

                                 hipBLASLt   CK GEMM
        4096^3                     1.29x      1.20x
        8192x4096x11008            1.09x      1.11x
        4096x4096x1024             1.49x      1.47x
        4096x1280x1280             0.96x      1.14x
        1024^3                     1.02x      1.21x
        1x4096x4096 (decode)       0.60x      0.90x

    hipBLASLt owns the large end and collapses on single-token decode --
    0.60x, where its heuristic picks a throughput kernel that never gets to
    use its throughput. CK is flatter and ahead everywhere hipBLASLt is
    behind. Two rows are outright regressions, which is exactly why this
    goes through kernel_select rather than a fixed tier order: stock is an
    ordinary candidate and wins the shapes it deserves to win.

    A fourth candidate, splitk_gemm_ops (see that module's docstring),
    targets exactly the row both existing tiers regress on: single-token/
    small-batch decode, where a plain tiled GEMM has too few output tiles
    to occupy this card's CUs regardless of per-tile throughput. It splits
    the K-reduction across more thread blocks instead, and -- like every
    candidate here -- only ever wins if kernel_select's own numerical
    verification against stock actually agrees; unmeasured on real
    hardware, so it may simply always lose its own contest here, which is
    the safe failure mode this whole module exists to guarantee.
    """
    orig = _ORIGINALS[(F, "linear")]
    if not _usable(input, weight):
        return orig(input, weight, bias)
    if input.dim() < 2 or weight.dim() != 2:
        return orig(input, weight, bias)

    def _accelerated(input, weight, bias):
        if kernel_select.enabled():
            # Keyed on the GEMM's actual identity (M, K, N and dtype), not
            # on the caller's leading dims: [4, 128, K] and [512, K] are one
            # problem to every candidate here, and keying on the raw shape
            # would re-run the contest for each of them.
            key = (input.dtype, int(input.numel() // input.size(-1)),
                   tuple(weight.shape), bias is not None)
            won = kernel_select.cached_key("linear", key)
            if won == "stock":
                return orig(input, weight, bias)
            if won is None:
                out = kernel_select.pick_key("linear", key, [
                    ("hipblaslt", lambda: compile_ops.hipblaslt_linear(
                        input, weight, bias,
                        hipblaslt_ops.EPILOGUE_BIAS if bias is not None
                        else hipblaslt_ops.EPILOGUE_NONE)),
                    ("ck_gemm", lambda: compile_ops.ck_gemm_linear(
                        input, weight, bias,
                        ck_gemm_ops.EPILOGUE_BIAS if bias is not None
                        else ck_gemm_ops.EPILOGUE_NONE)),
                    # Decode-shaped (small-M) candidate only -- see
                    # splitk_gemm_ops.py's module docstring for why
                    # hipBLASLt/CK both regress at M=1 and what this does
                    # about it. Declines (returns None) for any M outside
                    # its target range, same as every other candidate here.
                    ("splitk", lambda: splitk_gemm_ops.linear(input, weight, bias)),
                    ("aiter", lambda: _linear_aiter(input, weight, bias)),
                    ("stock", lambda: orig(input, weight, bias)),
                ])
                if out is not None:
                    return out
            elif won == "hipblaslt":
                # compile_ops.hipblaslt_linear/ck_gemm_linear (not
                # hipblaslt_ops.linear/ck_gemm_ops.linear directly): this is
                # the hot path a cached contest winner takes on every
                # subsequent call, i.e. the actual monkeypatched-F.linear
                # call torch.compile would trace through -- see
                # compile_ops.py's own docstring for why it wraps these two
                # as torch.library custom ops.
                out = compile_ops.hipblaslt_linear(input, weight, bias,
                                           hipblaslt_ops.EPILOGUE_BIAS if bias is not None
                                           else hipblaslt_ops.EPILOGUE_NONE)
                if out is not None:
                    return out
            elif won == "ck_gemm":
                out = compile_ops.ck_gemm_linear(input, weight, bias,
                                         ck_gemm_ops.EPILOGUE_BIAS if bias is not None
                                         else ck_gemm_ops.EPILOGUE_NONE)
                if out is not None:
                    return out
            elif won == "splitk":
                out = splitk_gemm_ops.linear(input, weight, bias)
                if out is not None:
                    return out

        out = _linear_aiter(input, weight, bias)
        return orig(input, weight, bias) if out is None else out

    if _grad_safe(input, weight, bias):
        return _accelerated(input, weight, bias)
    # A gradient is needed -- route through _LinearFn so the accelerated
    # forward above still runs, with a real (textbook, stock-matmul-based)
    # backward instead of skipping acceleration entirely. See the
    # TRAINING-TIME GAP note above _LinearFn's definition.
    return _LinearFn.apply(input, weight, bias, _accelerated)


def _linear_aiter(input, weight, bias):
    """The pre-existing aiter path, as a candidate that may decline.

    Returns None rather than raising so it loses the contest instead of
    breaking the call -- which on gfx1100 is what always happens, since
    aiter ships no RDNA3 tuning config.
    """
    try:
        return compile_ops.linear_fp16(input, weight, bias)
    except (RuntimeError, TypeError, AssertionError, KeyError):
        return None


def _patched_matmul(input, other, *, out=None):
    """torch.matmul over the SAME hipBLASLt/aiter/stock contest _patched_bmm
    already uses, for the batched cases matmul reduces to a bmm problem: 2D
    @ 2D, 3D @ 3D, and >=4D @ >=4D where both operands share the exact same
    batch shape (no broadcasting -- neither hipBLASLt's nor aiter's batched
    kernels have broadcast semantics of their own). Every candidate,
    including stock, operates on the reshaped-to-3D view; the result is
    reshaped back to matmul's own shape convention once, after the contest
    has already picked a winner, so the contest itself only ever has to
    reason about one shape family (matching _patched_bmm's own key).

    This used to call aiter directly with no fallback to hipBLASLt/CK at
    all -- the same gap _patched_linear/_patched_bmm closed once hipBLASLt
    and CK GEMM existed as real candidates, just never closed here too.

    STRUCTURED-FACTOR OPERANDS. `other` may also be butterfly or Monarch
    FACTORS (a list/tuple/ParameterList of weight tensors), in which case the
    call is routed to flash_mm_kernel -- see the block comment on that branch
    below for why only the list form is intercepted."""
    orig = _ORIGINALS[(torch, "matmul")]

    # ---- structured factors (butterfly / Monarch) -------------------------
    # flash_mm_kernel.matmul is deliberately spelled like torch.matmul, but
    # nothing routed to it until now: it had to be called by name. This makes
    # `torch.matmul(x, factors)` reach it.
    #
    # ONLY THE LIST/TUPLE FORM IS INTERCEPTED, AND THAT IS NOT AN OVERSIGHT.
    # torch.matmul rejects a list outright today ("argument 'other' must be
    # Tensor, not list"), so treating one as factors cannot change the result
    # of any call that works at present -- it only gives meaning to a call
    # that currently raises. A PRE-STACKED (E, L, 2) factor tensor is left
    # alone on purpose: that is a perfectly valid 3D operand for an ordinary
    # batched matmul, and hijacking it on shape alone would silently change
    # the meaning of real dense calls. Callers holding stacked factors should
    # call flash_mm_kernel.matmul directly.
    #
    # A DENSE `other` CANNOT BE ROUTED HERE AT ALL. Going from a dense (L, L)
    # matrix to butterfly factors is a factorization problem, not a reshape,
    # and most matrices admit no exact butterfly factorization -- so there is
    # no version of this that accelerates ordinary dense torch.matmul calls.
    # The structured path is only ever reachable when the caller already
    # holds its weights factored.
    #
    # is_eligible() is the non-raising precondition check (right factor
    # count, power-of-2 L for butterfly, matching in_dims product for
    # Monarch); anything it declines falls through to `orig`, which raises
    # the same TypeError it always did.
    if isinstance(other, (list, tuple, torch.nn.ParameterList)):
        if isinstance(input, torch.Tensor) and flash_mm_kernel.is_eligible(input, other):
            return flash_mm_kernel.matmul(input, other, out=out)
        return orig(input, other, out=out) if out is not None else orig(input, other)

    # Sparse-input fast path: tests `input`'s LAYOUT (torch.sparse_csr/csc/
    # coo vs. the ordinary torch.strided every other branch below assumes),
    # not its content -- unlike flexgemm_ops's occupancy-gated conv2d/conv3d
    # fast paths, there is no reduction to pay here, a sparse tensor's own
    # layout already says so. See rocsparse_ops.py's module docstring for
    # what's supported (2D CSR/CSC/COO sparse @ 2D dense only) and why.
    # Checked FIRST, before _usable/the reshape-to-3D dance below: those
    # assume a plain strided tensor (.unsqueeze/.reshape on a sparse CSR
    # tensor is not the batching operation they need it to be), so a sparse
    # `input`/`other` must never reach them either way, whether or not
    # rocSPARSE ends up handling the call. Same _grad_safe gate
    # flexgemm_ops's sparse conv fast paths need for the same reason:
    # rocsparse_spmm has no backward, so this only fires when no gradient is
    # needed -- otherwise stock's own (correct, autograd-aware) sparse
    # matmul handles it instead, same as if rocSPARSE were unavailable.
    #
    # rocSPARSE vs. stock is MEASURED, not assumed, exactly like every other
    # tier in this file (conv2d/conv3d's kernel_select contest, fftconv's
    # -- see that section's own comment for why this project doesn't trust
    # "our kernel is a real vendor library" as a substitute for a benchmark
    # here either: ATen's own generic sparse dispatch on ROCm may or may not
    # be the naive fallback rocSPARSE would beat). Keyed on (dtype, both
    # operands' layout, both shapes) rather than shared with the dense
    # (a3.dtype, a3.shape, b3.shape) key the contest below uses -- a sparse
    # `input`'s layout changes which candidates even apply, so it needs its
    # own kernel_select "kind" and key shape rather than colliding with the
    # dense matmul contest's cache entries.
    if out is None and isinstance(input, torch.Tensor) and isinstance(other, torch.Tensor) and (
            input.layout != torch.strided or other.layout != torch.strided):
        if _grad_safe(input, other):
            if kernel_select.enabled():
                key = (input.dtype, str(input.layout), str(other.layout),
                       tuple(input.shape), tuple(other.shape))
                won = kernel_select.cached_key("matmul_sparse", key)
                if won is None:
                    _out = kernel_select.pick_key("matmul_sparse", key, [
                        ("rocsparse", lambda: rocsparse_ops.maybe_spmm(input, other)),
                        ("stock", lambda: orig(input, other)),
                    ])
                    if _out is not None:
                        return _out
                elif won == "rocsparse":
                    _out = rocsparse_ops.maybe_spmm(input, other)
                    if _out is not None:
                        return _out
                # won == "stock", or a cached "rocsparse" winner declined this
                # call (e.g. AMD_TUNED_TORCH_ROCSPARSE_SPMM=0 toggled off since
                # it was cached) -- either way, stock below is always correct.
            else:
                _sparse_out = rocsparse_ops.maybe_spmm(input, other)
                if _sparse_out is not None:
                    return _sparse_out
        return orig(input, other)

    if out is not None or not _usable(input, other):
        return orig(input, other) if out is None else orig(input, other, out=out)
    if input.dtype != other.dtype:
        return orig(input, other)

    if input.dim() == 2 and other.dim() == 2:
        a3, b3 = input.unsqueeze(0), other.unsqueeze(0)
        def _reshape_back(o3): return o3.squeeze(0)
    elif input.dim() == 3 and other.dim() == 3:
        a3, b3 = input, other
        def _reshape_back(o3): return o3
    elif input.dim() == other.dim() >= 4 and input.shape[:-2] == other.shape[:-2]:
        batch_shape = input.shape[:-2]
        a3 = input.reshape(-1, *input.shape[-2:])
        b3 = other.reshape(-1, *other.shape[-2:])
        def _reshape_back(o3): return o3.view(*batch_shape, *o3.shape[-2:])
    else:
        return orig(input, other)

    # matmul on the reshaped 3D tensors is bmm's own semantics exactly (no
    # broadcasting possible once both operands are forced to the same
    # leading batch dim), so `orig` applied here is a safe, cheap stock
    # reference -- no need to reach for a separate "true" matmul call on
    # the un-reshaped tensors.
    def _stock3(a3, b3): return orig(a3, b3)

    def _accelerated(a3, b3):
        if kernel_select.enabled():
            key = (a3.dtype, tuple(a3.shape), tuple(b3.shape))
            won = kernel_select.cached_key("matmul", key)
            if won == "stock":
                return _stock3(a3, b3)
            if won == "hipblaslt":
                # compile_ops.hipblaslt_bmm, not hipblaslt_ops.bmm directly
                # -- same reasoning as _patched_linear's cached-winner
                # branches above.
                got = compile_ops.hipblaslt_bmm(a3, b3)
                if got is not None:
                    return got
            elif won is None:
                got = kernel_select.pick_key("matmul", key, [
                    ("hipblaslt", lambda: compile_ops.hipblaslt_bmm(a3, b3)),
                    ("aiter", lambda: _bmm_aiter(a3, b3)),
                    ("stock", lambda: _stock3(a3, b3)),
                ])
                if got is not None:
                    return got

        got = _bmm_aiter(a3, b3)
        return _stock3(a3, b3) if got is None else got

    if _grad_safe(input, other):
        return _reshape_back(_accelerated(a3, b3))
    # See _LinearFn/_patched_linear's identical reasoning. a3/b3 are
    # themselves ordinary autograd-tracked views of input/other (unsqueeze
    # or reshape), so _BmmFn's backward composes automatically with their
    # own (stock, already-correct) backward -- nothing extra needed for the
    # 2D/>=4D reshape cases here.
    return _reshape_back(_BmmFn.apply(a3, b3, _accelerated))


def _patched_bmm(input, mat2, *, out=None):
    """torch.bmm over a per-shape contest: hipBLASLt, aiter, stock.

    Same reasoning as _patched_linear -- hipBLASLt is a real candidate on
    this card where aiter is not -- but with one candidate fewer: the CK
    GEMM tier compiles no batched instances. That was a build-cost decision
    (see src/cuda/ck_gemm_fwd.hpp), taken because hipBLASLt already covers
    batched shapes well; the attention-shaped GEMM it exists for measures
    1.49x fp16 / 1.73x bf16 against stock.
    """
    orig = _ORIGINALS[(torch, "bmm")]
    if out is not None or not _usable(input, mat2):
        return orig(input, mat2) if out is None else orig(input, mat2, out=out)
    if input.dtype != mat2.dtype:
        return orig(input, mat2)

    def _accelerated(input, mat2):
        if kernel_select.enabled():
            key = (input.dtype, tuple(input.shape), tuple(mat2.shape))
            won = kernel_select.cached_key("bmm", key)
            if won == "stock":
                return orig(input, mat2)
            if won == "hipblaslt":
                # compile_ops.hipblaslt_bmm, not hipblaslt_ops.bmm directly
                # -- same reasoning as _patched_linear's cached-winner
                # branches above.
                got = compile_ops.hipblaslt_bmm(input, mat2)
                if got is not None:
                    return got
            elif won is None:
                got = kernel_select.pick_key("bmm", key, [
                    ("hipblaslt", lambda: compile_ops.hipblaslt_bmm(input, mat2)),
                    ("aiter", lambda: _bmm_aiter(input, mat2)),
                    ("stock", lambda: orig(input, mat2)),
                ])
                if got is not None:
                    return got

        got = _bmm_aiter(input, mat2)
        return orig(input, mat2) if got is None else got

    if _grad_safe(input, mat2):
        return _accelerated(input, mat2)
    # See _LinearFn/_patched_linear's identical reasoning -- _BmmFn's
    # backward is the textbook bmm derivative, computed with plain `@`.
    return _BmmFn.apply(input, mat2, _accelerated)


def _bmm_aiter(input, mat2):
    """The pre-existing aiter path, as a candidate that may decline."""
    try:
        return compile_ops.bmm_fp16(input, mat2)
    except (RuntimeError, TypeError, AssertionError, KeyError):
        return None


def _is_pointwise_conv2d(weight, stride, padding, dilation) -> bool:
    """True for a 1x1/stride1/pad0/dilation1 conv2d -- a pure per-pixel
    channel-mixing GEMM (Y = W @ X over channels) with no spatial reduction
    at all, unlike every other conv2d shape this module handles.

    Measured directly on RX 7900 XTX (see miopen_amd_log.txt): for a
    512->256ch, 1024x512 fp16 1x1 conv, MIOpen's own rocBLAS strided-batched
    GEMM solver (GemmFwd1x1_0_1) took 1.69ms, beating its own best Winograd/
    asm-direct kernel (5.74ms, 3.4x slower) and its naive direct kernel
    (759.9ms, 450x slower). rocBLAS wins this by such a wide margin because
    its GEMM primitive consumes the input's existing NCHW strides directly
    (channel-major, spatially-contiguous is already a valid (C_out,C_in) @
    (C_in, H*W) matmul via lda/ldb) with zero data movement -- neither of
    this module's own tiers can match that for K=1: the native im2col-tiled
    kernel below has no spatial window to exploit, and aiter's Triton WMMA
    GEMM is a channels-last kernel that would need an actual NCHW->NHWC
    transpose copy first. So this case is deliberately left on stock rather
    than routed through either tier."""
    return (tuple(weight.shape[2:]) == (1, 1)
            and tuple(aiter_ops._pair(stride)) == (1, 1)
            and tuple(aiter_ops._pair(padding)) == (0, 0)
            and tuple(aiter_ops._pair(dilation)) == (1, 1))


def _fftconv_conv_candidate(input, weight, bias, stride, padding, dilation, groups,
                             *, ndim: int):
    """The FFT-conv tier (amd_tuned_torch.fftconv_ops) as a (name, thunk)
    kernel_select candidate for conv2d/conv3d, or None when this call
    isn't worth contesting it for at all.

    WHY A PRE-FILTER AND NOT JUST ANOTHER CANDIDATE. FFT-conv's
    O(N log N)-per-spatial-dim win only arrives once the kernel is wide
    enough to pay for the transform (see fftconv_ops' own module
    docstring). Measured on gfx1100 against stock, fp32:

      conv2d 4x32x256x256   7x7  1.31ms -> 18.79ms  (14.3x LOSS)
                           15x15  2.50ms -> 16.93ms  ( 6.8x loss)
                           31x31 10.17ms -> 19.59ms  ( 1.9x loss)
                           63x63 38.65ms -> 38.85ms  (even)
      conv3d 1x32x32x64x64   3^3  2.71ms -> 13.26ms  ( 4.9x loss)
                             7^3 33.76ms -> 30.24ms  ( 1.1x win)
                            15^3 6907ms  -> 38.77ms  ( 178x WIN)

    Direct convolution costs K^ndim multiplies per output element while
    the transform costs the same regardless, so 3D crosses over an order
    of magnitude lower in kernel width than 2D -- hence two separate
    thresholds rather than one shared "large kernel" number. Both sit far
    above the 3x3-style kernels this package's other conv tiers exist
    for, and offering the candidate down there would buy nothing while
    costing every distinct small shape one contest measurement plus a
    full padded-transform allocation. Same posture as
    fftconv_ops._FFTCONV1D_MIN_KERNEL on the conv1d path: a cheap
    pre-filter, NOT the win/lose decision, which stays measured per shape.

    None when the tier is switched off (AMD_TUNED_TORCH_FFTCONV2D=0 /
    AMD_TUNED_TORCH_FFTCONV3D=0), `input`/`weight` aren't `ndim`-spatial
    conv tensors, any spatial kernel extent is below the threshold,
    `input` is a small volume (conv3d only -- see
    AMD_TUNED_TORCH_FFTCONV3D_MIN_POSITIONS above this function), `padding`
    is one of F.convNd's string modes ("same"/"valid" -- the contest's
    shape key and fft_conv's own padding handling both want numbers), or
    the dtype/device isn't one this tier covers. The thunk itself returns
    None (declining, per kernel_select.pick's contract) rather than raising
    if the call fails -- including on OOM, which is a
    realistic outcome here specifically: the padded transform is by far
    the most memory-hungry tier in this file (measured ~1.5GB peak on a
    B=8 C=512 L=8192 conv1d whose inputs are 134MB), and a tier that
    can't fit should lose the contest, not kill the process."""
    if ndim == 2:
        if not _FFTCONV_CONV2D_ENABLED:
            return None
        min_kernel, fn = _FFTCONV_CONV2D_MIN_KERNEL, fftconv_ops.fft_conv2d
    else:
        if not _FFTCONV_CONV3D_ENABLED:
            return None
        min_kernel, fn = _FFTCONV_CONV3D_MIN_KERNEL, fftconv_ops.fft_conv3d
    if input.dim() != ndim + 2 or weight.dim() != ndim + 2:
        return None
    if ndim == 3 and input.shape[0] * math.prod(input.shape[2:]) < _FFTCONV_CONV3D_MIN_POSITIONS:
        return None
    if isinstance(padding, str):
        return None
    if min(weight.shape[2:]) < min_kernel:
        return None
    if not _usable(input, weight, dtypes=_FFTCONV_CONV_DTYPES):
        return None

    def _thunk():
        try:
            return fn(input, weight, bias=bias, padding=padding, stride=stride,
                      dilation=dilation, groups=groups)
        except (RuntimeError, ValueError, TypeError):
            return None

    return ("fftconv", _thunk)


def _fftconv_contest_tolerance(fft_candidate, input):
    """kernel_select verification tolerance for a conv contest that has the
    FFT tier in it, or None to keep kernel_select's per-dtype default for
    a contest that doesn't.

    Only the RELATIVE part differs here. FFT-conv doesn't merely round
    differently from direct convolution, it rounds BETTER -- measured
    against a float64 reference on a 1x8x1024 fp32 conv1d, at K=255
    direct is off by 2.6e-4 and FFT by 7.9e-5; at K=1023, 8.6e-4 vs
    1.5e-4 -- but it is a different algorithm, so its disagreement with
    direct convolution is larger than fp32's default rtol of 1e-4 admits
    (fftconv_ops.fftconv_tolerance carries the measured pairs and the
    reasoning). The ABSOLUTE part is no longer this function's problem:
    kernel_select._verify now scales its atol floor by the reference
    output's own RMS, from the reference tensor it already holds, which
    is both more accurate than estimating the output magnitude from the
    operands here and fixes the same defect for every other tier at once
    (see _verify's docstring for the three kernels a purely absolute atol
    was rejecting)."""
    if fft_candidate is None:
        return None
    return fftconv_ops.fftconv_tolerance(input.dtype)


def _conv2d_backward_data_stock(grad_output, weight, input_size, stride, padding, dilation, groups):
    """Reference (always-correct) conv2d backward-data via the same ATen op
    autograd itself would have used had this module never patched
    F.conv2d. `input` is passed as an uninitialized same-shape/dtype/device
    tensor -- grad_input is mathematically independent of the original
    input's VALUES (only grad_output/weight/stride/padding/dilation/groups
    matter), so nothing ever reads it; output_mask requests only this one
    output, matching the convention set by the CK candidate it's contested
    against (see _conv2d_backward_data)."""
    dummy_input = grad_output.new_empty(input_size)
    grad_input, _, _ = torch.ops.aten.convolution_backward(
        grad_output, dummy_input, weight, None,
        list(aiter_ops._pair(stride)), list(aiter_ops._pair(padding)),
        list(aiter_ops._pair(dilation)), False, [0, 0], groups, [True, False, False])
    return grad_input


def _conv2d_backward_weight_stock(input, grad_output, weight_size, stride, padding, dilation, groups):
    """Reference conv2d backward-weight -- see _conv2d_backward_data_stock,
    same reasoning mirrored for the weight-gradient output (dummy_weight's
    values are never read; grad_weight depends only on input/grad_output)."""
    dummy_weight = input.new_empty(weight_size)
    _, grad_weight, _ = torch.ops.aten.convolution_backward(
        grad_output, input, dummy_weight, None,
        list(aiter_ops._pair(stride)), list(aiter_ops._pair(padding)),
        list(aiter_ops._pair(dilation)), False, [0, 0], groups, [False, True, False])
    return grad_weight


def _conv2d_backward_data(grad_output, weight, input_size, stride, padding, dilation, groups):
    """dL/dInput for _Conv2dFn.backward: CK's WMMA backward-data device op
    (ck_ops.conv2d_backward_data) contested against stock via
    kernel_select, same "never assume our kernel wins" policy the forward
    tier already uses (see kernel_select.py's module docstring) -- CK's
    forward WMMA conv already loses to stock's Winograd solver for
    symmetric-channel/stride-1 fp16 shapes and only opens a real gap on the
    asymmetric-channel/stride-2 shapes MIOpen backward falls back to
    im2col+GEMM for (see the TRAINING-TIME GAP comment above
    _patched_conv2d), so a fixed preference either way would be wrong for
    some shape -- exactly what kernel_select measures instead of guessing."""
    stock = lambda: _conv2d_backward_data_stock(
        grad_output, weight, input_size, stride, padding, dilation, groups)
    if groups != 1 or not (ck_ops.available() and _usable(grad_output, weight, dtypes=_CK_CONV_DTYPES)):
        return stock()
    ck = lambda: ck_ops.conv2d_backward_data(grad_output, weight, input_size, stride, padding, dilation)
    if not kernel_select.enabled():
        out = ck()
        return out if out is not None else stock()
    key = ("conv2d_bwd_data", grad_output.dtype, tuple(grad_output.shape), tuple(weight.shape),
           tuple(aiter_ops._pair(stride)), tuple(aiter_ops._pair(padding)),
           tuple(aiter_ops._pair(dilation)))
    out = kernel_select.pick_key("conv2d_bwd_data", key, [("ck", ck), ("stock", stock)])
    return out if out is not None else stock()


def _conv2d_backward_weight(input, grad_output, weight_size, stride, padding, dilation, groups):
    """dL/dWeight for _Conv2dFn.backward -- see _conv2d_backward_data,
    same contest shape, CK's backward-weight device op vs stock."""
    stock = lambda: _conv2d_backward_weight_stock(
        input, grad_output, weight_size, stride, padding, dilation, groups)
    if groups != 1 or not (ck_ops.available() and _usable(input, grad_output, dtypes=_CK_CONV_DTYPES)):
        return stock()
    ck = lambda: ck_ops.conv2d_backward_weight(input, grad_output, weight_size, stride, padding, dilation)
    if not kernel_select.enabled():
        out = ck()
        return out if out is not None else stock()
    key = ("conv2d_bwd_weight", input.dtype, tuple(input.shape), tuple(grad_output.shape),
           tuple(weight_size), tuple(aiter_ops._pair(stride)), tuple(aiter_ops._pair(padding)),
           tuple(aiter_ops._pair(dilation)))
    out = kernel_select.pick_key("conv2d_bwd_weight", key, [("ck", ck), ("stock", stock)])
    return out if out is not None else stock()


class _Conv2dFn(torch.autograd.Function):
    """Makes _patched_conv2d's accelerated forward usable when a gradient
    is needed, with a REAL backward -- unlike _LinearFn/_BmmFn, conv2d's
    backward is not reducible to calling conv2d again the same simple way
    (see _patched_conv2d's TRAINING-TIME GAP comment: backward-data/
    backward-weight are algorithmically distinct convolution variants, not
    GEMMs), so this routes through CK's WMMA backward-data/backward-weight
    device ops (source/cmp_ext_turing/src/cuda/ck_conv_bwd*.{hpp,cu} via
    ck_ops.conv2d_backward_data/_weight) instead, each contested against
    PyTorch's own stock backward via kernel_select
    (_conv2d_backward_data/_weight above). Bias gradient is a plain
    reduction, not a kernel choice, computed the same way regardless of
    which candidate wins the data/weight contests.

    fwd_fn is _patched_conv2d's own accelerated-candidate selection,
    called unchanged -- only the backward is new here. Scoped to conv2d
    only: conv3d has no corresponding CK backward tier (see
    _patched_conv3d), so it keeps falling back to stock whenever grad is
    needed, same as before this class existed."""

    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups, fwd_fn):
        ctx.save_for_backward(input, weight)
        ctx.has_bias = bias is not None
        ctx.stride, ctx.padding, ctx.dilation, ctx.groups = stride, padding, dilation, groups
        return fwd_fn(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        stride, padding, dilation, groups = ctx.stride, ctx.padding, ctx.dilation, ctx.groups
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = _conv2d_backward_data(
                grad_output, weight, tuple(input.shape), stride, padding, dilation, groups)
        if ctx.needs_input_grad[1]:
            grad_weight = _conv2d_backward_weight(
                input, grad_output, tuple(weight.shape), stride, padding, dilation, groups)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=(0, 2, 3))
        return grad_input, grad_weight, grad_bias, None, None, None, None, None


def _patched_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # Three tiers: native HIP kernel (fp16/fp32, this project's own
    # src/cuda/conv2d_fp{16,32}.cu) first, then aiter's Triton conv2d
    # (fp16/bf16 only -- covers the bf16 case the native kernel doesn't),
    # then stock. The native kernel is always available (hard-required at
    # import time, same as group_norm); aiter is optional.
    orig = _ORIGINALS[(F, "conv2d")]
    if groups != 1:
        return orig(input, weight, bias, stride, padding, dilation, groups)
    if _is_pointwise_conv2d(weight, stride, padding, dilation):
        return orig(input, weight, bias, stride, padding, dilation, groups)

    def _accelerated(input, weight, bias, stride, padding, dilation, groups):
        return _conv2d_accelerated_forward(input, weight, bias, stride, padding, dilation, groups, orig)

    if _grad_safe(input, weight, bias):
        return _accelerated(input, weight, bias, stride, padding, dilation, groups)
    # A gradient is needed. UNTIL RECENTLY this always meant falling
    # straight back to stock -- CONFIRMED VIA A REAL SDXL LoRA TRAINING RUN
    # (MIOpen verbose log, gfx1100, character-mode LoKr targeting
    # to_q/to_k/to_v): this branch is not a rare edge case during training,
    # it is EVERY conv2d call. Frozen conv weights don't matter here --
    # _grad_safe() also declines whenever the *input activation*
    # requires_grad, and once any upstream param anywhere in the graph
    # needs a gradient (any LoRA adapter, anywhere in the UNet), autograd
    # must be able to backprop THROUGH every frozen conv layer between it
    # and the loss, so its activations requires_grad=True too. The logged
    # run showed zero native/CK/kernel_select decisions and 100% stock
    # MIOpen calls for every single conv2d, forward and backward.
    #
    # That same log is also where stock itself doesn't have a fast path:
    # every logged backward-data call with symmetric in==out channels and
    # stride 1 got MIOpen's Winograd assembly kernel
    # (miopenSp3AsmConv_v30_3_1_gfx11_fp16_dot2_f2x3_stride1) -- but every
    # call with asymmetric channels (concatenated-skip-connection ResNet
    # convs: 1280->640, 1920->640, 1920->1280, 2560->1280) or stride 2
    # (downsampling convs: 640->640 H64xW37, 320->320 H128xW73) fell back
    # to the slower Col2Im2dU (im2col+GEMM) kernel instead. Those specific
    # shapes -- not the symmetric stride-1 case, where stock's Winograd
    # already wins by 4.6x per ck_ops.py's own fp16 benchmark -- are where
    # CK's real backward-data/backward-weight device ops (_Conv2dFn,
    # _conv2d_backward_data/_weight above) have an actual opening: they're
    # contested against stock per shape via kernel_select rather than
    # assumed to win, exactly like the forward tier already is, so the
    # symmetric-stride-1 case simply keeps picking stock instead of
    # regressing. _grad_safe()'s decline is now the entry into that
    # contest instead of a dead end.
    return _Conv2dFn.apply(input, weight, bias, stride, padding, dilation, groups, _accelerated)


def _conv2d_accelerated_forward(input, weight, bias, stride, padding, dilation, groups, orig):
    # Sparse-pixel fast path, same occupancy-gated design as _patched_conv3d's
    # equivalent check above it (see flexgemm_ops.maybe_sparse_conv2d's
    # docstring) -- content-dependent, so re-checked every call rather than
    # folded into kernel_select's shape cache below. The check itself is
    # cheap on the dense activations a U-Net is made of: 0.0035ms against a
    # 0.326ms conv on 1x320x64x64, i.e. 1.1%, and it declines immediately.
    # AMD_TUNED_TORCH_SPARSE_CONV2D=0 disables it. It also declines unless
    # flex_gemm's native extension is built: this used to run the
    # pure-Python gather/scatter instead, which measured 20-60x SLOWER than
    # stock on exactly the sparse inputs it accepted (numbers in
    # maybe_sparse_conv2d's docstring).
    # _is_plain_tensor, not just dim(): this branch runs BEFORE any _usable
    # check in this function, and FlexGEMM's kernels read dense storage the
    # same way the GEMM tiers do. A Tensor subclass (a quantized weight, say)
    # reaching them fails the way it did in F.linear -- an illegal memory
    # access from inside the subclass's own dispatch fallback, not a catchable
    # exception. See _PLAIN_TENSOR_TYPES.
    if (flexgemm_ops.sparse_conv2d_enabled() and input.dim() == 4 and weight.dim() == 4
            and _is_plain_tensor(input) and _is_plain_tensor(weight)):
        _sparse_out = flexgemm_ops.maybe_sparse_conv2d(
            input, weight, bias, stride=aiter_ops._pair(stride),
            padding=aiter_ops._pair(padding), dilation=aiter_ops._pair(dilation))
        if _sparse_out is not None:
            return _sparse_out

    # Which kernel wins is measured per shape, stock included as a
    # candidate -- see kernel_select.py. This is deliberately NOT a fixed
    # ordering: on gfx1100 stock beats our kernels for fp16/fp32 conv2d
    # (MIOpen has an assembly Winograd solver) and loses badly for bf16
    # (it has none), so any static order is wrong for some dtype.
    if kernel_select.enabled():
        # Fast path: once this shape has been decided, go straight to the
        # winner. Building the candidate thunks below costs ~0.09ms per
        # call -- real money on the many small convs a U-Net is made of,
        # and pointless when the answer is already known.
        _won = kernel_select.cached("conv2d", input, weight, stride, padding, dilation)
        if _won == "stock":
            return orig(input, weight, bias, stride, padding, dilation, groups)
        if _won == "ck":
            _out = ck_ops.conv2d(input, weight, bias, stride, padding, dilation)
            if _out is not None:
                return _out
        elif _won == "native":
            try:
                return compile_ops.conv2d_native(input, weight, bias, stride, padding, dilation)
            except (RuntimeError, TypeError):
                pass
        elif _won == "fftconv":
            _fft_won = _fftconv_conv_candidate(input, weight, bias, stride, padding,
                                                dilation, groups, ndim=2)
            if _fft_won is not None:
                _out = _fft_won[1]()
                if _out is not None:
                    return _out

        candidates = []
        if ck_ops.available() and _usable(input, weight, dtypes=_CK_CONV_DTYPES):
            candidates.append(
                ("ck", lambda: ck_ops.conv2d(input, weight, bias, stride, padding, dilation)))
        if (input.is_contiguous() and weight.is_contiguous()
                and _usable(input, weight, dtypes=_CONV_NATIVE_DTYPES)):
            candidates.append(
                ("native", lambda: compile_ops.conv2d_native(
                    input, weight, bias, stride, padding, dilation)))
        # Large-kernel tier: only contested at all above a measured kernel
        # width (see _fftconv_conv_candidate), so a U-Net's 3x3 convs never
        # pay for it.
        _fft = _fftconv_conv_candidate(input, weight, bias, stride, padding,
                                        dilation, groups, ndim=2)
        if _fft is not None:
            candidates.append(_fft)
        candidates.append(
            ("stock", lambda: orig(input, weight, bias, stride, padding, dilation, groups)))
        out = kernel_select.pick("conv2d", input, weight, stride, padding, dilation, candidates,
                                  tolerance=_fftconv_contest_tolerance(_fft, input))
        if out is not None:
            return out

    # conv_select disabled (AMD_TUNED_TORCH_MEASURE_KERNELS=0): fall back to the
    # old fixed ordering, our kernels first. The FFT tier goes first of all
    # for a kernel wide enough to have passed its pre-filter: with no contest
    # to measure anything, that pre-filter is all the evidence there is, and
    # it is the same static-heuristic posture fftconv_ops.maybe_fft_conv1d
    # keeps for callers who don't want to pay a contest either.
    _fft = _fftconv_conv_candidate(input, weight, bias, stride, padding,
                                    dilation, groups, ndim=2)
    if _fft is not None:
        _out = _fft[1]()
        if _out is not None:
            return _out
    if ck_ops.available() and _usable(input, weight, dtypes=_CK_CONV_DTYPES):
        try:
            _ck = ck_ops.conv2d(input, weight, bias, stride, padding, dilation)
            if _ck is not None:
                return _ck
        except (RuntimeError, TypeError):
            pass
    if (input.is_contiguous() and weight.is_contiguous()
            and _usable(input, weight, dtypes=_CONV_NATIVE_DTYPES)):
        try:
            return compile_ops.conv2d_native(input, weight, bias, stride, padding, dilation)
        except (RuntimeError, TypeError):
            pass
    if aiter_ops.available() and _usable(input, weight):
        try:
            return compile_ops.conv2d_fp16(input, weight, bias, stride, padding, dilation)
        except (RuntimeError, TypeError, AssertionError):
            pass
    return orig(input, weight, bias, stride, padding, dilation, groups)


def _patched_conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # Two tiers now: Composable Kernel's WMMA conv (fp16/bf16) then this
    # project's hand-written HIP kernel (fp16/fp32,
    # src/cuda/conv3d_fp{16,32}.cu). Neither aiter nor TransformerEngine
    # cover conv3d at all, so CK is what fills that gap -- and it is also
    # the first conv3d bf16 path here that isn't stock.
    orig = _ORIGINALS[(F, "conv3d")]
    if groups != 1 or not _grad_safe(input, weight, bias):
        return orig(input, weight, bias, stride, padding, dilation, groups)

    # Sparse-voxel fast path: routes to flex_gemm's sparse_conv3d instead of
    # a dense kernel when `input` is mostly empty (e.g. a voxelized surface
    # or point cloud, as opposed to a diffusion U-Net's normally-dense
    # activations). This is a CONTENT-dependent decision -- occupancy, not
    # shape -- so unlike the kernel_select contest just below, it can't be
    # cached by (dtype, shape, stride, padding, dilation): two calls with an
    # identical shape key can have very different occupancy. It is instead
    # re-checked on every eligible call via a single cheap reduction over
    # `input` (see flexgemm_ops.maybe_sparse_conv3d's docstring for that
    # cost and the unvalidated default occupancy threshold). Declines
    # (returns None) instantly back to the dense contest below whenever
    # flex_gemm isn't installed, AMD_TUNED_TORCH_SPARSE_CONV3D=0 has
    # disabled it, or occupancy is at/above the threshold -- so a normal
    # dense conv3d call pays only the one reduction pass, never a wasted
    # sparse-kernel attempt.
    # Same subclass guard as the conv2d sparse path above.
    if (flexgemm_ops.sparse_conv3d_enabled() and input.dim() == 5 and weight.dim() == 5
            and _is_plain_tensor(input) and _is_plain_tensor(weight)):
        _sparse_out = flexgemm_ops.maybe_sparse_conv3d(
            input, weight, bias, stride=_triple(stride), padding=_triple(padding),
            dilation=_triple(dilation))
        if _sparse_out is not None:
            return _sparse_out

    # Measured per shape with stock as a candidate, exactly as in
    # _patched_conv2d. conv3d is where our kernels look best (CK 1.16ms vs
    # stock 3.02ms at fp16) but fp32 is within noise of stock (0.95x), so
    # the same contest applies rather than a fixed order.
    if kernel_select.enabled():
        # Fast path: once this shape has been decided, go straight to the
        # winner. Building the candidate thunks below costs ~0.09ms per
        # call -- real money on the many small convs a U-Net is made of,
        # and pointless when the answer is already known.
        _won = kernel_select.cached("conv3d", input, weight, stride, padding, dilation)
        if _won == "stock":
            return orig(input, weight, bias, stride, padding, dilation, groups)
        if _won == "ck":
            _out = ck_ops.conv3d(input, weight, bias, stride, padding, dilation)
            if _out is not None:
                return _out
        elif _won == "native":
            try:
                return compile_ops.conv3d_native(input, weight, bias, stride, padding, dilation)
            except (RuntimeError, TypeError):
                pass
        elif _won == "fftconv":
            _fft_won = _fftconv_conv_candidate(input, weight, bias, stride, padding,
                                                dilation, groups, ndim=3)
            if _fft_won is not None:
                _out = _fft_won[1]()
                if _out is not None:
                    return _out

        candidates = []
        if ck_ops.available() and _usable(input, weight, dtypes=_CK_CONV_DTYPES):
            candidates.append(
                ("ck", lambda: ck_ops.conv3d(input, weight, bias, stride, padding, dilation)))
        if (input.is_contiguous() and weight.is_contiguous()
                and _usable(input, weight, dtypes=_CONV_NATIVE_DTYPES)):
            candidates.append(
                ("native", lambda: compile_ops.conv3d_native(
                    input, weight, bias, stride, padding, dilation)))
        # Large-kernel tier, and the one place in this file where it wins
        # by orders of magnitude rather than percent: direct conv3d pays
        # K^3 multiplies per output element (measured 15^3 fp32: stock
        # 6907ms vs 38.8ms here). See _fftconv_conv_candidate.
        _fft = _fftconv_conv_candidate(input, weight, bias, stride, padding,
                                        dilation, groups, ndim=3)
        if _fft is not None:
            candidates.append(_fft)
        candidates.append(
            ("stock", lambda: orig(input, weight, bias, stride, padding, dilation, groups)))
        out = kernel_select.pick("conv3d", input, weight, stride, padding, dilation, candidates,
                                  tolerance=_fftconv_contest_tolerance(_fft, input))
        if out is not None:
            return out

    # conv_select disabled: old fixed ordering, our kernels first -- with
    # the FFT tier ahead of them for a wide-enough kernel, same reasoning
    # as _patched_conv2d's equivalent branch.
    _fft = _fftconv_conv_candidate(input, weight, bias, stride, padding,
                                    dilation, groups, ndim=3)
    if _fft is not None:
        _out = _fft[1]()
        if _out is not None:
            return _out
    if ck_ops.available() and _usable(input, weight, dtypes=_CK_CONV_DTYPES):
        try:
            _ck = ck_ops.conv3d(input, weight, bias, stride, padding, dilation)
            if _ck is not None:
                return _ck
        except (RuntimeError, TypeError):
            pass
    # groups/grad-safety already checked above.
    if (not (input.is_contiguous() and weight.is_contiguous())
            or not _usable(input, weight, dtypes=_CONV_NATIVE_DTYPES)):
        return orig(input, weight, bias, stride, padding, dilation, groups)
    try:
        return compile_ops.conv3d_native(input, weight, bias, stride, padding, dilation)
    except (RuntimeError, TypeError):
        return orig(input, weight, bias, stride, padding, dilation, groups)


# ---------------------------------------------------------------------------
# group_norm: native HIP kernel (no autograd, same fallback pattern as above).
# ---------------------------------------------------------------------------

def _group_norm_native(input, num_groups, weight, bias, eps):
    """The native group_norm kernel, routed to avoid paying for
    torch.compile safety in eager.

    compile_ops.group_norm wraps the kernel in a torch.library custom op so
    it doesn't graph-break under torch.compile (see compile_ops.py). That
    wrapper costs ~15us of dispatch, which is nothing for a conv but is
    most of this kernel's margin: raw it beats stock 1.10x (0.150ms vs
    0.166ms at the bench shape), through the custom op it loses. Since the
    graph-break protection only matters while Dynamo is actually tracing,
    take the custom op then and the direct call otherwise -- identical
    numerics either way.
    """
    if _is_compiling():
        return compile_ops.group_norm(input, num_groups, weight, bias, eps)
    return ops.group_norm(input, num_groups, weight, bias, eps)


def _patched_group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    """F.group_norm over a per-shape contest: native HIP, CK, stock.

    The CK candidate only enters for CHANNELS-LAST input, and that is a
    measured restriction rather than a conservative one. CK's normalization
    instances index the tensor as [N, S1, S2, G, C], which is a free reshape
    of channels-last memory and a permute-in-and-out of contiguous NCHW.
    Measured against stock on plain GroupNorm:

        2x320x64x64  fp16    NHWC 2.09x    NCHW 0.85x
        2x640x32x32  fp16    NHWC 1.37x    NCHW 0.48x
        2x1280x16x16 fp32    NHWC 1.26x    NCHW 0.40x

    So it is offered where it wins and withheld where it loses. NCHW keeps
    the native kernel, which is what it was already getting.

    The larger win from this tier is not reachable here at all: fusing the
    SiLU that follows GroupNorm in every diffusion ResBlock is 1.6-2.1x, and
    F.group_norm cannot know an activation follows it. See
    ck_norm_ops.group_norm_silu for the call a model can make directly.
    """
    orig = _ORIGINALS[(F, "group_norm")]
    channels_last = ck_norm_ops._channels_last(input)
    if not ((input.is_contiguous() or channels_last) and _grad_safe(input, weight, bias)
            and _usable(input, dtypes=_GROUPNORM_DTYPES)):
        return orig(input, num_groups, weight, bias, eps)

    # Measured per shape like conv, rather than assuming the native kernel
    # wins. It usually does, but only by 1.03-1.23x depending on dtype --
    # small enough that "assume ours is faster" is not a safe default, and
    # the contest costs nothing after the first call for a shape.
    if kernel_select.enabled():
        # Layout belongs in the key: it decides which candidates are even
        # eligible, not merely how fast they are.
        key = (input.dtype, tuple(input.shape), int(num_groups), channels_last)
        won = kernel_select.cached_key("group_norm", key)
        if won == "stock":
            return orig(input, num_groups, weight, bias, eps)
        if won == "ck":
            out = ck_norm_ops.group_norm(input, num_groups, weight, bias, eps)
            if out is not None:
                return out
        elif won != "native":
            candidates = []
            if channels_last:
                candidates.append(
                    ("ck", lambda: ck_norm_ops.group_norm(input, num_groups, weight, bias, eps)))
            if input.is_contiguous():
                candidates.append(
                    ("native", lambda: _group_norm_native(input, num_groups, weight, bias, eps)))
            candidates.append(("stock", lambda: orig(input, num_groups, weight, bias, eps)))
            out = kernel_select.pick_key("group_norm", key, candidates)
            if out is not None:
                return out

    if input.is_contiguous():
        try:
            return _group_norm_native(input, num_groups, weight, bias, eps)
        except (RuntimeError, TypeError):
            pass
    return orig(input, num_groups, weight, bias, eps)


# ---------------------------------------------------------------------------
# Wrappers backed by TransformerEngine (real autograd -- no grad-safety
# check needed, active for training as well as inference).
# ---------------------------------------------------------------------------

def _patched_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    orig = _ORIGINALS[(F, "layer_norm")]
    if weight is None or bias is None or not _usable(input, weight, bias, dtypes=_TE_DTYPES):
        return orig(input, normalized_shape, weight, bias, eps)
    try:
        return te_ops.layer_norm(input, normalized_shape, weight, bias, eps)
    except (RuntimeError, TypeError):
        return orig(input, normalized_shape, weight, bias, eps)


def _patched_rms_norm(input, normalized_shape, weight=None, eps=None):
    orig = _ORIGINALS[(F, "rms_norm")]
    if weight is None or not _usable(input, weight, dtypes=_TE_DTYPES):
        return orig(input, normalized_shape, weight, eps)
    try:
        return te_ops.rms_norm(input, normalized_shape, weight, eps)
    except (RuntimeError, TypeError):
        return orig(input, normalized_shape, weight, eps)


def _patched_gelu(input, approximate="none"):
    orig = _ORIGINALS[(F, "gelu")]
    # tex.gelu computes the tanh approximation only -- patching the default
    # (exact, erf-based) approximate="none" path would silently change
    # numerics for anything expecting exact GELU.
    if approximate != "tanh" or not _usable(input, dtypes=_TE_DTYPES):
        return orig(input, approximate=approximate)
    try:
        return te_ops.gelu(input)
    except (RuntimeError, TypeError):
        return orig(input, approximate=approximate)


def _patched_silu(input, inplace=False):
    orig = _ORIGINALS[(F, "silu")]
    if inplace or not _usable(input, dtypes=_SILU_DTYPES):
        return orig(input, inplace=inplace)
    try:
        return te_ops.silu(input)
    except (RuntimeError, TypeError):
        return orig(input, inplace=inplace)


def _patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                   scale=None, **kwargs):
    orig = _ORIGINALS[(F, "scaled_dot_product_attention")]
    # Cheap guards first: dropout_p/dim/dtype are plain Python attribute
    # reads. te_ops.is_bottom_right_causal_mask (below) is NOT cheap -- it
    # ends in torch.equal(), which forces a GPU sync to pull the boolean
    # result back to host. Checking it before these guards would mean
    # paying for that sync on every training-mode call (dropout_p != 0)
    # only to immediately discard the result and fall back to stock anyway.
    if (dropout_p != 0.0 or query.dim() != 4
            or not _usable(query, key, value, dtypes=_TE_DTYPES)):
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                     is_causal=is_causal, scale=scale, **kwargs)
    # attn_mask is allowed through only when it structurally matches the
    # bottom-right causal pattern (see te_ops.is_bottom_right_causal_mask) --
    # e.g. dflash and HF's sdpa_attention_forward build an explicit boolean
    # mask for causal/KV-cache decoding instead of using is_causal=True.
    # Anything else (padding, sliding-window, genuinely arbitrary masks)
    # still falls back to stock: TE's fused backends don't accept arbitrary
    # tensors, only named attn_mask_type strings.
    mask_ok = attn_mask is None or (
        not is_causal and te_ops.is_bottom_right_causal_mask(attn_mask)
    )
    if not mask_ok:
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                     is_causal=is_causal, scale=scale, **kwargs)
    try:
        return te_ops.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, scale=scale, is_causal=is_causal
        )
    except (RuntimeError, TypeError):
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                     is_causal=is_causal, scale=scale, **kwargs)


def enable() -> None:
    """Patch torch/torch.nn.functional to route eligible calls through amd_tuned_torch."""
    global _ENABLED
    if _ENABLED:
        return

    # linear/matmul/bmm. Installed when ANY of the three GEMM candidates is
    # available -- aiter's Triton kernels, hipBLASLt, or the CK WMMA GEMM
    # tier -- since each of the three patches runs a contest that includes
    # stock and will simply pick stock if nothing else can serve the shape.
    #
    # This used to be gated on aiter alone, which was right when aiter was
    # the only candidate and wrong the moment it stopped being. On gfx1100
    # aiter_ops.available() is False (it ships no RDNA3 tuning config), so
    # that gate silently withheld the patch entirely -- and with it
    # hipBLASLt's 1.29x and CK's 1.21x -- long after there was something to
    # gain. A capability check has to name the capability it is checking
    # for, not one dependency that used to imply it.
    if aiter_ops.available() or hipblaslt_ops.available() or ck_gemm_ops.available():
        _install(F, "linear", _patched_linear)
        _install(torch, "matmul", _patched_matmul)
        _install(torch, "bmm", _patched_bmm)

    # Native HIP kernels (see src/main_rocm.cpp). Always installed -- the
    # native extension is hard-required at import time (see the ImportError
    # guard above), unlike the aiter-gated and TE-gated patches. conv2d/conv3d
    # try the native kernel first (fp16/fp32; see _patched_conv2d/
    # _patched_conv3d for the tier order and fallbacks).
    _install(F, "group_norm", _patched_group_norm)
    _install(F, "conv2d", _patched_conv2d)
    if hasattr(F, "conv3d"):
        _install(F, "conv3d", _patched_conv3d)

    # TransformerEngine-backed (see amd_tuned_torch/te_ops.py). Skipped entirely if
    # TE isn't installed -- these ops just stay on stock PyTorch/ROCm.
    #
    # F.layer_norm intentionally NOT installed: TE's layernorm_fwd
    # benchmarked slower than stock on RX 7900 XTX for both fp16 (0.71x)
    # and fp32 (0.61x), see benchmark.json -- _patched_layer_norm stays
    # available for manual use.
    if te_ops.available():
        if hasattr(F, "rms_norm"):
            _install(F, "rms_norm", _patched_rms_norm)
        _install(F, "gelu", _patched_gelu)
        _install(F, "silu", _patched_silu)
        if hasattr(F, "scaled_dot_product_attention"):
            _install(F, "scaled_dot_product_attention", _patched_sdpa)

    _ENABLED = True


def disable() -> None:
    """Restore every patched function to the stock PyTorch implementation."""
    global _ENABLED
    for target, name in list(_ORIGINALS.keys()):
        _restore(target, name)
    _ENABLED = False


def is_enabled() -> bool:
    return _ENABLED


# ---------------------------------------------------------------------------
# TorchScript compatibility.
#
# THE PROBLEM. torch.jit.script compiles the Python SOURCE of whatever it
# finds behind a name, so `torch.bmm(a, b)` inside a scripted function
# compiles whatever `torch.bmm` is bound to at that moment. Once enable()
# has run that is a wrapper of ours, and none of these wrappers are
# scriptable -- nor should they be. TorchScript rejects each for a
# different reason:
#
#     torch.bmm, torch.matmul   keyword-only argument with a default (out=None)
#     F.linear, F.group_norm    lambdas (the kernel_select candidate thunks)
#     F.conv2d                  try blocks
#
# and the ones it does not reject it would compile wrongly, since the body
# reaches into a dict keyed by module objects and calls into a pybind
# extension. This is not a defect in any single wrapper; it is what
# monkeypatching costs, and no rewriting of the wrappers fixes it.
#
# It is also not new. F.conv2d has been unscriptable since it was first
# patched -- it simply went unnoticed, because a library has to script code
# that calls a patched op before anything breaks. kornia scripts
# kornia.geometry.boxes at import time and that calls torch.bmm, which
# stayed harmless only for as long as enable() declined to install the
# torch.bmm patch (it was gated on aiter, absent on gfx1100). Fixing that
# gate turned a latent incompatibility into an import-time crash for every
# stable-diffusion-webui launch.
#
# THE FIX. Compile against stock. torch.jit.script and torch.jit.trace are
# wrapped so the patches are lifted for the duration of the call and
# restored after, which makes the compiler bind to the real aten ops. The
# cost is that code inside a scripted or traced region runs stock kernels
# rather than ours -- correct, and the only honest option, since a
# TorchScript graph cannot call a Python dispatch wrapper anyway.
#
# Tracing needs this at least as much as scripting, and more quietly:
# torch.jit.trace records the aten ops a function executes, and these
# wrappers execute pybind calls that are not traceable ops. A trace taken
# with the patches live would bake in a constant instead of a computation
# and be silently wrong, where scripting at least fails loudly.
#
# The guard is installed at import time, not by enable(), because the
# breakage is caused by the patches existing rather than by anyone opting
# in -- and because this module is auto-loaded before user code (see the
# sitecustomize hook), so libraries that script at import time are already
# covered.
_ORIG_JIT_SCRIPT = torch.jit.script
_ORIG_JIT_TRACE = torch.jit.trace


def _script_with_stock_ops(obj, *args, **kwargs):
    """torch.jit.script, compiled against stock ops.

    _frames_up IS NOT OPTIONAL BOOKKEEPING. When the object being scripted
    is a class, torch's _script_impl builds its name-resolution callback
    with createResolutionCallbackFromFrame(_frames_up + 1) -- it walks a
    fixed number of stack frames up to find the caller's module globals.
    This wrapper is one extra frame, so without compensating, every name a
    scripted class refers to is looked up in amd_tuned_torch's namespace
    instead of the caller's. That surfaces as `undefined value <helper>` on
    a perfectly ordinary module-level function, which is how kornia's
    Boxes3D failed the first time this guard was written. Functions resolve
    through createResolutionCallbackFromClosure and are unaffected, which
    is exactly why the bug hides until something scripts a class.

    The adjustment is unconditional: this frame exists whether or not the
    patches were active.
    """
    if len(args) >= 2:
        # positional: script(obj, optimize, _frames_up, ...)
        args = args[:1] + (args[1] + 1,) + args[2:]
    else:
        kwargs["_frames_up"] = kwargs.get("_frames_up", 0) + 1

    if not _ENABLED:
        return _ORIG_JIT_SCRIPT(obj, *args, **kwargs)
    disable()
    try:
        return _ORIG_JIT_SCRIPT(obj, *args, **kwargs)
    finally:
        enable()


def _trace_with_stock_ops(*args, **kwargs):
    """torch.jit.trace, recorded against stock ops.

    No frame compensation needed here: tracing executes the function and
    records the aten ops it runs, rather than resolving names out of the
    caller's namespace.
    """
    if not _ENABLED:
        return _ORIG_JIT_TRACE(*args, **kwargs)
    disable()
    try:
        return _ORIG_JIT_TRACE(*args, **kwargs)
    finally:
        enable()


_script_with_stock_ops.__doc__ = _ORIG_JIT_SCRIPT.__doc__
_trace_with_stock_ops.__doc__ = _ORIG_JIT_TRACE.__doc__
torch.jit.script = _script_with_stock_ops
torch.jit.trace = _trace_with_stock_ops


# Re-exported at the package top level via `from ._dispatch import *` in
# __init__.py -- includes underscore-prefixed names because tests and other
# submodules already reach into them directly (amd_tuned_torch._grad_safe,
# amd_tuned_torch._usable, amd_tuned_torch._ORIGINALS, etc.), and this split
# must not change what amd_tuned_torch.X resolves to for any X below.
__all__ = [
    "ops",
    "_GEMM_DTYPES", "_GROUPNORM_DTYPES", "_TE_DTYPES", "_CONV_NATIVE_DTYPES",
    "_CK_CONV_DTYPES", "_FFTCONV_CONV_DTYPES",
    "_FFTCONV_CONV2D_ENABLED", "_FFTCONV_CONV3D_ENABLED",
    "_FFTCONV_CONV2D_MIN_KERNEL", "_FFTCONV_CONV3D_MIN_KERNEL",
    "_fftconv_calibration_data", "_calibrated_fftconv_default",
    "_FFTCONV_CONV3D_MIN_POSITIONS", "_SILU_DTYPES",
    "_ORIGINALS", "_ENABLED", "_HAS_INFERENCE_MODE",
    "_is_compiling", "_grad_safe", "_usable", "_install", "_restore", "_triple",
    "_LinearFn", "_BmmFn", "_patched_linear", "_linear_aiter",
    "_patched_matmul", "_patched_bmm", "_bmm_aiter",
    "_is_pointwise_conv2d", "_fftconv_conv_candidate", "_fftconv_contest_tolerance",
    "_Conv2dFn", "_conv2d_accelerated_forward",
    "_conv2d_backward_data", "_conv2d_backward_weight",
    "_conv2d_backward_data_stock", "_conv2d_backward_weight_stock",
    "_patched_conv2d", "_patched_conv3d",
    "_group_norm_native", "_patched_group_norm",
    "_patched_layer_norm", "_patched_rms_norm", "_patched_gelu", "_patched_silu",
    "_patched_sdpa",
    "enable", "disable", "is_enabled",
    "_ORIG_JIT_SCRIPT", "_ORIG_JIT_TRACE",
    "_script_with_stock_ops", "_trace_with_stock_ops",
]
