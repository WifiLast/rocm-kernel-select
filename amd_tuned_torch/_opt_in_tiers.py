"""amd_tuned_torch._opt_in_tiers -- everything enable()/disable() does NOT
install: INT8 (W8A8) quantized linear, conv3d Winograd fp16, FlashAttention
(rocWMMA), RMSNorm (triton-kernels), and BOFT's fast_block_diag, each
toggled independently via its own enable_*()/disable_*() pair -- plus the
standalone, never-monkeypatched ops a model calls directly because there is
no torch.nn.functional entry point for them to intercept (mla_decode,
fused_add_rms_norm, rotary_embedding, compute_rope_cos_sin_cache,
fused_linear_cross_entropy, bias_swiglu, fast_block_diag), and the
SmoothQuant calibration re-exports
(calibrate_smoothquant, compute_smoothquant_scale, set_smooth_scale,
is_int8_linear_enabled).

Split out of amd_tuned_torch/__init__.py (previously one ~2300-line file)
purely for readability -- no behavior changed by this split. Every name
defined here is re-exported at the package top level
(`from ._opt_in_tiers import *` in __init__.py, with this module's __all__
below listing every name), so `amd_tuned_torch.X` resolves exactly as it
did before the split, for every X defined here.

See amd_tuned_torch/_dispatch.py for the always-on tiers enable() DOES
install, and for _grad_safe/_usable/_triple, the shared helpers this module
imports from there rather than duplicating.
"""
from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F

from . import _dispatch
from . import aiter_ops
from . import compile_ops
from . import kernel_select
from . import flash_attn_rocwmma_ops
from . import triton_kernels_ops
from . import mla_ops
from . import fused_norm_ops
from . import rope_ops
from . import fused_ce_ops
from . import boft_ops
from . import swiglu_ops

_INT8_LINEAR_DTYPES = (torch.float16, torch.bfloat16, torch.float32)  # aiter quantizes from any of these

# ---------------------------------------------------------------------------
# INT8 (W8A8) quantized linear -- aiter-backed, opt-in only (see
# amd_tuned_torch/aiter_ops.py). Unlike everything enable() installs, this changes
# numerics, so it's a separate toggle, never installed automatically and
# never bundled into enable()/disable().
#
# This composes with enable()/disable() regardless of call order: it
# captures whatever F.linear currently *is* (stock, or already
# _patched_linear if enable() ran first) as its own fallback, rather than
# going through the shared _ORIGINALS bookkeeping enable()/disable() use --
# so toggling this on/off never disturbs, and is never disturbed by, the
# main patch set.
# ---------------------------------------------------------------------------

_INT8_LINEAR_ENABLED = False
_int8_linear_fallback: Callable | None = None


def _patched_linear_int8(input, weight, bias=None):
    if not (_dispatch._grad_safe(input, weight, bias) and _dispatch._usable(input, weight, dtypes=_INT8_LINEAR_DTYPES)):
        return _int8_linear_fallback(input, weight, bias)
    if input.dim() < 2 or weight.dim() != 2:
        return _int8_linear_fallback(input, weight, bias)
    try:
        return compile_ops.linear_int8(input, weight, bias)
    except (RuntimeError, TypeError, AssertionError):
        return _int8_linear_fallback(input, weight, bias)


def enable_int8_linear() -> None:
    """Opt-in: route F.linear through aiter's W8A8 (int8 activation, int8
    weight) quantized GEMM, which auto-routes to a pure-Triton kernel on
    gfx11/RDNA3 (see amd_tuned_torch/aiter_ops.py). This trades accuracy for speed
    -- unlike every other patch amd_tuned_torch installs, this is NOT numerically
    transparent. No-ops with a warning if aiter isn't installed."""
    global _INT8_LINEAR_ENABLED, _int8_linear_fallback
    if _INT8_LINEAR_ENABLED:
        return
    if not aiter_ops.available():
        import warnings
        warnings.warn("amd_tuned_torch.enable_int8_linear(): aiter not installed, F.linear left as-is")
        return
    _int8_linear_fallback = F.linear
    F.linear = _patched_linear_int8
    _INT8_LINEAR_ENABLED = True


def disable_int8_linear() -> None:
    """Restore whatever F.linear was before enable_int8_linear() -- stock,
    or _patched_linear if the main enable() had already run."""
    global _INT8_LINEAR_ENABLED, _int8_linear_fallback
    if not _INT8_LINEAR_ENABLED:
        return
    F.linear = _int8_linear_fallback
    _int8_linear_fallback = None
    _INT8_LINEAR_ENABLED = False


# ---------------------------------------------------------------------------
# SmoothQuant calibration -- opt-in on top of enable_int8_linear(), never run
# automatically. See amd_tuned_torch/aiter_ops.py's module docstring and the
# SmoothQuant section there for the full derivation; these are thin
# re-exports so `amd_tuned_torch.calibrate_smoothquant(...)` reads the same as
# `amd_tuned_torch.enable_int8_linear()` rather than requiring
# `amd_tuned_torch.aiter_ops.calibrate_smoothquant(...)`.
# ---------------------------------------------------------------------------

def calibrate_smoothquant(forward_fn: Callable[[], Any], alpha: float = 0.5) -> None:
    """Run `forward_fn()` with torch.nn.Linear.forward temporarily patched
    to collect per-input-channel activation statistics, then bake a
    SmoothQuant scale into every weight seen so subsequent linear_int8
    calls (enable_int8_linear() must already be on) route activation
    quantization through aiter's fused smoothquant_quantize kernel instead
    of plain per-token quantization. See amd_tuned_torch/aiter_ops.py for details."""
    aiter_ops.calibrate_smoothquant(forward_fn, alpha=alpha)


# ---------------------------------------------------------------------------
# Conv3d fp16 Winograd -- opt-in only, same "separate toggle, never bundled
# into enable()" shape as enable_int8_linear() above, for the same reason:
# this changes which kernel actually runs, and unlike every tier enable()
# installs, it has never been validated. src/cuda/templates/
# conv3d_fp16_winograd.cu.tmpl (tools/kernelgen/) has not been run,
# correctness-checked, or benchmarked on any hardware -- see that
# template's header. Composes with enable()/disable() regardless of call
# order (captures whatever F.conv3d currently *is* as its own fallback),
# same mechanism as enable_int8_linear().
# ---------------------------------------------------------------------------

_CONV3D_WINOGRAD_FP16_ENABLED = False
_conv3d_winograd_fp16_fallback: Callable | None = None


def _is_winograd_eligible_conv3d(input, weight, stride, padding, dilation) -> bool:
    """True only for the exact shape family
    src/cuda/templates/conv3d_fp16_winograd.cu.tmpl's F(2x2x2,3x3x3) kernel
    applies to -- mirrors that template's launcher guard exactly (kernel=
    3x3x3, stride=1, padding=1, dilation=1, batch=1, even D_in/H_in/W_in,
    and a resulting tile-count/C_out both divisible by that kernel's
    BT_TILES/BT_COUT=8 register-blocking factor) so this predicate never
    calls into the kernel for a shape it would decline anyway.

    UNLIKE _is_pointwise_conv2d above, this is NOT informed by a real
    measurement on this hardware -- satisfying this predicate is necessary
    but not sufficient for the Winograd tier to run; see
    enable_conv3d_winograd_fp16's docstring for what to verify first."""
    if input.dim() != 5 or weight.dim() != 5:
        return False
    if input.shape[0] != 1:
        return False
    if tuple(weight.shape[2:]) != (3, 3, 3):
        return False
    if (_dispatch._triple(stride) != (1, 1, 1) or _dispatch._triple(padding) != (1, 1, 1)
            or _dispatch._triple(dilation) != (1, 1, 1)):
        return False
    _, _, D_in, H_in, W_in = input.shape
    if D_in % 2 or H_in % 2 or W_in % 2:
        return False
    # stride=1/padding=1/kernel=3 -> output spatial size equals input's.
    total_tiles = (D_in // 2) * (H_in // 2) * (W_in // 2)
    C_out = weight.shape[0]
    return total_tiles % 8 == 0 and C_out % 8 == 0


def _patched_conv3d_winograd_fp16(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if (groups != 1
            or not _dispatch._grad_safe(input, weight, bias)
            or not (input.is_contiguous() and weight.is_contiguous())
            or not _dispatch._usable(input, weight, dtypes=(torch.float16,))
            or not _is_winograd_eligible_conv3d(input, weight, stride, padding, dilation)):
        return _conv3d_winograd_fp16_fallback(input, weight, bias, stride, padding, dilation, groups)
    try:
        out = compile_ops.conv3d_fp16_winograd_bt8_bc8(input, weight, bias, stride, padding, dilation)
        if out is not None:
            return out
    except (RuntimeError, TypeError):
        pass
    return _conv3d_winograd_fp16_fallback(input, weight, bias, stride, padding, dilation, groups)


def enable_conv3d_winograd_fp16() -> None:
    """Opt-in: try the codegen'd F(2x2x2,3x3x3) Winograd fp16 conv3d kernel
    first for eligible shapes (see _is_winograd_eligible_conv3d), falling
    back to whatever F.conv3d currently is -- stock, or _patched_conv3d if
    the main enable() already ran -- for any other shape, or if the kernel
    declines/raises.

    DO NOT call this without first, on your actual RX 7900 XTX:
      1. Running tests_hardware/test_conv_kernels.py (or an equivalent
         direct comparison against F.conv3d, ideally cross-checked in
         fp64) to confirm correctness at a tolerance you're comfortable
         with. Winograd's error profile (few large-magnitude-swing terms
         per output vs. this project's other conv kernels' many small
         ordered accumulations) means their tolerances should NOT be
         assumed to transfer -- determine one empirically.
      2. Running tools/kernelgen/autotune.py to confirm this is actually
         faster than both stock and the native WMMA conv3d kernel for
         your shapes -- nothing here benchmarks it before turning it on.
    This kernel has never been run on any hardware; both checks are
    unverified until you perform them.

    Disable ordering, if you also use the main enable()/disable(): undo
    these LIFO -- call disable_conv3d_winograd_fp16() BEFORE disable(),
    not after. disable() restores F.conv3d from its own bookkeeping
    (_ORIGINALS) unconditionally and clears that bookkeeping; doing so
    while this is still enabled leaves _conv3d_winograd_fp16_fallback
    pointing at an now-orphaned _patched_conv3d, and a later
    disable_conv3d_winograd_fp16() call would re-install that orphaned
    reference into F.conv3d -- which then raises KeyError on its own
    _ORIGINALS lookup the next time anything calls it. (Confirmed by
    triggering it: see TestConv3dWinogradFp16Dispatch.
    test_composes_on_top_of_main_patched_conv3d in
    tests/test_amd_tuned_torch_monkeypatch.py.)"""
    global _CONV3D_WINOGRAD_FP16_ENABLED, _conv3d_winograd_fp16_fallback
    if _CONV3D_WINOGRAD_FP16_ENABLED:
        return
    _conv3d_winograd_fp16_fallback = F.conv3d
    F.conv3d = _patched_conv3d_winograd_fp16
    _CONV3D_WINOGRAD_FP16_ENABLED = True


def disable_conv3d_winograd_fp16() -> None:
    """Restore whatever F.conv3d was before enable_conv3d_winograd_fp16()."""
    global _CONV3D_WINOGRAD_FP16_ENABLED, _conv3d_winograd_fp16_fallback
    if not _CONV3D_WINOGRAD_FP16_ENABLED:
        return
    F.conv3d = _conv3d_winograd_fp16_fallback
    _conv3d_winograd_fp16_fallback = None
    _CONV3D_WINOGRAD_FP16_ENABLED = False


# ---------------------------------------------------------------------------
# FlashAttention (rocWMMA) -- DEFAULT ON as of the bottom-of-file
# AMD_TUNED_TORCH_FLASH_ATTN_ROCWMMA trigger (set it to "0" to opt back
# out), even though flash_attn_rocwmma_ops wraps a vendored kernel
# (amd_tuned_torch/_vendor/rocwmma_fattn/) that has never been run,
# correctness-checked, or benchmarked on any hardware this project has
# access to -- see that module's docstring, and enable_flash_attn_rocwmma's
# docstring below for exactly what remains unverified. This was a
# deliberate choice to accept that risk by default, not an oversight --
# the eligibility gate (_is_flash_attn_rocwmma_eligible) and the
# try/except fallback to whatever F.scaled_dot_product_attention already
# was mean a bad/incompatible build degrades to a warning at worst (see
# enable_flash_attn_rocwmma's `if not flash_attn_rocwmma_ops.available()`
# branch), not a hard failure, which is what makes default-on tolerable
# here in a way it wouldn't be for a kernel with no such fallback.
# Composes with enable()/disable() regardless of call order the same way
# enable_conv3d_winograd_fp16 does (captures whatever
# F.scaled_dot_product_attention currently is as its own fallback) -- the
# SAME disable-ordering caveat applies here too (disable this before the
# main disable(), not after); see enable_conv3d_winograd_fp16's docstring
# for the mechanism/why.
# ---------------------------------------------------------------------------

_FLASH_ATTN_ROCWMMA_ENABLED = False
_flash_attn_rocwmma_fallback: Callable | None = None

_FLASH_ATTN_ROCWMMA_DTYPES = (torch.float16, torch.bfloat16)


def _is_flash_attn_rocwmma_eligible(query, key, value, attn_mask, dropout_p, is_causal) -> bool:
    """True only for shapes flash_attn_rocwmma_ops's vendored kernel
    actually supports: no attn_mask at all (its host.cpp forward/backward
    signatures take only a `causal: bool` -- there is no mask tensor
    input, unlike TE's fused path, which at least recognizes named
    attn_mask_type strings), no dropout, 4D (batch, heads, seq, head_dim)
    query/key/value, and -- if is_causal -- query and key the same
    sequence length.

    That last constraint is required, not a simplification: reading the
    vendored kernel (kernel_fp16.cu's causal masking, a compile-time
    template bool compared against plain block-relative positions, with
    no q_len/kv_len offset parameter anywhere in host.cpp's
    fwd_parm/bwd_parm) shows this is plain top-left causal masking, not
    the KV-cache-offset-aware bottom-right causal masking
    te_ops.is_bottom_right_causal_mask exists to detect. Top-left and
    bottom-right causal are only identical when query_len == key_len, so
    is_causal is only honored in that case -- anything else (e.g.
    KV-cache decoding, q_len < kv_len) falls back rather than silently
    computing the wrong mask.

    The equal-head-count requirement is there for the same reason and is
    just as load-bearing. The vendored kernel has no MQA/GQA support at
    all: host.cpp's fwd_parm/bwd_parm carry a single head count, and the
    kernel indexes K and V by the QUERY head index. Hand it a K/V with
    fewer heads than Q -- the standard GQA layout, and what
    F.scaled_dot_product_attention(..., enable_gqa=True) passes -- and it
    reads past the end of those tensors. Measured on gfx1100, Q(2,8,512,64)
    against K/V(2,2,512,64) in bf16 returns non-finite values, and the same
    out-of-bounds access intermittently traps as

        HIP error: an illegal memory access was encountered

    which poisons the HIP context for the rest of the process, so the
    failure usually surfaces later, in whatever unrelated CUDA call
    synchronizes next.

    Note this is not merely about which kernel wins: kernel_select's
    contest CALLS each candidate to time it, so on a cold cache an
    ineligible-but-not-rejected shape faults during measurement even when
    the fallback would have been chosen. The guard therefore has to be
    here, in eligibility, not in the ranking."""
    if attn_mask is not None or dropout_p != 0.0 or query.dim() != 4:
        return False
    if not _dispatch._usable(query, key, value, dtypes=_FLASH_ATTN_ROCWMMA_DTYPES):
        return False
    if key.dim() != 4 or value.dim() != 4:
        return False
    # No MQA/GQA: Q, K and V must agree on head count (dim 1) and head_dim
    # (dim 3), and K/V must agree on sequence length (dim 2).
    if not (query.shape[1] == key.shape[1] == value.shape[1]):
        return False
    if not (query.shape[3] == key.shape[3] == value.shape[3]):
        return False
    if key.shape[2] != value.shape[2]:
        return False
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        return False
    if is_causal and query.shape[2] != key.shape[2]:
        return False
    return True


def _patched_sdpa_flash_attn_rocwmma(query, key, value, attn_mask=None, dropout_p=0.0,
                                      is_causal=False, scale=None, **kwargs):
    """Eligible calls go through the SAME kernel_select contest linear/bmm/
    conv2d/conv3d/group_norm already use, instead of always preferring the
    vendored kernel unconditionally -- there was never a benchmark backing
    that preference (see enable_flash_attn_rocwmma's docstring: "no reason
    to assume this beats TE just because it's a dedicated kernel"), and the
    only measurement this project HAS made for a different op (conv2d fp16)
    found its own hand-written kernel losing to stock outright. 'fallback'
    is whatever F.scaled_dot_product_attention was before this wrapper
    installed itself -- TE-patched if the main enable() already ran, stock
    otherwise -- and always succeeds, so it's the last-resort candidate the
    same way stock is for every other contest in this file."""
    def _fallback():
        return _flash_attn_rocwmma_fallback(query, key, value, attn_mask=attn_mask,
                                             dropout_p=dropout_p, is_causal=is_causal,
                                             scale=scale, **kwargs)

    if not _is_flash_attn_rocwmma_eligible(query, key, value, attn_mask, dropout_p, is_causal):
        return _fallback()

    def _try_flash_rocwmma():
        try:
            return flash_attn_rocwmma_ops.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, scale=scale, is_causal=is_causal
            )
        except (RuntimeError, TypeError):
            return None

    if not kernel_select.enabled():
        # AMD_TUNED_TORCH_MEASURE_KERNELS=0: restore the old fixed
        # always-prefer-flash-when-eligible ordering, same escape hatch
        # every other contest in this file offers.
        out = _try_flash_rocwmma()
        return _fallback() if out is None else out

    # No stride/padding/dilation here (this isn't a conv) -- built directly
    # rather than through kernel_select.pick, the same way _patched_linear/
    # _patched_bmm/_patched_group_norm build their own keys. is_causal is
    # part of the key because it changes which candidate is even eligible
    # (_is_flash_attn_rocwmma_eligible's query_len==key_len requirement),
    # not just how fast the winner is; attn_mask's actual content isn't --
    # flash_rocwmma never accepts one at all (already excluded above), and
    # the fallback candidate handles any mask shape identically regardless
    # of contest outcome.
    contest_key = (query.dtype, tuple(query.shape), tuple(key.shape), tuple(value.shape),
                   bool(is_causal))
    won = kernel_select.cached_key("sdpa", contest_key)
    if won == "fallback":
        return _fallback()
    if won is None:
        out = kernel_select.pick_key("sdpa", contest_key, [
            ("flash_rocwmma", _try_flash_rocwmma),
            ("fallback", _fallback),
        ])
        if out is not None:
            return out
    elif won == "flash_rocwmma":
        out = _try_flash_rocwmma()
        if out is not None:
            return out
    return _fallback()


def enable_flash_attn_rocwmma() -> None:
    """OPT-IN, and not recommended -- set
    AMD_TUNED_TORCH_FLASH_ATTN_ROCWMMA=1 to install it (it used to be
    default-on; see the trigger at the bottom of this file, and the
    measurements below, for why it no longer is). Tries
    flash_attn_rocwmma_ops's
    vendored rocWMMA FlashAttention-2 kernel first for eligible calls
    (see _is_flash_attn_rocwmma_eligible: no attn_mask, no dropout, 4D
    query/key/value, causal only when query_len == key_len), falling back
    to whatever F.scaled_dot_product_attention currently is -- stock, or
    TE-patched if the main enable() already ran -- for anything else, if
    the kernel isn't available, or if it raises.

    VALIDATED, and now correct but still slower. This docstring used to
    say the backward was BROKEN, which it was: the checks below were first
    run on a real RX 7900 XTX (gfx1100) via
    tools/bench_flash_attn_rocwmma.py and step 4 failed outright. Six real
    bugs in the vendored kernel were found and fixed as a result (dQ
    missing the ln(2) rescale out of the exp2 domain, dQ accumulated
    across workgroups that did not own it, dO read with the padded Q's
    strides while never padded in its sequence dimension, bf16's infinite
    mask sentinel turning fully-masked padded rows into NaN, guards
    deleted by -ffinite-math-only, and gradients accumulated in half
    precision) -- see _vendor/rocwmma_fattn/NOTICE.md for the itemised
    list and each FIX comment in the .cu sources. Re-run results:

      1. BUILD: fine. The JIT build succeeds and
         flash_attn_rocwmma_ops.available() is True.
      2. FORWARD NUMERICS: fine. Max error vs an fp32 reference is within
         the benchmark's 8x-stock line on every shape, which is ordinary
         for a different summation order in half precision.
      3. FORWARD SPEED: still loses on every shape measured, 0.52x-1.00x
         of stock across all 12. There is no shape where it wins, so the
         kernel_select contest below can only ever pick the fallback.
      4. BACKWARD NUMERICS: now fine. dQ/dK/dV max error vs fp32 is
         5.2e-4 to 3.4e-2 against stock's 2.1e-4 to 1.9e-2 on the
         identical shapes -- 1.2x to 3.8x stock, where it used to be three
         orders of magnitude out. The benchmark's summary line is
         "numerics: no shape exceeded 8x stock's own error vs the fp32
         reference".
      5. BACKWARD SPEED: loses, 0.35x-0.73x of stock. Part of that is the
         fix for the dQ race: dQ now gets its own pass over the tile grid,
         which costs one extra recomputation of Si and dPi per tile.

    Beyond the benchmark, the kernel is checked by
    tools/ (see the scratch test harness referenced in
    flash_attn_rocwmma_ops.py's docstring): 56 shape/dtype/causal
    combinations forward and backward, plus a determinism sweep (144
    combinations x 5 repeats, bitwise identical) and a clean run under
    AMD_SERIALIZE_KERNEL=3 with HIP_LAUNCH_BLOCKING=1.

    So: it is no longer unsafe, but it is still not faster. Turn it on to
    work on it, or if you specifically want this code path; there is
    currently no performance reason to prefer it over stock.

    Separately, and now fixed in _is_flash_attn_rocwmma_eligible: the
    kernel has no MQA/GQA support and used to accept those shapes, reading
    past the end of K/V and intermittently trapping as "HIP error: an
    illegal memory access was encountered".

    So: do not turn this on to use it. Turn it on only to work on it --
    fixing the backward is the prerequisite for this tier being worth
    anything, and step 4 above is the test to fix it against.

    Benchmarking against stock/TE is not something you need to do by hand:
    eligible calls go through the same kernel_select
         contest linear/bmm/conv2d/conv3d/group_norm already use (see
         _patched_sdpa_flash_attn_rocwmma) -- the first call for each
         distinct (dtype, Q/K/V shape, is_causal) measures this kernel
         against whatever F.scaled_dot_product_attention would otherwise
         have been (TE if enabled, stock otherwise) and caches whichever
         actually won, in memory and on disk (kernel_select.debug_winners()
         shows the current decisions). Set AMD_TUNED_TORCH_MEASURE_KERNELS=0
         to go back to the old unconditional-prefer-this-kernel ordering,
         or AMD_TUNED_TORCH_FLASH_ATTN_ROCWMMA=0 to disable this tier
         entirely.

    Disable ordering, if you also use the main enable()/disable(): undo
    these LIFO -- call disable_flash_attn_rocwmma() BEFORE disable(), not
    after. See enable_conv3d_winograd_fp16's docstring for exactly why
    (the same _ORIGINALS-bookkeeping hazard applies here)."""
    global _FLASH_ATTN_ROCWMMA_ENABLED, _flash_attn_rocwmma_fallback
    if _FLASH_ATTN_ROCWMMA_ENABLED:
        return
    if not flash_attn_rocwmma_ops.available():
        import warnings
        warnings.warn(
            "amd_tuned_torch.enable_flash_attn_rocwmma(): JIT build failed, "
            f"F.scaled_dot_product_attention left as-is ({flash_attn_rocwmma_ops.load_error()})"
        )
        return
    _flash_attn_rocwmma_fallback = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = _patched_sdpa_flash_attn_rocwmma
    _FLASH_ATTN_ROCWMMA_ENABLED = True


def disable_flash_attn_rocwmma() -> None:
    """Restore whatever F.scaled_dot_product_attention was before
    enable_flash_attn_rocwmma()."""
    global _FLASH_ATTN_ROCWMMA_ENABLED, _flash_attn_rocwmma_fallback
    if not _FLASH_ATTN_ROCWMMA_ENABLED:
        return
    F.scaled_dot_product_attention = _flash_attn_rocwmma_fallback
    _flash_attn_rocwmma_fallback = None
    _FLASH_ATTN_ROCWMMA_ENABLED = False


# ---------------------------------------------------------------------------
# RMSNorm (triton-kernels) -- DEFAULT ON, same
# AMD_TUNED_TORCH_TRITON_KERNELS_RMSNORM bottom-of-file trigger shape as
# enable_flash_attn_rocwmma above, and safe for the same reason: it
# degrades to a warning + no-op when triton_kernels isn't installed (see
# enable_triton_kernels_rmsnorm's `if not triton_kernels_ops.available()`
# branch) rather than failing hard -- on this checkout specifically,
# source/triton-kernels isn't vendored/installed, so this trigger fires
# and immediately no-ops with a warning until that sibling package is
# installed (see triton_kernels_ops.py's module docstring). Unlike
# flash_attn_rocwmma, this ALSO has no backward pass at all (unlike
# te_ops.rms_norm's real torch.autograd.Function), so this tier is
# grad-gated the same way linear_fp16/bmm_fp16/conv2d/group_norm are
# (_grad_safe), and nothing here has been benchmarked against either TE's
# rms_norm or stock F.rms_norm on RX 7900 XTX -- default-on does not mean
# validated, see enable_triton_kernels_rmsnorm's docstring. Composes with
# enable()/disable() regardless of call order the same way
# enable_flash_attn_rocwmma does (captures whatever F.rms_norm currently is
# as its own fallback) -- same disable-ordering caveat: disable this before
# the main disable(), not after (see enable_conv3d_winograd_fp16's
# docstring for the mechanism/why).
# ---------------------------------------------------------------------------

_TRITON_KERNELS_RMSNORM_ENABLED = False
_triton_kernels_rmsnorm_fallback: Callable | None = None

_TRITON_KERNELS_RMSNORM_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _is_triton_kernels_rmsnorm_eligible(input, weight) -> bool:
    """True only for the shape/dtype/autograd combination
    triton_kernels.rmsnorm.rmsnorm actually supports: a weight tensor (not
    None -- F.rms_norm's own signature allows omitting it, but the kernel
    requires one), same last-dim size as input, and no live autograd
    (_grad_safe -- the kernel has no backward pass)."""
    if weight is None or input.shape[-1] != weight.shape[0]:
        return False
    if not _dispatch._grad_safe(input, weight):
        return False
    return _dispatch._usable(input, weight, dtypes=_TRITON_KERNELS_RMSNORM_DTYPES)


def _patched_rms_norm_triton_kernels(input, normalized_shape, weight=None, eps=None):
    if not _is_triton_kernels_rmsnorm_eligible(input, weight):
        return _triton_kernels_rmsnorm_fallback(input, normalized_shape, weight, eps)
    try:
        return triton_kernels_ops.rms_norm(input, weight, eps)
    except (RuntimeError, TypeError):
        return _triton_kernels_rmsnorm_fallback(input, normalized_shape, weight, eps)


def enable_triton_kernels_rmsnorm() -> None:
    """Called automatically at import time (see the
    AMD_TUNED_TORCH_TRITON_KERNELS_RMSNORM trigger at the bottom of this
    file; set it to "0" to opt out). Tries triton_kernels.rmsnorm.rmsnorm
    first for eligible calls (see _is_triton_kernels_rmsnorm_eligible: a
    weight tensor, matching last-dim size, no live autograd), falling back
    to whatever F.rms_norm currently is -- stock, or TE-patched if the
    main enable() already ran -- for anything else, if triton_kernels
    isn't installed, or if it raises.

    Being default-on does NOT mean this has been validated -- it still
    has not, on any hardware this project has access to. Before trusting
    output from this path on your actual RX 7900 XTX:
      1. Confirm `import triton_kernels` succeeds --
         amd_tuned_torch.triton_kernels_ops.available(). If it doesn't
         (e.g. source/triton-kernels isn't installed, the default state
         of this checkout), this function already no-ops with a warning
         and F.rms_norm is untouched.
      2. Compare its output against F.rms_norm numerically -- nothing in
         this package has verified correctness of triton_kernels' rmsnorm
         against PyTorch's own.
      3. Benchmark it against both stock and TE's rms_norm (if
         available) for your actual shapes -- there's no reason to assume
         this beats either just because it's a dedicated Triton kernel.
         Set AMD_TUNED_TORCH_TRITON_KERNELS_RMSNORM=0 if it doesn't.

    Disable ordering, if you also use the main enable()/disable(): undo
    these LIFO -- call disable_triton_kernels_rmsnorm() BEFORE disable(),
    not after. See enable_conv3d_winograd_fp16's docstring for exactly why
    (the same _ORIGINALS-bookkeeping hazard applies here)."""
    global _TRITON_KERNELS_RMSNORM_ENABLED, _triton_kernels_rmsnorm_fallback
    if _TRITON_KERNELS_RMSNORM_ENABLED:
        return
    if not hasattr(F, "rms_norm"):
        return
    if not triton_kernels_ops.available():
        import warnings
        warnings.warn(
            "amd_tuned_torch.enable_triton_kernels_rmsnorm(): triton_kernels not installed, "
            "F.rms_norm left as-is"
        )
        return
    _triton_kernels_rmsnorm_fallback = F.rms_norm
    F.rms_norm = _patched_rms_norm_triton_kernels
    _TRITON_KERNELS_RMSNORM_ENABLED = True


def disable_triton_kernels_rmsnorm() -> None:
    """Restore whatever F.rms_norm was before enable_triton_kernels_rmsnorm()."""
    global _TRITON_KERNELS_RMSNORM_ENABLED, _triton_kernels_rmsnorm_fallback
    if not _TRITON_KERNELS_RMSNORM_ENABLED:
        return
    F.rms_norm = _triton_kernels_rmsnorm_fallback
    _triton_kernels_rmsnorm_fallback = None
    _TRITON_KERNELS_RMSNORM_ENABLED = False


# ---------------------------------------------------------------------
# BOFT (Butterfly Orthogonal Fine-Tuning) fast_block_diag
# ---------------------------------------------------------------------
# Unlike every other tier in this file, this one patches a THIRD-PARTY
# package (peft.tuners.boft.layer), not torch.nn.functional -- BOFT's
# block-diagonal assembly has no F.* entry point to intercept. The
# enable/disable bookkeeping is the same shape regardless: capture what was
# there, install, restore exactly that on the way out.
_BOFT_ENABLED = False
_boft_fallback: tuple | None = None


def enable_boft() -> None:
    """Point source/peft's BOFT tuner at this package's compiled
    fast_block_diag kernel (src/cuda/fast_block_diag.cu) instead of the
    runtime torch.utils.cpp_extension.load() JIT build it would otherwise
    attempt -- see amd_tuned_torch.boft_ops's module docstring for the full
    motivation, and patch_peft_boft()'s for the mechanism.

    NOT called at import time, and deliberately so -- unlike every other
    enable_* here, this one reaches into `peft`, so firing it automatically
    would drag peft (and transformers behind it) into the import of any
    program that so much as imports amd_tuned_torch. Opt in explicitly, or
    set AMD_TUNED_TORCH_BOFT=1 (see the trigger in __init__.py).

    TIMING MATTERS -- call this BEFORE get_peft_model(model, BOFTConfig(...)).
    Each BOFTLayer reads get_fbd_cuda() ONCE, in its own __init__, to decide
    its fbd_cuda_available flag; layers built before this call keep whatever
    they decided then. See patch_peft_boft()'s docstring.

    WHAT IT BUYS, measured on this project's own hardware (RX 7900 XTX,
    ROCm 7.2 -- tools/bench_boft.py, tests_hardware/test_boft_ops.py):

      * PEFT's own JIT extension DOES NOT BUILD against torch 2.15/ROCm 7.2
        (its hipified fbd kernel fails on torch's headeronly Dispatch.h), so
        upstream silently falls back to a Python torch.block_diag loop. This
        tier is currently the only way a BOFT adapter gets a compiled
        fast_block_diag here at all.
      * 19-160x faster than that fallback on real boft_R shapes (forward),
        3-31x forward+backward, bitwise identical output in fp16/bf16/fp32.
      * Multi-factor BOFT becomes reachable: upstream clamps
        boft_n_butterfly_factor to 1 whenever get_fbd_cuda() is falsy,
        because its fallback squeezes dim 0 and can only build ONE
        block-diagonal.
      * bf16 works, which upstream's own extension never supported
        (AT_DISPATCH_FLOATING_TYPES_AND_HALF covers fp16/fp32/fp64 only).

    No-ops with a warning if peft isn't installed, same posture as
    enable_triton_kernels_rmsnorm() with triton_kernels."""
    global _BOFT_ENABLED, _boft_fallback
    if _BOFT_ENABLED:
        return
    try:
        from peft.tuners.boft import layer as _boft_layer
    except ImportError:
        import warnings
        warnings.warn(
            "amd_tuned_torch.enable_boft(): peft not installed, nothing patched"
        )
        return
    # Captured BEFORE patching so disable_boft() restores exactly what was
    # here -- including a JIT build a previous BOFTLayer already resolved.
    # Reading these attributes does not invoke upstream's loader.
    _boft_fallback = (_boft_layer._FBD_CUDA, _boft_layer.get_fbd_cuda)
    if not boft_ops.patch_peft_boft():
        _boft_fallback = None
        return
    _BOFT_ENABLED = True


def disable_boft() -> None:
    """Restore peft.tuners.boft.layer's own get_fbd_cuda/_FBD_CUDA."""
    global _BOFT_ENABLED, _boft_fallback
    if not _BOFT_ENABLED:
        return
    from peft.tuners.boft import layer as _boft_layer
    _boft_layer._FBD_CUDA, _boft_layer.get_fbd_cuda = _boft_fallback
    _boft_fallback = None
    _BOFT_ENABLED = False


def is_boft_enabled() -> bool:
    return _BOFT_ENABLED


def fast_block_diag(input):
    """block_diag(input[z, 0], ..., input[z, N-1]) for every z at once --
    [z, N, b, b] -> [z, N*b, N*b], with a real backward. Never
    monkeypatched (torch.block_diag takes varargs matrices, not a batched
    4D tensor, so there is no stock signature to intercept), so a caller
    wanting the kernel outside PEFT calls this directly. See
    amd_tuned_torch.boft_ops's module docstring.

    Bitwise identical to torch.block_diag over the unbound blocks in
    fp16/bf16/fp32/fp64 -- it is pure data movement, no arithmetic. Note
    that it is bandwidth-bound on the ZEROED OUTPUT, not on the scatter:
    the (N*b, N*b) result is mostly zeros and costs a full memset to
    materialize, which is ~98% of the runtime at realistic sizes. If you
    are applying a block-diagonal to something, a batched matmul over the
    (b, b) blocks avoids materializing it at all and is asymptotically
    better -- see flash_mm_kernel._monarch_group_kernel, which does exactly
    that. This op is for when you genuinely need the dense matrix."""
    return boft_ops.fast_block_diag(input)


def mla_decode(x, kv_cache, kv_len, q_proj_down_weight, q_proj_up_weight,
                kv_proj_down_weight, kv_proj_up_weight, wo_weight,
                n_heads, nope_dim, rope_dim, v_dim, rope_theta: float = 10000.0):
    """DeepSeek-V3-style Multi-Head Latent Attention decode step (query
    sequence length 1), against a compressed KV cache -- never
    monkeypatched (there is no F.* op for this to replace), so model code
    using MLA calls this directly. See amd_tuned_torch.mla_ops's module
    docstring for the weight-absorption algorithm, argument shapes, and
    what to verify before relying on it (UNVALIDATED on real hardware)."""
    return mla_ops.mla_decode(
        x, kv_cache, kv_len, q_proj_down_weight, q_proj_up_weight,
        kv_proj_down_weight, kv_proj_up_weight, wo_weight,
        n_heads, nope_dim, rope_dim, v_dim, rope_theta=rope_theta,
    )


def fused_add_rms_norm(x, residual, weight, eps: float = 1e-6):
    """Fused `residual = residual + x; rms_norm(residual) * weight` in one
    Triton kernel pass instead of two, with a REAL backward pass -- never
    monkeypatched (there is no F.* op combining add+rmsnorm to
    intercept), so model code calls this directly from its decoder-
    layer's pre-norm epilogue. See amd_tuned_torch.fused_norm_ops's
    module docstring for the exact calling convention, the backward
    formula (Liger-Kernel's RMSNorm backward, extended for this op's
    second "new residual" output), and what to verify before relying on
    it (UNVALIDATED on real hardware)."""
    return fused_norm_ops.fused_add_rms_norm(x, residual, weight, eps=eps)


def rotary_embedding(positions, query, key, head_size: int, cos_sin_cache):
    """Fused NeoX-style rotary embedding (Triton), applied to query/key IN
    PLACE, with a REAL backward pass -- never monkeypatched (there is no
    F.* op for RoPE to intercept), so model code calls this directly from
    its attention layer's Q/K projection. Full-head rotation only
    (rotary_dim == head_size); NOT for MLA's RoPE (see
    amd_tuned_torch.mla_decode instead -- incompatible shape conventions,
    explained in amd_tuned_torch.rope_ops's module docstring). See that
    docstring for the exact calling convention, why the backward pass is
    just this same rotation with sin negated, and what to verify before
    relying on this (UNVALIDATED on gfx1100 specifically)."""
    return rope_ops.rotary_embedding(positions, query, key, head_size, cos_sin_cache)


def compute_rope_cos_sin_cache(base: float, rotary_dim: int, max_position_embeddings: int):
    """[max_position_embeddings, rotary_dim] cos/sin cache for
    rotary_embedding() -- build once (e.g. at model init), not per call.
    See amd_tuned_torch.rope_ops.compute_cos_sin_cache."""
    return rope_ops.compute_cos_sin_cache(base, rotary_dim, max_position_embeddings)


def fused_linear_cross_entropy(input, weight, target, bias=None, ce_weight=None,
                                ignore_index: int = -100, label_smoothing: float = 0.0,
                                reduction: str = "mean", softcap=None):
    """Fused final-linear-layer + cross-entropy loss with a REAL backward
    pass -- the first training-capable (has-a-backward) kernel in this
    package. Never monkeypatched (F.cross_entropy takes already-computed
    logits, not hidden_states + a weight matrix), so call this directly
    from a training loop's loss computation in place of
    `F.cross_entropy(hidden @ weight.T + bias, target)`. See
    amd_tuned_torch.fused_ce_ops's module docstring for the memory-saving
    mechanism (never materializes the full (batch*seq, vocab) logits
    tensor), the exact calling convention, what was deliberately not
    ported from upstream Liger-Kernel (amd_tuned_torch/_vendor/
    liger_fused_ce/NOTICE.md), and what to verify before relying on this
    for real training (UNVALIDATED on gfx1100 specifically)."""
    return fused_ce_ops.fused_linear_cross_entropy(
        input, weight, target, bias=bias, ce_weight=ce_weight,
        ignore_index=ignore_index, label_smoothing=label_smoothing,
        reduction=reduction, softcap=softcap,
    )


def bias_swiglu(input, bias=None, clamp_value=None):
    """Fused bias-add + SwiGLU with a real backward pass -- never
    monkeypatched (no F.* op for "bias-add then SwiGLU" to intercept), so
    call this directly from an FFN/MoE block's forward. Unlike every
    other Triton-backed op in this package, this one is plain PyTorch (no
    triton dependency, always available) -- see
    amd_tuned_torch.swiglu_ops's module docstring for the fusion boundary
    this covers (aiter's fused_silu_mul and TE's silu both require the
    bias already added), the calling convention, and what to verify
    before relying on this (UNVALIDATED on real hardware)."""
    return swiglu_ops.bias_swiglu(input, bias=bias, clamp_value=clamp_value)


def compute_smoothquant_scale(weight: torch.Tensor, alpha: float = 0.5):
    """The per-input-channel SmoothQuant scale for `weight`, or None if
    calibrate_smoothquant() hasn't collected activation data for it yet."""
    return aiter_ops.compute_smoothquant_scale(weight, alpha=alpha)


def set_smooth_scale(weight: torch.Tensor, smooth_scale: torch.Tensor) -> None:
    """Register a precomputed SmoothQuant scale for `weight` directly,
    bypassing calibrate_smoothquant()'s forward-hook data collection."""
    aiter_ops.set_smooth_scale(weight, smooth_scale)


def is_int8_linear_enabled() -> bool:
    return _INT8_LINEAR_ENABLED


# Re-exported at the package top level via `from ._opt_in_tiers import *` in
# __init__.py -- this split must not change what amd_tuned_torch.X resolves
# to for any X below.
__all__ = [
    "_INT8_LINEAR_DTYPES",
    "_INT8_LINEAR_ENABLED", "_int8_linear_fallback", "_patched_linear_int8",
    "enable_int8_linear", "disable_int8_linear",
    "calibrate_smoothquant",
    "_CONV3D_WINOGRAD_FP16_ENABLED", "_conv3d_winograd_fp16_fallback",
    "_is_winograd_eligible_conv3d", "_patched_conv3d_winograd_fp16",
    "enable_conv3d_winograd_fp16", "disable_conv3d_winograd_fp16",
    "_FLASH_ATTN_ROCWMMA_ENABLED", "_flash_attn_rocwmma_fallback",
    "_FLASH_ATTN_ROCWMMA_DTYPES", "_is_flash_attn_rocwmma_eligible",
    "_patched_sdpa_flash_attn_rocwmma",
    "enable_flash_attn_rocwmma", "disable_flash_attn_rocwmma",
    "_TRITON_KERNELS_RMSNORM_ENABLED", "_triton_kernels_rmsnorm_fallback",
    "_TRITON_KERNELS_RMSNORM_DTYPES", "_is_triton_kernels_rmsnorm_eligible",
    "_patched_rms_norm_triton_kernels",
    "enable_triton_kernels_rmsnorm", "disable_triton_kernels_rmsnorm",
    "_BOFT_ENABLED", "_boft_fallback",
    "enable_boft", "disable_boft", "is_boft_enabled", "fast_block_diag",
    "mla_decode", "fused_add_rms_norm", "rotary_embedding",
    "compute_rope_cos_sin_cache", "fused_linear_cross_entropy", "bias_swiglu",
    "compute_smoothquant_scale", "set_smooth_scale", "is_int8_linear_enabled",
]
