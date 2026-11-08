"""amd_tuned_torch.conv_fusion_ops -- fuses the elementwise work AROUND a
convolution (GroupNorm -> SiLU -> Conv, and Conv -> +residual) into fewer
kernels, for SDXL-shaped UNet/VAE workloads on gfx1100.

WHAT IS ACTUALLY FUSED, AND WHY IT IS NOT THE CONVOLUTION ITSELF.
"Conv fusion" on AMD usually means MIOpen's fusion API
(miopenCreateFusionPlan + miopenCreateOpConvForward/OpBiasForward/
OpActivationForward). That route is CLOSED here, and not for a reason any
environment variable can fix: PyTorch does not link it. `libtorch_hip.so`
imports 76 MIOpen symbols and not one fusion-API symbol, so
MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_BIAS_ACTIV and its neighbours gate solvers
that nothing in this stack can ever select. Confirmed against the webui's
own libtorch; tools/analyse_miopen_log.py re-checks it on any log.

So the convolution stays a single MIOpen call and this module fuses what
surrounds it. Which turns out to be where the time is anyway.

WHERE THE TIME ACTUALLY IS, MEASURED. On the 2,648 forward convs of one
SDXL 2048x1024 generation (log/miopen/miopen-conv-20260917-133654.log,
analysed with tools/analyse_miopen_log.py):

  * the conv EPILOGUE (bias + residual add) costs ~3-6% of conv time --
    155GB of elementwise traffic against 229 TFLOP of convolution.
  * the conv PROLOGUE (GroupNorm + SiLU, which in SDXL's ResBlock runs
    BEFORE every conv: sgm/modules/diffusionmodules/openaimodel.py orders
    them GroupNorm -> SiLU -> Conv) costs ~17% of conv time.

The prologue is the prize, and it is the one a patched F.conv2d can never
reach: by the time conv2d is called, GroupNorm and SiLU have already run
and written their outputs to memory. That is why this module works at
nn.Module level (replace_conv_blocks) instead of joining _dispatch.py's
function patches. Fusion needs to see more than one op; monkeypatching one
op cannot.

THREE CANDIDATES, AND LAYOUT DECIDES BETWEEN THEM.

  ck            Composable Kernel's GroupNorm with SiLU fused
                (src/cuda/ck_norm_fwd*.cu via ck_norm_ops). CK HAS this
                kernel -- DeviceNormalizationFwd takes a
                YElementwiseOperation epilogue and CK ships Silu/Swish --
                and this package already builds it. Nothing new had to be
                written for the case that matters.
  fused_triton  the Triton kernel in this file. NCHW only.
  stock         GroupNorm then SiLU, two kernels.

Measured against stock, fp16, RX 7900 XTX, all three interleaved in one
session:

    shape              layout    stock    triton       ck
    c1280 64x32        NCHW     0.106m   0.084m    declines
    c1280 64x32        NHWC     0.215m   0.268m     0.118m
    c640 128x64        NHWC     0.176m   0.256m     0.158m
    c320 256x128       NHWC     0.395m   0.608m     0.313m
    c1280 32x16        NHWC     0.044m   0.082m     0.019m

CK wins every channels_last shape (1.12x-2.39x) and declines NCHW
outright -- its own docstring's measurements show the permutes into and
out of NHWC cost more than the fusion saves there. The Triton kernel is
the opposite: a small win in NCHW, and SLOWER THAN STOCK in channels_last,
which is why it now declines that layout rather than quietly calling
.contiguous() and paying for a transpose nobody asked for.

So the contest key includes the MEMORY FORMAT, not just the shape. A key
that ignored it would cache one layout's winner and then apply it to the
other, which is precisely backwards given the table above.

AND THEN THE CATCH: CHANNELS_LAST IS UNUSABLE ON THIS BOX, SO THE CK
COLUMN ABOVE IS UNREACHABLE IN PRACTICE. Reaching it means putting the
model in channels_last (`--opt-channelslast` in the webui), and on this
gfx1100 that makes CONVOLUTION 239x-753x slower -- measured on stock
torch.conv2d, fp16, steady state, on this workload's own shapes:

    n2 c1280 32x16 k1280    NCHW   0.663 ms    NHWC   158.3 ms    239x
    n2 c1280 64x32 k1280    NCHW   2.003 ms    NHWC   944.7 ms    472x
    n2 c640 128x64 k640     NCHW   1.359 ms    NHWC  1023.0 ms    753x
    n2 c320 256x128 k320    NCHW   1.464 ms    NHWC   981.5 ms    670x

First call is ~0.2s, so that is not a solver search being amortised: the
NHWC kernels themselves are pathological here. In NCHW, MIOpen runs
miopenSp3AsmConv_v30_3_1_gfx11_fp16_dot2_f2x3_stride1 (Winograd) for 77%
of calls and falls off it entirely in NHWC.

The convolution is ~95% of the work in these blocks and the norm fusion is
worth a few percent, so trading the former for the latter is never right
on this hardware. Note also that PYTORCH_MIOPEN_SUGGEST_NHWC=1 being set
in the environment does NOT put tensors in channels_last -- it is a hint to
MIOpen about kernel selection. The captured workload is 2864/2864 NCHW.

So on this machine the live candidates are fused_triton and stock, and the
CK path is dead code that stays in the contest only because it costs one
decline to ask. Do not go looking for a way to enable it.

MEASURE THE DOMINANT COST CENTRE FIRST. This whole detour happened because
a layout change was recommended on the strength of the NORM numbers without
first measuring what it did to the CONV, which is 95% of the time. That is
the general lesson worth more than the specific figures above.

END-TO-END BLOCK SPEEDUP IS STILL NOT ESTABLISHED. The per-kernel numbers
above are consistent across runs, but what replace_conv_blocks does to a
whole ResBlock is not measurable on this machine right now. A NULL TEST --
the identical unmodified model on BOTH sides of the same interleaved A/B
harness -- reports

    c320 256x128  0.55x        c640 128x64  0.90x
    c1280 64x32   1.00x        c1280 32x16  1.00x

i.e. a noise floor up to +-45%, which swamps the few percent a block-level
win would be. Not thermal (40C, 19W) -- the card sits at Performance Level
"auto" and idles at 96MHz mclk / 0MHz sclk, so a short benchmark burst is
timed somewhere on the clock ramp. Pin the clocks before believing any
end-to-end number:

    rocm-smi --setperflevel high        # and --setperflevel auto after

Established: the kernels are NUMERICALLY CORRECT (checked against the ATen
builtin on every shape, dtype and layout) and the contest picks per shape
AND layout. The block-level win remains predicted, not observed.

RELATIONSHIP TO liger_group_norm_ops. That module also replaces GroupNorm
modules (replace_group_norm_modules) with a Triton kernel, and it has a
real backward pass, so it is the one to use for TRAINING. This module is
forward-only and fuses across the GroupNorm/SiLU boundary, which Liger's
does not -- it normalizes and returns, leaving SiLU as a second kernel and
a second round trip. For inference the extra fusion is the point; for
training use liger_group_norm_ops and leave this off.

A NOTE ON THE STOCK REFERENCE. _dispatch.enable() patches F.group_norm,
so `_stock_group_norm_silu` below deliberately reaches for the ORIGINAL
captured in _ORIGINALS (falling back to torch.group_norm, the ATen builtin)
rather than calling whatever is currently installed. Not because the
patched path is slow -- measured with an interleaved A/B it is 1.00x on
every shape this workload uses, i.e. indistinguishable from stock, with
only the smallest (n2 c1280 32x16) showing 1.19x of dispatch overhead --
but because a contest should measure against the real baseline rather than
against another tier that happens to be installed. An earlier version of
this comment claimed the patched path was 3-5x SLOWER, from a
one-then-the-other measurement; that number was an artifact of the clock
drift documented above and is retracted.

STATUS. The Triton kernel here is new and has been run on gfx1100, but the
end-to-end claim ("an SDXL generation gets faster") has NOT been measured
-- only the per-block microbenchmarks above, which do not capture
allocator pressure or what else is resident during a real generation.
Validate on a real run before believing a number.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import kernel_select
from . import ck_norm_ops

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without Triton
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


_ENV = "AMD_TUNED_TORCH_CONV_FUSION"


def available() -> bool:
    """True if the fused path can run at all: Triton importable and a GPU
    visible. Everything public here falls back to stock when this is
    False, so callers never have to check."""
    return _TRITON_AVAILABLE and torch.cuda.is_available()


def enabled() -> bool:
    """Whether fusion is allowed to be installed. Default ON when
    available; AMD_TUNED_TORCH_CONV_FUSION=0 turns it off without
    editing model code."""
    return available() and os.environ.get(_ENV, "1") != "0"


# ---------------------------------------------------------------------------
# Fused GroupNorm + SiLU
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:

    @triton.jit
    def _gn_silu_kernel(
        X, Y, W, B,
        C, HW, groups, channels_per_group, eps,
        BLOCK: tl.constexpr,
        HAS_AFFINE: tl.constexpr,
    ):
        """One program per (batch, group). Two passes over the group's
        elements: statistics, then normalize+affine+SiLU.

        EVERY ACCUMULATOR IS fp32, DELIBERATELY. tl.sum over a fp16 tensor
        returns fp16, and a GroupNorm group at SDXL sizes is 40 channels x
        64x32 = 81,920 elements -- the sum of squares overflows fp16's
        65,504 ceiling long before the reduction finishes, silently
        producing inf and then NaN output. Loads are cast to fp32 on the
        way in and only the final store narrows back."""
        pid = tl.program_id(0)
        n = pid // groups
        g = pid % groups

        group_elems = channels_per_group * HW
        base = n * C * HW + g * channels_per_group * HW

        acc = tl.zeros([BLOCK], dtype=tl.float32)
        acc_sq = tl.zeros([BLOCK], dtype=tl.float32)
        for off in range(0, group_elems, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            mask = idx < group_elems
            v = tl.load(X + base + idx, mask=mask, other=0.0).to(tl.float32)
            acc += v
            acc_sq += v * v

        total = tl.sum(acc, axis=0)
        total_sq = tl.sum(acc_sq, axis=0)
        mean = total / group_elems
        var = total_sq / group_elems - mean * mean
        rstd = 1.0 / tl.sqrt(var + eps)

        for off in range(0, group_elems, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            mask = idx < group_elems
            v = tl.load(X + base + idx, mask=mask, other=0.0).to(tl.float32)
            h = (v - mean) * rstd
            if HAS_AFFINE:
                ch = g * channels_per_group + idx // HW
                w = tl.load(W + ch, mask=mask, other=1.0).to(tl.float32)
                b = tl.load(B + ch, mask=mask, other=0.0).to(tl.float32)
                h = h * w + b
            # SiLU fused in: no second kernel, no round trip through memory.
            h = h * tl.sigmoid(h)
            tl.store(Y + base + idx, h.to(Y.dtype.element_ty), mask=mask)


def _fused_group_norm_silu(x: torch.Tensor, num_groups: int,
                           weight: Optional[torch.Tensor],
                           bias: Optional[torch.Tensor], eps: float):
    """Triton GroupNorm+SiLU. Returns None (declines) for channels-last.

    This kernel indexes NCHW: channel = (offset within group) // HW. Feeding
    it a channels_last tensor used to "work" because of the .contiguous()
    below, which silently transposes the whole tensor first -- and that
    transpose costs more than the fusion saves. Measured on channels_last
    it came out SLOWER than stock on every shape tried (0.268ms vs 0.215ms
    at n2 c1280 64x32), i.e. the fallback was paying for a layout change
    nobody asked for. Channels-last belongs to the CK candidate, which has
    a real NHWC kernel."""
    if x.is_contiguous(memory_format=torch.channels_last) and not x.is_contiguous():
        return None
    x = x.contiguous()
    n, c = x.shape[0], x.shape[1]
    hw = 1
    for d in x.shape[2:]:
        hw *= d
    y = torch.empty_like(x)
    cpg = c // num_groups
    block = 1024
    _gn_silu_kernel[(n * num_groups,)](
        x, y, weight, bias,
        c, hw, num_groups, cpg, eps,
        BLOCK=block,
        HAS_AFFINE=weight is not None and bias is not None,
        num_warps=8,
    )
    return y


def _stock_group_norm_silu(x, num_groups, weight, bias, eps):
    """The unfused reference: GroupNorm then SiLU, two kernels.

    Reaches for _dispatch's captured ORIGINAL F.group_norm rather than
    whatever is installed. enable() replaces F.group_norm with a patched
    version that is 3-5x SLOWER than stock on these exact shapes (see the
    module docstring), and contesting against that would make this module
    look good for the wrong reason."""
    try:
        from . import _dispatch
        gn = _dispatch._ORIGINALS.get((F, "group_norm"))
    except Exception:
        gn = None
    if gn is None:
        # torch.group_norm is the ATen builtin. Prefer it over F.group_norm
        # as the fallback: F.group_norm is exactly the symbol enable()
        # replaces, so reading it here can pick up the patched version --
        # which is both the thing being contested against and, on these
        # shapes, slower than real stock.
        gn = torch.group_norm
    return F.silu(gn(x, num_groups, weight, bias, eps))


def group_norm_silu(x: torch.Tensor, num_groups: int,
                    weight: Optional[torch.Tensor] = None,
                    bias: Optional[torch.Tensor] = None,
                    eps: float = 1e-5) -> torch.Tensor:
    """SiLU(GroupNorm(x)) in one kernel where that is faster, stock where
    it is not -- decided per shape by kernel_select's contest, not by a
    threshold in this file. See the module docstring for the 0.59x-to-2.40x
    spread that makes the contest necessary.

    Forward only: falls back to stock whenever a gradient is live, so an
    autograd graph is never built through the Triton kernel."""
    if (not enabled() or not x.is_cuda or x.dim() < 3
            or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or num_groups <= 0 or x.shape[1] % num_groups != 0
            or torch.is_grad_enabled() and (x.requires_grad
                                            or (weight is not None and weight.requires_grad))):
        return _stock_group_norm_silu(x, num_groups, weight, bias, eps)

    def ck_fused():
        # CK's own GroupNorm with SiLU fused (src/cuda/ck_norm_fwd*.cu via
        # ck_norm_ops). It returns None for layouts/dtypes it will not serve
        # -- notably plain NCHW, where its docstring's measurements show the
        # permutes cost more than the fusion saves -- and the contest treats
        # that as a decline.
        if not ck_norm_ops.available():
            return None
        return ck_norm_ops.group_norm_silu(x, num_groups, weight, bias, eps)

    def triton_fused():
        return _fused_group_norm_silu(x, num_groups, weight, bias, eps)

    def stock():
        return _stock_group_norm_silu(x, num_groups, weight, bias, eps)

    # Keyed on the LAYOUT as well as the shape: which candidate wins flips
    # entirely between NCHW and channels_last (CK wins every channels_last
    # shape measured, by 1.1x-2.4x, and declines NCHW outright), so a key
    # that ignored memory format would cache one layout's winner and apply
    # it to the other.
    layout = "nhwc" if (x.is_contiguous(memory_format=torch.channels_last)
                        and not x.is_contiguous()) else "nchw"
    key = ("gn_silu", tuple(x.shape), str(x.dtype), num_groups,
           weight is not None, bias is not None, layout)
    out = kernel_select.pick_key("group_norm_silu", key,
                                 [("ck", ck_fused),
                                  ("fused_triton", triton_fused),
                                  ("stock", stock)])
    return stock() if out is None else out


# ---------------------------------------------------------------------------
# Module-level installation
# ---------------------------------------------------------------------------

class FusedGroupNormSiLUConv2d(nn.Module):
    """Drop-in for nn.Sequential(GroupNorm, SiLU, Conv2d).

    Holds the ORIGINAL GroupNorm and Conv2d submodules rather than copying
    their parameters, so this is safe to install on a live model: the same
    Parameter objects stay referenced, an optimizer holding them keeps
    working, and a state_dict saved after this call still round-trips
    (the submodule names are preserved as "0" and "2", matching the
    nn.Sequential this replaced -- index "1" was the stateless SiLU)."""

    def __init__(self, norm: nn.GroupNorm, conv: nn.Conv2d):
        super().__init__()
        # Names chosen to match the nn.Sequential indices they replace, so
        # existing checkpoints load without a key remap.
        self.add_module("0", norm)
        self.add_module("2", conv)

    @property
    def norm(self) -> nn.GroupNorm:
        return getattr(self, "0")

    @property
    def conv(self) -> nn.Conv2d:
        return getattr(self, "2")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = self.norm
        h = group_norm_silu(x, norm.num_groups, norm.weight, norm.bias, norm.eps)
        return self.conv(h)

    def extra_repr(self) -> str:
        return "fused=GroupNorm+SiLU"


def _is_target(seq: nn.Module) -> bool:
    if not isinstance(seq, nn.Sequential) or len(seq) != 3:
        return False
    a, b, c = seq[0], seq[1], seq[2]
    return (isinstance(a, nn.GroupNorm) and a.affine
            and isinstance(b, nn.SiLU)
            and isinstance(c, nn.Conv2d))


def replace_conv_blocks(model: nn.Module) -> int:
    """Replace every nn.Sequential(GroupNorm, SiLU, Conv2d) in `model`
    (recursively, in place) with a FusedGroupNormSiLUConv2d. Returns how
    many were replaced.

    This is the shape SDXL's ResBlock builds for both its in_layers and
    out_layers, so on an SDXL UNet this finds every ResBlock conv. It
    matches structurally rather than by class name, so it does not depend
    on which repository's ResBlock is in use.

    Returns 0 and changes nothing when enabled() is False, so it is safe to
    call unconditionally. Idempotent: an already-fused block is not a
    Sequential any more and will not match twice.

    NOT called by amd_tuned_torch.enable(). enable() patches functions and
    can run at import time; this rewrites a model tree, which needs a model
    and is the caller's decision to make after they have built one."""
    if not enabled():
        return 0
    replaced = 0
    for name, child in list(model.named_children()):
        if _is_target(child):
            setattr(model, name, FusedGroupNormSiLUConv2d(child[0], child[2]))
            replaced += 1
        else:
            replaced += replace_conv_blocks(child)
    return replaced


def unfuse_conv_blocks(model: nn.Module) -> int:
    """Undo replace_conv_blocks, restoring the original nn.Sequential with
    a fresh nn.SiLU in the middle. Returns how many were restored. The
    GroupNorm and Conv2d objects are the same ones, so parameters and
    optimizer state survive the round trip."""
    restored = 0
    for name, child in list(model.named_children()):
        if isinstance(child, FusedGroupNormSiLUConv2d):
            setattr(model, name, nn.Sequential(child.norm, nn.SiLU(), child.conv))
            restored += 1
        else:
            restored += unfuse_conv_blocks(child)
    return restored


def describe() -> str:
    """One-line status, for rocm_env_check-style reporting."""
    if not _TRITON_AVAILABLE:
        return "conv_fusion: unavailable -- Triton not importable"
    if not torch.cuda.is_available():
        return "conv_fusion: unavailable -- no GPU"
    if not enabled():
        return f"conv_fusion: disabled by {_ENV}=0"
    return ("conv_fusion: available -- GroupNorm+SiLU fused, per-shape contest "
            "against stock via kernel_select")
