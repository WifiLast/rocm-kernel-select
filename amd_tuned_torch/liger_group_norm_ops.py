"""Real-backward Triton GroupNorm, ported from LinkedIn's Liger Kernel --
see amd_tuned_torch/_vendor/liger_group_norm/NOTICE.md for exact
provenance (commit, what changed, why this is close to a direct copy
rather than a reformulated port like this package's other Liger-derived
modules).

WHY THIS EXISTS. amd_tuned_torch's own native HIP group_norm kernel
(src/cuda/group_norm.cu, reached via amd_tuned_torch.ops.group_norm /
the default F.group_norm monkeypatch) is FORWARD-ONLY -- `_grad_safe()`
falls back to stock the instant any input requires_grad, and during LoRA/
LyCORIS training that is true for essentially every call in the network
(the same "some upstream adapter needs a gradient, so every activation
downstream of it does too" finding already documented for conv2d/linear/
bmm's own grad-safety gates in this package). SDXL's UNet is built almost
entirely out of GroupNorm-normalized ResNet blocks, so the package's
default GroupNorm tier gives a LoRA training script exactly zero
acceleration on that op today -- not a narrow gap, the common case. This
module is a SEPARATE tier with a genuine `torch.autograd.Function`
backward, specifically so a training script can opt into GroupNorm
acceleration where the default monkeypatch structurally cannot help.

WHY A DROP-IN nn.Module INSTEAD OF A MONKEYPATCH. F.group_norm has no
"which candidate is faster AND differentiable" contest the way conv2d/
linear do (kernel_select verifies candidates against a no-grad reference
and this kernel's whole reason to exist is the grad-enabled case that
contest doesn't cover), and unconditionally monkeypatching F.group_norm
package-wide would apply it to every caller including ones with no LoRA
adapter anywhere near them, not just this one training script's UNet.
`LigerGroupNorm` is instead a plain nn.GroupNorm-compatible module a
caller swaps into a specific model (see replace_group_norm_modules()
below) -- explicit opt-in on the exact module tree it's meant for, same
posture as this package's other "plain library surface, not a monkeypatch
target" tiers (cumesh_ops, flexgemm_ops's sparse-3D ops, flash_mm_kernel).

VALIDATION STATUS -- READ BEFORE TRUSTING THIS ON HARDWARE. Unlike most
"unvalidated" ports in this package, the kernel this was copied from DOES
have real upstream AMD CI (Liger-Kernel ships official, actively-tested
ROCm support -- see its own README's AMD CI badge and "Full AMD support"
v0.4.0 release note) -- this is not a "no ROCm claim at all" situation
like Megatron's SwiGLU (swiglu_ops.py) or the vendored flash-attention
kernel. What IS still unvalidated is THIS SPECIFIC COPY: it has never been
compiled or run in this project's own environment (no Triton install, no
CUDA/ROCm device here -- see available() below), on gfx1100 specifically,
or against this package's own dtype/shape conventions. Before relying on
this for real training: compare `group_norm(x, ...)` output AND gradients
(x.grad, weight.grad, bias.grad) against `F.group_norm`'s own for your
actual (batch, channels, *spatial, num_groups, dtype) combinations -- a
plain numerical mismatch is the realistic failure mode for a kernel this
close to verbatim-copied from a well-tested source, not a subtle one.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def available() -> bool:
    """True if Triton is importable and a CUDA or ROCm device is visible.
    LigerGroupNorm/group_norm() both fall back to stock F.group_norm
    (real backward either way, just not this module's fused kernel) when
    this is False, so callers never have to gate on it themselves."""
    return _TRITON_AVAILABLE and torch.cuda.is_available()


_MAX_FUSED_SIZE = 65536  # no NPU backend to special-case smaller for, unlike upstream


if _TRITON_AVAILABLE:
    import operator

    def _compare_version(op, other):
        parts = lambda s: tuple(int(p) for p in s.split(".")[:2] if p.isdigit())
        return op(parts(triton.__version__), parts(other))

    if _compare_version(operator.ge, "3.0.0"):
        try:
            # Correctly backend-dispatching (CUDA or ROCm) already -- see
            # this module's own NOTICE.md for why it's kept as upstream
            # wrote it rather than branched on AMD specifically.
            from triton.language.extra.libdevice import rsqrt as _rsqrt
        except ModuleNotFoundError:
            from triton.language.extra.cuda.libdevice import rsqrt as _rsqrt
    else:
        from triton.language.math import rsqrt as _rsqrt

    @triton.jit
    def _group_norm_forward_kernel(
        Y_ptr, Y_row_stride, Y_col_stride,
        X_ptr, X_row_stride, X_col_stride,
        Mean_ptr, Mean_row_stride, Mean_col_stride,
        RSTD_ptr, RSTD_row_stride, RSTD_col_stride,
        W_ptr, B_ptr,
        hidden_size, channels_per_group, eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        """References: https://nn.labml.ai/normalization/group_norm/index.html"""
        batch_idx = tl.program_id(0)
        group_idx = tl.program_id(1)

        X_ptr += batch_idx * X_row_stride + group_idx * X_col_stride
        Y_ptr += batch_idx * Y_row_stride + group_idx * Y_col_stride

        block_range = tl.arange(0, BLOCK_SIZE)

        s = 0.0
        squared_sum = 0.0
        for i in tl.range(0, hidden_size, BLOCK_SIZE):
            hidden_size_offsets = i + block_range
            mask = hidden_size_offsets < hidden_size
            X = tl.load(X_ptr + hidden_size_offsets, mask=mask, other=0.0)
            s += tl.sum(X)
            squared_sum += tl.sum(X * X)

        m = s / hidden_size
        variance = (squared_sum / hidden_size) - (m * m)
        rstd = _rsqrt(variance + eps.to(tl.float32))

        # Flat loop over full hidden_size (not per-channel) -- avoids the
        # nested channel x per_channel_hidden loop where BLOCK_SIZE >>
        # hidden_size_per_channel wastes padding.
        hidden_size_per_channel = hidden_size // channels_per_group
        for i in tl.range(0, hidden_size, BLOCK_SIZE):
            hidden_size_offsets = i + block_range
            mask = hidden_size_offsets < hidden_size
            X = tl.load(X_ptr + hidden_size_offsets, mask=mask, other=m)
            local_channel = hidden_size_offsets // hidden_size_per_channel
            global_channel = group_idx * channels_per_group + local_channel
            W = tl.load(W_ptr + global_channel, mask=mask)
            B = tl.load(B_ptr + global_channel, mask=mask)
            Y = (X - m) * rstd * W + B
            tl.store(Y_ptr + hidden_size_offsets, Y, mask=mask)

        tl.store(Mean_ptr + batch_idx * Mean_row_stride + group_idx * Mean_col_stride, m)
        tl.store(RSTD_ptr + batch_idx * RSTD_row_stride + group_idx * RSTD_col_stride, rstd)

    @triton.jit
    def _group_norm_backward_kernel(
        X_ptr, X_row_stride, X_col_stride,
        W_ptr,
        Mean_ptr, Mean_ptr_row_stride, Mean_ptr_col_stride,
        RSTD_ptr,
        DX_ptr, DW_ptr, DB_ptr,
        UPSTREAM_ptr,
        hidden_size: tl.constexpr, channels_per_group: tl.constexpr,
        BLOCK_SIZE: tl.constexpr, dtype: tl.constexpr,
    ):
        """References:
        https://nn.labml.ai/normalization/group_norm/index.html
        https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md

        Same backprop equations as layer_norm -- the mean/rstd here are
        computed over channels_per_group * hidden_size elements (all
        channels in the group), not hidden_size alone."""
        batch_idx = tl.program_id(0)
        group_idx = tl.program_id(1)

        X_ptr += batch_idx * X_row_stride
        DX_ptr += batch_idx * X_row_stride
        UPSTREAM_ptr += batch_idx * X_row_stride

        mean = tl.load(Mean_ptr + batch_idx * Mean_ptr_row_stride + group_idx * Mean_ptr_col_stride)
        rstd = tl.load(RSTD_ptr + batch_idx * Mean_ptr_row_stride + group_idx * Mean_ptr_col_stride)

        c1 = 0.0
        c2 = 0.0
        block_range = tl.arange(0, BLOCK_SIZE)

        for channel_idx in range(group_idx * channels_per_group, (group_idx + 1) * channels_per_group):
            dW = 0.0
            dB = 0.0
            W = tl.load(W_ptr + channel_idx)
            for i in tl.range(0, hidden_size, BLOCK_SIZE):
                hidden_size_offsets = i + block_range
                mask = hidden_size_offsets < hidden_size
                X = tl.load(X_ptr + channel_idx * X_col_stride + hidden_size_offsets, mask=mask, other=0.0)
                UPSTREAM_grad = tl.load(UPSTREAM_ptr + channel_idx * X_col_stride + hidden_size_offsets,
                                        mask=mask, other=0.0)

                x_hat = (X - mean) * rstd
                dW += tl.sum(UPSTREAM_grad * x_hat)
                dB += tl.sum(UPSTREAM_grad)

                wdy = W * UPSTREAM_grad
                c1 += tl.sum(x_hat * wdy)
                c2 += tl.sum(wdy)

            # Additions to the same channel from different groups' programs
            # must be atomic.
            tl.atomic_add(DW_ptr + channel_idx, dW.to(dtype))
            tl.atomic_add(DB_ptr + channel_idx, dB.to(dtype))

        N = hidden_size * channels_per_group
        c1 = c1 / N
        c2 = c2 / N

        for channel_idx in tl.range(group_idx * channels_per_group, (group_idx + 1) * channels_per_group):
            W = tl.load(W_ptr + channel_idx)
            for i in range(0, hidden_size, BLOCK_SIZE):
                hidden_size_offsets = i + block_range
                mask = hidden_size_offsets < hidden_size
                X = tl.load(X_ptr + channel_idx * X_col_stride + hidden_size_offsets, mask=mask, other=0.0)
                UPSTREAM_grad = tl.load(UPSTREAM_ptr + channel_idx * X_col_stride + hidden_size_offsets,
                                        mask=mask, other=0.0)

                x_hat = (X - mean) * rstd
                wdy = W * UPSTREAM_grad
                dx = (wdy - (x_hat * c1 + c2)) * rstd
                tl.store(DX_ptr + channel_idx * X_col_stride + hidden_size_offsets, dx, mask=mask)


def _group_norm_forward(X, num_channels, num_groups, W, B, eps):
    shape = X.shape
    batch_size = shape[0]
    channels_per_group = num_channels // num_groups
    X = X.view(batch_size, num_groups, -1).contiguous()
    hidden_size = X.shape[-1]
    BLOCK_SIZE = min(_MAX_FUSED_SIZE, triton.next_power_of_2(hidden_size))
    Y = torch.empty((batch_size, num_groups, hidden_size), dtype=X.dtype, device=X.device)
    Mean = torch.zeros((batch_size, num_groups), dtype=X.dtype, device=X.device)
    RSTD = torch.zeros((batch_size, num_groups), dtype=X.dtype, device=X.device)

    _group_norm_forward_kernel[(batch_size, num_groups)](
        Y, Y.stride(0), Y.stride(1),
        X, X.stride(0), X.stride(1),
        Mean, Mean.stride(0), Mean.stride(1),
        RSTD, RSTD.stride(0), RSTD.stride(1),
        W, B, hidden_size, channels_per_group, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return Y.view(*shape), X.view(*shape), Mean, RSTD


def _group_norm_backward(dY, X, W, B, Mean, RSTD, num_channels, num_groups):
    shape = dY.shape
    batch_size = shape[0]
    hidden_size = dY.shape[-1]
    channels_per_group = num_channels // num_groups
    dY = dY.view(batch_size, num_groups, -1)
    DX = torch.empty((batch_size, num_groups, hidden_size * channels_per_group),
                      dtype=X.dtype, device=X.device)
    DW = torch.zeros((num_channels,), dtype=W.dtype, device=W.device)
    DB = torch.zeros((num_channels,), dtype=B.dtype, device=B.device)
    triton_dtype = tl.float32 if X.dtype == torch.float32 else tl.bfloat16

    BLOCK_SIZE = min(_MAX_FUSED_SIZE, triton.next_power_of_2(hidden_size))
    _group_norm_backward_kernel[(batch_size, num_groups)](
        X, X.stride(0), X.stride(1),
        W, Mean, Mean.stride(0), Mean.stride(1), RSTD,
        DX, DW, DB, dY,
        hidden_size, channels_per_group,
        BLOCK_SIZE=BLOCK_SIZE, dtype=triton_dtype,
    )
    return DX.view(*shape), DW, DB


class _LigerGroupNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, bias, num_channels, num_groups, eps):
        X = X.contiguous()
        Y, X, Mean, RSTD = _group_norm_forward(X, num_channels, num_groups, weight, bias, eps)
        ctx.num_channels = num_channels
        ctx.num_groups = num_groups
        ctx.save_for_backward(X, weight, bias, Mean, RSTD)
        return Y

    @staticmethod
    def backward(ctx, dY):
        X, W, B, Mean, RSTD = ctx.saved_tensors
        DX, DW, DB = _group_norm_backward(dY.contiguous(), X, W, B, Mean, RSTD,
                                          ctx.num_channels, ctx.num_groups)
        return DX, DW, DB, None, None, None


def group_norm(x: torch.Tensor, num_groups: int, weight: torch.Tensor, bias: torch.Tensor,
               eps: float = 1e-5) -> torch.Tensor:
    """Drop-in for F.group_norm(x, num_groups, weight, bias, eps), with a
    real backward through x/weight/bias either way. Uses the Triton kernel
    when available() is True; falls back to F.group_norm itself otherwise
    (same numerics contract, just not this module's fused kernel) -- so
    callers never need to check available() themselves."""
    if not available() or not x.is_cuda:
        return F.group_norm(x, num_groups, weight, bias, eps)
    num_channels = x.shape[1]
    try:
        return _LigerGroupNormFunction.apply(x, weight, bias, num_channels, num_groups, eps)
    except (RuntimeError, TypeError):
        return F.group_norm(x, num_groups, weight, bias, eps)


class LigerGroupNorm(nn.GroupNorm):
    """nn.GroupNorm drop-in replacement backed by group_norm() above --
    same constructor signature/parameters as nn.GroupNorm (in fact IS one,
    so an existing module's state_dict loads into this unchanged), only
    forward() differs. See replace_group_norm_modules() to swap every
    nn.GroupNorm in an existing model tree for one of these in place."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return group_norm(x, self.num_groups, self.weight, self.bias, self.eps)


def replace_group_norm_modules(model: nn.Module) -> int:
    """Replaces every nn.GroupNorm submodule in `model` (recursively, in
    place) with a LigerGroupNorm carrying the SAME weight/bias Parameters
    (not copies -- the original Parameter objects are reused, so this is
    safe to call on an already-training model, an optimizer already
    holding references to those Parameters keeps working, and a
    checkpoint saved before/after this call has identical keys/values for
    them) and the same num_groups/eps/affine. Returns the number of
    modules replaced -- 0 means either the model has no GroupNorm layers,
    or (see group_norm()'s own fallback) LigerGroupNorm will just call
    stock F.group_norm every time, e.g. when available() is False.

    Only affine (`elementwise_affine=True`, i.e. has weight/bias) GroupNorm
    layers are replaced -- LigerGroupNormFunction always expects tensors for
    weight/bias (upstream Liger's own contract), and every GroupNorm this
    package's own training scripts target (SDXL's UNet) is affine, so this
    is not expected to skip anything in practice."""
    replaced = 0
    for name, child in list(model.named_children()):
        if isinstance(child, nn.GroupNorm) and not isinstance(child, LigerGroupNorm) and child.affine:
            new_module = LigerGroupNorm(child.num_groups, child.num_channels, eps=child.eps, affine=True)
            new_module.weight = child.weight
            new_module.bias = child.bias
            setattr(model, name, new_module)
            replaced += 1
        else:
            replaced += replace_group_norm_modules(child)
    return replaced
