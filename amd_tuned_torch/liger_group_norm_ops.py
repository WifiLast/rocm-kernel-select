"""Real-backward Triton GroupNorm, ported from LinkedIn's Liger Kernel --
see amd_tuned_torch/_vendor/liger_group_norm/NOTICE.md for exact
provenance (commit, what changed, why this is close to a direct copy
rather than a reformulated port like this package's other Liger-derived
modules).

WHY THIS EXISTS. amd_tuned_torch's own native HIP group_norm kernel
(src/cuda/group_norm.cu, reached via amd_tuned_torch.ops.group_norm /
the default F.group_norm monkeypatch) is FORWARD-ONLY. That used to mean
`_grad_safe()` sent every GroupNorm call straight back to stock during
training -- essentially every call in the network, since once one LoRA/
LyCORIS adapter needs a gradient autograd must backprop through every
frozen layer between it and the loss, so nearly every activation
requires_grad. SDXL's UNet and VAE are built almost entirely out of
GroupNorm-normalized ResNet blocks, so the default tier gave a training
script zero GroupNorm acceleration: not a narrow gap, the common case,
and the reason this module was written.

THAT IS NO LONGER THE ONLY WAY IN. `_GroupNormFn` in _dispatch.py now
wraps the default tier's accelerated forward in a real
torch.autograd.Function whose backward is GroupNorm's textbook
derivative in plain torch ops, so the default F.group_norm monkeypatch
accelerates the FORWARD during training too, without this module. What
this module still offers over that is a fused Triton kernel on both
sides -- one kernel for the forward, one for the backward, instead of an
accelerated forward plus a backward built out of several separate
reduction/elementwise ops. Which one actually wins on a given shape is
not measured here, so treat this tier as an experiment to benchmark
rather than an automatic upgrade; train_lycoris_mod.py keeps it behind
--enable_liger_groupnorm for that reason as much as for the bug history
below.

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

VALIDATION STATUS. This copy HAS now been run on gfx1100 (RX 7900 XTX,
torch 2.15 / ROCm 7.2, Triton), and the check this section used to ask for
-- output plus x.grad/weight.grad/bias.grad against F.group_norm, on the
real 4D SDXL GroupNorm shapes in fp32 and fp16 -- is a test:
source/cmp_ext_turing/tests/test_liger_group_norm_ops_gpu.py. It skips
itself when available() is False, so a CPU-only checkout still gets the
F.group_norm-fallback coverage in test_liger_group_norm_ops.py and nothing
else.

That first hardware run found three real bugs, all of them in the paths a
LoRA/LyCORIS SDXL training script actually takes, and all now fixed:

  1. DX under-allocated in _group_norm_backward (HIP 700). The backward
     kept upstream Liger's `hidden_size = dY.shape[-1]`, which is the
     per-channel element count only for the 3D inputs upstream documents;
     for 4D NCHW it is just W, while the backward kernel indexes per
     channel with a H*W stride. DX was allocated at 40,960 elements for
     (1, 320, 128, 128) with 32 groups and written ~5.2M elements in --
     an out-of-bounds device write, hipErrorIllegalAddress on the first
     backward pass of every run using this tier. See that function's own
     SHAPE CONTRACT docstring.
  2. fp16 reductions overflowing to inf. tl.sum over an fp16 block
     returns fp16, and one SDXL group is up to 163,840 elements; the sum
     of that many unit-variance squares exceeds fp16's 65504 ceiling, so
     variance/rstd came out garbage and ~15% of the forward output was
     wrong by up to 13.1 absolute -- silent corruption of an fp16
     training run, which is the default precision for SDXL. Both kernels
     now load into fp32 and Mean/RSTD are fp32 buffers.
  3. bf16 atomics into an fp16 buffer. `dtype` was tl.bfloat16 for every
     non-fp32 input, so an fp16 run atomically added bf16 values into an
     fp16 DW/DB. Those are fp32 accumulators now, cast back to the
     parameter dtype on return.

Bugs 1 and 2 were both invisible to the `except RuntimeError -> stock
F.group_norm` guard in group_norm(): an async illegal address has already
poisoned the HIP context by the time anything raises, and wrong numbers
raise nothing at all.

Even so, train_lycoris_mod.py keeps this tier OPT-IN (--enable_liger_groupnorm,
default off) rather than trusting the green test: stock nn.GroupNorm is
always correct, and this is the one tier that has taken the device down.
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

        # .to(tl.float32) on every load in this kernel is load-bearing,
        # not defensive: tl.sum over an fp16 block returns fp16, and one
        # SDXL group is channels_per_group * H * W elements -- 10 * 128 *
        # 128 = 163,840 for the first UNet block. Summing that many
        # unit-variance squares in fp16 OVERFLOWS (fp16 tops out at
        # 65504): squared_sum became inf, variance and rstd garbage, and
        # ~15% of the output tensor came back wrong by up to 13.1
        # absolute against F.group_norm -- silent training corruption,
        # not noise. Stock GroupNorm accumulates in fp32 for exactly this
        # reason; Mean/RSTD are fp32 buffers now too (see
        # _group_norm_forward), so the backward pass reads them back at
        # full precision as well.
        s = 0.0
        squared_sum = 0.0
        for i in tl.range(0, hidden_size, BLOCK_SIZE):
            hidden_size_offsets = i + block_range
            mask = hidden_size_offsets < hidden_size
            X = tl.load(X_ptr + hidden_size_offsets, mask=mask, other=0.0).to(tl.float32)
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
            # other=0.0 (not other=m) since these lanes are masked out
            # of the store below anyway, and mixing an fp32 `m` into an
            # fp16 load's `other` is exactly the kind of implicit
            # promotion this kernel should not be relying on.
            X = tl.load(X_ptr + hidden_size_offsets, mask=mask, other=0.0).to(tl.float32)
            local_channel = hidden_size_offsets // hidden_size_per_channel
            global_channel = group_idx * channels_per_group + local_channel
            W = tl.load(W_ptr + global_channel, mask=mask).to(tl.float32)
            B = tl.load(B_ptr + global_channel, mask=mask).to(tl.float32)
            Y = (X - m) * rstd * W + B
            # Y is fp32; the store converts to Y_ptr's own dtype.
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
            W = tl.load(W_ptr + channel_idx).to(tl.float32)
            for i in tl.range(0, hidden_size, BLOCK_SIZE):
                hidden_size_offsets = i + block_range
                mask = hidden_size_offsets < hidden_size
                # fp32 for the same overflow reason as the forward
                # kernel -- these reductions run over the whole group.
                X = tl.load(X_ptr + channel_idx * X_col_stride + hidden_size_offsets,
                            mask=mask, other=0.0).to(tl.float32)
                UPSTREAM_grad = tl.load(UPSTREAM_ptr + channel_idx * X_col_stride + hidden_size_offsets,
                                        mask=mask, other=0.0).to(tl.float32)

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
            W = tl.load(W_ptr + channel_idx).to(tl.float32)
            for i in range(0, hidden_size, BLOCK_SIZE):
                hidden_size_offsets = i + block_range
                mask = hidden_size_offsets < hidden_size
                X = tl.load(X_ptr + channel_idx * X_col_stride + hidden_size_offsets,
                            mask=mask, other=0.0).to(tl.float32)
                UPSTREAM_grad = tl.load(UPSTREAM_ptr + channel_idx * X_col_stride + hidden_size_offsets,
                                        mask=mask, other=0.0).to(tl.float32)

                x_hat = (X - mean) * rstd
                wdy = W * UPSTREAM_grad
                dx = (wdy - (x_hat * c1 + c2)) * rstd
                # dx is fp32; the store converts to DX_ptr's own dtype.
                tl.store(DX_ptr + channel_idx * X_col_stride + hidden_size_offsets, dx, mask=mask)


def _group_norm_forward(X, num_channels, num_groups, W, B, eps):
    shape = X.shape
    batch_size = shape[0]
    channels_per_group = num_channels // num_groups
    X = X.view(batch_size, num_groups, -1).contiguous()
    hidden_size = X.shape[-1]
    BLOCK_SIZE = min(_MAX_FUSED_SIZE, triton.next_power_of_2(hidden_size))
    Y = torch.empty((batch_size, num_groups, hidden_size), dtype=X.dtype, device=X.device)
    # fp32 regardless of X's dtype: these are the per-group mean and
    # reciprocal std the backward pass reads back, and rounding them to
    # fp16 throws away precision the kernel just spent fp32 accumulators
    # earning (upstream keeps them in X.dtype).
    Mean = torch.zeros((batch_size, num_groups), dtype=torch.float32, device=X.device)
    RSTD = torch.zeros((batch_size, num_groups), dtype=torch.float32, device=X.device)

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
    """Gradients for _group_norm_forward.

    SHAPE CONTRACT -- the reason this is not upstream Liger's version.
    _group_norm_backward_kernel indexes PER CHANNEL: `channel_idx *
    X_col_stride + offsets`, with `offsets < hidden_size`. So its
    `hidden_size` is the per-CHANNEL element count (H*W for 4D NCHW, H for
    3D) and `X_col_stride` the per-channel stride -- unlike
    _group_norm_forward, which was rewritten here to walk each group's
    whole flat span at once and whose `hidden_size` is therefore
    per-GROUP (channels_per_group * H * W).

    Upstream Liger takes `hidden_size = dY.shape[-1]`, which is only the
    per-channel count for the 3D (batch, channels, hidden) inputs it
    documents. Handed SDXL's 4D NCHW GroupNorm activations that is just W:
    the kernel then walked 1/H of each channel and, worse, DX was
    allocated at `hidden_size * channels_per_group` elements per group
    while the kernel's own stores reach `channel_idx * H*W + W` -- for the
    first SDXL block, (1, 320, 128, 128) with 32 groups, a 40,960-element
    buffer written up to ~5.2M elements in. That is an out-of-bounds
    device write: HIP 700 / hipErrorIllegalAddress on the first backward
    pass of any run using this tier, which is what it did on gfx1100. The
    `except RuntimeError -> F.group_norm` guard in group_norm() cannot
    catch it either, since an async illegal address has already poisoned
    the HIP context by the time anything raises.

    DX therefore has to be X's own full shape, and (since the kernel
    reuses X_row_stride/X_col_stride to address DX) laid out exactly like
    X -- empty_like on a contiguous X, with the strides passed explicitly
    below rather than read off X, so the two can't drift apart again.
    """
    shape = X.shape
    batch_size = shape[0]
    channels_per_group = num_channels // num_groups
    # Per-channel element count: H*W for 4D NCHW, H for 3D, 1 for 2D --
    # derived from numel so it holds for any spatial rank, unlike
    # shape[-1].
    hidden_size = X.numel() // (batch_size * num_channels)

    X = X.contiguous()
    dY = dY.contiguous()
    DX = torch.empty_like(X)
    # fp32 accumulators for the per-channel weight/bias gradients, cast
    # back at the end. Upstream atomically adds these in the parameter
    # dtype, which here would be a genuine correctness bug and a
    # precision one: `dtype` was tl.bfloat16 for every non-fp32 input, so
    # an fp16 run (SDXL mixed precision) atomically added bf16 values
    # into an fp16 buffer; and summing H*W*batch terms into one fp16
    # scalar loses accuracy badly even when the types do line up.
    DW = torch.zeros((num_channels,), dtype=torch.float32, device=W.device)
    DB = torch.zeros((num_channels,), dtype=torch.float32, device=B.device)

    BLOCK_SIZE = min(_MAX_FUSED_SIZE, triton.next_power_of_2(hidden_size))
    _group_norm_backward_kernel[(batch_size, num_groups)](
        X, num_channels * hidden_size, hidden_size,
        W, Mean, Mean.stride(0), Mean.stride(1), RSTD,
        DX, DW, DB, dY,
        hidden_size, channels_per_group,
        BLOCK_SIZE=BLOCK_SIZE, dtype=tl.float32,
    )
    return DX.view(*shape), DW.to(W.dtype), DB.to(B.dtype)


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
