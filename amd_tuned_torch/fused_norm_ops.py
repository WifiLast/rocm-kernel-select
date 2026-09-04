"""Fused residual-add + RMSNorm -- the standard pre-norm transformer
epilogue (`hidden = residual + sublayer_out; normed = rmsnorm(hidden)`,
with `hidden` becoming the next layer's residual) done as a single kernel
instead of two full read-modify-write passes over (batch*seq, hidden_dim)
tensors. Every decoder layer of every transformer model runs this once
per sublayer (attention and FFN), so on RX 7900 XTX's memory bandwidth
this is a real bottleneck at inference batch sizes where compute is
otherwise cheap.

This is the first in-repo `@triton.jit` kernel in this package -- every
other Triton-backed op here (aiter_ops.py, triton_kernels_ops.py) wraps
an externally-installed Triton-kernel package instead of defining its own
kernel. Consequently, unlike ck_norm_ops.py's group_norm (a native HIP
kernel built by setup.py) or hipblaslt_ops.py (linked against a system
library), `triton` itself is NOT a declared dependency of this package
anywhere (no requirements.txt/pyproject.toml exists, and setup.py never
mentions it) -- it happens to be importable in a dev environment that
also has PyTorch's Inductor backend set up, but that is not guaranteed by
anything this package installs. available() reports False cleanly if
`import triton` fails, same shape as every other optional backend here.

HAS A REAL BACKWARD PASS -- added after this module's initial forward-
only version. The backward formula is LinkedIn Liger-Kernel's own
(`src/liger_kernel/ops/rms_norm.py`'s `_rms_norm_backward_kernel`,
BSD-2-Clause -- see amd_tuned_torch/_vendor/liger_rms_norm/NOTICE.md for
full provenance and what was adapted), applied to this kernel's
fused-add calling convention:

    dhidden = rstd*(dout*w) - (1/N)*rstd^3*sum((dout*w)*hidden)*hidden
            + dnew_residual        (identity gradient through hidden = x + residual)
    dx = dresidual = dhidden       (both x and residual feed hidden identically)
    dweight = sum over rows of (dout * hidden * rstd)

The `+ dnew_residual` term is this kernel's own addition, not something
Liger's plain (non-fused-add) RMSNorm needs: this op returns TWO outputs
(the normed value AND the updated residual for the next layer), and
`new_residual` is exactly `hidden` before normalization -- so a gradient
arriving through that second output path is an identity pass-through
into `dhidden`, summed with whatever the normalization math itself
contributes. Getting this term right (not just the RMSNorm formula in
isolation) is precisely what tests_hardware/test_fused_norm_ops.py's
multi-step decode-loop-style backward test is checking.

dweight accumulation uses `tl.atomic_add` across all rows into one fp32
buffer, a deliberate simplification vs. Liger's own
per-SM-scratch-buffer-plus-host-side-sum approach (which avoids atomics
entirely for better performance under high row-count contention, at the
cost of needing `sm_count`/`rows_per_program` grid-partitioning logic).
Atomics are simpler to reason about and verify by hand at the cost of
some contention under very large batch*seq; revisit if profiling on real
hardware shows this mattering.

UNVALIDATED, same caveat as every other kernel in this package ported or
written without ROCm hardware available while writing it: verify
numerically (loss AND gradients) against `residual + x` followed by
stock F.rms_norm/manual RMSNorm + autograd before trusting this for
anything, and benchmark against the unfused version for your actual
shapes -- see tests_hardware/test_fused_norm_ops.py.
"""
from __future__ import annotations

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def available() -> bool:
    return _TRITON_AVAILABLE


if _TRITON_AVAILABLE:

    @triton.jit
    def _fused_add_rmsnorm_fwd_kernel(
        x_ptr, residual_ptr, weight_ptr, out_ptr, new_residual_ptr, rstd_ptr,
        n_cols,
        x_row_stride, residual_row_stride, out_row_stride, new_residual_row_stride,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        # One program per row (i.e. per token): load that row's x and
        # residual, add in fp32 (more accurate than adding in fp16/bf16
        # and then computing variance on the rounded result -- this is
        # the same accumulate-in-fp32 convention every RMSNorm
        # implementation in this package already uses, e.g.
        # ck_norm_ops's native kernel), write the new residual back out
        # in the working dtype, then normalize. rstd is cached per row
        # for backward, same reason Liger's own forward kernel caches it
        # instead of recomputing it there.
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + row * residual_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        hidden = x + residual

        out_dtype = x_ptr.dtype.element_ty
        tl.store(new_residual_ptr + row * new_residual_row_stride + cols, hidden.to(out_dtype), mask=mask)

        variance = tl.sum(hidden * hidden, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(variance + eps)
        tl.store(rstd_ptr + row, rstd)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = hidden * rstd * weight

        tl.store(out_ptr + row * out_row_stride + cols, out.to(out_dtype), mask=mask)

    @triton.jit
    def _fused_add_rmsnorm_bwd_kernel(
        grad_out_ptr, grad_new_residual_ptr, new_residual_ptr, weight_ptr, rstd_ptr,
        grad_x_ptr, grad_residual_ptr, grad_weight_ptr,
        n_cols,
        grad_out_row_stride, grad_new_residual_row_stride, new_residual_row_stride,
        grad_x_row_stride, grad_residual_row_stride,
        HAS_RESIDUAL_GRAD: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        # See module docstring for the formula. `hidden` is read back from
        # new_residual (the forward already wrote it there in the working
        # dtype -- no separate fp32 save, same tradeoff Liger's own
        # backward makes by re-upcasting its saved X rather than keeping
        # an extra fp32 buffer around).
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        grad_out = tl.load(grad_out_ptr + row * grad_out_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        hidden = tl.load(new_residual_ptr + row * new_residual_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + row)

        m = grad_out * weight
        dot = tl.sum(m * hidden, axis=0)
        dhidden = rstd * m - (1.0 / n_cols) * rstd * rstd * rstd * dot * hidden

        if HAS_RESIDUAL_GRAD:
            grad_new_residual = tl.load(
                grad_new_residual_ptr + row * grad_new_residual_row_stride + cols, mask=mask, other=0.0
            ).to(tl.float32)
            dhidden += grad_new_residual

        out_dtype = grad_x_ptr.dtype.element_ty
        dhidden_out = dhidden.to(out_dtype)
        tl.store(grad_x_ptr + row * grad_x_row_stride + cols, dhidden_out, mask=mask)
        tl.store(grad_residual_ptr + row * grad_residual_row_stride + cols, dhidden_out, mask=mask)

        dweight = grad_out * hidden * rstd
        tl.atomic_add(grad_weight_ptr + cols, dweight, mask=mask)


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


_MAX_BLOCK_SIZE = 65536


def _num_warps(block_size: int) -> int:
    num_warps = 4
    if block_size >= 2048:
        num_warps = 8
    if block_size >= 8192:
        num_warps = 16
    return num_warps


if _TRITON_AVAILABLE:

    class _FusedAddRMSNormFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x2d, residual2d, weight, eps, block_size, n_rows):
            out = torch.empty_like(x2d)
            new_residual = torch.empty_like(x2d)
            rstd = torch.empty(n_rows, dtype=torch.float32, device=x2d.device)
            num_warps = _num_warps(block_size)

            _fused_add_rmsnorm_fwd_kernel[(n_rows,)](
                x2d, residual2d, weight, out, new_residual, rstd,
                x2d.shape[-1],
                x2d.stride(0), residual2d.stride(0), out.stride(0), new_residual.stride(0),
                eps,
                BLOCK_SIZE=block_size, num_warps=num_warps,
            )
            ctx.save_for_backward(new_residual, weight, rstd)
            ctx.block_size = block_size
            ctx.num_warps = num_warps
            return out, new_residual

        @staticmethod
        def backward(ctx, grad_out, grad_new_residual):
            new_residual, weight, rstd = ctx.saved_tensors
            n_rows, n_cols = new_residual.shape
            if grad_out is None:
                grad_out = torch.zeros_like(new_residual)

            grad_x = torch.empty_like(new_residual)
            grad_residual = torch.empty_like(new_residual)
            grad_weight_fp32 = torch.zeros(n_cols, dtype=torch.float32, device=weight.device)
            has_residual_grad = grad_new_residual is not None

            _fused_add_rmsnorm_bwd_kernel[(n_rows,)](
                grad_out, grad_new_residual, new_residual, weight, rstd,
                grad_x, grad_residual, grad_weight_fp32,
                n_cols,
                grad_out.stride(0),
                grad_new_residual.stride(0) if has_residual_grad else 0,
                new_residual.stride(0),
                grad_x.stride(0), grad_residual.stride(0),
                HAS_RESIDUAL_GRAD=has_residual_grad,
                BLOCK_SIZE=ctx.block_size, num_warps=ctx.num_warps,
            )
            grad_weight = grad_weight_fp32.to(weight.dtype)
            return grad_x, grad_residual, grad_weight, None, None, None


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """residual = residual + x; return (rms_norm(residual) * weight, residual)
    -- the returned second element is the new residual to carry into the
    next sublayer, matching the standard pre-norm decoder-layer pattern:

        attn_out = self_attn(x)
        x, residual = fused_add_rms_norm(attn_out, residual, ln1_weight, eps)
        ffn_out = ffn(x)
        x, residual = fused_add_rms_norm(ffn_out, residual, ln2_weight, eps)

    x/residual: any shape ending in hidden_dim (flattened internally to
    (rows, hidden_dim) and reshaped back). weight: (hidden_dim,).

    Has a real backward pass (see module docstring) -- gradients flow
    correctly through x, residual, and weight, including when the
    returned `residual` is itself used downstream (e.g. by the next
    layer's own fused_add_rms_norm call)."""
    if not available():
        raise RuntimeError("fused_norm_ops.fused_add_rms_norm: triton is not importable")
    assert x.shape == residual.shape, "x and residual must have the same shape"
    assert x.shape[-1] == weight.shape[0], "last dim of x must match weight's length"
    assert x.dtype in _DTYPES, f"unsupported dtype {x.dtype}"
    assert residual.dtype == x.dtype, "x and residual must share a dtype"
    assert x.is_cuda and residual.is_cuda and weight.is_cuda, "fused_add_rms_norm requires CUDA/ROCm tensors"

    orig_shape = x.shape
    hidden_dim = orig_shape[-1]
    x2d = x.reshape(-1, hidden_dim).contiguous()
    residual2d = residual.reshape(-1, hidden_dim).contiguous()
    weight = weight.contiguous()
    n_rows = x2d.shape[0]

    block_size = _next_pow2(hidden_dim)
    if block_size > _MAX_BLOCK_SIZE:
        raise RuntimeError(
            f"fused_add_rms_norm: hidden_dim={hidden_dim} needs a {block_size}-wide "
            f"block, over this single-block implementation's {_MAX_BLOCK_SIZE} limit"
        )

    out, new_residual = _FusedAddRMSNormFunction.apply(x2d, residual2d, weight, eps, block_size, n_rows)
    return out.view(orig_shape), new_residual.view(orig_shape)
