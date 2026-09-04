"""Fused linear + cross-entropy loss, with a REAL backward pass -- adapted
from LinkedIn's Liger Kernel (see
amd_tuned_torch/_vendor/liger_fused_ce/NOTICE.md for exactly what was
kept/dropped from upstream and why). This is the first training-capable
(has-a-backward) kernel in this package: every other Triton kernel added
so far (fused_norm_ops.fused_add_rms_norm, rope_ops.rotary_embedding) is
explicitly forward-only.

THE MEMORY PROBLEM THIS SOLVES: for a causal LM's final layer, computing
loss the ordinary way materializes a (batch*seq, vocab) logits tensor --
at typical LLM vocab sizes (32k-150k+) this is often the single largest
activation tensor in the whole forward pass, and autograd keeps it alive
for backward on top of that. Since cross-entropy's gradient with respect
to the *logits* has a closed form (softmax(x) - one_hot(target), scaled),
that gradient can be computed immediately in the forward pass, and this
package's linear-layer connection to a *separately* recoverable
grad_input/grad_weight/grad_bias means the full logits tensor never has
to be kept around for backward at all -- see fused_linear_cross_entropy()
below. The forward pass instead processes tokens in chunks small enough
that only one chunk's logits are ever live at a time.

Not a monkeypatch target (there's no F.cross_entropy-taking-a-weight-
matrix-directly stock op to intercept -- F.cross_entropy takes
already-computed logits, this takes hidden_states + the final linear
layer's weight instead), so call this directly from your training loop's
loss computation, replacing something like:

    logits = hidden_states @ lm_head.weight.T
    loss = F.cross_entropy(logits, targets)

with:

    loss = amd_tuned_torch.fused_linear_cross_entropy(
        hidden_states, lm_head.weight, targets)

available() gates on `triton` importability, same reasoning as
fused_norm_ops.py/rope_ops.py: triton is not a declared dependency of
this package anywhere.

UNVALIDATED ON GFX1100 specifically: Liger-Kernel has official AMD
support (dedicated MI300/CDNA3 CI) using this exact Triton kernel
unchanged, but MI300 is not RX 7900 XTX -- there is no known
architectural reason this wouldn't work identically on RDNA3 (no
fp8/tensor-core dependency, plain elementwise + reduction Triton), but
"no known reason it would fail" is not "verified here." Before relying on
this for real training:
  1. Compare its loss AND gradients (grad_input, grad_weight, grad_bias)
     against plain `F.cross_entropy(hidden @ weight.T + bias, target)` +
     `.backward()` numerically, for your actual vocab size / hidden dim /
     dtype / batch shape.
  2. Benchmark peak memory and wall-clock against that same unfused path
     -- the whole point of this kernel is a memory win at large vocab
     sizes; confirm it actually materializes for your shapes before
     switching a real training run over to it.
"""
from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def available() -> bool:
    return _TRITON_AVAILABLE


def _is_hip() -> bool:
    return torch.version.hip is not None


# Historical note carried from upstream: the hard Triton limit on a single
# tensor's element count is 1048576, but 65536 measures faster in practice
# (less register spilling) -- this is the same MAX_FUSED_SIZE constant
# Liger-Kernel's LayerNorm/CrossEntropy kernels both use.
_MAX_FUSED_SIZE = 65536 // 2

# Chunk-sizing budget for the token-chunk loop in fused_linear_cross_entropy:
# widens the transient (chunk_size x V) logits tensor's memory budget to
# _CHUNK_MEM_CONST x BT x H instead of a bare BT x H floor, trading a bit
# more transient memory for far fewer, larger kernel launches at big vocab
# sizes. Inherited from upstream unmeasured on gfx1100 -- see
# fused_linear_cross_entropy's docstring point 2 for what to check before
# assuming this constant is well-tuned for this hardware too.
_CHUNK_MEM_CONST = 16


if _TRITON_AVAILABLE:

    LOG2_E: tl.constexpr = 1.4426950408889634

    @triton.jit
    def _element_mul_kernel(X_ptr, X_stride, grad_output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
        """In-place X *= grad_output (a 0-dim tensor) -- used to rescale
        the gradients this module already computed in forward by whatever
        upstream scalar grad_output backward actually receives."""
        program_id = tl.program_id(0).to(tl.int64)
        X_ptr += program_id * X_stride
        grad_output = tl.load(grad_output_ptr)
        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            block = tl.load(X_ptr + offsets, mask=mask)
            tl.store(X_ptr + offsets, block * grad_output, mask=mask)

    @triton.jit
    def _fused_ce_kernel(
        X_ptr, X_stride, Y_ptr, Y_stride, weight_ptr, loss_ptr, loss_stride,
        n_cols, n_non_ignore, sum_non_ignore_weight, weight_sum, ignore_index,
        label_smoothing: tl.constexpr,
        reduction: tl.constexpr,
        softcap,
        HAS_WEIGHT: tl.constexpr,
        HAS_SOFTCAPPING: tl.constexpr,
        HAS_GRADIENTS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """One program per row (token): computes that row's cross-entropy
        loss, and -- if HAS_GRADIENTS -- overwrites X's own row in place
        with d(loss)/d(X), the "gradient trick" that lets the caller skip
        ever allocating a second (chunk_size, n_cols)-shaped buffer for
        it. Two-pass online softmax (max+sum, then the actual softmax),
        same numerically-stable algorithm any cross-entropy kernel needs;
        see https://arxiv.org/pdf/1805.02867 Algorithm 3."""
        row = tl.program_id(0).to(tl.int64)
        Y_ptr += row * Y_stride
        y = tl.load(Y_ptr)
        X_ptr += row * X_stride

        if y == ignore_index:
            for i in range(0, n_cols, BLOCK_SIZE):
                offsets = i + tl.arange(0, BLOCK_SIZE)
                tl.store(X_ptr + offsets, 0.0, mask=offsets < n_cols)
            return

        loss_ptr += row * loss_stride

        if HAS_SOFTCAPPING:
            softcap = softcap.to(tl.float32)
        if HAS_WEIGHT:
            sum_non_ignore_weight = sum_non_ignore_weight.to(tl.float32)
            weight_sum = weight_sum.to(tl.float32)
            weight_y = tl.load(weight_ptr + y).to(tl.float32)

        m = float("-inf")
        d = 0.0
        ori_X_y = tl.load(X_ptr + y).to(tl.float32)
        if HAS_SOFTCAPPING:
            ori_X_y = softcap * tl.math.tanh(ori_X_y / softcap)

        scaled_x_sum = 0.0
        eps = label_smoothing / n_cols

        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            block = tl.load(X_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)
            if HAS_SOFTCAPPING:
                block = softcap * tl.math.tanh(block / softcap)
            block_max = tl.max(block)
            if label_smoothing > 0:
                if HAS_WEIGHT:
                    weight_block = tl.load(weight_ptr + offsets, mask=mask)
                    scaled_x_sum += tl.sum(tl.where(mask, -eps * block * weight_block, 0.0))
                else:
                    scaled_x_sum += tl.sum(tl.where(mask, -eps * block, 0.0))
            m_new = tl.maximum(m, block_max)
            d = d * tl.exp2((m - m_new) * LOG2_E) + tl.sum(tl.exp2((block - m_new) * LOG2_E))
            m = m_new

        lse = m + tl.log(d)

        if HAS_GRADIENTS:
            for i in range(0, n_cols, BLOCK_SIZE):
                offsets = i + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_cols
                block = tl.load(X_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)
                if HAS_SOFTCAPPING:
                    intermediate = tl.math.tanh(block / softcap)
                    block = softcap * intermediate

                if not HAS_WEIGHT:
                    softmax_x = tl.exp2((block - m) * LOG2_E) / d
                    grad = softmax_x - eps
                    if reduction == "mean":
                        grad = grad / n_non_ignore
                else:
                    weight_block = tl.load(weight_ptr + offsets, mask=mask)
                    softmax_x = tl.exp2((block - m) * LOG2_E) / d
                    dloss_ori = (1 - label_smoothing) * softmax_x * weight_y
                    dloss_smooth = eps * (-weight_block + softmax_x * weight_sum)
                    if reduction == "mean":
                        dloss_ori = dloss_ori / sum_non_ignore_weight
                        dloss_smooth = dloss_smooth / sum_non_ignore_weight
                    grad = dloss_ori + dloss_smooth

                if HAS_SOFTCAPPING:
                    grad = grad * (1 - intermediate * intermediate)
                tl.store(X_ptr + offsets, grad, mask=mask)

            # True-class correction: dx_y needs an extra -(1 - label_smoothing)
            # term the loop above doesn't apply. Recomputed once in fp32 from
            # the ORIGINAL logit (ori_X_y) rather than read back the value the
            # loop just stored, since (softmax(x_y) - 1) cancels catastrophically
            # as softmax(x_y) -> 1 -- rounding to the output dtype before this
            # correction would lose most of the significant bits in fp16/bf16.
            tl.debug_barrier()
            softmax_x_y = tl.exp2((ori_X_y - m) * LOG2_E) / d
            if not HAS_WEIGHT:
                dx_y = softmax_x_y - eps - (1 - label_smoothing)
                if reduction == "mean":
                    dx_y = dx_y / n_non_ignore
            else:
                dloss_ori_y = ((1 - label_smoothing) * softmax_x_y - (1 - label_smoothing)) * weight_y
                dloss_smooth_y = eps * (-weight_y + softmax_x_y * weight_sum)
                if reduction == "mean":
                    dloss_ori_y = dloss_ori_y / sum_non_ignore_weight
                    dloss_smooth_y = dloss_smooth_y / sum_non_ignore_weight
                dx_y = dloss_ori_y + dloss_smooth_y
            if HAS_SOFTCAPPING:
                t_y = ori_X_y / softcap
                dx_y = dx_y * (1 - t_y * t_y)
            tl.store(X_ptr + y, dx_y)

        tl.debug_barrier()

        loss = lse - ori_X_y
        if HAS_WEIGHT:
            loss = weight_y * loss
        if label_smoothing > 0:
            if HAS_WEIGHT:
                smooth_loss = scaled_x_sum + eps * lse * weight_sum
            else:
                smooth_loss = scaled_x_sum + label_smoothing * lse
            loss = loss * (1 - label_smoothing) + smooth_loss
        if reduction == "mean":
            loss = loss / (sum_non_ignore_weight if HAS_WEIGHT else n_non_ignore)
        tl.store(loss_ptr, loss)


def _chunked_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: Optional[torch.Tensor],
    ce_weight: Optional[torch.Tensor],
    ignore_index: int,
    label_smoothing: float,
    reduction: str,
    softcap: Optional[float],
    input_requires_grad: bool,
    weight_requires_grad: bool,
):
    device = input.device
    BT, H = input.shape
    V = weight.shape[0]

    inc_factor = triton.cdiv(V, _CHUNK_MEM_CONST * H)
    chunk_size = min(triton.next_power_of_2(triton.cdiv(BT, inc_factor)), BT)
    num_chunks = triton.cdiv(BT, chunk_size)

    grad_input = torch.zeros_like(input) if input_requires_grad else None
    grad_weight = torch.zeros_like(weight, dtype=torch.float32) if weight_requires_grad else None
    grad_bias = torch.zeros_like(bias, dtype=torch.float32) if bias is not None else None
    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)

    target_mask = target != ignore_index
    n_non_ignore = target_mask.sum().item()
    sum_non_ignore_weight = float(n_non_ignore)
    weight_sum = 0.0
    if ce_weight is not None:
        sum_non_ignore_weight = torch.gather(ce_weight, 0, target.masked_select(target_mask)).sum().item()
        weight_sum = ce_weight.sum().item()
        if ce_weight.stride(-1) != 1:
            ce_weight = ce_weight.contiguous()

    for chunk_id in range(num_chunks):
        start, end = chunk_id * chunk_size, min((chunk_id + 1) * chunk_size, BT)
        input_chunk = input[start:end]
        logits_chunk = input_chunk @ weight.t()
        if bias is not None:
            logits_chunk = logits_chunk + bias
        logits_chunk = logits_chunk.contiguous()
        target_chunk = target[start:end].contiguous()

        block_size = min(_MAX_FUSED_SIZE, triton.next_power_of_2(V))
        loss_slice = loss_1d[start:end]
        _fused_ce_kernel[(end - start,)](
            X_ptr=logits_chunk, X_stride=logits_chunk.stride(-2),
            Y_ptr=target_chunk, Y_stride=target_chunk.stride(-1),
            weight_ptr=ce_weight, loss_ptr=loss_slice, loss_stride=loss_slice.stride(-1),
            n_cols=V, n_non_ignore=n_non_ignore, sum_non_ignore_weight=sum_non_ignore_weight,
            weight_sum=weight_sum, ignore_index=ignore_index,
            label_smoothing=label_smoothing, reduction=reduction, softcap=softcap,
            HAS_WEIGHT=ce_weight is not None, HAS_SOFTCAPPING=softcap is not None,
            HAS_GRADIENTS=input_requires_grad,
            BLOCK_SIZE=block_size, num_warps=16 if _is_hip() else 32,
        )
        loss_1d[start:end] = loss_slice
        grad_logits_chunk = logits_chunk  # the kernel overwrote this in place

        if input_requires_grad:
            grad_input[start:end] = grad_logits_chunk @ weight
        if grad_weight is not None:
            # `@` (torch.matmul), not torch.mm: amd_tuned_torch.__init__._patched_matmul
            # intercepts torch.matmul/@ to contest hipBLASLt/CK/aiter against
            # stock, but does NOT patch torch.mm -- using torch.mm here would
            # silently opt this GEMM out of that tier system while the
            # logits_chunk/grad_input GEMMs above (already written with `@`)
            # benefit from it automatically whenever amd_tuned_torch.enable()
            # is active.
            grad_weight += (grad_logits_chunk.t() @ input_chunk).float()
        if grad_bias is not None:
            grad_bias += grad_logits_chunk.sum(dim=0).float()

    loss = torch.sum(loss_1d)
    if grad_weight is not None:
        grad_weight = grad_weight.to(weight.dtype)
    if grad_bias is not None:
        grad_bias = grad_bias.to(bias.dtype)
    return loss, grad_input, grad_weight, grad_bias


def _rescale_grads(grad_output, grad_input, grad_weight, grad_bias):
    if torch.equal(grad_output, torch.ones_like(grad_output)):
        return grad_input, grad_weight, grad_bias  # loss is the graph's leaf -> grad_output is 1.0, nothing to scale
    block_size = min(_MAX_FUSED_SIZE, triton.next_power_of_2(grad_input.shape[-1]))
    num_warps = 16 if _is_hip() else 32
    _element_mul_kernel[(grad_input.shape[0],)](
        grad_input, grad_input.stride(-2), grad_output, grad_input.shape[-1],
        BLOCK_SIZE=block_size, num_warps=num_warps,
    )
    if grad_weight is not None:
        _element_mul_kernel[(grad_weight.shape[0],)](
            grad_weight, grad_weight.stride(-2), grad_output, grad_weight.shape[-1],
            BLOCK_SIZE=block_size, num_warps=num_warps,
        )
    if grad_bias is not None:
        _element_mul_kernel[(grad_bias.shape[0],)](
            grad_bias, grad_bias.stride(-1), grad_output, 1,
            BLOCK_SIZE=block_size, num_warps=num_warps,
        )
    return grad_input, grad_weight, grad_bias


class _FusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, target, bias, ce_weight, ignore_index, label_smoothing, reduction, softcap):
        loss, grad_input, grad_weight, grad_bias = _chunked_forward(
            input, weight, target, bias, ce_weight, ignore_index, label_smoothing, reduction, softcap,
            input_requires_grad=input.requires_grad,
            weight_requires_grad=weight.requires_grad,
        )
        ctx.save_for_backward(grad_input, grad_weight, grad_bias)
        ctx.has_bias = bias is not None
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight, grad_bias = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = _rescale_grads(grad_output, grad_input, grad_weight, grad_bias)
        return grad_input, grad_weight, None, grad_bias, None, None, None, None, None


def fused_linear_cross_entropy(
    input: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    ce_weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    softcap: Optional[float] = None,
) -> torch.Tensor:
    """Fuses `logits = input @ weight.T (+ bias); loss = F.cross_entropy(logits, target, ...)`
    into one call that never materializes the full (BT, V) logits tensor
    for backward -- see module docstring for the memory-saving mechanism
    and why this needs its own autograd.Function rather than being
    expressible as ordinary ops.

    input: (BT, H) -- already-flattened (batch*seq, hidden_dim); flatten
        a (batch, seq, hidden_dim) tensor with .view(-1, hidden_dim) first.
    weight: (V, H) -- the LM head's weight (F.linear layout, not pre-transposed).
    target: (BT,) int64, values in [0, V) or ignore_index.
    bias: (V,), optional.
    ce_weight: (V,) per-class rescaling weight, optional (matches
        F.cross_entropy's `weight` argument).
    ignore_index: target value to skip (no loss/gradient contribution).
    label_smoothing: as in F.cross_entropy.
    reduction: "mean" or "sum" ONLY -- "none" is not supported by this
        port (see amd_tuned_torch/_vendor/liger_fused_ce/NOTICE.md for why).
    softcap: if set, logits are passed through `softcap * tanh(x / softcap)`
        before the loss (Gemma2-style logit softcapping) -- both the loss
        and its gradient account for this transform.

    Returns a scalar loss tensor with a working backward -- input/weight/
    bias all receive correct gradients through ordinary `.backward()`."""
    if not available():
        raise RuntimeError("fused_ce_ops.fused_linear_cross_entropy: triton is not importable")
    assert input.dim() == 2, "input must be 2D (BT, H) -- flatten batch/seq dims first"
    assert weight.dim() == 2 and weight.shape[1] == input.shape[1], "weight must be (V, H) matching input's H"
    assert target.dim() == 1 and target.shape[0] == input.shape[0], "target must be (BT,) matching input's BT"
    assert reduction in ("mean", "sum"), "fused_linear_cross_entropy only supports reduction='mean' or 'sum'"
    if bias is not None:
        assert bias.shape == (weight.shape[0],), "bias must be (V,)"
    if ce_weight is not None:
        assert ce_weight.shape == (weight.shape[0],), "ce_weight must be (V,)"
        assert torch.is_floating_point(ce_weight), "ce_weight must be floating point"

    return _FusedLinearCrossEntropyFunction.apply(
        input, weight, target, bias, ce_weight, ignore_index, label_smoothing, reduction, softcap
    )
