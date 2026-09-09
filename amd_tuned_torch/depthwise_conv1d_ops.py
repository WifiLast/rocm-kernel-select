"""General depthwise Conv1d for amd_tuned_torch, backed by a hand-written
HIP kernel vendored from FlashFFTConv (see
amd_tuned_torch/_vendor/flashfftconv_depthwise_conv1d/NOTICE.md for exact
provenance and why it's vendored independently of
third_party/FlashFFTConv/ -- that package's own actual FFT-conv kernels are
tensor-core-only and needed a hand rewrite against rocWMMA; this kernel has
no WMMA/tensor-core usage at all and builds via ordinary hipify).

WHAT THIS COVERS THAT causal_conv1d/aiter/CK DON'T. This package already
has a depthwise conv1d fast path -- amd_tuned_torch.miopen_fallback's
CAUSAL_CONV1D FAST PATH, backed by source/cmp_ext_turing/src/causal-conv1d-amd
-- but it only matches the narrow *causal* idiom (kernel width 2-4,
padding == width - 1). This module's kernel is the general case: any odd
kernel width, any symmetric padding value, dilation/stride fixed at 1
(same restriction upstream's own conv1d_fwd enforces -- see conv1d.h's
`TORCH_CHECK(k % 2 == 1, ...)`), mixed input/weight dtypes (fp16/bf16/fp32
in any combination), with a real backward pass. Its output matches stock
`F.conv1d(input, weight, bias, padding=padding, groups=channels)` exactly
(same formula, no truncation the way causal_conv1d_fn's output has), so
unlike causal_conv1d -- wired in via a structural pattern match because its
truncated output isn't a drop-in replacement -- this kernel can be entered
directly into kernel_select's measure-and-cache contest against stock
conv1d, the same way fftconv_ops.fftconv1d_candidate already is. See
miopen_fallback.py's "DEPTHWISE CONV1D FAST PATH" section for the wiring
and the eligibility check (groups == in_channels == out_channels, odd
kernel width, stride == dilation == 1).

FLASHATTENTION-3-INSPIRED WRAPPER PRACTICES. FA3's core kernel work itself
(warp-specialized producer/consumer pipelining, TMA, FP8 with incoherent
processing) is Hopper-instruction-level and doesn't transfer to a plain
depthwise-conv HIP kernel -- there's no softmax-like non-matmul bottleneck
here to hide behind async copies. What DOES transfer are two of its
Python-wrapper-level engineering practices (see
source/flash-attention/hopper/flash_attn_interface.py, not vendored, just
the pattern):

1. `maybe_contiguous(x)` -- calls `.contiguous()` only when
   `x.stride(-1) != 1`, instead of unconditionally. The native kernel only
   needs contiguity in the last (fastest-varying) dimension; forcing a full
   default-stride copy on a tensor that already satisfies that (e.g. one
   sliced/transposed upstream but still contiguous in its last dim) is a
   wasted copy. `DepthwiseConv1dFn.backward` used to call
   `dout.contiguous()` unconditionally -- now uses this instead, same as
   every tensor argument FA3's `_flash_attn_forward`/`_flash_attn_backward`
   massage this way before reaching the native call.

2. Wrapping the raw native forward/backward calls as
   `torch.library.custom_op`s with `register_fake` meta implementations --
   the same pattern amd_tuned_torch/compile_ops.py already uses for
   linear_fp16/bmm_fp16/conv2d_fp16/group_norm/conv2d_native/conv3d_native
   (itself ported from NVIDIA TransformerEngine's custom_ops.py; see that
   module's docstring), and the exact shape FA3 uses for
   `flash_attn_3::_flash_attn_forward`/`_flash_attn_backward`: register the
   raw kernel calls as opaque ops with shape/dtype-only fake stand-ins, and
   keep an ordinary `torch.autograd.Function` on top of them for the actual
   gradient wiring (FA3 does this too -- `FlashAttnFunc.forward`/`backward`
   call `torch.ops.flash_attn_3._flash_attn_forward`/`_backward`, not
   `register_autograd`). Registered HERE, not in compile_ops.py, because
   every op already there is grad-unsafe (no backward pass at all, always
   called under a `_grad_safe()` guard) -- this op has a REAL backward
   pass, a materially different category worth keeping self-contained
   rather than blending into that file's "no backward pass" framing.
   Without this, Dynamo has no registered op to place as a graph node for
   either the forward or backward native call and falls back to inlining
   through a pybind11 extension boundary it can't trace past -- a graph
   break under plain torch.compile(model), or a hard
   torch._dynamo.exc.Unsupported under fullgraph=True. Falls back to a
   plain function call (today's behavior, numerically identical, no
   graph-node benefit) on PyTorch builds without
   torch.library.custom_op (added in 2.4), same fallback convention
   compile_ops.py uses.

WHAT'S EXPOSED. `conv1d_forward`/`conv1d_backward` (thin wrappers, now
torch.compile-safe, over the native `_native_depthwise_conv1d` extension),
`DepthwiseConv1dFn` (the torch.autograd.Function pairing them), and
`depthwise_conv1d_candidate` (the kernel_select-contest entry point, taking
the same (input, weight, bias, stride, padding, dilation, groups) shape
every other conv1d candidate in miopen_fallback.py uses -- stride/dilation/
groups are accepted for that uniform calling convention but not otherwise
used here, since eligibility already pins them to 1/1/channels before this
is ever called).
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

try:
    from . import _native_depthwise_conv1d as _C

    _DEPTHWISE_CONV1D_AVAILABLE = True
except ImportError:
    _C = None
    _DEPTHWISE_CONV1D_AVAILABLE = False


def available() -> bool:
    """True if the `_native_depthwise_conv1d` extension (built from
    amd_tuned_torch/_vendor/flashfftconv_depthwise_conv1d/) is importable."""
    return _DEPTHWISE_CONV1D_AVAILABLE


def maybe_contiguous(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Same one-liner as flash-attention/hopper/flash_attn_interface.py's
    own maybe_contiguous: only pay for a `.contiguous()` copy when the last
    dimension actually isn't contiguous, rather than unconditionally
    forcing the tensor's full default stride layout."""
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _out_len(length: int, padding: int, width: int) -> int:
    """Matches conv1d_cuda_bhl/conv1d_cuda_blh's own
    `l_out = l + 2 * padding - k + 1` exactly (stride/dilation fixed at 1
    -- see conv1d.h) -- used by the forward fake below so FakeTensor/
    torch.compile shape propagation agrees with the real kernel without
    ever running it."""
    return length + 2 * padding - width + 1


_HAS_CUSTOM_OP = hasattr(torch.library, "custom_op")


def _already_registered(name: str) -> bool:
    return hasattr(torch.ops, "amd_tuned_torch") and hasattr(torch.ops.amd_tuned_torch, name)


if _HAS_CUSTOM_OP and not _already_registered("depthwise_conv1d_forward"):

    @torch.library.custom_op("amd_tuned_torch::depthwise_conv1d_forward", mutates_args=())
    def _depthwise_conv1d_forward_op(
        input_: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, padding: int, is_bhl: bool,
    ) -> torch.Tensor:
        return _C.conv1d_forward(input_, weight, bias, padding, is_bhl)

    @_depthwise_conv1d_forward_op.register_fake
    def _depthwise_conv1d_forward_fake(
        input_: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, padding: int, is_bhl: bool,
    ) -> torch.Tensor:
        del bias
        if is_bhl:
            b, d, length = input_.shape
            width = weight.shape[1]
            return input_.new_empty(b, d, _out_len(length, padding, width))
        b, length, d = input_.shape
        width = weight.shape[0]
        return input_.new_empty(b, _out_len(length, padding, width), d)

    @torch.library.custom_op("amd_tuned_torch::depthwise_conv1d_backward", mutates_args=())
    def _depthwise_conv1d_backward_op(
        dout: torch.Tensor, input_: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
        padding: int, is_bhl: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        du, dweight, dbias = _C.conv1d_backward(dout, input_, weight, bias, padding, is_bhl)
        return du, dweight, dbias

    @_depthwise_conv1d_backward_op.register_fake
    def _depthwise_conv1d_backward_fake(
        dout: torch.Tensor, input_: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
        padding: int, is_bhl: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del padding, is_bhl
        # du/dweight match input/weight's own shape AND dtype exactly (the
        # real kernel's du is torch::empty(..., u.options()), dweight is
        # explicitly .to(weight.type())'d); dbias matches bias's SHAPE but
        # dout's dtype (dbias = dout.sum(...), never cast to bias's own
        # dtype by the real kernel -- see conv1d_bwd_cuda_{bhl,blh}.cu) --
        # a real, if easy to miss, mixed-dtype subtlety upstream's own
        # kernel has whenever bias's dtype differs from input's.
        du = input_.new_empty(input_.shape)
        dweight = weight.new_empty(weight.shape)
        dbias = dout.new_empty(bias.shape)
        return du, dweight, dbias

    _conv1d_forward_impl = torch.ops.amd_tuned_torch.depthwise_conv1d_forward
    _conv1d_backward_impl = torch.ops.amd_tuned_torch.depthwise_conv1d_backward

elif _already_registered("depthwise_conv1d_forward"):
    # Already registered by an earlier import of this module in this
    # process -- stay idempotent rather than raising on a stray reload,
    # same guard compile_ops.py uses.
    _conv1d_forward_impl = torch.ops.amd_tuned_torch.depthwise_conv1d_forward
    _conv1d_backward_impl = torch.ops.amd_tuned_torch.depthwise_conv1d_backward

else:
    # PyTorch build predates torch.library.custom_op (added in 2.4) -- same
    # numerics, just no torch.compile graph-node benefit.
    def _conv1d_forward_impl(input_, weight, bias, padding, is_bhl):
        return _C.conv1d_forward(input_, weight, bias, padding, is_bhl)

    def _conv1d_backward_impl(dout, input_, weight, bias, padding, is_bhl):
        return _C.conv1d_backward(dout, input_, weight, bias, padding, is_bhl)


def conv1d_forward(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, padding: int, is_bhl: bool = True,
) -> torch.Tensor:
    """`weight` is the squeezed depthwise kernel, shape [channels, width]
    (is_bhl -- `input` layout (B, C, L), the standard torch.nn.Conv1d
    layout) or [width, channels] (not is_bhl, layout (B, L, C)). `bias` is
    required (not Optional) by the native kernel -- see
    depthwise_conv1d_candidate for the None-bias convenience wrapper."""
    input, weight, bias = maybe_contiguous(input), maybe_contiguous(weight), maybe_contiguous(bias)
    return _conv1d_forward_impl(input, weight, bias, padding, is_bhl)


def conv1d_backward(
    dout: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
    padding: int, is_bhl: bool = True,
):
    dout, input, weight, bias = (
        maybe_contiguous(dout), maybe_contiguous(input), maybe_contiguous(weight), maybe_contiguous(bias))
    return _conv1d_backward_impl(dout, input, weight, bias, padding, is_bhl)


class DepthwiseConv1dFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, padding, is_bhl):
        ctx.padding = padding
        ctx.is_bhl = is_bhl
        ctx.save_for_backward(input, weight, bias)
        return conv1d_forward(input, weight, bias, padding, is_bhl)

    @staticmethod
    def backward(ctx, dout):
        input, weight, bias = ctx.saved_tensors
        du, dweight, dbias = conv1d_backward(dout, input, weight, bias, ctx.padding, ctx.is_bhl)
        return du, dweight, dbias, None, None


def depthwise_conv1d_candidate(
    input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
    stride: Any, padding: Any, dilation: Any, groups: int,
) -> torch.Tensor:
    """kernel_select candidate matching stock F.conv1d's own calling
    convention: `weight` is the un-squeezed [channels, 1, width] shape,
    `bias` may be None. `stride`/`dilation`/`groups` are accepted only to
    match every other conv1d candidate's lambda signature in
    miopen_fallback.py -- the caller's eligibility check has already
    confirmed stride == dilation == 1 and groups == channels before this
    is ever invoked, so they're not re-validated here."""
    channels = weight.shape[0]
    squeezed = weight.squeeze(1)
    if bias is None:
        bias = torch.zeros(channels, device=input.device, dtype=weight.dtype)
    pad = padding[0] if isinstance(padding, (tuple, list)) else padding
    return DepthwiseConv1dFn.apply(input, squeezed, bias, pad, True)
