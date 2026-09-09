"""Tests for amd_tuned_torch.depthwise_conv1d_ops's FlashAttention-3-inspired
wrapper practices: maybe_contiguous (avoid an unconditional `.contiguous()`
copy) and the torch.library.custom_op + register_fake registration for
depthwise_conv1d_forward/backward (so torch.compile/Dynamo/FakeTensorMode
gets correct shape propagation without ever invoking the real, unbuilt
native extension). Does not cover the real native kernel call itself (no
ROCm toolchain in this environment) -- see depthwise_conv1d_ops.py's module
docstring for exactly what was ported from
source/flash-attention/hopper/flash_attn_interface.py and why.

Run with:

    python -m pytest source/cmp_ext_turing/tests
"""
from __future__ import annotations

import pytest
import torch

import amd_tuned_torch.depthwise_conv1d_ops as depthwise_conv1d_ops


class TestMaybeContiguous:
    def test_none_passes_through(self):
        assert depthwise_conv1d_ops.maybe_contiguous(None) is None

    def test_already_last_dim_contiguous_returned_as_is(self):
        x = torch.randn(2, 3, 4)
        assert x.stride(-1) == 1
        out = depthwise_conv1d_ops.maybe_contiguous(x)
        assert out is x

    def test_non_last_dim_contiguous_gets_copied(self):
        # A transpose swaps strides so the new last dim is no longer
        # stride-1 -- exactly the case maybe_contiguous exists to fix up,
        # and plain .is_contiguous() would also have flagged (this case
        # doesn't distinguish the two checks, see the next test for that).
        x = torch.randn(2, 4, 3).transpose(1, 2)
        assert x.stride(-1) != 1
        out = depthwise_conv1d_ops.maybe_contiguous(x)
        assert out is not x
        assert out.stride(-1) == 1
        assert torch.equal(out, x)

    def test_last_dim_contiguous_but_not_is_contiguous_skips_the_copy(self):
        """The actual point of using stride(-1) instead of .is_contiguous()
        (matching flash-attention's own maybe_contiguous exactly): a tensor
        can have a non-default overall stride layout while still being
        contiguous in its last dimension, e.g. a batch slice out of a
        larger contiguous tensor. .is_contiguous() would be False here (the
        strides aren't the tensor's own default C-contiguous set), but the
        kernel only cares that the last dim has stride 1, so no copy should
        happen."""
        base = torch.randn(2, 5, 4)
        x = base[:, :3, :]  # shape (2, 3, 4); stride(-1) == 1, but not
        # .is_contiguous() since dim 1's stride is still 4 (the parent's),
        # not 3 (what a fresh (2,3,4) tensor would have).
        assert x.stride(-1) == 1
        assert x.is_contiguous() is False
        out = depthwise_conv1d_ops.maybe_contiguous(x)
        assert out is x


class TestOutLen:
    def test_matches_conv1d_cuda_formula(self):
        # l_out = l + 2*padding - k + 1, verified directly against
        # conv1d_cuda_bhl/conv1d_cuda_blh's own formula (conv1d_bhl.cu /
        # conv1d_blh.cu) -- stride/dilation are always 1 for this kernel.
        assert depthwise_conv1d_ops._out_len(length=10, padding=1, width=3) == 10
        assert depthwise_conv1d_ops._out_len(length=10, padding=0, width=3) == 8
        assert depthwise_conv1d_ops._out_len(length=100, padding=5, width=11) == 100


@pytest.mark.skipif(
    not depthwise_conv1d_ops._HAS_CUSTOM_OP,
    reason="torch.library.custom_op requires PyTorch >= 2.4",
)
class TestCustomOpFakeShapes:
    """Exercises the registered fake (meta) implementations directly under
    FakeTensorMode -- this dispatches straight to register_fake, never
    touching the real (unbuilt, in this environment) _native_depthwise_conv1d
    extension, the same way Dynamo/torch.compile tracing would."""

    def test_forward_fake_bhl_shape(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x = torch.randn(2, 4, 10)
            w = torch.randn(4, 3)
            b = torch.randn(4)
            out = torch.ops.amd_tuned_torch.depthwise_conv1d_forward(x, w, b, 1, True)

        assert out.shape == (2, 4, 10)
        assert out.dtype == x.dtype

    def test_forward_fake_blh_shape(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x = torch.randn(2, 10, 4)
            w = torch.randn(3, 4)
            b = torch.randn(4)
            out = torch.ops.amd_tuned_torch.depthwise_conv1d_forward(x, w, b, 0, False)

        assert out.shape == (2, 8, 4)
        assert out.dtype == x.dtype

    def test_backward_fake_shapes_and_dtypes(self):
        """Regression coverage for the mixed-dtype subtlety noted in the
        fake's own comment: dbias takes dout's dtype, NOT bias's -- the
        real kernel (conv1d_bwd_cuda_bhl.cu: `dbias = dout.sum(-1).sum(0)`)
        never casts it to bias's own dtype, unlike dweight which is
        explicitly `.to(weight.type())`'d."""
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x = torch.randn(2, 4, 10, dtype=torch.float32)
            w = torch.randn(4, 3, dtype=torch.float16)
            b = torch.randn(4, dtype=torch.float16)
            dout = torch.randn(2, 4, 10, dtype=torch.float32)
            du, dweight, dbias = torch.ops.amd_tuned_torch.depthwise_conv1d_backward(
                dout, x, w, b, 1, True)

        assert du.shape == x.shape and du.dtype == x.dtype
        assert dweight.shape == w.shape and dweight.dtype == w.dtype
        assert dbias.shape == b.shape
        assert dbias.dtype == dout.dtype
