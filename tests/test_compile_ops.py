"""Tests for amd_tuned_torch's torch.compile / Dynamo integration
(amd_tuned_torch/compile_ops.py) -- torch.library.custom_op registration and
register_fake shape inference for linear_fp16/bmm_fp16/conv2d_fp16/
group_norm/conv2d_native/conv3d_native.

No GPU, aiter, or TransformerEngine required: dispatch-correctness tests
stub aiter_ops's/the native extension's underlying kernels directly (same
style as test_amd_tuned_torch_monkeypatch.py), and the register_fake/torch.compile
tests use plain CPU-friendly stand-ins -- only tracing/dispatch correctness
is under test here, not real-kernel numerics.

Run with:

    python -m pytest source/cmp_ext_turing/tests
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

import amd_tuned_torch.aiter_ops as aiter_ops
import amd_tuned_torch.ck_gemm_ops as ck_gemm_ops
import amd_tuned_torch.compile_ops as compile_ops
import amd_tuned_torch.hipblaslt_ops as hipblaslt_ops

pytestmark = pytest.mark.skipif(
    not compile_ops._HAS_CUSTOM_OP,
    reason="torch.library.custom_op requires PyTorch >= 2.4",
)


class TestRegistration:
    def test_ops_registered_under_amd_tuned_torch_namespace(self):
        assert hasattr(torch.ops, "amd_tuned_torch")
        for name in ("linear_fp16", "bmm_fp16", "conv2d_fp16", "group_norm",
                     "conv2d_native", "conv3d_native",
                     "hipblaslt_linear", "ck_gemm_linear", "hipblaslt_bmm"):
            assert hasattr(torch.ops.amd_tuned_torch, name)

    def test_already_registered_guard_reflects_reality(self):
        assert compile_ops._already_registered("linear_fp16") is True
        assert compile_ops._already_registered("not_a_real_op") is False


class TestRegisterFakeShapes:
    """Directly exercises register_fake under FakeTensorMode -- the real
    kernel body is never invoked here (that's the point of FakeTensorMode),
    only the shape-only stand-in, so no aiter/native mocking is needed."""

    def test_linear_fp16_fake_shape(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x = torch.randn(2, 4)
            w = torch.randn(3, 4)
            out = torch.ops.amd_tuned_torch.linear_fp16(x, w, None)
        assert out.shape == (2, 3)
        assert out.dtype == x.dtype

    def test_linear_fp16_fake_shape_preserves_leading_dims(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x = torch.randn(2, 5, 4)
            w = torch.randn(3, 4)
            out = torch.ops.amd_tuned_torch.linear_fp16(x, w, None)
        assert out.shape == (2, 5, 3)

    def test_bmm_fp16_fake_shape(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x = torch.randn(5, 2, 4)
            y = torch.randn(5, 4, 3)
            out = torch.ops.amd_tuned_torch.bmm_fp16(x, y)
        assert out.shape == (5, 2, 3)

    def test_conv2d_fp16_fake_shape(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x = torch.randn(1, 3, 8, 8)
            w = torch.randn(4, 3, 3, 3)
            out = torch.ops.amd_tuned_torch.conv2d_fp16(x, w, None, [1, 1], [1, 1], [1, 1])
        assert out.shape == (1, 4, 8, 8)

    def test_conv2d_fp16_fake_shape_with_stride(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x = torch.randn(1, 3, 8, 8)
            w = torch.randn(4, 3, 3, 3)
            out = torch.ops.amd_tuned_torch.conv2d_fp16(x, w, None, [2, 2], [1, 1], [1, 1])
        expected = F.conv2d(
            torch.randn(1, 3, 8, 8), torch.randn(4, 3, 3, 3), stride=2, padding=1
        )
        assert out.shape == expected.shape

    def test_group_norm_fake_shape(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x = torch.randn(2, 4, 3, 3)
            out = torch.ops.amd_tuned_torch.group_norm(x, 2, None, None, 1e-5)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_conv2d_native_fake_shape(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x = torch.randn(1, 3, 8, 8)
            w = torch.randn(4, 3, 3, 3)
            out = torch.ops.amd_tuned_torch.conv2d_native(x, w, None, [1, 1], [1, 1], [1, 1])
        assert out.shape == (1, 4, 8, 8)

    def test_conv3d_native_fake_shape(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x = torch.randn(1, 3, 4, 8, 8)
            w = torch.randn(4, 3, 3, 3, 3)
            out = torch.ops.amd_tuned_torch.conv3d_native(
                x, w, None, [1, 1, 1], [1, 1, 1], [1, 1, 1]
            )
        assert out.shape == (1, 4, 4, 8, 8)


class TestOptionalReturnFakeShapes:
    """hipblaslt_linear/ck_gemm_linear/hipblaslt_bmm are the only ops here
    that can legitimately return None -- see compile_ops.py's own docstring
    for why they need an explicit `schema=".. -> Tensor?"` string rather
    than relying on infer_schema, and why their fakes replicate
    hipblaslt_ops.is_linear_eligible/is_bmm_eligible and
    ck_gemm_ops.is_eligible exactly (pulled into their own functions
    specifically for this reuse)."""

    def test_hipblaslt_linear_fake_shape_when_eligible(self, monkeypatch):
        monkeypatch.setattr(hipblaslt_ops, "is_linear_eligible", lambda *a, **k: True)
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x = torch.randn(2, 4)
            w = torch.randn(3, 4)
            out = torch.ops.amd_tuned_torch.hipblaslt_linear(x, w, None, 0)
        assert out.shape == (2, 3)
        assert out.dtype == x.dtype

    def test_hipblaslt_linear_fake_none_when_ineligible(self):
        # is_linear_eligible not patched -- conftest.py's global
        # has_hipblaslt()=False stub makes available() (and therefore
        # is_linear_eligible) False by default.
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x = torch.randn(2, 4)
            w = torch.randn(3, 4)
            out = torch.ops.amd_tuned_torch.hipblaslt_linear(x, w, None, 0)
        assert out is None

    def test_ck_gemm_linear_fake_shape_when_eligible(self, monkeypatch):
        monkeypatch.setattr(ck_gemm_ops, "is_eligible", lambda *a, **k: True)
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x = torch.randn(2, 4)
            w = torch.randn(3, 4)
            out = torch.ops.amd_tuned_torch.ck_gemm_linear(x, w, None, 0)
        assert out.shape == (2, 3)
        assert out.dtype == x.dtype

    def test_ck_gemm_linear_fake_none_when_ineligible(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x = torch.randn(2, 4)
            w = torch.randn(3, 4)
            out = torch.ops.amd_tuned_torch.ck_gemm_linear(x, w, None, 0)
        assert out is None

    def test_hipblaslt_bmm_fake_shape_when_eligible(self, monkeypatch):
        monkeypatch.setattr(hipblaslt_ops, "is_bmm_eligible", lambda *a, **k: True)
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x = torch.randn(5, 2, 4)
            y = torch.randn(5, 4, 3)
            out = torch.ops.amd_tuned_torch.hipblaslt_bmm(x, y)
        assert out.shape == (5, 2, 3)

    def test_hipblaslt_bmm_fake_none_when_ineligible(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x = torch.randn(5, 2, 4)
            y = torch.randn(5, 4, 3)
            out = torch.ops.amd_tuned_torch.hipblaslt_bmm(x, y)
        assert out is None


class TestDispatchCorrectness:
    """compile_ops.* must actually call through to aiter_ops/the native
    extension, same as calling them directly -- registering as a custom op
    must not change *what* gets computed, only whether torch.compile can
    trace it."""

    def test_linear_fp16_dispatches_to_aiter(self, monkeypatch):
        x = torch.randn(2, 4)
        w = torch.randn(3, 4)
        expected = torch.zeros(2, 3)
        fake = MagicMock(return_value=expected)
        monkeypatch.setattr(aiter_ops, "linear_fp16", fake)
        out = compile_ops.linear_fp16(x, w, None)
        assert torch.equal(out, expected)
        fake.assert_called_once_with(x, w, None)

    def test_bmm_fp16_dispatches_to_aiter(self, monkeypatch):
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 4, 5)
        expected = torch.zeros(2, 3, 5)
        fake = MagicMock(return_value=expected)
        monkeypatch.setattr(aiter_ops, "bmm_fp16", fake)
        out = compile_ops.bmm_fp16(x, y)
        assert torch.equal(out, expected)
        fake.assert_called_once_with(x, y)

    def test_conv2d_fp16_dispatches_to_aiter_with_normalized_args(self, monkeypatch):
        x = torch.randn(1, 3, 8, 8)
        w = torch.randn(4, 3, 3, 3)
        expected = torch.zeros(1, 4, 8, 8)
        fake = MagicMock(return_value=expected)
        monkeypatch.setattr(aiter_ops, "conv2d_fp16", fake)
        out = compile_ops.conv2d_fp16(x, w, None, stride=1, padding=1, dilation=1)
        assert torch.equal(out, expected)
        # A bare int stride/padding/dilation (as F.conv2d itself accepts)
        # must be normalized to a list before reaching the registered op's
        # list[int]-typed schema.
        fake.assert_called_once_with(x, w, None, [1, 1], [1, 1], [1, 1])

    def test_group_norm_dispatches_to_native(self, monkeypatch):
        x = torch.randn(2, 4, 3, 3)
        w = torch.randn(4)
        b = torch.randn(4)
        expected = torch.zeros_like(x)
        fake = MagicMock(return_value=expected)
        monkeypatch.setattr(compile_ops._C, "group_norm", fake)
        out = compile_ops.group_norm(x, 2, w, b, 1e-5)
        assert torch.equal(out, expected)
        fake.assert_called_once_with(x, 2, w, b, 1e-5)

    def test_conv2d_native_dispatches_to_native_with_normalized_args(self, monkeypatch):
        x = torch.randn(1, 3, 8, 8)
        w = torch.randn(4, 3, 3, 3)
        expected = torch.zeros(1, 4, 8, 8)
        fake = MagicMock(return_value=expected)
        monkeypatch.setattr(compile_ops._C, "conv2d", fake)
        out = compile_ops.conv2d_native(x, w, None, stride=1, padding=1, dilation=1)
        assert torch.equal(out, expected)
        fake.assert_called_once_with(x, w, None, [1, 1], [1, 1], [1, 1])

    def test_conv3d_native_dispatches_to_native_with_normalized_args(self, monkeypatch):
        x = torch.randn(1, 3, 4, 8, 8)
        w = torch.randn(4, 3, 3, 3, 3)
        expected = torch.zeros(1, 4, 4, 8, 8)
        fake = MagicMock(return_value=expected)
        monkeypatch.setattr(compile_ops._C, "conv3d", fake)
        out = compile_ops.conv3d_native(x, w, None, stride=1, padding=1, dilation=1)
        assert torch.equal(out, expected)
        fake.assert_called_once_with(x, w, None, [1, 1, 1], [1, 1, 1], [1, 1, 1])

    def test_hipblaslt_linear_dispatches_to_hipblaslt_ops(self, monkeypatch):
        x = torch.randn(2, 4)
        w = torch.randn(3, 4)
        expected = torch.zeros(2, 3)
        fake = MagicMock(return_value=expected)
        monkeypatch.setattr(hipblaslt_ops, "linear", fake)
        out = compile_ops.hipblaslt_linear(x, w, None)
        assert torch.equal(out, expected)
        fake.assert_called_once_with(x, w, None, hipblaslt_ops.EPILOGUE_NONE)

    def test_hipblaslt_linear_returns_none_when_hipblaslt_ops_declines(self, monkeypatch):
        monkeypatch.setattr(hipblaslt_ops, "linear", MagicMock(return_value=None))
        out = compile_ops.hipblaslt_linear(torch.randn(2, 4), torch.randn(3, 4), None)
        assert out is None

    def test_ck_gemm_linear_dispatches_to_ck_gemm_ops(self, monkeypatch):
        x = torch.randn(2, 4)
        w = torch.randn(3, 4)
        expected = torch.zeros(2, 3)
        fake = MagicMock(return_value=expected)
        monkeypatch.setattr(ck_gemm_ops, "linear", fake)
        out = compile_ops.ck_gemm_linear(x, w, None)
        assert torch.equal(out, expected)
        fake.assert_called_once_with(x, w, None, ck_gemm_ops.EPILOGUE_NONE)

    def test_hipblaslt_bmm_dispatches_to_hipblaslt_ops(self, monkeypatch):
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 4, 5)
        expected = torch.zeros(2, 3, 5)
        fake = MagicMock(return_value=expected)
        monkeypatch.setattr(hipblaslt_ops, "bmm", fake)
        out = compile_ops.hipblaslt_bmm(x, y)
        assert torch.equal(out, expected)
        fake.assert_called_once_with(x, y)


class TestTorchCompileFullGraph:
    """The actual point of compile_ops.py: torch.compile(fullgraph=True)
    must not graph-break or raise Unsupported on a call to one of these
    ops. aiter_ops's/the native extension's real kernels are stood in for
    with plain CPU ops here -- only tracing/dispatch behavior is under
    test, not real-kernel numerics (covered by the mocked-aiter tests in
    test_amd_tuned_torch_monkeypatch.py and the shape tests above)."""

    def test_linear_fp16_traces_under_fullgraph_compile(self, monkeypatch):
        monkeypatch.setattr(
            aiter_ops, "linear_fp16", lambda input_, weight, bias: F.linear(input_, weight, bias)
        )

        def fn(x, w, b):
            return compile_ops.linear_fp16(x, w, b)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x, w, b = torch.randn(2, 4), torch.randn(3, 4), torch.randn(3)
        assert torch.allclose(compiled(x, w, b), F.linear(x, w, b))

    def test_bmm_fp16_traces_under_fullgraph_compile(self, monkeypatch):
        monkeypatch.setattr(aiter_ops, "bmm_fp16", lambda input_, mat2: torch.bmm(input_, mat2))

        def fn(x, y):
            return compile_ops.bmm_fp16(x, y)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x, y = torch.randn(2, 3, 4), torch.randn(2, 4, 5)
        assert torch.allclose(compiled(x, y), torch.bmm(x, y))

    def test_conv2d_fp16_traces_under_fullgraph_compile(self, monkeypatch):
        monkeypatch.setattr(
            aiter_ops,
            "conv2d_fp16",
            lambda input_, weight, bias, stride, padding, dilation: F.conv2d(
                input_, weight, bias, stride, padding, dilation
            ),
        )

        def fn(x, w, b):
            return compile_ops.conv2d_fp16(x, w, b, stride=1, padding=1, dilation=1)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x, w, b = torch.randn(1, 3, 8, 8), torch.randn(4, 3, 3, 3), torch.randn(4)
        assert torch.allclose(
            compiled(x, w, b), F.conv2d(x, w, b, stride=1, padding=1, dilation=1)
        )

    def test_group_norm_traces_under_fullgraph_compile(self, monkeypatch):
        monkeypatch.setattr(
            compile_ops._C,
            "group_norm",
            lambda input_, num_groups, weight, bias, eps: F.group_norm(
                input_, num_groups, weight, bias, eps
            ),
        )

        def fn(x, w, b):
            return compile_ops.group_norm(x, 2, w, b, 1e-5)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x, w, b = torch.randn(2, 4, 3, 3), torch.randn(4), torch.randn(4)
        assert torch.allclose(compiled(x, w, b), F.group_norm(x, 2, w, b, 1e-5))

    def test_conv2d_native_traces_under_fullgraph_compile(self, monkeypatch):
        monkeypatch.setattr(
            compile_ops._C,
            "conv2d",
            lambda input_, weight, bias, stride, padding, dilation: F.conv2d(
                input_, weight, bias, stride, padding, dilation
            ),
        )

        def fn(x, w, b):
            return compile_ops.conv2d_native(x, w, b, stride=1, padding=1, dilation=1)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x, w, b = torch.randn(1, 3, 8, 8), torch.randn(4, 3, 3, 3), torch.randn(4)
        assert torch.allclose(
            compiled(x, w, b), F.conv2d(x, w, b, stride=1, padding=1, dilation=1)
        )

    def test_conv3d_native_traces_under_fullgraph_compile(self, monkeypatch):
        monkeypatch.setattr(
            compile_ops._C,
            "conv3d",
            lambda input_, weight, bias, stride, padding, dilation: F.conv3d(
                input_, weight, bias, stride, padding, dilation
            ),
        )

        def fn(x, w, b):
            return compile_ops.conv3d_native(x, w, b, stride=1, padding=1, dilation=1)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x, w, b = torch.randn(1, 3, 4, 8, 8), torch.randn(4, 3, 3, 3, 3), torch.randn(4)
        assert torch.allclose(
            compiled(x, w, b), F.conv3d(x, w, b, stride=1, padding=1, dilation=1)
        )

    def test_hipblaslt_linear_traces_under_fullgraph_compile(self, monkeypatch):
        """The real point of the schema="..-> Tensor?" string: a genuine
        Dynamo fullgraph trace, not just FakeTensorMode called by hand,
        through an op whose fake sometimes legitimately returns None."""
        monkeypatch.setattr(hipblaslt_ops, "is_linear_eligible", lambda *a, **k: True)
        monkeypatch.setattr(
            hipblaslt_ops, "linear",
            lambda input_, weight, bias, epilogue: F.linear(input_, weight, bias),
        )

        def fn(x, w, b):
            return compile_ops.hipblaslt_linear(x, w, b)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x, w, b = torch.randn(2, 4), torch.randn(3, 4), torch.randn(3)
        assert torch.allclose(compiled(x, w, b), F.linear(x, w, b))
