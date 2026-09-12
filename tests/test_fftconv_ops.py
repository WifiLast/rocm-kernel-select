"""Tests for amd_tuned_torch.fftconv_ops -- pure PyTorch (no extension, no
GPU required), so unlike test_flexgemm_ops.py this exercises real numerics
against F.convNd rather than mocking an external kernel.

Run with:

    python -m pytest source/cmp_ext_turing/tests
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import amd_tuned_torch.fftconv_ops as fftconv_ops


def _rand(*shape, dtype=torch.float32):
    return torch.randn(*shape, dtype=dtype)


class TestAvailable:
    def test_always_available(self):
        assert fftconv_ops.available() is True


class TestFftConv1dMatchesDirect:
    def test_basic(self):
        x = _rand(2, 3, 200)
        w = _rand(5, 3, 33)
        b = _rand(5)
        expected = F.conv1d(x, w, bias=b, padding=4)
        actual = fftconv_ops.fft_conv1d(x, w, bias=b, padding=4)
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)

    def test_stride_and_groups(self):
        x = _rand(1, 4, 128)
        w = _rand(4, 1, 17)
        expected = F.conv1d(x, w, stride=2, padding=8, groups=4)
        actual = fftconv_ops.fft_conv1d(x, w, stride=2, padding=8, groups=4)
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)

    def test_no_bias(self):
        x = _rand(1, 2, 64)
        w = _rand(6, 2, 9)
        expected = F.conv1d(x, w, padding=2)
        actual = fftconv_ops.fft_conv1d(x, w, padding=2)
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)

    def test_preserves_input_dtype(self):
        x = _rand(1, 2, 64, dtype=torch.float16)
        w = _rand(3, 2, 9, dtype=torch.float16)
        out = fftconv_ops.fft_conv1d(x, w, padding=4)
        assert out.dtype == torch.float16

    def test_rejects_wrong_ndim(self):
        with pytest.raises(ValueError):
            fftconv_ops.fft_conv1d(_rand(1, 2, 4, 4), _rand(1, 2, 3, 3))


class TestFftConv2d3dMatchDirect:
    def test_conv2d(self):
        x = _rand(1, 2, 20, 20)
        w = _rand(3, 2, 5, 5)
        expected = F.conv2d(x, w, padding=2)
        actual = fftconv_ops.fft_conv2d(x, w, padding=2)
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)

    def test_conv3d(self):
        x = _rand(1, 1, 10, 10, 10)
        w = _rand(2, 1, 3, 3, 3)
        expected = F.conv3d(x, w, padding=1)
        actual = fftconv_ops.fft_conv3d(x, w, padding=1)
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)


class TestGroupedConv:
    """`groups` reaches fft_conv unfiltered from both the public
    fft_convNd entry points and the conv1d/2d/3d auto-patch candidates, so
    every grouping shape has to work. The `Cout/groups > 1` cases below
    used to raise `RuntimeError: view size is not compatible ...` out of
    complex_matmul's final `view` -- see that function's docstring."""

    @pytest.mark.parametrize("groups,cin,cout", [(2, 4, 6), (8, 16, 16), (4, 8, 4)])
    def test_matches_direct_conv1d(self, groups, cin, cout):
        x = _rand(2, cin, 128)
        w = _rand(cout, cin // groups, 17)
        b = _rand(cout)
        expected = F.conv1d(x, w, bias=b, padding=8, groups=groups)
        actual = fftconv_ops.fft_conv1d(x, w, bias=b, padding=8, groups=groups)
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)

    def test_depthwise_with_channel_multiplier(self):
        """Cin/groups == 1 but Cout > groups -- not the elementwise
        depthwise case, so it must take complex_matmul's einsum branch."""
        x = _rand(2, 4, 96)
        w = _rand(8, 1, 9)
        expected = F.conv1d(x, w, padding=4, groups=4)
        actual = fftconv_ops.fft_conv1d(x, w, padding=4, groups=4)
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)

    def test_matches_direct_conv2d(self):
        x = _rand(1, 4, 24, 24)
        w = _rand(6, 2, 5, 5)
        expected = F.conv2d(x, w, padding=2, groups=2)
        actual = fftconv_ops.fft_conv2d(x, w, padding=2, groups=2)
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)


class TestComplexMatmul:
    """complex_matmul is the frequency-domain contraction rewritten as an
    einsum for layout reasons (6-80x on gfx1100, see its docstring). This
    pins it against the reshaped-`@` formulation it replaced, which is the
    definition of the contraction it has to keep computing."""

    @staticmethod
    def _reference(a, b, groups):
        a = a.view(a.size(0), groups, -1, *a.shape[2:])
        b = b.view(groups, -1, *b.shape[1:])
        a = torch.movedim(a, 2, a.dim() - 1).unsqueeze(-2)
        b = torch.movedim(b, (1, 2), (b.dim() - 1, b.dim() - 2))
        c = torch.movedim(a @ b, -1, 2).squeeze(-1)
        return c.reshape(c.size(0), -1, *c.shape[3:])

    @pytest.mark.parametrize("groups,cin,cout", [(1, 8, 6), (2, 8, 6), (4, 4, 4), (4, 4, 8)])
    def test_matches_reshaped_matmul(self, groups, cin, cout):
        a = torch.randn(3, cin, 17, dtype=torch.complex64)
        b = torch.randn(cout, cin // groups, 17, dtype=torch.complex64)
        torch.testing.assert_close(
            fftconv_ops.complex_matmul(a, b, groups=groups),
            self._reference(a, b, groups),
        )

    def test_preserves_nd_frequency_shape(self):
        a = torch.randn(2, 4, 5, 7, dtype=torch.complex64)
        b = torch.randn(6, 2, 5, 7, dtype=torch.complex64)
        out = fftconv_ops.complex_matmul(a, b, groups=2)
        assert out.shape == (2, 6, 5, 7)
        torch.testing.assert_close(out, self._reference(a, b, 2))

    def test_is_differentiable(self):
        """fft_conv's autograd is built entirely from differentiable
        primitives (module docstring) -- both branches here included."""
        for groups, cin, cout in ((1, 4, 4), (4, 4, 4)):
            a = torch.randn(2, cin, 9, dtype=torch.complex64, requires_grad=True)
            b = torch.randn(cout, cin // groups, 9, dtype=torch.complex64, requires_grad=True)
            fftconv_ops.complex_matmul(a, b, groups=groups).abs().sum().backward()
            assert a.grad is not None and b.grad is not None


class TestMixedPrecisionWindow:
    """fft_conv's MIXED PRECISION contract: dilation expansion (torch.kron)
    and padding (F.pad) stay in the caller's storage dtype (fp16/bf16),
    only rfftn/complex_matmul/irfftn run in float32 -- see fft_conv's own
    docstring for why. These spy on the real calls rather than mock them,
    so correctness is exercised at the same time as the dtype claim."""

    def test_kron_and_pad_stay_in_storage_dtype(self, monkeypatch):
        kron_dtypes = []
        orig_kron = torch.kron

        def spy_kron(a, b):
            kron_dtypes.append(a.dtype)
            return orig_kron(a, b)

        pad_dtypes = []
        orig_pad = F.pad

        def spy_pad(x, *args, **kwargs):
            pad_dtypes.append(x.dtype)
            return orig_pad(x, *args, **kwargs)

        monkeypatch.setattr(fftconv_ops.torch, "kron", spy_kron)
        monkeypatch.setattr(fftconv_ops.F, "pad", spy_pad)

        x = _rand(1, 2, 64, dtype=torch.float16)
        w = _rand(3, 2, 9, dtype=torch.float16)
        fftconv_ops.fft_conv1d(x, w, padding=4)

        assert kron_dtypes and all(d == torch.float16 for d in kron_dtypes)
        assert pad_dtypes and all(d == torch.float16 for d in pad_dtypes)

    def test_rfftn_runs_in_float32(self, monkeypatch):
        rfftn_dtypes = []
        orig_rfftn = fftconv_ops.rfftn

        def spy_rfftn(t, dim):
            rfftn_dtypes.append(t.dtype)
            return orig_rfftn(t, dim=dim)

        monkeypatch.setattr(fftconv_ops, "rfftn", spy_rfftn)

        x = _rand(1, 2, 64, dtype=torch.float16)
        w = _rand(3, 2, 9, dtype=torch.float16)
        fftconv_ops.fft_conv1d(x, w, padding=4)

        assert rfftn_dtypes and all(d == torch.float32 for d in rfftn_dtypes)

    def test_fp16_matches_direct_conv(self):
        x = _rand(2, 4, 256, dtype=torch.float16)
        w = _rand(4, 4, 33, dtype=torch.float16)
        b = _rand(4, dtype=torch.float16)
        expected = F.conv1d(x, w, bias=b, padding=16)
        actual = fftconv_ops.fft_conv1d(x, w, bias=b, padding=16)
        assert actual.dtype == torch.float16
        torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)

    def test_bf16_matches_direct_conv(self):
        x = _rand(1, 2, 128, dtype=torch.bfloat16)
        w = _rand(2, 2, 17, dtype=torch.bfloat16)
        expected = F.conv1d(x, w, padding=8)
        actual = fftconv_ops.fft_conv1d(x, w, padding=8)
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual, expected, rtol=8e-2, atol=8e-2)


class TestMaybeFftConv1d:
    def test_declines_below_min_kernel(self, monkeypatch):
        monkeypatch.setattr(fftconv_ops, "fft_conv1d_enabled", lambda: True)
        x = _rand(1, 2, 64)
        w = _rand(2, 2, 9)
        assert fftconv_ops.maybe_fft_conv1d(x, w, min_kernel=128) is None

    def test_engages_above_min_kernel(self, monkeypatch):
        monkeypatch.setattr(fftconv_ops, "fft_conv1d_enabled", lambda: True)
        x = _rand(1, 2, 512)
        w = _rand(2, 2, 129)
        out = fftconv_ops.maybe_fft_conv1d(x, w, padding=(64,), min_kernel=128)
        assert out is not None
        expected = F.conv1d(x, w, padding=64)
        torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-4)

    def test_disabled_globally(self, monkeypatch):
        monkeypatch.setattr(fftconv_ops, "fft_conv1d_enabled", lambda: False)
        x = _rand(1, 2, 512)
        w = _rand(2, 2, 129)
        assert fftconv_ops.maybe_fft_conv1d(x, w, min_kernel=128) is None

    def test_declines_for_non_conv1d_shapes(self, monkeypatch):
        monkeypatch.setattr(fftconv_ops, "fft_conv1d_enabled", lambda: True)
        assert fftconv_ops.maybe_fft_conv1d(_rand(1, 2, 4, 4), _rand(1, 2, 3, 3)) is None
