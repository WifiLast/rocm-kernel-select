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
