"""Tests for amd_tuned_torch.ck_gemm_ops.is_eligible -- pulled out of
linear()'s body into its own function so amd_tuned_torch.compile_ops's
register_fake for ck_gemm_linear can reuse the exact same eligibility
decision at trace time (see compile_ops.py's own docstring). Regression
coverage for that extraction: linear() must behave identically to before.

conftest.py stubs has_ck()=False globally, so available() is False unless
a test monkeypatches it -- same convention as every other CK/hipBLASLt test
in this suite. is_eligible also requires input.is_cuda/weight.is_cuda,
which is False for every plain CPU tensor in this test environment --
_patch_is_cuda monkeypatches torch.Tensor.is_cuda to True for the "should
be eligible" cases, same technique
tests/test_miopen_fallback.py::TestIsDepthwiseConv1dEligible uses for the
identical situation.

Run with:

    python -m pytest source/cmp_ext_turing/tests
"""
from __future__ import annotations

import torch

import amd_tuned_torch.ck_gemm_ops as ck_gemm_ops


def _patch_is_cuda(monkeypatch, value: bool = True) -> None:
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: value))


class TestIsEligible:
    def test_ineligible_when_unavailable(self, monkeypatch):
        _patch_is_cuda(monkeypatch)
        # available() is False by conftest.py's global has_ck() stub, even
        # though is_cuda is patched True -- confirms available() is checked.
        x = torch.randn(2, 4, dtype=torch.float16)
        w = torch.randn(3, 4, dtype=torch.float16)
        assert ck_gemm_ops.is_eligible(x, w) is False

    def test_ineligible_on_cpu(self, monkeypatch):
        monkeypatch.setattr(ck_gemm_ops, "available", lambda: True)
        # is_cuda NOT patched here -- real CPU tensors, real (unpatched)
        # is_cuda gate.
        x = torch.randn(2, 4, dtype=torch.float16)
        w = torch.randn(3, 4, dtype=torch.float16)
        assert x.is_cuda is False
        assert ck_gemm_ops.is_eligible(x, w) is False

    def test_eligible_fp16_when_available(self, monkeypatch):
        _patch_is_cuda(monkeypatch)
        monkeypatch.setattr(ck_gemm_ops, "available", lambda: True)
        x = torch.randn(2, 4, dtype=torch.float16)
        w = torch.randn(3, 4, dtype=torch.float16)
        assert ck_gemm_ops.is_eligible(x, w) is True

    def test_ineligible_for_fp32(self, monkeypatch):
        _patch_is_cuda(monkeypatch)
        monkeypatch.setattr(ck_gemm_ops, "available", lambda: True)
        x = torch.randn(2, 4, dtype=torch.float32)
        w = torch.randn(3, 4, dtype=torch.float32)
        assert ck_gemm_ops.is_eligible(x, w) is False

    def test_ineligible_for_mismatched_dtypes(self, monkeypatch):
        _patch_is_cuda(monkeypatch)
        monkeypatch.setattr(ck_gemm_ops, "available", lambda: True)
        x = torch.randn(2, 4, dtype=torch.float16)
        w = torch.randn(3, 4, dtype=torch.bfloat16)
        assert ck_gemm_ops.is_eligible(x, w) is False

    def test_ineligible_for_non_2d_weight(self, monkeypatch):
        _patch_is_cuda(monkeypatch)
        monkeypatch.setattr(ck_gemm_ops, "available", lambda: True)
        x = torch.randn(2, 4, dtype=torch.float16)
        w = torch.randn(1, 3, 4, dtype=torch.float16)
        assert ck_gemm_ops.is_eligible(x, w) is False

    def test_ineligible_for_bias_shape_mismatch(self, monkeypatch):
        _patch_is_cuda(monkeypatch)
        monkeypatch.setattr(ck_gemm_ops, "available", lambda: True)
        x = torch.randn(2, 4, dtype=torch.float16)
        w = torch.randn(3, 4, dtype=torch.float16)
        bad_bias = torch.randn(5, dtype=torch.float16)
        assert ck_gemm_ops.is_eligible(x, w, bad_bias) is False

    def test_eligible_with_matching_bias(self, monkeypatch):
        _patch_is_cuda(monkeypatch)
        monkeypatch.setattr(ck_gemm_ops, "available", lambda: True)
        x = torch.randn(2, 4, dtype=torch.float16)
        w = torch.randn(3, 4, dtype=torch.float16)
        bias = torch.randn(3, dtype=torch.float16)
        assert ck_gemm_ops.is_eligible(x, w, bias) is True


class TestLinearUsesIsEligible:
    def test_linear_declines_when_ineligible(self):
        x = torch.randn(2, 4, dtype=torch.float16)
        w = torch.randn(3, 4, dtype=torch.float16)
        assert ck_gemm_ops.linear(x, w) is None
