"""Tests for amd_tuned_torch.hipblaslt_ops.is_linear_eligible/
is_bmm_eligible -- pulled out of linear()/bmm()'s bodies into their own
functions so amd_tuned_torch.compile_ops's register_fake for
hipblaslt_linear/hipblaslt_bmm can reuse the exact same eligibility
decision at trace time (see compile_ops.py's own docstring). Regression
coverage for that extraction: linear()/bmm() must behave identically to
before.

conftest.py stubs has_hipblaslt()=False globally, so available() is False
unless a test monkeypatches it. is_linear_eligible/is_bmm_eligible also
require is_cuda, False for every plain CPU tensor here -- _patch_is_cuda
monkeypatches torch.Tensor.is_cuda, same technique
tests/test_miopen_fallback.py::TestIsDepthwiseConv1dEligible uses for the
identical situation.

Run with:

    python -m pytest source/cmp_ext_turing/tests
"""
from __future__ import annotations

import torch

import amd_tuned_torch.hipblaslt_ops as hipblaslt_ops


def _patch_is_cuda(monkeypatch, value: bool = True) -> None:
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: value))


class TestIsLinearEligible:
    def test_ineligible_when_unavailable(self, monkeypatch):
        _patch_is_cuda(monkeypatch)
        x = torch.randn(2, 4, dtype=torch.float16)
        w = torch.randn(3, 4, dtype=torch.float16)
        assert hipblaslt_ops.is_linear_eligible(x, w) is False

    def test_ineligible_on_cpu(self, monkeypatch):
        monkeypatch.setattr(hipblaslt_ops, "available", lambda: True)
        x = torch.randn(2, 4, dtype=torch.float16)
        w = torch.randn(3, 4, dtype=torch.float16)
        assert x.is_cuda is False
        assert hipblaslt_ops.is_linear_eligible(x, w) is False

    def test_eligible_when_available(self, monkeypatch):
        _patch_is_cuda(monkeypatch)
        monkeypatch.setattr(hipblaslt_ops, "available", lambda: True)
        x = torch.randn(2, 4, dtype=torch.float16)
        w = torch.randn(3, 4, dtype=torch.float16)
        assert hipblaslt_ops.is_linear_eligible(x, w) is True

    def test_eligible_for_fp32_unlike_ck_gemm(self, monkeypatch):
        # hipblaslt_ops supports fp32 (ck_gemm_ops doesn't -- gfx1100 has no
        # fp32 WMMA path at all, see ck_gemm_ops.py's own docstring).
        _patch_is_cuda(monkeypatch)
        monkeypatch.setattr(hipblaslt_ops, "available", lambda: True)
        x = torch.randn(2, 4, dtype=torch.float32)
        w = torch.randn(3, 4, dtype=torch.float32)
        assert hipblaslt_ops.is_linear_eligible(x, w) is True

    def test_ineligible_for_non_2d_weight(self, monkeypatch):
        _patch_is_cuda(monkeypatch)
        monkeypatch.setattr(hipblaslt_ops, "available", lambda: True)
        x = torch.randn(2, 4, dtype=torch.float16)
        w = torch.randn(1, 3, 4, dtype=torch.float16)
        assert hipblaslt_ops.is_linear_eligible(x, w) is False

    def test_ineligible_for_bias_shape_mismatch(self, monkeypatch):
        _patch_is_cuda(monkeypatch)
        monkeypatch.setattr(hipblaslt_ops, "available", lambda: True)
        x = torch.randn(2, 4, dtype=torch.float16)
        w = torch.randn(3, 4, dtype=torch.float16)
        bad_bias = torch.randn(5, dtype=torch.float16)
        assert hipblaslt_ops.is_linear_eligible(x, w, bad_bias) is False


class TestIsBmmEligible:
    def test_ineligible_when_unavailable(self, monkeypatch):
        _patch_is_cuda(monkeypatch)
        x = torch.randn(2, 3, 4, dtype=torch.float16)
        y = torch.randn(2, 4, 5, dtype=torch.float16)
        assert hipblaslt_ops.is_bmm_eligible(x, y) is False

    def test_eligible_when_available(self, monkeypatch):
        _patch_is_cuda(monkeypatch)
        monkeypatch.setattr(hipblaslt_ops, "available", lambda: True)
        x = torch.randn(2, 3, 4, dtype=torch.float16)
        y = torch.randn(2, 4, 5, dtype=torch.float16)
        assert hipblaslt_ops.is_bmm_eligible(x, y) is True

    def test_ineligible_for_non_3d_input(self, monkeypatch):
        _patch_is_cuda(monkeypatch)
        monkeypatch.setattr(hipblaslt_ops, "available", lambda: True)
        x = torch.randn(3, 4, dtype=torch.float16)
        y = torch.randn(4, 5, dtype=torch.float16)
        assert hipblaslt_ops.is_bmm_eligible(x, y) is False

    def test_ineligible_for_mismatched_dtypes(self, monkeypatch):
        _patch_is_cuda(monkeypatch)
        monkeypatch.setattr(hipblaslt_ops, "available", lambda: True)
        x = torch.randn(2, 3, 4, dtype=torch.float16)
        y = torch.randn(2, 4, 5, dtype=torch.bfloat16)
        assert hipblaslt_ops.is_bmm_eligible(x, y) is False


class TestLinearAndBmmUseEligibilityHelpers:
    def test_linear_declines_when_ineligible(self):
        x = torch.randn(2, 4, dtype=torch.float16)
        w = torch.randn(3, 4, dtype=torch.float16)
        assert hipblaslt_ops.linear(x, w) is None

    def test_bmm_declines_when_ineligible(self):
        x = torch.randn(2, 3, 4, dtype=torch.float16)
        y = torch.randn(2, 4, 5, dtype=torch.float16)
        assert hipblaslt_ops.bmm(x, y) is None
