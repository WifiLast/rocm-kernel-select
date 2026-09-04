"""CPU-safe tests for fused_norm_ops -- pure Python guard-clause logic
only (shape/dtype/device/autograd assertions, and the block-size helper).
The actual @triton.jit kernel needs a real GPU to launch (Triton compiles
for the target backend), so the numerical-correctness check against an
unfused reference lives in tests_hardware/test_fused_norm_ops.py instead,
same split as tests_hardware/test_ck_conv.py."""
from __future__ import annotations

import pytest
import torch

import amd_tuned_torch
from amd_tuned_torch import fused_norm_ops


class TestAvailable:
    def test_matches_triton_importability(self):
        assert fused_norm_ops.available() is fused_norm_ops._TRITON_AVAILABLE


class TestNextPow2:
    @pytest.mark.parametrize("n, expected", [(1, 1), (2, 2), (3, 4), (4, 4), (5, 8),
                                              (4096, 4096), (4097, 8192), (7168, 8192)])
    def test_next_pow2(self, n, expected):
        assert fused_norm_ops._next_pow2(n) == expected


class TestFusedAddRmsNormGuards:
    """These run regardless of whether triton is installed here: every
    guard below raises/asserts before the function ever reaches a
    _fused_add_rmsnorm_kernel[...] launch, so they're pure-Python checks
    exercisable on CPU tensors."""

    def _unavailable(self, monkeypatch):
        monkeypatch.setattr(fused_norm_ops, "_TRITON_AVAILABLE", False)

    def test_raises_when_triton_unavailable(self, monkeypatch):
        self._unavailable(monkeypatch)
        x = torch.randn(2, 8)
        residual = torch.randn(2, 8)
        weight = torch.randn(8)
        with pytest.raises(RuntimeError, match="triton is not importable"):
            fused_norm_ops.fused_add_rms_norm(x, residual, weight)

    def test_requires_grad_no_longer_rejected(self, monkeypatch):
        # fused_add_rms_norm now has a real backward pass (see module
        # docstring) -- a requires_grad=True input must reach the same
        # device assertion every other guard test hits, NOT an
        # autograd-specific rejection (that guard existed only in this
        # module's original forward-only version).
        monkeypatch.setattr(fused_norm_ops, "_TRITON_AVAILABLE", True)
        x = torch.randn(2, 8, requires_grad=True)
        residual = torch.randn(2, 8)
        weight = torch.randn(8, requires_grad=True)
        with pytest.raises(AssertionError, match="CUDA/ROCm"):
            fused_norm_ops.fused_add_rms_norm(x, residual, weight)

    def test_raises_on_shape_mismatch(self, monkeypatch):
        monkeypatch.setattr(fused_norm_ops, "_TRITON_AVAILABLE", True)
        x = torch.randn(2, 8)
        residual = torch.randn(2, 16)
        weight = torch.randn(8)
        with pytest.raises(AssertionError, match="same shape"):
            fused_norm_ops.fused_add_rms_norm(x, residual, weight)

    def test_raises_on_weight_dim_mismatch(self, monkeypatch):
        monkeypatch.setattr(fused_norm_ops, "_TRITON_AVAILABLE", True)
        x = torch.randn(2, 8)
        residual = torch.randn(2, 8)
        weight = torch.randn(16)
        with pytest.raises(AssertionError, match="last dim"):
            fused_norm_ops.fused_add_rms_norm(x, residual, weight)

    def test_raises_on_unsupported_dtype(self, monkeypatch):
        monkeypatch.setattr(fused_norm_ops, "_TRITON_AVAILABLE", True)
        x = torch.randint(0, 10, (2, 8)).to(torch.int32)
        residual = torch.randint(0, 10, (2, 8)).to(torch.int32)
        weight = torch.randn(8)
        with pytest.raises(AssertionError, match="unsupported dtype"):
            fused_norm_ops.fused_add_rms_norm(x, residual, weight)

    def test_raises_on_dtype_mismatch(self, monkeypatch):
        monkeypatch.setattr(fused_norm_ops, "_TRITON_AVAILABLE", True)
        x = torch.randn(2, 8, dtype=torch.float16)
        residual = torch.randn(2, 8, dtype=torch.float32)
        weight = torch.randn(8, dtype=torch.float16)
        with pytest.raises(AssertionError, match="share a dtype"):
            fused_norm_ops.fused_add_rms_norm(x, residual, weight)

    def test_raises_on_cpu_tensors(self, monkeypatch):
        monkeypatch.setattr(fused_norm_ops, "_TRITON_AVAILABLE", True)
        x = torch.randn(2, 8)
        residual = torch.randn(2, 8)
        weight = torch.randn(8)
        with pytest.raises(AssertionError, match="CUDA/ROCm"):
            fused_norm_ops.fused_add_rms_norm(x, residual, weight)


class TestModuleWiring:
    def test_top_level_delegates(self, monkeypatch):
        called = {}

        def fake(x, residual, weight, eps=1e-6):
            called["args"] = (x, residual, weight, eps)
            return "sentinel"

        monkeypatch.setattr(amd_tuned_torch.fused_norm_ops, "fused_add_rms_norm", fake)
        x, residual, weight = object(), object(), object()
        result = amd_tuned_torch.fused_add_rms_norm(x, residual, weight, eps=1e-5)
        assert result == "sentinel"
        assert called["args"] == (x, residual, weight, 1e-5)
