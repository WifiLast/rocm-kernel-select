"""CPU-safe tests for fused_ce_ops -- guard-clause assertions only. The
Triton kernel needs a real GPU to launch, so numerical correctness
(loss AND gradients, against F.cross_entropy + autograd) lives in
tests_hardware/test_fused_ce_ops.py instead, same split as every other
Triton module in this package."""
from __future__ import annotations

import pytest
import torch

import amd_tuned_torch
from amd_tuned_torch import fused_ce_ops


class TestAvailable:
    def test_matches_triton_importability(self):
        assert fused_ce_ops.available() is fused_ce_ops._TRITON_AVAILABLE


class TestFusedLinearCrossEntropyGuards:
    def _unavailable(self, monkeypatch):
        monkeypatch.setattr(fused_ce_ops, "_TRITON_AVAILABLE", False)

    def _args(self, BT=4, H=8, V=16):
        input = torch.randn(BT, H)
        weight = torch.randn(V, H)
        target = torch.randint(0, V, (BT,))
        return input, weight, target

    def test_raises_when_triton_unavailable(self, monkeypatch):
        self._unavailable(monkeypatch)
        input, weight, target = self._args()
        with pytest.raises(RuntimeError, match="triton is not importable"):
            fused_ce_ops.fused_linear_cross_entropy(input, weight, target)

    def test_raises_on_non_2d_input(self, monkeypatch):
        monkeypatch.setattr(fused_ce_ops, "_TRITON_AVAILABLE", True)
        input = torch.randn(2, 4, 8)
        weight = torch.randn(16, 8)
        target = torch.randint(0, 16, (2,))
        with pytest.raises(AssertionError, match="input must be 2D"):
            fused_ce_ops.fused_linear_cross_entropy(input, weight, target)

    def test_raises_on_hidden_dim_mismatch(self, monkeypatch):
        monkeypatch.setattr(fused_ce_ops, "_TRITON_AVAILABLE", True)
        input, weight, target = self._args(H=8)
        weight = torch.randn(16, 12)  # wrong H
        with pytest.raises(AssertionError, match="weight must be"):
            fused_ce_ops.fused_linear_cross_entropy(input, weight, target)

    def test_raises_on_target_bt_mismatch(self, monkeypatch):
        monkeypatch.setattr(fused_ce_ops, "_TRITON_AVAILABLE", True)
        input, weight, _ = self._args(BT=4)
        target = torch.randint(0, 16, (5,))
        with pytest.raises(AssertionError, match="target must be"):
            fused_ce_ops.fused_linear_cross_entropy(input, weight, target)

    def test_raises_on_unsupported_reduction(self, monkeypatch):
        monkeypatch.setattr(fused_ce_ops, "_TRITON_AVAILABLE", True)
        input, weight, target = self._args()
        with pytest.raises(AssertionError, match="reduction='mean' or 'sum'"):
            fused_ce_ops.fused_linear_cross_entropy(input, weight, target, reduction="none")

    def test_raises_on_wrong_bias_shape(self, monkeypatch):
        monkeypatch.setattr(fused_ce_ops, "_TRITON_AVAILABLE", True)
        input, weight, target = self._args(V=16)
        bias = torch.randn(8)  # should be (V=16,)
        with pytest.raises(AssertionError, match="bias must be"):
            fused_ce_ops.fused_linear_cross_entropy(input, weight, target, bias=bias)

    def test_raises_on_wrong_ce_weight_shape(self, monkeypatch):
        monkeypatch.setattr(fused_ce_ops, "_TRITON_AVAILABLE", True)
        input, weight, target = self._args(V=16)
        ce_weight = torch.randn(8)  # should be (V=16,)
        with pytest.raises(AssertionError, match="ce_weight must be"):
            fused_ce_ops.fused_linear_cross_entropy(input, weight, target, ce_weight=ce_weight)

    def test_raises_on_non_float_ce_weight(self, monkeypatch):
        monkeypatch.setattr(fused_ce_ops, "_TRITON_AVAILABLE", True)
        input, weight, target = self._args(V=16)
        ce_weight = torch.randint(0, 2, (16,))
        with pytest.raises(AssertionError, match="floating point"):
            fused_ce_ops.fused_linear_cross_entropy(input, weight, target, ce_weight=ce_weight)


class TestModuleWiring:
    def test_top_level_delegates(self, monkeypatch):
        called = {}

        def fake(input, weight, target, bias=None, ce_weight=None, ignore_index=-100,
                 label_smoothing=0.0, reduction="mean", softcap=None):
            called["args"] = (input, weight, target, bias, ce_weight, ignore_index,
                               label_smoothing, reduction, softcap)
            return "sentinel"

        monkeypatch.setattr(amd_tuned_torch.fused_ce_ops, "fused_linear_cross_entropy", fake)
        i, w, t = object(), object(), object()
        result = amd_tuned_torch.fused_linear_cross_entropy(i, w, t, label_smoothing=0.1)
        assert result == "sentinel"
        assert called["args"] == (i, w, t, None, None, -100, 0.1, "mean", None)
