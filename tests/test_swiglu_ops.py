"""Tests for swiglu_ops -- unlike every other Triton-backed module in
this package, this one is plain PyTorch, so it runs (and can be fully
verified) on CPU with no GPU/Triton at all. torch.autograd.gradcheck
compares the hand-derived backward (_swiglu_bwd/_clamped_swiglu_bwd)
against numerical (finite-difference) differentiation of the forward --
a stronger correctness signal than comparing against a second
hand-written reference, since it doesn't share any derivation mistakes
the forward and a reference might have in common."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import amd_tuned_torch
from amd_tuned_torch import swiglu_ops


class TestAvailable:
    def test_always_available(self):
        assert swiglu_ops.available() is True


def _double_input(*shape, requires_grad=True):
    return torch.randn(*shape, dtype=torch.float64, requires_grad=requires_grad)


class TestGradcheckPlain:
    def test_no_bias(self):
        x = _double_input(3, 16)
        assert torch.autograd.gradcheck(lambda x: swiglu_ops.bias_swiglu(x), (x,))

    def test_with_bias_matching_last_dim(self):
        x = _double_input(3, 16)
        b = _double_input(16)
        assert torch.autograd.gradcheck(lambda x, b: swiglu_ops.bias_swiglu(x, b), (x, b))

    def test_with_bias_broadcast_over_leading_dims(self):
        # input has TWO leading dims (batch, seq) beyond bias's own shape --
        # exercises _reduce_bias_grad summing over more than one dim.
        x = _double_input(2, 5, 16)
        b = _double_input(16)
        assert torch.autograd.gradcheck(lambda x, b: swiglu_ops.bias_swiglu(x, b), (x, b))


class TestGradcheckClamped:
    def test_no_bias(self):
        # Keep inputs well inside +/-clamp_value so no random sample lands
        # on the clamp's non-differentiable boundary.
        x = _double_input(3, 16) * 0.1
        assert torch.autograd.gradcheck(
            lambda x: swiglu_ops.bias_swiglu(x, clamp_value=5.0), (x,)
        )

    def test_with_bias(self):
        x = _double_input(3, 16) * 0.1
        b = _double_input(16) * 0.1
        assert torch.autograd.gradcheck(
            lambda x, b: swiglu_ops.bias_swiglu(x, b, clamp_value=5.0), (x, b)
        )

    def test_clamp_actually_engages(self):
        """A gradcheck pass alone can't tell you the clamp branch was ever
        exercised (small random inputs may never reach it) -- this checks
        the forward value directly against a hand-evaluated clamped case."""
        y1 = torch.tensor([10.0, -10.0])  # both would be clamped
        y2 = torch.tensor([10.0, -10.0])
        y = torch.cat([y1, y2])
        clamp_value = 2.0
        out = swiglu_ops.bias_swiglu(y, clamp_value=clamp_value)
        y1c = y1.clamp(max=clamp_value)
        y2c = y2.clamp(min=-clamp_value, max=clamp_value)
        expected = F.silu(y1c) * y2c
        assert torch.allclose(out, expected)


class TestMatchesUnfusedForward:
    """Forward-value check against the plain unfused computation, in
    ordinary fp32 (gradcheck above already covers the gradient; this
    covers the forward at the dtype callers will actually use)."""

    def test_matches_plain_swiglu(self):
        torch.manual_seed(0)
        x = torch.randn(4, 8, 32)
        bias = torch.randn(32)
        out = swiglu_ops.bias_swiglu(x, bias)
        y = x + bias
        y1, y2 = y.chunk(2, dim=-1)
        expected = F.silu(y1) * y2
        assert torch.allclose(out, expected, atol=1e-6)

    def test_matches_no_bias(self):
        torch.manual_seed(1)
        x = torch.randn(4, 32)
        out = swiglu_ops.bias_swiglu(x)
        y1, y2 = x.chunk(2, dim=-1)
        expected = F.silu(y1) * y2
        assert torch.allclose(out, expected, atol=1e-6)


class TestGuards:
    def test_raises_on_odd_last_dim(self):
        x = torch.randn(2, 7)
        with pytest.raises(AssertionError, match="even"):
            swiglu_ops.bias_swiglu(x)

    def test_raises_on_bias_dim_mismatch(self):
        x = torch.randn(2, 16)
        bias = torch.randn(8)
        with pytest.raises(AssertionError, match="last dim must match"):
            swiglu_ops.bias_swiglu(x, bias)


class TestModuleWiring:
    def test_top_level_delegates(self, monkeypatch):
        called = {}

        def fake(input, bias=None, clamp_value=None):
            called["args"] = (input, bias, clamp_value)
            return "sentinel"

        monkeypatch.setattr(amd_tuned_torch.swiglu_ops, "bias_swiglu", fake)
        x, b = object(), object()
        result = amd_tuned_torch.bias_swiglu(x, b, clamp_value=3.0)
        assert result == "sentinel"
        assert called["args"] == (x, b, 3.0)
