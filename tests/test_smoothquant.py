"""Tests for amd_tuned_torch's SmoothQuant calibration (amd_tuned_torch/aiter_ops.py).

Unlike the aiter-kernel-backed ops in test_amd_tuned_torch_monkeypatch.py, the
calibration/scale-math logic here is pure PyTorch and is exercised for
real, not mocked -- only the actual aiter GEMM/kernel calls inside
linear_int8 are stubbed, same as the rest of this suite (no GPU, aiter, or
TransformerEngine required).

Run with:

    python -m pytest source/cmp_ext_turing/tests
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

import amd_tuned_torch
import amd_tuned_torch.aiter_ops as aiter_ops


class TestWeakTensorKeyDict:
    """_WeakTensorKeyDict exists because plain weakref.WeakKeyDictionary
    keyed on raw tensors is unsafe: on a hash collision between two
    *distinct* multi-element tensors (not equal ones -- any two landing in
    the same bucket), CPython's weakref.ref equality falls back to
    `referent1 == referent2`, and torch.Tensor.__eq__ returns an elementwise
    tensor there, not a bool -- crashing with "RuntimeError: Boolean value
    of Tensor with more than one value is ambiguous". This is exactly what
    happened, unforced, once this project's own test suite created enough
    distinct weight tensors in one process to hit a collision (tensor
    hashes are identity-based, so collisions are rare but real). Forcing an
    actual collision deterministically in a test isn't reliable here --
    torch.Tensor's hash is a C-level tp_hash slot that a Python-level
    `Tensor.__hash__ = ...` monkeypatch doesn't reach -- so these tests
    verify correctness directly rather than trying to reproduce the
    original crash.
    """

    def test_distinct_tensors_get_distinct_values(self):
        d = aiter_ops._WeakTensorKeyDict()
        t1, t2 = torch.randn(4, 8), torch.randn(4, 8)
        d[t1] = "first"
        d[t2] = "second"
        assert d.get(t1) == "first"
        assert d.get(t2) == "second"
        assert d.get(torch.randn(4, 8)) is None

    def test_content_equal_tensors_are_still_distinct_keys(self):
        # Two tensors with identical values are still different objects --
        # a dict keyed by id() (this class) must not conflate them, unlike
        # a hypothetical value-based cache would.
        t1, t2 = torch.ones(4, 8), torch.ones(4, 8)
        d = aiter_ops._WeakTensorKeyDict()
        d[t1] = "first"
        assert d.get(t2) is None
        d[t2] = "second"
        assert d.get(t1) == "first"
        assert d.get(t2) == "second"

    def test_getitem_and_contains(self):
        d = aiter_ops._WeakTensorKeyDict()
        t = torch.randn(4, 8)
        assert t not in d
        d[t] = "value"
        assert t in d
        assert d[t] == "value"

    def test_getitem_raises_keyerror_when_missing(self):
        d = aiter_ops._WeakTensorKeyDict()
        with pytest.raises(KeyError):
            d[torch.randn(4, 8)]

    def test_pop_removes_entry(self):
        d = aiter_ops._WeakTensorKeyDict()
        t = torch.randn(4, 8)
        d[t] = "value"
        assert d.pop(t) == "value"
        assert d.get(t) is None
        assert d.pop(t, "default") == "default"

    def test_evicted_on_garbage_collection(self):
        d = aiter_ops._WeakTensorKeyDict()
        t = torch.randn(4, 8)
        d[t] = "value"
        assert t in d
        del t
        import gc

        gc.collect()
        assert len(d._data) == 0


class TestComputeSmoothquantScale:
    def test_none_without_calibration(self):
        weight = torch.randn(4, 8)
        assert aiter_ops.compute_smoothquant_scale(weight) is None

    def test_matches_formula_after_calibration(self):
        torch.manual_seed(0)
        weight = torch.randn(4, 8)
        x = torch.randn(16, 8)
        aiter_ops._update_calib_amax(weight, x)

        alpha = 0.5
        scale = aiter_ops.compute_smoothquant_scale(weight, alpha=alpha)
        assert scale is not None
        assert scale.shape == (8,)

        weight_amax = weight.abs().amax(dim=0).clamp(min=1e-8)
        act_amax = x.abs().amax(dim=0).clamp(min=1e-8)
        expected = (weight_amax.pow(1 - alpha) / act_amax.pow(alpha)).clamp(min=1e-4, max=1e4)
        assert torch.allclose(scale, expected, atol=1e-5)

    def test_preserves_matmul_result_before_quantization(self):
        # The whole point of SmoothQuant: x_smooth @ w_smooth^T == x @ w^T
        # exactly (up to fp precision), before any int8 rounding happens.
        torch.manual_seed(1)
        weight = torch.randn(4, 8, dtype=torch.float64)
        x = torch.randn(16, 8, dtype=torch.float64)
        aiter_ops._update_calib_amax(weight, x)
        scale = aiter_ops.compute_smoothquant_scale(weight).to(torch.float64)

        x_smooth = x * scale
        w_smooth = weight / scale
        assert torch.allclose(x_smooth @ w_smooth.t(), x @ weight.t(), atol=1e-10)

    def test_amax_accumulates_across_multiple_calibration_batches(self):
        weight = torch.randn(2, 4)
        aiter_ops._update_calib_amax(weight, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
        aiter_ops._update_calib_amax(weight, torch.tensor([[5.0, 1.0, 1.0, 1.0]]))
        amax = aiter_ops._calib_act_amax[weight]
        assert torch.equal(amax, torch.tensor([5.0, 2.0, 3.0, 4.0]))


class TestSetSmoothScale:
    def test_registers_scale(self):
        weight = torch.randn(4, 8)
        scale = torch.ones(8)
        aiter_ops.set_smooth_scale(weight, scale)
        assert aiter_ops._smooth_scale_cache[weight] is scale

    def test_evicts_stale_weight_cache_entry(self):
        # Otherwise linear_int8/_quantize_weight would keep serving a
        # quantization computed before smoothing was registered.
        weight = torch.randn(4, 8)
        aiter_ops._weight_cache[weight] = ("stale_q", "stale_scale")
        aiter_ops.set_smooth_scale(weight, torch.ones(8))
        assert weight not in aiter_ops._weight_cache


class TestQuantizeWeightWithSmoothing:
    def test_no_smoothing_when_no_scale_registered(self):
        weight = torch.randn(4, 8)
        q, scale = aiter_ops._quantize_weight(weight)
        expected_amax = weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        expected_row_scale = expected_amax / aiter_ops._INT8_MAX
        expected_q = (
            (weight.float() / expected_row_scale).round().clamp(-127, 127).to(torch.int8)
        )
        assert torch.equal(q, expected_q)
        assert torch.allclose(scale, expected_row_scale.t())

    def test_applies_smoothing_when_scale_registered(self):
        torch.manual_seed(2)
        weight = torch.randn(4, 8)
        smooth_scale = torch.rand(8) + 0.5  # keep scales away from zero
        aiter_ops.set_smooth_scale(weight, smooth_scale)

        q, scale = aiter_ops._quantize_weight(weight)

        w_smoothed = weight.float() / smooth_scale
        expected_amax = w_smoothed.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        expected_row_scale = expected_amax / aiter_ops._INT8_MAX
        expected_q = (
            (w_smoothed / expected_row_scale).round().clamp(-127, 127).to(torch.int8)
        )
        assert torch.equal(q, expected_q)
        assert torch.allclose(scale, expected_row_scale.t())

    def test_result_is_cached(self):
        weight = torch.randn(4, 8)
        first = aiter_ops._quantize_weight(weight)
        second = aiter_ops._quantize_weight(weight)
        assert first[0] is second[0]


class TestLinearInt8Smoothing:
    def test_uses_plain_quantize_activation_without_smoothing(self, monkeypatch):
        weight = torch.randn(4, 8)
        x = torch.randn(2, 8)
        fake_aiter = MagicMock()
        fake_aiter.gemm_a8w8.return_value = torch.zeros(2, 4)
        monkeypatch.setattr(aiter_ops, "aiter", fake_aiter)
        spy = MagicMock(wraps=aiter_ops._quantize_activation)
        monkeypatch.setattr(aiter_ops, "_quantize_activation", spy)

        aiter_ops.linear_int8(x, weight)
        spy.assert_called_once()

    def test_uses_smoothquant_kernel_when_calibrated(self, monkeypatch):
        weight = torch.randn(4, 8)
        x = torch.randn(2, 8)
        fake_aiter = MagicMock()
        fake_aiter.gemm_a8w8.return_value = torch.zeros(2, 4)
        monkeypatch.setattr(aiter_ops, "aiter", fake_aiter)
        aiter_ops.set_smooth_scale(weight, torch.ones(8))

        fake_smoothquant = MagicMock(
            return_value=(torch.zeros(2, 8, dtype=torch.int8), torch.ones(2))
        )
        monkeypatch.setattr(aiter_ops, "_smoothquant_quantize", fake_smoothquant)
        spy_plain = MagicMock(wraps=aiter_ops._quantize_activation)
        monkeypatch.setattr(aiter_ops, "_quantize_activation", spy_plain)

        aiter_ops.linear_int8(x, weight)

        fake_smoothquant.assert_called_once()
        spy_plain.assert_not_called()

    def test_falls_back_to_plain_quantize_when_kernel_unavailable(self, monkeypatch):
        # A smooth_scale is registered, but aiter's kernel isn't importable
        # in this environment (_smoothquant_quantize is None) -- must not
        # crash, just skip smoothing for the activation side.
        weight = torch.randn(4, 8)
        x = torch.randn(2, 8)
        fake_aiter = MagicMock()
        fake_aiter.gemm_a8w8.return_value = torch.zeros(2, 4)
        monkeypatch.setattr(aiter_ops, "aiter", fake_aiter)
        monkeypatch.setattr(aiter_ops, "_smoothquant_quantize", None)
        aiter_ops.set_smooth_scale(weight, torch.ones(8))

        spy_plain = MagicMock(wraps=aiter_ops._quantize_activation)
        monkeypatch.setattr(aiter_ops, "_quantize_activation", spy_plain)

        aiter_ops.linear_int8(x, weight)
        spy_plain.assert_called_once()


class TestCalibrateSmoothquant:
    """calibrate_smoothquant() patches torch.nn.Linear.forward directly --
    a module-level patch, not another functional one -- so it observes
    (weight, input) pairs regardless of how the surrounding model code
    invokes the module."""

    def test_patches_and_restores_linear_forward(self):
        orig_forward = torch.nn.Linear.forward
        linear = nn.Linear(4, 2)
        patched_during_call = {}

        def run():
            patched_during_call["is_patched"] = torch.nn.Linear.forward is not orig_forward
            linear(torch.randn(3, 4))

        amd_tuned_torch.calibrate_smoothquant(run)
        assert patched_during_call["is_patched"] is True
        assert torch.nn.Linear.forward is orig_forward

    def test_restores_forward_even_if_forward_fn_raises(self):
        orig_forward = torch.nn.Linear.forward

        def boom():
            raise RuntimeError("calibration batch failed")

        with pytest.raises(RuntimeError):
            amd_tuned_torch.calibrate_smoothquant(boom)
        assert torch.nn.Linear.forward is orig_forward

    def test_collects_amax_and_sets_scale_for_every_linear_seen(self):
        linear = nn.Linear(4, 2, bias=False)
        x = torch.randn(5, 4)

        amd_tuned_torch.calibrate_smoothquant(lambda: linear(x))

        scale = aiter_ops._smooth_scale_cache.get(linear.weight)
        assert scale is not None
        assert scale.shape == (4,)

    def test_does_not_touch_weights_never_seen(self):
        untouched = nn.Linear(4, 2)
        seen = nn.Linear(4, 2)
        amd_tuned_torch.calibrate_smoothquant(lambda: seen(torch.randn(2, 4)))
        assert untouched.weight not in aiter_ops._smooth_scale_cache

    def test_still_produces_correct_output_while_calibrating(self):
        # The patched forward must still return the real Linear output, not
        # just observe it silently.
        linear = nn.Linear(4, 2)
        x = torch.randn(3, 4)
        expected = linear(x)
        got = {}

        amd_tuned_torch.calibrate_smoothquant(lambda: got.__setitem__("out", linear(x)))

        assert torch.equal(got["out"], expected)
