"""Tests for _Conv2dFn / _conv2d_backward_data / _conv2d_backward_weight --
the real conv2d backward this project gained once CK's WMMA backward-data/
backward-weight device ops existed to accelerate it (see _patched_conv2d's
TRAINING-TIME GAP comment in amd_tuned_torch/_dispatch.py). Before this,
`not _grad_safe(...)` always fell straight back to stock; now it enters a
kernel_select contest between CK's backward device ops and PyTorch's own
stock backward (torch.ops.aten.convolution_backward), same policy the
forward tier already uses.

No ROCm hardware, aiter, or built CK extension is required: amd_tuned_torch.
ck_ops.available() is False by default in this suite (conftest.py fakes
amd_tuned_torch._native_ck.has_ck() -> False), so
TestConv2dBackwardStockOnly below exercises the real stock path for real,
comparing gradients against plain (unpatched) F.conv2d's own autograd on
CPU tensors. TestConv2dBackwardKernelSelectContest mocks ck_ops.available()
and ck_ops.conv2d_backward_data/_weight (making them delegate to the SAME
stock helper internally, so their output is numerically correct rather
than a meaningless sentinel) and stubs kernel_select._time the way
test_amd_tuned_torch_monkeypatch.py's TestFlashAttnRocwmmaKernelSelectContest
already does for the same real-timing-needs-a-GPU reason, to check the
*selection policy* (does the faster candidate's gradient come back, is
"stock" still returned when it is the reference and CK ties/loses).

Run with:

    python -m pytest source/cmp_ext_turing/tests
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

import amd_tuned_torch
from conftest import force_eligible


def _random_conv2d_inputs(dtype=torch.float32, bias=True):
    x = torch.randn(2, 4, 8, 8, dtype=dtype, requires_grad=True)
    w = torch.randn(6, 4, 3, 3, dtype=dtype, requires_grad=True)
    b = torch.randn(6, dtype=dtype, requires_grad=True) if bias else None
    return x, w, b


class TestConv2dBackwardStockOnly:
    """ck_ops.available() is False here (no compiled CK extension in this
    dev environment) -- every gradient below is computed by the real
    stock-fallback path (_conv2d_backward_data_stock/_weight_stock via
    torch.ops.aten.convolution_backward), giving a genuine, not mocked,
    correctness check against plain F.conv2d's own autograd.

    _ORIGINALS[(F, "conv2d")] (read by _patched_conv2d) is only populated
    while amd_tuned_torch is "enabled" -- explicit enable()/disable() here
    rather than relying on some other test file leaving it installed keeps
    this file order-independent (a real, pre-existing fragility elsewhere
    in this suite -- see this project's own notes on a similar
    _ORIGINALS[(F, "linear")] KeyError-in-isolation quirk)."""

    def setup_method(self):
        amd_tuned_torch.enable()

    def teardown_method(self):
        amd_tuned_torch.disable()

    def test_matches_stock_autograd_input_weight_bias(self, monkeypatch):
        force_eligible(monkeypatch)
        x, w, b = _random_conv2d_inputs()
        x_ref, w_ref, b_ref = x.detach().clone().requires_grad_(), \
            w.detach().clone().requires_grad_(), b.detach().clone().requires_grad_()

        out = amd_tuned_torch._patched_conv2d(x, w, b, padding=1)
        out_ref = F.conv2d(x_ref, w_ref, b_ref, padding=1)
        assert torch.allclose(out, out_ref)

        out.sum().backward()
        out_ref.sum().backward()
        assert torch.allclose(x.grad, x_ref.grad)
        assert torch.allclose(w.grad, w_ref.grad)
        assert torch.allclose(b.grad, b_ref.grad)

    def test_matches_stock_autograd_no_bias(self, monkeypatch):
        force_eligible(monkeypatch)
        x, w, _ = _random_conv2d_inputs(bias=False)
        x_ref, w_ref = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()

        out = amd_tuned_torch._patched_conv2d(x, w, stride=2)
        out_ref = F.conv2d(x_ref, w_ref, stride=2)

        out.sum().backward()
        out_ref.sum().backward()
        assert torch.allclose(x.grad, x_ref.grad)
        assert torch.allclose(w.grad, w_ref.grad)

    def test_frozen_weight_only_computes_input_grad(self, monkeypatch):
        """A LoRA-style frozen conv layer: weight.requires_grad is False,
        only the input activation needs a gradient so autograd can keep
        backpropagating through it -- exactly the case the TRAINING-TIME
        GAP comment describes as "every conv2d call" during LoRA
        training."""
        force_eligible(monkeypatch)
        x = torch.randn(2, 4, 8, 8, requires_grad=True)
        w = torch.randn(6, 4, 3, 3, requires_grad=False)
        x_ref = x.detach().clone().requires_grad_()

        out = amd_tuned_torch._patched_conv2d(x, w, padding=1)
        out_ref = F.conv2d(x_ref, w, padding=1)
        out.sum().backward()
        out_ref.sum().backward()
        assert torch.allclose(x.grad, x_ref.grad)
        assert w.grad is None

    def test_groups_not_1_still_uses_stock_autograd(self, monkeypatch):
        """groups != 1 bypasses _Conv2dFn entirely (see _patched_conv2d) --
        confirm the plain stock path still produces correct gradients for
        that case too."""
        force_eligible(monkeypatch)
        x = torch.randn(2, 4, 8, 8, requires_grad=True)
        w = torch.randn(4, 1, 3, 3, requires_grad=True)
        x_ref, w_ref = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()

        out = amd_tuned_torch._patched_conv2d(x, w, groups=4)
        out_ref = F.conv2d(x_ref, w_ref, groups=4)
        out.sum().backward()
        out_ref.sum().backward()
        assert torch.allclose(x.grad, x_ref.grad)
        assert torch.allclose(w.grad, w_ref.grad)


class TestConv2dBackwardKernelSelectContest:
    """Unit-level tests of _conv2d_backward_data/_conv2d_backward_weight
    directly (not through the full _patched_conv2d/_Conv2dFn pipeline --
    that also runs the unrelated FORWARD contest, which would either need
    its own ck_ops.conv2d mock or interfere with a bare
    kernel_select._time stub meant only for the backward contest).

    Mirrors TestFlashAttnRocwmmaKernelSelectContest's pattern (see that
    class's docstring in test_amd_tuned_torch_monkeypatch.py): real timing
    needs a GPU, so kernel_select._time is stubbed to a fake that tags
    outputs by which candidate produced them, and correctness verification
    is disabled since it isn't what these tests are checking. Unlike that
    class's torch.zeros sentinel, the mocked "ck" candidates here delegate
    to the real stock helper internally, so their output IS numerically
    correct."""

    def setup_method(self):
        amd_tuned_torch.kernel_select._ENABLED = True
        amd_tuned_torch.kernel_select._VERIFY_ENABLED = False
        amd_tuned_torch.kernel_select.reset()

    def teardown_method(self):
        amd_tuned_torch.kernel_select.reset()
        amd_tuned_torch.kernel_select._VERIFY_ENABLED = True
        amd_tuned_torch.kernel_select._ENABLED = False

    def _mock_ck(self, monkeypatch):
        """ck_ops.available() -> True, and conv2d_backward_data/_weight
        delegate to the real stock helpers (so their output is correct),
        wrapped in a MagicMock so call counts/args can be asserted."""
        monkeypatch.setattr(amd_tuned_torch.ck_ops, "available", lambda: True)

        def fake_backward_data(grad_output, weight, input_size, stride, padding, dilation):
            return amd_tuned_torch._conv2d_backward_data_stock(
                grad_output, weight, input_size, stride, padding, dilation, 1)

        def fake_backward_weight(input, grad_output, weight_size, stride, padding, dilation):
            return amd_tuned_torch._conv2d_backward_weight_stock(
                input, grad_output, weight_size, stride, padding, dilation, 1)

        mock_data = MagicMock(side_effect=fake_backward_data)
        mock_weight = MagicMock(side_effect=fake_backward_weight)
        monkeypatch.setattr(amd_tuned_torch.ck_ops, "conv2d_backward_data", mock_data)
        monkeypatch.setattr(amd_tuned_torch.ck_ops, "conv2d_backward_weight", mock_weight)
        return mock_data, mock_weight

    def test_ck_wins_when_measured_faster(self, monkeypatch):
        force_eligible(monkeypatch)
        mock_data, mock_weight = self._mock_ck(monkeypatch)
        monkeypatch.setattr(amd_tuned_torch.kernel_select, "_time",
                             lambda fn: None if fn() is None else 1.0)

        x = torch.randn(2, 4, 8, 8, dtype=torch.float16)
        w = torch.randn(6, 4, 3, 3, dtype=torch.float16)
        grad_output = torch.randn(2, 6, 8, 8, dtype=torch.float16)

        grad_input = amd_tuned_torch._conv2d_backward_data(
            grad_output, w, tuple(x.shape), 1, 1, 1, 1)
        grad_weight = amd_tuned_torch._conv2d_backward_weight(
            x, grad_output, tuple(w.shape), 1, 1, 1, 1)

        mock_data.assert_called()
        mock_weight.assert_called()
        expected_input = amd_tuned_torch._conv2d_backward_data_stock(
            grad_output, w, tuple(x.shape), 1, 1, 1, 1)
        expected_weight = amd_tuned_torch._conv2d_backward_weight_stock(
            x, grad_output, tuple(w.shape), 1, 1, 1, 1)
        assert torch.equal(grad_input, expected_input)
        assert torch.equal(grad_weight, expected_weight)
        assert amd_tuned_torch.kernel_select.debug_winners()
        assert all(v == "ck" for v in amd_tuned_torch.kernel_select.debug_winners().values())

    def test_stock_wins_when_ck_measured_slower(self, monkeypatch):
        force_eligible(monkeypatch)
        mock_data, mock_weight = self._mock_ck(monkeypatch)
        # Candidates are timed in list order (ck first, stock second -- see
        # _conv2d_backward_data/_weight's candidate lists), so tag "ck
        # always measures slower" by call parity rather than introspecting
        # which thunk ran.
        calls = {"n": 0}

        def fake_time(fn):
            calls["n"] += 1
            out = fn()
            if out is None:
                return None
            return 2.0 if calls["n"] % 2 == 1 else 1.0

        monkeypatch.setattr(amd_tuned_torch.kernel_select, "_time", fake_time)

        x = torch.randn(2, 4, 8, 8, dtype=torch.float16)
        w = torch.randn(6, 4, 3, 3, dtype=torch.float16)
        grad_output = torch.randn(2, 6, 8, 8, dtype=torch.float16)

        grad_input = amd_tuned_torch._conv2d_backward_data(
            grad_output, w, tuple(x.shape), 1, 1, 1, 1)
        grad_weight = amd_tuned_torch._conv2d_backward_weight(
            x, grad_output, tuple(w.shape), 1, 1, 1, 1)

        expected_input = amd_tuned_torch._conv2d_backward_data_stock(
            grad_output, w, tuple(x.shape), 1, 1, 1, 1)
        expected_weight = amd_tuned_torch._conv2d_backward_weight_stock(
            x, grad_output, tuple(w.shape), 1, 1, 1, 1)
        assert torch.equal(grad_input, expected_input)
        assert torch.equal(grad_weight, expected_weight)
        # The losing "ck" candidate is still measured once (that's how a
        # contest decides), but must never be the one whose output wins.
        mock_data.assert_called_once()
        mock_weight.assert_called_once()
        assert all(v == "stock" for v in amd_tuned_torch.kernel_select.debug_winners().values())

    def test_ck_unavailable_never_enters_contest(self, monkeypatch):
        force_eligible(monkeypatch)
        monkeypatch.setattr(amd_tuned_torch.ck_ops, "available", lambda: False)
        monkeypatch.setattr(
            amd_tuned_torch.kernel_select, "_time",
            lambda fn: pytest.fail("contest must not run when ck_ops is unavailable"))

        x = torch.randn(2, 4, 8, 8, dtype=torch.float16)
        w = torch.randn(6, 4, 3, 3, dtype=torch.float16)
        grad_output = torch.randn(2, 6, 8, 8, dtype=torch.float16)

        grad_input = amd_tuned_torch._conv2d_backward_data(
            grad_output, w, tuple(x.shape), 1, 1, 1, 1)
        grad_weight = amd_tuned_torch._conv2d_backward_weight(
            x, grad_output, tuple(w.shape), 1, 1, 1, 1)
        assert grad_input is not None
        assert grad_weight is not None

    def test_groups_not_1_never_enters_contest(self, monkeypatch):
        """CK's forward/backward instances are groups=1 only (see
        ck_conv_fwd.hpp's ConvProblem.G comment) -- confirm the backward
        contest short-circuits to stock the same way rather than ever
        offering CK a grouped-conv problem it was never built to accept."""
        force_eligible(monkeypatch)
        self._mock_ck(monkeypatch)
        monkeypatch.setattr(
            amd_tuned_torch.kernel_select, "_time",
            lambda fn: pytest.fail("contest must not run for groups != 1"))

        x = torch.randn(2, 4, 8, 8, dtype=torch.float16)
        w = torch.randn(4, 1, 3, 3, dtype=torch.float16)
        grad_output = torch.randn(2, 4, 8, 8, dtype=torch.float16)

        grad_input = amd_tuned_torch._conv2d_backward_data(
            grad_output, w, tuple(x.shape), 1, 1, 1, 4)
        grad_weight = amd_tuned_torch._conv2d_backward_weight(
            x, grad_output, tuple(w.shape), 1, 1, 1, 4)
        assert grad_input is not None
        assert grad_weight is not None
