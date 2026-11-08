"""Tests for the two autograd wrappers that close the last TRAINING-TIME
GAPs in _dispatch: _GroupNormFn and _Conv3dFn.

Both ops used to decline outright the moment any input required a
gradient, which during LoRA/LyCORIS training is essentially every call
(once one adapter needs a gradient, autograd has to backprop through every
frozen layer between it and the loss, so nearly every activation in the
forward pass requires_grad). group_norm mattered most -- SDXL's UNet and
VAE are almost entirely GroupNorm-normalised ResNet blocks -- and was the
op _dispatch's own comments called the one remaining dead end.

CPU-safe, like the rest of this suite: the numeric tests pin the new
backward math against autograd's own gradients through stock
F.group_norm/F.conv3d (real torch ops, no device needed), and the wiring
tests use the fake native kernel from conftest to check that a live
gradient now reaches the accelerated forward instead of falling back to
stock.

Run with:

    python -m pytest source/cmp_ext_turing/tests/test_dispatch_training_autograd.py
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import amd_tuned_torch
from amd_tuned_torch import _dispatch

# The ATen ops, NOT F.group_norm/F.conv3d: this package patches the F.*
# names, so a stand-in forward that called F.group_norm would be calling
# _patched_group_norm, which calls the stand-in, which... -- recursion,
# not a stock reference. torch.group_norm/torch.conv3d are never patched
# (_install only touches the F.* entries plus torch.matmul/torch.bmm), so
# they are the same stock implementation no matter what is installed.
_STOCK_GROUP_NORM = torch.group_norm
_STOCK_CONV3D = torch.conv3d


def _eligible_but_grad_live(monkeypatch):
    """Like conftest's force_eligible, except _grad_safe stays False --
    that is the whole point here: the dtype/device gates must pass so the
    tier is reachable, while a live gradient routes through the autograd
    wrapper rather than the direct call."""
    monkeypatch.setattr(_dispatch, "_usable", lambda *a, **k: True)
    monkeypatch.setattr(_dispatch, "_grad_safe", lambda *a, **k: False)


def _stock_group_norm_grads(x0, w0, b0, num_groups, dy):
    """Gradients autograd computes through stock F.group_norm."""
    x = x0.clone().requires_grad_(True)
    w = None if w0 is None else w0.clone().requires_grad_(True)
    b = None if b0 is None else b0.clone().requires_grad_(True)
    _STOCK_GROUP_NORM(x, num_groups, w, b, 1e-5).backward(dy)
    return x.grad, None if w is None else w.grad, None if b is None else b.grad


class TestGroupNormBackwardMath:
    """_group_norm_backward on its own, against autograd's own answer."""

    @pytest.mark.parametrize("shape,num_groups", [
        ((2, 8, 5, 5), 4),      # 4D NCHW, the SDXL case
        ((1, 320, 8, 8), 32),   # SDXL channel/group counts
        ((2, 8, 16), 4),        # 3D
        ((2, 8, 3, 4, 5), 2),   # 5D -- the (N, G, M, S) view holds for any rank
        ((2, 8), 4),            # 2D, no spatial dims at all
    ])
    def test_matches_autograd_through_stock(self, shape, num_groups):
        torch.manual_seed(0)
        C = shape[1]
        x0 = torch.randn(*shape, dtype=torch.float64)
        w0 = torch.randn(C, dtype=torch.float64)
        b0 = torch.randn(C, dtype=torch.float64)
        dy = torch.randn(*shape, dtype=torch.float64)

        got = _dispatch._group_norm_backward(dy, x0, num_groups, w0, 1e-5,
                                             (True, True, True))
        want = _stock_group_norm_grads(x0, w0, b0, num_groups, dy)
        for name, g, r in zip(("grad_input", "grad_weight", "grad_bias"), got, want):
            torch.testing.assert_close(g, r, msg=lambda m: f"{name}: {m}")

    def test_non_affine_has_no_weight_gradient(self):
        torch.manual_seed(0)
        x0 = torch.randn(2, 8, 4, 4, dtype=torch.float64)
        dy = torch.randn(2, 8, 4, 4, dtype=torch.float64)
        grad_input, grad_weight, grad_bias = _dispatch._group_norm_backward(
            dy, x0, 4, None, 1e-5, (True, True, True))
        assert grad_weight is None
        want_input, _, _ = _stock_group_norm_grads(x0, None, None, 4, dy)
        torch.testing.assert_close(grad_input, want_input)

    @pytest.mark.parametrize("needs", [
        (True, False, False), (False, True, False), (False, False, True),
        (False, False, False), (True, True, False),
    ])
    def test_needs_mask_returns_none_for_unwanted_gradients(self, needs):
        torch.manual_seed(0)
        x0 = torch.randn(2, 8, 4, 4, dtype=torch.float64)
        w0 = torch.randn(8, dtype=torch.float64)
        dy = torch.randn(2, 8, 4, 4, dtype=torch.float64)
        out = _dispatch._group_norm_backward(dy, x0, 4, w0, 1e-5, needs)
        for wanted, got in zip(needs, out):
            assert (got is not None) == wanted

    def test_fp16_statistics_survive_a_group_too_large_for_fp16_sums(self):
        """The fp32-accumulation contract. A group of 1x64x32x32 / 8 groups
        is 8192 elements; in fp16 the sum of that many unit-variance
        squares is already within an order of magnitude of fp16's 65504
        ceiling, and the SDXL shapes this runs on are 20x larger. Accumulate
        in fp16 and rstd comes back inf/garbage -- see the liger kernel's
        own bug 2. Here it must stay finite and close to stock."""
        torch.manual_seed(0)
        x0 = torch.randn(1, 64, 32, 32, dtype=torch.float16)
        w0 = torch.randn(64, dtype=torch.float16)
        dy = torch.randn(1, 64, 32, 32, dtype=torch.float16)
        grad_input, grad_weight, grad_bias = _dispatch._group_norm_backward(
            dy, x0, 8, w0, 1e-5, (True, True, True))
        for name, g in (("grad_input", grad_input), ("grad_weight", grad_weight),
                        ("grad_bias", grad_bias)):
            assert torch.isfinite(g).all(), f"{name} is not finite"
        want_input, want_w, want_b = _stock_group_norm_grads(x0, w0, w0, 8, dy)
        torch.testing.assert_close(grad_input.float(), want_input.float(), rtol=5e-2, atol=5e-2)
        torch.testing.assert_close(grad_weight.float(), want_w.float(), rtol=5e-2, atol=5e-2)


class TestPatchedGroupNormWithLiveGradient:
    def test_reaches_the_accelerated_forward_instead_of_stock(self, monkeypatch, native):
        _eligible_but_grad_live(monkeypatch)
        x = torch.randn(2, 4, 3, 3, requires_grad=True)
        w = torch.randn(4, requires_grad=True)
        b = torch.randn(4, requires_grad=True)
        native.group_norm.return_value = torch.zeros(2, 4, 3, 3)

        out = _dispatch._patched_group_norm(x, 2, w, b, 1e-5)

        native.group_norm.assert_called_once()
        assert out.requires_grad, "result must carry a backward, not be a detached kernel output"
        assert type(out.grad_fn).__name__.startswith("_GroupNormFn")

    def test_gradients_match_stock_group_norm_end_to_end(self, monkeypatch):
        """Through _patched_group_norm itself, with the accelerated forward
        standing in as stock (the fake native kernel returns a MagicMock,
        not numbers) -- so what is under test is the wrapper's backward."""
        _eligible_but_grad_live(monkeypatch)
        monkeypatch.setattr(
            _dispatch, "_group_norm_accelerated_forward",
            lambda input, num_groups, weight, bias, eps: _STOCK_GROUP_NORM(
                input, num_groups, weight, bias, eps))

        torch.manual_seed(0)
        x0 = torch.randn(2, 8, 5, 5, dtype=torch.float64)
        w0 = torch.randn(8, dtype=torch.float64)
        b0 = torch.randn(8, dtype=torch.float64)
        dy = torch.randn(2, 8, 5, 5, dtype=torch.float64)

        x = x0.clone().requires_grad_(True)
        w = w0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        out = _dispatch._patched_group_norm(x, 4, w, b, 1e-5)
        out.backward(dy)

        want = _stock_group_norm_grads(x0, w0, b0, 4, dy)
        for name, g, r in zip(("x.grad", "w.grad", "b.grad"), (x.grad, w.grad, b.grad), want):
            torch.testing.assert_close(g, r, msg=lambda m: f"{name}: {m}")

    def test_bias_gradient_keeps_its_own_dtype(self, monkeypatch):
        """An fp32 bias under fp16 activations is normal, and autograd
        requires each gradient in its own input's dtype."""
        _eligible_but_grad_live(monkeypatch)
        monkeypatch.setattr(
            _dispatch, "_group_norm_accelerated_forward",
            lambda input, num_groups, weight, bias, eps: _STOCK_GROUP_NORM(
                input, num_groups, weight.to(input.dtype), bias.to(input.dtype), eps))
        x = torch.randn(2, 4, 3, 3, dtype=torch.float16, requires_grad=True)
        w = torch.randn(4, dtype=torch.float32, requires_grad=True)
        b = torch.randn(4, dtype=torch.float32, requires_grad=True)
        _dispatch._patched_group_norm(x, 2, w, b, 1e-5).backward(
            torch.randn(2, 4, 3, 3, dtype=torch.float16))
        assert x.grad.dtype is torch.float16
        assert w.grad.dtype is torch.float32
        assert b.grad.dtype is torch.float32


class TestPatchedConv3dWithLiveGradient:
    def test_reaches_the_accelerated_forward_instead_of_stock(self, monkeypatch, native):
        _eligible_but_grad_live(monkeypatch)
        monkeypatch.setattr(amd_tuned_torch.kernel_select, "enabled", lambda: False)
        x = torch.randn(1, 2, 4, 4, 4, requires_grad=True)
        w = torch.randn(3, 2, 3, 3, 3, requires_grad=True)
        native.conv3d.return_value = torch.zeros(1, 3, 2, 2, 2)

        out = _dispatch._patched_conv3d(x, w)

        assert out.requires_grad
        assert type(out.grad_fn).__name__.startswith("_Conv3dFn")

    def test_grouped_conv_still_declines_to_stock(self, monkeypatch, native):
        """No tier here supports groups != 1, so there is nothing to wrap
        -- stock must handle forward AND backward, as before."""
        _eligible_but_grad_live(monkeypatch)
        x = torch.randn(1, 4, 4, 4, 4, requires_grad=True)
        w = torch.randn(4, 2, 3, 3, 3, requires_grad=True)

        out = _dispatch._patched_conv3d(x, w, None, 1, 0, 1, 2)

        native.conv3d.assert_not_called()
        torch.testing.assert_close(out, _STOCK_CONV3D(x, w, None, 1, 0, 1, 2))

    @pytest.mark.parametrize("stride,padding,dilation,bias", [
        (1, 0, 1, True),
        (2, 1, 1, True),
        (1, 1, 2, False),
        ((1, 2, 2), (1, 0, 1), 1, True),
    ])
    def test_gradients_match_stock_conv3d(self, monkeypatch, stride, padding, dilation, bias):
        _eligible_but_grad_live(monkeypatch)
        monkeypatch.setattr(
            _dispatch, "_conv3d_accelerated_forward",
            lambda i, w, b, s, p, d, g: _STOCK_CONV3D(i, w, b, s, p, d, g))

        torch.manual_seed(0)
        x0 = torch.randn(2, 3, 6, 6, 6, dtype=torch.float64)
        w0 = torch.randn(4, 3, 3, 3, 3, dtype=torch.float64)
        b0 = torch.randn(4, dtype=torch.float64) if bias else None

        outs = []
        for use_ours in (True, False):
            x = x0.clone().requires_grad_(True)
            w = w0.clone().requires_grad_(True)
            b = None if b0 is None else b0.clone().requires_grad_(True)
            fn = _dispatch._patched_conv3d if use_ours else _STOCK_CONV3D
            out = fn(x, w, b, stride, padding, dilation, 1)
            out.sum().backward()
            outs.append((out.detach(), x.grad, w.grad, None if b is None else b.grad))

        for name, g, r in zip(("out", "x.grad", "w.grad", "b.grad"), outs[0], outs[1]):
            if g is None:
                continue
            torch.testing.assert_close(g, r, msg=lambda m: f"{name}: {m}")

    def test_only_requested_gradients_are_computed(self, monkeypatch):
        """A frozen weight (requires_grad=False) must not get a gradient,
        and the output_mask must reflect that rather than computing one."""
        _eligible_but_grad_live(monkeypatch)
        monkeypatch.setattr(
            _dispatch, "_conv3d_accelerated_forward",
            lambda i, w, b, s, p, d, g: _STOCK_CONV3D(i, w, b, s, p, d, g))
        x = torch.randn(1, 2, 4, 4, 4, dtype=torch.float64, requires_grad=True)
        w = torch.randn(3, 2, 3, 3, 3, dtype=torch.float64)  # frozen
        _dispatch._patched_conv3d(x, w).sum().backward()
        assert x.grad is not None
        assert w.grad is None
