"""Tests for amd_tuned_torch.liger_group_norm_ops -- CPU-safe, no Triton/GPU
required. available() is False without Triton+CUDA/ROCm (true in this dev
environment), so group_norm()/LigerGroupNorm always take their F.group_norm
fallback path here -- exercised for real (not mocked), including a real
backward pass, since that path has no hardware dependency at all. The
Triton kernel itself needs a real device to compile/launch and is not
exercised here -- see the module's own VALIDATION STATUS docstring section.

Run with:

    python -m pytest source/cmp_ext_turing/tests
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from amd_tuned_torch import liger_group_norm_ops as gn


class TestAvailable:
    def test_false_without_triton_or_gpu(self):
        # This dev environment has neither -- available() must not raise
        # either way, and must be a plain bool.
        assert gn.available() is False


class TestGroupNormFallback:
    def test_matches_f_group_norm_forward(self):
        torch.manual_seed(0)
        x = torch.randn(2, 8, 5, 5)
        weight = torch.randn(8)
        bias = torch.randn(8)
        out = gn.group_norm(x, 4, weight, bias, eps=1e-5)
        ref = F.group_norm(x, 4, weight, bias, eps=1e-5)
        assert torch.allclose(out, ref)

    def test_real_backward_matches_f_group_norm(self):
        torch.manual_seed(0)
        x = torch.randn(2, 8, 5, 5, requires_grad=True)
        weight = torch.randn(8, requires_grad=True)
        bias = torch.randn(8, requires_grad=True)

        out = gn.group_norm(x, 4, weight, bias)
        out.sum().backward()
        x_grad, w_grad, b_grad = x.grad.clone(), weight.grad.clone(), bias.grad.clone()

        x.grad = weight.grad = bias.grad = None
        ref = F.group_norm(x, 4, weight, bias)
        ref.sum().backward()

        assert torch.allclose(x_grad, x.grad)
        assert torch.allclose(w_grad, weight.grad)
        assert torch.allclose(b_grad, bias.grad)


class TestLigerGroupNormModule:
    def test_is_a_real_nn_group_norm_subclass(self):
        # State_dict compatibility with plain nn.GroupNorm depends on this.
        m = gn.LigerGroupNorm(4, 16)
        assert isinstance(m, nn.GroupNorm)

    def test_forward_matches_nn_group_norm(self):
        torch.manual_seed(0)
        x = torch.randn(2, 16, 4, 4)
        plain = nn.GroupNorm(4, 16)
        liger = gn.LigerGroupNorm(4, 16)
        liger.load_state_dict(plain.state_dict())
        assert torch.allclose(liger(x), plain(x))


class TestReplaceGroupNormModules:
    def _toy_model(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(8, 16, 3, padding=1)
                self.norm = nn.GroupNorm(4, 16)
                self.inner = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.GroupNorm(4, 16))
                self.non_affine_norm = nn.GroupNorm(4, 16, affine=False)

            def forward(self, x):
                x = self.norm(self.conv(x))
                x = self.inner(x)
                return x
        return Toy()

    def test_replaces_every_affine_group_norm_recursively(self):
        m = self._toy_model()
        n = gn.replace_group_norm_modules(m)
        assert n == 2  # self.norm and inner[1], NOT non_affine_norm
        assert isinstance(m.norm, gn.LigerGroupNorm)
        assert isinstance(m.inner[1], gn.LigerGroupNorm)

    def test_skips_non_affine_group_norm(self):
        m = self._toy_model()
        gn.replace_group_norm_modules(m)
        assert type(m.non_affine_norm) is nn.GroupNorm  # untouched, not replaced

    def test_reuses_same_parameter_objects(self):
        m = self._toy_model()
        original_weight, original_bias = m.norm.weight, m.norm.bias
        gn.replace_group_norm_modules(m)
        assert m.norm.weight is original_weight
        assert m.norm.bias is original_bias

    def test_forward_and_backward_unchanged_after_replacement(self):
        torch.manual_seed(0)
        m = self._toy_model()
        x = torch.randn(2, 8, 5, 5, requires_grad=True)

        ref_out = m(x)
        ref_out.sum().backward()
        ref_x_grad = x.grad.clone()
        ref_w_grad = m.norm.weight.grad.clone()
        x.grad = None
        m.norm.weight.grad = None
        m.norm.bias.grad = None

        gn.replace_group_norm_modules(m)
        out = m(x)
        out.sum().backward()

        assert torch.allclose(out, ref_out)
        assert torch.allclose(x.grad, ref_x_grad)
        assert torch.allclose(m.norm.weight.grad, ref_w_grad)

    def test_idempotent_does_not_double_wrap(self):
        m = self._toy_model()
        gn.replace_group_norm_modules(m)
        n_second_pass = gn.replace_group_norm_modules(m)
        assert n_second_pass == 0
