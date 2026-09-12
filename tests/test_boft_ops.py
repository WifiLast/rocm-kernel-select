"""CPU-safe tests for boft_ops -- FastBlockDiag/fast_block_diag are thin
autograd.Function wrappers around the native fast_block_diag_forward/
backward kernels (mocked here via the `native` fixture, same convention as
test_iu4_gemm_ops.py); the kernels' own numerical output is pure data
movement that needs a real device to actually launch, not exercised here.
patch_peft_boft()'s wiring is checked against a fake peft.tuners.boft.layer
module, so none of this needs peft actually installed."""
from __future__ import annotations

import sys
import types

import torch

from amd_tuned_torch import boft_ops


class TestAvailable:
    def test_true_when_native_exposes_both_ops(self, native):
        assert boft_ops.available() is True

    def test_false_when_extension_lacks_the_ops(self, monkeypatch):
        monkeypatch.setattr(boft_ops, "_C", types.SimpleNamespace())
        assert boft_ops.available() is False


class TestFastBlockDiag:
    def test_forward_calls_native_with_input_and_returns_its_result(self, native):
        input_ = torch.randn(2, 3, 4, 4)
        expected = torch.randn(2, 12, 12)
        native.fast_block_diag_forward.return_value = expected
        out = boft_ops.fast_block_diag(input_)
        native.fast_block_diag_forward.assert_called_once_with(input_)
        assert out is expected

    def test_backward_calls_native_with_grad_output_and_saved_input(self, native):
        input_ = torch.randn(2, 3, 4, 4, requires_grad=True)
        native.fast_block_diag_forward.return_value = torch.zeros(2, 12, 12)
        y = boft_ops.FastBlockDiag.apply(input_)

        grad_out = torch.randn(2, 12, 12)
        expected_grad = torch.randn(2, 3, 4, 4)
        native.fast_block_diag_backward.return_value = expected_grad
        y.backward(grad_out)

        native.fast_block_diag_backward.assert_called_once()
        called_args = native.fast_block_diag_backward.call_args[0]
        assert called_args[0] is grad_out
        assert called_args[1] is input_
        assert torch.equal(input_.grad, expected_grad)


def _install_fake_peft_boft_module(monkeypatch):
    """Registers a minimal fake peft.tuners.boft.layer module in sys.modules
    so `from peft.tuners.boft import layer` resolves without peft actually
    being installed. Returns the fake layer module for the caller to
    inspect after patch_peft_boft() runs."""
    fake_layer = types.ModuleType("peft.tuners.boft.layer")
    fake_layer._FBD_CUDA = None
    fake_layer.get_fbd_cuda = lambda: None  # stand-in for upstream's own JIT loader

    fake_boft_pkg = types.ModuleType("peft.tuners.boft")
    fake_boft_pkg.layer = fake_layer
    fake_tuners_pkg = types.ModuleType("peft.tuners")
    fake_tuners_pkg.boft = fake_boft_pkg
    fake_peft_pkg = types.ModuleType("peft")
    fake_peft_pkg.tuners = fake_tuners_pkg

    monkeypatch.setitem(sys.modules, "peft", fake_peft_pkg)
    monkeypatch.setitem(sys.modules, "peft.tuners", fake_tuners_pkg)
    monkeypatch.setitem(sys.modules, "peft.tuners.boft", fake_boft_pkg)
    monkeypatch.setitem(sys.modules, "peft.tuners.boft.layer", fake_layer)
    return fake_layer


class TestPatchPeftBoft:
    def test_returns_false_when_peft_not_importable(self, monkeypatch):
        # Forces ImportError regardless of whether a real `peft` happens to
        # be installed in whatever environment runs this test.
        monkeypatch.setitem(sys.modules, "peft", None)
        assert boft_ops.patch_peft_boft() is False

    def test_patches_get_fbd_cuda_to_this_packages_kernel(self, monkeypatch, native):
        fake_layer = _install_fake_peft_boft_module(monkeypatch)

        assert boft_ops.patch_peft_boft() is True

        shim = fake_layer.get_fbd_cuda()
        assert shim is fake_layer._FBD_CUDA

        input_ = torch.randn(1, 2, 4, 4)
        expected_fwd = torch.randn(1, 8, 8)
        native.fast_block_diag_forward.return_value = expected_fwd
        # Same call shape BOFTLayer's own FastBlockDiag.forward uses:
        # get_fbd_cuda().forward(input)[0].
        assert shim.forward(input_) == [expected_fwd]
        native.fast_block_diag_forward.assert_called_once_with(input_)

        grad_out = torch.randn(1, 8, 8)
        expected_bwd = torch.randn(1, 2, 4, 4)
        native.fast_block_diag_backward.return_value = expected_bwd
        assert shim.backward(grad_out, input_) == [expected_bwd]
        native.fast_block_diag_backward.assert_called_once_with(grad_out, input_)
