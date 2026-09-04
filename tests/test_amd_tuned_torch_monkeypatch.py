"""Tests for amd_tuned_torch's monkeypatch layer (source/cmp_ext_turing), ROCm build.

These test the *Python dispatch logic* in amd_tuned_torch/__init__.py -- eligibility
gating (grad-safety for the aiter/native-backed ops, dtype/device for all of
them, shape/argument constraints), correct wiring to aiter/native/
TransformerEngine, and fallback-to-stock on ineligibility or a
RuntimeError/TypeError from the underlying call. They do not require an
RX 7900 XTX, aiter, or TransformerEngine: amd_tuned_torch._native (group_norm) is
replaced with a MagicMock stub (see conftest.py), and amd_tuned_torch.aiter_ops /
amd_tuned_torch.te_ops are stubbed the same way via the `aiter` / `te` fixtures. The
CUDA/dtype gate is bypassed on demand via ``force_eligible`` so the branch
logic can be exercised with plain CPU tensors.

Run with:

    python -m pytest source/cmp_ext_turing/tests
"""
from __future__ import annotations

import importlib
import os
import sys
import types
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

import amd_tuned_torch
import amd_tuned_torch.te_ops as te_ops_module
from conftest import STOCK_UNPATCHED, force_eligible
from _helpers import assert_called_once_with_tensors


# ---------------------------------------------------------------------------
# TE disabled-by-default gate (amd_tuned_torch/te_ops.py) -- a broken/ABI-mismatched
# TE install can segfault the whole process on `import transformer_engine`,
# which no try/except can catch, so the import is gated behind
# AMD_TUNED_TORCH_ENABLE_TE and off unless explicitly set. These tests reload
# te_ops.py under each env-var state; the "enabled" case injects FAKE
# transformer_engine modules into sys.modules first so the test never risks
# touching a real (possibly broken) TE install.
# ---------------------------------------------------------------------------

class TestTeDisabledByDefault:
    def teardown_method(self):
        # Force TE off for this reload regardless of monkeypatch's own
        # cleanup timing relative to this xunit-style hook (unspecified,
        # and not worth depending on) -- never leave a reload free to
        # attempt a real `import transformer_engine` during teardown, which
        # is exactly the crash risk these tests exist to guard against.
        # Save/restore rather than delete, so monkeypatch's own finalizer
        # (whenever it runs) still sees and undoes whatever it set.
        prev = os.environ.get("AMD_TUNED_TORCH_ENABLE_TE")
        os.environ["AMD_TUNED_TORCH_ENABLE_TE"] = "0"
        for name in ("transformer_engine_torch", "transformer_engine.pytorch",
                     "transformer_engine.pytorch.constants", "transformer_engine"):
            sys.modules.pop(name, None)
        importlib.reload(te_ops_module)
        if prev is None:
            os.environ.pop("AMD_TUNED_TORCH_ENABLE_TE", None)
        else:
            os.environ["AMD_TUNED_TORCH_ENABLE_TE"] = prev

    def test_te_not_imported_without_env_var(self, monkeypatch):
        monkeypatch.delenv("AMD_TUNED_TORCH_ENABLE_TE", raising=False)
        importlib.reload(te_ops_module)
        assert te_ops_module.available() is False
        assert te_ops_module.tex is None
        assert "transformer_engine_torch" not in sys.modules

    def test_te_not_imported_when_explicitly_zero(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_ENABLE_TE", "0")
        importlib.reload(te_ops_module)
        assert te_ops_module.available() is False

    def test_te_import_attempted_when_enabled(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_ENABLE_TE", "1")
        fake_tex = types.ModuleType("transformer_engine_torch")
        fake_te_pkg = types.ModuleType("transformer_engine")
        fake_te_pytorch = types.ModuleType("transformer_engine.pytorch")
        fake_constants = types.ModuleType("transformer_engine.pytorch.constants")
        fake_constants.TE_DType = {}
        sys.modules["transformer_engine_torch"] = fake_tex
        sys.modules["transformer_engine"] = fake_te_pkg
        sys.modules["transformer_engine.pytorch"] = fake_te_pytorch
        sys.modules["transformer_engine.pytorch.constants"] = fake_constants

        importlib.reload(te_ops_module)

        assert te_ops_module.available() is True
        assert te_ops_module.tex is fake_tex


# ---------------------------------------------------------------------------
# _grad_safe / _usable helpers
# ---------------------------------------------------------------------------

class TestGradSafe:
    def test_true_when_grad_disabled(self):
        x = torch.randn(2, requires_grad=True)
        with torch.no_grad():
            assert amd_tuned_torch._grad_safe(x) is True

    def test_true_in_inference_mode(self):
        x = torch.randn(2, requires_grad=True)
        with torch.inference_mode():
            assert amd_tuned_torch._grad_safe(x) is True

    def test_false_when_any_tensor_requires_grad(self):
        x = torch.randn(2, requires_grad=True)
        y = torch.randn(2)
        assert torch.is_grad_enabled()
        assert amd_tuned_torch._grad_safe(y, x) is False

    def test_true_when_no_tensor_requires_grad(self):
        x = torch.randn(2)
        y = torch.randn(2)
        assert amd_tuned_torch._grad_safe(x, y) is True

    def test_non_tensor_args_are_ignored(self):
        assert amd_tuned_torch._grad_safe(1, "x", None) is True


class TestUsable:
    """Default dtype set (_GEMM_DTYPES) is fp16/bf16 only -- aiter's Triton
    WMMA GEMM kernels backing linear/matmul/bmm don't cover fp32. group_norm
    and every TE-backed op pass a wider `dtypes=` explicitly (see
    _GROUPNORM_DTYPES / _TE_DTYPES)."""

    def test_true_for_non_tensor_args(self):
        assert amd_tuned_torch._usable(1, "x", None) is True

    def test_false_for_cpu_tensor(self):
        x = torch.randn(2)
        assert x.is_cuda is False
        assert amd_tuned_torch._usable(x) is False

    def test_false_for_unsupported_dtype_on_fake_cuda_tensor(self, monkeypatch):
        x = torch.zeros(2, dtype=torch.int64)
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
        assert amd_tuned_torch._usable(x) is False

    def test_true_for_fp16_on_fake_cuda_tensor(self, monkeypatch):
        x = torch.zeros(2, dtype=torch.float16)
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
        assert amd_tuned_torch._usable(x) is True

    def test_true_for_bf16_on_fake_cuda_tensor(self, monkeypatch):
        # RDNA3's WMMA units have native bf16 matrix throughput (unlike
        # Turing), so bf16 goes straight to aiter's Triton GEMM, no fp32
        # conversion.
        x = torch.zeros(2, dtype=torch.bfloat16)
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
        assert amd_tuned_torch._usable(x) is True

    def test_false_for_fp32_by_default(self, monkeypatch):
        # aiter's Triton WMMA GEMM kernels are fp16/bf16 only -- fp32 falls
        # back to stock rocBLAS/hipBLASLt for linear/matmul/bmm.
        x = torch.zeros(2, dtype=torch.float32)
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
        assert amd_tuned_torch._usable(x) is False

    def test_true_for_fp32_with_groupnorm_dtypes(self, monkeypatch):
        x = torch.zeros(2, dtype=torch.float32)
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
        assert amd_tuned_torch._usable(x, dtypes=amd_tuned_torch._GROUPNORM_DTYPES) is True

    def test_true_for_fp32_with_te_dtypes(self, monkeypatch):
        x = torch.zeros(2, dtype=torch.float32)
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
        assert amd_tuned_torch._usable(x, dtypes=amd_tuned_torch._TE_DTYPES) is True


# ---------------------------------------------------------------------------
# _install / _restore / enable / disable / is_enabled
# ---------------------------------------------------------------------------

class TestEnableDisable:
    def test_enable_installs_aiter_gemm_when_available(self, aiter):
        assert not amd_tuned_torch.is_enabled()
        amd_tuned_torch.enable()
        try:
            assert F.linear is amd_tuned_torch._patched_linear
            assert torch.matmul is amd_tuned_torch._patched_matmul
            assert torch.bmm is amd_tuned_torch._patched_bmm
        finally:
            amd_tuned_torch.disable()

    def test_enable_skips_aiter_gemm_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(amd_tuned_torch.aiter_ops, "available", lambda: False)
        orig_linear = F.linear
        assert not amd_tuned_torch.is_enabled()
        amd_tuned_torch.enable()
        try:
            assert F.linear is orig_linear
        finally:
            amd_tuned_torch.disable()

    def test_enable_always_installs_native_kernels(self):
        # group_norm/conv2d/conv3d are the native HIP kernels
        # (src/main_rocm.cpp), independent of aiter/TE availability --
        # amd_tuned_torch._native is hard-required at import time (see the
        # ImportError guard above), so these are unconditional, unlike the
        # aiter-gated and TE-gated patches.
        assert not amd_tuned_torch.is_enabled()
        amd_tuned_torch.enable()
        try:
            assert F.group_norm is amd_tuned_torch._patched_group_norm
            assert F.conv2d is amd_tuned_torch._patched_conv2d
            if hasattr(F, "conv3d"):
                assert F.conv3d is amd_tuned_torch._patched_conv3d
        finally:
            amd_tuned_torch.disable()

    def test_enable_skips_te_backed_ops_when_te_unavailable(self, monkeypatch):
        monkeypatch.setattr(amd_tuned_torch.te_ops, "available", lambda: False)
        orig_rms_norm = F.rms_norm if hasattr(F, "rms_norm") else None
        assert not amd_tuned_torch.is_enabled()
        amd_tuned_torch.enable()
        try:
            if orig_rms_norm is not None:
                assert F.rms_norm is orig_rms_norm
        finally:
            amd_tuned_torch.disable()

    def test_enable_installs_te_backed_ops_when_available(self, te):
        assert not amd_tuned_torch.is_enabled()
        amd_tuned_torch.enable()
        try:
            assert F.gelu is amd_tuned_torch._patched_gelu
            assert F.silu is amd_tuned_torch._patched_silu
            assert F.scaled_dot_product_attention is amd_tuned_torch._patched_sdpa
        finally:
            amd_tuned_torch.disable()

    def test_enable_never_installs_layer_norm(self, aiter, te):
        # TE's layernorm_fwd benchmarked slower than stock on RX 7900 XTX
        # (see benchmark.json) -- enable() leaves it on stock regardless of
        # aiter/TE availability.
        orig_layer_norm = F.layer_norm
        assert not amd_tuned_torch.is_enabled()
        amd_tuned_torch.enable()
        try:
            assert F.layer_norm is orig_layer_norm
        finally:
            amd_tuned_torch.disable()

    def test_disable_restores_stock_functions(self, aiter):
        for (target, name), orig in STOCK_UNPATCHED.items():
            assert getattr(target, name) is orig
        amd_tuned_torch.enable()
        amd_tuned_torch.disable()
        for (target, name), orig in STOCK_UNPATCHED.items():
            assert getattr(target, name) is orig, f"{name} was not restored"
        assert F.linear is not amd_tuned_torch._patched_linear

    def test_enable_is_idempotent(self, aiter):
        amd_tuned_torch.enable()
        try:
            patched_linear = F.linear
            amd_tuned_torch.enable()
            assert F.linear is patched_linear
        finally:
            amd_tuned_torch.disable()

    def test_stock_unpatched_ops_never_touched(self, aiter):
        amd_tuned_torch.enable()
        try:
            for (target, name), orig in STOCK_UNPATCHED.items():
                assert getattr(target, name) is orig
        finally:
            amd_tuned_torch.disable()


# ---------------------------------------------------------------------------
# aiter-backed GEMM: linear, matmul, bmm
# ---------------------------------------------------------------------------

class TestPatchedLinear:
    def test_falls_back_when_ineligible(self, monkeypatch, aiter):
        monkeypatch.setattr(amd_tuned_torch, "_usable", lambda *a, **k: False)
        x = torch.randn(2, 4)
        w = torch.randn(3, 4)
        out = amd_tuned_torch._patched_linear(x, w, None)
        assert torch.equal(out, F.linear(x, w, None))
        aiter.linear_fp16.assert_not_called()

    def test_falls_back_for_weight_not_2d(self, monkeypatch, aiter):
        force_eligible(monkeypatch)
        x = torch.randn(2, 4)
        w = torch.randn(4)
        with pytest.raises(RuntimeError):
            # F.linear itself rejects a 1D weight; confirms we routed to stock.
            amd_tuned_torch._patched_linear(x, w, None)
        aiter.linear_fp16.assert_not_called()

    def test_calls_aiter_when_eligible(self, monkeypatch, aiter):
        force_eligible(monkeypatch)
        x = torch.randn(2, 4)
        w = torch.randn(3, 4)
        aiter.linear_fp16.return_value = torch.zeros(2, 3)
        out = amd_tuned_torch._patched_linear(x, w, None)
        assert out is aiter.linear_fp16.return_value
        assert_called_once_with_tensors(aiter.linear_fp16, x, w, None)

    def test_falls_back_on_aiter_runtime_error(self, monkeypatch, aiter):
        force_eligible(monkeypatch)
        aiter.linear_fp16.side_effect = RuntimeError("no Triton config for this shape")
        x = torch.randn(2, 4)
        w = torch.randn(3, 4)
        out = amd_tuned_torch._patched_linear(x, w, None)
        assert torch.equal(out, F.linear(x, w, None))


class TestPatchedMatmul:
    def test_calls_aiter_bmm_for_2d_inputs(self, monkeypatch, aiter):
        force_eligible(monkeypatch)
        x = torch.randn(2, 4)
        y = torch.randn(4, 3)
        aiter.bmm_fp16.return_value = torch.zeros(1, 2, 3)
        out = amd_tuned_torch._patched_matmul(x, y)
        assert torch.equal(out, torch.zeros(2, 3))
        assert aiter.bmm_fp16.call_count == 1

    def test_calls_aiter_bmm_for_3d_inputs(self, monkeypatch, aiter):
        force_eligible(monkeypatch)
        x = torch.randn(5, 2, 4)
        y = torch.randn(5, 4, 3)
        aiter.bmm_fp16.return_value = torch.zeros(5, 2, 3)
        out = amd_tuned_torch._patched_matmul(x, y)
        assert out is aiter.bmm_fp16.return_value

    def test_falls_back_for_out_kwarg(self, monkeypatch, aiter):
        force_eligible(monkeypatch)
        x = torch.randn(2, 4)
        y = torch.randn(4, 3)
        out_tensor = torch.empty(2, 3)
        amd_tuned_torch._patched_matmul(x, y, out=out_tensor)
        aiter.bmm_fp16.assert_not_called()

    def test_falls_back_for_mismatched_dtype(self, monkeypatch, aiter):
        force_eligible(monkeypatch)
        x = torch.randn(2, 4, dtype=torch.float32)
        y = torch.randn(4, 3, dtype=torch.float32).half()
        out = amd_tuned_torch._patched_matmul(x, y.float())
        aiter.bmm_fp16.assert_not_called()

    def test_calls_aiter_bmm_for_4d_inputs_by_flattening_batch_dims(self, monkeypatch, aiter):
        # The shape attention actually uses: (batch, heads, seq, head_dim).
        # Previously fell through to stock -- only dim()==2/3 were handled.
        force_eligible(monkeypatch)
        x = torch.randn(2, 3, 5, 4)
        y = torch.randn(2, 3, 4, 6)
        aiter.bmm_fp16.return_value = torch.arange(2 * 3 * 5 * 6, dtype=torch.float32).view(6, 5, 6)
        out = amd_tuned_torch._patched_matmul(x, y)
        call_args = aiter.bmm_fp16.call_args[0]
        assert call_args[0].shape == (6, 5, 4)
        assert call_args[1].shape == (6, 4, 6)
        assert out.shape == (2, 3, 5, 6)
        assert torch.equal(out, aiter.bmm_fp16.return_value.view(2, 3, 5, 6))

    def test_falls_back_for_5d_inputs_by_flattening_batch_dims(self, monkeypatch, aiter):
        # Not just 4D -- any rank >= 3 with matching batch shape flattens.
        force_eligible(monkeypatch)
        x = torch.randn(2, 3, 1, 5, 4)
        y = torch.randn(2, 3, 1, 4, 6)
        aiter.bmm_fp16.return_value = torch.zeros(6, 5, 6)
        out = amd_tuned_torch._patched_matmul(x, y)
        assert out.shape == (2, 3, 1, 5, 6)

    def test_falls_back_for_4d_inputs_needing_broadcast(self, monkeypatch, aiter):
        # Batch shapes differ (3 vs 1 heads) -- broadcastable by torch.matmul,
        # but aiter's batched_gemm_bf16 has no broadcasting of its own, so
        # this must not take the fast path.
        force_eligible(monkeypatch)
        x = torch.randn(2, 3, 5, 4)
        y = torch.randn(1, 3, 4, 6)
        out = amd_tuned_torch._patched_matmul(x, y)
        aiter.bmm_fp16.assert_not_called()
        assert torch.allclose(out, torch.matmul(x, y))


class TestPatchedBmm:
    def test_calls_aiter_when_eligible(self, monkeypatch, aiter):
        force_eligible(monkeypatch)
        x = torch.randn(5, 2, 4)
        y = torch.randn(5, 4, 3)
        aiter.bmm_fp16.return_value = torch.zeros(5, 2, 3)
        out = amd_tuned_torch._patched_bmm(x, y)
        assert out is aiter.bmm_fp16.return_value

    def test_falls_back_on_aiter_runtime_error(self, monkeypatch, aiter):
        force_eligible(monkeypatch)
        aiter.bmm_fp16.side_effect = RuntimeError("no Triton config for this shape")
        x = torch.randn(5, 2, 4)
        y = torch.randn(5, 4, 3)
        out = amd_tuned_torch._patched_bmm(x, y)
        assert torch.equal(out, torch.bmm(x, y))


class TestPatchedGroupNorm:
    def test_falls_back_when_not_contiguous(self, monkeypatch, native):
        force_eligible(monkeypatch)
        x = torch.randn(2, 4, 3, 3).transpose(1, 2)
        assert not x.is_contiguous()
        w = torch.randn(4)
        b = torch.randn(4)
        amd_tuned_torch._patched_group_norm(x, 2, w, b, 1e-5)
        native.group_norm.assert_not_called()

    def test_calls_native_when_eligible(self, monkeypatch, native):
        force_eligible(monkeypatch)
        x = torch.randn(2, 4, 3, 3)
        w = torch.randn(4)
        b = torch.randn(4)
        native.group_norm.return_value = torch.zeros_like(x)
        out = amd_tuned_torch._patched_group_norm(x, 2, w, b, 1e-5)
        assert out is native.group_norm.return_value
        assert_called_once_with_tensors(native.group_norm, x, 2, w, b, 1e-5)


class TestPatchedConv2d:
    """Three tiers: native HIP kernel (fp16/fp32) -> aiter Triton conv2d
    (fp16/bf16, the bf16 case the native kernel doesn't cover) -> stock."""

    def test_falls_back_when_not_contiguous(self, monkeypatch, native, aiter):
        force_eligible(monkeypatch)
        x = torch.randn(2, 4, 8, 8).transpose(1, 2)
        assert not x.is_contiguous()
        w = torch.randn(4, 2, 3, 3)
        amd_tuned_torch._patched_conv2d(x, w)
        native.conv2d.assert_not_called()

    def test_falls_back_for_groups_not_1(self, monkeypatch, native, aiter):
        force_eligible(monkeypatch)
        x = torch.randn(2, 4, 8, 8)
        w = torch.randn(4, 1, 3, 3)
        amd_tuned_torch._patched_conv2d(x, w, groups=4)
        native.conv2d.assert_not_called()
        aiter.conv2d_fp16.assert_not_called()

    def test_pointwise_1x1_skips_both_tiers(self, monkeypatch, native, aiter):
        """1x1/stride1/pad0/dilation1 conv2d ("pointwise") is a pure
        channel-mixing GEMM -- miopen_amd_log.txt shows MIOpen's own
        rocBLAS GEMM solver winning that case by 3.4x over its best
        Winograd kernel, so it's routed straight to stock instead of
        through either of this module's tiers (see _is_pointwise_conv2d)."""
        force_eligible(monkeypatch)
        x = torch.randn(2, 4, 8, 8)
        w = torch.randn(4, 4, 1, 1)
        out = amd_tuned_torch._patched_conv2d(x, w)
        native.conv2d.assert_not_called()
        aiter.conv2d_fp16.assert_not_called()
        assert torch.equal(out, F.conv2d(x, w))

    def test_1x1_with_nondefault_stride_still_uses_native(self, monkeypatch, native, aiter):
        """Only a *pure* pointwise conv (stride1/pad0/dilation1 as well as
        1x1 kernel) skips the tiers -- a strided 1x1 conv still downsamples
        spatially, so it isn't the zero-spatial-reduction GEMM case
        miopen_amd_log.txt measured, and stays on the native kernel."""
        force_eligible(monkeypatch)
        x = torch.randn(2, 4, 8, 8)
        w = torch.randn(4, 4, 1, 1)
        native.conv2d.return_value = torch.zeros(2, 4, 4, 4)
        out = amd_tuned_torch._patched_conv2d(x, w, stride=2)
        assert out is native.conv2d.return_value
        native.conv2d.assert_called_once()

    def test_calls_native_when_eligible(self, monkeypatch, native, aiter):
        force_eligible(monkeypatch)
        x = torch.randn(2, 4, 8, 8)
        w = torch.randn(4, 4, 3, 3)
        native.conv2d.return_value = torch.zeros(2, 4, 8, 8)
        out = amd_tuned_torch._patched_conv2d(x, w, padding=1)
        assert out is native.conv2d.return_value
        native.conv2d.assert_called_once()
        aiter.conv2d_fp16.assert_not_called()

    def test_falls_back_to_aiter_on_native_runtime_error(self, monkeypatch, native, aiter):
        force_eligible(monkeypatch)
        native.conv2d.side_effect = RuntimeError("unsupported shape")
        x = torch.randn(2, 4, 8, 8)
        w = torch.randn(4, 4, 3, 3)
        aiter.conv2d_fp16.return_value = torch.zeros(2, 4, 8, 8)
        out = amd_tuned_torch._patched_conv2d(x, w, padding=1)
        assert out is aiter.conv2d_fp16.return_value

    def test_falls_back_to_stock_when_aiter_unavailable(self, monkeypatch, native):
        force_eligible(monkeypatch)
        monkeypatch.setattr(amd_tuned_torch.aiter_ops, "available", lambda: False)
        native.conv2d.side_effect = RuntimeError("unsupported shape")
        x = torch.randn(2, 4, 8, 8)
        w = torch.randn(4, 4, 3, 3)
        out = amd_tuned_torch._patched_conv2d(x, w, padding=1)
        assert torch.equal(out, F.conv2d(x, w, padding=1))

    def test_falls_back_to_stock_on_aiter_assertion_error(self, monkeypatch, native, aiter):
        force_eligible(monkeypatch)
        native.conv2d.side_effect = RuntimeError("unsupported shape")
        aiter.conv2d_fp16.side_effect = AssertionError("groups != 1")
        x = torch.randn(2, 4, 8, 8)
        w = torch.randn(4, 4, 3, 3)
        out = amd_tuned_torch._patched_conv2d(x, w, padding=1)
        assert torch.equal(out, F.conv2d(x, w, padding=1))


class TestPatchedConv3d:
    """Native HIP kernel only (fp16/fp32) -> stock -- neither aiter nor TE
    cover conv3d at all, so there's no second tier."""

    def test_falls_back_when_not_contiguous(self, monkeypatch, native):
        force_eligible(monkeypatch)
        x = torch.randn(2, 4, 3, 8, 8).transpose(1, 2)
        assert not x.is_contiguous()
        w = torch.randn(4, 4, 3, 3, 3)
        amd_tuned_torch._patched_conv3d(x, w)
        native.conv3d.assert_not_called()

    def test_falls_back_for_groups_not_1(self, monkeypatch, native):
        force_eligible(monkeypatch)
        x = torch.randn(2, 4, 3, 8, 8)
        w = torch.randn(4, 1, 3, 3, 3)
        amd_tuned_torch._patched_conv3d(x, w, groups=4)
        native.conv3d.assert_not_called()

    def test_calls_native_when_eligible(self, monkeypatch, native):
        force_eligible(monkeypatch)
        x = torch.randn(2, 4, 3, 8, 8)
        w = torch.randn(4, 4, 3, 3, 3)
        native.conv3d.return_value = torch.zeros(2, 4, 3, 8, 8)
        out = amd_tuned_torch._patched_conv3d(x, w, padding=1)
        assert out is native.conv3d.return_value
        native.conv3d.assert_called_once()

    def test_falls_back_to_stock_on_native_runtime_error(self, monkeypatch, native):
        force_eligible(monkeypatch)
        native.conv3d.side_effect = RuntimeError("unsupported shape")
        x = torch.randn(2, 4, 3, 8, 8)
        w = torch.randn(4, 4, 3, 3, 3)
        out = amd_tuned_torch._patched_conv3d(x, w, padding=1)
        assert torch.equal(out, F.conv3d(x, w, padding=1))


# ---------------------------------------------------------------------------
# Conv3d fp16 Winograd -- opt-in only (amd_tuned_torch.enable_conv3d_winograd_fp16),
# never bundled into enable(). _is_winograd_eligible_conv3d mirrors
# src/cuda/templates/conv3d_fp16_winograd.cu.tmpl's launcher guard exactly;
# this is the one piece of that kernel's logic testable without real
# hardware (the kernel itself is numerically unvalidated -- see that
# template's header -- these tests only cover the Python-side dispatch).
# ---------------------------------------------------------------------------

class TestIsWinogradEligibleConv3d:
    def _shape(self, B=1, C_in=8, D_in=8, H_in=8, W_in=8, C_out=8, K=3):
        x = torch.randn(B, C_in, D_in, H_in, W_in)
        w = torch.randn(C_out, C_in, K, K, K)
        return x, w

    def test_eligible_shape_passes(self):
        x, w = self._shape()
        assert amd_tuned_torch._is_winograd_eligible_conv3d(x, w, 1, 1, 1)

    def test_batch_greater_than_1_ineligible(self):
        x, w = self._shape(B=2)
        assert not amd_tuned_torch._is_winograd_eligible_conv3d(x, w, 1, 1, 1)

    def test_non_3x3x3_kernel_ineligible(self):
        x, w = self._shape(K=5)
        assert not amd_tuned_torch._is_winograd_eligible_conv3d(x, w, 1, 1, 1)

    def test_stride_not_1_ineligible(self):
        x, w = self._shape()
        assert not amd_tuned_torch._is_winograd_eligible_conv3d(x, w, 2, 1, 1)

    def test_padding_not_1_ineligible(self):
        x, w = self._shape()
        assert not amd_tuned_torch._is_winograd_eligible_conv3d(x, w, 1, 0, 1)

    def test_dilation_not_1_ineligible(self):
        x, w = self._shape()
        assert not amd_tuned_torch._is_winograd_eligible_conv3d(x, w, 1, 1, 2)

    def test_odd_spatial_dim_ineligible(self):
        x, w = self._shape(D_in=9)
        assert not amd_tuned_torch._is_winograd_eligible_conv3d(x, w, 1, 1, 1)

    def test_tile_count_not_divisible_by_8_ineligible(self):
        # D_in=H_in=W_in=6 (even, so passes that check) -> D_out=6 (stride1/
        # pad1/k3 keeps spatial size) -> tiles = (6/2)^3 = 27, not
        # divisible by 8. Contrast with the default 8x8x8 shape (64 tiles).
        x, w = self._shape(D_in=6, H_in=6, W_in=6)
        assert not amd_tuned_torch._is_winograd_eligible_conv3d(x, w, 1, 1, 1)

    def test_cout_not_divisible_by_8_ineligible(self):
        x, w = self._shape(C_out=5)
        assert not amd_tuned_torch._is_winograd_eligible_conv3d(x, w, 1, 1, 1)


class TestConv3dWinogradFp16Dispatch:
    """enable_conv3d_winograd_fp16 -- off by default, composes with
    enable()/disable() regardless of call order (captures whatever
    F.conv3d currently is as its own fallback, same mechanism as
    enable_int8_linear -- see its docstring)."""

    def teardown_method(self):
        # Never leave F.conv3d patched across tests -- other test classes
        # (e.g. TestPatchedConv3d, which calls amd_tuned_torch._patched_conv3d
        # directly and doesn't go through F.conv3d, but other suites might)
        # assume it's whatever enable()/disable() left it as.
        amd_tuned_torch.disable_conv3d_winograd_fp16()

    def test_disabled_by_default(self):
        assert not amd_tuned_torch._CONV3D_WINOGRAD_FP16_ENABLED

    def test_falls_back_when_ineligible_shape(self, monkeypatch, native):
        force_eligible(monkeypatch)
        amd_tuned_torch.enable_conv3d_winograd_fp16()
        x = torch.randn(2, 4, 8, 8, 8)  # batch=2 -> ineligible
        w = torch.randn(8, 4, 3, 3, 3)
        native.conv3d_fp16_winograd_bt8_bc8.return_value = torch.zeros(2, 8, 8, 8, 8)
        F.conv3d(x, w, padding=1)
        native.conv3d_fp16_winograd_bt8_bc8.assert_not_called()

    def test_calls_winograd_when_eligible(self, monkeypatch, native):
        force_eligible(monkeypatch)
        amd_tuned_torch.enable_conv3d_winograd_fp16()
        x = torch.randn(1, 8, 8, 8, 8)
        w = torch.randn(8, 8, 3, 3, 3)
        native.conv3d_fp16_winograd_bt8_bc8.return_value = torch.zeros(1, 8, 8, 8, 8)
        out = F.conv3d(x, w, padding=1)
        assert out is native.conv3d_fp16_winograd_bt8_bc8.return_value
        native.conv3d_fp16_winograd_bt8_bc8.assert_called_once()

    def test_falls_back_when_kernel_declines(self, monkeypatch, native):
        force_eligible(monkeypatch)
        x = torch.randn(1, 8, 8, 8, 8)
        w = torch.randn(8, 8, 3, 3, 3)
        expected = F.conv3d(x, w, padding=1)  # stock, captured before patching
        amd_tuned_torch.enable_conv3d_winograd_fp16()
        native.conv3d_fp16_winograd_bt8_bc8.return_value = None  # kernel says "out of scope"
        out = F.conv3d(x, w, padding=1)
        assert torch.equal(out, expected)
        native.conv3d_fp16_winograd_bt8_bc8.assert_called_once()

    def test_falls_back_to_stock_on_winograd_runtime_error(self, monkeypatch, native):
        force_eligible(monkeypatch)
        x = torch.randn(1, 8, 8, 8, 8)
        w = torch.randn(8, 8, 3, 3, 3)
        expected = F.conv3d(x, w, padding=1)
        amd_tuned_torch.enable_conv3d_winograd_fp16()
        native.conv3d_fp16_winograd_bt8_bc8.side_effect = RuntimeError("unsupported shape")
        out = F.conv3d(x, w, padding=1)
        assert torch.equal(out, expected)

    def test_disable_restores_prior_f_conv3d(self, monkeypatch, native):
        force_eligible(monkeypatch)
        prior = F.conv3d
        amd_tuned_torch.enable_conv3d_winograd_fp16()
        assert F.conv3d is not prior
        amd_tuned_torch.disable_conv3d_winograd_fp16()
        assert F.conv3d is prior

    def test_composes_on_top_of_main_patched_conv3d(self, monkeypatch, native):
        """enable() first, then enable_conv3d_winograd_fp16() -- an
        ineligible/declined shape must fall through to _patched_conv3d
        (the main tier), not straight to stock, per the docstring's
        call-order-independence claim.

        Cleanup must undo these LIFO (winograd first, then the outer
        enable()) -- doing it the other way round actually crashes a
        LATER, unrelated test (test_compile_ops.py's fullgraph-compile
        test): the outer disable() blindly restores F.conv3d from
        _ORIGINALS and clears that bookkeeping, leaving
        _conv3d_winograd_fp16_fallback pointing at the now-orphaned
        _patched_conv3d; disable_conv3d_winograd_fp16() (in this class's
        teardown_method) would then re-install that orphaned reference,
        and _patched_conv3d's own `_ORIGINALS[(F, "conv3d")]` lookup
        raises KeyError the next time anything calls F.conv3d. Confirmed
        by actually triggering it before adding the try/finally below --
        see amd_tuned_torch.enable_conv3d_winograd_fp16's docstring for
        the general version of this caveat."""
        force_eligible(monkeypatch)
        was_enabled = amd_tuned_torch.is_enabled()
        if not was_enabled:
            amd_tuned_torch.enable()
        try:
            amd_tuned_torch.enable_conv3d_winograd_fp16()
            try:
                x = torch.randn(2, 4, 8, 8, 8)  # batch=2 -> Winograd-ineligible
                w = torch.randn(4, 4, 3, 3, 3)
                native.conv3d.return_value = torch.zeros(2, 4, 8, 8, 8)
                out = F.conv3d(x, w, padding=1)
                assert out is native.conv3d.return_value
                native.conv3d_fp16_winograd_bt8_bc8.assert_not_called()
                native.conv3d.assert_called_once()
            finally:
                amd_tuned_torch.disable_conv3d_winograd_fp16()
        finally:
            if not was_enabled:
                amd_tuned_torch.disable()


class TestIsFlashAttnRocwmmaEligible:
    def _qk(self, B=1, H=4, Nq=8, Nk=8, D=32, dtype=torch.float16):
        q = torch.randn(B, H, Nq, D, dtype=dtype)
        k = torch.randn(B, H, Nk, D, dtype=dtype)
        return q, k

    def test_eligible_shape_passes(self, monkeypatch):
        force_eligible(monkeypatch)
        q, k = self._qk()
        assert amd_tuned_torch._is_flash_attn_rocwmma_eligible(q, k, None, 0.0, False)

    def test_attn_mask_given_ineligible(self, monkeypatch):
        force_eligible(monkeypatch)
        q, k = self._qk()
        mask = torch.ones(1, 1, 8, 8, dtype=torch.bool)
        assert not amd_tuned_torch._is_flash_attn_rocwmma_eligible(q, k, mask, 0.0, False)

    def test_dropout_ineligible(self, monkeypatch):
        force_eligible(monkeypatch)
        q, k = self._qk()
        assert not amd_tuned_torch._is_flash_attn_rocwmma_eligible(q, k, None, 0.1, False)

    def test_non_4d_ineligible(self, monkeypatch):
        force_eligible(monkeypatch)
        q = torch.randn(4, 8, 32)
        k = torch.randn(4, 8, 32)
        assert not amd_tuned_torch._is_flash_attn_rocwmma_eligible(q, k, None, 0.0, False)

    def test_causal_with_mismatched_seqlen_ineligible(self, monkeypatch):
        """The vendored kernel's causal masking is plain top-left (no
        q_len/kv_len offset parameter anywhere in its host.cpp signature)
        -- only identical to bottom-right causal when q_len == kv_len, so
        is_causal must decline otherwise rather than compute the wrong
        mask silently."""
        force_eligible(monkeypatch)
        q, k = self._qk(Nq=4, Nk=8)
        assert not amd_tuned_torch._is_flash_attn_rocwmma_eligible(q, k, None, 0.0, True)

    def test_causal_with_matched_seqlen_eligible(self, monkeypatch):
        force_eligible(monkeypatch)
        q, k = self._qk(Nq=8, Nk=8)
        assert amd_tuned_torch._is_flash_attn_rocwmma_eligible(q, k, None, 0.0, True)

    def test_non_causal_with_mismatched_seqlen_still_eligible(self, monkeypatch):
        # non-causal doesn't care about seqlen matching at all -- only
        # is_causal triggers the top-left-vs-bottom-right ambiguity.
        force_eligible(monkeypatch)
        q, k = self._qk(Nq=4, Nk=8)
        assert amd_tuned_torch._is_flash_attn_rocwmma_eligible(q, k, None, 0.0, False)


class TestFlashAttnRocwmmaDispatch:
    """enable_flash_attn_rocwmma -- off by default, composes with
    enable()/disable() regardless of call order (see its docstring, and
    TestConv3dWinogradFp16Dispatch's identical LIFO-ordering lesson --
    same _ORIGINALS-bookkeeping hazard applies here)."""

    def teardown_method(self):
        amd_tuned_torch.disable_flash_attn_rocwmma()

    def test_disabled_by_default(self):
        assert not amd_tuned_torch._FLASH_ATTN_ROCWMMA_ENABLED

    def test_noop_with_warning_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(amd_tuned_torch.flash_attn_rocwmma_ops, "available", lambda: False)
        monkeypatch.setattr(
            amd_tuned_torch.flash_attn_rocwmma_ops, "load_error",
            lambda: RuntimeError("no rocwmma headers"),
        )
        prior = F.scaled_dot_product_attention
        with pytest.warns(UserWarning):
            amd_tuned_torch.enable_flash_attn_rocwmma()
        assert F.scaled_dot_product_attention is prior
        assert not amd_tuned_torch._FLASH_ATTN_ROCWMMA_ENABLED

    def test_falls_back_when_ineligible_shape(self, monkeypatch):
        force_eligible(monkeypatch)
        monkeypatch.setattr(amd_tuned_torch.flash_attn_rocwmma_ops, "available", lambda: True)
        mocked_sdpa = MagicMock(return_value=torch.zeros(1, 4, 8, 32))
        monkeypatch.setattr(
            amd_tuned_torch.flash_attn_rocwmma_ops, "scaled_dot_product_attention", mocked_sdpa
        )
        amd_tuned_torch.enable_flash_attn_rocwmma()
        q = torch.randn(1, 4, 8, 32)
        k = torch.randn(1, 4, 8, 32)
        v = torch.randn(1, 4, 8, 32)
        mask = torch.ones(1, 1, 8, 8, dtype=torch.bool)  # explicit mask -> always ineligible
        F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        mocked_sdpa.assert_not_called()

    def test_calls_flash_attn_when_eligible(self, monkeypatch):
        force_eligible(monkeypatch)
        monkeypatch.setattr(amd_tuned_torch.flash_attn_rocwmma_ops, "available", lambda: True)
        expected = torch.zeros(1, 4, 8, 32)
        mocked_sdpa = MagicMock(return_value=expected)
        monkeypatch.setattr(
            amd_tuned_torch.flash_attn_rocwmma_ops, "scaled_dot_product_attention", mocked_sdpa
        )
        amd_tuned_torch.enable_flash_attn_rocwmma()
        q = torch.randn(1, 4, 8, 32)
        k = torch.randn(1, 4, 8, 32)
        v = torch.randn(1, 4, 8, 32)
        out = F.scaled_dot_product_attention(q, k, v)
        assert out is expected
        mocked_sdpa.assert_called_once()

    def test_falls_back_to_stock_on_runtime_error(self, monkeypatch):
        force_eligible(monkeypatch)
        monkeypatch.setattr(amd_tuned_torch.flash_attn_rocwmma_ops, "available", lambda: True)
        q = torch.randn(1, 4, 8, 32)
        k = torch.randn(1, 4, 8, 32)
        v = torch.randn(1, 4, 8, 32)
        expected = F.scaled_dot_product_attention(q, k, v)  # stock, captured before patching
        monkeypatch.setattr(
            amd_tuned_torch.flash_attn_rocwmma_ops, "scaled_dot_product_attention",
            MagicMock(side_effect=RuntimeError("unsupported shape")),
        )
        amd_tuned_torch.enable_flash_attn_rocwmma()
        out = F.scaled_dot_product_attention(q, k, v)
        assert torch.equal(out, expected)

    def test_disable_restores_prior_f_sdpa(self, monkeypatch):
        monkeypatch.setattr(amd_tuned_torch.flash_attn_rocwmma_ops, "available", lambda: True)
        prior = F.scaled_dot_product_attention
        amd_tuned_torch.enable_flash_attn_rocwmma()
        assert F.scaled_dot_product_attention is not prior
        amd_tuned_torch.disable_flash_attn_rocwmma()
        assert F.scaled_dot_product_attention is prior

    def test_composes_on_top_of_main_patched_sdpa(self, monkeypatch, te):
        """enable() first, then enable_flash_attn_rocwmma() -- a shape our
        tier declines (causal with mismatched seqlen) must fall through
        to _patched_sdpa (the TE tier), not straight to stock, per the
        docstring's call-order-independence claim. Cleanup undoes these
        LIFO, same lesson as TestConv3dWinogradFp16Dispatch."""
        force_eligible(monkeypatch)
        monkeypatch.setattr(amd_tuned_torch.flash_attn_rocwmma_ops, "available", lambda: True)
        was_enabled = amd_tuned_torch.is_enabled()
        if not was_enabled:
            amd_tuned_torch.enable()
        try:
            amd_tuned_torch.enable_flash_attn_rocwmma()
            try:
                q = torch.randn(1, 4, 8, 32)
                k = torch.randn(1, 4, 16, 32)  # mismatched seqlen -> our tier declines when causal
                v = torch.randn(1, 4, 16, 32)
                te.scaled_dot_product_attention.return_value = torch.zeros(1, 4, 8, 32)
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                assert out is te.scaled_dot_product_attention.return_value
                te.scaled_dot_product_attention.assert_called_once()
            finally:
                amd_tuned_torch.disable_flash_attn_rocwmma()
        finally:
            if not was_enabled:
                amd_tuned_torch.disable()


class TestIsTritonKernelsRmsnormEligible:
    def test_eligible_shape_passes(self, monkeypatch):
        force_eligible(monkeypatch)
        x = torch.randn(2, 4, 8)
        w = torch.randn(8)
        assert amd_tuned_torch._is_triton_kernels_rmsnorm_eligible(x, w)

    def test_missing_weight_ineligible(self, monkeypatch):
        force_eligible(monkeypatch)
        x = torch.randn(2, 4, 8)
        assert not amd_tuned_torch._is_triton_kernels_rmsnorm_eligible(x, None)

    def test_mismatched_hidden_dim_ineligible(self, monkeypatch):
        force_eligible(monkeypatch)
        x = torch.randn(2, 4, 8)
        w = torch.randn(16)
        assert not amd_tuned_torch._is_triton_kernels_rmsnorm_eligible(x, w)

    def test_autograd_live_ineligible(self):
        # No force_eligible here -- _grad_safe must run for real against a
        # requires_grad tensor (triton_kernels' rmsnorm has no backward pass).
        x = torch.randn(2, 4, 8, requires_grad=True)
        w = torch.randn(8)
        assert not amd_tuned_torch._is_triton_kernels_rmsnorm_eligible(x, w)


class TestTritonKernelsRmsnormDispatch:
    """enable_triton_kernels_rmsnorm -- off by default, composes with
    enable()/disable() regardless of call order, same LIFO-disable caveat
    as TestFlashAttnRocwmmaDispatch/TestConv3dWinogradFp16Dispatch."""

    def teardown_method(self):
        amd_tuned_torch.disable_triton_kernels_rmsnorm()

    def test_disabled_by_default(self):
        assert not amd_tuned_torch._TRITON_KERNELS_RMSNORM_ENABLED

    def test_noop_with_warning_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(amd_tuned_torch.triton_kernels_ops, "available", lambda: False)
        prior = F.rms_norm
        with pytest.warns(UserWarning):
            amd_tuned_torch.enable_triton_kernels_rmsnorm()
        assert F.rms_norm is prior
        assert not amd_tuned_torch._TRITON_KERNELS_RMSNORM_ENABLED

    def test_falls_back_when_ineligible(self, monkeypatch):
        force_eligible(monkeypatch)
        monkeypatch.setattr(amd_tuned_torch.triton_kernels_ops, "available", lambda: True)
        mocked_rms_norm = MagicMock(return_value=torch.zeros(2, 4, 8))
        monkeypatch.setattr(amd_tuned_torch.triton_kernels_ops, "rms_norm", mocked_rms_norm)
        amd_tuned_torch.enable_triton_kernels_rmsnorm()
        x = torch.randn(2, 4, 8)
        F.rms_norm(x, [8], None)  # no weight -> always ineligible
        mocked_rms_norm.assert_not_called()

    def test_calls_triton_kernels_when_eligible(self, monkeypatch):
        force_eligible(monkeypatch)
        monkeypatch.setattr(amd_tuned_torch.triton_kernels_ops, "available", lambda: True)
        expected = torch.zeros(2, 4, 8)
        mocked_rms_norm = MagicMock(return_value=expected)
        monkeypatch.setattr(amd_tuned_torch.triton_kernels_ops, "rms_norm", mocked_rms_norm)
        amd_tuned_torch.enable_triton_kernels_rmsnorm()
        x = torch.randn(2, 4, 8)
        w = torch.randn(8)
        out = F.rms_norm(x, [8], w)
        assert out is expected
        mocked_rms_norm.assert_called_once()

    def test_falls_back_to_stock_on_runtime_error(self, monkeypatch):
        force_eligible(monkeypatch)
        monkeypatch.setattr(amd_tuned_torch.triton_kernels_ops, "available", lambda: True)
        x = torch.randn(2, 4, 8)
        w = torch.randn(8)
        expected = F.rms_norm(x, [8], w)  # stock, captured before patching
        monkeypatch.setattr(
            amd_tuned_torch.triton_kernels_ops, "rms_norm",
            MagicMock(side_effect=RuntimeError("unsupported shape")),
        )
        amd_tuned_torch.enable_triton_kernels_rmsnorm()
        out = F.rms_norm(x, [8], w)
        assert torch.equal(out, expected)

    def test_disable_restores_prior_f_rms_norm(self, monkeypatch):
        monkeypatch.setattr(amd_tuned_torch.triton_kernels_ops, "available", lambda: True)
        prior = F.rms_norm
        amd_tuned_torch.enable_triton_kernels_rmsnorm()
        assert F.rms_norm is not prior
        amd_tuned_torch.disable_triton_kernels_rmsnorm()
        assert F.rms_norm is prior

    def test_composes_on_top_of_main_patched_rms_norm(self, monkeypatch, te):
        """enable() first, then enable_triton_kernels_rmsnorm() -- a call
        our tier declines (autograd live -- triton_kernels' rmsnorm has no
        backward pass) but TE's real torch.autograd.Function-backed
        rms_norm does not decline for that reason, so it must fall through
        to _patched_rms_norm (the TE tier), not straight to stock. Cleanup
        undoes these LIFO, same lesson as TestConv3dWinogradFp16Dispatch."""
        monkeypatch.setattr(amd_tuned_torch, "_usable", lambda *a, **k: True)
        monkeypatch.setattr(amd_tuned_torch.triton_kernels_ops, "available", lambda: True)
        was_enabled = amd_tuned_torch.is_enabled()
        if not was_enabled:
            amd_tuned_torch.enable()
        try:
            amd_tuned_torch.enable_triton_kernels_rmsnorm()
            try:
                x = torch.randn(2, 4, 8, requires_grad=True)
                w = torch.randn(8)
                te.rms_norm.return_value = torch.zeros(2, 4, 8)
                out = F.rms_norm(x, [8], w)  # autograd live -> our tier declines
                assert out is te.rms_norm.return_value
                te.rms_norm.assert_called_once()
            finally:
                amd_tuned_torch.disable_triton_kernels_rmsnorm()
        finally:
            if not was_enabled:
                amd_tuned_torch.disable()


class TestFlashAttnRocwmmaKernelSelectContest:
    """_patched_sdpa_flash_attn_rocwmma routes eligible calls through the
    same kernel_select contest linear/bmm/conv2d/conv3d/group_norm already
    use, instead of always preferring the vendored kernel unconditionally
    whenever eligible. tests/conftest.py forces
    AMD_TUNED_TORCH_MEASURE_KERNELS=0 for the rest of this suite (see
    test_kernel_select.py's own module docstring for why: the contest
    would call every candidate several times and break call-count
    assertions elsewhere) -- this class opts back in explicitly, the same
    way test_kernel_select.py's `_clean` fixture does, and stubs
    kernel_select._time the same way test_kernel_select.py's `timings`
    fixture does (real timing needs a GPU and asserts a benchmark, not a
    behaviour). Correctness verification (see kernel_select's own
    CORRECTNESS VERIFICATION docstring section) is disabled here too, same
    reason and same pattern as test_kernel_select.py's `_clean` fixture:
    `flash_out` below is a bare torch.zeros(...) sentinel, not a real
    attention output, so it would never numerically match the real stock
    fallback -- these tests are about SELECTION policy (does the faster
    candidate win, is the decision cached), not about verification, which
    test_kernel_select.py::TestCorrectnessVerification covers directly."""

    def setup_method(self):
        amd_tuned_torch.kernel_select._ENABLED = True
        amd_tuned_torch.kernel_select._VERIFY_ENABLED = False
        amd_tuned_torch.kernel_select.reset()

    def teardown_method(self):
        amd_tuned_torch.disable_flash_attn_rocwmma()
        amd_tuned_torch.kernel_select.reset()
        amd_tuned_torch.kernel_select._VERIFY_ENABLED = True
        amd_tuned_torch.kernel_select._ENABLED = False

    def _qkv(self, dtype=torch.float16):
        q = torch.randn(1, 4, 8, 32, dtype=dtype)
        k = torch.randn(1, 4, 8, 32, dtype=dtype)
        v = torch.randn(1, 4, 8, 32, dtype=dtype)
        return q, k, v

    def test_flash_wins_when_measured_faster(self, monkeypatch):
        force_eligible(monkeypatch)
        monkeypatch.setattr(amd_tuned_torch.flash_attn_rocwmma_ops, "available", lambda: True)
        flash_out = torch.zeros(1, 4, 8, 32)
        mocked_flash = MagicMock(return_value=flash_out)
        monkeypatch.setattr(amd_tuned_torch.flash_attn_rocwmma_ops,
                             "scaled_dot_product_attention", mocked_flash)

        def fake_time(fn):
            out = fn()
            return None if out is None else (1.0 if out is flash_out else 2.0)

        monkeypatch.setattr(amd_tuned_torch.kernel_select, "_time", fake_time)

        amd_tuned_torch.enable_flash_attn_rocwmma()
        q, k, v = self._qkv()
        out = F.scaled_dot_product_attention(q, k, v)
        assert out is flash_out
        mocked_flash.assert_called()
        assert amd_tuned_torch.kernel_select.debug_winners()

    def test_fallback_wins_when_measured_faster(self, monkeypatch):
        force_eligible(monkeypatch)
        monkeypatch.setattr(amd_tuned_torch.flash_attn_rocwmma_ops, "available", lambda: True)
        q, k, v = self._qkv()
        expected = F.scaled_dot_product_attention(q, k, v)  # stock, captured before patching

        flash_out = torch.zeros(1, 4, 8, 32)
        mocked_flash = MagicMock(return_value=flash_out)
        monkeypatch.setattr(amd_tuned_torch.flash_attn_rocwmma_ops,
                             "scaled_dot_product_attention", mocked_flash)

        def fake_time(fn):
            out = fn()
            return None if out is None else (2.0 if out is flash_out else 1.0)

        monkeypatch.setattr(amd_tuned_torch.kernel_select, "_time", fake_time)

        amd_tuned_torch.enable_flash_attn_rocwmma()
        out = F.scaled_dot_product_attention(q, k, v)
        assert torch.equal(out, expected)
        # The losing candidate is still measured once (that's how a contest
        # decides), but its output must never be the one returned.
        mocked_flash.assert_called()
        assert out is not flash_out

    def test_decision_is_cached_and_not_re_timed(self, monkeypatch):
        force_eligible(monkeypatch)
        monkeypatch.setattr(amd_tuned_torch.flash_attn_rocwmma_ops, "available", lambda: True)
        flash_out = torch.zeros(1, 4, 8, 32)
        mocked_flash = MagicMock(return_value=flash_out)
        monkeypatch.setattr(amd_tuned_torch.flash_attn_rocwmma_ops,
                             "scaled_dot_product_attention", mocked_flash)

        time_calls = []

        def fake_time(fn):
            out = fn()
            time_calls.append(1)
            return None if out is None else (1.0 if out is flash_out else 2.0)

        monkeypatch.setattr(amd_tuned_torch.kernel_select, "_time", fake_time)

        amd_tuned_torch.enable_flash_attn_rocwmma()
        q, k, v = self._qkv()
        F.scaled_dot_product_attention(q, k, v)
        assert len(time_calls) > 0
        first_round = len(time_calls)

        out2 = F.scaled_dot_product_attention(q, k, v)
        assert out2 is flash_out
        assert len(time_calls) == first_round  # no re-measurement on the cached path

    def test_ineligible_call_skips_the_contest_entirely(self, monkeypatch):
        force_eligible(monkeypatch)
        monkeypatch.setattr(amd_tuned_torch.flash_attn_rocwmma_ops, "available", lambda: True)
        mocked_flash = MagicMock(return_value=torch.zeros(1, 4, 8, 32))
        monkeypatch.setattr(amd_tuned_torch.flash_attn_rocwmma_ops,
                             "scaled_dot_product_attention", mocked_flash)
        monkeypatch.setattr(amd_tuned_torch.kernel_select, "_time",
                             lambda fn: pytest.fail("contest must not run for an ineligible call"))

        amd_tuned_torch.enable_flash_attn_rocwmma()
        q = torch.randn(1, 4, 8, 32)
        k = torch.randn(1, 4, 8, 32)
        v = torch.randn(1, 4, 8, 32)
        mask = torch.ones(1, 1, 8, 8, dtype=torch.bool)  # explicit mask -> always ineligible
        F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        mocked_flash.assert_not_called()

    def test_kernel_select_disabled_restores_old_unconditional_preference(self, monkeypatch):
        # AMD_TUNED_TORCH_MEASURE_KERNELS=0 (or kernel_select._ENABLED
        # False, as here) must fall back to the pre-contest behaviour:
        # always prefer flash_rocwmma when eligible, no timing at all.
        amd_tuned_torch.kernel_select._ENABLED = False
        force_eligible(monkeypatch)
        monkeypatch.setattr(amd_tuned_torch.flash_attn_rocwmma_ops, "available", lambda: True)
        flash_out = torch.zeros(1, 4, 8, 32)
        mocked_flash = MagicMock(return_value=flash_out)
        monkeypatch.setattr(amd_tuned_torch.flash_attn_rocwmma_ops,
                             "scaled_dot_product_attention", mocked_flash)
        monkeypatch.setattr(amd_tuned_torch.kernel_select, "_time",
                             lambda fn: pytest.fail("contest must not run when kernel_select is disabled"))

        amd_tuned_torch.enable_flash_attn_rocwmma()
        q, k, v = self._qkv()
        out = F.scaled_dot_product_attention(q, k, v)
        assert out is flash_out


# ---------------------------------------------------------------------------
# TransformerEngine-backed ops: layer_norm, rms_norm, gelu, silu, sdpa
# ---------------------------------------------------------------------------

class TestPatchedLayerNorm:
    def test_falls_back_without_weight_or_bias(self, te):
        x = torch.randn(2, 4)
        out = amd_tuned_torch._patched_layer_norm(x, [4], None, None, 1e-5)
        assert torch.equal(out, F.layer_norm(x, [4], None, None, 1e-5))
        te.layer_norm.assert_not_called()

    def test_calls_te_when_eligible(self, monkeypatch, te):
        monkeypatch.setattr(amd_tuned_torch, "_usable", lambda *a, **k: True)
        x = torch.randn(2, 4)
        w = torch.randn(4)
        b = torch.randn(4)
        te.layer_norm.return_value = torch.zeros(2, 4)
        out = amd_tuned_torch._patched_layer_norm(x, [4], w, b, 1e-5)
        assert out is te.layer_norm.return_value
        assert_called_once_with_tensors(te.layer_norm, x, [4], w, b, 1e-5)

    def test_falls_back_on_te_runtime_error(self, monkeypatch, te):
        monkeypatch.setattr(amd_tuned_torch, "_usable", lambda *a, **k: True)
        te.layer_norm.side_effect = RuntimeError("TE not built for this shape")
        x = torch.randn(2, 4)
        w = torch.randn(4)
        b = torch.randn(4)
        out = amd_tuned_torch._patched_layer_norm(x, [4], w, b, 1e-5)
        assert torch.equal(out, F.layer_norm(x, [4], w, b, 1e-5))


class TestPatchedRmsNorm:
    def test_falls_back_without_weight(self, te):
        if not hasattr(F, "rms_norm"):
            pytest.skip("this torch build has no F.rms_norm")
        x = torch.randn(2, 4)
        te.rms_norm.assert_not_called()
        amd_tuned_torch._patched_rms_norm(x, [4], None, None)
        te.rms_norm.assert_not_called()

    def test_calls_te_when_eligible(self, monkeypatch, te):
        monkeypatch.setattr(amd_tuned_torch, "_usable", lambda *a, **k: True)
        x = torch.randn(2, 4)
        w = torch.randn(4)
        te.rms_norm.return_value = torch.zeros(2, 4)
        out = amd_tuned_torch._patched_rms_norm(x, [4], w, 1e-5)
        assert out is te.rms_norm.return_value
        assert_called_once_with_tensors(te.rms_norm, x, [4], w, 1e-5)


class TestPatchedGelu:
    def test_falls_back_for_exact_gelu(self, monkeypatch, te):
        # tex.gelu is the tanh approximation only; patching approximate=
        # "none" (the default) would silently change numerics.
        monkeypatch.setattr(amd_tuned_torch, "_usable", lambda *a, **k: True)
        x = torch.randn(4)
        out = amd_tuned_torch._patched_gelu(x, approximate="none")
        assert torch.equal(out, F.gelu(x, approximate="none"))
        te.gelu.assert_not_called()

    def test_calls_te_for_tanh_approximate(self, monkeypatch, te):
        monkeypatch.setattr(amd_tuned_torch, "_usable", lambda *a, **k: True)
        x = torch.randn(4)
        te.gelu.return_value = torch.zeros(4)
        out = amd_tuned_torch._patched_gelu(x, approximate="tanh")
        assert out is te.gelu.return_value
        te.gelu.assert_called_once()


class TestPatchedSilu:
    def test_falls_back_for_inplace(self, monkeypatch, te):
        monkeypatch.setattr(amd_tuned_torch, "_usable", lambda *a, **k: True)
        x = torch.randn(4)
        amd_tuned_torch._patched_silu(x.clone(), inplace=True)
        te.silu.assert_not_called()

    def test_calls_te_when_eligible(self, monkeypatch, te):
        monkeypatch.setattr(amd_tuned_torch, "_usable", lambda *a, **k: True)
        x = torch.randn(4)
        te.silu.return_value = torch.zeros(4)
        out = amd_tuned_torch._patched_silu(x)
        assert out is te.silu.return_value


class TestIsBottomRightCausalMask:
    """amd_tuned_torch.te_ops.is_bottom_right_causal_mask -- pure tensor logic, no TE
    or hardware needed. Used by _patched_sdpa to decide whether an explicit
    attn_mask tensor (as opposed to is_causal=True) can still reach TE's
    fused "causal_bottom_right" path."""

    def test_true_for_square_causal(self):
        mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))
        assert te_ops_module.is_bottom_right_causal_mask(mask) is True

    def test_true_for_kv_cache_shaped_causal(self):
        # q_len=2, kv_len=5 -- a KV-cache decode step, where the query's
        # true position is offset by (kv_len - q_len).
        kv_len, q_len = 5, 2
        query_position = kv_len - q_len + torch.arange(q_len)[:, None]
        key_position = torch.arange(kv_len)[None, :]
        mask = key_position <= query_position
        assert te_ops_module.is_bottom_right_causal_mask(mask) is True

    def test_true_when_broadcast_over_batch_and_heads(self):
        mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))
        mask = mask[None, None, :, :].expand(2, 3, 4, 4)
        assert te_ops_module.is_bottom_right_causal_mask(mask) is True

    def test_false_for_non_boolean_mask(self):
        # An additive float mask (e.g. 0/-inf), not a boolean attend mask.
        mask = torch.zeros(4, 4)
        assert te_ops_module.is_bottom_right_causal_mask(mask) is False

    def test_false_for_all_false_mask(self):
        mask = torch.zeros(4, 4, dtype=torch.bool)
        assert te_ops_module.is_bottom_right_causal_mask(mask) is False

    def test_false_for_top_left_causal_when_kv_longer(self):
        # Top-left-aligned causal (no KV-cache offset) with kv_len > q_len is
        # NOT the bottom-right pattern -- must not match.
        kv_len, q_len = 5, 2
        query_position = torch.arange(q_len)[:, None]
        key_position = torch.arange(kv_len)[None, :]
        mask = key_position <= query_position
        assert te_ops_module.is_bottom_right_causal_mask(mask) is False

    def test_false_for_1d_tensor(self):
        assert te_ops_module.is_bottom_right_causal_mask(torch.ones(4, dtype=torch.bool)) is False

    def test_false_for_near_miss_mask(self):
        # Off by a single element -- exact match is required, not "mostly
        # causal" (e.g. a genuinely arbitrary block-diffusion mask that
        # happens to look causal almost everywhere must not be misrouted).
        mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))
        mask[0, 3] = True
        assert te_ops_module.is_bottom_right_causal_mask(mask) is False


class TestPatchedSdpa:
    def test_falls_back_with_attn_mask(self, monkeypatch, te):
        monkeypatch.setattr(amd_tuned_torch, "_usable", lambda *a, **k: True)
        q = k = v = torch.randn(1, 2, 3, 4)
        mask = torch.zeros(3, 3, dtype=torch.bool)
        amd_tuned_torch._patched_sdpa(q, k, v, attn_mask=mask)
        te.scaled_dot_product_attention.assert_not_called()

    def test_falls_back_with_dropout(self, monkeypatch, te):
        monkeypatch.setattr(amd_tuned_torch, "_usable", lambda *a, **k: True)
        q = k = v = torch.randn(1, 2, 3, 4)
        amd_tuned_torch._patched_sdpa(q, k, v, dropout_p=0.1)
        te.scaled_dot_product_attention.assert_not_called()

    def test_calls_te_when_eligible(self, monkeypatch, te):
        monkeypatch.setattr(amd_tuned_torch, "_usable", lambda *a, **k: True)
        q = k = v = torch.randn(1, 2, 3, 4)
        te.scaled_dot_product_attention.return_value = torch.zeros(1, 2, 3, 4)
        out = amd_tuned_torch._patched_sdpa(q, k, v, is_causal=True)
        assert out is te.scaled_dot_product_attention.return_value
        te.scaled_dot_product_attention.assert_called_once_with(
            q, k, v, attn_mask=None, scale=None, is_causal=True
        )

    def test_falls_back_on_te_runtime_error(self, monkeypatch, te):
        monkeypatch.setattr(amd_tuned_torch, "_usable", lambda *a, **k: True)
        te.scaled_dot_product_attention.side_effect = RuntimeError("unsupported head_dim")
        q = k = v = torch.randn(1, 2, 3, 4)
        out = amd_tuned_torch._patched_sdpa(q, k, v)
        assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v))

    def test_routes_bottom_right_causal_mask_to_te(self, monkeypatch, te):
        # dflash / HF's sdpa_attention_forward build an explicit boolean
        # causal mask instead of setting is_causal=True -- this must still
        # reach TE, not silently fall back to stock forever.
        monkeypatch.setattr(amd_tuned_torch, "_usable", lambda *a, **k: True)
        q = k = v = torch.randn(1, 2, 3, 4)
        mask = torch.tril(torch.ones(3, 3, dtype=torch.bool))
        te.scaled_dot_product_attention.return_value = torch.zeros(1, 2, 3, 4)
        out = amd_tuned_torch._patched_sdpa(q, k, v, attn_mask=mask)
        assert out is te.scaled_dot_product_attention.return_value
        te.scaled_dot_product_attention.assert_called_once_with(
            q, k, v, attn_mask=mask, scale=None, is_causal=False
        )

    def test_falls_back_when_both_is_causal_and_attn_mask_set(self, monkeypatch, te):
        monkeypatch.setattr(amd_tuned_torch, "_usable", lambda *a, **k: True)
        q = k = v = torch.randn(1, 2, 3, 4)
        mask = torch.tril(torch.ones(3, 3, dtype=torch.bool))
        amd_tuned_torch._patched_sdpa(q, k, v, attn_mask=mask, is_causal=True)
        te.scaled_dot_product_attention.assert_not_called()


# ---------------------------------------------------------------------------
# INT8 (W8A8) linear -- aiter-backed, opt-in only (never installed by
# enable()/disable()). See amd_tuned_torch/aiter_ops.py and enable_int8_linear()'s
# docstring: this changes numerics, unlike everything else in this file.
# ---------------------------------------------------------------------------

class TestEnableDisableInt8Linear:
    def test_disabled_by_default(self):
        assert not amd_tuned_torch.is_int8_linear_enabled()

    def test_noop_when_aiter_unavailable(self, monkeypatch):
        monkeypatch.setattr(amd_tuned_torch.aiter_ops, "available", lambda: False)
        orig_linear = F.linear
        amd_tuned_torch.enable_int8_linear()
        try:
            assert not amd_tuned_torch.is_int8_linear_enabled()
            assert F.linear is orig_linear
        finally:
            amd_tuned_torch.disable_int8_linear()

    def test_enable_installs_patched_linear(self, aiter):
        assert not amd_tuned_torch.is_int8_linear_enabled()
        amd_tuned_torch.enable_int8_linear()
        try:
            assert amd_tuned_torch.is_int8_linear_enabled()
            assert F.linear is amd_tuned_torch._patched_linear_int8
        finally:
            amd_tuned_torch.disable_int8_linear()

    def test_disable_restores_prior_linear(self, aiter):
        orig_linear = F.linear
        amd_tuned_torch.enable_int8_linear()
        amd_tuned_torch.disable_int8_linear()
        assert F.linear is orig_linear
        assert not amd_tuned_torch.is_int8_linear_enabled()

    def test_composes_on_top_of_main_enable(self, aiter):
        # enable_int8_linear() after enable(): falls back to the aiter
        # Triton fp16/bf16 path (_patched_linear), not straight to stock,
        # when ineligible.
        amd_tuned_torch.enable()
        amd_tuned_torch.enable_int8_linear()
        try:
            assert F.linear is amd_tuned_torch._patched_linear_int8
            assert amd_tuned_torch._int8_linear_fallback is amd_tuned_torch._patched_linear
        finally:
            amd_tuned_torch.disable_int8_linear()
            amd_tuned_torch.disable()

    def test_is_idempotent(self, aiter):
        amd_tuned_torch.enable_int8_linear()
        try:
            fallback = amd_tuned_torch._int8_linear_fallback
            amd_tuned_torch.enable_int8_linear()
            assert amd_tuned_torch._int8_linear_fallback is fallback
        finally:
            amd_tuned_torch.disable_int8_linear()


class TestPatchedLinearInt8:
    def test_falls_back_when_ineligible(self, monkeypatch, aiter):
        monkeypatch.setattr(amd_tuned_torch, "_usable", lambda *a, **k: False)
        fallback = MagicMock(return_value=torch.zeros(2, 3))
        monkeypatch.setattr(amd_tuned_torch, "_int8_linear_fallback", fallback)
        x = torch.randn(2, 4)
        w = torch.randn(3, 4)
        out = amd_tuned_torch._patched_linear_int8(x, w, None)
        assert out is fallback.return_value
        aiter.linear_int8.assert_not_called()
        fallback.assert_called_once_with(x, w, None)

    def test_calls_aiter_when_eligible(self, monkeypatch, aiter):
        monkeypatch.setattr(amd_tuned_torch, "_usable", lambda *a, **k: True)
        x = torch.randn(2, 4)
        w = torch.randn(3, 4)
        aiter.linear_int8.return_value = torch.zeros(2, 3)
        out = amd_tuned_torch._patched_linear_int8(x, w, None)
        assert out is aiter.linear_int8.return_value
        aiter.linear_int8.assert_called_once_with(x, w, None)

    def test_falls_back_on_aiter_runtime_error(self, monkeypatch, aiter):
        monkeypatch.setattr(amd_tuned_torch, "_usable", lambda *a, **k: True)
        aiter.linear_int8.side_effect = RuntimeError("aiter not built for this shape")
        fallback = MagicMock(return_value=torch.zeros(2, 3))
        monkeypatch.setattr(amd_tuned_torch, "_int8_linear_fallback", fallback)
        x = torch.randn(2, 4)
        w = torch.randn(3, 4)
        out = amd_tuned_torch._patched_linear_int8(x, w, None)
        assert out is fallback.return_value
