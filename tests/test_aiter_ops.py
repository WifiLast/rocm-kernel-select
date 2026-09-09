"""Tests for amd_tuned_torch.aiter_ops's manually-callable helpers that aren't
wired into any amd_tuned_torch.__init__ dispatch layer (no stock F.* op to
intercept). Currently just fused_silu_mul -- linear_fp16/bmm_fp16/
conv2d_fp16/linear_int8 are exercised indirectly via the mocked `aiter`
fixture in test_amd_tuned_torch_monkeypatch.py, since those go through
_patched_linear etc.; fused_silu_mul has no such wrapper to test through,
so it's tested directly here instead.

Run with:

    python -m pytest source/cmp_ext_turing/tests
"""
from __future__ import annotations

from unittest.mock import MagicMock

import torch

import amd_tuned_torch.aiter_ops as aiter_ops


class TestFusedSiluMul:
    """fused_silu_mul is a thin, direct pass-through to
    aiter.ops.triton.activation.fused_silu_mul -- imported locally at
    amd_tuned_torch.aiter_ops module load time (see that module's try/except
    ImportError block), no network/Hub dependency at all."""

    def test_calls_local_aiter_function_and_returns_its_result(self, monkeypatch):
        expected = torch.zeros(4, 4)
        fake_fused_silu_mul = MagicMock(return_value=expected)
        monkeypatch.setattr(aiter_ops, "_fused_silu_mul", fake_fused_silu_mul)

        x = torch.randn(4, 8)
        out = aiter_ops.fused_silu_mul(x)
        assert out is expected
        fake_fused_silu_mul.assert_called_once_with(x, out=None)

    def test_forwards_the_out_argument(self, monkeypatch):
        fake_fused_silu_mul = MagicMock()
        monkeypatch.setattr(aiter_ops, "_fused_silu_mul", fake_fused_silu_mul)

        x = torch.randn(4, 8)
        preallocated = torch.empty(4, 4)
        aiter_ops.fused_silu_mul(x, out=preallocated)
        fake_fused_silu_mul.assert_called_once_with(x, out=preallocated)


class TestQuantizeWeightInferenceTensor:
    """Same inference-tensor hazard the occupancy cache had (see
    aiter_ops._tensor_version): a weight cast inside an inference_mode
    block has no version counter, and reading one for the cache key would
    raise instead of quantizing."""

    def test_quantize_weight_handles_an_inference_tensor(self):
        with torch.inference_mode():
            w = torch.randn(4, 8).half().float()   # created here -> inference tensor
            assert aiter_ops._tensor_version(w) is None
            q, scale = aiter_ops._quantize_weight(w)
        assert q.dtype == torch.int8 and q.shape == (4, 8)
        assert scale.shape == (1, 4)

    def test_ordinary_weight_still_caches(self):
        w = torch.randn(4, 8)
        first_q, first_scale = aiter_ops._quantize_weight(w)
        second_q, second_scale = aiter_ops._quantize_weight(w)
        assert second_q is first_q and second_scale is first_scale
