"""CPU-safe tests for iu4_gemm_ops -- pack_int4_rows/_quantize_rowwise are
pure tensor math (no native kernel launch) and iu8_linear/iu4_linear decline
(return None) before ever calling the native GEMM for every case tested
here (wrong device, wrong shape, unavailable). The native kernels'
actual numerical output needs a real gfx1100 device to launch -- see
tests_hardware/test_iu4_gemm_ops.py."""
from __future__ import annotations

import torch

from amd_tuned_torch import iu4_gemm_ops


class TestQuantizeRowwise:
    def test_round_trip_within_one_quantum(self):
        torch.manual_seed(0)
        x = torch.randn(8, 32) * 3.0
        x_int8, scale = iu4_gemm_ops._quantize_rowwise(x, qmax=127)
        assert x_int8.dtype == torch.int8
        assert x_int8.abs().max() <= 127
        reconstructed = x_int8.float() * scale.unsqueeze(-1)
        assert torch.allclose(reconstructed, x, atol=(scale.max() * 1.0).item())

    def test_clamped_to_qmax(self):
        x = torch.tensor([[100.0, -100.0, 1.0, -1.0]])
        x_int8, _ = iu4_gemm_ops._quantize_rowwise(x, qmax=7)
        assert x_int8.abs().max() <= 7

    def test_all_zero_row_does_not_divide_by_zero(self):
        x = torch.zeros(2, 16)
        x_int8, scale = iu4_gemm_ops._quantize_rowwise(x, qmax=127)
        assert torch.isfinite(scale).all()
        assert torch.equal(x_int8, torch.zeros_like(x_int8))


class TestPackInt4Rows:
    def test_matches_manual_nibble_layout(self):
        # k=4, values chosen so lo/hi nibbles are unambiguous: row0 = [1,2,3,4]
        x = torch.tensor([[1, 2, 3, 4]], dtype=torch.int8)
        packed = iu4_gemm_ops.pack_int4_rows(x)
        # k padded to 16 -> 8 bytes/row. Byte0 = nibble(k=0)=1 | nibble(k=1)=2<<4
        assert packed.shape == (1, 8)
        assert int(packed[0, 0]) == (1 | (2 << 4))
        assert int(packed[0, 1]) == (3 | (4 << 4))
        assert torch.equal(packed[0, 2:], torch.zeros(6, dtype=torch.uint8))

    def test_negative_values_use_twos_complement_nibble(self):
        # -1 as a signed 4-bit value is nibble 0xF.
        x = torch.tensor([[-1, 0]], dtype=torch.int8)
        packed = iu4_gemm_ops.pack_int4_rows(x)
        assert int(packed[0, 0]) == 0x0F

    def test_no_padding_needed_when_k_already_multiple_of_16(self):
        x = torch.zeros(1, 16, dtype=torch.int8)
        packed = iu4_gemm_ops.pack_int4_rows(x)
        assert packed.shape == (1, 8)


class TestAvailable:
    def test_false_when_native_reports_unsupported(self, native):
        native.iu4_gemm_supported.return_value = False
        assert iu4_gemm_ops.available() is False

    def test_true_when_native_reports_supported(self, native):
        native.iu4_gemm_supported.return_value = True
        assert iu4_gemm_ops.available() is True


class TestLinearGuards:
    def test_iu8_linear_none_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(iu4_gemm_ops, "available", lambda: False)
        x = torch.randn(4, 32)
        w = torch.randn(8, 32)
        assert iu4_gemm_ops.iu8_linear(x, w) is None

    def test_iu4_linear_none_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(iu4_gemm_ops, "available", lambda: False)
        x = torch.randn(4, 32)
        w = torch.randn(8, 32)
        assert iu4_gemm_ops.iu4_linear(x, w) is None

    def test_none_on_cpu_tensors_even_when_available(self, monkeypatch):
        monkeypatch.setattr(iu4_gemm_ops, "available", lambda: True)
        x = torch.randn(4, 32)
        w = torch.randn(8, 32)
        assert iu4_gemm_ops.iu8_linear(x, w) is None
        assert iu4_gemm_ops.iu4_linear(x, w) is None

    def test_none_on_k_mismatch(self, monkeypatch):
        monkeypatch.setattr(iu4_gemm_ops, "available", lambda: True)
        x = torch.randn(4, 32)
        w = torch.randn(8, 16)  # weight's K doesn't match input's K
        assert iu4_gemm_ops.iu8_linear(x, w) is None

    def test_none_on_weight_dim_mismatch(self, monkeypatch):
        monkeypatch.setattr(iu4_gemm_ops, "available", lambda: True)
        x = torch.randn(4, 32)
        w = torch.randn(8, 32, 1)  # 3D weight, invalid
        assert iu4_gemm_ops.iu8_linear(x, w) is None
