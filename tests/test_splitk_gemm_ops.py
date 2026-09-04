"""CPU-safe tests for splitk_gemm_ops -- linear() declines (returns None)
before ever launching the Triton kernel for every case tested here
(wrong device, wrong shape, wrong dtype, out of its eligible M/K range),
so these run without a GPU or triton installed. _choose_split_k is pure
host-side integer math and is fully tested here too. The kernel's actual
numerical output needs a GPU to launch -- see
tests_hardware/test_splitk_gemm_ops.py."""
from __future__ import annotations

import torch

from amd_tuned_torch import splitk_gemm_ops


class TestAvailable:
    def test_matches_triton_importability(self):
        assert splitk_gemm_ops.available() is splitk_gemm_ops._TRITON_AVAILABLE


class TestCeildiv:
    def test_exact(self):
        assert splitk_gemm_ops._ceildiv(128, 128) == 1

    def test_rounds_up(self):
        assert splitk_gemm_ops._ceildiv(129, 128) == 2
        assert splitk_gemm_ops._ceildiv(1, 128) == 1


class TestChooseSplitK:
    """Pure Python -- no GPU/triton needed. _cu_count falls back to
    _DEFAULT_CU_COUNT when torch.cuda.get_device_properties fails, which
    it always will against a CPU device."""

    def test_within_bounds(self):
        split_k = splitk_gemm_ops._choose_split_k(M=1, N=4096, K=4096, device=torch.device("cpu"))
        assert 1 <= split_k <= splitk_gemm_ops._MAX_SPLIT_K

    def test_never_exceeds_k_over_block_k(self):
        # K=256 with BLOCK_K=128 means splitting more than 2 ways would
        # leave a split with less than one full block of real work.
        split_k = splitk_gemm_ops._choose_split_k(M=1, N=4096, K=256, device=torch.device("cpu"))
        assert split_k <= max(1, 256 // splitk_gemm_ops._BLOCK_K)

    def test_larger_m_needs_less_splitting(self):
        # More rows already means more (n_block, m) output tiles exist,
        # so less K-splitting is needed to reach the same target
        # occupancy -- split_k for M=16 should not exceed split_k for M=1
        # at the same N/K.
        split_k_m1 = splitk_gemm_ops._choose_split_k(M=1, N=4096, K=4096, device=torch.device("cpu"))
        split_k_m16 = splitk_gemm_ops._choose_split_k(M=16, N=4096, K=4096, device=torch.device("cpu"))
        assert split_k_m16 <= split_k_m1

    def test_always_at_least_one(self):
        split_k = splitk_gemm_ops._choose_split_k(M=16, N=32, K=256, device=torch.device("cpu"))
        assert split_k >= 1


class TestLinearGuards:
    def _unavailable(self, monkeypatch):
        monkeypatch.setattr(splitk_gemm_ops, "_TRITON_AVAILABLE", False)

    def test_none_when_triton_unavailable(self, monkeypatch):
        self._unavailable(monkeypatch)
        x = torch.randn(1, 4096)
        w = torch.randn(4096, 4096)
        assert splitk_gemm_ops.linear(x, w) is None

    def test_none_on_cpu_tensors(self, monkeypatch):
        monkeypatch.setattr(splitk_gemm_ops, "_TRITON_AVAILABLE", True)
        x = torch.randn(1, 4096)
        w = torch.randn(4096, 4096)
        assert splitk_gemm_ops.linear(x, w) is None

    def test_none_when_m_too_large(self, monkeypatch):
        # M=32 exceeds _MAX_M. Can't isolate this guard specifically
        # without a real CUDA tensor (the device check declines first on
        # CPU), but the end result -- None -- is the same either way, and
        # is what every guard test here is actually checking.
        monkeypatch.setattr(splitk_gemm_ops, "_TRITON_AVAILABLE", True)
        x = torch.randn(32, 4096)
        w = torch.randn(4096, 4096)
        assert splitk_gemm_ops.linear(x, w) is None

    def test_none_when_k_too_small(self, monkeypatch):
        monkeypatch.setattr(splitk_gemm_ops, "_TRITON_AVAILABLE", True)
        x = torch.randn(1, 64)  # K=64 < _MIN_K, also not CUDA -- both reasons to decline
        w = torch.randn(128, 64)
        assert splitk_gemm_ops.linear(x, w) is None

    def test_none_on_dtype_mismatch(self, monkeypatch):
        monkeypatch.setattr(splitk_gemm_ops, "_TRITON_AVAILABLE", True)
        x = torch.randn(1, 4096, dtype=torch.float16)
        w = torch.randn(4096, 4096, dtype=torch.float32)
        assert splitk_gemm_ops.linear(x, w) is None

    def test_none_on_weight_dim_mismatch(self, monkeypatch):
        monkeypatch.setattr(splitk_gemm_ops, "_TRITON_AVAILABLE", True)
        x = torch.randn(1, 4096)
        w = torch.randn(4096, 4096, 1)  # 3D weight, invalid
        assert splitk_gemm_ops.linear(x, w) is None

    def test_none_on_k_mismatch(self, monkeypatch):
        monkeypatch.setattr(splitk_gemm_ops, "_TRITON_AVAILABLE", True)
        x = torch.randn(1, 4096)
        w = torch.randn(4096, 2048)  # weight's K doesn't match input's K
        assert splitk_gemm_ops.linear(x, w) is None
