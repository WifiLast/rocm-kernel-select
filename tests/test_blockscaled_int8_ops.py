"""CPU-safe tests for blockscaled_int8_ops -- the PyTorch references are
checked against independent per-block loops, and the public functions' CPU
routing (reference fallback) runs without a GPU. The Triton kernels are
checked against these references in tests_hardware/test_blockscaled_int8_ops.py."""
from __future__ import annotations

import pytest
import torch

from amd_tuned_torch import blockscaled_int8_ops as ops


def _loop_quant(x):
    M, K = x.shape
    B = ops.QBLOCK
    q = torch.empty(M, K, dtype=torch.int8)
    s = torch.empty((M + B - 1) // B, (K + B - 1) // B)
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            blk = x[i * B:(i + 1) * B, j * B:(j + 1) * B].float()
            amax = max(blk.abs().max().item(), 1e-8)
            s[i, j] = amax / 127.0
            q[i * B:(i + 1) * B, j * B:(j + 1) * B] = torch.round(blk * (127.0 / amax)).to(torch.int8)
    return q, s


class TestAvailable:
    def test_matches_triton_importability(self):
        assert ops.available() is ops._TRITON_AVAILABLE


@pytest.mark.parametrize("shape", [(128, 128), (130, 300), (1, 77), (256, 129)])
def test_quant_reference_matches_loop(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape) * 3
    q, s = ops.int8_quant_reference(x)
    q2, s2 = _loop_quant(x)
    assert torch.equal(q, q2)
    assert torch.allclose(s, s2, rtol=1e-6, atol=0)


def test_quant_zero_block_uses_amax_floor():
    q, s = ops.int8_quant_reference(torch.zeros(4, 4))
    assert torch.equal(q, torch.zeros(4, 4, dtype=torch.int8))
    assert s.item() == pytest.approx(1e-8 / 127.0)


def test_quant_max_maps_to_127():
    x = torch.tensor([[2.0, -2.0, 1.0, 0.0]])
    q, _ = ops.int8_quant_reference(x)
    assert q.tolist() == [[127, -127, 64, 0]]  # 63.5 rounds half-to-even -> 64


@pytest.mark.parametrize("M,N,K", [(64, 96, 128), (130, 257, 300), (5, 128, 77)])
def test_gemm_reference_matches_dequantised_matmul(M, N, K):
    torch.manual_seed(1)
    x = torch.randn(M, K)
    w = torch.randn(N, K)
    xq, xs = ops.int8_quant_reference(x)
    wq, ws = ops.int8_quant_reference(w)
    B = ops.QBLOCK
    deq = lambda q, s: q.double() * s.double().repeat_interleave(B, 0).repeat_interleave(B, 1)[:q.shape[0], :q.shape[1]]  # noqa: E731
    want = (deq(xq, xs) @ deq(wq, ws).T).float()
    got = ops.int8_gemm_reference(xq, xs, wq, ws)
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-5)
    # and it approximates the float product
    assert (got - x @ w.T).abs().max() < 0.05 * (x @ w.T).abs().max()


def test_gemm_reference_bias_and_dtype():
    torch.manual_seed(2)
    xq, xs = ops.int8_quant_reference(torch.randn(8, 64))
    wq, ws = ops.int8_quant_reference(torch.randn(16, 64))
    bias = torch.randn(16)
    y = ops.int8_gemm_reference(xq, xs, wq, ws, torch.bfloat16, bias)
    assert y.dtype == torch.bfloat16
    assert torch.allclose(y.float(), ops.int8_gemm_reference(xq, xs, wq, ws) + bias, rtol=1e-2, atol=1e-2)


def test_cpu_tensors_take_reference_path():
    torch.manual_seed(3)
    x = torch.randn(3, 5, 64)
    w = torch.randn(32, 64)
    wq, ws = ops.int8_quant(w)
    y = ops.int8_linear(x, wq, ws)
    assert y.shape == (3, 5, 32) and y.dtype == x.dtype
    xq, xs = ops.int8_quant_reference(x.reshape(-1, 64))
    assert torch.allclose(y.reshape(-1, 32), ops.int8_gemm_reference(xq, xs, wq, ws), rtol=1e-6, atol=1e-6)
    assert not ops._kernel_failed
