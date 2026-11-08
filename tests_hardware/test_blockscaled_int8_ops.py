"""Numerical correctness of amd_tuned_torch.blockscaled_int8_ops (Triton
quant + int8 GEMM) against the PyTorch references, on real hardware -- see
tests/test_blockscaled_int8_ops.py for the CPU-side tests.

Run with:

    pytest tests_hardware/test_blockscaled_int8_ops.py
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA/ROCm device available"
)
pytest.importorskip(
    "amd_tuned_torch",
    reason="amd_tuned_torch native extension not built -- run "
           "`pip install -e . --no-build-isolation` first",
)
from amd_tuned_torch import blockscaled_int8_ops as ops  # noqa: E402

if not ops.available():
    pytest.skip("triton is not importable in this environment", allow_module_level=True)


@pytest.mark.parametrize("shape", [(128, 128), (1000, 3072), (77, 4096), (1, 300), (4097, 129)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_quant_matches_reference(shape, dtype):
    torch.manual_seed(0)
    x = (torch.randn(*shape, device="cuda") * 4).to(dtype)
    q, s = ops.int8_quant(x)
    q2, s2 = ops.int8_quant_reference(x)
    assert not ops._kernel_failed
    assert torch.equal(s, s2)
    assert torch.equal(q, q2)


def test_quant_strided_rows():
    torch.manual_seed(1)
    wide = torch.randn(256, 1024, device="cuda", dtype=torch.float16)
    x = wide[:, 128:640]
    q, s = ops.int8_quant(x)
    q2, s2 = ops.int8_quant_reference(x)
    assert torch.equal(s, s2) and torch.equal(q, q2)


@pytest.mark.parametrize("M,N,K", [(4096, 3072, 3072), (75600, 1536, 128), (77, 4096, 4096),
                                   (1, 1536, 1536), (333, 257, 300), (8, 128, 64)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_matches_reference(M, N, K, dtype):
    if M * N > 50_000_000:
        M = 50_000_000 // N
    torch.manual_seed(2)
    xq, xs = ops.int8_quant(torch.randn(M, K, device="cuda", dtype=dtype))
    wq, ws = ops.int8_quant(torch.randn(N, K, device="cuda", dtype=dtype) * 0.02)
    bias = torch.randn(N, device="cuda", dtype=dtype)
    got = ops.int8_gemm(xq, xs, wq, ws, out_dtype=dtype, bias=bias)
    want = ops.int8_gemm_reference(xq, xs, wq, ws, torch.float32, bias)
    assert not ops._kernel_failed
    assert got.dtype == dtype
    tol = 1e-2 if dtype == torch.float16 else 2e-2
    assert torch.allclose(got.float(), want, rtol=tol, atol=tol * want.abs().mean().item())


def test_linear_close_to_float():
    torch.manual_seed(3)
    x = torch.randn(2, 1000, 1536, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16) * 0.02
    wq, ws = ops.int8_quant(w)
    y = ops.int8_linear(x, wq, ws)
    ref = (x.float() @ w.float().T)
    rel = (y.float() - ref).norm() / ref.norm()
    assert rel < 0.02, rel
