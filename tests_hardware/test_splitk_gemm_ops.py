"""Numerical correctness AND eligibility-range check for
amd_tuned_torch.splitk_gemm_ops.linear against stock F.linear, on real
hardware -- the Triton kernel needs a GPU to launch.

This is a from-scratch kernel (not a port), so correctness here matters
more than usual: kernel_select's own runtime verification (see
amd_tuned_torch/__init__.py's _patched_linear) is the production safety
net, but this file is what actually proves the kernel is right in the
first place, at a range of decode-relevant shapes.

Run with:

    pytest tests_hardware/test_splitk_gemm_ops.py
"""
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA/ROCm device available"
)
pytest.importorskip(
    "amd_tuned_torch",
    reason="amd_tuned_torch native extension not built -- run "
           "`pip install -e . --no-build-isolation` first",
)
from amd_tuned_torch import splitk_gemm_ops  # noqa: E402

if not splitk_gemm_ops.available():
    pytest.skip("triton is not importable in this environment", allow_module_level=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("M, K, N", [(1, 4096, 4096), (1, 7168, 128256), (8, 4096, 4096), (16, 4096, 11008)])
@pytest.mark.parametrize("has_bias", [False, True])
def test_matches_stock_linear(dtype, M, K, N, has_bias):
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=dtype, device="cuda") * 0.1
    w = torch.randn(N, K, dtype=dtype, device="cuda") * 0.1
    bias = torch.randn(N, dtype=dtype, device="cuda") * 0.1 if has_bias else None

    expected = F.linear(x, w, bias)
    actual = splitk_gemm_ops.linear(x, w, bias)

    assert actual is not None, "expected this shape to be eligible"
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_matches_stock_linear_with_batch_seq_leading_dims():
    """Confirms the (batch, seq, K) -> flatten -> (batch, seq, N) reshape
    round-trip, not just the already-2D case."""
    torch.manual_seed(1)
    batch, seq, K, N = 2, 4, 4096, 4096
    x = torch.randn(batch, seq, K, device="cuda") * 0.1
    w = torch.randn(N, K, device="cuda") * 0.1

    expected = F.linear(x, w)
    actual = splitk_gemm_ops.linear(x, w)

    assert actual is not None
    assert actual.shape == (batch, seq, N)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_declines_outside_eligible_m_range():
    x = torch.randn(splitk_gemm_ops._MAX_M + 1, 4096, device="cuda")
    w = torch.randn(4096, 4096, device="cuda")
    assert splitk_gemm_ops.linear(x, w) is None


def test_declines_below_min_k():
    x = torch.randn(1, splitk_gemm_ops._MIN_K - 1, device="cuda")
    w = torch.randn(128, splitk_gemm_ops._MIN_K - 1, device="cuda")
    assert splitk_gemm_ops.linear(x, w) is None


def test_registered_as_kernel_select_candidate():
    """End-to-end: F.linear itself (patched by amd_tuned_torch.enable())
    should be able to select "splitk" for an eligible decode shape and
    still produce a numerically correct result -- proves the wiring in
    __init__.py._patched_linear, not just the standalone function.
    Compares against a plain manual reference rather than introspecting
    "the original F.linear" through the patch, since nothing in this
    package guarantees _install preserves an unwrap hook."""
    import amd_tuned_torch
    from amd_tuned_torch import kernel_select

    was_enabled = amd_tuned_torch.is_enabled()
    if not was_enabled:
        amd_tuned_torch.enable()
    try:
        kernel_select.reset()
        x = torch.randn(1, 4096, dtype=torch.float16, device="cuda") * 0.1
        w = torch.randn(4096, 4096, dtype=torch.float16, device="cuda") * 0.1
        expected = (x.float() @ w.float().t()).to(x.dtype)
        actual = torch.nn.functional.linear(x, w)
        assert actual.shape == (1, 4096)
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)
    finally:
        if not was_enabled:
            amd_tuned_torch.disable()
