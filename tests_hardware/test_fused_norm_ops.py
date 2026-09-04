"""Numerical correctness of amd_tuned_torch.fused_norm_ops.fused_add_rms_norm
against an unfused PyTorch reference, on real hardware -- the Triton
kernel needs a GPU to actually launch, so this can't run in the CPU-only
tests/ suite (see tests/test_fused_norm_ops.py for the guard-clause tests
that do run there).

Run with:

    pytest tests_hardware/test_fused_norm_ops.py
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
from amd_tuned_torch import fused_norm_ops  # noqa: E402

if not fused_norm_ops.available():
    pytest.skip("triton is not importable in this environment", allow_module_level=True)


def _reference(x, residual, weight, eps):
    hidden = (x.float() + residual.float())
    new_residual = hidden.to(x.dtype)
    variance = hidden.pow(2).mean(-1, keepdim=True)
    normed = hidden * torch.rsqrt(variance + eps)
    out = (normed * weight.float()).to(x.dtype)
    return out, new_residual


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batch, seq, hidden_dim", [(1, 1, 128), (2, 8, 4096), (4, 16, 7168)])
def test_matches_unfused_reference(dtype, batch, seq, hidden_dim):
    torch.manual_seed(0)
    x = torch.randn(batch, seq, hidden_dim, dtype=dtype, device="cuda")
    residual = torch.randn(batch, seq, hidden_dim, dtype=dtype, device="cuda")
    weight = torch.randn(hidden_dim, dtype=dtype, device="cuda")
    eps = 1e-6

    expected_out, expected_residual = _reference(x, residual, weight, eps)
    actual_out, actual_residual = fused_norm_ops.fused_add_rms_norm(x, residual, weight, eps=eps)

    assert actual_out.shape == x.shape
    assert actual_residual.shape == x.shape
    torch.testing.assert_close(actual_residual, expected_residual, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual_out, expected_out, atol=2e-2, rtol=2e-2)


def test_decode_loop_matches_reference_across_steps():
    """Simulates a few decode steps (batch=1, seq=1, carrying `residual`
    across calls) since that's the actual usage pattern this optimizes."""
    torch.manual_seed(0)
    dtype = torch.bfloat16
    hidden_dim = 4096
    residual = torch.randn(2, 1, hidden_dim, dtype=dtype, device="cuda")
    weight = torch.randn(hidden_dim, dtype=dtype, device="cuda")
    eps = 1e-6

    ref_residual = residual.clone()
    for _ in range(4):
        x = torch.randn(2, 1, hidden_dim, dtype=dtype, device="cuda")
        expected_out, ref_residual = _reference(x, ref_residual, weight, eps)
        actual_out, residual = fused_norm_ops.fused_add_rms_norm(x, residual, weight, eps=eps)
        torch.testing.assert_close(actual_out, expected_out, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(residual, ref_residual, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batch, seq, hidden_dim", [(2, 4, 256), (3, 8, 4096)])
def test_backward_matches_reference(dtype, batch, seq, hidden_dim):
    """_reference() above is built entirely from ordinary differentiable
    PyTorch ops (add/pow/mean/rsqrt/mul, plus dtype casts, all of which
    have well-defined gradients), so it already has a correct backward
    via plain autograd -- compare our kernel's gradients (x, residual,
    AND weight) against that, not against a second hand-derived backward
    formula that could share the same mistake as the kernel's own."""
    torch.manual_seed(1)
    grad_out_seed = torch.randn(batch, seq, hidden_dim, dtype=dtype, device="cuda")
    grad_residual_seed = torch.randn(batch, seq, hidden_dim, dtype=dtype, device="cuda")
    x_seed = torch.randn(batch, seq, hidden_dim, dtype=dtype, device="cuda")
    residual_seed = torch.randn(batch, seq, hidden_dim, dtype=dtype, device="cuda")
    weight_seed = torch.randn(hidden_dim, dtype=dtype, device="cuda")
    eps = 1e-6

    x_ref = x_seed.clone().requires_grad_()
    residual_ref = residual_seed.clone().requires_grad_()
    weight_ref = weight_seed.clone().requires_grad_()
    out_ref, new_residual_ref = _reference(x_ref, residual_ref, weight_ref, eps)
    ((out_ref * grad_out_seed).sum() + (new_residual_ref.float() * grad_residual_seed.float()).sum()).backward()

    x_fused = x_seed.clone().requires_grad_()
    residual_fused = residual_seed.clone().requires_grad_()
    weight_fused = weight_seed.clone().requires_grad_()
    out_fused, new_residual_fused = fused_norm_ops.fused_add_rms_norm(x_fused, residual_fused, weight_fused, eps=eps)
    ((out_fused * grad_out_seed).sum() + (new_residual_fused.float() * grad_residual_seed.float()).sum()).backward()

    torch.testing.assert_close(x_fused.grad, x_ref.grad, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(residual_fused.grad, residual_ref.grad, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(weight_fused.grad, weight_ref.grad, atol=3e-2, rtol=3e-2)


def test_backward_without_residual_grad():
    """Confirms HAS_RESIDUAL_GRAD=False path (grad_new_residual is None,
    e.g. the last layer's residual is never used again) doesn't crash and
    still produces correct x/weight gradients -- only via the `out` path."""
    torch.manual_seed(2)
    batch, hidden_dim = 4, 512
    eps = 1e-6
    x_seed = torch.randn(batch, hidden_dim, device="cuda")
    residual_seed = torch.randn(batch, hidden_dim, device="cuda")
    weight_seed = torch.randn(hidden_dim, device="cuda")

    x_ref = x_seed.clone().requires_grad_()
    residual_ref = residual_seed.clone().requires_grad_()
    weight_ref = weight_seed.clone().requires_grad_()
    out_ref, _ = _reference(x_ref, residual_ref, weight_ref, eps)
    out_ref.sum().backward()

    x_fused = x_seed.clone().requires_grad_()
    residual_fused = residual_seed.clone().requires_grad_()
    weight_fused = weight_seed.clone().requires_grad_()
    out_fused, _ = fused_norm_ops.fused_add_rms_norm(x_fused, residual_fused, weight_fused, eps=eps)
    out_fused.sum().backward()

    torch.testing.assert_close(x_fused.grad, x_ref.grad, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(residual_fused.grad, residual_ref.grad, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(weight_fused.grad, weight_ref.grad, atol=3e-2, rtol=3e-2)
