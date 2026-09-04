"""Numerical correctness of amd_tuned_torch.rope_ops.rotary_embedding
against an unfused PyTorch reference, on real hardware -- the Triton
kernel needs a GPU to actually launch, so this can't run in the CPU-only
tests/ suite (see tests/test_rope_ops.py for the guard-clause and
compute_cos_sin_cache tests that do run there).

Run with:

    pytest tests_hardware/test_rope_ops.py
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
from amd_tuned_torch import rope_ops  # noqa: E402

if not rope_ops.available():
    pytest.skip("triton is not importable in this environment", allow_module_level=True)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _reference(positions, query, key, head_size, cos_sin_cache):
    """NeoX-style RoPE reference: (batch=num_tokens, num_heads, head_size)
    view, cos/sin gathered per-token via positions, rotate-half applied --
    independent implementation of the same math the kernel computes, not
    a copy of it."""
    num_tokens = query.shape[0]
    num_heads = query.shape[-1] // head_size
    num_kv_heads = key.shape[-1] // head_size

    cos, sin = cos_sin_cache.index_select(0, positions).to(query.dtype).chunk(2, dim=-1)

    q = query.view(num_tokens, num_heads, head_size)
    cos_q = cos.unsqueeze(1)
    sin_q = sin.unsqueeze(1)
    q_rot = q * cos_q.repeat(1, 1, 2) + _rotate_half(q) * sin_q.repeat(1, 1, 2)

    k = key.view(num_tokens, num_kv_heads, head_size)
    cos_k = cos.unsqueeze(1)
    sin_k = sin.unsqueeze(1)
    k_rot = k * cos_k.repeat(1, 1, 2) + _rotate_half(k) * sin_k.repeat(1, 1, 2)

    return q_rot.view_as(query), k_rot.view_as(key)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_tokens, num_heads, num_kv_heads, head_size",
                          [(1, 4, 4, 64), (8, 32, 8, 128), (17, 4, 4, 256)])
def test_matches_unfused_reference(dtype, num_tokens, num_heads, num_kv_heads, head_size):
    torch.manual_seed(0)
    max_position = 2048
    cache = rope_ops.compute_cos_sin_cache(10000.0, head_size, max_position).to("cuda")
    positions = torch.randint(0, max_position, (num_tokens,), device="cuda")
    query = torch.randn(num_tokens, num_heads * head_size, dtype=dtype, device="cuda")
    key = torch.randn(num_tokens, num_kv_heads * head_size, dtype=dtype, device="cuda")

    expected_q, expected_k = _reference(positions, query.clone(), key.clone(), head_size, cache)
    actual_q, actual_k = rope_ops.rotary_embedding(positions, query, key, head_size, cache)

    assert actual_q is query and actual_k is key  # in-place, per the documented contract
    torch.testing.assert_close(actual_q, expected_q, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual_k, expected_k, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_tokens, num_heads, num_kv_heads, head_size",
                          [(1, 4, 4, 64), (8, 32, 8, 128)])
def test_backward_matches_reference(dtype, num_tokens, num_heads, num_kv_heads, head_size):
    """_reference() above is built entirely from ordinary differentiable
    PyTorch ops (view/chunk/cat/mul/add), so it already has a correct
    backward via plain autograd -- compare our kernel's gradients against
    that, not against a second hand-derived backward formula, since a
    hand-derived reference backward could share the same mistake as the
    kernel's own hand-derived one."""
    torch.manual_seed(2)
    max_position = 2048
    cache = rope_ops.compute_cos_sin_cache(10000.0, head_size, max_position).to("cuda")
    positions = torch.randint(0, max_position, (num_tokens,), device="cuda")
    query_seed = torch.randn(num_tokens, num_heads * head_size, dtype=dtype, device="cuda")
    key_seed = torch.randn(num_tokens, num_kv_heads * head_size, dtype=dtype, device="cuda")
    grad_seed_q = torch.randn(num_tokens, num_heads * head_size, dtype=dtype, device="cuda")
    grad_seed_k = torch.randn(num_tokens, num_kv_heads * head_size, dtype=dtype, device="cuda")

    query_ref = query_seed.clone().requires_grad_()
    key_ref = key_seed.clone().requires_grad_()
    q_out_ref, k_out_ref = _reference(positions, query_ref, key_ref, head_size, cache)
    ((q_out_ref * grad_seed_q).sum() + (k_out_ref * grad_seed_k).sum()).backward()

    query_fused = query_seed.clone().requires_grad_()
    key_fused = key_seed.clone().requires_grad_()
    q_out_fused, k_out_fused = rope_ops.rotary_embedding(positions, query_fused, key_fused, head_size, cache)
    ((q_out_fused * grad_seed_q).sum() + (k_out_fused * grad_seed_k).sum()).backward()

    torch.testing.assert_close(query_fused.grad, query_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(key_fused.grad, key_ref.grad, atol=2e-2, rtol=2e-2)
