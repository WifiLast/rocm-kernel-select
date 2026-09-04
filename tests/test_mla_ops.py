"""mla_ops.mla_decode -- weight-absorption MLA decode. The property worth
testing isn't shape plumbing, it's that the absorption reordering
(q_nope @ kup_nope folded into the query, and the output-side vup
expansion deferred until after the attention-weighted sum) is an exact
algebraic identity versus materializing k_nope/v explicitly and running
ordinary per-head attention. See amd_tuned_torch/mla_ops.py's module
docstring and amd_tuned_torch/_vendor/radeonflow_mla/NOTICE.md."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import amd_tuned_torch
from amd_tuned_torch import mla_ops

torch.manual_seed(0)

HIDDEN_DIM = 16
N_HEADS = 2
NOPE_DIM = 4
ROPE_DIM = 4
V_DIM = 4
Q_LORA_DIM = 6
KV_LORA_DIM = 5
MAX_SEQ_LEN = 8


def _make_weights(dtype=torch.float32):
    return dict(
        q_proj_down_weight=torch.randn(Q_LORA_DIM, HIDDEN_DIM, dtype=dtype),
        q_proj_up_weight=torch.randn(N_HEADS * (NOPE_DIM + ROPE_DIM), Q_LORA_DIM, dtype=dtype),
        kv_proj_down_weight=torch.randn(KV_LORA_DIM + ROPE_DIM, HIDDEN_DIM, dtype=dtype),
        kv_proj_up_weight=torch.randn(N_HEADS * (NOPE_DIM + V_DIM), KV_LORA_DIM, dtype=dtype),
        wo_weight=torch.randn(HIDDEN_DIM, N_HEADS * V_DIM, dtype=dtype),
    )


def _naive_mla_decode(x, kv_cache, kv_len, weights):
    """Same math as mla_ops.mla_decode, but materializes k_nope/v via
    KV_proj_up explicitly instead of absorbing kup_nope/vup into the
    query/output -- the "obviously correct, obviously slow" version this
    test checks the absorbed version against."""
    kv_and_rope = F.linear(x, weights["kv_proj_down_weight"])
    kv_cache = kv_cache.clone()
    kv_cache[:, kv_len - 1: kv_len, :] = kv_and_rope
    kv_lora, k_rope = kv_cache[:, :kv_len, :].split([KV_LORA_DIM, ROPE_DIM], dim=-1)

    kv_up = F.linear(kv_lora, weights["kv_proj_up_weight"]).view(
        kv_lora.shape[0], kv_len, N_HEADS, NOPE_DIM + V_DIM
    )
    k_nope, v = kv_up.split([NOPE_DIM, V_DIM], dim=-1)
    k_nope = k_nope.permute(0, 2, 1, 3)  # (batch, n_heads, kv_len, nope_dim)
    v = v.permute(0, 2, 1, 3)            # (batch, n_heads, kv_len, v_dim)

    q_lora = F.linear(x, weights["q_proj_down_weight"])
    q_up = F.linear(q_lora, weights["q_proj_up_weight"]).view(
        x.shape[0], 1, N_HEADS, NOPE_DIM + ROPE_DIM
    )
    q_nope, q_rope = q_up.split([NOPE_DIM, ROPE_DIM], dim=-1)
    q_nope = q_nope.permute(0, 2, 1, 3)  # (batch, n_heads, 1, nope_dim)
    q_rope = q_rope.permute(0, 2, 1, 3)  # (batch, n_heads, 1, rope_dim)

    cos, sin = mla_ops._rope_cache(kv_len, ROPE_DIM, 10000.0, x.dtype, x.device)
    q_rope = mla_ops._apply_rope(q_rope, cos[kv_len - 1: kv_len], sin[kv_len - 1: kv_len])
    k_rope = mla_ops._apply_rope(k_rope, cos[:kv_len], sin[:kv_len])

    attn_nope = torch.einsum("b h s d, b h p d -> b h s p", q_nope, k_nope)
    attn_rope = torch.einsum("b h s d, b p d -> b h s p", q_rope, k_rope)
    attention = torch.softmax((attn_nope + attn_rope) / math.sqrt(NOPE_DIM + ROPE_DIM), dim=-1)

    o = torch.einsum("b h s p, b h p v -> b h s v", attention, v)
    wo = weights["wo_weight"].view(HIDDEN_DIM, N_HEADS, V_DIM)
    return torch.einsum("b h s v, d h v -> b s d", o, wo)


class TestAvailable:
    def test_always_available(self):
        assert mla_ops.available() is True


class TestMlaDecodeMatchesNaiveReference:
    def test_single_step_matches_naive(self):
        batch = 3
        prefill = 4
        kv_len = prefill + 1
        weights = _make_weights()
        x = torch.randn(batch, 1, HIDDEN_DIM)
        kv_cache_absorbed = torch.zeros(batch, MAX_SEQ_LEN, KV_LORA_DIM + ROPE_DIM)
        kv_cache_absorbed[:, :prefill, :] = torch.randn(batch, prefill, KV_LORA_DIM + ROPE_DIM)
        kv_cache_naive = kv_cache_absorbed.clone()

        expected = _naive_mla_decode(x, kv_cache_naive, kv_len, weights)
        actual = mla_ops.mla_decode(
            x, kv_cache_absorbed, kv_len,
            weights["q_proj_down_weight"], weights["q_proj_up_weight"],
            weights["kv_proj_down_weight"], weights["kv_proj_up_weight"],
            weights["wo_weight"], N_HEADS, NOPE_DIM, ROPE_DIM, V_DIM,
        )
        assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)

    def test_multi_step_decode_matches_naive(self):
        """Simulates a short decode loop -- kv_cache carried across calls,
        kv_len incrementing each step -- since that's how mla_decode is
        actually meant to be used (one call per generated token)."""
        batch = 2
        prefill = 2
        weights = _make_weights()
        kv_cache_absorbed = torch.zeros(batch, MAX_SEQ_LEN, KV_LORA_DIM + ROPE_DIM)
        kv_cache_absorbed[:, :prefill, :] = torch.randn(batch, prefill, KV_LORA_DIM + ROPE_DIM)
        kv_cache_naive = kv_cache_absorbed.clone()

        for step in range(3):
            kv_len = prefill + step + 1
            x = torch.randn(batch, 1, HIDDEN_DIM)
            expected = _naive_mla_decode(x, kv_cache_naive, kv_len, weights)
            kv_cache_naive[:, kv_len - 1: kv_len, :] = F.linear(x, weights["kv_proj_down_weight"])
            actual = mla_ops.mla_decode(
                x, kv_cache_absorbed, kv_len,
                weights["q_proj_down_weight"], weights["q_proj_up_weight"],
                weights["kv_proj_down_weight"], weights["kv_proj_up_weight"],
                weights["wo_weight"], N_HEADS, NOPE_DIM, ROPE_DIM, V_DIM,
            )
            assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4), f"mismatch at step {step}"


class TestMlaDecodeShape:
    def test_output_shape(self):
        batch = 2
        kv_len = 3
        weights = _make_weights()
        x = torch.randn(batch, 1, HIDDEN_DIM)
        kv_cache = torch.zeros(batch, MAX_SEQ_LEN, KV_LORA_DIM + ROPE_DIM)
        out = mla_ops.mla_decode(
            x, kv_cache, kv_len,
            weights["q_proj_down_weight"], weights["q_proj_up_weight"],
            weights["kv_proj_down_weight"], weights["kv_proj_up_weight"],
            weights["wo_weight"], N_HEADS, NOPE_DIM, ROPE_DIM, V_DIM,
        )
        assert out.shape == (batch, 1, HIDDEN_DIM)

    def test_writes_new_token_into_kv_cache(self):
        batch = 1
        kv_len = 2
        weights = _make_weights()
        x = torch.randn(batch, 1, HIDDEN_DIM)
        kv_cache = torch.zeros(batch, MAX_SEQ_LEN, KV_LORA_DIM + ROPE_DIM)
        mla_ops.mla_decode(
            x, kv_cache, kv_len,
            weights["q_proj_down_weight"], weights["q_proj_up_weight"],
            weights["kv_proj_down_weight"], weights["kv_proj_up_weight"],
            weights["wo_weight"], N_HEADS, NOPE_DIM, ROPE_DIM, V_DIM,
        )
        expected_new_slot = F.linear(x, weights["kv_proj_down_weight"])
        assert torch.allclose(kv_cache[:, kv_len - 1: kv_len, :], expected_new_slot)


class TestModuleWiring:
    def test_top_level_mla_decode_delegates(self):
        batch = 1
        kv_len = 1
        weights = _make_weights()
        x = torch.randn(batch, 1, HIDDEN_DIM)
        kv_cache_a = torch.zeros(batch, MAX_SEQ_LEN, KV_LORA_DIM + ROPE_DIM)
        kv_cache_b = kv_cache_a.clone()

        direct = mla_ops.mla_decode(
            x, kv_cache_a, kv_len,
            weights["q_proj_down_weight"], weights["q_proj_up_weight"],
            weights["kv_proj_down_weight"], weights["kv_proj_up_weight"],
            weights["wo_weight"], N_HEADS, NOPE_DIM, ROPE_DIM, V_DIM,
        )
        via_top_level = amd_tuned_torch.mla_decode(
            x, kv_cache_b, kv_len,
            weights["q_proj_down_weight"], weights["q_proj_up_weight"],
            weights["kv_proj_down_weight"], weights["kv_proj_up_weight"],
            weights["wo_weight"], N_HEADS, NOPE_DIM, ROPE_DIM, V_DIM,
        )
        assert torch.equal(direct, via_top_level)
