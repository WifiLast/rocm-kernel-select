"""CPU-safe tests for rope_ops -- compute_cos_sin_cache is plain PyTorch
(no GPU needed, fully tested here) and rotary_embedding()'s guard clauses
run before ever launching the Triton kernel. The kernel itself needs a
real GPU to launch, so its numerical-correctness check against the
reference math lives in tests_hardware/test_rope_ops.py instead, same
split as tests_hardware/test_fused_norm_ops.py."""
from __future__ import annotations

import pytest
import torch

import amd_tuned_torch
from amd_tuned_torch import rope_ops


class TestAvailable:
    def test_matches_triton_importability(self):
        assert rope_ops.available() is rope_ops._TRITON_AVAILABLE


class TestComputeCosSinCache:
    def test_shape(self):
        cache = rope_ops.compute_cos_sin_cache(base=10000.0, rotary_dim=64, max_position_embeddings=128)
        assert cache.shape == (128, 64)

    def test_position_zero_is_all_ones_cos_zero_sin(self):
        # At position 0, every frequency's angle is 0 -> cos=1, sin=0.
        cache = rope_ops.compute_cos_sin_cache(base=10000.0, rotary_dim=8, max_position_embeddings=4)
        cos0, sin0 = cache[0].chunk(2, dim=-1)
        assert torch.allclose(cos0, torch.ones_like(cos0))
        assert torch.allclose(sin0, torch.zeros_like(sin0), atol=1e-6)

    def test_matches_hand_computed_frequencies(self):
        base, rotary_dim, max_pos = 10000.0, 4, 3
        cache = rope_ops.compute_cos_sin_cache(base, rotary_dim, max_pos)
        inv_freq = torch.tensor([1.0 / (base ** (0 / rotary_dim)), 1.0 / (base ** (2 / rotary_dim))])
        for pos in range(max_pos):
            expected_angles = pos * inv_freq
            cos, sin = cache[pos].chunk(2, dim=-1)
            assert torch.allclose(cos, expected_angles.cos(), atol=1e-5)
            assert torch.allclose(sin, expected_angles.sin(), atol=1e-5)


class TestRotaryEmbeddingGuards:
    def _unavailable(self, monkeypatch):
        monkeypatch.setattr(rope_ops, "_TRITON_AVAILABLE", False)

    def test_raises_when_triton_unavailable(self, monkeypatch):
        self._unavailable(monkeypatch)
        positions = torch.zeros(2, dtype=torch.long)
        query = torch.randn(2, 8)
        key = torch.randn(2, 8)
        cache = rope_ops.compute_cos_sin_cache(10000.0, 8, 4)
        with pytest.raises(RuntimeError, match="triton is not importable"):
            rope_ops.rotary_embedding(positions, query, key, 8, cache)

    def test_raises_on_non_1d_positions(self, monkeypatch):
        monkeypatch.setattr(rope_ops, "_TRITON_AVAILABLE", True)
        positions = torch.zeros(2, 1, dtype=torch.long)
        query = torch.randn(2, 8)
        key = torch.randn(2, 8)
        cache = rope_ops.compute_cos_sin_cache(10000.0, 8, 4)
        with pytest.raises(AssertionError, match="positions must be 1D"):
            rope_ops.rotary_embedding(positions, query, key, 8, cache)

    def test_raises_on_non_2d_query(self, monkeypatch):
        monkeypatch.setattr(rope_ops, "_TRITON_AVAILABLE", True)
        positions = torch.zeros(2, dtype=torch.long)
        query = torch.randn(2, 1, 8)
        key = torch.randn(2, 8)
        cache = rope_ops.compute_cos_sin_cache(10000.0, 8, 4)
        with pytest.raises(AssertionError, match="must be 2D"):
            rope_ops.rotary_embedding(positions, query, key, 8, cache)

    def test_raises_on_num_tokens_mismatch(self, monkeypatch):
        monkeypatch.setattr(rope_ops, "_TRITON_AVAILABLE", True)
        positions = torch.zeros(3, dtype=torch.long)
        query = torch.randn(2, 8)
        key = torch.randn(2, 8)
        cache = rope_ops.compute_cos_sin_cache(10000.0, 8, 4)
        with pytest.raises(AssertionError, match="num_tokens must match"):
            rope_ops.rotary_embedding(positions, query, key, 8, cache)

    def test_raises_on_partial_rotary_dim(self, monkeypatch):
        monkeypatch.setattr(rope_ops, "_TRITON_AVAILABLE", True)
        positions = torch.zeros(2, dtype=torch.long)
        query = torch.randn(2, 8)
        key = torch.randn(2, 8)
        cache = rope_ops.compute_cos_sin_cache(10000.0, 4, 4)  # rot_dim=4 != head_size=8
        with pytest.raises(AssertionError, match="full-head rotation"):
            rope_ops.rotary_embedding(positions, query, key, 8, cache)

    def test_raises_on_non_multiple_of_head_size(self, monkeypatch):
        monkeypatch.setattr(rope_ops, "_TRITON_AVAILABLE", True)
        positions = torch.zeros(2, dtype=torch.long)
        query = torch.randn(2, 12)  # not a multiple of head_size=8
        key = torch.randn(2, 8)
        cache = rope_ops.compute_cos_sin_cache(10000.0, 8, 4)
        with pytest.raises(AssertionError, match="exact multiple of head_size"):
            rope_ops.rotary_embedding(positions, query, key, 8, cache)


class TestModuleWiring:
    def test_top_level_rotary_embedding_delegates(self, monkeypatch):
        called = {}

        def fake(positions, query, key, head_size, cos_sin_cache):
            called["args"] = (positions, query, key, head_size, cos_sin_cache)
            return "sentinel"

        monkeypatch.setattr(amd_tuned_torch.rope_ops, "rotary_embedding", fake)
        p, q, k, cache = object(), object(), object(), object()
        result = amd_tuned_torch.rotary_embedding(p, q, k, 8, cache)
        assert result == "sentinel"
        assert called["args"] == (p, q, k, 8, cache)

    def test_top_level_compute_cache_delegates(self, monkeypatch):
        called = {}

        def fake(base, rotary_dim, max_position_embeddings):
            called["args"] = (base, rotary_dim, max_position_embeddings)
            return "sentinel"

        monkeypatch.setattr(amd_tuned_torch.rope_ops, "compute_cos_sin_cache", fake)
        result = amd_tuned_torch.compute_rope_cos_sin_cache(10000.0, 64, 2048)
        assert result == "sentinel"
        assert called["args"] == (10000.0, 64, 2048)
