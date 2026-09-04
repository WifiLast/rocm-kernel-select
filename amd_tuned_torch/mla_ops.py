"""Multi-Head Latent Attention (MLA) decode -- the DeepSeek-V3-style
attention variant, with the "weight absorption" optimization that lets
single-token decode run entirely against the compressed KV cache
(kv_lora_dim, e.g. 512) instead of ever materializing full per-head K/V
tensors (n_heads * nope_dim, e.g. 128 * 128 = 16384). Ported from
RadeonFlow_Kernels (see amd_tuned_torch/_vendor/radeonflow_mla/NOTICE.md
for exactly what was changed from upstream and why only this component --
not that project's GEMM/MoE/FP8-GEMM/dist-infer kernels -- was ported: MLA
is implemented there in plain PyTorch/ATen ops, so it runs on gfx1100
unmodified, whereas the rest of that project is hand-written CDNA3/gfx942
HIP (and, for FP8-GEMM, depends on matrix-core hardware gfx1100 lacks).

Unlike every other module in this package, mla_ops has no F.* stock op to
monkeypatch -- there is no torch.nn.functional.multi_head_latent_attention
to intercept -- so this is a manually-callable helper, never auto-patched,
same shape as triton_kernels_ops.swiglu_fused(): a model that uses MLA
(only DeepSeek-V3-architecture models and close derivatives do) calls
mla_decode() directly from its own attention layer.

UNVALIDATED: adapted from a competition submission's reference
implementation, not verified against a real model's weights/activations
on any hardware this project has access to. Before relying on this:
  1. Compare its output against a from-scratch (non-absorbed) MLA decode
     step -- materialize k_nope/v explicitly via KV_proj_up and run
     ordinary multi-head attention -- numerically, for your actual model's
     dimensions and dtype. The weight-absorption reordering is an exact
     algebraic identity (associativity of matrix multiplication), so any
     mismatch beyond dtype-rounding means a porting bug, not an
     approximation.
  2. Benchmark against whatever attention path your model currently uses
     for the same shapes -- decode-only (seq_len=1) attention against a
     long KV cache is a narrow, latency-sensitive case; there's no
     tools/kernelgen/-style autotuning for it here.

Only decode (query sequence length 1) is implemented -- prefill (building
the initial KV cache from a full prompt) is a different, compute-bound
shape this weight-absorption trick does not help (absorbing kup_nope into
q_nope turns an O(nope_dim) contraction into an O(kv_lora_dim) one, which
is a win only when kv_lora_dim < nope_dim * kv_len, i.e. long decode
context -- for prefill, where kv_len == query length, plain per-head
attention is cheaper). Use ordinary attention (e.g. this package's
flash_attn_rocwmma_ops, or stock F.scaled_dot_product_attention) for
prefill and mla_decode() only for the token-at-a-time decode loop after.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def available() -> bool:
    """Always True: mla_decode() is plain PyTorch (linear/einsum/softmax),
    with no HIP/CK/hipBLASLt/Triton dependency and no separate build step,
    so it runs on any device/dtype combination stock PyTorch itself
    supports -- gfx1100 included. Kept as a function, rather than callers
    checking nothing at all, for the same reason every other backend in
    this package exposes one: a uniform way to ask "can I use this here"
    before wiring a tier in, even when the answer never varies."""
    return True


_ROPE_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_ROPE_CACHE_MIN_LEN = 256


def _rope_cache(seq_len: int, dim: int, theta: float, dtype: torch.dtype,
                 device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables for NeoX-style (rotate-half) RoPE, grown on demand
    and cached by (dim, theta, dtype, device) -- NOT by seq_len, unlike
    upstream's `rope_cache = precompute_rope_cache(6145, 64)` module-level
    global (sized for exactly one benchmark's prefill length; any longer
    sequence would silently index past the end of that table). Regrows
    (replacing the cached tables) only when a longer sequence is
    requested, the usual rotary-embedding-cache growth pattern."""
    key = (dim, theta, dtype, device)
    cos, sin = _ROPE_CACHE.get(key, (None, None))
    if cos is None or cos.shape[0] < seq_len:
        grown_len = max(seq_len, _ROPE_CACHE_MIN_LEN)
        position = torch.arange(grown_len, device=device, dtype=torch.float32)
        inv_freq = theta ** (-torch.arange(0, dim // 2, device=device, dtype=torch.float32) / (dim // 2))
        angles = torch.outer(position, inv_freq)
        angles = torch.cat([angles, angles], dim=-1)
        cos, sin = angles.cos().to(dtype), angles.sin().to(dtype)
        _ROPE_CACHE[key] = (cos, sin)
    return cos[:seq_len], sin[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + _rotate_half(x) * sin


def mla_decode(
    x: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_len: int,
    q_proj_down_weight: torch.Tensor,
    q_proj_up_weight: torch.Tensor,
    kv_proj_down_weight: torch.Tensor,
    kv_proj_up_weight: torch.Tensor,
    wo_weight: torch.Tensor,
    n_heads: int,
    nope_dim: int,
    rope_dim: int,
    v_dim: int,
    rope_theta: float = 10000.0,
) -> torch.Tensor:
    """One decode step (query sequence length 1) of DeepSeek-V3-style MLA,
    against a KV cache stored in its compressed (kv_lora_dim + rope_dim)
    form -- never expanded to full per-head K/V.

    x: (batch, 1, hidden_dim) -- the current token's hidden state.
    kv_cache: (batch, max_seq_len, kv_lora_dim + rope_dim), pre-allocated
        by the caller. This call writes the new token's projected
        kv_lora/k_rope into kv_cache[:, kv_len - 1] and reads
        kv_cache[:, :kv_len] back out -- i.e. kv_cache[:, :kv_len - 1]
        must already hold every prior token's projection.
    kv_len: sequence length including the current token (so the new
        token's slot is kv_cache[:, kv_len - 1]).
    q_proj_down_weight: (q_lora_dim, hidden_dim)
    q_proj_up_weight: (n_heads * (nope_dim + rope_dim), q_lora_dim)
    kv_proj_down_weight: (kv_lora_dim + rope_dim, hidden_dim)
    kv_proj_up_weight: (n_heads * (nope_dim + v_dim), kv_lora_dim)
    wo_weight: (hidden_dim, n_heads * v_dim)

    Returns (batch, 1, hidden_dim). kv_cache is updated in place.
    """
    hidden_dim = x.shape[-1]
    kv_lora_dim = kv_proj_up_weight.shape[1]
    q_lora_dim = q_proj_up_weight.shape[1]
    assert q_proj_up_weight.shape[0] == n_heads * (nope_dim + rope_dim), \
        "q_proj_up_weight shape does not match n_heads * (nope_dim + rope_dim)"
    assert kv_proj_up_weight.shape[0] == n_heads * (nope_dim + v_dim), \
        "kv_proj_up_weight shape does not match n_heads * (nope_dim + v_dim)"
    assert kv_cache.shape[-1] == kv_lora_dim + rope_dim, \
        "kv_cache last dim does not match kv_lora_dim + rope_dim"

    # Step 1: down-project into the compressed query/KV latents, and
    # write this token's KV latent into the cache.
    q_lora = F.linear(x, q_proj_down_weight)
    kv_and_rope = F.linear(x, kv_proj_down_weight)
    kv_cache[:, kv_len - 1: kv_len, :] = kv_and_rope
    kv_lora, k_rope = kv_cache[:, :kv_len, :].split([kv_lora_dim, rope_dim], dim=-1)

    # Step 2: up-project the query, then absorb KV_proj_up's nope half
    # (kup_nope) into the query itself -- q_nope @ kup_nope^T folded in
    # here, rather than materializing k_nope = kv_lora @ kup_nope^T for
    # every cached token. Associativity of matmul makes this an exact
    # identity, not an approximation (see module docstring, point 1).
    qup = q_proj_up_weight.view(n_heads, nope_dim + rope_dim, q_lora_dim)
    kup_nope, vup = kv_proj_up_weight.view(n_heads, nope_dim + v_dim, kv_lora_dim).split(
        [nope_dim, v_dim], dim=-2
    )
    q_nope, q_rope = torch.einsum("b s l, h d l -> b s h d", q_lora, qup).split(
        [nope_dim, rope_dim], dim=-1
    )
    q_absorb = torch.einsum("b s h d, h d l -> b s h l", q_nope, kup_nope)
    q_rope = q_rope.permute(0, 2, 1, 3)      # (batch, n_heads, 1, rope_dim)
    q_absorb = q_absorb.permute(0, 2, 1, 3)  # (batch, n_heads, 1, kv_lora_dim)

    # Step 3: RoPE on the rope half, then attention scores computed
    # directly against the compressed kv_lora/k_rope cache.
    cos, sin = _rope_cache(kv_len, rope_dim, rope_theta, x.dtype, x.device)
    q_rope = _apply_rope(q_rope, cos[kv_len - 1: kv_len], sin[kv_len - 1: kv_len])
    k_rope = _apply_rope(k_rope, cos[:kv_len], sin[:kv_len])

    attn_nope = torch.einsum("b h s l, b p l -> b h s p", q_absorb, kv_lora)
    attn_rope = torch.einsum("b h s d, b p d -> b h s p", q_rope, k_rope)
    attention = torch.softmax((attn_nope + attn_rope) / math.sqrt(nope_dim + rope_dim), dim=-1)

    # Step 4: weighted-sum against kv_lora (still compressed), and only
    # now expand to v_dim via vup -- the output-side mirror of step 2's
    # absorption, again an exact associativity identity.
    o = torch.einsum("b h s p, b p l -> b h s l", attention, kv_lora)
    o = torch.einsum("b h s l, h v l -> b h s v", o, vup)
    wo = wo_weight.view(hidden_dim, n_heads, v_dim)
    return torch.einsum("b h s v, d h v -> b s d", o, wo)
