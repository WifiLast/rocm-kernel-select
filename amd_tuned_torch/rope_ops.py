"""Fused rotary positional embedding (RoPE), NeoX-style, applied in-place
to query and key -- vendored from Conch (see
amd_tuned_torch/_vendor/conch_rope/NOTICE.md for exactly what changed and
why only this kernel, not the rest of that project, was ported).

Unlike mla_ops.py's own `_apply_rope`/`_rope_cache` (four unfused
elementwise PyTorch ops -- multiply, rotate-half via `cat`, multiply,
add -- run twice per decode step), this is a single Triton kernel pass
per call, and it's usable by any NeoX-style-RoPE model (Llama, Qwen,
Mistral, ...), not just MLA's specific calling convention. It is
DELIBERATELY NOT used to replace mla_ops._apply_rope in this pass: the
two have incompatible shape conventions --

  - mla_ops._apply_rope: q_rope is (batch, n_heads, 1, rope_dim) and
    k_rope is (batch, kv_len, rope_dim) with NO per-head dimension at all
    (MLA shares one k_rope across every head, expanded via einsum
    broadcasting rather than physically replicated) -- see mla_ops.py's
    module docstring.
  - rope_ops.rotary_embedding (this module): query/key are BOTH
    [num_tokens, num_heads * head_size] with a per-head stride baked into
    the kernel's addressing math (see _rotary_embedding_kernel), and every
    head is assumed to need rotation -- there is no "one shared,
    non-per-head key" case this kernel's launcher can express.

Reconciling those would need either a second, MLA-shaped kernel variant or
a shape-adapting wrapper (materializing k_rope's missing head dimension,
which defeats the whole point of MLA's compressed-KV-cache design) --
out of scope for this port. Use this module directly from an ordinary
(non-MLA) attention layer's Q/K projection instead.

NeoX-style, full-head rotation ONLY (rotary_dim == head_size) -- upstream
Conch's own Triton kernel has no partial-rotary-dim support either (only
its separate *reference* implementation does, via a query_pass/key_pass
split this kernel's addressing math has no equivalent for), so a model
using partial rotary_pct < 1.0 (e.g. some GPT-NeoX/Phi configurations)
cannot use this as-is -- see rotary_embedding()'s assertion.

available() gates on `triton` importability, same reasoning as
fused_norm_ops.py: triton is not a declared dependency of this package
anywhere (no requirements.txt/pyproject.toml, and setup.py never
mentions it).

HAS A REAL BACKWARD PASS -- added in this package, not present in
upstream Conch (whose own Triton kernel, like vLLM's op it's a port of,
is forward-only; there is no `rotary_embedding_backward` anywhere in
Conch). A rotation's inverse is its transpose (rotation matrices are
orthogonal), so the gradient w.r.t. query/key is exactly the same
rotation applied to the incoming gradient with `sin` negated -- see
_rotary_embedding_kernel's INVERSE parameter and
_RotaryEmbeddingFunction below. This makes rotary_embedding() usable
inside an ordinary training loop, unlike fused_add_rms_norm (still
forward-only; RMSNorm's backward needs the row's variance/mean, which
this rotation-only kernel doesn't compute at all, so the same trick
doesn't transfer there).

UNVALIDATED ON GFX1100 specifically: Conch's own README lists AMD MI300X
(CDNA3, ROCm 6.2.4) as its tested AMD platform, not RX 7900 XTX/RDNA3.
Unlike RadeonFlow_Kernels' FP8-GEMM, this kernel has no matrix-core/fp8
dependency at all -- plain elementwise tl.load/tl.store/multiply/add --
so there is no known *architectural* reason it wouldn't work on gfx1100,
but "no known reason it would fail" is not the same as "verified." Before
relying on this:
  1. Compare its output against compute_cos_sin_cache +
     torch.rotary-by-hand (or simply against mla_ops-style rotate-half
     RoPE restricted to the full-head case, which is mathematically the
     same NeoX-style rotation) numerically, for your actual model's
     head_size/num_heads/dtype.
  2. Benchmark against whatever unfused RoPE application your model
     currently does -- there's no tools/kernelgen/-style autotuning here,
     and the block size is fixed to next_pow2(head_size) with no tuning
     beyond that.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def available() -> bool:
    return _TRITON_AVAILABLE


if _TRITON_AVAILABLE:

    @triton.jit
    def _rotary_embedding_kernel(
        positions_ptr, query_ptr, key_ptr, cos_sin_cache_ptr,
        rot_dim, query_stride, key_stride,
        num_heads, num_kv_heads, head_size,
        cxpr_block_size: tl.constexpr,
        INVERSE: tl.constexpr = False,
    ):
        # One program per token. cos_sin_cache_ptr is [max_position,
        # rot_dim] with rot_dim split into [cos (rot_dim/2) | sin
        # (rot_dim/2)] -- see compute_cos_sin_cache below for the layout
        # this expects.
        #
        # INVERSE applies the rotation's inverse (transpose, since a
        # rotation matrix is orthogonal: R(theta)^-1 == R(theta)^T ==
        # R(-theta)) instead of the rotation itself -- i.e. this same
        # kernel IS rotary_embedding's backward pass, just called with
        # sin negated. See _RotaryEmbeddingFunction.backward below for
        # why that's exactly the gradient this op needs, not merely a
        # convenient reuse.
        token_idx = tl.program_id(0)
        pos = tl.load(positions_ptr + token_idx)
        rot_cache_ptr = cos_sin_cache_ptr + pos * rot_dim

        embed_dim = rot_dim // 2
        cos_ptr = rot_cache_ptr
        sin_ptr = rot_cache_ptr + embed_dim

        # Query: num_heads * embed_dim rotation pairs for this token,
        # walked cxpr_block_size at a time.
        nq = num_heads * embed_dim
        query_token_offset = token_idx.to(tl.int64) * query_stride
        i = tl.arange(0, cxpr_block_size)
        for _ in tl.range(0, nq, cxpr_block_size):
            head_idx = i // embed_dim
            token_head = query_token_offset + head_idx * head_size
            rot_offset = i % embed_dim

            x_index = rot_offset
            y_index = rot_offset + embed_dim
            query_offset = query_ptr + token_head
            mask = i < nq
            x = tl.load(query_offset + x_index, mask=mask)
            y = tl.load(query_offset + y_index, mask=mask)
            cos = tl.load(cos_ptr + x_index, mask=mask)
            sin = tl.load(sin_ptr + x_index, mask=mask)
            if INVERSE:
                sin = -sin
            x_rot = x * cos - y * sin
            y_rot = y * cos + x * sin
            tl.store(query_offset + x_index, x_rot, mask=mask)
            tl.store(query_offset + y_index, y_rot, mask=mask)
            i += cxpr_block_size

        # Key: same shape of loop, independent head count (GQA/MQA: fewer
        # KV heads than query heads).
        nk = num_kv_heads * embed_dim
        key_token_offset = token_idx.to(tl.int64) * key_stride
        j = tl.arange(0, cxpr_block_size)
        for _ in tl.range(0, nk, cxpr_block_size):
            head_idx = j // embed_dim
            token_head = key_token_offset + head_idx * head_size
            rot_offset = j % embed_dim

            x_index = rot_offset
            y_index = rot_offset + embed_dim
            key_offset = key_ptr + token_head
            mask = j < nk
            x = tl.load(key_offset + x_index, mask=mask)
            y = tl.load(key_offset + y_index, mask=mask)
            cos = tl.load(cos_ptr + x_index, mask=mask)
            sin = tl.load(sin_ptr + x_index, mask=mask)
            if INVERSE:
                sin = -sin
            x_rot = x * cos - y * sin
            y_rot = y * cos + x * sin
            tl.store(key_offset + x_index, x_rot, mask=mask)
            tl.store(key_offset + y_index, y_rot, mask=mask)
            j += cxpr_block_size


def compute_cos_sin_cache(base: float, rotary_dim: int, max_position_embeddings: int) -> torch.Tensor:
    """[max_position_embeddings, rotary_dim] cache, laid out as
    cat([cos, sin], dim=-1) -- the layout rotary_embedding()'s kernel
    expects. Build once (e.g. at model init) and reuse across every
    decode/prefill step; recomputing per call defeats the point of a
    fused kernel."""
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    t = torch.arange(max_position_embeddings, dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


if _TRITON_AVAILABLE:

    class _RotaryEmbeddingFunction(torch.autograd.Function):
        """Real backward pass: RoPE's forward is `y = R(theta) @ x` for
        each rotated pair, and a rotation matrix is orthogonal
        (R(theta)^-1 == R(theta)^T == R(-theta)), so the gradient w.r.t.
        x is exactly `R(-theta) @ grad_y` -- the SAME kernel, called again
        with INVERSE=True (sin negated), not a separately-derived
        computation. This is the identical relationship Megatron-LM's own
        MLA RoPE kernels (fused_mla_yarn_rope_apply.py) use between their
        forward and backward kernels."""

        @staticmethod
        def forward(ctx, positions, query, key, head_size, cos_sin_cache):
            rot_dim = cos_sin_cache.shape[-1]
            num_heads = query.shape[-1] // head_size
            num_kv_heads = key.shape[-1] // head_size
            query_stride = query.stride(-2)
            key_stride = key.stride(-2)
            block = triton.next_power_of_2(head_size)
            _rotary_embedding_kernel[(query.shape[0],)](
                positions, query, key, cos_sin_cache,
                rot_dim, query_stride, key_stride,
                num_heads, num_kv_heads, head_size,
                block, INVERSE=False,
            )
            ctx.mark_dirty(query, key)
            ctx.save_for_backward(positions, cos_sin_cache)
            ctx.head_size = head_size
            ctx.rot_dim = rot_dim
            ctx.num_heads = num_heads
            ctx.num_kv_heads = num_kv_heads
            ctx.block = block
            return query, key

        @staticmethod
        def backward(ctx, grad_query, grad_key):
            positions, cos_sin_cache = ctx.saved_tensors
            # In-place, same convention as fused_ce_ops._rescale_grads --
            # grad_query/grad_key are freshly produced by the next layer's
            # backward with nothing else depending on their pre-rotation
            # values, so there is nothing to preserve by allocating new
            # output tensors instead.
            _rotary_embedding_kernel[(grad_query.shape[0],)](
                positions, grad_query, grad_key, cos_sin_cache,
                ctx.rot_dim, grad_query.stride(-2), grad_key.stride(-2),
                ctx.num_heads, ctx.num_kv_heads, ctx.head_size,
                ctx.block, INVERSE=True,
            )
            return None, grad_query, grad_key, None, None


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply NeoX-style rotary embedding to query and key IN PLACE
    (matches upstream vLLM/Conch semantics -- the same tensors passed in
    are mutated and also returned, for call-site convenience), with a
    real backward pass (see _RotaryEmbeddingFunction above).

    positions: [num_tokens] token position indices into cos_sin_cache.
    query: [num_tokens, num_heads * head_size].
    key: [num_tokens, num_kv_heads * head_size] -- num_kv_heads may differ
        from num_heads (GQA/MQA), derived from key.shape[-1] // head_size.
    head_size: must equal cos_sin_cache.shape[-1] (full-head rotation
        only -- see module docstring for why there's no partial-rotary
        support here).
    cos_sin_cache: [max_position, head_size] from compute_cos_sin_cache().

    Both query and key must be 2D (unbatched: batch and sequence already
    flattened into num_tokens) -- reshape a (batch, seq, num_heads,
    head_size) tensor to (batch*seq, num_heads*head_size) before calling,
    same convention as vLLM's own rotary_embedding op."""
    if not available():
        raise RuntimeError("rope_ops.rotary_embedding: triton is not importable")
    assert positions.dim() == 1, "positions must be 1D [num_tokens]"
    assert query.dim() == 2 and key.dim() == 2, "query/key must be 2D [num_tokens, num_heads*head_size]"
    assert query.shape[0] == key.shape[0] == positions.shape[0], "num_tokens must match across positions/query/key"
    rot_dim = cos_sin_cache.shape[-1]
    assert rot_dim == head_size, (
        "rope_ops only supports full-head rotation (rotary_dim == head_size); "
        f"got cos_sin_cache rot_dim={rot_dim}, head_size={head_size}"
    )
    assert query.shape[-1] % head_size == 0 and key.shape[-1] % head_size == 0, \
        "query/key last dim must be an exact multiple of head_size"

    return _RotaryEmbeddingFunction.apply(positions, query, key, head_size, cos_sin_cache)
