# Provenance

`amd_tuned_torch/rope_ops.py`'s `_rotary_embedding_kernel`/`rotary_embedding_launcher`
are vendored, with only cosmetic changes (renamed a couple of locals,
folded the module docstring into this package's convention), from:

  https://github.com/stackav-oss/conch
  `conch/kernels/embedding/rotary_embedding.py` (itself a "port of vllm
  rotary_embedding to Triton", per that file's own docstring), and
  `conch/reference/embedding/rotary_embedding.py`'s `compute_cos_sin_cache`/
  `_compute_inv_freq` (ported as `rope_ops.compute_cos_sin_cache`).
  Apache License 2.0, Copyright 2025 Stack AV Co. See LICENSE in this
  directory for the full license text.

Conch is a Triton kernel library that explicitly supports AMD (its own
README lists "AMD MI300X, ROCm 6.2.4" as a validated platform, alongside
Nvidia) -- this kernel has no fp8/tensor-core dependency at all (plain
elementwise load/store/multiply/add), so it carries none of the
CDNA3-vs-RDNA3 matrix-core portability risk that blocked porting
RadeonFlow_Kernels' FP8-GEMM (see
amd_tuned_torch/_vendor/radeonflow_mla/NOTICE.md for that project's
comparison). Still UNVALIDATED specifically on gfx1100/RX 7900 XTX, since
Conch's own README only claims MI300X as the tested AMD target -- see
rope_ops.py's module docstring for what to verify before trusting this.

What changed from upstream:
  - `rotary_embedding_launcher` dropped conch's own `assert is_neox` /
    batch-shape assertions in favor of this package's usual
    assert-with-message style; behavior is unchanged (NeoX-style rotation
    only, unbatched [num_tokens, num_heads*head_size] query/key layout
    only -- both are upstream's own limitations, not something this port
    removed capability from).
  - Nothing else in the kernel itself changed -- the block/grid layout,
    the per-token/per-head indexing math, and the in-place
    query/key mutation semantics are identical to upstream.
  - `compute_cos_sin_cache` is included as-is (a few lines of plain
    torch.arange/einsum/cos/sin, no kernel involved) because the Triton
    kernel is unusable without a cache built in this exact
    [max_position, rotary_dim] `cat([cos, sin], dim=-1)` layout, and there
    is no other cos/sin-cache helper anywhere in this package to reuse
    instead (mla_ops.py's own `_rope_cache` builds a *different* on-demand
    grow-by-position cache shape for a different calling convention --
    see rope_ops.py's module docstring for why these two RoPE
    implementations were not unified).
  - ADDED (not present upstream): a real backward pass. Neither Conch's
    kernel nor the vLLM op it's a port of has one -- both are forward-
    only/inference-oriented. `_rotary_embedding_kernel` gained an
    `INVERSE: tl.constexpr` parameter (negates `sin` before the same
    rotation math runs) and `rotary_embedding()` is now wrapped in a
    `_RotaryEmbeddingFunction(torch.autograd.Function)`; backward re-runs
    the identical kernel with `INVERSE=True` on the incoming
    grad_query/grad_key, since a rotation matrix's inverse is its
    transpose. This is the same forward/backward relationship Megatron-
    LM's own MLA RoPE kernels (`fused_mla_yarn_rope_apply.py`, surveyed
    but not ported into this package -- see
    amd_tuned_torch/_vendor/megatron_swiglu/NOTICE.md for why) use
    between their own forward and backward kernels, applied here to the
    much simpler full-head (non-MLA, non-THD, non-context-parallel) case
    this module already covers.

NOT ported: conch's `envs`/`platforms` module dependencies (this
package's own `available()`-gating convention replaces conch's
`CONCH_ENABLE_VLLM`/`current_platform` checks, which exist to select
between a vLLM CUDA reference and a pure-PyTorch reference -- irrelevant
here since this package has no vLLM dependency at all), and conch's
`is_neox=False` / partial-rotary-dim code paths that exist in conch's own
*reference* implementation (`_rotary_embedding_pytorch_ref`) but were
never implemented in conch's own Triton kernel either -- see
rope_ops.py's module docstring for this limitation.
