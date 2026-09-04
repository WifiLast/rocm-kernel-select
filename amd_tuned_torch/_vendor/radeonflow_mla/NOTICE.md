# Provenance

`amd_tuned_torch/mla_ops.py` reimplements the weight-absorption Multi-Head
Latent Attention (MLA) decode algorithm from:

  https://github.com/RadeonFlow/RadeonFlow_Kernels
  `tests/mla/submit/submission.py`'s `custom_kernel_impl` (and the
  `RoPE`/`precompute_rope_cache` helpers from `tests/mla/reference.py`),
  MIT License, Copyright (c) 2025 RadeonFlow. See LICENSE in this
  directory for the full license text.

RadeonFlow_Kernels was the Grand Prize Winner of AMD's 2025 Developer
Challenge (GPUMode "Inference Sprint", MLA problem), targeting MI300X
(gfx942/CDNA3). Only its MLA component is vendored here: the GEMM, MoE,
FP8-GEMM, and multi-GPU (dist-infer) kernels in that project are all
hardcoded to gfx942 (wavefront-64 MFMA tile shapes, XCD cache-swizzle
constants, and -- for FP8 -- matrix-core hardware RDNA3/gfx1100 does not
have) and were not ported, since MLA decode is the only piece of that
project implemented in plain PyTorch/ATen ops (einsum, softmax, linear)
rather than hand-written HIP/CDNA3 kernels, making it the only component
that runs on gfx1100 without modification.

What changed from upstream:
  - Removed all competition-harness-specific code: the `input_t`/`output_t`
    dataclasses, `Config`/`KVCache` wrapper classes, CUDA-graph capture
    (`custom_kernel`/`custom_kernel_step_1`/`_step_2`), and the benchmark's
    hardcoded global dimensions (`hidden_dim=7168`, `dq=1536`, etc., from
    `src/mla/mla.h` / `tests/mla/reference.py`'s module-level constants).
  - `mla_decode()` takes every dimension (n_heads, nope_dim, rope_dim,
    v_dim) and every projection weight as an explicit argument instead of
    reading them off a global `Config`/`InputParams` struct, so it isn't
    tied to one specific model's shape.
  - `precompute_rope_cache`'s hardcoded `rope_cache = precompute_rope_cache(6145, 64)`
    module-level global (sized for one benchmark's exact prefill length)
    was replaced with `_rope_cache()`, a grow-on-demand cache keyed by
    (rope_dim, theta, dtype, device) that reallocates only when a
    longer sequence is requested -- the usual rotary-embedding-cache
    pattern, needed here because upstream's version would silently
    produce wrong (too-short) cos/sin tables for any sequence longer than
    the one benchmark shape it was written for.
  - Dropped the `torch.compile` decorators upstream applies to each step;
    callers of `mla_decode()` can wrap it in `torch.compile` themselves.
    Not applied by default here so this stays a plain function usable
    without a compilation warmup cost on first call.

The algorithm itself (the "weight absorption" associative-law reordering
that lets decode attention run against the compressed kv_lora cache
directly, without ever materializing full per-head K/V tensors) is
unchanged from upstream -- see mla_ops.py's module docstring for what
that means concretely.

UNVALIDATED, same as amd_tuned_torch/_vendor/rocwmma_fattn/: this has not
been run against a real DeepSeek-V3-architecture model on any hardware
this project has access to. See enable/available notes in mla_ops.py.
