# Provenance

`amd_tuned_torch/swiglu_ops.py` is adapted from NVIDIA Megatron-LM:

  https://github.com/NVIDIA/Megatron-LM
  `megatron/core/fusions/fused_bias_swiglu.py` (`swiglu`, `bias_swiglu`,
  `swiglu_back`, `bias_swiglu_back`, `clamped_swiglu`,
  `bias_clamped_swiglu`, `clamped_swiglu_back`,
  `bias_clamped_swiglu_back`, `BiasSwiGLUFunction`, `SwiGLUFunction`).
  BSD-3-Clause License, Copyright (c) 2024, NVIDIA CORPORATION. See
  LICENSE in this directory for the full license text.

Unlike every other Triton kernel in this package, this file has NO Triton
dependency at all -- upstream's own implementation is plain PyTorch
wrapped in `@jit_fuser` (Megatron's own decorator, `torch.jit.script` by
default, upgraded to `torch.compile` on torch>=2.2). This port drops that
decorator entirely (see below) rather than reimplement `jit_fuser`, so
`amd_tuned_torch.swiglu_ops.available()` always returns True -- there is
nothing to gate on.

Megatron-LM upstream (the checkout this was ported from) makes NO ROCm
support claim and has zero `is_hip`/ROCm-specific code anywhere in
`megatron/core/fusions/` -- this is unlike Liger-Kernel (official AMD
support, dedicated MI300 CI) or Conch (AMD listed as a supported
platform). That said, this specific file has no CUDA/apex dependency
either (unlike `fused_layer_norm.py`/`fused_softmax.py` in the same
directory, which hard-require `apex`'s compiled CUDA extensions and were
NOT ported for exactly that reason -- see the investigation that led to
this port for the full comparison) -- it's ordinary elementwise PyTorch
math (chunk/silu/sigmoid/mul/cat), so there is no known architectural
reason it wouldn't produce identical results on gfx1100. UNVALIDATED
here regardless -- see swiglu_ops.py's module docstring.

What changed from upstream, and why:

  - Dropped `@jit_fuser` (i.e. `torch.jit.script`/`torch.compile`)
    entirely. Megatron applies this because `jit_fuser` is intended to
    fuse the chunk/silu/sigmoid/mul/cat sequence into fewer kernel
    launches at the TorchScript/Inductor level, on top of the memory
    savings the custom autograd.Function itself already provides.
    Applying `torch.jit.script` to a freshly-ported, unvalidated function
    adds a second, independent source of possible failure (scripting can
    reject or silently miscompile code that runs correctly in eager mode)
    on top of the porting risk this file already carries -- not
    something to add speculatively. A caller wrapping their whole model
    in `torch.compile` gets the same fusion Inductor would apply to any
    other Triton-less PyTorch function, without this file needing to
    force it.
  - Dropped `fp8_input_store` (stores the saved-for-backward activation
    in fp8 for memory savings) -- RDNA3 has no fp8 matrix-core support,
    and this feature exists purely as an activation-memory optimization,
    not a compute one, orthogonal to whether the actual SwiGLU math runs
    correctly; adding it back would need real validation that a
    HIP-side `torch.float8_e4m3fn` round-trip behaves as expected, which
    is out of scope here.
  - Dropped `cpu_offload_input` (activation-checkpointing/CPU-offload
    integration tied to Megatron's own offloading machinery) -- this
    package has no equivalent activation-offloading infrastructure to
    hook into.
  - Dropped SiTU-GLU (`situ_glu`/`gate_clamp_scale`/`linear_clamp_scale`)
    and the token-weighted MoE variant (`weighted_swiglu`,
    `WeightedSwiGLUFunction`) -- niche architecture-specific variants
    (SiTU-GLU is a specific stability trick; the weighted variant is for
    MoE expert-output scaling) this package has no current use for
    (cmp_ext_turing has no MoE support). Kept: plain SwiGLU and the
    hard-clamped variant (`clamp_value`), which is a more commonly
    encountered stability technique (e.g. some Grok/OLMoE-family models).
  - The bias gradient in upstream's `BiasSwiGLUFunction.backward` returns
    the UNREDUCED per-token gradient tensor for both `input` and `bias`
    (`return tmp, tmp, None, ...`) -- correct only if Megatron's calling
    convention always passes a `bias` tensor already broadcast/expanded
    to input's full shape at that call site (plausible given Megatron's
    distributed bias-parallel GEMM plumbing, but not verified from this
    file alone, and this package has no equivalent calling convention to
    match against). This port instead implements standard trailing-dim
    broadcasting (bias shape is a suffix of input's shape, exactly
    nn.Linear's own bias convention) and explicitly sums the gradient
    over every leading dimension bias didn't have -- the standard,
    unambiguous rule, safer than assuming upstream's context-specific
    shape convention holds here too.
