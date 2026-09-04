# Provenance

`amd_tuned_torch/fused_ce_ops.py` is adapted from LinkedIn's Liger Kernel:

  https://github.com/linkedin/Liger-Kernel
  `src/liger_kernel/ops/cross_entropy.py` (`liger_cross_entropy_kernel`,
  `LigerCrossEntropyFunction`) and
  `src/liger_kernel/ops/fused_linear_cross_entropy.py`
  (`fused_linear_cross_entropy_forward`/`_backward`,
  `LigerFusedLinearCrossEntropyFunction`).
  BSD 2-Clause License, Copyright (c) 2024 LinkedIn Corporation. See
  LICENSE in this directory for the full license text.

Liger-Kernel's own `NOTICE`/`licenses/` additionally credit
`https://github.com/mgmalek/efficient_cross_entropy` (MIT) as the
reference algorithm this kernel implements (computing the logits
gradient in the forward pass, before the full B*T x V logits tensor
would otherwise need to be materialized for backward) -- carried forward
here for the same reason Liger-Kernel itself carries it. Liger-Kernel's
`ops/utils.py` also separately credits Unsloth (Apache-2.0) for the
`is_hip()`-gated num_warps tuning idea this port's kernel launches reuse.

Liger-Kernel has official AMD support (README: "2024/11/6 v0.4.0: Full
AMD support", credits EmbeddedLLM for AMD stabilization work, runs
dedicated MI300 CI) using the same generic Triton kernel this port
copies -- no CUDA-only intrinsics, no fp8/tensor-core dependency, only an
`is_hip()` check that tunes `num_warps` (16 vs 32). UNVALIDATED
specifically on gfx1100/RX 7900 XTX (Liger's own CI targets MI300/CDNA3,
not RDNA3) -- see fused_ce_ops.py's module docstring for what to verify.

What was deliberately NOT ported, and why (this is a large, heavily
feature-flagged kernel upstream -- porting all of it would multiply the
testing/maintenance surface for capability this package has no use for):

  - Multi-backend dispatch (`ce_impl`/`ce_mode`, `liger_kernel.backends.dispatch`):
    upstream can route to alternate NVIDIA-only CuTeDSL/cuTile
    implementations on Hopper/Blackwell. This package has exactly one
    backend (Triton), so that whole dispatch layer has nothing to
    dispatch to and was dropped entirely -- fused_ce_ops.py always uses
    the plain Triton kernel upstream calls its `ce_impl=None` default.
  - `return_z_loss` (PaLM-style z-loss auxiliary regularizer,
    `lse_square_scale`), `return_token_accuracy`, `return_predicted_tokens`:
    training-stability/metrics features orthogonal to the core
    memory-saving mechanism. Each interleaves extra branches into the
    online-softmax kernel loop upstream; dropping all three keeps the
    ported kernel's control flow close to a plain cross-entropy backward
    derivation, easier to verify against a from-scratch reference.
  - `use_token_scaling`, `accum_dtype` (as a user-facing parameter),
    `token_grad_output`/`compute_gradients`/`weight_requires_grad`
    re-entrant machinery, and `reduction="none"` support: upstream's
    `reduction="none"` defers gradient computation to backward (which
    re-invokes forward with `token_grad_output` set, recomputing the
    chunked logits) because the per-token upstream gradient isn't known
    until backward runs and must be folded in before the weight/bias
    projections sum over tokens. This port only supports
    `reduction="mean"`/`"sum"`, where gradients are fully known at
    forward time -- the entire re-entrant deferred-gradient code path
    (roughly half of upstream's `LigerFusedLinearCrossEntropyFunction`)
    does not exist here.
  - `accum_dtype=None`'s upstream default (grad_weight/grad_bias
    accumulate in the parameter's own dtype, e.g. bf16, chunk by chunk)
    was replaced with an unconditional fp32 accumulator (matching what
    upstream calls `accum_dtype=torch.float32`, its own recommended
    setting "if training is unstable") -- always-fp32 accumulation avoids
    compounding bf16/fp16 rounding error across `num_chunks` partial
    sums, at the cost of the small amount of memory an fp32-sized
    grad_weight/grad_bias buffer needs over a bf16-sized one (V*H
    elements -- negligible next to the B*T*V logits tensor this whole
    kernel exists to avoid materializing). Not configurable in this port;
    add an `accum_dtype` parameter back if a future model needs the
    lower-memory bf16-accumulator path.
  - The Ampere+-only `torch.addmm(..., out_dtype=...)` CUDA fast path
    (upstream lines ~284-304 of `fused_linear_cross_entropy.py`) was
    dropped -- it is gated on `torch.cuda.get_device_capability(...)[0] >= 8`
    and would never fire on a ROCm device (`get_device_capability` isn't
    the ROCm-relevant check at all); the portable `torch.mm(...).float()`
    fallback upstream itself falls through to is the only path this port
    implements.
