# Provenance

`host.cpp`, `kernel_fp16.cu`, `kernel_bf16.cu`, and `LICENSE` in this
directory are vendored from:

- Upstream: https://github.com/Repeerc/flash-attention-v2-RDNA3-minimal
- Path: `rocwmma_fattn/`
- Commit: `76d4433f1a06c7bb015c621ba9f8a3722e0489b4` (2024-08-25)
- License: Apache License 2.0 (see `LICENSE` in this directory)

## What was changed from upstream (Apache 2.0 §4(b) requires marking this)

- `host.cpp`: **not modified** -- copied verbatim, only a short
  attribution header comment added at the top pointing back here.
- `kernel_fp16.cu`, `kernel_bf16.cu`: **modified**. Beyond the
  attribution header, these carry correctness and performance fixes made
  against real gfx1100 hardware; each is marked with a `FIX` comment at
  the site. The correctness ones, in the order they were found:

  1. **Forward**: K/V's sequence dimension was never padded to a multiple
     of `Bc` even though the launch config and every `Kj`/`Vj` read assume
     a full `Tc*Bc` allocation (read past the end of K/V for any `n_kv`
     not a multiple of `Bc`, e.g. CLIP's 77-token cross-attention).
  2. **Backward**: `Q`/`K`/`V`/`O`/`dO`/`L` `.contiguous()` calls were
     commented out while every kernel does packed-layout pointer
     arithmetic.
  3. **Backward**: `dQ` never received the `ln(2)` rescale that converts
     out of the exp2-domain the softmax works in -- `mul_add_AT_B` applies
     it for `dK` but `mul_add_A_B`, the `dQ` GEMM, takes no scale argument
     at all. `dQ` came out `log2(e)` = 1.4427x too large in every shape
     and both dtypes. Now folded into `dSi` once, so `dK`'s GEMM scale
     drops to `1.0f`.
  4. **Backward**: `dQ` was accumulated by workgroups indexed by `Tc_j`
     while `dQi` is selected by `Tr_i`, so all `Tc` workgroups of a
     `(b, h)` read-modify-wrote the same `dQ` rows non-atomically and all
     but one block's contribution was lost. The kernel now runs as two
     passes over the same tile grid -- `dK`/`dV` indexed by `Tc_j`, `dQ`
     by `Tr_i` -- so both accumulations are workgroup-private. Costs one
     extra recomputation of `Si` and `dPi` per tile.
  5. **Backward**: `dO` was never padded in its *sequence* dimension
     (`Nq_pad_sz` is derived from Q's already-padded `n`, so it is always
     0) yet is read with the padded Q's strides -- an out-of-bounds read,
     and the wrong batch entirely for `b`/`h` > 0, at any `n` that is not
     a multiple of `Br`.
  6. **bf16**: `MAX_NUM` was `INFINITY`, so a fully-masked padded row
     evaluated `exp2f(-inf - -inf)` = NaN and wrote NaN into `l_i`, `O`'s
     padding and `L`; the backward reloads `L`, so `dK`/`dV` came back
     all-NaN for any shape whose `n`/`n_kv` was not a multiple of
     `Br`/`Bc`. Now a finite sentinel, matching the fp16 file. (Guarding
     the NaN after the fact does not work under `-Ofast`/`-ffast-math`,
     which imply `-ffinite-math-only`: an `l_i > 0.0f` guard was measured
     being compiled away. The `is_finite_f32` helper added alongside tests
     the exponent bits instead, for the guards that remain.)

  Plus forward performance work: parallelising the fp16 softmax epilogue
  across the workgroup, and a causal early-exit in the backward for tiles
  entirely above the diagonal.
- `FlashAttn.py` was **not** vendored verbatim. Its logic (the JIT
  `torch.utils.cpp_extension.load(...)` call, `FlashAttentionFunction`'s
  forward/backward `Br`/`Bc` selection) was reimplemented in
  `amd_tuned_torch/flash_attn_rocwmma_ops.py`, with two deliberate
  changes:
  1. The `sys.platform.startswith("win32")` / ZLUDA compatibility branch
     (`zluda_hijack_torch_hip_ext`) was dropped entirely --
     `amd_tuned_torch` is Linux-only (see the main README's "Build
     prerequisites"), so that branch would be dead code here.
  2. The JIT build is **lazy** (only triggered by `available()` or an
     actual attention call) instead of running eagerly at module-import
     time the way upstream's `FlashAttn.py` does at its top level -- so
     that `import amd_tuned_torch` never pays JIT-compile latency, or
     risks a build failure, for a backend nobody asked to use. See
     `amd_tuned_torch.enable_flash_attn_rocwmma`'s docstring for the
     opt-in rationale.

## Why this backend, not the alternative considered

`source/sd-webui-flash-attention2-rdna3-rocm` (a different vendored repo
in this project) also contains a hand-written rocWMMA flash-attention
kernel, but its Python integration code is an AUTOMATIC1111
webui-specific extension (imports `modules.*`/`ldm`/`sgm` directly, only
importable inside a running webui process) and, critically, that repo
ships with **no LICENSE file at all** -- default copyright applies, with
no clear grant to vendor its source. `rocwmma_fattn` here is
framework-agnostic, already shaped as a plain `torch.autograd.Function`,
and has a real Apache 2.0 grant, hence the choice.
