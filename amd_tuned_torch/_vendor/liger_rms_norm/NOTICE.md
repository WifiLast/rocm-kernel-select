# Provenance

`amd_tuned_torch/fused_norm_ops.py`'s `_fused_add_rmsnorm_bwd_kernel`
implements the RMSNorm backward FORMULA (not the kernel code itself,
which is written against this file's own fused-add calling convention)
from LinkedIn's Liger Kernel:

  https://github.com/linkedin/Liger-Kernel
  `src/liger_kernel/ops/rms_norm.py`'s `_rms_norm_backward_kernel`
  (the per-row, non-block-parallel variant -- Liger also has a
  `_block_rms_norm_backward_kernel` grid-parallel-with-SM-count variant
  this port did not use, see below).
  BSD 2-Clause License, Copyright (c) 2024 LinkedIn Corporation. See
  LICENSE in this directory for the full license text. That file's own
  header additionally credits Unsloth (Apache-2.0) as the origin of the
  RMSNorm kernel approach it's based on -- carried forward here for the
  same reason Liger itself carries it.

This is distinct from `amd_tuned_torch/_vendor/liger_fused_ce/`, which
covers `fused_ce_ops.py`'s port of a DIFFERENT Liger file
(`fused_linear_cross_entropy.py`/`cross_entropy.py`) -- both are Liger
Kernel, but different source files, ported into different modules here,
so each gets its own attribution directory per this package's existing
one-vendor-dir-per-source-component convention.

The formula (from that kernel's own docstring):

    dx = (1/RMS) * [dy*(w+offset) - (1/N)*(1/RMS^2)*((dy*(w+offset)) . x)*x]
    dw = sum over rows of (dy * (x/RMS))

adapted here as (offset=0, since fused_add_rms_norm has no Gemma-style
weight offset; "x" renamed "hidden" since it's `x_input + residual`, not
a bare input):

    dhidden = rstd*(dout*w) - (1/N)*rstd^3*((dout*w) . hidden)*hidden
    dweight = sum over rows of (dout * hidden * rstd)

What changed from upstream, and why:

  - Liger's kernel handles THREE "casting modes" (llama/gemma/none,
    differing in which intermediate values get upcast to fp32) and an
    optional Gemma-style weight `offset`. This port implements only the
    "none, offset=0" case -- fused_add_rms_norm's own forward already
    always accumulates the residual-add and variance in fp32 regardless
    of input dtype (see this file's forward kernel), which is closest to
    Liger's "llama" mode (only the reduction is fp32) but not identical
    to any one of Liger's three modes exactly; re-deriving which of the
    three would best match a specific downstream model's numerics was
    out of scope, since this package has no casting-mode-selecting
    call site to match against (Liger's modes exist to bit-match specific
    HuggingFace model implementations, which this package doesn't target).
  - Liger's own backward computes dW into a `(sm_count, n_cols)` scratch
    buffer partitioned by SM count (`rows_per_program`), avoiding atomics
    entirely, then sums that buffer down to `(n_cols,)` on the host
    afterward -- a real performance-motivated design (atomic contention
    under a full-size grid can be significant at large batch*seq). This
    port instead accumulates dweight via `tl.atomic_add` directly from
    every row's program into one pre-zeroed fp32 `(n_cols,)` buffer -- a
    deliberate simplification: no `sm_count`/`rows_per_program`
    grid-partitioning logic to get right (and no hardware here to verify
    it against if introduced), at the cost of atomic contention this
    package hasn't measured. Revisit with Liger's SM-partitioned approach
    if profiling on real hardware shows this mattering.
  - The `+ dnew_residual` term in this port's dhidden computation has no
    Liger equivalent at all -- Liger's plain RMSNorm has one output (the
    normalized value) and one gradient input; this op's fused-add design
    has two outputs (normalized value AND the pre-normalization
    `hidden = x + residual`, returned as `new_residual` for the next
    layer), so its backward must also sum whatever gradient arrives
    through that second output path before splitting into dx/dresidual.
    See fused_norm_ops.py's module docstring for the full derivation.

UNVALIDATED, same as fused_norm_ops.py's own forward kernel: this
backward formula is copied faithfully from a production, AMD-CI-tested
library, but the ADAPTATION to this file's two-output calling convention
and the atomic-accumulation rewrite are new code with no test coverage
beyond tests_hardware/test_fused_norm_ops.py's gradient checks -- run
those on real gfx1100 hardware before trusting this for training.
