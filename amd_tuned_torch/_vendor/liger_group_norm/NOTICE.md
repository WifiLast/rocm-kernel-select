# Provenance

`amd_tuned_torch/liger_group_norm_ops.py` is a near-verbatim port of
LinkedIn's Liger Kernel:

  https://github.com/linkedin/Liger-Kernel
  `src/liger_kernel/ops/group_norm.py`
  commit `10acf80cf33d5b30f7979dac8e5b1c4d1d325522` (2026-08-13)
  BSD 2-Clause License, Copyright (c) 2024 LinkedIn Corporation. See
  LICENSE in this directory for the full license text.

Unlike `_vendor/liger_rms_norm/` (which re-derives Liger's RMSNorm
backward FORMULA against a different, fused-add calling convention this
package's own `fused_norm_ops.py` needed) or `_vendor/liger_fused_ce/`
(a substantially adapted port), this one is close to a direct copy: both
Triton kernels (`_group_norm_forward_kernel`, `_group_norm_backward_kernel`)
and the `group_norm_forward`/`group_norm_backward`/`LigerGroupNormFunction`
Python wrappers around them are carried over with no algorithmic changes,
because this package had no existing GroupNorm calling convention of its
own to reconcile against -- `amd_tuned_torch.ops.group_norm` (the
hand-written HIP kernel in `src/cuda/group_norm.cu`) is FORWARD-ONLY and
therefore not something a real-backward tier needs to match calling
conventions with.

WHAT CHANGED FROM UPSTREAM, AND WHY:
  - Renamed `LigerGroupNormFunction` -> `_LigerGroupNormFunction` (private,
    this module's own public entry point is the `group_norm()` function
    and `LigerGroupNorm` module class below it) and `group_norm_forward`/
    `group_norm_backward` -> `_group_norm_forward`/`_group_norm_backward`,
    matching this package's own leading-underscore convention for
    implementation helpers that aren't part of a module's public API.
  - Upstream's `is_npu_available()`/`infer_device()` device-dispatch
    helpers (from `liger_kernel.utils`) were inlined as a plain CUDA/ROCm
    check -- this package targets ROCm exclusively and has no NPU backend
    to dispatch for, so carrying that abstraction over would be dead code.
  - Upstream's triton-version-gated `rsqrt` import (`triton.language.extra.
    libdevice.rsqrt`, falling back to `.extra.cuda.libdevice` or
    `triton.language.math`) is kept as-is: it is already correctly
    backend-dispatching (the `extra.libdevice` path resolves per active
    Triton backend, CUDA or ROCm, without any AMD-specific branch needed)
    and changing it would be modifying working portability logic this
    project did not write and has no independent way to improve on.
  - Everything else (variable names, kernel grid shape, block-size
    selection, the atomic-add gradient accumulation for dW/dB) is
    unchanged.

WHY THIS PORT EXISTS AT ALL, GIVEN LIGER-KERNEL ALREADY SHIPS OFFICIAL,
CI-TESTED AMD SUPPORT (see its own README's AMD CI badge and "Full AMD
support" v0.4.0 release note -- this is NOT a "guess it probably works on
ROCm" port the way most of this project's other vendored kernels are).
This package's own convention (see `_vendor/liger_rms_norm/`,
`_vendor/liger_fused_ce/`, both predating this port) is to vendor the
specific Triton source it depends on rather than take a live `pip install
liger-kernel` dependency -- consistent with that, not a statement that
Liger's own AMD CI is untrusted. See `amd_tuned_torch/liger_group_norm_ops.py`'s
own module docstring for what "unvalidated" means for the specific
vendored copy here (this exact file has not been compiled/run on gfx1100,
even though the kernel it was copied from has been, elsewhere, by
Liger's own CI on different AMD hardware).
