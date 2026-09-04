# Provenance

`host.cpp`, `kernel_fp16.cu`, `kernel_bf16.cu`, and `LICENSE` in this
directory are vendored from:

- Upstream: https://github.com/Repeerc/flash-attention-v2-RDNA3-minimal
- Path: `rocwmma_fattn/`
- Commit: `76d4433f1a06c7bb015c621ba9f8a3722e0489b4` (2024-08-25)
- License: Apache License 2.0 (see `LICENSE` in this directory)

## What was changed from upstream (Apache 2.0 §4(b) requires marking this)

- `host.cpp`, `kernel_fp16.cu`, `kernel_bf16.cu`: **not modified** --
  copied verbatim, only a short attribution header comment added at the
  top of each file pointing back here.
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
