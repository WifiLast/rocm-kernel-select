"""FlashAttention-2-style attention backed by a vendored rocWMMA HIP
kernel (amd_tuned_torch/_vendor/rocwmma_fattn/, see NOTICE.md there for
provenance and exactly what was changed from upstream) -- an alternative
to te_ops.py's TransformerEngine-backed attention, for
amd_tuned_torch.enable_flash_attn_rocwmma() to opt into.

UNVALIDATED -- do not treat this as equivalent in trustworthiness to
te_ops.py. The upstream kernel's own README and benchmark numbers are all
Windows+ZLUDA; its Linux/ROCm code path (which this module uses) is real,
un-platform-gated code, but there is no evidence upstream ever ran it, and
nothing in this project has run or numerically verified it either (no
ROCm hardware was available while writing this). See
amd_tuned_torch.enable_flash_attn_rocwmma's docstring for what to check
before trusting this.

Unlike te_ops.py (gated behind AMD_TUNED_TORCH_ENABLE_TE, read once at
*import* time, because a broken TE install can segfault the whole process
before Python's exception machinery even runs -- see that module's
docstring), this module builds nothing at import time at all. The JIT
build (torch.utils.cpp_extension.load) only happens lazily, the first
time available() or scaled_dot_product_attention() is actually called --
a JIT compile failure here is an ordinary catchable exception, not a
segfault risk, so there's no need to keep even *attempting* it behind an
env-var gate the way TE's import is. It's still opt-in (never triggered
by plain `import amd_tuned_torch`, never installed by enable()) because
first use pays real JIT-compile wall-clock time and because the kernel
itself is unvalidated -- see amd_tuned_torch.enable_flash_attn_rocwmma.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

_VENDOR_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "_vendor", "rocwmma_fattn")

_flash_attn_wmma = None
_load_error: Optional[Exception] = None


def _ensure_loaded() -> None:
    global _flash_attn_wmma, _load_error
    if _flash_attn_wmma is not None or _load_error is not None:
        return
    try:
        import torch.utils.cpp_extension

        # Matches setup.py's own GPU_ARCH convention (default gfx1100,
        # overridable for other RDNA3/3.5 parts) rather than hardcoding
        # "gfx1100" the way upstream's FlashAttn.py does, so this JIT
        # build stays consistent with however the rest of this extension
        # was configured to build.
        os.environ.setdefault(
            "PYTORCH_ROCM_ARCH", os.environ.get("AMD_TUNED_TORCH_GPU_ARCH", "gfx1100")
        )
        build_dir = os.path.join(_VENDOR_DIR, "build")
        os.makedirs(build_dir, exist_ok=True)
        sources = [
            os.path.join(_VENDOR_DIR, "host.cpp"),
            os.path.join(_VENDOR_DIR, "kernel_bf16.cu"),
            os.path.join(_VENDOR_DIR, "kernel_fp16.cu"),
        ]
        _flash_attn_wmma = torch.utils.cpp_extension.load(
            name="amd_tuned_torch_flash_attn_wmma",
            sources=sources,
            extra_cuda_cflags=[
                "-Ofast",
                # Upstream's FlashAttn.py additionally forced rocWMMA's
                # architecture macros here by hand
                # (-DROCWMMA_ARCH_GFX1100=1 -DROCWMMA_ARCH_GFX11=0
                # -DROCWMMA_WAVE32_MODE=1
                # -DROCWMMA_BLOCK_DIM_16_SUPPORTED=1, plus =0 for the other
                # RDNA3 parts). Those are NOT passed here, deliberately, and
                # re-adding them breaks the build outright on a current
                # rocWMMA.
                #
                # rocwmma/internal/config.hpp derives every one of those
                # symbols itself, from __gfx1100__ and friends -- and those
                # compiler macros exist ONLY in the device pass. On the HOST
                # pass config.hpp instead takes its ROCWMMA_ARCH_HOST branch,
                # which sets ROCWMMA_BLOCK_DIM_32_SUPPORTED=1. Forcing
                # ROCWMMA_ARCH_GFX1100=1 on the command line applies to both
                # passes, so on the host pass gfx1100 and host are set at
                # once; config.hpp then derives ROCWMMA_ARCH_GFX11=1 (quietly
                # redefining the =0 above) and its gfx11 sanity check fires:
                #
                #   config.hpp:225: error: static assertion failed due to
                #   requirement '!(bool)(1)': rocWMMA supports only block
                #   size of 16 for gfx11 arch
                #
                # Upstream needed the hand-forcing because it builds under
                # ZLUDA on Windows, where the device pass never defines
                # __gfx1100__ at all. On Linux/ROCm --offload-arch=gfx1100
                # (which torch adds from PYTORCH_ROCM_ARCH, set just above)
                # does define it, so detection works and forcing only breaks
                # it. setup.py's ahead-of-time build of the other
                # rocWMMA-using kernels has always relied on that same
                # detection, and compiles clean.
                # Without this the JIT build cannot succeed at all, on any
                # ROCm PyTorch. torch's COMMON_HIPCC_FLAGS unconditionally
                # prepends -D__HIP_NO_HALF_CONVERSIONS__=1
                # (torch/utils/cpp_extension.py), which deletes __half's
                # constructor-from-float in <hip/hip_fp16.h>; rocWMMA's
                # vector.hpp then registers its hfloat16_t vector types
                # through a macro doing static_cast<hfloat16_t>(0.0f), so
                # including <rocwmma/rocwmma.hpp> under that define is a hard
                # compile error ("no matching conversion for static_cast from
                # 'float' to 'rocwmma::hfloat16_t'"), repeated once per
                # registered type. torch splices extra_cuda_cflags in *after*
                # COMMON_HIPCC_FLAGS, so the -U wins.
                #
                # setup.py carries this same flag with the same reasoning for
                # the ahead-of-time build of the codegen'd fp16 conv kernels;
                # this JIT path simply never got it, which is why
                # enable_flash_attn_rocwmma() had only ever been observed
                # degrading to its "JIT build failed" warning + no-op.
                "-U__HIP_NO_HALF_CONVERSIONS__",
                "-mcumode",
                "-ffast-math",
                "-fgpu-flush-denormals-to-zero",
            ],
            build_directory=build_dir,
        )
    except Exception as exc:  # noqa: BLE001 -- a first-time JIT build can
        # fail for many reasons (missing rocWMMA headers, a hipcc/ROCm
        # version mismatch, an unsupported GPU_ARCH, ...); every failure
        # mode here means the same thing to callers ("not available"),
        # exactly like every other optional backend in this package
        # (aiter_ops.available(), te_ops.available()) collapses its own
        # exception zoo down to a plain bool.
        _load_error = exc


def available() -> bool:
    _ensure_loaded()
    return _flash_attn_wmma is not None


def load_error() -> Optional[Exception]:
    """The exception _ensure_loaded() caught, if available() is False --
    for amd_tuned_torch.enable_flash_attn_rocwmma's warnings.warn message.
    None if available() hasn't been called yet, or the build succeeded."""
    return _load_error


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
#
# Reimplements rocwmma_fattn/FlashAttn.py's FlashAttentionFunction against
# the lazily-loaded module above instead of an eagerly-loaded one -- same
# Br/Bc tile-size selection, same forward/backward call shape. See
# _vendor/rocwmma_fattn/NOTICE.md for the full list of what changed vs.
# upstream (this file, not the vendored .cu/.cpp sources, is where that
# logic lives now).
#
# The kernel has no attn_mask input at all (host.cpp's fwd_parm/bwd_parm
# take only a `causal: bool`, no mask tensor) and no dropout support --
# amd_tuned_torch.__init__._is_flash_attn_rocwmma_eligible enforces both
# constraints before this is ever called; this module does not re-check
# them.

class _FlashAttentionRocwmmaFn(torch.autograd.Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, q, k, v, causal, scale):
        D = q.shape[3]
        N = q.shape[2]
        Nkv = k.shape[2]
        Br, Bc = 64, 128
        if scale is None:
            scale = D ** -0.5
        if D > 384:
            Br, Bc = 32, 128

        o, q_bwd, k_bwd, v_bwd, o_bwd, L = _flash_attn_wmma.forward(
            q, k, v, Br, Bc, causal, scale, False
        )

        if q.requires_grad:
            ctx.args = (causal, scale, N, Nkv, D)
            ctx.save_for_backward(q_bwd, k_bwd, v_bwd, o_bwd, L)
        return o

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):
        causal, scale, N, Nkv, D = ctx.args
        q, k, v, o, L = ctx.saved_tensors
        Br, Bc = 128, 128
        dQ, dK, dV = _flash_attn_wmma.backward(
            q, k, v, o, do, L, N, Nkv, D, Br, Bc, causal, scale, False
        )
        return dQ, dK, dV, None, None


def scaled_dot_product_attention(query, key, value, attn_mask: Optional[torch.Tensor] = None,
                                  scale: Optional[float] = None, is_causal: bool = False):
    """query/key/value: (batch, heads, seq, head_dim), matching
    F.scaled_dot_product_attention's layout -- same convention as
    te_ops.scaled_dot_product_attention. attn_mask must already be None by
    the time this is called (see amd_tuned_torch.__init__.
    _is_flash_attn_rocwmma_eligible -- the kernel has no mask input at
    all, unlike TE's fused path, which at least recognizes a named
    attn_mask_type)."""
    _ensure_loaded()
    if _flash_attn_wmma is None:
        raise RuntimeError(f"flash_attn_rocwmma_ops not available: {_load_error}")
    return _FlashAttentionRocwmmaFn.apply(
        query.contiguous(), key.contiguous(), value.contiguous(), is_causal, scale
    )
