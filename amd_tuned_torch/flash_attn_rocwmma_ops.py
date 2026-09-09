"""FlashAttention-2-style attention backed by a vendored rocWMMA HIP
kernel (amd_tuned_torch/_vendor/rocwmma_fattn/, see NOTICE.md there for
provenance and exactly what was changed from upstream) -- an alternative
to te_ops.py's TransformerEngine-backed attention, for
amd_tuned_torch.enable_flash_attn_rocwmma() to opt into.

Numerically validated on gfx1100 (RX 7900 XTX, ROCm 7.2) as of
2026-09-06: forward and backward, fp16 and bf16, causal and not, across
56 shape combinations including sequence lengths and head dims that are
not multiples of the Br=64 / Bc=128 / 32 tile sizes, each against an fp32
reference and toleranced against a same-precision naive attention
(flash-attention's own test criterion). Also checked for run-to-run
determinism over 144 combinations x 5 repeats, and clean under
AMD_SERIALIZE_KERNEL=3 with HIP_LAUNCH_BLOCKING=1.

That validation found six real bugs in the vendored kernel, all now
fixed and each marked with a FIX comment at its site -- see
_vendor/rocwmma_fattn/NOTICE.md for the itemised list. Before it, the
backward pass was wrong in every single configuration tested (dQ off by
log2(e) everywhere, dK/dV NaN or garbage at any ragged shape), which is
consistent with the upstream README's benchmark numbers all being
Windows+ZLUDA forward-only and with no evidence upstream ever exercised
the Linux/ROCm backward. Treat coverage outside the tested envelope
(permute_NH=True in particular, which this module never sets) as still
unvalidated.

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

    # Prefer the ahead-of-time build, if setup.py made one. It is these
    # same three sources with the same flags (setup.py keeps its nvcc list
    # in sync with the extra_cuda_cflags below deliberately -- see the
    # comment block there), just compiled at install time instead of
    # costing whichever process touches attention first a ~4 minute hipcc
    # run. required=False because every reason it can be absent is a
    # perfectly ordinary one -- a checkout built before setup.py grew this
    # tier, a build for a different torch version, or
    # AMD_TUNED_TORCH_FLASH_ATTN_WMMA=0 -- and all of them should fall
    # through to the JIT rather than surface as an error.
    # AMD_TUNED_TORCH_FLASH_ATTN_JIT=1 skips the prebuilt module and forces
    # the JIT path. Without it, editing the vendored .cu sources has no
    # visible effect once a prebuilt tier exists: the loader below would keep
    # returning the .so setup.py compiled, and the edit would look like it did
    # nothing. That is exactly the loop anyone tuning these kernels is in.
    _prebuilt = None
    if os.environ.get("AMD_TUNED_TORCH_FLASH_ATTN_JIT", "0") != "1":
        try:
            from . import _native_loader
            _prebuilt = _native_loader.load(
                "amd_tuned_torch",
                os.path.dirname(os.path.realpath(__file__)),
                module_name="_native_flash_attn_wmma",
                required=False,
            )
        except Exception:
            # A broken/ABI-mismatched prebuilt .so must not be the end of the
            # story: the JIT path below can still produce a working module,
            # and it is the one that was always here. Deliberately not
            # recorded in _load_error -- that is reserved for "no backend at
            # all".
            _prebuilt = None

    if _prebuilt is not None:
        _flash_attn_wmma = _prebuilt
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

        # The device-wide synchronize()s that used to bracket this call are
        # gone. They were added to contain an async fault whose reported site
        # kept moving after each earlier fix -- the signature of a real
        # out-of-bounds access, not of a stream-ordering problem. That access
        # has since been found and fixed: backward_fp16/backward_bf16 read dO
        # with the *padded* Q's strides while dO itself was only ever padded
        # in its head dimension, never its sequence dimension (Nq_pad_sz is
        # computed there from Q's already-padded n, so it is always 0). Any
        # sequence length that is not a multiple of Br walked off the end of
        # dO's storage and, for b/h > 0, indexed the wrong batch entirely. See
        # the dO_Npad_sz FIX comment in either kernel .cu.
        #
        # Every launch already takes at::cuda::getCurrentCUDAStream(), so
        # ordering against surrounding work is the stream's job, not a
        # synchronize()'s -- and a device sync per attention call is a real
        # cost in a step that makes many of them.
        o, q_bwd, k_bwd, v_bwd, o_bwd, L = _flash_attn_wmma.forward(
            q, k, v, Br, Bc, causal, scale, False
        )

        if q.requires_grad:
            # Br/Bc saved alongside N/Nkv/D so backward tiles with the exact
            # same Br forward used (was hardcoded to 128 here regardless of
            # forward's D-conditional 64/32 -- the same "forward and
            # backward don't agree on a shape assumption" bug class as the
            # K/V padding fix in kernel_fp16.cu/kernel_bf16.cu, just for the
            # query-tile size instead of the KV-tile size. backward_fp16/
            # backward_bf16 do re-derive and re-pad for their own Br, so a
            # mismatch isn't a proven crash by itself, but there is no
            # reason for it to exist and it's one less divergent assumption
            # in a kernel already found to have two of these.
            ctx.args = (causal, scale, N, Nkv, D, Br, Bc)
            ctx.save_for_backward(q_bwd, k_bwd, v_bwd, o_bwd, L)
        return o

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):
        causal, scale, N, Nkv, D, Br, Bc = ctx.args
        q, k, v, o, L = ctx.saved_tensors
        # No synchronize() here either; see forward() for why they went.
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
