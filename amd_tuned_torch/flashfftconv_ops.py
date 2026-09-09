"""FlashFFTConv-backed long-sequence FFT convolution for amd_tuned_torch --
a Monarch-matrix-decomposition FFT-conv (Fu, Kumbong et al., Stanford Hazy
Research) for the seqlen tiers listed in flashfftconv.conv.FlashFFTConv
(256 up to 4M), backed by hand-written HIP/rocWMMA tensor-core kernels.

WHERE THIS COMES FROM. `flashfftconv` is not vendored into this package's
own source -- it is a separate installable extension bundled at
third_party/FlashFFTConv (this package is standalone: everything needed to
build it, including this dependency, lives under source/cmp_ext_turing,
not in a sibling directory), built from its own setup.py (the `monarch_cuda`
CUDAExtension there). This module follows the same "thin adapter over a
locally-built external package" shape as cumesh_ops.py/nvdiffrast_ops.py --
nothing here is a copy of FlashFFTConv's code, just an import of it.

ROCm STATUS. Unlike CuMesh/nvdiffrast/FlexGEMM/torchsparse, this port
required rewriting the library's core kernels by hand rather than only
fixing HIP-specific guards or excluding a dataflow: FlashFFTConv's whole
algorithm (the Monarch FFT decomposition) is implemented with
`nvcuda::wmma` tensor-core intrinsics across ~50 CUDA header files (every
tile is a fixed 16x16x16 fp16/bf16 fragment -- the "16_16_16"/"32_16_16"/
etc. naming refers to the Monarch decomposition's N1xN2 recursion factors,
not different WMMA tile shapes). Ported to rocWMMA for gfx1100 (RDNA3, RX
7900 XTX): `wmma::` -> `rocwmma::` via a per-file namespace alias, and every
WMMA fragment's element type changed from `half`/`__nv_bfloat16` to
`rocwmma::float16_t`/`rocwmma::bfloat16_t` specifically -- gfx11's actual
WMMA instruction dispatch (rocwmma/internal/wmma_impl.hpp, vendored at
third_party/rocm-headers/rocwmma) only has amdgcn_wmma specializations for
those two types, not for `rocwmma::hfloat16_t` (`__half`), which is what
bare `half` resolves to. Same convention this package's own hardware
-validated kernel already uses (see
_vendor/rocwmma_fattn/kernel_fp16.cu's `ComputeType`). setup.py carries the
same BUILD_TARGET/GPU_ARCHS convention as FlexGEMM/torchsparse, plus the
`-U__HIP_NO_HALF_CONVERSIONS__` flag rocWMMA's half-type registration needs
under torch's default HIPCC flags (see that module's docstring for why).

UNVALIDATED ON HARDWARE. This port has not yet been built or run on a real
gfx1100 device -- there is no ROCm toolchain in the environment it was
written in. Treat every seqlen tier / dtype / real-vs-complex-vs-r2r /
padded / gated combination as unverified until exercised on real hardware;
this module makes no numerical claims. The rocWMMA flash-attention kernel
in this same package (_vendor/rocwmma_fattn) went through exactly this
"port blind, then fix against real hardware" cycle and needed six real bug
fixes after its first hardware test -- expect this one to need the same.

WHY NOT WIRED INTO kernel_select OR enable(). FlashFFTConv is reached for
explicitly by a caller who wants a specific long-conv FFT layer at a fixed
seqlen (`FlashFFTConv(seqlen)` is an nn.Module a model architecture holds
onto, not a drop-in replacement for a stock op at arbitrary shapes) -- there
is no stock torch.nn.functional call this package's dispatch tiers could
transparently contest it against, the same posture as cumesh_ops.py's mesh
ops. `amd_tuned_torch.fftconv_ops`/`.miopen_fallback` already cover the
"transparent conv1d fast path" use case (short/medium kernels, contested via
kernel_select) -- this module is for callers who explicitly want Monarch
FFT-conv's long-sequence tiers instead.

WHAT'S EXPOSED. The library's own `FlashFFTConv` module and
`FlashDepthWiseConv1d` helper, re-exported as-is (see
third_party/FlashFFTConv/flashfftconv/conv.py and depthwise_1d.py for their
full surface) -- this adapter adds nothing beyond `available()`.
"""
from __future__ import annotations

try:
    from flashfftconv import FlashFFTConv, FlashDepthWiseConv1d

    _FLASHFFTCONV_AVAILABLE = True
except ImportError:
    FlashFFTConv = None
    FlashDepthWiseConv1d = None
    _FLASHFFTCONV_AVAILABLE = False


def available() -> bool:
    """True if the `flashfftconv` extension (third_party/FlashFFTConv,
    its `monarch_cuda` native module specifically) is importable."""
    return _FLASHFFTCONV_AVAILABLE
