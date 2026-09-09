"""nvdiffrast-backed differentiable rasterization for amd_tuned_torch --
mesh rasterize / attribute interpolate / texture sample / antialias, all
torch-native and end-to-end differentiable.

WHERE THIS COMES FROM. `nvdiffrast` is not vendored into this package's own
source -- it is a separate installable extension bundled at
third_party/nvdiffrast (this package is standalone: everything needed to
build it, including this dependency, lives under source/cmp_ext_turing,
not in a sibling directory), built from its own setup.py. Same "thin
adapter over a locally-built external package" shape as aiter_ops.py/
te_ops.py/cumesh_ops.py/flexgemm_ops.py, not the _vendor/NOTICE.md
copied-source pattern.

ROCm STATUS -- the biggest port of the three sibling *_ops modules in this
package. Unlike CuMesh/FlexGEMM, upstream nvdiffrast's setup.py had zero
HIP awareness before this repo's port: it's now been brought in line with
the CuMesh/FlexGEMM BUILD_TARGET/GPU_ARCHS/IS_HIP_EXTENSION convention.
More importantly, nvdiffrast's rasterizer (csrc/common/cudaraster/) is
NVIDIA's own hand-written warp-synchronous software rasterizer -- no
OpenGL fallback exists in this checkout, so it had to actually work under
HIP, not just compile. Its impl/*.inl files leaned on ~35 raw PTX `asm()`
instructions with no HIP/AMDGPU equivalent (lanemask special registers,
bfind, packed-byte/half-word "video" instructions, prmt, slct, ...); all
are now reimplemented as portable C++/HIP intrinsics behind
`#if defined(__HIP_PLATFORM_AMD__)` guards in
csrc/common/cudaraster/impl/Util.inl, with the original PTX path left
byte-for-byte untouched for CUDA builds. See that file's guard comments
for exactly which ~35 functions and the semantics each one was verified
against at its actual call sites.

Two things NOT fully verified without real gfx1100/1101/1102 hardware
(flagged in Util.inl/FineRaster.inl/antialias.cu's own comments, not
silently assumed correct):
  - `__ballot_sync`/`__syncwarp` mask-narrowing chains in FineRaster.inl's
    ROP execution and antialias.cu's gradient kernel -- HIP ignores the
    mask argument (participation follows the real EXEC mask) where CUDA/
    Volta+ enforces it explicitly; every call site was traced and each
    mask is built from the actual active-lane set immediately before use,
    which should be self-consistent under both models, but reconvergence-
    timing edge cases can't be ruled out without running it.
  - The wave32 lane-count assumption baked into every shared-memory layout
    and lane-index computation matches HIP's default compute wavefront
    width on RDNA3, but is not something the source itself enforces --
    verify the ROCm build isn't forcing `-mwavefrontsize64`.

Build against a ROCm PyTorch with:

    cd third_party/nvdiffrast && BUILD_TARGET=rocm GPU_ARCHS=gfx1100 pip install --user -e .

before this module's available() reports True.

WHY NOT WIRED INTO kernel_select OR enable(). Differentiable rasterization
has no torch.nn.functional equivalent to intercept or contest against
stock -- there is nothing for enable() to monkeypatch, same posture as
cumesh_ops/flexgemm_ops. This is a plain library surface for a caller
building a differentiable-rendering pipeline (e.g. a mesh-to-image loss on
top of this package's other tuned dense ops for the surrounding network).

WHAT'S EXPOSED. A cached-per-device RasterizeCudaContext plus thin
pass-throughs of rasterize/interpolate/texture/antialias -- the full
forward-rendering pipeline nvdiffrast's own samples/torch/*.py demonstrate
(rasterize a mesh, interpolate per-vertex attributes across the rasterized
image, sample a texture at the interpolated UVs, antialias triangle
silhouette edges). Every op here is a real torch.autograd.Function under
the hood (nvdiffrast's own), so gradients flow through vertex positions,
attributes, and textures exactly as upstream nvdiffrast provides -- this
module changes none of that, only adds the availability gate and the
context-caching convenience.
"""
from __future__ import annotations

from typing import Optional

import torch

try:
    import nvdiffrast.torch as _dr

    _NVDIFFRAST_AVAILABLE = True
except ImportError:
    _dr = None
    _NVDIFFRAST_AVAILABLE = False


def available() -> bool:
    """True if the `nvdiffrast` extension (third_party/nvdiffrast) is
    importable. Does not by itself confirm the CUDA/HIP rasterizer context
    can actually be created on this device -- that's checked lazily by
    get_context(), same drop-out convention as every other tier here."""
    return _NVDIFFRAST_AVAILABLE


_contexts: dict = {}


def get_context(device: Optional[torch.device] = None):
    """A RasterizeCudaContext cached per device -- context creation isn't
    free, and unlike the rest of this module's stateless one-shot
    wrappers, nvdiffrast's own rasterize() takes a context as an explicit
    argument rather than creating one implicitly. Returns None if
    `nvdiffrast` isn't available or context creation fails on this device
    (e.g. no usable HIP context)."""
    if not available():
        return None
    key = str(device) if device is not None else torch.cuda.current_device()
    ctx = _contexts.get(key)
    if ctx is None:
        try:
            ctx = _dr.RasterizeCudaContext(device=device)
        except RuntimeError:
            return None
        _contexts[key] = ctx
    return ctx


def rasterize(pos: torch.Tensor, tri: torch.Tensor, resolution,
              ranges: Optional[torch.Tensor] = None, grad_db: bool = True,
              glctx=None):
    """Rasterize triangles -- see nvdiffrast.torch.rasterize for the full
    contract (pos [V,4] or [B,V,4] clip-space vertex positions, tri
    [T,3] int32 indices, resolution (H, W)). `glctx` defaults to
    get_context(pos.device). Returns (rast_out, rast_out_db) or None if
    unavailable/unsupported."""
    ctx = glctx if glctx is not None else get_context(pos.device if isinstance(pos, torch.Tensor) else None)
    if ctx is None:
        return None
    try:
        return _dr.rasterize(ctx, pos, tri, resolution, ranges=ranges, grad_db=grad_db)
    except RuntimeError:
        return None


def interpolate(attr: torch.Tensor, rast: torch.Tensor, tri: torch.Tensor,
                 rast_db: Optional[torch.Tensor] = None,
                 diff_attrs=None) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Interpolate per-vertex attributes across a rasterized image -- see
    nvdiffrast.torch.interpolate. None if unavailable/unsupported."""
    if not available():
        return None
    try:
        return _dr.interpolate(attr, rast, tri, rast_db=rast_db, diff_attrs=diff_attrs)
    except RuntimeError:
        return None


def texture(tex: torch.Tensor, uv: torch.Tensor, uv_da: Optional[torch.Tensor] = None,
            mip_level_bias: Optional[torch.Tensor] = None, mip=None,
            filter_mode: str = "auto", boundary_mode: str = "wrap",
            max_mip_level: Optional[int] = None) -> Optional[torch.Tensor]:
    """Sample a texture at interpolated UVs -- see nvdiffrast.torch.texture.
    None if unavailable/unsupported."""
    if not available():
        return None
    try:
        return _dr.texture(tex, uv, uv_da=uv_da, mip_level_bias=mip_level_bias, mip=mip,
                            filter_mode=filter_mode, boundary_mode=boundary_mode,
                            max_mip_level=max_mip_level)
    except RuntimeError:
        return None


def antialias(color: torch.Tensor, rast: torch.Tensor, pos: torch.Tensor, tri: torch.Tensor,
              topology_hash: Optional[torch.Tensor] = None,
              pos_gradient_boost: float = 1.0) -> Optional[torch.Tensor]:
    """Antialias triangle silhouette edges in a rendered image -- see
    nvdiffrast.torch.antialias. None if unavailable/unsupported."""
    if not available():
        return None
    try:
        return _dr.antialias(color, rast, pos, tri, topology_hash=topology_hash,
                              pos_gradient_boost=pos_gradient_boost)
    except RuntimeError:
        return None
