"""torchsparse-backed sparse 3D convolution for amd_tuned_torch -- a more
mature, feature-complete sparse-voxel library than this package's own
flexgemm_ops.sparse_conv3d (real SparseTensor abstraction, submanifold and
strided convolution, a tuned hashmap-based neighbor map, an autograd-
capable convolution with a real backward pass).

WHERE THIS COMES FROM. `torchsparse` (MIT-licensed, mit-han-lab) is not
vendored into this package's own source -- it is a separate installable
extension bundled at third_party/torchsparse (this package is standalone:
everything needed to build it, including this dependency, lives under
source/cmp_ext_turing, not in a sibling directory), built from its own
setup.py. Same "thin adapter over a locally-built external package" shape
as aiter_ops.py/te_ops.py/cumesh_ops.py/flexgemm_ops.py, not the
_vendor/NOTICE.md copied-source pattern.

ROCm STATUS -- READ BEFORE ASSUMING FEATURE PARITY WITH UPSTREAM CUDA.
torchsparse ships THREE convolution dataflows (see
third_party/torchsparse/torchsparse/nn/functional/conv/conv_config.py's
`Dataflow` enum): ImplicitGEMM and FetchOnDemand are both built on raw
NVIDIA tensor-core intrinsics (`wmma`/`mma.sync`/`nvcuda::`) with no direct
HIP equivalent -- rocWMMA is a similar-purpose but API-incompatible
library, and porting these five kernel files would mean rewriting them
from scratch against it, which needs real ROCm hardware to validate and
was out of scope for this port. Those five sources
(convolution_{forward,backward_wgrad}_implicit_gemm{,_sorted}_cuda.cu and
convolution_forward_fetch_on_demand_cuda.cu) are EXCLUDED from the HIP
build entirely (third_party/torchsparse/setup.py compiles them only for
CUDA); only the third dataflow, GatherScatter
(convolution_gather_scatter_cuda.cu -- confirmed to use no tensor-core/
CUB/cooperative-groups constructs), is available on ROCm.

Upstream's own default is ImplicitGEMM
(conv_config.py's `_default_conv_config`), which would silently try to
call into a kernel this build doesn't compile in. So `available()` below,
the first time it reports True, forces the global conv config to
GatherScatter for the lifetime of the process
(torchsparse.nn.functional.conv_config.set_global_conv_config) --
`sparse_conv3d`/`Conv3d` in this module always go through that forced
config; a caller reaching into `torchsparse.nn` directly instead of
through this module bypasses that and would need to force it themselves
(see `force_gather_scatter_dataflow` below, exposed for exactly that case).
GatherScatter is slower than a correctly-tuned ImplicitGEMM on NVIDIA
hardware (it doesn't use tensor cores at all), but it is the only
dataflow this build has -- there is no ROCm equivalent of the tensor-core
speedup to lose today, only somewhere to reach for if this project ever
undertakes an actual rocWMMA rewrite of the excluded kernels.

Build against a ROCm PyTorch with:

    cd third_party/torchsparse && BUILD_TARGET=rocm GPU_ARCHS=gfx1100 pip install --user -e .

before this module's available() reports True.

WHY NOT WIRED INTO flexgemm_ops's maybe_sparse_conv3d SWITCH. torchsparse's
SparseTensor is a different data abstraction from flex_gemm's raw
(feats, coords, shape) tuples -- coordinate convention, stride tracking,
and the kmap (kernel map / neighbor cache) it builds internally are all
torchsparse-specific -- so bolting it into the existing dense-F.conv3d
occupancy-gated switch would mean silently picking between two unrelated
sparse-tensor representations inside one function, which is more
surprising than useful. This stays a separate, explicit-opt-in module a
caller reaches for directly when building a point-cloud/voxel pipeline
around torchsparse's own richer SparseTensor API (submanifold vs. strided
convs, a real backward pass, multi-scale downsample/upsample) rather than
flex_gemm's narrower one-shot conv -- same posture as cumesh_ops/
nvdiffrast_ops relative to this package's own dense-op dispatch.

WHAT'S EXPOSED. SparseTensor (re-exported), sparse_conv3d (thin wrapper
over torchsparse.nn.functional.conv3d with GatherScatter forced),
sparse_quantize (voxelization: raw point coordinates -> deduplicated
integer voxel coordinates), voxelize/devoxelize (scatter point features
into voxels and back, e.g. for a learned per-voxel feature from irregular
point positions). See third_party/torchsparse/README.md and
torchsparse/nn/modules/conv.py's Conv3d module for the fuller API this
module's one-shot wrappers sit on top of -- nothing here is a ceiling on
what's reachable via `import torchsparse` directly once available() is True.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch

try:
    import torchsparse
    from torchsparse import SparseTensor
    from torchsparse.nn import functional as _spf
    from torchsparse.nn.functional.conv import conv_config as _conv_config
    from torchsparse.nn.functional.conv.conv_config import Dataflow as _Dataflow
    from torchsparse.utils.quantize import sparse_quantize as _sparse_quantize

    _TORCHSPARSE_AVAILABLE = True
except ImportError:
    torchsparse = None
    SparseTensor = None
    _spf = None
    _conv_config = None
    _Dataflow = None
    _sparse_quantize = None
    _TORCHSPARSE_AVAILABLE = False

_dataflow_forced = False


def force_gather_scatter_dataflow() -> None:
    """Forces torchsparse's GLOBAL conv dataflow to GatherScatter -- the
    only dataflow this ROCm build compiles in (see module docstring).
    Idempotent; safe to call even if `torchsparse` isn't available (a
    no-op then). available() calls this automatically the first time it
    reports True, so sparse_conv3d below never needs a caller to remember
    this -- exposed separately only for a caller that reaches into
    `torchsparse.nn` directly instead of through this module."""
    global _dataflow_forced
    if not _TORCHSPARSE_AVAILABLE or _dataflow_forced:
        return
    # Start from upstream's own defaults (kmap_mode, downsample_mode, etc.)
    # and override only the two fields that actually matter here -- passing
    # a partial dict to set_global_conv_config would still work (its own
    # keys_check() fills in anything missing from _default_conv_config),
    # but noisily prints a "Missing fields" warning on every call, which
    # available() making itself is not the place for.
    config = dict(_conv_config._default_conv_config)
    config["dataflow"] = _Dataflow.GatherScatter
    config["ifsort"] = False
    _conv_config.set_global_conv_config(config)
    _dataflow_forced = True


def available() -> bool:
    """True if the `torchsparse` extension (third_party/torchsparse) is
    importable. Also forces the GatherScatter dataflow globally the first
    time this returns True (see module docstring) -- checking availability
    is therefore not free of side effects here, unlike most available()
    functions in this package, but the side effect is idempotent and only
    ever narrows torchsparse's own behavior to what this build actually
    supports."""
    if _TORCHSPARSE_AVAILABLE:
        force_gather_scatter_dataflow()
    return _TORCHSPARSE_AVAILABLE


def make_sparse_tensor(feats: torch.Tensor, coords: torch.Tensor,
                        stride: Union[int, Tuple[int, ...]] = 1) -> Optional["SparseTensor"]:
    """feats [N,C] + coords [N,4] (batch,x,y,z), torchsparse's own
    coordinate convention -- see torchsparse.SparseTensor. None if
    `torchsparse` isn't available."""
    if not available():
        return None
    return SparseTensor(feats=feats, coords=coords, stride=stride)


def sparse_conv3d(input_tensor: "SparseTensor", weight: torch.Tensor,
                   kernel_size: Union[int, List[int], Tuple[int, ...]],
                   bias: Optional[torch.Tensor] = None,
                   stride: Union[int, List[int], Tuple[int, ...]] = 1,
                   padding: Union[int, Tuple[int, ...]] = 0,
                   dilation: Union[int, Tuple[int, ...]] = 1,
                   transposed: bool = False) -> Optional["SparseTensor"]:
    """Sparse 3D convolution over a torchsparse SparseTensor -- see
    torchsparse.nn.functional.conv3d for the full contract (submanifold
    when stride == 1, strided down/up-sampling otherwise). Always runs
    through the GatherScatter dataflow on this build (see module
    docstring) regardless of what upstream's own default would pick.

    Unlike flexgemm_ops.sparse_conv3d, this has a REAL backward pass
    (torchsparse's own autograd.Function) -- usable during training, not
    just inference.

    None if `torchsparse` isn't available or the call fails for any
    reason -- same drop-out convention as every other tier in this package."""
    if not available():
        return None
    try:
        return _spf.conv3d(input_tensor, weight, kernel_size, bias=bias,
                            stride=stride, padding=padding, dilation=dilation,
                            transposed=transposed)
    except (RuntimeError, TypeError, ValueError):
        return None


def sparse_quantize(coords, voxel_size: Union[float, Tuple[float, ...]] = 1,
                     return_index: bool = False, return_inverse: bool = False):
    """Raw point coordinates -> deduplicated integer voxel coordinates
    (torchsparse.utils.quantize.sparse_quantize, numpy in/out -- the usual
    first step turning a raw point cloud into SparseTensor-ready voxel
    coordinates). None if `torchsparse` isn't available."""
    if not available():
        return None
    return _sparse_quantize(coords, voxel_size, return_index=return_index,
                             return_inverse=return_inverse)


def voxelize(feats: torch.Tensor, coords: torch.Tensor, counts: torch.Tensor) -> Optional[torch.Tensor]:
    """Scatter per-point features into per-voxel features (mean-pooled by
    `counts`, the number of points per voxel) -- torchsparse's own
    spvoxelize, a real torch.autograd.Function. None if `torchsparse`
    isn't available or the call fails."""
    if not available():
        return None
    try:
        return _spf.spvoxelize(feats, coords, counts)
    except (RuntimeError, TypeError, ValueError):
        return None


def devoxelize(feats: torch.Tensor, idx_query: torch.Tensor, weights: torch.Tensor) -> Optional[torch.Tensor]:
    """Inverse of voxelize: trilinearly-interpolated per-point features
    gathered back from per-voxel features -- torchsparse's own
    spdevoxelize (see calc_ti_weights for building `weights`). None if
    `torchsparse` isn't available or the call fails."""
    if not available():
        return None
    try:
        return _spf.spdevoxelize(feats, idx_query, weights)
    except (RuntimeError, TypeError, ValueError):
        return None


def _conv3d_output_size(size: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def sparse_conv3d_from_dense(input: torch.Tensor, weight: torch.Tensor,
                              bias: Optional[torch.Tensor] = None,
                              stride: Tuple[int, int, int] = (1, 1, 1),
                              padding: Tuple[int, int, int] = (0, 0, 0),
                              dilation: Tuple[int, int, int] = (1, 1, 1)
                              ) -> Optional[torch.Tensor]:
    """F.conv3d-equivalent, dense tensor in and out, computed through
    torchsparse's SparseTensor + sparse_conv3d -- deliberately mirrors
    flexgemm_ops.sparse_conv3d_from_dense's exact contract (occupied-
    coordinate extraction, sparse convolution, scatter back into a dense
    tensor with every untouched output position filled with `bias` alone,
    not zero) so the two are directly comparable --
    tools/benchmark_sparse_conv.py does exactly that, timing both against
    real F.conv3d for the same synthetic input.

    WEIGHT LAYOUT CONVERSION. torchsparse's own Conv3d module stores its
    kernel as [kernel_volume, Ci, Co] -- every kernel OFFSET flattened into
    one leading dimension -- rather than PyTorch's dense
    [Co,Ci,Kd,Kh,Kw] (see third_party/torchsparse/torchsparse/nn/modules/
    conv.py's Conv3d.__init__). This converts via permute+reshape,
    ASSUMING torchsparse's internal kernel-offset enumeration is the same
    row-major order (Kd outermost, Kw innermost) every other convolution
    in this project uses (F.conv3d itself, flex_gemm) -- consistent with
    every other library here, but never independently verified against
    torchsparse's own C++ kmap-building code on real hardware.

    None if `torchsparse` isn't available, `input`/`weight` aren't 5D dense
    conv3d tensors, or the call fails for any reason -- same drop-out
    convention as every other tier in this package. No autograd support on
    THIS dense-wrapper path specifically (the mask-based coordinate
    extraction and the scatter-back are not verified to produce correct
    gradients) -- caller must ensure grad-safety first, same posture as
    flexgemm_ops.sparse_conv3d_from_dense; sparse_conv3d itself, called
    directly on an already-sparse input, IS autograd-capable."""
    if not available():
        return None
    if input.dim() != 5 or weight.dim() != 5:
        return None
    try:
        b, c_in, d, h, w = input.shape
        c_out, w_c_in, kd, kh, kw = weight.shape
        if w_c_in != c_in:
            return None

        with torch.no_grad():
            x_bdhwc = input.detach().permute(0, 2, 3, 4, 1).contiguous()
            mask = x_bdhwc.abs().amax(dim=-1) > 1e-12
            coords = mask.nonzero(as_tuple=False).to(torch.int32)
            feats = x_bdhwc[mask]

        d_out = _conv3d_output_size(d, kd, stride[0], padding[0], dilation[0])
        h_out = _conv3d_output_size(h, kh, stride[1], padding[1], dilation[1])
        w_out = _conv3d_output_size(w, kw, stride[2], padding[2], dilation[2])
        if d_out <= 0 or h_out <= 0 or w_out <= 0:
            return None

        out_dtype = bias.dtype if bias is not None else input.dtype
        out_bdhwc = torch.zeros(b, d_out, h_out, w_out, c_out,
                                 dtype=out_dtype, device=input.device)
        if bias is not None:
            out_bdhwc += bias.to(dtype=out_dtype, device=input.device).view(1, 1, 1, 1, -1)

        if feats.numel() > 0:
            sparse_input = make_sparse_tensor(feats, coords)
            weight_flat = weight.permute(2, 3, 4, 1, 0).reshape(kd * kh * kw, c_in, c_out).contiguous()
            out_sparse = sparse_conv3d(sparse_input, weight_flat, (kd, kh, kw), bias=bias,
                                        stride=stride, padding=padding, dilation=dilation)
            if out_sparse is None:
                return None
            oc = out_sparse.coords.long()
            out_bdhwc[oc[:, 0], oc[:, 1], oc[:, 2], oc[:, 3]] = out_sparse.feats.to(out_dtype)

        return out_bdhwc.permute(0, 4, 1, 2, 3).contiguous().to(input.dtype)
    except (RuntimeError, TypeError, ValueError, IndexError):
        return None
