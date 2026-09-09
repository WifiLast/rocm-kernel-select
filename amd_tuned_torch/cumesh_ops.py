"""CuMesh-backed mesh processing for amd_tuned_torch -- GPU mesh cleanup,
simplification, UV unwrapping, and BVH queries (ray tracing / distance
fields), all torch-native.

WHERE THIS COMES FROM. `cumesh` is not vendored into this package's own
source -- it is a separate installable extension bundled at
third_party/CuMesh (this package is standalone: everything needed to
build it, including this dependency, lives under source/cmp_ext_turing,
not in a sibling directory), built from its own setup.py (CuMesh, cubvh,
and xatlas are three independent CUDAExtension modules there). This module
follows the same "thin adapter over a locally-built external package"
shape as aiter_ops.py and te_ops.py, not the _vendor/NOTICE.md pattern
those two modules' sibling gfx1100_iu4_gemm/rocwmma_fattn use for copied
source snippets -- nothing here is a copy of CuMesh's code, just an import
of it.

ROCm STATUS. third_party/CuMesh's setup.py already branches on
torch.utils.cpp_extension.IS_HIP_EXTENSION (BUILD_TARGET=auto/cuda/rocm,
GPU_ARCHS for --offload-arch) the same way this package's own setup.py
does. Build it against a ROCm PyTorch with:

    cd third_party/CuMesh && BUILD_TARGET=rocm GPU_ARCHS=gfx1100 pip install --user -e .

before this module's available() reports True. No hand-written CUDA-only
construct blocks a HIP build (audited: no CUB usage outside what hipify's
mapping table already translates to hipcub, no texture/surface objects, no
inline PTX, no cooperative groups -- see the fix applied to
third_party/CuMesh/third_party/cubvh/include/gpu/gpu_memory.h for the one
real HIP-specific guard this tree needed, an __HIP_DEVICE_COMPILE__ check
alongside the existing __CUDA_ARCH__ one in a __host__ __device__
destructor).

WHY NOT WIRED INTO kernel_select OR enable(). Every op here (mesh
simplification, hole filling, UV atlas packing, BVH ray/distance queries)
has no stock torch.nn.functional equivalent to contest or transparently
replace -- unlike linear/matmul/conv2d/group_norm, there is nothing for
enable() to monkeypatch. These are plain library calls a caller reaches
for explicitly, the same posture as aiter_ops.fused_silu_mul and
iu4_gemm_ops.iu4_linear.

WHAT'S EXPOSED. Not the full CuMesh/cuBVH/Atlas class surface (see
third_party/CuMesh/cumesh/{cumesh,bvh,xatlas}.py for that) -- just the
one-shot, stateless entry points a caller reaches for most often
(simplify a mesh, fill holes, query a BVH). For anything else (chart
computation, connectivity queries, incremental atlas packing across
multiple meshes, ...) the underlying classes are re-exported below
(CuMesh, cuBVH, Atlas) so nothing here is a ceiling on what's reachable.
"""
from __future__ import annotations

from typing import Optional

import torch

try:
    from cumesh import CuMesh, cuBVH, Atlas

    _CUMESH_AVAILABLE = True
except ImportError:
    CuMesh = None
    cuBVH = None
    Atlas = None
    _CUMESH_AVAILABLE = False


def available() -> bool:
    """True if the `cumesh` extension (third_party/CuMesh) is importable."""
    return _CUMESH_AVAILABLE


def simplify_mesh(vertices: torch.Tensor, faces: torch.Tensor, target_num_faces: int,
                   verbose: bool = False, options: Optional[dict] = None
                   ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """One-shot mesh simplification: (vertices [V,3], faces [F,3]) ->
    (vertices', faces') with at most target_num_faces triangles. None if
    `cumesh` isn't available or the call fails -- same drop-out convention
    as every other opt-in tier in this package (never raises for an
    ordinary ineligible call)."""
    if not available():
        return None
    try:
        mesh = CuMesh()
        mesh.init(vertices, faces)
        mesh.simplify(target_num_faces, verbose=verbose, options=options or {})
        return mesh.read()
    except (RuntimeError, TypeError):
        return None


def clean_mesh(vertices: torch.Tensor, faces: torch.Tensor,
                fill_hole_perimeter: Optional[float] = 3e-2,
                remove_small_components_min_area: Optional[float] = None
                ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """One-shot mesh cleanup: remove degenerate/duplicate/non-manifold
    faces, drop unreferenced vertices, optionally fill small holes and/or
    drop small connected components, then unify face orientations. Common
    prep step before simplify_mesh/uv_unwrap on a mesh from an arbitrary
    (e.g. marching-cubes or dual-contouring) source. None on failure or if
    unavailable."""
    if not available():
        return None
    try:
        mesh = CuMesh()
        mesh.init(vertices, faces)
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_non_manifold_faces()
        mesh.remove_unreferenced_vertices()
        if fill_hole_perimeter is not None:
            mesh.fill_holes(max_hole_perimeter=fill_hole_perimeter)
        if remove_small_components_min_area is not None:
            mesh.remove_small_connected_components(remove_small_components_min_area)
        mesh.unify_face_orientations()
        return mesh.read()
    except (RuntimeError, TypeError):
        return None


def uv_unwrap(vertices: torch.Tensor, faces: torch.Tensor
              ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """One-shot UV unwrap via CuMesh's own chart computation + unwrap
    (xatlas-backed packing). Returns (vertices', faces', uvs) with vertices'
    possibly re-indexed to carry one UV per (vertex, chart) pair -- see
    CuMesh.uv_unwrap's own docstring in third_party/CuMesh/cumesh/cumesh.py for
    the exact seam-splitting semantics. None on failure or if unavailable."""
    if not available():
        return None
    try:
        mesh = CuMesh()
        mesh.init(vertices, faces)
        mesh.compute_charts()
        return mesh.uv_unwrap()
    except (RuntimeError, TypeError):
        return None


def signed_distance(vertices: torch.Tensor, faces: torch.Tensor, positions: torch.Tensor,
                     mode: str = "watertight", return_uvw: bool = False):
    """Signed distance from `positions` [N,3] to the mesh (vertices, faces)
    via a one-shot cuBVH build + query. Builds a fresh BVH every call --
    for repeated queries against the same mesh, build a cuBVH directly
    (amd_tuned_torch.cumesh_ops.cuBVH, re-exported from `cumesh`) and call
    .signed_distance() on it instead of paying the build cost per query.
    None on failure or if unavailable."""
    if not available():
        return None
    try:
        bvh = cuBVH(vertices, faces)
        return bvh.signed_distance(positions, return_uvw=return_uvw, mode=mode)
    except (RuntimeError, TypeError):
        return None
