"""FlexGEMM-backed sparse 3D convolution / grid-sample / Morton-order
serialization for amd_tuned_torch -- the sparse-voxel-workload counterpart
to this package's dense conv2d/conv3d/GEMM tiers.

WHERE THIS COMES FROM. `flex_gemm` is not vendored into this package's own
source -- it is a separate installable extension bundled at
third_party/FlexGEMM (this package is standalone: everything needed to
build it, including this dependency, lives under source/cmp_ext_turing,
not in a sibling directory), built from its own setup.py. Same "thin
adapter over a locally-built external package" shape as
aiter_ops.py/te_ops.py/cumesh_ops.py, not the _vendor/NOTICE.md
copied-source pattern.

ROCm STATUS. FlexGEMM's compute is split two ways:
  - The hand-written CUDA kernels under flex_gemm/kernels/cuda/ (hashmap,
    Morton/Hilbert encode-decode, grid-sample, sparse-conv neighbor-map
    construction) use the same IS_HIP_EXTENSION/BUILD_TARGET/GPU_ARCHS
    convention as third_party/CuMesh's setup.py. Audited clean for HIP: no
    CUB, no texture/surface objects, no inline PTX, no half/bfloat16
    atomics -- the one warp-sync call found (migemm_neighmap_pp.cu) already
    carried a pre-existing __HIP_PLATFORM_AMD__ __syncwarp guard and needed
    no further change.
  - The actual GEMM/conv math in flex_gemm/kernels/triton/ is Triton, which
    already has its own ROCm backend -- no source changes needed there.

Build against a ROCm PyTorch with:

    cd third_party/FlexGEMM && BUILD_TARGET=rocm GPU_ARCHS=gfx1100 pip install --user -e .

before this module's available() reports True.

WHY NOT WIRED INTO kernel_select OR enable(). Sparse 3D convolution over
voxel coordinates has no torch.nn.functional equivalent at all (F.conv3d
is dense) -- there's nothing for enable() to monkeypatch or contest
against stock. Same posture as cumesh_ops/aiter_ops.fused_silu_mul: a
plain library surface a caller reaches for explicitly, e.g. a sparse-voxel
encoder/decoder (3D generative or point-cloud pipeline) built on this
package's other tuned dense ops for its 2D/attention layers.

WHAT'S EXPOSED. sparse_conv3d / sparse_submanifold_conv3d (dense-weight
sparse convolution with and without the submanifold sparsity-preserving
restriction), grid_sample_3d (trilinear/nearest sampling of a sparse
tensor at arbitrary query points), and encode_seq/decode_seq (Morton/
Hilbert-order coordinate<->code serialization, useful for sorting sparse
voxels into a cache-friendly or attention-window-friendly order). See
third_party/FlexGEMM/flex_gemm/ops/{spconv,grid_sample,serialize}.py for the
full class-based API (neighbor-cache reuse across repeated calls with the
same coordinates, algorithm/hashmap-ratio tuning knobs) this module's
one-shot wrappers sit on top of.

sparse_conv2d / sparse_submanifold_conv2d are DIFFERENT from everything
above: pure PyTorch, no flex_gemm/native extension involved, so
available() is irrelevant to them -- they work with or without
third_party/FlexGEMM installed, on CPU or ROCm identically, since they are
plain tensor-indexing ops with no custom kernel. FlexGEMM ships no
hand-written 2D sparse-conv CUDA/HIP kernel, so these are instead built
from two external templates in this repo: the per-kernel-position
"gather_scatter" method in source/sparse_convolution (gather each kernel
offset's shifted neighbor, accumulate into a dense-shaped accumulator --
there for single-channel scipy-sparse 2D arrays and a fixed non-trainable
kernel) generalized here to multi-channel deep-learning sparse tensors
(trainable [Co,Ci,Kh,Kw] weight, coords+feats rather than a CSR matrix),
using source/spconv's coordinate/feature/weight convention (coords
[N,ndim+1] batch-first, feats [N,C], a `subm`/non-`subm` split matching
its SubMConv/SparseConv naming) for the calling shape.

sparse_conv2d_native / sparse_submanifold_conv2d_native are a THIRD thing:
same contract and calling convention as the pure-Python pair above, but
computed on a REAL ROCm/HIP kernel -- not a new one. 2D convolution is
exactly a 3D convolution whose depth axis has size 1, so these lift
coords/shape/weight by one dummy spatial dimension and call
third_party/FlexGEMM's existing, already ROCm-ported sparse_conv3d /
sparse_submanifold_conv3d HIP kernel, rather than writing and compiling a
brand-new 2D kernel this repo has no ROCm hardware to validate. These DO
require available(). sparse_conv2d_from_dense (and therefore
maybe_sparse_conv2d and amd_tuned_torch.__init__._patched_conv2d's sparse
fast path) prefers the *_native kernel and falls back to the pure-Python
version when `flex_gemm` isn't installed or the native call declines --
so the production path runs on real hardware when available and degrades
to a portable software implementation rather than failing outright. See
each function's own docstring for specifics, and maybe_sparse_conv3d for
the same occupancy-threshold on-the-fly switch design (same
AMD_TUNED_TORCH_SPARSE_CONV2D on/off env var here).
"""
from __future__ import annotations

import math
import os
import threading
import weakref
from typing import Dict, Optional, Tuple

import torch

from . import sparse_conv_calibration

try:
    from flex_gemm.ops.spconv import sparse_conv3d as _sparse_conv3d
    from flex_gemm.ops.spconv import sparse_submanifold_conv3d as _sparse_submanifold_conv3d
    from flex_gemm.ops.grid_sample import grid_sample_3d as _grid_sample_3d
    from flex_gemm.ops.serialize import encode_seq as _encode_seq
    from flex_gemm.ops.serialize import decode_seq as _decode_seq

    _FLEX_GEMM_AVAILABLE = True
except ImportError:
    _sparse_conv3d = None
    _sparse_submanifold_conv3d = None
    _grid_sample_3d = None
    _encode_seq = None
    _decode_seq = None
    _FLEX_GEMM_AVAILABLE = False


def available() -> bool:
    """True if the `flex_gemm` extension (third_party/FlexGEMM) is importable."""
    return _FLEX_GEMM_AVAILABLE


def _env_flag(name: str, default: str = "1") -> bool:
    """Same on/off-string parsing as miopen_fallback._env_flag -- not
    imported from there to avoid a cross-module dependency for one
    four-line helper."""
    return os.environ.get(name, default).strip().lower() not in ("0", "", "false", "no", "off")


# ---------------------------------------------------------------------------
# Measured calibration (amd_tuned_torch.sparse_conv_calibration, written by
# tools/benchmark_sparse_conv.py) for the two per-dimensionality guesses
# maybe_sparse_conv{1,2,3}d makes -- min_positions (the "Minimum-size gate"
# section further down) and max_occupancy (the "On-the-fly dense<->sparse
# ...d switching" sections). Loaded ONCE here, before either constant is
# defined for any dimensionality, since all three dimensionalities' env-var
# defaults need it. Precedence for each resulting env var's default is
# explicit env var (checked at each os.environ.get call site below, not
# here) > this measured calibration > the hardcoded guess as a last resort
# for a GPU/build that hasn't been benchmarked yet. See
# sparse_conv_calibration.py's module docstring for the disk format, the
# AMD_TUNED_TORCH_SPARSE_CONV_CALIBRATION on/off gate, and the
# AMD_TUNED_TORCH_SPARSE_CONV_CALIBRATION_RESET "go back to the guess until
# re-benchmarked" flag.
# ---------------------------------------------------------------------------

_sparse_conv_calibration = sparse_conv_calibration.load()


def _calibrated_default(dim_key: str, field: str, hardcoded_default: str) -> str:
    """The env-var default for one (dim_key, field) pair -- e.g.
    ("conv2d", "min_positions") for AMD_TUNED_TORCH_SPARSE_CONV2D_MIN_POSITIONS,
    ("conv1d", "max_occupancy") for AMD_TUNED_TORCH_SPARSE_CONV1D_MAX_OCCUPANCY.
    Returns a measured calibration value for that pair if one was loaded,
    else `hardcoded_default`, always as a string since it feeds straight
    into os.environ.get(name, default) at each call site -- this only ever
    decides what the *default* is, never overrides an explicit env var."""
    value = _sparse_conv_calibration.get(dim_key, {}).get(field)
    return str(value) if value is not None else hardcoded_default


# ---------------------------------------------------------------------------
# On-the-fly dense<->sparse conv3d switching -- see maybe_sparse_conv3d's
# docstring for the full design and amd_tuned_torch.__init__._patched_conv3d
# for where this is called from. ON by default (AMD_TUNED_TORCH_SPARSE_CONV3D=0
# to disable and fall back to this package's existing dense-only conv3d
# tiers unconditionally, same posture as every other default-on gate in
# this package, e.g. AMD_TUNED_TORCH_MIOPEN_CONV1D_FALLBACK).
# ---------------------------------------------------------------------------

_SPARSE_CONV3D_ENABLED = _env_flag("AMD_TUNED_TORCH_SPARSE_CONV3D", default="1")
# Fraction of spatial positions with any nonzero channel, at or above which
# maybe_sparse_conv3d declines and leaves the call to the dense tiers.
# 0.1 is a hardcoded fallback guess (sparse convolution's per-point
# overhead -- coordinate math, hashmap neighbor lookups -- plausibly only
# pays for itself somewhere below "most of the volume is occupied") used
# only until tools/benchmark_sparse_conv.py has measured this GPU's actual
# crossover (see _calibrated_default above). Override per call via
# maybe_sparse_conv3d's max_occupancy argument, or globally via this env var.
_SPARSE_CONV3D_MAX_OCCUPANCY = float(
    os.environ.get("AMD_TUNED_TORCH_SPARSE_CONV3D_MAX_OCCUPANCY",
                    _calibrated_default("conv3d", "max_occupancy", "0.1")))


def sparse_conv3d_enabled() -> bool:
    """True unless AMD_TUNED_TORCH_SPARSE_CONV3D=0. Read once at import
    time, same posture as every other env-var gate in this package. Does
    not by itself mean maybe_sparse_conv3d will actually route to the
    sparse kernel for any given call -- see that function for the
    occupancy check that decides per call."""
    return _SPARSE_CONV3D_ENABLED


# ---------------------------------------------------------------------------
# FlexGEMM's kernel-volume ceiling.
#
# The default spconv algorithm (MASKED_IMPLICIT_GEMM_SPLITK, see
# third_party/FlexGEMM/flex_gemm/ops/spconv/__init__.py) encodes which of a
# kernel's taps each neighbour contributes to as a bitmask in ONE uint32, so a
# kernel with more than 32 elements has no representation. FlexGEMM enforces
# that with a bare `assert` (sparse_conv3d.py:177, submanifold_conv3d.py:84),
# not an exception type a caller can tell apart from a bug -- and AssertionError
# is not in the "unsupported shape, fall back to dense" except-tuples this
# module uses everywhere else, so it escaped every layer and killed the calling
# application. Observed from Hunyuan3D-2's mesh_render.back_project(), which
# convolves a mask with a (2/512 * resolution * 2 + 1)-square box kernel: 17x17
# = 289 taps at 2048px, 9x9 = 81 at 1024px. Only a resolution of 512 or below
# stays inside 32.
#
# Volume is checked here rather than left to FlexGEMM for two reasons: the
# maybe_* switches can then decline BEFORE paying for the dense->sparse
# conversion, and the lower-level wrappers keep this module's documented
# "None when unsupported" contract instead of raising. AssertionError is also
# added to those wrappers' except-tuples as a backstop, since a bare assert is
# how FlexGEMM reports unsupported configurations generally -- the same reason
# every aiter-backed wrapper in this package already catches it.
# ---------------------------------------------------------------------------

_FLEX_GEMM_MAX_KERNEL_VOLUME = 32


def _kernel_volume_supported(*kernel_dims) -> bool:
    """True when a kernel with these spatial extents fits FlexGEMM's uint32
    tap mask. Pass the kernel's spatial dimensions only -- not Co/Ci."""
    volume = 1
    for dim in kernel_dims:
        volume *= int(dim)
    return volume <= _FLEX_GEMM_MAX_KERNEL_VOLUME


def _weight_kernel_volume_supported(weight, *dim_indices) -> bool:
    """_kernel_volume_supported for a weight tensor's spatial dims, reading
    them at `dim_indices` (which differ between FlexGEMM's own
    [Co,Kw,Kh,Kd,Ci] layout and torch's [Co,Ci,...] one).

    PERMISSIVE when the dims cannot be read at all -- a `weight` that is not
    a tensor, or has too few dims. The guard exists to stop an
    AssertionError escaping this module; a guard that raised AttributeError
    on an odd argument would just be the same bug wearing a different
    exception. Anything it waves through still meets FlexGEMM's own assert,
    which the except-clauses below now catch.
    """
    try:
        dims = [weight.shape[i] for i in dim_indices]
    except (AttributeError, IndexError, TypeError):
        return True
    return _kernel_volume_supported(*dims)


def sparse_conv3d(feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size,
                   weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                   stride: Tuple[int, int, int] = (1, 1, 1),
                   padding: Tuple[int, int, int] = (0, 0, 0),
                   dilation: Tuple[int, int, int] = (1, 1, 1)):
    """feats [N,C] + coords [N,4] (batch,x,y,z) sparse 3D convolution with
    dense weight [Co,Kw,Kh,Kd,Ci]. Returns (out_feats, out_coords) or None
    if unavailable/unsupported -- for repeated calls against the same
    coordinates (e.g. a fixed sparse-voxel grid across training steps),
    build a flex_gemm.ops.spconv.SparseConv3dNeighborCache directly and
    call flex_gemm.ops.spconv.sparse_conv3d with it instead of recomputing
    the neighbor map every call, which is what this one-shot wrapper does."""
    if not available():
        return None
    # weight is FlexGEMM's own [Co,Kw,Kh,Kd,Ci] layout here, so the spatial
    # extents are dims 1..3.
    if not _weight_kernel_volume_supported(weight, 1, 2, 3):
        return None
    try:
        out_feats, out_coords, _cache = _sparse_conv3d(
            feats, coords, shape, weight, bias=bias,
            stride=stride, padding=padding, dilation=dilation)
        return out_feats, out_coords
    except (RuntimeError, TypeError, AssertionError):
        return None


def sparse_submanifold_conv3d(feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size,
                               weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                               dilation: Tuple[int, int, int] = (1, 1, 1)
                               ) -> Optional[torch.Tensor]:
    """Same as sparse_conv3d but sparsity-preserving (output occupies
    exactly the input's coordinates, stride fixed at 1) -- the standard
    building block for a sparse-voxel U-Net's per-resolution blocks, as
    opposed to sparse_conv3d's role at down/up-sampling transitions. None
    if unavailable/unsupported."""
    if not available():
        return None
    if not _weight_kernel_volume_supported(weight, 1, 2, 3):
        return None
    try:
        out_feats, _cache = _sparse_submanifold_conv3d(
            feats, coords, shape, weight, bias=bias, dilation=dilation)
        return out_feats
    except (RuntimeError, TypeError, AssertionError):
        return None


def grid_sample_3d(feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size,
                    grid: torch.Tensor, mode: str = "trilinear") -> Optional[torch.Tensor]:
    """Sample a sparse tensor (feats [N,C] at coords [N,4]) at arbitrary
    query points (grid [B,L,3], normalized like F.grid_sample) -> [B,L,C].
    None if unavailable/unsupported."""
    if not available():
        return None
    try:
        return _grid_sample_3d(feats, coords, shape, grid, mode=mode)
    except (RuntimeError, TypeError):
        return None


def encode_seq(coords: torch.Tensor, shape: torch.Size,
                mode: str = "z_order") -> Optional[torch.Tensor]:
    """3D voxel coordinates [N,4] -> a Morton (z_order) or Hilbert space-
    filling-curve code per point, e.g. for sorting sparse voxels into a
    cache- or attention-window-friendly order before a serialized/windowed
    attention layer. None if unavailable/unsupported."""
    if not available():
        return None
    try:
        return _encode_seq(coords, shape, mode=mode)
    except (RuntimeError, TypeError, ValueError, AssertionError):
        return None


def decode_seq(code: torch.Tensor, shape: torch.Size,
                mode: str = "z_order") -> Optional[torch.Tensor]:
    """Inverse of encode_seq: code [N] -> coordinates [N,4]. None if
    unavailable/unsupported."""
    if not available():
        return None
    try:
        return _decode_seq(code, shape, mode=mode)
    except (RuntimeError, TypeError, ValueError, AssertionError):
        return None


# ---------------------------------------------------------------------------
# Dense F.conv3d <-> sparse_conv3d switching. See maybe_sparse_conv3d's
# docstring for the design and amd_tuned_torch.__init__._patched_conv3d for
# the call site. FORWARD ONLY -- unlike flex_gemm's own
# SparseConv3dFunction (a real torch.autograd.Function), the dense<->sparse
# reconstruction below (boolean-mask extraction, advanced-index scatter)
# is not verified to produce correct gradients, so _patched_conv3d only
# calls this after its own _grad_safe() check has already passed, exactly
# like this package's other non-autograd conv3d tiers (native HIP, CK).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Occupancy estimation shared by maybe_sparse_conv{1,2,3}d -- deciding
# whether to even ATTEMPT the sparse path is itself content-dependent (see
# maybe_sparse_conv3d's docstring), so it can't be skipped or cached by
# shape the way kernel_select caches a dense-vs-dense winner. Two things
# keep that check cheap instead of a full O(B*C*prod(spatial)) scan on
# every single eligible call, including ones that end up staying dense:
#
#   (a) IDENTITY+VERSION CACHING (_OccupancyCache/_cached_occupancy): a
#       tensor object reused across several calls -- a fixed mask, or the
#       same activation handed to more than one patched conv in a single
#       forward pass -- is estimated once, not once per call, until it's
#       mutated in place (torch's own `._version` counter, the same
#       staleness signal amd_tuned_torch.aiter_ops._quantize_weight's cache
#       uses for a cached quantized weight) or garbage collected (evicted
#       via a weakref finalizer -- this cache tracks live tensors, not a
#       fixed-size/LRU window).
#   (b) BOUNDED SAMPLING (_estimate_occupancy): on a cache miss, at most
#       AMD_TUNED_TORCH_SPARSE_OCCUPANCY_SAMPLE_SIZE (default 4096)
#       randomly chosen spatial positions are scanned, not every position
#       -- turning the cost into a fixed O(sample_size) regardless of
#       tensor size, at the price of a statistical estimate rather than an
#       exact count for any tensor larger than the sample size. A tensor
#       with fewer spatial positions than the sample size is scanned in
#       full instead -- nothing to save by sampling less than everything,
#       and it keeps small tensors (and this file's own tests) getting an
#       exact rather than approximate answer.
#
# Neither of these changes what maybe_sparse_conv{1,2,3}d ultimately does
# with the number -- same threshold comparison, same fallback-to-dense
# behavior on a miss -- they only change how the number itself gets
# computed. Generic over spatial-dimension count (1D/2D/3D all reduce to
# "flatten every dim except channel"), so _occupancy/_occupancy2d/
# _occupancy1d below are now thin, dimensionality-named wrappers around one
# shared implementation, kept as separate names since maybe_sparse_conv1d/
# 2d/3d and this section's own docstrings already refer to them that way.
# ---------------------------------------------------------------------------


# Resolved once at import, same posture as
# amd_tuned_torch.__init__._HAS_INFERENCE_MODE: this is checked on every
# occupancy lookup, i.e. on every eligible F.conv2d call.
_HAS_IS_INFERENCE = hasattr(torch, "is_inference")


def _tensor_version(tensor: torch.Tensor) -> Optional[int]:
    """torch's in-place-mutation counter for `tensor`, or None when it
    doesn't have one.

    A tensor created inside `torch.inference_mode()` is an INFERENCE
    tensor: torch keeps no version-counter bookkeeping for it and raises
    `RuntimeError: Inference tensors do not track version counter.`
    (c10/core/TensorImpl.h) on any `._version` access. That is not an
    exotic case for this module, it is the normal state of every
    activation inside an inference_mode block -- how real pipelines run
    inference -- and amd_tuned_torch._grad_safe deliberately treats
    inference mode as SAFE to patch, so those calls do reach the
    occupancy cache below. Reading `._version` there unguarded took down
    the whole convolution: observed crashing Hunyuan3D-2's DINOv2
    patch-embedding conv2d (_patched_conv2d -> maybe_sparse_conv2d ->
    _cached_occupancy -> _OccupancyCache.set), i.e. a plain
    `F.conv2d` under `torch.inference_mode()` raising RuntimeError from
    what is only meant to be a cache-key detail.

    None means "uncacheable", not "unusable": the caller re-estimates
    (bounded work -- see _estimate_occupancy's sampling) and stores
    nothing. Caching such a tensor by identity alone is not an option,
    because without a version counter an in-place rewrite of the same
    tensor object would silently keep serving the old occupancy."""
    if _HAS_IS_INFERENCE and torch.is_inference(tensor):
        return None
    try:
        return tensor._version
    except RuntimeError:
        return None


class _OccupancyCache:
    """Caches an occupancy estimate keyed by identity + torch's in-place-
    mutation version counter (+ the `eps` it was computed with, so a call
    with a non-default eps for the same tensor object can't silently reuse
    a value computed with a different one). Not a
    weakref.WeakKeyDictionary, for the same reason
    amd_tuned_torch.aiter_ops._WeakTensorKeyDict isn't one: WeakKeyDictionary's
    hash-bucket collision handling falls back to `==`, and
    torch.Tensor.__eq__ returns an elementwise tensor for anything with
    more than one element -- a hash collision between two unrelated
    multi-element tensors would crash with "Boolean value of Tensor with
    more than one element is ambiguous" the moment both are keys here.
    This keys by id(tensor) (a plain int, no custom __eq__ to trip over)
    with a weakref finalizer per entry to evict it when the tensor is
    freed, matching WeakKeyDictionary's auto-eviction without ever calling
    `==` on a tensor for cache bookkeeping.

    Guarded by a plain threading.Lock, same as
    amd_tuned_torch.kernel_select's own _winners/_bad_candidates dicts --
    maybe_sparse_conv{1,2,3}d can be called concurrently from more than one
    thread (e.g. a multi-threaded inference server), and an unguarded dict
    mutation here would race the same way an unguarded kernel_select cache
    would. The weakref finalizer below also takes the lock: it can fire
    from garbage collection at a point unrelated to any get()/set() call
    (CPython's GC can run on any thread that happens to trigger it), so it
    needs the same protection as an ordinary caller."""

    def __init__(self) -> None:
        self._data: Dict[int, Tuple[object, int, float, float]] = {}
        self._lock = threading.Lock()

    def get(self, tensor: torch.Tensor, eps: float) -> Optional[float]:
        version = _tensor_version(tensor)
        if version is None:  # uncacheable, always a miss -- see _tensor_version
            return None
        with self._lock:
            entry = self._data.get(id(tensor))
            if entry is None:
                return None
            _ref, cached_version, cached_eps, occupancy = entry
        if cached_version != version or cached_eps != eps:
            return None
        return occupancy

    def set(self, tensor: torch.Tensor, eps: float, occupancy: float) -> None:
        version = _tensor_version(tensor)
        if version is None:  # uncacheable -- see _tensor_version
            return
        key_id = id(tensor)

        def _on_collected(_ref, data=self._data, key_id=key_id, lock=self._lock):
            with lock:
                data.pop(key_id, None)

        with self._lock:
            self._data[key_id] = (weakref.ref(tensor, _on_collected), version, eps, occupancy)


_occupancy_cache = _OccupancyCache()
_OCCUPANCY_SAMPLE_SIZE = int(os.environ.get("AMD_TUNED_TORCH_SPARSE_OCCUPANCY_SAMPLE_SIZE", "4096"))


def _estimate_occupancy(input: torch.Tensor, eps: float = 1e-12) -> float:
    """(b) Bounded-sample occupancy estimate: fraction of spatial positions
    (every dim except batch and channel) with at least one nonzero
    channel, from at most _OCCUPANCY_SAMPLE_SIZE randomly sampled
    positions rather than a full scan. Generic over 1D/2D/3D -- `input`
    can be [B,C,L], [B,C,H,W], or [B,C,D,H,W]; `movedim(1,-1).reshape(-1,C)`
    flattens whichever it is down to one [num_positions, C] table before
    sampling from it."""
    with torch.no_grad():
        c = input.shape[1]
        flat = input.detach().movedim(1, -1).reshape(-1, c)
        total = flat.shape[0]
        if total > _OCCUPANCY_SAMPLE_SIZE:
            idx = torch.randint(0, total, (_OCCUPANCY_SAMPLE_SIZE,), device=input.device)
            flat = flat[idx]
        return (flat.abs().amax(dim=-1) > eps).float().mean().item()


def _cached_occupancy(input: torch.Tensor, eps: float = 1e-12) -> float:
    """(a) Identity+version cache in front of _estimate_occupancy -- see
    this section's module comment above for the full (a)+(b) design."""
    cached = _occupancy_cache.get(input, eps)
    if cached is not None:
        return cached
    value = _estimate_occupancy(input, eps)
    _occupancy_cache.set(input, eps, value)
    return value


def _occupancy(input: torch.Tensor, eps: float = 1e-12) -> float:
    """Occupancy estimate for a [B,C,D,H,W] conv3d input -- see this
    section's module comment above _OccupancyCache for the caching +
    sampling design that makes this cheaper than a full scan on repeat/
    large-tensor calls, and _estimate_occupancy for the underlying
    computation."""
    return _cached_occupancy(input, eps)


# ---------------------------------------------------------------------------
# Minimum-size gate -- checked BEFORE occupancy, not after. Even a cached/
# sampled occupancy estimate isn't free (a cache lookup, or a kernel launch
# for the sample on a miss), and if the sparse path DOES end up engaging,
# its own bookkeeping (building a dense index grid the size of the input's
# spatial extent, torch.unique to dedup output coordinates, stacking
# Kh*Kw/Kd*Kh*Kw gathered copies) has a largely fixed per-call cost that a
# small enough dense conv already beats outright, occupancy notwithstanding
# -- the same reasoning _is_pointwise_conv2d in amd_tuned_torch/__init__.py
# already applies to skip its own tiers for a shape stock wins decisively
# on. So maybe_sparse_conv{1,2,3}d check this FIRST and skip straight back
# to the dense path without spending anything on occupancy at all when the
# input is too small for sparsity to matter either way.
#
# The threshold itself: precedence is explicit env var > MEASURED
# calibration (amd_tuned_torch.sparse_conv_calibration, written by
# tools/benchmark_sparse_conv.py -- an actual sweep of dense-vs-sparse
# timings at a favorable/sparse occupancy, finding the size below which
# sparse loses even in its best case) > the hardcoded "1024" guess as a
# last resort for a GPU/build this hasn't been benchmarked on yet -- see
# _calibrated_default above (defined once, shared with the max_occupancy
# constants elsewhere in this file) and sparse_conv_calibration.py's module
# docstring for the disk format, the AMD_TUNED_TORCH_SPARSE_CONV_CALIBRATION
# on/off gate, and the AMD_TUNED_TORCH_SPARSE_CONV_CALIBRATION_RESET "go
# back to the guess until re-benchmarked" flag.
# ---------------------------------------------------------------------------

_MIN_SPATIAL_POSITIONS_DEFAULT = "1024"  # hardcoded guess, no measured calibration yet -- see precedence above

_SPARSE_CONV1D_MIN_POSITIONS = int(
    os.environ.get("AMD_TUNED_TORCH_SPARSE_CONV1D_MIN_POSITIONS",
                    _calibrated_default("conv1d", "min_positions", _MIN_SPATIAL_POSITIONS_DEFAULT)))
_SPARSE_CONV2D_MIN_POSITIONS = int(
    os.environ.get("AMD_TUNED_TORCH_SPARSE_CONV2D_MIN_POSITIONS",
                    _calibrated_default("conv2d", "min_positions", _MIN_SPATIAL_POSITIONS_DEFAULT)))
_SPARSE_CONV3D_MIN_POSITIONS = int(
    os.environ.get("AMD_TUNED_TORCH_SPARSE_CONV3D_MIN_POSITIONS",
                    _calibrated_default("conv3d", "min_positions", _MIN_SPATIAL_POSITIONS_DEFAULT)))


def _n_spatial_positions(input: torch.Tensor) -> int:
    """Batch * every dim after channel -- the size of the occupancy mask
    and of the per-kernel-offset index-grid lookups the sparse path would
    build, independent of channel count (channels affect the matmul's
    compute depth, not this bookkeeping). [B,C,L] -> B*L, [B,C,H,W] ->
    B*H*W, [B,C,D,H,W] -> B*D*H*W."""
    return input.shape[0] * math.prod(input.shape[2:])


def _dense_to_sparse_bdhwc(input: torch.Tensor, eps: float = 1e-12):
    """[B,C,D,H,W] dense -> (feats [N,C], coords [N,4] as (b,d,h,w)),
    keeping only spatial positions with at least one nonzero channel.
    coords' column order matches dims 2,3,4 of `input` directly (whatever
    those axes physically represent -- this makes no assumption about
    which is "width" vs "depth"), which is all sparse_conv3d needs as long
    as the `shape` and `weight` passed alongside it use the same order,
    which sparse_conv3d_from_dense below always does."""
    with torch.no_grad():
        x = input.detach().permute(0, 2, 3, 4, 1).contiguous()  # [B,D,H,W,C]
        mask = x.abs().amax(dim=-1) > eps
        coords = mask.nonzero(as_tuple=False).to(torch.int32)
        feats = x[mask]
    return feats, coords


def _conv3d_output_size(size: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def sparse_conv3d_from_dense(input: torch.Tensor, weight: torch.Tensor,
                              bias: Optional[torch.Tensor] = None,
                              stride: Tuple[int, int, int] = (1, 1, 1),
                              padding: Tuple[int, int, int] = (0, 0, 0),
                              dilation: Tuple[int, int, int] = (1, 1, 1)
                              ) -> Optional[torch.Tensor]:
    """F.conv3d-equivalent, dense tensor in and out, computed entirely
    through sparse_conv3d: extracts the occupied (b,d,h,w) coordinates from
    `input` (see _dense_to_sparse_bdhwc), runs the sparse convolution, and
    scatters the result back into a dense [B,Co,D',H',W'] tensor.

    CORRECTNESS. Every output position sparse_conv3d does not return (no
    occupied input voxel's receptive field reaches it) is filled with
    `bias` alone, broadcast per output channel -- the exact value a real
    dense conv3d computes there too (all-zero receptive field times
    weight, summed, plus bias) -- not left at zero. So the reconstructed
    tensor matches what F.conv3d would have produced everywhere, not just
    at the occupied positions, PROVIDED sparse_conv3d's own output is
    correct (unvalidated on real hardware -- see
    amd_tuned_torch/nvdiffrast_ops.py-style caveats in this package's other
    third_party adapters for the general posture on unverified kernels).

    None if unavailable, `input`/`weight` aren't 5D dense conv3d tensors,
    or the call fails for any reason -- same drop-out convention as every
    other tier in this package, so callers can always fall back to a dense
    conv3d tier unconditionally. No autograd support (see this section's
    module comment above) -- caller must ensure grad-safety first."""
    if not available():
        return None
    if input.dim() != 5 or weight.dim() != 5:
        return None
    try:
        b, c_in, d, h, w = input.shape
        c_out, w_c_in, kd, kh, kw = weight.shape
        if w_c_in != c_in:
            return None

        feats, coords = _dense_to_sparse_bdhwc(input)

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
            weight_perm = weight.permute(0, 2, 3, 4, 1).contiguous()  # [Co,Kd,Kh,Kw,Ci]
            shape = torch.Size((b, c_in, d, h, w))
            out_feats, out_coords, _cache = _sparse_conv3d(
                feats, coords, shape, weight_perm, bias=bias,
                stride=stride, padding=padding, dilation=dilation)
            oc = out_coords.long()
            out_bdhwc[oc[:, 0], oc[:, 1], oc[:, 2], oc[:, 3]] = out_feats.to(out_dtype)

        return out_bdhwc.permute(0, 4, 1, 2, 3).contiguous().to(input.dtype)
    except (RuntimeError, TypeError, ValueError, IndexError, AssertionError):
        return None


def maybe_sparse_conv3d(input: torch.Tensor, weight: torch.Tensor,
                         bias: Optional[torch.Tensor] = None,
                         stride: Tuple[int, int, int] = (1, 1, 1),
                         padding: Tuple[int, int, int] = (0, 0, 0),
                         dilation: Tuple[int, int, int] = (1, 1, 1),
                         max_occupancy: Optional[float] = None,
                         min_positions: Optional[int] = None) -> Optional[torch.Tensor]:
    """The on-the-fly dense/sparse switch amd_tuned_torch's F.conv3d patch
    calls before its own dense-only contest: first checks `input` isn't too
    small for sparsity to matter either way (see this module's "Minimum-
    size gate" section, `min_positions` default
    AMD_TUNED_TORCH_SPARSE_CONV3D_MIN_POSITIONS, 1024), THEN estimates
    occupancy (see _occupancy) and routes to sparse_conv3d_from_dense only
    when it is below `max_occupancy` (default
    AMD_TUNED_TORCH_SPARSE_CONV3D_MAX_OCCUPANCY, 0.1 -- see this module's
    comment above that constant for why it's an unvalidated guess, not a
    measured crossover). The size check runs first specifically so a small
    input skips paying for an occupancy estimate at all, not just for the
    sparse kernel itself.

    This is a content-dependent decision, unlike kernel_select's per-shape
    cache in the rest of this package -- two calls with an identical shape
    key (dtype, shape, stride, padding, dilation) can have very different
    occupancy (e.g. a diffusion U-Net's activations across denoising
    timesteps, or simply different inputs), so occupancy is re-estimated
    every call rather than cached by shape. See _occupancy's docstring for
    that estimate's own cost.

    None (falls through to the dense contest) when AMD_TUNED_TORCH_SPARSE_CONV3D=0
    (sparse_conv3d_enabled() is False), `flex_gemm` isn't available,
    `input` has fewer than `min_positions` spatial positions, occupancy is
    at or above the threshold, or sparse_conv3d_from_dense itself declines
    for any reason."""
    if not (sparse_conv3d_enabled() and available()):
        return None
    if input.dim() != 5 or weight.dim() != 5:
        return None
    # Declined here, not inside sparse_conv3d_from_dense, so an oversized
    # kernel costs nothing: the dense->sparse conversion below is O(B*C*D*H*W).
    if not _kernel_volume_supported(weight.shape[2], weight.shape[3], weight.shape[4]):
        return None
    min_pos = _SPARSE_CONV3D_MIN_POSITIONS if min_positions is None else min_positions
    if _n_spatial_positions(input) < min_pos:
        return None
    threshold = _SPARSE_CONV3D_MAX_OCCUPANCY if max_occupancy is None else max_occupancy
    if _occupancy(input) > threshold:
        return None
    return sparse_conv3d_from_dense(input, weight, bias, stride=stride,
                                     padding=padding, dilation=dilation)


# ---------------------------------------------------------------------------
# 2D sparse convolution -- pure PyTorch, no native/HIP kernel, no
# `available()` gate (works whether or not third_party/FlexGEMM is
# installed). See this module's docstring for the source/sparse_convolution
# + source/spconv template this is built from.
# ---------------------------------------------------------------------------


def _coords_to_keys(coords: torch.Tensor, spatial: Tuple[int, int]) -> torch.Tensor:
    """coords [N,3] (batch,h,w) -> [N] int64 keys: a batch-major, row-major
    flat index over (batch,h,w). Unique per distinct coordinate given
    `spatial`'s bounds -- every coordinate here is guaranteed
    0<=h<spatial[0], 0<=w<spatial[1] by construction of the sparse tensor
    it came from -- so this is a perfect (collision-free) key, not a hash
    needing collision handling."""
    coords_l = coords.long()
    h, w = spatial
    return (coords_l[:, 0] * h + coords_l[:, 1]) * w + coords_l[:, 2]


class _CoordsIndex:
    """coords [N,3] -> row-index lookup via a sorted-keys binary search
    (torch.sort once, torch.searchsorted per query) instead of a dense
    [B,H,W] grid (this class replaces what used to be a module-level
    _coords_index_grid function that built exactly that grid).

    WHY. A dense grid costs O(B*H*W) memory AND time to allocate and fill
    with an "unoccupied" sentinel -- paid up front, before a single lookup
    happens, regardless of how few points are actually occupied. For a
    genuinely sparse tensor (the entire reason this module exists) that
    can be WORSE than the dense conv path this is supposed to beat: a
    512x512 grid with a few hundred occupied points already costs 262144
    int64s (2MB) just to hold -1 everywhere, before any real work starts.
    third_party/torchsparse's own hashmap-based neighbor lookup
    (torchsparse/backend/hashmap/hashmap_cuda.cuh) makes the same
    O(N)-not-O(volume) trade -- this class is the same idea, implemented
    with a sort instead of an actual hash table since building a real
    GPU hashmap from plain PyTorch ops has no natural expression (no
    atomic-insert primitive at this level), while torch.sort/
    torch.searchsorted are both native ops needing no custom kernel either
    way -- exactly the "works identically on CPU or ROCm, no custom
    kernel" property every other function in this section already has.

    Costs O(N log N) to build (one sort) and O(M log N) per batch of M
    lookups (one binary search) instead of O(1) per lookup into a
    pre-built dense grid -- trading a log-factor per lookup for
    asymptotically better memory and build cost, which is the right side
    of that trade whenever N is small relative to the spatial volume --
    again, the entire premise of this module engaging at all (see
    maybe_sparse_conv2d's occupancy gate further down)."""

    def __init__(self, coords: torch.Tensor, spatial: Tuple[int, int]):
        self.spatial = spatial
        keys = _coords_to_keys(coords, spatial)
        self.sorted_keys, self.sort_perm = torch.sort(keys)

    def lookup(self, batch: torch.Tensor, hh: torch.Tensor, ww: torch.Tensor) -> torch.Tensor:
        """batch/hh/ww: same-shape long tensors, already clamped into
        [0,spatial) by the caller -- an out-of-bounds coordinate clamped to
        a valid one could otherwise alias a real point's key, so callers
        must separately mask out-of-bounds queries to -1 AFTER this call,
        same as every call site here already did against the dense grid
        (a clamped-but-invalid coordinate looks up SOME position's real
        row there too -- that's exactly why the `valid` bounds mask has
        always been applied after the lookup, not instead of it). Returns
        row indices into the coords/feats this index was built from, or -1
        where no matching coordinate exists."""
        h, w = self.spatial
        key = (batch * h + hh) * w + ww
        pos = torch.searchsorted(self.sorted_keys, key)
        pos_c = pos.clamp(max=self.sorted_keys.shape[0] - 1)
        found = (pos < self.sorted_keys.shape[0]) & (self.sorted_keys[pos_c] == key)
        return torch.where(found, self.sort_perm[pos_c], torch.full_like(key, -1))


def _pad_with_zero_row(feats: torch.Tensor) -> torch.Tensor:
    """Appends one zero row to `feats`, ONCE per conv call -- pulled out of
    _gather_with_sentinel (which used to do this `torch.cat`, an O(N*Ci)
    copy of `feats`, itself on every one of a kernel-offset loop's Kh*Kw
    iterations) so the whole loop pads `feats` a single time and every
    iteration only pays for a cheap index_select against the same already-
    padded tensor."""
    zero_row = torch.zeros(1, feats.shape[-1], dtype=feats.dtype, device=feats.device)
    return torch.cat([feats, zero_row], dim=0)


def _gather_with_sentinel(padded_feats: torch.Tensor, index: torch.Tensor, n_real: int) -> torch.Tensor:
    """padded_feats: `feats` with one extra zero row appended at index
    n_real (see _pad_with_zero_row -- call it once per conv call, not once
    per kernel offset). index: long tensor of any shape, -1 meaning "no
    such point" -> remapped to n_real, the padded zero row."""
    safe_index = torch.where(index < 0, torch.full_like(index, n_real), index)
    return padded_feats[safe_index]


def _batched_kernel_matmul(neighbor_feats_stack: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Sums `neighbor_feats_stack[k] @ weight[:,:,i,j].T` over every kernel
    position k=(i,j) in ONE torch.bmm call instead of Kh*Kw separate small
    matmuls in a Python loop.

    Args:
        neighbor_feats_stack: [K,P,Ci] -- P gathered-neighbor rows (P is
            the point count doing the accumulating: N for
            sparse_submanifold_conv2d, M/out_coords for sparse_conv2d) for
            each of the K=Kh*Kw kernel positions, stacked in the same
            row-major (i,j) order as weight's own Kh,Kw axes (i.e. entry
            k = i*Kw + j -- the same order `for i in range(kh): for j in
            range(kw)` produces).
        weight: [Co,Ci,Kh,Kw].

    Returns: [P,Co].

    WHY THIS MATTERS ON ROCm, NOT JUST AS A LOOP-UNROLL. If
    amd_tuned_torch.enable() has patched torch.bmm (see
    amd_tuned_torch.__init__._patched_bmm), EVERY `@` in this file used to be
    its own separate call into that dispatch -- hipBLASLt/CK/aiter contested
    against stock, Kh*Kw times per conv call, each paying kernel_select's
    own per-call bookkeeping (~0.09ms, see kernel_select.py) for what is
    often a small GEMM. Stacking first means exactly ONE dispatch per conv
    call, covering the whole kernel window at once -- a single larger GEMM
    a tuned kernel can actually keep this card's compute units busy on,
    instead of Kh*Kw dispatches whose overhead can exceed what any of them
    individually saves. Trades memory for it: all K gathered copies exist
    at once instead of one at a time, same tradeoff torch.nn.Unfold/im2col
    already makes for dense convolution.
    """
    c_out, c_in, kh, kw = weight.shape
    weight_flat = weight.permute(2, 3, 1, 0).reshape(kh * kw, c_in, c_out).contiguous()
    per_offset = torch.bmm(neighbor_feats_stack, weight_flat)  # [K,P,Co]
    return per_offset.sum(dim=0)


def sparse_submanifold_conv2d(feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size,
                               weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                               dilation: Tuple[int, int] = (1, 1)) -> torch.Tensor:
    """Submanifold 2D sparse convolution: output occupies EXACTLY `coords`
    (stride fixed at 1, kernel centered so shape is preserved) -- the 2D
    analogue of sparse_submanifold_conv3d's contract, same "sparsity-
    preserving building block" role for a sparse-pixel U-Net's per-
    resolution blocks.

    Args:
        feats: [N,Ci] input features.
        coords: [N,3] input coordinates, (batch, h, w) -- spconv's
            batch-first coordinate convention.
        shape: (B, Ci, H, W), the dense shape this sparse tensor represents.
        weight: [Co, Ci, Kh, Kw], REQUIRES odd Kh and Kw (so a centered
            kernel maps output position p to input positions p +
            (i - Kh//2)*dilation_h, matching how spconv/every submanifold
            implementation defines "same shape" -- there is no padding
            argument here for the same reason sparse_submanifold_conv3d
            has none: padding is implied by the centered kernel).
        bias: Optional [Co].
        dilation: (dilation_h, dilation_w).

    Returns:
        [N,Co] output features, same coordinate order as `feats`/`coords`.

    ALGORITHM (source/sparse_convolution's 'gather_scatter' method,
    generalized to multi-channel + trainable weight): build one sorted-key
    index mapping occupied (b,h,w) -> row index (_CoordsIndex -- O(N), not
    a dense O(B*H*W) grid, see that class's docstring), then for each of
    the Kh*Kw kernel positions, gather every point's shifted neighbor (zero
    if out of bounds or unoccupied -- _gather_with_sentinel). The Kh*Kw
    gathered neighbor sets are stacked and reduced in ONE torch.bmm call
    (_batched_kernel_matmul) rather than accumulated via Kh*Kw separate
    small matmuls -- see that function's docstring for why that's not just
    a loop-unroll on ROCm. O(Kh*Kw) tensor ops for the gather step, each
    O(N log N). Works identically on CPU or ROCm since every op here is
    plain indexing/sort/matmul, no custom kernel.
    """
    c_out = weight.shape[0]
    if feats.shape[0] == 0:
        return torch.zeros(0, c_out, dtype=feats.dtype, device=feats.device)
    b, c_in, h, w = shape
    w_c_in, kh, kw = weight.shape[1], weight.shape[2], weight.shape[3]
    if w_c_in != c_in:
        raise ValueError(f"weight in_channels {w_c_in} != shape's C {c_in}")
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("sparse_submanifold_conv2d requires an odd kernel "
                          f"size to preserve coordinates, got ({kh}, {kw})")
    dh, dw = dilation
    coords_l = coords.long()
    index = _CoordsIndex(coords, (h, w))
    padded_feats = _pad_with_zero_row(feats)
    neighbor_feats_stack = []
    for i in range(kh):
        oh = (i - kh // 2) * dh
        nb_h = coords_l[:, 1] + oh
        for j in range(kw):
            ow = (j - kw // 2) * dw
            nb_w = coords_l[:, 2] + ow
            valid = (nb_h >= 0) & (nb_h < h) & (nb_w >= 0) & (nb_w < w)
            neighbor_idx = index.lookup(coords_l[:, 0], nb_h.clamp(0, h - 1), nb_w.clamp(0, w - 1))
            neighbor_idx = torch.where(valid, neighbor_idx, torch.full_like(neighbor_idx, -1))
            neighbor_feats_stack.append(_gather_with_sentinel(padded_feats, neighbor_idx, feats.shape[0]))
    out = _batched_kernel_matmul(torch.stack(neighbor_feats_stack, dim=0), weight)
    if bias is not None:
        out = out + bias
    return out


def _conv_output_size(size: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def sparse_conv2d(feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size,
                   weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                   stride: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (0, 0),
                   dilation: Tuple[int, int] = (1, 1)
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """General 2D sparse convolution (stride/padding, unlike
    sparse_submanifold_conv2d): output coordinates are computed from scratch
    -- every position touched by at least one occupied input point's
    receptive field -- rather than reusing the input's coordinates. The 2D
    analogue of sparse_conv3d's contract (down/up-sampling transitions in a
    sparse-pixel U-Net, as opposed to sparse_submanifold_conv2d's role at
    fixed resolution).

    Args:
        feats: [N,Ci] input features.
        coords: [N,3] input coordinates, (batch, h, w).
        shape: (B, Ci, H, W), the dense shape this sparse tensor represents.
        weight: [Co, Ci, Kh, Kw] (any kernel size, unlike the submanifold
            variant -- no shape-preservation constraint here).
        bias: Optional [Co].
        stride, padding, dilation: (h, w) pairs, same semantics as F.conv2d.

    Returns:
        (out_feats [M,Co], out_coords [M,3]) -- M is however many distinct
        output positions were actually touched, determined by the input's
        occupancy (may be more or fewer than N depending on stride/padding).

    ALGORITHM: for each occupied input point p and each of the Kh*Kw kernel
    offsets, the output position q that would read p through that offset is
    q = (p + padding - offset*dilation) / stride (integer division only
    when it divides evenly); collecting these across every point and offset
    and de-duplicating (torch.unique) gives exactly the set of touched
    output coordinates, same idea as sparse_conv3d's own out_coords
    computation. From there it's the same per-kernel-position gather as
    sparse_submanifold_conv2d (gathering from a _CoordsIndex over the
    INPUT coordinates into the newly-built output coordinate set instead
    of back into itself), reduced via the same single-torch.bmm
    _batched_kernel_matmul instead of Kh*Kw separate matmuls -- see that
    function's docstring.
    """
    c_out = weight.shape[0]
    empty = (torch.zeros(0, c_out, dtype=feats.dtype, device=feats.device),
              torch.zeros(0, 3, dtype=torch.int32, device=feats.device))
    if feats.shape[0] == 0:
        return empty
    b, c_in, h, w = shape
    w_c_in, kh, kw = weight.shape[1], weight.shape[2], weight.shape[3]
    if w_c_in != c_in:
        raise ValueError(f"weight in_channels {w_c_in} != shape's C {c_in}")
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    h_out = _conv_output_size(h, kh, sh, ph, dh)
    w_out = _conv_output_size(w, kw, sw, pw, dw)
    if h_out <= 0 or w_out <= 0:
        return empty

    coords_l = coords.long()
    device = feats.device
    ii, jj = torch.meshgrid(torch.arange(kh, device=device), torch.arange(kw, device=device),
                             indexing="ij")
    ii = ii.reshape(1, -1)  # [1,K]
    jj = jj.reshape(1, -1)
    p_h = coords_l[:, 1:2]  # [N,1]
    p_w = coords_l[:, 2:3]
    num_h = p_h + ph - ii * dh  # [N,K]
    num_w = p_w + pw - jj * dw
    valid = (num_h % sh == 0) & (num_w % sw == 0)
    q_h = num_h // sh
    q_w = num_w // sw
    valid = valid & (q_h >= 0) & (q_h < h_out) & (q_w >= 0) & (q_w < w_out)
    b_col = coords_l[:, 0:1].expand(-1, ii.shape[1])
    cand = torch.stack([b_col[valid], q_h[valid], q_w[valid]], dim=1)
    if cand.shape[0] == 0:
        return empty
    out_coords = torch.unique(cand, dim=0)

    in_index = _CoordsIndex(coords, (h, w))

    ob, oh, ow = out_coords[:, 0], out_coords[:, 1], out_coords[:, 2]
    padded_feats = _pad_with_zero_row(feats)
    neighbor_feats_stack = []
    for i in range(kh):
        src_h = oh * sh - ph + i * dh
        h_valid = (src_h >= 0) & (src_h < h)
        for j in range(kw):
            src_w = ow * sw - pw + j * dw
            valid_ij = h_valid & (src_w >= 0) & (src_w < w)
            neighbor_idx = in_index.lookup(ob, src_h.clamp(0, h - 1), src_w.clamp(0, w - 1))
            neighbor_idx = torch.where(valid_ij, neighbor_idx, torch.full_like(neighbor_idx, -1))
            neighbor_feats_stack.append(_gather_with_sentinel(padded_feats, neighbor_idx, feats.shape[0]))
    out = _batched_kernel_matmul(torch.stack(neighbor_feats_stack, dim=0), weight)
    if bias is not None:
        out = out + bias
    return out, out_coords.to(torch.int32)


# ---------------------------------------------------------------------------
# ROCm/HIP-native 2D sparse convolution -- NOT a new kernel. 2D convolution
# is mathematically a 3D convolution whose depth axis has size 1, kernel
# depth 1, stride 1, padding 0, dilation 1 -- so these lift coords/shape/
# weight by one dummy spatial dimension and call third_party/FlexGEMM's
# real, already ROCm-ported sparse_conv3d/sparse_submanifold_conv3d HIP
# kernel (hashmap + neighbor-map + GEMM), rather than writing, building,
# and (with no ROCm hardware in this environment) leaving untested a
# brand-new 2D HIP kernel. Same [Co,Ci,Kh,Kw]/[N,3]/(B,C,H,W) calling
# convention as sparse_conv2d/sparse_submanifold_conv2d above -- these are
# drop-in alternates, not a different API -- and the same drop-out
# convention (None on failure/unavailability) so callers always have the
# pure-Python versions above as a fallback.
# ---------------------------------------------------------------------------


def _lift_weight_2d_to_3d(weight: torch.Tensor) -> torch.Tensor:
    """[Co,Ci,Kh,Kw] -> the [Co,K(dim2),K(dim3),K(dim4),Ci] layout
    sparse_conv3d_from_dense's own weight.permute(0,2,3,4,1) produces from a
    dense [Co,Ci,Kd,Kh,Kw] conv3d weight -- i.e. append a size-1 kernel-depth
    axis (weight.unsqueeze(-1) -> [Co,Ci,Kh,Kw,1], matching coords' (h,w,depth)
    column order below) then apply that exact same permute."""
    return weight.unsqueeze(-1).permute(0, 2, 3, 4, 1).contiguous()


def _lift_coords_2d_to_3d(coords: torch.Tensor) -> torch.Tensor:
    """[N,3] (b,h,w) -> [N,4] (b,h,w,0) -- a size-1 depth column, matching
    _lift_weight_2d_to_3d's kernel-depth axis and shape5's depth size of 1
    below. Zero is the only valid depth coordinate once that axis has size 1."""
    zeros = torch.zeros(coords.shape[0], 1, dtype=coords.dtype, device=coords.device)
    return torch.cat([coords, zeros], dim=1)


def sparse_submanifold_conv2d_native(feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size,
                                      weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                                      dilation: Tuple[int, int] = (1, 1)
                                      ) -> Optional[torch.Tensor]:
    """Same contract as sparse_submanifold_conv2d, computed on
    third_party/FlexGEMM's real HIP kernel via the depth=1 lift described
    in this section's module comment. None if `flex_gemm` isn't available,
    the kernel size isn't odd (same shape-preservation requirement as the
    pure-Python version), or the call fails for any reason."""
    if not available():
        return None
    c_out = weight.shape[0]
    if feats.shape[0] == 0:
        return torch.zeros(0, c_out, dtype=feats.dtype, device=feats.device)
    kh, kw = weight.shape[2], weight.shape[3]
    if kh % 2 == 0 or kw % 2 == 0:
        return None
    # The depth-1 lift below makes the 3D volume kh * kw * 1.
    if not _kernel_volume_supported(kh, kw):
        return None
    try:
        b, c_in, h, w = shape
        dh, dw = dilation
        out_feats, _cache = _sparse_submanifold_conv3d(
            feats, _lift_coords_2d_to_3d(coords), torch.Size((b, c_in, h, w, 1)),
            _lift_weight_2d_to_3d(weight), bias=bias, dilation=(dh, dw, 1))
        return out_feats
    except (RuntimeError, TypeError, ValueError, AssertionError):
        return None


def sparse_conv2d_native(feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size,
                          weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                          stride: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (0, 0),
                          dilation: Tuple[int, int] = (1, 1)
                          ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Same contract as sparse_conv2d, computed on third_party/FlexGEMM's
    real HIP kernel via the depth=1 lift described in this section's module
    comment (stride/padding/dilation get a matching depth-axis entry of
    (1, 0, 1) -- the identity transform for a size-1 axis). None if
    `flex_gemm` isn't available or the call fails for any reason."""
    if not available():
        return None
    c_out = weight.shape[0]
    if not _kernel_volume_supported(weight.shape[2], weight.shape[3]):
        return None
    if feats.shape[0] == 0:
        return (torch.zeros(0, c_out, dtype=feats.dtype, device=feats.device),
                torch.zeros(0, 3, dtype=torch.int32, device=feats.device))
    try:
        b, c_in, h, w = shape
        sh, sw = stride
        ph, pw = padding
        dh, dw = dilation
        out_feats, out_coords4, _cache = _sparse_conv3d(
            feats, _lift_coords_2d_to_3d(coords), torch.Size((b, c_in, h, w, 1)),
            _lift_weight_2d_to_3d(weight), bias=bias,
            stride=(sh, sw, 1), padding=(ph, pw, 0), dilation=(dh, dw, 1))
        return out_feats, out_coords4[:, :3].to(torch.int32)
    except (RuntimeError, TypeError, ValueError, AssertionError):
        return None


# ---------------------------------------------------------------------------
# On-the-fly dense F.conv2d <-> sparse_conv2d switching -- same design as
# maybe_sparse_conv3d above (occupancy estimate, content-dependent so not
# folded into kernel_select's shape cache). sparse_conv2d_from_dense below
# prefers the ROCm-native *_native kernels just above and falls back to the
# pure-Python sparse_conv2d/sparse_submanifold_conv2d when `flex_gemm` isn't
# available or the native call declines -- so, unlike maybe_sparse_conv3d,
# this path works with or without the native extension installed. ON by
# default (AMD_TUNED_TORCH_SPARSE_CONV2D=0 to disable).
# ---------------------------------------------------------------------------

_SPARSE_CONV2D_ENABLED = _env_flag("AMD_TUNED_TORCH_SPARSE_CONV2D", default="1")
_SPARSE_CONV2D_MAX_OCCUPANCY = float(
    os.environ.get("AMD_TUNED_TORCH_SPARSE_CONV2D_MAX_OCCUPANCY",
                    _calibrated_default("conv2d", "max_occupancy", "0.1")))


def sparse_conv2d_enabled() -> bool:
    """True unless AMD_TUNED_TORCH_SPARSE_CONV2D=0. Read once at import
    time, same posture as sparse_conv3d_enabled()."""
    return _SPARSE_CONV2D_ENABLED


def _occupancy2d(input: torch.Tensor, eps: float = 1e-12) -> float:
    """Occupancy estimate for a [B,C,H,W] conv2d input -- see _occupancy's
    module comment (this file's "Occupancy estimation" section, above
    sparse_conv3d_from_dense) for the identity+version caching and bounded-
    sampling design shared by all three dimensionalities."""
    return _cached_occupancy(input, eps)


def sparse_conv2d_from_dense(input: torch.Tensor, weight: torch.Tensor,
                              bias: Optional[torch.Tensor] = None,
                              stride: Tuple[int, int] = (1, 1),
                              padding: Tuple[int, int] = (0, 0),
                              dilation: Tuple[int, int] = (1, 1)
                              ) -> Optional[torch.Tensor]:
    """F.conv2d-equivalent, dense tensor in and out -- the 2D analogue of
    sparse_conv3d_from_dense; see that function's docstring for the full
    design (occupied-coordinate extraction, sparse convolution, scatter
    back into a dense tensor with every untouched output position filled
    with `bias` alone -- the exact value a real dense conv2d computes there
    too -- not zero).

    Tries sparse_conv2d_native (third_party/FlexGEMM's real HIP kernel)
    first, falling back to the pure-Python sparse_conv2d whenever
    `flex_gemm` isn't available or the native call declines for any
    reason -- so this always produces a result if occupancy is low enough,
    with or without the native extension installed, and always the SAME
    result either way (both compute the identical convolution; the native
    path is a real GPU kernel, the fallback is pure PyTorch, not a
    different approximation).

    None if `input`/`weight` aren't 4D dense conv2d tensors or the call
    fails for any reason. No autograd support (see sparse_conv3d_from_dense's
    equivalent note) -- caller must ensure grad-safety first."""
    if input.dim() != 4 or weight.dim() != 4:
        return None
    try:
        b, c_in, h, w = input.shape
        c_out, w_c_in, kh, kw = weight.shape
        if w_c_in != c_in:
            return None

        with torch.no_grad():
            # No autograd support here (see docstring), so x_bhwc can be
            # detached once and reused for both the mask and feats -- this
            # used to permute+contiguous `input` a second time just to
            # index it for feats, an extra O(B*C*H*W) copy for no reason.
            x_bhwc = input.detach().permute(0, 2, 3, 1).contiguous()
            mask = x_bhwc.abs().amax(dim=-1) > 1e-12
            coords = mask.nonzero(as_tuple=False).to(torch.int32)
            feats = x_bhwc[mask]

        h_out = _conv_output_size(h, kh, stride[0], padding[0], dilation[0])
        w_out = _conv_output_size(w, kw, stride[1], padding[1], dilation[1])
        if h_out <= 0 or w_out <= 0:
            return None

        out_dtype = bias.dtype if bias is not None else input.dtype
        out_bhwc = torch.zeros(b, h_out, w_out, c_out, dtype=out_dtype, device=input.device)
        if bias is not None:
            out_bhwc += bias.to(dtype=out_dtype, device=input.device).view(1, 1, 1, -1)

        if feats.numel() > 0:
            shape = torch.Size((b, c_in, h, w))
            native_result = sparse_conv2d_native(
                feats, coords, shape, weight, bias=bias,
                stride=stride, padding=padding, dilation=dilation)
            if native_result is not None:
                out_feats, out_coords = native_result
            else:
                out_feats, out_coords = sparse_conv2d(
                    feats, coords, shape, weight, bias=bias,
                    stride=stride, padding=padding, dilation=dilation)
            oc = out_coords.long()
            out_bhwc[oc[:, 0], oc[:, 1], oc[:, 2]] = out_feats.to(out_dtype)

        return out_bhwc.permute(0, 3, 1, 2).contiguous().to(input.dtype)
    except (RuntimeError, TypeError, ValueError, IndexError, AssertionError):
        return None


def maybe_sparse_conv2d(input: torch.Tensor, weight: torch.Tensor,
                         bias: Optional[torch.Tensor] = None,
                         stride: Tuple[int, int] = (1, 1),
                         padding: Tuple[int, int] = (0, 0),
                         dilation: Tuple[int, int] = (1, 1),
                         max_occupancy: Optional[float] = None,
                         min_positions: Optional[int] = None) -> Optional[torch.Tensor]:
    """The on-the-fly dense/sparse switch for F.conv2d -- see
    maybe_sparse_conv3d's docstring for the full design (content-dependent,
    re-checked every call, not part of kernel_select's shape cache; a
    minimum-size check runs before occupancy so a small input skips paying
    for an occupancy estimate at all). `min_positions` defaults to
    AMD_TUNED_TORCH_SPARSE_CONV2D_MIN_POSITIONS (1024). Occupancy threshold
    defaults to AMD_TUNED_TORCH_SPARSE_CONV2D_MAX_OCCUPANCY (0.1, same
    unvalidated-guess caveat as the conv3d threshold).

    None (falls through to the dense contest) when
    AMD_TUNED_TORCH_SPARSE_CONV2D=0, `flex_gemm`'s native extension isn't
    available, `input` has fewer than `min_positions` spatial positions,
    occupancy is at or above the threshold, or sparse_conv2d_from_dense
    declines for any reason.

    THE available() GATE. This function used to skip that check, on the
    reasoning that sparse_conv2d_from_dense checks it internally to decide
    native vs. pure-Python and produces a correct result either way, so
    there was nothing for this caller to gate on. Correct either way, yes
    -- but the pure-Python gather/scatter is not a fast path at all.
    Measured on gfx1100 with the native extension absent, fp16, against
    stock MIOpen on the very inputs the occupancy gate accepts:

      1x64x128x128 -> 64  k3, occupancy 0.047:   0.070ms -> 4.24ms  (61x)
      1x3x1024x1024 -> 320 k3, occupancy 0.033:  2.997ms -> 62.79ms (21x)
      1x4x128x128 -> 512  k3, occupancy 0.094:   0.071ms -> 4.44ms  (63x)

    and numerically clean enough (fp16 max error 0.87% of the output's
    RMS) that nothing else would ever flag it -- a silent 20-60x
    regression on exactly the content this path exists to accelerate. A
    mostly-black ControlNet canny/scribble hint is that third row. So the
    fast path now requires the kernel that makes it fast, matching
    maybe_sparse_conv3d, and sparse_conv2d_from_dense stays available to
    callers who want the pure-Python implementation deliberately."""
    if not sparse_conv2d_enabled():
        return None
    if not available():  # see the docstring's available()-gate section
        return None
    if input.dim() != 4 or weight.dim() != 4:
        return None
    # See maybe_sparse_conv3d: declined before the O(B*C*H*W) conversion.
    if not _kernel_volume_supported(weight.shape[2], weight.shape[3]):
        return None
    min_pos = _SPARSE_CONV2D_MIN_POSITIONS if min_positions is None else min_positions
    if _n_spatial_positions(input) < min_pos:
        return None
    threshold = _SPARSE_CONV2D_MAX_OCCUPANCY if max_occupancy is None else max_occupancy
    if _occupancy2d(input) > threshold:
        return None
    return sparse_conv2d_from_dense(input, weight, bias, stride=stride,
                                     padding=padding, dilation=dilation)


# ---------------------------------------------------------------------------
# 1D sparse convolution. NOT a third implementation: 1D convolution is
# exactly a 2D convolution whose width axis has size 1, kernel width 1,
# stride 1, padding 0, dilation 1 -- the same "convolution is separable
# across a size-1 spatial axis" fact sparse_conv2d_native/
# sparse_submanifold_conv2d_native already use to reuse the 3D kernel. So
# every function below is a thin coords/shape/weight adapter around its 2D
# counterpart (pure-Python or *_native) -- no new gather-scatter loop, no
# new lift-to-3D code, and therefore no new source of numerical bugs beyond
# "is the dummy axis appended/stripped correctly", which the tests for this
# section check directly against real F.conv1d.
# ---------------------------------------------------------------------------


def _lift_coords_1d_to_2d(coords: torch.Tensor) -> torch.Tensor:
    """[N,2] (b,x) -> [N,3] (b,x,0) -- a size-1 width column, matching the
    weight/shape lifts below."""
    zeros = torch.zeros(coords.shape[0], 1, dtype=coords.dtype, device=coords.device)
    return torch.cat([coords, zeros], dim=1)


def sparse_submanifold_conv1d(feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size,
                               weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                               dilation: Tuple[int] = (1,)) -> torch.Tensor:
    """1D analogue of sparse_submanifold_conv2d (output occupies EXACTLY
    `coords`, odd kernel required) -- computed by lifting to
    sparse_submanifold_conv2d with a dummy width=1 axis (kernel size 1 is
    itself odd, so it satisfies that function's own odd-kernel requirement
    without special-casing it here).

    Args:
        feats: [N,Ci]. coords: [N,2] (batch, x). shape: (B, Ci, L).
        weight: [Co, Ci, K], K odd. bias: Optional [Co]. dilation: (dilation_x,).

    Returns: [N,Co].
    """
    b, c_in, l = shape
    out = sparse_submanifold_conv2d(
        feats, _lift_coords_1d_to_2d(coords), torch.Size((b, c_in, l, 1)),
        weight.unsqueeze(-1), bias=bias, dilation=(dilation[0], 1))
    return out


def sparse_conv1d(feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size,
                   weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                   stride: Tuple[int] = (1,), padding: Tuple[int] = (0,),
                   dilation: Tuple[int] = (1,)) -> Tuple[torch.Tensor, torch.Tensor]:
    """1D analogue of sparse_conv2d (general stride/padding, own output
    coordinate set) -- computed by lifting to sparse_conv2d with a dummy
    width=1 axis, then dropping that axis back off the returned coordinates.

    Args:
        feats: [N,Ci]. coords: [N,2] (batch, x). shape: (B, Ci, L).
        weight: [Co, Ci, K]. bias: Optional [Co].
        stride, padding, dilation: (x,) 1-tuples, same semantics as F.conv1d.

    Returns: (out_feats [M,Co], out_coords [M,2]).
    """
    b, c_in, l = shape
    out_feats, out_coords2 = sparse_conv2d(
        feats, _lift_coords_1d_to_2d(coords), torch.Size((b, c_in, l, 1)),
        weight.unsqueeze(-1), bias=bias, stride=(stride[0], 1),
        padding=(padding[0], 0), dilation=(dilation[0], 1))
    return out_feats, out_coords2[:, :2]


def sparse_submanifold_conv1d_native(feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size,
                                      weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                                      dilation: Tuple[int] = (1,)) -> Optional[torch.Tensor]:
    """Same contract as sparse_submanifold_conv1d, computed on
    third_party/FlexGEMM's real HIP kernel: lifts to
    sparse_submanifold_conv2d_native with a dummy width=1 axis, which
    itself lifts to the 3D kernel with a dummy depth=1 axis -- two size-1
    spatial axes total, matching "1D conv is 3D conv with two spatial dims
    of size 1" exactly. None if `flex_gemm` isn't available, the kernel
    size isn't odd, or the call fails for any reason."""
    b, c_in, l = shape
    return sparse_submanifold_conv2d_native(
        feats, _lift_coords_1d_to_2d(coords), torch.Size((b, c_in, l, 1)),
        weight.unsqueeze(-1), bias=bias, dilation=(dilation[0], 1))


def sparse_conv1d_native(feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size,
                          weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                          stride: Tuple[int] = (1,), padding: Tuple[int] = (0,),
                          dilation: Tuple[int] = (1,)
                          ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Same contract as sparse_conv1d, computed on third_party/FlexGEMM's
    real HIP kernel via the same double lift as
    sparse_submanifold_conv1d_native. None if `flex_gemm` isn't available
    or the call fails for any reason."""
    b, c_in, l = shape
    result = sparse_conv2d_native(
        feats, _lift_coords_1d_to_2d(coords), torch.Size((b, c_in, l, 1)),
        weight.unsqueeze(-1), bias=bias, stride=(stride[0], 1),
        padding=(padding[0], 0), dilation=(dilation[0], 1))
    if result is None:
        return None
    out_feats, out_coords2 = result
    return out_feats, out_coords2[:, :2]


# ---------------------------------------------------------------------------
# On-the-fly dense F.conv1d <-> sparse_conv1d switching -- same design as
# maybe_sparse_conv2d/maybe_sparse_conv3d above. sparse_conv1d_from_dense
# prefers sparse_conv1d_native and falls back to the pure-Python
# sparse_conv1d, same as the 2D version. ON by default
# (AMD_TUNED_TORCH_SPARSE_CONV1D=0 to disable). Unlike conv2d/conv3d, F.conv1d
# has no dedicated _patched_conv1d in amd_tuned_torch/__init__.py -- its dispatch
# lives in amd_tuned_torch/miopen_fallback.py's _miopen_safe_conv wrapper instead
# (a crash-rescue + causal-conv1d fast path, not a kernel_select contest),
# so that module calls maybe_sparse_conv1d directly rather than this one
# being wired from __init__.py.
# ---------------------------------------------------------------------------

_SPARSE_CONV1D_ENABLED = _env_flag("AMD_TUNED_TORCH_SPARSE_CONV1D", default="1")
_SPARSE_CONV1D_MAX_OCCUPANCY = float(
    os.environ.get("AMD_TUNED_TORCH_SPARSE_CONV1D_MAX_OCCUPANCY",
                    _calibrated_default("conv1d", "max_occupancy", "0.1")))


def sparse_conv1d_enabled() -> bool:
    """True unless AMD_TUNED_TORCH_SPARSE_CONV1D=0. Read once at import
    time, same posture as sparse_conv2d_enabled()/sparse_conv3d_enabled()."""
    return _SPARSE_CONV1D_ENABLED


def _occupancy1d(input: torch.Tensor, eps: float = 1e-12) -> float:
    """Occupancy estimate for a [B,C,L] conv1d input -- see _occupancy's
    module comment (this file's "Occupancy estimation" section, above
    sparse_conv3d_from_dense) for the identity+version caching and bounded-
    sampling design shared by all three dimensionalities."""
    return _cached_occupancy(input, eps)


def sparse_conv1d_from_dense(input: torch.Tensor, weight: torch.Tensor,
                              bias: Optional[torch.Tensor] = None,
                              stride: Tuple[int] = (1,), padding: Tuple[int] = (0,),
                              dilation: Tuple[int] = (1,)) -> Optional[torch.Tensor]:
    """F.conv1d-equivalent, dense tensor in and out -- the 1D analogue of
    sparse_conv2d_from_dense/sparse_conv3d_from_dense; see those functions'
    docstrings for the full design (occupied-coordinate extraction, sparse
    convolution, scatter back with every untouched output position filled
    with `bias` alone -- not zero). Tries sparse_conv1d_native first,
    falling back to the pure-Python sparse_conv1d.

    None if `input`/`weight` aren't 3D dense conv1d tensors or the call
    fails for any reason. No autograd support -- caller must ensure
    grad-safety first."""
    if input.dim() != 3 or weight.dim() != 3:
        return None
    try:
        b, c_in, l = input.shape
        c_out, w_c_in, k = weight.shape
        if w_c_in != c_in:
            return None

        with torch.no_grad():
            # See sparse_conv2d_from_dense's identical comment: no autograd
            # here, so x_blc can be reused for feats instead of
            # permute+contiguous-ing `input` a second time.
            x_blc = input.detach().permute(0, 2, 1).contiguous()
            mask = x_blc.abs().amax(dim=-1) > 1e-12
            coords = mask.nonzero(as_tuple=False).to(torch.int32)
            feats = x_blc[mask]

        l_out = _conv_output_size(l, k, stride[0], padding[0], dilation[0])
        if l_out <= 0:
            return None

        out_dtype = bias.dtype if bias is not None else input.dtype
        out_blc = torch.zeros(b, l_out, c_out, dtype=out_dtype, device=input.device)
        if bias is not None:
            out_blc += bias.to(dtype=out_dtype, device=input.device).view(1, 1, -1)

        if feats.numel() > 0:
            shape = torch.Size((b, c_in, l))
            native_result = sparse_conv1d_native(
                feats, coords, shape, weight, bias=bias,
                stride=stride, padding=padding, dilation=dilation)
            if native_result is not None:
                out_feats, out_coords = native_result
            else:
                out_feats, out_coords = sparse_conv1d(
                    feats, coords, shape, weight, bias=bias,
                    stride=stride, padding=padding, dilation=dilation)
            oc = out_coords.long()
            out_blc[oc[:, 0], oc[:, 1]] = out_feats.to(out_dtype)

        return out_blc.permute(0, 2, 1).contiguous().to(input.dtype)
    except (RuntimeError, TypeError, ValueError, IndexError, AssertionError):
        return None


def maybe_sparse_conv1d(input: torch.Tensor, weight: torch.Tensor,
                         bias: Optional[torch.Tensor] = None,
                         stride: Tuple[int] = (1,), padding: Tuple[int] = (0,),
                         dilation: Tuple[int] = (1,),
                         max_occupancy: Optional[float] = None,
                         min_positions: Optional[int] = None) -> Optional[torch.Tensor]:
    """The on-the-fly dense/sparse switch for F.conv1d -- see
    maybe_sparse_conv2d/maybe_sparse_conv3d's docstrings for the full
    design (a minimum-size check runs before occupancy so a small input
    skips paying for an occupancy estimate at all). `min_positions`
    defaults to AMD_TUNED_TORCH_SPARSE_CONV1D_MIN_POSITIONS (1024).
    Occupancy threshold defaults to
    AMD_TUNED_TORCH_SPARSE_CONV1D_MAX_OCCUPANCY (0.1, same unvalidated-
    guess caveat).

    None (caller should fall through to its own dense path) when
    AMD_TUNED_TORCH_SPARSE_CONV1D=0, `flex_gemm`'s native extension isn't
    available, `input` has fewer than `min_positions` spatial positions,
    occupancy is at or above the threshold, or sparse_conv1d_from_dense
    declines for any reason. The available() gate is there for the reason
    measured in maybe_sparse_conv2d's docstring -- without the native
    kernel the "fast" path is the pure-Python gather/scatter, which ran
    20-60x SLOWER than stock on the inputs the occupancy gate accepts,
    and this one is reachable from amd_tuned_torch.miopen_fallback's
    conv1d fast path on every long, mostly-empty signal."""
    if not sparse_conv1d_enabled():
        return None
    if not available():  # see maybe_sparse_conv2d's docstring
        return None
    if input.dim() != 3 or weight.dim() != 3:
        return None
    # A 1D kernel lifts to a 3D volume of k * 1 * 1, so the ceiling bites at
    # k > 32 -- reachable for the long depthwise kernels this path targets.
    if not _kernel_volume_supported(weight.shape[2]):
        return None
    min_pos = _SPARSE_CONV1D_MIN_POSITIONS if min_positions is None else min_positions
    if _n_spatial_positions(input) < min_pos:
        return None
    threshold = _SPARSE_CONV1D_MAX_OCCUPANCY if max_occupancy is None else max_occupancy
    if _occupancy1d(input) > threshold:
        return None
    return sparse_conv1d_from_dense(input, weight, bias, stride=stride,
                                     padding=padding, dilation=dilation)
