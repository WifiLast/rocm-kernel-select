"""amd_tuned_torch.rocfft_ops -- thin ctypes wrapper around the
SYSTEM-INSTALLED rocFFT library (`librocfft.so`, part of every ROCm
install -- not vendored/built here) with a persistent, per-shape PLAN
CACHE, for callers who want lower-level control than `torch.fft` exposes.

WHY THIS EXISTS, AND WHY IT ISN'T "port rocFFT". `source/rocm-libraries/
projects/rocfft` is the full, real rocFFT source tree -- ~63K lines across
152 files in `library/src` alone, plus its own kernel-code-generator
pipeline and JIT/runtime-compile machinery, a full CMake build with its
own test/bench clients. Building that from source here would be a
ROCm-library-scale undertaking, wildly out of proportion to every other
`third_party/` entry in this package (CuMesh, FlexGEMM, nvdiffrast,
FlashFFTConv, torchsparse -- all small novel research kernels not
otherwise available on ROCm), and redundant besides: rocFFT already ships
as `librocfft.so` with every standard ROCm install, and it's exactly what
`torch.fft.*`/`torch.stft` already call into on ROCm (via hipFFT, the
cuFFT-API-compatible shim over rocFFT -- the same plumbing CUDA PyTorch
uses for cuFFT). `amd_tuned_torch.fftconv_ops` already goes through this
path via plain `torch.fft.rfftn`/`irfftn`.

So this module does NOT vendor or build rocFFT. It `ctypes.CDLL`-loads
whatever `librocfft.so` is already on the system (gracefully degrading,
`available()` False, if it isn't found -- same posture as every other
optional tier in this package), and binds just the small slice of
rocFFT's public C API (read directly from `source/rocm-libraries/projects/
rocfft/library/include/rocfft/rocfft.h` -- exact enum ordinals, struct/
pointer types, and argument order transcribed from that header, not
guessed) needed to create/cache/execute plans.

THE ACTUAL GAP THIS TARGETS. `torch.fft.rfftn`/`irfftn` go through
PyTorch's own `CuFFTPlanCache` (an LRU cache keyed by shape/dtype/etc.,
present on both CUDA and ROCm builds) -- so "PyTorch replans every call"
is NOT an established problem in this codebase; no profiling here has
ever shown it to be one (see the project memory this module's own
integration history lives in for that investigation). What PyTorch's
plan cache does NOT give a caller is: an explicit, inspectable,
externally-owned plan handle (useful for holding a small fixed set of
plans alive deliberately rather than trusting an LRU eviction policy
sized for general-purpose use), or direct access to rocFFT's own
advanced layout controls (`rocfft_plan_description_set_data_layout`'s
custom strides/offsets/distances, scale factors, planar-vs-interleaved
array types) that torch.fft's Python API doesn't surface at all. This
module exists to make THAT surface reachable, not because a measured
bottleneck demands it.

STATUS -- READ BEFORE RELYING ON THIS. Unvalidated on real hardware, same
as every other freshly-written module in this package: no ROCm device (so
no librocfft.so to actually dlopen) exists in the environment this was
written in. `available()` will report False here and this module's public
functions will simply raise -- by design, the same "gracefully absent,
never silently wrong" contract every other optional tier in this package
follows. What IS checked without hardware: every ctypes signature and
enum ordinal is transcribed directly from the vendored rocfft.h (not
memorized/guessed), and the pure-Python plan-key/shape/batch-count
arithmetic is unit-tested directly (see
tests/test_rocfft_ops.py) by exercising it against a MOCKED ctypes
library object, so the surrounding control flow is exercised for real even
though the actual FFT math obviously cannot be. The one thing that
CANNOT be checked at all without a real device is whether the ctypes
argtypes/struct layout actually match what the real compiled library
expects at the ABI level -- a mismatch there is a segfault, not a
Python exception, and would only surface on first real use. Confirm this
works with a small correctness check against `torch.fft` before trusting
it for anything real.

SCOPE. Real<->complex multi-dimensional transforms only
(`rfftn`/`irfftn`, single/double precision), batched over every leading
dimension not named in `dim`, on packed/contiguous data -- the same shape
of transform `fftconv_ops.fft_conv1d/2d/3d` actually needs. Complex-to-
complex transforms, custom strides/offsets/scale factors, planar
(non-interleaved) arrays, half precision, multi-GPU fields/bricks, and
JIT load/store callbacks are all real rocFFT features this module's
public API does NOT expose -- `_create_plan`'s `description` handle
support is there for a caller who wants to reach rocFFT's C API more
directly (see its docstring), but nothing here builds a friendlier layer
over those features without a concrete need driving it.

NOT WIRED INTO ANY DISPATCH PATH. Like cumesh_ops/nvdiffrast_ops/
flash_mm_kernel, this has no torch.nn.functional equivalent to intercept and
is not installed by enable()/disable() -- call
amd_tuned_torch.rocfft_ops.rfftn/irfftn directly. Do NOT wire this into
fftconv_ops.py's kernel_select contest without first actually measuring a
real plan-caching win on real hardware against torch.fft's own cache --
see "THE ACTUAL GAP" above: that gap is currently a capability gap
(features torch.fft doesn't expose), not a demonstrated performance one.
"""
from __future__ import annotations

import atexit
import ctypes
import ctypes.util
import threading
from typing import Optional

import torch

_LIB: Optional[ctypes.CDLL] = None
_LIB_LOAD_ERROR: Optional[Exception] = None
_SETUP_DONE = False
_lock = threading.RLock()

# ---------------------------------------------------------------------------
# Enum ordinals, transcribed directly from library/include/rocfft/rocfft.h
# (plain C enums with no explicit values -> 0-based declaration order).
# Do not hand-edit these without re-checking that header.
# ---------------------------------------------------------------------------
ROCFFT_STATUS_SUCCESS = 0
ROCFFT_STATUS_FAILURE = 1
ROCFFT_STATUS_INVALID_ARG_VALUE = 2
ROCFFT_STATUS_INVALID_DIMENSIONS = 3
ROCFFT_STATUS_INVALID_ARRAY_TYPE = 4
ROCFFT_STATUS_INVALID_STRIDES = 5
ROCFFT_STATUS_INVALID_DISTANCE = 6
ROCFFT_STATUS_INVALID_OFFSET = 7
ROCFFT_STATUS_INVALID_WORK_BUFFER = 8
_STATUS_NAMES = {
    ROCFFT_STATUS_SUCCESS: "success",
    ROCFFT_STATUS_FAILURE: "failure",
    ROCFFT_STATUS_INVALID_ARG_VALUE: "invalid_arg_value",
    ROCFFT_STATUS_INVALID_DIMENSIONS: "invalid_dimensions",
    ROCFFT_STATUS_INVALID_ARRAY_TYPE: "invalid_array_type",
    ROCFFT_STATUS_INVALID_STRIDES: "invalid_strides",
    ROCFFT_STATUS_INVALID_DISTANCE: "invalid_distance",
    ROCFFT_STATUS_INVALID_OFFSET: "invalid_offset",
    ROCFFT_STATUS_INVALID_WORK_BUFFER: "invalid_work_buffer",
}

ROCFFT_TRANSFORM_TYPE_COMPLEX_FORWARD = 0
ROCFFT_TRANSFORM_TYPE_COMPLEX_INVERSE = 1
ROCFFT_TRANSFORM_TYPE_REAL_FORWARD = 2
ROCFFT_TRANSFORM_TYPE_REAL_INVERSE = 3

ROCFFT_PRECISION_SINGLE = 0
ROCFFT_PRECISION_DOUBLE = 1
ROCFFT_PRECISION_HALF = 2

ROCFFT_PLACEMENT_INPLACE = 0
ROCFFT_PLACEMENT_NOTINPLACE = 1

ROCFFT_ARRAY_TYPE_COMPLEX_INTERLEAVED = 0
ROCFFT_ARRAY_TYPE_COMPLEX_PLANAR = 1
ROCFFT_ARRAY_TYPE_REAL = 2
ROCFFT_ARRAY_TYPE_HERMITIAN_INTERLEAVED = 3
ROCFFT_ARRAY_TYPE_HERMITIAN_PLANAR = 4
ROCFFT_ARRAY_TYPE_UNSET = 5

_TORCH_DTYPE_TO_PRECISION = {
    torch.float32: ROCFFT_PRECISION_SINGLE,
    torch.float64: ROCFFT_PRECISION_DOUBLE,
}
_PRECISION_TO_COMPLEX_DTYPE = {
    ROCFFT_PRECISION_SINGLE: torch.complex64,
    ROCFFT_PRECISION_DOUBLE: torch.complex128,
}
_PRECISION_TO_REAL_DTYPE = {
    ROCFFT_PRECISION_SINGLE: torch.float32,
    ROCFFT_PRECISION_DOUBLE: torch.float64,
}


class RocfftError(RuntimeError):
    """Raised for a non-success rocfft_status return, or when the library
    can't be used at all (not found, setup failed)."""


def _check(status: int, what: str) -> None:
    if status != ROCFFT_STATUS_SUCCESS:
        raise RocfftError(f"{what} failed: rocfft_status={_STATUS_NAMES.get(status, status)}")


def _configure_signatures(lib: ctypes.CDLL) -> None:
    """argtypes/restype for exactly the C functions this module calls,
    matching rocfft.h's declared signatures. Opaque handles (rocfft_plan,
    rocfft_plan_description, rocfft_execution_info) are all `struct X*`
    typedefs in the header -- bound here as plain c_void_p, with the
    actual handle value tracked on the Python side as whatever ctypes
    hands back through an output c_void_p parameter."""
    lib.rocfft_setup.argtypes = []
    lib.rocfft_setup.restype = ctypes.c_int

    lib.rocfft_cleanup.argtypes = []
    lib.rocfft_cleanup.restype = ctypes.c_int

    lib.rocfft_plan_create.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # plan*
        ctypes.c_int,  # placement
        ctypes.c_int,  # transform_type
        ctypes.c_int,  # precision
        ctypes.c_size_t,  # dimensions
        ctypes.POINTER(ctypes.c_size_t),  # lengths
        ctypes.c_size_t,  # number_of_transforms
        ctypes.c_void_p,  # description (NULL for simple transforms)
    ]
    lib.rocfft_plan_create.restype = ctypes.c_int

    lib.rocfft_plan_destroy.argtypes = [ctypes.c_void_p]
    lib.rocfft_plan_destroy.restype = ctypes.c_int

    lib.rocfft_execution_info_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.rocfft_execution_info_create.restype = ctypes.c_int

    lib.rocfft_execution_info_destroy.argtypes = [ctypes.c_void_p]
    lib.rocfft_execution_info_destroy.restype = ctypes.c_int

    lib.rocfft_execution_info_set_stream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.rocfft_execution_info_set_stream.restype = ctypes.c_int

    lib.rocfft_execute.argtypes = [
        ctypes.c_void_p,  # plan
        ctypes.POINTER(ctypes.c_void_p),  # in_buffer[]
        ctypes.POINTER(ctypes.c_void_p),  # out_buffer[]
        ctypes.c_void_p,  # info
    ]
    lib.rocfft_execute.restype = ctypes.c_int

    lib.rocfft_get_version_string.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    lib.rocfft_get_version_string.restype = ctypes.c_int


def _load_library() -> None:
    global _LIB, _LIB_LOAD_ERROR
    if _LIB is not None or _LIB_LOAD_ERROR is not None:
        return
    candidates = []
    found = ctypes.util.find_library("rocfft")
    if found:
        candidates.append(found)
    candidates += ["librocfft.so", "librocfft.so.0"]
    last_error: Optional[Exception] = None
    for name in candidates:
        try:
            lib = ctypes.CDLL(name)
            _configure_signatures(lib)
            _LIB = lib
            return
        except OSError as e:
            last_error = e
    _LIB_LOAD_ERROR = last_error or OSError("librocfft.so not found")


def _ensure_setup() -> None:
    """rocfft_setup() must be called at least once before any plan/execute
    call, and rocfft_cleanup() exactly as many times as setup afterward --
    called once here (lazily, on first real use) and paired with a single
    atexit cleanup, rather than once per plan, matching the library's own
    documented contract."""
    global _SETUP_DONE
    with _lock:
        if _SETUP_DONE:
            return
        _load_library()
        if _LIB is None:
            raise RocfftError(f"librocfft.so could not be loaded: {_LIB_LOAD_ERROR}")
        _check(_LIB.rocfft_setup(), "rocfft_setup")
        _SETUP_DONE = True
        atexit.register(_cleanup_at_exit)


def _cleanup_at_exit() -> None:
    global _SETUP_DONE
    with _lock:
        if not _SETUP_DONE or _LIB is None:
            return
        for plan_handle in list(_PLAN_CACHE.values()):
            try:
                _LIB.rocfft_plan_destroy(plan_handle)
            except Exception:
                pass
        _PLAN_CACHE.clear()
        try:
            _LIB.rocfft_cleanup()
        except Exception:
            pass
        _SETUP_DONE = False


def available() -> bool:
    """True if librocfft.so was found and rocfft_setup() succeeded. Never
    raises -- callers that want the reason a load/setup failed can inspect
    load_error()."""
    try:
        _ensure_setup()
        return True
    except RocfftError:
        return False


def load_error() -> Optional[str]:
    """Why available() is False, or None if it's True / hasn't been
    checked yet."""
    return str(_LIB_LOAD_ERROR) if _LIB_LOAD_ERROR is not None else None


def version_string() -> str:
    """rocFFT's own reported version -- raises RocfftError if unavailable."""
    _ensure_setup()
    buf = ctypes.create_string_buffer(64)
    _check(_LIB.rocfft_get_version_string(buf, 64), "rocfft_get_version_string")
    return buf.value.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Plan cache. Keyed on everything that changes which plan is needed --
# transform type, precision, dimensionality/lengths, and batch count --
# NOT on the tensor's actual data or its batch-dimension SHAPE beyond the
# flattened count, same "identity, not shape" precedent kernel_select.py's
# own key functions already establish for this package's other contests.
# ---------------------------------------------------------------------------
_PLAN_CACHE: dict = {}


def _plan_key(transform_type: int, precision: int, lengths: tuple, number_of_transforms: int) -> tuple:
    return (transform_type, precision, lengths, number_of_transforms)


def _get_or_create_plan(transform_type: int, precision: int, lengths: tuple,
                         number_of_transforms: int) -> ctypes.c_void_p:
    """lengths must already be in rocFFT's own column-major order (fastest
    -varying dimension first) -- see rfftn/irfftn's docstrings for the
    reversal from PyTorch's row-major dim order, done once there rather
    than inside every cache lookup."""
    key = _plan_key(transform_type, precision, lengths, number_of_transforms)
    with _lock:
        cached = _PLAN_CACHE.get(key)
        if cached is not None:
            return cached
    _ensure_setup()
    plan = ctypes.c_void_p()
    lengths_arr = (ctypes.c_size_t * len(lengths))(*lengths)
    placement = ROCFFT_PLACEMENT_NOTINPLACE  # every transform this module builds is real<->complex, different sizes
    _check(
        _LIB.rocfft_plan_create(
            ctypes.byref(plan), placement, transform_type, precision,
            len(lengths), lengths_arr, number_of_transforms, None,
        ),
        "rocfft_plan_create",
    )
    with _lock:
        existing = _PLAN_CACHE.get(key)
        if existing is not None:
            # Lost a race with another thread building the same plan --
            # keep the one already published, destroy this redundant one.
            _LIB.rocfft_plan_destroy(plan)
            return existing
        _PLAN_CACHE[key] = plan
    return plan


def _execute(plan: ctypes.c_void_p, in_ptr: int, out_ptr: int) -> None:
    _ensure_setup()
    info = ctypes.c_void_p()
    _check(_LIB.rocfft_execution_info_create(ctypes.byref(info)), "rocfft_execution_info_create")
    try:
        # Run on whatever stream PyTorch is currently using, so this
        # transform's write is correctly stream-ordered against every op
        # queued before/after it -- the same assumption any other
        # HIP-stream-aware custom op in this package relies on. No
        # explicit synchronize here: the caller's own subsequent use of
        # the output tensor is what enforces ordering, on this stream.
        stream = torch.cuda.current_stream().cuda_stream
        _check(_LIB.rocfft_execution_info_set_stream(info, ctypes.c_void_p(stream)),
               "rocfft_execution_info_set_stream")
        in_buf = (ctypes.c_void_p * 1)(ctypes.c_void_p(in_ptr))
        out_buf = (ctypes.c_void_p * 1)(ctypes.c_void_p(out_ptr))
        _check(_LIB.rocfft_execute(plan, in_buf, out_buf, info), "rocfft_execute")
    finally:
        _LIB.rocfft_execution_info_destroy(info)


def reset() -> None:
    """Destroys every cached plan and re-runs setup/cleanup. For tests and
    for reclaiming plan memory explicitly; ordinary callers never need
    this -- plans are cheap to keep once built, that's the whole point of
    caching them."""
    _cleanup_at_exit()


# ---------------------------------------------------------------------------
# rfftn / irfftn -- the transform shape fftconv_ops.fft_conv1d/2d/3d
# actually needs: real<->complex, batched over every leading dim, packed
# (contiguous) layout. `dim` follows torch.fft's own convention (defaults
# to every dimension); unlike torch.fft.rfftn, `dim` here must be the
# TRAILING dimensions specifically (validated below) -- this module has no
# need for the fully general "any subset of dims, any order" case
# torch.fft.rfftn allows, since fftconv_ops (and every realistic caller)
# always transforms the trailing spatial dims and batches over the rest.
# ---------------------------------------------------------------------------
def _resolve_trailing_dims(ndim: int, dim) -> tuple:
    if dim is None:
        return tuple(range(ndim))
    dim = tuple(d % ndim for d in dim)
    n = len(dim)
    expected = tuple(range(ndim - n, ndim))
    if dim != expected:
        raise ValueError(
            f"rocfft_ops only transforms the TRAILING dimensions (got dim={dim} "
            f"for a {ndim}-D tensor, expected the last {n} dims {expected}) -- "
            "use torch.fft.rfftn directly for an arbitrary dim subset/order."
        )
    return dim


def rfftn(x: torch.Tensor, dim: Optional[tuple] = None) -> torch.Tensor:
    """Real-to-complex forward transform over the trailing `len(dim)`
    dimensions of `x` (every dimension, if `dim` is None), batched over
    whatever's left. Raises RocfftError if rocFFT isn't available (see
    available()) -- callers wanting a fallback should check that
    themselves, same convention as this package's other manual-call-only
    modules (cumesh_ops, nvdiffrast_ops)."""
    if x.dtype not in _TORCH_DTYPE_TO_PRECISION:
        raise TypeError(f"rfftn: unsupported real dtype {x.dtype} (float32/float64 only)")
    if not x.is_cuda:
        raise RocfftError("rfftn: x must be a CUDA/ROCm tensor")
    ndim = x.dim()
    dims = _resolve_trailing_dims(ndim, dim)
    n_transform = len(dims)
    transform_shape = tuple(x.shape[-n_transform:])
    batch_shape = tuple(x.shape[:-n_transform]) if n_transform < ndim else ()
    n_batch = 1
    for s in batch_shape:
        n_batch *= s

    precision = _TORCH_DTYPE_TO_PRECISION[x.dtype]
    # rocFFT lengths are column-major (fastest dim first) -- the reverse
    # of PyTorch's row-major shape order.
    lengths = tuple(reversed(transform_shape))
    plan = _get_or_create_plan(ROCFFT_TRANSFORM_TYPE_REAL_FORWARD, precision, lengths, max(n_batch, 1))

    out_transform_shape = transform_shape[:-1] + (transform_shape[-1] // 2 + 1,)
    out_shape = batch_shape + out_transform_shape
    complex_dtype = _PRECISION_TO_COMPLEX_DTYPE[precision]

    x_c = x.contiguous()
    out = torch.empty(out_shape, dtype=complex_dtype, device=x.device)
    _execute(plan, x_c.data_ptr(), out.data_ptr())
    return out


def irfftn(x: torch.Tensor, s: tuple, dim: Optional[tuple] = None) -> torch.Tensor:
    """Complex-to-real inverse transform, the counterpart to rfftn(). `s`
    is the REAL output's transform-dimension sizes (required -- unlike
    torch.fft.irfftn, this module never infers the last dimension's size
    from the hermitian input's own shape, since rocFFT's plan needs the
    real lengths explicit).

    NOTE ON SCALING: rocFFT's inverse transforms are UNNORMALIZED by
    default (they do not divide by the transform size) -- unlike
    torch.fft.irfftn, which always normalizes ("backward" norm by
    default). This function divides by the product of `s` itself before
    returning, matching torch.fft.irfftn's default normalization exactly,
    so it is a drop-in numerical match for torch.fft.irfftn(x, s=s,
    dim=dim) rather than requiring the caller to know rocFFT's own
    convention."""
    if x.dtype not in (torch.complex64, torch.complex128):
        raise TypeError(f"irfftn: unsupported complex dtype {x.dtype}")
    if not x.is_cuda:
        raise RocfftError("irfftn: x must be a CUDA/ROCm tensor")
    ndim = x.dim()
    n_transform = len(s)
    dims = _resolve_trailing_dims(ndim, dim if dim is not None else tuple(range(ndim - n_transform, ndim)))
    if len(dims) != n_transform:
        raise ValueError(f"irfftn: len(s)={n_transform} must match len(dim)={len(dims)}")
    batch_shape = tuple(x.shape[:-n_transform]) if n_transform < ndim else ()
    n_batch = 1
    for b in batch_shape:
        n_batch *= b

    precision = ROCFFT_PRECISION_SINGLE if x.dtype == torch.complex64 else ROCFFT_PRECISION_DOUBLE
    lengths = tuple(reversed(s))
    plan = _get_or_create_plan(ROCFFT_TRANSFORM_TYPE_REAL_INVERSE, precision, lengths, max(n_batch, 1))

    real_dtype = _PRECISION_TO_REAL_DTYPE[precision]
    out_shape = batch_shape + tuple(s)

    x_c = x.contiguous()
    out = torch.empty(out_shape, dtype=real_dtype, device=x.device)
    _execute(plan, x_c.data_ptr(), out.data_ptr())

    n_elements = 1
    for dim_size in s:
        n_elements *= dim_size
    out /= n_elements
    return out
