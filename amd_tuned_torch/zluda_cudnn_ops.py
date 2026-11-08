"""amd_tuned_torch.zluda_cudnn_ops -- optional ctypes binding to ZLUDA's
cuDNN-ABI shim (third_party/ZLUDA's zluda_dnn9 -> `libcudnn.so.9`), which
is NOT NVIDIA cuDNN and does NOT run cuDNN kernels.

READ THIS FIRST, BECAUSE THE NAME PROMISES SOMETHING IT CANNOT DELIVER.
"Use cuDNN kernels on AMD" is not a thing ZLUDA does, or could do. NVIDIA
ships cuDNN as precompiled SASS for NVIDIA SM targets -- no source, no
retargetable IR -- so there is nothing for ZLUDA's PTX compiler to
translate (that compiler handles YOUR kernels, not NVIDIA's shipped
binaries). What third_party/ZLUDA/zluda_dnn actually is: a
REIMPLEMENTATION of a slice of the cuDNN C API that forwards to MIOpen.
Look at zluda_dnn/Cargo.toml -- its dependencies are `miopen-sys` and
`hip_runtime-sys`, and zluda_dnn/src/lib.rs's bottom half binds
miopenFindSolutions/miopenRunSolution/miopenCreateConvProblem directly.

So the call chain this module opens is:

    python -> ctypes -> ZLUDA libcudnn.so.9 -> MIOpen -> gfx1100

and the kernels that run at the end of it are the SAME MIOpen kernels
(/opt/rocm/lib/libMIOpenCKGroupedConv_gfx1100.so et al) that stock
PyTorch-on-ROCm already reaches when you call torch.nn.functional.conv2d.
This module is therefore NOT a performance tier and is deliberately NOT
wired into enable()/disable() or kernel_select.py's contests: routing
torch -> cuDNN-shim -> MIOpen to reach kernels torch already calls
directly can only add descriptor-translation overhead. Anyone who wires
this into the dispatch path without first measuring a win on real
hardware is adding a translation layer for nothing. Same posture as
rocfft_ops: a capability surface, not a speedup.

WHAT IT IS ACTUALLY GOOD FOR.
  * A SECOND, INDEPENDENT ROUTE TO MIOpen conv that does not go through
    PyTorch's own ATen/MIOpen integration -- useful as a cross-check when
    a conv result or a MIOpen algorithm choice is under suspicion (see
    miopen_fallback.py, which exists precisely because MIOpen's algorithm
    search misbehaves on this hardware for some shapes). algorithms()
    below exposes MIOpen's ranked solution list through cuDNN's
    GetConvolution*Algorithm_v7 surface, which is a different view of the
    solver set than anything torch surfaces.
  * VALIDATING THE ZLUDA BUILD ITSELF. If you are building ZLUDA in this
    tree for its real purpose (running unmodified CUDA binaries on the
    7900 XTX), available() plus conv2d() against a torch reference is a
    fast end-to-end check that the cuDNN half of that build works.

SCOPE -- SMALL, AND FIXED BY ZLUDA, NOT BY THIS FILE. ZLUDA declares 220
cuDNN 9 entry points (cuda_macros/src/cudnn9.rs) and implements 34 of
them (zluda_dnn/src/lib.rs's `dual_function_declarations!` block); every
other one resolves to `impl::unimplemented()`. Two of those 34 are unsafe to call at all (landmine 2 below), leaving 32
this module binds. The implemented set is convolution and nothing else:

    conv fwd / bwd-data / bwd-filter, tensor+filter+convolution
    descriptors, the three workspace-size queries, the three
    *Algorithm_v7 queries, handle and stream management.

NOT implemented, and each of these is a hard wall, not a slow path:
the ENTIRE cudnnBackend* graph API (BackendExecute/SetAttribute/
Finalize/...), which is what the cuDNN frontend and modern PyTorch conv
and attention paths actually use; plus RNN, batchnorm, softmax,
activation, pooling, dropout, LRN, cudnnAddTensor, cudnnTransformTensor
and multi-head attention. There is no bias-add here because cuDNN's own
ConvolutionForward has none and cudnnAddTensor is not implemented --
conv2d() below adds bias with plain torch arithmetic and says so.

FOUR ABI LANDMINES, TRANSCRIBED FROM ZLUDA'S SOURCE, NOT GUESSED:

  1. BUILD --release, NEVER debug. zluda_dnn/src/impl.rs:363 defines
     `unimplemented()` as Rust's `unimplemented!()` macro under
     `#[cfg(debug_assertions)]` -- that PANICS, and a panic across the C
     ABI aborts the process. Only the `#[cfg(not(debug_assertions))]`
     arm returns CUDNN_STATUS_NOT_SUPPORTED. A debug-built libcudnn.so.9
     turns every unsupported call into a hard process kill instead of an
     error code. _load_library() below refuses a path under target/debug
     unless AMD_TUNED_TORCH_ZLUDA_CUDNN_ALLOW_DEBUG=1.

  2. TWO "IMPLEMENTED" FUNCTIONS ABORT IN EVERY PROFILE.
     `cudnnGetMaxDeviceVersion` and `cudnnGetCudartVersion` are listed
     under `implemented_no_conversion` but their bodies are `todo!()`
     (impl.rs's get_max_device_version/get_cudart_version), which panics
     regardless of debug_assertions. This module never binds or calls
     either one. Do not add them.

  3. CROSS_CORRELATION ONLY. zluda_common/src/lib.rs:851 maps
     cudnnConvolutionMode_t -> miopenConvolutionMode_t for
     CUDNN_CROSS_CORRELATION (1) alone and returns NOT_SUPPORTED for
     CUDNN_CONVOLUTION (0). That happens to be exactly what
     torch.nn.functional.conv2d means by "convolution", so this module
     hardcodes CUDNN_CROSS_CORRELATION and does not expose the choice.

  4. 2D ONLY. impl.rs's set_convolution_nd_descriptor returns
     miopenStatusErrorNotImplemented unless `array_length == 2`. There is
     no conv1d or conv3d through this path at all, whatever the cuDNN Nd
     API appears to offer. Use this package's own conv3d tiers (ck_ops,
     compile_ops.conv3d_native, fftconv_ops) for those.

STREAMS -- WHY THIS WORKS WITHOUT ZLUDA'S libcuda. zluda_dnn takes the
`cudaStream_t` handed to cudnnSetStream and passes it straight to
miopenSetStream as a `hipStream_t` (impl.rs:1174, `hipStream_t(stream)`
at :1035) -- it is a reinterpret, not a translation through ZLUDA's CUDA
driver shim. So a real HIP stream pointer out of
torch.cuda.current_stream().cuda_stream is directly valid here, and this
module can be used from an ordinary ROCm PyTorch process with no other
part of ZLUDA loaded. That is the single fact that makes this binding
viable at all.

BUILDING IT. ZLUDA is a Rust workspace and is NOT built by this
package's setup.py (which builds HIP/C++ extensions; a cargo workspace is
a different toolchain with a very long build, and this package's
convention is that third_party/ entries build themselves -- same as
torchsparse and TransformerEngine). There is no cargo in this
environment as written, so this has never been built here:

    tools/build_zluda_cudnn.sh          # cargo build --release -p zluda_dnn9

or by hand:

    cd third_party/ZLUDA && cargo build --release -p zluda_dnn9

Either leaves target/release/libcudnn.so.9 (the crate's `linux_symlinks`
metadata names libcudnn.so and libcudnn.so.9; a plain `cargo build`
produces libcudnn64_9.so and the helper script makes the symlink).

OPT-IN, AND WHY IT IS OFF BY DEFAULT. Nothing here loads unless
AMD_TUNED_TORCH_ENABLE_ZLUDA_CUDNN=1, same gate shape as
AMD_TUNED_TORCH_ENABLE_TE for te_ops. Reasons: dlopening a shim that
binds MIOpen and creates its own MIOpen handle is a real side effect to
take on merely because someone imported this package; landmine 1 above
means a wrongly-built .so can abort the process rather than fail
gracefully; and available() should not go looking for a library nobody
asked for.

STATUS -- UNVALIDATED ON HARDWARE, AND BE HONEST ABOUT WHICH HALF.
Checked for real, without a GPU: every enum ordinal, struct layout and
argument order below is transcribed from ZLUDA's own generated bindings
(cuda_types/src/cudnn{,9}.rs and cuda_macros/src/cudnn9.rs) rather than
recalled, and tests/test_zluda_cudnn_ops.py exercises the shape/output-
size/descriptor arithmetic against a mocked ctypes library. NOT checked,
and uncheckable without a built .so on a real gfx1100: whether the
ctypes argtypes and the AlgoPerf struct layout match the compiled
library at the ABI level. A mismatch there is a segfault, not a Python
exception. Run conv2d() against torch.nn.functional.conv2d on small
shapes before trusting it for anything.
"""
from __future__ import annotations

import ctypes
import os
import threading
from typing import List, Optional, Sequence, Tuple

import torch

# ---------------------------------------------------------------------------
# Enum ordinals and struct layouts, transcribed from
# third_party/ZLUDA/cuda_types/src/cudnn.rs and cudnn9.rs. These are C enum
# values with no symbolic checking possible from Python; do not hand-edit
# without re-reading those files. tests/test_zluda_cudnn_ops.py pins them.
# ---------------------------------------------------------------------------

# cudnnStatus_t is `Result<(), cudnnError_t>` transmuted to u32 in ZLUDA
# (cudnn9.rs:2654), so success is 0 and the error values are cudnnError_t's
# own NonZeroU32 constants. cuDNN 9 renumbered these away from cuDNN 8's
# 1..14 -- these are the v9 numbers, which is what libcudnn.so.9 returns.
CUDNN_STATUS_SUCCESS = 0
CUDNN_STATUS_NOT_INITIALIZED = 1001
CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH = 1002
CUDNN_STATUS_SERIALIZATION_VERSION_MISMATCH = 1003
CUDNN_STATUS_DEPRECATED = 1004
CUDNN_STATUS_LICENSE_ERROR = 1005
CUDNN_STATUS_RUNTIME_IN_PROGRESS = 1006
CUDNN_STATUS_RUNTIME_FP_OVERFLOW = 1007
CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED = 1008
CUDNN_STATUS_BAD_PARAM = 2000
CUDNN_STATUS_BAD_PARAM_NULL_POINTER = 2002
CUDNN_STATUS_BAD_PARAM_MISALIGNED_POINTER = 2003
CUDNN_STATUS_BAD_PARAM_NOT_FINALIZED = 2004
CUDNN_STATUS_BAD_PARAM_OUT_OF_BOUND = 2005
CUDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT = 2006
CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH = 2007
CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH = 2008
CUDNN_STATUS_NOT_SUPPORTED = 3000
CUDNN_STATUS_NOT_SUPPORTED_SHAPE = 3002
CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE = 3003
CUDNN_STATUS_NOT_SUPPORTED_LAYOUT = 3004
CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH = 3007
CUDNN_STATUS_INTERNAL_ERROR = 4000
CUDNN_STATUS_EXECUTION_FAILED = 5000

_STATUS_NAMES = {
    CUDNN_STATUS_SUCCESS: "CUDNN_STATUS_SUCCESS",
    CUDNN_STATUS_NOT_INITIALIZED: "CUDNN_STATUS_NOT_INITIALIZED",
    CUDNN_STATUS_BAD_PARAM: "CUDNN_STATUS_BAD_PARAM",
    CUDNN_STATUS_BAD_PARAM_NULL_POINTER: "CUDNN_STATUS_BAD_PARAM_NULL_POINTER",
    CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH: "CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH",
    CUDNN_STATUS_NOT_SUPPORTED: "CUDNN_STATUS_NOT_SUPPORTED",
    CUDNN_STATUS_NOT_SUPPORTED_SHAPE: "CUDNN_STATUS_NOT_SUPPORTED_SHAPE",
    CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE: "CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE",
    CUDNN_STATUS_NOT_SUPPORTED_LAYOUT: "CUDNN_STATUS_NOT_SUPPORTED_LAYOUT",
    CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH: "CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH",
    CUDNN_STATUS_INTERNAL_ERROR: "CUDNN_STATUS_INTERNAL_ERROR",
    CUDNN_STATUS_EXECUTION_FAILED: "CUDNN_STATUS_EXECUTION_FAILED",
}

# cudnnDataType_t -- cudnn9.rs
CUDNN_DATA_FLOAT = 0
CUDNN_DATA_DOUBLE = 1
CUDNN_DATA_HALF = 2
CUDNN_DATA_INT8 = 3
CUDNN_DATA_INT32 = 4
CUDNN_DATA_BFLOAT16 = 9
CUDNN_DATA_INT64 = 10

# cudnnTensorFormat_t -- cudnn.rs:52
CUDNN_TENSOR_NCHW = 0
CUDNN_TENSOR_NHWC = 1
CUDNN_TENSOR_NCHW_VECT_C = 2

# cudnnConvolutionMode_t -- cudnn.rs:143. Landmine 3: only the second of
# these two survives ZLUDA's FromCuda mapping.
CUDNN_CONVOLUTION = 0
CUDNN_CROSS_CORRELATION = 1

# cudnnMathType_t -- cudnn.rs:26. Note the gap: there is no 2.
CUDNN_DEFAULT_MATH = 0
CUDNN_TENSOR_OP_MATH = 1
CUDNN_FMA_MATH = 3

# cudnnDeterminism_t -- cudnn.rs:518
CUDNN_NON_DETERMINISTIC = 0
CUDNN_DETERMINISTIC = 1

# cudnnConvolutionFwdAlgo_t -- cudnn.rs:718
CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0
CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1
CUDNN_CONVOLUTION_FWD_ALGO_GEMM = 2
CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = 3
CUDNN_CONVOLUTION_FWD_ALGO_FFT = 4
CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = 5
CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = 6
CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = 7
CUDNN_CONVOLUTION_FWD_ALGO_COUNT = 8

# cudnnConvolutionBwdFilterAlgo_t -- cudnn.rs:766
CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0
CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = 1
CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = 2
CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = 3
CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD = 4
CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = 5
CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING = 6
CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT = 7

# cudnnConvolutionBwdDataAlgo_t -- cudnn.rs:809
CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0
CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = 1
CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = 2
CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = 3
CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = 4
CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = 5
CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT = 6

# cuda_types/src/cudnn9.rs:18 -- what ZLUDA's cudnnGetVersion() reports.
ZLUDA_REPORTED_CUDNN_VERSION = 91301

_TORCH_DTYPE_TO_CUDNN = {
    torch.float32: CUDNN_DATA_FLOAT,
    torch.float64: CUDNN_DATA_DOUBLE,
    torch.float16: CUDNN_DATA_HALF,
    torch.bfloat16: CUDNN_DATA_BFLOAT16,
}


class _AlgoPerf(ctypes.Structure):
    """cudnnConvolutionFwdAlgoPerfStruct, cudnn9.rs:2290. The bwd-data and
    bwd-filter perf structs (cudnn9.rs:2302 and :2324) have byte-identical
    layout -- only the first field's enum type differs, and all three are
    c_uint -- so one Structure serves all three queries."""

    _fields_ = [
        ("algo", ctypes.c_uint),
        ("status", ctypes.c_uint),
        ("time", ctypes.c_float),
        ("memory", ctypes.c_size_t),
        ("determinism", ctypes.c_uint),
        ("mathType", ctypes.c_uint),
        ("reserved", ctypes.c_int * 3),
    ]


_LIB: Optional[ctypes.CDLL] = None
_LIB_PATH: Optional[str] = None
_LIB_LOAD_ERROR: Optional[Exception] = None
_HANDLES = {}  # device index -> cudnnHandle_t
_lock = threading.RLock()

_ENABLE_ENV = "AMD_TUNED_TORCH_ENABLE_ZLUDA_CUDNN"
_LIB_ENV = "AMD_TUNED_TORCH_ZLUDA_CUDNN_LIB"
_ROOT_ENV = "AMD_TUNED_TORCH_ZLUDA_ROOT"
_ALLOW_DEBUG_ENV = "AMD_TUNED_TORCH_ZLUDA_CUDNN_ALLOW_DEBUG"


class ZludaCudnnError(RuntimeError):
    pass


class ZludaCudnnUnavailable(ZludaCudnnError):
    """The shim could not be loaded -- not built, not opted in, or not
    found. Distinct from ZludaCudnnError so a caller can tell "this tier
    is absent" from "the call failed"."""


def _status_name(status: int) -> str:
    return _STATUS_NAMES.get(status, f"cudnnStatus_t({status})")


def _check(status: int, what: str) -> None:
    if status != CUDNN_STATUS_SUCCESS:
        detail = ""
        # cudnnGetLastErrorString is one of the three implemented
        # no-conversion entry points and is safe to call (unlike its two
        # todo!() neighbours -- see landmine 2 in the module docstring).
        if _LIB is not None:
            try:
                buf = ctypes.create_string_buffer(512)
                _LIB.cudnnGetLastErrorString(buf, 512)
                text = buf.value.decode("utf-8", errors="replace").strip()
                if text:
                    detail = f": {text}"
            except Exception:
                pass
        if status == CUDNN_STATUS_NOT_SUPPORTED:
            detail += (
                "  -- CUDNN_STATUS_NOT_SUPPORTED from ZLUDA usually means the "
                "entry point is one of the 186 it does not implement, or a "
                "parameter outside the conv-only subset (see this module's "
                "docstring)."
            )
        raise ZludaCudnnError(f"{what} failed: {_status_name(status)}{detail}")


# ---------------------------------------------------------------------------
# Library discovery and loading
# ---------------------------------------------------------------------------

def _zluda_root() -> str:
    override = os.environ.get(_ROOT_ENV)
    if override:
        return override
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "third_party", "ZLUDA")


def _candidate_paths() -> List[str]:
    """Where a built ZLUDA libcudnn might be, most-preferred first. The
    explicit env override wins; then release builds; then, only so the
    error message can name it, the debug build that _load_library refuses."""
    explicit = os.environ.get(_LIB_ENV)
    if explicit:
        return [explicit]
    root = _zluda_root()
    names = ("libcudnn.so.9", "libcudnn.so", "libcudnn64_9.so")
    out = []
    for profile in ("release", "release-lto", "debug"):
        for name in names:
            out.append(os.path.join(root, "target", profile, name))
    return out


def _configure_signatures(lib) -> None:
    """Bind argtypes/restype for the 32 entry points ZLUDA actually
    implements AND that are safe to call.

    Deliberately absent: cudnnGetMaxDeviceVersion and
    cudnnGetCudartVersion. Both are in ZLUDA's `implemented_no_conversion`
    list, and both have `todo!()` bodies that panic across the C ABI and
    abort the process. Binding them here would be an invitation to call
    them. See landmine 2 in the module docstring."""
    c_int = ctypes.c_int
    c_uint = ctypes.c_uint
    c_void_p = ctypes.c_void_p
    c_size_t = ctypes.c_size_t
    p_int = ctypes.POINTER(ctypes.c_int)
    p_uint = ctypes.POINTER(ctypes.c_uint)
    p_size = ctypes.POINTER(ctypes.c_size_t)
    p_void_p = ctypes.POINTER(ctypes.c_void_p)
    p_perf = ctypes.POINTER(_AlgoPerf)

    sigs = {
        # handle / stream
        "cudnnCreate": ([p_void_p], c_uint),
        "cudnnDestroy": ([c_void_p], c_uint),
        "cudnnSetStream": ([c_void_p, c_void_p], c_uint),
        "cudnnGetStream": ([c_void_p, p_void_p], c_uint),
        # version / errors -- cudnnGetVersion returns size_t, not a status
        "cudnnGetVersion": ([], c_size_t),
        "cudnnGetErrorString": ([c_uint], ctypes.c_char_p),
        "cudnnGetLastErrorString": ([ctypes.c_char_p, c_size_t], None),
        # tensor descriptors
        "cudnnCreateTensorDescriptor": ([p_void_p], c_uint),
        "cudnnDestroyTensorDescriptor": ([c_void_p], c_uint),
        "cudnnSetTensor4dDescriptor": (
            [c_void_p, c_uint, c_uint, c_int, c_int, c_int, c_int], c_uint),
        "cudnnSetTensorNdDescriptor": (
            [c_void_p, c_uint, c_int, p_int, p_int], c_uint),
        "cudnnGetTensorNdDescriptor": (
            [c_void_p, c_int, p_uint, p_int, p_int, p_int], c_uint),
        # filter descriptors
        "cudnnCreateFilterDescriptor": ([p_void_p], c_uint),
        "cudnnDestroyFilterDescriptor": ([c_void_p], c_uint),
        "cudnnSetFilter4dDescriptor": (
            [c_void_p, c_uint, c_uint, c_int, c_int, c_int, c_int], c_uint),
        "cudnnSetFilterNdDescriptor": (
            [c_void_p, c_uint, c_uint, c_int, p_int], c_uint),
        "cudnnGetFilterNdDescriptor": (
            [c_void_p, c_int, p_uint, p_uint, p_int, p_int], c_uint),
        # convolution descriptors
        "cudnnCreateConvolutionDescriptor": ([p_void_p], c_uint),
        "cudnnDestroyConvolutionDescriptor": ([c_void_p], c_uint),
        "cudnnSetConvolution2dDescriptor": (
            [c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_uint, c_uint],
            c_uint),
        "cudnnSetConvolutionNdDescriptor": (
            [c_void_p, c_int, p_int, p_int, p_int, c_uint, c_uint], c_uint),
        "cudnnSetConvolutionGroupCount": ([c_void_p, c_int], c_uint),
        "cudnnSetConvolutionMathType": ([c_void_p, c_uint], c_uint),
        # algorithm queries
        "cudnnGetConvolutionForwardAlgorithm_v7": (
            [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, p_int, p_perf],
            c_uint),
        "cudnnGetConvolutionBackwardDataAlgorithm_v7": (
            [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, p_int, p_perf],
            c_uint),
        "cudnnGetConvolutionBackwardFilterAlgorithm_v7": (
            [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, p_int, p_perf],
            c_uint),
        # workspace sizes
        "cudnnGetConvolutionForwardWorkspaceSize": (
            [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_uint, p_size], c_uint),
        "cudnnGetConvolutionBackwardDataWorkspaceSize": (
            [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_uint, p_size], c_uint),
        "cudnnGetConvolutionBackwardFilterWorkspaceSize": (
            [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_uint, p_size], c_uint),
        # execution
        "cudnnConvolutionForward": (
            [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
             c_uint, c_void_p, c_size_t, c_void_p, c_void_p, c_void_p], c_uint),
        "cudnnConvolutionBackwardData": (
            [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
             c_uint, c_void_p, c_size_t, c_void_p, c_void_p, c_void_p], c_uint),
        "cudnnConvolutionBackwardFilter": (
            [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
             c_uint, c_void_p, c_size_t, c_void_p, c_void_p, c_void_p], c_uint),
    }
    for name, (argtypes, restype) in sigs.items():
        fn = getattr(lib, name)  # AttributeError here = a typo or a stale .so
        fn.argtypes = argtypes
        fn.restype = restype


def implemented_functions() -> Tuple[str, ...]:
    """The cuDNN entry points this module binds -- ZLUDA's implemented set
    minus the two that abort. Everything else in cuDNN returns
    CUDNN_STATUS_NOT_SUPPORTED from the shim."""
    return (
        "cudnnCreate", "cudnnDestroy", "cudnnSetStream", "cudnnGetStream",
        "cudnnGetVersion", "cudnnGetErrorString", "cudnnGetLastErrorString",
        "cudnnCreateTensorDescriptor", "cudnnDestroyTensorDescriptor",
        "cudnnSetTensor4dDescriptor", "cudnnSetTensorNdDescriptor",
        "cudnnGetTensorNdDescriptor",
        "cudnnCreateFilterDescriptor", "cudnnDestroyFilterDescriptor",
        "cudnnSetFilter4dDescriptor", "cudnnSetFilterNdDescriptor",
        "cudnnGetFilterNdDescriptor",
        "cudnnCreateConvolutionDescriptor", "cudnnDestroyConvolutionDescriptor",
        "cudnnSetConvolution2dDescriptor", "cudnnSetConvolutionNdDescriptor",
        "cudnnSetConvolutionGroupCount", "cudnnSetConvolutionMathType",
        "cudnnGetConvolutionForwardAlgorithm_v7",
        "cudnnGetConvolutionBackwardDataAlgorithm_v7",
        "cudnnGetConvolutionBackwardFilterAlgorithm_v7",
        "cudnnGetConvolutionForwardWorkspaceSize",
        "cudnnGetConvolutionBackwardDataWorkspaceSize",
        "cudnnGetConvolutionBackwardFilterWorkspaceSize",
        "cudnnConvolutionForward", "cudnnConvolutionBackwardData",
        "cudnnConvolutionBackwardFilter",
    )


def _load_library() -> None:
    global _LIB, _LIB_PATH, _LIB_LOAD_ERROR
    if _LIB is not None:
        return
    if _LIB_LOAD_ERROR is not None:
        raise ZludaCudnnUnavailable(str(_LIB_LOAD_ERROR))

    if os.environ.get(_ENABLE_ENV, "0") != "1":
        _LIB_LOAD_ERROR = ZludaCudnnUnavailable(
            f"not enabled -- set {_ENABLE_ENV}=1 to opt in. This tier dlopens "
            "ZLUDA's cuDNN-ABI shim and creates a MIOpen handle, which is not "
            "something to do merely because amd_tuned_torch was imported."
        )
        raise _LIB_LOAD_ERROR

    allow_debug = os.environ.get(_ALLOW_DEBUG_ENV, "0") == "1"
    candidates = _candidate_paths()
    tried = []
    for path in candidates:
        if not os.path.exists(path):
            tried.append(f"{path} (absent)")
            continue
        if os.sep + "debug" + os.sep in path and not allow_debug:
            tried.append(
                f"{path} (REFUSED: debug build -- ZLUDA's unimplemented() is "
                f"Rust's unimplemented!() under debug_assertions and ABORTS the "
                f"process instead of returning CUDNN_STATUS_NOT_SUPPORTED. "
                f"Rebuild with --release, or set {_ALLOW_DEBUG_ENV}=1 if you "
                f"really want that.)"
            )
            continue
        try:
            lib = ctypes.CDLL(path, mode=ctypes.RTLD_LOCAL)
        except OSError as exc:
            tried.append(f"{path} ({exc})")
            continue
        try:
            _configure_signatures(lib)
        except AttributeError as exc:
            tried.append(f"{path} (missing symbol: {exc})")
            continue
        _LIB = lib
        _LIB_PATH = path
        return

    looked = "\n".join(f"      {t}" for t in tried)
    _LIB_LOAD_ERROR = ZludaCudnnUnavailable(
        "ZLUDA's libcudnn.so.9 was not loadable.\n"
        f"  Looked in:\n{looked}\n\n"
        "  To build it:  tools/build_zluda_cudnn.sh\n"
        "                (or: cd third_party/ZLUDA && "
        "cargo build --release -p zluda_dnn9)\n"
        f"  To point at an existing build:  {_LIB_ENV}=/path/to/libcudnn.so.9\n"
    )
    raise _LIB_LOAD_ERROR


def available() -> bool:
    """True if the opt-in flag is set and ZLUDA's libcudnn.so.9 loaded with
    every expected symbol present. Never raises -- load_error() explains a
    False."""
    try:
        _load_library()
        return True
    except ZludaCudnnError:
        return False


def load_error() -> Optional[str]:
    """Why available() is False, or None if it is True / not yet checked."""
    return str(_LIB_LOAD_ERROR) if _LIB_LOAD_ERROR is not None else None


def library_path() -> Optional[str]:
    """The .so actually loaded, or None."""
    return _LIB_PATH


def version() -> int:
    """What the shim reports for cudnnGetVersion(). This is a CONSTANT
    baked into ZLUDA (cuda_types/src/cudnn9.rs:18 -- 91301, i.e. "cuDNN
    9.13.1"), not a probe of anything real; it exists so CUDA software
    doing a version check accepts the shim. It says nothing about MIOpen's
    version or this machine."""
    _load_library()
    return int(_LIB.cudnnGetVersion())


def error_string(status: int) -> str:
    """ZLUDA's own text for a cudnnStatus_t value."""
    _load_library()
    raw = _LIB.cudnnGetErrorString(ctypes.c_uint(status))
    return raw.decode("utf-8", errors="replace") if raw else _status_name(status)


# ---------------------------------------------------------------------------
# Handles. One per device, created lazily, reused. ZLUDA's cudnnCreate makes
# a MIOpen handle, which is not cheap enough to do per call.
# ---------------------------------------------------------------------------

def _handle_for_current_device() -> ctypes.c_void_p:
    _load_library()
    index = torch.cuda.current_device()
    with _lock:
        handle = _HANDLES.get(index)
        if handle is None:
            handle = ctypes.c_void_p()
            _check(_LIB.cudnnCreate(ctypes.byref(handle)), "cudnnCreate")
            _HANDLES[index] = handle
        # Re-bound on every call, not just at creation: torch's current
        # stream can change between calls (autograd, torch.cuda.stream()
        # blocks), and a handle left on a stale stream would race the
        # caller's own work. The pointer goes straight through to
        # miopenSetStream as a hipStream_t -- see "STREAMS" above.
        stream = torch.cuda.current_stream().cuda_stream
        _check(_LIB.cudnnSetStream(handle, ctypes.c_void_p(stream)), "cudnnSetStream")
        return handle


def reset() -> None:
    """Destroy every cached handle. Safe to call when unavailable."""
    global _HANDLES
    with _lock:
        if _LIB is not None:
            for handle in _HANDLES.values():
                try:
                    _LIB.cudnnDestroy(handle)
                except Exception:
                    pass
        _HANDLES = {}


# ---------------------------------------------------------------------------
# Descriptor helpers
# ---------------------------------------------------------------------------

def _cudnn_dtype(dtype: torch.dtype) -> int:
    try:
        return _TORCH_DTYPE_TO_CUDNN[dtype]
    except KeyError:
        raise ZludaCudnnError(
            f"unsupported dtype {dtype} -- this path handles "
            f"{sorted(str(d) for d in _TORCH_DTYPE_TO_CUDNN)}"
        ) from None


def _pair(value, name: str) -> Tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    values = tuple(int(v) for v in value)
    if len(values) != 2:
        raise ZludaCudnnError(f"{name} must be an int or a pair, got {value!r}")
    return values


class _Descriptors:
    """Creates the four descriptors a conv2d call needs and destroys them on
    exit. cuDNN descriptors are process-global heap objects; leaking them on
    an exception path would be a slow leak with no error message."""

    def __init__(self):
        self._tensors = []
        self._filters = []
        self._convs = []

    def tensor4d(self, n, c, h, w, dtype) -> ctypes.c_void_p:
        desc = ctypes.c_void_p()
        _check(_LIB.cudnnCreateTensorDescriptor(ctypes.byref(desc)),
               "cudnnCreateTensorDescriptor")
        self._tensors.append(desc)
        _check(
            _LIB.cudnnSetTensor4dDescriptor(
                desc, CUDNN_TENSOR_NCHW, _cudnn_dtype(dtype),
                int(n), int(c), int(h), int(w)),
            "cudnnSetTensor4dDescriptor")
        return desc

    def filter4d(self, k, c, r, s, dtype) -> ctypes.c_void_p:
        desc = ctypes.c_void_p()
        _check(_LIB.cudnnCreateFilterDescriptor(ctypes.byref(desc)),
               "cudnnCreateFilterDescriptor")
        self._filters.append(desc)
        _check(
            _LIB.cudnnSetFilter4dDescriptor(
                desc, _cudnn_dtype(dtype), CUDNN_TENSOR_NCHW,
                int(k), int(c), int(r), int(s)),
            "cudnnSetFilter4dDescriptor")
        return desc

    def convolution(self, padding, stride, dilation, groups, compute_dtype,
                    math_type) -> ctypes.c_void_p:
        desc = ctypes.c_void_p()
        _check(_LIB.cudnnCreateConvolutionDescriptor(ctypes.byref(desc)),
               "cudnnCreateConvolutionDescriptor")
        self._convs.append(desc)
        # Set2d rather than SetNd: ZLUDA implements both, both end at
        # miopenInitConvolutionDescriptor, and SetNd additionally rejects
        # anything but arrayLength == 2 anyway (landmine 4). The 2d form
        # says that constraint in the name.
        _check(
            _LIB.cudnnSetConvolution2dDescriptor(
                desc, padding[0], padding[1], stride[0], stride[1],
                dilation[0], dilation[1],
                CUDNN_CROSS_CORRELATION,  # landmine 3: the only accepted mode
                _cudnn_dtype(compute_dtype)),
            "cudnnSetConvolution2dDescriptor")
        if groups != 1:
            _check(_LIB.cudnnSetConvolutionGroupCount(desc, int(groups)),
                   "cudnnSetConvolutionGroupCount")
        if math_type is not None:
            _check(_LIB.cudnnSetConvolutionMathType(desc, int(math_type)),
                   "cudnnSetConvolutionMathType")
        return desc

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for desc in self._tensors:
            try:
                _LIB.cudnnDestroyTensorDescriptor(desc)
            except Exception:
                pass
        for desc in self._filters:
            try:
                _LIB.cudnnDestroyFilterDescriptor(desc)
            except Exception:
                pass
        for desc in self._convs:
            try:
                _LIB.cudnnDestroyConvolutionDescriptor(desc)
            except Exception:
                pass
        return False


def conv2d_output_shape(x_shape, w_shape, stride, padding, dilation) -> Tuple[int, ...]:
    """Standard conv2d output shape. Kept as a plain function so the
    arithmetic is unit-testable without a library present."""
    n, _, h, w = x_shape
    k, _, r, s = w_shape
    out_h = (h + 2 * padding[0] - dilation[0] * (r - 1) - 1) // stride[0] + 1
    out_w = (w + 2 * padding[1] - dilation[1] * (s - 1) - 1) // stride[1] + 1
    return (n, k, out_h, out_w)


def _scalar_ptr(value: float, dtype: torch.dtype):
    """cuDNN's alpha/beta convention: the host scalar is a double only when
    the compute type is double, and a float for every other compute type
    including half and bfloat16. Passing the wrong width here reads garbage
    off the stack rather than failing."""
    if dtype == torch.float64:
        return ctypes.byref(ctypes.c_double(float(value)))
    return ctypes.byref(ctypes.c_float(float(value)))


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """cuDNN accumulates half/bfloat16 convolutions in float32 -- the
    convolution descriptor's computeType, distinct from the tensors' own
    data type."""
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _require_contiguous(t: torch.Tensor, name: str) -> torch.Tensor:
    if not t.is_cuda:
        raise ZludaCudnnError(
            f"{name} must be on the GPU -- this calls MIOpen through a HIP "
            f"stream, there is no CPU path here")
    return t.contiguous()


def _pick_algo(perfs: Sequence[_AlgoPerf], count: int, workspace_limit: int,
               what: str) -> Tuple[int, int]:
    """First entry whose own status is SUCCESS and whose workspace fits.
    The *Algorithm_v7 queries return results already ranked by MIOpen's
    solver ordering, and entries that MIOpen could not solve carry a
    non-success status in the struct rather than being omitted -- so
    "first usable", not "best time", is the right rule and matches what
    cuDNN callers normally do with v7 results."""
    for i in range(count):
        perf = perfs[i]
        if perf.status != CUDNN_STATUS_SUCCESS:
            continue
        if perf.memory > workspace_limit:
            continue
        return int(perf.algo), int(perf.memory)
    raise ZludaCudnnError(
        f"{what}: none of the {count} algorithm(s) MIOpen returned is usable "
        f"within the {workspace_limit} byte workspace limit. Raise "
        f"workspace_limit, or use stock torch for this shape.")


def algorithms(x_shape, w_shape, dtype=torch.float32, stride=1, padding=0,
               dilation=1, groups=1, direction="forward",
               max_results: int = 8) -> List[dict]:
    """MIOpen's ranked solution list for a conv2d problem, as seen through
    cuDNN's GetConvolution*Algorithm_v7 surface.

    `direction` is "forward", "backward_data" or "backward_filter". Shapes
    are (N, C, H, W) and (K, C/groups, R, S). Returns one dict per entry
    with algo/status/time/memory/determinism/math_type -- `time` is
    MIOpen's own estimate or measurement, not a benchmark this module ran.

    This is the diagnostic the module is really for: a view of the solver
    set that PyTorch does not expose."""
    _load_library()
    stride = _pair(stride, "stride")
    padding = _pair(padding, "padding")
    dilation = _pair(dilation, "dilation")
    if direction not in ("forward", "backward_data", "backward_filter"):
        raise ZludaCudnnError(f"unknown direction {direction!r}")

    handle = _handle_for_current_device()
    y_shape = conv2d_output_shape(x_shape, w_shape, stride, padding, dilation)
    with _Descriptors() as d:
        x_desc = d.tensor4d(*x_shape, dtype)
        y_desc = d.tensor4d(*y_shape, dtype)
        w_desc = d.filter4d(*w_shape, dtype)
        conv_desc = d.convolution(padding, stride, dilation, groups,
                                  _compute_dtype(dtype), None)
        perfs = (_AlgoPerf * max_results)()
        returned = ctypes.c_int(0)
        if direction == "forward":
            fn, args = _LIB.cudnnGetConvolutionForwardAlgorithm_v7, (
                handle, x_desc, w_desc, conv_desc, y_desc)
        elif direction == "backward_data":
            fn, args = _LIB.cudnnGetConvolutionBackwardDataAlgorithm_v7, (
                handle, w_desc, y_desc, conv_desc, x_desc)
        else:
            fn, args = _LIB.cudnnGetConvolutionBackwardFilterAlgorithm_v7, (
                handle, x_desc, y_desc, conv_desc, w_desc)
        _check(fn(*args, max_results, ctypes.byref(returned), perfs),
               f"cudnnGetConvolution{direction}Algorithm_v7")
        return [
            {
                "algo": int(perfs[i].algo),
                "status": _status_name(int(perfs[i].status)),
                "time": float(perfs[i].time),
                "memory": int(perfs[i].memory),
                "deterministic": bool(perfs[i].determinism == CUDNN_DETERMINISTIC),
                "math_type": int(perfs[i].mathType),
            }
            for i in range(returned.value)
        ]


# ---------------------------------------------------------------------------
# The three convolution entry points
# ---------------------------------------------------------------------------

_DEFAULT_WORKSPACE_LIMIT = 1 << 30  # 1 GiB


def conv2d(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
           stride=1, padding=0, dilation=1, groups: int = 1,
           algo: Optional[int] = None,
           workspace_limit: int = _DEFAULT_WORKSPACE_LIMIT,
           math_type: Optional[int] = None) -> torch.Tensor:
    """conv2d forward through ZLUDA's cuDNN shim into MIOpen.

    Signature matches torch.nn.functional.conv2d so the two can be swapped
    in a comparison, with three differences worth knowing:

      * NO AUTOGRAD. This is a raw library call, not an autograd.Function.
        Use conv2d_backward_data/conv2d_backward_filter explicitly if you
        need gradients. (Wiring it into autograd would imply this belongs
        in a training path, which the module docstring argues against.)
      * BIAS IS ADDED IN TORCH, not by cuDNN. cudnnConvolutionForward has
        no bias parameter and ZLUDA does not implement cudnnAddTensor, so
        `bias` here costs an extra elementwise kernel that a real fused
        cuDNN path would not.
      * NCHW CONTIGUOUS ONLY. Inputs are made contiguous; channels_last is
        not plumbed through (ZLUDA's get_tensor_nd_descriptor hardcodes
        CUDNN_TENSOR_NCHW on readback, impl.rs:559).
    """
    _load_library()
    stride = _pair(stride, "stride")
    padding = _pair(padding, "padding")
    dilation = _pair(dilation, "dilation")
    x = _require_contiguous(x, "x")
    weight = _require_contiguous(weight, "weight")
    if x.dim() != 4 or weight.dim() != 4:
        raise ZludaCudnnError(
            f"conv2d needs 4D x and weight, got {tuple(x.shape)} and "
            f"{tuple(weight.shape)}. ZLUDA's conv path is 2D-only -- see "
            f"landmine 4 in this module's docstring.")
    if x.dtype != weight.dtype:
        raise ZludaCudnnError(
            f"x and weight must share a dtype, got {x.dtype} and {weight.dtype}")

    handle = _handle_for_current_device()
    y_shape = conv2d_output_shape(tuple(x.shape), tuple(weight.shape),
                                  stride, padding, dilation)
    if y_shape[2] <= 0 or y_shape[3] <= 0:
        raise ZludaCudnnError(
            f"degenerate output shape {y_shape} for x={tuple(x.shape)} "
            f"w={tuple(weight.shape)} stride={stride} padding={padding} "
            f"dilation={dilation}")
    y = torch.empty(y_shape, dtype=x.dtype, device=x.device)
    compute = _compute_dtype(x.dtype)

    with _Descriptors() as d:
        x_desc = d.tensor4d(*x.shape, x.dtype)
        y_desc = d.tensor4d(*y_shape, x.dtype)
        w_desc = d.filter4d(*weight.shape, weight.dtype)
        conv_desc = d.convolution(padding, stride, dilation, groups, compute,
                                  math_type)

        if algo is None:
            perfs = (_AlgoPerf * CUDNN_CONVOLUTION_FWD_ALGO_COUNT)()
            returned = ctypes.c_int(0)
            _check(
                _LIB.cudnnGetConvolutionForwardAlgorithm_v7(
                    handle, x_desc, w_desc, conv_desc, y_desc,
                    CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
                    ctypes.byref(returned), perfs),
                "cudnnGetConvolutionForwardAlgorithm_v7")
            algo, workspace_bytes = _pick_algo(
                perfs, returned.value, workspace_limit, "conv2d forward")
        else:
            size = ctypes.c_size_t(0)
            _check(
                _LIB.cudnnGetConvolutionForwardWorkspaceSize(
                    handle, x_desc, w_desc, conv_desc, y_desc, algo,
                    ctypes.byref(size)),
                "cudnnGetConvolutionForwardWorkspaceSize")
            workspace_bytes = size.value

        workspace, ws_ptr = _workspace(workspace_bytes, x.device)
        _check(
            _LIB.cudnnConvolutionForward(
                handle,
                _scalar_ptr(1.0, compute),
                x_desc, ctypes.c_void_p(x.data_ptr()),
                w_desc, ctypes.c_void_p(weight.data_ptr()),
                conv_desc, algo,
                ws_ptr, workspace_bytes,
                _scalar_ptr(0.0, compute),
                y_desc, ctypes.c_void_p(y.data_ptr())),
            "cudnnConvolutionForward")
        del workspace

    if bias is not None:
        # Not fused -- see the docstring. Shaped for NCHW broadcast.
        y = y + bias.reshape(1, -1, 1, 1).to(y.dtype)
    return y


def conv2d_backward_data(weight: torch.Tensor, grad_output: torch.Tensor,
                         input_shape: Sequence[int], stride=1, padding=0,
                         dilation=1, groups: int = 1, algo: Optional[int] = None,
                         workspace_limit: int = _DEFAULT_WORKSPACE_LIMIT,
                         math_type: Optional[int] = None) -> torch.Tensor:
    """dL/dx for conv2d. `input_shape` is the forward input's (N, C, H, W),
    which cuDNN needs because it is not recoverable from the other two."""
    _load_library()
    stride = _pair(stride, "stride")
    padding = _pair(padding, "padding")
    dilation = _pair(dilation, "dilation")
    weight = _require_contiguous(weight, "weight")
    grad_output = _require_contiguous(grad_output, "grad_output")
    input_shape = tuple(int(v) for v in input_shape)
    if len(input_shape) != 4:
        raise ZludaCudnnError(f"input_shape must be 4D, got {input_shape}")

    handle = _handle_for_current_device()
    grad_input = torch.empty(input_shape, dtype=grad_output.dtype,
                             device=grad_output.device)
    compute = _compute_dtype(grad_output.dtype)

    with _Descriptors() as d:
        dx_desc = d.tensor4d(*input_shape, grad_output.dtype)
        dy_desc = d.tensor4d(*grad_output.shape, grad_output.dtype)
        w_desc = d.filter4d(*weight.shape, weight.dtype)
        conv_desc = d.convolution(padding, stride, dilation, groups, compute,
                                  math_type)

        if algo is None:
            perfs = (_AlgoPerf * CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT)()
            returned = ctypes.c_int(0)
            _check(
                _LIB.cudnnGetConvolutionBackwardDataAlgorithm_v7(
                    handle, w_desc, dy_desc, conv_desc, dx_desc,
                    CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT,
                    ctypes.byref(returned), perfs),
                "cudnnGetConvolutionBackwardDataAlgorithm_v7")
            algo, workspace_bytes = _pick_algo(
                perfs, returned.value, workspace_limit, "conv2d backward-data")
        else:
            size = ctypes.c_size_t(0)
            _check(
                _LIB.cudnnGetConvolutionBackwardDataWorkspaceSize(
                    handle, w_desc, dy_desc, conv_desc, dx_desc, algo,
                    ctypes.byref(size)),
                "cudnnGetConvolutionBackwardDataWorkspaceSize")
            workspace_bytes = size.value

        workspace, ws_ptr = _workspace(workspace_bytes, grad_output.device)
        _check(
            _LIB.cudnnConvolutionBackwardData(
                handle,
                _scalar_ptr(1.0, compute),
                w_desc, ctypes.c_void_p(weight.data_ptr()),
                dy_desc, ctypes.c_void_p(grad_output.data_ptr()),
                conv_desc, algo,
                ws_ptr, workspace_bytes,
                _scalar_ptr(0.0, compute),
                dx_desc, ctypes.c_void_p(grad_input.data_ptr())),
            "cudnnConvolutionBackwardData")
        del workspace
    return grad_input


def conv2d_backward_filter(x: torch.Tensor, grad_output: torch.Tensor,
                           weight_shape: Sequence[int], stride=1, padding=0,
                           dilation=1, groups: int = 1, algo: Optional[int] = None,
                           workspace_limit: int = _DEFAULT_WORKSPACE_LIMIT,
                           math_type: Optional[int] = None) -> torch.Tensor:
    """dL/dw for conv2d. `weight_shape` is (K, C/groups, R, S)."""
    _load_library()
    stride = _pair(stride, "stride")
    padding = _pair(padding, "padding")
    dilation = _pair(dilation, "dilation")
    x = _require_contiguous(x, "x")
    grad_output = _require_contiguous(grad_output, "grad_output")
    weight_shape = tuple(int(v) for v in weight_shape)
    if len(weight_shape) != 4:
        raise ZludaCudnnError(f"weight_shape must be 4D, got {weight_shape}")

    handle = _handle_for_current_device()
    grad_weight = torch.empty(weight_shape, dtype=x.dtype, device=x.device)
    compute = _compute_dtype(x.dtype)

    with _Descriptors() as d:
        x_desc = d.tensor4d(*x.shape, x.dtype)
        dy_desc = d.tensor4d(*grad_output.shape, grad_output.dtype)
        dw_desc = d.filter4d(*weight_shape, x.dtype)
        conv_desc = d.convolution(padding, stride, dilation, groups, compute,
                                  math_type)

        if algo is None:
            perfs = (_AlgoPerf * CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT)()
            returned = ctypes.c_int(0)
            _check(
                _LIB.cudnnGetConvolutionBackwardFilterAlgorithm_v7(
                    handle, x_desc, dy_desc, conv_desc, dw_desc,
                    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT,
                    ctypes.byref(returned), perfs),
                "cudnnGetConvolutionBackwardFilterAlgorithm_v7")
            algo, workspace_bytes = _pick_algo(
                perfs, returned.value, workspace_limit, "conv2d backward-filter")
        else:
            size = ctypes.c_size_t(0)
            _check(
                _LIB.cudnnGetConvolutionBackwardFilterWorkspaceSize(
                    handle, x_desc, dy_desc, conv_desc, dw_desc, algo,
                    ctypes.byref(size)),
                "cudnnGetConvolutionBackwardFilterWorkspaceSize")
            workspace_bytes = size.value

        workspace, ws_ptr = _workspace(workspace_bytes, x.device)
        _check(
            _LIB.cudnnConvolutionBackwardFilter(
                handle,
                _scalar_ptr(1.0, compute),
                x_desc, ctypes.c_void_p(x.data_ptr()),
                dy_desc, ctypes.c_void_p(grad_output.data_ptr()),
                conv_desc, algo,
                ws_ptr, workspace_bytes,
                _scalar_ptr(0.0, compute),
                dw_desc, ctypes.c_void_p(grad_weight.data_ptr())),
            "cudnnConvolutionBackwardFilter")
        del workspace
    return grad_weight


def _workspace(nbytes: int, device) -> Tuple[Optional[torch.Tensor], ctypes.c_void_p]:
    """Workspace out of torch's own caching allocator rather than a raw
    hipMalloc, so it participates in the same pool as everything else and
    empty_cache() can reclaim it -- the fragmentation concern
    miopen_fallback.py documents applies here too. The tensor is returned
    alongside the pointer so the caller keeps it alive across the call."""
    if nbytes <= 0:
        return None, ctypes.c_void_p(0)
    buf = torch.empty(nbytes, dtype=torch.uint8, device=device)
    return buf, ctypes.c_void_p(buf.data_ptr())


def describe() -> str:
    """One-paragraph status line for rocm_env_check-style reporting."""
    if not available():
        return f"zluda_cudnn: unavailable -- {load_error()}"
    return (
        f"zluda_cudnn: {library_path()} reporting cudnnGetVersion()={version()} "
        f"(a constant baked into ZLUDA, not a probe); {len(implemented_functions())} "
        f"entry points bound, conv2d-only, backed by MIOpen"
    )
