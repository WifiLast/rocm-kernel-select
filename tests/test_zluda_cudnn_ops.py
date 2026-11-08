"""Tests for amd_tuned_torch.zluda_cudnn_ops.

There is no cargo toolchain and no ROCm device in this environment, so
ZLUDA's libcudnn.so.9 has never been built here and nothing in this file
can exercise a real convolution. What IS covered for real:

  * The opt-in gate. available() is False and load_error() explains why
    when AMD_TUNED_TORCH_ENABLE_ZLUDA_CUDNN is unset -- and the module
    does NOT dlopen anything in that state, which is the whole point of
    the gate.
  * The debug-build refusal. A target/debug path is rejected with a
    message naming the reason, because ZLUDA's unimplemented() panics
    under debug_assertions and aborts the process.
  * Every enum ordinal, and the _AlgoPerf struct's field order, against
    third_party/ZLUDA/cuda_types/src/cudnn{,9}.rs. These are C values with
    no symbolic checking possible from Python, so a transcription slip
    would otherwise only surface as wrong results on hardware.
  * _configure_signatures runs against a stand-in without raising, and
    does NOT bind the two entry points whose bodies are todo!() and would
    abort the process.
  * conv2d_output_shape / _pair / _pick_algo / _scalar_ptr / _compute_dtype
    -- the arithmetic and policy this module adds on top of the C API.

Run with:

    python -m pytest source/cmp_ext_turing/tests
"""
from __future__ import annotations

import ctypes
import os
from unittest.mock import MagicMock

import pytest
import torch

import amd_tuned_torch.zluda_cudnn_ops as z


def _reset_module_state():
    z._LIB = None
    z._LIB_PATH = None
    z._LIB_LOAD_ERROR = None
    z._HANDLES = {}


@pytest.fixture(autouse=True)
def clean_state():
    _reset_module_state()
    yield
    _reset_module_state()


# ---------------------------------------------------------------------------
# Enum ordinals -- third_party/ZLUDA/cuda_types/src/cudnn.rs and cudnn9.rs
# ---------------------------------------------------------------------------

def test_status_ordinals_match_cudnn9_bindings():
    # cudnnStatus_t is Result<(), cudnnError_t> transmuted to u32, so Ok is 0
    # and the rest are cudnnError_t's NonZeroU32 values (cudnn9.rs). cuDNN 9
    # renumbered these away from cuDNN 8's 1..14 -- these are the v9 numbers.
    assert z.CUDNN_STATUS_SUCCESS == 0
    assert z.CUDNN_STATUS_NOT_INITIALIZED == 1001
    assert z.CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH == 1002
    assert z.CUDNN_STATUS_SERIALIZATION_VERSION_MISMATCH == 1003
    assert z.CUDNN_STATUS_DEPRECATED == 1004
    assert z.CUDNN_STATUS_LICENSE_ERROR == 1005
    assert z.CUDNN_STATUS_RUNTIME_IN_PROGRESS == 1006
    assert z.CUDNN_STATUS_RUNTIME_FP_OVERFLOW == 1007
    assert z.CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED == 1008
    assert z.CUDNN_STATUS_BAD_PARAM == 2000
    assert z.CUDNN_STATUS_BAD_PARAM_NULL_POINTER == 2002
    assert z.CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH == 2008
    assert z.CUDNN_STATUS_NOT_SUPPORTED == 3000
    assert z.CUDNN_STATUS_NOT_SUPPORTED_SHAPE == 3002
    assert z.CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE == 3003
    assert z.CUDNN_STATUS_NOT_SUPPORTED_LAYOUT == 3004
    assert z.CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH == 3007
    assert z.CUDNN_STATUS_INTERNAL_ERROR == 4000


def test_data_type_ordinals_match_cudnn9_bindings():
    assert z.CUDNN_DATA_FLOAT == 0
    assert z.CUDNN_DATA_DOUBLE == 1
    assert z.CUDNN_DATA_HALF == 2
    assert z.CUDNN_DATA_INT8 == 3
    assert z.CUDNN_DATA_INT32 == 4
    assert z.CUDNN_DATA_BFLOAT16 == 9
    assert z.CUDNN_DATA_INT64 == 10


def test_format_mode_and_math_ordinals():
    assert z.CUDNN_TENSOR_NCHW == 0
    assert z.CUDNN_TENSOR_NHWC == 1
    assert z.CUDNN_TENSOR_NCHW_VECT_C == 2

    assert z.CUDNN_CONVOLUTION == 0
    assert z.CUDNN_CROSS_CORRELATION == 1

    # cudnn.rs:26 -- note the deliberate gap, there is no 2.
    assert z.CUDNN_DEFAULT_MATH == 0
    assert z.CUDNN_TENSOR_OP_MATH == 1
    assert z.CUDNN_FMA_MATH == 3

    assert z.CUDNN_NON_DETERMINISTIC == 0
    assert z.CUDNN_DETERMINISTIC == 1


def test_conv_algo_ordinals_and_counts():
    assert z.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM == 0
    assert z.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM == 1
    assert z.CUDNN_CONVOLUTION_FWD_ALGO_GEMM == 2
    assert z.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT == 3
    assert z.CUDNN_CONVOLUTION_FWD_ALGO_FFT == 4
    assert z.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING == 5
    assert z.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD == 6
    assert z.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED == 7
    assert z.CUDNN_CONVOLUTION_FWD_ALGO_COUNT == 8

    assert z.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING == 6
    assert z.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT == 7

    assert z.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED == 5
    assert z.CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT == 6


def test_algo_perf_struct_layout_matches_cudnn9():
    # cudnn9.rs:2290 -- algo, status, time, memory, determinism, mathType,
    # reserved[3]. The bwd-data (:2302) and bwd-filter (:2324) structs are
    # byte-identical, which is why one Structure serves all three queries.
    names = [name for name, _ in z._AlgoPerf._fields_]
    assert names == ["algo", "status", "time", "memory", "determinism",
                     "mathType", "reserved"]
    types = dict(z._AlgoPerf._fields_)
    assert types["algo"] is ctypes.c_uint
    assert types["status"] is ctypes.c_uint
    assert types["time"] is ctypes.c_float
    assert types["memory"] is ctypes.c_size_t
    assert types["determinism"] is ctypes.c_uint
    assert types["mathType"] is ctypes.c_uint
    # reserved is c_int[3]; a wrong length silently shifts nothing here but
    # would make sizeof() disagree with the compiled struct.
    assert ctypes.sizeof(types["reserved"]) == 3 * ctypes.sizeof(ctypes.c_int)


def test_reported_version_is_zludas_baked_in_constant():
    # cuda_types/src/cudnn9.rs:18. Not a probe of anything -- see version().
    assert z.ZLUDA_REPORTED_CUDNN_VERSION == 91301


# ---------------------------------------------------------------------------
# The opt-in gate and the debug-build refusal
# ---------------------------------------------------------------------------

class TestAvailability:
    def test_unavailable_without_opt_in_and_never_dlopens(self, monkeypatch):
        monkeypatch.delenv(z._ENABLE_ENV, raising=False)
        cdll = MagicMock(side_effect=AssertionError("must not dlopen when gated off"))
        monkeypatch.setattr(ctypes, "CDLL", cdll)

        assert z.available() is False
        err = z.load_error()
        assert err is not None and z._ENABLE_ENV in err
        cdll.assert_not_called()

    def test_unavailable_when_opted_in_but_nothing_built(self, monkeypatch):
        monkeypatch.setenv(z._ENABLE_ENV, "1")
        monkeypatch.setenv(z._LIB_ENV, "/nonexistent/libcudnn.so.9")

        assert z.available() is False
        err = z.load_error()
        assert "/nonexistent/libcudnn.so.9" in err
        # The error has to tell you how to fix it, not just that it failed.
        assert "build_zluda_cudnn.sh" in err

    def test_debug_build_is_refused(self, monkeypatch, tmp_path):
        debug_dir = tmp_path / "target" / "debug"
        debug_dir.mkdir(parents=True)
        fake_so = debug_dir / "libcudnn.so.9"
        fake_so.write_bytes(b"")

        monkeypatch.setenv(z._ENABLE_ENV, "1")
        monkeypatch.setenv(z._ROOT_ENV, str(tmp_path))
        monkeypatch.delenv(z._LIB_ENV, raising=False)
        monkeypatch.delenv(z._ALLOW_DEBUG_ENV, raising=False)
        cdll = MagicMock(side_effect=AssertionError("must not dlopen a debug build"))
        monkeypatch.setattr(ctypes, "CDLL", cdll)

        assert z.available() is False
        err = z.load_error()
        assert "REFUSED" in err and "debug" in err
        # Why it is refused, not just that it is: a debug ZLUDA aborts the
        # process instead of returning CUDNN_STATUS_NOT_SUPPORTED.
        assert "ABORTS" in err
        cdll.assert_not_called()

    def test_debug_build_allowed_under_escape_hatch(self, monkeypatch, tmp_path):
        debug_dir = tmp_path / "target" / "debug"
        debug_dir.mkdir(parents=True)
        (debug_dir / "libcudnn.so.9").write_bytes(b"")

        monkeypatch.setenv(z._ENABLE_ENV, "1")
        monkeypatch.setenv(z._ROOT_ENV, str(tmp_path))
        monkeypatch.setenv(z._ALLOW_DEBUG_ENV, "1")
        monkeypatch.delenv(z._LIB_ENV, raising=False)
        monkeypatch.setattr(ctypes, "CDLL", MagicMock(return_value=MagicMock()))

        assert z.available() is True
        assert z.library_path().endswith(os.path.join("debug", "libcudnn.so.9"))

    def test_release_build_preferred_over_debug(self, monkeypatch, tmp_path):
        for profile in ("release", "debug"):
            d = tmp_path / "target" / profile
            d.mkdir(parents=True)
            (d / "libcudnn.so.9").write_bytes(b"")

        monkeypatch.setenv(z._ENABLE_ENV, "1")
        monkeypatch.setenv(z._ROOT_ENV, str(tmp_path))
        monkeypatch.delenv(z._LIB_ENV, raising=False)
        monkeypatch.setattr(ctypes, "CDLL", MagicMock(return_value=MagicMock()))

        assert z.available() is True
        assert os.path.join("target", "release") in z.library_path()

    def test_missing_symbol_is_reported_not_raised(self, monkeypatch, tmp_path):
        rel = tmp_path / "target" / "release"
        rel.mkdir(parents=True)
        (rel / "libcudnn.so.9").write_bytes(b"")

        lib = MagicMock()
        # A .so that is missing an entry point this module needs -- e.g. a
        # stale build from before zluda_dnn9 gained it.
        del lib.cudnnConvolutionForward
        type(lib).cudnnConvolutionForward = property(
            lambda self: (_ for _ in ()).throw(AttributeError("cudnnConvolutionForward")))

        monkeypatch.setenv(z._ENABLE_ENV, "1")
        monkeypatch.setenv(z._ROOT_ENV, str(tmp_path))
        monkeypatch.delenv(z._LIB_ENV, raising=False)
        monkeypatch.setattr(ctypes, "CDLL", MagicMock(return_value=lib))

        assert z.available() is False
        assert "missing symbol" in z.load_error()


# ---------------------------------------------------------------------------
# Signature binding
# ---------------------------------------------------------------------------

def test_configure_signatures_does_not_raise():
    fake = MagicMock()
    z._configure_signatures(fake)  # AttributeError here = a typo'd name
    assert fake.cudnnConvolutionForward.argtypes is not None
    assert fake.cudnnGetVersion.restype is ctypes.c_size_t
    # cudnnGetLastErrorString returns void, not a status.
    assert fake.cudnnGetLastErrorString.restype is None


class _RecordingLib:
    """Stand-in for a CDLL that records which symbols were looked up.

    A plain MagicMock is no good here: _configure_signatures reaches
    symbols with getattr, which MagicMock does not record in
    method_calls, so an assertion built on that would pass no matter
    what got bound."""

    def __init__(self):
        self.looked_up = []

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        self.looked_up.append(name)
        fn = MagicMock(name=name)
        # Cache it so a second lookup returns the same object, the way a
        # real CDLL does -- otherwise argtypes set here would be lost.
        object.__setattr__(self, name, fn)
        return fn


def test_aborting_entry_points_are_never_bound():
    """cudnnGetMaxDeviceVersion and cudnnGetCudartVersion are in ZLUDA's
    `implemented_no_conversion` list but have todo!() bodies, which panic
    across the C ABI and kill the process in EVERY build profile. Binding
    them would invite a caller to try."""
    lib = _RecordingLib()
    z._configure_signatures(lib)
    assert lib.looked_up, "the stand-in recorded nothing -- test is vacuous"
    for forbidden in ("cudnnGetMaxDeviceVersion", "cudnnGetCudartVersion"):
        assert forbidden not in lib.looked_up
        assert forbidden not in z.implemented_functions()


def test_implemented_functions_matches_what_is_actually_bound():
    """implemented_functions() is public API that callers and describe()
    report from -- it must not drift from the real binding table."""
    lib = _RecordingLib()
    z._configure_signatures(lib)
    assert sorted(lib.looked_up) == sorted(z.implemented_functions())
    # No symbol bound twice, which would mean a duplicated dict key.
    assert len(lib.looked_up) == len(set(lib.looked_up))


def test_every_bound_symbol_gets_argtypes_and_restype():
    lib = _RecordingLib()
    z._configure_signatures(lib)
    for name in lib.looked_up:
        fn = getattr(lib, name)
        assert fn.argtypes is not None, name
        # cudnnGetLastErrorString returns void, so restype None is correct
        # there and only there.
        if name != "cudnnGetLastErrorString":
            assert fn.restype is not None, name


# ---------------------------------------------------------------------------
# Arithmetic and policy this module adds on top of the C API
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "x_shape,w_shape,stride,padding,dilation,expected",
    [
        ((1, 3, 32, 32), (16, 3, 3, 3), (1, 1), (1, 1), (1, 1), (1, 16, 32, 32)),
        ((8, 64, 56, 56), (128, 64, 1, 1), (1, 1), (0, 0), (1, 1), (8, 128, 56, 56)),
        ((2, 16, 28, 28), (32, 16, 3, 3), (2, 2), (1, 1), (1, 1), (2, 32, 14, 14)),
        ((1, 8, 17, 19), (4, 8, 5, 3), (1, 1), (0, 0), (2, 2), (1, 4, 9, 15)),
        ((4, 3, 224, 224), (64, 3, 7, 7), (2, 2), (3, 3), (1, 1), (4, 64, 112, 112)),
    ],
)
def test_conv2d_output_shape_matches_torch(x_shape, w_shape, stride, padding,
                                           dilation, expected):
    got = z.conv2d_output_shape(x_shape, w_shape, stride, padding, dilation)
    assert got == expected
    # And cross-check against torch's own conv2d on CPU, so this isn't just
    # asserting the formula against itself.
    ref = torch.nn.functional.conv2d(
        torch.zeros(x_shape), torch.zeros(w_shape),
        stride=stride, padding=padding, dilation=dilation)
    assert tuple(ref.shape) == expected


def test_pair_accepts_int_and_pair():
    assert z._pair(3, "stride") == (3, 3)
    assert z._pair((2, 5), "stride") == (2, 5)
    assert z._pair([2, 5], "stride") == (2, 5)
    with pytest.raises(z.ZludaCudnnError):
        z._pair((1, 2, 3), "stride")


def test_compute_dtype_promotes_low_precision_to_fp32():
    # cuDNN accumulates half/bf16 convolutions in fp32 -- the descriptor's
    # computeType, distinct from the tensors' data type.
    assert z._compute_dtype(torch.float16) is torch.float32
    assert z._compute_dtype(torch.bfloat16) is torch.float32
    assert z._compute_dtype(torch.float32) is torch.float32
    assert z._compute_dtype(torch.float64) is torch.float64


def test_scalar_ptr_width_follows_compute_type():
    # cuDNN reads alpha/beta as double ONLY for double compute; float for
    # everything else including half. The wrong width reads stack garbage.
    assert z._scalar_ptr(1.0, torch.float64)._obj.__class__ is ctypes.c_double
    assert z._scalar_ptr(1.0, torch.float32)._obj.__class__ is ctypes.c_float
    assert z._scalar_ptr(0.0, torch.float16)._obj.__class__ is ctypes.c_float


def test_cudnn_dtype_mapping_and_rejection():
    assert z._cudnn_dtype(torch.float16) == z.CUDNN_DATA_HALF
    assert z._cudnn_dtype(torch.bfloat16) == z.CUDNN_DATA_BFLOAT16
    assert z._cudnn_dtype(torch.float32) == z.CUDNN_DATA_FLOAT
    assert z._cudnn_dtype(torch.float64) == z.CUDNN_DATA_DOUBLE
    with pytest.raises(z.ZludaCudnnError):
        z._cudnn_dtype(torch.int8)


def _perf(algo, status, memory, time=1.0):
    p = z._AlgoPerf()
    p.algo = algo
    p.status = status
    p.memory = memory
    p.time = time
    return p


class TestPickAlgo:
    def test_picks_first_successful_entry_that_fits(self):
        perfs = [
            _perf(0, z.CUDNN_STATUS_NOT_SUPPORTED, 0),
            _perf(1, z.CUDNN_STATUS_SUCCESS, 1 << 40),   # too much workspace
            _perf(2, z.CUDNN_STATUS_SUCCESS, 4096),      # this one
            _perf(3, z.CUDNN_STATUS_SUCCESS, 0),
        ]
        algo, ws = z._pick_algo(perfs, 4, 1 << 20, "fwd")
        assert (algo, ws) == (2, 4096)

    def test_respects_returned_count_not_array_length(self):
        # Entries past returnedAlgoCount are uninitialised garbage in the
        # real API; reading them would be a correctness bug.
        perfs = [
            _perf(0, z.CUDNN_STATUS_NOT_SUPPORTED, 0),
            _perf(1, z.CUDNN_STATUS_SUCCESS, 128),
        ]
        with pytest.raises(z.ZludaCudnnError):
            z._pick_algo(perfs, 1, 1 << 20, "fwd")

    def test_raises_when_nothing_fits(self):
        perfs = [_perf(0, z.CUDNN_STATUS_SUCCESS, 1 << 40)]
        with pytest.raises(z.ZludaCudnnError, match="workspace"):
            z._pick_algo(perfs, 1, 1024, "fwd")

    def test_raises_when_all_unsupported(self):
        perfs = [_perf(i, z.CUDNN_STATUS_NOT_SUPPORTED, 0) for i in range(3)]
        with pytest.raises(z.ZludaCudnnError):
            z._pick_algo(perfs, 3, 1 << 30, "fwd")


def test_workspace_of_zero_bytes_allocates_nothing():
    buf, ptr = z._workspace(0, torch.device("cpu"))
    assert buf is None
    assert ptr.value in (None, 0)


def test_cpu_tensors_are_rejected():
    with pytest.raises(z.ZludaCudnnError, match="GPU"):
        z._require_contiguous(torch.zeros(1, 1, 4, 4), "x")


def test_describe_is_safe_when_unavailable(monkeypatch):
    monkeypatch.delenv(z._ENABLE_ENV, raising=False)
    text = z.describe()
    assert text.startswith("zluda_cudnn: unavailable")


def test_reset_is_safe_when_nothing_loaded():
    z.reset()  # must not raise with _LIB None
    assert z._HANDLES == {}


def test_status_name_falls_back_for_unknown_codes():
    assert z._status_name(z.CUDNN_STATUS_NOT_SUPPORTED) == "CUDNN_STATUS_NOT_SUPPORTED"
    assert "12345" in z._status_name(12345)


# ---------------------------------------------------------------------------
# The binding list, checked against ZLUDA's own source rather than a
# hand-maintained number. If someone updates third_party/ZLUDA and
# zluda_dnn implements more of cuDNN (or stops implementing something),
# this is what notices.
# ---------------------------------------------------------------------------

def _zluda_implemented_from_source():
    """Parse the `dual_function_declarations!` block in
    zluda_dnn/src/lib.rs -- the single place ZLUDA states which cuDNN
    entry points are real and which fall through to unimplemented()."""
    import re

    root = os.environ.get(
        z._ROOT_ENV,
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(z.__file__))),
                     "third_party", "ZLUDA"),
    )
    lib_rs = os.path.join(root, "zluda_dnn", "src", "lib.rs")
    if not os.path.exists(lib_rs):
        pytest.skip(f"no ZLUDA checkout at {root}")
    src = open(lib_rs).read()
    block = src[src.index("dual_function_declarations! {"):]
    block = block[: block.index("\n}\n")]
    names = set()
    pattern = r"(\(implemented_dnn8\|implemented_dnn9\)|implemented_no_conversion|implemented)\s*<=\s*\[(.*?)\]"
    for m in re.finditer(pattern, block, re.S):
        names |= {n.strip() for n in m.group(2).split(",") if n.strip()}
    return names


# The two whose bodies are `todo!()` in zluda_dnn/src/impl.rs. They are
# listed as implemented but panic across the C ABI, which aborts the
# process in every build profile -- so they are deliberately not bound.
_ABORTING = {"cudnnGetMaxDeviceVersion", "cudnnGetCudartVersion"}


def test_binding_list_matches_zluda_source_exactly():
    implemented = _zluda_implemented_from_source()
    bound = set(z.implemented_functions())

    # Nothing bound that ZLUDA does not implement -- that would be a call
    # into unimplemented(), i.e. CUDNN_STATUS_NOT_SUPPORTED at best and a
    # process abort from a debug build at worst.
    assert bound - implemented == set()

    # Nothing implemented left unbound except the two that abort.
    assert implemented - bound == _ABORTING


def test_implemented_count_is_as_documented():
    implemented = _zluda_implemented_from_source()
    # The numbers the module and package docstrings quote. If ZLUDA is
    # updated and these move, update the prose too -- that is the point.
    assert len(implemented) == 34
    assert len(implemented - _ABORTING) == 32
    assert len(z.implemented_functions()) == 32
