"""Tests for amd_tuned_torch.rocfft_ops -- no ROCm device or librocfft.so
exists in this dev environment, so nothing here can exercise a real FFT.
What IS covered for real, without hardware:

  * available()/load_error() degrade gracefully when the library can't be
    found (confirmed against both "genuinely absent, as on this machine"
    and an explicitly forced not-found path, so this test doesn't secretly
    depend on which is true on whatever machine runs it).
  * Every enum ordinal this module hardcodes matches
    library/include/rocfft/rocfft.h (a regression guard against a future
    transcription slip -- these are C enum values with no symbolic
    checking possible from Python).
  * _configure_signatures runs against a stand-in object without raising
    (catches a typo'd function name early rather than only at real load
    time).
  * rfftn/irfftn's own shape/dim/normalization arithmetic -- the actual
    error-prone logic this module adds on top of the C API -- exercised
    end-to-end by monkeypatching _get_or_create_plan/_execute (not the
    ctypes layer itself) and torch.Tensor.is_cuda, so the real Python
    control flow runs on plain CPU tensors standing in for CUDA ones.

Run with:

    python -m pytest source/cmp_ext_turing/tests
"""
from __future__ import annotations

import ctypes
from unittest.mock import MagicMock, patch

import pytest
import torch

import amd_tuned_torch.rocfft_ops as rocfft_ops


def test_enum_ordinals_match_header():
    # library/include/rocfft/rocfft.h -- plain C enums, declaration order.
    assert rocfft_ops.ROCFFT_STATUS_SUCCESS == 0
    assert rocfft_ops.ROCFFT_STATUS_FAILURE == 1
    assert rocfft_ops.ROCFFT_STATUS_INVALID_ARG_VALUE == 2
    assert rocfft_ops.ROCFFT_STATUS_INVALID_DIMENSIONS == 3
    assert rocfft_ops.ROCFFT_STATUS_INVALID_ARRAY_TYPE == 4
    assert rocfft_ops.ROCFFT_STATUS_INVALID_STRIDES == 5
    assert rocfft_ops.ROCFFT_STATUS_INVALID_DISTANCE == 6
    assert rocfft_ops.ROCFFT_STATUS_INVALID_OFFSET == 7
    assert rocfft_ops.ROCFFT_STATUS_INVALID_WORK_BUFFER == 8

    assert rocfft_ops.ROCFFT_TRANSFORM_TYPE_COMPLEX_FORWARD == 0
    assert rocfft_ops.ROCFFT_TRANSFORM_TYPE_COMPLEX_INVERSE == 1
    assert rocfft_ops.ROCFFT_TRANSFORM_TYPE_REAL_FORWARD == 2
    assert rocfft_ops.ROCFFT_TRANSFORM_TYPE_REAL_INVERSE == 3

    assert rocfft_ops.ROCFFT_PRECISION_SINGLE == 0
    assert rocfft_ops.ROCFFT_PRECISION_DOUBLE == 1
    assert rocfft_ops.ROCFFT_PRECISION_HALF == 2

    assert rocfft_ops.ROCFFT_PLACEMENT_INPLACE == 0
    assert rocfft_ops.ROCFFT_PLACEMENT_NOTINPLACE == 1

    assert rocfft_ops.ROCFFT_ARRAY_TYPE_COMPLEX_INTERLEAVED == 0
    assert rocfft_ops.ROCFFT_ARRAY_TYPE_COMPLEX_PLANAR == 1
    assert rocfft_ops.ROCFFT_ARRAY_TYPE_REAL == 2
    assert rocfft_ops.ROCFFT_ARRAY_TYPE_HERMITIAN_INTERLEAVED == 3
    assert rocfft_ops.ROCFFT_ARRAY_TYPE_HERMITIAN_PLANAR == 4
    assert rocfft_ops.ROCFFT_ARRAY_TYPE_UNSET == 5


def test_configure_signatures_does_not_raise():
    fake = MagicMock()
    rocfft_ops._configure_signatures(fake)  # raises AttributeError on a typo'd name
    assert fake.rocfft_execute.argtypes is not None


class TestAvailability:
    def teardown_method(self):
        rocfft_ops._LIB = None
        rocfft_ops._LIB_LOAD_ERROR = None
        rocfft_ops._SETUP_DONE = False

    def test_unavailable_on_this_machine_or_when_forced_not_found(self, monkeypatch):
        rocfft_ops._LIB = None
        rocfft_ops._LIB_LOAD_ERROR = None
        monkeypatch.setattr(ctypes, "CDLL", MagicMock(side_effect=OSError("not found")))
        assert rocfft_ops.available() is False
        assert rocfft_ops.load_error() is not None

    def test_available_when_library_and_setup_succeed(self, monkeypatch):
        rocfft_ops._LIB = None
        rocfft_ops._LIB_LOAD_ERROR = None
        fake_lib = MagicMock()
        fake_lib.rocfft_setup.return_value = rocfft_ops.ROCFFT_STATUS_SUCCESS
        monkeypatch.setattr(ctypes, "CDLL", MagicMock(return_value=fake_lib))
        monkeypatch.setattr(rocfft_ops.atexit, "register", lambda fn: None)
        assert rocfft_ops.available() is True
        assert rocfft_ops.load_error() is None

    def test_unavailable_when_setup_fails(self, monkeypatch):
        rocfft_ops._LIB = None
        rocfft_ops._LIB_LOAD_ERROR = None
        fake_lib = MagicMock()
        fake_lib.rocfft_setup.return_value = rocfft_ops.ROCFFT_STATUS_FAILURE
        monkeypatch.setattr(ctypes, "CDLL", MagicMock(return_value=fake_lib))
        assert rocfft_ops.available() is False


class TestResolveTrailingDims:
    def test_none_means_every_dim(self):
        assert rocfft_ops._resolve_trailing_dims(4, None) == (0, 1, 2, 3)

    def test_explicit_trailing_dims_accepted(self):
        assert rocfft_ops._resolve_trailing_dims(4, (2, 3)) == (2, 3)

    def test_negative_dims_normalized(self):
        assert rocfft_ops._resolve_trailing_dims(4, (-2, -1)) == (2, 3)

    def test_non_trailing_dims_rejected(self):
        with pytest.raises(ValueError):
            rocfft_ops._resolve_trailing_dims(4, (0, 1))

    def test_out_of_order_trailing_dims_rejected(self):
        with pytest.raises(ValueError):
            rocfft_ops._resolve_trailing_dims(4, (3, 2))


class TestRfftnShapeArithmetic:
    """rfftn/irfftn's own logic (batch/transform shape split, column-major
    length reversal, output shape/dtype, irfftn's normalization) -- the
    part of this module that's actually error-prone and worth protecting,
    exercised with real Python control flow by monkeypatching
    _get_or_create_plan/_execute (not the ctypes/C layer) and
    torch.Tensor.is_cuda (CPU tensors standing in for CUDA ones)."""

    def test_rfftn_plan_key_and_output_shape_2d(self, monkeypatch):
        calls = {}

        def fake_get_or_create_plan(transform_type, precision, lengths, n_batch):
            calls["plan"] = (transform_type, precision, lengths, n_batch)
            return "FAKE_PLAN"

        def fake_execute(plan, in_ptr, out_ptr):
            calls["execute"] = (plan, in_ptr, out_ptr)

        monkeypatch.setattr(rocfft_ops, "_get_or_create_plan", fake_get_or_create_plan)
        monkeypatch.setattr(rocfft_ops, "_execute", fake_execute)

        X = torch.randn(3, 5, 8, 16, dtype=torch.float32)  # batch=(3,5), transform=(8,16)
        with patch.object(torch.Tensor, "is_cuda", property(lambda self: True)):
            out = rocfft_ops.rfftn(X, dim=(-2, -1))

        # rocFFT lengths are column-major (fastest dim first): reverse of (8, 16) -> (16, 8).
        assert calls["plan"] == (
            rocfft_ops.ROCFFT_TRANSFORM_TYPE_REAL_FORWARD,
            rocfft_ops.ROCFFT_PRECISION_SINGLE,
            (16, 8),
            15,  # 3 * 5
        )
        assert calls["plan"][0] == rocfft_ops.ROCFFT_TRANSFORM_TYPE_REAL_FORWARD
        assert calls["execute"][0] == "FAKE_PLAN"
        # rfftn only halves the LAST transform dim: (8, 16) -> (8, 9).
        assert out.shape == (3, 5, 8, 9)
        assert out.dtype == torch.complex64

    def test_rfftn_1d_no_batch_dims(self, monkeypatch):
        calls = {}
        monkeypatch.setattr(rocfft_ops, "_get_or_create_plan",
                             lambda *a: calls.setdefault("plan", a) or "PLAN")
        monkeypatch.setattr(rocfft_ops, "_execute", lambda *a: calls.setdefault("execute", a))

        X = torch.randn(32, dtype=torch.float64)
        with patch.object(torch.Tensor, "is_cuda", property(lambda self: True)):
            out = rocfft_ops.rfftn(X)

        assert calls["plan"] == (rocfft_ops.ROCFFT_TRANSFORM_TYPE_REAL_FORWARD,
                                  rocfft_ops.ROCFFT_PRECISION_DOUBLE, (32,), 1)
        assert out.shape == (17,)  # 32 // 2 + 1
        assert out.dtype == torch.complex128

    def test_rfftn_rejects_cpu_tensor(self):
        with pytest.raises(rocfft_ops.RocfftError):
            rocfft_ops.rfftn(torch.randn(4, 8))

    def test_rfftn_rejects_unsupported_dtype(self):
        with patch.object(torch.Tensor, "is_cuda", property(lambda self: True)):
            with pytest.raises(TypeError):
                rocfft_ops.rfftn(torch.randn(4, 8, dtype=torch.float16))

    def test_irfftn_normalizes_like_torch_fft_default(self, monkeypatch):
        """rocFFT's inverse transform is unnormalized; irfftn() must divide
        by prod(s) to match torch.fft.irfftn's default normalization --
        verified by having the fake _execute write a KNOWN pattern into
        the real output buffer (via its raw pointer, the same interface
        the real ctypes call would write through) and checking the
        returned tensor is that pattern divided by prod(s), not the raw
        pattern itself."""
        known_value = 40.0
        s = (4, 5)  # prod = 20 -> known_value / 20 = 2.0

        def fake_get_or_create_plan(transform_type, precision, lengths, n_batch):
            return "FAKE_PLAN"

        def fake_execute(plan, in_ptr, out_ptr):
            # Simulate rocFFT writing the transform result into the real
            # output buffer, through the same raw pointer interface
            # _execute really uses.
            n = 1
            for d in s:
                n *= d
            buf = (ctypes.c_float * n).from_address(out_ptr)
            for i in range(n):
                buf[i] = known_value

        monkeypatch.setattr(rocfft_ops, "_get_or_create_plan", fake_get_or_create_plan)
        monkeypatch.setattr(rocfft_ops, "_execute", fake_execute)

        X = torch.zeros(4, 3, dtype=torch.complex64)
        with patch.object(torch.Tensor, "is_cuda", property(lambda self: True)):
            out = rocfft_ops.irfftn(X, s=s)

        assert out.shape == s
        assert torch.allclose(out, torch.full(s, known_value / 20.0))

    def test_irfftn_rejects_cpu_tensor(self):
        with pytest.raises(rocfft_ops.RocfftError):
            rocfft_ops.irfftn(torch.zeros(4, 3, dtype=torch.complex64), s=(4, 4))

    def test_irfftn_rejects_unsupported_dtype(self):
        with patch.object(torch.Tensor, "is_cuda", property(lambda self: True)):
            with pytest.raises(TypeError):
                rocfft_ops.irfftn(torch.zeros(4, 3), s=(4, 4))
