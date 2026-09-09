"""Numerical correctness for amd_tuned_torch.iu4_gemm_ops on real gfx1100
hardware -- the native WMMA kernels (src/cuda/iu4_gemm_fwd.cu) need a real
device to launch. See that module's docstring for why this tier is
EXPERIMENTAL and opt-in only, never a kernel_select candidate: unlike
tests_hardware/test_splitk_gemm_ops.py, "matches stock F.linear" is not the
right bar here -- INT8/INT4 quantization is deliberately lossy, so this
checks the packed-integer GEMM against an int32 CPU reference computed from
the SAME quantized values (i.e. proves the kernel is exact given its
inputs), and separately checks that dequantized end-to-end error against
an unquantized reference stays within a generous, quantization-aware
tolerance -- not that it matches bit-for-bit or even to fp16 tolerance.

NOTE: no conftest.py in this directory -- see test_conv_kernels.py's
docstring for why (tests/conftest.py's MagicMock stub of amd_tuned_torch._native
would otherwise leak in if both directories are collected together). Same
guard applied here.

Run with:

    pytest tests_hardware/test_iu4_gemm_ops.py -v
"""
from unittest.mock import MagicMock

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA/ROCm device available"
)
amd_tuned_torch = pytest.importorskip(
    "amd_tuned_torch",
    reason="amd_tuned_torch native extension not built -- run "
           "`pip install -e . --no-build-isolation` first",
)

if isinstance(amd_tuned_torch.ops.iu4_gemm_supported, MagicMock):
    pytest.skip(
        "amd_tuned_torch.ops is tests/conftest.py's MagicMock stub, not the real "
        "compiled extension -- run this file on its own: "
        "pytest tests_hardware/test_iu4_gemm_ops.py",
        allow_module_level=True,
    )

from amd_tuned_torch import iu4_gemm_ops  # noqa: E402

if not iu4_gemm_ops.available():
    pytest.skip("iu4/iu8 GEMM requires a gfx1100 device", allow_module_level=True)


def _int_reference(a_int8: torch.Tensor, b_int8: torch.Tensor) -> torch.Tensor:
    return (a_int8.float() @ b_int8.float().t()).to(torch.int32)


@pytest.mark.parametrize("m, k, n", [(1, 32, 16), (17, 64, 33), (128, 256, 128)])
def test_iu8_gemm_matches_int_reference(m, k, n):
    torch.manual_seed(0)
    a = torch.randint(-127, 128, (m, k), dtype=torch.int8, device="cuda")
    b = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
    got = amd_tuned_torch.ops.iu8_gemm(a, b)
    expected = _int_reference(a, b)
    assert torch.equal(got, expected)


@pytest.mark.parametrize("m, k, n", [(1, 32, 16), (17, 48, 33), (128, 256, 128)])
def test_iu4_gemm_matches_int_reference(m, k, n):
    torch.manual_seed(1)
    a = torch.randint(-7, 8, (m, k), dtype=torch.int8, device="cuda")
    b = torch.randint(-7, 8, (n, k), dtype=torch.int8, device="cuda")
    a_packed = iu4_gemm_ops.pack_int4_rows(a)
    b_packed = iu4_gemm_ops.pack_int4_rows(b)
    got = amd_tuned_torch.ops.iu4_gemm(a_packed, b_packed, k)
    expected = _int_reference(a, b)
    assert torch.equal(got, expected)


def test_dot4_i8_gemm_matches_int_reference():
    torch.manual_seed(2)
    m, k, n = 33, 128, 65
    a = torch.randint(-127, 128, (m, k), dtype=torch.int8, device="cuda")
    b = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
    got = amd_tuned_torch.ops.dot4_i8_gemm(a, b)
    expected = _int_reference(a, b)
    assert torch.equal(got, expected)


@pytest.mark.parametrize("m, k, n, has_bias", [
    (4, 256, 128, False), (4, 256, 128, True), (1, 4096, 4096, False),
])
def test_iu8_linear_reasonable_error_vs_unquantized_reference(m, k, n, has_bias):
    torch.manual_seed(3)
    x = torch.randn(m, k, dtype=torch.float16, device="cuda") * 0.1
    w = torch.randn(n, k, dtype=torch.float16, device="cuda") * 0.1
    bias = torch.randn(n, dtype=torch.float16, device="cuda") * 0.1 if has_bias else None

    expected = torch.nn.functional.linear(x, w, bias)
    actual = iu4_gemm_ops.iu8_linear(x, w, bias)

    assert actual is not None
    assert actual.shape == expected.shape
    # int8 per-row quantization on both operands: generous, quantization-
    # aware tolerance, not fp16 rounding tolerance.
    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)


def test_iu4_linear_reasonable_error_vs_unquantized_reference():
    torch.manual_seed(4)
    m, k, n = 4, 256, 128
    x = torch.randn(m, k, dtype=torch.float16, device="cuda") * 0.1
    w = torch.randn(n, k, dtype=torch.float16, device="cuda") * 0.1

    expected = torch.nn.functional.linear(x, w)
    actual = iu4_gemm_ops.iu4_linear(x, w)

    assert actual is not None
    assert actual.shape == expected.shape
    # int4 (16-level) quantization on both operands is materially lossier
    # than iu8_linear's -- this tolerance is wide on purpose.
    torch.testing.assert_close(actual, expected, atol=2e-1, rtol=2e-1)
