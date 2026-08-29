"""Correctness of the Composable Kernel conv tier (amd_tuned_torch.ck_ops)
against stock ROCm, on real hardware.

Accuracy is asserted RELATIVE TO STOCK rather than against a fixed
tolerance. A fixed tolerance would be the wrong instrument here: CK
accumulates in fp32 and so does MIOpen, so the meaningful question is
whether CK is as accurate as the kernel it replaces, not whether it clears
some absolute bar that bf16's ~3 decimal digits would make either
meaningless or unachievable depending on which number was picked. Each
case computes an fp32 reference and requires CK's mean relative error to
be within a small factor of stock's own error on the identical inputs.

Every case also runs in both memory formats. That is not redundant: CK's
WMMA conv instances are channels-last only (see src/cuda/ck_conv_fwd.hpp),
so the NCHW cases exercise the wrapper's conversion-in/convert-back path
while the channels-last cases exercise the zero-copy path, and those are
genuinely different code.

Run with:

    pytest tests_hardware/test_ck_conv.py
"""
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA/ROCm device available"
)
amd_tuned_torch = pytest.importorskip(
    "amd_tuned_torch",
    reason="amd_tuned_torch native extension not built -- run "
           "`pip install -e . --no-build-isolation` first",
)

# Same guard as test_conv_kernels.py: refuse to run against tests/
# conftest.py's MagicMock stub of the native extension.
if isinstance(amd_tuned_torch.ops.conv2d, MagicMock):
    pytest.skip(
        "amd_tuned_torch.ops is tests/conftest.py's MagicMock stub, not the real "
        "compiled extension -- run this file on its own: "
        "pytest tests_hardware/test_ck_conv.py",
        allow_module_level=True,
    )

from amd_tuned_torch import ck_ops  # noqa: E402  (after the importorskip)

if not ck_ops.available():
    pytest.skip(
        "extension built without Composable Kernel -- set "
        "AMD_TUNED_TORCH_CK_ROOT and rebuild to exercise this tier",
        allow_module_level=True,
    )

device = torch.device("cuda")

# CK must be no worse than this multiple of stock's own relative error, with
# a small floor so shapes where stock is near-exact don't make the bound
# unreasonably tight.
ERROR_FACTOR = 3.0
ERROR_FLOOR = 5e-3

# (N, C_in, spatial, C_out, K, stride, padding, dilation)
CONV2D_SHAPES = [
    pytest.param(2, 64, (32, 32), 64, 3, 1, 1, 1, id="plain_3x3"),
    pytest.param(1, 64, (17, 19), 32, 3, 1, 1, 1, id="odd_hw"),
    pytest.param(2, 32, (28, 28), 64, 3, 2, 1, 1, id="stride2"),
    pytest.param(1, 64, (32, 32), 64, 3, 1, 2, 2, id="dilation2"),
    pytest.param(1, 64, (32, 32), 64, 1, 1, 0, 1, id="pointwise_1x1"),
]

CONV3D_SHAPES = [
    pytest.param(1, 64, (4, 16, 16), 64, 3, 1, 1, 1, id="plain_3x3x3"),
    pytest.param(1, 32, (4, 8, 8), 64, 3, 2, 1, 1, id="stride2"),
]


def _check(ndim, x, w, b, stride, padding, dilation, channels_last):
    conv = F.conv2d if ndim == 2 else F.conv3d
    ck_conv = ck_ops.conv2d if ndim == 2 else ck_ops.conv3d
    fmt = torch.channels_last if ndim == 2 else torch.channels_last_3d

    if channels_last:
        x = x.contiguous(memory_format=fmt)
        w = w.contiguous(memory_format=fmt)

    got = ck_conv(x, w, b, stride, padding, dilation)
    if got is None:
        # A routine outcome, not a failure: CK rejects problems no compiled
        # instance supports (vector-load alignment, tile divisibility) and
        # the dispatch falls back. Nothing to verify.
        pytest.skip("no compiled CK instance supports this problem")

    ref = conv(x.float(), w.float(), None if b is None else b.float(),
               stride=stride, padding=padding, dilation=dilation)
    stock = conv(x, w, b, stride=stride, padding=padding, dilation=dilation)

    assert got.shape == ref.shape
    assert got.dtype == x.dtype
    # The wrapper must hand back the format it was given, in both
    # directions -- a channels-last caller must not be silently handed NCHW
    # (it would force a copy on every downstream op) and an NCHW caller must
    # not be handed channels-last (it would break `.view()`-style callers).
    if channels_last:
        assert got.is_contiguous(memory_format=fmt)
    else:
        assert got.is_contiguous()

    scale = ref.abs().mean().item()
    ck_err = (got.float() - ref).abs().mean().item() / scale
    stock_err = (stock.float() - ref).abs().mean().item() / scale
    assert ck_err <= max(ERROR_FACTOR * stock_err, ERROR_FLOOR), (
        f"CK mean relative error {ck_err:.6f} exceeds bound "
        f"(stock {stock_err:.6f}, factor {ERROR_FACTOR}, floor {ERROR_FLOOR})"
    )


@pytest.mark.parametrize("channels_last", [False, True], ids=["nchw", "channels_last"])
@pytest.mark.parametrize("with_bias", [True, False], ids=["bias", "no_bias"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("N,C_in,spatial,C_out,K,stride,padding,dilation", CONV2D_SHAPES)
def test_ck_conv2d_matches_stock_accuracy(
    N, C_in, spatial, C_out, K, stride, padding, dilation, dtype, with_bias, channels_last
):
    torch.manual_seed(0)
    x = torch.randn(N, C_in, *spatial, device=device, dtype=dtype)
    w = torch.randn(C_out, C_in, K, K, device=device, dtype=dtype)
    b = torch.randn(C_out, device=device, dtype=dtype) if with_bias else None
    _check(2, x, w, b, stride, padding, dilation, channels_last)


@pytest.mark.parametrize("channels_last", [False, True], ids=["nchw", "channels_last"])
@pytest.mark.parametrize("with_bias", [True, False], ids=["bias", "no_bias"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("N,C_in,spatial,C_out,K,stride,padding,dilation", CONV3D_SHAPES)
def test_ck_conv3d_matches_stock_accuracy(
    N, C_in, spatial, C_out, K, stride, padding, dilation, dtype, with_bias, channels_last
):
    torch.manual_seed(0)
    x = torch.randn(N, C_in, *spatial, device=device, dtype=dtype)
    w = torch.randn(C_out, C_in, K, K, K, device=device, dtype=dtype)
    b = torch.randn(C_out, device=device, dtype=dtype) if with_bias else None
    _check(3, x, w, b, stride, padding, dilation, channels_last)


def test_ck_declines_unsupported_dtype_rather_than_raising():
    """fp32 has no CK path at all (gfx1100's WMMA units have no fp32 mode),
    and the tier must decline by returning None so the caller's fallback
    runs -- not raise, which would surface as a hard failure through
    _patched_conv2d."""
    x = torch.randn(1, 32, 16, 16, device=device, dtype=torch.float32)
    w = torch.randn(32, 32, 3, 3, device=device, dtype=torch.float32)
    assert ck_ops.conv2d(x, w, None, 1, 1, 1) is None


def test_ck_declines_grouped_conv():
    """groups != 1 is not routed here; the shape check (weight C != input C)
    must decline rather than silently compute something wrong."""
    x = torch.randn(1, 64, 16, 16, device=device, dtype=torch.float16)
    w = torch.randn(64, 32, 3, 3, device=device, dtype=torch.float16)  # groups=2 weight
    assert ck_ops.conv2d(x, w, None, 1, 1, 1) is None
