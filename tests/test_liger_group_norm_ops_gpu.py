"""Hardware tests for amd_tuned_torch.liger_group_norm_ops' Triton kernels.

The sibling test_liger_group_norm_ops.py covers the F.group_norm fallback
path on CPU; this file is the other half the module's VALIDATION STATUS
docstring section asks for -- "compare group_norm(x, ...) output AND
gradients (x.grad, weight.grad, bias.grad) against F.group_norm's own for
your actual (batch, channels, *spatial, num_groups, dtype) combinations"
-- and it skips itself entirely unless available() reports a real
Triton + GPU install, so it stays inert in a CPU-only checkout.

The 4D NCHW cases are the ones that matter: every GroupNorm in SDXL's UNet
and VAE is 4D, and the backward pass used to take upstream Liger's 3D
`hidden_size = dY.shape[-1]` for them, under-allocating DX by a factor of
H and writing ~5.2M elements into a 40,960-element buffer -- HIP 700,
hipErrorIllegalAddress, on the first backward of any run using this tier.
See _group_norm_backward's own SHAPE CONTRACT docstring.

Run with:

    python -m pytest source/cmp_ext_turing/tests/test_liger_group_norm_ops_gpu.py
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from amd_tuned_torch import liger_group_norm_ops as gn

pytestmark = pytest.mark.skipif(
    not gn.available(), reason="needs a real Triton + CUDA/ROCm install"
)

# The ATen op, NOT F.group_norm. amd_tuned_torch patches the F.* name, and in
# this suite the patched path is backed by conftest's fake MagicMock native
# kernel -- so F.group_norm here returns a MagicMock, not numbers, and the
# comparison fails with "No comparison pair was able to handle inputs of
# type Tensor and MagicMock". It used to return real numbers only because
# _patched_group_norm declined whenever a gradient was live; now that
# _GroupNormFn handles that case (which is the point of it), the reference
# has to be the op nothing patches.
_STOCK_GROUP_NORM = torch.group_norm


# (shape, num_groups) -- the first three are the real SDXL UNet GroupNorm
# activation shapes at 1024x1024 with batch 1; then a 2-batch case, a
# small sanity case, and the 3D shape upstream Liger documents (which
# must keep working, since the fix derives the per-channel count from
# numel rather than assuming a spatial rank).
CASES = [
    ((1, 320, 128, 128), 32),
    ((1, 640, 64, 64), 32),
    ((1, 1280, 32, 32), 32),
    ((2, 640, 32, 32), 32),
    ((2, 8, 5, 5), 4),
    ((2, 8, 16), 4),
]


def _run(shape, num_groups, dtype):
    """group_norm() and F.group_norm on identical inputs; returns
    (out, x.grad, w.grad, b.grad) for each, in that order."""
    torch.manual_seed(0)
    num_channels = shape[1]
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    w = torch.randn(num_channels, device="cuda", dtype=dtype)
    b = torch.randn(num_channels, device="cuda", dtype=dtype)
    # A non-uniform upstream gradient -- dY = ones would hide any error
    # that depends on the per-element indexing being right.
    torch.manual_seed(1)
    dY = torch.randn(*shape, device="cuda", dtype=dtype)

    got, ref = [], []
    for fn in (gn.group_norm, _STOCK_GROUP_NORM):
        xx = x.clone().requires_grad_(True)
        ww = w.clone().requires_grad_(True)
        bb = b.clone().requires_grad_(True)
        out = fn(xx, num_groups, ww, bb, 1e-5)
        out.backward(dY)
        (got if fn is gn.group_norm else ref).extend(
            [out.detach(), xx.grad, ww.grad, bb.grad]
        )
    return got, ref


@pytest.mark.parametrize("shape,num_groups", CASES)
def test_fp32_matches_f_group_norm(shape, num_groups):
    got, ref = _run(shape, num_groups, torch.float32)
    names = ["output", "x.grad", "weight.grad", "bias.grad"]
    for name, g, r in zip(names, got, ref):
        torch.testing.assert_close(g, r, rtol=2e-3, atol=2e-3, msg=lambda m: f"{name}: {m}")


@pytest.mark.parametrize("shape,num_groups", [CASES[0], CASES[4], CASES[5]])
def test_fp16_matches_f_group_norm(shape, num_groups):
    """fp16 is what SDXL mixed-precision training actually feeds this.
    Tolerances are loose because F.group_norm upcasts its accumulation to
    fp32 while this kernel keeps Mean/RSTD in the input dtype -- the point
    of the test is that the result is still the same tensor within fp16
    noise, not bit-identical."""
    got, ref = _run(shape, num_groups, torch.float16)
    names = ["output", "x.grad", "weight.grad", "bias.grad"]
    for name, g, r in zip(names, got, ref):
        assert g.dtype == r.dtype, f"{name}: dtype {g.dtype} != {r.dtype}"
        torch.testing.assert_close(g.float(), r.float(), rtol=5e-2, atol=5e-2,
                                   msg=lambda m: f"{name}: {m}")


def test_module_replacement_trains_on_a_4d_unet_like_block():
    """End-to-end through LigerGroupNorm itself (not the private helpers),
    on a conv-GroupNorm-conv block shaped like an SDXL ResNet block: a
    real optimizer step must run without a device fault and move the
    weights."""
    import torch.nn as nn

    torch.manual_seed(0)
    block = nn.Sequential(
        nn.Conv2d(320, 320, 3, padding=1),
        nn.GroupNorm(32, 320),
        nn.SiLU(),
        nn.Conv2d(320, 320, 3, padding=1),
    ).cuda()
    assert gn.replace_group_norm_modules(block) == 1
    before = block[1].weight.detach().clone()

    opt = torch.optim.SGD(block.parameters(), lr=1e-2)
    out = block(torch.randn(1, 320, 64, 64, device="cuda"))
    out.square().mean().backward()
    assert block[1].weight.grad is not None
    assert torch.isfinite(block[1].weight.grad).all()
    opt.step()
    torch.cuda.synchronize()
    assert not torch.equal(before, block[1].weight.detach())
