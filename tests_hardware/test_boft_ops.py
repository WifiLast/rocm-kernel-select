"""Numerical correctness for amd_tuned_torch.boft_ops on real gfx1100
hardware -- the native fast_block_diag kernels (src/cuda/fast_block_diag.cu)
need a real device to launch, so tests/test_boft_ops.py can only check the
autograd.Function/monkeypatch wiring against a MagicMock'd extension. This
file closes exactly the gap boft_ops.py's own "VALIDATION STATUS" docstring
section asks for: build the extension and cross-check fast_block_diag(x)
against a real block_diag construction for real 4D input.

WHAT THE BAR IS HERE. Unlike test_conv_kernels.py or test_splitk_gemm_ops.py,
"matches stock within fp16 tolerance" is the wrong bar: fast_block_diag does
no arithmetic at all, only data movement (a plain element copy), so every
dtype -- fp16 and bf16 included -- must match the reference EXACTLY,
bit-for-bit. Any tolerance at all would hide a real indexing bug. The
reference is torch.block_diag() over the unbound blocks, i.e. precisely
PEFT's own portable fallback path (fbd_cuda_available=False).

PEFT'S OWN EXTENSION DOES NOT BUILD HERE (observed 2026-09-12, ROCm 7.2 /
torch 2.15.0.dev+rocm7.2). Constructing a BOFTLayer WITHOUT patch_peft_boft()
runs upstream's own torch.utils.cpp_extension.load(), whose hipified
fbd_hip_kernel.hip fails to compile against this torch's headeronly
Dispatch.h ("no matching function ... ::detail::scalar_type"), so PEFT warns
and silently sets fbd_cuda_available=False. That is not a hypothetical
point of failure -- on this machine the patch is currently the ONLY way a
BOFT adapter gets a compiled fast_block_diag at all. It also means the
fallback path these tests compare against is a real, reachable path here,
not a contrived one.

bf16 IS THE POINT OF THE PORT. Upstream PEFT's kernel dispatches via
AT_DISPATCH_FLOATING_TYPES_AND_HALF, which never covered bf16 -- see
boft_ops.py's docstring. The bf16 parametrizations below are the only thing
proving this port's added launcher actually works rather than merely
compiling.

NOTE: no conftest.py in this directory -- see test_conv_kernels.py's
docstring for why (tests/conftest.py's MagicMock stub of
amd_tuned_torch._native would otherwise leak in if both directories are
collected together). Same guard applied here.

Run with:

    pytest tests_hardware/test_boft_ops.py -v
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

from amd_tuned_torch import boft_ops  # noqa: E402

if isinstance(boft_ops._C, MagicMock):
    pytest.skip(
        "amd_tuned_torch._native is tests/conftest.py's MagicMock stub, not the "
        "real compiled extension -- run this file on its own: "
        "pytest tests_hardware/test_boft_ops.py",
        allow_module_level=True,
    )

if not boft_ops.available():
    pytest.skip(
        "this build of amd_tuned_torch._native predates the fast_block_diag "
        "kernel (src/cuda/fast_block_diag.cu) -- rebuild the extension",
        allow_module_level=True,
    )

# fp16/bf16 included deliberately: pure data movement must be exact for them
# too (see this module's docstring).
DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]

# (z, N, b). Covers the degenerate 1x1 block, b/N not a multiple of the
# kThreads=512 block size, an N*b*b that lands exactly on a thread-block
# boundary (2*8*8*4 = 512 threads/batch), and a big-enough case to need many
# thread blocks.
SHAPES = [(1, 1, 1), (1, 2, 4), (3, 5, 2), (2, 8, 8), (4, 3, 7), (2, 128, 4)]


def _reference(input_: torch.Tensor) -> torch.Tensor:
    """PEFT's own portable fallback, block_diag over the unbound blocks --
    peft.tuners.boft.layer's fbd_cuda_available=False path."""
    return torch.stack([torch.block_diag(*torch.unbind(x, dim=0)) for x in input_])


def _randn(z, N, b, dtype, **kw):
    # randn then cast: keeps every element exactly representable in the
    # target dtype, so an exact comparison tests the kernel's indexing
    # rather than a cast that happened on one side only.
    return torch.randn(z, N, b, b, device="cuda", dtype=torch.float32).to(dtype, **kw)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("z, N, b", SHAPES)
def test_forward_matches_block_diag_exactly(z, N, b, dtype):
    torch.manual_seed(0)
    x = _randn(z, N, b, dtype)
    got = boft_ops.fast_block_diag(x)
    assert torch.equal(got, _reference(x))


@pytest.mark.parametrize("dtype", DTYPES)
def test_forward_metadata(dtype):
    x = _randn(3, 5, 2, dtype)
    got = boft_ops.fast_block_diag(x)
    assert got.shape == (3, 5 * 2, 5 * 2)
    assert got.dtype == dtype
    assert got.device == x.device
    assert got.is_contiguous()


def test_forward_zeroes_everything_off_the_block_diagonal():
    # The kernel only ever writes the N*b*b diagonal elements; the caller's
    # torch::zeros() is what makes the rest zero. An empty() there would
    # pass every equality check above only by luck of a fresh allocation.
    x = torch.ones(2, 4, 3, 3, device="cuda")
    got = boft_ops.fast_block_diag(x)
    mask = _reference(torch.ones_like(x)).bool()
    assert torch.equal(got[~mask], torch.zeros_like(got[~mask]))
    assert got.sum().item() == pytest.approx(x.numel())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("z, N, b", SHAPES)
def test_backward_gathers_the_blocks_back_out(z, N, b, dtype):
    torch.manual_seed(1)
    x = _randn(z, N, b, dtype, copy=True).requires_grad_(True)
    y = boft_ops.fast_block_diag(x)
    grad_out = torch.randn_like(y, dtype=torch.float32).to(dtype)
    y.backward(grad_out)
    # Backward is the exact inverse gather, so the expected grad is just the
    # diagonal blocks of grad_output read back out.
    expected = torch.stack([
        torch.stack([g[i * b:(i + 1) * b, i * b:(i + 1) * b] for i in range(N)])
        for g in grad_out
    ])
    assert torch.equal(x.grad, expected)


@pytest.mark.parametrize("dtype", DTYPES)
def test_backward_matches_autograd_through_the_reference_path(dtype):
    torch.manual_seed(2)
    x = _randn(2, 6, 3, dtype, copy=True).requires_grad_(True)
    ref_x = x.detach().clone().requires_grad_(True)
    grad_out = torch.randn(2, 18, 18, device="cuda", dtype=torch.float32).to(dtype)

    boft_ops.fast_block_diag(x).backward(grad_out)
    _reference(ref_x).backward(grad_out)

    assert torch.equal(x.grad, ref_x.grad)


def test_forward_backward_roundtrip_is_the_identity():
    torch.manual_seed(3)
    x = torch.randn(3, 7, 4, 4, device="cuda")
    y = boft_ops.fast_block_diag(x)
    assert torch.equal(boft_ops._C.fast_block_diag_backward(y, x), x)


def test_noncontiguous_input_is_made_contiguous_not_misread():
    # main_rocm.cpp calls .contiguous() before handing data_ptr() to the
    # kernel; without that, a transposed view's strides would be read as if
    # they were the contiguous [z, N, b, b] layout.
    torch.manual_seed(4)
    base = torch.randn(3, 5, 4, 4, device="cuda")
    x = base.transpose(2, 3)
    assert not x.is_contiguous()
    assert torch.equal(boft_ops.fast_block_diag(x), _reference(x.contiguous()))


def test_noncontiguous_grad_output_is_made_contiguous():
    torch.manual_seed(5)
    x = torch.randn(2, 4, 3, 3, device="cuda", requires_grad=True)
    y = boft_ops.fast_block_diag(x)
    grad_out = torch.randn(2, 12, 12, device="cuda").transpose(1, 2)
    assert not grad_out.is_contiguous()
    y.backward(grad_out)
    expected = torch.stack([
        torch.stack([g[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] for i in range(4)])
        for g in grad_out.contiguous()
    ])
    assert torch.equal(x.grad, expected)


class TestRejectsBadInput:
    def test_non_4d(self):
        with pytest.raises(RuntimeError, match="4D"):
            boft_ops.fast_block_diag(torch.randn(5, 4, 4, device="cuda"))

    def test_non_square_blocks(self):
        with pytest.raises(RuntimeError, match="last two dims"):
            boft_ops.fast_block_diag(torch.randn(2, 3, 4, 5, device="cuda"))

    def test_unsupported_dtype(self):
        x = torch.randint(0, 8, (2, 3, 4, 4), device="cuda", dtype=torch.int32)
        with pytest.raises(RuntimeError, match="Unsupported dtype"):
            boft_ops.fast_block_diag(x)

    def test_backward_dtype_mismatch(self):
        x = torch.randn(2, 3, 4, 4, device="cuda", dtype=torch.float16)
        grad_out = torch.randn(2, 12, 12, device="cuda", dtype=torch.float32)
        with pytest.raises(RuntimeError, match="dtype mismatch"):
            boft_ops._C.fast_block_diag_backward(grad_out, x)


# ----------------------------------------------------------------------
# patch_peft_boft() against the REAL installed peft, not tests/'s fake
# module -- the wiring test there proves the shim is installed where PEFT
# looks, this proves PEFT's own BOFT layer actually computes the right
# thing through it.
# ----------------------------------------------------------------------

peft_boft_layer = pytest.importorskip(
    "peft.tuners.boft.layer", reason="peft not installed"
)


@pytest.fixture
def restore_peft_boft():
    """patch_peft_boft() mutates peft's module globals process-wide."""
    saved = (peft_boft_layer._FBD_CUDA, peft_boft_layer.get_fbd_cuda)
    yield
    peft_boft_layer._FBD_CUDA, peft_boft_layer.get_fbd_cuda = saved


def test_patch_peft_boft_installs_the_shim_on_real_peft(restore_peft_boft):
    assert boft_ops.patch_peft_boft() is True
    shim = peft_boft_layer.get_fbd_cuda()
    assert shim is peft_boft_layer._FBD_CUDA

    x = torch.randn(2, 4, 3, 3, device="cuda")
    # Exactly how BOFTLayer's own FastBlockDiag calls it: .forward(input)[0].
    assert torch.equal(shim.forward(x)[0], _reference(x))
    grad_out = torch.randn(2, 12, 12, device="cuda")
    assert torch.equal(
        shim.backward(grad_out, x)[0], boft_ops._C.fast_block_diag_backward(grad_out, x)
    )


def _boft_model(seed=0):
    from peft import BOFTConfig, get_peft_model

    torch.manual_seed(seed)
    base = torch.nn.Sequential(torch.nn.Linear(8, 8, bias=False)).cuda()
    model = get_peft_model(
        base, BOFTConfig(target_modules=["0"], boft_block_size=2, boft_n_butterfly_factor=1)
    )
    # BOFT initializes its rotation to the identity, so an untouched adapter
    # would make every comparison below trivially true whatever the kernel
    # returns. Randomize the butterfly factors first.
    randomized = []
    for name, p in model.named_parameters():
        if "boft_R" in name or "boft_s" in name:
            torch.nn.init.normal_(p, std=0.1)
            randomized.append(name)
    # Guards the comparisons below against a future peft renaming these:
    # an all-identity adapter would weaken (not break) them, silently.
    assert randomized, "no boft_R/boft_s parameter found to randomize"
    return model.cuda()


def test_real_boft_layer_output_matches_pefts_own_fallback_path(restore_peft_boft):
    """End-to-end: a real BOFT adapter running through this package's kernel
    must produce the same output as the same adapter with
    fbd_cuda_available=False (torch.block_diag), which is what PEFT falls
    back to when no extension is available."""
    assert boft_ops.patch_peft_boft() is True
    x = torch.randn(4, 8, device="cuda")

    patched = _boft_model()
    boft_layers = [m for m in patched.modules() if hasattr(m, "fbd_cuda_available")]
    assert boft_layers, "no BOFTLayer found -- peft's target_modules didn't match"
    assert all(m.fbd_cuda_available for m in boft_layers), (
        "BOFTLayer didn't pick up the shim -- patch_peft_boft() must run BEFORE "
        "layer construction (see its docstring's TIMING MATTERS)"
    )
    with torch.no_grad():
        got = patched(x)

    for m in boft_layers:
        m.fbd_cuda_available = False
    with torch.no_grad():
        expected = patched(x)

    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_real_boft_layer_backward_matches_pefts_own_fallback_path(restore_peft_boft):
    assert boft_ops.patch_peft_boft() is True
    x = torch.randn(4, 8, device="cuda")

    model = _boft_model()
    boft_layers = [m for m in model.modules() if hasattr(m, "fbd_cuda_available")]

    model(x).sum().backward()
    got = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    assert got, "no BOFT parameter received a gradient"

    model.zero_grad(set_to_none=True)
    for m in boft_layers:
        m.fbd_cuda_available = False
    model(x).sum().backward()
    expected = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}

    assert got.keys() == expected.keys()
    for name in got:
        torch.testing.assert_close(got[name], expected[name], rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_real_boft_layer_in_half_precision_matches_the_fallback(dtype, restore_peft_boft):
    """Half precision gets the same bitwise-parity bar as fp32, not a
    smoke test: fp16 is the dtype a BOFT adapter is most likely to actually
    train in, and bf16 is the case upstream's own extension crashes on (its
    AT_DISPATCH never covered it) -- see boft_ops.py's WHAT THIS FIXES.
    Both must equal what PEFT's torch.block_diag fallback produces, exactly,
    forward and backward."""
    assert boft_ops.patch_peft_boft() is True
    model = _boft_model().to(dtype)
    boft_layers = [m for m in model.modules() if hasattr(m, "fbd_cuda_available")]
    assert all(m.fbd_cuda_available for m in boft_layers)
    x = torch.randn(4, 8, device="cuda", dtype=dtype)

    out = model(x)
    assert out.dtype == dtype
    assert torch.isfinite(out).all()
    got_fwd = out.detach().clone()
    out.sum().backward()
    got_grad = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    model.zero_grad(set_to_none=True)
    for m in boft_layers:
        m.fbd_cuda_available = False
    expected = model(x)
    expected.sum().backward()

    torch.testing.assert_close(got_fwd, expected.detach(), rtol=0, atol=0)
    assert got_grad, "no BOFT parameter received a gradient"
    for name, grad in got_grad.items():
        torch.testing.assert_close(
            grad, dict(model.named_parameters())[name].grad, rtol=0, atol=0, msg=name
        )


def test_patch_unlocks_multi_factor_boft(restore_peft_boft):
    """What the compiled kernel actually BUYS beyond speed. PEFT's
    pure-PyTorch fallback squeezes dim 0 (layer.py's
    `torch.block_diag(*torch.unbind(orth_rotate_butterfly.squeeze(0)))`), so
    it can only ever build ONE block-diagonal -- which is why upstream
    forces boft_n_butterfly_factor back to 1 whenever get_fbd_cuda() is
    falsy (layer.py:286). boft_R is [n_butterfly_factor + 1, ...], so with
    no compiled kernel the z dimension is pinned at 1 and multi-factor BOFT
    -- the butterfly factorization the method is named for -- is simply
    unavailable. With the shim installed it isn't.
    """
    from peft import BOFTConfig, get_peft_model

    def build():
        torch.manual_seed(0)
        base = torch.nn.Sequential(torch.nn.Linear(64, 64, bias=False)).cuda()
        cfg = BOFTConfig(target_modules=["0"], boft_block_size=8, boft_n_butterfly_factor=2)
        model = get_peft_model(base, cfg).cuda()
        boft_R = next(p for n, p in model.named_parameters() if "boft_R" in n)
        return model, boft_R

    # Without a compiled kernel: upstream clamps to a single factor. Stubbed
    # rather than left to the real get_fbd_cuda(), which would attempt a
    # multi-second JIT build (that fails on this ROCm/torch pair anyway).
    peft_boft_layer._FBD_CUDA = None
    peft_boft_layer.get_fbd_cuda = lambda: None
    _, boft_R_fallback = build()
    assert boft_R_fallback.shape[0] == 1, "expected upstream's clamp to one factor"

    assert boft_ops.patch_peft_boft() is True
    model, boft_R_patched = build()
    # boft_R is [internal_factor + 1, ...] and layer.py:291 stores
    # config.boft_n_butterfly_factor - 1, so z ends up equal to the config
    # value: 2 here, versus the 1 the fallback is clamped to above.
    assert boft_R_patched.shape[0] == 2

    # And the z > 1 tensor actually flows through the kernel.
    out = model(torch.randn(4, 64, device="cuda"))
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()
