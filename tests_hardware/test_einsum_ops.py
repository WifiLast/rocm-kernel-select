"""torch.einsum through amd_tuned_torch.enable()'s _patched_einsum on a real
RX 7900 XTX: results vs stock einsum, the contest's cached decision, and
(with --bench via `python tests_hardware/test_einsum_ops.py --bench`) the
rewrite-vs-stock timings quoted in _dispatch._patched_einsum's docstring.

Run with:

    pytest tests_hardware/test_einsum_ops.py
"""
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA/ROCm device available"
)
amd_tuned_torch = pytest.importorskip("amd_tuned_torch")
from amd_tuned_torch import einsum_ops, kernel_select  # noqa: E402

STOCK = einsum_ops._STOCK_EINSUM

CASES = [
    ("bhqd,bhkd->bhqk", [(2, 10, 1024, 64), (2, 10, 1024, 64)], torch.float16),
    ("bhqk,bhkd->bhqd", [(2, 10, 1024, 1024), (2, 10, 1024, 64)], torch.float16),
    ("bij,bjk->bik", [(64, 256, 256), (64, 256, 256)], torch.float32),
    ("bnc,cd->bnd", [(4, 4096, 320), (320, 320)], torch.float16),
    ("bij,bjk->bik", [(20000, 24, 24), (20000, 24, 3)], torch.float32),
    ("ij,jk->ik", [(2048, 2048), (2048, 2048)], torch.bfloat16),
    ("b i d, b j d -> b i j", [(8, 77, 64), (8, 77, 64)], torch.bfloat16),
]
_TOL = {torch.float16: 2e-3, torch.bfloat16: 1.6e-2, torch.float32: 1e-4}


@pytest.fixture(scope="module")
def enabled():
    was = amd_tuned_torch.is_enabled()
    amd_tuned_torch.enable()
    yield
    if not was:
        amd_tuned_torch.disable()


def _ops(shapes, dtype):
    torch.manual_seed(0)
    return [torch.randn(s, device="cuda", dtype=dtype) for s in shapes]


@pytest.mark.parametrize("eq, shapes, dtype", CASES)
def test_patched_einsum_matches_stock(enabled, eq, shapes, dtype):
    assert torch.einsum is amd_tuned_torch._dispatch._patched_einsum
    ops = _ops(shapes, dtype)
    ref = STOCK(eq, *ops).float()
    for _ in range(2):                   # contest call, then cached-winner call
        got = torch.einsum(eq, *ops).float()
        err = ((got - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()
        assert got.shape == ref.shape and err < _TOL[dtype], err
    if kernel_select.enabled():
        key = (dtype, eq.replace(" ", ""), tuple(tuple(s) for s in shapes))
        assert kernel_select.cached_key("einsum", key) in ("bmm_plan", "stock")


def test_unsupported_equation_still_works(enabled):
    x = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(torch.einsum("ii->i", x), torch.diagonal(x))
    a = torch.randn(2, 3, 8, 16, device="cuda", dtype=torch.float16)
    b = torch.randn(2, 3, 16, 4, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(torch.einsum("...ij,...jk->...ik", a, b), STOCK("...ij,...jk->...ik", a, b))


def test_autograd_goes_to_stock(enabled):
    a = torch.randn(4, 32, 16, device="cuda", requires_grad=True)
    b = torch.randn(4, 16, 8, device="cuda")
    torch.einsum("bij,bjk->bik", a, b).sum().backward()
    torch.testing.assert_close(a.grad, b.sum(-1).unsqueeze(1).expand(4, 32, 16))


def _bench():
    amd_tuned_torch.enable()

    def t(fn, it=20):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(it):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / it

    for eq, shapes, dtype in CASES:
        ops = _ops(shapes, dtype)
        ts = t(lambda: STOCK(eq, *ops))
        tp = t(lambda: torch.einsum(eq, *ops))
        key = (dtype, eq.replace(" ", ""), tuple(tuple(s) for s in shapes))
        print(f"{eq:22s} {str(dtype)[6:]:8s} {str(shapes):42s} stock {ts:7.3f} ms  patched {tp:7.3f} ms "
              f"x{ts / tp:5.2f}  winner {kernel_select.cached_key('einsum', key)}", flush=True)


if __name__ == "__main__" and "--bench" in sys.argv:
    _bench()
