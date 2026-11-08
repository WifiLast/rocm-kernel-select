"""The te_extra_ops kernels enable() installs over stock by default:
torch.optim Adam/AdamW (foreach default path), clip_grad_norm_/get_total_norm,
F.dropout (exact p).

Each test runs the same thing with the patches off (stock) and on, and checks
both that TE was actually reached (a call counter on te_extra_ops) and that the
result matches stock. Declined cases check the opposite: stock ran.

Needs AMD_TUNED_TORCH_ENABLE_TE=1 (and AMD_TUNED_TORCH_AUTOPATCH=0 so the
fixture, not import, decides when patches are live):

    AMD_TUNED_TORCH_ENABLE_TE=1 AMD_TUNED_TORCH_AUTOPATCH=0 \\
        pytest tests_hardware/test_te_default_patches.py
"""
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")

import amd_tuned_torch  # noqa: E402
from amd_tuned_torch import te_extra_ops, _dispatch  # noqa: E402

if not te_extra_ops.available():
    pytest.skip("TE not enabled (AMD_TUNED_TORCH_ENABLE_TE=1)", allow_module_level=True)


@pytest.fixture
def patches():
    was = amd_tuned_torch.is_enabled()
    amd_tuned_torch.disable()
    yield
    amd_tuned_torch.disable()
    if was:
        amd_tuned_torch.enable()


@pytest.fixture
def counter(monkeypatch):
    calls = {}

    def wrap(name):
        real = getattr(te_extra_ops, name)

        def counted(*a, **k):
            calls[name] = calls.get(name, 0) + 1
            return real(*a, **k)
        monkeypatch.setattr(te_extra_ops, name, counted)
    for n in ("multi_tensor_adam", "multi_tensor_l2norm", "dropout"):
        wrap(n)
    return calls


def _model_params(dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    shapes = [(64, 32), (64,), (128, 64, 3, 3), (7,), (300, 5)]
    return [torch.randn(s, generator=g).to("cuda", dtype) for s in shapes]


def _run_opt(opt_cls, base, dtype, steps=5, grad_round=None, **kw):
    ps = [torch.nn.Parameter(b.clone()) for b in base]
    opt = opt_cls(ps, lr=1e-2, weight_decay=0.05, **kw)
    g = torch.Generator().manual_seed(1)
    for _ in range(steps):
        for p in ps:
            gr = torch.randn(p.shape, generator=g).to("cuda", grad_round or dtype)
            p.grad = gr.to(dtype)
        opt.step()
    return ps, opt


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("opt_cls", [torch.optim.AdamW, torch.optim.Adam])
@pytest.mark.parametrize("mode", [{}, {"fused": True}])
def test_adam_matches_stock_and_uses_te(patches, counter, dtype, opt_cls, mode):
    base = _model_params(dtype)
    ref, ref_opt = _run_opt(opt_cls, base, dtype, **mode)
    amd_tuned_torch.enable()
    got, got_opt = _run_opt(opt_cls, base, dtype, **mode)
    if mode.get("fused"):
        # fused=True deliberately stays on torch's C++ kernel (no win once the
        # wrapper's step bookkeeping is counted, and its steps live on GPU).
        assert counter.get("multi_tensor_adam", 0) == 0
        for a, b in zip(got, ref):
            torch.testing.assert_close(a.detach(), b.detach())
        return
    assert counter.get("multi_tensor_adam", 0) == 5
    for pa, pb in zip(got, ref):
        sa, sb = got_opt.state[pa], ref_opt.state[pb]
        assert float(sa["step"]) == float(sb["step"]) == 5.0
        assert sa["step"].device == sb["step"].device
    if dtype == torch.float32:
        for a, b in zip(got, ref):
            torch.testing.assert_close(a.detach(), b.detach(), rtol=1e-5, atol=1e-5)
        return
    # bf16: TE does the update in fp32 and rounds once, stock foreach rounds
    # every intermediate -- they legitimately differ by a few ulps. The
    # meaningful check: TE is no further from an fp32 run than stock is.
    ref32, _ = _run_opt(opt_cls, [b.float() for b in base], torch.float32,
                        grad_round=dtype, **{k: v for k, v in mode.items()})
    err_te = max((a.float() - r).abs().max().item() for a, r in zip(got, ref32))
    err_stock = max((b.float() - r).abs().max().item() for b, r in zip(ref, ref32))
    assert err_te <= err_stock * 1.01 + 1e-6, (err_te, err_stock)


@pytest.mark.parametrize("kw,dtype", [({"amsgrad": True}, torch.float32),
                                      ({"maximize": True}, torch.float32),
                                      ({}, torch.float16)])
def test_adam_declines_unsupported(patches, counter, kw, dtype):
    base = _model_params(dtype)
    ref, _ = _run_opt(torch.optim.AdamW, base, dtype, **kw)
    amd_tuned_torch.enable()
    got, _ = _run_opt(torch.optim.AdamW, base, dtype, **kw)
    assert counter.get("multi_tensor_adam", 0) == 0
    for a, b in zip(got, ref):
        torch.testing.assert_close(a.detach(), b.detach())


def test_adam_resumes_from_a_stock_state_dict(patches, counter):
    """Same state layout as stock: a checkpoint saved without the patch
    continues identically with it."""
    base = _model_params(torch.float32)
    ref, ref_opt = _run_opt(torch.optim.AdamW, base, torch.float32, steps=6)
    ps, opt = _run_opt(torch.optim.AdamW, base, torch.float32, steps=3)
    amd_tuned_torch.enable()
    g = torch.Generator().manual_seed(1)
    for _ in range(3):  # replay the generator to step 3
        for p in ps:
            torch.randn(p.shape, generator=g)
    for _ in range(3):
        for p in ps:
            p.grad = torch.randn(p.shape, generator=g).to("cuda")
        opt.step()
    assert counter.get("multi_tensor_adam", 0) == 3
    for a, b in zip(ps, ref):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_clip_grad_norm_matches_stock(patches, counter, dtype):
    base = _model_params(dtype)

    def run():
        ps = [torch.nn.Parameter(b.clone()) for b in base]
        for i, p in enumerate(ps):
            p.grad = (torch.arange(p.numel(), device="cuda").reshape(p.shape) % 7 - 3
                      ).to(dtype) * 0.1 * (i + 1)
        n = torch.nn.utils.clip_grad_norm_(ps, max_norm=1.0)
        return n, [p.grad.clone() for p in ps]

    n_ref, g_ref = run()
    amd_tuned_torch.enable()
    n_got, g_got = run()
    assert counter.get("multi_tensor_l2norm", 0) == 1
    assert n_got.shape == n_ref.shape == () and n_got.dtype == n_ref.dtype
    tol = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(n_got, n_ref, rtol=tol, atol=tol)
    for a, b in zip(g_got, g_ref):
        torch.testing.assert_close(a, b, rtol=tol, atol=tol)
    # public alias goes through TE too
    torch.nn.utils.get_total_norm([g for g in g_got])
    assert counter["multi_tensor_l2norm"] == 2


def test_clip_grad_norm_nonfinite_and_other_norms(patches, counter):
    amd_tuned_torch.enable()
    p = torch.nn.Parameter(torch.ones(10, device="cuda"))
    p.grad = torch.full((10,), float("inf"), device="cuda")
    with pytest.raises(RuntimeError, match="non-finite"):
        torch.nn.utils.clip_grad_norm_([p], 1.0, error_if_nonfinite=True)
    p.grad = torch.ones(10, device="cuda")
    n = torch.nn.utils.clip_grad_norm_([p], 100.0, norm_type=float("inf"))
    assert n.item() == 1.0  # inf-norm declined to stock
    assert counter.get("multi_tensor_l2norm", 0) == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_dropout_exact_p_uses_te(patches, counter, dtype):
    amd_tuned_torch.enable()
    x = torch.randn(1000, 333, device="cuda", dtype=dtype).requires_grad_()
    y = F.dropout(x, 0.5, training=True)
    y.backward(torch.ones_like(y))
    assert counter.get("dropout", 0) == 1
    keep = x.grad != 0
    assert abs(keep.float().mean().item() - 0.5) < 0.01
    torch.testing.assert_close(x.grad[keep].float(),
                               torch.full_like(x.grad[keep].float(), 2.0))
    torch.testing.assert_close(y[keep], (x * 2)[keep])
    assert bool((y[~keep] == 0).all())
    # nn.Dropout reaches F.dropout
    torch.nn.Dropout(0.25).train()(x.detach())
    assert counter["dropout"] == 2


@pytest.mark.parametrize("p,training,inplace", [(0.1, True, False), (0.5, False, False),
                                                 (0.5, True, True)])
def test_dropout_declines(patches, counter, p, training, inplace):
    amd_tuned_torch.enable()
    x = torch.randn(64, 64, device="cuda")
    F.dropout(x, p, training=training, inplace=inplace)
    assert counter.get("dropout", 0) == 0


# --- te_extra_ops bugs the defaults would otherwise inherit -----------------

def test_l2norm_result_is_not_overwritten_by_the_next_call():
    n1, _ = te_extra_ops.multi_tensor_l2norm([torch.ones(1000, device="cuda")])
    v1 = n1.item()
    te_extra_ops.multi_tensor_l2norm([torch.full((1000,), 2.0, device="cuda")])
    assert n1.item() == v1


def test_nonfinite_l2norm_does_not_stall_later_sgd():
    te_extra_ops.multi_tensor_l2norm([torch.tensor([float("inf")], device="cuda")])
    p = [torch.zeros(4, device="cuda")]
    te_extra_ops.multi_tensor_sgd([torch.ones(4, device="cuda")], p,
                                  [torch.zeros(4, device="cuda")], lr=1.0, first_run=True)
    assert p[0].tolist() == [-1.0] * 4


# --- RMSNorm on TE's ROCm Triton kernel -------------------------------------

@pytest.fixture
def rms_backend(monkeypatch):
    from amd_tuned_torch import te_ops
    seen = []
    real = te_ops.rms_norm

    def spy(*a, **k):
        seen.append(k.get("backend", "hip"))
        return real(*a, **k)
    monkeypatch.setattr(te_ops, "rms_norm", spy)
    return seen


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cols", [4096, 8192])
def test_rms_norm_training_uses_triton_and_matches_stock(patches, rms_backend, dtype, cols):
    from amd_tuned_torch import te_ops
    if not te_ops.triton_rms_norm_available():
        pytest.skip("TE Triton norms not importable")
    torch.manual_seed(0)
    x0 = torch.randn(16384, cols, device="cuda", dtype=dtype)
    w0 = torch.rand(cols, device="cuda", dtype=dtype) + 0.5
    gy = torch.randn_like(x0)

    def run():
        x = x0.clone().requires_grad_()
        w = w0.clone().requires_grad_()
        y = F.rms_norm(x, (cols,), w, 1e-5)
        y.backward(gy)
        return y.detach(), x.grad, w.grad

    ref = run()
    amd_tuned_torch.enable()
    got = run()
    assert rms_backend == ["triton"]
    # dw sums 16384 rows: compare relative to its magnitude
    for g, r in zip(got, ref):
        rel = ((g.float() - r.float()).abs().max() / r.float().abs().max()).item()
        assert rel < 2e-2, rel


@pytest.mark.parametrize("shape,grad", [((2048, 2048), True),     # too few rows
                                        ((16384, 2048), True),    # hidden: hip wins
                                        ((16384, 4096), False)])  # inference
def test_rms_norm_triton_declines(patches, rms_backend, shape, grad):
    amd_tuned_torch.enable()
    x = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=grad)
    w = torch.ones(shape[1], device="cuda", dtype=torch.float16, requires_grad=grad)
    F.rms_norm(x, (shape[1],), w, 1e-5)
    assert "triton" not in rms_backend
