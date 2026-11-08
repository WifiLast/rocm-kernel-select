"""Numerical correctness of amd_tuned_torch.te_extra_ops against unfused
PyTorch references, on real hardware.

The CPU suite (tests/test_te_extra_ops.py) covers the Python layer -- which
tex binding each entry point reaches for, in what argument order, and the
shape/dtype eligibility rules. None of that touches a kernel. This file is
the other half: the kernels actually run and their results are compared
against references written the slow, obvious way.

First run on the RX 7900 XTX 2026-11-06 against the vendored TE build
(gfx1100, torch 2.15 rocm7.14): all green after two wrapper fixes this file
now guards -- softmax backward aliasing grad_output, and swap_first_dims'
keyword-only `out`. Timings: tools/bench_te_extra_ops.py.

Run with:

    pytest tests_hardware/test_te_extra_ops.py

Run it on its own, not combined with tests/ -- tests/conftest.py installs a
MagicMock amd_tuned_torch._native that leaks in otherwise (same note as
test_conv_kernels.py).
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA/ROCm device available"
)
pytest.importorskip(
    "amd_tuned_torch",
    reason="amd_tuned_torch native extension not built -- run "
           "`pip install -e . --no-build-isolation` first",
)
from amd_tuned_torch import te_extra_ops  # noqa: E402

if not te_extra_ops.available():
    pytest.skip(
        "TransformerEngine is not available -- set AMD_TUNED_TORCH_ENABLE_TE=1 and "
        "verify `python -c 'import transformer_engine.pytorch'` works",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Multi-tensor optimizers
# ---------------------------------------------------------------------------


def _param_set(shapes, dtype=torch.float32, seed=0):
    """Matched (params, grads, exp_avg, exp_avg_sq) lists plus a reference
    copy of the params, so the fused step and torch's can start identical."""
    torch.manual_seed(seed)
    params = [torch.randn(s, device="cuda", dtype=dtype) for s in shapes]
    grads = [torch.randn(s, device="cuda", dtype=dtype) for s in shapes]
    reference = [p.detach().clone() for p in params]
    exp_avg = [torch.zeros_like(p) for p in params]
    exp_avg_sq = [torch.zeros_like(p) for p in params]
    return params, grads, exp_avg, exp_avg_sq, reference


SHAPES = [(64,), (128, 32), (7, 13), (256,)]


@pytest.mark.parametrize("adam_w_mode", [True, False])
def test_multi_tensor_adam_matches_torch(adam_w_mode):
    """Three steps, because bias correction only diverges after the first --
    a kernel that ignored `step` would still pass a single-step check."""
    params, grads, exp_avg, exp_avg_sq, reference = _param_set(SHAPES)
    lr, beta1, beta2, eps, wd = 1e-2, 0.9, 0.999, 1e-8, 0.01

    ref_params = [p.detach().clone().requires_grad_(True) for p in reference]
    opt_cls = torch.optim.AdamW if adam_w_mode else torch.optim.Adam
    opt = opt_cls(ref_params, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=wd)

    for step in range(1, 4):
        for rp, g in zip(ref_params, grads):
            rp.grad = g.detach().clone()
        opt.step()

        te_extra_ops.multi_tensor_adam(
            grads, params, exp_avg, exp_avg_sq,
            lr=lr, beta1=beta1, beta2=beta2, eps=eps, step=step,
            adam_w_mode=adam_w_mode, bias_correction=True, weight_decay=wd,
        )

    for fused, ref in zip(params, ref_params):
        torch.testing.assert_close(fused, ref.detach(), rtol=1e-4, atol=1e-5)


def test_multi_tensor_sgd_matches_torch():
    params, grads, _, _, reference = _param_set(SHAPES, seed=1)
    momentums = [torch.zeros_like(p) for p in params]
    lr, momentum, wd = 0.1, 0.9, 1e-4

    ref_params = [p.detach().clone().requires_grad_(True) for p in reference]
    opt = torch.optim.SGD(ref_params, lr=lr, momentum=momentum, weight_decay=wd)

    for step in range(3):
        for rp, g in zip(ref_params, grads):
            rp.grad = g.detach().clone()
        opt.step()
        te_extra_ops.multi_tensor_sgd(
            grads, params, momentums,
            lr=lr, momentum=momentum, weight_decay=wd,
            # torch seeds the momentum buffer from the first gradient rather
            # than blending into zeros; first_run is how TE says the same.
            first_run=(step == 0),
        )

    for fused, ref in zip(params, ref_params):
        torch.testing.assert_close(fused, ref.detach(), rtol=1e-4, atol=1e-5)


def test_multi_tensor_l2norm_matches_torch():
    tensors = [torch.randn(s, device="cuda") for s in SHAPES]
    total, per_tensor = te_extra_ops.multi_tensor_l2norm(tensors, per_tensor=True)

    expected_total = torch.linalg.vector_norm(torch.cat([t.flatten() for t in tensors]))
    torch.testing.assert_close(total.float().squeeze(), expected_total, rtol=1e-5, atol=1e-5)

    expected_each = torch.stack([torch.linalg.vector_norm(t) for t in tensors])
    torch.testing.assert_close(
        per_tensor[: len(tensors)].float(), expected_each, rtol=1e-5, atol=1e-5
    )


def test_multi_tensor_l2norm_unscales_without_writing_back():
    """The AMP contract: the reported norm is of the un-scaled values, but
    the stored tensors keep their scaled values."""
    tensors = [torch.randn(s, device="cuda") * 8.0 for s in SHAPES]
    before = [t.clone() for t in tensors]
    inv_scale = torch.tensor([0.125], device="cuda")

    total, _ = te_extra_ops.multi_tensor_l2norm(tensors, inv_scale=inv_scale)

    expected = torch.linalg.vector_norm(torch.cat([t.flatten() for t in before]) * 0.125)
    torch.testing.assert_close(total.float().squeeze(), expected, rtol=1e-4, atol=1e-5)
    for t, b in zip(tensors, before):
        # Un-scaling happens inside the norm reduction only; the stored
        # tensors keep the scaled values the caller handed in.
        torch.testing.assert_close(t, b)


def test_multi_tensor_scale_matches_and_flags_overflow():
    src = [torch.randn(s, device="cuda") for s in SHAPES]
    dst = [torch.empty_like(t) for t in src]
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")

    te_extra_ops.multi_tensor_scale(src, dst, 0.25, noop_flag=flag)
    for out, inp in zip(dst, src):
        torch.testing.assert_close(out, inp * 0.25)
    assert int(flag.item()) == 0, "no overflow for a finite input"

    src[0][0] = float("inf")
    te_extra_ops.multi_tensor_scale(src, dst, 2.0, noop_flag=flag)
    assert int(flag.item()) != 0, "an inf must set the overflow flag"


# ---------------------------------------------------------------------------
# Fused softmax
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 4, 128, 128), (1, 8, 64, 256)])
def test_scaled_softmax_matches_torch(dtype, shape):
    scale = 0.125
    scores = torch.randn(shape, device="cuda", dtype=dtype)
    assert te_extra_ops.kernel_available(scores), f"{shape}/{dtype} should be eligible"

    out = te_extra_ops.scaled_softmax(scores, None, scale, "no_mask")
    expected = torch.softmax(scores.float() * scale, dim=-1).to(dtype)
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-3)


def test_scaled_masked_softmax_matches_torch():
    scale = 0.125
    b, np_, sq, sk = 2, 4, 128, 128
    scores = torch.randn(b, np_, sq, sk, device="cuda", dtype=torch.float16)
    # TE's mask convention is "True means masked out", the same as
    # F.scaled_dot_product_attention's boolean mask inverted.
    mask = torch.zeros(b, 1, sq, sk, device="cuda", dtype=torch.bool)
    mask[:, :, :, sk // 2:] = True
    assert te_extra_ops.kernel_available(scores, mask, "padding")

    out = te_extra_ops.scaled_softmax(scores, mask, scale, "padding")
    expected = torch.softmax(
        (scores.float() * scale).masked_fill(mask, float("-inf")), dim=-1
    ).to(torch.float16)
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-3)


def test_scaled_softmax_backward_matches_torch():
    scale = 0.5
    scores = torch.randn(2, 4, 128, 128, device="cuda", dtype=torch.float16,
                         requires_grad=True)
    ref = scores.detach().clone().requires_grad_(True)

    te_extra_ops.scaled_softmax(scores, None, scale, "no_mask").sum().backward()
    torch.softmax(ref.float() * scale, dim=-1).to(torch.float16).sum().backward()

    torch.testing.assert_close(scores.grad, ref.grad, rtol=2e-2, atol=2e-3)


@pytest.mark.parametrize("variant", ["no_mask", "padding", "upper_triang"])
def test_softmax_backward_with_real_grad_and_no_aliasing(variant):
    """Regression: TE's softmax backward kernels write dL/dx into grad_output
    in place. With a contiguous, caller-owned grad (y.backward(g)) the wrapper
    used to hand that buffer straight to TE, so g was overwritten and the
    returned gradient was wrong by 1-5000x. A random upstream grad also makes
    this non-trivial: with .sum().backward() the softmax grad is ~0 anyway."""
    scale = 0.125
    torch.manual_seed(0)
    if variant == "upper_triang":
        shape = (8, 128, 128)
        mask = torch.ones(128, 128, device="cuda", dtype=torch.bool).triu(1)
        fn = lambda t: te_extra_ops.upper_triang_causal_softmax(t, scale)
    else:
        shape = (2, 4, 128, 128)
        mask = None
        if variant == "padding":
            mask = torch.zeros(2, 1, 128, 128, device="cuda", dtype=torch.bool)
            mask[..., 100:] = True
        fn = lambda t: te_extra_ops.scaled_softmax(t, mask, scale, variant)
    x = (torch.randn(shape, device="cuda", dtype=torch.float16) * 3).requires_grad_()
    g = torch.randn(shape, device="cuda", dtype=torch.float16)
    g_before = g.clone()

    fn(x).backward(g)

    assert torch.equal(g, g_before), "the caller's grad tensor was modified"
    x32 = x.detach().float().requires_grad_()
    s = x32 * scale
    if mask is not None:
        s = s.masked_fill(mask, float("-inf"))
    torch.softmax(s, dim=-1).backward(g.float())
    ref = x32.grad
    rel = ((x.grad.float() - ref).abs().max() / ref.abs().max()).item()
    assert rel < 5e-3, f"grad rel err {rel:.2e} vs fp32 reference"


def test_scaled_softmax_or_torch_falls_back_without_crashing():
    """sq == 1 is the decode shape the kernel asserts against -- the wrapper
    has to route it to torch rather than let AT_ASSERTM fire."""
    scores = torch.randn(2, 4, 1, 64, device="cuda", dtype=torch.float16)
    assert not te_extra_ops.kernel_available(scores)
    out = te_extra_ops.scaled_softmax_or_torch(scores, scale=1.0)
    expected = torch.softmax(scores.float(), dim=-1).to(torch.float16)
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-3)


def test_upper_triang_causal_softmax_matches_torch():
    scale = 0.125
    attn_batches, s = 8, 128
    scores = torch.randn(attn_batches, s, s, device="cuda", dtype=torch.float16)

    out = te_extra_ops.upper_triang_causal_softmax(scores, scale)

    causal = torch.triu(torch.ones(s, s, device="cuda", dtype=torch.bool), diagonal=1)
    expected = torch.softmax(
        (scores.float() * scale).masked_fill(causal, float("-inf")), dim=-1
    ).to(torch.float16)
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-3)


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------


def test_dropout_scales_survivors_and_zeroes_the_rest():
    """Statistical, not exact: the mask is TE's own RNG, so the check is that
    the survivors carry the 1/(1-p) scaling and the drop rate is close to p."""
    p = 0.25
    x = torch.ones(1024, 1024, device="cuda", dtype=torch.float16)
    out = te_extra_ops.dropout(x, p, training=True)

    survivors = out != 0
    dropped_fraction = 1.0 - survivors.float().mean().item()
    assert abs(dropped_fraction - p) < 0.02, f"drop rate {dropped_fraction} vs p={p}"
    torch.testing.assert_close(
        out[survivors],
        torch.full_like(out[survivors], 1.0 / (1.0 - p)),
        rtol=2e-3, atol=2e-3,
    )


def test_dropout_backward_follows_the_same_mask():
    """The gradient must be zero exactly where the forward zeroed, and carry
    the same scaling everywhere else -- that is what the saved bit-mask is
    for."""
    p = 0.25
    x = torch.randn(512, 512, device="cuda", dtype=torch.float16, requires_grad=True)
    out = te_extra_ops.dropout(x, p, training=True)
    out.sum().backward()

    dropped = out == 0
    assert torch.all(x.grad[dropped] == 0)
    kept = ~dropped
    torch.testing.assert_close(
        x.grad[kept],
        torch.full_like(x.grad[kept], 1.0 / (1.0 - p)),
        rtol=2e-3, atol=2e-3,
    )


# ---------------------------------------------------------------------------
# Sequence-layout utilities
# ---------------------------------------------------------------------------


def _ragged(seqlens, h=4, d=8, dtype=torch.float16):
    offsets = [0]
    for length in seqlens:
        offsets.append(offsets[-1] + length)
    cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    total = int(cu[-1])
    return torch.randn(total, h, d, device="cuda", dtype=dtype), cu


def test_thd_to_bshd_places_each_sequence_at_its_own_row():
    seqlens = [3, 7, 2]
    thd, cu = _ragged(seqlens)
    max_seqlen = max(seqlens)

    bshd = te_extra_ops.thd_to_bshd(thd, cu, max_seqlen)

    assert bshd.shape == (len(seqlens), max_seqlen, thd.shape[1], thd.shape[2])
    for i, length in enumerate(seqlens):
        start = int(cu[i])
        torch.testing.assert_close(bshd[i, :length], thd[start:start + length])


def test_bshd_to_thd_is_the_inverse():
    seqlens = [3, 7, 2]
    thd, cu = _ragged(seqlens)
    round_trip = te_extra_ops.bshd_to_thd(
        te_extra_ops.thd_to_bshd(thd, cu, max(seqlens)), cu
    )
    torch.testing.assert_close(round_trip, thd)


def test_thd_to_bshd_backward_drops_padding_gradient():
    """Padding rows correspond to no input token, so their gradient has
    nowhere to go -- the round trip of a gradient must land back on exactly
    the real tokens."""
    seqlens = [3, 7, 2]
    thd, cu = _ragged(seqlens, dtype=torch.float32)
    thd.requires_grad_(True)

    te_extra_ops.thd_to_bshd(thd, cu, max(seqlens)).sum().backward()

    torch.testing.assert_close(thd.grad, torch.ones_like(thd))


def test_pad_and_unpad_rows_round_trip():
    cols = 16
    input_rows = [2, 5]
    padded_rows = [4, 8]
    src = torch.randn(sum(input_rows), cols, device="cuda", dtype=torch.float16)
    padded = torch.zeros(sum(padded_rows), cols, device="cuda", dtype=torch.float16)

    te_extra_ops.pad_rows(src, padded, input_rows, padded_rows)

    torch.testing.assert_close(padded[: input_rows[0]], src[: input_rows[0]])
    torch.testing.assert_close(
        padded[padded_rows[0]: padded_rows[0] + input_rows[1]], src[input_rows[0]:]
    )

    back = torch.zeros_like(src)
    te_extra_ops.unpad_rows(padded, back, padded_rows, input_rows)
    torch.testing.assert_close(back, src)


def test_swap_first_dims_as_first_te_call_does_not_segfault():
    """Regression: swap_first_dims' binding releases the GIL, and as the first
    TE call in a process TE's lazy init_extension() then imports Python
    without it -> SIGSEGV. Needs a fresh interpreter: in this pytest process
    some other TE call has always run first and hides it."""
    import os, subprocess, sys, pathlib
    code = ("import torch, amd_tuned_torch.te_extra_ops as tx\n"
            "x = torch.randn(4, 6, 8, device='cuda', dtype=torch.float16)\n"
            "assert torch.equal(tx.swap_first_dims(x), x.transpose(0, 1).contiguous())\n")
    env = dict(os.environ, AMD_TUNED_TORCH_ENABLE_TE="1", AMD_TUNED_TORCH_AUTOPATCH="0")
    root = str(pathlib.Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-c", code], env=env, cwd=root,
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, f"rc={r.returncode}\n{r.stderr[-2000:]}"


def test_swap_first_dims_matches_transpose():
    x = torch.randn(4, 6, 8, device="cuda", dtype=torch.float16)
    out = te_extra_ops.swap_first_dims(x)
    torch.testing.assert_close(out, x.transpose(0, 1).contiguous())
    assert out.is_contiguous(), "the whole point is a contiguous result"
