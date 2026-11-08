"""Numerical correctness of flash-attention's Triton ROCm backend on real
hardware (RX 7900 XTX / gfx1100).

``third_party/flash-attention`` is installed editable with **no C++ kernels
built** -- there is no ``flash_attn_2_cuda``, so ``flash_attn_interface``
falls back to aiter's Triton kernels
(``aiter.ops.triton._triton_kernels.flash_attn_triton_amd``). Every kernel
here JITs on first use and needs a GPU to launch at all, so this can't live
in the CPU-only ``tests/`` suite.

Run with:

    pytest tests_hardware/test_flash_attn.py

**The first run takes ~5 minutes**, almost all of it Triton JIT-compiling one
kernel per distinct (headdim, causal, dtype, varlen/kvcache) config. Those
land in ``~/.triton/cache`` and are reused across runs, so every run after
that is ~50s -- of which ~35s is just importing torch + flash_attn + aiter,
and ~15s is the tests themselves. If a run suddenly costs minutes again,
the cache was cleared or a kernel signature changed; that is not a hang.

Tolerance follows flash-attention's own upstream idiom rather than a fixed
atol: compute the same attention three ways -- the kernel, an fp32
reference, and the *same* reference re-run in the test dtype -- and require
the kernel's error against fp32 to be no worse than twice PyTorch's own.
A fixed atol would either be too loose to catch a real bug at short
seqlens or too tight to pass at long ones, since fp16 softmax error grows
with the number of accumulated terms.
"""
import os

# Must be set before torch (and hence triton) is imported. This box has both
# an AMD gfx1100 and an NVIDIA card visible, and triton refuses to pick a
# driver on its own ("2 active drivers ... There should only be one").
# Note: do NOT "fix" this with CUDA_VISIBLE_DEVICES="" -- that hides the AMD
# device from torch's HIP runtime too, since ROCm torch reads the same var.
os.environ.setdefault("TRITON_DEFAULT_BACKEND", "amd")
# This env's torch is locally patched to autoload amd_tuned_torch from the
# bottom of torch/__init__.py, which patches F/torch ops the moment torch is
# imported -- so without this the reference below would be measured against
# amd_tuned_torch's kernels, not stock PyTorch, and a bug shared by both
# would cancel out. Stock torch is the only honest baseline here.
os.environ.setdefault("TORCH_DISABLE_AMD_TUNED_TORCH", "1")
# Not required (flash_attn_interface warns and falls back to Triton on its
# own when flash_attn_2_cuda is absent) -- set only to silence that warning.
os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA/ROCm device available"
)
pytest.importorskip(
    "flash_attn",
    reason="flash_attn not installed -- see third_party/flash-attention "
           "(skip its setup.py aiter step, its pin needs a nonexistent flydsl)",
)
from flash_attn import (  # noqa: E402
    flash_attn_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)


# ---------------------------------------------------------------------------
# reference implementation
# ---------------------------------------------------------------------------

def _mask(seqlen_q, seqlen_k, causal, window_size, device):
    """FA2's mask for a (seqlen_q, seqlen_k) score block, as a keep-bool.

    Both ``causal`` and ``window_size`` are **bottom-right aligned** in FA2:
    query row ``i`` sits at absolute position ``i + (seqlen_k - seqlen_q)``.
    This is what makes a KV-cache decode step (seqlen_q=1, seqlen_k=n) attend
    to the whole cache instead of just the first token, and it is where
    ``F.scaled_dot_product_attention(is_causal=True)`` disagrees -- SDPA
    aligns top-left. They only coincide when seqlen_q == seqlen_k, which is
    why this builds the mask explicitly rather than deferring to SDPA.
    """
    if not causal and window_size == (-1, -1):
        return None
    row = torch.arange(seqlen_q, device=device).unsqueeze(1) + (seqlen_k - seqlen_q)
    col = torch.arange(seqlen_k, device=device).unsqueeze(0)
    left, right = window_size
    if causal:
        right = 0
    keep = torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=device)
    if right >= 0:
        keep &= (col - row) <= right
    if left >= 0:
        keep &= (row - col) <= left
    return keep


def _attn_ref(q, k, v, causal=False, window_size=(-1, -1), softmax_scale=None, upcast=True):
    """Unfused attention over FA2's (batch, seqlen, nheads, headdim) layout.

    Built from ordinary differentiable PyTorch ops, so it doubles as the
    backward reference via plain autograd -- a hand-derived reference
    backward could share a mistake with the kernel's own hand-derived one.
    """
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    _, seqlen_q, nheads, headdim = q.shape
    seqlen_k, nheads_k = k.shape[1], k.shape[2]
    if nheads != nheads_k:  # GQA/MQA: each KV head is shared by nheads/nheads_k query heads
        assert nheads % nheads_k == 0
        k = k.repeat_interleave(nheads // nheads_k, dim=2)
        v = v.repeat_interleave(nheads // nheads_k, dim=2)
    scale = softmax_scale if softmax_scale is not None else headdim ** -0.5
    scores = torch.einsum("bshd,bthd->bhst", q, k) * scale
    keep = _mask(seqlen_q, seqlen_k, causal, window_size, q.device)
    if keep is not None:
        scores = scores.masked_fill(~keep, float("-inf"))
    out = torch.einsum("bhst,bthd->bshd", scores.softmax(dim=-1), v)
    return out.to(dtype_og)


def _qkv(batch, seqlen_q, seqlen_k, nheads, nheads_k, headdim, dtype, requires_grad=False):
    g = torch.Generator(device="cuda").manual_seed(0)

    def rand(*shape):
        t = torch.randn(*shape, generator=g, device="cuda", dtype=dtype)
        return t.requires_grad_(requires_grad)

    return (rand(batch, seqlen_q, nheads, headdim),
            rand(batch, seqlen_k, nheads_k, headdim),
            rand(batch, seqlen_k, nheads_k, headdim))


def assert_no_worse_than_pytorch(out, out_ref, out_pt, slack=2.0, atol=1e-5, what="out"):
    """out vs the fp32 reference, budgeted against fp16-PyTorch's own error."""
    err = (out.float() - out_ref.float()).abs().max().item()
    err_pt = (out_pt.float() - out_ref.float()).abs().max().item()
    assert err <= slack * err_pt + atol, (
        f"{what}: flash-attn max abs err {err:.3e} exceeds {slack}x "
        f"PyTorch's own {err_pt:.3e} (+{atol:.0e})"
    )


# ---------------------------------------------------------------------------
# backend sanity
# ---------------------------------------------------------------------------

def test_triton_rocm_backend_is_the_one_in_use():
    """Guards the whole file: if a C++ ``flash_attn_2_cuda`` ever gets built
    into this env, every test below silently stops covering the Triton path
    it was written for."""
    from flash_attn import flash_attn_interface

    assert flash_attn_interface.USE_TRITON_ROCM, "expected the Triton ROCm backend"
    assert flash_attn_interface.flash_attn_gpu.__name__.startswith("aiter.ops.triton"), (
        f"expected aiter's Triton kernels, got {flash_attn_interface.flash_attn_gpu.__name__}"
    )
    assert torch.version.hip is not None, "this suite targets ROCm, not CUDA"


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "batch, seqlen_q, seqlen_k, nheads, headdim",
    [
        (2, 1024, 1024, 8, 64),    # the common square case
        (1, 239, 239, 5, 128),     # seqlen and nheads not multiples of any block size
        (3, 64, 512, 4, 32),       # seqlen_q < seqlen_k -- exercises bottom-right alignment
    ],
)
def test_forward_matches_reference(dtype, causal, batch, seqlen_q, seqlen_k, nheads, headdim):
    q, k, v = _qkv(batch, seqlen_q, seqlen_k, nheads, nheads, headdim, dtype)

    out = flash_attn_func(q, k, v, causal=causal)
    out_ref = _attn_ref(q, k, v, causal=causal)
    out_pt = _attn_ref(q, k, v, causal=causal, upcast=False)

    assert out.shape == q.shape
    assert out.dtype == dtype
    assert_no_worse_than_pytorch(out, out_ref, out_pt)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("nheads, nheads_k", [(8, 2), (8, 1)])  # GQA, MQA
def test_gqa_mqa_matches_reference(causal, nheads, nheads_k):
    q, k, v = _qkv(2, 512, 512, nheads, nheads_k, 64, torch.float16)

    out = flash_attn_func(q, k, v, causal=causal)
    out_ref = _attn_ref(q, k, v, causal=causal)
    out_pt = _attn_ref(q, k, v, causal=causal, upcast=False)

    assert_no_worse_than_pytorch(out, out_ref, out_pt)


@pytest.mark.parametrize("window_size", [(128, 0), (64, 64)])
def test_sliding_window_matches_reference(window_size):
    """Local attention. (128, 0) is the causal sliding window Mistral-style
    models use; (64, 64) is a symmetric non-causal band."""
    q, k, v = _qkv(2, 512, 512, 4, 4, 64, torch.float16)

    out = flash_attn_func(q, k, v, causal=False, window_size=window_size)
    out_ref = _attn_ref(q, k, v, window_size=window_size)
    out_pt = _attn_ref(q, k, v, window_size=window_size, upcast=False)

    assert_no_worse_than_pytorch(out, out_ref, out_pt)


def test_softmax_scale_is_honoured():
    """A non-default scale is easy to drop on the way into a kernel, and a
    dropped one still produces plausible-looking output."""
    q, k, v = _qkv(1, 256, 256, 4, 4, 64, torch.float16)
    scale = 0.05  # deliberately far from the 1/sqrt(64) = 0.125 default

    out = flash_attn_func(q, k, v, softmax_scale=scale)
    out_ref = _attn_ref(q, k, v, softmax_scale=scale)
    out_pt = _attn_ref(q, k, v, softmax_scale=scale, upcast=False)

    assert_no_worse_than_pytorch(out, out_ref, out_pt)
    default = flash_attn_func(q, k, v)
    assert not torch.allclose(out, default, atol=1e-2), "softmax_scale looks ignored"


def test_qkvpacked_matches_unpacked():
    qkv = torch.randn(2, 256, 3, 4, 64, device="cuda", dtype=torch.float16)
    q, k, v = qkv.unbind(dim=2)

    torch.testing.assert_close(
        flash_attn_qkvpacked_func(qkv, causal=True),
        flash_attn_func(q, k, v, causal=True),
        atol=0, rtol=0,  # same kernel, same inputs -- must be bit-identical
    )


# ---------------------------------------------------------------------------
# backward
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("causal", [False, True])
def test_backward_matches_reference(causal):
    q, k, v = _qkv(2, 512, 512, 4, 4, 64, torch.float16, requires_grad=True)
    g = torch.randn(2, 512, 4, 64, device="cuda", dtype=torch.float16)

    grads = torch.autograd.grad(flash_attn_func(q, k, v, causal=causal), (q, k, v), g)
    grads_ref = torch.autograd.grad(_attn_ref(q, k, v, causal=causal), (q, k, v), g)
    grads_pt = torch.autograd.grad(
        _attn_ref(q, k, v, causal=causal, upcast=False), (q, k, v), g
    )

    for name, got, ref, pt in zip("qkv", grads, grads_ref, grads_pt):
        # dk/dv accumulate across the whole query axis, so their fp16 error
        # is larger than dq's -- which is exactly what budgeting against
        # PyTorch's own error handles, per-tensor.
        assert_no_worse_than_pytorch(got, ref, pt, slack=3.0, what=f"d{name}")


# ---------------------------------------------------------------------------
# varlen
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("causal", [False, True])
def test_varlen_matches_per_sequence_reference(causal):
    """Ragged batch: sequences packed end-to-end with no padding, indexed by
    cu_seqlens. A wrong offset here shows up as one sequence attending into
    its neighbour, which a same-length batch can never catch."""
    seqlens = [128, 1, 333, 64]
    nheads, headdim, dtype = 4, 64, torch.float16
    total = sum(seqlens)
    g = torch.Generator(device="cuda").manual_seed(1)
    q, k, v = (torch.randn(total, nheads, headdim, generator=g, device="cuda", dtype=dtype)
               for _ in range(3))
    cu_seqlens = torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0)),
                              device="cuda", dtype=torch.int32)

    out = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens,
                                 max(seqlens), max(seqlens), causal=causal)

    for i, n in enumerate(seqlens):
        s = slice(cu_seqlens[i], cu_seqlens[i + 1])
        qi, ki, vi = (t[s].unsqueeze(0) for t in (q, k, v))  # (1, n, h, d)
        ref = _attn_ref(qi, ki, vi, causal=causal)
        pt = _attn_ref(qi, ki, vi, causal=causal, upcast=False)
        assert_no_worse_than_pytorch(out[s].unsqueeze(0), ref, pt, what=f"seq{i} (len {n})")


# ---------------------------------------------------------------------------
# kv cache / decode
# ---------------------------------------------------------------------------

def test_kvcache_decode_step_appends_and_attends():
    """One incremental decode step: a single new token is written into the
    cache *in place* and attended over the whole valid prefix. Ragged
    cache_seqlens is the point -- the padding past each sequence's length
    must not be attended to, and must not be overwritten at the wrong row.
    """
    batch, cache_len, nheads, headdim, dtype = 3, 256, 4, 64, torch.float16
    lengths = [200, 17, 255]  # ragged, and one that fills the cache exactly
    cache_seqlens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    g = torch.Generator(device="cuda").manual_seed(2)

    def rand(*shape):
        return torch.randn(*shape, generator=g, device="cuda", dtype=dtype)

    q = rand(batch, 1, nheads, headdim)
    k_cache, v_cache = rand(batch, cache_len, nheads, headdim), rand(batch, cache_len, nheads, headdim)
    k_new, v_new = rand(batch, 1, nheads, headdim), rand(batch, 1, nheads, headdim)
    k_before, v_before = k_cache.clone(), v_cache.clone()

    out = flash_attn_with_kvcache(q, k_cache, v_cache, k_new, v_new,
                                  cache_seqlens=cache_seqlens, causal=True)

    # This backend advances cache_seqlens itself, in place, by the number of
    # new tokens -- so `lengths` above is read from the snapshot, not from
    # the tensor. Pinned because a decode loop that also increments its own
    # counter would skip every other cache slot, and nothing about the call
    # signature hints that the argument is an in/out parameter.
    torch.testing.assert_close(
        cache_seqlens.cpu(),
        torch.tensor([n + 1 for n in lengths], dtype=torch.int32),
        atol=0, rtol=0,
    )

    for b, n in enumerate(lengths):
        # the new token landed at row n, in place, and nothing else moved
        torch.testing.assert_close(k_cache[b, n], k_new[b, 0], atol=0, rtol=0)
        torch.testing.assert_close(v_cache[b, n], v_new[b, 0], atol=0, rtol=0)
        torch.testing.assert_close(k_cache[b, :n], k_before[b, :n], atol=0, rtol=0)
        torch.testing.assert_close(k_cache[b, n + 1:], k_before[b, n + 1:], atol=0, rtol=0)
        torch.testing.assert_close(v_cache[b, n + 1:], v_before[b, n + 1:], atol=0, rtol=0)

        # ...and the output attends over exactly [0, n], nothing past it
        qb = q[b:b + 1]
        kb, vb = k_cache[b:b + 1, :n + 1], v_cache[b:b + 1, :n + 1]
        ref = _attn_ref(qb, kb, vb, causal=True)
        pt = _attn_ref(qb, kb, vb, causal=True, upcast=False)
        assert_no_worse_than_pytorch(out[b:b + 1], ref, pt, what=f"batch{b} (cache len {n})")
