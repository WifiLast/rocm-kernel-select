"""Tests for amd_tuned_torch.flash_mm_kernel -- pure PyTorch (no Triton, no GPU
required), so like test_fftconv_ops.py this exercises real numerics rather
than mocking an external kernel. Only flash_butterfly_mm_torch and matmul's
preprocessing are exercised here (the validated path); the compiled
@triton.jit kernel is skipped unless flash_kernel.available() -- see that
module's docstring for why.

Run with:

    python -m pytest source/cmp_ext_turing/tests
"""
from __future__ import annotations

import math
import time
from unittest.mock import MagicMock

import pytest
import torch

import amd_tuned_torch.flash_mm_kernel as flash_kernel
from amd_tuned_torch.flash_mm_kernel import (
    MonarchLinear,
    flash_butterfly_mm_torch,
    is_eligible,
    matmul,
)


def _butterfly_mm_ref(X, W_par, rightmost=True):
    """Independent O(L^2) reference, mirrors
    source/butterfly_matrix_kernel/reference_impl.py's butterfly_mm_ref --
    duplicated here (not imported) so this test suite has no dependency on
    that sibling directory."""
    B, F, L = X.shape
    e = int(math.log2(L))
    Y = X.clone()
    stage_order = range(e) if rightmost else reversed(range(e))
    for i in stage_order:
        stride = 1 << i
        for base in range(0, L, 2 * stride):
            j0 = slice(base, base + stride)
            j1 = slice(base + stride, base + 2 * stride)
            v0 = Y[:, :, j0].clone()
            v1 = Y[:, :, j1].clone()
            a0 = W_par[i, j0, 0].unsqueeze(0).unsqueeze(0)
            a1 = W_par[i, j0, 1].unsqueeze(0).unsqueeze(0)
            b0 = W_par[i, j1, 0].unsqueeze(0).unsqueeze(0)
            b1 = W_par[i, j1, 1].unsqueeze(0).unsqueeze(0)
            Y[:, :, j0] = a0 * v0 + b0 * v1
            Y[:, :, j1] = a1 * v0 + b1 * v1
    return Y


@pytest.mark.parametrize("B,F,L", [(1, 1, 2), (1, 1, 16), (2, 8, 32), (3, 5, 64)])
@pytest.mark.parametrize("rightmost", [True, False])
def test_flash_butterfly_mm_torch_matches_reference(B, F, L, rightmost):
    torch.manual_seed(0)
    e = int(math.log2(L))
    X = torch.randn(B, F, L, dtype=torch.float64)
    W_par = torch.randn(e, L, 2, dtype=torch.float64)

    Y_ref = _butterfly_mm_ref(X, W_par, rightmost=rightmost)
    Y = flash_butterfly_mm_torch(X, W_par, rightmost=rightmost)

    assert (Y_ref - Y).abs().max().item() < 1e-10


class TestIsEligible:
    def test_true_for_matching_shapes(self):
        L = 32
        e = int(math.log2(L))
        X = torch.randn(4, L)
        W_par = torch.randn(e, L, 2)
        assert is_eligible(X, W_par) is True

    def test_true_for_list_of_parameters(self):
        L = 16
        e = int(math.log2(L))
        X = torch.randn(L)
        W_list = [torch.randn(L, 2) for _ in range(e)]
        assert is_eligible(X, W_list) is True

    def test_false_for_non_power_of_two_L(self):
        X = torch.randn(3, 12)
        W_par = torch.randn(3, 12, 2)
        assert is_eligible(X, W_par) is False

    def test_false_for_wrong_stage_count(self):
        L = 16
        X = torch.randn(2, L)
        W_par = torch.randn(3, L, 2)  # e should be 4 for L=16
        assert is_eligible(X, W_par) is False

    def test_false_for_scalar_input(self):
        X = torch.tensor(1.0)
        W_par = torch.randn(0, 1, 2)
        assert is_eligible(X, W_par) is False

    def test_false_for_non_tensor_input(self):
        assert is_eligible([1, 2, 3], torch.randn(1, 2, 2)) is False


class TestMatmul:
    """matmul()'s own job is shape preprocessing (arbitrary leading dims,
    like torch.matmul) around the already-validated flash_butterfly_mm_torch
    -- these tests focus on that reshape/restore behavior, not the
    butterfly math itself (covered above)."""

    def test_matches_reference_for_1d_vector(self):
        torch.manual_seed(0)
        L = 16
        e = int(math.log2(L))
        X = torch.randn(L, dtype=torch.float64)
        W_par = torch.randn(e, L, 2, dtype=torch.float64)

        Y = matmul(X, W_par)
        Y_ref = _butterfly_mm_ref(X.reshape(1, 1, L), W_par).reshape(L)

        assert Y.shape == (L,)
        assert (Y - Y_ref).abs().max().item() < 1e-10

    def test_matches_reference_for_2d_matrix(self):
        torch.manual_seed(0)
        M, L = 5, 32
        e = int(math.log2(L))
        X = torch.randn(M, L, dtype=torch.float64)
        W_par = torch.randn(e, L, 2, dtype=torch.float64)

        Y = matmul(X, W_par)
        Y_ref = _butterfly_mm_ref(X.reshape(1, M, L), W_par).reshape(M, L)

        assert Y.shape == (M, L)
        assert (Y - Y_ref).abs().max().item() < 1e-10

    def test_matches_reference_for_arbitrarily_batched_input(self):
        """(..., L) with more leading dims than flash_butterfly_mm_torch's
        own hardcoded (B, F, L) accepts directly -- this is the actual
        preprocessing being tested: matmul() must flatten a 4D (2,3,4,L)
        input matmul()'s underlying kernel doesn't natively take."""
        torch.manual_seed(0)
        d0, d1, d2, L = 2, 3, 4, 16
        e = int(math.log2(L))
        X = torch.randn(d0, d1, d2, L, dtype=torch.float64)
        W_par = torch.randn(e, L, 2, dtype=torch.float64)

        Y = matmul(X, W_par)
        Y_ref = _butterfly_mm_ref(X.reshape(1, d0 * d1 * d2, L), W_par).reshape(d0, d1, d2, L)

        assert Y.shape == (d0, d1, d2, L)
        assert (Y - Y_ref).abs().max().item() < 1e-10

    def test_accepts_list_of_parameters(self):
        torch.manual_seed(0)
        M, L = 3, 16
        e = int(math.log2(L))
        X = torch.randn(M, L, dtype=torch.float64)
        W_list = [torch.randn(L, 2, dtype=torch.float64) for _ in range(e)]

        Y = matmul(X, W_list)
        Y_ref = matmul(X, torch.stack(W_list, dim=0))

        assert (Y - Y_ref).abs().max().item() < 1e-10

    def test_out_parameter_is_filled_and_returned(self):
        torch.manual_seed(0)
        M, L = 2, 16
        e = int(math.log2(L))
        X = torch.randn(M, L, dtype=torch.float64)
        W_par = torch.randn(e, L, 2, dtype=torch.float64)
        out = torch.empty(M, L, dtype=torch.float64)

        result = matmul(X, W_par, out=out)
        expected = matmul(X, W_par)

        assert result is out
        assert (out - expected).abs().max().item() < 1e-10

    def test_rightmost_false_matches_reference(self):
        torch.manual_seed(0)
        M, L = 3, 32
        e = int(math.log2(L))
        X = torch.randn(M, L, dtype=torch.float64)
        W_par = torch.randn(e, L, 2, dtype=torch.float64)

        Y = matmul(X, W_par, rightmost=False)
        Y_ref = _butterfly_mm_ref(X.reshape(1, M, L), W_par, rightmost=False).reshape(M, L)

        assert (Y - Y_ref).abs().max().item() < 1e-10

    def test_rejects_non_power_of_two_L(self):
        X = torch.randn(2, 12)
        W_par = torch.randn(3, 12, 2)
        with pytest.raises(ValueError):
            matmul(X, W_par)

    def test_rejects_dense_matrix_as_other(self):
        """The documented, deliberate non-feature: `other` must be
        butterfly factors, not an arbitrary dense (L, L) weight."""
        L = 16
        X = torch.randn(2, L)
        dense_w = torch.randn(L, L)
        with pytest.raises(ValueError):
            matmul(X, dense_w)

    def test_rejects_non_tensor_input(self):
        with pytest.raises(TypeError):
            matmul([1.0, 2.0], torch.randn(1, 2, 2))

    def test_rejects_scalar_input(self):
        W_par = torch.randn(0, 1, 2)
        with pytest.raises(ValueError):
            matmul(torch.tensor(1.0), W_par)


def _build_dense(W_par: torch.Tensor, L: int, dtype: torch.dtype) -> torch.Tensor:
    """Exact dense L x L matrix equivalent to W_par's butterfly factors,
    built by running the (already-validated) butterfly transform on the L x
    L identity matrix -- treating each of its L rows as one "row" of a
    (1, L, L) batch. Costs O(L^2 log L), the same complexity class as the
    butterfly transform itself, unlike
    reference_impl.build_dense_from_stages's O(L^3) chained-matmul
    construction (e matmuls of (L,L)@(L,L)) -- which is far too slow to use
    for the "big" L this benchmark needs. This makes the benchmark below
    apples-to-apples: both sides compute the exact same L x L linear map,
    just via two different algorithms."""
    I = torch.eye(L, dtype=dtype)
    return flash_butterfly_mm_torch(I.reshape(1, L, L), W_par)[0]


@pytest.mark.parametrize("L", [4096, 8192])
def test_benchmark_big_matrix_vs_torch_matmul(L):
    """Benchmarks matmul() against torch.matmul for a real, square
    (L, L) @ (L, L) problem -- run with `-s` to see the printed numbers.

    L in {4096, 8192} keeps this in the routine suite (both finish in low
    single-digit seconds); a THIRD point, L=16384, was measured separately
    (not run here -- ~94s total, too slow for the routine suite) and is
    recorded below because the trend across all three is the actual
    finding worth keeping.

    MEASURED ON THIS CPU-ONLY MACHINE (float32, rows=L, no GPU -- this only
    exercises matmul()'s validated torch-fallback path, never the
    unvalidated compiled Triton kernel; see flash_mm_kernel.py's VALIDATION
    STATUS section):

        L      matmul()   torch.matmul   ratio   FLOP ratio (dense/butterfly)
        4096     0.44s        0.71s      0.61x        341x
        8192     3.09s        5.96s      0.52x        630x
        16384   23.84s       46.36s      0.51x       1170x

    The ratio IMPROVES (drops) as L grows: torch.matmul is a single fused,
    multi-threaded BLAS SGEMM call (O(L^3) work for this square case) while
    matmul()'s CPU fallback is log2(L) sequential eager-PyTorch ops
    (reshape/index/stack per stage, O(L^2 log L) work) -- eager per-call
    overhead dominates at small L, but the cubic-vs-quasi-quadratic growth
    gap eventually wins out even without any kernel fusion, so matmul()
    pulls further ahead every time L doubles. This is a real, measured
    effect of the algorithm, not a benchmarking artifact -- but it is still
    the CPU-fallback path outrunning eager BLAS by constant factors (0.5x),
    nowhere near the FLOP-count ratio (630x-1170x): the large FLOP-count
    win this kernel exists to deliver is expected to show up as a single
    fused GPU kernel launch (the still-unvalidated Triton path), not as
    sequential eager CPU ops -- don't extrapolate a GPU speedup claim from
    these CPU numbers. Error also grows with L (fp32 accumulation over more
    stages), but stays tiny relative to the output's own scale at every
    size measured (max relative error ~8e-7 at L=16384).

    This test hard-asserts correctness at scale and reports CPU-fallback
    timing for the record.
    """
    torch.manual_seed(0)
    rows = L
    e = int(math.log2(L))
    dtype = torch.float32

    W_par = torch.randn(e, L, 2, dtype=dtype)
    X = torch.randn(rows, L, dtype=dtype)

    t0 = time.perf_counter()
    W_dense = _build_dense(W_par, L, dtype)
    t_build_dense = time.perf_counter() - t0

    t0 = time.perf_counter()
    Y_flash = matmul(X, W_par)
    t_flash = time.perf_counter() - t0

    t0 = time.perf_counter()
    Y_stock = torch.matmul(X, W_dense)
    t_stock = time.perf_counter() - t0

    err = (Y_flash - Y_stock).abs().max().item()
    scale = Y_stock.abs().max().item()
    flop_ratio = L / e  # dense O(L^2) vs. butterfly O(L log L) per row

    print(f"\n[benchmark] L={L} rows={rows} dtype={dtype}")
    print(f"  dense-matrix build (one-off, for this test only): {t_build_dense:.3f}s")
    print(f"  matmul() (butterfly, torch-fallback path):        {t_flash:.3f}s")
    print(f"  torch.matmul (dense, stock BLAS SGEMM):            {t_stock:.3f}s")
    print(f"  wall-time ratio matmul()/torch.matmul: {t_flash / t_stock:.2f}x")
    print(f"  theoretical per-row FLOP ratio (dense/butterfly): {flop_ratio:.0f}x")
    print(f"  max abs error vs. torch.matmul: {err:.2e} (output scale {scale:.2e})")

    assert Y_flash.shape == (rows, L)
    assert err < 1e-2 + 1e-2 * scale, (
        f"butterfly result diverged from the dense-matmul reference: "
        f"err={err:.2e}, scale={scale:.2e}"
    )


class TestFlashButterflyMmKernelSelectContest:
    """flash_butterfly_mm routes the compiled-Triton-vs-torch-fallback
    decision through the same kernel_select contest linear/bmm/conv2d/
    conv3d/group_norm/attention already use (see
    _patched_sdpa_flash_attn_rocwmma / TestFlashAttnRocwmmaKernelSelectContest
    in test_amd_tuned_torch_monkeypatch.py, whose pattern this mirrors
    exactly), instead of always preferring the compiled kernel whenever
    `available()` is True.

    Exercised here entirely on plain CPU tensors, with no real GPU: the
    eligibility gate is monkeypatched via `_flash_triton_eligible` (the one
    function flash_mm_kernel.py factors that check into specifically so it can
    be overridden as a unit -- see its own docstring), the "compiled
    kernel" is a MagicMock standing in for flash_butterfly_mm_triton (never
    actually invoked), and kernel_select._time is stubbed the same way
    test_amd_tuned_torch_monkeypatch.py's equivalent class stubs it (real
    timing needs a GPU and asserts a benchmark, not a behaviour). These
    tests are about SELECTION POLICY -- does the faster candidate win, is
    the decision cached, does disabling kernel_select restore the old
    unconditional preference -- not about the kernel's own correctness,
    which flash_butterfly_mm_torch's tests above already cover."""

    def setup_method(self):
        flash_kernel.kernel_select._ENABLED = True
        flash_kernel.kernel_select._VERIFY_ENABLED = False
        flash_kernel.kernel_select.reset()

    def teardown_method(self):
        flash_kernel.kernel_select.reset()
        flash_kernel.kernel_select._VERIFY_ENABLED = True
        flash_kernel.kernel_select._ENABLED = False

    def _xw(self, L=16, dtype=torch.float32):
        e = int(math.log2(L))
        X = torch.randn(2, 4, L, dtype=dtype)
        W_par = torch.randn(e, L, 2, dtype=dtype)
        return X, W_par

    def test_triton_wins_when_measured_faster(self, monkeypatch):
        monkeypatch.setattr(flash_kernel, "_flash_triton_eligible", lambda X: True)
        triton_out = torch.zeros(2, 4, 16)
        mocked_triton = MagicMock(return_value=triton_out)
        monkeypatch.setattr(flash_kernel, "flash_butterfly_mm_triton", mocked_triton)

        def fake_time(fn):
            out = fn()
            return None if out is None else (1.0 if out is triton_out else 2.0)

        monkeypatch.setattr(flash_kernel.kernel_select, "_time", fake_time)

        X, W_par = self._xw()
        out = flash_kernel.flash_butterfly_mm(X, W_par)
        assert out is triton_out
        mocked_triton.assert_called()
        assert flash_kernel.kernel_select.debug_winners()

    def test_torch_wins_when_measured_faster(self, monkeypatch):
        monkeypatch.setattr(flash_kernel, "_flash_triton_eligible", lambda X: True)
        X, W_par = self._xw()
        expected = flash_butterfly_mm_torch(X, W_par)  # captured before mocking

        triton_out = torch.zeros(2, 4, 16)
        mocked_triton = MagicMock(return_value=triton_out)
        monkeypatch.setattr(flash_kernel, "flash_butterfly_mm_triton", mocked_triton)

        def fake_time(fn):
            out = fn()
            return None if out is None else (2.0 if out is triton_out else 1.0)

        monkeypatch.setattr(flash_kernel.kernel_select, "_time", fake_time)

        out = flash_kernel.flash_butterfly_mm(X, W_par)
        assert torch.equal(out, expected)
        # The losing candidate is still measured once (that's how a contest
        # decides), but its output must never be the one returned.
        mocked_triton.assert_called()
        assert out is not triton_out

    def test_decision_is_cached_and_not_re_timed(self, monkeypatch):
        monkeypatch.setattr(flash_kernel, "_flash_triton_eligible", lambda X: True)
        triton_out = torch.zeros(2, 4, 16)
        mocked_triton = MagicMock(return_value=triton_out)
        monkeypatch.setattr(flash_kernel, "flash_butterfly_mm_triton", mocked_triton)

        time_calls = []

        def fake_time(fn):
            out = fn()
            time_calls.append(1)
            return None if out is None else (1.0 if out is triton_out else 2.0)

        monkeypatch.setattr(flash_kernel.kernel_select, "_time", fake_time)

        X, W_par = self._xw()
        flash_kernel.flash_butterfly_mm(X, W_par)
        assert len(time_calls) > 0
        first_round = len(time_calls)

        out2 = flash_kernel.flash_butterfly_mm(X, W_par)
        assert out2 is triton_out
        assert len(time_calls) == first_round  # no re-measurement on the cached path

    def test_ineligible_call_skips_the_contest_entirely(self, monkeypatch):
        monkeypatch.setattr(flash_kernel, "_flash_triton_eligible", lambda X: False)
        mocked_triton = MagicMock(return_value=torch.zeros(2, 4, 16))
        monkeypatch.setattr(flash_kernel, "flash_butterfly_mm_triton", mocked_triton)
        monkeypatch.setattr(flash_kernel.kernel_select, "_time",
                             lambda fn: pytest.fail("contest must not run when ineligible"))

        X, W_par = self._xw()
        expected = flash_butterfly_mm_torch(X, W_par)
        out = flash_kernel.flash_butterfly_mm(X, W_par)
        assert torch.equal(out, expected)
        mocked_triton.assert_not_called()

    def test_kernel_select_disabled_restores_old_unconditional_preference(self, monkeypatch):
        flash_kernel.kernel_select._ENABLED = False
        monkeypatch.setattr(flash_kernel, "_flash_triton_eligible", lambda X: True)
        triton_out = torch.zeros(2, 4, 16)
        mocked_triton = MagicMock(return_value=triton_out)
        monkeypatch.setattr(flash_kernel, "flash_butterfly_mm_triton", mocked_triton)
        monkeypatch.setattr(flash_kernel.kernel_select, "_time",
                             lambda fn: pytest.fail("contest must not run when kernel_select is disabled"))

        X, W_par = self._xw()
        out = flash_kernel.flash_butterfly_mm(X, W_par)
        assert out is triton_out

    def test_triton_exception_falls_back_to_torch_without_crashing(self, monkeypatch):
        """A concrete stand-in for 'the never-hardware-tested kernel is
        actually broken' -- kernel_select's contest must not let that
        surface as an exception to the caller."""
        monkeypatch.setattr(flash_kernel, "_flash_triton_eligible", lambda X: True)
        monkeypatch.setattr(flash_kernel, "flash_butterfly_mm_triton",
                             MagicMock(side_effect=RuntimeError("illegal memory access")))
        monkeypatch.setattr(flash_kernel.kernel_select, "_time",
                             lambda fn: (None if fn() is None else 1.0))

        X, W_par = self._xw()
        expected = flash_butterfly_mm_torch(X, W_par)
        out = flash_kernel.flash_butterfly_mm(X, W_par)
        assert torch.equal(out, expected)


@pytest.mark.skipif(not flash_kernel.available(), reason="requires Triton + a visible CUDA/ROCm device")
def test_matmul_matches_triton_kernel():
    from amd_tuned_torch.flash_mm_kernel import flash_butterfly_mm_triton

    torch.manual_seed(0)
    M, L = 4, 64
    e = int(math.log2(L))
    X = torch.randn(M, L, device="cuda", dtype=torch.float32)
    W_par = torch.randn(e, L, 2, device="cuda", dtype=torch.float32)

    Y_matmul = matmul(X, W_par)
    Y_kern = flash_butterfly_mm_triton(X.reshape(1, M, L), W_par).reshape(M, L)

    assert (Y_matmul - Y_kern).abs().max().item() < 1e-4


def _probe_dense_matrix(layer: MonarchLinear, in_features: int, dtype=torch.float64) -> torch.Tensor:
    """Builds the (in_features, out_features) matrix `layer` implements by
    feeding the standard basis through it -- an algorithm-agnostic
    correctness check (independent of how the layer's internals are
    wired) that `layer(x) == x @ this_matrix` for any fresh x, i.e. that
    the layer really is one fixed linear map applied consistently."""
    with torch.no_grad():
        return layer(torch.eye(in_features, dtype=dtype))


class TestMonarchLinear:
    """MonarchLinear (ported from source/monarch.py -- see flash_mm_kernel.py's
    own MONARCH MATRICES docstring section for what this is and why it's
    here) -- a structured-matrix class unrelated to the butterfly kernel
    above, included in this file on request. Unlike the butterfly tests,
    every test here runs the real numerics directly (no CUDA/Triton
    needed at all -- MonarchLinear is plain torch.bmm/permute/reshape)."""

    def test_matches_independent_einsum_reference_2d(self):
        """Hand-derived reference for the classic 2-factor case, built
        without reusing any of MonarchLinear's own permute/reshape
        choreography -- an einsum with explicit index labels instead."""
        torch.manual_seed(0)
        in_dims, out_dims = (3, 5), (4, 6)  # not powers of two
        in_f, out_f = math.prod(in_dims), math.prod(out_dims)
        layer = MonarchLinear(in_f, out_f, in_dims, out_dims, bias=False).double()
        w0, w1 = layer.weights[0].detach(), layer.weights[1].detach()

        x = torch.randn(2, 7, in_f, dtype=torch.float64)
        y = layer(x)

        x_r = x.reshape(-1, *in_dims)
        r1 = torch.einsum("bij,jik->bkj", x_r, w0)
        r2 = torch.einsum("bkj,kjl->bkl", r1, w1)
        y_ref = r2.reshape(*x.shape[:-1], out_f)

        assert (y - y_ref).abs().max().item() < 1e-12

    @pytest.mark.parametrize("in_dims,out_dims", [
        ((3, 5), (4, 6)),        # 2D, non-power-of-2, rectangular
        ((2, 3, 5), (3, 4, 5)),  # 3D generalization, non-power-of-2
        ((7, 11), (11, 7)),      # 2D, primes, in/out swapped
    ])
    def test_dense_matrix_self_consistency(self, in_dims, out_dims):
        """layer(x) must equal x @ (the dense matrix the layer itself
        implements, recovered by probing it with the identity) -- verifies
        linearity, batching, and shape handling independent of the
        specific factorization."""
        torch.manual_seed(0)
        in_f, out_f = math.prod(in_dims), math.prod(out_dims)
        layer = MonarchLinear(in_f, out_f, in_dims, out_dims, bias=False).double()
        dense = _probe_dense_matrix(layer, in_f)

        x = torch.randn(5, in_f, dtype=torch.float64)
        assert (layer(x) - x @ dense).abs().max().item() < 1e-10

    def test_gradcheck(self):
        torch.manual_seed(0)
        in_dims, out_dims = (3, 5), (4, 6)
        in_f, out_f = math.prod(in_dims), math.prod(out_dims)
        layer = MonarchLinear(in_f, out_f, in_dims, out_dims, bias=True).double()
        x = torch.randn(2, in_f, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(layer, (x,), eps=1e-6, atol=1e-4)

    def test_checkpoint_matches_no_checkpoint_forward_and_backward(self):
        torch.manual_seed(0)
        in_dims, out_dims = (3, 5), (4, 6)
        in_f, out_f = math.prod(in_dims), math.prod(out_dims)
        layer_ckpt = MonarchLinear(in_f, out_f, in_dims, out_dims, bias=True, checkpoint=True)
        layer_plain = MonarchLinear(in_f, out_f, in_dims, out_dims, bias=True)
        layer_plain.load_state_dict(layer_ckpt.state_dict())

        x_ckpt = torch.randn(3, in_f, requires_grad=True)
        x_plain = x_ckpt.detach().clone().requires_grad_(True)
        out_ckpt = layer_ckpt(x_ckpt)
        out_plain = layer_plain(x_plain)
        out_ckpt.sum().backward()
        out_plain.sum().backward()

        assert torch.allclose(out_ckpt, out_plain, atol=1e-6)
        assert torch.allclose(x_ckpt.grad, x_plain.grad, atol=1e-6)
        assert all(torch.allclose(a.grad, b.grad, atol=1e-6)
                   for a, b in zip(layer_ckpt.weights, layer_plain.weights))

    def test_rejects_mismatched_dims_length(self):
        with pytest.raises(AssertionError):
            MonarchLinear(15, 96, (3, 5), (4, 4, 6))  # len(out_dims)=3 != len(in_dims)=2

    def test_rejects_dims_not_matching_features(self):
        with pytest.raises(AssertionError):
            MonarchLinear(16, 24, (3, 5), (4, 6))  # 3*5=15 != in_features=16

    def test_rejects_single_factor(self):
        # len(in_dims) > 1 is required -- a single factor is trivially just
        # a dense nn.Linear, not a genuine Monarch decomposition.
        with pytest.raises(AssertionError):
            MonarchLinear(7, 11, (7,), (11,))

    def test_bias_changes_output(self):
        torch.manual_seed(0)
        in_dims, out_dims = (3, 5), (4, 6)
        in_f, out_f = math.prod(in_dims), math.prod(out_dims)
        layer = MonarchLinear(in_f, out_f, in_dims, out_dims, bias=True)
        x = torch.randn(2, in_f)
        with torch.no_grad():
            layer.bias.zero_()
            out_zero_bias = layer(x)
            layer.bias.add_(1.0)
            out_nonzero_bias = layer(x)
        assert not torch.allclose(out_zero_bias, out_nonzero_bias)
        assert torch.allclose(out_nonzero_bias - out_zero_bias, torch.ones_like(out_zero_bias))

    def test_no_bias(self):
        in_dims, out_dims = (3, 5), (4, 6)
        in_f, out_f = math.prod(in_dims), math.prod(out_dims)
        layer = MonarchLinear(in_f, out_f, in_dims, out_dims, bias=False)
        assert layer.bias is None
        assert "bias" not in dict(layer.named_parameters())

    def test_leading_batch_dims_preserved(self):
        """Arbitrarily-shaped leading dims (not just a single flat batch)
        must round-trip through the internal `.view(-1, *in_dims)` flatten
        correctly."""
        torch.manual_seed(0)
        in_dims, out_dims = (3, 5), (4, 6)
        in_f, out_f = math.prod(in_dims), math.prod(out_dims)
        layer = MonarchLinear(in_f, out_f, in_dims, out_dims, bias=False).double()
        x = torch.randn(2, 3, 4, in_f, dtype=torch.float64)
        out = layer(x)
        assert out.shape == (2, 3, 4, out_f)

        dense = _probe_dense_matrix(layer, in_f)
        assert (out - x @ dense).abs().max().item() < 1e-10


class TestMatmulMonarchDispatch:
    """matmul()'s automatic dispatch to Monarch factors, detected via
    _is_monarch_factors (a list/tuple of >=2 3D weight tensors) instead
    of butterfly's (E, L, 2) convention -- see flash_mm_kernel.py's MONARCH
    MATRICES docstring section."""

    def test_matches_monarchlinear_forward_exactly(self):
        torch.manual_seed(0)
        in_dims, out_dims = (3, 5), (4, 6)
        in_f, out_f = math.prod(in_dims), math.prod(out_dims)
        layer = MonarchLinear(in_f, out_f, in_dims, out_dims, bias=False).double()
        x = torch.randn(2, 7, in_f, dtype=torch.float64)

        y_module = layer(x)
        y_matmul = matmul(x, list(layer.weights))

        assert (y_module - y_matmul).abs().max().item() == 0.0

    def test_matches_monarchlinear_with_bias(self):
        torch.manual_seed(0)
        in_dims, out_dims = (3, 5), (4, 6)
        in_f, out_f = math.prod(in_dims), math.prod(out_dims)
        layer = MonarchLinear(in_f, out_f, in_dims, out_dims, bias=True).double()
        x = torch.randn(4, in_f, dtype=torch.float64)

        assert torch.equal(layer(x), flash_kernel.monarch_matmul(x, list(layer.weights), layer.bias))

    def test_rejects_mismatched_input_dim(self):
        torch.manual_seed(0)
        in_dims, out_dims = (3, 5), (4, 6)
        layer = MonarchLinear(math.prod(in_dims), math.prod(out_dims), in_dims, out_dims, bias=False)
        with pytest.raises(ValueError):
            matmul(torch.randn(2, 20), list(layer.weights))

    def test_gradient_flows_through_matmul_dispatch(self):
        torch.manual_seed(0)
        in_dims, out_dims = (3, 5), (4, 6)
        in_f = math.prod(in_dims)
        layer = MonarchLinear(in_f, math.prod(out_dims), in_dims, out_dims, bias=False).double()
        x = torch.randn(2, in_f, dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradcheck(lambda inp: matmul(inp, list(layer.weights)), (x,),
                                         eps=1e-6, atol=1e-4)

    def test_butterfly_dispatch_unaffected(self):
        """Regression check: adding Monarch detection to matmul() must not
        change anything about the pre-existing butterfly path."""
        torch.manual_seed(0)
        L = 16
        e = int(math.log2(L))
        X = torch.randn(3, L, dtype=torch.float64)
        W_par = torch.randn(e, L, 2, dtype=torch.float64)

        Y = matmul(X, W_par)
        Y_ref = _butterfly_mm_ref(X.reshape(1, 3, L), W_par).reshape(3, L)

        assert (Y - Y_ref).abs().max().item() < 1e-10

    def test_is_eligible_recognizes_monarch_factors(self):
        in_dims, out_dims = (3, 5), (4, 6)
        layer = MonarchLinear(math.prod(in_dims), math.prod(out_dims), in_dims, out_dims, bias=False)
        x_ok = torch.randn(2, math.prod(in_dims))
        x_wrong = torch.randn(2, math.prod(in_dims) + 1)

        assert is_eligible(x_ok, list(layer.weights)) is True
        assert is_eligible(x_wrong, list(layer.weights)) is False

    def test_is_monarch_factors_does_not_misclassify_butterfly(self):
        L = 16
        e = int(math.log2(L))
        assert flash_kernel._is_monarch_factors(torch.randn(e, L, 2)) is False
        assert flash_kernel._is_monarch_factors([torch.randn(L, 2) for _ in range(e)]) is False
        assert flash_kernel._is_monarch_factors([torch.randn(4, 3, 5)]) is False  # only 1 factor
