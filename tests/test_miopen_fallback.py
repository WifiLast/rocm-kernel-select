"""Tests for amd_tuned_torch.miopen_fallback's sparse conv1d fast path
(_try_sparse_conv1d_fastpath) and FFT-conv1d fast path
(_try_fftconv1d_fastpath), and their ordering inside _miopen_safe_conv's
wrapped() relative to the pre-existing causal-conv1d fast path. Does not
cover the MIOpen retry/CPU-fallback machinery itself (untested before this
file existed) -- see the module docstring for that design; out of scope
here.

Run with:

    python -m pytest source/cmp_ext_turing/tests
"""
from __future__ import annotations

from unittest.mock import MagicMock

import torch

import amd_tuned_torch.miopen_fallback as miopen_fallback_module


class TestTrySparseConv1dFastpath:
    def test_declines_for_grouped_conv(self, monkeypatch):
        fake_maybe = MagicMock(return_value=torch.zeros(1, 4, 8))
        monkeypatch.setattr(miopen_fallback_module.flexgemm_ops, "maybe_sparse_conv1d", fake_maybe)
        x = torch.randn(1, 4, 10)
        w = torch.randn(4, 1, 3)
        out = miopen_fallback_module._try_sparse_conv1d_fastpath(x, w, None, 1, 1, 1, groups=4)
        assert out is None
        fake_maybe.assert_not_called()

    def test_delegates_for_groups_1(self, monkeypatch):
        sparse_result = torch.zeros(1, 4, 8)
        fake_maybe = MagicMock(return_value=sparse_result)
        monkeypatch.setattr(miopen_fallback_module.flexgemm_ops, "maybe_sparse_conv1d", fake_maybe)
        x = torch.randn(1, 2, 10)
        w = torch.randn(4, 2, 3)
        bias = torch.randn(4)

        out = miopen_fallback_module._try_sparse_conv1d_fastpath(x, w, bias, 1, 1, 1, groups=1)

        assert out is sparse_result
        fake_maybe.assert_called_once_with(x, w, bias, stride=(1,), padding=(1,), dilation=(1,))

    def test_declines_when_input_requires_grad(self, monkeypatch):
        """Regression test: flexgemm_ops's sparse conv1d path has no
        backward pass (extracts `input` via .detach() internally, per
        sparse_conv1d_from_dense), so a grad-tracked call must never reach
        it -- see _grad_safe's docstring for the silently-broken-backward
        failure mode this prevents. This check was originally missing
        entirely."""
        fake_maybe = MagicMock(return_value=torch.zeros(1, 4, 8))
        monkeypatch.setattr(miopen_fallback_module.flexgemm_ops, "maybe_sparse_conv1d", fake_maybe)
        x = torch.randn(1, 2, 10, requires_grad=True)
        w = torch.randn(4, 2, 3)

        out = miopen_fallback_module._try_sparse_conv1d_fastpath(x, w, None, 1, 1, 1, groups=1)

        assert out is None
        fake_maybe.assert_not_called()

    def test_declines_when_weight_requires_grad(self, monkeypatch):
        fake_maybe = MagicMock(return_value=torch.zeros(1, 4, 8))
        monkeypatch.setattr(miopen_fallback_module.flexgemm_ops, "maybe_sparse_conv1d", fake_maybe)
        x = torch.randn(1, 2, 10)
        w = torch.randn(4, 2, 3, requires_grad=True)

        out = miopen_fallback_module._try_sparse_conv1d_fastpath(x, w, None, 1, 1, 1, groups=1)

        assert out is None
        fake_maybe.assert_not_called()

    def test_declines_when_bias_requires_grad(self, monkeypatch):
        fake_maybe = MagicMock(return_value=torch.zeros(1, 4, 8))
        monkeypatch.setattr(miopen_fallback_module.flexgemm_ops, "maybe_sparse_conv1d", fake_maybe)
        x = torch.randn(1, 2, 10)
        w = torch.randn(4, 2, 3)
        bias = torch.randn(4, requires_grad=True)

        out = miopen_fallback_module._try_sparse_conv1d_fastpath(x, w, bias, 1, 1, 1, groups=1)

        assert out is None
        fake_maybe.assert_not_called()

    def test_delegates_when_grad_disabled_even_if_requires_grad(self, monkeypatch):
        """torch.no_grad()/inference_mode: a tensor can carry
        requires_grad=True but not actually be tracked -- must still be
        treated as safe, same as amd_tuned_torch.__init__._grad_safe."""
        sparse_result = torch.zeros(1, 4, 8)
        fake_maybe = MagicMock(return_value=sparse_result)
        monkeypatch.setattr(miopen_fallback_module.flexgemm_ops, "maybe_sparse_conv1d", fake_maybe)
        x = torch.randn(1, 2, 10, requires_grad=True)
        w = torch.randn(4, 2, 3)

        with torch.no_grad():
            out = miopen_fallback_module._try_sparse_conv1d_fastpath(x, w, None, 1, 1, 1, groups=1)

        assert out is sparse_result
        fake_maybe.assert_called_once()

    def test_unwraps_tuple_args_via_scalar(self, monkeypatch):
        fake_maybe = MagicMock(return_value=None)
        monkeypatch.setattr(miopen_fallback_module.flexgemm_ops, "maybe_sparse_conv1d", fake_maybe)
        x = torch.randn(1, 2, 10)
        w = torch.randn(4, 2, 3)

        miopen_fallback_module._try_sparse_conv1d_fastpath(
            x, w, None, stride=(2,), padding=(1,), dilation=(1,), groups=1)

        fake_maybe.assert_called_once_with(x, w, None, stride=(2,), padding=(1,), dilation=(1,))

    def test_returns_none_when_maybe_sparse_declines(self, monkeypatch):
        monkeypatch.setattr(miopen_fallback_module.flexgemm_ops, "maybe_sparse_conv1d",
                             MagicMock(return_value=None))
        x = torch.randn(1, 2, 10)
        w = torch.randn(4, 2, 3)
        assert miopen_fallback_module._try_sparse_conv1d_fastpath(x, w, None, 1, 0, 1, groups=1) is None


class TestMiopenSafeConvOrdering:
    """wrapped() tries the causal-conv1d fast path first, then the FFT-conv1d
    fast path, then the sparse fast path, before ever calling the real op --
    verify all three are consulted in that order and any one of them
    short-circuits the rest."""

    def test_causal_fastpath_wins_when_both_could_apply(self, monkeypatch):
        causal_result = torch.zeros(1, 4, 8)
        monkeypatch.setattr(miopen_fallback_module, "_try_causal_conv1d_fastpath",
                             MagicMock(return_value=causal_result))
        fake_fftconv = MagicMock(return_value=torch.zeros(1, 4, 8))
        monkeypatch.setattr(miopen_fallback_module, "_try_fftconv1d_fastpath", fake_fftconv)
        fake_sparse = MagicMock(return_value=torch.zeros(1, 4, 8))
        monkeypatch.setattr(miopen_fallback_module, "_try_sparse_conv1d_fastpath", fake_sparse)
        orig_fn = MagicMock()

        wrapped = miopen_fallback_module._miopen_safe_conv("conv1d", orig_fn)
        x = torch.randn(1, 4, 10)
        w = torch.randn(4, 1, 3)

        out = wrapped(x, w)

        assert out is causal_result
        fake_fftconv.assert_not_called()
        fake_sparse.assert_not_called()
        orig_fn.assert_not_called()

    def test_fftconv_fastpath_used_when_causal_declines(self, monkeypatch):
        monkeypatch.setattr(miopen_fallback_module, "_try_causal_conv1d_fastpath",
                             MagicMock(return_value=None))
        fftconv_result = torch.zeros(1, 4, 8)
        monkeypatch.setattr(miopen_fallback_module, "_try_fftconv1d_fastpath",
                             MagicMock(return_value=fftconv_result))
        fake_sparse = MagicMock(return_value=torch.zeros(1, 4, 8))
        monkeypatch.setattr(miopen_fallback_module, "_try_sparse_conv1d_fastpath", fake_sparse)
        orig_fn = MagicMock()

        wrapped = miopen_fallback_module._miopen_safe_conv("conv1d", orig_fn)
        x = torch.randn(1, 2, 10)
        w = torch.randn(4, 2, 3)

        out = wrapped(x, w)

        assert out is fftconv_result
        fake_sparse.assert_not_called()
        orig_fn.assert_not_called()

    def test_sparse_fastpath_used_when_causal_and_fftconv_decline(self, monkeypatch):
        monkeypatch.setattr(miopen_fallback_module, "_try_causal_conv1d_fastpath",
                             MagicMock(return_value=None))
        monkeypatch.setattr(miopen_fallback_module, "_try_fftconv1d_fastpath",
                             MagicMock(return_value=None))
        sparse_result = torch.zeros(1, 4, 8)
        monkeypatch.setattr(miopen_fallback_module, "_try_sparse_conv1d_fastpath",
                             MagicMock(return_value=sparse_result))
        orig_fn = MagicMock()

        wrapped = miopen_fallback_module._miopen_safe_conv("conv1d", orig_fn)
        x = torch.randn(1, 2, 10)
        w = torch.randn(4, 2, 3)

        out = wrapped(x, w)

        assert out is sparse_result
        orig_fn.assert_not_called()

    def test_falls_through_to_real_op_when_all_decline(self, monkeypatch):
        monkeypatch.setattr(miopen_fallback_module, "_try_causal_conv1d_fastpath",
                             MagicMock(return_value=None))
        monkeypatch.setattr(miopen_fallback_module, "_try_fftconv1d_fastpath",
                             MagicMock(return_value=None))
        monkeypatch.setattr(miopen_fallback_module, "_try_sparse_conv1d_fastpath",
                             MagicMock(return_value=None))
        real_result = torch.zeros(1, 4, 8)
        orig_fn = MagicMock(return_value=real_result)

        wrapped = miopen_fallback_module._miopen_safe_conv("conv1d", orig_fn)
        x = torch.randn(1, 2, 10)
        w = torch.randn(4, 2, 3)

        out = wrapped(x, w)

        assert out is real_result
        orig_fn.assert_called_once()


class TestTryFftconv1dFastpath:
    """_try_fftconv1d_fastpath contests fftconv_ops.fftconv1d_candidate
    against `orig_fn` (stock conv1d) through kernel_select.pick -- these
    mock kernel_select.pick itself rather than exercise a real contest, same
    discipline test_miopen_fallback.py already uses for flexgemm_ops/
    causal_conv1d (mocked, no real extension involved)."""

    def test_declines_when_disabled(self, monkeypatch):
        monkeypatch.setattr(miopen_fallback_module.fftconv_ops, "fft_conv1d_enabled", lambda: False)
        fake_pick = MagicMock()
        monkeypatch.setattr(miopen_fallback_module.kernel_select, "pick", fake_pick)
        x = torch.randn(1, 2, 200)
        w = torch.randn(2, 2, 129)
        orig_fn = MagicMock()

        out = miopen_fallback_module._try_fftconv1d_fastpath(
            x, w, None, 1, 0, 1, groups=1, orig_fn=orig_fn)

        assert out is None
        fake_pick.assert_not_called()

    def test_delegates_to_kernel_select_pick_with_fftconv_tolerance(self, monkeypatch):
        monkeypatch.setattr(miopen_fallback_module.fftconv_ops, "fft_conv1d_enabled", lambda: True)
        fake_pick = MagicMock(return_value="picked")
        monkeypatch.setattr(miopen_fallback_module.kernel_select, "pick", fake_pick)
        x = torch.randn(1, 2, 200)
        w = torch.randn(2, 2, 129)
        orig_fn = MagicMock()

        out = miopen_fallback_module._try_fftconv1d_fastpath(
            x, w, None, 1, 0, 1, groups=1, orig_fn=orig_fn)

        assert out == "picked"
        args, kwargs = fake_pick.call_args
        assert args[0] == "conv1d"
        assert kwargs["tolerance"] == miopen_fallback_module.fftconv_ops.fftconv_tolerance(x.dtype)

    def test_candidate_thunks_wire_to_fftconv_and_orig_fn(self, monkeypatch):
        monkeypatch.setattr(miopen_fallback_module.fftconv_ops, "fft_conv1d_enabled", lambda: True)
        captured = {}

        def fake_pick(kind, input, weight, stride, padding, dilation, candidates, tolerance=None):
            captured["candidates"] = dict(candidates)
            return None

        monkeypatch.setattr(miopen_fallback_module.kernel_select, "pick", fake_pick)
        monkeypatch.setattr(miopen_fallback_module.fftconv_ops, "fftconv1d_candidate",
                             MagicMock(return_value="fft_out"))
        x = torch.randn(1, 2, 200)
        w = torch.randn(2, 2, 129)
        orig_fn = MagicMock(return_value="stock_out")

        miopen_fallback_module._try_fftconv1d_fastpath(
            x, w, None, 1, 0, 1, groups=1, orig_fn=orig_fn)

        assert captured["candidates"]["fftconv"]() == "fft_out"
        assert captured["candidates"]["stock"]() == "stock_out"
        orig_fn.assert_called_once_with(x, w, None, 1, 0, 1, 1)


class TestIsDepthwiseConv1dEligible:
    """_is_depthwise_conv1d_eligible gates the depthwise candidate folded
    into _try_fftconv1d_fastpath's contest -- see that function's and the
    module's DEPTHWISE CONV1D FAST PATH docstring section.

    The function's first gate is `input.is_cuda`, which is False for every
    plain torch.randn(...) tensor in this CPU-only test environment --
    _patch_is_cuda monkeypatches torch.Tensor.is_cuda to True for the
    "should be eligible" cases so the rest of the condition is actually
    exercised, and test_ineligible_on_cpu checks the real (unpatched)
    default is correctly rejected."""

    @staticmethod
    def _patch_is_cuda(monkeypatch, value: bool) -> None:
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: value))

    def test_eligible_depthwise_odd_width(self, monkeypatch):
        self._patch_is_cuda(monkeypatch, True)
        x = torch.randn(1, 4, 10)
        w = torch.randn(4, 1, 3)
        assert miopen_fallback_module._is_depthwise_conv1d_eligible(
            x, w, stride=1, dilation=1, groups=4) is True

    def test_ineligible_when_not_depthwise(self, monkeypatch):
        self._patch_is_cuda(monkeypatch, True)
        x = torch.randn(1, 4, 10)
        w = torch.randn(4, 2, 3)
        assert miopen_fallback_module._is_depthwise_conv1d_eligible(
            x, w, stride=1, dilation=1, groups=2) is False

    def test_ineligible_for_even_width(self, monkeypatch):
        self._patch_is_cuda(monkeypatch, True)
        x = torch.randn(1, 4, 10)
        w = torch.randn(4, 1, 4)
        assert miopen_fallback_module._is_depthwise_conv1d_eligible(
            x, w, stride=1, dilation=1, groups=4) is False

    def test_ineligible_for_stride_or_dilation(self, monkeypatch):
        self._patch_is_cuda(monkeypatch, True)
        x = torch.randn(1, 4, 10)
        w = torch.randn(4, 1, 3)
        assert miopen_fallback_module._is_depthwise_conv1d_eligible(
            x, w, stride=2, dilation=1, groups=4) is False
        assert miopen_fallback_module._is_depthwise_conv1d_eligible(
            x, w, stride=1, dilation=2, groups=4) is False

    def test_ineligible_on_cpu(self):
        x = torch.randn(1, 4, 10)
        w = torch.randn(4, 1, 3)
        assert x.is_cuda is False
        assert miopen_fallback_module._is_depthwise_conv1d_eligible(
            x, w, stride=1, dilation=1, groups=4) is False


class TestTryFftconv1dFastpathDepthwiseCandidate:
    """The depthwise candidate _try_fftconv1d_fastpath folds into its
    contest when depthwise_conv1d_ops is available and the call is
    depthwise-eligible -- see that function's docstring for why this is
    one contest rather than a second sequential fast path."""

    def test_depthwise_candidate_added_when_eligible_and_available(self, monkeypatch):
        TestIsDepthwiseConv1dEligible._patch_is_cuda(monkeypatch, True)
        monkeypatch.setattr(miopen_fallback_module.fftconv_ops, "fft_conv1d_enabled", lambda: True)
        monkeypatch.setattr(miopen_fallback_module.depthwise_conv1d_ops, "available", lambda: True)
        monkeypatch.setattr(miopen_fallback_module.depthwise_conv1d_ops, "depthwise_conv1d_candidate",
                             MagicMock(return_value="depthwise_out"))
        monkeypatch.setattr(miopen_fallback_module.fftconv_ops, "fftconv1d_candidate",
                             MagicMock(return_value="fft_out"))
        captured = {}

        def fake_pick(kind, input, weight, stride, padding, dilation, candidates, tolerance=None):
            captured["candidates"] = dict(candidates)
            return None

        monkeypatch.setattr(miopen_fallback_module.kernel_select, "pick", fake_pick)
        x = torch.randn(1, 4, 10)
        w = torch.randn(4, 1, 3)
        orig_fn = MagicMock(return_value="stock_out")

        miopen_fallback_module._try_fftconv1d_fastpath(
            x, w, None, 1, 1, 1, groups=4, orig_fn=orig_fn)

        assert set(captured["candidates"]) == {"fftconv", "depthwise", "stock"}
        assert captured["candidates"]["depthwise"]() == "depthwise_out"

    def test_depthwise_candidate_not_added_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(miopen_fallback_module.fftconv_ops, "fft_conv1d_enabled", lambda: True)
        monkeypatch.setattr(miopen_fallback_module.depthwise_conv1d_ops, "available", lambda: False)
        monkeypatch.setattr(miopen_fallback_module.fftconv_ops, "fftconv1d_candidate",
                             MagicMock(return_value="fft_out"))
        captured = {}

        def fake_pick(kind, input, weight, stride, padding, dilation, candidates, tolerance=None):
            captured["candidates"] = dict(candidates)
            return None

        monkeypatch.setattr(miopen_fallback_module.kernel_select, "pick", fake_pick)
        x = torch.randn(1, 4, 10)
        w = torch.randn(4, 1, 3)
        orig_fn = MagicMock(return_value="stock_out")

        miopen_fallback_module._try_fftconv1d_fastpath(
            x, w, None, 1, 1, 1, groups=4, orig_fn=orig_fn)

        assert set(captured["candidates"]) == {"fftconv", "stock"}

    def test_contest_runs_with_only_depthwise_when_fftconv_disabled(self, monkeypatch):
        TestIsDepthwiseConv1dEligible._patch_is_cuda(monkeypatch, True)
        monkeypatch.setattr(miopen_fallback_module.fftconv_ops, "fft_conv1d_enabled", lambda: False)
        monkeypatch.setattr(miopen_fallback_module.depthwise_conv1d_ops, "available", lambda: True)
        monkeypatch.setattr(miopen_fallback_module.depthwise_conv1d_ops, "depthwise_conv1d_candidate",
                             MagicMock(return_value="depthwise_out"))
        captured = {}

        def fake_pick(kind, input, weight, stride, padding, dilation, candidates, tolerance=None):
            captured["candidates"] = dict(candidates)
            return "picked"

        monkeypatch.setattr(miopen_fallback_module.kernel_select, "pick", fake_pick)
        x = torch.randn(1, 4, 10)
        w = torch.randn(4, 1, 3)
        orig_fn = MagicMock(return_value="stock_out")

        out = miopen_fallback_module._try_fftconv1d_fastpath(
            x, w, None, 1, 1, 1, groups=4, orig_fn=orig_fn)

        assert out == "picked"
        assert set(captured["candidates"]) == {"depthwise", "stock"}


class TestGradSafe:
    """Same semantics as amd_tuned_torch.__init__._grad_safe, duplicated
    into miopen_fallback.py to avoid a circular import -- see that
    function's docstring for the real bug this exists to prevent."""

    def test_safe_when_no_tensor_requires_grad(self):
        assert miopen_fallback_module._grad_safe(torch.randn(2, 2), torch.randn(2, 2)) is True

    def test_unsafe_when_any_tensor_requires_grad(self):
        assert miopen_fallback_module._grad_safe(
            torch.randn(2, 2), torch.randn(2, 2, requires_grad=True)) is False

    def test_safe_inside_no_grad_even_with_requires_grad_tensor(self):
        t = torch.randn(2, 2, requires_grad=True)
        with torch.no_grad():
            assert miopen_fallback_module._grad_safe(t) is True

    def test_safe_inside_inference_mode(self):
        t = torch.randn(2, 2, requires_grad=True)
        with torch.inference_mode():
            assert miopen_fallback_module._grad_safe(t) is True

    def test_ignores_non_tensor_arguments(self):
        assert miopen_fallback_module._grad_safe(None, "not a tensor", 42) is True


class TestSparseConv1dDeclinesTensorSubclasses:
    """FlexGEMM's kernels read dense storage, so a Tensor subclass reaching
    them fails as an illegal memory access from inside the subclass's own
    dispatch fallback -- not as a catchable exception. Same guard, and same
    reason, as amd_tuned_torch._dispatch._usable."""

    class _FakeQuantizedTensor(torch.Tensor):
        pass

    def test_predicate_accepts_plain_tensors_and_none(self):
        assert miopen_fallback_module._is_plain_tensor(torch.zeros(2)) is True
        assert miopen_fallback_module._is_plain_tensor(torch.nn.Parameter(torch.zeros(2))) is True
        assert miopen_fallback_module._is_plain_tensor(None) is True, "bias is legitimately None"

    def test_predicate_rejects_a_subclass(self):
        assert miopen_fallback_module._is_plain_tensor(
            self._FakeQuantizedTensor(torch.zeros(2))) is False

    def test_fastpath_declines_a_quantized_weight(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            miopen_fallback_module.flexgemm_ops, "maybe_sparse_conv1d",
            lambda *a, **k: called.append(1))

        x = torch.zeros(1, 4, 64)
        w = self._FakeQuantizedTensor(torch.zeros(4, 4, 3))
        assert miopen_fallback_module._try_sparse_conv1d_fastpath(
            x, w, None, 1, 0, 1, 1) is None
        assert not called, "FlexGEMM must never be reached with a subclass"

    def test_fastpath_still_reaches_flexgemm_for_plain_tensors(self, monkeypatch):
        """The guard must not disable the fast path for ordinary tensors."""
        sentinel = torch.zeros(1, 4, 62)
        monkeypatch.setattr(
            miopen_fallback_module.flexgemm_ops, "maybe_sparse_conv1d",
            lambda *a, **k: sentinel)

        x = torch.zeros(1, 4, 64)
        w = torch.zeros(4, 4, 3)
        assert miopen_fallback_module._try_sparse_conv1d_fastpath(
            x, w, None, 1, 0, 1, 1) is sentinel
