"""Tests for amd_tuned_torch.flexgemm_ops -- the `flex_gemm`
(third_party/FlexGEMM) adapter. No real flex_gemm/CUDA/HIP extension anywhere
here: every underlying call is mocked, same discipline as aiter/TE in
test_amd_tuned_torch_monkeypatch.py.

Run with:

    python -m pytest source/cmp_ext_turing/tests
"""
from __future__ import annotations

import importlib
import os
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

import amd_tuned_torch.flexgemm_ops as flexgemm_ops_module


class TestAvailable:
    def test_unavailable_when_package_missing(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "_FLEX_GEMM_AVAILABLE", False)
        assert flexgemm_ops_module.available() is False

    def test_available_when_package_present(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "_FLEX_GEMM_AVAILABLE", True)
        assert flexgemm_ops_module.available() is True


class TestDropOutWhenUnavailable:
    def test_sparse_conv3d_returns_none(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: False)
        assert flexgemm_ops_module.sparse_conv3d(None, None, None, None) is None

    def test_sparse_submanifold_conv3d_returns_none(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: False)
        assert flexgemm_ops_module.sparse_submanifold_conv3d(None, None, None, None) is None

    def test_grid_sample_3d_returns_none(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: False)
        assert flexgemm_ops_module.grid_sample_3d(None, None, None, None) is None

    def test_encode_seq_returns_none(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: False)
        assert flexgemm_ops_module.encode_seq(None, None) is None

    def test_decode_seq_returns_none(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: False)
        assert flexgemm_ops_module.decode_seq(None, None) is None


class TestSparseConv3d:
    def test_unpacks_neighbor_cache_from_result(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: True)
        fake = MagicMock(return_value=("out_feats", "out_coords", "cache"))
        monkeypatch.setattr(flexgemm_ops_module, "_sparse_conv3d", fake)

        result = flexgemm_ops_module.sparse_conv3d("feats", "coords", (1, 1, 2, 2, 2), "weight")

        assert result == ("out_feats", "out_coords")
        fake.assert_called_once_with(
            "feats", "coords", (1, 1, 2, 2, 2), "weight", bias=None,
            stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1))

    def test_runtime_error_returns_none(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: True)
        monkeypatch.setattr(flexgemm_ops_module, "_sparse_conv3d",
                             MagicMock(side_effect=RuntimeError("bad shape")))
        assert flexgemm_ops_module.sparse_conv3d("f", "c", (1, 1, 2, 2, 2), "w") is None


class TestSparseSubmanifoldConv3d:
    def test_unpacks_feats_only(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: True)
        fake = MagicMock(return_value=("out_feats", "cache"))
        monkeypatch.setattr(flexgemm_ops_module, "_sparse_submanifold_conv3d", fake)

        result = flexgemm_ops_module.sparse_submanifold_conv3d(
            "feats", "coords", (1, 1, 2, 2, 2), "weight")

        assert result == "out_feats"


class TestGridSample3d:
    def test_delegates(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: True)
        fake = MagicMock(return_value="sampled")
        monkeypatch.setattr(flexgemm_ops_module, "_grid_sample_3d", fake)

        result = flexgemm_ops_module.grid_sample_3d("feats", "coords", (1, 1, 2, 2, 2), "grid")

        assert result == "sampled"
        fake.assert_called_once_with("feats", "coords", (1, 1, 2, 2, 2), "grid", mode="trilinear")


class TestEncodeDecodeSeq:
    def test_encode_delegates(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: True)
        fake = MagicMock(return_value="codes")
        monkeypatch.setattr(flexgemm_ops_module, "_encode_seq", fake)

        result = flexgemm_ops_module.encode_seq("coords", (1, 1, 2, 2, 2))

        assert result == "codes"
        fake.assert_called_once_with("coords", (1, 1, 2, 2, 2), mode="z_order")

    def test_decode_delegates(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: True)
        fake = MagicMock(return_value="coords")
        monkeypatch.setattr(flexgemm_ops_module, "_decode_seq", fake)

        result = flexgemm_ops_module.decode_seq("codes", (1, 1, 2, 2, 2), mode="hilbert")

        assert result == "coords"
        fake.assert_called_once_with("codes", (1, 1, 2, 2, 2), mode="hilbert")

    def test_value_error_returns_none(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: True)
        monkeypatch.setattr(flexgemm_ops_module, "_encode_seq",
                             MagicMock(side_effect=ValueError("too large")))
        assert flexgemm_ops_module.encode_seq("coords", (1, 1, 2, 2, 2)) is None


# ---------------------------------------------------------------------------
# On-the-fly dense/sparse conv3d switching.
# ---------------------------------------------------------------------------


class TestSparseConv3dEnabledFlag:
    def teardown_method(self):
        os.environ.pop("AMD_TUNED_TORCH_SPARSE_CONV3D", None)
        importlib.reload(flexgemm_ops_module)

    def test_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("AMD_TUNED_TORCH_SPARSE_CONV3D", raising=False)
        importlib.reload(flexgemm_ops_module)
        assert flexgemm_ops_module.sparse_conv3d_enabled() is True

    def test_disabled_when_explicitly_zero(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SPARSE_CONV3D", "0")
        importlib.reload(flexgemm_ops_module)
        assert flexgemm_ops_module.sparse_conv3d_enabled() is False

    def test_disabled_when_false_string(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SPARSE_CONV3D", "false")
        importlib.reload(flexgemm_ops_module)
        assert flexgemm_ops_module.sparse_conv3d_enabled() is False


class TestOccupancy:
    def test_fully_dense_is_one(self):
        x = torch.ones(1, 2, 3, 3, 3)
        assert flexgemm_ops_module._occupancy(x) == 1.0

    def test_fully_empty_is_zero(self):
        x = torch.zeros(1, 2, 3, 3, 3)
        assert flexgemm_ops_module._occupancy(x) == 0.0

    def test_partial_occupancy(self):
        x = torch.zeros(1, 1, 2, 2, 2)
        x[0, 0, 0, 0, 0] = 1.0  # 1 of 8 spatial positions occupied
        assert flexgemm_ops_module._occupancy(x) == 0.125

    def test_occupancy2d_and_1d_agree_with_occupancy3d_shape(self):
        x2 = torch.zeros(1, 1, 2, 2)
        x2[0, 0, 0, 0] = 1.0
        assert flexgemm_ops_module._occupancy2d(x2) == 0.25
        x1 = torch.zeros(1, 1, 4)
        x1[0, 0, 0] = 1.0
        assert flexgemm_ops_module._occupancy1d(x1) == 0.25


class TestEstimateOccupancy:
    """(b) bounded sampling -- small tensors (below the sample-size cap)
    get an exact answer via the full-scan fallback; large tensors get a
    statistical estimate from a fixed-size random sample instead of a full
    scan."""

    def test_small_tensor_is_exact(self):
        x = torch.zeros(1, 1, 2, 2, 2)
        x[0, 0, 0, 0, 0] = 1.0
        assert flexgemm_ops_module._estimate_occupancy(x) == 0.125

    def test_large_tensor_uses_bounded_sample_not_full_scan(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "_OCCUPANCY_SAMPLE_SIZE", 16)
        torch.manual_seed(0)
        x = torch.zeros(1, 1, 64, 64)  # 4096 spatial positions >> sample size 16
        x[0, 0, 0, 0] = 1.0

        seen_sizes = []
        real_randint = torch.randint

        def spy_randint(*args, **kwargs):
            result = real_randint(*args, **kwargs)
            seen_sizes.append(result.numel())
            return result

        monkeypatch.setattr(torch, "randint", spy_randint)
        flexgemm_ops_module._estimate_occupancy(x)

        assert seen_sizes == [16]  # sampled exactly the capped size, not all 4096 positions

    def test_large_all_dense_tensor_estimates_close_to_one(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "_OCCUPANCY_SAMPLE_SIZE", 1000)
        torch.manual_seed(1)
        x = torch.ones(1, 1, 64, 64)
        assert flexgemm_ops_module._estimate_occupancy(x) == 1.0  # every position occupied -> exact regardless of sampling


class TestOccupancyCache:
    """(a) identity+version caching in front of _estimate_occupancy."""

    def test_repeated_call_on_same_tensor_hits_cache(self, monkeypatch):
        x = torch.zeros(1, 1, 2, 2, 2)
        x[0, 0, 0, 0, 0] = 1.0
        spy = MagicMock(wraps=flexgemm_ops_module._estimate_occupancy)
        monkeypatch.setattr(flexgemm_ops_module, "_estimate_occupancy", spy)

        first = flexgemm_ops_module._cached_occupancy(x)
        second = flexgemm_ops_module._cached_occupancy(x)

        assert first == second == 0.125
        spy.assert_called_once()

    def test_in_place_mutation_invalidates_cache(self, monkeypatch):
        x = torch.zeros(1, 1, 2, 2, 2)
        assert flexgemm_ops_module._cached_occupancy(x) == 0.0
        x[0, 0, 0, 0, 0] = 1.0  # in-place write bumps x._version
        assert flexgemm_ops_module._cached_occupancy(x) == 0.125

    def test_different_tensor_objects_are_cached_independently(self):
        a = torch.zeros(1, 1, 2, 2, 2)
        b = torch.ones(1, 1, 2, 2, 2)
        assert flexgemm_ops_module._cached_occupancy(a) == 0.0
        assert flexgemm_ops_module._cached_occupancy(b) == 1.0

    def test_different_eps_for_same_tensor_recomputes(self, monkeypatch):
        x = torch.full((1, 1, 2, 2, 2), 1e-10)
        spy = MagicMock(wraps=flexgemm_ops_module._estimate_occupancy)
        monkeypatch.setattr(flexgemm_ops_module, "_estimate_occupancy", spy)

        low_eps = flexgemm_ops_module._cached_occupancy(x, eps=1e-12)
        high_eps = flexgemm_ops_module._cached_occupancy(x, eps=1e-9)

        assert low_eps == 1.0   # 1e-10 > 1e-12 -> counted as occupied
        assert high_eps == 0.0  # 1e-10 < 1e-9 -> counted as empty
        assert spy.call_count == 2  # different eps, not served from the eps=1e-12 entry

    def test_cache_entry_evicted_when_tensor_is_garbage_collected(self):
        cache = flexgemm_ops_module._occupancy_cache
        x = torch.zeros(1, 1, 2, 2, 2)
        flexgemm_ops_module._cached_occupancy(x)
        key = id(x)
        assert key in cache._data
        del x
        import gc
        gc.collect()
        assert key not in cache._data

    def test_concurrent_get_set_does_not_corrupt_or_raise(self):
        """Regression test for the missing lock: many threads hammering
        get()/set() on a shared set of tensors (plus a background thread
        churning throwaway tensors to trigger weakref-finalizer evictions
        concurrently with the others' get()/set() calls) must never raise
        (a torn dict read/write) and every read must come back internally
        consistent (never a value that doesn't match _estimate_occupancy
        for that exact tensor)."""
        import gc
        import concurrent.futures

        tensors = []
        for i in range(8):
            t = torch.zeros(1, 1, 4, 4, 4)
            if i % 2 == 0:
                t[0, 0, 0, 0, 0] = 1.0
            tensors.append(t)
        expected = [flexgemm_ops_module._estimate_occupancy(t) for t in tensors]

        errors = []

        def hammer(_n):
            try:
                for _ in range(200):
                    for t, exp in zip(tensors, expected):
                        got = flexgemm_ops_module._cached_occupancy(t)
                        if got != exp:
                            errors.append((exp, got))
            except Exception as e:  # noqa: BLE001 -- any exception is a failure here
                errors.append(e)

        def churn(_n):
            try:
                for _ in range(200):
                    throwaway = torch.zeros(1, 1, 2, 2, 2)
                    flexgemm_ops_module._cached_occupancy(throwaway)
                    del throwaway
                    gc.collect()
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(hammer, i) for i in range(6)]
            futures += [pool.submit(churn, i) for i in range(2)]
            for f in futures:
                f.result()

        assert errors == []

    def test_maybe_sparse_conv3d_uses_cached_occupancy(self, monkeypatch):
        """End-to-end: maybe_sparse_conv3d goes through _occupancy, which
        goes through the cache -- a second call with the same input tensor
        object shouldn't re-estimate occupancy from scratch."""
        # available()=True: maybe_sparse_conv3d only reaches the occupancy
        # check at all once its own available() gate passes (unlike
        # maybe_sparse_conv2d/1d, which don't gate on it before checking
        # occupancy) -- a fully-dense input still declines afterward via
        # the occupancy threshold, so sparse_conv3d_from_dense is never
        # actually invoked either way.
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv3d_enabled", lambda: True)
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: True)
        spy = MagicMock(wraps=flexgemm_ops_module._estimate_occupancy)
        monkeypatch.setattr(flexgemm_ops_module, "_estimate_occupancy", spy)

        x = torch.ones(1, 2, 4, 4, 4)  # fully dense -> both calls decline
        weight = torch.randn(3, 2, 3, 3, 3)

        # min_positions=1: bypass the minimum-size gate (this tensor is
        # far below the 1024 default) so both calls actually reach the
        # occupancy check this test is about.
        flexgemm_ops_module.maybe_sparse_conv3d(x, weight, min_positions=1)
        flexgemm_ops_module.maybe_sparse_conv3d(x, weight, min_positions=1)

        spy.assert_called_once()


class TestDenseToSparse:
    def test_extracts_only_occupied_positions(self):
        x = torch.zeros(1, 2, 2, 2, 2)
        x[0, :, 0, 1, 0] = torch.tensor([5.0, 6.0])
        x[0, :, 1, 1, 1] = torch.tensor([7.0, 8.0])

        feats, coords = flexgemm_ops_module._dense_to_sparse_bdhwc(x)

        assert feats.shape == (2, 2)
        assert coords.shape == (2, 4)
        assert coords.dtype == torch.int32
        rows = {tuple(c.tolist()): tuple(f.tolist()) for c, f in zip(coords, feats)}
        assert rows[(0, 0, 1, 0)] == (5.0, 6.0)
        assert rows[(0, 1, 1, 1)] == (7.0, 8.0)

    def test_empty_input_yields_no_points(self):
        x = torch.zeros(1, 3, 2, 2, 2)
        feats, coords = flexgemm_ops_module._dense_to_sparse_bdhwc(x)
        assert feats.shape[0] == 0
        assert coords.shape[0] == 0


class TestConv3dOutputSize:
    def test_matches_pytorch_formula(self):
        # stride=1, padding=1, dilation=1, kernel=3 -> same spatial size.
        assert flexgemm_ops_module._conv3d_output_size(8, 3, 1, 1, 1) == 8
        # stride=2, no padding, kernel=2 -> halves.
        assert flexgemm_ops_module._conv3d_output_size(8, 2, 2, 0, 1) == 4


def _touched_output_mask(occ_mask: torch.Tensor, kernel_size, stride, padding, dilation) -> torch.Tensor:
    """[B,D,H,W] bool occupancy -> [B,D',H',W'] bool: True at every output
    position whose receptive field overlaps at least one occupied input
    voxel -- i.e. exactly the positions a real sparse convolution would
    return coordinates for. Computed with plain dense F.conv3d over the
    occupancy mask as a 1-channel volume against an all-ones kernel, purely
    as test scaffolding (no flex_gemm/GPU involved)."""
    kd, kh, kw = kernel_size
    ones_kernel = torch.ones(1, 1, kd, kh, kw)
    overlap = F.conv3d(occ_mask.float().unsqueeze(1), ones_kernel,
                        stride=stride, padding=padding, dilation=dilation)
    return overlap.squeeze(1) > 0


class TestSparseConv3dFromDense:
    """Validates the dense<->sparse reconstruction (mask extraction, scatter,
    bias-fill for untouched positions) against a real F.conv3d reference --
    _sparse_conv3d itself is mocked to return values gathered from that same
    reference at the positions a real sparse kernel would touch, so these
    tests catch bugs in sparse_conv3d_from_dense's own glue code without
    needing the actual flex_gemm/HIP kernel."""

    def _mock_sparse_conv3d_from_reference(self, monkeypatch, reference: torch.Tensor,
                                            occ_mask: torch.Tensor, kernel_size,
                                            stride, padding, dilation):
        touched = _touched_output_mask(occ_mask, kernel_size, stride, padding, dilation)
        ref_bdhwc = reference.permute(0, 2, 3, 4, 1)  # [B,D',H',W',Co]

        def _fake_sparse_conv3d(feats, coords, shape, weight, bias, stride, padding, dilation):
            out_coords = touched.nonzero(as_tuple=False).to(torch.int32)
            out_feats = ref_bdhwc[touched]
            return out_feats, out_coords, None

        monkeypatch.setattr(flexgemm_ops_module, "_sparse_conv3d", _fake_sparse_conv3d)
        return touched

    def test_matches_dense_conv3d_including_untouched_positions(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: True)
        torch.manual_seed(0)
        b, c_in, c_out = 1, 2, 3
        d = h = w = 6
        kernel_size = (3, 3, 3)
        stride, padding, dilation = (1, 1, 1), (1, 1, 1), (1, 1, 1)

        weight = torch.randn(c_out, c_in, *kernel_size)
        bias = torch.randn(c_out)

        x = torch.zeros(b, c_in, d, h, w)
        # Sparse: only a handful of occupied voxels.
        for (dd, hh, ww) in [(1, 1, 1), (4, 2, 3), (0, 5, 5)]:
            x[0, :, dd, hh, ww] = torch.randn(c_in)
        occ_mask = (x.abs().amax(dim=1) > 1e-12)

        reference = F.conv3d(x, weight, bias, stride=stride, padding=padding, dilation=dilation)
        self._mock_sparse_conv3d_from_reference(
            monkeypatch, reference, occ_mask, kernel_size, stride, padding, dilation)

        result = flexgemm_ops_module.sparse_conv3d_from_dense(
            x, weight, bias, stride=stride, padding=padding, dilation=dilation)

        assert result is not None
        assert result.shape == reference.shape
        torch.testing.assert_close(result, reference, atol=1e-5, rtol=1e-5)

    def test_fully_empty_input_returns_pure_bias(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: True)
        c_in, c_out = 2, 3
        weight = torch.randn(c_out, c_in, 3, 3, 3)
        bias = torch.randn(c_out)
        x = torch.zeros(1, c_in, 5, 5, 5)

        called = MagicMock()
        monkeypatch.setattr(flexgemm_ops_module, "_sparse_conv3d", called)

        reference = F.conv3d(x, weight, bias, stride=1, padding=1, dilation=1)
        result = flexgemm_ops_module.sparse_conv3d_from_dense(
            x, weight, bias, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1))

        torch.testing.assert_close(result, reference, atol=1e-6, rtol=1e-6)
        called.assert_not_called()  # skipped the kernel call entirely -- see docstring.

    def test_unavailable_returns_none(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: False)
        x = torch.zeros(1, 2, 4, 4, 4)
        weight = torch.randn(3, 2, 3, 3, 3)
        assert flexgemm_ops_module.sparse_conv3d_from_dense(x, weight) is None

    def test_wrong_dims_returns_none(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: True)
        assert flexgemm_ops_module.sparse_conv3d_from_dense(
            torch.zeros(2, 2), torch.zeros(3, 2, 3, 3, 3)) is None

    def test_kernel_runtime_error_returns_none(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: True)
        monkeypatch.setattr(flexgemm_ops_module, "_sparse_conv3d",
                             MagicMock(side_effect=RuntimeError("bad shape")))
        x = torch.zeros(1, 2, 4, 4, 4)
        x[0, :, 0, 0, 0] = 1.0
        weight = torch.randn(3, 2, 3, 3, 3)
        assert flexgemm_ops_module.sparse_conv3d_from_dense(x, weight) is None


class TestMaybeSparseConv3d:
    def test_declines_when_disabled(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv3d_enabled", lambda: False)
        x = torch.zeros(1, 2, 4, 4, 4)
        weight = torch.randn(3, 2, 3, 3, 3)
        assert flexgemm_ops_module.maybe_sparse_conv3d(x, weight) is None

    def test_declines_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv3d_enabled", lambda: True)
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: False)
        x = torch.zeros(1, 2, 4, 4, 4)
        weight = torch.randn(3, 2, 3, 3, 3)
        assert flexgemm_ops_module.maybe_sparse_conv3d(x, weight) is None

    def test_declines_above_occupancy_threshold(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv3d_enabled", lambda: True)
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: True)
        fake_from_dense = MagicMock(return_value="sparse_result")
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv3d_from_dense", fake_from_dense)

        x = torch.ones(1, 2, 4, 4, 4)  # fully dense -> occupancy 1.0
        weight = torch.randn(3, 2, 3, 3, 3)

        # min_positions=1: bypass the minimum-size gate (this tensor is far
        # below the 1024 default) -- this test is about the occupancy
        # threshold specifically, not the size gate (see TestMinSpatialPositionsGate).
        result = flexgemm_ops_module.maybe_sparse_conv3d(
            x, weight, max_occupancy=0.3, min_positions=1)

        assert result is None
        fake_from_dense.assert_not_called()

    def test_routes_to_sparse_below_threshold(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv3d_enabled", lambda: True)
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: True)
        fake_from_dense = MagicMock(return_value="sparse_result")
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv3d_from_dense", fake_from_dense)

        x = torch.zeros(1, 2, 4, 4, 4)
        x[0, :, 0, 0, 0] = 1.0  # 1/64 occupancy
        weight = torch.randn(3, 2, 3, 3, 3)
        bias = torch.randn(3)

        result = flexgemm_ops_module.maybe_sparse_conv3d(
            x, weight, bias, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1),
            max_occupancy=0.3, min_positions=1)

        assert result == "sparse_result"
        fake_from_dense.assert_called_once_with(
            x, weight, bias, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1))

    def test_wrong_dims_returns_none(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv3d_enabled", lambda: True)
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: True)
        assert flexgemm_ops_module.maybe_sparse_conv3d(torch.zeros(2, 2), torch.zeros(3, 2)) is None


# ---------------------------------------------------------------------------
# 2D sparse convolution -- pure PyTorch, no mocking needed anywhere below:
# sparse_conv2d/sparse_submanifold_conv2d are real, self-contained
# implementations (see flexgemm_ops.py's module docstring for the
# source/sparse_convolution + source/spconv template), so these tests check
# them directly against real F.conv2d.
# ---------------------------------------------------------------------------


def _dense_to_sparse2d(x: torch.Tensor, eps: float = 1e-12):
    """[B,C,H,W] -> (feats [N,C], coords [N,3] as (b,h,w)) -- test helper
    mirroring flexgemm_ops.sparse_conv2d_from_dense's own extraction."""
    x_bhwc = x.permute(0, 2, 3, 1)
    mask = x_bhwc.abs().amax(dim=-1) > eps
    coords = mask.nonzero(as_tuple=False).to(torch.int32)
    feats = x_bhwc[mask]
    return feats, coords


def _touched_output_mask2d(occ_mask: torch.Tensor, kernel_size, stride, padding, dilation) -> torch.Tensor:
    """[B,H,W] bool occupancy -> [B,H',W'] bool: True at every output
    position whose receptive field overlaps at least one occupied input
    pixel -- exactly the coordinates sparse_conv2d should return. Computed
    via plain dense F.conv2d over the mask against an all-ones kernel, test
    scaffolding only."""
    kh, kw = kernel_size
    ones_kernel = torch.ones(1, 1, kh, kw)
    overlap = F.conv2d(occ_mask.float().unsqueeze(1), ones_kernel,
                        stride=stride, padding=padding, dilation=dilation)
    return overlap.squeeze(1) > 0


class TestCoordsToKeys:
    def test_distinct_coords_get_distinct_keys(self):
        coords = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.int32)
        keys = flexgemm_ops_module._coords_to_keys(coords, (4, 4))
        assert keys.unique().numel() == coords.shape[0]

    def test_matches_manual_flat_index(self):
        coords = torch.tensor([[2, 3, 1]], dtype=torch.int32)
        h, w = 5, 6
        expected = (2 * h + 3) * w + 1
        assert flexgemm_ops_module._coords_to_keys(coords, (h, w)).item() == expected


class TestCoordsIndex:
    def test_lookup_finds_exact_matches(self):
        coords = torch.tensor([[0, 1, 2], [0, 3, 0], [1, 0, 0]], dtype=torch.int32)
        index = flexgemm_ops_module._CoordsIndex(coords, (4, 4))

        result = index.lookup(
            torch.tensor([0, 0, 1]), torch.tensor([1, 3, 0]), torch.tensor([2, 0, 0]))

        # Each query matches a distinct row of `coords`; check by re-deriving
        # feats identity via the returned row index rather than assuming order.
        coords_l = coords.long()
        for q in range(3):
            row = result[q].item()
            assert torch.equal(coords_l[row], torch.tensor(
                [[0, 1, 2], [0, 3, 0], [1, 0, 0]])[q])

    def test_lookup_returns_minus_one_for_absent_coords(self):
        coords = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        index = flexgemm_ops_module._CoordsIndex(coords, (4, 4))

        result = index.lookup(torch.tensor([0]), torch.tensor([3]), torch.tensor([3]))

        assert result.item() == -1

    def test_does_not_allocate_a_dense_grid(self):
        """The whole point of this class: no tensor sized to the spatial
        volume is ever created, even for a spatial extent far too large to
        materialize as a dense [B,H,W] grid (10000x10000 = 1e8 int64s,
        800MB -- this test would OOM or take a very long time if
        _CoordsIndex still built one)."""
        coords = torch.tensor([[0, 5000, 5000], [0, 1, 1], [0, 9999, 0]], dtype=torch.int32)
        index = flexgemm_ops_module._CoordsIndex(coords, (10000, 10000))
        assert index.sorted_keys.numel() == 3  # O(N), not O(H*W)

        result = index.lookup(torch.tensor([0, 0]), torch.tensor([1, 0]), torch.tensor([1, 0]))
        assert result[0].item() == 1  # (0,1,1) is coords row 1
        assert result[1].item() == -1  # (0,0,0) was never occupied


class TestSparseSubmanifoldConv2d:
    def test_matches_dense_conv2d_at_occupied_positions(self):
        torch.manual_seed(0)
        b, c_in, c_out, h, w = 1, 3, 4, 8, 8
        weight = torch.randn(c_out, c_in, 3, 3)
        bias = torch.randn(c_out)

        x = torch.zeros(b, c_in, h, w)
        for (hh, ww) in [(0, 0), (3, 5), (7, 7), (2, 2)]:
            x[0, :, hh, ww] = torch.randn(c_in)

        reference = F.conv2d(x, weight, bias, stride=1, padding=1)
        feats, coords = _dense_to_sparse2d(x)

        out = flexgemm_ops_module.sparse_submanifold_conv2d(
            feats, coords, torch.Size((b, c_in, h, w)), weight, bias=bias)

        ref_bhwc = reference.permute(0, 2, 3, 1)
        for row, (bb, hh, ww) in enumerate(coords.tolist()):
            torch.testing.assert_close(out[row], ref_bhwc[bb, hh, ww], atol=1e-5, rtol=1e-5)

    def test_even_kernel_rejected(self):
        feats = torch.randn(2, 3)
        coords = torch.tensor([[0, 0, 0], [0, 1, 1]], dtype=torch.int32)
        weight = torch.randn(4, 3, 2, 2)
        with pytest.raises(ValueError):
            flexgemm_ops_module.sparse_submanifold_conv2d(feats, coords, torch.Size((1, 3, 4, 4)), weight)

    def test_empty_input_returns_no_rows(self):
        feats = torch.zeros(0, 3)
        coords = torch.zeros(0, 3, dtype=torch.int32)
        weight = torch.randn(4, 3, 3, 3)
        out = flexgemm_ops_module.sparse_submanifold_conv2d(feats, coords, torch.Size((1, 3, 4, 4)), weight)
        assert out.shape == (0, 4)


class TestSparseConv2d:
    def test_matches_dense_conv2d_stride2(self):
        torch.manual_seed(1)
        b, c_in, c_out, h, w = 1, 2, 3, 8, 8
        weight = torch.randn(c_out, c_in, 3, 3)
        bias = torch.randn(c_out)
        stride, padding, dilation = (2, 2), (1, 1), (1, 1)

        x = torch.zeros(b, c_in, h, w)
        for (hh, ww) in [(0, 0), (3, 5), (7, 7), (2, 2), (5, 1)]:
            x[0, :, hh, ww] = torch.randn(c_in)
        occ_mask = x.abs().amax(dim=1) > 1e-12

        reference = F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation)
        feats, coords = _dense_to_sparse2d(x)

        out_feats, out_coords = flexgemm_ops_module.sparse_conv2d(
            feats, coords, torch.Size((b, c_in, h, w)), weight, bias=bias,
            stride=stride, padding=padding, dilation=dilation)

        expected_touched = _touched_output_mask2d(occ_mask, (3, 3), stride, padding, dilation)
        assert out_coords.shape[0] == int(expected_touched.sum())

        ref_bhwc = reference.permute(0, 2, 3, 1)
        for row, (bb, hh, ww) in enumerate(out_coords.tolist()):
            assert expected_touched[bb, hh, ww]
            torch.testing.assert_close(out_feats[row], ref_bhwc[bb, hh, ww], atol=1e-5, rtol=1e-5)

    def test_matches_dense_conv2d_with_dilation(self):
        torch.manual_seed(2)
        b, c_in, c_out, h, w = 1, 2, 2, 10, 10
        weight = torch.randn(c_out, c_in, 3, 3)
        stride, padding, dilation = (1, 1), (2, 2), (2, 2)

        x = torch.zeros(b, c_in, h, w)
        x[0, :, 4, 4] = torch.randn(c_in)
        x[0, :, 1, 8] = torch.randn(c_in)

        reference = F.conv2d(x, weight, None, stride=stride, padding=padding, dilation=dilation)
        feats, coords = _dense_to_sparse2d(x)

        out_feats, out_coords = flexgemm_ops_module.sparse_conv2d(
            feats, coords, torch.Size((b, c_in, h, w)), weight,
            stride=stride, padding=padding, dilation=dilation)

        ref_bhwc = reference.permute(0, 2, 3, 1)
        for row, (bb, hh, ww) in enumerate(out_coords.tolist()):
            torch.testing.assert_close(out_feats[row], ref_bhwc[bb, hh, ww], atol=1e-5, rtol=1e-5)

    def test_empty_input_returns_empty(self):
        feats = torch.zeros(0, 3)
        coords = torch.zeros(0, 3, dtype=torch.int32)
        weight = torch.randn(4, 3, 3, 3)
        out_feats, out_coords = flexgemm_ops_module.sparse_conv2d(
            feats, coords, torch.Size((1, 3, 4, 4)), weight)
        assert out_feats.shape == (0, 4)
        assert out_coords.shape == (0, 3)


def _fake_sparse_submanifold_conv3d_via_2d(feats, coords4, shape5, weight5, bias, dilation):
    """Stand-in for the real (uninstalled here) `_sparse_submanifold_conv3d`
    HIP kernel: unlifts its depth=1-lifted arguments back to 2D and
    delegates to the already-validated (against real F.conv2d)
    sparse_submanifold_conv2d. Used to test sparse_submanifold_conv2d_native's
    own lift/unlift arithmetic (coord padding, weight permute, dilation
    triple) without needing the actual compiled kernel -- any bug in that
    arithmetic shows up as a shape mismatch or wrong numbers here, since
    this fake's own math is a real, independently-checked 2D convolution."""
    coords3 = coords4[:, :3]
    b, c_in, h, w, _d = shape5
    weight2 = weight5[:, :, :, 0, :].permute(0, 3, 1, 2).contiguous()
    out_feats = flexgemm_ops_module.sparse_submanifold_conv2d(
        feats, coords3, torch.Size((b, c_in, h, w)), weight2,
        bias=bias, dilation=(dilation[0], dilation[1]))
    return out_feats, None


def _fake_sparse_conv3d_via_2d(feats, coords4, shape5, weight5, bias, stride, padding, dilation):
    """Same idea as _fake_sparse_submanifold_conv3d_via_2d, for
    sparse_conv2d_native / the general (non-submanifold) case."""
    coords3 = coords4[:, :3]
    b, c_in, h, w, _d = shape5
    weight2 = weight5[:, :, :, 0, :].permute(0, 3, 1, 2).contiguous()
    out_feats, out_coords3 = flexgemm_ops_module.sparse_conv2d(
        feats, coords3, torch.Size((b, c_in, h, w)), weight2, bias=bias,
        stride=(stride[0], stride[1]), padding=(padding[0], padding[1]),
        dilation=(dilation[0], dilation[1]))
    zeros = torch.zeros(out_coords3.shape[0], 1, dtype=out_coords3.dtype, device=out_coords3.device)
    out_coords4 = torch.cat([out_coords3, zeros], dim=1)
    return out_feats, out_coords4, None


class TestSparseSubmanifoldConv2dNative:
    def test_unavailable_returns_none(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: False)
        feats = torch.randn(2, 3)
        coords = torch.tensor([[0, 0, 0], [0, 1, 1]], dtype=torch.int32)
        weight = torch.randn(4, 3, 3, 3)
        assert flexgemm_ops_module.sparse_submanifold_conv2d_native(
            feats, coords, torch.Size((1, 3, 4, 4)), weight) is None

    def test_even_kernel_returns_none(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: True)
        feats = torch.randn(2, 3)
        coords = torch.tensor([[0, 0, 0], [0, 1, 1]], dtype=torch.int32)
        weight = torch.randn(4, 3, 2, 2)
        assert flexgemm_ops_module.sparse_submanifold_conv2d_native(
            feats, coords, torch.Size((1, 3, 4, 4)), weight) is None

    def test_matches_dense_conv2d_via_lift(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: True)
        monkeypatch.setattr(flexgemm_ops_module, "_sparse_submanifold_conv3d",
                             _fake_sparse_submanifold_conv3d_via_2d)
        torch.manual_seed(10)
        b, c_in, c_out, h, w = 1, 3, 4, 8, 8
        weight = torch.randn(c_out, c_in, 3, 3)
        bias = torch.randn(c_out)
        x = torch.zeros(b, c_in, h, w)
        for (hh, ww) in [(0, 0), (3, 5), (7, 7)]:
            x[0, :, hh, ww] = torch.randn(c_in)

        reference = F.conv2d(x, weight, bias, stride=1, padding=1)
        feats, coords = _dense_to_sparse2d(x)

        out = flexgemm_ops_module.sparse_submanifold_conv2d_native(
            feats, coords, torch.Size((b, c_in, h, w)), weight, bias=bias)

        assert out is not None
        ref_bhwc = reference.permute(0, 2, 3, 1)
        for row, (bb, hh, ww) in enumerate(coords.tolist()):
            torch.testing.assert_close(out[row], ref_bhwc[bb, hh, ww], atol=1e-5, rtol=1e-5)


class TestSparseConv2dNative:
    def test_unavailable_returns_none(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: False)
        feats = torch.randn(2, 3)
        coords = torch.tensor([[0, 0, 0], [0, 1, 1]], dtype=torch.int32)
        weight = torch.randn(4, 3, 3, 3)
        assert flexgemm_ops_module.sparse_conv2d_native(
            feats, coords, torch.Size((1, 3, 4, 4)), weight) is None

    def test_matches_dense_conv2d_via_lift(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: True)
        monkeypatch.setattr(flexgemm_ops_module, "_sparse_conv3d", _fake_sparse_conv3d_via_2d)
        torch.manual_seed(11)
        b, c_in, c_out, h, w = 1, 2, 3, 8, 8
        weight = torch.randn(c_out, c_in, 3, 3)
        bias = torch.randn(c_out)
        stride, padding, dilation = (2, 2), (1, 1), (1, 1)
        x = torch.zeros(b, c_in, h, w)
        for (hh, ww) in [(0, 0), (3, 5), (7, 7), (5, 1)]:
            x[0, :, hh, ww] = torch.randn(c_in)

        reference = F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation)
        feats, coords = _dense_to_sparse2d(x)

        result = flexgemm_ops_module.sparse_conv2d_native(
            feats, coords, torch.Size((b, c_in, h, w)), weight, bias=bias,
            stride=stride, padding=padding, dilation=dilation)

        assert result is not None
        out_feats, out_coords = result
        ref_bhwc = reference.permute(0, 2, 3, 1)
        for row, (bb, hh, ww) in enumerate(out_coords.tolist()):
            torch.testing.assert_close(out_feats[row], ref_bhwc[bb, hh, ww], atol=1e-5, rtol=1e-5)

    def test_kernel_runtime_error_returns_none(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: True)
        monkeypatch.setattr(flexgemm_ops_module, "_sparse_conv3d",
                             MagicMock(side_effect=RuntimeError("bad shape")))
        feats = torch.randn(2, 3)
        coords = torch.tensor([[0, 0, 0], [0, 1, 1]], dtype=torch.int32)
        weight = torch.randn(4, 3, 3, 3)
        assert flexgemm_ops_module.sparse_conv2d_native(
            feats, coords, torch.Size((1, 3, 4, 4)), weight) is None


class TestSparseConv2dFromDensePrefersNative:
    def test_uses_native_result_when_available(self, monkeypatch):
        native_result = (torch.randn(2, 4), torch.tensor([[0, 0, 0], [0, 1, 1]], dtype=torch.int32))
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv2d_native",
                             MagicMock(return_value=native_result))
        fake_pure_python = MagicMock()
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv2d", fake_pure_python)

        x = torch.zeros(1, 3, 4, 4)
        x[0, :, 0, 0] = 1.0
        weight = torch.randn(4, 3, 3, 3)
        result = flexgemm_ops_module.sparse_conv2d_from_dense(x, weight, stride=(1, 1), padding=(1, 1))

        assert result is not None
        fake_pure_python.assert_not_called()

    def test_falls_back_to_pure_python_when_native_declines(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv2d_native", MagicMock(return_value=None))
        x = torch.zeros(1, 3, 4, 4)
        x[0, :, 0, 0] = 1.0
        weight = torch.randn(4, 3, 3, 3)
        bias = torch.randn(4)

        reference = F.conv2d(x, weight, bias, stride=1, padding=1)
        result = flexgemm_ops_module.sparse_conv2d_from_dense(x, weight, bias, stride=(1, 1), padding=(1, 1))

        torch.testing.assert_close(result, reference, atol=1e-5, rtol=1e-5)


class TestSparseConv2dFromDense:
    def test_matches_dense_conv2d_including_untouched_positions(self):
        torch.manual_seed(3)
        b, c_in, c_out, h, w = 1, 2, 3, 8, 8
        weight = torch.randn(c_out, c_in, 3, 3)
        bias = torch.randn(c_out)

        x = torch.zeros(b, c_in, h, w)
        for (hh, ww) in [(1, 1), (4, 2), (0, 5)]:
            x[0, :, hh, ww] = torch.randn(c_in)

        reference = F.conv2d(x, weight, bias, stride=1, padding=1)
        result = flexgemm_ops_module.sparse_conv2d_from_dense(
            x, weight, bias, stride=(1, 1), padding=(1, 1), dilation=(1, 1))

        assert result is not None
        torch.testing.assert_close(result, reference, atol=1e-5, rtol=1e-5)

    def test_fully_empty_input_returns_pure_bias(self):
        c_in, c_out = 2, 3
        weight = torch.randn(c_out, c_in, 3, 3)
        bias = torch.randn(c_out)
        x = torch.zeros(1, c_in, 5, 5)

        reference = F.conv2d(x, weight, bias, stride=1, padding=1)
        result = flexgemm_ops_module.sparse_conv2d_from_dense(
            x, weight, bias, stride=(1, 1), padding=(1, 1), dilation=(1, 1))

        torch.testing.assert_close(result, reference, atol=1e-6, rtol=1e-6)

    def test_wrong_dims_returns_none(self):
        assert flexgemm_ops_module.sparse_conv2d_from_dense(torch.zeros(2, 2), torch.zeros(3, 2)) is None


class TestSparseConv2dEnabledFlag:
    def teardown_method(self):
        os.environ.pop("AMD_TUNED_TORCH_SPARSE_CONV2D", None)
        importlib.reload(flexgemm_ops_module)

    def test_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("AMD_TUNED_TORCH_SPARSE_CONV2D", raising=False)
        importlib.reload(flexgemm_ops_module)
        assert flexgemm_ops_module.sparse_conv2d_enabled() is True

    def test_disabled_when_explicitly_zero(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SPARSE_CONV2D", "0")
        importlib.reload(flexgemm_ops_module)
        assert flexgemm_ops_module.sparse_conv2d_enabled() is False


class TestMaybeSparseConv2d:
    def test_declines_when_disabled(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv2d_enabled", lambda: False)
        x = torch.zeros(1, 2, 4, 4)
        weight = torch.randn(3, 2, 3, 3)
        assert flexgemm_ops_module.maybe_sparse_conv2d(x, weight) is None

    def test_declines_above_occupancy_threshold(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv2d_enabled", lambda: True)
        fake_from_dense = MagicMock(return_value="sparse_result")
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv2d_from_dense", fake_from_dense)

        x = torch.ones(1, 2, 4, 4)  # fully dense
        weight = torch.randn(3, 2, 3, 3)

        # min_positions=1: bypass the minimum-size gate -- see the 3D
        # version of this test for why.
        result = flexgemm_ops_module.maybe_sparse_conv2d(
            x, weight, max_occupancy=0.3, min_positions=1)

        assert result is None
        fake_from_dense.assert_not_called()

    def test_routes_to_sparse_below_threshold(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv2d_enabled", lambda: True)
        # The path is gated on the native kernel now -- without it the
        # pure-Python fallback is 21-63x slower than stock (measured; see
        # maybe_sparse_conv2d's docstring), so routing is what needs the
        # precondition stated, not declining.
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: True)
        fake_from_dense = MagicMock(return_value="sparse_result")
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv2d_from_dense", fake_from_dense)

        x = torch.zeros(1, 2, 4, 4)
        x[0, :, 0, 0] = 1.0  # 1/16 occupancy
        weight = torch.randn(3, 2, 3, 3)
        bias = torch.randn(3)

        result = flexgemm_ops_module.maybe_sparse_conv2d(
            x, weight, bias, stride=(1, 1), padding=(1, 1), dilation=(1, 1),
            max_occupancy=0.3, min_positions=1)

        assert result == "sparse_result"
        fake_from_dense.assert_called_once_with(
            x, weight, bias, stride=(1, 1), padding=(1, 1), dilation=(1, 1))

    def test_wrong_dims_returns_none(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv2d_enabled", lambda: True)
        assert flexgemm_ops_module.maybe_sparse_conv2d(torch.zeros(2, 2), torch.zeros(3, 2)) is None


# ---------------------------------------------------------------------------
# 1D sparse convolution -- every function here is a thin lift/unlift
# adapter around its 2D counterpart (see flexgemm_ops.py's module comment
# above sparse_submanifold_conv1d), so these tests check real F.conv1d
# directly rather than re-deriving a separate ground truth.
# ---------------------------------------------------------------------------


def _dense_to_sparse1d(x: torch.Tensor, eps: float = 1e-12):
    """[B,C,L] -> (feats [N,C], coords [N,2] as (b,x))."""
    x_blc = x.permute(0, 2, 1)
    mask = x_blc.abs().amax(dim=-1) > eps
    coords = mask.nonzero(as_tuple=False).to(torch.int32)
    feats = x_blc[mask]
    return feats, coords


def _touched_output_mask1d(occ_mask: torch.Tensor, kernel_size, stride, padding, dilation) -> torch.Tensor:
    """[B,L] bool occupancy -> [B,L'] bool touched-output mask, via dense
    F.conv1d over the mask against an all-ones kernel."""
    (k,) = kernel_size
    ones_kernel = torch.ones(1, 1, k)
    overlap = F.conv1d(occ_mask.float().unsqueeze(1), ones_kernel,
                        stride=stride, padding=padding, dilation=dilation)
    return overlap.squeeze(1) > 0


class TestSparseSubmanifoldConv1d:
    def test_matches_dense_conv1d_at_occupied_positions(self):
        torch.manual_seed(20)
        b, c_in, c_out, l = 1, 3, 4, 16
        weight = torch.randn(c_out, c_in, 3)
        bias = torch.randn(c_out)

        x = torch.zeros(b, c_in, l)
        for xx in [0, 3, 7, 15]:
            x[0, :, xx] = torch.randn(c_in)

        reference = F.conv1d(x, weight, bias, stride=1, padding=1)
        feats, coords = _dense_to_sparse1d(x)

        out = flexgemm_ops_module.sparse_submanifold_conv1d(
            feats, coords, torch.Size((b, c_in, l)), weight, bias=bias)

        ref_blc = reference.permute(0, 2, 1)
        for row, (bb, xx) in enumerate(coords.tolist()):
            torch.testing.assert_close(out[row], ref_blc[bb, xx], atol=1e-5, rtol=1e-5)


class TestSparseConv1d:
    def test_matches_dense_conv1d_stride2(self):
        torch.manual_seed(21)
        b, c_in, c_out, l = 1, 2, 3, 16
        weight = torch.randn(c_out, c_in, 3)
        bias = torch.randn(c_out)
        stride, padding, dilation = (2,), (1,), (1,)

        x = torch.zeros(b, c_in, l)
        for xx in [0, 3, 7, 15, 9]:
            x[0, :, xx] = torch.randn(c_in)
        occ_mask = x.abs().amax(dim=1) > 1e-12

        reference = F.conv1d(x, weight, bias, stride=stride, padding=padding, dilation=dilation)
        feats, coords = _dense_to_sparse1d(x)

        out_feats, out_coords = flexgemm_ops_module.sparse_conv1d(
            feats, coords, torch.Size((b, c_in, l)), weight, bias=bias,
            stride=stride, padding=padding, dilation=dilation)

        expected_touched = _touched_output_mask1d(occ_mask, (3,), stride, padding, dilation)
        assert out_coords.shape[0] == int(expected_touched.sum())

        ref_blc = reference.permute(0, 2, 1)
        for row, (bb, xx) in enumerate(out_coords.tolist()):
            assert expected_touched[bb, xx]
            torch.testing.assert_close(out_feats[row], ref_blc[bb, xx], atol=1e-5, rtol=1e-5)


class TestSparseConv1dFromDense:
    def test_matches_dense_conv1d_including_untouched_positions(self):
        torch.manual_seed(22)
        b, c_in, c_out, l = 1, 2, 3, 16
        weight = torch.randn(c_out, c_in, 3)
        bias = torch.randn(c_out)

        x = torch.zeros(b, c_in, l)
        for xx in [1, 4, 12]:
            x[0, :, xx] = torch.randn(c_in)

        reference = F.conv1d(x, weight, bias, stride=1, padding=1)
        result = flexgemm_ops_module.sparse_conv1d_from_dense(
            x, weight, bias, stride=(1,), padding=(1,), dilation=(1,))

        assert result is not None
        torch.testing.assert_close(result, reference, atol=1e-5, rtol=1e-5)

    def test_wrong_dims_returns_none(self):
        assert flexgemm_ops_module.sparse_conv1d_from_dense(torch.zeros(2, 2), torch.zeros(3, 2)) is None


class TestSparseConv1dNative:
    def test_unavailable_returns_none(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: False)
        feats = torch.randn(2, 3)
        coords = torch.tensor([[0, 0], [0, 5]], dtype=torch.int32)
        weight = torch.randn(4, 3, 3)
        assert flexgemm_ops_module.sparse_conv1d_native(
            feats, coords, torch.Size((1, 3, 8)), weight) is None

    def test_matches_dense_conv1d_via_double_lift(self, monkeypatch):
        # Mocks the 2D-native entry points sparse_conv1d_native lifts
        # through, using the already-F.conv2d-validated pure-Python
        # sparse_conv2d as the fake kernel -- same technique
        # TestSparseConv2dNative uses for the 2D->3D lift, one level up.
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: True)
        monkeypatch.setattr(flexgemm_ops_module, "_sparse_conv3d", _fake_sparse_conv3d_via_2d)
        torch.manual_seed(23)
        b, c_in, c_out, l = 1, 2, 3, 16
        weight = torch.randn(c_out, c_in, 3)
        bias = torch.randn(c_out)
        stride, padding, dilation = (2,), (1,), (1,)

        x = torch.zeros(b, c_in, l)
        for xx in [0, 3, 7, 15]:
            x[0, :, xx] = torch.randn(c_in)

        reference = F.conv1d(x, weight, bias, stride=stride, padding=padding, dilation=dilation)
        feats, coords = _dense_to_sparse1d(x)

        result = flexgemm_ops_module.sparse_conv1d_native(
            feats, coords, torch.Size((b, c_in, l)), weight, bias=bias,
            stride=stride, padding=padding, dilation=dilation)

        assert result is not None
        out_feats, out_coords = result
        ref_blc = reference.permute(0, 2, 1)
        for row, (bb, xx) in enumerate(out_coords.tolist()):
            torch.testing.assert_close(out_feats[row], ref_blc[bb, xx], atol=1e-5, rtol=1e-5)


class TestSparseConv1dEnabledFlag:
    def teardown_method(self):
        os.environ.pop("AMD_TUNED_TORCH_SPARSE_CONV1D", None)
        importlib.reload(flexgemm_ops_module)

    def test_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("AMD_TUNED_TORCH_SPARSE_CONV1D", raising=False)
        importlib.reload(flexgemm_ops_module)
        assert flexgemm_ops_module.sparse_conv1d_enabled() is True

    def test_disabled_when_explicitly_zero(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SPARSE_CONV1D", "0")
        importlib.reload(flexgemm_ops_module)
        assert flexgemm_ops_module.sparse_conv1d_enabled() is False


class TestMaybeSparseConv1d:
    def test_declines_when_disabled(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv1d_enabled", lambda: False)
        x = torch.zeros(1, 2, 8)
        weight = torch.randn(3, 2, 3)
        assert flexgemm_ops_module.maybe_sparse_conv1d(x, weight) is None

    def test_declines_above_occupancy_threshold(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv1d_enabled", lambda: True)
        fake_from_dense = MagicMock(return_value="sparse_result")
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv1d_from_dense", fake_from_dense)

        x = torch.ones(1, 2, 8)  # fully dense
        weight = torch.randn(3, 2, 3)

        # min_positions=1: bypass the minimum-size gate -- see the 3D
        # version of this test for why.
        result = flexgemm_ops_module.maybe_sparse_conv1d(
            x, weight, max_occupancy=0.3, min_positions=1)

        assert result is None
        fake_from_dense.assert_not_called()

    def test_routes_to_sparse_below_threshold(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv1d_enabled", lambda: True)
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: True)
        fake_from_dense = MagicMock(return_value="sparse_result")
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv1d_from_dense", fake_from_dense)

        x = torch.zeros(1, 2, 8)
        x[0, :, 0] = 1.0
        weight = torch.randn(3, 2, 3)
        bias = torch.randn(3)

        result = flexgemm_ops_module.maybe_sparse_conv1d(
            x, weight, bias, stride=(1,), padding=(1,), dilation=(1,),
            max_occupancy=0.3, min_positions=1)

        assert result == "sparse_result"
        fake_from_dense.assert_called_once_with(
            x, weight, bias, stride=(1,), padding=(1,), dilation=(1,))

    def test_wrong_dims_returns_none(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv1d_enabled", lambda: True)
        assert flexgemm_ops_module.maybe_sparse_conv1d(torch.zeros(2, 2), torch.zeros(3, 2)) is None


# ---------------------------------------------------------------------------
# Minimum-size gate -- checked before occupancy in all three
# maybe_sparse_conv{1,2,3}d, so a small input skips paying for an occupancy
# estimate at all, not just for the sparse kernel.
# ---------------------------------------------------------------------------


class TestNSpatialPositions:
    def test_counts_batch_times_spatial_dims_only(self):
        assert flexgemm_ops_module._n_spatial_positions(torch.zeros(2, 3, 4)) == 8       # 1D: B*L
        assert flexgemm_ops_module._n_spatial_positions(torch.zeros(2, 3, 4, 5)) == 40    # 2D: B*H*W
        assert flexgemm_ops_module._n_spatial_positions(torch.zeros(2, 3, 4, 5, 6)) == 240  # 3D: B*D*H*W


class TestMinSpatialPositionsGate:
    def test_conv3d_declines_below_min_positions_without_checking_occupancy(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv3d_enabled", lambda: True)
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: True)
        occupancy_spy = MagicMock(wraps=flexgemm_ops_module._occupancy)
        monkeypatch.setattr(flexgemm_ops_module, "_occupancy", occupancy_spy)

        x = torch.zeros(1, 2, 4, 4, 4)  # 64 spatial positions, empty -> would pass occupancy
        x[0, :, 0, 0, 0] = 1.0
        weight = torch.randn(3, 2, 3, 3, 3)

        result = flexgemm_ops_module.maybe_sparse_conv3d(x, weight, min_positions=100)

        assert result is None
        occupancy_spy.assert_not_called()  # declined before ever estimating occupancy

    def test_conv3d_proceeds_at_or_above_min_positions(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv3d_enabled", lambda: True)
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: True)
        occupancy_spy = MagicMock(wraps=flexgemm_ops_module._occupancy)
        monkeypatch.setattr(flexgemm_ops_module, "_occupancy", occupancy_spy)

        x = torch.zeros(1, 2, 4, 4, 4)  # 64 spatial positions
        x[0, :, 0, 0, 0] = 1.0
        weight = torch.randn(3, 2, 3, 3, 3)

        flexgemm_ops_module.maybe_sparse_conv3d(x, weight, min_positions=64)

        occupancy_spy.assert_called_once()

    def test_conv2d_declines_below_min_positions_without_checking_occupancy(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv2d_enabled", lambda: True)
        occupancy_spy = MagicMock(wraps=flexgemm_ops_module._occupancy2d)
        monkeypatch.setattr(flexgemm_ops_module, "_occupancy2d", occupancy_spy)

        x = torch.zeros(1, 2, 4, 4)  # 16 spatial positions
        x[0, :, 0, 0] = 1.0
        weight = torch.randn(3, 2, 3, 3)

        result = flexgemm_ops_module.maybe_sparse_conv2d(x, weight, min_positions=100)

        assert result is None
        occupancy_spy.assert_not_called()

    def test_conv1d_declines_below_min_positions_without_checking_occupancy(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv1d_enabled", lambda: True)
        occupancy_spy = MagicMock(wraps=flexgemm_ops_module._occupancy1d)
        monkeypatch.setattr(flexgemm_ops_module, "_occupancy1d", occupancy_spy)

        x = torch.zeros(1, 2, 8)  # 8 spatial positions
        x[0, :, 0] = 1.0
        weight = torch.randn(3, 2, 3)

        result = flexgemm_ops_module.maybe_sparse_conv1d(x, weight, min_positions=100)

        assert result is None
        occupancy_spy.assert_not_called()

    def test_default_threshold_is_read_from_env_at_import(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv2d_enabled", lambda: True)
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: True)
        occupancy_spy = MagicMock(wraps=flexgemm_ops_module._occupancy2d)
        monkeypatch.setattr(flexgemm_ops_module, "_occupancy2d", occupancy_spy)
        monkeypatch.setattr(flexgemm_ops_module, "_SPARSE_CONV2D_MIN_POSITIONS", 8)

        x = torch.zeros(1, 2, 4, 4)  # 16 spatial positions >= 8 -> proceeds
        x[0, :, 0, 0] = 1.0
        weight = torch.randn(3, 2, 3, 3)

        flexgemm_ops_module.maybe_sparse_conv2d(x, weight)  # no explicit min_positions

        occupancy_spy.assert_called_once()


class TestMinPositionsCalibrationPrecedence:
    """AMD_TUNED_TORCH_SPARSE_CONV2D_MIN_POSITIONS's own default: explicit env
    var > sparse_conv_calibration.load() > the hardcoded "1024" guess. Only
    conv2d is exercised here -- conv1d/conv3d wire the identical pattern,
    covered by inspection rather than tripling this reload dance."""

    def teardown_method(self):
        os.environ.pop("AMD_TUNED_TORCH_SPARSE_CONV2D_MIN_POSITIONS", None)
        importlib.reload(flexgemm_ops_module)

    def test_uses_hardcoded_default_when_no_calibration(self, monkeypatch):
        monkeypatch.delenv("AMD_TUNED_TORCH_SPARSE_CONV2D_MIN_POSITIONS", raising=False)
        monkeypatch.setattr(flexgemm_ops_module.sparse_conv_calibration, "load", lambda: {})
        importlib.reload(flexgemm_ops_module)
        assert flexgemm_ops_module._SPARSE_CONV2D_MIN_POSITIONS == 1024

    def test_uses_calibrated_value_when_present(self, monkeypatch):
        monkeypatch.delenv("AMD_TUNED_TORCH_SPARSE_CONV2D_MIN_POSITIONS", raising=False)
        monkeypatch.setattr(flexgemm_ops_module.sparse_conv_calibration, "load",
                             lambda: {"conv2d": {"min_positions": 777}})
        importlib.reload(flexgemm_ops_module)
        assert flexgemm_ops_module._SPARSE_CONV2D_MIN_POSITIONS == 777

    def test_calibration_for_other_dims_does_not_affect_this_one(self, monkeypatch):
        monkeypatch.delenv("AMD_TUNED_TORCH_SPARSE_CONV2D_MIN_POSITIONS", raising=False)
        monkeypatch.setattr(flexgemm_ops_module.sparse_conv_calibration, "load",
                             lambda: {"conv1d": {"min_positions": 111}, "conv3d": {"min_positions": 333}})
        importlib.reload(flexgemm_ops_module)
        assert flexgemm_ops_module._SPARSE_CONV2D_MIN_POSITIONS == 1024

    def test_calibration_with_only_max_occupancy_does_not_affect_min_positions(self, monkeypatch):
        """A dim entry can carry only one of the two fields (e.g.
        tools/benchmark_sparse_conv.py's --sweep occupancy) -- the missing
        field must fall back to the hardcoded default, not crash."""
        monkeypatch.delenv("AMD_TUNED_TORCH_SPARSE_CONV2D_MIN_POSITIONS", raising=False)
        monkeypatch.setattr(flexgemm_ops_module.sparse_conv_calibration, "load",
                             lambda: {"conv2d": {"max_occupancy": 0.2}})
        importlib.reload(flexgemm_ops_module)
        assert flexgemm_ops_module._SPARSE_CONV2D_MIN_POSITIONS == 1024

    def test_explicit_env_var_wins_over_calibration(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SPARSE_CONV2D_MIN_POSITIONS", "999")
        monkeypatch.setattr(flexgemm_ops_module.sparse_conv_calibration, "load",
                             lambda: {"conv2d": {"min_positions": 777}})
        importlib.reload(flexgemm_ops_module)
        assert flexgemm_ops_module._SPARSE_CONV2D_MIN_POSITIONS == 999


class TestMaxOccupancyCalibrationPrecedence:
    """Same precedence as TestMinPositionsCalibrationPrecedence, for
    AMD_TUNED_TORCH_SPARSE_CONV2D_MAX_OCCUPANCY. Only conv2d is exercised
    here -- conv1d/conv3d wire the identical pattern."""

    def teardown_method(self):
        os.environ.pop("AMD_TUNED_TORCH_SPARSE_CONV2D_MAX_OCCUPANCY", None)
        importlib.reload(flexgemm_ops_module)

    def test_uses_hardcoded_default_when_no_calibration(self, monkeypatch):
        monkeypatch.delenv("AMD_TUNED_TORCH_SPARSE_CONV2D_MAX_OCCUPANCY", raising=False)
        monkeypatch.setattr(flexgemm_ops_module.sparse_conv_calibration, "load", lambda: {})
        importlib.reload(flexgemm_ops_module)
        assert flexgemm_ops_module._SPARSE_CONV2D_MAX_OCCUPANCY == 0.1

    def test_uses_calibrated_value_when_present(self, monkeypatch):
        monkeypatch.delenv("AMD_TUNED_TORCH_SPARSE_CONV2D_MAX_OCCUPANCY", raising=False)
        monkeypatch.setattr(flexgemm_ops_module.sparse_conv_calibration, "load",
                             lambda: {"conv2d": {"max_occupancy": 0.17}})
        importlib.reload(flexgemm_ops_module)
        assert flexgemm_ops_module._SPARSE_CONV2D_MAX_OCCUPANCY == 0.17

    def test_calibration_with_only_min_positions_does_not_affect_max_occupancy(self, monkeypatch):
        monkeypatch.delenv("AMD_TUNED_TORCH_SPARSE_CONV2D_MAX_OCCUPANCY", raising=False)
        monkeypatch.setattr(flexgemm_ops_module.sparse_conv_calibration, "load",
                             lambda: {"conv2d": {"min_positions": 8000}})
        importlib.reload(flexgemm_ops_module)
        assert flexgemm_ops_module._SPARSE_CONV2D_MAX_OCCUPANCY == 0.1

    def test_explicit_env_var_wins_over_calibration(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SPARSE_CONV2D_MAX_OCCUPANCY", "0.55")
        monkeypatch.setattr(flexgemm_ops_module.sparse_conv_calibration, "load",
                             lambda: {"conv2d": {"max_occupancy": 0.17}})
        importlib.reload(flexgemm_ops_module)
        assert flexgemm_ops_module._SPARSE_CONV2D_MAX_OCCUPANCY == 0.55


class TestInferenceTensorOccupancy:
    """A tensor created inside `torch.inference_mode()` tracks no version
    counter, and `._version` on one raises. The occupancy cache read it
    unguarded for its cache key, so a plain `F.conv2d` under
    inference_mode died with "Inference tensors do not track version
    counter." from inside a caching detail -- see
    flexgemm_ops._tensor_version's docstring and crash_log.log (Hunyuan3D-2's
    DINOv2 patch-embedding conv2d). Inference mode is the normal way to run
    inference and amd_tuned_torch._grad_safe deliberately allows patching
    there, so these calls must work, not merely not crash by luck."""

    def test_version_is_none_for_an_inference_tensor(self):
        with torch.inference_mode():
            x = torch.zeros(1, 1, 2, 2)
            assert flexgemm_ops_module._tensor_version(x) is None

    def test_version_is_the_counter_for_an_ordinary_tensor(self):
        x = torch.zeros(1, 1, 2, 2)
        assert flexgemm_ops_module._tensor_version(x) == x._version
        x[0, 0, 0, 0] = 1.0
        assert flexgemm_ops_module._tensor_version(x) == x._version

    def test_cache_get_and_set_do_not_raise_on_an_inference_tensor(self):
        cache = flexgemm_ops_module._OccupancyCache()
        with torch.inference_mode():
            x = torch.zeros(1, 1, 2, 2)
            assert cache.get(x, 1e-12) is None
            cache.set(x, 1e-12, 0.5)          # must be a no-op, not a crash
            assert cache.get(x, 1e-12) is None
        assert cache._data == {}

    def test_cached_occupancy_recomputes_instead_of_caching(self, monkeypatch):
        spy = MagicMock(wraps=flexgemm_ops_module._estimate_occupancy)
        monkeypatch.setattr(flexgemm_ops_module, "_estimate_occupancy", spy)
        with torch.inference_mode():
            x = torch.zeros(1, 1, 2, 2, 2)
            x_ = x  # keep the same object across both calls
            assert flexgemm_ops_module._cached_occupancy(x_) == 0.0
            assert flexgemm_ops_module._cached_occupancy(x_) == 0.0
        # uncacheable, so no cache hit to skip the second estimate
        assert spy.call_count == 2

    def test_maybe_sparse_conv2d_runs_under_inference_mode(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv2d_enabled", lambda: True)
        with torch.inference_mode():
            x = torch.zeros(1, 2, 8, 8)
            x[0, :, 0, 0] = 1.0            # sparse enough to take the fast path
            w = torch.randn(2, 2, 3, 3)
            out = flexgemm_ops_module.maybe_sparse_conv2d(x, w, None, padding=(1, 1))
        assert out is None or out.shape == (1, 2, 8, 8)


class TestSparseFastPathNeedsTheNativeKernel:
    """maybe_sparse_conv1d/2d take the sparse path only when flex_gemm's
    native kernel is there to make it fast. Without it,
    sparse_conv{1,2}d_from_dense still returns a CORRECT result via its
    pure-Python gather/scatter -- which is why these two used to skip the
    available() check -- but correct is not the same as fast: measured on
    gfx1100, that path ran 21-63x slower than stock MIOpen on the very
    inputs the occupancy gate accepts, with an fp16 error of 0.87% of the
    output's RMS, i.e. silently. See maybe_sparse_conv2d's docstring."""

    @staticmethod
    def _sparse_2d():
        x = torch.zeros(1, 4, 128, 128)
        x[:, :, ::16, ::16] = 1.0          # occupancy ~0.004, 16384 positions
        return x, torch.randn(4, 4, 3, 3)

    @staticmethod
    def _sparse_1d():
        x = torch.zeros(1, 4, 8192)
        x[:, :, ::16] = 1.0
        return x, torch.randn(4, 4, 3)

    def test_conv2d_declines_without_the_native_extension(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv2d_enabled", lambda: True)
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: False)
        x, w = self._sparse_2d()
        assert flexgemm_ops_module.maybe_sparse_conv2d(x, w, None, padding=(1, 1)) is None

    def test_conv1d_declines_without_the_native_extension(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv1d_enabled", lambda: True)
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: False)
        x, w = self._sparse_1d()
        assert flexgemm_ops_module.maybe_sparse_conv1d(x, w, None, padding=(1,)) is None

    def test_conv2d_still_takes_the_path_when_the_kernel_is_there(self, monkeypatch):
        """The gate must not disable the feature for anyone who built the
        extension -- with available() True, a sparse input still routes."""
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv2d_enabled", lambda: True)
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: True)
        x, w = self._sparse_2d()
        out = flexgemm_ops_module.maybe_sparse_conv2d(x, w, None, padding=(1, 1))
        assert out is not None
        torch.testing.assert_close(out, F.conv2d(x, w, None, 1, 1), rtol=1e-4, atol=1e-4)

    def test_dense_input_still_declines_with_the_kernel_present(self, monkeypatch):
        monkeypatch.setattr(flexgemm_ops_module, "sparse_conv2d_enabled", lambda: True)
        monkeypatch.setattr(flexgemm_ops_module, "available", lambda: True)
        x = torch.randn(1, 4, 128, 128)     # occupancy 1.0
        assert flexgemm_ops_module.maybe_sparse_conv2d(
            x, torch.randn(4, 4, 3, 3), None, padding=(1, 1)) is None
