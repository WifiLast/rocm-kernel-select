"""Tests for amd_tuned_torch.magcache -- the generic MagCache skip/cache decision
engine (source/MagCache, arXiv:2506.09045). Pure PyTorch/Python, no aiter,
TransformerEngine, or GPU required -- these tests run identically on CPU.

Run with:

    python -m pytest source/cmp_ext_turing/tests
"""
from __future__ import annotations

import pytest
import torch

import amd_tuned_torch.magcache as magcache


class TestNearestInterp:
    def test_identity_when_lengths_match(self):
        src = [1.0, 2.0, 3.0]
        assert magcache.nearest_interp(src, 3) == src

    def test_target_length_one_returns_last_element(self):
        assert magcache.nearest_interp([1.0, 2.0, 3.0], 1) == [3.0]

    def test_upsamples_by_nearest_neighbor(self):
        # 3 -> 5: matches the reference nearest_interp's own index math
        # (scale = (3-1)/(5-1) = 0.5, indices = round([0,0.5,1,1.5,2]) = [0,0,1,2,2]).
        src = [10.0, 20.0, 30.0]
        assert magcache.nearest_interp(src, 5) == [10.0, 10.0, 20.0, 30.0, 30.0]

    def test_downsamples_by_nearest_neighbor(self):
        # 5 -> 3: scale = (5-1)/(3-1) = 2, indices = round([0,2,4]) = [0,2,4].
        src = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert magcache.nearest_interp(src, 3) == [1.0, 3.0, 5.0]


class TestMagCacheCalibrator:
    def test_first_record_does_not_append_ratio(self):
        calib = magcache.MagCacheCalibrator(num_steps=3)
        calib.record(torch.ones(2, 4))
        assert calib.norm_ratios == []
        assert calib.cnt == 1

    def test_records_ratio_std_and_cos_dist_from_second_call_on(self):
        calib = magcache.MagCacheCalibrator(num_steps=3)
        r1 = torch.ones(2, 4) * 2.0
        r2 = torch.ones(2, 4) * 3.0
        calib.record(r1)
        calib.record(r2)
        assert len(calib.norm_ratios) == 1
        assert calib.norm_ratios[0] == pytest.approx(1.5, abs=1e-4)
        assert len(calib.norm_stds) == 1
        assert len(calib.cos_dists) == 1
        # r1 and r2 point the same direction (both positive multiples of
        # the same vector) -- cosine similarity is 1, so cos_dist ~ 0.
        assert calib.cos_dists[0] == pytest.approx(0.0, abs=1e-3)

    def test_finalize_prefixes_with_one(self):
        calib = magcache.MagCacheCalibrator(num_steps=3)
        calib.record(torch.ones(2, 4))
        calib.record(torch.ones(2, 4) * 2.0)
        result = calib.finalize()
        assert result[0] == 1.0
        assert result[1:] == calib.norm_ratios


class TestMagCacheConstruction:
    def test_raises_when_mag_ratios_too_short(self):
        with pytest.raises(ValueError, match="mag_ratios"):
            magcache.MagCache(num_steps=5, mag_ratios=[1.0, 1.0])

    def test_accepts_exact_length(self):
        magcache.MagCache(num_steps=3, mag_ratios=[1.0, 1.0, 1.0])

    def test_accepts_longer_table(self):
        magcache.MagCache(num_steps=3, mag_ratios=[1.0] * 10)


class TestShouldSkip:
    def test_false_with_no_cached_residual_yet(self):
        mc = magcache.MagCache(num_steps=10, mag_ratios=[1.0] * 10, retention_ratio=0.0)
        assert mc.should_skip() is False

    def test_false_during_retention_period(self):
        mc = magcache.MagCache(num_steps=10, mag_ratios=[1.0] * 10, retention_ratio=0.5)
        mc.record(torch.ones(2, 4))
        # retention covers int(0.5*10+0.5) = 5 steps; cnt starts at 0.
        assert mc.should_skip() is False

    def test_true_when_ratios_close_to_one_and_within_budget(self):
        mc = magcache.MagCache(
            num_steps=10, mag_ratios=[1.0] * 10, retention_ratio=0.0, thresh=0.5, K=5
        )
        mc.record(torch.ones(2, 4))
        assert mc.should_skip() is True

    def test_false_and_resets_when_error_exceeds_threshold(self):
        mc = magcache.MagCache(
            num_steps=10, mag_ratios=[2.0] * 10, retention_ratio=0.0, thresh=0.5, K=5
        )
        mc.record(torch.ones(2, 4))
        # accumulated_ratio becomes 2.0, accumulated_err = |1-2.0| = 1.0 > thresh=0.5.
        assert mc.should_skip() is False
        assert mc.accumulated_ratio == 1.0
        assert mc.accumulated_steps == 0
        assert mc.accumulated_err == 0.0

    def test_false_and_resets_when_exceeding_max_consecutive_skips(self):
        # thresh generous enough to never trip, but K=1 caps consecutive skips.
        mc = magcache.MagCache(
            num_steps=10, mag_ratios=[1.0] * 10, retention_ratio=0.0, thresh=100.0, K=1
        )
        mc.record(torch.ones(2, 4))
        assert mc.should_skip() is True  # accumulated_steps -> 1, within K
        assert mc.should_skip() is False  # would be accumulated_steps -> 2, exceeds K=1
        assert mc.accumulated_steps == 0

    def test_each_call_advances_accumulators_even_when_not_skipping(self):
        # should_skip() itself accumulates state on every call, not just
        # accepted skips -- this matters because callers must call it
        # exactly once per step (paired with advance()), never speculatively.
        mc = magcache.MagCache(
            num_steps=2, mag_ratios=[1.5, 1.5], retention_ratio=0.0, thresh=100.0, K=5
        )
        mc.record(torch.ones(2, 4))
        mc.should_skip()
        assert mc.accumulated_ratio == pytest.approx(1.5)
        mc.cnt = 1
        mc.should_skip()
        assert mc.accumulated_ratio == pytest.approx(2.25)


class TestRecordAndAdvance:
    def test_record_sets_cached_residual(self):
        mc = magcache.MagCache(num_steps=5, mag_ratios=[1.0] * 5)
        residual = torch.randn(2, 4)
        mc.record(residual)
        assert mc.cached_residual is residual

    def test_advance_increments_cnt(self):
        mc = magcache.MagCache(num_steps=5, mag_ratios=[1.0] * 5)
        mc.advance()
        assert mc.cnt == 1

    def test_advance_wraps_and_resets_accumulators_but_not_residual(self):
        mc = magcache.MagCache(num_steps=3, mag_ratios=[1.0] * 3, retention_ratio=0.0)
        residual = torch.randn(2, 4)
        mc.record(residual)
        mc.should_skip()  # perturbs accumulators away from defaults
        mc.cnt = 2
        mc.advance()
        assert mc.cnt == 0
        assert mc.accumulated_ratio == 1.0
        assert mc.accumulated_steps == 0
        assert mc.accumulated_err == 0.0
        assert mc.cached_residual is residual  # not cleared on wraparound


class TestEndToEndSimulation:
    """Simulates the exact usage pattern from the module/README docstring:
    a fake 'run_transformer_blocks' stands in for a real model, and this
    verifies both that skipping actually reduces the number of real calls
    *and* that skipped steps produce numerically correct output."""

    def test_skips_reduce_real_compute_calls_and_stay_numerically_correct(self):
        num_steps = 10
        mc = magcache.MagCache(
            num_steps=num_steps, mag_ratios=[1.0] * num_steps, retention_ratio=0.2,
            thresh=0.5, K=3,
        )
        hidden_states = torch.zeros(2, 4)
        real_compute_calls = 0
        skipped_steps = 0

        def run_transformer_blocks(h):
            nonlocal real_compute_calls
            real_compute_calls += 1
            return h + 1.0  # a fixed, deterministic "residual" of 1.0 per real step

        for _ in range(num_steps):
            if mc.should_skip():
                skipped_steps += 1
                hidden_states = hidden_states + mc.cached_residual
            else:
                ori = hidden_states
                hidden_states = run_transformer_blocks(hidden_states)
                mc.record(hidden_states - ori)
            mc.advance()

        assert skipped_steps > 0
        assert real_compute_calls < num_steps
        assert real_compute_calls + skipped_steps == num_steps
        # Every step (real or skipped) adds exactly 1.0 in this simulation,
        # real or reused -- output must match regardless of which path ran.
        assert torch.equal(hidden_states, torch.full((2, 4), float(num_steps)))

    def test_never_skips_when_thresh_is_zero(self):
        num_steps = 10
        mc = magcache.MagCache(
            num_steps=num_steps, mag_ratios=[1.01] * num_steps, retention_ratio=0.0, thresh=0.0
        )
        real_compute_calls = 0
        hidden_states = torch.zeros(2, 4)
        for _ in range(num_steps):
            if mc.should_skip():
                hidden_states = hidden_states + mc.cached_residual
            else:
                real_compute_calls += 1
                ori = hidden_states
                hidden_states = ori + 1.0
                mc.record(hidden_states - ori)
            mc.advance()
        assert real_compute_calls == num_steps
