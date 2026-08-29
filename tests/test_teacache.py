"""Tests for amd_tuned_torch.teacache -- the generic TeaCache skip/cache decision
engine (source/TeaCache, arXiv:2411.19108). Pure PyTorch/Python, no aiter,
TransformerEngine, or GPU required -- these tests run identically on CPU.

Run with:

    python -m pytest source/cmp_ext_turing/tests
"""
from __future__ import annotations

import pytest
import torch

import amd_tuned_torch.teacache as teacache


class TestEvaluatePolynomial:
    def test_matches_numpy_poly1d_convention(self):
        # x^2 + 0x + 0, evaluated at x=3 -> 9 (numpy.poly1d([1, 0, 0])(3) == 9).
        assert teacache.evaluate_polynomial([1, 0, 0], 3) == pytest.approx(9.0)

    def test_linear_polynomial(self):
        # 2x + 3 at x=5 -> 13 (numpy.poly1d([2, 3])(5) == 13).
        assert teacache.evaluate_polynomial([2, 3], 5) == pytest.approx(13.0)

    def test_constant_polynomial(self):
        assert teacache.evaluate_polynomial([7.0], 100.0) == pytest.approx(7.0)

    def test_matches_flux_coefficients_at_a_sample_point(self):
        coeffs = [4.98651651e02, -2.83781631e02, 5.58554382e01, -3.82021401e00, 2.64230861e-01]
        # Cross-checked against numpy.poly1d(coeffs)(0.05).
        import numpy as np

        expected = float(np.poly1d(coeffs)(0.05))
        assert teacache.evaluate_polynomial(coeffs, 0.05) == pytest.approx(expected, rel=1e-9)


class TestShouldSkipForcedSteps:
    def test_first_step_never_skips(self):
        tc = teacache.TeaCache(num_steps=5, rel_l1_thresh=0.6)
        tc.record(torch.ones(2, 4))  # pretend a previous run left a cached residual
        assert tc.should_skip(torch.randn(2, 4)) is False
        assert tc.accumulated_rel_l1_distance == 0.0

    def test_last_step_never_skips(self):
        tc = teacache.TeaCache(num_steps=5, rel_l1_thresh=0.6)
        tc.cnt = 4  # num_steps - 1
        tc.record(torch.ones(2, 4))
        tc.previous_modulated_input = torch.randn(2, 4)
        assert tc.should_skip(torch.randn(2, 4)) is False

    def test_records_modulated_input_even_when_forced(self):
        tc = teacache.TeaCache(num_steps=5, rel_l1_thresh=0.6)
        probe = torch.randn(2, 4)
        tc.should_skip(probe)
        assert tc.previous_modulated_input is probe


class TestShouldSkipMiddleSteps:
    def test_false_without_cached_residual_even_if_distance_is_tiny(self):
        tc = teacache.TeaCache(num_steps=5, rel_l1_thresh=0.6)
        tc.cnt = 1
        tc.previous_modulated_input = torch.ones(2, 4)
        # no record() called yet -- cached_residual is still None
        assert tc.should_skip(torch.ones(2, 4)) is False

    def test_true_when_distance_stays_under_threshold(self):
        tc = teacache.TeaCache(num_steps=5, rel_l1_thresh=0.6)
        tc.record(torch.ones(2, 4))
        tc.cnt = 1
        tc.previous_modulated_input = torch.ones(2, 4)
        # identical modulated input -> raw distance 0, well under thresh=0.6
        assert tc.should_skip(torch.ones(2, 4)) is True

    def test_false_and_resets_when_distance_exceeds_threshold(self):
        tc = teacache.TeaCache(num_steps=5, rel_l1_thresh=0.1)
        tc.record(torch.ones(2, 4))
        tc.cnt = 1
        tc.previous_modulated_input = torch.ones(2, 4)
        # new input is 3x the old one -> raw distance = |3-1|/1 = 2.0 > thresh=0.1
        assert tc.should_skip(torch.ones(2, 4) * 3) is False
        assert tc.accumulated_rel_l1_distance == 0.0

    def test_polynomial_rescale_changes_the_decision(self):
        # Same raw distance, but a rescale polynomial that amplifies it
        # past the threshold flips the decision from skip to no-skip.
        tc_identity = teacache.TeaCache(num_steps=5, rel_l1_thresh=0.6, coefficients=None)
        tc_amplified = teacache.TeaCache(num_steps=5, rel_l1_thresh=0.6, coefficients=[100.0, 0.0])

        for tc in (tc_identity, tc_amplified):
            tc.record(torch.ones(2, 4))
            tc.cnt = 1
            tc.previous_modulated_input = torch.ones(2, 4)

        probe = torch.ones(2, 4) * 1.01  # small raw distance: 0.01
        assert tc_identity.should_skip(probe) is True
        assert tc_amplified.should_skip(probe) is False  # 100 * 0.01 = 1.0 > 0.6

    def test_accumulates_across_consecutive_middle_steps(self):
        # should_skip() unconditionally sets previous_modulated_input to
        # whatever was just passed in, so reproducing a *constant* 0.4
        # relative distance on every call means growing the probe by the
        # same 1.4x factor each time, relative to the previous call's
        # value -- not reusing one static tensor.
        tc = teacache.TeaCache(num_steps=10, rel_l1_thresh=1.0)
        tc.record(torch.ones(2, 4))
        tc.cnt = 1
        tc.previous_modulated_input = torch.ones(2, 4)
        probe = torch.ones(2, 4) * 1.4  # raw distance |1.4-1|/1 = 0.4
        assert tc.should_skip(probe) is True
        assert tc.accumulated_rel_l1_distance == pytest.approx(0.4, abs=1e-4)
        tc.cnt = 2
        probe = probe * 1.4  # raw distance |1.4-1|/1 = 0.4 again, relative to the last probe
        assert tc.should_skip(probe) is True
        assert tc.accumulated_rel_l1_distance == pytest.approx(0.8, abs=1e-4)
        tc.cnt = 3
        probe = probe * 1.4
        # third consecutive step pushes accumulated (1.2) past thresh=1.0 -> forces recompute
        assert tc.should_skip(probe) is False
        assert tc.accumulated_rel_l1_distance == 0.0


class TestRecordAndAdvance:
    def test_record_sets_cached_residual(self):
        tc = teacache.TeaCache(num_steps=5)
        residual = torch.randn(2, 4)
        tc.record(residual)
        assert tc.cached_residual is residual

    def test_advance_increments_cnt(self):
        tc = teacache.TeaCache(num_steps=5)
        tc.advance()
        assert tc.cnt == 1

    def test_advance_wraps_and_resets_accumulated_distance(self):
        tc = teacache.TeaCache(num_steps=3)
        tc.accumulated_rel_l1_distance = 0.5
        tc.cnt = 2
        tc.advance()
        assert tc.cnt == 0
        assert tc.accumulated_rel_l1_distance == 0.0


class TestEndToEndSimulation:
    """Mirrors the exact usage pattern from the module/README docstring: a
    fake 'run_transformer_blocks' and a fake modulation probe stand in for
    a real model, verifying both that skipping reduces real compute calls
    *and* that skipped steps produce numerically correct output."""

    def test_skips_reduce_real_compute_calls_and_stay_numerically_correct(self):
        num_steps = 10
        tc = teacache.TeaCache(num_steps=num_steps, rel_l1_thresh=1.0)
        hidden_states = torch.zeros(2, 4)
        real_compute_calls = 0
        skipped_steps = 0
        constant_modulated_input = torch.ones(2, 4)  # unchanging probe -> raw distance always 0

        def run_transformer_blocks(h):
            nonlocal real_compute_calls
            real_compute_calls += 1
            return h + 1.0

        for _ in range(num_steps):
            if tc.should_skip(constant_modulated_input):
                skipped_steps += 1
                hidden_states = hidden_states + tc.cached_residual
            else:
                ori = hidden_states
                hidden_states = run_transformer_blocks(hidden_states)
                tc.record(hidden_states - ori)
            tc.advance()

        assert skipped_steps > 0
        assert real_compute_calls < num_steps
        assert real_compute_calls + skipped_steps == num_steps
        assert torch.equal(hidden_states, torch.full((2, 4), float(num_steps)))

    def test_never_skips_when_thresh_is_zero(self):
        num_steps = 10
        tc = teacache.TeaCache(num_steps=num_steps, rel_l1_thresh=0.0)
        real_compute_calls = 0
        hidden_states = torch.zeros(2, 4)
        probe = torch.ones(2, 4)
        for _ in range(num_steps):
            if tc.should_skip(probe):
                hidden_states = hidden_states + tc.cached_residual
            else:
                real_compute_calls += 1
                ori = hidden_states
                hidden_states = ori + 1.0
                tc.record(hidden_states - ori)
            tc.advance()
            probe = probe * 1.001  # tiny drift so the raw distance is never exactly 0
        assert real_compute_calls == num_steps
