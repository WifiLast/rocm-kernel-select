"""Tests for amd_tuned_torch.keyframe_residual -- the reference
implementation of the keyframe-residual image-set scheme described in
plan/spartialal_2d.txt.

These are the section 18 "first practical prototype" measurements turned
into assertions. Two kinds of test live here and they are worth telling
apart:

  * CORRECTNESS tests, which must always pass: the residual split round
    trips, and the residual convolution path is mathematically identical
    to the per-image baseline. These pin down the algebra the whole plan
    rests on -- if Conv(K) + Conv(D) ever stops equalling Conv(K + D),
    every benchmark downstream is measuring the wrong thing.

  * MEASUREMENT tests, which encode what the plan *expects* to be true
    of the data: that a burst's residual is mostly near zero, that an
    unrelated batch's is not, and that global alignment is what rescues
    the shifted case. These run on synthetic sets with known structure,
    so their thresholds are deliberately loose -- they are regression
    guards on the measurement machinery, NOT evidence about real
    imagery. Section 15's question is only answered by real bursts.

No GPU required; the one hardware test is skipped without ROCm. Sizes
are the section 18 prototype sizes (N=8, C=16, 64x64), which keeps the
whole file well under a second.

Run with:

    python -m pytest source/cmp_ext_turing/tests/test_keyframe_residual.py
"""
from __future__ import annotations

import pytest
import torch

from amd_tuned_torch import keyframe_residual as kr

# Section 18 prototype shape.
N, C, H, W = 8, 16, 64, 64


@pytest.fixture
def stack():
    """A correlated set, the case the scheme is designed for."""
    return kr.make_synthetic_set("burst", n=N, channels=C, height=H, width=W, seed=0)


@pytest.fixture
def weight():
    generator = torch.Generator().manual_seed(1)
    return torch.randn(8, C, 3, 3, generator=generator) * 0.1


@pytest.fixture
def bias():
    generator = torch.Generator().manual_seed(2)
    return torch.randn(8, generator=generator)


class TestResidualRoundTrip:
    """Section 3: the split has to be lossless before anything else matters."""

    def test_anchored_round_trips_exactly(self, stack):
        keyframe, residuals = kr.anchored_residuals(stack)
        assert torch.equal(kr.reconstruct_anchored(keyframe, residuals), stack)

    def test_anchored_keyframe_residual_is_zero(self, stack):
        _, residuals = kr.anchored_residuals(stack, keyframe_index=3)
        assert torch.count_nonzero(residuals[3]) == 0

    def test_anchored_honours_a_non_zero_keyframe_index(self, stack):
        """Section 16 trains with a random keyframe, so any index must work."""
        keyframe, residuals = kr.anchored_residuals(stack, keyframe_index=5)
        assert torch.equal(keyframe, stack[5])
        assert torch.equal(kr.reconstruct_anchored(keyframe, residuals), stack)

    def test_anchored_accepts_a_negative_keyframe_index(self, stack):
        keyframe, _ = kr.anchored_residuals(stack, keyframe_index=-1)
        assert torch.equal(keyframe, stack[-1])

    def test_anchored_rejects_an_out_of_range_keyframe_index(self, stack):
        with pytest.raises(IndexError):
            kr.anchored_residuals(stack, keyframe_index=N)

    def test_chained_round_trips(self, stack):
        keyframe, residuals = kr.chained_residuals(stack)
        torch.testing.assert_close(kr.reconstruct_chained(keyframe, residuals), stack)

    def test_both_forms_reject_a_non_4d_stack(self):
        with pytest.raises(ValueError):
            kr.anchored_residuals(torch.zeros(N, H, W))
        with pytest.raises(ValueError):
            kr.chained_residuals(torch.zeros(N, H, W))


class TestConvolutionLinearity:
    """Section 8: Conv(K) + Conv(D) == Conv(K + D). The load-bearing identity."""

    def test_residual_path_matches_per_image_baseline(self, stack, weight):
        baseline = kr.per_image_conv2d(stack, weight)
        residual = kr.keyframe_residual_conv2d(stack, weight)
        torch.testing.assert_close(residual, baseline, rtol=1e-5, atol=1e-5)

    def test_residual_path_matches_baseline_with_bias(self, stack, weight, bias):
        """The bias must be added once, not once per path.

        Convolution is linear but an affine layer is not: adding the bias
        on the residual path too would offset every non-keyframe output
        by exactly one extra bias vector. That error is invisible in a
        loss curve and fatal to the scheme, so it gets its own test.
        """
        baseline = kr.per_image_conv2d(stack, weight, bias)
        residual = kr.keyframe_residual_conv2d(stack, weight, bias)
        torch.testing.assert_close(residual, baseline, rtol=1e-5, atol=1e-5)

    def test_double_counted_bias_would_be_caught(self, stack, weight, bias):
        """Guard on the guard: show the above test can actually fail.

        If the residual path did add the bias twice, non-keyframe outputs
        would differ from the baseline by one bias vector. Constructing
        that wrong answer here proves the assertion above has teeth
        rather than passing because the tolerance is loose.
        """
        baseline = kr.per_image_conv2d(stack, weight, bias)
        wrong = baseline + bias.view(1, -1, 1, 1)
        with pytest.raises(AssertionError):
            torch.testing.assert_close(wrong, baseline, rtol=1e-5, atol=1e-5)

    def test_matches_baseline_for_any_keyframe_index(self, stack, weight, bias):
        baseline = kr.per_image_conv2d(stack, weight, bias)
        for index in (0, 3, N - 1, -1):
            residual = kr.keyframe_residual_conv2d(stack, weight, bias, keyframe_index=index)
            torch.testing.assert_close(residual, baseline, rtol=1e-5, atol=1e-5)

    def test_matches_baseline_with_padding_and_stride(self, stack, weight, bias):
        baseline = kr.per_image_conv2d(stack, weight, bias, padding=1, stride=2)
        residual = kr.keyframe_residual_conv2d(stack, weight, bias, padding=1, stride=2)
        torch.testing.assert_close(residual, baseline, rtol=1e-5, atol=1e-5)

    def test_holds_on_an_uncorrelated_set_too(self, weight, bias):
        """The identity is algebra, not a property of the data: it must
        hold just as exactly for the unrelated control set, where the
        scheme has no performance advantage whatsoever."""
        unrelated = kr.make_synthetic_set("unrelated", n=N, channels=C, height=H, width=W, seed=7)
        baseline = kr.per_image_conv2d(unrelated, weight, bias)
        residual = kr.keyframe_residual_conv2d(unrelated, weight, bias)
        torch.testing.assert_close(residual, baseline, rtol=1e-5, atol=1e-5)

    def test_single_image_set_is_just_the_keyframe(self, weight, bias):
        single = kr.make_synthetic_set("burst", n=1, channels=C, height=H, width=W, seed=0)
        torch.testing.assert_close(
            kr.keyframe_residual_conv2d(single, weight, bias),
            kr.per_image_conv2d(single, weight, bias),
        )


class TestThresholding:
    """Section 10 arm C: thresholding is the only step that loses information."""

    def test_zero_threshold_is_lossless(self, stack):
        _, residuals = kr.anchored_residuals(stack)
        assert torch.equal(kr.threshold_residuals(residuals, 0.0), residuals)

    def test_surviving_elements_keep_their_exact_value(self, stack):
        _, residuals = kr.anchored_residuals(stack)
        sparse = kr.threshold_residuals(residuals, 0.01)
        kept = sparse != 0
        assert torch.equal(sparse[kept], residuals[kept])

    def test_reconstruction_error_is_bounded_by_the_threshold(self, stack):
        """Hard thresholding drops elements below the threshold and keeps
        the rest exactly, so no element can move by more than it."""
        keyframe, residuals = kr.anchored_residuals(stack)
        threshold = 0.01
        approximation = kr.reconstruct_anchored(keyframe, kr.threshold_residuals(residuals, threshold))
        assert (approximation - stack).abs().max().item() < threshold

    def test_occupancy_falls_monotonically_with_the_threshold(self, stack):
        _, residuals = kr.anchored_residuals(stack)
        occupancies = [
            kr.residual_stats(kr.threshold_residuals(residuals, t), threshold=1e-9)["occupancy"]
            for t in (0.0, 0.005, 0.01, 0.05, 1.0)
        ]
        assert occupancies == sorted(occupancies, reverse=True)
        assert occupancies[-1] == 0.0

    def test_thresholded_conv_stays_close_to_the_exact_result(self, stack, weight, bias):
        """The approximation has to be *usable*, not just bounded: a
        threshold that kills the residual entirely would satisfy the
        bound above and destroy the output."""
        exact = kr.keyframe_residual_conv2d(stack, weight, bias)
        approximate = kr.keyframe_residual_conv2d(stack, weight, bias, threshold=0.005)
        relative = (approximate - exact).abs().max() / exact.abs().max()
        assert relative.item() < 0.05


class TestResidualStatistics:
    """Section 9: the numbers that decide whether any of this is worth it."""

    def test_stats_report_every_documented_field(self, stack):
        _, residuals = kr.anchored_residuals(stack)
        stats = kr.residual_stats(residuals)
        assert set(stats) == {"near_zero_fraction", "occupancy", "mean_abs",
                              "max_abs", "var", "clustering"}

    def test_near_zero_fraction_and_occupancy_are_complementary(self, stack):
        _, residuals = kr.anchored_residuals(stack)
        stats = kr.residual_stats(residuals)
        assert stats["near_zero_fraction"] + stats["occupancy"] == pytest.approx(1.0)

    def test_an_all_zero_residual_reports_full_sparsity(self):
        stats = kr.residual_stats(torch.zeros(N, C, H, W))
        assert stats["occupancy"] == 0.0
        assert stats["mean_abs"] == 0.0
        assert stats["clustering"] == 0.0

    def test_clustering_separates_a_blob_from_scattered_noise(self):
        """Section 9 cares as much about *where* the non-zeros are as how
        many: a solid blob is dense-tileable, salt-and-pepper is not.

        Both patterns below have exactly 400 non-zeros on a 128x128
        canvas, so occupancy cannot tell them apart and only the
        clustering term can. At that 2.4% density a uniformly scattered
        point has an 8-neighbour by chance about 18% of the time
        (1 - 0.976^8), which is what sets the bound used here; the
        earlier 64x64 version of this test was too dense to discriminate
        at all -- random points there cluster half the time.
        """
        side = 128
        blob = torch.zeros(1, 1, side, side)
        blob[..., 20:40, 20:40] = 1.0           # 400 contiguous non-zeros

        scattered = torch.zeros(1, 1, side, side)
        generator = torch.Generator().manual_seed(3)
        flat = scattered.view(-1)
        picks = torch.randperm(flat.numel(), generator=generator)[:400]
        flat[picks] = 1.0

        blob_stats = kr.residual_stats(blob)
        scattered_stats = kr.residual_stats(scattered)
        # Same number of non-zeros, opposite spatial structure.
        assert blob_stats["occupancy"] == pytest.approx(scattered_stats["occupancy"], rel=0.05)
        assert blob_stats["clustering"] > 0.95
        assert scattered_stats["clustering"] < 0.35

    @pytest.mark.parametrize("kind", ["burst", "bracket", "multiview", "unrelated"])
    def test_every_domain_produces_finite_statistics(self, kind):
        images = kr.make_synthetic_set(kind, n=N, channels=C, height=H, width=W, seed=0)
        _, residuals = kr.anchored_residuals(images)
        stats = kr.residual_stats(residuals)
        assert all(v == v for v in stats.values())          # no NaN
        assert 0.0 <= stats["occupancy"] <= 1.0

    def test_a_burst_is_far_sparser_than_the_unrelated_control(self):
        """The section 1 control case. If this ever inverts, the
        measurement is broken -- an unrelated batch cannot possibly
        benefit from a keyframe."""
        burst = kr.make_synthetic_set("burst", n=N, channels=C, height=H, width=W, seed=0)
        unrelated = kr.make_synthetic_set("unrelated", n=N, channels=C, height=H, width=W, seed=0)

        _, burst_residuals = kr.anchored_residuals(burst)
        _, unrelated_residuals = kr.anchored_residuals(unrelated)

        burst_stats = kr.residual_stats(burst_residuals, threshold=0.02)
        unrelated_stats = kr.residual_stats(unrelated_residuals, threshold=0.02)

        assert burst_stats["occupancy"] < 0.05
        assert unrelated_stats["occupancy"] > 0.5
        # The measured gap is ~8x for this generator (sensor noise of
        # 0.01 against two independent smoothed uniform fields). The
        # bound is deliberately below that: the point is the sign and
        # order of the effect, not the exact ratio, which is a property
        # of make_synthetic_set rather than of any real burst.
        assert burst_stats["mean_abs"] < unrelated_stats["mean_abs"] / 5


class TestAnchoredVersusChained:
    """Section 3: which form wins depends on whether the set is ordered."""

    def test_chained_beats_anchored_on_an_ordered_set(self):
        """A bracket ramps monotonically, so successive differences are
        much smaller than differences against the first image. This is
        the argument for the chained form in the video case."""
        ordered = kr.make_synthetic_set("bracket", n=N, channels=C, height=H, width=W, seed=0)
        _, anchored = kr.anchored_residuals(ordered)
        _, chained = kr.chained_residuals(ordered)
        assert chained.abs().mean() < anchored.abs().mean()

    def test_anchored_does_not_drift(self):
        """Chained reconstruction is a cumulative sum, so rounding error
        accumulates along the chain; anchored reconstruction cannot
        drift because every image is one subtraction from the base."""
        ordered = kr.make_synthetic_set("bracket", n=N, channels=C, height=H, width=W, seed=0)
        keyframe, residuals = kr.anchored_residuals(ordered)
        assert torch.equal(kr.reconstruct_anchored(keyframe, residuals), ordered)


class TestSpatialIndexLayout:
    """Sections 2 and 4: the [Spatial, N] rearrangement, both orientations."""

    @pytest.mark.parametrize("axis,expected", [("W", (H, C, W, N)), ("H", (W, C, H, N))])
    def test_layout_has_the_documented_shape(self, stack, axis, expected):
        assert tuple(kr.to_spatial_index(stack, axis=axis).shape) == expected

    @pytest.mark.parametrize("axis", ["W", "H"])
    def test_layout_round_trips(self, stack, axis):
        view = kr.to_spatial_index(stack, axis=axis)
        assert torch.equal(kr.from_spatial_index(view, axis=axis), stack)

    @pytest.mark.parametrize("axis", ["W", "H"])
    def test_the_last_axis_is_the_image_index(self, stack, axis):
        """A Conv2D kernel's last dimension must run over images, not
        pixels -- that is the entire point of the layout."""
        view = kr.to_spatial_index(stack, axis=axis)
        assert view.shape[-1] == N
        for n in range(N):
            assert not torch.equal(view[..., n], view[..., (n + 1) % N])

    def test_a_conv2d_over_the_layout_spans_spatial_and_index(self, stack):
        """Smoke test of the plan's central layer: a 3x3 kernel on this
        layout mixes 3 spatial positions with 3 images."""
        view = kr.to_spatial_index(stack, axis="W")
        generator = torch.Generator().manual_seed(4)
        kernel = torch.randn(4, C, 3, 3, generator=generator) * 0.1
        out = torch.nn.functional.conv2d(view, kernel)
        assert tuple(out.shape) == (H, 4, W - 2, N - 2)

    def test_rejects_an_unknown_axis(self, stack):
        with pytest.raises(ValueError):
            kr.to_spatial_index(stack, axis="T")
        with pytest.raises(ValueError):
            kr.from_spatial_index(stack, axis="T")


class TestAlignment:
    """Section 15: the step that decides whether the residuals are sparse."""

    def test_phase_correlation_recovers_a_known_shift(self):
        reference = kr.make_synthetic_set("burst", n=1, channels=1, height=H, width=W, seed=5)[0, 0]
        for shift in ((0, 0), (3, 0), (0, -4), (5, 7)):
            shifted = torch.roll(reference, shifts=shift, dims=(-2, -1))
            assert kr.estimate_global_shift(reference, shifted) == shift

    def test_alignment_makes_a_shifted_set_dramatically_sparser(self):
        """The plan's central practical claim: a multi-view set looks
        dense in raw pixels and sparse once registered, even though the
        content barely changed."""
        views = kr.make_synthetic_set("multiview", n=N, channels=C, height=H, width=W, seed=0)

        _, raw = kr.anchored_residuals(views)
        _, aligned = kr.anchored_residuals(kr.align_to_keyframe(views))

        raw_occupancy = kr.residual_stats(raw, threshold=0.02)["occupancy"]
        aligned_occupancy = kr.residual_stats(aligned, threshold=0.02)["occupancy"]

        assert raw_occupancy > 0.3
        assert aligned_occupancy < raw_occupancy / 10

    def test_alignment_leaves_an_already_aligned_set_alone(self):
        burst = kr.make_synthetic_set("burst", n=N, channels=C, height=H, width=W, seed=0)
        torch.testing.assert_close(kr.align_to_keyframe(burst), burst)

    def test_alignment_preserves_shape_and_keyframe(self):
        views = kr.make_synthetic_set("multiview", n=N, channels=C, height=H, width=W, seed=0)
        aligned = kr.align_to_keyframe(views, keyframe_index=2)
        assert aligned.shape == views.shape
        assert torch.equal(aligned[2], views[2])

    def test_shift_estimation_rejects_non_2d_input(self):
        with pytest.raises(ValueError):
            kr.estimate_global_shift(torch.zeros(C, H, W), torch.zeros(C, H, W))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a ROCm GPU")
class TestOnDevice:
    """The identity has to survive fp16 on the real target (RX 7900 XTX).

    Separate from the CPU tests because fp16 accumulation is where an
    exact-in-theory decomposition can quietly stop being exact: the
    residual path sums two rounded convolutions where the baseline
    rounds once.
    """

    def test_residual_path_matches_baseline_in_fp16(self):
        device = torch.device("cuda")
        images = kr.make_synthetic_set("burst", n=N, channels=C, height=H, width=W,
                                       seed=0, device=device, dtype=torch.float16)
        generator = torch.Generator().manual_seed(1)
        weight = (torch.randn(8, C, 3, 3, generator=generator) * 0.1).to(device, torch.float16)
        bias = torch.randn(8, generator=generator).to(device, torch.float16)

        baseline = kr.per_image_conv2d(images, weight, bias)
        residual = kr.keyframe_residual_conv2d(images, weight, bias)
        torch.testing.assert_close(residual, baseline, rtol=2e-2, atol=2e-2)

    def test_statistics_work_on_device(self):
        device = torch.device("cuda")
        images = kr.make_synthetic_set("burst", n=N, channels=C, height=H, width=W, device=device)
        _, residuals = kr.anchored_residuals(images)
        stats = kr.residual_stats(residuals)
        assert 0.0 <= stats["occupancy"] <= 1.0
