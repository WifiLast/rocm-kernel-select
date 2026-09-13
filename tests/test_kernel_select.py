"""Tests for amd_tuned_torch.kernel_select -- the per-shape contest that lets
stock ROCm win conv2d/conv3d when it is actually faster.

These exercise the selection and caching logic only, with the timing
function stubbed. That is deliberate: the real `_time` needs a GPU and,
worse, its results are hardware- and shape-dependent, so a test that
depended on it would be asserting a benchmark rather than a behaviour.
What matters and is testable anywhere is the policy -- fastest candidate
wins, the decision is cached per shape, a declining candidate loses instead
of breaking the call, and the env kill-switch works.
"""
from __future__ import annotations

import json
import warnings

import pytest
import torch

from amd_tuned_torch import kernel_select


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    # tests/conftest.py forces AMD_TUNED_TORCH_MEASURE_KERNELS=0 so the
    # dispatch-logic suite keeps testing the fixed tier ordering (the
    # contest would call every candidate several times and break its
    # call-count assertions). This suite is the one that DOES test the
    # contest, so it opts back in explicitly.
    monkeypatch.setattr(kernel_select, "_ENABLED", True)
    # This file's candidates return tagged string sentinels ("out:ck"), not
    # real tensors -- correctness verification (see kernel_select's own
    # CORRECTNESS VERIFICATION docstring section) can't meaningfully compare
    # those, so it's disabled here to keep this file testing SELECTION
    # policy in isolation, the same way `timings` isolates it from real GPU
    # timing. TestCorrectnessVerification below tests verification itself,
    # with real tensors, and turns it back on explicitly.
    monkeypatch.setattr(kernel_select, "_VERIFY_ENABLED", False)
    kernel_select.reset()
    yield
    kernel_select.reset()


@pytest.fixture
def timings(monkeypatch):
    """Stubs kernel_select._time with a table keyed by candidate name.

    Returns the call-count dict so tests can assert how often each
    candidate was actually measured.
    """
    calls: dict = {}

    def install(table: dict):
        def fake_time(fn):
            name = fn._name  # set by make_candidates below
            calls[name] = calls.get(name, 0) + 1
            # Mirror the real _time contract: it runs the thunk during
            # warmup and reports None if the candidate declines, so a
            # declining candidate must not appear timeable here either.
            if fn() is None:
                return None
            return table.get(name)
        monkeypatch.setattr(kernel_select, "_time", fake_time)
        return calls

    return install


def make_candidates(names, declines=()):
    """(name, thunk) pairs whose thunks return a tagged sentinel, or None
    for names in `declines` (mimicking the CK tier refusing a problem)."""
    out = []
    for n in names:
        def thunk(n=n):
            return None if n in declines else f"out:{n}"
        thunk._name = n
        out.append((n, thunk))
    return out


def _args():
    x = torch.zeros(1, 4, 8, 8)
    w = torch.zeros(4, 4, 3, 3)
    return x, w, 1, 1, 1


def test_picks_fastest_not_first(timings):
    """The whole point: a slower candidate listed first must not win."""
    calls = timings({"ck": 5.0, "native": 9.0, "stock": 1.0})
    x, w, s, p, d = _args()
    out = kernel_select.pick("conv2d", x, w, s, p, d,
                           make_candidates(["ck", "native", "stock"]))
    assert out == "out:stock"
    # every candidate measured exactly once
    assert calls == {"ck": 1, "native": 1, "stock": 1}


def test_decision_is_cached_per_shape(timings):
    calls = timings({"ck": 1.0, "stock": 4.0})
    x, w, s, p, d = _args()
    for _ in range(5):
        assert kernel_select.pick("conv2d", x, w, s, p, d,
                                make_candidates(["ck", "stock"])) == "out:ck"
    # measured on the first call only
    assert calls == {"ck": 1, "stock": 1}
    assert list(kernel_select.debug_winners().values()) == ["ck"]


def test_different_shapes_measured_separately(timings):
    calls = timings({"ck": 1.0, "stock": 4.0})
    w = torch.zeros(4, 4, 3, 3)
    kernel_select.pick("conv2d", torch.zeros(1, 4, 8, 8), w, 1, 1, 1,
                     make_candidates(["ck", "stock"]))
    kernel_select.pick("conv2d", torch.zeros(1, 4, 16, 16), w, 1, 1, 1,
                     make_candidates(["ck", "stock"]))
    assert calls == {"ck": 2, "stock": 2}
    assert len(kernel_select.debug_winners()) == 2


def test_same_shape_different_stride_is_a_different_key(timings):
    """stride/padding/dilation change the work, so they must key separately
    -- otherwise a strided conv inherits a decision made for stride 1."""
    timings({"ck": 1.0, "stock": 4.0})
    x, w, _, p, d = _args()
    kernel_select.pick("conv2d", x, w, 1, p, d, make_candidates(["ck", "stock"]))
    kernel_select.pick("conv2d", x, w, 2, p, d, make_candidates(["ck", "stock"]))
    assert len(kernel_select.debug_winners()) == 2


def test_conv2d_and_conv3d_keys_do_not_collide(timings):
    timings({"ck": 1.0, "stock": 4.0})
    x, w, s, p, d = _args()
    kernel_select.pick("conv2d", x, w, s, p, d, make_candidates(["ck", "stock"]))
    kernel_select.pick("conv3d", x, w, s, p, d, make_candidates(["ck", "stock"]))
    assert len(kernel_select.debug_winners()) == 2


def test_unrunnable_candidate_loses_instead_of_winning(timings):
    """_time returns None for a candidate that can't run; it must be skipped,
    not treated as infinitely fast."""
    timings({"ck": None, "stock": 7.0})
    x, w, s, p, d = _args()
    out = kernel_select.pick("conv2d", x, w, s, p, d,
                           make_candidates(["ck", "stock"]))
    assert out == "out:stock"


def test_cached_winner_that_later_declines_is_remeasured(timings):
    """A cached winner returning None (e.g. CK hitting a workspace
    allocation failure) must fall back and re-decide, not return None to the
    caller -- returning None would surface as a wrong result, not a slow one."""
    calls = timings({"ck": 1.0, "stock": 4.0})
    x, w, s, p, d = _args()
    assert kernel_select.pick("conv2d", x, w, s, p, d,
                            make_candidates(["ck", "stock"])) == "out:ck"
    assert calls == {"ck": 1, "stock": 1}
    # now ck declines
    out = kernel_select.pick("conv2d", x, w, s, p, d,
                           make_candidates(["ck", "stock"], declines=("ck",)))
    assert out == "out:stock"
    assert kernel_select.debug_winners() and \
        list(kernel_select.debug_winners().values()) == ["stock"]


def test_all_candidates_unrunnable_returns_none(timings):
    """pick() returning None tells the caller to run its own fallback."""
    timings({"ck": None, "stock": None})
    x, w, s, p, d = _args()
    assert kernel_select.pick("conv2d", x, w, s, p, d,
                            make_candidates(["ck", "stock"])) is None


def test_env_kill_switch(monkeypatch):
    """AMD_TUNED_TORCH_MEASURE_KERNELS=0 must restore the old behaviour, which
    the dispatch implements by treating pick() as unavailable."""
    monkeypatch.setattr(kernel_select, "_ENABLED", False)
    assert not kernel_select.enabled()
    x, w, s, p, d = _args()
    assert kernel_select.pick("conv2d", x, w, s, p, d,
                            make_candidates(["ck", "stock"])) is None


def test_reset_forces_remeasurement(timings):
    calls = timings({"ck": 1.0, "stock": 4.0})
    x, w, s, p, d = _args()
    kernel_select.pick("conv2d", x, w, s, p, d, make_candidates(["ck", "stock"]))
    kernel_select.reset()
    assert kernel_select.debug_winners() == {}
    kernel_select.pick("conv2d", x, w, s, p, d, make_candidates(["ck", "stock"]))
    assert calls == {"ck": 2, "stock": 2}


# --- generic (non-conv) key path, used by group_norm --------------------

def test_pick_key_runs_a_contest_for_a_custom_key(timings):
    """group_norm doesn't key on (stride, padding, dilation), so it uses
    pick_key with its own key. Same policy must apply."""
    calls = timings({"native": 4.0, "stock": 1.0})
    key = (torch.float16, (32, 128, 64, 64), 32)
    out = kernel_select.pick_key("group_norm", key,
                                 make_candidates(["native", "stock"]))
    assert out == "out:stock"
    assert calls == {"native": 1, "stock": 1}


def test_pick_key_caches_and_cached_key_reads_it_back(timings):
    calls = timings({"native": 1.0, "stock": 4.0})
    key = (torch.float16, (32, 128, 64, 64), 32)
    assert kernel_select.cached_key("group_norm", key) is None
    kernel_select.pick_key("group_norm", key, make_candidates(["native", "stock"]))
    assert kernel_select.cached_key("group_norm", key) == "native"
    # a different group count is a different contest
    other = (torch.float16, (32, 128, 64, 64), 16)
    assert kernel_select.cached_key("group_norm", other) is None
    kernel_select.pick_key("group_norm", other, make_candidates(["native", "stock"]))
    assert calls == {"native": 2, "stock": 2}


def test_conv_and_group_norm_keys_do_not_collide(timings):
    timings({"native": 1.0, "stock": 4.0})
    x, w, s, p, d = _args()
    kernel_select.pick("conv2d", x, w, s, p, d, make_candidates(["native", "stock"]))
    kernel_select.pick_key("group_norm", (torch.float16, (1, 4, 8, 8), 2),
                           make_candidates(["native", "stock"]))
    assert len(kernel_select.debug_winners()) == 2


def test_pick_key_honours_the_kill_switch(monkeypatch):
    monkeypatch.setattr(kernel_select, "_ENABLED", False)
    assert kernel_select.pick_key("group_norm", (torch.float16, (1, 4, 8, 8), 2),
                                  make_candidates(["native", "stock"])) is None


def _tagged(name, fn):
    fn._name = name
    return name, fn


class TestCorrectnessVerification:
    """kernel_select._contest additionally verifies a non-reference
    candidate's OUTPUT against the reference before crowning it winner --
    see the module docstring's CORRECTNESS VERIFICATION section. Real
    tensors here, unlike the rest of this file's string sentinels, since
    the whole point is exercising torch.allclose; _VERIFY_ENABLED is
    turned back on per-test (the file-wide `_clean` fixture turns it off
    for everything else -- see that fixture's own comment)."""

    def _reference(self):
        return torch.ones(4, 4)

    def _correct(self):
        return torch.ones(4, 4) + 1e-6  # well within float32 tolerance

    def _wrong(self):
        return torch.zeros(4, 4)  # far outside any real tolerance

    def test_correct_fast_candidate_wins(self, monkeypatch, timings):
        monkeypatch.setattr(kernel_select, "_VERIFY_ENABLED", True)
        ref, correct = self._reference(), self._correct()
        candidates = [_tagged("fast", lambda: correct), _tagged("stock", lambda: ref)]
        timings({"fast": 1.0, "stock": 5.0})
        out = kernel_select.pick_key("linear", ("k1",), candidates)
        assert out is correct
        assert kernel_select.cached_key("linear", ("k1",)) == "fast"

    def test_wrong_fast_candidate_is_excluded_and_reference_used(self, monkeypatch, timings):
        monkeypatch.setattr(kernel_select, "_VERIFY_ENABLED", True)
        ref, wrong = self._reference(), self._wrong()
        candidates = [_tagged("fast", lambda: wrong), _tagged("stock", lambda: ref)]
        timings({"fast": 1.0, "stock": 5.0})
        with pytest.warns(UserWarning, match="didn't match"):
            out = kernel_select.pick_key("linear", ("k2",), candidates)
        assert out is ref
        assert kernel_select.cached_key("linear", ("k2",)) == "stock"
        assert "fast" in kernel_select.debug_bad_candidates()[("linear", "k2")]

    def test_excluded_candidate_never_timed_again(self, monkeypatch, timings):
        monkeypatch.setattr(kernel_select, "_VERIFY_ENABLED", True)
        ref, wrong = self._reference(), self._wrong()
        candidates = [_tagged("fast", lambda: wrong), _tagged("stock", lambda: ref)]
        calls = timings({"fast": 1.0, "stock": 5.0})
        with pytest.warns(UserWarning):
            kernel_select.pick_key("linear", ("k3",), candidates)
        assert calls["fast"] == 1
        kernel_select.pick_key("linear", ("k3",), candidates)  # cached "stock" winner path
        assert calls["fast"] == 1  # never re-timed once verified bad

    def test_reference_never_needs_verification(self, monkeypatch, timings):
        monkeypatch.setattr(kernel_select, "_VERIFY_ENABLED", True)
        ref = self._reference()
        candidates = [_tagged("stock", lambda: ref)]
        timings({"stock": 1.0})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = kernel_select.pick_key("linear", ("k4",), candidates)
        assert out is ref

    def test_verify_disabled_trusts_fastest_regardless(self, monkeypatch, timings):
        monkeypatch.setattr(kernel_select, "_VERIFY_ENABLED", False)
        ref, wrong = self._reference(), self._wrong()
        candidates = [_tagged("fast", lambda: wrong), _tagged("stock", lambda: ref)]
        timings({"fast": 1.0, "stock": 5.0})
        out = kernel_select.pick_key("linear", ("k5",), candidates)
        assert out is wrong  # old behaviour: fastest wins unconditionally


class TestMagnitudeRelativeVerification:
    """_verify's tolerance is relative to the reference output's own RMS,
    not absolute. A fixed atol rejected three correct, faster kernels on
    gfx1100 -- always on a handful of near-zero elements out of millions
    (see _verify's docstring for the measurements) -- and because a
    rejection is a permanent per-shape blacklist, each one permanently
    lost the speedup: 1.8x for CK conv3d fp16, 157x for FFT conv3d fp32,
    2.24x for CK conv2d fp16 on a real downsampling conv."""

    def test_near_zero_elements_of_a_large_output_pass(self):
        """The shape of all three real failures: a big output, agreement
        that is excellent relative to its magnitude, and a few near-zero
        elements whose absolute error exceeds a fixed atol."""
        # fp16, as measured: CK conv2d on 2x640x128x64->640 k3 stride2,
        # output RMS 75.4, worst disagreement 0.375 on a near-zero element.
        ref = (torch.randn(4096) * 75.0).half()
        ref[0] = 0.06
        cand = ref.clone()
        cand[0] += 0.375
        rtol, atol = kernel_select._TOLERANCES[torch.float16]
        assert not torch.allclose(cand.float(), ref.float(), rtol=rtol, atol=atol)  # old bar
        assert kernel_select._verify(cand, ref)                                     # new bar

    def test_the_relative_bar_tracks_the_dtype_it_was_calibrated_for(self):
        """fp32's rtol is 100x tighter than fp16's, so the same absolute
        disagreement that is fine for an fp16 accumulation is not fine for
        an fp32 one -- the scaling must not flatten that distinction."""
        ref = (torch.randn(4096) * 75.0)
        cand = ref.clone()
        cand[0] += 0.375
        assert not kernel_select._verify(cand, ref)                  # fp32: rejected
        assert kernel_select._verify(cand.half(), ref.half())        # fp16: accepted

    def test_a_wrong_candidate_is_still_rejected(self):
        ref = torch.randn(4096) * 75.0
        assert not kernel_select._verify(torch.zeros_like(ref), ref)
        assert not kernel_select._verify(ref * 1.5, ref)

    def test_a_subtly_wrong_candidate_is_still_rejected(self):
        """Order RMS/sqrt(C) -- a dropped channel's worth of error -- must
        stay well above the scaled bar, or the bar is useless."""
        ref = (torch.randn(64, 64) * 75.0)
        cand = ref.clone()
        cand[:, 0] = 0.0                    # one channel of 64 dropped
        assert not kernel_select._verify(cand, ref)

    def test_scale_floor_is_never_below_the_fixed_tolerance(self):
        """An all-but-zero output gives a tiny RMS; the fixed atol is a
        floor, so the bar can't collapse to zero and start rejecting
        ordinary rounding."""
        ref = torch.zeros(4096)
        cand = torch.zeros(4096)
        cand[0] = 5e-3                      # under fp32's atol floor of 1e-5? no
        assert kernel_select._reference_scale(ref) is None
        assert not kernel_select._verify(cand, ref)
        cand[0] = 1e-6
        assert kernel_select._verify(cand, ref)

    def test_reference_scale_samples_without_upcasting_everything(self):
        x = torch.full((4096,), 3.0)
        assert abs(kernel_select._reference_scale(x) - 3.0) < 1e-5
        assert kernel_select._reference_scale(torch.zeros(8)) is None
        assert kernel_select._reference_scale("not a tensor") is None


class TestVerifyBarVersioning:
    """A blacklist written under a superseded bar is a stale judgement, not
    evidence: it must not outlive the bar change, and the winner it caused
    must be re-contested rather than served from cache forever."""

    def test_stale_blacklist_and_its_winner_are_dropped_on_load(self, tmp_path, monkeypatch):
        key = ("conv2d", "shape-a")
        encoded = json.dumps([kernel_select._encode(x) for x in key])
        other = ("conv2d", "shape-b")
        encoded_other = json.dumps([kernel_select._encode(x) for x in other])
        path = tmp_path / "cache.json"
        path.write_text(json.dumps({
            "verify_bar": kernel_select._VERIFY_BAR_VERSION - 1,
            "winners": {encoded: "stock", encoded_other: "ck"},
            "bad": {encoded: ["ck"]},
        }))
        monkeypatch.setattr(kernel_select, "_cache_path", lambda: str(path))
        kernel_select._winners.clear()
        kernel_select._bad_candidates.clear()
        kernel_select._load_disk_cache()
        assert kernel_select._bad_candidates == {}          # judgement dropped
        assert key not in kernel_select._winners            # so is its winner
        assert kernel_select._winners.get(other) == "ck"    # untouched
        kernel_select._winners.clear()

    def test_current_bar_keeps_the_blacklist(self, tmp_path, monkeypatch):
        key = ("conv2d", "shape-c")
        encoded = json.dumps([kernel_select._encode(x) for x in key])
        path = tmp_path / "cache.json"
        path.write_text(json.dumps({
            "verify_bar": kernel_select._VERIFY_BAR_VERSION,
            "winners": {encoded: "stock"},
            "bad": {encoded: ["ck"]},
        }))
        monkeypatch.setattr(kernel_select, "_cache_path", lambda: str(path))
        kernel_select._winners.clear()
        kernel_select._bad_candidates.clear()
        kernel_select._load_disk_cache()
        assert kernel_select._bad_candidates.get(key) == {"ck"}
        assert kernel_select._winners.get(key) == "stock"
        kernel_select._winners.clear()
        kernel_select._bad_candidates.clear()


class TestOomIsNotACorrectnessVerdict:
    """An out-of-memory failure inside the verification step used to be
    laundered into a permanent, disk-persisted "this kernel is numerically
    wrong" blacklist entry.

    The chain: _verify upcast BOTH outputs to fp32 with .float(), so
    verifying a 2,000,000x1024 fp16 linear asked for 8.19 GB twice. On a
    24 GB card already holding a 23 GB model that OOMs;
    torch.OutOfMemoryError is a RuntimeError SUBCLASS, so _verify's
    except-clause caught it and returned False; _contest read False as
    "disagrees" and blacklisted the candidate forever. Every `linear` entry
    in the shipped blacklist was of exactly that kind -- M from 174593 to
    2000000, i.e. precisely the shapes whose fp32 copies are gigabytes.
    """

    def _oom(self):
        cls = getattr(torch, "OutOfMemoryError", None)
        if isinstance(cls, type):
            return cls("CUDA out of memory. Tried to allocate 8.19 GiB")
        return RuntimeError("CUDA out of memory. Tried to allocate 8.19 GiB")

    def test_oom_is_recognised_however_torch_spells_it(self):
        assert kernel_select._is_oom(self._oom())
        assert kernel_select._is_oom(RuntimeError("HIP out of memory"))
        assert not kernel_select._is_oom(RuntimeError("invalid configuration argument"))
        assert not kernel_select._is_oom(TypeError("bad dtype"))

    def test_verify_returns_none_not_false_on_oom(self, monkeypatch):
        """None means INDETERMINATE. False would be a verdict, and a verdict
        is what gets written to the permanent blacklist."""
        def boom(*args, **kwargs):
            raise self._oom()

        monkeypatch.setattr(torch, "allclose", boom)
        a = torch.ones(8, 4)
        assert kernel_select._verify(a, a.clone()) is None

    def test_verify_still_returns_false_for_a_real_mismatch(self):
        a = torch.zeros(8, 4)
        b = torch.ones(8, 4)
        assert kernel_select._verify(a, b) is False

    def test_verify_still_returns_true_for_agreement(self):
        a = torch.randn(8, 4)
        assert kernel_select._verify(a, a.clone()) is True

    def test_verify_returns_false_for_a_shape_mismatch(self):
        """Still a verdict about the candidate, not a resource problem."""
        assert kernel_select._verify(torch.zeros(8, 4), torch.zeros(8, 5)) is False

    def test_verify_never_upcasts_the_whole_tensor(self, monkeypatch):
        """The actual fix: peak comparison memory must be bounded by the
        chunk size, not by the output size. Recorded by watching how large
        each .float() the comparison performs actually is."""
        seen = []
        original = torch.Tensor.float

        def spy(self, *args, **kwargs):
            seen.append(self.numel())
            return original(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "float", spy)
        rows = 4 * (kernel_select._VERIFY_CHUNK_ELEMENTS // 16) + 7
        a = torch.zeros(rows, 16)
        kernel_select._verify(a, a.clone())

        assert seen, "the comparison should have upcast something"
        assert max(seen) <= kernel_select._VERIFY_CHUNK_ELEMENTS, (
            f"upcast {max(seen)} elements at once, chunk limit is "
            f"{kernel_select._VERIFY_CHUNK_ELEMENTS}"
        )
        assert max(seen) < a.numel(), "a whole-tensor upcast is the bug being fixed"

    def test_chunked_comparison_still_catches_a_mismatch_in_the_last_chunk(self):
        """Chunking is exact, not a sample -- an element wrong anywhere must
        still fail, including in a final short chunk."""
        rows = 3 * (kernel_select._VERIFY_CHUNK_ELEMENTS // 16) + 5
        a = torch.zeros(rows, 16)
        b = a.clone()
        b[-1, -1] = 1e9
        assert kernel_select._verify(a, b) is False

    def test_contest_does_not_blacklist_on_an_oom(self, monkeypatch):
        """The whole point. An OOM during verification must leave the
        blacklist untouched -- an entry there is permanent and persisted."""
        key = ("linear", torch.float16, 2000000, (1024, 1024), False)
        monkeypatch.setattr(kernel_select, "_winners", {})
        monkeypatch.setattr(kernel_select, "_bad_candidates", {})
        monkeypatch.setattr(kernel_select, "_save_disk_cache", lambda: None)
        monkeypatch.setattr(kernel_select, "_time", lambda fn: 1.0)
        monkeypatch.setattr(kernel_select, "_VERIFY_ENABLED", True)
        monkeypatch.setattr(kernel_select, "_verify", lambda *a, **k: None)

        fast = torch.ones(4, 4)
        reference = torch.zeros(4, 4)
        with pytest.warns(UserWarning, match="ran out of memory verifying"):
            out = kernel_select._contest(
                key, [("hipblaslt", lambda: fast), ("stock", lambda: reference)])

        assert out is reference, "the reference is correct by convention -- use it"
        assert kernel_select._bad_candidates == {}, "no judgement may be recorded"
        assert kernel_select._winners == {}, "nor a winner chosen under pressure"

    def test_contest_still_blacklists_a_genuine_mismatch(self, monkeypatch):
        """The guard must not disarm the real correctness check."""
        key = ("linear", torch.float16, 128, (64, 64), False)
        monkeypatch.setattr(kernel_select, "_winners", {})
        monkeypatch.setattr(kernel_select, "_bad_candidates", {})
        monkeypatch.setattr(kernel_select, "_save_disk_cache", lambda: None)
        monkeypatch.setattr(kernel_select, "_time", lambda fn: 1.0)
        monkeypatch.setattr(kernel_select, "_VERIFY_ENABLED", True)

        wrong = torch.ones(4, 4)
        reference = torch.zeros(4, 4)
        with pytest.warns(UserWarning, match="didn't match"):
            out = kernel_select._contest(
                key, [("hipblaslt", lambda: wrong), ("stock", lambda: reference)])

        assert out is reference
        assert "hipblaslt" in kernel_select._bad_candidates[key]

    def test_a_candidate_oom_does_not_escape_the_contest(self, monkeypatch):
        """An OOM raised by a candidate thunk used to propagate out and kill
        the application, even though the reference could still serve."""
        key = ("linear", torch.float16, 2000000, (1024, 1024), False)
        monkeypatch.setattr(kernel_select, "_winners", {})
        monkeypatch.setattr(kernel_select, "_bad_candidates", {})
        monkeypatch.setattr(kernel_select, "_save_disk_cache", lambda: None)
        monkeypatch.setattr(kernel_select, "_time", lambda fn: 1.0)
        monkeypatch.setattr(kernel_select, "_VERIFY_ENABLED", True)

        reference = torch.zeros(4, 4)

        def oom_thunk():
            raise self._oom()

        out = kernel_select._contest(
            key, [("hipblaslt", oom_thunk), ("stock", lambda: reference)])
        assert out is reference
        assert kernel_select._bad_candidates == {}

    def test_a_non_oom_runtime_error_from_a_candidate_still_propagates(self, monkeypatch):
        """The guard is for memory pressure only -- a genuine bug in a
        candidate must not be swallowed into a silent fallback."""
        key = ("linear", torch.float16, 128, (64, 64), False)
        monkeypatch.setattr(kernel_select, "_winners", {})
        monkeypatch.setattr(kernel_select, "_bad_candidates", {})
        monkeypatch.setattr(kernel_select, "_save_disk_cache", lambda: None)
        monkeypatch.setattr(kernel_select, "_time", lambda fn: 1.0)
        monkeypatch.setattr(kernel_select, "_VERIFY_ENABLED", True)

        def broken():
            raise RuntimeError("invalid configuration argument")

        with pytest.raises(RuntimeError, match="invalid configuration"):
            kernel_select._contest(
                key, [("hipblaslt", broken), ("stock", lambda: torch.zeros(4, 4))])

    def test_the_bar_version_bump_drops_the_old_oom_written_entries(self):
        """Entries written at bar version 2 were judgements made by a
        comparison that could OOM into a False, so they are not evidence
        under the current bar. The module's existing versioning is what
        retires them."""
        assert kernel_select._VERIFY_BAR_VERSION >= 3
