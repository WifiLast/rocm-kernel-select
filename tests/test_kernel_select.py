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
