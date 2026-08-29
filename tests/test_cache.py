"""Tests for amd_tuned_torch.cache -- the generic similarity-gated memoization
cache (SimilarityCache / similarity_cached). Pure PyTorch/Python, no
aiter, TransformerEngine, or GPU required -- these tests run identically
on CPU.

Run with:

    python -m pytest source/cmp_ext_turing/tests
"""
from __future__ import annotations

import importlib
import logging
import os
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

import amd_tuned_torch.cache as cache_module


class TestEnvFlag:
    def test_default_off_when_unset(self, monkeypatch):
        monkeypatch.delenv("AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE", raising=False)
        assert cache_module._env_flag("AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE") is False

    @pytest.mark.parametrize("value", ["0", "", "false", "False"])
    def test_off_for_falsy_strings(self, monkeypatch, value):
        monkeypatch.setenv("AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE", value)
        assert cache_module._env_flag("AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE") is False

    @pytest.mark.parametrize("value", ["1", "true", "yes"])
    def test_on_for_other_strings(self, monkeypatch, value):
        monkeypatch.setenv("AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE", value)
        assert cache_module._env_flag("AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE") is True


class TestConstruction:
    def test_reads_env_flag_by_default(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE", "1")
        assert cache_module.SimilarityCache().enabled is True
        monkeypatch.setenv("AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE", "0")
        assert cache_module.SimilarityCache().enabled is False

    def test_explicit_enabled_overrides_env(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE", "0")
        assert cache_module.SimilarityCache(enabled=True).enabled is True
        monkeypatch.setenv("AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE", "1")
        assert cache_module.SimilarityCache(enabled=False).enabled is False


class TestIsCloseEnough:
    def test_false_with_no_previous_input(self):
        sc = cache_module.SimilarityCache(thresh=0.5, enabled=True)
        assert sc._is_close_enough(torch.ones(2, 4)) is False

    def test_false_on_shape_mismatch(self):
        sc = cache_module.SimilarityCache(thresh=0.5, enabled=True)
        sc.previous_input = torch.ones(2, 4)
        assert sc._is_close_enough(torch.ones(3, 4)) is False

    def test_true_when_within_threshold(self):
        sc = cache_module.SimilarityCache(thresh=0.5, enabled=True)
        sc.previous_input = torch.ones(2, 4)
        assert sc._is_close_enough(torch.ones(2, 4) * 1.1) is True

    def test_false_when_outside_threshold(self):
        sc = cache_module.SimilarityCache(thresh=0.05, enabled=True)
        sc.previous_input = torch.ones(2, 4)
        assert sc._is_close_enough(torch.ones(2, 4) * 2.0) is False

    def test_false_when_distance_computation_raises_instead_of_propagating(self, monkeypatch):
        # Simulates whatever real-world dtype/device combination might
        # raise inside the distance math (exact cases vary by PyTorch
        # version) -- the contract is: never let that crash a call that
        # would otherwise succeed, treat it as a cache miss instead.
        sc = cache_module.SimilarityCache(thresh=0.5, enabled=True)
        sc.previous_input = torch.ones(2, 4)

        def _boom(self):
            raise RuntimeError("simulated dtype/device mismatch")

        monkeypatch.setattr(torch.Tensor, "abs", _boom)
        assert sc._is_close_enough(torch.ones(2, 4)) is False


class TestCall:
    def test_disabled_always_calls_fn(self):
        sc = cache_module.SimilarityCache(thresh=1e9, enabled=False)
        calls = []

        def fn(x):
            calls.append(x)
            return x * 2

        x = torch.ones(2, 4)
        sc.call(fn, x)
        sc.call(fn, x)
        assert len(calls) == 2

    def test_non_tensor_input_always_calls_fn(self):
        sc = cache_module.SimilarityCache(enabled=True, thresh=1e9)
        calls = []

        def fn(x):
            calls.append(x)
            return x

        sc.call(fn, "not a tensor")
        sc.call(fn, "not a tensor")
        assert len(calls) == 2

    def test_enabled_skips_call_when_similar(self):
        sc = cache_module.SimilarityCache(thresh=0.5, enabled=True)
        calls = []

        def fn(x):
            calls.append(x)
            return x * 2

        out1 = sc.call(fn, torch.ones(2, 4))
        out2 = sc.call(fn, torch.ones(2, 4) * 1.05)  # within thresh -> should skip
        assert len(calls) == 1
        assert out2 is out1

    def test_enabled_calls_again_when_dissimilar(self):
        sc = cache_module.SimilarityCache(thresh=0.1, enabled=True)
        calls = []

        def fn(x):
            calls.append(x)
            return x * 2

        sc.call(fn, torch.ones(2, 4))
        sc.call(fn, torch.ones(2, 4) * 5.0)  # far outside thresh -> real call
        assert len(calls) == 2

    def test_passes_through_extra_args_and_kwargs(self):
        sc = cache_module.SimilarityCache(thresh=1e9, enabled=False)
        received = {}

        def fn(x, y, z=None):
            received["y"] = y
            received["z"] = z
            return x

        sc.call(fn, torch.ones(2, 4), "aux", z="kw")
        assert received == {"y": "aux", "z": "kw"}

    def test_cached_input_is_detached_clone_not_the_live_reference(self):
        sc = cache_module.SimilarityCache(thresh=0.5, enabled=True)

        def fn(x):
            return x * 2

        x = torch.ones(2, 4)
        sc.call(fn, x)
        x.add_(100.0)  # mutate the caller's own tensor in place afterward
        # the cache's comparison reference must be unaffected by that mutation
        assert torch.equal(sc.previous_input, torch.ones(2, 4))


class TestMaxBytes:
    def test_none_means_no_limit(self):
        sc = cache_module.SimilarityCache(thresh=0.5, enabled=True, max_bytes=None)
        assert sc._exceeds_size_limit(torch.ones(1000, 1000)) is False

    def test_bypasses_caching_for_oversized_input(self):
        # float32 (2, 4) tensor = 32 bytes; cap it below that.
        sc = cache_module.SimilarityCache(thresh=1e9, enabled=True, max_bytes=16)
        calls = []

        def fn(x):
            calls.append(x)
            return x * 2

        sc.call(fn, torch.ones(2, 4))
        sc.call(fn, torch.ones(2, 4))  # would be a hit if not oversized
        assert len(calls) == 2
        assert sc.previous_input is None
        assert sc.previous_output is None

    def test_undersized_input_still_caches_normally(self):
        sc = cache_module.SimilarityCache(thresh=0.5, enabled=True, max_bytes=1024)
        calls = []

        def fn(x):
            calls.append(x)
            return x * 2

        sc.call(fn, torch.ones(2, 4))
        sc.call(fn, torch.ones(2, 4) * 1.01)  # within thresh -> should skip
        assert len(calls) == 1

    def test_oversized_call_does_not_clobber_existing_small_cache(self):
        sc = cache_module.SimilarityCache(thresh=1e9, enabled=True, max_bytes=16)
        sc.call(lambda x: x * 2, torch.ones(2))  # 8 bytes, fits -> cached
        assert sc.previous_input is not None
        sc.call(lambda x: x * 2, torch.ones(1000))  # oversized -> bypassed
        assert torch.equal(sc.previous_input, torch.ones(2))


class TestMinBytes:
    def test_none_means_no_floor(self):
        sc = cache_module.SimilarityCache(thresh=0.5, enabled=True, min_bytes=None)
        assert sc._below_size_limit(torch.ones(1)) is False

    def test_bypasses_caching_for_undersized_input(self):
        # float32 (2,) tensor = 8 bytes; floor it above that.
        sc = cache_module.SimilarityCache(thresh=1e9, enabled=True, min_bytes=1024)
        calls = []

        def fn(x):
            calls.append(x)
            return x * 2

        sc.call(fn, torch.ones(2))
        sc.call(fn, torch.ones(2))  # would be a hit if not undersized
        assert len(calls) == 2
        assert sc.previous_input is None
        assert sc.previous_output is None

    def test_oversized_relative_to_floor_still_caches_normally(self):
        sc = cache_module.SimilarityCache(thresh=0.5, enabled=True, min_bytes=16)
        calls = []

        def fn(x):
            calls.append(x)
            return x * 2

        sc.call(fn, torch.ones(1000))
        sc.call(fn, torch.ones(1000) * 1.01)  # within thresh -> should skip
        assert len(calls) == 1

    def test_undersized_call_does_not_clobber_existing_cache(self):
        sc = cache_module.SimilarityCache(thresh=1e9, enabled=True, min_bytes=1024)
        sc.call(lambda x: x * 2, torch.ones(1000))  # 4000 bytes, above floor -> cached
        assert sc.previous_input is not None
        sc.call(lambda x: x * 2, torch.ones(2))  # 8 bytes, below floor -> bypassed
        assert torch.equal(sc.previous_input, torch.ones(1000))

    def test_min_and_max_bytes_combine_as_a_window(self):
        sc = cache_module.SimilarityCache(thresh=1e9, enabled=True, min_bytes=16, max_bytes=4096)
        calls = []

        def fn(x):
            calls.append(x)
            return x * 2

        sc.call(fn, torch.ones(2))       # 8 bytes -- below floor, bypassed
        sc.call(fn, torch.ones(2000))    # 8000 bytes -- above ceiling, bypassed
        sc.call(fn, torch.ones(100))     # 400 bytes -- within window, cached
        sc.call(fn, torch.ones(100) * 1.0)  # identical -> hit
        assert len(calls) == 3  # two bypassed real calls + one real call before the hit


class TestMaxConsecutiveMisses:
    def test_none_never_auto_disables(self):
        sc = cache_module.SimilarityCache(thresh=1e-9, enabled=True, max_consecutive_misses=None)
        for i in range(20):
            sc.call(lambda x: x * 2, torch.ones(2, 4) * i)  # always dissimilar -> always a miss
        assert sc._auto_disabled is False

    def test_auto_disables_after_streak_of_misses(self):
        sc = cache_module.SimilarityCache(thresh=1e-9, enabled=True, max_consecutive_misses=5)
        for i in range(5):
            sc.call(lambda x: x * 2, torch.ones(2, 4) * i)  # 5 consecutive dissimilar calls
        assert sc._auto_disabled is True

    def test_not_yet_disabled_before_streak_reached(self):
        sc = cache_module.SimilarityCache(thresh=1e-9, enabled=True, max_consecutive_misses=5)
        for i in range(4):
            sc.call(lambda x: x * 2, torch.ones(2, 4) * i)
        assert sc._auto_disabled is False

    def test_hit_resets_the_miss_streak(self):
        sc = cache_module.SimilarityCache(thresh=0.5, enabled=True, max_consecutive_misses=3)
        sc.call(lambda x: x * 2, torch.ones(2, 4) * 1)   # miss 1
        sc.call(lambda x: x * 2, torch.ones(2, 4) * 100)  # miss 2
        sc.call(lambda x: x * 2, torch.ones(2, 4) * 100 * 1.01)  # within thresh -> hit, resets streak
        sc.call(lambda x: x * 2, torch.ones(2, 4) * 200)  # miss (streak restarts at 1)
        sc.call(lambda x: x * 2, torch.ones(2, 4) * 300)  # miss (streak = 2, still < 3)
        assert sc._auto_disabled is False

    def test_auto_disabled_calls_still_compute_correctly(self):
        sc = cache_module.SimilarityCache(thresh=1e-9, enabled=True, max_consecutive_misses=2)
        calls = []

        def fn(x):
            calls.append(x)
            return x * 2

        sc.call(fn, torch.ones(2, 4) * 1)  # miss 1
        sc.call(fn, torch.ones(2, 4) * 2)  # miss 2 -> trips breaker
        assert sc._auto_disabled is True
        out = sc.call(fn, torch.ones(2, 4) * 3)  # breaker tripped -> real call, no distance check
        assert torch.equal(out, torch.ones(2, 4) * 6)
        assert len(calls) == 3

    def test_auto_disabled_skips_the_distance_check(self):
        sc = cache_module.SimilarityCache(thresh=1e-9, enabled=True, max_consecutive_misses=2)
        sc.call(lambda x: x * 2, torch.ones(2, 4) * 1)
        sc.call(lambda x: x * 2, torch.ones(2, 4) * 2)
        assert sc._auto_disabled is True

        def boom(self, x):
            raise AssertionError("_is_close_enough must not be called once auto-disabled")

        sc._is_close_enough = boom.__get__(sc, cache_module.SimilarityCache)
        sc.call(lambda x: x * 2, torch.ones(2, 4) * 3)  # must not raise

    def test_auto_disabled_still_uses_compiled_fn(self, monkeypatch):
        import amd_tuned_torch.torch_compile as torch_compile_module

        compiled = MagicMock(side_effect=lambda x: x * 2)
        monkeypatch.setattr(torch_compile_module, "compile_fn", MagicMock(return_value=compiled))
        monkeypatch.setattr(torch_compile_module, "available", lambda: True)

        sc = cache_module.SimilarityCache(thresh=1e-9, enabled=True, max_consecutive_misses=2, compile=True)
        sc.call(lambda x: x * 2, torch.ones(2, 4) * 1)
        sc.call(lambda x: x * 2, torch.ones(2, 4) * 2)
        assert sc._auto_disabled is True
        sc.call(lambda x: x * 2, torch.ones(2, 4) * 3)
        assert compiled.call_count == 3  # every real call, including post-trip ones, uses the compiled fn

    def test_oversized_bypass_does_not_count_toward_streak(self):
        sc = cache_module.SimilarityCache(thresh=1e-9, enabled=True, max_bytes=16, max_consecutive_misses=2)
        for _ in range(10):
            sc.call(lambda x: x * 2, torch.ones(1000))  # always oversized -> bypassed, never a "miss"
        assert sc._auto_disabled is False

    def test_reset_re_arms_the_breaker(self):
        sc = cache_module.SimilarityCache(thresh=1e-9, enabled=True, max_consecutive_misses=2)
        sc.call(lambda x: x * 2, torch.ones(2, 4) * 1)
        sc.call(lambda x: x * 2, torch.ones(2, 4) * 2)
        assert sc._auto_disabled is True
        sc.reset()
        assert sc._auto_disabled is False
        assert sc._consecutive_misses == 0

    def test_trip_reason_records_consecutive_misses(self):
        sc = cache_module.SimilarityCache(thresh=1e-9, enabled=True, max_consecutive_misses=3)
        sc.call(lambda x: x * 2, torch.ones(2, 4) * 1)
        sc.call(lambda x: x * 2, torch.ones(2, 4) * 2)
        sc.call(lambda x: x * 2, torch.ones(2, 4) * 3)
        assert sc._auto_disabled is True
        assert "3 consecutive misses" in sc.trip_reason


class _FakeClock:
    """A controllable stand-in for time.monotonic() -- lets min_hit_rate
    tests exercise the rolling time window deterministically instead of
    depending on real wall-clock sleeps (slow and flaky)."""

    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


class TestMinHitRate:
    """SimilarityCache(min_hit_rate=...) -- the rate-based circuit breaker,
    tracking hit/miss outcomes over a rolling wall-clock time window
    instead of requiring an unbroken miss streak. Exists specifically
    because occasional coincidental hits (e.g. from a coarse aggregate-mean
    similarity signal over spatially-coherent dense-grid chunks) reset
    max_consecutive_misses's streak counter and can indefinitely postpone
    that breaker -- see the module docstring's AMD_TUNED_TORCH_SIMILARITY_CACHE_
    MIN_HIT_RATE section."""

    def _cache(self, monkeypatch, **kwargs):
        clock = _FakeClock()
        monkeypatch.setattr(cache_module.time, "monotonic", clock)
        sc = cache_module.SimilarityCache(enabled=True, max_consecutive_misses=None, **kwargs)
        return sc, clock

    def test_none_means_disabled(self, monkeypatch):
        sc, clock = self._cache(monkeypatch, thresh=1e9, min_hit_rate=None,
                                 hit_rate_min_samples=1)
        for i in range(100):
            sc.call(lambda x: x * 2, torch.ones(2, 4) * i)
            clock.advance(0.01)
        assert sc._auto_disabled is False

    def test_does_not_trip_before_min_samples_reached(self, monkeypatch):
        # thresh/hit-vs-miss pattern is irrelevant here -- the point is
        # that with fewer than hit_rate_min_samples calls, the rate is
        # never even evaluated, regardless of how bad it would look.
        sc, clock = self._cache(monkeypatch, thresh=1e9, min_hit_rate=0.9,
                                 hit_rate_min_samples=20, hit_rate_window_seconds=10.0)
        for i in range(19):  # one short of hit_rate_min_samples
            sc.call(lambda x: x * 2, torch.ones(2, 4) * i)
            clock.advance(0.01)
        assert sc._auto_disabled is False

    def test_trips_once_min_samples_reached_with_low_hit_rate(self, monkeypatch):
        sc, clock = self._cache(monkeypatch, thresh=1e-9, min_hit_rate=0.5,
                                 hit_rate_min_samples=20, hit_rate_window_seconds=10.0)
        for i in range(20):  # always a miss -- 0% hit rate, well below 50%
            sc.call(lambda x: x * 2, torch.ones(2, 4) * (i + 1))
            clock.advance(0.01)
        assert sc._auto_disabled is True
        assert "hit rate" in sc.trip_reason

    def test_does_not_trip_when_hit_rate_above_threshold(self, monkeypatch):
        sc, clock = self._cache(monkeypatch, thresh=0.5, min_hit_rate=0.3,
                                 hit_rate_min_samples=10, hit_rate_window_seconds=10.0)
        # Alternate miss/hit -- roughly 50% hit rate, above the 30% floor.
        sc.call(lambda x: x * 2, torch.ones(2, 4))  # miss, seeds previous_input
        for _ in range(20):
            sc.call(lambda x: x * 2, torch.ones(2, 4))  # identical input -> hit every time
            clock.advance(0.01)
        assert sc._auto_disabled is False

    def test_coincidental_hits_do_not_indefinitely_block_the_trip(self, monkeypatch):
        # The actual real-world scenario this feature exists for: mostly
        # misses, punctuated by just enough coincidental hits to keep
        # max_consecutive_misses's streak from ever completing (verified
        # separately below) -- min_hit_rate must still trip on the low
        # *overall* rate.
        sc, clock = self._cache(monkeypatch, thresh=1e9, min_hit_rate=0.5,
                                 hit_rate_min_samples=10, hit_rate_window_seconds=10.0)
        for i in range(20):
            is_hit_call = (i % 4 == 3)  # 1 hit every 4 calls -> 25% hit rate, below 50%
            sc._record_outcome_and_maybe_trip(is_hit_call)
            clock.advance(0.01)
        assert sc._auto_disabled is True
        assert "hit rate" in sc.trip_reason

    def test_max_consecutive_misses_never_trips_with_periodic_hits(self):
        # Companion to the test above: proves *why* the rate breaker is
        # needed -- a hit every other call keeps resetting the streak
        # counter, so max_consecutive_misses alone never fires even after
        # many calls, no matter how low the overall hit rate ends up being.
        sc = cache_module.SimilarityCache(thresh=1e-9, enabled=True, max_consecutive_misses=3)
        calls = []

        def fn(x):
            calls.append(x)
            return x * 2

        for i in range(40):
            if i % 2 == 1 and sc.previous_input is not None:
                x = sc.previous_input.clone()  # identical to previous -> guaranteed hit
            else:
                x = torch.ones(2, 4) * (i + 1)  # always different -> guaranteed miss
            sc.call(fn, x)
        assert sc._auto_disabled is False

    def test_old_samples_pruned_outside_window(self, monkeypatch):
        sc, clock = self._cache(monkeypatch, thresh=1e-9, min_hit_rate=0.5,
                                 hit_rate_min_samples=5, hit_rate_window_seconds=1.0)
        for i in range(5):
            sc.call(lambda x: x * 2, torch.ones(2, 4) * (i + 1))  # 5 misses, fills the window
        assert len(sc._outcomes) == 5
        clock.advance(2.0)  # well past the 1.0s window
        sc.call(lambda x: x * 2, torch.ones(2, 4) * 99)  # one more miss -- old ones must be pruned
        assert len(sc._outcomes) == 1

    def test_reset_clears_outcome_history(self, monkeypatch):
        sc, clock = self._cache(monkeypatch, thresh=1e-9, min_hit_rate=0.5,
                                 hit_rate_min_samples=5, hit_rate_window_seconds=10.0)
        for i in range(5):
            sc.call(lambda x: x * 2, torch.ones(2, 4) * (i + 1))  # always a miss -- trips at sample 5
        assert sc._auto_disabled is True
        sc.reset()
        assert len(sc._outcomes) == 0
        assert sc._outcome_hits == 0
        assert sc._auto_disabled is False

    def test_zero_overhead_data_structures_when_disabled(self):
        sc = cache_module.SimilarityCache(thresh=0.5, enabled=True, min_hit_rate=None)
        sc.call(lambda x: x * 2, torch.ones(2, 4))
        assert len(sc._outcomes) == 0  # never populated when min_hit_rate is None


class TestCounters:
    def test_hit_and_miss_counters(self):
        sc = cache_module.SimilarityCache(thresh=0.5, enabled=True)
        sc.call(lambda x: x * 2, torch.ones(2, 4))       # miss
        sc.call(lambda x: x * 2, torch.ones(2, 4) * 1.01)  # hit (within thresh)
        sc.call(lambda x: x * 2, torch.ones(2, 4) * 100)   # miss
        assert sc.hits == 1
        assert sc.misses == 2
        assert sc.size_bypassed == 0
        assert sc.breaker_bypassed == 0

    def test_size_bypassed_counter(self):
        sc = cache_module.SimilarityCache(thresh=1e9, enabled=True, max_bytes=16)
        sc.call(lambda x: x * 2, torch.ones(1000))
        sc.call(lambda x: x * 2, torch.ones(1000))
        assert sc.size_bypassed == 2
        assert sc.hits == 0
        assert sc.misses == 0

    def test_breaker_bypassed_counter(self):
        sc = cache_module.SimilarityCache(thresh=1e-9, enabled=True, max_consecutive_misses=2)
        sc.call(lambda x: x * 2, torch.ones(2, 4) * 1)  # miss 1
        sc.call(lambda x: x * 2, torch.ones(2, 4) * 2)  # miss 2 -> trips
        sc.call(lambda x: x * 2, torch.ones(2, 4) * 3)  # breaker bypass
        sc.call(lambda x: x * 2, torch.ones(2, 4) * 4)  # breaker bypass
        assert sc.misses == 2
        assert sc.breaker_bypassed == 2

    def test_debug_name_used_in_label(self):
        sc = cache_module.SimilarityCache(thresh=0.5, enabled=True, debug_name="my_module")
        assert sc._label() == "my_module"

    def test_default_label_falls_back_to_id(self):
        sc = cache_module.SimilarityCache(thresh=0.5, enabled=True)
        assert sc._label() == f"SimilarityCache@{id(sc):#x}"


class TestCompileIntegration:
    """SimilarityCache(compile=True) -- lazily torch.compile `fn` on its
    first real (cache-miss) call via amd_tuned_torch.torch_compile.compile_fn, then
    reuse that compiled callable for every later miss. Mocks
    amd_tuned_torch.torch_compile directly (same style as conftest.py's aiter/te
    fixtures mock amd_tuned_torch.aiter_ops/te_ops) rather than exercising a real
    torch.compile, so no Triton/Inductor backend is required to run these.
    """

    def test_compile_false_never_touches_torch_compile(self, monkeypatch):
        import amd_tuned_torch.torch_compile as torch_compile_module

        spy = MagicMock(side_effect=AssertionError("should not be called"))
        monkeypatch.setattr(torch_compile_module, "compile_fn", spy)

        sc = cache_module.SimilarityCache(thresh=1e9, enabled=True, compile=False)
        sc.call(lambda x: x * 2, torch.ones(2, 4))
        spy.assert_not_called()

    def test_compile_true_compiles_on_first_real_call_only(self, monkeypatch):
        import amd_tuned_torch.torch_compile as torch_compile_module

        real_calls = []

        def fn(x):
            real_calls.append(x)
            return x * 2

        compile_spy = MagicMock(side_effect=lambda f, **kw: f)
        monkeypatch.setattr(torch_compile_module, "compile_fn", compile_spy)
        monkeypatch.setattr(torch_compile_module, "available", lambda: True)

        sc = cache_module.SimilarityCache(thresh=0.05, enabled=True, compile=True)
        sc.call(fn, torch.zeros(2, 4))  # miss: first real call -> compiles
        sc.call(fn, torch.ones(2, 4) * 100.0)  # far outside thresh -> another miss
        assert compile_spy.call_count == 1

    def test_compile_true_reuses_compiled_fn_across_different_fn_objects(self, monkeypatch):
        # Mirrors _cached_module_call: a fresh `_forward` closure is built
        # per call, but they're all behaviorally the same underlying
        # computation, so the first-compiled version must still be reused.
        import amd_tuned_torch.torch_compile as torch_compile_module

        compiled_marker = MagicMock(side_effect=lambda x: x * 2)
        monkeypatch.setattr(torch_compile_module, "compile_fn", MagicMock(return_value=compiled_marker))
        monkeypatch.setattr(torch_compile_module, "available", lambda: True)

        sc = cache_module.SimilarityCache(thresh=0.05, enabled=True, compile=True)

        def fn_a(x):
            return x * 2

        def fn_b(x):
            return x * 2

        sc.call(fn_a, torch.zeros(2, 4))
        sc.call(fn_b, torch.ones(2, 4) * 100.0)  # different fn object, still a miss
        assert torch_compile_module.compile_fn.call_count == 1
        assert compiled_marker.call_count == 2

    def test_compile_true_falls_back_to_fn_when_torch_compile_unavailable(self, monkeypatch):
        import amd_tuned_torch.torch_compile as torch_compile_module

        monkeypatch.setattr(torch_compile_module, "available", lambda: False)
        spy = MagicMock(side_effect=AssertionError("compile_fn should not be reached"))
        monkeypatch.setattr(torch_compile_module, "compile_fn", spy)

        real_calls = []

        def fn(x):
            real_calls.append(x)
            return x * 2

        sc = cache_module.SimilarityCache(thresh=0.05, enabled=True, compile=True)
        out = sc.call(fn, torch.zeros(2, 4))
        assert len(real_calls) == 1
        assert torch.equal(out, torch.zeros(2, 4))
        spy.assert_not_called()

    def test_compile_kwargs_forwarded(self, monkeypatch):
        import amd_tuned_torch.torch_compile as torch_compile_module

        spy = MagicMock(side_effect=lambda f, **kw: f)
        monkeypatch.setattr(torch_compile_module, "compile_fn", spy)
        monkeypatch.setattr(torch_compile_module, "available", lambda: True)

        sc = cache_module.SimilarityCache(
            thresh=0.05, enabled=True, compile=True,
            compile_kwargs={"fullgraph": True, "dynamic": True},
        )
        sc.call(lambda x: x * 2, torch.zeros(2, 4))
        _, kwargs = spy.call_args
        assert kwargs == {"fullgraph": True, "dynamic": True}

    def test_reset_does_not_clear_compiled_fn(self, monkeypatch):
        import amd_tuned_torch.torch_compile as torch_compile_module

        monkeypatch.setattr(torch_compile_module, "compile_fn", MagicMock(side_effect=lambda f, **kw: f))
        monkeypatch.setattr(torch_compile_module, "available", lambda: True)

        sc = cache_module.SimilarityCache(thresh=0.05, enabled=True, compile=True)
        sc.call(lambda x: x * 2, torch.zeros(2, 4))
        assert sc._compiled_fn is not None
        sc.reset()
        assert sc._compiled_fn is not None  # compiling isn't stale cached data

    def test_cache_hit_never_invokes_compiled_fn(self, monkeypatch):
        import amd_tuned_torch.torch_compile as torch_compile_module

        compiled = MagicMock(side_effect=lambda x: x * 2)
        monkeypatch.setattr(torch_compile_module, "compile_fn", MagicMock(return_value=compiled))
        monkeypatch.setattr(torch_compile_module, "available", lambda: True)

        sc = cache_module.SimilarityCache(thresh=0.5, enabled=True, compile=True)
        sc.call(lambda x: x * 2, torch.ones(2, 4))
        sc.call(lambda x: x * 2, torch.ones(2, 4) * 1.01)  # within thresh -> hit, no call at all
        assert compiled.call_count == 1


class TestDecoratorUsage:
    def test_similarity_cached_decorator(self):
        calls = []

        @cache_module.similarity_cached(thresh=0.5, enabled=True)
        def fn(x):
            calls.append(x)
            return x * 2

        fn(torch.ones(2, 4))
        fn(torch.ones(2, 4) * 1.01)
        assert len(calls) == 1

    def test_decorator_exposes_cache_attribute(self):
        @cache_module.similarity_cached(thresh=0.5, enabled=True)
        def fn(x):
            return x

        assert isinstance(fn.cache, cache_module.SimilarityCache)

    def test_functools_wraps_preserves_name(self):
        @cache_module.similarity_cached()
        def my_named_function(x):
            return x

        assert my_named_function.__name__ == "my_named_function"

    def test_direct_instance_as_decorator(self):
        sc = cache_module.SimilarityCache(thresh=0.5, enabled=True)
        calls = []

        @sc
        def fn(x):
            calls.append(x)
            return x * 2

        fn(torch.ones(2, 4))
        fn(torch.ones(2, 4) * 1.01)
        assert len(calls) == 1


class TestKeyedCache:
    """KeyedCache -- exact, identity-keyed memoization, the deliberate
    non-approximating sibling of SimilarityCache. No tensor comparison at
    all: a plain dict keyed by whatever the caller passes as `key`."""

    def test_disabled_always_calls_fn(self):
        kc = cache_module.KeyedCache(enabled=False)
        calls = []

        def fn(x):
            calls.append(x)
            return x * 2

        kc.call(fn, "same-key", torch.ones(2, 4))
        kc.call(fn, "same-key", torch.ones(2, 4))
        assert len(calls) == 2  # disabled -- key is irrelevant, always a real call

    def test_second_call_with_same_key_is_a_hit(self):
        kc = cache_module.KeyedCache(enabled=True)
        calls = []

        def fn(x):
            calls.append(x)
            return x * 2

        out1 = kc.call(fn, "chunk-0", torch.ones(2, 4))
        out2 = kc.call(fn, "chunk-0", torch.zeros(2, 4))  # different tensor, SAME key -> still a hit
        assert len(calls) == 1
        assert out2 is out1  # the *first* call's output, not recomputed

    def test_different_keys_are_always_misses(self):
        kc = cache_module.KeyedCache(enabled=True)
        calls = []

        def fn(x):
            calls.append(x)
            return x * 2

        kc.call(fn, "chunk-0", torch.ones(2, 4))
        kc.call(fn, "chunk-1", torch.ones(2, 4))  # identical tensor, DIFFERENT key -> still a miss
        assert len(calls) == 2

    def test_key_not_passed_to_fn(self):
        kc = cache_module.KeyedCache(enabled=True)
        received = {}

        def fn(x, y=None):
            received["x"] = x
            received["y"] = y
            return x

        kc.call(fn, "some-key", "arg-x", y="arg-y")
        assert received == {"x": "arg-x", "y": "arg-y"}

    def test_hits_and_misses_counters(self):
        kc = cache_module.KeyedCache(enabled=True)
        kc.call(lambda x: x, "a", 1)   # miss
        kc.call(lambda x: x, "a", 1)   # hit
        kc.call(lambda x: x, "b", 2)   # miss
        assert kc.hits == 1
        assert kc.misses == 2

    def test_max_entries_stops_storing_new_keys(self):
        kc = cache_module.KeyedCache(enabled=True, max_entries=2)
        calls = []

        def fn(x):
            calls.append(x)
            return x

        kc.call(fn, "a", 1)
        kc.call(fn, "b", 2)
        kc.call(fn, "c", 3)  # cache already at max_entries=2 -- computed but not stored
        assert kc.not_stored == 1
        assert len(kc._cache) == 2

        kc.call(fn, "c", 3)  # "c" was never stored -> miss again, recomputes
        assert len(calls) == 4

    def test_max_entries_does_not_evict_existing_keys(self):
        kc = cache_module.KeyedCache(enabled=True, max_entries=2)
        calls = []

        def fn(x):
            calls.append(x)
            return x

        kc.call(fn, "a", 1)
        kc.call(fn, "b", 2)
        kc.call(fn, "c", 3)  # not stored, cache stays at 2 entries
        kc.call(fn, "a", 1)  # "a" still cached -> hit
        assert kc.hits == 1

    def test_reset_clears_cache_and_counters(self):
        kc = cache_module.KeyedCache(enabled=True)
        kc.call(lambda x: x, "a", 1)
        kc.call(lambda x: x, "a", 1)
        assert kc.hits == 1
        kc.reset()
        assert kc.hits == 0
        assert kc.misses == 0
        assert kc.not_stored == 0
        assert len(kc._cache) == 0
        # after reset, a previously-cached key is a fresh miss
        calls = []
        kc.call(lambda x: (calls.append(x), x)[1], "a", 1)
        assert len(calls) == 1

    def test_debug_name_used_in_label(self):
        kc = cache_module.KeyedCache(enabled=True, debug_name="geo_decoder_cache")
        assert kc._label() == "geo_decoder_cache"

    def test_default_label_falls_back_to_id(self):
        kc = cache_module.KeyedCache(enabled=True)
        assert kc._label() == f"KeyedCache@{id(kc):#x}"

    def test_enable_disable_toggle(self):
        kc = cache_module.KeyedCache(enabled=False)
        assert kc.enabled is False
        kc.enable()
        assert kc.enabled is True
        kc.disable()
        assert kc.enabled is False

    def test_reads_env_flag_by_default(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE", "1")
        assert cache_module.KeyedCache().enabled is True
        monkeypatch.setenv("AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE", "0")
        assert cache_module.KeyedCache().enabled is False


class TestKeyedCachedSugar:
    def test_returns_keyed_cache_instance(self):
        kc = cache_module.keyed_cached(enabled=True)
        assert isinstance(kc, cache_module.KeyedCache)

    def test_forwards_max_entries(self):
        kc = cache_module.keyed_cached(enabled=True, max_entries=5)
        assert kc.max_entries == 5

    def test_usable_like_a_direct_construction(self):
        kc = cache_module.keyed_cached(enabled=True)
        calls = []

        def fn(x):
            calls.append(x)
            return x * 2

        kc.call(fn, "k", torch.ones(2, 4))
        kc.call(fn, "k", torch.ones(2, 4))
        assert len(calls) == 1


class TestResetEnableDisable:
    def test_reset_clears_cached_state(self):
        sc = cache_module.SimilarityCache(thresh=0.5, enabled=True)
        sc.call(lambda x: x * 2, torch.ones(2, 4))
        assert sc.previous_input is not None
        sc.reset()
        assert sc.previous_input is None
        assert sc.previous_output is None

    def test_enable_disable_toggle(self):
        sc = cache_module.SimilarityCache(enabled=False)
        assert sc.enabled is False
        sc.enable()
        assert sc.enabled is True
        sc.disable()
        assert sc.enabled is False


class TestEndToEndSimulation:
    def test_skips_reduce_real_calls_and_stay_numerically_correct(self):
        sc = cache_module.SimilarityCache(thresh=0.05, enabled=True)
        real_calls = 0

        def run_expensive_step(x):
            nonlocal real_calls
            real_calls += 1
            return x + 1.0

        hidden_states = torch.zeros(2, 4)
        # first three calls drift by less than thresh=0.05 relative to each
        # other once nonzero, then a big jump forces a real recompute.
        inputs = [
            torch.zeros(2, 4),
            torch.zeros(2, 4),
            torch.zeros(2, 4),
            torch.ones(2, 4) * 100.0,
        ]
        outputs = [sc.call(run_expensive_step, x) for x in inputs]
        # call 1: real (no previous input yet). calls 2-3: distance 0
        # against the all-zero previous_input (falls back through the
        # clamp) -> skipped, reusing call 1's output. call 4: a huge jump
        # forces a real recompute.
        assert real_calls == 2
        assert torch.equal(outputs[0], torch.ones(2, 4))
        assert outputs[1] is outputs[0]
        assert outputs[2] is outputs[0]
        assert torch.equal(outputs[3], torch.ones(2, 4) * 101.0)

    def test_never_skips_when_disabled(self):
        sc = cache_module.SimilarityCache(thresh=1e9, enabled=False)
        real_calls = 0

        def run_expensive_step(x):
            nonlocal real_calls
            real_calls += 1
            return x + 1.0

        x = torch.ones(2, 4)
        for _ in range(5):
            sc.call(run_expensive_step, x)
        assert real_calls == 5


class TestModuleCacheLifecycle:
    """enable_module_cache()/disable_module_cache() patch
    torch.nn.Module.__call__ process-wide -- every test here must leave
    that restored, or every other test in this whole suite (this file and
    every other one) that constructs/calls an nn.Module would silently run
    through the cache wrapper afterward."""

    def teardown_method(self):
        cache_module.disable_module_cache()

    def test_disabled_by_default_in_this_module_object(self):
        # This file's own import of amd_tuned_torch.cache happens with
        # AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE unset in the test environment
        # (conftest.py doesn't set it), so the auto-install at the bottom
        # of cache.py should not have fired for this test run.
        assert cache_module.is_module_cache_enabled() is False

    def test_enable_patches_module_call(self):
        orig_call = torch.nn.Module.__call__
        cache_module.enable_module_cache()
        assert cache_module.is_module_cache_enabled() is True
        assert torch.nn.Module.__call__ is not orig_call

    def test_disable_restores_stock_call(self):
        orig_call = torch.nn.Module.__call__
        cache_module.enable_module_cache()
        cache_module.disable_module_cache()
        assert cache_module.is_module_cache_enabled() is False
        assert torch.nn.Module.__call__ is orig_call

    def test_enable_is_idempotent(self):
        cache_module.enable_module_cache(thresh=0.3)
        cache_module.enable_module_cache(thresh=0.9)  # must not override the first call's thresh
        assert cache_module._module_cache_thresh == pytest.approx(0.3)

    def test_disable_without_enable_is_a_no_op(self):
        cache_module.disable_module_cache()  # must not raise
        assert cache_module.is_module_cache_enabled() is False


class TestModuleCacheBehavior:
    def teardown_method(self):
        cache_module.disable_module_cache()

    def test_still_computes_correct_output(self):
        cache_module.enable_module_cache(thresh=0.5, min_bytes=None)
        linear = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            linear.weight.fill_(1.0)
        x = torch.ones(1, 4)
        out = linear(x)
        assert torch.equal(out, torch.full((1, 2), 4.0))

    def test_skips_real_forward_when_input_is_similar(self):
        cache_module.enable_module_cache(thresh=0.5, min_bytes=None)
        calls = []
        real_forward = nn.Linear.forward

        def counting_forward(self, x):
            calls.append(x)
            return real_forward(self, x)

        linear = nn.Linear(4, 2)
        linear.forward = counting_forward.__get__(linear, nn.Linear)

        with torch.no_grad():
            out1 = linear(torch.ones(1, 4))
            out2 = linear(torch.ones(1, 4) * 1.01)  # within thresh=0.5
        assert len(calls) == 1
        assert out2 is out1

    def test_different_module_instances_have_independent_caches(self):
        cache_module.enable_module_cache(thresh=0.5, min_bytes=None)
        a = nn.Linear(4, 2)
        b = nn.Linear(4, 2)
        with torch.no_grad():
            a.weight.fill_(1.0)
            b.weight.fill_(2.0)
            out_a = a(torch.ones(1, 4))
            # calling a completely different module/input must not be
            # treated as "similar to a's previous call" just because it's
            # the next call to a similarity-cached __call__.
            out_b = b(torch.ones(1, 4))
        assert not torch.equal(out_a, out_b)

    def test_opt_out_attribute_bypasses_caching(self):
        cache_module.enable_module_cache(thresh=0.5, min_bytes=None)
        calls = []
        real_forward = nn.Linear.forward

        def counting_forward(self, x):
            calls.append(x)
            return real_forward(self, x)

        linear = nn.Linear(4, 2)
        linear.forward = counting_forward.__get__(linear, nn.Linear)
        linear._amd_tuned_torch_no_cache = True

        with torch.no_grad():
            linear(torch.ones(1, 4))
            linear(torch.ones(1, 4) * 1.01)  # would be a cache hit if not opted out
        assert len(calls) == 2

    def test_does_not_cache_when_grad_is_needed(self):
        cache_module.enable_module_cache(thresh=1e9, min_bytes=None)  # would always "match" if it cached
        calls = []
        real_forward = nn.Linear.forward

        def counting_forward(self, x):
            calls.append(x)
            return real_forward(self, x)

        linear = nn.Linear(4, 2)
        linear.forward = counting_forward.__get__(linear, nn.Linear)

        x1 = torch.ones(1, 4, requires_grad=True)
        x2 = torch.ones(1, 4, requires_grad=True) * 1.001
        linear(x1)
        linear(x2)
        assert len(calls) == 2

    def test_oversized_module_input_bypasses_caching(self):
        cache_module.enable_module_cache(thresh=1e9, max_bytes=16)  # float32 (1,4) = 16 bytes
        calls = []
        real_forward = nn.Linear.forward

        def counting_forward(self, x):
            calls.append(x)
            return real_forward(self, x)

        linear = nn.Linear(8, 2)
        linear.forward = counting_forward.__get__(linear, nn.Linear)

        with torch.no_grad():
            linear(torch.ones(1, 8))  # float32 (1,8) = 32 bytes -> oversized
            linear(torch.ones(1, 8))  # would be a hit if not oversized
        assert len(calls) == 2

    def test_non_tensor_first_argument_bypasses_caching(self):
        cache_module.enable_module_cache(thresh=1e9)
        calls = []

        class Weird(nn.Module):
            def forward(self, flag):
                calls.append(flag)
                return flag

        m = Weird()
        with torch.no_grad():
            m(True)
            m(True)
        assert len(calls) == 2

    def test_circuit_breaker_trips_on_never_similar_inputs(self):
        # Mirrors the real regression this was added for: a module called
        # repeatedly (e.g. once per spatial chunk during volume decoding)
        # with inputs that are never similar to each other, min_bytes-sized
        # tensors -- every call should be a genuine miss, and after
        # max_consecutive_misses of those the module's cache should stop
        # computing the (GPU-sync-forcing) distance check altogether.
        cache_module.enable_module_cache(thresh=1e-9, min_bytes=None, max_consecutive_misses=3)
        linear = nn.Linear(4, 2)

        with torch.no_grad():
            for i in range(3):
                linear(torch.ones(1, 4) * (i + 1))  # always dissimilar -> always a miss

        sc = cache_module._module_caches.get(linear)
        assert sc is not None
        assert sc._auto_disabled is True

        # Further calls must still compute correctly, just without ever
        # reaching the distance check again.
        def boom(self, x):
            raise AssertionError("_is_close_enough must not run once auto-disabled")

        sc._is_close_enough = boom.__get__(sc, cache_module.SimilarityCache)
        with torch.no_grad():
            out = linear(torch.ones(1, 4) * 99)  # must not raise
        assert out.shape == (1, 2)

    def test_min_hit_rate_trips_when_max_consecutive_misses_cannot(self, monkeypatch):
        # The actual scenario min_hit_rate exists for: a periodic
        # coincidental hit (simulating a dense-grid chunk's coarse
        # aggregate-mean similarity occasionally landing within thresh)
        # resets max_consecutive_misses's streak every time, so with a high
        # enough max_consecutive_misses that breaker never fires -- but the
        # overall hit rate is still low enough for min_hit_rate to catch it.
        clock = _FakeClock()
        monkeypatch.setattr(cache_module.time, "monotonic", clock)
        cache_module.enable_module_cache(
            thresh=1e-9, min_bytes=None, max_consecutive_misses=100,
            min_hit_rate=0.5, hit_rate_min_samples=10, hit_rate_window_seconds=10.0,
        )
        linear = nn.Linear(4, 2)

        with torch.no_grad():
            for i in range(20):
                if i % 4 == 3:
                    sc = cache_module._module_caches.get(linear)
                    x = sc.previous_input.clone()  # identical -> guaranteed hit (25% hit rate overall)
                else:
                    x = torch.ones(1, 4) * (i + 1)  # always different -> guaranteed miss
                linear(x)
                clock.advance(0.01)

        sc = cache_module._module_caches.get(linear)
        assert sc is not None
        assert sc._auto_disabled is True
        assert "hit rate" in sc.trip_reason


class TestModuleCacheThreshEnv:
    def teardown_method(self):
        cache_module.disable_module_cache()

    def test_reads_thresh_env_var(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_THRESH", "0.42")
        cache_module.enable_module_cache()
        assert cache_module._module_cache_thresh == pytest.approx(0.42)

    def test_falls_back_to_default_on_invalid_value(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_THRESH", "not-a-float")
        cache_module.enable_module_cache()
        assert cache_module._module_cache_thresh == pytest.approx(0.1)

    def test_explicit_thresh_overrides_env(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_THRESH", "0.42")
        cache_module.enable_module_cache(thresh=0.7)
        assert cache_module._module_cache_thresh == pytest.approx(0.7)


class TestModuleCacheMaxBytesEnv:
    def teardown_method(self):
        cache_module.disable_module_cache()

    def test_defaults_to_64mib_when_unset(self, monkeypatch):
        monkeypatch.delenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_MAX_BYTES", raising=False)
        cache_module.enable_module_cache()
        assert cache_module._module_cache_max_bytes == 64 * 1024 * 1024

    def test_reads_max_bytes_env(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_MAX_BYTES", "1024")
        cache_module.enable_module_cache()
        assert cache_module._module_cache_max_bytes == 1024

    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_zero_or_negative_env_means_no_limit(self, monkeypatch, value):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_MAX_BYTES", value)
        cache_module.enable_module_cache()
        assert cache_module._module_cache_max_bytes is None

    def test_falls_back_to_default_on_invalid_value(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_MAX_BYTES", "not-an-int")
        cache_module.enable_module_cache()
        assert cache_module._module_cache_max_bytes == 64 * 1024 * 1024

    def test_explicit_none_overrides_env_to_no_limit(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_MAX_BYTES", "1024")
        cache_module.enable_module_cache(max_bytes=None)
        assert cache_module._module_cache_max_bytes is None

    def test_explicit_value_overrides_env(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_MAX_BYTES", "1024")
        cache_module.enable_module_cache(max_bytes=2048)
        assert cache_module._module_cache_max_bytes == 2048


class TestModuleCacheMinBytesEnv:
    def teardown_method(self):
        cache_module.disable_module_cache()

    def test_defaults_to_4kib_when_unset(self, monkeypatch):
        monkeypatch.delenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_MIN_BYTES", raising=False)
        cache_module.enable_module_cache()
        assert cache_module._module_cache_min_bytes == 4 * 1024

    def test_reads_min_bytes_env(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_MIN_BYTES", "1024")
        cache_module.enable_module_cache()
        assert cache_module._module_cache_min_bytes == 1024

    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_zero_or_negative_env_means_no_floor(self, monkeypatch, value):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_MIN_BYTES", value)
        cache_module.enable_module_cache()
        assert cache_module._module_cache_min_bytes is None

    def test_falls_back_to_default_on_invalid_value(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_MIN_BYTES", "not-an-int")
        cache_module.enable_module_cache()
        assert cache_module._module_cache_min_bytes == 4 * 1024

    def test_explicit_none_overrides_env_to_no_floor(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_MIN_BYTES", "1024")
        cache_module.enable_module_cache(min_bytes=None)
        assert cache_module._module_cache_min_bytes is None

    def test_explicit_value_overrides_env(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_MIN_BYTES", "1024")
        cache_module.enable_module_cache(min_bytes=2048)
        assert cache_module._module_cache_min_bytes == 2048


class TestModuleCacheMaxMissesEnv:
    def teardown_method(self):
        cache_module.disable_module_cache()

    def test_defaults_to_5_when_unset(self, monkeypatch):
        monkeypatch.delenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_MAX_MISSES", raising=False)
        cache_module.enable_module_cache()
        assert cache_module._module_cache_max_misses == 5

    def test_reads_max_misses_env(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_MAX_MISSES", "10")
        cache_module.enable_module_cache()
        assert cache_module._module_cache_max_misses == 10

    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_zero_or_negative_env_means_never_auto_disable(self, monkeypatch, value):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_MAX_MISSES", value)
        cache_module.enable_module_cache()
        assert cache_module._module_cache_max_misses is None

    def test_falls_back_to_default_on_invalid_value(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_MAX_MISSES", "not-an-int")
        cache_module.enable_module_cache()
        assert cache_module._module_cache_max_misses == 5

    def test_explicit_none_overrides_env_to_never_auto_disable(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_MAX_MISSES", "10")
        cache_module.enable_module_cache(max_consecutive_misses=None)
        assert cache_module._module_cache_max_misses is None

    def test_explicit_value_overrides_env(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_MAX_MISSES", "10")
        cache_module.enable_module_cache(max_consecutive_misses=3)
        assert cache_module._module_cache_max_misses == 3

    def test_new_module_caches_inherit_max_misses(self):
        cache_module.enable_module_cache(thresh=0.5, max_consecutive_misses=7)
        linear = nn.Linear(4, 2)
        with torch.no_grad():
            linear(torch.ones(1, 4))
        sc = cache_module._module_caches.get(linear)
        assert sc is not None
        assert sc.max_consecutive_misses == 7


class TestModuleCacheMinHitRateEnv:
    def teardown_method(self):
        cache_module.disable_module_cache()

    def test_defaults_to_point2_when_unset(self, monkeypatch):
        monkeypatch.delenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_MIN_HIT_RATE", raising=False)
        cache_module.enable_module_cache()
        assert cache_module._module_cache_min_hit_rate == pytest.approx(0.2)

    def test_reads_min_hit_rate_env(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_MIN_HIT_RATE", "0.4")
        cache_module.enable_module_cache()
        assert cache_module._module_cache_min_hit_rate == pytest.approx(0.4)

    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_zero_or_negative_env_means_disabled(self, monkeypatch, value):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_MIN_HIT_RATE", value)
        cache_module.enable_module_cache()
        assert cache_module._module_cache_min_hit_rate is None

    def test_falls_back_to_default_on_invalid_value(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_MIN_HIT_RATE", "not-a-float")
        cache_module.enable_module_cache()
        assert cache_module._module_cache_min_hit_rate == pytest.approx(0.2)

    def test_explicit_none_overrides_env_to_disabled(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_MIN_HIT_RATE", "0.4")
        cache_module.enable_module_cache(min_hit_rate=None)
        assert cache_module._module_cache_min_hit_rate is None

    def test_explicit_value_overrides_env(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_MIN_HIT_RATE", "0.4")
        cache_module.enable_module_cache(min_hit_rate=0.7)
        assert cache_module._module_cache_min_hit_rate == pytest.approx(0.7)

    def test_new_module_caches_inherit_min_hit_rate_and_window_settings(self):
        cache_module.enable_module_cache(thresh=0.5, min_hit_rate=0.35,
                                          hit_rate_window_seconds=3.5, hit_rate_min_samples=15)
        linear = nn.Linear(4, 2)
        with torch.no_grad():
            linear(torch.ones(1, 4))
        sc = cache_module._module_caches.get(linear)
        assert sc is not None
        assert sc.min_hit_rate == pytest.approx(0.35)
        assert sc.hit_rate_window_seconds == pytest.approx(3.5)
        assert sc.hit_rate_min_samples == 15


class TestModuleCacheAutoExcludeEnv:
    def teardown_method(self):
        cache_module.disable_module_cache()

    def test_defaults_off_when_unset(self, monkeypatch):
        monkeypatch.delenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_AUTO_EXCLUDE", raising=False)
        cache_module.enable_module_cache()
        assert cache_module._module_cache_auto_exclude is False

    @pytest.mark.parametrize("value", ["1", "true", "yes"])
    def test_reads_auto_exclude_env(self, monkeypatch, value):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_AUTO_EXCLUDE", value)
        cache_module.enable_module_cache()
        assert cache_module._module_cache_auto_exclude is True

    def test_explicit_value_overrides_env(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_AUTO_EXCLUDE", "1")
        cache_module.enable_module_cache(auto_exclude=False)
        assert cache_module._module_cache_auto_exclude is False

    def test_module_gets_excluded_after_trip(self):
        cache_module.enable_module_cache(thresh=1e-9, min_bytes=None,
                                          max_consecutive_misses=3, auto_exclude=True)
        linear = nn.Linear(4, 2)
        with torch.no_grad():
            for i in range(3):
                linear(torch.ones(1, 4) * (i + 1))  # always dissimilar -> trips at 3
        assert linear._amd_tuned_torch_no_cache is True

    def test_module_stays_wrapped_before_trip(self):
        cache_module.enable_module_cache(thresh=0.5, min_bytes=None,
                                          max_consecutive_misses=5, auto_exclude=True)
        linear = nn.Linear(4, 2)
        with torch.no_grad():
            linear(torch.ones(1, 4))  # single miss -- nowhere near tripping
        assert getattr(linear, "_amd_tuned_torch_no_cache", False) is False

    def test_no_exclusion_when_disabled(self):
        cache_module.enable_module_cache(thresh=1e-9, min_bytes=None,
                                          max_consecutive_misses=3, auto_exclude=False)
        linear = nn.Linear(4, 2)
        with torch.no_grad():
            for i in range(3):
                linear(torch.ones(1, 4) * (i + 1))
        sc = cache_module._module_caches.get(linear)
        assert sc is not None
        assert sc._auto_disabled is True  # breaker still trips...
        assert getattr(linear, "_amd_tuned_torch_no_cache", False) is False  # ...just no auto-exclude

    def test_compiled_module_not_excluded(self, monkeypatch):
        import amd_tuned_torch.torch_compile as torch_compile_module
        monkeypatch.setattr(torch_compile_module, "compile_fn", MagicMock(side_effect=lambda f, **kw: f))
        monkeypatch.setattr(torch_compile_module, "available", lambda: True)

        cache_module.enable_module_cache(thresh=1e-9, min_bytes=None,
                                          max_consecutive_misses=3, auto_exclude=True, compile=True)
        linear = nn.Linear(4, 2)
        with torch.no_grad():
            for i in range(3):
                linear(torch.ones(1, 4) * (i + 1))
        sc = cache_module._module_caches.get(linear)
        assert sc is not None
        assert sc._auto_disabled is True
        # compile=True means the auto_disabled fast-path still benefits from
        # the compiled callable -- don't throw that away by excluding the
        # module from the wrapper entirely.
        assert getattr(linear, "_amd_tuned_torch_no_cache", False) is False


class TestModuleCacheClassCooldownEnv:
    def teardown_method(self):
        cache_module.disable_module_cache()
        cache_module._class_cooldown_until.clear()

    def test_defaults_to_disabled_when_unset(self, monkeypatch):
        monkeypatch.delenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_CLASS_COOLDOWN", raising=False)
        cache_module.enable_module_cache()
        assert cache_module._module_cache_class_cooldown_seconds is None

    def test_reads_class_cooldown_env(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_CLASS_COOLDOWN", "20")
        cache_module.enable_module_cache()
        assert cache_module._module_cache_class_cooldown_seconds == pytest.approx(20.0)

    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_zero_or_negative_env_means_disabled(self, monkeypatch, value):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_CLASS_COOLDOWN", value)
        cache_module.enable_module_cache()
        assert cache_module._module_cache_class_cooldown_seconds is None

    def test_falls_back_to_disabled_on_invalid_value(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_CLASS_COOLDOWN", "not-a-float")
        cache_module.enable_module_cache()
        assert cache_module._module_cache_class_cooldown_seconds is None

    def test_explicit_value_overrides_env(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_CLASS_COOLDOWN", "20")
        cache_module.enable_module_cache(class_cooldown_seconds=5.0)
        assert cache_module._module_cache_class_cooldown_seconds == pytest.approx(5.0)

    def test_trip_puts_other_same_class_instances_on_cooldown(self, monkeypatch):
        clock = _FakeClock()
        monkeypatch.setattr(cache_module.time, "monotonic", clock)
        cache_module.enable_module_cache(thresh=1e-9, min_bytes=None,
                                          max_consecutive_misses=3, class_cooldown_seconds=20.0)
        a = nn.Linear(4, 2)
        b = nn.Linear(4, 2)  # same class as `a`, independent instance

        with torch.no_grad():
            for i in range(3):
                a(torch.ones(1, 4) * (i + 1))  # trips a's breaker
                clock.advance(0.01)

        sc_a = cache_module._module_caches.get(a)
        assert sc_a is not None and sc_a._auto_disabled is True
        assert nn.Linear in cache_module._class_cooldown_until

        # b has never been called before -- no SimilarityCache exists for it
        # yet -- but the class cooldown should route it straight through
        # without ever creating one.
        assert cache_module._module_caches.get(b) is None
        with torch.no_grad():
            out = b(torch.ones(1, 4))
        assert out.shape == (1, 2)
        assert cache_module._module_caches.get(b) is None  # still never created

    def test_cooldown_expires_and_resumes_per_instance_tracking(self, monkeypatch):
        clock = _FakeClock()
        monkeypatch.setattr(cache_module.time, "monotonic", clock)
        cache_module.enable_module_cache(thresh=1e-9, min_bytes=None,
                                          max_consecutive_misses=3, class_cooldown_seconds=1.0)
        a = nn.Linear(4, 2)
        b = nn.Linear(4, 2)

        with torch.no_grad():
            for i in range(3):
                a(torch.ones(1, 4) * (i + 1))  # trips a -> b on cooldown
            clock.advance(2.0)  # past the 1.0s cooldown
            b(torch.ones(1, 4))  # cooldown expired -- b gets its own cache now

        assert cache_module._module_caches.get(b) is not None

    def test_no_cooldown_when_disabled(self, monkeypatch):
        clock = _FakeClock()
        monkeypatch.setattr(cache_module.time, "monotonic", clock)
        cache_module.enable_module_cache(thresh=1e-9, min_bytes=None,
                                          max_consecutive_misses=3, class_cooldown_seconds=None)
        a = nn.Linear(4, 2)
        b = nn.Linear(4, 2)

        with torch.no_grad():
            for i in range(3):
                a(torch.ones(1, 4) * (i + 1))
            b(torch.ones(1, 4))  # no cooldown -- b gets its own cache immediately

        assert cache_module._module_caches.get(b) is not None
        assert cache_module._class_cooldown_until == {}


class TestModuleCacheCompileEnv:
    def teardown_method(self):
        cache_module.disable_module_cache()

    def test_defaults_off_when_unset(self, monkeypatch):
        monkeypatch.delenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_COMPILE", raising=False)
        cache_module.enable_module_cache()
        assert cache_module._module_cache_compile is False

    @pytest.mark.parametrize("value", ["1", "true", "yes"])
    def test_reads_compile_env(self, monkeypatch, value):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_COMPILE", value)
        cache_module.enable_module_cache()
        assert cache_module._module_cache_compile is True

    def test_explicit_value_overrides_env(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_COMPILE", "1")
        cache_module.enable_module_cache(compile=False)
        assert cache_module._module_cache_compile is False

    def test_new_module_caches_inherit_compile_flag(self, monkeypatch):
        import amd_tuned_torch.torch_compile as torch_compile_module

        monkeypatch.setattr(torch_compile_module, "compile_fn", MagicMock(side_effect=lambda f, **kw: f))
        monkeypatch.setattr(torch_compile_module, "available", lambda: True)

        cache_module.enable_module_cache(thresh=0.5, compile=True)
        linear = nn.Linear(4, 2)
        with torch.no_grad():
            linear(torch.ones(1, 4))
        sc = cache_module._module_caches.get(linear)
        assert sc is not None
        assert sc.compile is True


class TestAutoInstallAtImportTime:
    """Mirrors test_amd_tuned_torch_monkeypatch.py's TestTeDisabledByDefault pattern:
    reload cache.py under each env-var state to test the module-import-time
    auto-install, always restoring stock torch.nn.Module.__call__ and
    reloading once more afterward so no other test in the suite ever sees
    a leftover patched state."""

    def teardown_method(self):
        cache_module.disable_module_cache()
        os.environ.pop("AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE", None)
        importlib.reload(cache_module)

    def test_not_installed_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE", raising=False)
        importlib.reload(cache_module)
        assert cache_module.is_module_cache_enabled() is False

    def test_installed_when_env_set(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE", "1")
        importlib.reload(cache_module)
        assert cache_module.is_module_cache_enabled() is True


class TestDebugAutoHandler:
    """AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG=1 must be sufficient on its own to
    produce visible output -- no logging.basicConfig() required by the host
    application (that was the actual bug report: setting the env var alone
    produced nothing during a real run). Reload-based, same pattern as
    TestAutoInstallAtImportTime, since the handler is attached once at
    import time."""

    def teardown_method(self):
        os.environ.pop("AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG", None)
        cache_module.log.handlers.clear()
        importlib.reload(cache_module)

    def test_no_handler_attached_when_debug_unset(self, monkeypatch):
        monkeypatch.delenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG", raising=False)
        cache_module.log.handlers.clear()
        importlib.reload(cache_module)
        assert cache_module.log.handlers == []

    def test_handler_attached_when_debug_set(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG", "1")
        cache_module.log.handlers.clear()
        importlib.reload(cache_module)
        assert len(cache_module.log.handlers) == 1
        assert isinstance(cache_module.log.handlers[0], logging.StreamHandler)
        assert cache_module.log.level == logging.DEBUG

    def test_reload_does_not_duplicate_handlers(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG", "1")
        cache_module.log.handlers.clear()
        importlib.reload(cache_module)
        importlib.reload(cache_module)
        assert len(cache_module.log.handlers) == 1

    def test_debug_output_visible_without_any_logging_setup(self, monkeypatch, capsys):
        # Simulates the real-world report exactly: no logging.basicConfig()
        # anywhere, just the env var, then a normal cache call.
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG", "1")
        cache_module.log.handlers.clear()
        importlib.reload(cache_module)

        cache_module.SimilarityCache(thresh=0.5, enabled=True, debug_name="visible_test")
        err = capsys.readouterr().err  # StreamHandler defaults to stderr
        assert "visible_test" in err


class TestDebugFileHandler:
    """AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG_FILE -- writes the same lifecycle
    logging to a file, independent of (and sufficient on its own, without)
    AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG's stderr handler. Same reload-based
    pattern as TestDebugAutoHandler."""

    def teardown_method(self):
        os.environ.pop("AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG", None)
        os.environ.pop("AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG_FILE", None)
        for h in list(cache_module.log.handlers):
            h.close()
        cache_module.log.handlers.clear()
        importlib.reload(cache_module)

    def test_file_var_alone_turns_debug_on(self, monkeypatch, tmp_path):
        monkeypatch.delenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG", raising=False)
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG_FILE", str(tmp_path / "debug.log"))
        cache_module.log.handlers.clear()
        importlib.reload(cache_module)
        assert cache_module._DEBUG is True

    def test_file_handler_attached_without_stream_handler(self, monkeypatch, tmp_path):
        monkeypatch.delenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG", raising=False)
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG_FILE", str(tmp_path / "debug.log"))
        cache_module.log.handlers.clear()
        importlib.reload(cache_module)
        assert len(cache_module.log.handlers) == 1
        assert isinstance(cache_module.log.handlers[0], logging.FileHandler)

    def test_both_env_vars_attach_both_handlers(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG", "1")
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG_FILE", str(tmp_path / "debug.log"))
        cache_module.log.handlers.clear()
        importlib.reload(cache_module)
        kinds = [type(h) for h in cache_module.log.handlers]
        assert logging.FileHandler in kinds
        assert any(t is logging.StreamHandler for t in kinds)

    def test_reload_does_not_duplicate_file_handler(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG_FILE", str(tmp_path / "debug.log"))
        cache_module.log.handlers.clear()
        importlib.reload(cache_module)
        importlib.reload(cache_module)
        file_handlers = [h for h in cache_module.log.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1

    def test_log_output_actually_written_to_file(self, monkeypatch, tmp_path):
        log_path = tmp_path / "debug.log"
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG_FILE", str(log_path))
        cache_module.log.handlers.clear()
        importlib.reload(cache_module)

        cache_module.SimilarityCache(thresh=0.5, enabled=True, debug_name="file_test")
        for h in cache_module.log.handlers:
            h.flush()
        assert "file_test" in log_path.read_text()

    def test_file_handler_appends_not_truncates(self, monkeypatch, tmp_path):
        log_path = tmp_path / "debug.log"
        log_path.write_text("pre-existing line\n")
        monkeypatch.setenv("AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG_FILE", str(log_path))
        cache_module.log.handlers.clear()
        importlib.reload(cache_module)

        cache_module.SimilarityCache(thresh=0.5, enabled=True, debug_name="append_test")
        for h in cache_module.log.handlers:
            h.flush()
        content = log_path.read_text()
        assert "pre-existing line" in content
        assert "append_test" in content


class TestDebugLogging:
    """AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG -- gates lifecycle logging through the
    standard `logging` module (logger "amd_tuned_torch.cache"). cache_module._DEBUG
    is read once at import time, same pattern as every other env-flag
    module attribute here, so these tests flip it directly rather than
    reloading the module (reload would also re-run the import-time
    enable_module_cache() auto-install based on a *different* env var)."""

    def teardown_method(self):
        cache_module.disable_module_cache()
        cache_module._DEBUG = False

    def test_no_log_output_when_debug_off(self, caplog):
        cache_module._DEBUG = False
        with caplog.at_level("DEBUG", logger="amd_tuned_torch.cache"):
            sc = cache_module.SimilarityCache(thresh=1e-9, enabled=True, max_consecutive_misses=1)
            sc.call(lambda x: x * 2, torch.ones(2, 4))
        assert caplog.records == []

    def test_logs_creation_when_debug_on(self, caplog):
        cache_module._DEBUG = True
        with caplog.at_level("DEBUG", logger="amd_tuned_torch.cache"):
            cache_module.SimilarityCache(thresh=0.5, enabled=True, debug_name="my_module")
        assert any("my_module" in r.message for r in caplog.records)

    def test_logs_breaker_trip_when_debug_on(self, caplog):
        cache_module._DEBUG = True
        sc = cache_module.SimilarityCache(thresh=1e-9, enabled=True, max_consecutive_misses=2,
                                           debug_name="trip_target")
        with caplog.at_level("INFO", logger="amd_tuned_torch.cache"):
            sc.call(lambda x: x * 2, torch.ones(2, 4) * 1)
            sc.call(lambda x: x * 2, torch.ones(2, 4) * 2)  # trips here
        assert any("auto-disabled" in r.message and "trip_target" in r.message for r in caplog.records)

    def test_logs_reset_of_tripped_breaker_when_debug_on(self, caplog):
        cache_module._DEBUG = True
        sc = cache_module.SimilarityCache(thresh=1e-9, enabled=True, max_consecutive_misses=1,
                                           debug_name="reset_target")
        sc.call(lambda x: x * 2, torch.ones(2, 4))  # trips immediately
        assert sc._auto_disabled is True
        with caplog.at_level("INFO", logger="amd_tuned_torch.cache"):
            sc.reset()
        assert any("reset_target" in r.message and "reset" in r.message for r in caplog.records)


class TestDebugSummary:
    def teardown_method(self):
        cache_module.disable_module_cache()

    def test_empty_when_no_module_caches_recorded(self, capsys):
        rows = cache_module.debug_summary(print_output=False)
        assert rows == []

    def test_prints_placeholder_when_empty(self, capsys):
        cache_module.debug_summary(print_output=True)
        out = capsys.readouterr().out
        assert "no module caches recorded" in out

    def test_reports_counters_per_module(self, capsys):
        cache_module.enable_module_cache(thresh=0.5, min_bytes=None)
        linear = nn.Linear(4, 2)
        with torch.no_grad():
            linear(torch.ones(1, 4))
            linear(torch.ones(1, 4) * 1.01)  # hit

        rows = cache_module.debug_summary(print_output=False)
        assert len(rows) == 1
        name, hits, misses, size_bp, breaker_bp, auto_dis, reason = rows[0]
        assert hits == 1
        assert misses == 1
        assert size_bp == 0
        assert breaker_bp == 0
        assert auto_dis is False
        assert reason is None
        assert "Linear" in name

    def test_sorted_by_total_calls_descending(self, capsys):
        cache_module.enable_module_cache(thresh=1e-9, min_bytes=None)
        busy = nn.Linear(4, 2)
        quiet = nn.Linear(4, 2)
        with torch.no_grad():
            for i in range(5):
                busy(torch.ones(1, 4) * i)  # 5 distinct misses
            quiet(torch.ones(1, 4))  # 1 miss

        rows = cache_module.debug_summary(print_output=False)
        assert len(rows) == 2
        assert rows[0][2] == 5  # busiest module's miss count listed first
        assert rows[1][2] == 1

    def test_top_n_limits_results(self, capsys):
        cache_module.enable_module_cache(thresh=0.5, min_bytes=None)
        with torch.no_grad():
            for _ in range(5):
                nn.Linear(4, 2)(torch.ones(1, 4))  # 5 distinct instances

        rows = cache_module.debug_summary(top_n=2, print_output=False)
        assert len(rows) == 2

    def test_prints_a_table_when_requested(self, capsys):
        cache_module.enable_module_cache(thresh=0.5, min_bytes=None)
        with torch.no_grad():
            nn.Linear(4, 2)(torch.ones(1, 4))
        cache_module.debug_summary(print_output=True)
        out = capsys.readouterr().out
        assert "hits" in out
        assert "Linear" in out

    def test_reports_trip_reason(self, capsys):
        cache_module.enable_module_cache(thresh=1e-9, min_bytes=None, max_consecutive_misses=3)
        linear = nn.Linear(4, 2)
        with torch.no_grad():
            for i in range(3):
                linear(torch.ones(1, 4) * (i + 1))  # always dissimilar -> trips at 3

        rows = cache_module.debug_summary(print_output=False)
        assert len(rows) == 1
        _name, _hits, _misses, _size_bp, _breaker_bp, auto_dis, reason = rows[0]
        assert auto_dis is True
        assert "3 consecutive misses" in reason

        cache_module.debug_summary(print_output=True)
        out = capsys.readouterr().out
        assert "consecutive misses" in out
