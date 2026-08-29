"""Tests for amd_tuned_torch.hub_ops -- the optional Hugging Face Hub kernel
backend. No real network/Hub access anywhere here: kernels.get_kernel
itself is always mocked, same discipline as aiter/TE in
test_amd_tuned_torch_monkeypatch.py.

Run with:

    python -m pytest source/cmp_ext_turing/tests
"""
from __future__ import annotations

import importlib
import os
import types
from unittest.mock import MagicMock

import pytest

import amd_tuned_torch.hub_ops as hub_ops_module


# ---------------------------------------------------------------------------
# AMD_TUNED_TORCH_ENABLE_HUB_KERNELS gate -- read once at import time, same posture
# as AMD_TUNED_TORCH_ENABLE_TE (see TestTeDisabledByDefault in
# test_amd_tuned_torch_monkeypatch.py for the reference pattern this mirrors).
# ---------------------------------------------------------------------------


class TestEnabledByDefaultOff:
    def teardown_method(self):
        os.environ.pop("AMD_TUNED_TORCH_ENABLE_HUB_KERNELS", None)
        importlib.reload(hub_ops_module)

    def test_disabled_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("AMD_TUNED_TORCH_ENABLE_HUB_KERNELS", raising=False)
        importlib.reload(hub_ops_module)
        assert hub_ops_module._ENABLED is False

    def test_disabled_when_explicitly_zero(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_ENABLE_HUB_KERNELS", "0")
        importlib.reload(hub_ops_module)
        assert hub_ops_module._ENABLED is False

    def test_enabled_when_set(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_ENABLE_HUB_KERNELS", "1")
        importlib.reload(hub_ops_module)
        assert hub_ops_module._ENABLED is True

    def test_available_requires_both_package_and_env(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_ENABLE_HUB_KERNELS", "1")
        importlib.reload(hub_ops_module)
        monkeypatch.setattr(hub_ops_module, "_KERNELS_PACKAGE_AVAILABLE", False)
        assert hub_ops_module.available() is False
        monkeypatch.setattr(hub_ops_module, "_KERNELS_PACKAGE_AVAILABLE", True)
        assert hub_ops_module.available() is True


# ---------------------------------------------------------------------------
# get_hub_kernel: caching + broad failure handling.
# ---------------------------------------------------------------------------


class TestGetHubKernel:
    def teardown_method(self):
        hub_ops_module._kernel_cache.clear()
        hub_ops_module._failed_repos.clear()

    def test_returns_none_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(hub_ops_module, "available", lambda: False)
        assert hub_ops_module.get_hub_kernel("some/repo", version=1) is None

    def test_returns_and_caches_successful_fetch(self, monkeypatch):
        monkeypatch.setattr(hub_ops_module, "available", lambda: True)
        fake_module = types.ModuleType("fake_kernel")
        fake_get_kernel = MagicMock(return_value=fake_module)
        monkeypatch.setattr(hub_ops_module, "_hub_kernels", MagicMock(get_kernel=fake_get_kernel))

        result1 = hub_ops_module.get_hub_kernel("some/repo", version=1)
        result2 = hub_ops_module.get_hub_kernel("some/repo", version=1)

        assert result1 is fake_module
        assert result2 is fake_module
        fake_get_kernel.assert_called_once_with("some/repo", version=1, revision=None)

    def test_different_keys_are_cached_independently(self, monkeypatch):
        monkeypatch.setattr(hub_ops_module, "available", lambda: True)
        modules = {1: types.ModuleType("v1"), 2: types.ModuleType("v2")}
        fake_get_kernel = MagicMock(side_effect=lambda repo_id, version, revision: modules[version])
        monkeypatch.setattr(hub_ops_module, "_hub_kernels", MagicMock(get_kernel=fake_get_kernel))

        r1 = hub_ops_module.get_hub_kernel("some/repo", version=1)
        r2 = hub_ops_module.get_hub_kernel("some/repo", version=2)
        assert r1 is modules[1]
        assert r2 is modules[2]
        assert fake_get_kernel.call_count == 2

    def test_any_exception_returns_none_instead_of_raising(self, monkeypatch):
        monkeypatch.setattr(hub_ops_module, "available", lambda: True)

        class SomeWeirdHubError(Exception):
            pass

        fake_get_kernel = MagicMock(side_effect=SomeWeirdHubError("no internet"))
        monkeypatch.setattr(hub_ops_module, "_hub_kernels", MagicMock(get_kernel=fake_get_kernel))

        result = hub_ops_module.get_hub_kernel("some/repo", version=1)
        assert result is None

    def test_failed_fetch_is_cached_and_not_retried(self, monkeypatch):
        monkeypatch.setattr(hub_ops_module, "available", lambda: True)
        fake_get_kernel = MagicMock(side_effect=RuntimeError("network down"))
        monkeypatch.setattr(hub_ops_module, "_hub_kernels", MagicMock(get_kernel=fake_get_kernel))

        hub_ops_module.get_hub_kernel("some/repo", version=1)
        hub_ops_module.get_hub_kernel("some/repo", version=1)

        assert fake_get_kernel.call_count == 1

    # Note: no concrete kernel repo is wired up as a backend in this module.
    # kernels-community/aiter-kernels (a real ROCm-native repackaging of
    # aiter, confirmed via its file tree) was checked and turned out to be
    # the exact same aiter.ops.triton.activation.fused_silu_mul already
    # importable locally -- see amd_tuned_torch/aiter_ops.py and
    # tests/test_aiter_ops.py instead, no network dependency needed.
