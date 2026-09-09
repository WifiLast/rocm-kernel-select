"""Tests for amd_tuned_torch/rocm_env_check.py -- pure predicate logic over a
mocked _Env/_Tiers pair, no real ROCm install or GPU needed (see that
module's own docstring for why the rules are structured this way)."""
from __future__ import annotations

import warnings

import pytest

from amd_tuned_torch import rocm_env_check as rec


def _env(**kv):
    return rec._Env(dict(kv))


def _tiers(aiter=False, ck_gemm=False, hipblaslt=False, autopatch=True):
    return rec._Tiers(aiter=aiter, ck_gemm=ck_gemm, hipblaslt=hipblaslt, autopatch=autopatch)


def _rule(name):
    for rule in rec._RULES:
        if rule.name == name:
            return rule
    raise KeyError(name)


class TestTruthy:
    def test_unset_is_false(self):
        assert rec._truthy(None) is False

    @pytest.mark.parametrize("value", ["0", "", "false", "False"])
    def test_falsy_strings(self, value):
        assert rec._truthy(value) is False

    @pytest.mark.parametrize("value", ["1", "2", "yes", "true"])
    def test_truthy_strings(self, value):
        assert rec._truthy(value) is True


class TestHipLaunchBlocking:
    def test_triggers_with_autopatch(self):
        rule = _rule("hip_launch_blocking")
        assert rule.predicate(_env(HIP_LAUNCH_BLOCKING="1"), _tiers(autopatch=True))

    def test_not_triggered_without_autopatch(self):
        rule = _rule("hip_launch_blocking")
        assert not rule.predicate(_env(HIP_LAUNCH_BLOCKING="1"), _tiers(autopatch=False))

    def test_not_triggered_when_unset(self):
        rule = _rule("hip_launch_blocking")
        assert not rule.predicate(_env(), _tiers(autopatch=True))


class TestMiopenFindEnforceSearch:
    @pytest.mark.parametrize("enforce", ["2", "3", "4", "SEARCH", "SEARCH_DB_UPDATE"])
    def test_triggers_on_search_values_alone(self, enforce, monkeypatch):
        # No MIOPEN_DEBUG_DISABLE_FIND_DB needed -- SEARCH mode bypasses the
        # find-db for deciding regardless of whether it's also disabled.
        monkeypatch.delenv("AMD_TUNED_TORCH_MEASURE_KERNELS", raising=False)
        rule = _rule("miopen_find_enforce_search")
        env = _env(MIOPEN_FIND_ENFORCE=enforce)
        assert rule.predicate(env, _tiers())

    def test_still_triggers_with_find_db_also_disabled(self, monkeypatch):
        monkeypatch.delenv("AMD_TUNED_TORCH_MEASURE_KERNELS", raising=False)
        rule = _rule("miopen_find_enforce_search")
        env = _env(MIOPEN_DEBUG_DISABLE_FIND_DB="1", MIOPEN_FIND_ENFORCE="SEARCH")
        assert rule.predicate(env, _tiers())

    def test_not_triggered_with_mild_enforce_value(self, monkeypatch):
        monkeypatch.delenv("AMD_TUNED_TORCH_MEASURE_KERNELS", raising=False)
        rule = _rule("miopen_find_enforce_search")
        env = _env(MIOPEN_FIND_ENFORCE="1")
        assert not rule.predicate(env, _tiers())

    def test_not_triggered_when_kernel_select_disabled(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_MEASURE_KERNELS", "0")
        rule = _rule("miopen_find_enforce_search")
        env = _env(MIOPEN_FIND_ENFORCE="SEARCH")
        assert not rule.predicate(env, _tiers())


class TestKernelSelectDisabledWastesTiers:
    def test_triggers_when_ck_gemm_available(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_MEASURE_KERNELS", "0")
        rule = _rule("kernel_select_disabled_wastes_compiled_tiers")
        assert rule.predicate(_env(), _tiers(ck_gemm=True))

    def test_triggers_when_hipblaslt_available(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_MEASURE_KERNELS", "0")
        rule = _rule("kernel_select_disabled_wastes_compiled_tiers")
        assert rule.predicate(_env(), _tiers(hipblaslt=True))

    def test_not_triggered_when_neither_tier_available(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_MEASURE_KERNELS", "0")
        rule = _rule("kernel_select_disabled_wastes_compiled_tiers")
        assert not rule.predicate(_env(), _tiers(ck_gemm=False, hipblaslt=False))

    def test_not_triggered_when_kernel_select_enabled(self, monkeypatch):
        monkeypatch.delenv("AMD_TUNED_TORCH_MEASURE_KERNELS", raising=False)
        rule = _rule("kernel_select_disabled_wastes_compiled_tiers")
        assert not rule.predicate(_env(), _tiers(ck_gemm=True))


class TestTunableopNumericalCheck:
    def test_triggers_when_both_set(self):
        rule = _rule("tunableop_numerical_check")
        env = _env(PYTORCH_TUNABLEOP_ENABLED="1", PYTORCH_TUNABLEOP_NUMERICAL_CHECK="1")
        assert rule.predicate(env, _tiers())

    def test_not_triggered_with_only_enabled(self):
        rule = _rule("tunableop_numerical_check")
        env = _env(PYTORCH_TUNABLEOP_ENABLED="1")
        assert not rule.predicate(env, _tiers())


class TestHsaOverrideGfxVersion:
    def test_triggers_when_set(self):
        rule = _rule("hsa_override_gfx_version")
        assert rule.predicate(_env(HSA_OVERRIDE_GFX_VERSION="10.3.0"), _tiers())

    def test_not_triggered_when_unset(self):
        rule = _rule("hsa_override_gfx_version")
        assert not rule.predicate(_env(), _tiers())

    def test_message_includes_the_actual_value(self):
        rule = _rule("hsa_override_gfx_version")
        env = _env(HSA_OVERRIDE_GFX_VERSION="10.3.0")
        assert "10.3.0" in rule.message(env, _tiers())


class TestHsaEnableInterruptDisabled:
    def test_triggers_with_autopatch(self):
        rule = _rule("hsa_enable_interrupt_disabled")
        assert rule.predicate(_env(HSA_ENABLE_INTERRUPT="0"), _tiers(autopatch=True))

    def test_not_triggered_without_autopatch(self):
        rule = _rule("hsa_enable_interrupt_disabled")
        assert not rule.predicate(_env(HSA_ENABLE_INTERRUPT="0"), _tiers(autopatch=False))

    def test_not_triggered_when_enabled(self):
        rule = _rule("hsa_enable_interrupt_disabled")
        assert not rule.predicate(_env(HSA_ENABLE_INTERRUPT="1"), _tiers(autopatch=True))


class TestAmdSerializeKernel:
    def test_triggers_with_autopatch(self):
        rule = _rule("amd_serialize_kernel")
        assert rule.predicate(_env(AMD_SERIALIZE_KERNEL="3"), _tiers(autopatch=True))

    def test_not_triggered_without_autopatch(self):
        rule = _rule("amd_serialize_kernel")
        assert not rule.predicate(_env(AMD_SERIALIZE_KERNEL="3"), _tiers(autopatch=False))

    def test_not_triggered_when_zero(self):
        rule = _rule("amd_serialize_kernel")
        assert not rule.predicate(_env(AMD_SERIALIZE_KERNEL="0"), _tiers(autopatch=True))


class TestGpuMaxHwQueuesSerialized:
    def test_triggers_when_one_with_autopatch(self):
        rule = _rule("gpu_max_hw_queues_serialized")
        assert rule.predicate(_env(GPU_MAX_HW_QUEUES="1"), _tiers(autopatch=True))

    def test_not_triggered_without_autopatch(self):
        rule = _rule("gpu_max_hw_queues_serialized")
        assert not rule.predicate(_env(GPU_MAX_HW_QUEUES="1"), _tiers(autopatch=False))

    def test_not_triggered_with_higher_value(self):
        rule = _rule("gpu_max_hw_queues_serialized")
        assert not rule.predicate(_env(GPU_MAX_HW_QUEUES="4"), _tiers(autopatch=True))


class TestHipMemoryCachingDisabled:
    def test_triggers_with_kernel_select_enabled(self, monkeypatch):
        monkeypatch.delenv("AMD_TUNED_TORCH_MEASURE_KERNELS", raising=False)
        monkeypatch.delenv("AMD_TUNED_TORCH_CONV_MEASURE", raising=False)
        monkeypatch.delenv("AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE", raising=False)
        rule = _rule("hip_memory_caching_disabled")
        assert rule.predicate(_env(PYTORCH_NO_HIP_MEMORY_CACHING="1"), _tiers())

    def test_triggers_with_similarity_cache_enabled(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_MEASURE_KERNELS", "0")
        monkeypatch.setenv("AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE", "1")
        rule = _rule("hip_memory_caching_disabled")
        assert rule.predicate(_env(PYTORCH_NO_HIP_MEMORY_CACHING="1"), _tiers())

    def test_not_triggered_when_both_disabled(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_MEASURE_KERNELS", "0")
        monkeypatch.delenv("AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE", raising=False)
        rule = _rule("hip_memory_caching_disabled")
        assert not rule.predicate(_env(PYTORCH_NO_HIP_MEMORY_CACHING="1"), _tiers())

    def test_not_triggered_when_caching_not_disabled(self, monkeypatch):
        monkeypatch.delenv("AMD_TUNED_TORCH_MEASURE_KERNELS", raising=False)
        rule = _rule("hip_memory_caching_disabled")
        assert not rule.predicate(_env(), _tiers())


class TestMiopenDbPathNotWritable:
    def test_triggers_for_unwritable_user_path(self, monkeypatch):
        monkeypatch.setattr(rec, "_is_writable_dir", lambda p: p != "/bad/path")
        rule = _rule("miopen_db_path_not_writable")
        env = _env(MIOPEN_USER_DB_PATH="/bad/path")
        assert rule.predicate(env, _tiers())
        assert "MIOPEN_USER_DB_PATH=/bad/path" in rule.message(env, _tiers())

    def test_not_triggered_for_writable_path(self, monkeypatch):
        monkeypatch.setattr(rec, "_is_writable_dir", lambda p: True)
        rule = _rule("miopen_db_path_not_writable")
        assert not rule.predicate(_env(MIOPEN_USER_DB_PATH="/good/path"), _tiers())

    def test_not_triggered_when_unset(self):
        rule = _rule("miopen_db_path_not_writable")
        assert not rule.predicate(_env(), _tiers())

    def test_message_names_the_actually_bad_var_not_the_other_one(self, monkeypatch):
        # Regression test: the message must never blame SYSTEM_DB_PATH for
        # a USER_DB_PATH failure or vice versa.
        monkeypatch.setattr(rec, "_is_writable_dir", lambda p: p != "/bad/user/path")
        rule = _rule("miopen_db_path_not_writable")
        env = _env(MIOPEN_USER_DB_PATH="/bad/user/path",
                    MIOPEN_SYSTEM_DB_PATH="/good/system/path")
        msg = rule.message(env, _tiers())
        assert "MIOPEN_USER_DB_PATH=/bad/user/path" in msg
        assert "MIOPEN_SYSTEM_DB_PATH" not in msg


class TestRocmProfilerAttached:
    @pytest.mark.parametrize("var", ["HSA_TOOLS_LIB", "ROCPROFILER_METRICS_PATH", "HIP_TRACE_API"])
    def test_triggers_with_autopatch(self, var):
        rule = _rule("rocm_profiler_attached")
        assert rule.predicate(_env(**{var: "/some/path"}), _tiers(autopatch=True))

    def test_not_triggered_without_autopatch(self):
        rule = _rule("rocm_profiler_attached")
        assert not rule.predicate(_env(HSA_TOOLS_LIB="/some/path"), _tiers(autopatch=False))

    def test_not_triggered_when_none_set(self):
        rule = _rule("rocm_profiler_attached")
        assert not rule.predicate(_env(), _tiers(autopatch=True))


class TestSparseConvCalibrationResetLeftOn:
    def test_triggers_when_set(self):
        rule = _rule("sparse_conv_calibration_reset_left_on")
        assert rule.predicate(
            _env(AMD_TUNED_TORCH_SPARSE_CONV_CALIBRATION_RESET="1"), _tiers())

    def test_not_triggered_when_unset(self):
        rule = _rule("sparse_conv_calibration_reset_left_on")
        assert not rule.predicate(_env(), _tiers())

    def test_not_triggered_when_explicitly_zero(self):
        rule = _rule("sparse_conv_calibration_reset_left_on")
        assert not rule.predicate(
            _env(AMD_TUNED_TORCH_SPARSE_CONV_CALIBRATION_RESET="0"), _tiers())

    def test_does_not_depend_on_autopatch(self):
        """Unlike most rules here, this one is about
        sparse_conv_calibration.load() (consulted by flexgemm_ops at import
        time, independent of whether amd_tuned_torch.enable() has patched
        anything), so it must trigger the same way regardless of tiers.autopatch."""
        rule = _rule("sparse_conv_calibration_reset_left_on")
        env = _env(AMD_TUNED_TORCH_SPARSE_CONV_CALIBRATION_RESET="1")
        assert rule.predicate(env, _tiers(autopatch=True))
        assert rule.predicate(env, _tiers(autopatch=False))


class TestIsWritableDir:
    def test_existing_writable_dir(self, tmp_path):
        assert rec._is_writable_dir(str(tmp_path)) is True

    def test_nonexistent_path_falls_back_to_parent(self, tmp_path):
        assert rec._is_writable_dir(str(tmp_path / "not_yet_created")) is True

    def test_bogus_path_does_not_raise(self):
        # A malformed env var value must never break the whole check() pass
        # -- the exact result doesn't matter here (a NUL-containing string
        # has no real parent to fall back to), only that this returns a
        # plain bool instead of raising.
        assert isinstance(rec._is_writable_dir("\x00bad"), bool)


class TestCheckFunction:
    def test_no_findings_on_clean_environment(self, monkeypatch):
        for name in (
            "HIP_LAUNCH_BLOCKING", "MIOPEN_DEBUG_DISABLE_FIND_DB", "MIOPEN_FIND_ENFORCE",
            "ROCBLAS_LAYER", "AMD_LOG_LEVEL", "AMD_TUNED_TORCH_MEASURE_KERNELS",
            "AMD_TUNED_TORCH_CONV_MEASURE", "PYTORCH_TUNABLEOP_ENABLED",
            "PYTORCH_TUNABLEOP_NUMERICAL_CHECK", "HSA_ENABLE_SDMA",
            "AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE", "HSA_OVERRIDE_GFX_VERSION",
            "HSA_ENABLE_INTERRUPT", "AMD_SERIALIZE_KERNEL", "GPU_MAX_HW_QUEUES",
            "PYTORCH_NO_HIP_MEMORY_CACHING", "MIOPEN_USER_DB_PATH", "MIOPEN_SYSTEM_DB_PATH",
            "HSA_TOOLS_LIB", "ROCPROFILER_METRICS_PATH", "HIP_TRACE_API",
        ):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(rec, "_tiers", lambda: _tiers())
        assert rec.check(warn=False) == []

    def test_finding_returned_and_warned(self, monkeypatch):
        monkeypatch.setenv("HIP_LAUNCH_BLOCKING", "1")
        monkeypatch.setattr(rec, "_tiers", lambda: _tiers(autopatch=True))
        with pytest.warns(UserWarning):
            findings = rec.check(warn=True)
        assert any(f.name == "hip_launch_blocking" for f in findings)

    def test_warn_false_suppresses_warnings(self, monkeypatch):
        monkeypatch.setenv("HIP_LAUNCH_BLOCKING", "1")
        monkeypatch.setattr(rec, "_tiers", lambda: _tiers(autopatch=True))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            findings = rec.check(warn=False)
        assert any(f.name == "hip_launch_blocking" for f in findings)

    def test_a_broken_rule_does_not_break_check(self, monkeypatch):
        def _boom(env, tiers):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(rec, "_RULES", rec._RULES + [
            rec._Rule(name="broken", severity="warning", predicate=_boom, message="x")
        ])
        monkeypatch.setattr(rec, "_tiers", lambda: _tiers())
        # Must not raise, and must not report the broken rule as triggered.
        findings = rec.check(warn=False)
        assert not any(f.name == "broken" for f in findings)
