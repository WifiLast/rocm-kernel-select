"""Tests for amd_tuned_torch.sparse_conv_calibration -- the persisted,
measured-crossover cache flexgemm_ops.py loads to override its hardcoded
AMD_TUNED_TORCH_SPARSE_CONV{1,2,3}D_MIN_POSITIONS/_MAX_OCCUPANCY defaults. No
real GPU benchmarking here: this only tests the load/save/reset
persistence mechanics against a temp directory, never
tools/benchmark_sparse_conv.py's actual measurement.

Run with:

    python -m pytest source/cmp_ext_turing/tests
"""
from __future__ import annotations

import importlib
import json
import os

import amd_tuned_torch.sparse_conv_calibration as calibration_module


class TestDeviceKey:
    def test_returns_a_nonempty_string(self):
        assert isinstance(calibration_module.device_key(), str)
        assert len(calibration_module.device_key()) > 0

    def test_includes_torch_and_python_version(self):
        import sys
        import torch
        key = calibration_module.device_key()
        assert f"cp{sys.version_info.major}{sys.version_info.minor}" in key
        assert torch.__version__.replace("+", "_") in key


class TestLoadSaveRoundTrip:
    def setup_method(self):
        self._orig_dir = calibration_module._CALIBRATION_DIR

    def teardown_method(self, tmp_path=None):
        calibration_module._CALIBRATION_DIR = self._orig_dir

    def test_load_returns_empty_dict_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(calibration_module, "_CALIBRATION_DIR", str(tmp_path))
        assert calibration_module.load() == {}

    def test_save_then_load_round_trips_min_positions(self, tmp_path, monkeypatch):
        monkeypatch.setattr(calibration_module, "_CALIBRATION_DIR", str(tmp_path))
        calibration_module.save({"conv2d": {"min_positions": 12345}})
        assert calibration_module.load() == {"conv2d": {"min_positions": 12345}}

    def test_save_then_load_round_trips_max_occupancy(self, tmp_path, monkeypatch):
        monkeypatch.setattr(calibration_module, "_CALIBRATION_DIR", str(tmp_path))
        calibration_module.save({"conv2d": {"max_occupancy": 0.2}})
        assert calibration_module.load() == {"conv2d": {"max_occupancy": 0.2}}

    def test_save_then_load_round_trips_both_fields(self, tmp_path, monkeypatch):
        monkeypatch.setattr(calibration_module, "_CALIBRATION_DIR", str(tmp_path))
        calibration_module.save({"conv2d": {"min_positions": 1000, "max_occupancy": 0.15}})
        assert calibration_module.load() == {"conv2d": {"min_positions": 1000, "max_occupancy": 0.15}}

    def test_save_merges_rather_than_overwrites_other_dims(self, tmp_path, monkeypatch):
        monkeypatch.setattr(calibration_module, "_CALIBRATION_DIR", str(tmp_path))
        calibration_module.save({"conv1d": {"min_positions": 100}})
        calibration_module.save({"conv2d": {"min_positions": 200}})
        assert calibration_module.load() == {
            "conv1d": {"min_positions": 100},
            "conv2d": {"min_positions": 200},
        }

    def test_save_merges_fields_within_same_dim(self, tmp_path, monkeypatch):
        """Calibrating min_positions today and max_occupancy tomorrow (or
        via separate --sweep runs) must not clobber the other field."""
        monkeypatch.setattr(calibration_module, "_CALIBRATION_DIR", str(tmp_path))
        calibration_module.save({"conv2d": {"min_positions": 1000}})
        calibration_module.save({"conv2d": {"max_occupancy": 0.25}})
        assert calibration_module.load() == {"conv2d": {"min_positions": 1000, "max_occupancy": 0.25}}

    def test_save_overwrites_same_field(self, tmp_path, monkeypatch):
        monkeypatch.setattr(calibration_module, "_CALIBRATION_DIR", str(tmp_path))
        calibration_module.save({"conv3d": {"min_positions": 100}})
        calibration_module.save({"conv3d": {"min_positions": 200}})
        assert calibration_module.load() == {"conv3d": {"min_positions": 200}}

    def test_extra_metadata_is_stored_but_not_returned_by_load(self, tmp_path, monkeypatch):
        monkeypatch.setattr(calibration_module, "_CALIBRATION_DIR", str(tmp_path))
        calibration_module.save({"conv2d": {"min_positions": 500}},
                                 extra_metadata={"conv2d": {"occupancy": 0.05}})
        assert calibration_module.load() == {"conv2d": {"min_positions": 500}}
        with open(calibration_module.calibration_path()) as f:
            raw = json.load(f)
        assert raw["conv2d"]["occupancy"] == 0.05

    def test_unknown_dim_key_is_ignored(self, tmp_path, monkeypatch):
        monkeypatch.setattr(calibration_module, "_CALIBRATION_DIR", str(tmp_path))
        calibration_module.save({"conv2d": {"min_positions": 10}, "not_a_real_dim": {"min_positions": 20}})
        assert calibration_module.load() == {"conv2d": {"min_positions": 10}}

    def test_unknown_field_key_is_ignored(self, tmp_path, monkeypatch):
        monkeypatch.setattr(calibration_module, "_CALIBRATION_DIR", str(tmp_path))
        calibration_module.save({"conv2d": {"min_positions": 10, "not_a_real_field": 999}})
        assert calibration_module.load() == {"conv2d": {"min_positions": 10}}

    def test_corrupt_file_returns_empty_dict(self, tmp_path, monkeypatch):
        monkeypatch.setattr(calibration_module, "_CALIBRATION_DIR", str(tmp_path))
        path = calibration_module.calibration_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("{not valid json")
        assert calibration_module.load() == {}

    def test_dim_present_with_no_valid_fields_is_omitted(self, tmp_path, monkeypatch):
        monkeypatch.setattr(calibration_module, "_CALIBRATION_DIR", str(tmp_path))
        path = calibration_module.calibration_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"conv2d": {"min_positions": "not an int"}}, f)
        assert calibration_module.load() == {}

    def test_reset_deletes_the_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(calibration_module, "_CALIBRATION_DIR", str(tmp_path))
        calibration_module.save({"conv2d": {"min_positions": 10}})
        assert os.path.exists(calibration_module.calibration_path())
        calibration_module.reset()
        assert not os.path.exists(calibration_module.calibration_path())
        assert calibration_module.load() == {}

    def test_reset_on_missing_file_is_a_noop(self, tmp_path, monkeypatch):
        monkeypatch.setattr(calibration_module, "_CALIBRATION_DIR", str(tmp_path))
        calibration_module.reset()  # must not raise


class TestEnabledFlag:
    def teardown_method(self):
        os.environ.pop("AMD_TUNED_TORCH_SPARSE_CONV_CALIBRATION", None)
        importlib.reload(calibration_module)

    def test_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("AMD_TUNED_TORCH_SPARSE_CONV_CALIBRATION", raising=False)
        importlib.reload(calibration_module)
        assert calibration_module.enabled() is True

    def test_disabled_when_explicitly_zero(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_SPARSE_CONV_CALIBRATION", "0")
        importlib.reload(calibration_module)
        assert calibration_module.enabled() is False

    def test_load_returns_empty_when_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setattr(calibration_module, "_CALIBRATION_DIR", str(tmp_path))
        calibration_module.save({"conv2d": {"min_positions": 10}})
        monkeypatch.setattr(calibration_module, "_ENABLED", False)
        assert calibration_module.load() == {}


class TestResetFlag:
    def test_load_ignores_and_deletes_file_when_reset_flag_set(self, tmp_path, monkeypatch):
        monkeypatch.setattr(calibration_module, "_CALIBRATION_DIR", str(tmp_path))
        calibration_module.save({"conv2d": {"min_positions": 10}})
        monkeypatch.setattr(calibration_module, "_RESET", True)

        assert calibration_module.load() == {}
        assert not os.path.exists(calibration_module.calibration_path())
