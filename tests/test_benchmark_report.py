"""Tests for amd_tuned_torch.benchmark_report -- combines whatever
sparse_conv_calibration.py/fftconv_calibration.py have already written on
this machine, plus system metadata (GPU/ROCm/torch/Python version), into
one JSON report. No real GPU benchmarking here: this only tests the
collect/save mechanics against temp calibration directories, never an
actual benchmark tool's measurement.

Run with:

    python -m pytest source/cmp_ext_turing/tests
"""
from __future__ import annotations

import json
import os

import pytest
import torch

import amd_tuned_torch.benchmark_report as benchmark_report
import amd_tuned_torch.fftconv_calibration as fftconv_calibration
import amd_tuned_torch.sparse_conv_calibration as sparse_conv_calibration


class TestSystemMetadata:
    def test_has_expected_keys(self):
        meta = benchmark_report.system_metadata()
        assert set(meta) == {
            "gpu_name", "rocm_version", "cuda_version",
            "torch_version", "python_version", "platform", "pcie",
        }

    def test_pcie_is_gpu_pcie_infos_own_result(self, monkeypatch):
        sentinel = {"link_speed": "16.0 GT/s PCIe", "link_width": "16"}
        monkeypatch.setattr(benchmark_report, "gpu_pcie_info", lambda: sentinel)
        assert benchmark_report.system_metadata()["pcie"] is sentinel

    def test_torch_version_matches_real_torch(self):
        assert benchmark_report.system_metadata()["torch_version"] == torch.__version__

    def test_rocm_and_cuda_version_match_torch_version_attrs(self):
        """Not asserting a specific value (this test environment has
        neither ROCm nor CUDA) -- just that these are read straight from
        torch.version.hip/.cuda, not hardcoded or guessed."""
        meta = benchmark_report.system_metadata()
        assert meta["rocm_version"] == torch.version.hip
        assert meta["cuda_version"] == torch.version.cuda

    def test_gpu_name_is_none_without_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert benchmark_report.system_metadata()["gpu_name"] is None


class TestDeviceKey:
    def test_returns_a_nonempty_string(self):
        assert isinstance(benchmark_report.device_key(), str)
        assert len(benchmark_report.device_key()) > 0

    def test_matches_calibration_modules_own_device_key(self):
        """The combined report's device_key must be the SAME key
        sparse_conv_calibration/fftconv_calibration use for their own
        per-device filenames -- otherwise collect() would look in the
        wrong place, or a reader matching files across the three by name
        would silently fail to."""
        assert benchmark_report.device_key() == sparse_conv_calibration.device_key()
        assert benchmark_report.device_key() == fftconv_calibration.device_key()


class TestCollect:
    def setup_method(self):
        self._orig_sparse_dir = sparse_conv_calibration._CALIBRATION_DIR
        self._orig_fftconv_dir = fftconv_calibration._CALIBRATION_DIR

    def teardown_method(self):
        sparse_conv_calibration._CALIBRATION_DIR = self._orig_sparse_dir
        fftconv_calibration._CALIBRATION_DIR = self._orig_fftconv_dir

    def test_omits_domains_with_no_calibration_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sparse_conv_calibration, "_CALIBRATION_DIR", str(tmp_path / "sparse"))
        monkeypatch.setattr(fftconv_calibration, "_CALIBRATION_DIR", str(tmp_path / "fftconv"))
        report = benchmark_report.collect()
        assert "sparse_conv" not in report
        assert "fftconv" not in report

    def test_always_has_metadata_even_with_nothing_calibrated(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sparse_conv_calibration, "_CALIBRATION_DIR", str(tmp_path / "sparse"))
        monkeypatch.setattr(fftconv_calibration, "_CALIBRATION_DIR", str(tmp_path / "fftconv"))
        report = benchmark_report.collect()
        assert report["schema_version"] == 1
        assert report["device_key"] == benchmark_report.device_key()
        assert "generated_at" in report
        assert report["system"]["torch_version"] == torch.__version__

    def test_includes_sparse_conv_domain_when_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sparse_conv_calibration, "_CALIBRATION_DIR", str(tmp_path / "sparse"))
        monkeypatch.setattr(fftconv_calibration, "_CALIBRATION_DIR", str(tmp_path / "fftconv"))
        sparse_conv_calibration.save({"conv2d": {"min_positions": 500, "max_occupancy": 0.12}})
        report = benchmark_report.collect()
        assert report["sparse_conv"]["conv2d"] == {"min_positions": 500, "max_occupancy": 0.12}
        assert "fftconv" not in report

    def test_includes_fftconv_domain_when_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sparse_conv_calibration, "_CALIBRATION_DIR", str(tmp_path / "sparse"))
        monkeypatch.setattr(fftconv_calibration, "_CALIBRATION_DIR", str(tmp_path / "fftconv"))
        fftconv_calibration.save({"conv3d": {"min_positions": 32768}})
        report = benchmark_report.collect()
        assert report["fftconv"]["conv3d"] == {"min_positions": 32768}
        assert "sparse_conv" not in report

    def test_includes_both_domains_and_their_extra_metadata(self, tmp_path, monkeypatch):
        """collect() surfaces the RAW on-disk file (extra_metadata sweep
        details included), not just each module's own filtered load()
        view -- "all the results", not only the two final numbers."""
        monkeypatch.setattr(sparse_conv_calibration, "_CALIBRATION_DIR", str(tmp_path / "sparse"))
        monkeypatch.setattr(fftconv_calibration, "_CALIBRATION_DIR", str(tmp_path / "fftconv"))
        sparse_conv_calibration.save(
            {"conv2d": {"min_positions": 500}},
            extra_metadata={"conv2d": {"size_sweep": {"occupancy": 0.05, "sweep": [1, 2, 3]}}})
        fftconv_calibration.save(
            {"conv3d": {"min_positions": 32768}},
            extra_metadata={"conv3d": {"min_positions_sweep": {"kernel_width": 15}}})

        report = benchmark_report.collect()

        assert report["sparse_conv"]["conv2d"]["size_sweep"] == {"occupancy": 0.05, "sweep": [1, 2, 3]}
        assert report["fftconv"]["conv3d"]["min_positions_sweep"] == {"kernel_width": 15}


class TestSaveRoundTrip:
    def setup_method(self):
        self._orig_report_dir = benchmark_report._REPORT_DIR
        self._orig_sparse_dir = sparse_conv_calibration._CALIBRATION_DIR
        self._orig_fftconv_dir = fftconv_calibration._CALIBRATION_DIR

    def teardown_method(self):
        benchmark_report._REPORT_DIR = self._orig_report_dir
        sparse_conv_calibration._CALIBRATION_DIR = self._orig_sparse_dir
        fftconv_calibration._CALIBRATION_DIR = self._orig_fftconv_dir

    def test_save_writes_readable_json_matching_collect(self, tmp_path, monkeypatch):
        monkeypatch.setattr(benchmark_report, "_REPORT_DIR", str(tmp_path / "report"))
        monkeypatch.setattr(sparse_conv_calibration, "_CALIBRATION_DIR", str(tmp_path / "sparse"))
        monkeypatch.setattr(fftconv_calibration, "_CALIBRATION_DIR", str(tmp_path / "fftconv"))
        fftconv_calibration.save({"conv3d": {"min_positions": 32768}})

        path = benchmark_report.save()

        assert os.path.exists(path)
        with open(path) as f:
            on_disk = json.load(f)
        assert on_disk["fftconv"]["conv3d"] == {"min_positions": 32768}
        assert on_disk == benchmark_report.collect()

    def test_save_path_uses_report_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(benchmark_report, "_REPORT_DIR", str(tmp_path / "report"))
        monkeypatch.setattr(sparse_conv_calibration, "_CALIBRATION_DIR", str(tmp_path / "sparse"))
        monkeypatch.setattr(fftconv_calibration, "_CALIBRATION_DIR", str(tmp_path / "fftconv"))
        assert benchmark_report.save() == benchmark_report.report_path()

    def test_save_overwrites_stale_domain_after_a_calibration_reset(self, tmp_path, monkeypatch):
        """A domain that no longer has an on-disk calibration file (e.g.
        after sparse_conv_calibration.reset()) must disappear from a
        re-saved report, not linger from a previous save()."""
        monkeypatch.setattr(benchmark_report, "_REPORT_DIR", str(tmp_path / "report"))
        monkeypatch.setattr(sparse_conv_calibration, "_CALIBRATION_DIR", str(tmp_path / "sparse"))
        monkeypatch.setattr(fftconv_calibration, "_CALIBRATION_DIR", str(tmp_path / "fftconv"))
        sparse_conv_calibration.save({"conv2d": {"min_positions": 500}})
        benchmark_report.save()

        sparse_conv_calibration.reset()
        path = benchmark_report.save()

        with open(path) as f:
            on_disk = json.load(f)
        assert "sparse_conv" not in on_disk


class TestGpuPcieInfo:
    def test_returns_none_values_on_non_linux(self, monkeypatch):
        monkeypatch.setattr(benchmark_report.sys, "platform", "win32")
        assert benchmark_report.gpu_pcie_info() == {"link_speed": None, "link_width": None}

    def test_returns_none_values_when_no_matching_sysfs_entry(self, monkeypatch):
        monkeypatch.setattr(benchmark_report.sys, "platform", "linux")
        monkeypatch.setattr(benchmark_report.glob, "glob", lambda pattern: [])
        assert benchmark_report.gpu_pcie_info() == {"link_speed": None, "link_width": None}

    def test_reads_speed_and_width_from_sysfs(self, tmp_path, monkeypatch):
        device_dir = tmp_path / "card0" / "device"
        device_dir.mkdir(parents=True)
        (device_dir / "current_link_speed").write_text("16.0 GT/s PCIe\n")
        (device_dir / "current_link_width").write_text("16\n")

        monkeypatch.setattr(benchmark_report.sys, "platform", "linux")
        monkeypatch.setattr(benchmark_report.glob, "glob", lambda pattern: [str(device_dir)])

        assert benchmark_report.gpu_pcie_info() == {
            "link_speed": "16.0 GT/s PCIe", "link_width": "16",
        }

    def test_skips_a_card_dir_missing_one_of_the_two_files(self, tmp_path, monkeypatch):
        incomplete = tmp_path / "card0" / "device"
        incomplete.mkdir(parents=True)
        (incomplete / "current_link_speed").write_text("8.0 GT/s PCIe\n")
        # current_link_width deliberately absent.

        complete = tmp_path / "card1" / "device"
        complete.mkdir(parents=True)
        (complete / "current_link_speed").write_text("16.0 GT/s PCIe\n")
        (complete / "current_link_width").write_text("16\n")

        monkeypatch.setattr(benchmark_report.sys, "platform", "linux")
        monkeypatch.setattr(benchmark_report.glob, "glob",
                             lambda pattern: [str(incomplete), str(complete)])

        assert benchmark_report.gpu_pcie_info() == {
            "link_speed": "16.0 GT/s PCIe", "link_width": "16",
        }


class TestCaptureMiopenLog:
    def test_captures_fd2_writes(self):
        with benchmark_report.CaptureMiopenLog() as cap:
            os.write(2, b"hello from fd 2\n")
        assert cap.text == "hello from fd 2\n"

    def test_empty_when_nothing_written(self):
        with benchmark_report.CaptureMiopenLog() as cap:
            pass
        assert cap.text == ""

    def test_restores_fd2_after_exit(self):
        # A second capture must see only what happens INSIDE its own
        # block -- proof fd 2 was correctly restored after the first one.
        with benchmark_report.CaptureMiopenLog():
            os.write(2, b"first block\n")
        with benchmark_report.CaptureMiopenLog() as cap2:
            os.write(2, b"second block\n")
        assert cap2.text == "second block\n"

    def test_does_not_suppress_exceptions(self):
        with pytest.raises(ValueError):
            with benchmark_report.CaptureMiopenLog():
                raise ValueError("boom")


class TestMiopenLogPersistence:
    def setup_method(self):
        self._orig_report_dir = benchmark_report._REPORT_DIR

    def teardown_method(self):
        benchmark_report._REPORT_DIR = self._orig_report_dir

    def test_save_then_read_back_round_trips(self, tmp_path, monkeypatch):
        monkeypatch.setattr(benchmark_report, "_REPORT_DIR", str(tmp_path))
        benchmark_report.save_miopen_log("fftconv", "some log text\n")
        assert benchmark_report._read_miopen_logs() == {"fftconv": "some log text\n"}

    def test_multiple_domains_coexist(self, tmp_path, monkeypatch):
        monkeypatch.setattr(benchmark_report, "_REPORT_DIR", str(tmp_path))
        benchmark_report.save_miopen_log("sparse_conv", "sparse log\n")
        benchmark_report.save_miopen_log("fftconv", "fftconv log\n")
        assert benchmark_report._read_miopen_logs() == {
            "sparse_conv": "sparse log\n", "fftconv": "fftconv log\n",
        }

    def test_overwrites_only_the_same_domain(self, tmp_path, monkeypatch):
        monkeypatch.setattr(benchmark_report, "_REPORT_DIR", str(tmp_path))
        benchmark_report.save_miopen_log("sparse_conv", "sparse log v1\n")
        benchmark_report.save_miopen_log("fftconv", "fftconv log\n")
        benchmark_report.save_miopen_log("sparse_conv", "sparse log v2\n")
        assert benchmark_report._read_miopen_logs() == {
            "sparse_conv": "sparse log v2\n", "fftconv": "fftconv log\n",
        }

    def test_no_logs_returns_empty_dict(self, tmp_path, monkeypatch):
        monkeypatch.setattr(benchmark_report, "_REPORT_DIR", str(tmp_path))
        assert benchmark_report._read_miopen_logs() == {}

    def test_long_log_is_truncated_inline_but_kept_in_full_on_disk(self, tmp_path, monkeypatch):
        monkeypatch.setattr(benchmark_report, "_REPORT_DIR", str(tmp_path))
        monkeypatch.setattr(benchmark_report, "_MAX_INLINE_LOG_BYTES", 100)
        full_text = "x" * 500
        path = benchmark_report.save_miopen_log("fftconv", full_text)

        logs = benchmark_report._read_miopen_logs()

        assert len(logs["fftconv"]) < 500
        assert "truncated" in logs["fftconv"]
        assert "500" in logs["fftconv"]
        with open(path) as f:
            assert f.read() == full_text  # the file on disk is never truncated

    def test_miopen_log_path_is_keyed_by_device_and_domain(self):
        path = benchmark_report.miopen_log_path("fftconv")
        assert benchmark_report.device_key() in path
        assert "fftconv" in path
        assert path.endswith(".miopen.log")


class TestCollectIncludesMiopenLogs:
    def setup_method(self):
        self._orig_report_dir = benchmark_report._REPORT_DIR
        self._orig_sparse_dir = sparse_conv_calibration._CALIBRATION_DIR
        self._orig_fftconv_dir = fftconv_calibration._CALIBRATION_DIR

    def teardown_method(self):
        benchmark_report._REPORT_DIR = self._orig_report_dir
        sparse_conv_calibration._CALIBRATION_DIR = self._orig_sparse_dir
        fftconv_calibration._CALIBRATION_DIR = self._orig_fftconv_dir

    def test_collect_omits_miopen_logs_key_when_none_exist(self, tmp_path, monkeypatch):
        monkeypatch.setattr(benchmark_report, "_REPORT_DIR", str(tmp_path / "report"))
        monkeypatch.setattr(sparse_conv_calibration, "_CALIBRATION_DIR", str(tmp_path / "sparse"))
        monkeypatch.setattr(fftconv_calibration, "_CALIBRATION_DIR", str(tmp_path / "fftconv"))
        report = benchmark_report.collect()
        assert "miopen_logs" not in report

    def test_collect_includes_miopen_logs_when_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(benchmark_report, "_REPORT_DIR", str(tmp_path / "report"))
        monkeypatch.setattr(sparse_conv_calibration, "_CALIBRATION_DIR", str(tmp_path / "sparse"))
        monkeypatch.setattr(fftconv_calibration, "_CALIBRATION_DIR", str(tmp_path / "fftconv"))
        benchmark_report.save_miopen_log("fftconv", "some real miopen output\n")

        report = benchmark_report.collect()

        assert report["miopen_logs"] == {"fftconv": "some real miopen output\n"}
