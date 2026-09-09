"""Tests for amd_tuned_torch.nvdiffrast_ops -- the `nvdiffrast`
(third_party/nvdiffrast) adapter. No real nvdiffrast/CUDA/HIP extension or GPU
context anywhere here: every underlying call is mocked, same discipline as
aiter/TE in test_amd_tuned_torch_monkeypatch.py.

Run with:

    python -m pytest source/cmp_ext_turing/tests
"""
from __future__ import annotations

from unittest.mock import MagicMock

import amd_tuned_torch.nvdiffrast_ops as nvdiffrast_ops_module


class TestAvailable:
    def test_unavailable_when_package_missing(self, monkeypatch):
        monkeypatch.setattr(nvdiffrast_ops_module, "_NVDIFFRAST_AVAILABLE", False)
        assert nvdiffrast_ops_module.available() is False

    def test_available_when_package_present(self, monkeypatch):
        monkeypatch.setattr(nvdiffrast_ops_module, "_NVDIFFRAST_AVAILABLE", True)
        assert nvdiffrast_ops_module.available() is True


class TestGetContext:
    def teardown_method(self):
        nvdiffrast_ops_module._contexts.clear()

    def test_returns_none_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(nvdiffrast_ops_module, "available", lambda: False)
        assert nvdiffrast_ops_module.get_context() is None

    def test_creates_and_caches_per_device(self, monkeypatch):
        monkeypatch.setattr(nvdiffrast_ops_module, "available", lambda: True)
        fake_ctx = MagicMock()
        fake_dr = MagicMock(RasterizeCudaContext=MagicMock(return_value=fake_ctx))
        monkeypatch.setattr(nvdiffrast_ops_module, "_dr", fake_dr)

        ctx1 = nvdiffrast_ops_module.get_context(device="cuda:0")
        ctx2 = nvdiffrast_ops_module.get_context(device="cuda:0")

        assert ctx1 is fake_ctx
        assert ctx2 is fake_ctx
        fake_dr.RasterizeCudaContext.assert_called_once_with(device="cuda:0")

    def test_context_creation_failure_returns_none(self, monkeypatch):
        monkeypatch.setattr(nvdiffrast_ops_module, "available", lambda: True)
        fake_dr = MagicMock(RasterizeCudaContext=MagicMock(side_effect=RuntimeError("no HIP context")))
        monkeypatch.setattr(nvdiffrast_ops_module, "_dr", fake_dr)

        assert nvdiffrast_ops_module.get_context(device="cuda:0") is None


class TestDropOutWhenUnavailable:
    def test_interpolate_returns_none(self, monkeypatch):
        monkeypatch.setattr(nvdiffrast_ops_module, "available", lambda: False)
        assert nvdiffrast_ops_module.interpolate(None, None, None) is None

    def test_texture_returns_none(self, monkeypatch):
        monkeypatch.setattr(nvdiffrast_ops_module, "available", lambda: False)
        assert nvdiffrast_ops_module.texture(None, None) is None

    def test_antialias_returns_none(self, monkeypatch):
        monkeypatch.setattr(nvdiffrast_ops_module, "available", lambda: False)
        assert nvdiffrast_ops_module.antialias(None, None, None, None) is None

    def test_rasterize_returns_none_without_context(self, monkeypatch):
        monkeypatch.setattr(nvdiffrast_ops_module, "get_context", lambda device=None: None)
        assert nvdiffrast_ops_module.rasterize("pos", "tri", (64, 64)) is None


class TestRasterize:
    def test_delegates_with_explicit_context(self, monkeypatch):
        fake_dr = MagicMock(rasterize=MagicMock(return_value=("rast", "rast_db")))
        monkeypatch.setattr(nvdiffrast_ops_module, "_dr", fake_dr)
        fake_ctx = MagicMock()

        result = nvdiffrast_ops_module.rasterize("pos", "tri", (64, 64), glctx=fake_ctx)

        assert result == ("rast", "rast_db")
        fake_dr.rasterize.assert_called_once_with(fake_ctx, "pos", "tri", (64, 64),
                                                    ranges=None, grad_db=True)

    def test_runtime_error_returns_none(self, monkeypatch):
        fake_dr = MagicMock(rasterize=MagicMock(side_effect=RuntimeError("bad shape")))
        monkeypatch.setattr(nvdiffrast_ops_module, "_dr", fake_dr)
        fake_ctx = MagicMock()

        assert nvdiffrast_ops_module.rasterize("pos", "tri", (64, 64), glctx=fake_ctx) is None


class TestInterpolateTextureAntialias:
    def test_interpolate_delegates(self, monkeypatch):
        monkeypatch.setattr(nvdiffrast_ops_module, "available", lambda: True)
        fake_dr = MagicMock(interpolate=MagicMock(return_value=("out", "out_da")))
        monkeypatch.setattr(nvdiffrast_ops_module, "_dr", fake_dr)

        result = nvdiffrast_ops_module.interpolate("attr", "rast", "tri")

        assert result == ("out", "out_da")

    def test_texture_delegates(self, monkeypatch):
        monkeypatch.setattr(nvdiffrast_ops_module, "available", lambda: True)
        fake_dr = MagicMock(texture=MagicMock(return_value="sampled"))
        monkeypatch.setattr(nvdiffrast_ops_module, "_dr", fake_dr)

        result = nvdiffrast_ops_module.texture("tex", "uv")

        assert result == "sampled"

    def test_antialias_delegates(self, monkeypatch):
        monkeypatch.setattr(nvdiffrast_ops_module, "available", lambda: True)
        fake_dr = MagicMock(antialias=MagicMock(return_value="aa_color"))
        monkeypatch.setattr(nvdiffrast_ops_module, "_dr", fake_dr)

        result = nvdiffrast_ops_module.antialias("color", "rast", "pos", "tri")

        assert result == "aa_color"
