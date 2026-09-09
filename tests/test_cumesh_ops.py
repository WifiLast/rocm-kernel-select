"""Tests for amd_tuned_torch.cumesh_ops -- the `cumesh` (third_party/CuMesh)
adapter. No real CuMesh/CUDA/HIP extension anywhere here: CuMesh/cuBVH are
always mocked, same discipline as aiter/TE in
test_amd_tuned_torch_monkeypatch.py.

Run with:

    python -m pytest source/cmp_ext_turing/tests
"""
from __future__ import annotations

from unittest.mock import MagicMock

import amd_tuned_torch.cumesh_ops as cumesh_ops_module


class TestAvailable:
    def test_unavailable_when_package_missing(self, monkeypatch):
        monkeypatch.setattr(cumesh_ops_module, "_CUMESH_AVAILABLE", False)
        assert cumesh_ops_module.available() is False

    def test_available_when_package_present(self, monkeypatch):
        monkeypatch.setattr(cumesh_ops_module, "_CUMESH_AVAILABLE", True)
        assert cumesh_ops_module.available() is True


class TestDropOutWhenUnavailable:
    def test_simplify_mesh_returns_none(self, monkeypatch):
        monkeypatch.setattr(cumesh_ops_module, "available", lambda: False)
        assert cumesh_ops_module.simplify_mesh(None, None, 100) is None

    def test_clean_mesh_returns_none(self, monkeypatch):
        monkeypatch.setattr(cumesh_ops_module, "available", lambda: False)
        assert cumesh_ops_module.clean_mesh(None, None) is None

    def test_uv_unwrap_returns_none(self, monkeypatch):
        monkeypatch.setattr(cumesh_ops_module, "available", lambda: False)
        assert cumesh_ops_module.uv_unwrap(None, None) is None

    def test_signed_distance_returns_none(self, monkeypatch):
        monkeypatch.setattr(cumesh_ops_module, "available", lambda: False)
        assert cumesh_ops_module.signed_distance(None, None, None) is None


class TestSimplifyMesh:
    def test_delegates_to_cumesh_and_reads_result(self, monkeypatch):
        monkeypatch.setattr(cumesh_ops_module, "available", lambda: True)
        fake_mesh = MagicMock()
        fake_mesh.read.return_value = ("verts", "faces")
        monkeypatch.setattr(cumesh_ops_module, "CuMesh", MagicMock(return_value=fake_mesh))

        result = cumesh_ops_module.simplify_mesh("v", "f", 500, verbose=True)

        fake_mesh.init.assert_called_once_with("v", "f")
        fake_mesh.simplify.assert_called_once_with(500, verbose=True, options={})
        assert result == ("verts", "faces")

    def test_runtime_error_returns_none(self, monkeypatch):
        monkeypatch.setattr(cumesh_ops_module, "available", lambda: True)
        fake_mesh = MagicMock()
        fake_mesh.simplify.side_effect = RuntimeError("bad mesh")
        monkeypatch.setattr(cumesh_ops_module, "CuMesh", MagicMock(return_value=fake_mesh))

        assert cumesh_ops_module.simplify_mesh("v", "f", 500) is None


class TestCleanMesh:
    def test_runs_full_cleanup_pipeline_in_order(self, monkeypatch):
        monkeypatch.setattr(cumesh_ops_module, "available", lambda: True)
        fake_mesh = MagicMock()
        fake_mesh.read.return_value = ("verts", "faces")
        monkeypatch.setattr(cumesh_ops_module, "CuMesh", MagicMock(return_value=fake_mesh))

        result = cumesh_ops_module.clean_mesh(
            "v", "f", fill_hole_perimeter=0.1, remove_small_components_min_area=0.01)

        fake_mesh.remove_degenerate_faces.assert_called_once()
        fake_mesh.fill_holes.assert_called_once_with(max_hole_perimeter=0.1)
        fake_mesh.remove_small_connected_components.assert_called_once_with(0.01)
        fake_mesh.unify_face_orientations.assert_called_once()
        assert result == ("verts", "faces")

    def test_skips_optional_steps_when_not_requested(self, monkeypatch):
        monkeypatch.setattr(cumesh_ops_module, "available", lambda: True)
        fake_mesh = MagicMock()
        monkeypatch.setattr(cumesh_ops_module, "CuMesh", MagicMock(return_value=fake_mesh))

        cumesh_ops_module.clean_mesh("v", "f", fill_hole_perimeter=None,
                                      remove_small_components_min_area=None)

        fake_mesh.fill_holes.assert_not_called()
        fake_mesh.remove_small_connected_components.assert_not_called()


class TestSignedDistance:
    def test_builds_bvh_and_queries(self, monkeypatch):
        monkeypatch.setattr(cumesh_ops_module, "available", lambda: True)
        fake_bvh = MagicMock()
        fake_bvh.signed_distance.return_value = "distances"
        monkeypatch.setattr(cumesh_ops_module, "cuBVH", MagicMock(return_value=fake_bvh))

        result = cumesh_ops_module.signed_distance("v", "f", "p", mode="raystab")

        cumesh_ops_module.cuBVH.assert_called_once_with("v", "f")
        fake_bvh.signed_distance.assert_called_once_with("p", return_uvw=False, mode="raystab")
        assert result == "distances"
