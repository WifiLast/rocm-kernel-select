"""Tests for amd_tuned_torch.torchsparse_ops -- the `torchsparse`
(third_party/torchsparse) adapter. No real torchsparse/CUDA/HIP extension
anywhere here: every underlying call is mocked, same discipline as
aiter/TE in test_amd_tuned_torch_monkeypatch.py.

Run with:

    python -m pytest source/cmp_ext_turing/tests
"""
from __future__ import annotations

from unittest.mock import MagicMock

import torch
import torch.nn.functional as F

import amd_tuned_torch.torchsparse_ops as torchsparse_ops_module


class TestAvailable:
    def test_unavailable_when_package_missing(self, monkeypatch):
        monkeypatch.setattr(torchsparse_ops_module, "_TORCHSPARSE_AVAILABLE", False)
        assert torchsparse_ops_module.available() is False

    def test_available_when_package_present(self, monkeypatch):
        monkeypatch.setattr(torchsparse_ops_module, "_TORCHSPARSE_AVAILABLE", True)
        monkeypatch.setattr(torchsparse_ops_module, "force_gather_scatter_dataflow", lambda: None)
        assert torchsparse_ops_module.available() is True


class TestForceGatherScatterDataflow:
    def teardown_method(self):
        torchsparse_ops_module._dataflow_forced = False

    def test_noop_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(torchsparse_ops_module, "_TORCHSPARSE_AVAILABLE", False)
        fake_conv_config = MagicMock()
        monkeypatch.setattr(torchsparse_ops_module, "_conv_config", fake_conv_config)
        torchsparse_ops_module.force_gather_scatter_dataflow()
        fake_conv_config.set_global_conv_config.assert_not_called()

    def test_sets_gather_scatter_dataflow(self, monkeypatch):
        monkeypatch.setattr(torchsparse_ops_module, "_TORCHSPARSE_AVAILABLE", True)
        torchsparse_ops_module._dataflow_forced = False
        fake_conv_config = MagicMock()
        fake_conv_config._default_conv_config = {
            "dataflow": "ImplicitGEMM", "ifsort": False, "kmap_mode": "hashmap_on_the_fly",
        }
        monkeypatch.setattr(torchsparse_ops_module, "_conv_config", fake_conv_config)
        fake_dataflow = MagicMock(GatherScatter="GATHER_SCATTER_SENTINEL")
        monkeypatch.setattr(torchsparse_ops_module, "_Dataflow", fake_dataflow)

        torchsparse_ops_module.force_gather_scatter_dataflow()

        fake_conv_config.set_global_conv_config.assert_called_once()
        called_config = fake_conv_config.set_global_conv_config.call_args[0][0]
        assert called_config["dataflow"] == "GATHER_SCATTER_SENTINEL"
        assert called_config["ifsort"] is False
        assert called_config["kmap_mode"] == "hashmap_on_the_fly"  # preserved from defaults

    def test_idempotent_only_sets_once(self, monkeypatch):
        monkeypatch.setattr(torchsparse_ops_module, "_TORCHSPARSE_AVAILABLE", True)
        torchsparse_ops_module._dataflow_forced = False
        fake_conv_config = MagicMock()
        fake_conv_config._default_conv_config = {"dataflow": None, "ifsort": False}
        monkeypatch.setattr(torchsparse_ops_module, "_conv_config", fake_conv_config)
        monkeypatch.setattr(torchsparse_ops_module, "_Dataflow", MagicMock(GatherScatter="X"))

        torchsparse_ops_module.force_gather_scatter_dataflow()
        torchsparse_ops_module.force_gather_scatter_dataflow()

        fake_conv_config.set_global_conv_config.assert_called_once()

    def test_available_triggers_force(self, monkeypatch):
        monkeypatch.setattr(torchsparse_ops_module, "_TORCHSPARSE_AVAILABLE", True)
        torchsparse_ops_module._dataflow_forced = False
        spy = MagicMock()
        monkeypatch.setattr(torchsparse_ops_module, "force_gather_scatter_dataflow", spy)

        assert torchsparse_ops_module.available() is True
        spy.assert_called_once()


class TestDropOutWhenUnavailable:
    def test_make_sparse_tensor_returns_none(self, monkeypatch):
        monkeypatch.setattr(torchsparse_ops_module, "available", lambda: False)
        assert torchsparse_ops_module.make_sparse_tensor(None, None) is None

    def test_sparse_conv3d_returns_none(self, monkeypatch):
        monkeypatch.setattr(torchsparse_ops_module, "available", lambda: False)
        assert torchsparse_ops_module.sparse_conv3d(None, None, 3) is None

    def test_sparse_quantize_returns_none(self, monkeypatch):
        monkeypatch.setattr(torchsparse_ops_module, "available", lambda: False)
        assert torchsparse_ops_module.sparse_quantize(None) is None

    def test_voxelize_returns_none(self, monkeypatch):
        monkeypatch.setattr(torchsparse_ops_module, "available", lambda: False)
        assert torchsparse_ops_module.voxelize(None, None, None) is None

    def test_devoxelize_returns_none(self, monkeypatch):
        monkeypatch.setattr(torchsparse_ops_module, "available", lambda: False)
        assert torchsparse_ops_module.devoxelize(None, None, None) is None


class TestMakeSparseTensor:
    def test_delegates_to_sparse_tensor_class(self, monkeypatch):
        monkeypatch.setattr(torchsparse_ops_module, "available", lambda: True)
        fake_tensor = MagicMock()
        fake_cls = MagicMock(return_value=fake_tensor)
        monkeypatch.setattr(torchsparse_ops_module, "SparseTensor", fake_cls)

        result = torchsparse_ops_module.make_sparse_tensor("feats", "coords", stride=2)

        assert result is fake_tensor
        fake_cls.assert_called_once_with(feats="feats", coords="coords", stride=2)


class TestSparseConv3d:
    def test_delegates_to_functional_conv3d(self, monkeypatch):
        monkeypatch.setattr(torchsparse_ops_module, "available", lambda: True)
        fake_spf = MagicMock()
        fake_spf.conv3d.return_value = "conv_result"
        monkeypatch.setattr(torchsparse_ops_module, "_spf", fake_spf)

        result = torchsparse_ops_module.sparse_conv3d(
            "input_tensor", "weight", 3, bias="bias", stride=2, padding=1, dilation=1)

        assert result == "conv_result"
        fake_spf.conv3d.assert_called_once_with(
            "input_tensor", "weight", 3, bias="bias", stride=2, padding=1, dilation=1,
            transposed=False)

    def test_runtime_error_returns_none(self, monkeypatch):
        monkeypatch.setattr(torchsparse_ops_module, "available", lambda: True)
        fake_spf = MagicMock()
        fake_spf.conv3d.side_effect = RuntimeError("bad shape")
        monkeypatch.setattr(torchsparse_ops_module, "_spf", fake_spf)

        assert torchsparse_ops_module.sparse_conv3d("input_tensor", "weight", 3) is None


class TestSparseQuantize:
    def test_delegates(self, monkeypatch):
        monkeypatch.setattr(torchsparse_ops_module, "available", lambda: True)
        fake_quantize = MagicMock(return_value="quantized")
        monkeypatch.setattr(torchsparse_ops_module, "_sparse_quantize", fake_quantize)

        result = torchsparse_ops_module.sparse_quantize("coords", voxel_size=0.05)

        assert result == "quantized"
        fake_quantize.assert_called_once_with("coords", 0.05, return_index=False, return_inverse=False)


class TestVoxelizeDevoxelize:
    def test_voxelize_delegates(self, monkeypatch):
        monkeypatch.setattr(torchsparse_ops_module, "available", lambda: True)
        fake_spf = MagicMock()
        fake_spf.spvoxelize.return_value = "voxelized"
        monkeypatch.setattr(torchsparse_ops_module, "_spf", fake_spf)

        result = torchsparse_ops_module.voxelize("feats", "coords", "counts")

        assert result == "voxelized"
        fake_spf.spvoxelize.assert_called_once_with("feats", "coords", "counts")

    def test_voxelize_runtime_error_returns_none(self, monkeypatch):
        monkeypatch.setattr(torchsparse_ops_module, "available", lambda: True)
        fake_spf = MagicMock()
        fake_spf.spvoxelize.side_effect = RuntimeError("bad shape")
        monkeypatch.setattr(torchsparse_ops_module, "_spf", fake_spf)

        assert torchsparse_ops_module.voxelize("feats", "coords", "counts") is None

    def test_devoxelize_delegates(self, monkeypatch):
        monkeypatch.setattr(torchsparse_ops_module, "available", lambda: True)
        fake_spf = MagicMock()
        fake_spf.spdevoxelize.return_value = "devoxelized"
        monkeypatch.setattr(torchsparse_ops_module, "_spf", fake_spf)

        result = torchsparse_ops_module.devoxelize("feats", "idx_query", "weights")

        assert result == "devoxelized"
        fake_spf.spdevoxelize.assert_called_once_with("feats", "idx_query", "weights")

    def test_devoxelize_runtime_error_returns_none(self, monkeypatch):
        monkeypatch.setattr(torchsparse_ops_module, "available", lambda: True)
        fake_spf = MagicMock()
        fake_spf.spdevoxelize.side_effect = RuntimeError("bad shape")
        monkeypatch.setattr(torchsparse_ops_module, "_spf", fake_spf)

        assert torchsparse_ops_module.devoxelize("feats", "idx_query", "weights") is None


class _FakeSparseTensor:
    """Minimal stand-in for torchsparse.SparseTensor -- only .feats/.coords
    are read by sparse_conv3d_from_dense."""

    def __init__(self, feats, coords):
        self.feats = feats
        self.coords = coords


def _touched_output_mask3d(occ_mask: torch.Tensor, kernel_size, stride, padding, dilation) -> torch.Tensor:
    kd, kh, kw = kernel_size
    ones_kernel = torch.ones(1, 1, kd, kh, kw)
    overlap = F.conv3d(occ_mask.float().unsqueeze(1), ones_kernel,
                        stride=stride, padding=padding, dilation=dilation)
    return overlap.squeeze(1) > 0


class TestSparseConv3dFromDense:
    def test_unavailable_returns_none(self, monkeypatch):
        monkeypatch.setattr(torchsparse_ops_module, "available", lambda: False)
        x = torch.zeros(1, 2, 4, 4, 4)
        weight = torch.randn(3, 2, 3, 3, 3)
        assert torchsparse_ops_module.sparse_conv3d_from_dense(x, weight) is None

    def test_wrong_dims_returns_none(self, monkeypatch):
        monkeypatch.setattr(torchsparse_ops_module, "available", lambda: True)
        assert torchsparse_ops_module.sparse_conv3d_from_dense(torch.zeros(2, 2), torch.zeros(3, 2)) is None

    def test_fully_empty_input_returns_pure_bias_without_calling_sparse_conv3d(self, monkeypatch):
        monkeypatch.setattr(torchsparse_ops_module, "available", lambda: True)
        fake_conv = MagicMock()
        monkeypatch.setattr(torchsparse_ops_module, "sparse_conv3d", fake_conv)
        c_in, c_out = 2, 3
        weight = torch.randn(c_out, c_in, 3, 3, 3)
        bias = torch.randn(c_out)
        x = torch.zeros(1, c_in, 5, 5, 5)

        reference = F.conv3d(x, weight, bias, stride=1, padding=1)
        result = torchsparse_ops_module.sparse_conv3d_from_dense(
            x, weight, bias, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1))

        torch.testing.assert_close(result, reference, atol=1e-6, rtol=1e-6)
        fake_conv.assert_not_called()

    def test_matches_dense_conv3d_via_weight_layout_conversion(self, monkeypatch):
        """Mocks sparse_conv3d with a fake that: (1) checks the weight it
        receives is actually in torchsparse's [kernel_volume,Ci,Co] layout
        (not PyTorch's dense [Co,Ci,Kd,Kh,Kw]), and (2) returns feats/coords
        computed from a REAL F.conv3d reference at exactly the positions a
        real sparse conv would touch -- so this validates
        sparse_conv3d_from_dense's own weight-layout conversion and
        scatter-back arithmetic against ground truth, without needing the
        actual torchsparse kernel."""
        monkeypatch.setattr(torchsparse_ops_module, "available", lambda: True)
        monkeypatch.setattr(torchsparse_ops_module, "make_sparse_tensor",
                             lambda feats, coords, **kw: _FakeSparseTensor(feats, coords))

        torch.manual_seed(0)
        b, c_in, c_out = 1, 2, 3
        d = h = w = 6
        kd = kh = kw = 3
        stride, padding, dilation = (1, 1, 1), (1, 1, 1), (1, 1, 1)

        weight = torch.randn(c_out, c_in, kd, kh, kw)
        bias = torch.randn(c_out)
        x = torch.zeros(b, c_in, d, h, w)
        for (dd, hh, ww) in [(1, 1, 1), (4, 2, 3), (0, 5, 5)]:
            x[0, :, dd, hh, ww] = torch.randn(c_in)
        occ_mask = x.abs().amax(dim=1) > 1e-12

        reference = F.conv3d(x, weight, bias, stride=stride, padding=padding, dilation=dilation)
        touched = _touched_output_mask3d(occ_mask, (kd, kh, kw), stride, padding, dilation)
        ref_bdhwc = reference.permute(0, 2, 3, 4, 1)

        def _fake_sparse_conv3d(input_tensor, weight_flat, kernel_size, bias, stride, padding, dilation):
            assert tuple(weight_flat.shape) == (kd * kh * kw, c_in, c_out)
            assert kernel_size == (kd, kh, kw)
            out_coords = touched.nonzero(as_tuple=False).to(torch.int32)
            out_feats = ref_bdhwc[touched]
            return _FakeSparseTensor(out_feats, out_coords)

        monkeypatch.setattr(torchsparse_ops_module, "sparse_conv3d", _fake_sparse_conv3d)

        result = torchsparse_ops_module.sparse_conv3d_from_dense(
            x, weight, bias, stride=stride, padding=padding, dilation=dilation)

        assert result is not None
        torch.testing.assert_close(result, reference, atol=1e-5, rtol=1e-5)

    def test_sparse_conv3d_declining_returns_none(self, monkeypatch):
        monkeypatch.setattr(torchsparse_ops_module, "available", lambda: True)
        monkeypatch.setattr(torchsparse_ops_module, "make_sparse_tensor",
                             lambda feats, coords, **kw: _FakeSparseTensor(feats, coords))
        monkeypatch.setattr(torchsparse_ops_module, "sparse_conv3d", lambda *a, **k: None)

        x = torch.zeros(1, 2, 4, 4, 4)
        x[0, :, 0, 0, 0] = 1.0
        weight = torch.randn(3, 2, 3, 3, 3)

        assert torchsparse_ops_module.sparse_conv3d_from_dense(x, weight) is None
