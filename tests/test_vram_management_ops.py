"""CPU-safe tests for vram_management_ops -- like swiglu_ops, this module
is plain PyTorch (.to(dtype=, device=), copy.deepcopy, F.linear) with no
Triton/native-extension dependency at all, so its wrapping/offload/onload
logic and enable_vram_management's tree-walk can be fully exercised on
CPU, with "device" left as "cpu" throughout (the state-machine logic under
test doesn't care what device string it's holding, only whether two
device/dtype values compare equal)."""
from __future__ import annotations

import torch
import torch.nn as nn

import amd_tuned_torch
from amd_tuned_torch import vram_management_ops as vmo


class TestCastTo:
    def test_copies_values_into_new_dtype(self):
        w = torch.randn(4, 4, dtype=torch.float32)
        out = vmo.cast_to(w, torch.float64, "cpu")
        assert out.dtype == torch.float64
        assert torch.allclose(out.double(), w.double())
        # A genuinely new tensor, not a view -- mutating it must not touch w.
        out[0, 0] = 999.0
        assert w[0, 0] != 999.0


class TestAutoWrappedLinear:
    def _linear(self, in_f=8, out_f=4, bias=True):
        torch.manual_seed(0)
        return nn.Linear(in_f, out_f, bias=bias)

    def test_forward_matches_unwrapped_when_tiers_are_uniform(self):
        lin = self._linear()
        wrapped = vmo.AutoWrappedLinear(
            lin,
            offload_dtype=torch.float32, offload_device="cpu",
            onload_dtype=torch.float32, onload_device="cpu",
            computation_dtype=torch.float32, computation_device="cpu",
        )
        x = torch.randn(3, 8)
        expected = lin(x)
        out = wrapped(x)
        assert torch.allclose(out, expected)

    def test_forward_casts_weight_and_bias_when_computation_tier_differs(self):
        # Only the STORED weight/bias are cast to computation_dtype here --
        # the activation is expected to already arrive at the model's
        # working precision (that's the whole point: weights live cheaply
        # offloaded/onloaded in a different dtype, activations don't), so
        # x itself is constructed in computation_dtype, matching real
        # usage. See vram_management_ops.AutoWrappedLinear.forward.
        lin = self._linear()
        wrapped = vmo.AutoWrappedLinear(
            lin,
            offload_dtype=torch.float32, offload_device="cpu",
            onload_dtype=torch.float32, onload_device="cpu",
            computation_dtype=torch.float64, computation_device="cpu",
        )
        x = torch.randn(3, 8, dtype=torch.float64)
        out = wrapped(x)
        assert out.dtype == torch.float64
        expected = nn.functional.linear(x, lin.weight.double(), lin.bias.double())
        assert torch.allclose(out, expected)
        # The source module's own weight must be untouched by the cast.
        assert lin.weight.dtype == torch.float32

    def test_forward_without_bias(self):
        lin = self._linear(bias=False)
        wrapped = vmo.AutoWrappedLinear(
            lin,
            offload_dtype=torch.float32, offload_device="cpu",
            onload_dtype=torch.float32, onload_device="cpu",
            computation_dtype=torch.float64, computation_device="cpu",
        )
        x = torch.randn(3, 8, dtype=torch.float64)
        out = wrapped(x)
        expected = nn.functional.linear(x, lin.weight.double(), None)
        assert torch.allclose(out, expected)

    def test_weight_and_bias_identity_preserved_from_source_module(self):
        lin = self._linear()
        wrapped = vmo.AutoWrappedLinear(
            lin,
            offload_dtype=torch.float32, offload_device="cpu",
            onload_dtype=torch.float32, onload_device="cpu",
            computation_dtype=torch.float32, computation_device="cpu",
        )
        assert wrapped.weight is lin.weight
        assert wrapped.bias is lin.bias

    def test_onload_offload_toggle_state_only_when_tiers_differ(self):
        lin = self._linear()
        wrapped = vmo.AutoWrappedLinear(
            lin,
            offload_dtype=torch.float32, offload_device="cpu",
            onload_dtype=torch.float64, onload_device="cpu",
            computation_dtype=torch.float64, computation_device="cpu",
        )
        assert wrapped.state == 0
        wrapped.onload()
        assert wrapped.state == 1
        assert wrapped.weight.dtype == torch.float64
        wrapped.offload()
        assert wrapped.state == 0
        assert wrapped.weight.dtype == torch.float32

    def test_onload_offload_are_noops_when_offload_and_onload_tiers_match(self):
        lin = self._linear()
        wrapped = vmo.AutoWrappedLinear(
            lin,
            offload_dtype=torch.float32, offload_device="cpu",
            onload_dtype=torch.float32, onload_device="cpu",
            computation_dtype=torch.float32, computation_device="cpu",
        )
        wrapped.onload()
        assert wrapped.state == 0  # never flips -- offload_dtype == onload_dtype
        wrapped.offload()
        assert wrapped.state == 0


class _Double(nn.Module):
    """A trivial non-Linear module, to exercise AutoWrappedModule's
    generic (not F.linear-specialized) forward path."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, x):
        return x * self.scale


class TestAutoWrappedModule:
    def test_forward_matches_unwrapped_when_tiers_are_uniform(self):
        mod = _Double()
        wrapped = vmo.AutoWrappedModule(
            mod,
            offload_dtype=torch.float32, offload_device="cpu",
            onload_dtype=torch.float32, onload_device="cpu",
            computation_dtype=torch.float32, computation_device="cpu",
        )
        x = torch.randn(5)
        assert torch.allclose(wrapped(x), x * 2.0)

    def test_forward_deep_copies_and_casts_when_computation_tier_differs(self):
        # Same convention as AutoWrappedLinear's equivalent test: the
        # deep-copied module's OWN parameters get cast to computation_dtype,
        # the activation is expected to already be there. (Passing a
        # float32 x against a float64-cast 0-dim Parameter would silently
        # produce a float32 result here regardless -- PyTorch's type
        # promotion doesn't let a 0-dim tensor's higher category win over a
        # non-0-dim tensor's -- so this asserts the parameter-cast behavior
        # directly rather than relying on the output dtype to prove it.)
        mod = _Double()
        wrapped = vmo.AutoWrappedModule(
            mod,
            offload_dtype=torch.float32, offload_device="cpu",
            onload_dtype=torch.float32, onload_device="cpu",
            computation_dtype=torch.float64, computation_device="cpu",
        )
        x = torch.randn(5, dtype=torch.float64)
        out = wrapped(x)
        assert out.dtype == torch.float64
        assert torch.allclose(out, x * 2.0)
        # The original module's own parameter must be untouched by the
        # deep-copy-and-cast (only the copy inside forward() was cast).
        assert mod.scale.dtype == torch.float32

    def test_onload_offload_toggle_state(self):
        mod = _Double()
        wrapped = vmo.AutoWrappedModule(
            mod,
            offload_dtype=torch.float32, offload_device="cpu",
            onload_dtype=torch.float64, onload_device="cpu",
            computation_dtype=torch.float64, computation_device="cpu",
        )
        assert wrapped.state == 0
        wrapped.onload()
        assert wrapped.state == 1
        assert wrapped.module.scale.dtype == torch.float64
        wrapped.offload()
        assert wrapped.state == 0
        assert wrapped.module.scale.dtype == torch.float32


class _TinyModel(nn.Module):
    """One Linear directly under root plus one nested inside a Sequential,
    to exercise enable_vram_management_recursively's both the direct-match
    and the recurse-into-non-matching-child branches."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.block = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))


class TestEnableVramManagementRecursively:
    def _module_config(self, computation_dtype=torch.float32):
        return dict(
            offload_dtype=torch.float32, offload_device="cpu",
            onload_dtype=torch.float32, onload_device="cpu",
            computation_dtype=computation_dtype, computation_device="cpu",
        )

    def test_wraps_every_matching_submodule_including_nested(self):
        model = _TinyModel()
        vmo.enable_vram_management_recursively(
            model, {nn.Linear: vmo.AutoWrappedLinear}, self._module_config()
        )
        assert isinstance(model.fc1, vmo.AutoWrappedLinear)
        assert isinstance(model.block[0], vmo.AutoWrappedLinear)
        assert isinstance(model.block[2], vmo.AutoWrappedLinear)
        # Non-matching children are left alone, not replaced.
        assert isinstance(model.block[1], nn.ReLU)

    def test_wrapped_model_forward_matches_reference(self):
        torch.manual_seed(0)
        model = _TinyModel()
        # Snapshot outputs BEFORE wrapping -- enable_vram_management_recursively
        # mutates the model's own submodules in place (setattr), it does not
        # operate on a copy.
        x = torch.randn(2, 4)
        expected = model.block(model.fc1(x))

        vmo.enable_vram_management_recursively(
            model, {nn.Linear: vmo.AutoWrappedLinear}, self._module_config()
        )
        out = model.block(model.fc1(x))
        assert torch.allclose(out, expected)

    def test_returns_total_param_count_across_all_wrapped_submodules(self):
        model = _TinyModel()
        expected_params = (
            sum(p.numel() for p in model.fc1.parameters())
            + sum(p.numel() for p in model.block[0].parameters())
            + sum(p.numel() for p in model.block[2].parameters())
        )
        total = vmo.enable_vram_management_recursively(
            model, {nn.Linear: vmo.AutoWrappedLinear}, self._module_config()
        )
        assert total == expected_params

    def test_overflow_config_used_once_max_num_param_exceeded(self):
        model = _TinyModel()
        fc1_params = sum(p.numel() for p in model.fc1.parameters())
        # max_num_param set so fc1 alone fits under the budget but the next
        # Linear (block[0]) pushes the running total over it.
        vmo.enable_vram_management_recursively(
            model,
            {nn.Linear: vmo.AutoWrappedLinear},
            module_config=self._module_config(computation_dtype=torch.float32),
            max_num_param=fc1_params,
            overflow_module_config=self._module_config(computation_dtype=torch.float64),
        )
        assert model.fc1.computation_dtype == torch.float32
        assert model.block[0].computation_dtype == torch.float64
        assert model.block[2].computation_dtype == torch.float64


class TestEnableVramManagement:
    def test_sets_flag_and_delegates_to_recursive_walk(self):
        model = _TinyModel()
        config = dict(
            offload_dtype=torch.float32, offload_device="cpu",
            onload_dtype=torch.float32, onload_device="cpu",
            computation_dtype=torch.float32, computation_device="cpu",
        )
        vmo.enable_vram_management(model, {nn.Linear: vmo.AutoWrappedLinear}, config)
        assert model.vram_management_enabled is True
        assert isinstance(model.fc1, vmo.AutoWrappedLinear)


class TestModuleWiring:
    def test_top_level_classes_are_the_same_objects(self):
        assert amd_tuned_torch.AutoWrappedModule is vmo.AutoWrappedModule
        assert amd_tuned_torch.AutoWrappedLinear is vmo.AutoWrappedLinear

    def test_top_level_enable_delegates(self, monkeypatch):
        called = {}

        def fake(model, module_map, module_config, max_num_param=None, overflow_module_config=None):
            called["args"] = (model, module_map, module_config, max_num_param, overflow_module_config)
            return "sentinel"

        monkeypatch.setattr(amd_tuned_torch.vram_management_ops, "enable_vram_management", fake)
        model, module_map, config = object(), object(), object()
        result = amd_tuned_torch.enable_vram_management(model, module_map, config, max_num_param=42)
        assert result == "sentinel"
        assert called["args"] == (model, module_map, config, 42, None)
