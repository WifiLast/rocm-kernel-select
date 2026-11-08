"""Per-module VRAM offload/onload management -- ported near-verbatim from
NVIDIA Cosmos-Predict2's cosmos_predict2/vram_management/layers.py. See
amd_tuned_torch/_vendor/cosmos_vram_management/NOTICE.md for exactly what
was changed from upstream and why (short version: nothing but a private
inline copy of one small helper). Unlike every other _ops module here,
this is not a ROCm kernel port: the original is plain PyTorch (.to(dtype=
..., device=...), copy.deepcopy, torch.nn.functional.linear) with zero
CUDA-specific calls, so nothing needed adapting for RDNA3/ROCm -- it is
reused as-is.

WHAT THIS IS FOR. A model built from ordinary nn.Module/nn.Linear layers can
be walked and selectively wrapped so that most of its weights live on
offload_device (typically "cpu") in offload_dtype, and only get moved to
onload_device/onload_dtype -- and optionally further cast to
computation_dtype/computation_device for the actual matmul -- around each
forward call. This is the same "layer swapping" trick diffusers/ComfyUI-style
low-VRAM inference uses, applied per-nn.Module instead of per-whole-model, so
a big DiT (e.g. cosmos-predict2's video2world model) can run on a card too
small to hold every weight at once.

Nothing here is auto-patched or auto-enabled -- it is a model-authoring-time
opt-in a caller applies explicitly to their own model, not a torch.nn.
functional interception, so there is no monkeypatch/eligibility-gate
machinery to speak of (unlike flash_attn_rocwmma_ops or te_ops). See
amd_tuned_torch.enable_vram_management's docstring in _opt_in_tiers.py for
the calling convention.

UNVALIDATED on this project's hardware (RX 7900 XTX / gfx1100) in the sense
that no benchmark here measures the offload/onload transfer cost against a
PCIe link -- that number is platform-specific (PCIe generation/lanes) rather
than something a kernel choice affects, so it is out of scope for this
package's usual "measured against stock" bar. The wrapped forward path
itself (an ordinary nn.Linear/F.linear call once weights are on
computation_device) already goes through amd_tuned_torch's own F.linear
patch like any other call, so no separate correctness work was needed there
either.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager

import torch

__all__ = [
    "cast_to",
    "AutoWrappedModule",
    "AutoWrappedLinear",
    "enable_vram_management_recursively",
    "enable_vram_management",
]


@contextmanager
def _init_weights_on_device(device=torch.device("meta"), include_buffers: bool = False):
    """Private copy of cosmos_predict2.models.utils.init_weights_on_device --
    small enough (and self-contained, no other cosmos_predict2 dependency)
    that vendoring it here avoids making amd_tuned_torch depend on the
    third_party/cosmos-predict2 checkout at import time. Makes
    torch.nn.Module.register_parameter/register_buffer and (optionally)
    torch.empty/zeros/ones/full allocate on `device` instead of wherever
    they'd normally land, for the duration of the context -- used below so
    AutoWrappedLinear's meta-device nn.Linear.__init__ never actually
    allocates real storage for weights it is about to discard in favor of
    the wrapped module's own."""
    old_register_parameter = torch.nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = torch.nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}

    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


def cast_to(weight, dtype, device):
    r = torch.empty_like(weight, dtype=dtype, device=device)
    r.copy_(weight)
    return r


class AutoWrappedModule(torch.nn.Module):
    """Wraps an arbitrary nn.Module so it lives on offload_device/
    offload_dtype at rest and is moved to onload_device/onload_dtype (and,
    if that differs again, further cast to computation_dtype/
    computation_device via a deep copy) only around forward()."""

    def __init__(
        self,
        module: torch.nn.Module,
        offload_dtype,
        offload_device,
        onload_dtype,
        onload_device,
        computation_dtype,
        computation_device,
    ):
        super().__init__()
        self.module = module.to(dtype=offload_dtype, device=offload_device)
        self.offload_dtype = offload_dtype
        self.offload_device = offload_device
        self.onload_dtype = onload_dtype
        self.onload_device = onload_device
        self.computation_dtype = computation_dtype
        self.computation_device = computation_device
        self.state = 0

    def offload(self):
        if self.state == 1 and (self.offload_dtype != self.onload_dtype or self.offload_device != self.onload_device):
            self.module.to(dtype=self.offload_dtype, device=self.offload_device)
            self.state = 0

    def onload(self):
        if self.state == 0 and (self.offload_dtype != self.onload_dtype or self.offload_device != self.onload_device):
            self.module.to(dtype=self.onload_dtype, device=self.onload_device)
            self.state = 1

    def forward(self, *args, **kwargs):
        if self.onload_dtype == self.computation_dtype and self.onload_device == self.computation_device:
            module = self.module
        else:
            module = copy.deepcopy(self.module).to(dtype=self.computation_dtype, device=self.computation_device)
        return module(*args, **kwargs)


class AutoWrappedLinear(torch.nn.Linear):
    """Same offload/onload/computation-tier scheme as AutoWrappedModule,
    specialized for nn.Linear: instead of deep-copying the whole submodule
    for the computation tier, it casts just the weight/bias tensors
    (cast_to) and calls F.linear directly -- cheaper than
    AutoWrappedModule's copy.deepcopy for the common case of a plain
    Linear, and the one path that benefits from amd_tuned_torch's own
    F.linear patch (aiter_ops.linear_fp16) once weight/bias land on
    computation_device in fp16/bf16."""

    def __init__(
        self,
        module: torch.nn.Linear,
        offload_dtype,
        offload_device,
        onload_dtype,
        onload_device,
        computation_dtype,
        computation_device,
    ):
        with _init_weights_on_device(device=torch.device("meta")):
            super().__init__(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                dtype=offload_dtype,
                device=offload_device,
            )
        self.weight = module.weight
        self.bias = module.bias
        self.offload_dtype = offload_dtype
        self.offload_device = offload_device
        self.onload_dtype = onload_dtype
        self.onload_device = onload_device
        self.computation_dtype = computation_dtype
        self.computation_device = computation_device
        self.state = 0

    def offload(self):
        if self.state == 1 and (self.offload_dtype != self.onload_dtype or self.offload_device != self.onload_device):
            self.to(dtype=self.offload_dtype, device=self.offload_device)
            self.state = 0

    def onload(self):
        if self.state == 0 and (self.offload_dtype != self.onload_dtype or self.offload_device != self.onload_device):
            self.to(dtype=self.onload_dtype, device=self.onload_device)
            self.state = 1

    def forward(self, x, *args, **kwargs):
        if self.onload_dtype == self.computation_dtype and self.onload_device == self.computation_device:
            weight, bias = self.weight, self.bias
        else:
            weight = cast_to(self.weight, self.computation_dtype, self.computation_device)
            bias = None if self.bias is None else cast_to(self.bias, self.computation_dtype, self.computation_device)
        return torch.nn.functional.linear(x, weight, bias)


def enable_vram_management_recursively(
    model: torch.nn.Module,
    module_map: dict,
    module_config: dict,
    max_num_param=None,
    overflow_module_config: dict = None,
    total_num_param=0,
):
    for name, module in model.named_children():
        for source_module, target_module in module_map.items():
            if isinstance(module, source_module):
                num_param = sum(p.numel() for p in module.parameters())
                if max_num_param is not None and total_num_param + num_param > max_num_param:
                    module_config_ = overflow_module_config
                else:
                    module_config_ = module_config
                module_ = target_module(module, **module_config_)
                setattr(model, name, module_)
                total_num_param += num_param
                break
        else:
            total_num_param = enable_vram_management_recursively(
                module, module_map, module_config, max_num_param, overflow_module_config, total_num_param
            )
    return total_num_param


def enable_vram_management(
    model: torch.nn.Module,
    module_map: dict,
    module_config: dict,
    max_num_param=None,
    overflow_module_config: dict = None,
):
    enable_vram_management_recursively(
        model, module_map, module_config, max_num_param, overflow_module_config, total_num_param=0
    )
    model.vram_management_enabled = True
