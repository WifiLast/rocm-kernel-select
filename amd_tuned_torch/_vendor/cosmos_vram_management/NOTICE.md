# Provenance

`amd_tuned_torch/vram_management_ops.py` is a near-verbatim port of:

  https://github.com/nvidia-cosmos/cosmos-predict2
  `cosmos_predict2/vram_management/layers.py` (`cast_to`,
  `AutoWrappedModule`, `AutoWrappedLinear`,
  `enable_vram_management_recursively`, `enable_vram_management`), plus
  `cosmos_predict2/models/utils.py`'s `init_weights_on_device` (vendored
  privately into the same file as `_init_weights_on_device`, to avoid a
  runtime dependency on the third_party/cosmos-predict2 checkout).
  Apache License 2.0, Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
  See LICENSE in this directory for the full license text. The exact
  checkout this was ported from lives in this repo at
  `source/cmp_ext_turing/third_party/cosmos-predict2/cosmos_predict2/
  vram_management/layers.py`.

What changed from upstream, and why:

  - `init_weights_on_device` was copied in as a private
    `_init_weights_on_device` instead of importing
    `cosmos_predict2.models.utils` -- the only other thing that function
    depended on was `torch`, so inlining it avoids amd_tuned_torch taking
    on a dependency on the third_party/cosmos-predict2 checkout (which is
    a separately-vendored NVIDIA project analyzed for portability, not a
    library amd_tuned_torch is meant to import from) just for one small,
    self-contained context manager.
  - Nothing else changed. Upstream is plain PyTorch (`.to(dtype=,
    device=)`, `copy.deepcopy`, `torch.nn.functional.linear`) with zero
    CUDA-specific calls anywhere in this file -- no `.cuda()`, no
    `torch.cuda.*`, no compiled extension -- so there was no ROCm
    adaptation to do. The wrapped `forward()` paths (`AutoWrappedLinear`'s
    `F.linear` call in particular) transparently pick up whichever
    F.linear tier amd_tuned_torch's own dispatch installs once weights
    reach `computation_device`, without this file needing to know that.

UNVALIDATED in the sense that no benchmark in this package measures the
offload/onload transfer cost itself (a PCIe-link-speed number, not a
kernel-choice one, and therefore out of scope for the "measured against
stock" bar the rest of this package holds itself to) -- but the ported
code's correctness is not novel: it is the same `.to()`/`F.linear` calls
this package already exercises elsewhere, just sequenced around a state
flag.
