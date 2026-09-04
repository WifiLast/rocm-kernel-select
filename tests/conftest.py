"""Pytest fixtures shared by the amd_tuned_torch monkeypatch tests.

``amd_tuned_torch/__init__.py`` hard-imports its compiled HIP extension
(``amd_tuned_torch._native``: group_norm, conv2d, conv3d) at module load time and raises
``ImportError`` if it isn't present. Building that extension requires
hipcc and a ROCm toolchain, not available on every machine that runs these
tests, so before ``amd_tuned_torch`` is ever imported we install a fake
``amd_tuned_torch._native`` module into ``sys.modules`` whose every kernel is a
``MagicMock``. Likewise, ``amd_tuned_torch.te_ops`` and ``amd_tuned_torch.aiter_ops`` degrade
gracefully when TransformerEngine/aiter aren't installed (``available()``
returns False), so tests that want to exercise those dispatch branches
monkeypatch ``available`` and stub the relevant functions.

This lets the *Python-level dispatch logic* (grad-safety checks, dtype/shape
eligibility, fallback-on-error) be tested anywhere, independent of whether a
real RX 7900 XTX, aiter, or TransformerEngine are available.

``AMD_TUNED_TORCH_AUTOPATCH`` is forced to ``"0"`` before import so importing the
package doesn't silently patch global ``torch``/``torch.nn.functional``
state as a side effect of collecting these tests; individual tests opt in
via the ``patched`` fixture instead.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

# Every kernel exported by src/main_rocm.cpp / amd_tuned_torch.ops.
# conv3d_fp16_winograd_bt8_bc8 is testing/opt-in-only (see
# amd_tuned_torch.enable_conv3d_winograd_fp16) -- still exported on
# amd_tuned_torch.ops like every other kernel here, so it needs a mock too.
NATIVE_OPS = ["group_norm", "conv2d", "conv3d", "conv3d_fp16_winograd_bt8_bc8"]

# Every function exported by amd_tuned_torch/te_ops.py (besides `available`).
TE_OPS = ["layer_norm", "rms_norm", "gelu", "silu", "scaled_dot_product_attention"]

# Every function exported by amd_tuned_torch/aiter_ops.py (besides `available`).
AITER_OPS = ["linear_fp16", "bmm_fp16", "linear_int8", "conv2d_fp16"]

# SmoothQuant calibration functions in amd_tuned_torch/aiter_ops.py -- not part of
# the `aiter` fixture's blanket mock (unlike AITER_OPS above, these are pure
# Python/PyTorch logic worth exercising for real, not stubbing out).

os.environ.setdefault("AMD_TUNED_TORCH_AUTOPATCH", "0")
# Same reasoning for the conv contest: kernel_select.pick() TIMES each
# candidate on a real GPU and calls every candidate several times, which
# both needs hardware and destroys the call-count assertions these
# dispatch-logic tests are built on. Forced off here so this suite keeps
# testing the fixed tier ordering; the contest has its own dedicated,
# hardware-free suite in tests/test_kernel_select.py.
os.environ.setdefault("AMD_TUNED_TORCH_MEASURE_KERNELS", "0")

if "amd_tuned_torch._native" not in sys.modules:
    _fake_native = types.ModuleType("amd_tuned_torch._native")
    for _name in NATIVE_OPS:
        setattr(_fake_native, _name, MagicMock(name=_name))
    sys.modules["amd_tuned_torch._native"] = _fake_native

# has_ck()/has_hipblaslt() must be a real False, not a MagicMock: a truthy
# mock would make ck_ops.available()/hipblaslt_ops.available() report their
# tier as present in every test, and e.g. _native_ck.ck_conv() would then
# return a MagicMock rather than None -- i.e. the tier would silently
# "succeed" and swallow every conv2d/conv3d dispatch these tests are
# checking. Tests that want the CK/hipBLASLt path monkeypatch that module's
# `available` themselves, the same way they do for aiter and TE. CK and
# hipBLASLt are their own extensions now (amd_tuned_torch._native_ck /
# ._native_hipblaslt, see setup.py) -- faked here the same way _native is,
# so ck_ops.py/ck_gemm_ops.py/ck_norm_ops.py/hipblaslt_ops.py's own
# `from . import _native_ck as _C` / `_native_hipblaslt as _C` resolve
# deterministically regardless of what's actually built on this machine.
if "amd_tuned_torch._native_ck" not in sys.modules:
    _fake_native_ck = types.ModuleType("amd_tuned_torch._native_ck")
    _fake_native_ck.has_ck = MagicMock(name="has_ck", return_value=False)
    sys.modules["amd_tuned_torch._native_ck"] = _fake_native_ck

if "amd_tuned_torch._native_hipblaslt" not in sys.modules:
    _fake_native_hipblaslt = types.ModuleType("amd_tuned_torch._native_hipblaslt")
    _fake_native_hipblaslt.has_hipblaslt = MagicMock(name="has_hipblaslt", return_value=False)
    sys.modules["amd_tuned_torch._native_hipblaslt"] = _fake_native_hipblaslt

import amd_tuned_torch  # noqa: E402 -- must come after the sys.modules stub above
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# Ops amd_tuned_torch deliberately never patches: everything outside the core
# linear/matmul/bmm/conv2d/conv3d/group_norm/attention/rms_norm/gelu/silu
# set, plus layer_norm specifically -- it has a working patch
# (_patched_layer_norm) that enable() just never installs, because it
# benchmarked slower than stock on RX 7900 XTX (see benchmark.json). See the
# module docstring in amd_tuned_torch/__init__.py for why.
STOCK_UNPATCHED = {
    (F, "layer_norm"): F.layer_norm,
    (F, "softmax"): F.softmax,
}


@pytest.fixture
def native():
    """The fake native module (``amd_tuned_torch.ops``), reset before each test."""
    for name in NATIVE_OPS:
        getattr(amd_tuned_torch.ops, name).reset_mock(return_value=True, side_effect=True)
    return amd_tuned_torch.ops


@pytest.fixture
def te(monkeypatch):
    """Stubs amd_tuned_torch.te_ops as available with every function mocked, reset
    before each test. Use together with `force_eligible` or a real CUDA
    fp16/bf16/fp32 tensor to reach the TE-backed dispatch branches."""
    monkeypatch.setattr(amd_tuned_torch.te_ops, "available", lambda: True)
    for name in TE_OPS:
        monkeypatch.setattr(amd_tuned_torch.te_ops, name, MagicMock(name=name))
    return amd_tuned_torch.te_ops


@pytest.fixture
def aiter(monkeypatch):
    """Stubs amd_tuned_torch.aiter_ops as available with every function mocked
    (linear_fp16/bmm_fp16 -- the default GEMM backend installed by
    enable() -- plus conv2d_fp16, the bf16-only second tier behind
    _patched_conv2d's native kernel, and linear_int8, the opt-in-only one),
    reset before each test. Use together with `force_eligible` or a real
    CUDA fp16/bf16 tensor to reach the aiter-backed dispatch branches."""
    monkeypatch.setattr(amd_tuned_torch.aiter_ops, "available", lambda: True)
    for name in AITER_OPS:
        monkeypatch.setattr(amd_tuned_torch.aiter_ops, name, MagicMock(name=name))
    return amd_tuned_torch.aiter_ops


@pytest.fixture
def hub(monkeypatch):
    """Stubs amd_tuned_torch.hub_ops as available with fused_silu_mul mocked --
    unlike aiter/te, no real network/Hub access is ever exercised by this
    fixture; get_hub_kernel() itself is left real (it's pure Python cache/
    error-handling logic, worth testing for real) but fused_silu_mul is
    mocked since it's the one function that would otherwise reach into an
    actual Hub-downloaded module's API surface."""
    monkeypatch.setattr(amd_tuned_torch.hub_ops, "available", lambda: True)
    monkeypatch.setattr(amd_tuned_torch.hub_ops, "fused_silu_mul", MagicMock(name="fused_silu_mul"))
    return amd_tuned_torch.hub_ops


def force_eligible(monkeypatch):
    """Makes every wrapper's grad-safety/dtype/device gate pass.

    amd_tuned_torch's eligibility checks (`_usable`) require a real CUDA tensor of a
    supported dtype. Forcing them to pass lets the *rest* of each wrapper's
    dispatch logic (shape/argument-specific fallbacks, native/TE-call
    wiring, error fallback) be exercised with plain CPU tensors on any
    machine.
    """
    monkeypatch.setattr(amd_tuned_torch, "_usable", lambda *a, **k: True)
    monkeypatch.setattr(amd_tuned_torch, "_grad_safe", lambda *a, **k: True)
