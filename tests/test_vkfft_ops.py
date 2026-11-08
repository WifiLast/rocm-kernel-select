"""Tests for amd_tuned_torch.vkfft_ops -- VkFFT via pyvkfft's OpenCL backend.

Everything that needs a GPU is skipped when `available()` is False, which
on this box most often means `OCL_ICD_VENDORS` is unset: the ROCm OpenCL
ICD is not registered in /etc/OpenCL/vendors, so an OpenCL process sees
only the NVIDIA platform. Run the hardware half with

    OCL_ICD_VENDORS=$CONDA_PREFIX/etc/OpenCL/vendors \\
        python -m pytest tests/test_vkfft_ops.py

What runs WITHOUT any of that: the graceful-degradation contract, which is
the part most likely to break a caller (an optional tier must report
absence, never raise on import or on `available()`).
"""
import numpy as np
import pytest
import torch

from amd_tuned_torch import vkfft_ops

requires_vkfft = pytest.mark.skipif(
    not vkfft_ops.available(),
    reason=f"vkfft_ops unavailable: {vkfft_ops.load_error()}")


class TestDegradesGracefully:
    def test_available_never_raises(self):
        assert isinstance(vkfft_ops.available(), bool)

    def test_load_error_explains_unavailability(self):
        if not vkfft_ops.available():
            assert vkfft_ops.load_error()

    def test_version_string_safe_either_way(self):
        assert isinstance(vkfft_ops.version_string(), str)

    def test_disabled_by_env(self, monkeypatch):
        monkeypatch.setattr(vkfft_ops, "_ENABLED", False)
        assert vkfft_ops.available() is False

    def test_torch_entry_points_raise_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(vkfft_ops, "_ENABLED", False)
        x = torch.randn(8, dtype=torch.complex64)
        with pytest.raises(RuntimeError, match="unavailable"):
            vkfft_ops.fftn(x)


@requires_vkfft
class TestMatchesTorchFFT:
    """The only claim that matters for a drop-in FFT provider: same answer
    as torch.fft. Shapes deliberately include a prime length and a
    non-power-of-2 smooth one, since radix coverage is where two FFT
    libraries actually diverge."""

    @pytest.mark.parametrize("shape,dim", [
        ((1024,), None),
        ((8, 1024), (-1,)),
        ((64, 1000), (-1,)),
        ((997,), None),          # prime -> Bluestein
        ((32, 31, 17), None),
        ((4, 128, 128), (-2, -1)),
    ])
    def test_fftn(self, shape, dim):
        x = torch.randn(*shape, dtype=torch.complex64, device="cuda")
        got = vkfft_ops.fftn(x, dim=dim)
        want = torch.fft.fftn(x, dim=dim) if dim else torch.fft.fftn(x)
        assert got.shape == want.shape
        assert got.dtype == x.dtype
        rel = (got - want).abs().max().item() / want.abs().max().item()
        assert rel < 1e-4, f"rel err {rel:.2e}"

    def test_ifftn_roundtrips(self):
        x = torch.randn(4, 512, dtype=torch.complex64, device="cuda")
        back = vkfft_ops.ifftn(vkfft_ops.fftn(x, dim=(-1,)), dim=(-1,))
        rel = (back - x).abs().max().item() / x.abs().max().item()
        assert rel < 1e-4

    def test_output_lands_on_the_input_device(self):
        x = torch.randn(256, dtype=torch.complex64, device="cuda")
        assert vkfft_ops.fftn(x).device == x.device


@requires_vkfft
class TestContracts:
    def test_rejects_real_input(self):
        with pytest.raises(TypeError, match="complex"):
            vkfft_ops.fftn(torch.randn(256, device="cuda"))

    def test_rejects_non_trailing_dims(self):
        x = torch.randn(8, 16, 32, dtype=torch.complex64, device="cuda")
        with pytest.raises(ValueError, match="trailing"):
            vkfft_ops.fftn(x, dim=(0,))

    def test_picks_an_amd_device_by_default(self):
        """Guards the mistake this module is designed against: this box also
        exposes an NVIDIA CMP 50HX through OpenCL, and benchmarking VkFFT
        there while rocFFT runs on the Radeon would compare two cards."""
        import os
        if os.environ.get("AMD_TUNED_TORCH_VKFFT_DEVICE"):
            pytest.skip("device explicitly overridden")
        name = vkfft_ops.device_name().lower()
        assert "gfx" in name or "amd" in name, f"picked {name!r}"

    def test_plan_cache_reuses_apps(self):
        x = torch.randn(4, 256, dtype=torch.complex64, device="cuda")
        vkfft_ops.fftn(x, dim=(-1,))
        n = len(vkfft_ops._apps)
        vkfft_ops.fftn(x, dim=(-1,))
        assert len(vkfft_ops._apps) == n, "identical shape rebuilt its plan"


@requires_vkfft
class TestZeroCopyPath:
    """fftn_cl is the configuration VkFFT is actually competitive in -- a
    caller already holding pyopencl arrays, with no host round trip."""

    def test_fftn_cl_matches_numpy(self):
        import pyopencl.array as cla
        a = (np.random.rand(4, 512) + 1j * np.random.rand(4, 512)).astype(np.complex64)
        ga = cla.to_device(vkfft_ops._queue, a.copy())
        out = vkfft_ops.fftn_cl(ga, ndim=1).get()
        want = np.fft.fftn(a, axes=(-1,))
        rel = np.abs(out - want).max() / np.abs(want).max()
        assert rel < 1e-4, f"rel err {rel:.2e}"
