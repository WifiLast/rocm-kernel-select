"""amd_tuned_torch.vkfft_ops -- VkFFT (via pyvkfft's OpenCL backend) as an
FFT provider, alongside `rocfft_ops`' direct rocFFT binding and the
`torch.fft` path both `fftconv_ops` shims already contest.

READ THIS BEFORE WIRING IT INTO ANYTHING. Unlike `rocfft_ops`, this tier
CANNOT operate on a torch tensor in place, and the reason is structural
rather than a missing optimization:

  * `rocfft_ops` binds librocfft's C API, which takes a raw device
    address. A torch ROCm tensor's `data_ptr()` IS that address, so the
    transform runs on the caller's own memory with nothing copied.
  * pyvkfft's OpenCL backend takes `src.data.int_ptr` -- an OpenCL
    `cl_mem` HANDLE, not a device address (see pyvkfft/opencl.py's
    `fft`/`ifft`). A `cl_mem` is an opaque object owned by the OpenCL
    runtime; there is no API, in pyopencl or in OpenCL itself, that wraps
    an externally allocated HIP pointer as one. `cl.Buffer.from_int_ptr`
    takes another cl_mem, not an address, and `USE_HOST_PTR` against a
    device pointer fails with `INVALID_HOST_PTR` (verified, not assumed).

HIP and OpenCL sit on the same HSA runtime on ROCm, but torch allocates
through `hipMalloc` and pyopencl through `clCreateBuffer`, and the two
allocations are not mutually addressable. So every torch<->VkFFT call here
pays a **round trip through host memory**: device->host, host->OpenCL
buffer, transform, and back. That is not a tax that tuning removes, and it
is large enough to dominate the transform for every shape measured -- see
`tools/benchmark_vkfft_vs_rocfft.py` and the numbers in this module's
integration notes. THIS TIER IS THEREFORE NOT WIRED INTO `fftconv_ops`'
rfftn/irfftn contest, and should not be: adding a candidate that must
copy the data twice to a contest whose other candidates do not copy it at
all would only spend measurement time discovering that, once per shape.

WHAT IT IS ACTUALLY FOR. Two things:

  1. A caller ALREADY HOLDING pyopencl arrays (an OpenCL pipeline that
     never had a torch tensor) gets VkFFT with no copy at all, via
     `fftn_cl`/`ifftn_cl` below, which take and return `pyopencl.array.
     Array`. That is the configuration VkFFT is competitive in, and the
     only one where this module's numbers are worth anything.
  2. A like-for-like measurement of VkFFT's KERNELS against rocFFT's,
     with transfers excluded on both sides, which is what
     `tools/benchmark_vkfft_vs_rocfft.py` reports separately from the
     end-to-end torch numbers.

ENVIRONMENT. pyvkfft is installed OpenCL-only here (it has CUDA and
OpenCL backends, no HIP, so OpenCL is the only route to gfx1100). The
ROCm OpenCL ICD is NOT registered in /etc/OpenCL/vendors, so an OpenCL
process sees only the NVIDIA platform unless `OCL_ICD_VENDORS` points at a
vendors directory naming libamdocl64.so by ABSOLUTE path -- one lives at
`$CONDA_PREFIX/etc/OpenCL/vendors`. `available()` returns False (never
raises) when pyopencl/pyvkfft are missing or no OpenCL GPU is visible, so
an unconfigured environment degrades quietly, same contract as every other
optional tier in this package.

DEVICE SELECTION. `_pick_device` prefers an AMD/gfx device so this lands
on the same GPU torch is using. If none is visible and torch is a ROCm
build, it reports UNAVAILABLE rather than falling back to another vendor's
GPU: this box also exposes an NVIDIA CMP 50HX through OpenCL, so a
fallback would transform on a different card than the one holding the
caller's tensors -- right answers, meaningless timings, and a silent extra
PCIe hop. In practice that situation means the ROCm ICD is unregistered
rather than absent, and `load_error()` says so. Override deliberately with
AMD_TUNED_TORCH_VKFFT_DEVICE (a substring matched against the device
name).
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

_ENABLED = os.environ.get("AMD_TUNED_TORCH_VKFFT", "1").strip().lower() not in (
    "0", "", "false", "no", "off")

_load_error: Optional[str] = None
_cl = None
_cla = None
_VkFFTApp = None
_ctx = None
_queue = None
_device = None
_apps: dict = {}


def _load():
    """Import pyopencl/pyvkfft once, recording why it failed rather than
    raising -- `available()` is the only thing callers should branch on."""
    global _cl, _cla, _VkFFTApp, _load_error
    if _cl is not None or _load_error is not None:
        return
    try:
        import pyopencl as cl
        import pyopencl.array as cla
        from pyvkfft.opencl import VkFFTApp
    except Exception as exc:  # ImportError, or a broken OpenCL install
        _load_error = f"{type(exc).__name__}: {exc}"
        return
    _cl, _cla, _VkFFTApp = cl, cla, VkFFTApp


def _pick_device():
    """First OpenCL GPU whose name matches AMD_TUNED_TORCH_VKFFT_DEVICE, or
    the first AMD/gfx one, or failing both the first GPU of any vendor."""
    want = os.environ.get("AMD_TUNED_TORCH_VKFFT_DEVICE", "").strip().lower()
    gpus = []
    for plat in _cl.get_platforms():
        try:
            gpus += list(plat.get_devices(_cl.device_type.GPU))
        except Exception:
            continue  # a platform with no GPU raises rather than returning []
    if not gpus:
        return None
    if want:
        for d in gpus:
            if want in d.name.lower():
                return d
    for d in gpus:
        if "gfx" in d.name.lower() or "amd" in d.vendor.lower():
            return d
    # No AMD device visible. On a ROCm torch build, falling back to whatever
    # other vendor's GPU OpenCL happens to expose is WORSE than reporting
    # nothing: this box also shows an NVIDIA CMP 50HX through OpenCL, so the
    # fallback would silently transform on a different card than the one
    # holding the caller's tensors -- correct numbers, meaningless timings,
    # and an extra PCIe hop nobody asked for. The overwhelmingly likely cause
    # is an unregistered ROCm ICD rather than a genuine absence, so say so.
    if getattr(torch.version, "hip", None):
        global _load_error
        _load_error = (
            "no AMD OpenCL device visible (found only: "
            + ", ".join(sorted({d.name for d in gpus}))
            + "). The ROCm OpenCL ICD is not registered in /etc/OpenCL/vendors; "
            "set OCL_ICD_VENDORS to a vendors directory naming libamdocl64.so "
            "by absolute path, e.g. $CONDA_PREFIX/etc/OpenCL/vendors. Set "
            "AMD_TUNED_TORCH_VKFFT_DEVICE to use a non-AMD device deliberately.")
        return None
    return gpus[0]


def _ensure_context() -> bool:
    global _ctx, _queue, _device, _load_error
    if _queue is not None:
        return True
    _load()
    if _cl is None:
        return False
    try:
        _device = _pick_device()
        if _device is None:
            # _pick_device may already have recorded a far more specific
            # reason (wrong-vendor devices present); don't clobber it.
            if not _load_error:
                _load_error = ("no OpenCL GPU device visible -- check "
                               "OCL_ICD_VENDORS")
            return False
        _ctx = _cl.Context([_device])
        _queue = _cl.CommandQueue(_ctx)
    except Exception as exc:
        _load_error = f"{type(exc).__name__}: {exc}"
        return False
    return True


def available() -> bool:
    """True if pyopencl + pyvkfft import and an OpenCL GPU is reachable.
    Never raises; see `load_error()` for why it is False."""
    if not _ENABLED:
        return False
    return _ensure_context()


def load_error() -> Optional[str]:
    """Why `available()` is False, or None if it isn't."""
    return _load_error


def device_name() -> Optional[str]:
    """Name of the OpenCL device this module transforms on, or None."""
    return _device.name if (available() and _device is not None) else None


def version_string() -> str:
    """pyvkfft and VkFFT versions, or a message saying why they're absent."""
    if not available():
        return f"unavailable ({load_error()})"
    from pyvkfft.version import __version__, vkfft_version
    return f"pyvkfft {__version__} / VkFFT {vkfft_version()}"


def reset() -> None:
    """Drop every cached VkFFTApp (each holds device-side plan buffers) and
    the context. Mainly for tests that change device selection."""
    global _ctx, _queue, _device
    _apps.clear()
    _ctx = _queue = _device = None


def _get_app(shape: Tuple[int, ...], dtype, ndim: int, r2c: bool, inplace: bool):
    """Cached VkFFTApp. Building one compiles kernels, so this is the
    difference between measuring VkFFT and measuring its compiler --
    equivalent in purpose to rocfft_ops' plan cache."""
    key = (tuple(shape), dtype, ndim, r2c, inplace)
    app = _apps.get(key)
    if app is None:
        app = _VkFFTApp(tuple(shape), dtype, queue=_queue, ndim=ndim,
                        inplace=inplace, r2c=r2c)
        _apps[key] = app
    return app


# ---------------------------------------------------------------------------
# Native OpenCL entry points -- no copy, for a caller already in OpenCL land.
# ---------------------------------------------------------------------------
def fftn_cl(arr, ndim: Optional[int] = None):
    """Forward complex-to-complex transform of a pyopencl Array, in place.
    Returns `arr`. This is the zero-copy path; `ndim` defaults to the
    array's own rank."""
    if not available():
        raise RuntimeError(f"vkfft_ops unavailable: {load_error()}")
    n = arr.ndim if ndim is None else ndim
    return _get_app(arr.shape, arr.dtype, n, r2c=False, inplace=True).fft(arr)


def ifftn_cl(arr, ndim: Optional[int] = None):
    """Inverse of `fftn_cl`, in place. Returns `arr`."""
    if not available():
        raise RuntimeError(f"vkfft_ops unavailable: {load_error()}")
    n = arr.ndim if ndim is None else ndim
    return _get_app(arr.shape, arr.dtype, n, r2c=False, inplace=True).ifft(arr)


def to_cl(x: torch.Tensor):
    """Copy a torch tensor into a fresh pyopencl Array -- device->host->
    device. Exposed because it is the cost this module is about, and a
    caller comparing tiers should be able to time it on its own."""
    if not available():
        raise RuntimeError(f"vkfft_ops unavailable: {load_error()}")
    return _cla.to_device(_queue, x.detach().cpu().numpy())


def from_cl(arr, device=None, dtype=None) -> torch.Tensor:
    """Inverse of `to_cl`."""
    out = torch.from_numpy(arr.get())
    if device is not None:
        out = out.to(device)
    return out.to(dtype) if dtype is not None else out


# ---------------------------------------------------------------------------
# torch entry points -- correct, and dominated by the host round trip.
# ---------------------------------------------------------------------------
def fftn(x: torch.Tensor, dim: Optional[tuple] = None) -> torch.Tensor:
    """`torch.fft.fftn`-equivalent served by VkFFT, for a COMPLEX input.

    Transforms the trailing `len(dim)` axes, batched over the leading ones
    (VkFFT's own `ndim` convention), so `dim` must name trailing axes --
    same restriction `rocfft_ops` documents.

    Every call copies the input to the host and the result back. That is
    inherent, not incidental: see the module docstring. Use `fftn_cl` if
    your data is already in an OpenCL buffer."""
    if not available():
        raise RuntimeError(f"vkfft_ops unavailable: {load_error()}")
    if not x.is_complex():
        raise TypeError("vkfft_ops.fftn expects a complex tensor; "
                        "use torch.fft.rfftn for real input.")
    ndim = x.dim() if dim is None else len(dim)
    if dim is not None:
        want = tuple(range(x.dim() - len(dim), x.dim()))
        if tuple(d % x.dim() for d in dim) != want:
            raise ValueError("vkfft_ops.fftn transforms trailing dims only, "
                             f"got dim={dim} for a {x.dim()}D tensor.")
    arr = to_cl(x.contiguous())
    out = _get_app(arr.shape, arr.dtype, ndim, r2c=False, inplace=True).fft(arr)
    return from_cl(out, device=x.device, dtype=x.dtype)


def ifftn(x: torch.Tensor, dim: Optional[tuple] = None) -> torch.Tensor:
    """Inverse of `fftn`, same trailing-dims restriction and same host
    round trip."""
    if not available():
        raise RuntimeError(f"vkfft_ops unavailable: {load_error()}")
    if not x.is_complex():
        raise TypeError("vkfft_ops.ifftn expects a complex tensor.")
    ndim = x.dim() if dim is None else len(dim)
    arr = to_cl(x.contiguous())
    out = _get_app(arr.shape, arr.dtype, ndim, r2c=False, inplace=True).ifft(arr)
    return from_cl(out, device=x.device, dtype=x.dtype)


# ---------------------------------------------------------------------------
# rfftn / irfftn -- the exact transform pair fftconv_ops' shims contest, so
# this tier can be entered as a kernel_select candidate next to rocfft_ops
# and torch.fft. Same signatures as rocfft_ops.rfftn/irfftn.
#
# These carry the same unavoidable host round trip as fftn/ifftn above, and
# exist so the claim "VkFFT cannot win from a torch tensor" is something
# kernel_select can MEASURE per shape rather than something this module
# asserts. See AS A CONTEST CANDIDATE in the module docstring.
# ---------------------------------------------------------------------------
def _np_real(dtype):
    import numpy as np
    return {torch.float32: np.float32, torch.float64: np.float64}.get(dtype)


def rfftn(x: torch.Tensor, dim: Optional[tuple] = None) -> torch.Tensor:
    """Real-to-complex forward transform over the trailing `len(dim)` axes,
    batched over the leading ones. Returns a complex tensor whose last axis
    is `x.shape[-1] // 2 + 1`, matching torch.fft.rfftn."""
    if not available():
        raise RuntimeError(f"vkfft_ops unavailable: {load_error()}")
    if x.is_complex():
        raise TypeError("vkfft_ops.rfftn expects a real tensor.")
    npdt = _np_real(x.dtype)
    if npdt is None:
        raise TypeError(f"vkfft_ops.rfftn: unsupported dtype {x.dtype}")
    ndim = x.dim() if dim is None else len(dim)
    if dim is not None:
        want = tuple(range(x.dim() - len(dim), x.dim()))
        if tuple(d % x.dim() for d in dim) != want:
            raise ValueError("vkfft_ops.rfftn transforms trailing dims only, "
                             f"got dim={dim} for a {x.dim()}D tensor.")
    x = x.contiguous()
    src = _cla.to_device(_queue, x.detach().cpu().numpy().astype(npdt, copy=False))
    out_shape = tuple(x.shape[:-1]) + (x.shape[-1] // 2 + 1,)
    import numpy as np
    dest = _cla.empty(_queue, out_shape,
                      np.complex64 if npdt is np.float32 else np.complex128)
    app = _get_app(tuple(x.shape), npdt, ndim, r2c=True, inplace=False)
    app.fft(src, dest)
    return from_cl(dest, device=x.device)


def irfftn(x: torch.Tensor, s: tuple, dim: Optional[tuple] = None) -> torch.Tensor:
    """Complex-to-real inverse transform, the counterpart to `rfftn`. `s` is
    the real output's transformed-axis sizes, as torch.fft.irfftn takes."""
    if not available():
        raise RuntimeError(f"vkfft_ops unavailable: {load_error()}")
    if not x.is_complex():
        raise TypeError("vkfft_ops.irfftn expects a complex tensor.")
    import numpy as np
    npdt = np.float32 if x.dtype == torch.complex64 else np.float64
    ndim = len(s) if dim is None else len(dim)
    real_shape = tuple(x.shape[:-len(s)]) + tuple(int(v) for v in s)
    x = x.contiguous()
    src = _cla.to_device(_queue, x.detach().cpu().numpy())
    dest = _cla.empty(_queue, real_shape, npdt)
    app = _get_app(real_shape, npdt, ndim, r2c=True, inplace=False)
    app.ifft(src, dest)
    out = from_cl(dest, device=x.device)
    return out.to(torch.float32 if npdt is np.float32 else torch.float64)
