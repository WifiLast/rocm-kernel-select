"""FFT-based N-D convolution for amd_tuned_torch -- the large-kernel
counterpart to this package's dense conv1d/2d/3d tiers, which are all
tuned for small (3x3-style) kernels.

WHERE THIS COMES FROM. The core algorithm (`fft_conv`/`complex_matmul`/
`to_ntuple`) is vendored from `fft-conv-pytorch` into
`_vendor/fft_conv_pytorch/` -- see that directory's NOTICE.md for the full
provenance and exactly what changed from upstream (mainly: this version
preserves the caller's dtype instead of silently upcasting to float32).
Unlike `cumesh_ops.py`/`flexgemm_ops.py`/`nvdiffrast_ops.py`/
`torchsparse_ops.py`, there is no `third_party/` extension behind this
module at all -- upstream is pure PyTorch (`torch.fft.rfftn`/`irfftn`,
`F.pad`, `torch.kron`, plain `@`), so `available()` below always returns
True: nothing to build, nothing that can fail to import.

WHY THIS EXISTS INSTEAD OF PORTING flash-fft-conv. `source/flash-fft-conv`
(Stanford's FlashFFTConv, "Efficient Convolutions for Long Sequences with
Tensor Cores") was evaluated and rejected as a porting candidate: its FFT
is computed as a Monarch matrix decomposition implemented ENTIRELY via CUDA
`wmma::fragment`/`mma_sync` tensor-core intrinsics (butterfly FFT/IFFT,
Monarch matmul, fwd+bwd, fp16+bf16 -- every kernel file, not an optional
fast path), with no non-tensor-core dataflow to salvage the way
`torchsparse`'s `GatherScatter` path or `nvdiffrast`'s software rasterizer
were. Porting it would mean rewriting every WMMA call against rocWMMA/MFMA
-- real matrix-core work, and only runs on hardware with matrix cores
(CDNA, or RDNA3+ via rocWMMA) at all. `fft-conv-pytorch` gets the same
algorithmic win (FFT beats direct/im2col convolution for large kernels,
since FFT-conv is O(N log N) per spatial dim vs. direct conv's O(N*K))
onto ROCm today, with no porting risk, at the cost of not having
flash-fft-conv's tensor-core speedup on top of that algorithmic win. See
`_vendor/fft_conv_pytorch/NOTICE.md` for the full comparison.

WHY MEASURED, NOT JUST GATED BY KERNEL SIZE. FFT-conv's O(N log N)
asymptotic win over direct convolution's O(N*K) only pays for itself once
K (the kernel's spatial extent) is large enough that the FFT/complex-GEMM/
IFFT overhead is smaller than the multiplies a direct kernel would have
saved by NOT touching every frequency bin. Upstream's own benchmarks put
the crossover around 100+ kernel elements per spatial dim for 1D, but
that is a guess from a different GPU/workload, not a measurement of THIS
card running THIS shape -- so rather than trust it outright (a static
threshold this module used originally, kept below as `maybe_fft_conv1d`
for a caller that wants it), the conv1d auto-patch instead runs
`fftconv1d_candidate` through `amd_tuned_torch.kernel_select`'s per-shape
contest against stock conv1d -- the same benchmark-once-then-cache policy
already used for conv2d/conv3d/linear/bmm/attention (see
kernel_select.py's own module docstring) -- and uses whichever actually
measures faster for that (dtype, shape, stride, padding, dilation)
combination, paid once per distinct shape then cached (in memory and on
disk) like every other kernel_select-gated tier. `AMD_TUNED_TORCH_FFTCONV1D_
MIN_KERNEL` (default 2) survives only as a cheap pre-filter skipping the
contest entirely for kernels obviously too small to bother timing, not as
the win/lose decision itself anymore.

FFT-conv's numerical rounding differs enough from direct convolution's
(an absolute margin that scales with signal magnitude, not the tighter
relative-only bar direct-vs-direct contests tolerate) that `kernel_select`'s
shared per-dtype verification tolerance is too strict for this contest --
see `fftconv_tolerance`'s module comment (near `_FFTCONV_TOLERANCES` below)
for the measured numbers and `kernel_select.pick`'s `tolerance` parameter
this contest passes to override it.

WHAT'S EXPOSED. `fft_conv1d`/`fft_conv2d`/`fft_conv3d` -- explicit,
`F.convNd`-shaped library functions a caller building a Hyena/long-conv/
global-convolution block (the flash-fft-conv/Hyena use case this module
takes its cue from) can reach for directly, at any kernel size, same
"plain library surface" posture as `cumesh_ops`/`flexgemm_ops`'s sparse-3D
ops. `fftconv1d_candidate` is the thunk `amd_tuned_torch.miopen_fallback`
contests against stock conv1d via `kernel_select.pick` before ever trying
MIOpen -- see that module's CAUSAL_CONV1D/FFTCONV1D/SPARSE CONV1D sections
for the full fast-path ordering; unlike the sparse fast path,
`fft_conv`'s real autograd (built entirely from differentiable primitives:
`rfftn`/`irfftn`/`pad`/`kron`/`@`, no custom kernel, no `.detach()`
anywhere) means this needs no `_grad_safe` check -- it stays correct,
gradient and all, whether or not the call is under `torch.no_grad()`.

ALL THREE DIMENSIONALITIES ARE NOW AUTO-PATCHED, not just conv1d (this
paragraph used to say otherwise -- conv2d/conv3d's wiring was added later,
in `amd_tuned_torch/__init__.py`, and this docstring went stale). conv1d's
contest lives here, in `miopen_fallback.py`'s `_try_fftconv1d_fastpath`, as
described above. conv2d/conv3d each get their own FFT candidate instead --
`amd_tuned_torch._fftconv_conv_candidate` -- folded into `_patched_conv2d`/
`_patched_conv3d`'s existing native/CK/aiter/stock contest rather than a
separate sequential fast path, gated by `AMD_TUNED_TORCH_FFTCONV2D`/
`AMD_TUNED_TORCH_FFTCONV3D` (default on) and their own measured-on-gfx1100
minimum kernel widths (`_FFTCONV_CONV2D_MIN_KERNEL`=32,
`_FFTCONV_CONV3D_MIN_KERNEL`=7 -- see that function's own docstring for the
measurements behind them and why 3D's threshold sits an order of magnitude
below 2D's). conv3d also declines outright for a small input volume
(`_FFTCONV_CONV3D_MIN_POSITIONS`, default 2048 spatial positions -- a
hardcoded guess, overridable per GPU by running
tools/benchmark_fftconv3d_min_positions.py once and letting
`amd_tuned_torch.fftconv_calibration` persist the measured crossover, same
precedence -- explicit env var > calibration file > hardcoded guess --
`flexgemm_ops.py`'s own sparse-conv gates already use) before ever
contesting kernel width, since the padded transform's fixed
memory/allocation overhead can't pay for itself on a small volume
regardless of kernel size -- conv2d has no equivalent gate. See
`_fftconv_conv_candidate`'s own docstring in `__init__.py` for the full
eligibility list (dtype/device via `_usable`, no string padding modes,
etc.).

Set AMD_TUNED_TORCH_FFTCONV1D=0 to disable the conv1d auto-patch fast path
entirely (fft_conv1d/2d/3d themselves are unaffected by this flag -- it
only gates fftconv1d_candidate/the kernel_select contest). conv2d/conv3d
have their own independent `AMD_TUNED_TORCH_FFTCONV2D`/`_FFTCONV3D` flags,
not this one.
"""
from __future__ import annotations

import os
from math import ceil, floor
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.fft import irfftn, rfftn


def available() -> bool:
    """Always True -- this module has no compiled extension or optional
    dependency to gate on (see module docstring)."""
    return True


def _env_flag(name: str, default: str = "1") -> bool:
    """Same on/off-string parsing as flexgemm_ops._env_flag -- not imported
    from there to avoid a cross-module dependency for one four-line
    helper."""
    return os.environ.get(name, default).strip().lower() not in ("0", "", "false", "no", "off")


def to_ntuple(val: Union[int, Iterable[int]], n: int) -> Tuple[int, ...]:
    """Casts to a tuple of length `n`. Vendored from fft-conv-pytorch's
    `utils.to_ntuple` (see _vendor/fft_conv_pytorch/NOTICE.md) with one
    change: a bare `str` is excluded from the `Iterable` branch even though
    Python considers it iterable, so `to_ntuple("same", n=2)` raises
    instead of silently iterating characters -- not reachable through this
    module's own call sites (padding="same" is handled before to_ntuple
    ever sees the string) but not a foot-gun worth keeping in a vendored
    copy either."""
    if isinstance(val, Iterable) and not isinstance(val, str):
        out = tuple(val)
        if len(out) == n:
            return out
        raise ValueError(f"Cannot cast tuple of length {len(out)} to length {n}.")
    return n * (val,)


def complex_matmul(a: Tensor, b: Tensor, groups: int = 1) -> Tensor:
    """Grouped complex-valued matrix multiplication.

    DIFFERS FROM UPSTREAM (see _vendor/fft_conv_pytorch/NOTICE.md): fft-conv-
    pytorch's original manually expands this into 4 separate real matmuls
    (a.real@b.real, a.imag@b.imag, a.real@b.imag, a.imag@b.real) plus a
    fresh torch.zeros(..., dtype=complex64) allocation to reassemble the
    real/imag halves into a complex tensor. `a`/`b` are already complex64
    here (sliced straight out of rfftn's output) -- `@`/torch.matmul
    supports complex dtypes directly and dispatches to a single native
    complex GEMM (cgemm, same call vendor BLAS uses for a real complex-valued
    workload -- rocBLAS/hipBLAS on ROCm, cuBLAS on CUDA, MKL/OpenBLAS on
    CPU), so doing the split by hand here was strictly more work for an
    identical result: 4 kernel launches instead of 1, plus the intermediate
    real/imag tensors and the extra allocation to recombine them -- none of
    which the complex GEMM path needs. Numerically identical (both compute
    the same complex product; which multiply/add count the BLAS backend
    uses internally is its own implementation detail, not something this
    call site controls either way)."""
    a = a.view(a.size(0), groups, -1, *a.shape[2:])
    b = b.view(groups, -1, *b.shape[1:])

    a = torch.movedim(a, 2, a.dim() - 1).unsqueeze(-2)
    b = torch.movedim(b, (1, 2), (b.dim() - 1, b.dim() - 2))

    c = a @ b
    c = torch.movedim(c, c.dim() - 1, 2).squeeze(-1)

    return c.view(c.size(0), -1, *c.shape[3:])


def fft_conv(
    signal: Tensor,
    kernel: Tensor,
    bias: Optional[Tensor] = None,
    padding: Union[int, Iterable[int], str] = 0,
    padding_mode: str = "constant",
    stride: Union[int, Iterable[int]] = 1,
    dilation: Union[int, Iterable[int]] = 1,
    groups: int = 1,
) -> Tensor:
    """N-D convolution via FFT: signal_fr = rfft(signal), kernel_fr =
    rfft(kernel), multiply in the frequency domain, irfft back. Wins over
    direct/im2col convolution for large kernels (see module docstring) --
    this project's other conv tiers already cover the small-kernel case.

    Same call signature and semantics as F.convNd (padding="same" also
    supported, stride=1/dilation=1 only), N inferred from
    `signal.ndim - 2`. Algorithm vendored from fft-conv-pytorch (see
    _vendor/fft_conv_pytorch/NOTICE.md).

    MIXED PRECISION: fp16/bf16 STORAGE, fp32 TRANSFORM. Upstream forces
    both `signal` and `kernel` to float32 before doing ANYTHING (dilation
    expansion, padding), then leaves the whole result in float32 for the
    caller to deal with. This version instead keeps `signal`/`kernel` in
    the caller's own dtype through every pure-data-movement step -- the
    `torch.kron` dilation expansion and both `F.pad` calls, which touch
    the largest tensors in this function (the padded signal in
    particular, since a long-kernel/long-sequence workload is exactly
    what this module targets) and carry no precision-sensitive math of
    their own -- keeping the memory-bandwidth win a fp16/bf16 activation
    tensor is supposed to have. Only `rfftn`/`complex_matmul`/`irfftn` (the
    actual transform and frequency-domain pointwise multiply, where low
    precision would dominate the numerical error, and which
    `torch.fft.rfftn`/`irfftn` only support in float32/float64 regardless)
    run in float32 -- cast to float32 right before `rfftn`, cast back to
    `signal`'s original dtype immediately after `irfftn`, so nothing after
    that point (the stride/kernel-size crop, the bias add) pays for
    float32 either. Net external contract is unchanged from before this
    optimization (output dtype always matches `signal`'s -- upstream
    itself silently upcasts a fp16/bf16 caller, still fixed here); what
    changed is WHEN the float32 window opens and closes, narrowed to just
    the three ops that actually need it."""
    n = signal.ndim - 2
    stride_ = to_ntuple(stride, n=n)
    dilation_ = to_ntuple(dilation, n=n)
    if isinstance(padding, str):
        if padding != "same":
            raise ValueError(f"Padding mode {padding} not supported.")
        if stride != 1 or dilation != 1:
            raise ValueError("stride must be 1 for padding='same'.")
        padding_ = tuple((k - 1) / 2 for k in kernel.shape[2:])
    else:
        padding_ = to_ntuple(padding, n=n)

    out_dtype = signal.dtype

    # --- fp16/bf16 storage: dilation expansion + padding, still in
    # `signal`'s original dtype (see docstring's MIXED PRECISION section).
    offset = torch.zeros(1, 1, *dilation_, device=signal.device, dtype=signal.dtype)
    offset[(slice(None), slice(None), *((0,) * n))] = 1.0

    cutoff = tuple(slice(None, -d + 1 if d != 1 else None) for d in dilation_)
    kernel = torch.kron(kernel, offset)[(slice(None), slice(None)) + cutoff]

    signal_padding = [r(p) for p in padding_[::-1] for r in (floor, ceil)]
    signal = F.pad(signal, signal_padding, mode=padding_mode)

    signal_size = signal.size()
    if signal.size(-1) % 2 != 0:
        signal = F.pad(signal, [0, 1])

    kernel_padding = [
        pad
        for i in reversed(range(2, signal.ndim))
        for pad in [0, signal.size(i) - kernel.size(i)]
    ]
    padded_kernel = F.pad(kernel, kernel_padding)

    # --- fp32 transform + pointwise multiply: the only part of this
    # function where precision actually matters, and the only part
    # torch.fft requires it for regardless of `signal`'s own dtype.
    signal_fr = rfftn(signal.float(), dim=tuple(range(2, signal.ndim)))
    kernel_fr = rfftn(padded_kernel.float(), dim=tuple(range(2, signal.ndim)))

    kernel_fr.imag *= -1
    output_fr = complex_matmul(signal_fr, kernel_fr, groups=groups)
    output = irfftn(output_fr, dim=tuple(range(2, signal.ndim)))

    # --- back to fp16/bf16 storage immediately: the crop and bias-add
    # below are again pure data movement over what's now an
    # activation-sized tensor, not a reason to keep paying for float32.
    output = output.to(out_dtype)

    crop_slices = (slice(None), slice(None)) + tuple(
        slice(0, (signal_size[i] - kernel.size(i) + 1), stride_[i - 2])
        for i in range(2, signal.ndim)
    )
    output = output[crop_slices].contiguous()

    if bias is not None:
        bias_shape = tuple([1, -1] + (signal.ndim - 2) * [1])
        output = output + bias.to(out_dtype).view(bias_shape)

    return output


def fft_conv1d(signal: Tensor, kernel: Tensor, bias: Optional[Tensor] = None,
                padding: Union[int, Iterable[int], str] = 0, padding_mode: str = "constant",
                stride: Union[int, Iterable[int]] = 1, dilation: Union[int, Iterable[int]] = 1,
                groups: int = 1) -> Tensor:
    """F.conv1d-equivalent via FFT. `signal` [B,Ci,L], `kernel` [Co,Ci/groups,K]."""
    if signal.dim() != 3 or kernel.dim() != 3:
        raise ValueError("fft_conv1d expects 3D [B,C,L] signal and kernel tensors.")
    return fft_conv(signal, kernel, bias=bias, padding=padding, padding_mode=padding_mode,
                     stride=stride, dilation=dilation, groups=groups)


def fft_conv2d(signal: Tensor, kernel: Tensor, bias: Optional[Tensor] = None,
                padding: Union[int, Iterable[int], str] = 0, padding_mode: str = "constant",
                stride: Union[int, Iterable[int]] = 1, dilation: Union[int, Iterable[int]] = 1,
                groups: int = 1) -> Tensor:
    """F.conv2d-equivalent via FFT. `signal` [B,Ci,H,W], `kernel` [Co,Ci/groups,Kh,Kw].
    `padding_mode="reflection"` is not supported for the 3D case, matching
    upstream (see fft_conv's own padding_mode argument)."""
    if signal.dim() != 4 or kernel.dim() != 4:
        raise ValueError("fft_conv2d expects 4D [B,C,H,W] signal and kernel tensors.")
    return fft_conv(signal, kernel, bias=bias, padding=padding, padding_mode=padding_mode,
                     stride=stride, dilation=dilation, groups=groups)


def fft_conv3d(signal: Tensor, kernel: Tensor, bias: Optional[Tensor] = None,
                padding: Union[int, Iterable[int], str] = 0, padding_mode: str = "constant",
                stride: Union[int, Iterable[int]] = 1, dilation: Union[int, Iterable[int]] = 1,
                groups: int = 1) -> Tensor:
    """F.conv3d-equivalent via FFT. `signal` [B,Ci,D,H,W], `kernel` [Co,Ci/groups,Kd,Kh,Kw]."""
    if signal.dim() != 5 or kernel.dim() != 5:
        raise ValueError("fft_conv3d expects 5D [B,C,D,H,W] signal and kernel tensors.")
    return fft_conv(signal, kernel, bias=bias, padding=padding, padding_mode=padding_mode,
                     stride=stride, dilation=dilation, groups=groups)


# ---------------------------------------------------------------------------
# amd_tuned_torch.kernel_select CONTEST -- see module docstring's "WHY GATED BY
# KERNEL SIZE" section for the background this superseded. Originally this
# module made the direct-vs-FFT decision itself via a hardcoded kernel-width
# guess (100+ elements is where fft-conv-pytorch's own upstream benchmarks
# say FFT-conv starts winning for 1D). That guess is now retired in favor of
# actually measuring both on the first call for each distinct shape and
# caching the winner -- the same policy kernel_select already applies to
# conv2d/conv3d/linear/bmm/attention (see kernel_select.py's own module
# docstring) -- via fftconv1d_candidate below, a thunk-friendly candidate
# amd_tuned_torch.miopen_fallback._try_fftconv1d_fastpath contests against
# stock conv1d through kernel_select.pick. AMD_TUNED_TORCH_FFTCONV1D_MIN_KERNEL
# survives only as a cheap PRE-filter (default 2 -- effectively "any valid
# kernel"): skip building/timing the FFT candidate at all below this width,
# rather than pay a contest for kernels obviously too small to ever win
# (the causal-conv1d fast path already claims width 2-4 depthwise convs
# before this module ever sees them; this is a backstop for the rest).
# Raise it if a workload's own measurement says the contest overhead isn't
# worth paying below some larger width.
# ---------------------------------------------------------------------------

_FFTCONV1D_ENABLED = _env_flag("AMD_TUNED_TORCH_FFTCONV1D", default="1")
_FFTCONV1D_MIN_KERNEL = int(os.environ.get("AMD_TUNED_TORCH_FFTCONV1D_MIN_KERNEL", "2"))

# Looser-than-kernel_select's-default verification tolerance for the
# fftconv-vs-stock contest (passed as kernel_select.pick's `tolerance`
# argument -- see that function's docstring for why a per-contest override
# exists at all). kernel_select._TOLERANCES is calibrated for contests
# between algorithmically SIMILAR candidates (different tile sizes,
# different GEMM instances, all doing direct convolution's same arithmetic
# in a different order) -- FFT-based convolution is a different algorithm
# entirely (frequency-domain complex multiply vs. direct sliding-window
# accumulation), and accumulates floating-point rounding differently: an
# ABSOLUTE margin that scales with signal magnitude, measured up to ~1e-4
# for fp32 in this module's own tests -- well within what any real
# workload would call correct, but outside kernel_select's default fp32
# bar of (rtol=1e-4, atol=1e-5), which would otherwise permanently
# blacklist a numerically-fine candidate the first time it's measured (see
# kernel_select.py's CORRECTNESS VERIFICATION section for what
# "permanently blacklist" means in practice).
_FFTCONV_TOLERANCES = {
    torch.float16: (2e-2, 2e-2),
    torch.bfloat16: (3e-2, 3e-2),
    torch.float32: (1e-3, 1e-4),
}
_FFTCONV_DEFAULT_TOLERANCE = (1e-3, 1e-4)


def fft_conv1d_enabled() -> bool:
    """True unless AMD_TUNED_TORCH_FFTCONV1D=0. Read once at import time, same
    posture as every other env-var gate in this package. Gates the ENTIRE
    fftconv1d_candidate/contest machinery -- when False,
    amd_tuned_torch.miopen_fallback never builds or times the FFT candidate
    at all, not just declines to pick it."""
    return _FFTCONV1D_ENABLED


def fftconv_tolerance(dtype: torch.dtype) -> Tuple[float, float]:
    """(rtol, atol) for verifying an fftconv candidate's output against
    stock conv1d's, for `dtype` -- see this section's module comment above
    _FFTCONV_TOLERANCES for why this differs from
    kernel_select._TOLERANCES's shared defaults. Falls back to
    _FFTCONV_DEFAULT_TOLERANCE for any dtype not in the table (matching
    kernel_select._verify's own fallback-to-default posture for an
    unlisted dtype)."""
    return _FFTCONV_TOLERANCES.get(dtype, _FFTCONV_DEFAULT_TOLERANCE)


def fftconv1d_candidate(input: Tensor, weight: Tensor, bias: Optional[Tensor],
                         stride: int, padding: int, dilation: int, groups: int
                         ) -> Optional[Tensor]:
    """Thunk-friendly fft_conv1d wrapper for kernel_select's contest (see
    amd_tuned_torch.miopen_fallback._try_fftconv1d_fastpath, which builds
    the (name, thunk) candidate list kernel_select.pick times and verifies).
    Returns None instead of raising so an ineligible or failing call simply
    loses the contest rather than breaking it -- matching every other
    kernel_select candidate in this project (see kernel_select.pick's own
    docstring: "a thunk returns None to decline").

    `stride`/`padding`/`dilation` are plain ints here (not tuples) --
    conv1d has exactly one spatial axis, so the caller (miopen_fallback's
    `_scalar` helper) has already reduced whatever F.conv1d received to a
    scalar before this is called.

    None if `input`/`weight` aren't 3D conv1d tensors, `weight`'s kernel
    width is below AMD_TUNED_TORCH_FFTCONV1D_MIN_KERNEL (a cheap pre-filter,
    not a fastest-vs-stock decision -- see this section's module comment),
    or the call fails for any reason (e.g. an unsupported padding_mode/
    padding string)."""
    if input.dim() != 3 or weight.dim() != 3:
        return None
    if weight.shape[-1] < _FFTCONV1D_MIN_KERNEL:
        return None
    try:
        return fft_conv1d(input, weight, bias=bias, padding=padding,
                           stride=stride, dilation=dilation, groups=groups)
    except (RuntimeError, ValueError, TypeError):
        return None


def maybe_fft_conv1d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
                      stride: Tuple[int, ...] = (1,), padding: Tuple[int, ...] = (0,),
                      dilation: Tuple[int, ...] = (1,), groups: int = 1,
                      min_kernel: Optional[int] = None) -> Optional[Tensor]:
    """Static-heuristic direct/FFT conv1d switch: routes to fft_conv1d only
    when `weight`'s kernel width is at or above `min_kernel` (default 128
    -- fft-conv-pytorch's own upstream benchmarks put FFT-conv's crossover
    around 100+ kernel elements for 1D), no timing involved. NOT what
    amd_tuned_torch.miopen_fallback's auto-patch calls -- that now goes
    through kernel_select's measured contest instead (fftconv1d_candidate
    above), which empirically decides per shape rather than trusting this
    one upstream-benchmark-derived guess. This function remains for a
    caller that wants FFT-conv's algorithmic-complexity argument applied
    directly, without paying a contest's one-time-per-shape measurement
    cost.

    Unlike flexgemm_ops.maybe_sparse_conv1d, this needs no `_grad_safe`
    check -- fft_conv's entire computation is ordinary differentiable
    PyTorch (rfftn/irfftn/pad/kron/@), so it has a real, correct backward
    pass and stays safe to call under autograd.

    None when AMD_TUNED_TORCH_FFTCONV1D=0 (fft_conv1d_enabled() is False),
    `weight`'s kernel width is below `min_kernel`, or the call fails for
    any reason (e.g. an unsupported padding_mode/padding string)."""
    if not fft_conv1d_enabled():
        return None
    if input.dim() != 3 or weight.dim() != 3:
        return None
    threshold = 128 if min_kernel is None else min_kernel
    if weight.shape[-1] < threshold:
        return None
    try:
        return fft_conv1d(input, weight, bias=bias, padding=padding[0],
                           stride=stride[0], dilation=dilation[0], groups=groups)
    except (RuntimeError, ValueError, TypeError):
        return None
