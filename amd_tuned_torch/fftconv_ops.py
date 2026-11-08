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
That property is now load-bearing in a second place: this module's
`rfftn`/`irfftn` are shims that can serve a transform straight from
librocfft via `rocfft_ops` (see the ROCFFT ROUTING comment above
`_rocfft_usable`), and those raw ctypes calls produce no grad_fn, so the
shims hand back to torch.fft whenever a gradient could be involved. The
autograd path through `fft_conv` is therefore unchanged and still entirely
torch.fft's -- verified bit-identical gradients with the routing on and
off.

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
from functools import lru_cache
from math import ceil, floor
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.fft import irfftn as _torch_irfftn, rfftn as _torch_rfftn

from . import kernel_select, rocfft_ops, vkfft_ops


def available() -> bool:
    """Always True -- this module has no compiled extension or optional
    dependency to gate on (see module docstring)."""
    return True


def _env_flag(name: str, default: str = "1") -> bool:
    """Same on/off-string parsing as flexgemm_ops._env_flag -- not imported
    from there to avoid a cross-module dependency for one four-line
    helper."""
    return os.environ.get(name, default).strip().lower() not in ("0", "", "false", "no", "off")


# ---------------------------------------------------------------------------
# rocFFT routing for this module's two transforms.
#
# `rocfft_ops` calls the SAME librocfft.so that torch.fft already reaches
# through hipFFT, so this is not a different FFT -- measured bit-identical
# output (max relative error exactly 0.0) on gfx1100 / rocFFT 1.0.36 for 1D,
# 2D and 3D. What differs is how much work happens around the transform,
# and WHICH SIDE THAT FAVOURS DEPENDS ENTIRELY ON THE SHAPE. Measured on a
# Radeon RX 7900 XTX, fp32, median of repeated interleaved runs, as
# torch.fft time / rocfft_ops time (>1 means rocFFT wins) --
#
#     rfftn (2, 8, 65536)        dim=-1        0.28x   <- rocFFT 3.6x SLOWER
#     rfftn (16, 8, 65536)       dim=-1        0.72x   <- rocFFT slower
#     rfftn (4096, 8192)         dim=-1        1.23x
#     rfftn (32, 1024, 1024)     dim=(-2,-1)   1.06x
#     rfftn (2, 4, 96, 96, 96)   3D            1.21x
#     rfftn (1, 2, 160,160,160)  3D            9.34x   <- rocFFT much faster
#
# There is no clean rule here ("3D always wins" is FALSE -- 96^3 is 1.21x
# and 64^3 is 1.30x, while a large-batch 1D transform is a 3.6x regression),
# and routing everything to rocFFT unconditionally measurably slows
# `fft_conv1d` down. So the choice is made the way every other
# implementation choice in this package is made: per shape, by measuring
# once and caching the winner, via `kernel_select`. This is exactly the
# "actually measure a real win on real hardware first" precondition
# `rocfft_ops`' own docstring attaches to being wired in here.
#
# AUTOGRAD IS THE REASON THIS IS GUARDED RATHER THAN A PLAIN IMPORT SWAP.
# `fft_conv` below is differentiable by construction and this module's
# docstring depends on that ("no custom kernel, no `.detach()` anywhere ...
# it stays correct, gradient and all, whether or not the call is under
# `torch.no_grad()`"). `rocfft_ops.rfftn`/`irfftn` are raw ctypes calls into
# librocfft with no `torch.autograd.Function` wrapper, so a tensor produced
# by them carries no grad_fn: routing them in unconditionally would not
# error, it would silently return a graph-detached result and break backward
# through every FFT convolution. So the shims below hand back to torch.fft
# the moment a gradient could be involved. Everything else they check is a
# hard precondition of rocfft_ops' own API (CUDA-resident, float32/float64,
# trailing transform dims only) rather than a judgement call.
# ---------------------------------------------------------------------------
_ROCFFT_ENABLED = _env_flag("AMD_TUNED_TORCH_FFTCONV_ROCFFT", default="1")


def rocfft_enabled() -> bool:
    """True unless AMD_TUNED_TORCH_FFTCONV_ROCFFT=0. Read once at import
    time, same convention as `_FFTCONV1D_ENABLED` below."""
    return _ROCFFT_ENABLED


def _rocfft_usable(x: Tensor, dim: Tuple[int, ...], *, grad_operands=()) -> bool:
    """True if `rocfft_ops` can serve this exact transform with identical
    results AND without dropping a gradient on the floor."""
    if not (_ROCFFT_ENABLED and rocfft_ops.available()):
        return False
    if not x.is_cuda:
        return False
    # Any chance of a backward pass -> torch.fft, see the block comment.
    if torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (x,) + tuple(grad_operands)
    ):
        return False
    if x.requires_grad:
        return False
    # rocfft_ops transforms the trailing dims only.
    ndim = x.dim()
    return tuple(d % ndim for d in dim) == tuple(range(ndim - len(dim), ndim))


_VKFFT_CONTEST = _env_flag("AMD_TUNED_TORCH_FFTCONV_VKFFT", default="0")


def vkfft_contest_enabled() -> bool:
    """True only if AMD_TUNED_TORCH_FFTCONV_VKFFT is set on. OFF by default,
    unlike every other tier gate in this package, and the asymmetry is the
    point: VkFFT's OpenCL backend cannot address a torch tensor (see
    vkfft_ops' module docstring), so every candidate call copies the data
    device->host->device. Measured end to end that is 10-100x slower than
    rocFFT on every shape tried, so admitting it to the contest by default
    would buy nothing but measurement time, paid once per distinct shape.
    The flag exists so that claim can be re-measured on demand rather than
    taken on faith -- turn it on and kernel_select decides per shape."""
    return _VKFFT_CONTEST


def _vkfft_thunk(fn_name: str, *args, **kwargs):
    """A (name, thunk) entry for the contest, or None when the tier is off
    or unavailable. Declines (returns None) rather than raising on any
    failure, matching every other kernel_select candidate."""
    if not (_VKFFT_CONTEST and vkfft_ops.available()):
        return None

    def _thunk():
        try:
            return getattr(vkfft_ops, fn_name)(*args, **kwargs)
        except (RuntimeError, ValueError, TypeError):
            return None

    return ("vkfft", _thunk)


def _contest(kind: str, key, rocfft_thunk, torch_thunk, vkfft_entry=None):
    """Run `rocfft_thunk` vs `torch_thunk` through kernel_select's
    measure-once-cache-the-winner contest, falling back to torch.fft when
    the contest is disabled (AMD_TUNED_TORCH_MEASURE_KERNELS=0) or declines.

    torch.fft is the SECOND candidate deliberately: kernel_select verifies
    each candidate numerically against the first one, and torch.fft is the
    behaviour this module is required to preserve, so it must be the thing
    compared *to*, not the reference. Both call the same librocfft, and
    measured output is bit-identical, so this verification should never
    actually fire -- if it ever does, rocfft_ops is wrong for that shape and
    kernel_select excluding it permanently is the right outcome."""
    if not kernel_select.enabled():
        return torch_thunk()
    won = kernel_select.cached_key(kind, key)
    if won == "torch":
        return torch_thunk()
    if won == "rocfft":
        out = rocfft_thunk()
        if out is not None:
            return out
    elif won == "vkfft" and vkfft_entry is not None:
        out = vkfft_entry[1]()
        if out is not None:
            return out
    elif won is None:
        candidates = [("rocfft", rocfft_thunk)]
        if vkfft_entry is not None:
            candidates.append(vkfft_entry)
        # torch stays LAST: it is the reference every other candidate is
        # verified against (kernel_select._contest reads candidates[-1]).
        candidates.append(("torch", torch_thunk))
        out = kernel_select.pick_key(kind, key, candidates,
                                      tolerance=fftconv_tolerance(x_dtype(key)))
        if out is not None:
            return out
    return torch_thunk()


def x_dtype(key):
    """The dtype out of a contest key built by rfftn/irfftn below (both put
    it first), for picking the verification tolerance. VkFFT is a different
    FFT implementation than rocFFT rather than the same library reached two
    ways, so unlike the rocfft-vs-torch pair its output is close but not
    bit-identical -- which is exactly what fftconv_tolerance's looser bar is
    for."""
    return key[0] if key and isinstance(key[0], torch.dtype) else torch.float32


def rfftn(x: Tensor, dim: Tuple[int, ...]) -> Tensor:
    """`torch.fft.rfftn`, served by whichever of rocFFT-direct and torch.fft
    measures faster for this exact shape (see the block comment above).
    Signature is deliberately the narrow one this module actually calls, not
    torch.fft.rfftn's full one."""
    def _torch():
        return _torch_rfftn(x, dim=dim)

    if not (x.dtype in (torch.float32, torch.float64) and _rocfft_usable(x, dim)):
        return _torch()

    def _rocfft():
        try:
            return rocfft_ops.rfftn(x, dim=dim)
        except (rocfft_ops.RocfftError, TypeError, ValueError):
            return None

    return _contest("fftconv_rfftn", (x.dtype, tuple(x.shape), tuple(dim)),
                    _rocfft, _torch,
                    vkfft_entry=_vkfft_thunk("rfftn", x, dim=tuple(dim)))


def irfftn(x: Tensor, dim: Tuple[int, ...], s: Optional[Tuple[int, ...]] = None) -> Tensor:
    """`torch.fft.irfftn`, served by `rocfft_ops` when safe/applicable.

    torch.fft.irfftn infers the real output's last transform dimension as
    `2 * (n - 1)` when `s` is omitted; `rocfft_ops.irfftn` requires the real
    lengths explicitly (its plan needs them). Reproducing torch's inference
    here rather than passing the caller's own intended size keeps this a
    drop-in replacement -- for an odd-length signal those two differ, and
    silently "fixing" that would change this module's output."""
    if s is None:
        ndim = x.dim()
        dims = tuple(d % ndim for d in dim)
        s = tuple(x.shape[d] for d in dims[:-1]) + (2 * (x.shape[dims[-1]] - 1),)
    def _torch():
        return _torch_irfftn(x, s=s, dim=dim)

    if not (x.dtype in (torch.complex64, torch.complex128) and _rocfft_usable(x, dim)):
        return _torch()

    def _rocfft():
        try:
            return rocfft_ops.irfftn(x, s=s, dim=dim)
        except (rocfft_ops.RocfftError, TypeError, ValueError):
            return None

    return _contest("fftconv_irfftn", (x.dtype, tuple(x.shape), tuple(dim), tuple(s)),
                    _rocfft, _torch,
                    vkfft_entry=_vkfft_thunk("irfftn", x, s=tuple(s), dim=tuple(dim)))


# ---------------------------------------------------------------------------
# TRANSFORM-LENGTH PADDING -- making an ARBITRARY signal length fast, not
# merely legal.
#
# Non-power-of-2 lengths have always been CORRECT here: torch.fft/rocFFT
# factor any length into mixed-radix passes and fall back to Bluestein for
# a large prime factor, so `fft_conv` already accepted 63, 997, (31,17),
# (7,9,11) and so on with no special-casing. What they are not is FAST --
# a length whose factorisation rocFFT has no good radix kernel for costs
# several times a nearby smooth length on gfx1100 (16381 and 15625 both
# measured ~3-4x a padded 16384).
#
# ZERO-PADDING THE TRANSFORM UP TO A SMOOTHER LENGTH IS FREE OF SEMANTIC
# RISK, which is what makes this a legitimate thing to do behind the
# caller's back. `fft_conv` computes a CIRCULAR correlation of length N and
# keeps only outputs [0, signal_size - K], which are exactly the taps whose
# support never wraps; padding the signal with zeros to N' > N leaves every
# retained tap's support untouched (it only adds further valid taps past
# the crop) and `signal_size` is captured BEFORE any of this, so the crop is
# unmoved. Verified over 48 shape/stride/padding/groups combinations x 6 pad
# amounts: bit-for-bit within fp32 rounding of the unpadded result. The
# function already relied on this property in the small -- the pre-existing
# odd-length `% 2` pad below is the same trick with a fixed target of "even".
#
# WHICH smooth length is NOT a decision this module makes statically.
# Measurement on gfx1100 put no policy on top uniformly -- next-power-of-2
# won for 15625, 7-smooth for 16386, 5-smooth for 17011 -- and for lengths
# that are already smooth, padding is pure loss (a bigger transform for
# identical output). That is the same shape of answer that made rfftn/irfftn
# above a contest rather than a rule, so this is one too: the policies below
# are candidates, `_ORIGINAL_LEN` (pad nothing beyond the historical
# odd->even step) is the reference they must beat, and kernel_select
# measures once per shape and caches the winner. Set
# AMD_TUNED_TORCH_FFTCONV_SMOOTH_LEN=0 to skip the contest and always take
# the reference.
# ---------------------------------------------------------------------------
_FFTCONV_SMOOTH_LEN = _env_flag("AMD_TUNED_TORCH_FFTCONV_SMOOTH_LEN", default="1")

# Name -> allowed prime factors. 2 must be in every one of these: the
# next-power-of-2 upper bound `_next_smooth_len` starts its search from
# assumes it, and an odd target would fight the `% 2` even-length pad.
_SMOOTH_POLICIES = (
    ("pow2", (2,)),
    ("5smooth", (2, 3, 5)),
    ("7smooth", (2, 3, 5, 7)),
)
_ORIGINAL_LEN = "original"

# Below this many spatial positions, skip the contest and take the unpadded
# transform. The contest is not free -- it runs the convolution once per
# surviving policy to time it, plus once more for the reference to verify
# against -- and a transform this small is dominated by launch overhead that
# a smoother length does not remove, so there is nothing for the measurement
# to find. Same "don't contest what can't win" posture as
# _FFTCONV_CONV3D_MIN_POSITIONS in __init__.py, and likewise a guess rather
# than a measurement: override per workload if a shape proves otherwise.
_FFTCONV_SMOOTH_MIN_POSITIONS = int(
    os.environ.get("AMD_TUNED_TORCH_FFTCONV_SMOOTH_MIN_POSITIONS", "4096")
)


def _positions(sizes: Tuple[int, ...]) -> int:
    """Product of the transform extents -- the spatial volume one transform
    covers, ignoring batch and channels."""
    total = 1
    for s in sizes:
        total *= int(s)
    return total


def smooth_len_enabled() -> bool:
    """True unless AMD_TUNED_TORCH_FFTCONV_SMOOTH_LEN=0. Read once at import
    time, same posture as every other env-var gate in this module."""
    return _FFTCONV_SMOOTH_LEN


@lru_cache(maxsize=4096)
def _next_smooth_len(n: int, radices: Tuple[int, ...]) -> int:
    """Smallest `m >= n` whose prime factorisation uses only `radices`.

    Cached because it is called per conv call per spatial axis and the
    distinct (n, radices) pairs a process sees are few.

    The obvious "increment until smooth" loop is not viable here -- for
    radices=(2,) it walks ~16k integers to get from 16385 to 32768, on
    every call, in Python. This instead enumerates the smooth numbers
    directly by DFS over exponent tuples, seeded with the next power of two
    as an upper bound (always valid: 2 is in every policy) so branches that
    can no longer improve on the best candidate are cut."""
    if n <= 1:
        return 1
    best = 1 << (n - 1).bit_length()  # next power of 2 >= n; always smooth
    if radices == (2,):
        return best

    def walk(i: int, cur: int) -> None:
        nonlocal best
        if cur >= n:
            if cur < best:
                best = cur
            return
        if i == len(radices):
            return
        v = cur
        while v < best:
            walk(i + 1, v)
            v *= radices[i]

    walk(0, 1)
    return best


def _smooth_sizes(sizes: Tuple[int, ...], radices: Tuple[int, ...]) -> Tuple[int, ...]:
    """Per-axis `_next_smooth_len`. `sizes` are the transform extents of the
    already-padded signal, in tensor order."""
    return tuple(_next_smooth_len(int(s), radices) for s in sizes)


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
    """Grouped complex-valued frequency-domain multiply: for every frequency
    bin, `out[n, o] = sum_i a[n, i] * b[o, i]` summed over the input channels
    belonging to output channel `o`'s group. `a` is [N, Cin, *freq] out of
    rfftn, `b` is [Cout, Cin/groups, *freq]; the result is [N, Cout, *freq].

    WRITTEN AS AN EINSUM, NOT A RESHAPED `@`. Upstream (and this function
    until the layout was profiled) expresses the contraction by moving the
    channel axis to the end with `movedim`, adding a length-1 axis, and
    calling `@` -- which asks the BLAS backend for a batched GEMM over every
    frequency bin whose operands are non-contiguous views with a degenerate
    1-wide dimension. That is a pathological request: measured on gfx1100
    (complex64, the shapes fft_conv actually produces), it cost 27.7 ms for
    a dense 1D conv's [8,128] x [128,128] over 4609 bins and 21.0 ms for a
    2D conv's [8,64] x [64,64] over 12960 bins, where the same contraction
    written as an einsum over contiguous operands takes 3.5 ms and 2.7 ms
    -- 6-8x, for bit-identical output (same contraction order; verified
    exactly equal, not merely close). End to end that is fft_conv 25.8 ms
    -> 9.6 ms on the 2D shape and 52.3 ms -> 28.7 ms on the dense 1D one.
    The win is layout, not arithmetic: there are no fewer complex
    multiplies here than before.

    DEPTHWISE IS NOT A MATMUL AT ALL. When `groups == Cin` (Cin/groups == 1,
    Cout == groups -- the long depthwise conv1d that is this module's
    headline Hyena-style workload) the per-bin contraction is over a single
    input channel, i.e. a scalar complex product, and dispatching it as a
    batched GEMM over N*C*bins 1x1 matrices is the worst case of the
    paragraph above: 62.7 ms where the plain elementwise multiply below
    takes 0.78 ms (80x), taking fft_conv itself from 70.9 ms to 20.4 ms on
    B=8, C=256, L=16384, K=2048. Whether that is enough to beat whatever
    conv1d tier this shape would otherwise land on is still kernel_select's
    measurement to make, not this docstring's claim -- against unpatched
    F.conv1d the same shape is roughly a tie, and the point here is that
    the tier is no longer losing 3.5x of its own time to a layout mistake
    before that contest even starts.

    Both branches are ordinary differentiable primitives (`einsum`, `*`),
    so fft_conv's "real autograd, no custom kernel anywhere" property (see
    the module docstring) is unchanged.

    FIXES A GROUPED-CONV CRASH. The old `@` formulation ended with
    `c.view(...)` over a tensor `movedim` had just made non-contiguous,
    which raised `RuntimeError: view size is not compatible with input
    tensor's size and stride` for every grouped conv with `groups > 1` and
    `Cout/groups > 1` (depthwise survived only because its trailing axis is
    1 wide). That reached the public `fft_conv1d`/`2d`/`3d` and the
    auto-patch candidates, neither of which filters `groups` -- see
    tests/test_fftconv_ops.py::TestGroupedConv."""
    n, cin = a.shape[0], a.shape[1]
    cout, freq = b.shape[0], a.shape[2:]
    a = a.reshape(n, cin, -1)
    b = b.reshape(cout, b.shape[1], -1)

    if b.shape[1] == 1 and cout == groups:
        # Depthwise: one input channel per output channel, no sum to do.
        out = a * b.squeeze(1).unsqueeze(0)
    else:
        out = torch.einsum(
            "ngif,goif->ngof",
            a.view(n, groups, cin // groups, -1),
            b.view(groups, cout // groups, b.shape[1], -1),
        ).reshape(n, cout, -1)

    return out.reshape(n, cout, *freq)


def _length_contest(run, signal: Tensor, kernel: Tensor,
                    base_sizes: Tuple[int, ...], groups: int,
                    stride_: Tuple[int, ...]) -> Tensor:
    """Pick the transform extents for this shape by measuring, and cache the
    winner -- `run(fft_sizes)` computes the whole convolution at those
    extents. See the TRANSFORM-LENGTH PADDING comment above for why every
    candidate here is numerically interchangeable and why none of them can
    be chosen statically.

    `_ORIGINAL_LEN` is deliberately LAST: kernel_select verifies candidates
    against the final entry (`_contest`'s `reference_name = candidates[-1]`),
    and the unpadded transform is the behaviour this function is required to
    preserve, so it is the thing compared *to*. Verification runs at
    `fftconv_tolerance`'s looser fftconv bar rather than kernel_select's
    dtype defaults: a different transform length rounds differently, so the
    padded variants are close to the reference but not bit-identical to it,
    which is exactly the case that bar exists for.

    Hands straight back to the unpadded path -- building no thunks and
    touching no cache -- when the contest is off, when every policy is a
    no-op for these extents (all axes already smooth, the common case for
    power-of-2 workloads), or when the signal is small enough that the
    contest would cost more than it could ever recover."""
    if not (_FFTCONV_SMOOTH_LEN and kernel_select.enabled()):
        return run(base_sizes)
    # A gradient-carrying call must not be routed through the contest: it
    # would run the convolution several times to time it, building (and
    # discarding) a graph per candidate. Correct, but pointless work.
    if torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (signal, kernel)
    ):
        return run(base_sizes)
    if _positions(base_sizes) < _FFTCONV_SMOOTH_MIN_POSITIONS:
        return run(base_sizes)

    sizes = {}
    for name, radices in _SMOOTH_POLICIES:
        cand = _smooth_sizes(base_sizes, radices)
        if cand != base_sizes and cand not in sizes.values():
            sizes[name] = cand
    if not sizes:
        return run(base_sizes)  # already smooth on every axis; nothing to try

    key = (signal.dtype, tuple(signal.shape), tuple(kernel.shape),
           groups, tuple(stride_), tuple(base_sizes))

    won = kernel_select.cached_key("fftconv_len", key)
    if won is not None:
        return run(sizes.get(won, base_sizes))

    candidates = [(n, (lambda sz=sz: run(sz))) for n, sz in sizes.items()]
    candidates.append((_ORIGINAL_LEN, lambda: run(base_sizes)))
    out = kernel_select.pick_key("fftconv_len", key, candidates,
                                  tolerance=fftconv_tolerance(signal.dtype))
    return out if out is not None else run(base_sizes)


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

    # --- A kernel wider than the (already padded, already dilation-expanded)
    # signal has no valid output position at all. Everything below would
    # still "work" -- F.pad accepts the negative pad it implies and silently
    # TRUNCATES the kernel, and the crop's `signal_size - kernel + 1` goes
    # <= 0 and slices to empty -- so the call used to return a wrongly-shaped
    # tensor computed from a mutilated kernel instead of failing. F.convNd
    # rejects this outright; so does this now, with the same complaint.
    invalid = [
        (i - 2, int(signal_size[i]), kernel.size(i))
        for i in range(2, signal.ndim)
        if kernel.size(i) > signal_size[i]
    ]
    if invalid:
        got = " x ".join(str(int(signal_size[i])) for i in range(2, signal.ndim))
        want = " x ".join(str(kernel.size(i)) for i in range(2, signal.ndim))
        raise ValueError(
            f"Calculated padded input size per channel: ({got}). "
            f"Kernel size: ({want}). Kernel size can't be greater than "
            f"actual input size (axis {invalid[0][0]})."
        )

    if signal.size(-1) % 2 != 0:
        signal = F.pad(signal, [0, 1])

    # --- Transform extents. `_ORIGINAL_LEN` is what this function has always
    # used (the signal as padded above); the smooth-length policies enlarge it
    # further. See the TRANSFORM-LENGTH PADDING comment for why enlarging is
    # semantically free and why the choice is measured rather than fixed.
    base_sizes = tuple(signal.size(i) for i in range(2, signal.ndim))

    def _run(fft_sizes: Tuple[int, ...]) -> Tensor:
        sig = signal
        if fft_sizes != base_sizes:
            grow = [
                pad
                for i in reversed(range(len(fft_sizes)))
                for pad in [0, fft_sizes[i] - base_sizes[i]]
            ]
            sig = F.pad(sig, grow)

        kernel_padding = [
            pad
            for i in reversed(range(2, sig.ndim))
            for pad in [0, sig.size(i) - kernel.size(i)]
        ]
        padded_kernel = F.pad(kernel, kernel_padding)

        # --- fp32 transform + pointwise multiply: the only part of this
        # function where precision actually matters, and the only part
        # torch.fft requires it for regardless of `signal`'s own dtype.
        signal_fr = rfftn(sig.float(), dim=tuple(range(2, sig.ndim)))
        kernel_fr = rfftn(padded_kernel.float(), dim=tuple(range(2, sig.ndim)))

        kernel_fr.imag *= -1
        output_fr = complex_matmul(signal_fr, kernel_fr, groups=groups)
        out = irfftn(output_fr, dim=tuple(range(2, sig.ndim)))

        # --- back to fp16/bf16 storage immediately: the crop and bias-add
        # below are again pure data movement over what's now an
        # activation-sized tensor, not a reason to keep paying for float32.
        out = out.to(out_dtype)

        crop_slices = (slice(None), slice(None)) + tuple(
            slice(0, (signal_size[i] - kernel.size(i) + 1), stride_[i - 2])
            for i in range(2, sig.ndim)
        )
        out = out[crop_slices].contiguous()

        if bias is not None:
            bias_shape = tuple([1, -1] + (sig.ndim - 2) * [1])
            out = out + bias.to(out_dtype).view(bias_shape)

        return out

    return _length_contest(_run, signal, kernel, base_sizes, groups, stride_)


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


def fft_convnd(signal: Tensor, kernel: Tensor, bias: Optional[Tensor] = None,
                padding: Union[int, Iterable[int], str] = 0, padding_mode: str = "constant",
                stride: Union[int, Iterable[int]] = 1, dilation: Union[int, Iterable[int]] = 1,
                groups: int = 1) -> Tensor:
    """F.convNd-equivalent via FFT for ANY number of spatial axes -- `signal`
    [B,Ci,*spatial], `kernel` [Co,Ci/groups,*kernel_spatial], N inferred as
    `signal.dim() - 2`.

    `fft_conv` was always rank-generic; what did not exist was a public entry
    point saying so. The fixed-rank `fft_conv1d`/`2d`/`3d` above stay as they
    are -- their strict 3D/4D/5D checks are a deliberate F.convNd-shaped
    contract, and catching a misranked tensor at the call site beats letting
    it become a differently-misranked one four frames down -- but they cover
    only the ranks torch itself has a conv for. This covers the rest, which
    for an FFT convolution costs nothing extra: there is no direct-convolution
    kernel behind it whose absence would be the real limit, just an rfftn over
    however many axes were handed over (verified against a 4-spatial-axis
    conv, which `F.conv4d` does not exist to check against).

    Requires at least one spatial axis, the one thing `fft_conv` genuinely
    cannot do -- a rank-2 [B,C] input leaves rfftn no axis to transform and
    fails inside torch.fft with a message about the transform rather than
    about the argument."""
    if signal.dim() < 3:
        raise ValueError(
            f"fft_convnd expects a [B,C,*spatial] signal with at least one "
            f"spatial axis, got a {signal.dim()}D tensor."
        )
    if kernel.dim() != signal.dim():
        raise ValueError(
            f"fft_convnd expects signal and kernel of equal rank, got "
            f"{signal.dim()}D signal and {kernel.dim()}D kernel."
        )
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
    return fftconv_candidate(input, weight, bias, stride, padding, dilation, groups)


def fftconv_candidate(input: Tensor, weight: Tensor, bias: Optional[Tensor],
                       stride, padding, dilation, groups: int) -> Optional[Tensor]:
    """Rank-generic `fftconv1d_candidate`: same decline-don't-raise contract,
    for a conv of any spatial rank rather than 1D only.

    `fftconv1d_candidate` above is the conv1d auto-patch's entry point and
    keeps its own 3D check (miopen_fallback hands it conv1d tensors and a
    scalar stride/padding/dilation, and a candidate that quietly accepted a
    4D tensor there would be papering over a caller bug); this is what a
    caller with a conv2d/conv3d/N-D shape contests with. `stride`/`padding`/
    `dilation` may be scalars or per-axis sequences -- `fft_conv` runs them
    through `to_ntuple` either way.

    None if the ranks disagree, if the kernel is below
    AMD_TUNED_TORCH_FFTCONV1D_MIN_KERNEL on every spatial axis (the same
    cheap pre-filter, applied to the widest axis so an asymmetric kernel
    like 1x64 is not dismissed for its narrow one), or if the call fails
    for any reason -- including the kernel-wider-than-input case `fft_conv`
    now rejects outright."""
    if input.dim() < 3 or weight.dim() != input.dim():
        return None
    if max(weight.shape[2:]) < _FFTCONV1D_MIN_KERNEL:
        return None
    try:
        return fft_conv(input, weight, bias=bias, padding=padding,
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
    if input.dim() != 3 or weight.dim() != 3:
        return None
    return maybe_fft_conv(input, weight, bias=bias, stride=stride, padding=padding,
                           dilation=dilation, groups=groups, min_kernel=min_kernel)


def maybe_fft_conv(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
                    stride: Tuple[int, ...] = (1,), padding: Tuple[int, ...] = (0,),
                    dilation: Tuple[int, ...] = (1,), groups: int = 1,
                    min_kernel: Optional[int] = None) -> Optional[Tensor]:
    """Rank-generic `maybe_fft_conv1d` -- same static `min_kernel` heuristic
    and same None-on-decline contract, for a conv of any spatial rank.

    The threshold is compared against the WIDEST spatial axis of `weight`,
    not the last one: for a 1D kernel those are the same axis, and for a 2D/
    3D kernel the widest is the one that decides whether FFT-conv's O(N log N)
    argument has anything to work with. Note the default of 128 is 1D's
    crossover -- 2D and 3D cross over much earlier (see
    `_FFTCONV_CONV2D_MIN_KERNEL`=32 / `_FFTCONV_CONV3D_MIN_KERNEL`=7 in
    __init__.py, both measured on gfx1100), so a 2D/3D caller relying on this
    function's static route rather than a measured contest should pass its
    own `min_kernel` rather than inherit 1D's.

    `stride`/`padding`/`dilation` are per-axis sequences here, matching
    F.convNd's own normalised form; a scalar works too."""
    if not fft_conv1d_enabled():
        return None
    if input.dim() < 3 or weight.dim() != input.dim():
        return None
    threshold = 128 if min_kernel is None else min_kernel
    if max(weight.shape[2:]) < threshold:
        return None
    try:
        return fft_conv(input, weight, bias=bias, padding=padding,
                         stride=stride, dilation=dilation, groups=groups)
    except (RuntimeError, ValueError, TypeError):
        return None
