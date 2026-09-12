"""Reference implementation of the keyframe-residual image-set scheme
(see ``plan/spartialal_2d.txt``).

RESEARCH PROTOTYPE, not a dispatch tier. Nothing here is patched into
``torch``/``torch.nn.functional`` by ``amd_tuned_torch.enable()``, nothing here
touches a native extension, and every function is plain PyTorch that runs
on CPU. The point is to make the plan's claims *measurable* before any
HIP/Triton kernel work starts: the plan's sections 9-11 and 15 all hinge
on numbers (how sparse is the residual, really?) that can be obtained
without a GPU and without training anything.

The scheme: given a set of N correlated images, pick one as the keyframe
``K`` and represent every other image as a residual against it,

    D(n) = I(n) - K            (anchored -- the default for image sets)
    D(t) = F(t) - F(t-1)       (chained  -- the video special case, section 19)

then run the expensive convolution once on ``K`` and only cheap updates
on the residuals:

    Z(n) = Conv(K) + Conv(D(n))

That identity is exact for a convolution *without* bias, because
convolution is linear: ``Conv(K + D) == Conv(K) + Conv(D)``. With a bias
it holds only if the bias is added exactly once -- adding it on both the
base path and the residual path double-counts it. This is the single
most important correctness property of the whole approach and the reason
``keyframe_residual_conv2d`` below passes ``bias=None`` to the residual
convolution. See ``tests/test_keyframe_residual.py``.

What is deliberately NOT here: any learned encoder (plan section 6), any
training loop (section 16), and any fused kernel (section 13). The
feature-space variant needs a trained encoder to say anything
meaningful; the pixel-space variant below is what tells you whether it
is worth training one.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "anchored_residuals",
    "chained_residuals",
    "reconstruct_anchored",
    "reconstruct_chained",
    "threshold_residuals",
    "residual_stats",
    "to_spatial_index",
    "from_spatial_index",
    "per_image_conv2d",
    "keyframe_residual_conv2d",
    "estimate_global_shift",
    "align_to_keyframe",
    "make_synthetic_set",
]


# --------------------------------------------------------------------------
# Section 3 -- keyframe + residual
# --------------------------------------------------------------------------

def anchored_residuals(stack: torch.Tensor, keyframe_index: int = 0
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split ``stack`` [N, C, H, W] into (keyframe [C, H, W], residuals [N, C, H, W]).

    ``residuals[n] = stack[n] - keyframe`` for every n, including the
    keyframe's own slot, which is therefore exactly zero. Keeping the
    full N rows means residual index == image index everywhere, which is
    what makes ``reconstruct_anchored`` a one-liner; the zero row costs
    nothing because every consumer here skips it explicitly (see
    ``keyframe_residual_conv2d``).
    """
    if stack.dim() != 4:
        raise ValueError(f"expected a 4D [N, C, H, W] stack, got shape {tuple(stack.shape)}")
    if not -stack.shape[0] <= keyframe_index < stack.shape[0]:
        raise IndexError(f"keyframe_index {keyframe_index} out of range for N={stack.shape[0]}")
    keyframe = stack[keyframe_index]
    return keyframe, stack - keyframe.unsqueeze(0)


def chained_residuals(stack: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split ``stack`` into (first image, [N, C, H, W] of successive differences).

    ``residuals[0]`` is zero and ``residuals[t] = stack[t] - stack[t-1]``.
    This is the video form (plan section 19) and is kept here as the
    comparison arm for the anchored form: it produces smaller residuals
    on an ordered set but forces a sequential reconstruction and lets
    error accumulate along the chain.
    """
    if stack.dim() != 4:
        raise ValueError(f"expected a 4D [N, C, H, W] stack, got shape {tuple(stack.shape)}")
    residuals = torch.zeros_like(stack)
    residuals[1:] = stack[1:] - stack[:-1]
    return stack[0], residuals


def reconstruct_anchored(keyframe: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`anchored_residuals`."""
    return keyframe.unsqueeze(0) + residuals


def reconstruct_chained(keyframe: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`chained_residuals` -- a cumulative sum, hence the drift."""
    return keyframe.unsqueeze(0) + torch.cumsum(residuals, dim=0)


# --------------------------------------------------------------------------
# Sections 9/10 -- thresholding and residual statistics
# --------------------------------------------------------------------------

def threshold_residuals(residuals: torch.Tensor, threshold: float) -> torch.Tensor:
    """Zero every residual element with ``abs(x) < threshold`` (hard threshold).

    Hard, not soft: elements above the threshold keep their exact value,
    so the reconstruction error of any surviving element is 0 and the
    worst-case error of a dropped one is bounded by ``threshold``. A soft
    threshold would shrink every element and blur that bound, which
    makes the error analysis in the tests far less sharp.
    """
    if threshold <= 0:
        return residuals
    return torch.where(residuals.abs() < threshold, torch.zeros_like(residuals), residuals)


def residual_stats(residuals: torch.Tensor, threshold: float = 1e-3) -> Dict[str, float]:
    """The section 9 measurements for a residual tensor.

    ``clustering`` is the fraction of above-threshold elements that have
    at least one other above-threshold element in their 8-neighbourhood.
    It matters as much as the raw count: scattered non-zeros defeat
    block-sparse kernels, while non-zeros concentrated in a few regions
    can be processed as dense tiles.
    """
    flat = residuals.detach().float()
    magnitude = flat.abs()
    mask = (magnitude >= threshold).float()
    occupancy = mask.mean().item()

    if mask.dim() == 4:
        # A 3x3 sum-pool counts the centre itself, so subtract it back out:
        # what we want is "does this non-zero have a non-zero NEIGHBOUR".
        counts = F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) * 9.0
        has_neighbour = ((counts - mask) >= 1.0).float() * mask
        clustering = (has_neighbour.sum() / mask.sum()).item() if mask.sum() > 0 else 0.0
    else:
        clustering = float("nan")

    return {
        "near_zero_fraction": 1.0 - occupancy,
        "occupancy": occupancy,
        "mean_abs": magnitude.mean().item(),
        "max_abs": magnitude.max().item(),
        "var": flat.var(unbiased=False).item(),
        "clustering": clustering,
    }


# --------------------------------------------------------------------------
# Sections 2/4 -- the [Spatial, N] representation
# --------------------------------------------------------------------------

def to_spatial_index(stack: torch.Tensor, axis: str = "W") -> torch.Tensor:
    """Rearrange [N, C, H, W] into the plan's ``spatial x image-index`` 2D layout.

    ``axis="W"`` keeps width as the spatial axis and folds height into
    the batch dimension, giving [H, C, W, N]; ``axis="H"`` does the
    mirror, giving [W, C, H, N]. A plain Conv2D over the result then has
    one spatial kernel dimension and one image-index kernel dimension,
    which is the layer the whole plan is built around.
    """
    if stack.dim() != 4:
        raise ValueError(f"expected a 4D [N, C, H, W] stack, got shape {tuple(stack.shape)}")
    if axis == "W":
        return stack.permute(2, 1, 3, 0).contiguous()   # [H, C, W, N]
    if axis == "H":
        return stack.permute(3, 1, 2, 0).contiguous()   # [W, C, H, N]
    raise ValueError(f"axis must be 'W' or 'H', got {axis!r}")


def from_spatial_index(view: torch.Tensor, axis: str = "W") -> torch.Tensor:
    """Inverse of :func:`to_spatial_index`."""
    if axis == "W":
        return view.permute(3, 1, 0, 2).contiguous()    # [H, C, W, N] -> [N, C, H, W]
    if axis == "H":
        return view.permute(3, 1, 2, 0).contiguous()    # [W, C, H, N] -> [N, C, H, W]
    raise ValueError(f"axis must be 'W' or 'H', got {axis!r}")


# --------------------------------------------------------------------------
# Section 11 -- the baseline and the residual path
# --------------------------------------------------------------------------

def per_image_conv2d(stack: torch.Tensor, weight: torch.Tensor,
                     bias: torch.Tensor = None, **kwargs) -> torch.Tensor:
    """Baseline 1: convolve every image of the set independently.

    This is what anyone would write today and it is already well tuned
    on ROCm, so it -- not Conv3D -- is the bar the residual path has to
    clear.
    """
    return F.conv2d(stack, weight, bias, **kwargs)


def keyframe_residual_conv2d(stack: torch.Tensor, weight: torch.Tensor,
                             bias: torch.Tensor = None, keyframe_index: int = 0,
                             threshold: float = 0.0, **kwargs) -> torch.Tensor:
    """Convolve the set as ``Conv(K)`` once plus ``Conv(D(n))`` per image.

    With ``threshold == 0`` this is *exactly* equivalent to
    :func:`per_image_conv2d` up to floating-point rounding -- convolution
    is linear, so nothing is approximated. A non-zero ``threshold``
    (plan section 10, arm C) is the only thing that introduces error.

    The bias is added on the base path only. Adding it again on the
    residual path would double-count it for every non-keyframe image,
    which is the easiest way to get this scheme subtly wrong.
    """
    keyframe, residuals = anchored_residuals(stack, keyframe_index)
    residuals = threshold_residuals(residuals, threshold)

    base = F.conv2d(keyframe.unsqueeze(0), weight, bias, **kwargs)    # bias here, once

    n = stack.shape[0]
    keyframe_index %= n
    others = [i for i in range(n) if i != keyframe_index]
    out = base.expand(n, *base.shape[1:]).clone()
    if others:
        index = torch.tensor(others, device=stack.device)
        updates = F.conv2d(residuals[index], weight, None, **kwargs)  # no bias here
        out[index] = base + updates
    return out


# --------------------------------------------------------------------------
# Section 15 -- alignment
# --------------------------------------------------------------------------

def estimate_global_shift(reference: torch.Tensor, image: torch.Tensor) -> Tuple[int, int]:
    """Integer (dy, dx) such that ``image ~= roll(reference, (dy, dx))``.

    Phase correlation -- level B of the plan's alignment ladder, the
    cheapest thing that does anything useful for a hand-held burst. Both
    inputs are [H, W]; multi-channel callers should pass a single
    channel or the channel mean.
    """
    if reference.dim() != 2 or image.dim() != 2:
        raise ValueError("estimate_global_shift expects 2D [H, W] tensors")
    height, width = reference.shape
    ref_spectrum = torch.fft.rfft2(reference.float())
    img_spectrum = torch.fft.rfft2(image.float())
    cross = ref_spectrum.conj() * img_spectrum
    cross = cross / (cross.abs() + 1e-8)
    correlation = torch.fft.irfft2(cross, s=(height, width))

    peak = int(torch.argmax(correlation))
    dy, dx = divmod(peak, width)
    if dy > height // 2:
        dy -= height
    if dx > width // 2:
        dx -= width
    return dy, dx


def align_to_keyframe(stack: torch.Tensor, keyframe_index: int = 0) -> torch.Tensor:
    """Undo each image's estimated global translation against the keyframe.

    Uses ``torch.roll``, so content leaving one edge wraps around to the
    other. That is wrong at the borders but keeps the operation exact and
    shape-preserving; a real implementation would crop to the valid
    region, at the cost of a shrinking canvas across the set.
    """
    keyframe = stack[keyframe_index]
    reference = keyframe.mean(dim=0)
    aligned = stack.clone()
    for n in range(stack.shape[0]):
        if n == keyframe_index % stack.shape[0]:
            continue
        dy, dx = estimate_global_shift(reference, stack[n].mean(dim=0))
        aligned[n] = torch.roll(stack[n], shifts=(-dy, -dx), dims=(-2, -1))
    return aligned


# --------------------------------------------------------------------------
# Section 1 -- synthetic image sets, one per domain
# --------------------------------------------------------------------------

def make_synthetic_set(kind: str, n: int = 8, channels: int = 3, height: int = 64,
                       width: int = 64, seed: int = 0,
                       device=None, dtype=torch.float32) -> torch.Tensor:
    """A synthetic [N, C, H, W] set standing in for one domain of section 1.

    Real data is the only thing that settles the plan's hypothesis; these
    are for testing the *machinery* and for giving each measurement a
    known-correct expectation to be checked against. ``"unrelated"`` is
    the control: independent noise, where the method must show no
    advantage at all.

        burst      base scene + small sensor noise
        bracket    base scene * a global gain per image
        multiview  base scene shifted by a few pixels per image
        unrelated  independent random images
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.rand(channels, height, width, generator=generator)
    # A few smooth blobs, so the "scene" has structure rather than being
    # white noise -- clustering statistics are meaningless otherwise.
    base = F.avg_pool2d(base.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)

    if kind == "burst":
        noise = 0.01 * torch.randn(n, channels, height, width, generator=generator)
        stack = base.unsqueeze(0) + noise
        stack[0] = base
    elif kind == "bracket":
        gains = torch.linspace(1.0, 1.6, n).view(n, 1, 1, 1)
        stack = base.unsqueeze(0) * gains
    elif kind == "multiview":
        stack = torch.stack([torch.roll(base, shifts=(0, 2 * i), dims=(-2, -1))
                             for i in range(n)])
    elif kind == "unrelated":
        stack = torch.rand(n, channels, height, width, generator=generator)
        stack = F.avg_pool2d(stack, kernel_size=5, stride=1, padding=2)
    else:
        raise ValueError(f"unknown kind {kind!r}")

    return stack.to(device=device, dtype=dtype)
