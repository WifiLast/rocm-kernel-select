"""TeaCache -- a generic, model-agnostic port of the skip/cache decision
engine from Liu et al., "Timestep Embedding Tells: It's Time to Cache for
Video Diffusion Model" (arXiv:2411.19108), source/TeaCache.

Same family of technique as amd_tuned_torch.magcache (see that module first) --
skip a denoising step's full transformer-block-stack pass and reuse the
last real output residual instead of recomputing it, whenever the step is
predicted to change the output negligibly. The two differ in what signal
drives that prediction:

  - magcache: a precomputed, content-independent per-step lookup table
    (mag_ratios[step]) -- purely a function of which step you're on.
  - teacache (this module): a live, content-dependent signal computed
    fresh every step from the model's own timestep-embedding modulation --
    the relative L1 distance between this step's and the previous step's
    "modulated input" (e.g. transformer_blocks[0].norm1(hidden_states,
    emb=temb) in a diffusers-style DiT), passed through a per-model
    calibrated polynomial rescale (upstream fits this with np.poly1d;
    ported here as plain Horner evaluation to avoid a numpy dependency).

Like magcache, this has nothing to do with kernel dispatch and needs no
aiter/TransformerEngine/ROCm -- pure tensor arithmetic around whatever
block loop and modulation step the caller already has, so it works on any
device, any backend.

Upstream ships this as several hundred near-identical lines *per model*
(TeaCache4FLUX/teacache_flux.py, TeaCache4Wan2.1/teacache_generate.py,
TeaCache4HunyuanVideo/teacache_sample_video.py, ...), each hand-copying the
same cnt/accumulated_rel_l1_distance/previous_modulated_input/
previous_residual state machine into that model's own forward(). What
differs per model is only how to compute the cheap "modulated input" probe
and where to splice the skip/no-skip branch into the block loop -- the
decision engine itself never touches a model class or block type. This
module extracts exactly that engine.

Usage (replaces the pattern in e.g. teacache_flux.py's teacache_forward):

    import amd_tuned_torch

    teacache = amd_tuned_torch.teacache.TeaCache(
        num_steps=28, rel_l1_thresh=0.6, coefficients=my_calibrated_coefficients,
    )

    for step in range(num_inference_steps):
        modulated_input, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            transformer_blocks[0].norm1(hidden_states, emb=temb)  # your model's own first-block modulation
        )
        if teacache.should_skip(modulated_input):
            hidden_states = hidden_states + teacache.cached_residual
        else:
            ori_hidden_states = hidden_states
            hidden_states = run_transformer_blocks(hidden_states, ...)  # your model's own block loop
            teacache.record(hidden_states - ori_hidden_states)
        teacache.advance()

`coefficients` is a per-model calibrated polynomial (upstream publishes one
per supported model, e.g. FLUX's is hardcoded in
TeaCache4FLUX/teacache_flux.py as
[4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00,
2.64230861e-01]) -- there is no universal one. Leave it as None to use the
raw relative-L1 distance directly, unscaled, if you haven't calibrated one
for your model.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch


def evaluate_polynomial(coefficients: Sequence[float], x: float) -> float:
    """Horner's method, matching numpy.poly1d's coefficient convention
    (highest-degree term first) without needing numpy as a dependency for
    what's otherwise a handful of floating point multiplies."""
    result = 0.0
    for c in coefficients:
        result = result * x + c
    return result


class TeaCache:
    """The inference-time skip/cache decision engine.

    num_steps: total denoising steps in one generation run. `advance()`
        wraps the internal step counter once this many steps have passed,
        so a single TeaCache instance can be reused across repeated
        generations without reconstructing it each time.
    rel_l1_thresh: max allowed accumulated (rescaled) relative-L1 distance
        across consecutive skipped steps before forcing a real recompute.
        Lower = more conservative (fewer skips, closer to full quality).
        Upstream's published per-model values trade this off explicitly --
        e.g. FLUX's README lists 0.25/0.4/0.6/0.8 for roughly
        1.5x/1.8x/2.0x/2.25x speedups.
    coefficients: a per-model calibrated polynomial (highest-degree
        coefficient first, same convention as numpy.poly1d) that rescales
        the raw relative-L1 distance into a value that actually tracks
        output error -- the raw distance alone doesn't correlate linearly
        with how much the final output changes, which is why upstream fits
        this per model rather than using the raw distance directly. None
        (the default) skips rescaling and uses the raw distance as-is.

    The first and last step of every num_steps-length run always compute
    in full (never skipped) -- matching upstream exactly, not a tunable
    here: the first step has no previous modulated input to compare
    against, and forcing the last step keeps final quality from drifting
    right before the output is used.
    """

    def __init__(
        self,
        num_steps: int,
        rel_l1_thresh: float = 0.6,
        coefficients: Optional[Sequence[float]] = None,
    ):
        self.num_steps = num_steps
        self.rel_l1_thresh = rel_l1_thresh
        self.coefficients = coefficients
        self.cnt = 0
        self.accumulated_rel_l1_distance = 0.0
        self.previous_modulated_input: Optional[torch.Tensor] = None
        self.cached_residual: Optional[torch.Tensor] = None

    def _rescale(self, raw_distance: float) -> float:
        if self.coefficients is None:
            return raw_distance
        return evaluate_polynomial(self.coefficients, raw_distance)

    def should_skip(self, modulated_input: torch.Tensor) -> bool:
        """Call once per denoising step, before running the block stack,
        with the cheap per-step probe signal (e.g. the first block's
        AdaLN-modulated input) computed by the caller. True means: don't
        run the block stack, reuse cached_residual instead.

        Always records modulated_input as the reference for next step's
        comparison, whether or not this step is skipped -- the probe is
        cheap to compute every step regardless of whether the expensive
        block stack runs, unlike the residual itself (only updated via
        record(), which only happens on non-skipped steps).
        """
        skip = False
        if self.cnt == 0 or self.cnt == self.num_steps - 1:
            # Force a real compute at the first step (nothing to compare
            # against yet) and the last step (keep final quality from
            # drifting right before the output is used) -- also resets the
            # accumulator, same as upstream.
            self.accumulated_rel_l1_distance = 0.0
        else:
            # previous_modulated_input is guaranteed set here: cnt reaches
            # this branch only after cnt==0 has already run once (which
            # always sets it below), for any num_steps >= 2.
            raw_distance = (
                (modulated_input - self.previous_modulated_input).abs().mean()
                / self.previous_modulated_input.abs().mean()
            ).item()
            self.accumulated_rel_l1_distance += self._rescale(raw_distance)
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                skip = True
            else:
                self.accumulated_rel_l1_distance = 0.0
        self.previous_modulated_input = modulated_input
        return skip and self.cached_residual is not None

    def record(self, residual: torch.Tensor) -> None:
        """Call after actually running the block stack (i.e. whenever
        should_skip() returned False) with its output residual. Do not
        call this on a skipped step; cached_residual is being reused
        unchanged, there is no new residual to record."""
        self.cached_residual = residual

    def advance(self) -> None:
        """Call once per step, after should_skip()/record() -- advances the
        step counter and, once num_steps is reached, resets it for the next
        generation run. cached_residual and previous_modulated_input are
        left alone: harmless, since should_skip() forces a real recompute
        at cnt==0 regardless, which will replace both before either is read
        again."""
        self.cnt += 1
        if self.cnt >= self.num_steps:
            self.cnt = 0
            self.accumulated_rel_l1_distance = 0.0
