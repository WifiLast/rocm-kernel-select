"""MagCache -- a generic, model-agnostic port of the skip/cache decision
engine from Zehong Ma et al., "MagCache: Fast Video Generation with
Magnitude-Aware Cache" (arXiv:2506.09045, NeurIPS 2025), source/MagCache.

MagCache accelerates diffusion transformer inference (Flux, Wan, Hunyuan
Video, QwenImage, FramePack, ...) by skipping a denoising step's full
transformer-block-stack pass whenever the *magnitude* of that step's output
residual is predictable from a per-step calibration table -- reusing the
last real residual instead of recomputing it. This has nothing to do with
kernel dispatch (unlike every other module in amd_tuned_torch) and needs no aiter/
TransformerEngine/ROCm at all: it's pure tensor arithmetic around whatever
block loop the caller already has, so it works on any device, any backend.

Upstream ships this as several hundred near-identical lines per model
(MagCache4FLUX/magcache_flux.py, MagCache4Wan2.1/magcache_generate.py,
MagCache4HunyuanVideo/magcache_sample_video.py, ...), each hand-copying the
same cnt/accumulated_ratio/accumulated_err/accumulated_steps state machine
into that model's own forward() method. What's actually model-specific is
only "where do I call the block stack and what counts as its input/output
residual" -- the skip/cache *decision* itself never touches a model class,
block type, or tensor shape beyond a plain elementwise subtraction. This
module extracts exactly that decision engine so it can be shared instead of
re-copied: wrap your own block-loop call with should_skip()/record()/
advance() instead of forking one of upstream's per-model files.

Usage (replaces the pattern in e.g. magcache_flux.py's magcache_forward):

    import amd_tuned_torch

    magcache = amd_tuned_torch.magcache.MagCache(num_steps=28, mag_ratios=my_calibrated_table)

    for step in range(num_inference_steps):
        if magcache.should_skip():
            hidden_states = hidden_states + magcache.cached_residual
        else:
            ori_hidden_states = hidden_states
            hidden_states = run_transformer_blocks(hidden_states, ...)  # your model's own block loop
            magcache.record(hidden_states - ori_hidden_states)
        magcache.advance()

`mag_ratios` is a per-step calibration table -- there is no universal one,
it depends on the model. Get it either from upstream's published tables
for a supported model (e.g. FLUX's is hardcoded in
MagCache4FLUX/magcache_flux.py) or by running MagCacheCalibrator (below)
once over a representative generation with your own model, and hardcoding
the result the same way upstream does.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F


def nearest_interp(src_ratios: Sequence[float], target_length: int) -> List[float]:
    """Nearest-neighbor resample a mag_ratios table calibrated at
    len(src_ratios) steps to a different step count -- ported verbatim from
    upstream's nearest_interp (pure list/index arithmetic, no tensor
    involved). Needed because the calibration table is specific to the
    exact number of denoising steps it was recorded at; running inference
    with a different step count needs a resampled table, not a fresh
    calibration run.
    """
    src_length = len(src_ratios)
    if target_length == 1:
        return [src_ratios[-1]]
    scale = (src_length - 1) / (target_length - 1)
    return [src_ratios[round(i * scale)] for i in range(target_length)]


class MagCache:
    """The inference-time skip/cache decision engine.

    Parameters mirror upstream's per-model hyperparameters exactly:

    num_steps: total denoising steps in one generation run. `advance()`
        wraps the internal step counter (and resets the accumulators, but
        not the cached residual) once this many steps have passed, so a
        single MagCache instance can be reused across repeated generations
        without reconstructing it each time.
    mag_ratios: per-step calibration table (see module docstring). Indexed
        by the current step count, so must have at least `num_steps`
        entries -- use nearest_interp() first if it was calibrated at a
        different step count.
    retention_ratio: fraction of the *earliest* steps that always run in
        full, never skipped -- early denoising steps set overall structure
        and are the least redundant, so skipping them tends to hurt quality
        disproportionately. Upstream's default (0.1) means the first 10%
        of steps are never skip candidates.
    thresh: max allowed accumulated error (sum of |1 - accumulated_ratio|
        across consecutive skipped steps) before forcing a real recompute.
        Lower = more conservative (fewer skips, closer to full quality).
    K: max number of *consecutive* skipped steps before forcing a
        recompute regardless of accumulated error -- an upper bound on how
        stale the cached residual is allowed to get.
    """

    def __init__(
        self,
        num_steps: int,
        mag_ratios: Sequence[float],
        retention_ratio: float = 0.1,
        thresh: float = 0.24,
        K: int = 5,
    ):
        if len(mag_ratios) < num_steps:
            raise ValueError(
                f"mag_ratios has {len(mag_ratios)} entries, need at least "
                f"num_steps={num_steps} -- resample with nearest_interp() first "
                "if this table was calibrated at a different step count."
            )
        self.num_steps = num_steps
        self.mag_ratios = mag_ratios
        self.retention_ratio = retention_ratio
        self.thresh = thresh
        self.K = K
        self.cnt = 0
        self.cached_residual: Optional[torch.Tensor] = None
        self._reset_step_state()

    def _reset_step_state(self) -> None:
        self.accumulated_ratio = 1.0
        self.accumulated_err = 0.0
        self.accumulated_steps = 0

    def should_skip(self) -> bool:
        """Call once per denoising step, before running the block stack.
        True means: don't run it, reuse cached_residual instead. Every call
        advances the internal accumulators regardless of outcome (matching
        upstream: the accumulated ratio/error tracks consecutive candidate
        steps, not just accepted skips), so call this at most once per step
        -- pair with advance() at the end of the same step."""
        if self.cached_residual is None:
            return False
        if self.cnt < int(self.retention_ratio * self.num_steps + 0.5):
            return False
        self.accumulated_ratio *= self.mag_ratios[self.cnt]
        self.accumulated_steps += 1
        self.accumulated_err += abs(1 - self.accumulated_ratio)
        if self.accumulated_err <= self.thresh and self.accumulated_steps <= self.K:
            return True
        self._reset_step_state()
        return False

    def record(self, residual: torch.Tensor) -> None:
        """Call after actually running the block stack (i.e. whenever
        should_skip() returned False) with its output residual -- the
        exact quantity upstream calls `cur_residual = hidden_states -
        ori_hidden_states`. Do not call this on a skipped step; there is no
        new residual to record, cached_residual is being reused unchanged."""
        self.cached_residual = residual

    def advance(self) -> None:
        """Call once per step, after should_skip()/record() -- advances the
        step counter and, once num_steps is reached, resets it (and the
        accumulators) for the next generation run. cached_residual is left
        alone: harmless, since should_skip() won't consider skipping again
        until retention_ratio's steps have passed, by which point a real
        recompute will have replaced it anyway."""
        self.cnt += 1
        if self.cnt >= self.num_steps:
            self.cnt = 0
            self._reset_step_state()


class MagCacheCalibrator:
    """Run this instead of MagCache during a calibration pass over a
    representative generation: never skips (always records, never reuses),
    and accumulates the per-step magnitude-ratio statistics upstream prints
    at the end of magcache_calibration -- norm_ratio (this module's mean
    residual-norm ratio to the previous step), norm_std (its std, a rough
    stability signal), and cos_dist (1 - mean cosine similarity, a
    direction-change signal upstream logs alongside but that MagCache
    itself doesn't use for the skip decision).

    Usage:

        calib = amd_tuned_torch.magcache.MagCacheCalibrator(num_steps=28)
        for step in range(num_steps):
            ori_hidden_states = hidden_states
            hidden_states = run_transformer_blocks(hidden_states, ...)
            calib.record(hidden_states - ori_hidden_states)
        mag_ratios = calib.finalize()  # hardcode this for later MagCache(...) use
    """

    def __init__(self, num_steps: int):
        self.num_steps = num_steps
        self.cnt = 0
        self.previous_residual: Optional[torch.Tensor] = None
        self.norm_ratios: List[float] = []
        self.norm_stds: List[float] = []
        self.cos_dists: List[float] = []

    def record(self, residual: torch.Tensor) -> None:
        if self.previous_residual is not None:
            ratio = residual.norm(dim=-1) / self.previous_residual.norm(dim=-1)
            self.norm_ratios.append(round(ratio.mean().item(), 5))
            self.norm_stds.append(round(ratio.std().item(), 5))
            cos_dist = (
                1 - F.cosine_similarity(residual, self.previous_residual, dim=-1, eps=1e-8)
            ).mean().item()
            self.cos_dists.append(round(cos_dist, 5))
        self.previous_residual = residual
        self.cnt += 1

    def finalize(self) -> List[float]:
        """The mag_ratios table for MagCache(...). Prefixed with 1.0 to
        keep indices aligned with the calibration step count -- index 0
        corresponds to the first step, which has no previous residual to
        form a ratio against and is never actually looked up by
        should_skip() anyway (retention_ratio always covers it), but
        keeping the table's length equal to num_steps avoids an off-by-one
        surprise if retention_ratio is later set to 0."""
        return [1.0] + self.norm_ratios
