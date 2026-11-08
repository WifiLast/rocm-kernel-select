# Conv tiers vs stock on a real SDXL workload (gfx1100, 2026-09-17)

Source: `tools/bench_conv_from_miopen_log.py` over a MIOpen capture of one
SDXL 2048x1024 generation from stable-diffusion-webui-amdgpu
(`log/miopen/miopen-conv-20260917-165449.log`, 3,972 forward convs across
72 distinct shapes, 344 TFLOP, 100% NCHW).

## Headline

    call-weighted conv time: stock 2242.0 ms, best-tier 2226.1 ms
    -> 1.01x available

**This package's conv tiers have essentially nothing to offer this
workload.** Stock wins 44 of the 72 shapes outright, CK wins 5, the native
HIP kernel wins 1, and 22 have no applicable tier. On the 20 most-called
shapes, stock wins every one, and CK/native are frequently 2-4x SLOWER:

    shape                       calls   stock      ck   native   winner
    2x1280x32x16->1280 k3s1       450   0.524   0.545    1.017   stock
    2x1280x64x32->1280 k3s1       375   1.372   1.718    3.666   stock
    2x320x128x64->320 k3s1        315   0.457   0.556    1.052   stock
    2x640x128x64->640 k3s1        243   1.254   1.654    3.400   stock
    2x320x256x128->320 k3s1       231   1.272   2.218    4.150   stock

No shape shows a tier beating stock by more than 5%.

## Why stock is so hard to beat here

MIOpen serves 77.3% of these convolutions with
`miopenSp3AsmConv_v30_3_1_gfx11_fp16_dot2_f2x3_stride1` -- a hand-written
gfx11 assembly Winograd kernel -- and the remaining 22.7% (all 1x1) through
rocBLAS. Both are already near-optimal for NCHW fp16 at these sizes.

## What this means for the package

The contest machinery is doing its job: it measures, picks stock, and
caches. Nothing here is broken. But do not expect the conv tiers to move a
diffusion workload on this hardware, and do not spend effort tuning them
against these shapes without first re-running this benchmark.

## Things that were measured and found NOT to help

* `--opt-channelslast` / channels_last -- CATASTROPHIC here, 239x-753x
  slower convolution. See the memory note; never enable it on gfx1100 for
  this workload.
* `MIOPEN_FIND_MODE=1` (full search) vs the default `FAST(2)` -- a first
  measurement suggested 1.24x, but a NULL CONTROL (mode 2 measured twice:
  2.848ms / 2.836ms, against mode 1's 2.830ms / 2.872ms) showed the
  difference was entirely the cold first run. No effect once warm.
* MIOpen conv fusion -- structurally unreachable: PyTorch does not link
  MIOpen's fusion API. See `tools/analyse_miopen_log.py`.

## Measurement warning

This card idles at 96MHz mclk / 0MHz sclk at Performance Level "auto", and
a benchmark burst gets timed somewhere on the clock ramp. Three separate
"findings" in the session that produced this file were artifacts of
one-then-the-other timing (0.32x, -40%, and the 1.24x above). Always
interleave A and B, and always run a null control with the SAME
configuration on both sides. Pin clocks with `rocm-smi --setperflevel high`
for anything you intend to quote.
