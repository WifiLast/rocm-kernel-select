"""Kernel-variant definitions for every codegen'd conv2d_fp16/conv3d_fp16
kernel family (currently: the WMMA implicit-GEMM kernels, and the Winograd
conv3d kernel) -- single source of truth consumed by:
  - generate.py  -- renders each variant into src/cuda/generated/<op>_<suffix>.cu
                     from src/cuda/templates/<op>.cu.tmpl via
                     string.Template.substitute(SUFFIX=variant.suffix, **variant.params)
  - autotune.py  -- benchmarks these variants against real shapes on real
                     hardware and writes tools/kernelgen/tuned_shapes.json

`params` is a plain dict of the template's $-placeholders (besides
$SUFFIX, which every template gets automatically) -- deliberately generic
rather than fixed fields, since different kernel families have different
tunable parameters (BM/BN/BK/STAGES for the WMMA GEMM kernels,
BT_TILES/BT_COUT for Winograd's register-blocked accumulate step) and
there's no reason to force them into a shared schema.

To add a variant: append one below, run
`python tools/kernelgen/generate.py`, rebuild
(`pip install -e . --no-build-isolation`), then re-run autotune.py if you
want it considered by the offline dispatch table.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Variant:
    op: str        # must match a src/cuda/templates/<op>.cu.tmpl file
    suffix: str     # unique per op; becomes part of the generated filename and
                    # the kernel/launcher symbol names (e.g. op="conv2d_fp16",
                    # suffix="bm256_bn128_bk32_s2" ->
                    # src/cuda/generated/conv2d_fp16_bm256_bn128_bk32_s2.cu,
                    # launch_conv2d_fp16_bm256_bn128_bk32_s2)
    params: dict = field(default_factory=dict)  # template $-placeholders -> value


# The original hand-written WMMA kernels' tile shape (BM=256/BN=128/BK=32/
# STAGES=2), reproduced byte-for-byte -- kept as variant 0 / the safe
# default every op falls back to.
#
# conv2d_fp16 additionally has two shape-motivated variants, both
# real-hardware candidates in run_conv2d_fp16's live benchmark-and-cache
# dispatch (src/main_rocm.cpp) -- neither is assumed to win anywhere
# without that dispatch actually measuring it for the shape at hand:
#
#   bm256_bn64_bk32_s2 -- half-width N-tile. Targets C_out < 128 (e.g.
#   tools/bench.py's own conv2d benchmark shape, C_out=64 -- BN=128 wastes
#   half that tile there). Irrelevant to C_out=256 shapes (miopen_amd_log.txt's
#   three real shapes all have C_out=256, an exact multiple of every BN
#   candidate here -- 0% waste already at BN=128; see
#   tools/kernelgen/shapes.py) -- included for the shapes it DOES help,
#   not those.
#
#   bm128_bn128_bk32_s3 -- smaller M-tile, 3-deep pipeline instead of 2.
#   Targets K-depth, not M or N: RDNA3 has no cp.async-equivalent async
#   global->LDS copy, so every main-loop iteration's LOAD_A/LOAD_B has to
#   fully land in shared memory (behind a __syncthreads()) before that
#   iteration's WMMA compute can start, with only STAGES=2's one tile of
#   lookahead to hide that latency. miopen_amd_log.txt's two 3x3 shapes
#   (C_in=256, K_H=K_W=3 -> GEMM-K=2304) run 2304/32=72 of these
#   iterations per block -- deeper buffering gives the load/compute
#   overlap more slack for exactly that case. BM shrinks from 256 to 128
#   only because STAGES=3's extra buffer doesn't fit the 64KB LDS budget
#   at BM=256 (see tools/kernelgen/generate.py's rendered smem-budget
#   comment in the launcher) -- M is enormous for both real shapes
#   (524288 and 2097152 pixels), so a smaller M-tile costs nothing here,
#   just more blocks.
#
# conv3d_fp16_winograd: F(2x2x2,3x3x3) Winograd, ported from the existing
# fp32 conv3d_fp32_winograd.cu (mixed precision -- half storage for U/V/
# input/weight/output, float compute for every transform and the C_in
# reduction; see that template's header for why). BT_TILES=BT_COUT=8
# reproduces the fp32 kernel's register-blocking factor, which its own
# header already flags as tuned on the original CUDA/Turing build and
# never re-tuned for RDNA3 -- doubly true here since the dtype changed
# too. Not yet wired into any runtime dispatch (src/main_rocm.cpp) --
# exists as a generated, buildable kernel pending real-hardware validation.
VARIANTS = [
    Variant("conv2d_fp16", "bm256_bn128_bk32_s2",
            params=dict(BM=256, BN=128, BK=32, STAGES=2)),
    Variant("conv2d_fp16", "bm256_bn64_bk32_s2",
            params=dict(BM=256, BN=64, BK=32, STAGES=2)),
    Variant("conv2d_fp16", "bm128_bn128_bk32_s3",
            params=dict(BM=128, BN=128, BK=32, STAGES=3)),
    Variant("conv3d_fp16", "bm256_bn128_bk32_s2",
            params=dict(BM=256, BN=128, BK=32, STAGES=2)),
    Variant("conv3d_fp16_winograd", "bt8_bc8",
            params=dict(BT_TILES=8, BT_COUT=8)),
]


def variants_for(op):
    return [v for v in VARIANTS if v.op == op]
