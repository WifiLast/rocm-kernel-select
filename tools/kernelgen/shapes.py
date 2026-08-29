"""Example/benchmark input shapes for conv2d_fp16/conv3d_fp16 -- the
corpus tools/kernelgen/autotune.py sweeps tile-shape variants
(tools/kernelgen/variants.py) against to decide which one wins per shape.

conv2d shapes marked "miopen_amd_log.txt" are transcribed from a real
MIOpen trace captured on this project's own RX 7900 XTX (the log's cache
paths reference gfx1100_48.HIP.3_5_1_f322e9ab61 and
/home/wiffzack/.config/miopen/...). Every `Command [LogCmdConvolution]
./bin/MIOpenDriver convfp16 ...` line in that file was extracted and
cross-checked against its surrounding xDesc/wDesc tensor-descriptor dump
(not just trusted from the CLI flags alone), then deduplicated -- MIOpen's
find/benchmark process logs the same call multiple times while trying
different solvers. All three are fp16/NCHW/groups=1/stride=1/dilation=1:
a 512->256 channel 1x1 projection, a 256->256 3x3 conv repeated 4x at
1024x512, then the same 256->256 3x3 conv again at the doubled 2048x1024
resolution -- this reads as one upsampling pipeline over a large image
(a VAE decoder or a pixel-space upscaler, not a U-Net's latent-space
resolution). No conv3d calls appear anywhere in that log, so conv3d's
corpus below is entirely synthetic/from tools/bench.py.

Every shape not sourced from the log is hand-picked to stress a specific
tile-shape edge case the log's shapes don't cover -- see each entry's
`source` string.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Conv2dShape:
    label: str
    source: str
    B: int
    C_in: int
    H_in: int
    W_in: int
    C_out: int
    K: int
    stride: int
    padding: int
    dilation: int = 1


@dataclass(frozen=True)
class Conv3dShape:
    label: str
    source: str
    B: int
    C_in: int
    D_in: int
    H_in: int
    W_in: int
    C_out: int
    K: int
    stride: int
    padding: int
    dilation: int = 1


CONV2D_SHAPES = [
    # -- real, from miopen_amd_log.txt --
    Conv2dShape(
        "log_1x1_c512to256_1024x512", "miopen_amd_log.txt:170",
        B=1, C_in=512, H_in=1024, W_in=512, C_out=256, K=1, stride=1, padding=0,
    ),
    Conv2dShape(
        "log_3x3_c256to256_1024x512", "miopen_amd_log.txt:257,341,425,509",
        B=1, C_in=256, H_in=1024, W_in=512, C_out=256, K=3, stride=1, padding=1,
    ),
    Conv2dShape(
        "log_3x3_c256to256_2048x1024", "miopen_amd_log.txt:674",
        B=1, C_in=256, H_in=2048, W_in=1024, C_out=256, K=3, stride=1, padding=1,
    ),
    # -- synthetic: C_out=64 < BN=128 wastes half the default tile's N-width
    # (the concrete regression that motivated adding a narrow-N variant in
    # the first place -- see conv2d_fp16.cu.tmpl's header). --
    Conv2dShape(
        "synthetic_narrow_cout64", "tools/bench.py's existing conv2d benchmark shape",
        B=64, C_in=128, H_in=128, W_in=128, C_out=64, K=3, stride=1, padding=1,
    ),
    # -- synthetic: small spatial extent (M much smaller than one BM=256
    # tile) with wide channels -- opposite edge from the log's huge-M shapes. --
    Conv2dShape(
        "synthetic_small_spatial_wide_channels", "hand-picked",
        B=1, C_in=320, H_in=16, W_in=16, C_out=320, K=3, stride=1, padding=1,
    ),
    # -- synthetic: 1x1 pointwise at a small, U-Net-latent-typical resolution
    # (the log's only 1x1 shape is at pixel-space 1024x512; this contrasts
    # it against a much smaller, very common SD/SDXL-latent-scale shape). --
    Conv2dShape(
        "synthetic_1x1_latent_res", "hand-picked",
        B=2, C_in=640, H_in=64, W_in=64, C_out=320, K=1, stride=1, padding=0,
    ),
]

CONV3D_SHAPES = [
    # miopen_amd_log.txt has no conv3d invocations at all -- nothing to
    # source from there. Kept from tools/bench.py's existing conv3d
    # benchmark shape, plus one smaller/different-aspect synthetic shape.
    Conv3dShape(
        "bench_c512_d8_hw32", "tools/bench.py's existing conv3d benchmark shape",
        B=1, C_in=512, D_in=8, H_in=32, W_in=32, C_out=512, K=3, stride=1, padding=1,
    ),
    Conv3dShape(
        "synthetic_small_batch2_d16", "hand-picked",
        B=2, C_in=64, D_in=16, H_in=16, W_in=16, C_out=128, K=3, stride=1, padding=1,
    ),
]
