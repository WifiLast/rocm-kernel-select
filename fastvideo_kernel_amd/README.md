# fastvideo_kernel_amd

AMD ROCm version of `fastvideo-kernel` (FastVideo's kernel package,
`/mnt/data/image_to_video/gen/FastVideo/fastvideo-kernel`), for the
RX 7900 XTX (gfx1100) in the `py310_amd` env. Installs as `fastvideo-kernel`
`0.3.5+amd`, which still satisfies FastVideo's `fastvideo-kernel==0.3.5` pin,
so FastVideo imports it unchanged.

## Why a separate package

Upstream's compiled extension is CUDA only: the turbodiffusion int8
GEMM/quant uses CUTLASS/CuTe SM80 int8 MMA, and the attention kernels use
ThunderKittens (sm_90a) or Blackwell PTX. `build.sh --rocm` therefore builds no
AMD code and ships only the Python + Triton package. In that package:

- The 64-block VSA kernel used `tl.make_block_ptr` / `tl.advance`, which Triton 3.8
  (py310_amd's ROCm build) removed, so every VSA call failed to compile.
- int8 called a missing CUDA op.

What each API does there, and here:

| API | upstream on ROCm | here |
|---|---|---|
| `video_sparse_attn`, `block_sparse_attn*` (VSA, 64 blocks) | **compile error** on Triton 3.8 | fixed (block pointers → pointer tiles), gfx11 configs |
| `sliding_tile_attention` (STA) | Triton fallback | Triton fallback, gfx11 configs |
| VSA 128/256 blocks, H3 | Triton / CuTe (NVIDIA) | Triton where upstream has it |
| `FastRMSNorm`, `FastLayerNorm` | Triton | Triton (unchanged) |
| `int8_quant`, `int8_linear`, `Int8Linear` | **crash** (`quant_cuda` is None) | Triton int8 kernels from amd_tuned_torch |
| `moba_attn_varlen` (VMoBA) | flash-attn varlen | same (needs a ROCm flash-attn) |
| FP4 attn_qat_infer, Ulysses NCCL all-to-all | not built | not built (no AMD equivalent) |

## int8 kernels

The kernels are in the sibling package: `../amd_tuned_torch/blockscaled_int8_ops.py`.
Tests are in `tests/test_blockscaled_int8_ops.py` (CPU) and
`tests_hardware/test_blockscaled_int8_ops.py` (GPU) in that tree.
`fastvideo_kernel/_amd.py` loads that one file by its path relative to this directory. It does not import the
`amd_tuned_torch` package, which would patch torch globally, and py310_amd's
site-packages copy of it is older than the source tree.

Numerics match the CUDA kernels: 128×128 block scales `amax/127` (amax
floored at 1e-8), round-to-nearest-even, an int32 dot per 128-wide K block,
then the fp32 scale multiply. Weights quantised on either side load on the
other. On gfx1100 `tl.dot` on int8 lowers to WMMA iu8 (about 55 TOPS on a raw
4096³ probe).

## Environment variables

| var | effect |
|---|---|
| `FASTVIDEO_AMD_INT8_OPS=<path>` | load a different `blockscaled_int8_ops.py` |
| `FASTVIDEO_AMD_INT8=0` | no AMD int8 kernels (int8 calls fail like upstream) |
| `AMD_TUNED_TORCH_BLOCKSCALED_INT8=0` | int8 calls take the PyTorch reference path |
| `FASTVIDEO_AMD_FULL_AUTOTUNE=1` | upstream's full autotune config lists |

## Install / update

This directory is `amd_tools/source/cmp_ext_turing/fastvideo_kernel_amd`.
`/mnt/data/image_to_video/gen/FastVideo/fastvideo-kernel-amd` is a symlink to it.

```bash
uv pip install --python /mnt/data/miniconda3/envs/py310_amd/bin/python --no-deps -e .
```

`--no-deps` keeps the ROCm torch build: upstream pins `torch==2.12.0`, and this
package declares no dependencies.

The AMD changes are kept as `patches/amd.patch` against
`upstream_snapshot/python`, at the FastVideo commit in `upstream_snapshot/COMMIT`:

```bash
./sync_upstream.sh patch   # after editing python/: regenerate the patch
./sync_upstream.sh sync    # after a FastVideo git pull: new upstream + re-apply the patch
```

## Benchmark / tuning

```bash
env -u TORCH_LOGS -u TORCH_COMPILE_DEBUG \
  /mnt/data/miniconda3/envs/py310_amd/bin/python tools/bench_gfx11.py all [--sweep] [--quick]
```

Each run needs about 2–3 GB RAM and the GPU. Don't run it while Wan2GP or the
SD webui are generating: memory_watch.

## Results (RX 7900 XTX, torch 2.15.0.dev+rocm7.14, Triton 3.8, 2026-10-26)

Tests: upstream's suite (`tests/`) run against this package with
`pytest tests/<file>.py`:

| file | result |
|---|---|
| test_turbodiffusion | 23 passed, 4 skipped (compare against the CUDA norm ops) |
| test_vsa, test_vsa_combine, test_vsa_utils, test_vsa_tile_layout | all pass |
| test_vsa_triton_backward_scale, test_vsa128_backward, test_vsa256_triton | all pass (3 skipped, CUDA-only) |
| test_fused_compress_topk, test_sta, test_vmoba_correctness | all pass |
| test_vsa_varlen | 14/15. `test_backward_asymmetric_q_kv` is flaky (about 1 run in 4 fails): it compares two bf16 kernels with max/mean error < 5%, and with topk=1 most dK rows are 0. Neither kernel uses atomics. |

The amd_tools int8 tests also pass: quant is bit-identical to the reference.

Speed (`tools/bench_gfx11.py`, H=12, D=128, bf16):

| op | shape | time | dense SDPA |
|---|---|---|---|
| VSA fwd | S=32768, topk 51/512 (10 %) | 18.1 ms | 94.0 ms (5.2×) |
| VSA fwd | S=32768, topk 26/512 (5 %) | 9.6 ms | 94.0 ms |
| VSA fwd | S=65536, topk 102/1024 | 72.5 ms | 381.8 ms (5.3×) |
| VSA fwd+bwd | S=32768, topk 51 | 145.7 ms | |
| STA | Wan 18×48×80, window 3,3,3 | 69.4 ms | 429.9 ms (6.2×) |
| STA | Wan 18×48×80, window 3,1,5 | 38.6 ms | 429.9 ms |

gfx11 autotune sets (`_amd.py`): VSA fwd keeps `num_warps=4` with `num_stages` 2 and 3. Stage 2
wins at every shape; 8 warps is 1.5× slower; 4+ stages exceed the 64 KB LDS and
don't compile. STA keeps 4 of 36 configs (`BLOCK_Q` 64/128, `BLOCK_KV` 32/64,
1 stage). First-call autotuning drops accordingly: `test_sta` 6:48 → 2:56,
`test_vsa` 4:21 → 0:44.

int8 linear (quant + GEMM) against bf16 `x @ w.T` (hipBLASLt), M=32760:

| N×K | bf16 | int8 | |
|---|---|---|---|
| 1536×1536 | 1.77 ms | 2.75 ms | 0.64× |
| 8960×1536 | 9.64 ms | 12.65 ms | 0.76× |
| 5120×5120 | 17.06 ms | 24.87 ms | 0.69× |

int8 is **slower** on RDNA3: WMMA iu8 has the same peak rate as WMMA fp16/bf16,
so there is no compute gain, and quantising the activation costs an extra pass.
The int8 path is for running TurboDiffusion int8 checkpoints and for half-size
weights, not for speed. Relative error against bf16 is 1.35 %.
