# Provenance

The three device kernels in `src/cuda/iu4_gemm_fwd.cu`
(`iu4_gemm_candidate`, `iu8_gemm_control`, `dot4_i8_control`) are ported
from:

- Upstream: https://github.com/DrBearJew/trellis2-convrot-rocm
- Path: `experiments/gfx1100-iu4/iu4_gemm_probe.hip`
- Commit: `13ec09a8329cd607f762a7eb7407d7302fe17dcd` (2026-08-01)
- License: MIT (see `LICENSE` in this directory)

Deliberately NOT vendored from the same upstream project: the production
TRELLIS.2 ConvRot INT8 GEMM (`int8_fused_kernel.py`'s
`_int8_matmul_dequant_per_row_kernel`, a Triton kernel with per-row
activation/weight quantization and TRELLIS-shape-tuned static tile
configs). That kernel only exists as a patch against
`ComfyUI-INT8-Fast-ROCM`, which is **AGPL-3.0** -- upstream's own
`docs/ARCHITECTURE.md` deliberately isolates that backend behind a private
package boundary for exactly this reason. Copying it here would pull
AGPL-3.0-licensed code into this project; the WMMA probe above is the only
kernel in that upstream repository with a clean, permissive license.

## What was changed from upstream

- The three `__global__` kernels and their `v_wmma_i32_16x16x16_iu4/iu8`
  and `v_dot4_i32_i8` device helpers, the `kTile`/vector-type constants,
  and the nibble/byte packing helpers (`load_i4_nibble`,
  `pack_i4_tail_word`, `pack_i8_tail_word`) are copied essentially
  verbatim -- these are the actual kernel under test.
- Upstream's `main()`, CPU reference, benchmark harness, and host-side
  `pack_i4_rows`/`pad_i8_rows` test-data preparation were **not** ported --
  they exist only to drive the standalone `iu4_gemm_probe` binary.
  Equivalent packing for real use lives in `amd_tuned_torch/iu4_gemm_ops.py`
  instead (`torch` tensor ops, not a second copy of the C++ helpers).
- `launch_i4`/`launch_i8`/`launch_dot4_i8` were rewritten as
  `launch_iu4_gemm`/`launch_iu8_gemm`/`launch_dot4_i8_gemm` matching this
  project's `launch_*(..., hipStream_t stream)` convention (see
  `src/cuda/group_norm.cu`, `src/cuda/conv2d_fp32.cu`) instead of taking a
  `DeviceBuffer`/default-null-stream pair built for the standalone probe.

## Status -- unlike every other tier in this project, unvalidated on real
hardware and not wired into `kernel_select`

Upstream's own README calls this "an isolated candidate, not a promoted
TRELLIS runtime path" and "a correctness/roofline probe, not yet an
LDS-staged production GEMM" -- direct global-memory fragment loads, no
shared-memory staging, never benchmarked against anything outside its own
three variants. `amd_tuned_torch/iu4_gemm_ops.py` keeps that framing: it is
exposed as an explicit opt-in (`iu4_linear`/`iu8_linear`), never entered
into `kernel_select`'s stock-vs-candidate contest the way `ck_gemm_ops`/
`hipblaslt_ops`/`splitk_gemm_ops` are. That contest's correctness gate
(`torch.allclose` against an fp16/bf16 reference, see
`kernel_select.py`'s CORRECTNESS VERIFICATION) assumes a candidate computes
the *same* answer as stock to within rounding -- true for every other
tier, but never true for an INT8/INT4-quantized GEMM against an
unquantized reference, so it would fail verification on every shape and be
permanently blacklisted despite working as designed. Quantization error is
an expected, load-bearing property here, not a bug the contest should be
allowed to "catch".
