# Hardware tests (ROCm / RX 7900 XTX)

This directory previously held `test_attention_bf16_routing.py`, which
validated the Turing/CMP-specific bf16->fp16 conversion path in the old CUDA
attention kernel (`fp16_attention.cu`). That kernel and the conversion logic
it tested no longer exist -- RDNA3's WMMA units have native bf16 matrix
throughput, so there's no conversion boundary to validate, and attention is
now routed through TransformerEngine's own fused attention (CK/AOTriton
backends), which has its own test suite upstream
(`source/TransformerEngine/tests/pytorch/attention`).

No replacement hardware test has been written here yet. Anything added
should run against a real RX 7900 XTX with amd_tuned_torch's native extension built
and TransformerEngine installed -- validate at minimum:

- `amd_tuned_torch.ops.linear` / `.bmm` numerically match `F.linear` / `torch.bmm`
  (within fp16/bf16 tolerance) across a range of M/N/K shapes, including
  ones with no matching CK instance (confirm the RuntimeError fallback path
  in `amd_tuned_torch/__init__.py` actually triggers stock rocBLAS/hipBLASLt).
- `amd_tuned_torch.ops.group_norm` matches `F.group_norm` across group counts that
  don't divide evenly into a full wavefront (64 or 32 threads), which is
  where a block-reduction kernel is most likely to have an off-by-one.
- `amd_tuned_torch.ops.conv2d` / `.conv3d`: **now covered by
  `test_conv_kernels.py`** in this directory -- run it on its own (not
  combined with `tests/`, see the note at the top of that file about
  `tests/conftest.py`'s MagicMock stub leaking in otherwise):
  ```
  pytest tests_hardware/test_conv_kernels.py -v
  ```
  fp32 conv2d/conv3d (`src/cuda/conv{2,3}d_fp32.cu`) are ports of the
  original CMP-Turing project's CUDA kernels (`src/cuda/kernel_example/`).
  fp16 conv2d/conv3d are WMMA (gfx1100 matrix-core) kernels codegen'd from
  `src/cuda/templates/*.cu.tmpl` into `src/cuda/generated/conv{2,3}d_fp16_*.cu`
  by `tools/kernelgen/generate.py`, one file per tile-shape variant in
  `tools/kernelgen/variants.py` -- see `src/main_rocm.cpp`'s
  `run_conv2d_fp16`/`run_conv3d_fp16` for the runtime dispatch across
  variants (currently just one; see `tools/kernelgen/` for adding more).
  None of this has run on real RDNA3 hardware yet -- `test_conv_kernels.py`
  checks numerically against `F.conv2d`/`F.conv3d` across a range of shapes
  (odd `H_in`/`W_in`/`D_in` not a multiple of the tile, `C_in`/`C_out` not a
  multiple of the tile/`CTILE`, stride/padding/dilation != 1, with and
  without bias) but still needs a real run to confirm it actually passes,
  before trusting the fallback-on-error paths in
  `_patched_conv2d`/`_patched_conv3d` are the only thing standing between a
  bad shape and silently wrong output. The fp16 kernels' double-buffered
  shared-memory pipeline (`STAGES=2`) and WMMA fragment loads are
  particularly worth stress-testing under a profiler, not just
  `assert_close` -- correct on paper, but never executed on this
  architecture before. Also re-run `tools/bench.py`-style timing for both
  once built: `benchmark.json`'s existing conv2d/conv3d numbers predate the
  WMMA rewrite (conv2d_fp16 measured *slower* than stock, 0.93x -- see that
  file), so the real win from real matrix-core instructions is unverified
  against real numbers until this is re-run.
- `src/cuda/conv3d_fp32_winograd.cu` (fp32 conv3d, batch=1,
  kernel=3x3x3/stride1/pad1/dilation1 only) is a second HIP port layered on
  top of the above, never run on real RDNA3 hardware either: confirm it
  numerically matches `F.conv3d` (its own tolerance, since it's a different
  algorithm from the direct kernel, not just a different implementation of
  the same one) at a shape within its scope, and confirm `src/main_rocm.cpp`'s
  `run_conv3d_fp32` benchmark-and-cache dispatch actually settles on a
  consistent winner across repeated calls with the same shape (it should
  only pay the double-kernel benchmarking cost once per distinct shape, not
  every call -- watch for this with a profiler, not just wall-clock, since a
  caching bug here wouldn't produce wrong numbers, just quietly double the
  per-call cost forever). Also confirm a shape outside Winograd's scope
  (odd `D_in`/`H_in`/`W_in`, non-3x3x3 kernel, batch>1, non-unit
  stride/padding/dilation, or a tile-count/`C_out` not divisible by 8) falls
  straight through to the direct kernel without ever entering the benchmark
  path (cheap correctness check: the very first call for such a shape
  should not allocate a scratch tensor).
- End-to-end: run `amd_tuned_torch.enable()` against an actual diffusion or LLM
  inference pipeline and confirm output parity with `amd_tuned_torch` disabled.
- `torch.matmul`'s new >=4D path (`amd_tuned_torch/__init__.py:_patched_matmul`):
  confirm `aiter_ops.bmm_fp16` numerically matches stock `torch.matmul` on
  real `(batch, heads, seq, head_dim)` attention-shaped tensors (an eager
  attention implementation's Q@K^T and attn@V, e.g. Medusa's or Lookahead
  Decoding's manual `torch.matmul`-based attention), not just the 2D/3D
  shapes the existing dispatch-logic tests cover with mocks.
- TE attention's `attn_mask_type="causal_bottom_right"` path
  (`amd_tuned_torch/te_ops.py:scaled_dot_product_attention`, reached whenever
  `is_bottom_right_causal_mask()` recognizes an explicit causal mask tensor):
  confirm this numerically matches stock
  `F.scaled_dot_product_attention(..., attn_mask=<the same tensor>)` for both
  q_len == kv_len (prefill) and q_len < kv_len (KV-cache decode steps) --
  this mapping was derived by reading TE's source, not by running it against
  a real ROCm + TransformerEngine install, so it's unverified until tested
  here.
- SmoothQuant calibration (`amd_tuned_torch/aiter_ops.py:calibrate_smoothquant` /
  `linear_int8`): confirm end-to-end accuracy on a real model actually
  improves (lower perplexity / closer logits to the unquantized model)
  versus plain `enable_int8_linear()` without calibration, on a model with
  real LLM-typical activation outlier channels -- the calibration math and
  aiter's `smoothquant_quantize` dispatch are unit-tested (`tests/
  test_smoothquant.py`), but nothing here yet confirms the *accuracy
  benefit* the whole feature exists for, only that the numerics compose
  correctly.
- `amd_tuned_torch/compile_ops.py`'s `torch.library.custom_op` registrations: the
  dispatch/shape-inference logic is unit-tested on CPU with mocked
  aiter/native calls (`tests/test_compile_ops.py`), but the actual
  performance claim -- `torch.compile(model)` produces a more fused graph,
  and `fullgraph=True` doesn't raise, with the *real* aiter Triton kernels
  and native HIP group_norm extension in the loop, on real ROCm/RDNA3 --
  is unverified until tested here. Also worth confirming: `torch.compile`
  doesn't recompile/re-register these ops on every call (registration is a
  process-wide, one-time side effect; nothing here proves that holds up
  under repeated `torch.compile` invocations against different input shapes
  in a real training/inference loop).
