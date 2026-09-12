# Provenance

`amd_tuned_torch/fftconv_ops.py`'s `fft_conv`/`complex_matmul`/`to_ntuple`
are adapted from `fft-conv-pytorch`:

  https://github.com/fkodom/fft-conv-pytorch
  `fft_conv_pytorch/fft_conv.py` (`fft_conv`, `complex_matmul`, `to_ntuple`).
  MIT License, Copyright (c) 2015 Logan Spears. See LICENSE in this
  directory for the full license text.

Unlike every other `third_party/` dependency in this package (CuMesh,
FlexGEMM, nvdiffrast, torchsparse -- separate installable extensions with
their own compiled CUDA/HIP kernels), upstream `fft-conv-pytorch` is pure
PyTorch: N-D convolution expressed entirely through `torch.fft.rfftn`/
`irfftn`, `F.pad`, and `torch.kron`, no custom kernel of any kind. That
makes it backend-agnostic already -- nothing to HIPify, no
`third_party/`-style build step, no `available()` gate tied to whether an
extension compiled -- so it's vendored here (small, single-file,
permissively licensed) rather than kept as a separate installable
package, the same posture as `_vendor/megatron_swiglu` (also plain
PyTorch math with nothing to compile).

WHY THIS EXISTS ALONGSIDE flash-fft-conv, NOT INSTEAD OF IT. `source/
flash-fft-conv` (Stanford Hazy Research's FlashFFTConv, the tensor-core
Monarch-decomposition FFT-conv library) was evaluated first and rejected
as a porting candidate: its entire compute path (butterfly FFT/IFFT,
Monarch matmul, fwd+bwd, fp16+bf16) is built on CUDA `wmma::fragment`/
`mma_sync` tensor-core intrinsics with no non-tensor-core fallback --
porting it means rewriting every one of those kernels against rocWMMA/MFMA,
real matrix-core work this project's other ports (torchsparse, nvdiffrast)
didn't require because each of those kept at least one portable dataflow.
`fft-conv-pytorch` gets the same algorithmic idea (FFT-based convolution,
which beats direct/im2col convolution for large kernels) onto ROCm today,
at the cost of not having flash-fft-conv's tensor-core speedup -- see
`amd_tuned_torch/fftconv_ops.py`'s own module docstring for where that
trade is expected to still be worth it.

What changed from upstream, and why:

  - `fft_conv` here casts its output back to `signal`'s original dtype
    before returning, instead of upstream's forced `.float()` on both the
    signal and kernel BEFORE anything else (dilation expansion, padding)
    and returning that float32 result as-is -- fine for upstream's own
    examples and tests (which stay in fp32 throughout) but a correctness
    hazard for a drop-in `F.conv1d`-shaped adapter meant to sit in a
    mixed-precision model's forward pass, where a layer silently switching
    a fp16 tensor to fp32 downstream can not just tax memory/bandwidth but
    change what `torch.autocast`/absolute-tolerance checks elsewhere in
    the same forward pass see. This is the "dtype consistency" caveat
    flagged when `fft-conv-pytorch` was first surveyed (see
    `[[reference-source-dir-candidate-libs]]`), now actually handled --
    and narrowed further than just "restore the dtype at the end": the
    `torch.kron` dilation expansion and both `F.pad` calls (the largest
    tensors this function touches, for the long-kernel/long-sequence
    workload this module targets) now run in `signal`'s ORIGINAL dtype
    too, not just get cast back afterward -- float32 is used only for
    `rfftn`/`complex_matmul`/`irfftn` themselves (upcast right before
    `rfftn`, downcast right after `irfftn`), the one part of this function
    that (a) `torch.fft` requires float32/float64 for regardless of
    caller dtype and (b) actually accumulates enough rounding error to
    matter. fp16/bf16 storage, fp32 transform -- keeps the bandwidth win
    on the activation-sized padding/dilation tensors while keeping the
    accuracy where it's actually needed. See `fft_conv`'s own docstring's
    MIXED PRECISION section.
  - `to_ntuple`'s `Iterable` check widened to also exclude `str` explicitly
    (a bare string is technically `Iterable` in Python -- iterating its
    characters -- so `to_ntuple("same", n=2)` upstream would silently
    iterate characters instead of raising; not reachable through this
    file's own call sites today, since `padding="same"` is handled before
    `to_ntuple` ever sees the string, but tightened here since a vendored
    copy shouldn't keep a latent foot-gun upstream never exercises either).
  - Module-level `nn.Module` wrapper classes (`_FFTConv`, `FFTConv1d`/
    `2d`/`3d`) were NOT ported -- this package's convention is a
    functional adapter over `F.convNd`-shaped call sites
    (`amd_tuned_torch.miopen_fallback`'s conv1d dispatch, mirroring how
    `flexgemm_ops`/`torchsparse_ops` expose plain functions, not
    `nn.Module` subclasses a caller would need to swap into their model
    definition). A caller wanting an `nn.Conv1d`-shaped layer already gets
    this transparently once `amd_tuned_torch.miopen_fallback`'s F.conv1d
    patch routes an eligible large-kernel call here.
  - `complex_matmul` writes the frequency-domain contraction as a
    `torch.einsum` over contiguous operands (plus a plain elementwise
    multiply for the depthwise case), instead of upstream's `movedim`-
    reshaped `@` with a degenerate 1-wide dimension -- and instead of
    upstream's manual 4-real-matmul expansion (`a.real@b.real -
    a.imag@b.imag`, `a.imag@b.real + a.real@b.imag`, reassembled into a
    freshly allocated complex tensor), which this file replaced with a
    single native complex `@` first. Both of those earlier forms ask the
    BLAS backend for a batched GEMM per frequency bin over non-contiguous
    views; measured on gfx1100 the einsum is 6-8x faster for grouped/dense
    convs and 80x for depthwise (where the "matmul" is a scalar product
    dispatched as a 1x1 GEMM), bit-identical output. It also fixes a crash
    inherited from upstream: the final `view` over a `movedim`-produced
    non-contiguous tensor raised `RuntimeError: view size is not
    compatible with input tensor's size and stride` for any conv with
    `groups > 1` and `Cout/groups > 1`. See `complex_matmul`'s own
    docstring for the numbers and
    `tests/test_fftconv_ops.py::TestComplexMatmul`, which pins the new
    form against the old contraction.
  - `to_ntuple` is otherwise unchanged from upstream.
