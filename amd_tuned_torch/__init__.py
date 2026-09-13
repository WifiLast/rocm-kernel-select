"""
amd_tuned_torch -- monkeypatches PyTorch's ROCm op dispatch on RX 7900 XTX
(gfx1100/RDNA3) with the most optimal kernels available on this platform:

  - F.linear, torch.matmul, torch.bmm are backed by aiter's Triton WMMA
    GEMM kernels (amd_tuned_torch/aiter_ops.py: linear_fp16/bmm_fp16) -- gluon
    backend on gfx1250, Triton (WMMA-targeted) everywhere else including
    gfx1100/RDNA3, per aiter's own kernel docstrings. torch.matmul also
    covers >=4D batched inputs (e.g. attention's (batch, heads, seq,
    head_dim) Q@K^T / attn@V) by flattening the leading batch dims into one
    before calling aiter and reshaping back -- only when both operands
    share the exact same batch shape (no broadcasting; aiter's batched GEMM
    has none of its own).
  - F.conv2d (groups=1, fp16/fp32) is backed by a hand-written HIP kernel
    (src/cuda/conv2d_fp{16,32}.cu, via compile_ops.conv2d_native) -- a port
    of the original CMP-Turing project's register-blocked CUDA kernels (see
    each .cu file's header comment). aiter's Triton conv2d (fp16/bf16 only)
    benchmarked slower than stock MIOpen on RX 7900 XTX for fp16 (0.93x, see
    benchmark.json), so it's now the SECOND tier, used only for bf16 (which
    the native kernel doesn't support) -- see _patched_conv2d for the exact
    tier order. Non-contiguous inputs/weights and groups!=1 fall back to
    stock, same as before. 1x1/stride1/pad0/dilation1 convs ("pointwise")
    are a special case that skips BOTH tiers and goes straight to stock:
    they're a pure channel-mixing GEMM with no spatial reduction, and
    miopen_amd_log.txt shows MIOpen's own rocBLAS GEMM solver winning that
    case by 3.4x over its best Winograd kernel and 450x over naive direct --
    see _is_pointwise_conv2d's docstring for the full numbers and why
    neither of this module's tiers can realistically beat that.
  - F.conv3d (groups=1, fp16/fp32) is backed by a hand-written HIP kernel
    (src/cuda/conv3d_fp{16,32}.cu, via compile_ops.conv3d_native) -- also
    ported from the original CMP-Turing project. Neither aiter nor
    TransformerEngine cover conv3d at all; Composable Kernel's WMMA conv
    (ck_ops.conv3d, fp16/bf16) and FFT-conv (below) fill that gap, all
    measured against stock per shape (see _patched_conv3d).
  - F.conv2d/F.conv3d with a LARGE kernel additionally contest FFT-conv
    (amd_tuned_torch.fftconv_ops, frequency-domain convolution), the one
    tier here whose advantage is algorithmic rather than an
    implementation detail: direct convolution pays K^ndim multiplies per
    output element, an FFT pays for the transform once. It is gated by a
    measured per-ndim kernel-width pre-filter so ordinary 3x3 convs never
    pay for it, and it is where the largest win in this file lives
    (measured 15^3 fp32 conv3d: stock 6907ms vs 38.8ms). See
    _fftconv_conv_candidate's docstring for the crossover measurements
    and AMD_TUNED_TORCH_FFTCONV2D/3D[_MIN_KERNEL] to retune or disable it.
  - F.group_norm is a hand-written HIP kernel (src/cuda/group_norm.cu, via
    the native HIP/C++ extension in src/main_rocm.cpp) -- neither aiter nor
    TransformerEngine cover GroupNorm (a diffusion-U-Net-specific op, not a
    transformer/LLM one), so this one stays genuinely hand-rolled.
  - F.scaled_dot_product_attention, F.rms_norm, F.gelu
    (approximate="tanh" only), F.silu (fp16/bf16 only) are routed to
    TransformerEngine's ROCm-native bindings (amd_tuned_torch/te_ops.py) -- TE
    already ships tuned CK/AOTriton fused attention and fused
    RMSNorm/GELU/SiLU kernels for ROCm, so there's no reason to hand-roll
    those. F.layer_norm is NOT auto-patched: TE's layernorm_fwd benchmarked
    slower than stock for both fp16 (0.71x) and fp32 (0.61x), see
    benchmark.json -- _patched_layer_norm remains available for manual use.
    F.silu's fp32 case also benchmarked slower (0.80x, vs. fp16's 1.15x) and
    is excluded from the patch, falling back to stock per-call. Attention
    also fires for an explicit causal `attn_mask` tensor, not just
    `is_causal=True` -- HF model code and dflash both build one for
    KV-cache decoding instead of setting the flag -- but only when it
    structurally matches TE's fused "causal_bottom_right" pattern (see
    te_ops.is_bottom_right_causal_mask); every other mask shape (padding,
    sliding-window, arbitrary) still falls back to stock.

amd_tuned_torch.aiter_ops.fused_silu_mul is a manually-callable helper (not
auto-patched -- no stock F.* op matches its SwiGLU-style gate*up
semantics), backed by aiter.ops.triton.activation.fused_silu_mul directly,
same local-import dependency as every other aiter_ops function, no
network involved. (kernels-community/aiter-kernels, a Hugging Face Hub
repackaging of this exact aiter source, was investigated first via
amd_tuned_torch/hub_ops.py -- turned out redundant with what's already local, so
importing it directly here removes a network round-trip that would have
fetched the identical code. See amd_tuned_torch/hub_ops.py for that history, and
for why kernels-community/activation -- the *other* Hub kernel tried --
was dropped instead of adapted: zero published ROCm build variants, so it
could never actually engage on RDNA3.)

Composable Kernel is NOT a dependency of this package (an earlier version
routed linear/matmul/bmm through CK's DeviceGemm instances directly --
src/cuda/ck_gemm.cu, since deleted -- before switching to aiter's Triton
GEMM, which needs no separate C++ library build/link step).

This is a DIFFERENT premise from the original CMP-Turing version of this
project (see git history / README for that version). CMP mining cards
(Turing TU10x) had a driver-enforced FFMA/Tensor-Core instruction-pattern
throttle that made even trivial ops slow unless you dodged it with
alternate instruction sequences -- that doesn't exist on a consumer RDNA3
card. The reason to intercept these particular ops on RX 7900 XTX is
ordinary kernel-selection: pick the best available ROCm kernel per op
instead of whatever `torch.nn.functional` currently uses by default, for
the small set of ops that dominate diffusion/LLM inference time. Every op
NOT patched here (conv3d, embedding, softmax, upsample, grouped conv2d,
etc.) is intentionally left on stock PyTorch/ROCm (rocBLAS/hipBLASLt/MIOpen)
-- unlike on Turing, that stock path is not throttled, so there's no
correctness or performance reason to replace it until it's actually
measured as a bottleneck.

Native fp16 AND bf16 support (RDNA3's WMMA units have native bf16 matrix
throughput, unlike Turing -- no bf16->fp32 conversion at any boundary here).
fp32 inputs fall back to stock ops for linear/matmul/bmm/conv2d (aiter's
Triton kernels are fp16/bf16 only); group_norm and every TE-backed op
handle fp32 natively.

linear_fp16/bmm_fp16/conv2d_fp16/group_norm/linear_int8 are called through
amd_tuned_torch/compile_ops.py, not aiter_ops.py/the native extension directly --
each is registered as a torch.library.custom_op with a register_fake
shape-only stand-in (ported from NVIDIA TransformerEngine's own
torch.compile glue, transformer_engine/pytorch/attention/custom_ops.py),
so torch.compile can place a call to one of these as a real graph node
instead of graph-breaking trying to inline through a raw Triton kernel
launcher or pybind11 extension call (linear_int8 additionally avoids
Dynamo tracing into its own per-weight quantization cache lookup -- see
compile_ops.py). Numerically and behaviorally identical either way -- this
only affects whether torch.compile can trace through the call, not what
it computes.

amd_tuned_torch.magcache is a different kind of thing from everything above: not a
kernel swap, and not RDNA3/ROCm-specific at all. It's a generic port of
MagCache's (source/MagCache, arXiv:2506.09045) skip/cache decision engine
for diffusion transformer inference -- reusing a cached residual instead of
recomputing a denoising step's block stack when its output magnitude is
predictable from a calibration table. Model-agnostic by construction (it
never touches a model class or block type, just the residual tensor a
caller's own block loop already produces), so it's not installed by
enable()/disable() and has no ROCm/aiter/TE dependency -- see
amd_tuned_torch/magcache.py for usage.

amd_tuned_torch.teacache is the same idea from a different paper (TeaCache,
source/TeaCache, arXiv:2411.19108) -- reuses a cached residual the same
way, but decides whether to skip using a live, content-dependent signal
(relative L1 distance between consecutive steps' timestep-embedding
modulation, rescaled through a per-model calibrated polynomial) instead of
magcache's precomputed per-step lookup table. Same properties: not
installed by enable()/disable(), no ROCm/aiter/TE dependency -- see
amd_tuned_torch/teacache.py for usage.

amd_tuned_torch.cache generalizes teacache's mechanism beyond diffusion transformers
entirely: SimilarityCache/similarity_cached wrap *any* callable, skipping
it and reusing its last output whenever the current call's primary tensor
argument is close enough (relative L1) to the previous call's -- no step
counter, no calibration table, no residual structure required. Gated by
its own AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE flag (default off) on top of never
being auto-installed -- this is the least safe-by-default piece of
amd_tuned_torch, since "close enough" as a proxy for "safe to skip" only holds for
workloads with smoothly-drifting consecutive calls; see amd_tuned_torch/cache.py's
module docstring before reaching for it. amd_tuned_torch.cache.KeyedCache is the
sibling for the opposite case -- exact, identity-keyed memoization for
workloads where reuse depends on precise input identity rather than
tolerating drift (e.g. a decoder called once per spatial chunk of a grid,
where forcing SimilarityCache's aggregate-similarity metric to work via a
tighter thresh can't eliminate coincidental false-positive matches, only
shrink their odds).

amd_tuned_torch.miopen_fallback is a different kind of thing again: not a plain
kernel swap and not a skip/reuse decision, but a crash rescue -- now with
a real kernel swap bolted onto its front. MIOpen's conv algorithm search
can fail for a specific (shape, dtype, current allocator state)
combination that has a perfectly good solution on a fresh/idle GPU --
observed directly on a Conv1d call in a long-running multi-model pipeline
(Wan2GP's ACE-Step Oobleck audio VAE), where the exact failing shape
succeeds cleanly in isolation, most likely because MIOpen's workspace
needs one contiguous free block and a session's accumulated allocator
fragmentation can starve that even with plenty of total free VRAM.
enable_conv1d_fallback() now first checks whether the call is the
depthwise-causal Conv1d pattern used by Mamba/SSM-style blocks (groups ==
channels, kernel width 2-4, padding == width - 1) and, if so, routes it
through source/kernel/causal-conv1d-amd's HIP kernel directly -- faster
than MIOpen and immune to the fragmentation failure mode, so it never even
attempts MIOpen for that shape family. Everything else still goes through
MIOpen first, retries once after torch.cuda.empty_cache() (cheap, often
enough to defragment and succeed on GPU) if that raises a miopenStatus
error, then falls back to a CPU float32 round-trip as a last resort -- see
amd_tuned_torch/miopen_fallback.py's module docstring for the full diagnosis, the
causal-conv1d fast path's exact eligibility rule, and its output-shape
caveat. Installed by default at import time
(AMD_TUNED_TORCH_MIOPEN_CONV1D_FALLBACK=0 to opt out) but still not bundled into
enable()/disable() -- it patches torch.nn.functional.conv1d directly and
is toggled independently, same as amd_tuned_torch.cache.

amd_tuned_torch.torch_compile is a different kind of optimization from everything
above: not a kernel swap or a skip/reuse decision, but whole-graph capture
via torch.compile (Dynamo + Inductor) -- the runtime analogue of what a
tool like onnxslim does ahead-of-time to a static ONNX graph (constant
folding, operator fusion), except captured live on first call and
Triton-autotuned per shape instead of rewritten once on a serialized graph.
Built on top of compile_ops.py's custom-op registrations (which just stop
amd_tuned_torch's own patched ops from graph-breaking Dynamo -- they don't compile
anything by themselves). Like similarity caching, this is never installed
by enable()/disable(): compile_module()/compile_fn() are explicit,
per-callable opt-ins, since first-call compile latency and per-shape
recompilation cost are workload-specific judgment calls. See
amd_tuned_torch/torch_compile.py's module docstring for the aggressive defaults
(max-autotune, Inductor freezing, static shapes) and ROCm-specific caveats.

amd_tuned_torch.cumesh_ops, amd_tuned_torch.flexgemm_ops,
amd_tuned_torch.nvdiffrast_ops, and amd_tuned_torch.torchsparse_ops are
four more thin adapters over locally-built external packages bundled at
third_party/{CuMesh,FlexGEMM,nvdiffrast,torchsparse} (this package is
standalone -- these dependencies live under source/cmp_ext_turing, not in
a sibling directory -- all ROCm/HIP-ported alongside this package; see
each module's own docstring for exactly what that port involved), same
shape as aiter_ops/te_ops: gated by available(), never raise at import
time. Unlike aiter_ops/te_ops, cumesh_ops, nvdiffrast_ops, and
torchsparse_ops have NO torch.nn.functional equivalent to intercept at all
-- mesh simplification/UV-unwrapping/BVH queries, differentiable
rasterization, and torchsparse's own richer SparseTensor-based sparse
convolution are simply not things F.* covers -- so, like
magcache/teacache/cache, none of them is installed by enable()/disable():
each is a plain library surface a caller building a 3D/mesh/point-cloud
pipeline on top of this package's other tuned dense ops reaches for
explicitly. torchsparse_ops specifically forces torchsparse's own
GatherScatter dataflow globally the first time its available() is checked
-- this ROCm build has no tensor-core kernel for the other two dataflows
upstream defaults to (see that module's docstring for exactly which
source files were excluded and why).

flexgemm_ops is the one exception: alongside its own plain-library sparse
3D ops (sparse_conv3d, grid_sample_3d, encode_seq/decode_seq -- no F.*
equivalent, same posture as cumesh_ops/nvdiffrast_ops), it ALSO installs a
real occupancy-gated fast path inside _patched_conv2d/_patched_conv3d
themselves: on every eligible F.conv2d/F.conv3d call, a cheap occupancy
check (flexgemm_ops.maybe_sparse_conv2d/maybe_sparse_conv3d) routes to a
sparse convolution instead of the dense native/CK/stock contest whenever
the input tensor is mostly empty. This is content-dependent, not
shape-dependent, so it is checked on every call rather than folded into
kernel_select's per-shape cache -- see the "Sparse-pixel/-voxel fast path"
comments in _patched_conv2d/_patched_conv3d and flexgemm_ops.py's own
sparse_conv2d/sparse_conv3d_from_dense docstrings for the full design. ON
by default; AMD_TUNED_TORCH_SPARSE_CONV2D=0 / AMD_TUNED_TORCH_SPARSE_CONV3D=0
disable each independently. sparse_conv2d is pure PyTorch (no native
extension involved, built from source/sparse_convolution's per-kernel-
position gather-scatter method and source/spconv's coords/feats/weight
convention -- see flexgemm_ops.py's docstring); sparse_conv3d is backed by
third_party/FlexGEMM's native HIP kernel, so its fast path additionally
requires flexgemm_ops.available().

amd_tuned_torch.flash_mm_kernel is a different kind of thing from every module
above: not a port of an external library, but a locally-authored Triton
kernel (originally prototyped in source/butterfly_matrix_kernel, copied in
here verbatim -- both copies stay in sync by hand). It implements butterfly
matrix multiplication (a structured O(L log L) alternative to a dense
Linear/matmul, L a power of 2) with every log2(L) stage fused into ONE
kernel launch instead of one launch per stage, so a row is read from GPU
memory once and written back once -- the same IO-awareness idea
FlashAttention popularized, applied here to eliminate the e-1 redundant HBM
round trips a naive per-stage implementation pays. Like cumesh_ops/
nvdiffrast_ops/torchsparse_ops, it has NO torch.nn.functional equivalent to
intercept -- butterfly multiplication is a distinct structured-matrix
primitive a model architecture would hold onto explicitly (e.g. as a
Linear-layer replacement it opts into), not a drop-in accelerant for an
existing op -- so, same as those three, it is NOT installed by
enable()/disable(): call amd_tuned_torch.flash_mm_kernel.flash_butterfly_mm(...)
directly. flash_mm_kernel.available() gates the compiled `@triton.jit` path
specifically (Triton importable + a CUDA/ROCm device visible);
flash_butterfly_mm() falls back to flash_butterfly_mm_torch, a pure-PyTorch
mirror of the identical reshape/pack algorithm, whenever it's False, so the
module is correct and usable even without Triton installed. On CUDA, when
both are available, flash_butterfly_mm() does NOT simply prefer the
compiled kernel -- it contests Triton against the torch fallback through
kernel_select (this package's own per-shape measure-once-cache-the-winner
engine, same as linear/matmul/bmm/conv2d/conv3d/group_norm/attention),
including kernel_select's numerical verification of the winner against the
fallback. This matters more here than for those other tiers: the Triton
kernel isn't merely unmeasured against a rival, it has never been run at
all (see below), so both "is it actually faster" and "does it actually
compute the right answer" are open questions kernel_select's contest
answers empirically per shape rather than by assumption.
UNVALIDATED ON REAL HARDWARE: the compiled kernel itself has never been run
(no Triton install, no GPU, in the environment it was written in) -- only
the reshape/pack algorithm is validated (bit-exact against
reference_impl.butterfly_mm_ref on CPU, see flash_mm_kernel.py's own module
docstring and its test suite). Forward-only, no backward pass.

amd_tuned_torch.rocfft_ops is a thin ctypes wrapper around the
SYSTEM-INSTALLED rocFFT library (librocfft.so, part of every ROCm install
-- NOT vendored/built from source/rocm-libraries/projects/rocfft, which
is the full ~63K-line production library and would be a build
undertaking wildly out of proportion to every other tier here), adding a
persistent per-shape PLAN CACHE and direct rocFFT C-API access
(rfftn/irfftn) for callers who want more control than torch.fft exposes.
torch.fft.rfftn/irfftn (what fftconv_ops.py already uses) already call
into this exact same rocFFT library via hipFFT on ROCm, with their own
LRU plan cache -- this module exists for the CAPABILITY gap (an
externally-owned, inspectable plan handle; rocFFT's own advanced-layout
controls torch.fft's Python API doesn't surface), not a measured
performance one, and is NOT wired into fftconv_ops.py's kernel_select
contest or any dispatch path -- call
amd_tuned_torch.rocfft_ops.rfftn/irfftn directly. UNVALIDATED ON REAL
HARDWARE like flash_mm_kernel above: every ctypes signature and enum
ordinal is transcribed directly from the vendored rocfft.h (not
guessed), and the plan-key/shape/batch arithmetic is unit-tested against
a mocked library object, but no ROCm device exists in the environment
this was written in to dlopen a real librocfft.so against -- see
rocfft_ops.py's own module docstring for exactly what is and isn't
checked without hardware.

amd_tuned_torch.rocm_env_check is advisory only -- unlike everything else
listed here, it never touches torch/ROCm behavior at all, just reads
os.environ (and this package's own tier-availability state) at import time
and warns about ROCm/HIP/MIOpen/rocBLAS/PyTorch environment variables that
individually often look harmless (usually left over from debugging a
different problem) but compound into a real performance regression once
combined with each other or with how this package actually dispatches --
e.g. HIP_LAUNCH_BLOCKING=1 serializing the exact hot-path ops enable()
patches, or AMD_TUNED_TORCH_MEASURE_KERNELS=0 silently benching a compiled
CK/hipBLASLt tier out of the linear/bmm contest. Set
AMD_TUNED_TORCH_ROCM_ENV_CHECK=0 to skip it; see
amd_tuned_torch/rocm_env_check.py's module docstring for the full rule list
and what each one is based on.

Import this package to enable the patch:

    import amd_tuned_torch   # patches torch/torch.nn.functional on import

Or control it explicitly:

    import amd_tuned_torch
    amd_tuned_torch.disable()
    amd_tuned_torch.enable()

Set AMD_TUNED_TORCH_AUTOPATCH=0 in the environment to import without patching.

TransformerEngine is DISABLED BY DEFAULT (a separate gate from
AMD_TUNED_TORCH_AUTOPATCH): set AMD_TUNED_TORCH_ENABLE_TE=1 to opt in. See te_ops.py's
module docstring for why -- a TE build that's ABI-mismatched against the
installed PyTorch/ROCm can segfault the whole process just from `import
transformer_engine.pytorch`, which no amount of try/except in this package
can catch. With TE disabled (the default), `import amd_tuned_torch` can never
crash for that reason; enable()/te_ops.available() just skip the four
TE-backed patches (attention/rms_norm/gelu/silu) and those ops stay on
stock PyTorch. Verify
`python -c "import transformer_engine.pytorch"` works cleanly on its own
before setting AMD_TUNED_TORCH_ENABLE_TE=1.

The same AMD_TUNED_TORCH_ENABLE_TE gate also governs amd_tuned_torch.te_extra_ops,
which is TransformerEngine's *other* half: kernels reached by name rather
than by patching over a stock op. Nothing in it is installed by enable().
Four families, chosen because they are the parts of TE that survive on
gfx1100 (TE's fused attention is CDNA-only and its FP8 path needs gfx94x+,
so neither is reachable here) and that no other backend in this package
already covers -- fused multi-tensor optimizer steps
(multi_tensor_adam/sgd/l2norm/scale), fused scaled softmax
(scaled_softmax/scaled_softmax_or_torch), bit-masked dropout, and ragged
sequence-layout plumbing (thd_to_bshd/bshd_to_thd/pad_rows/copy_to_kv_cache).
See te_extra_ops.py's module docstring for why those four and not others.

SAFETY
------
linear/matmul/bmm (aiter Triton kernels), conv2d/conv3d (native HIP
kernels), and group_norm (native HIP kernel) have no backward pass --
every patched call for these six falls back to the stock op whenever
autograd is live for the tensors involved, exactly like the original
Turing version's fallback behavior.

attention/rms_norm/gelu/silu are DIFFERENT: te_ops.py wraps each one in a
real torch.autograd.Function backed by TransformerEngine's own
forward/backward bindings, so these stay active under autograd -- you get
the speedup during training too, not just torch.no_grad() inference.
layer_norm shares this autograd support but isn't auto-installed regardless
(see above).

Every wrapper falls back to the stock op on RuntimeError or TypeError from
the native/aiter/TE call (unsupported shape, wrong arg type from a
pybind11/TE binding, etc.). Every aiter-backed wrapper (linear/matmul/bmm,
plus conv2d's aiter tier and the opt-in int8 linear) additionally catches
AssertionError, since that's how aiter's own kernels signal an unsupported
case (e.g. groups != 1) -- a bare Python assert, not a raised exception
type the other two would already cover.
"""
from __future__ import annotations

import math
import os
from typing import Any, Callable

import torch
import torch.nn.functional as F

# The compiled extension is resolved per-PyTorch rather than by ordinary
# submodule import: one editable source tree can serve several environments,
# but a .so links one specific libtorch. See amd_tuned_torch/_native_loader.py
# -- it also keeps working for the single-environment case, where the .so
# simply sits beside this file.
from . import _native_loader

_package_dir = os.path.dirname(os.path.abspath(__file__))

_C = _native_loader.load(__name__, _package_dir)
_native = _C  # so `from . import _native` elsewhere in the package resolves

# CK and hipBLASLt each got split into their own extension (see setup.py and
# src/ck_native.cpp / src/hipblaslt_native.cpp) so that rebuilding one never
# relinks the core _native extension or each other. Both are optional tiers
# (required=False): a build predating this split, or one that never enabled
# either, degrades to "tier unavailable" here -- amd_tuned_torch/ck_ops.py,
# ck_gemm_ops.py, ck_norm_ops.py and hipblaslt_ops.py all already handle
# their own `_C` being None the same way they handle has_ck()/
# has_hipblaslt() reporting False (AttributeError -> unavailable), so no
# further guard is needed here.
_C_CK = _native_loader.load(__name__, _package_dir, module_name="_native_ck", required=False)
_native_ck = _C_CK  # so `from . import _native_ck` in ck_ops.py etc. resolves

_C_HIPBLASLT = _native_loader.load(__name__, _package_dir, module_name="_native_hipblaslt",
                                    required=False)
_native_hipblaslt = _C_HIPBLASLT  # so `from . import _native_hipblaslt` in hipblaslt_ops.py resolves

# Same split-extension, optional-tier posture as CK/hipBLASLt above (see
# setup.py's rocSPARSE tier and src/rocsparse_native.cpp) -- rocsparse_ops.py
# already handles its own `_C` being None the same way hipblaslt_ops.py
# handles has_hipblaslt() reporting False.
_C_ROCSPARSE = _native_loader.load(__name__, _package_dir, module_name="_native_rocsparse",
                                    required=False)
_native_rocsparse = _C_ROCSPARSE  # so `from . import _native_rocsparse` in rocsparse_ops.py resolves


def native_build_info() -> str:
    """Which _native*.so extensions are loaded and why each was chosen."""
    return "\n".join(
        _native_loader.describe(name)
        for name in ("_native", "_native_ck", "_native_hipblaslt", "_native_rocsparse")
    )

from . import rocm_env_check
from . import te_ops
from . import te_extra_ops
from . import aiter_ops
from . import flash_attn_rocwmma_ops
from . import triton_kernels_ops
from . import mla_ops
from . import fused_norm_ops
from . import rope_ops
from . import fused_ce_ops
from . import swiglu_ops
from . import splitk_gemm_ops
from . import ck_ops
from . import ck_gemm_ops
from . import ck_norm_ops
from . import hipblaslt_ops
from . import kernel_select
from . import compile_ops
from . import magcache
from . import teacache
from . import cache
from . import hub_ops
from . import torch_compile
from . import miopen_fallback
from . import cumesh_ops
from . import sparse_conv_calibration
from . import flexgemm_ops
from . import rocsparse_ops
from . import fftconv_calibration
from . import fftconv_ops
from . import nvdiffrast_ops
from . import torchsparse_ops
from . import flash_mm_kernel
from . import rocfft_ops
from . import boft_ops
from . import liger_group_norm_ops



# The core always-on dispatch tiers (enable()/disable(), linear/matmul/bmm,
# conv2d/conv3d, group_norm, TE-backed wrappers) and the opt-in-only tiers
# plus standalone ops live in their own files now -- split out of what used
# to be one ~2300-line __init__.py purely for readability, no behavior
# change. Both use __all__ to re-export every name (including
# underscore-prefixed ones existing tests/tools reach into directly, e.g.
# amd_tuned_torch._grad_safe) at this package's top level, so
# amd_tuned_torch.X resolves exactly as it did when everything lived here.
from ._dispatch import *  # noqa: F401,F403
from ._opt_in_tiers import *  # noqa: F401,F403

if os.environ.get("AMD_TUNED_TORCH_AUTOPATCH", "1") != "0":
    enable()

    # Both of these are nested under AUTOPATCH, not independent of it:
    # AUTOPATCH=0 is this package's existing "do not touch torch at
    # import time at all" contract (relied on by tests/conftest.py to
    # keep the test suite import-time-side-effect-free), and
    # enable_flash_attn_rocwmma() in particular can trigger a real
    # JIT hipcc/rocWMMA compile on first call (see
    # flash_attn_rocwmma_ops._ensure_loaded) -- that must never fire
    # just because someone imported this package, independent of whether
    # they asked for auto-patching at all. Each still has its own
    # opt-out on top of AUTOPATCH, for a user who wants the GEMM/conv/norm
    # tiers auto-installed but not these two specifically.
    #
    # Default OFF. This was default-ON, on the reasoning that
    # enable_flash_attn_rocwmma() degrades to a warning + no-op whenever the
    # kernel does not work, so default-on "costs nothing on a machine where
    # it doesn't work". That reasoning assumed the only failure mode was the
    # kernel being ABSENT or failing to build. It has now been run on the
    # hardware it was written for (gfx1100 / RX 7900 XTX) via
    # tools/bench_flash_attn_rocwmma.py, and it is present, builds, runs --
    # and is wrong:
    #
    #   * BACKWARD is numerically broken. dQ/dK/dV max error against an fp32
    #     reference is 1e-1 to 2.9 across all 12 benchmarked shapes, where
    #     stock's error on the same shapes is 1e-4 to 2e-2. An absolute
    #     gradient error of 2.4 is not precision noise; training on it
    #     silently produces a different model.
    #   * FORWARD is numerically fine but SLOWER than stock on every shape
    #     measured -- 0.36x to 0.71x. There is no shape where it wins.
    #
    # So the kernel degrades neither loudly nor safely: it returns wrong
    # gradients at a speed nobody wanted. A tier that can only lose does not
    # belong on by default. Set AMD_TUNED_TORCH_FLASH_ATTN_ROCWMMA=1 to opt
    # in anyway (e.g. to re-measure it after fixing the backward).
    if os.environ.get("AMD_TUNED_TORCH_FLASH_ATTN_ROCWMMA", "0") == "1":
        enable_flash_attn_rocwmma()

    # Default ON for the same reason as flash_attn_rocwmma above: no-ops
    # with a warning rather than failing when triton_kernels isn't
    # installed (the default state of this checkout -- see
    # triton_kernels_ops.py). Set AMD_TUNED_TORCH_TRITON_KERNELS_RMSNORM=0
    # to opt out.
    if os.environ.get("AMD_TUNED_TORCH_TRITON_KERNELS_RMSNORM", "1") != "0":
        enable_triton_kernels_rmsnorm()

    # Default OFF, and for a different reason than the two above: this one
    # patches `peft`, not torch, so enable_boft() has to import peft to do
    # anything -- and peft pulls transformers in behind it. Defaulting it on
    # would put that whole import cost on every program that imports this
    # package, including the overwhelming majority that never touch BOFT.
    # It is also the one tier here whose effect is order-sensitive
    # (BOFTLayer reads get_fbd_cuda() once, at construction), so a caller
    # who cares generally wants to place the call themselves. Set
    # AMD_TUNED_TORCH_BOFT=1 to opt in, or just call enable_boft() before
    # building the adapter.
    if os.environ.get("AMD_TUNED_TORCH_BOFT", "0") == "1":
        enable_boft()
