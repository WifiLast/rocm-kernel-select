"""BOFT (Butterfly Orthogonal Fine-Tuning) fast_block_diag kernel for
amd_tuned_torch, plus a monkeypatch to make source/peft's own BOFT tuner use
it.

WHERE THIS COMES FROM. `source/peft` is a full, unmodified checkout of
upstream huggingface/peft. Its `peft.tuners.boft` implements BOFT (Qiu et
al., "Parameter-Efficient Orthogonal Finetuning via Butterfly
Factorization") -- a LoRA ALTERNATIVE that reparameterizes a frozen weight's
update as a product of orthogonal butterfly-block-diagonal factors instead
of a low-rank `A @ B`. Its only piece of compiled code is
`peft/tuners/boft/fbd/{fbd_cuda.cpp,fbd_cuda_kernel.cu}` -- a small
(137-line), OPTIONAL CUDA extension implementing exactly one op: assemble a
batch of small (b, b) blocks into one big block-diagonal matrix (forward)
and the inverse gather (backward). This module is that same kernel, ported
into amd_tuned_torch's own always-built native extension
(src/cuda/fast_block_diag.cu, registered in src/main_rocm.cpp) rather than
left as PEFT's own runtime JIT build.

ALGORITHM (unchanged from upstream -- see src/cuda/fast_block_diag.cu's own
header for the full index arithmetic). Forward: input [z, N, b, b] -> output
[z, N*b, N*b], scattering block i onto the diagonal at [z, i*b:(i+1)*b,
i*b:(i+1)*b] -- i.e. `block_diag(input[z, 0], ..., input[z, N-1])` for every
z at once. Backward is the exact inverse gather. Pure data movement, no
arithmetic -- the compiled kernel exists purely to avoid materializing an
`(N*b, N*b)` zero tensor and writing N*b*b elements into it from Python
(`torch.block_diag(*torch.unbind(...))`, PEFT's own portable fallback, does
exactly that, one Python-level call per block).

WHY A SEPARATE PORT INSTEAD OF JUST USING PEFT'S OWN EXTENSION AS-IS. PEFT's
`get_fbd_cuda()` (peft/tuners/boft/layer.py) calls
`torch.utils.cpp_extension.load(...)` the first time any BOFTLayer is
constructed -- a real JIT compile, needing ninja and a working host compiler
AT THAT MOMENT, in whatever environment happens to be running training. A
missing ninja/toolchain is not a hard error there -- it warns and falls back
to the pure-PyTorch path -- but it is a real point of failure (a container
or CI image without a C++ toolchain) and a multi-second stall the first time
it works. This package already builds and ships the identical kernel as
part of its own extension (built once, at `pip install -e .` time, like
every other tier here) -- `patch_peft_boft()` below points BOFT at that
instead, so using BOFT together with amd_tuned_torch needs no separate JIT
step at all.

WHAT THIS FIXES ALONG THE WAY. Upstream's kernel dispatches dtype via
`AT_DISPATCH_FLOATING_TYPES_AND_HALF`, which is fp16/fp32/fp64 -- NOT
bfloat16. Nothing in `peft/tuners/boft/layer.py` guards against that: if the
extension happens to be built and a BOFT adapter runs in bf16 (a completely
ordinary choice on this project's own target hardware), upstream's kernel
raises at that specific call rather than falling back, since the
fbd_cuda_available decision was already made (at layer-construction time,
based only on whether the extension *built*) before any dtype was known.
This is an upstream bug, not a ROCm-specific one. `src/cuda/fast_block_diag.
cu` adds a real bf16 launcher (pure data movement, so there is no numerical
risk in doing so -- a bf16 element copies exactly, no rounding involved),
closing the gap for a BOFT adapter run through this port.

VALIDATION STATUS -- READ BEFORE TRUSTING THIS ON HARDWARE. Same posture as
every other port in this package written without ROCm hardware available:
the CUDA source was translated line-for-line from PEFT's own
fbd_cuda_kernel.cu (identical index arithmetic, only the dispatch mechanism
changed from a single AT_DISPATCH template to four typed launchers matching
this package's own group_norm.cu/conv2d_fp32.cu convention, plus the added
bf16 launcher), but the compiled kernel has never been built or run here --
no hipcc, no GPU in this dev environment. The operation itself carries very
little numerical risk (pure data movement, no floating-point arithmetic at
all), but "never compiled" still means never compiled -- build the
extension and cross-check `fast_block_diag(x)` against
`torch.block_diag(*torch.unbind(x, dim=0)... )`-style construction (or
simply PEFT's own pure-Python fallback path, `fbd_cuda_available=False`)
for a real 4D input before relying on it.

USAGE. `fast_block_diag(input)` works standalone (any CUDA/ROCm tensor,
fp16/bf16/fp32/fp64, real backward via `FastBlockDiag`). To make PEFT's OWN
BOFT tuner use this kernel instead of its runtime JIT build, call
`patch_peft_boft()` once, BEFORE constructing any BOFT layer (see that
function's own docstring for exactly why timing matters here)::

    import amd_tuned_torch
    amd_tuned_torch.boft_ops.patch_peft_boft()

    from peft import BOFTConfig, get_peft_model
    model = get_peft_model(model, BOFTConfig(...))
"""
from __future__ import annotations

import torch

from . import _native as _C


def available() -> bool:
    """True if the extension exposes both fast_block_diag kernels.

    Always True for any build of this package in practice -- unlike
    hipblaslt_ops/rocsparse_ops this is a CORE-extension kernel with no
    optional external dependency to be missing (see setup.py: it's built
    alongside group_norm/conv2d/conv3d unconditionally) -- but checked for
    real rather than assumed, matching every other *_ops module's own
    posture, and so a build predating this kernel's addition degrades
    cleanly instead of raising AttributeError deep inside FastBlockDiag."""
    return hasattr(_C, "fast_block_diag_forward") and hasattr(_C, "fast_block_diag_backward")


class FastBlockDiag(torch.autograd.Function):
    """Same contract as PEFT's own peft.tuners.boft.layer.FastBlockDiag,
    backed by this package's compiled kernel: input [z, N, b, b] ->
    block_diag(input[z, 0], ..., input[z, N-1]) per z, shape
    [z, N*b, N*b]. See this module's docstring for the algorithm and
    provenance."""

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return _C.fast_block_diag_forward(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        return _C.fast_block_diag_backward(grad_output, input)


def fast_block_diag(input: torch.Tensor) -> torch.Tensor:
    """block_diag(input[z, 0], ..., input[z, N-1]) for every z at once --
    [z, N, b, b] -> [z, N*b, N*b], real backward via FastBlockDiag.

    Unlike most *_ops functions in this package, this does NOT return None
    on ineligibility -- there is no stock-fallback contract to honour here
    (this op has no F.* equivalent to decline in favour of); shape/dtype
    validation happens on the C++ side (src/main_rocm.cpp's
    custom_fast_block_diag_forward/backward) via ordinary TORCH_CHECK,
    which raises like any other torch op given bad input. PEFT's own
    layer.py already supplies the fallback for when this kernel isn't
    available at all (torch.block_diag) -- see patch_peft_boft()."""
    return FastBlockDiag.apply(input)


def patch_peft_boft() -> bool:
    """Monkeypatch source/peft's peft.tuners.boft.layer module so BOFT uses
    THIS package's compiled fast_block_diag kernel instead of its own
    runtime torch.utils.cpp_extension.load() JIT build. See this module's
    docstring's WHY section for the full motivation (no separate build
    step, and the bf16 fix).

    HOW. peft.tuners.boft.layer.BOFTLayer's own FastBlockDiag.forward/
    backward call `get_fbd_cuda().forward(input)[0]` /
    `get_fbd_cuda().backward(grad_output, input)[0]` -- this replaces
    get_fbd_cuda (and the module-level _FBD_CUDA cache it reads/writes) with
    a tiny shim exposing that exact .forward(...)/.backward(...) -> [tensor]
    interface, backed by this module's own compiled kernel. BOFTLayer's own
    FastBlockDiag class and every call site are left completely untouched --
    only WHERE the compute happens changes, not how it's invoked. Resetting
    _FBD_CUDA (not just get_fbd_cuda) matters too: an earlier call in this
    same process (a previous BOFTLayer construction that already ran
    upstream's own JIT attempt, successfully or not) would otherwise shadow
    this patch via that cache.

    TIMING MATTERS. Each BOFTLayer decides its own `fbd_cuda_available`
    flag ONCE, in `__init__`, from whatever `get_fbd_cuda()` returns AT THAT
    MOMENT -- so this must be called BEFORE constructing any BOFTLayer /
    before `get_peft_model(model, BOFTConfig(...))`. Calling it after layers
    already exist does not retroactively speed them up (though it will
    affect any layer constructed afterward, e.g. a second adapter added
    later).

    Returns True if `peft.tuners.boft` was importable and the patch was
    applied, False if it isn't installed (nothing to patch) -- never
    raises, same optional-integration posture as every available()-style
    function in this package. Safe to call even when this module's own
    available() is False: the patch just installs a shim that will itself
    raise AttributeError on first real use in that case, exactly as calling
    fast_block_diag() directly would."""
    try:
        from peft.tuners.boft import layer as _boft_layer
    except ImportError:
        return False

    class _FbdCudaShim:
        """Stand-in for the object PEFT's own get_fbd_cuda() would JIT-build
        -- same .forward(input)/.backward(grad_output, input) -> [tensor]
        interface BOFTLayer's FastBlockDiag.forward/backward already call
        (the `[0]` indexing at those call sites is why single-element lists
        are returned here rather than bare tensors)."""

        @staticmethod
        def forward(input):
            return [_C.fast_block_diag_forward(input)]

        @staticmethod
        def backward(grad_output, input):
            return [_C.fast_block_diag_backward(grad_output, input)]

    _boft_layer._FBD_CUDA = _FbdCudaShim
    _boft_layer.get_fbd_cuda = lambda: _FbdCudaShim
    return True
