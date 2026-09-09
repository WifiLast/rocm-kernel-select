"""torch.compile / Dynamo integration for amd_tuned_torch's aiter- and native-backed
ops (linear_fp16, bmm_fp16, conv2d_fp16, group_norm, conv2d_native,
conv3d_native, linear_int8).

Ports the pattern from NVIDIA TransformerEngine's
transformer_engine/pytorch/attention/custom_ops.py -- its own docstring
says exactly why it exists: "Attention kernels wrapped as custom ops, so
they don't graph-break under torch.compile." Without this, Dynamo has no
special-case recognition for amd_tuned_torch's monkeypatched F.linear/torch.matmul/
F.conv2d/F.group_norm (that fast-path match is by identity against the
*original* torch function, lost the moment it's rebound), so it falls back
to inlining each as a plain Python function. Inlining succeeds through the
_grad_safe/_usable guard logic, but then hits a call into an aiter Triton
kernel launcher or amd_tuned_torch's own pybind11 group_norm extension -- neither
is a registered PyTorch op, so Dynamo can't trace through that boundary.
Result: a graph break at every one of these calls under plain
torch.compile(model) (still correct, just fragments the graph and loses
fusion across it), or a hard torch._dynamo.exc.Unsupported under
fullgraph=True.

torch.library.custom_op + register_fake fixes this the same way NVIDIA's
code does: the custom_op registration gives Dynamo a real op with a schema
to place as an opaque graph node (no inlining attempted), and register_fake
is a shape/dtype-only stand-in -- no real computation -- that FakeTensor
propagation calls instead of the real kernel while tracing. All six ops
here (excluding linear_int8, see below) are pure functions of their input
shapes (no data-dependent output shape), so each fake impl is a few lines --
conv2d_native/conv3d_native reuse aten's own conv2d/conv3d meta kernels for
the fake, same trick conv2d_fp16 already uses.

Unlike NVIDIA's version, these aren't registered with device_types="cuda":
amd_tuned_torch.__init__._usable() already checks `tensor.is_cuda` (true for ROCm
HIP tensors too, which PyTorch reports under the "cuda" dispatch key for
compatibility) before any of these ever get called, so a device restriction
at the op-registration level would be redundant -- and it would also make
these ops uncallable with plain CPU tensors, unlike every other piece of
dispatch logic in this project's test suite (aiter/TE are always mocked
out, exercised with CPU tensors and force_eligible()). Registering for all
device types keeps that same testability without weakening real eligibility
gating, which still happens exactly where it always has.

None of this changes the autograd story documented in amd_tuned_torch/__init__.py:
these six ops still have no backward pass. That's fine and unchanged --
amd_tuned_torch.__init__._grad_safe() already guarantees these are only ever called
in a context where no backward pass is needed (no_grad/inference_mode, or
no input actually requires_grad), same invariant as before this module
existed; torch.library.custom_op ops with no registered autograd formula
work correctly under exactly that condition.

linear_int8 (see amd_tuned_torch/aiter_ops.py) needs this more than the other six:
its Python body does a data-dependent WeakKeyDictionary lookup per call
(the cached per-weight quantized tensor/scale) before ever reaching
aiter.gemm_a8w8 -- without an opaque custom-op boundary, Dynamo would try
to trace or guard on that cache lookup directly (at best a graph break
right there, same as the other six uncompiled; at worst a guard baked in
against one specific cache state). Wrapping the whole call -- cache lookup
included -- as one opaque op sidesteps that the same way it sidesteps
tracing into the aiter kernel launcher for the other six; the op still
runs its real Python body (cache lookup included) every time it's
actually invoked, this only changes what Dynamo does while *tracing*.

Falls back to plain function calls (today's behavior, numerically
identical, just without the torch.compile graph-node benefit) on PyTorch
builds without torch.library.custom_op (added in PyTorch 2.4). Either way,
amd_tuned_torch.compile_ops.linear_fp16/bmm_fp16/conv2d_fp16/group_norm/
conv2d_native/conv3d_native/linear_int8 are always safe to call -- callers
never need to know which path is active underneath.

hipblaslt_linear/ck_gemm_linear/hipblaslt_bmm (added later, same idea
applied to amd_tuned_torch.hipblaslt_ops.linear/amd_tuned_torch.ck_gemm_ops.linear/
amd_tuned_torch.hipblaslt_ops.bmm -- the GEMM contest candidates
_patched_linear/_patched_bmm/_patched_matmul call directly once
kernel_select has already picked a winner for a shape, the same
monkeypatched-op hot path reasoning as every op above) needed one more
piece the seven above didn't: those three can legitimately return None
(the installed navi31 logic has no kernel for some dtype/epilogue/shape
combination -- see hipblaslt_ops.py's own docstring), where every op above
always succeeds once called. torch.library.infer_schema has no mapping for
a bare `Optional[torch.Tensor]` return annotation (raises ValueError,
confirmed empirically against this PyTorch version), so these three pass
an explicit `schema=".. -> Tensor?"` string instead of relying on
inference -- PyTorch's schema language itself supports an optional Tensor
return perfectly well, it is only the Python-type-annotation-based
inference path that doesn't. Each op's register_fake replicates the same
call's own eligibility check (hipblaslt_ops.is_linear_eligible/
is_bmm_eligible, ck_gemm_ops.is_eligible -- pulled out of linear/bmm into
their own functions specifically so the fake can reuse them verbatim
instead of duplicating the logic) to decide whether the fake should return
a shaped stand-in or None too; every one of those checks is static tensor
metadata (dtype/dim/is_cuda/shape), so it is exactly as valid to evaluate
under FakeTensorMode as for real. What the fake CANNOT replicate is a
decline made only inside the native call itself (an installed navi31 logic
gap for one specific shape/dtype/epilogue combination) -- in practice this
doesn't bite the compiled hot path: kernel_select only ever calls one of
these three directly, without going through kernel_select.pick's contest
again, for a (dtype, shape) key that has ALREADY been measured to succeed
with that exact candidate, and kernel selection inside hipBLASLt/CK is
itself shape/dtype-deterministic, so a key that won once keeps winning.
"""
from __future__ import annotations

from typing import Optional

import torch

from . import aiter_ops
from . import ck_gemm_ops
from . import hipblaslt_ops
from . import _native as _C

_HAS_CUSTOM_OP = hasattr(torch.library, "custom_op")


def _already_registered(name: str) -> bool:
    return hasattr(torch.ops, "amd_tuned_torch") and hasattr(torch.ops.amd_tuned_torch, name)


if _HAS_CUSTOM_OP and not _already_registered("linear_fp16"):

    @torch.library.custom_op("amd_tuned_torch::linear_fp16", mutates_args=())
    def _linear_fp16_op(
        input_: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return aiter_ops.linear_fp16(input_, weight, bias)

    @_linear_fp16_op.register_fake
    def _linear_fp16_fake(
        input_: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        del bias
        return input_.new_empty(*input_.shape[:-1], weight.shape[0])

    @torch.library.custom_op("amd_tuned_torch::bmm_fp16", mutates_args=())
    def _bmm_fp16_op(input_: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
        return aiter_ops.bmm_fp16(input_, mat2)

    @_bmm_fp16_op.register_fake
    def _bmm_fp16_fake(input_: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
        return input_.new_empty(input_.shape[0], input_.shape[1], mat2.shape[-1])

    @torch.library.custom_op("amd_tuned_torch::conv2d_fp16", mutates_args=())
    def _conv2d_fp16_op(
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        stride: list[int],
        padding: list[int],
        dilation: list[int],
    ) -> torch.Tensor:
        return aiter_ops.conv2d_fp16(input_, weight, bias, stride, padding, dilation)

    @_conv2d_fp16_op.register_fake
    def _conv2d_fp16_fake(
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        stride: list[int],
        padding: list[int],
        dilation: list[int],
    ) -> torch.Tensor:
        # aten::conv2d has its own meta/fake kernel with the real output-size
        # arithmetic -- reuse it instead of re-deriving stride/padding/dilation
        # math by hand here.
        return torch.ops.aten.conv2d.default(
            input_, weight, bias, stride, padding, dilation, 1
        )

    @torch.library.custom_op("amd_tuned_torch::group_norm", mutates_args=())
    def _group_norm_op(
        input_: torch.Tensor,
        num_groups: int,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        eps: float,
    ) -> torch.Tensor:
        return _C.group_norm(input_, num_groups, weight, bias, eps)

    @_group_norm_op.register_fake
    def _group_norm_fake(
        input_: torch.Tensor,
        num_groups: int,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        eps: float,
    ) -> torch.Tensor:
        del num_groups, weight, bias, eps
        return input_.new_empty(input_.shape)  # group_norm is shape-preserving

    @torch.library.custom_op("amd_tuned_torch::conv2d_native", mutates_args=())
    def _conv2d_native_op(
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        stride: list[int],
        padding: list[int],
        dilation: list[int],
    ) -> torch.Tensor:
        return _C.conv2d(input_, weight, bias, stride, padding, dilation)

    @_conv2d_native_op.register_fake
    def _conv2d_native_fake(
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        stride: list[int],
        padding: list[int],
        dilation: list[int],
    ) -> torch.Tensor:
        return torch.ops.aten.conv2d.default(
            input_, weight, bias, stride, padding, dilation, 1
        )

    @torch.library.custom_op("amd_tuned_torch::conv3d_native", mutates_args=())
    def _conv3d_native_op(
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        stride: list[int],
        padding: list[int],
        dilation: list[int],
    ) -> torch.Tensor:
        return _C.conv3d(input_, weight, bias, stride, padding, dilation)

    @_conv3d_native_op.register_fake
    def _conv3d_native_fake(
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        stride: list[int],
        padding: list[int],
        dilation: list[int],
    ) -> torch.Tensor:
        return torch.ops.aten.conv3d.default(
            input_, weight, bias, stride, padding, dilation, 1
        )

    @torch.library.custom_op("amd_tuned_torch::linear_int8", mutates_args=())
    def _linear_int8_op(
        input_: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return aiter_ops.linear_int8(input_, weight, bias)

    @_linear_int8_op.register_fake
    def _linear_int8_fake(
        input_: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # Same output shape as linear_fp16's fake -- linear_int8 is still
        # F.linear's shape contract, only the internal compute differs.
        del bias
        return input_.new_empty(*input_.shape[:-1], weight.shape[0])

    @torch.library.custom_op(
        "amd_tuned_torch::hipblaslt_linear", mutates_args=(),
        schema="(Tensor input, Tensor weight, Tensor? bias, int epilogue) -> Tensor?")
    def _hipblaslt_linear_op(
        input_: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], epilogue: int,
    ) -> Optional[torch.Tensor]:
        return hipblaslt_ops.linear(input_, weight, bias, epilogue)

    @_hipblaslt_linear_op.register_fake
    def _hipblaslt_linear_fake(
        input_: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], epilogue: int,
    ) -> Optional[torch.Tensor]:
        del epilogue
        if not hipblaslt_ops.is_linear_eligible(input_, weight, bias):
            return None
        return input_.new_empty(*input_.shape[:-1], weight.shape[0])

    @torch.library.custom_op(
        "amd_tuned_torch::ck_gemm_linear", mutates_args=(),
        schema="(Tensor input, Tensor weight, Tensor? bias, int epilogue) -> Tensor?")
    def _ck_gemm_linear_op(
        input_: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], epilogue: int,
    ) -> Optional[torch.Tensor]:
        return ck_gemm_ops.linear(input_, weight, bias, epilogue)

    @_ck_gemm_linear_op.register_fake
    def _ck_gemm_linear_fake(
        input_: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], epilogue: int,
    ) -> Optional[torch.Tensor]:
        del epilogue
        if not ck_gemm_ops.is_eligible(input_, weight, bias):
            return None
        return input_.new_empty(*input_.shape[:-1], weight.shape[0])

    @torch.library.custom_op(
        "amd_tuned_torch::hipblaslt_bmm", mutates_args=(),
        schema="(Tensor input, Tensor mat2) -> Tensor?")
    def _hipblaslt_bmm_op(input_: torch.Tensor, mat2: torch.Tensor) -> Optional[torch.Tensor]:
        return hipblaslt_ops.bmm(input_, mat2)

    @_hipblaslt_bmm_op.register_fake
    def _hipblaslt_bmm_fake(input_: torch.Tensor, mat2: torch.Tensor) -> Optional[torch.Tensor]:
        if not hipblaslt_ops.is_bmm_eligible(input_, mat2):
            return None
        return input_.new_empty(input_.shape[0], input_.shape[1], mat2.shape[-1])

    _linear_fp16_impl = torch.ops.amd_tuned_torch.linear_fp16
    _bmm_fp16_impl = torch.ops.amd_tuned_torch.bmm_fp16
    _conv2d_fp16_impl = torch.ops.amd_tuned_torch.conv2d_fp16
    _group_norm_impl = torch.ops.amd_tuned_torch.group_norm
    _conv2d_native_impl = torch.ops.amd_tuned_torch.conv2d_native
    _conv3d_native_impl = torch.ops.amd_tuned_torch.conv3d_native
    _linear_int8_impl = torch.ops.amd_tuned_torch.linear_int8
    _hipblaslt_linear_impl = torch.ops.amd_tuned_torch.hipblaslt_linear
    _ck_gemm_linear_impl = torch.ops.amd_tuned_torch.ck_gemm_linear
    _hipblaslt_bmm_impl = torch.ops.amd_tuned_torch.hipblaslt_bmm

elif _already_registered("linear_fp16"):
    # Already registered by an earlier import of this module in this
    # process (sys.modules caching normally prevents this, but stay
    # idempotent rather than raising on a stray reload).
    _linear_fp16_impl = torch.ops.amd_tuned_torch.linear_fp16
    _bmm_fp16_impl = torch.ops.amd_tuned_torch.bmm_fp16
    _conv2d_fp16_impl = torch.ops.amd_tuned_torch.conv2d_fp16
    _group_norm_impl = torch.ops.amd_tuned_torch.group_norm
    _conv2d_native_impl = torch.ops.amd_tuned_torch.conv2d_native
    _conv3d_native_impl = torch.ops.amd_tuned_torch.conv3d_native
    _linear_int8_impl = torch.ops.amd_tuned_torch.linear_int8
    _hipblaslt_linear_impl = torch.ops.amd_tuned_torch.hipblaslt_linear
    _ck_gemm_linear_impl = torch.ops.amd_tuned_torch.ck_gemm_linear
    _hipblaslt_bmm_impl = torch.ops.amd_tuned_torch.hipblaslt_bmm

else:
    # PyTorch build predates torch.library.custom_op (added in 2.4) --
    # same numerics, just no torch.compile graph-node benefit.
    _linear_fp16_impl = aiter_ops.linear_fp16
    _bmm_fp16_impl = aiter_ops.bmm_fp16
    _conv2d_fp16_impl = aiter_ops.conv2d_fp16
    _group_norm_impl = _C.group_norm
    _conv2d_native_impl = _C.conv2d
    _conv3d_native_impl = _C.conv3d
    _linear_int8_impl = aiter_ops.linear_int8
    _hipblaslt_linear_impl = hipblaslt_ops.linear
    _ck_gemm_linear_impl = ck_gemm_ops.linear
    _hipblaslt_bmm_impl = hipblaslt_ops.bmm


def linear_fp16(
    input_: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return _linear_fp16_impl(input_, weight, bias)


def bmm_fp16(input_: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    return _bmm_fp16_impl(input_, mat2)


def conv2d_fp16(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride=1,
    padding=0,
    dilation=1,
) -> torch.Tensor:
    # The custom op's schema declares stride/padding/dilation as list[int];
    # callers (amd_tuned_torch.__init__._patched_conv2d) may pass a bare int (e.g.
    # stride=1) the same way F.conv2d itself accepts -- normalize here so
    # the registered op always sees an actual list regardless.
    return _conv2d_fp16_impl(
        input_, weight, bias,
        list(aiter_ops._pair(stride)),
        list(aiter_ops._pair(padding)),
        list(aiter_ops._pair(dilation)),
    )


def group_norm(
    input_: torch.Tensor,
    num_groups: int,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    return _group_norm_impl(input_, num_groups, weight, bias, eps)


def conv2d_native(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride=1,
    padding=0,
    dilation=1,
) -> torch.Tensor:
    """Hand-written HIP Conv2d (src/cuda/conv2d_fp{16,32}.cu, groups=1,
    fp16/fp32 only) -- see amd_tuned_torch.__init__._patched_conv2d for where
    this sits relative to aiter's Triton conv2d_fp16 (bf16 support, and the
    fallback when this raises) and stock."""
    return _conv2d_native_impl(
        input_, weight, bias,
        list(aiter_ops._pair(stride)),
        list(aiter_ops._pair(padding)),
        list(aiter_ops._pair(dilation)),
    )


def conv3d_native(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride=1,
    padding=0,
    dilation=1,
) -> torch.Tensor:
    """Hand-written HIP Conv3d (src/cuda/conv3d_fp{16,32}.cu, groups=1,
    fp16/fp32 only). Neither aiter nor TransformerEngine cover conv3d at
    all, so this has no fallback besides stock."""
    def _triple(v):
        return (v, v, v) if isinstance(v, int) else tuple(v)

    return _conv3d_native_impl(
        input_, weight, bias, list(_triple(stride)), list(_triple(padding)), list(_triple(dilation)),
    )


def conv3d_fp16_winograd_bt8_bc8(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride=1,
    padding=0,
    dilation=1,
) -> Optional[torch.Tensor]:
    """Testing/opt-in-only: direct call to the codegen'd F(2x2x2,3x3x3)
    Winograd fp16 conv3d kernel (src/cuda/templates/conv3d_fp16_winograd.cu.tmpl,
    see tools/kernelgen/). Returns None (not a Tensor) if the shape is
    outside that kernel's narrow scope (kernel=3x3x3, stride=1, padding=1,
    dilation=1, batch=1, even D/H/W, tile-count/C_out divisible by 8) --
    see amd_tuned_torch.__init__._is_winograd_eligible_conv3d for the exact
    guard mirrored in Python, and amd_tuned_torch.enable_conv3d_winograd_fp16
    for how this gets wired into F.conv3d.

    Unlike conv2d_native/conv3d_native above, this is NOT registered as a
    torch.library.custom_op -- no torch.compile/Dynamo custom-op support,
    so torch.compile(fullgraph=True) will graph-break on it. An opt-in,
    numerically-unvalidated kernel (see that template's header) doesn't
    need that complexity yet; add it if/when this graduates to an
    always-on tier the way conv2d_native/conv3d_native are."""
    def _triple(v):
        return (v, v, v) if isinstance(v, int) else tuple(v)

    return _C.conv3d_fp16_winograd_bt8_bc8(
        input_, weight, bias, list(_triple(stride)), list(_triple(padding)), list(_triple(dilation)),
    )


def linear_int8(
    input_: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return _linear_int8_impl(input_, weight, bias)


def hipblaslt_linear(
    input_: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
    epilogue: int = hipblaslt_ops.EPILOGUE_NONE,
) -> Optional[torch.Tensor]:
    """Compile-safe wrapper for hipblaslt_ops.linear -- see this module's
    docstring for why it (unlike every op above) can return None."""
    return _hipblaslt_linear_impl(input_, weight, bias, int(epilogue))


def ck_gemm_linear(
    input_: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
    epilogue: int = hipblaslt_ops.EPILOGUE_NONE,
) -> Optional[torch.Tensor]:
    """Compile-safe wrapper for ck_gemm_ops.linear -- see this module's
    docstring for why it (unlike every op above) can return None."""
    return _ck_gemm_linear_impl(input_, weight, bias, int(epilogue))


def hipblaslt_bmm(input_: torch.Tensor, mat2: torch.Tensor) -> Optional[torch.Tensor]:
    """Compile-safe wrapper for hipblaslt_ops.bmm -- see this module's
    docstring for why it (unlike every op above) can return None."""
    return _hipblaslt_bmm_impl(input_, mat2)
