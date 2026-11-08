"""Block-scaled int8 quantisation and int8 GEMM (W8A8), TurboDiffusion layout,
as Triton kernels for ROCm.

    quant:  x [M, K] fp16/bf16  ->  q [M, K] int8,  s [ceil(M/128), ceil(K/128)] fp32
            per 128x128 block: amax = max(|x|, 1e-8), q = rint(x * 127 / amax),
            s = amax / 127
    gemm:   y[m, n] = sum over K blocks kb of
                      s_a[m // 128, kb] * s_b[n // 128, kb] * (q_a[m, kb] . q_b[n, kb])
            q_a [M, K], q_b [N, K] (weight, "TN"), y in the activation dtype

WHAT IT IS FOR: fastvideo-kernel's turbodiffusion ops (int8_quant /
int8_linear / Int8Linear). Upstream builds them from CUTLASS/CuTe
(SM80 int8 mma), which does not compile for HIP, so on ROCm those functions
called None. The AMD fastvideo-kernel (../fastvideo_kernel_amd) loads this
file in their place. Same numbers as the CUDA kernels: the scale layout, the 1e-8
amax floor and round-to-nearest-even all match quant.hpp / gemm kernel.hpp,
so weights quantised on either side load on the other.

The GEMM accumulates the int8 products of one 128-wide K block in int32
(exact; RDNA3 runs tl.dot int8 on v_wmma_i32_16x16x16_iu8) and only then
applies the two scales in fp32 -- one convert + multiply per 128 K instead
of per BLOCK_K step.

Limits: 2-D contiguous-last-dim inputs, int8 operands, fp32 scales in the
shape above, CUDA (ROCm) tensors. Anything else, or a kernel compile
failure, goes to the PyTorch references (int8_quant_reference /
int8_gemm_reference) -- same result, slower. Forward only.
AMD_TUNED_TORCH_BLOCKSCALED_INT8=0 forces the reference path.

available() gates on `triton` importability, same reasoning as
rope_ops.py / fused_norm_ops.py.

VALIDATED on gfx1100 (RX 7900 XTX, torch 2.15.0.dev+rocm7.14, Triton
3.8 AMD backend), see tests_hardware/test_blockscaled_int8_ops.py: quant is
bit-identical to the reference, GEMM within bf16 output rounding.

NOT FASTER than bf16 on RDNA3 (WMMA iu8 peaks at the same rate as WMMA
fp16/bf16, and quantising the activation is an extra pass). int8_linear vs
bf16 hipBLASLt x @ w.T, M=32760, 2026-10-26:

    N x K          bf16       int8 (quant+gemm)
    1536 x 1536    1.77 ms    2.75 ms   (0.64x)
    8960 x 1536    9.64 ms   12.65 ms   (0.76x)
    5120 x 5120   17.06 ms   24.87 ms   (0.69x)

Use it to run int8 (TurboDiffusion) checkpoints / halve weight memory, not
for speed. Relative error vs the bf16 product 1.35 %.
"""
from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

QBLOCK = 128
INT8_MAX = 127.0
AMAX_FLOOR = 1e-8
_kernel_failed = False


def available() -> bool:
    return _TRITON_AVAILABLE


def _enabled() -> bool:
    return (_TRITON_AVAILABLE and not _kernel_failed
            and os.environ.get("AMD_TUNED_TORCH_BLOCKSCALED_INT8", "1") != "0")


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


# --------------------------------------------------------------------------
# PyTorch references
# --------------------------------------------------------------------------

def int8_quant_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-128x128-block symmetric int8 quantisation of a 2-D tensor."""
    assert x.dim() == 2, "expected a 2-D tensor"
    M, K = x.shape
    MB, KB = _cdiv(M, QBLOCK), _cdiv(K, QBLOCK)
    xp = torch.zeros(MB * QBLOCK, KB * QBLOCK, dtype=torch.float32, device=x.device)
    xp[:M, :K] = x.float()
    blocks = xp.view(MB, QBLOCK, KB, QBLOCK)
    amax = blocks.abs().amax(dim=(1, 3)).clamp_min(AMAX_FLOOR)  # [MB, KB]
    # divide in float64, round once to fp32 = the correctly rounded fp32 quotient
    # (torch's fp32 GPU division on ROCm is 1 ulp off now and then)
    inv = (INT8_MAX / amax.double()).float()
    q = torch.round(blocks * inv[:, None, :, None]).clamp_(-128, 127)
    q = q.view(MB * QBLOCK, KB * QBLOCK)[:M, :K].to(torch.int8)
    return q.contiguous(), (amax.double() / INT8_MAX).float().contiguous()


def int8_gemm_reference(a_q, a_s, b_q, b_s, out_dtype=torch.float32, bias=None) -> torch.Tensor:
    """y = blockwise-dequantised a_q @ b_q^T, exact int32 per K block."""
    M, K = a_q.shape
    N = b_q.shape[0]
    y = torch.zeros(M, N, dtype=torch.float32, device=a_q.device)
    rows_a = torch.arange(M, device=a_q.device) // QBLOCK
    rows_b = torch.arange(N, device=a_q.device) // QBLOCK
    for kb in range(_cdiv(K, QBLOCK)):
        sl = slice(kb * QBLOCK, min(K, (kb + 1) * QBLOCK))
        # int8 products of one block stay below 2**24 per element, exact in fp32 too,
        # but the sum of 128 of them is not -> do the block in float64.
        part = a_q[:, sl].double() @ b_q[:, sl].double().T
        y += (part * a_s[rows_a, kb].double()[:, None] * b_s[rows_b, kb].double()[None, :]).float()
    if bias is not None:
        y += bias.float()
    return y.to(out_dtype)


# --------------------------------------------------------------------------
# Triton kernels
# --------------------------------------------------------------------------

if _TRITON_AVAILABLE:
    from triton.language.extra import libdevice as _libdevice

    @triton.jit
    def _quant_kernel(X, Q, S, M, K, stride_xm, stride_qm, KB,
                      BLOCK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        rm = pid_m * BLOCK + tl.arange(0, BLOCK)
        rk = pid_k * BLOCK + tl.arange(0, BLOCK)
        mask = (rm[:, None] < M) & (rk[None, :] < K)
        x = tl.load(X + rm[:, None] * stride_xm + rk[None, :], mask=mask, other=0.0).to(tl.float32)
        amax = tl.maximum(tl.max(tl.max(tl.abs(x), axis=1), axis=0), 1e-8)
        # correctly rounded division + rint: bit-identical to the reference; a hand-made
        # floor(x + .5) tie fix-up rounded exact .5 products the wrong way on gfx1100
        q = _libdevice.rint(x * tl.math.div_rn(127.0, amax))
        q = tl.minimum(tl.maximum(q, -128.0), 127.0)
        tl.store(Q + rm[:, None] * stride_qm + rk[None, :], q.to(tl.int8), mask=mask)
        tl.store(S + pid_m * KB + pid_k, tl.math.div_rn(amax, 127.0))

    def _gemm_configs():
        out = []
        for bm, bn, bk, w in [(128, 128, 64, 4), (128, 256, 64, 8), (128, 128, 128, 8),
                              (64, 128, 64, 4), (32, 128, 64, 4), (16, 128, 128, 4)]:
            out.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                                     num_stages=2, num_warps=w))
        return out

    def _prune(configs, named_args, **kwargs):
        # tiny M (per-token / short-text projections) only needs the narrow tiles
        m = named_args["M"]
        keep = [c for c in configs if c.kwargs["BLOCK_M"] <= max(16, triton.next_power_of_2(m))]
        return keep or configs[-1:]

    @triton.autotune(configs=_gemm_configs(), key=["M_BUCKET", "N", "K"],
                     prune_configs_by={"early_config_prune": _prune})
    @triton.jit
    def _gemm_kernel(A, AS, B, BS, C, BIAS, M, N, K, M_BUCKET,
                     stride_am, stride_bn, stride_cm, KB,
                     HAS_BIAS: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        mask_m = rm < M
        mask_n = rn < N
        sa_row = AS + (rm // 128) * KB
        sb_row = BS + (rn // 128) * KB
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kb in range(0, KB):
            acc_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
            for kk in tl.static_range(128 // BLOCK_K):
                k = kb * 128 + kk * BLOCK_K + rk
                mask_k = k < K
                a = tl.load(A + rm[:, None] * stride_am + k[None, :],
                            mask=mask_m[:, None] & mask_k[None, :], other=0)
                b = tl.load(B + rn[:, None] * stride_bn + k[None, :],
                            mask=mask_n[:, None] & mask_k[None, :], other=0)
                acc_i += tl.dot(a, tl.trans(b))
            sa = tl.load(sa_row + kb, mask=mask_m, other=0.0)
            sb = tl.load(sb_row + kb, mask=mask_n, other=0.0)
            acc += acc_i.to(tl.float32) * sa[:, None] * sb[None, :]
        if HAS_BIAS:
            acc += tl.load(BIAS + rn, mask=mask_n, other=0.0).to(tl.float32)[None, :]
        tl.store(C + rm[:, None] * stride_cm + rn[None, :], acc.to(C.dtype.element_ty),
                 mask=mask_m[:, None] & mask_n[None, :])


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def _quant_fusable(x: torch.Tensor) -> bool:
    return (x.is_cuda and x.dim() == 2 and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and x.stride(-1) == 1 and x.numel() > 0)


def int8_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(q int8 [M, K], scale fp32 [ceil(M/128), ceil(K/128)]); see module docstring."""
    global _kernel_failed
    if not (_enabled() and _quant_fusable(x)):
        return int8_quant_reference(x)
    M, K = x.shape
    MB, KB = _cdiv(M, QBLOCK), _cdiv(K, QBLOCK)
    q = torch.empty(M, K, dtype=torch.int8, device=x.device)
    s = torch.empty(MB, KB, dtype=torch.float32, device=x.device)
    try:
        _quant_kernel[(MB, KB)](x, q, s, M, K, x.stride(0), q.stride(0), KB,
                                BLOCK=QBLOCK, num_warps=8)
    except Exception:
        _kernel_failed = True
        return int8_quant_reference(x)
    return q, s


def _gemm_fusable(a_q, a_s, b_q, b_s) -> bool:
    if not (a_q.is_cuda and b_q.is_cuda and a_q.dtype == torch.int8 and b_q.dtype == torch.int8):
        return False
    if a_q.dim() != 2 or b_q.dim() != 2 or a_q.shape[1] != b_q.shape[1]:
        return False
    if a_q.stride(-1) != 1 or b_q.stride(-1) != 1:
        return False
    M, K = a_q.shape
    N = b_q.shape[0]
    KB = _cdiv(K, QBLOCK)
    return (tuple(a_s.shape) == (_cdiv(M, QBLOCK), KB) and tuple(b_s.shape) == (_cdiv(N, QBLOCK), KB)
            and a_s.dtype == torch.float32 and b_s.dtype == torch.float32
            and a_s.is_contiguous() and b_s.is_contiguous() and M > 0 and N > 0)


def int8_gemm(a_q, a_s, b_q, b_s, out_dtype=torch.float32, bias=None, out=None) -> torch.Tensor:
    """Block-scaled int8 GEMM, y [M, N] = deq(a_q) @ deq(b_q)^T (+ bias)."""
    global _kernel_failed
    if out is not None:
        out_dtype = out.dtype
    if not (_enabled() and _gemm_fusable(a_q, a_s, b_q, b_s)):
        y = int8_gemm_reference(a_q, a_s, b_q, b_s, out_dtype, bias)
        return out.copy_(y) if out is not None else y
    M, K = a_q.shape
    N = b_q.shape[0]
    y = out if out is not None and out.stride(-1) == 1 else torch.empty(M, N, dtype=out_dtype, device=a_q.device)
    m_bucket = min(triton.next_power_of_2(M), 4096)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))  # noqa: E731
    try:
        _gemm_kernel[grid](a_q, a_s, b_q, b_s, y, bias if bias is not None else a_s,
                           M, N, K, m_bucket, a_q.stride(0), b_q.stride(0), y.stride(0),
                           _cdiv(K, QBLOCK), HAS_BIAS=bias is not None)
    except Exception:
        _kernel_failed = True
        y2 = int8_gemm_reference(a_q, a_s, b_q, b_s, out_dtype, bias)
        return out.copy_(y2) if out is not None else y2
    if out is not None and y is not out:
        out.copy_(y)
        return out
    return y


def int8_linear(x: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor, bias=None) -> torch.Tensor:
    """Quantise x per 128x128 block, then int8 GEMM against (w_q, w_s); any leading dims."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    x_q, x_s = int8_quant(x2)
    y = int8_gemm(x_q, x_s, w_q, w_s, out_dtype=x.dtype, bias=bias)
    return y.reshape(*shape[:-1], w_q.shape[0])
