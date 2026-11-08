"""ROCm side of fastvideo-kernel-amd.

Upstream builds its compiled ops (turbodiffusion int8 GEMM/quant, TK and
Blackwell attention) from CUTLASS/CuTe/PTX, none of which compiles for HIP,
so on ROCm only the Triton paths exist. This module supplies the missing
pieces for AMD GPUs:

* the block-scaled int8 quant / GEMM kernels, loaded from the sibling
  amd_tuned_torch source tree by file path (only that one self-contained file: importing the
  amd_tuned_torch package would also patch torch globally, and the copy in
  py310_amd's site-packages is older than the source tree);
* the RDNA3 (gfx11) autotune config sets for the attention kernels.

FASTVIDEO_AMD_INT8_OPS=<path> points at another blockscaled_int8_ops.py,
FASTVIDEO_AMD_INT8=0 disables the int8 kernels (calls then fail like upstream),
FASTVIDEO_AMD_FULL_AUTOTUNE=1 restores upstream's full autotune search.
"""
from __future__ import annotations

import functools
import importlib.util
import os
from pathlib import Path

import torch

# this package lives in amd_tools/source/cmp_ext_turing/fastvideo_kernel_amd, next to amd_tuned_torch
DEFAULT_INT8_OPS = Path(__file__).resolve().parents[3] / "amd_tuned_torch" / "blockscaled_int8_ops.py"


def is_rocm() -> bool:
    return torch.version.hip is not None


@functools.lru_cache(maxsize=1)
def gfx_arch() -> str:
    """'gfx1100' etc., '' when there is no ROCm device."""
    if not is_rocm() or not torch.cuda.is_available():
        return ""
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


def is_rdna() -> bool:
    """RDNA (gfx10/11/12): wave32, WMMA, no MFMA -- needs its own tile configs."""
    arch = gfx_arch()
    return arch.startswith(("gfx10", "gfx11", "gfx12"))


def full_autotune() -> bool:
    return os.environ.get("FASTVIDEO_AMD_FULL_AUTOTUNE", "0") == "1"


@functools.lru_cache(maxsize=1)
def int8_ops():
    """The amd_tuned_torch blockscaled_int8_ops module, or None."""
    if not is_rocm() or os.environ.get("FASTVIDEO_AMD_INT8", "1") == "0":
        return None
    path = Path(os.environ.get("FASTVIDEO_AMD_INT8_OPS", DEFAULT_INT8_OPS)).expanduser()
    if path.is_file():
        spec = importlib.util.spec_from_file_location("fastvideo_kernel._amd_blockscaled_int8", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    try:  # an installed amd_tuned_torch new enough to carry the module
        from amd_tuned_torch import blockscaled_int8_ops as mod  # type: ignore
        return mod
    except Exception:
        return None


def quant_rocm(x, out=None, out_s=None):
    """Same contract as upstream quant_cuda(x, None, None) -> (q, s)."""
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("Unsupported input data type for quant_cuda (int8 quantization).")
    q, s = int8_ops().int8_quant(x)
    if out is not None:
        out.copy_(q)
        out_s.copy_(s)
        return out, out_s
    return q, s


def gemm_rocm(a_q, a_s, b_q, b_s, out):
    """Same contract as upstream gemm_cuda: writes deq(a) @ deq(b)^T into out."""
    int8_ops().int8_gemm(a_q, a_s, b_q, b_s, out=out)
    return out


# gfx1100 sweep (tools/bench_gfx11.py --sweep, RX 7900 XTX, 2026-10-26): only these
# configs ever won; num_stages >= 4 (w8) / >= 5 don't fit the 64 KB LDS and fail to compile.
# Autotuning upstream's full lists costs minutes per new shape (STA: 36 configs).
_RDNA_VSA_FWD = {(2, 4), (3, 4)}  # (num_stages, num_warps), BLOCK_M = BLOCK_N = 64 fixed
_RDNA_STA = {(64, 32, 4, 1), (128, 32, 8, 1), (64, 64, 4, 1), (128, 64, 8, 1)}  # (BLOCK_Q, BLOCK_KV, warps, stages)


def vsa_fwd_configs(configs):
    if not is_rdna() or full_autotune():
        return configs
    return [c for c in configs if (c.num_stages, c.num_warps) in _RDNA_VSA_FWD] or configs


def sta_configs(configs):
    if not is_rdna() or full_autotune():
        return configs
    keep = [c for c in configs
            if (c.kwargs["BLOCK_Q"], c.kwargs["BLOCK_KV"], c.num_warps, c.num_stages) in _RDNA_STA]
    return keep or configs
