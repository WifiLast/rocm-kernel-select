"""Minimal usage example for amd_tuned_torch.flash_mm_kernel.flash_butterfly_mm_monarch.

Shows the two pieces a caller has to build to use this function directly
(most callers should go through `flash_butterfly_mm`/`matmul` instead, which
pick this path automatically via kernel_select -- see this function's own
module for why): an `X` activation tensor shaped (B, F, L) and a `W_par`
butterfly-coefficient tensor shaped (num_stages, L, 2), where L is the
per-row size being transformed (power-of-2 only) and num_stages =
log2(L). `flash_butterfly_mm_monarch` then computes the SAME result as an
`X @ M` matmul against the (L, L) matrix those coefficients imply, just via
Monarch-grouped block-diagonal `tl.dot`s instead of one dense GEMM -- see
the PERFORMANCE section of flash_mm_kernel.py's own module docstring for
when that's actually faster (roughly L >= 512 at large batch, per the
measured gfx1100 numbers there).

REQUIRES a real Triton install and a visible CUDA/ROCm GPU -- there is
nothing to fall back to here, this function raises RuntimeError without
Triton and launches real GPU kernels otherwise (see monarch_eligible()).
This script checks both up front and explains what's missing rather than
letting the first kernel launch fail with a less legible error.

Correctness is checked against `flash_butterfly_mm_torch` (the plain-PyTorch
mirror of the same stage arithmetic, runs on CPU or GPU, no Triton needed)
in float64 -- the same reference every test in tests/test_flash_mm_kernel.py
checks this kernel against.

Run with:

    python tools/example_flash_butterfly_mm_monarch.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from amd_tuned_torch import flash_mm_kernel as fmk  # noqa: E402


def main() -> None:
    torch.manual_seed(0)

    # L=256 clears monarch_eligible's own gate (L >= 256 and >= 8 stages,
    # see flash_mm_kernel.AMD_TUNED_TORCH_FLASH_MM_MONARCH_MIN_L) --
    # smaller L routes to the plain fused butterfly kernel instead, not an
    # error, just a different code path than the one this example is about.
    B, F, L = 4, 16, 256
    e = fmk._num_stages(L)  # log2(L) == 8 for L=256

    if not fmk.available():
        print("flash_mm_kernel.available() is False here -- this needs Triton "
              "installed AND a visible CUDA/ROCm device to actually run "
              "flash_butterfly_mm_monarch. Nothing more to demonstrate on this "
              "machine.")
        return

    device = "cuda"
    X = torch.randn(B, F, L, device=device, dtype=torch.float32)
    if not fmk.monarch_eligible(X, L):
        print(f"monarch_eligible(X, L={L}) is False for this shape/dtype -- "
              "see AMD_TUNED_TORCH_FLASH_MM_MONARCH_MIN_L and monarch_eligible()'s "
              "own docstring for the exact gate. Try a larger L instead.")
        return
    # Butterfly-stage coefficients: w_stage[j, :] = [a0, a1] for the "low"
    # half of a pair, w_stage[j+s, :] = [b0, b1] for the "high" half (same
    # convention as triton_kernel.py/reference_impl.py) -- random here since
    # this example is about calling the function correctly, not about what
    # (L, L) matrix the coefficients represent.
    W_par = torch.randn(e, L, 2, device=device, dtype=torch.float32)

    Y = fmk.flash_butterfly_mm_monarch(X, W_par)
    print(f"X {tuple(X.shape)} @ (implicit {L}x{L} butterfly matrix) -> Y {tuple(Y.shape)}")

    # Cross-check against the plain-PyTorch mirror in float64, same
    # tolerance posture as this kernel's own tests.
    ref = fmk.flash_butterfly_mm_torch(X.double().cpu(), W_par.double().cpu())
    max_rel_err = ((Y.double().cpu() - ref).abs() / ref.abs().clamp_min(1e-12)).max().item()
    print(f"max relative error vs flash_butterfly_mm_torch: {max_rel_err:.3e}")


if __name__ == "__main__":
    main()
