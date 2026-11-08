"""Benchmark / tune fastvideo-kernel-amd on the local ROCm GPU.

    python tools/bench_gfx11.py [int8|vsa|sta|all] [--sweep] [--quick]

int8   block-scaled int8 linear vs bf16 torch matmul (Wan 1.3B / 14B linear shapes)
vsa    VSA 64-block sparse attention (Triton) fwd (+bwd) vs dense SDPA
sta    sliding tile attention (Triton), Wan 18x48x80 canvas
--sweep  time every config of the autotuned kernels one by one (upstream's full
         list, FASTVIDEO_AMD_FULL_AUTOTUNE semantics) and print the table,
         used to pick the gfx11 config sets in _amd.py.

Run with the shell's AMD debug env stripped (see README).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))


def timeit(fn, warmup=2, reps=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


def sweep(kernel, call, label):
    """Time each config of a triton Autotuner on its own."""
    orig = list(kernel.configs)
    rows = []
    for cfg in orig:
        kernel.configs = [cfg]
        kernel.cache.clear()
        try:
            ms = timeit(call, warmup=1, reps=5)
        except Exception as e:  # compile failures (e.g. too much LDS) are expected for some configs
            ms = float("nan")
            print(f"  {cfg}: FAIL {type(e).__name__}: {str(e)[:80]}")
        rows.append((ms, cfg))
    kernel.configs = orig
    kernel.cache.clear()
    rows.sort(key=lambda r: (r[0] != r[0], r[0]))
    print(f"--- sweep {label} ({len(rows)} configs), fastest first")
    for ms, cfg in rows[:12]:
        print(f"  {ms:9.3f} ms  {cfg}")
    return rows


def bench_int8(args):
    from fastvideo_kernel import int8_linear, int8_quant
    shapes = [(32760, 1536, 1536), (32760, 8960, 1536), (32760, 1536, 8960),
              (32760, 5120, 5120), (512, 1536, 4096)]
    if args.quick:
        shapes = shapes[:2]
    print("M N K | bf16 matmul ms | int8_linear ms (quant+gemm) | speedup | rel err")
    for M, N, K in shapes:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
        wq, ws = int8_quant(w)
        t_ref = timeit(lambda: x @ w.T)
        t_i8 = timeit(lambda: int8_linear(x, wq, ws))
        ref = (x @ w.T).float()
        err = ((int8_linear(x, wq, ws).float() - ref).norm() / ref.norm()).item()
        print(f"{M} {N} {K} | {t_ref:.3f} | {t_i8:.3f} | {t_ref / t_i8:.2f}x | {err:.4f}")
        del x, w, wq, ws, ref
        torch.cuda.empty_cache()


def bench_vsa(args):
    from fastvideo_kernel.block_sparse_attn import block_sparse_attn
    from fastvideo_kernel.triton_kernels import block_sparse_attn_triton as bsat
    from tests.utils import generate_block_sparse_mask_for_function
    H, D = 12, 128
    cases = [(512, 51), (512, 26), (1024, 102)] if not args.quick else [(512, 51)]
    for nb, topk in cases:
        S = nb * 64
        q, k, v = (torch.randn(1, H, S, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        vbs = torch.full((nb, ), 64, device="cuda", dtype=torch.int32)
        mask = generate_block_sparse_mask_for_function(H, nb, nb, topk, "cuda").unsqueeze(0)
        fwd = lambda: block_sparse_attn(q, k, v, mask, vbs)  # noqa: E731
        sdpa = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v)  # noqa: E731
        if args.sweep:
            sweep(bsat._attn_fwd_sparse, fwd, f"VSA fwd S={S} topk={topk}")
        t_fwd = timeit(fwd)
        t_sdpa = timeit(sdpa)
        o = fwd()[0]
        print(f"VSA S={S} H={H} D={D} topk={topk}/{nb}: fwd {t_fwd:.2f} ms, dense SDPA {t_sdpa:.2f} ms, "
              f"best config {getattr(bsat._attn_fwd_sparse, 'best_config', None)}, finite={bool(o.isfinite().all())}")
        if not args.quick:
            qg, kg, vg = (t.clone().requires_grad_() for t in (q, k, v))
            do = torch.randn_like(q)

            def fb():
                out = block_sparse_attn(qg, kg, vg, mask, vbs)[0]
                out.backward(do)
            t_fb = timeit(fb, reps=5)
            print(f"    fwd+bwd {t_fb:.2f} ms")
        del q, k, v
        torch.cuda.empty_cache()


def bench_sta(args):
    from fastvideo_kernel import sliding_tile_attention
    from fastvideo_kernel.triton_kernels import st_attn_triton as sta
    H, D = 12, 128
    S = 18 * 48 * 80
    q, k, v = (torch.randn(1, H, S, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    for win in ([3, 3, 3], [3, 1, 5]):
        call = lambda: sliding_tile_attention(q, k, v, [win] * H, 0, False, "18x48x80")  # noqa: E731
        if args.sweep:
            sweep(sta.triton_sta_kernel, call, f"STA window {win}")
        t = timeit(call, warmup=1, reps=5)
        print(f"STA 18x48x80 H={H} window {win}: {t:.2f} ms, best config "
              f"{getattr(sta.triton_sta_kernel, 'best_config', None)}")
    t_sdpa = timeit(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v), warmup=1, reps=3)
    print(f"dense SDPA same shape: {t_sdpa:.2f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all", choices=["int8", "vsa", "sta", "all"])
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.sweep:
        os.environ["FASTVIDEO_AMD_FULL_AUTOTUNE"] = "1"
    print(f"torch {torch.__version__}, device {torch.cuda.get_device_name(0)}")
    for name, fn in (("int8", bench_int8), ("vsa", bench_vsa), ("sta", bench_sta)):
        if args.what in (name, "all"):
            fn(args)


if __name__ == "__main__":
    main()
