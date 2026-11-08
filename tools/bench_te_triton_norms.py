"""TE's experimental ROCm Triton norms (NVTE_USE_LAYERNORM_TRITON /
NVTE_USE_RMSNORM_TRITON) vs the hipified TE kernels te_ops uses vs stock,
on the 7900 XTX. The env switches only affect TE's own modules, so the
Triton functions are called directly through the same autograd wrapper
te_ops uses (identical signatures to tex.*norm_fwd/bwd).

    AMD_TUNED_TORCH_ENABLE_TE=1 AMD_TUNED_TORCH_AUTOPATCH=0 PYTHONNOUSERSITE=1 \\
        TRITON_DEFAULT_BACKEND=amd python tools/bench_te_triton_norms.py
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(1, str(pathlib.Path(__file__).resolve().parent))
import torch, torch.nn.functional as F
import amd_tuned_torch.te_ops as te_ops
from bench_te_ops import _time_ms
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.triton_kernels.norms_common import (
    te_layernorm_fwd_triton, te_layernorm_bwd_triton, te_rmsnorm_fwd_triton, te_rmsnorm_bwd_triton)

def make_ln(fwd, bwd):
    class Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w, b, eps):
            y, mu, rs = fwd(x, w, b, eps, None, None, TE_DType[w.dtype], 0, False)
            ctx.save_for_backward(x, mu, rs, w); return y
        @staticmethod
        def backward(ctx, dy):
            x, mu, rs, w = ctx.saved_tensors
            dx, dw, db = bwd(dy.contiguous(), x, mu, rs, w, 0, False)
            return dx, dw, db, None
    return lambda x, w, b: Fn.apply(x, w, b, 1e-5)

def make_rms(fwd, bwd):
    class Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w, eps):
            y, _, rs = fwd(x, w, eps, None, None, TE_DType[w.dtype], 0, False)
            ctx.save_for_backward(x, rs, w); return y
        @staticmethod
        def backward(ctx, dy):
            x, rs, w = ctx.saved_tensors
            dx, dw = bwd(dy.contiguous(), x, rs, w, 0, False)
            return dx, dw, None
    return lambda x, w: Fn.apply(x, w, 1e-5)

IMPLS = {
  "layer": {"stock": lambda x, w, b: F.layer_norm(x, (x.shape[-1],), w, b, 1e-5),
            "te_hip": make_ln(tex.layernorm_fwd, tex.layernorm_bwd),
            "te_triton": make_ln(te_layernorm_fwd_triton, te_layernorm_bwd_triton)},
  "rms":   {"stock": lambda x, w: F.rms_norm(x, (x.shape[-1],), w, 1e-5),
            "te_hip": make_rms(tex.rmsnorm_fwd, tex.rmsnorm_bwd),
            "te_triton": make_rms(te_rmsnorm_fwd_triton, te_rmsnorm_bwd_triton)},
}

def run(kind, impl, x, w, b, gy):
    args = (x, w, b) if kind == "layer" else (x, w)
    for t in args: t.grad = None
    y = IMPLS[kind][impl](*args); y.backward(gy)
    return [y.detach().float()] + [t.grad.float() for t in args]

shapes = [(4096, 4096), (16384, 2048), (16384, 3072), (16384, 4096), (16384, 5120), (16384, 8192),
          (32768, 4096), (65536, 1024), (2048, 1024)]
for dt in (torch.float16, torch.bfloat16):
    print(f"\n=== {str(dt)[6:]} ===  (speedup vs stock; err = max rel err vs fp32 over out+grads)", flush=True)
    for kind in ("rms", "layer"):
        for r, c in shapes:
            x = (torch.randn(r, c, device="cuda") * 4 + 1).to(dt).requires_grad_()
            w = (torch.rand(c, device="cuda") + 0.5).to(dt).requires_grad_()
            b = torch.randn(c, device="cuda").to(dt).requires_grad_()
            gy = torch.randn(r, c, device="cuda").to(dt)
            x32, w32, b32 = (t.detach().float().requires_grad_() for t in (x, w, b))
            ref = run(kind, "stock", x32, w32, b32, gy.float())
            row = {}
            for impl in ("stock", "te_hip", "te_triton"):
                try:
                    got = run(kind, impl, x, w, b, gy)
                    err = max(((g - r_).abs().max() / r_.abs().max().clamp_min(1e-6)).item()
                              for g, r_ in zip(got, ref))
                    args = (x, w, b) if kind == "layer" else (x, w)
                    def fw(): 
                        with torch.no_grad(): IMPLS[kind][impl](*args)
                    def fb():
                        # timing only: no fp32 copies (an earlier version timed
                        # run(), whose .float() conversions skewed every ratio)
                        for t in args: t.grad = None
                        IMPLS[kind][impl](*args).backward(gy)
                    row[impl] = (_time_ms(fw, 20), _time_ms(fb, 20), err)
                except Exception as e:  # noqa: BLE001
                    row[impl] = None; print(f"  {kind} {r}x{c} {impl} FAILED: {type(e).__name__}: {str(e)[:150]}")
            s = row["stock"]
            parts = []
            for impl in ("te_hip", "te_triton"):
                v = row[impl]
                parts.append(f"{impl:<9} fwd {s[0]/v[0]:5.2f}x fwd+bwd {s[1]/v[1]:5.2f}x err {v[2]:.1e}" if v else f"{impl:<9} FAILED")
            print(f"{kind:<5} {r:>6}x{c:<5} stock fwd {s[0]:6.3f} fwd+bwd {s[1]:6.3f} ms err {s[2]:.1e} | " + " | ".join(parts), flush=True)
