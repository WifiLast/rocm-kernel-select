"""Re-derive te_ops routing thresholds on the current torch/ROCm: norms over
rows x hidden x dtype (stock vs TE hipified vs TE Triton RMSNorm), attention
over score-matrix work x mask x dtype. fwd+bwd (training) and fwd (no_grad).
Writes one line per measurement, flushed (CSV on stdout).

    AMD_TUNED_TORCH_ENABLE_TE=1 AMD_TUNED_TORCH_AUTOPATCH=0 PYTHONNOUSERSITE=1 \\
        TRITON_DEFAULT_BACKEND=amd python tools/sweep_te_thresholds.py [norm|attn]
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(1, str(pathlib.Path(__file__).resolve().parent))
import torch, torch.nn.functional as F
import amd_tuned_torch, amd_tuned_torch.te_ops as te_ops
from bench_te_ops import _time_ms
amd_tuned_torch.disable()
what = sys.argv[1:] or ["norm", "attn"]
DT = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

def t_fb(fn, leaves, gy, iters=10):
    def run():
        for t in leaves: t.grad = None
        fn().backward(gy)
    return _time_ms(run, iters)

def t_f(fn, iters=10):
    def run():
        with torch.no_grad(): fn()
    return _time_ms(run, iters)

if "norm" in what:
    print("kind,dtype,rows,cols,mode,stock_ms,hip_x,triton_x", flush=True)
    for dn, dt in DT.items():
        for kind in ("rms", "layer"):
            for rows in (2048, 4096, 8192, 16384, 32768, 65536):
                for cols in (768, 1024, 2048, 3072, 4096, 5120, 8192):
                    if rows * cols > 65536 * 4096: continue
                    x = torch.randn(rows, cols, device="cuda", dtype=dt, requires_grad=True)
                    w = (torch.rand(cols, device="cuda", dtype=dt) + .5).requires_grad_()
                    b = torch.randn(cols, device="cuda", dtype=dt, requires_grad=True)
                    gy = torch.randn_like(x)
                    if kind == "rms":
                        impl = {"stock": lambda: F.rms_norm(x, (cols,), w, 1e-5),
                                "hip": lambda: te_ops.rms_norm(x, (cols,), w, 1e-5),
                                "triton": lambda: te_ops.rms_norm(x, (cols,), w, 1e-5, backend="triton")}
                        leaves = (x, w)
                    else:
                        impl = {"stock": lambda: F.layer_norm(x, (cols,), w, b, 1e-5),
                                "hip": lambda: te_ops.layer_norm(x, (cols,), w, b, 1e-5)}
                        leaves = (x, w, b)
                    for mode in ("fwdbwd", "fwd"):
                        tm = {k: (t_fb(f, leaves, gy) if mode == "fwdbwd" else t_f(f)) for k, f in impl.items()}
                        tri = f"{tm['stock']/tm['triton']:.2f}" if "triton" in tm else ""
                        print(f"{kind},{dn},{rows},{cols},{mode},{tm['stock']:.4f},{tm['stock']/tm['hip']:.2f},{tri}", flush=True)
                    del x, w, b, gy
            torch.cuda.empty_cache()

if "attn" in what:
    print("dtype,B,H,S,D,work_M,mask,mode,stock_ms,te_x", flush=True)
    cfgs = [(1, 8, 512, 64), (2, 8, 512, 64), (1, 16, 1024, 64), (2, 8, 1024, 128), (2, 16, 1024, 64),
            (4, 16, 1024, 64), (1, 32, 2048, 128), (2, 32, 2048, 128), (1, 16, 4096, 64), (1, 32, 4096, 128)]
    for dn, dt in DT.items():
        for B, H, S, D in cfgs:
            q, k, v = (torch.randn(B, H, S, D, device="cuda", dtype=dt, requires_grad=True) for _ in range(3))
            gy = torch.randn(B, H, S, D, device="cuda", dtype=dt)
            br = torch.ones(S, S, device="cuda", dtype=torch.bool).tril()
            work = B * H * S * S / 1e6
            for mask in ("br_mask", "is_causal", "none"):
                kw = {"attn_mask": br} if mask == "br_mask" else ({"is_causal": True} if mask == "is_causal" else {})
                st = lambda: F.scaled_dot_product_attention(q, k, v, **kw)
                te = lambda: te_ops.scaled_dot_product_attention(q, k, v, **kw)
                for mode in ("fwdbwd", "fwd"):
                    try:
                        a = t_fb(st, (q, k, v), gy, 5) if mode == "fwdbwd" else t_f(st, 5)
                        c = t_fb(te, (q, k, v), gy, 5) if mode == "fwdbwd" else t_f(te, 5)
                        print(f"{dn},{B},{H},{S},{D},{work:.1f},{mask},{mode},{a:.4f},{a/c:.2f}", flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"{dn},{B},{H},{S},{D},{work:.1f},{mask},{mode},ERR,{type(e).__name__}", flush=True)
            del q, k, v, gy, br
            torch.cuda.empty_cache()
