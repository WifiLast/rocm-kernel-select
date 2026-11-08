"""End-to-end: F.rms_norm / F.layer_norm / F.scaled_dot_product_attention
fwd+bwd with amd_tuned_torch disabled vs enabled, over shapes on both sides
of every routing boundary. Prints the route taken (dispatch debug counts)."""
import pathlib, sys, os
os.environ.setdefault("AMD_TUNED_TORCH_DEBUG_DISPATCH", "0")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(1, str(pathlib.Path(__file__).resolve().parent))
import torch, torch.nn.functional as F, amd_tuned_torch
import amd_tuned_torch._dispatch as D, amd_tuned_torch.te_ops as te_ops
from bench_te_ops import _time_ms

def bench(label, fn, leaves, gy, route):
    def run():
        for t in leaves: t.grad = None
        fn().backward(gy)
    amd_tuned_torch.disable(); a = _time_ms(run, 10)
    amd_tuned_torch.enable(); b = _time_ms(run, 10); amd_tuned_torch.disable()
    print(f"{label:<44} route {route:<7} stock {a:7.3f} ms  patched {b:7.3f} ms  {a/b:5.2f}x", flush=True)

for dn, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)):
    for kind in ("rms", "layer"):
        for r, c in [(8192, 768), (8192, 2048), (8192, 5120), (16384, 1024), (16384, 2048), (16384, 3072),
                     (16384, 4096), (16384, 8192), (32768, 4096), (32768, 8192)]:
            x = torch.randn(r, c, device="cuda", dtype=dt, requires_grad=True)
            w = (torch.rand(c, device="cuda", dtype=dt) + .5).requires_grad_()
            b = torch.randn(c, device="cuda", dtype=dt, requires_grad=True)
            gy = torch.randn_like(x)
            route = D._te_norm_route(kind, x, w) or "stock"
            if kind == "rms":
                bench(f"rms   {dn} {r}x{c}", lambda: F.rms_norm(x, (c,), w, 1e-5), (x, w), gy, route)
            else:
                bench(f"layer {dn} {r}x{c}", lambda: F.layer_norm(x, (c,), w, b, 1e-5), (x, w, b), gy, route)
            del x, w, b, gy
    for B, H, S, Dh in [(2, 16, 1024, 64), (4, 16, 1024, 64), (2, 8, 1024, 128), (2, 32, 2048, 128), (1, 8, 512, 128)]:
        q, k, v = (torch.randn(B, H, S, Dh, device="cuda", dtype=dt, requires_grad=True) for _ in range(3))
        gy = torch.randn(B, H, S, Dh, device="cuda", dtype=dt)
        br = torch.ones(S, S, device="cuda", dtype=torch.bool).tril()
        for mask in ("br_mask", "none", "is_causal"):
            kw = {"attn_mask": br} if mask == "br_mask" else ({"is_causal": True} if mask == "is_causal" else {})
            work = B * H * S * S
            te_ok = (work >= D._TE_ATTN_MIN_WORK and Dh >= D._TE_ATTN_MIN_HEAD_DIM and
                     (mask == "br_mask" or (mask == "none" and dt == torch.float32)))
            bench(f"sdpa  {dn} B{B}H{H}S{S}D{Dh} {mask}", lambda: F.scaled_dot_product_attention(q, k, v, **kw),
                  (q, k, v), gy, "te" if te_ok else "stock")
        del q, k, v, gy, br
        torch.cuda.empty_cache()
