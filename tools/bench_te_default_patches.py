"""End-to-end check of the te_extra_ops patches enable() installs by default:
times the public torch APIs (opt.step(), clip_grad_norm_, nn.Dropout) with
amd_tuned_torch disabled vs enabled, so wrapper overhead is included.

    AMD_TUNED_TORCH_ENABLE_TE=1 AMD_TUNED_TORCH_AUTOPATCH=0 PYTHONNOUSERSITE=1 \\
        python tools/bench_te_default_patches.py
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(1, str(pathlib.Path(__file__).resolve().parent))
import torch
import amd_tuned_torch
from bench_te_ops import _time_ms

def params(kind, dtype):
    shapes = []
    if kind == "unet":
        for i in range(100):
            c = (64, 128, 256, 512)[i % 4]
            shapes += [(c, c, 3, 3), (c,), (c,), (c,), (c, c, 1, 1), (c,)]
    else:
        for _ in range(12):
            shapes += [(3072, 1024), (3072,), (1024, 1024), (1024,), (1024,), (1024,),
                       (4096, 1024), (4096,), (1024, 4096), (1024,), (1024,), (1024,)]
        shapes += [(32000, 1024)]
    ps = [torch.nn.Parameter(torch.randn(s, device="cuda", dtype=dtype)) for s in shapes]
    for p in ps:
        p.grad = torch.randn_like(p) * 1e-2
    return ps

def both(label, make_fn, iters=20):
    out = []
    for on in (False, True):
        amd_tuned_torch.enable() if on else amd_tuned_torch.disable()
        fn = make_fn()
        out.append(_time_ms(fn, iters))
    amd_tuned_torch.disable()
    print(f"{label:<44} stock {out[0]:8.3f} ms  patched {out[1]:8.3f} ms  {out[0]/out[1]:5.2f}x",
          flush=True)

for kind in ("unet", "llm"):
    for dt in (torch.float32, torch.bfloat16):
        tag = f"{kind} {str(dt)[6:]}"
        for mode in ({}, {"fused": True}):
            def mk(mode=mode, kind=kind, dt=dt):
                opt = torch.optim.AdamW(params(kind, dt), lr=1e-3, weight_decay=0.01, **mode)
                return opt.step
            both(f"AdamW{'(fused)' if mode else ''}.step  {tag}", mk)
        def mkc(kind=kind, dt=dt):
            ps = params(kind, dt)
            return lambda: torch.nn.utils.clip_grad_norm_(ps, 1e9)
        both(f"clip_grad_norm_  {tag}", mkc)
for dt in (torch.float16, torch.bfloat16):
    x = torch.randn(8192, 4096, device="cuda", dtype=dt, requires_grad=True)
    g = torch.randn_like(x)
    def mkd():
        d = torch.nn.Dropout(0.5).train()
        def run():
            x.grad = None
            d(x).backward(g)
        return run
    both(f"nn.Dropout(0.5) fwd+bwd 8192x4096 {str(dt)[6:]}", mkd)
