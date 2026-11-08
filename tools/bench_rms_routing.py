"""F.rms_norm fwd+bwd through amd_tuned_torch's patch: stock vs the routing
in effect (run once with AMD_TUNED_TORCH_TE_RMS_TRITON=0 and once without)."""
import pathlib, sys, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(1, str(pathlib.Path(__file__).resolve().parent))
import torch, torch.nn.functional as F, amd_tuned_torch
from bench_te_ops import _time_ms
tag = "triton-routing" if os.environ.get("AMD_TUNED_TORCH_TE_RMS_TRITON", "1") != "0" else "hip-only"
for dt in (torch.float16, torch.bfloat16):
    for r, c in [(16384, 2048), (16384, 4096), (16384, 8192), (65536, 1024), (2048, 2048)]:
        x = torch.randn(r, c, device="cuda", dtype=dt, requires_grad=True)
        w = torch.ones(c, device="cuda", dtype=dt, requires_grad=True)
        g = torch.randn_like(x)
        def run():
            x.grad = w.grad = None
            F.rms_norm(x, (c,), w, 1e-5).backward(g)
        amd_tuned_torch.disable(); t0 = _time_ms(run, 20)
        amd_tuned_torch.enable(); t1 = _time_ms(run, 20); amd_tuned_torch.disable()
        print(f"{tag:<15} {str(dt)[6:]:<9} {r:>6}x{c:<5} stock {t0:6.3f} ms  patched {t1:6.3f} ms  {t0/t1:5.2f}x", flush=True)
