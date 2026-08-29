import torch, torch.nn.functional as F, amd_tuned_torch
from amd_tuned_torch import conv_select

d = 'cuda'
def t(fn, n, warm=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / n

CASES = [("conv2d",2,64,128,(128,128),64,3), ("conv3d",3,1,512,(8,32,32),512,3)]
rows = []
for name, ndim, N, C, sp, K, k in CASES:
    for dt in (torch.float16, torch.bfloat16, torch.float32):
        x = torch.randn(N,C,*sp,device=d,dtype=dt)
        w = torch.randn(K,C,*([k]*ndim),device=d,dtype=dt)
        b = torch.randn(K,device=d,dtype=dt)
        n = 30 if ndim == 2 else 20
        conv = lambda: (F.conv2d if ndim==2 else F.conv3d)(x,w,b,stride=1,padding=1)

        amd_tuned_torch.disable()                 # true stock
        ts = t(conv, n)
        amd_tuned_torch.enable()                  # patched, with conv_select
        conv_select.reset()
        conv()                                    # first touch runs the contest
        picked = list(conv_select.debug_winners().values())
        tp = t(conv, n)
        rows.append((name, str(dt).replace('torch.',''), ts, tp, picked))
        del x, w, b; torch.cuda.empty_cache()

print(f"{'op':7s} {'dtype':5s} {'stock':>9s} {'F.conv patched':>15s} {'ratio':>7s}  picked")
for name, dt, ts, tp, picked in rows:
    flag = "" if tp <= ts * 1.05 else "   <-- SLOWER THAN STOCK"
    print(f"{name:7s} {dt:5s} {ts:9.3f} {tp:15.3f} {ts/tp:6.2f}x  {picked}{flag}")
