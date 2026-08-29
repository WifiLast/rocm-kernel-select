import torch, torch.nn.functional as F, amd_tuned_torch
from amd_tuned_torch import ck_ops
amd_tuned_torch.disable()
d='cuda'
def t(fn,n=30,warm=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s,e=torch.cuda.Event(True),torch.cuda.Event(True); s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/n

SH = [("conv2d", 2, 64,128,(128,128),64,3), ("conv3d", 3, 1,512,(8,32,32),512,3)]
print(f"{'op':7s} {'dtype':5s} {'stock':>8s} {'native':>8s} {'CK NCHW':>9s} {'CK chan-last':>13s}   best")
for name, ndim, N, C, sp, K, k in SH:
    for dt in (torch.float16, torch.bfloat16):
        x = torch.randn(N,C,*sp,device=d,dtype=dt)
        w = torch.randn(K,C,*([k]*ndim),device=d,dtype=dt)
        b = torch.randn(K,device=d,dtype=dt)
        conv = F.conv2d if ndim==2 else F.conv3d
        fmt = torch.channels_last if ndim==2 else torch.channels_last_3d
        n = 30 if ndim==2 else 20
        ts = t(lambda: conv(x,w,b,stride=1,padding=1), n)
        # native hand-written kernel (fp16 only; no bf16 kernel exists)
        try:
            tn = t(lambda: amd_tuned_torch.ops.conv2d(x,w,b,[1,1],[1,1],[1,1]) if ndim==2
                   else amd_tuned_torch.ops.conv3d(x,w,b,[1,1,1],[1,1,1],[1,1,1]), n)
        except Exception:
            tn = float('nan')
        ckf = ck_ops.conv2d if ndim==2 else ck_ops.conv3d
        tc_nchw = t(lambda: ckf(x,w,b,1,1,1), n)
        xcl = x.contiguous(memory_format=fmt); wcl = w.contiguous(memory_format=fmt)
        tc_cl = t(lambda: ckf(xcl,wcl,b,1,1,1), n)
        cands = {"stock":ts, "native":tn, "CK":min(tc_nchw,tc_cl)}
        best = min((v,kk) for kk,v in cands.items() if v==v)[1]
        nat = f"{tn:8.3f}" if tn==tn else "     n/a"
        print(f"{name:7s} {str(dt).replace('torch.',''):5s} {ts:8.3f} {nat} {tc_nchw:9.3f} {tc_cl:13.3f}   {best}"
              f"   (CK vs stock: {ts/tc_nchw:.2f}x NCHW, {ts/tc_cl:.2f}x chan-last)")
        del x,w,b,xcl,wcl; torch.cuda.empty_cache()
