"""What a plain `import torch` user actually gets, per op."""
import torch, torch.nn.functional as F, amd_tuned_torch
from amd_tuned_torch import kernel_select
d='cuda'
def t(fn,n,w=10):
    for _ in range(w): fn()
    torch.cuda.synchronize(); s,e=torch.cuda.Event(True),torch.cuda.Event(True); s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/n

print(f"{'op':10s} {'dtype':5s} {'stock':>8s} {'default (patched)':>18s} {'ratio':>7s}  picked")
CASES=[("conv2d",2,64,128,(128,128),64,3),("conv3d",3,1,512,(8,32,32),512,3)]
for name,ndim,N,C,sp,K,k in CASES:
    for dt in (torch.float16, torch.bfloat16, torch.float32):
        x=torch.randn(N,C,*sp,device=d,dtype=dt); w=torch.randn(K,C,*([k]*ndim),device=d,dtype=dt)
        b=torch.randn(K,device=d,dtype=dt); n=30 if ndim==2 else 20
        call=lambda: (F.conv2d if ndim==2 else F.conv3d)(x,w,b,stride=1,padding=1)
        kernel_select.reset(); amd_tuned_torch.enable(); call()
        picked=[v for kk,v in kernel_select.debug_winners().items() if kk[0]==name]
        tp=t(call,n)
        amd_tuned_torch.disable(); ts=t(call,n); amd_tuned_torch.enable()
        print(f"{name:10s} {str(dt).replace('torch.',''):5s} {ts:8.3f} {tp:18.3f} {ts/tp:6.2f}x  {picked}")
        del x,w,b; torch.cuda.empty_cache()
for dt in (torch.float16, torch.bfloat16, torch.float32):
    x=torch.randn(32,128,64,64,device=d,dtype=dt); g=torch.randn(128,device=d,dtype=dt); bb=torch.randn(128,device=d,dtype=dt)
    call=lambda: F.group_norm(x,32,g,bb,eps=1e-5)
    kernel_select.reset(); amd_tuned_torch.enable(); call()
    picked=[v for kk,v in kernel_select.debug_winners().items() if kk[0]=="group_norm"]
    tp=t(call,200,50); amd_tuned_torch.disable(); ts=t(call,200,50); amd_tuned_torch.enable()
    print(f"{'group_norm':10s} {str(dt).replace('torch.',''):5s} {ts:8.4f} {tp:18.4f} {ts/tp:6.2f}x  {picked}")
    del x,g,bb; torch.cuda.empty_cache()
