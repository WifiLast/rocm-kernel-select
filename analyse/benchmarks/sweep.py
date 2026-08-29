import torch, torch.nn.functional as F, amd_tuned_torch
from amd_tuned_torch import _native
amd_tuned_torch.disable()
d='cuda'
PEAK={torch.float16:122.9, torch.float32:61.4}

def t(fn,n,warm=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s,e=torch.cuda.Event(True),torch.cuda.Event(True); s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/n

SHAPES=[
 ("bench ref: 64x128->64 @128x128",      64,128,128,128, 64),
 ("channels 1024->512 @64x64  N=8",       8,1024,64,64, 512),
 ("channels 1024->512 @64x64  N=1",       1,1024,64,64, 512),
 ("channels 1024->512 @32x32  N=8",       8,1024,32,32, 512),
 ("spatial 1024x512, 64->64   N=1",       1,64,1024,512, 64),
 ("spatial 1024x512, 128->128 N=1",       1,128,1024,512,128),
 ("512->512 @128x128 N=2 (SD-ish)",       2,512,128,128,512),
 ("1024->1024 @32x32 N=8",                8,1024,32,32,1024),
]
for dt in (torch.float16, torch.float32):
    print(f"\n===== {str(dt).replace('torch.','')} =====")
    print(f"{'shape':34s} {'stock ms':>9s} {'native ms':>10s} {'ratio':>7s} {'stock%pk':>9s} {'nat%pk':>7s}  var")
    for name,N,Ci,H,W,Co in SHAPES:
        K=3
        try:
            x=torch.randn(N,Ci,H,W,device=d,dtype=dt)
            w=torch.randn(Co,Ci,K,K,device=d,dtype=dt)
            b=torch.randn(Co,device=d,dtype=dt)
        except RuntimeError as ex:
            print(f"{name:34s} OOM alloc"); continue
        fl=2*N*Ci*Co*H*W*K*K
        n = max(5, min(40, int(3e11/fl)))
        try:
            ts=t(lambda: F.conv2d(x,w,b,stride=1,padding=1),n)
            tc=t(lambda: amd_tuned_torch.ops.conv2d(x,w,b,[1,1],[1,1],[1,1]),n)
        except RuntimeError as ex:
            print(f"{name:34s} FAIL {str(ex)[:40]}"); del x,w,b; torch.cuda.empty_cache(); continue
        var=""
        if dt is torch.float16:
            v=_native.conv2d_fp16_cached_variant(N,Ci,H,W,Co,K,K,H,W,1,1,1,1,1,1)
            var=str(v)
        flag = " <== NATIVE WINS" if tc<ts else ""
        print(f"{name:34s} {ts:9.3f} {tc:10.3f} {ts/tc:6.2f}x {fl/(ts*1e-3)/1e12/PEAK[dt]*100:8.0f}% "
              f"{fl/(tc*1e-3)/1e12/PEAK[dt]*100:6.0f}%  {var}{flag}")
        del x,w,b; torch.cuda.empty_cache()
