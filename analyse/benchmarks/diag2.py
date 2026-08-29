import torch, torch.nn.functional as F, amd_tuned_torch
from amd_tuned_torch import _native
dev = torch.device("cuda")
print("auto-enabled on import:", amd_tuned_torch.is_enabled())
amd_tuned_torch.disable()
print("after disable():", amd_tuned_torch.is_enabled(), "-> F.conv2d is stock MIOpen\n")

def t(fn, n=30, warm=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    s,e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/n

PEAK = {torch.float16: 122.9, torch.float32: 61.4}  # RX 7900 XTX: 2x fp16 WMMA vs fp32 vector
rows = []
for dt in (torch.float16, torch.float32):
    N,Cin,Cout,H,W,K = 64,128,64,128,128,3
    fl2 = 2*N*Cin*Cout*H*W*K*K
    x = torch.randn(N,Cin,H,W,device=dev,dtype=dt); w = torch.randn(Cout,Cin,K,K,device=dev,dtype=dt); b = torch.randn(Cout,device=dev,dtype=dt)
    ts = t(lambda: F.conv2d(x,w,b,stride=1,padding=1))
    tc = t(lambda: amd_tuned_torch.ops.conv2d(x,w,b,[1,1],[1,1],[1,1]))
    rows.append(("conv2d", dt, ts, tc, None, fl2))
    if dt is torch.float16:
        v = _native.conv2d_fp16_cached_variant(N,Cin,H,W,Cout,K,K,1,1,1,1,1,1,1,1)
        print("conv2d fp16 selected variant index:", v)
        for i in range(3):
            r,slds,dlds,occ = _native.conv2d_fp16_variant_diagnostics(i)
            print(f"   variant {i}: VGPRs={r} static_LDS={slds} dyn_LDS={dlds} max_blocks/CU={occ}")
    del x,w,b; torch.cuda.empty_cache()

    N3,Cin3,D,H3,W3,Cout3 = 1,512,8,32,32,512
    fl3 = 2*N3*Cin3*Cout3*D*H3*W3*K*K*K
    x3 = torch.randn(N3,Cin3,D,H3,W3,device=dev,dtype=dt); w3 = torch.randn(Cout3,Cin3,K,K,K,device=dev,dtype=dt); b3 = torch.randn(Cout3,device=dev,dtype=dt)
    ts = t(lambda: F.conv3d(x3,w3,b3,stride=1,padding=1), n=20)
    tc = t(lambda: amd_tuned_torch.ops.conv3d(x3,w3,b3,[1,1,1],[1,1,1],[1,1,1]), n=20)
    tw = None
    if dt is torch.float16:
        tw = t(lambda: _native.conv3d_fp16_winograd_bt8_bc8(x3,w3,b3,[1,1,1],[1,1,1],[1,1,1]), n=20)
        # numerics: fp32 CPU-free reference via stock ROCm in fp32, compared in fp32
        ref = F.conv3d(x3.float(), w3.float(), b3.float(), stride=1, padding=1)
        got = _native.conv3d_fp16_winograd_bt8_bc8(x3,w3,b3,[1,1,1],[1,1,1],[1,1,1]).float()
        direct = amd_tuned_torch.ops.conv3d(x3,w3,b3,[1,1,1],[1,1,1],[1,1,1]).float()
        stock16 = F.conv3d(x3,w3,b3,stride=1,padding=1).float()
        den = ref.abs().mean().item()
        for nm, tt in (("stock fp16", stock16), ("native direct fp16", direct), ("native winograd fp16", got)):
            print(f"   numerics vs fp32 ref: {nm:22s} max|d|={(tt-ref).abs().max().item():9.4f}  "
                  f"mean|d|/mean|ref|={(tt-ref).abs().mean().item()/den:.5f}")
    rows.append(("conv3d", dt, ts, tc, tw, fl3))
    del x3,w3,b3; torch.cuda.empty_cache()

print(f"\n{'op':8s} {'dtype':8s} {'stock ms':>9s} {'native ms':>10s} {'speedup':>8s} {'stock %peak':>12s} {'native %peak':>13s}")
for op, dt, ts, tc, tw, fl in rows:
    dn = str(dt).replace("torch.","")
    print(f"{op:8s} {dn:8s} {ts:9.3f} {tc:10.3f} {ts/tc:7.2f}x {fl/(ts*1e-3)/1e12/PEAK[dt]*100:11.0f}% {fl/(tc*1e-3)/1e12/PEAK[dt]*100:12.0f}%")
    if tw: print(f"{'  +wino':8s} {dn:8s} {ts:9.3f} {tw:10.3f} {ts/tw:7.2f}x {'':11s} {fl/(tw*1e-3)/1e12/PEAK[dt]*100:12.0f}%")
