"""Numerical check of every route amd_tuned_torch/comfy_kitchen_ops.py adds.

Each case compares the comfy-kitchen result against the STOCK torch op it
stands in for, at the tolerance that route is dispatched under.
"""
import torch
import torch.nn.functional as F

import amd_tuned_torch as A
from amd_tuned_torch import comfy_kitchen_ops as ck

D = "cuda"
fails = []


def check(name, got, want, rtol, atol):
    if got is None:
        fails.append(f"{name}: DECLINED (returned None)")
        print(f"  {name:38s} DECLINED")
        return
    got32, want32 = got.float(), want.float()
    err = (got32 - want32).abs().max().item()
    scale = want32.abs().max().item() or 1.0
    ok = torch.allclose(got32, want32, rtol=rtol, atol=atol + rtol * scale)
    if not ok:
        fails.append(f"{name}: max abs err {err:.3g} (scale {scale:.3g})")
    print(f"  {name:38s} {'ok ' if ok else 'FAIL'}  max|err|={err:.3g} scale={scale:.3g}")


print("== layer_norm -> adaln ==")
for dt in (torch.float16, torch.bfloat16, torch.float32):
    x = torch.randn(4, 512, 1280, dtype=dt, device=D)
    w = (torch.rand(1280, dtype=dt, device=D) * 2)          # spans <1 and >1
    b = torch.randn(1280, dtype=dt, device=D)
    with torch.inference_mode():
        check(f"layer_norm {dt}", ck.layer_norm(x, (1280,), w, b, 1e-5),
              F.layer_norm(x, (1280,), w, b, 1e-5), 1e-2, 1e-3)

print("== layer_norm with a small weight (the fp32 scale argument) ==")
x = torch.randn(64, 1280, dtype=torch.float16, device=D)
w = torch.full((1280,), 0.001, dtype=torch.float16, device=D)
b = torch.zeros(1280, dtype=torch.float16, device=D)
with torch.inference_mode():
    check("layer_norm w=0.001", ck.layer_norm(x, (1280,), w, b, 1e-5),
          F.layer_norm(x, (1280,), w, b, 1e-5), 1e-2, 1e-5)

print("== rms_norm -> rms_adaln ==")
for dt in (torch.float16, torch.bfloat16, torch.float32):
    x = torch.randn(4, 512, 1280, dtype=dt, device=D)
    w = torch.rand(1280, dtype=dt, device=D) * 2
    with torch.inference_mode():
        for eps in (None, 1e-6):
            check(f"rms_norm {dt} eps={eps}", ck.rms_norm(x, (1280,), w, eps),
                  F.rms_norm(x, (1280,), w, eps), 1e-2, 1e-3)

print("== rms_norm must decline a multi-dim normalized_shape ==")
x = torch.randn(4, 8, 16, dtype=torch.float16, device=D)
w = torch.rand(8, 16, dtype=torch.float16, device=D)
with torch.inference_mode():
    print("  multi-dim normalized_shape declined:",
          ck.rms_norm(x, (8, 16), w, None) is None)

print("== linear -> WMMA GEMM (kernel envelope: K > 4096) ==")
for m, k, n, bias in ((4096, 8192, 4096, False), (512, 11008, 4096, True),
                      (1, 8192, 4096, False)):
    x = torch.randn(m, k, dtype=torch.float16, device=D) * 0.05
    w = torch.randn(n, k, dtype=torch.float16, device=D) * 0.05
    b = torch.randn(n, dtype=torch.float16, device=D) if bias else None
    with torch.inference_mode():
        check(f"linear {m}x{k}x{n} bias={bias}", ck.linear(x, w, b),
              F.linear(x, w, b), 1e-2, 1e-3)

print("== linear, 3D input reshape ==")
x = torch.randn(2, 256, 8192, dtype=torch.float16, device=D) * 0.05
w = torch.randn(4096, 8192, dtype=torch.float16, device=D) * 0.05
with torch.inference_mode():
    check("linear 3D", ck.linear(x, w, None), F.linear(x, w, None), 1e-2, 1e-3)

print("== linear must decline below the kernel envelope (K <= 4096) ==")
xs = torch.randn(512, 4096, dtype=torch.float16, device=D)
ws = torch.randn(4096, 4096, dtype=torch.float16, device=D)
with torch.inference_mode():
    print("  K=4096 declined:", ck.linear(xs, ws, None) is None)

print("== conv3d -> WMMA implicit GEMM ==")
for pad in (0, 1):
    x = torch.randn(1, 256, 8, 64, 64, dtype=torch.float16, device=D) * 0.1
    w = torch.randn(256, 256, 3, 3, 3, dtype=torch.float16, device=D) * 0.1
    b = torch.randn(256, dtype=torch.float16, device=D)
    with torch.inference_mode():
        got = ck.conv3d(x, w, b, 1, pad, 1)
        check(f"conv3d pad={pad}", got, F.conv3d(x, w, b, 1, pad, 1), 1e-2, 1e-2)
        if got is not None and got.is_contiguous(memory_format=torch.channels_last_3d) \
                and not got.is_contiguous():
            fails.append("conv3d returned channels_last_3d for a contiguous input")
        print(f"    layout preserved (contiguous NCDHW): "
              f"{got is not None and got.is_contiguous()}")

x2g = torch.randn(512, 8192, dtype=torch.float16, device=D)
w2g = torch.randn(4096, 8192, dtype=torch.float16, device=D)
print("== conv3d must decline dilation != 1 ==")
with torch.inference_mode():
    print("  dilation=2 declined:", ck.conv3d(x, w, b, 1, 2, 2) is None)

print("== AMD_TUNED_TORCH_COMFY_GEMM=0 must remove both GEMM tiers ==")
ck._GEMM_ENABLED = False
with torch.inference_mode():
    print("  linear declined:", ck.linear(x2g, w2g, None) is None)
    print("  conv3d declined:", ck.conv3d(x, w, b, 1, 1, 1) is None)
ck._GEMM_ENABLED = True

print("== sdpa_decode -> flash decode ==")
for bsz, hq, hk, s in ((2, 8, 8, 512), (1, 32, 8, 1024)):
    q = torch.randn(bsz, hq, 1, 128, dtype=torch.bfloat16, device=D)
    k = torch.randn(bsz, hk, s, 128, dtype=torch.bfloat16, device=D)
    v = torch.randn(bsz, hk, s, 128, dtype=torch.bfloat16, device=D)
    gqa = hq != hk
    with torch.inference_mode():
        want = F.scaled_dot_product_attention(q, k, v, enable_gqa=gqa)
        check(f"sdpa_decode B{bsz} Hq{hq} Hk{hk} S{s}",
              ck.sdpa_decode(q, k, v, None, False, None, gqa), want, 2e-2, 2e-3)

print("== sdpa_decode declines what it must ==")
q = torch.randn(2, 8, 1, 128, dtype=torch.bfloat16, device=D)
k = torch.randn(2, 8, 512, 128, dtype=torch.bfloat16, device=D)
with torch.inference_mode():
    print("  is_causal declined :", ck.sdpa_decode(q, k, k, None, True, None, False) is None)
    print("  attn_mask declined :",
          ck.sdpa_decode(q, k, k, torch.zeros(2, 8, 1, 512, device=D), False, None, False) is None)
    print("  odd scale declined :", ck.sdpa_decode(q, k, k, None, False, 0.5, False) is None)
    print("  q_len>1 declined   :",
          ck.sdpa_decode(torch.randn(2, 8, 4, 128, dtype=torch.bfloat16, device=D),
                         k, k, None, False, None, False) is None)
    print("  fp16 declined      :",
          ck.sdpa_decode(q.half(), k.half(), k.half(), None, False, None, False) is None)

print()
if fails:
    print("FAILURES:")
    for f in fails:
        print("  -", f)
    raise SystemExit(1)
print("ALL NUMERICAL CHECKS PASSED")
