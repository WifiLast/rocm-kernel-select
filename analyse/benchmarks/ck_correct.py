import torch, torch.nn.functional as F, amd_tuned_torch
from amd_tuned_torch import ck_ops
amd_tuned_torch.disable()
d = 'cuda'
torch.manual_seed(0)

CASES = [
    # (ndim, N, C, spatial, K, k, stride, pad, dil)
    (2, 2, 64, (32,32), 64, 3, 1, 1, 1),
    (2, 1, 128, (64,64), 256, 3, 1, 1, 1),
    (2, 2, 32, (28,28), 64, 3, 2, 1, 1),      # strided
    (2, 1, 64, (17,19), 32, 3, 1, 1, 1),      # odd spatial
    (2, 1, 64, (32,32), 64, 1, 1, 0, 1),      # 1x1
    (2, 1, 64, (32,32), 64, 3, 1, 2, 2),      # dilated
    (3, 1, 64, (4,16,16), 64, 3, 1, 1, 1),
    (3, 1, 128, (8,16,16), 128, 3, 1, 1, 1),
    (3, 1, 32, (4,8,8), 64, 3, 2, 1, 1),      # strided 3d
]
print(f"{'case':38s} {'dtype':6s} {'bias':5s} {'fmt':4s} {'max|d|':>9s} {'rel':>9s}  status")
nfail = 0
for (ndim,N,C,sp,K,k,st,pa,di) in CASES:
    for dt in (torch.float16, torch.bfloat16):
        for use_bias in (True, False):
            for cl in (False, True):
                x = torch.randn(N, C, *sp, device=d, dtype=dt)
                w = torch.randn(K, C, *([k]*ndim), device=d, dtype=dt)
                b = torch.randn(K, device=d, dtype=dt) if use_bias else None
                fmt = (torch.channels_last if ndim==2 else torch.channels_last_3d)
                if cl:
                    x = x.contiguous(memory_format=fmt); w = w.contiguous(memory_format=fmt)
                fn = ck_ops.conv2d if ndim==2 else ck_ops.conv3d
                got = fn(x, w, b, st, pa, di)
                name = f"{ndim}d N{N} C{C}{sp}->K{K} k{k} s{st} p{pa} d{di}"
                if got is None:
                    print(f"{name:38s} {str(dt).replace('torch.',''):6s} {str(use_bias):5s} "
                          f"{'CL' if cl else 'NCHW':4s} {'-':>9s} {'-':>9s}  unsupported (falls back)")
                    continue
                ref = (F.conv2d if ndim==2 else F.conv3d)(x.float(), w.float(),
                        b.float() if b is not None else None, stride=st, padding=pa, dilation=di)
                dif = (got.float() - ref).abs()
                rel = dif.mean().item() / ref.abs().mean().item()
                # fp16/bf16 accumulate in fp32 in CK; tolerance follows the
                # dtype's own rounding, matched against a stock run below.
                st_out = (F.conv2d if ndim==2 else F.conv3d)(x, w, b, stride=st, padding=pa, dilation=di)
                st_rel = (st_out.float()-ref).abs().mean().item()/ref.abs().mean().item()
                ok = rel <= max(3*st_rel, 5e-3) and got.shape == ref.shape
                if not ok: nfail += 1
                print(f"{name:38s} {str(dt).replace('torch.',''):6s} {str(use_bias):5s} "
                      f"{'CL' if cl else 'NCHW':4s} {dif.max().item():9.4f} {rel:9.5f}  "
                      f"{'OK' if ok else 'FAIL'} (stock rel {st_rel:.5f})")
print(f"\n{'ALL PASS' if nfail==0 else str(nfail)+' FAILURES'}")
