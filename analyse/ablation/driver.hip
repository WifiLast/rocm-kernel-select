#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <vector>

void launch_conv2d_fp16_bm256_bn64_bk32_s2(
    const void*, const void*, const void*, void*,
    int, int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, hipStream_t);

int main() {
    const int B=64, C_in=128, H=128, W=128, C_out=64, K=3;
    const int H_out=H, W_out=W;
    size_t n_in = (size_t)B*C_in*H*W, n_w = (size_t)C_out*C_in*K*K, n_out = (size_t)B*C_out*H_out*W_out;
    half *in, *wt, *bs, *out;
    (void)hipMalloc(&in, n_in*2); (void)hipMalloc(&wt, n_w*2);
    (void)hipMalloc(&bs, C_out*2); (void)hipMalloc(&out, n_out*2);
    (void)hipMemset(in, 0x3c, n_in*2); (void)hipMemset(wt, 0x30, n_w*2); (void)hipMemset(bs, 0, C_out*2);

    auto run = [&]{ launch_conv2d_fp16_bm256_bn64_bk32_s2(in,wt,bs,out,B,C_in,H,W,C_out,K,K,H_out,W_out,1,1,1,1,1,1,0); };
    for (int i=0;i<5;i++) run();
    (void)hipDeviceSynchronize();
    hipEvent_t a,b; (void)hipEventCreate(&a); (void)hipEventCreate(&b);
    (void)hipEventRecord(a,0);
    const int N=20; for (int i=0;i<N;i++) run();
    (void)hipEventRecord(b,0); (void)hipEventSynchronize(b);
    float ms=0; (void)hipEventElapsedTime(&ms,a,b);
    printf("%.3f\n", ms/N);
    return 0;
}
