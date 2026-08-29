// Re-tuning tool for the Composable Kernel conv tier: times EVERY tuned CK
// WMMA conv-fwd instance for one shape and prints them ranked, so
// src/cuda/ck_conv_fwd_impl.hpp's choice of which instance tuples to
// compile in can be re-decided against measurements rather than guessed.
//
// This exists because that choice is a real tradeoff: compiling all 38
// instances per (rank, dtype) is the single largest build cost in this
// project, but runtime selection can only pick from what was compiled in.
// The current answer (generic + part1 + part2) came from this tool --
// index 3 won both 2D dtypes and index 11 won both 3D dtypes, so part1
// alone would have missed the 3D winner by ~25%.
//
// Standalone on purpose: it links no torch and nothing from this
// extension, so it builds in minutes without touching the package.
//
//   CK=/path/to/composable_kernel
//   hipcc -O3 --offload-arch=gfx1100 -std=c++17 \
//     -D__HIP_PLATFORM_AMD__=1 -DCK_ENABLE_FP16=1 -DCK_ENABLE_BF16=1 \
//     -DNDIM=2 -DDT_PREFIX=device_grouped_conv_fwd_wmma_cshufflev3_bf16 \
//     '-DDT_NAME="bf16"' -I$CK/include -I$CK/library/include \
//     tools/ck_instance_sweep.hip -o /tmp/sweep && /tmp/sweep
//
// NDIM is 2 or 3; DT_PREFIX selects the dtype family
// (..._cshufflev3_f16 or ..._cshufflev3_bf16). Edit the shape in main().
// Note these instances are instantiated WITHOUT the fused bias that the
// extension uses (Empty_Tuple/PassThrough here vs Tuple<T>/Add there), so
// treat the absolute times as the no-bias lower bound; the ranking, which
// is what this tool is for, is unaffected.
#include <cstdio>
#include <array>
#include <tuple>
#include <string>
#include <hip/hip_runtime.h>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_instance.hpp"

using namespace ck::tensor_operation::device;
using namespace ck::tensor_operation::device::instance;
namespace ctl = ck::tensor_layout::convolution;

#ifndef NDIM
#define NDIM 2
#endif

#if NDIM == 2
using AL = ctl::NHWGC; using BL = ctl::GKYXC; using EL = ctl::NHWGK;
#else
using AL = ctl::NDHWGC; using BL = ctl::GKZYXC; using EL = ctl::NDHWGK;
#endif

#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)
#define TUP(SUF) CAT(DT_PREFIX, SUF)<NDIM, AL, BL, Empty_Tuple, EL, ConvFwdDefault>
using All = decltype(std::tuple_cat(
    std::declval<TUP(_generic_instances)>(),
    std::declval<TUP(_instances_part1)>(),
    std::declval<TUP(_instances_part2)>(),
    std::declval<TUP(_instances_part3)>(),
    std::declval<TUP(_instances_part4)>()));

struct Prob {
    int G,N,C,K;
    std::array<ck::index_t,NDIM> spatial, filt, stride, dil, lpad, rpad;
};

static void* pa; static void* pb; static void* pe;
static double gflop; static float best_ms = 1e30f; static std::string best_name;

template <typename Op>
void try_one(int idx, const Prob& p) {
    constexpr int R = NDIM + 3;
    std::array<ck::index_t,R> a_len, a_str, b_len, b_str, e_len, e_str;
    // index order: (G, N, C, spatial...) ; memory order: N, spatial..., G, C
    int sp = 1; for (int i=0;i<NDIM;i++) sp *= p.spatial[i];
    int osp = 1; std::array<ck::index_t,NDIM> out;
    for (int i=0;i<NDIM;i++){ out[i]=(p.spatial[i]+p.lpad[i]+p.rpad[i]-p.dil[i]*(p.filt[i]-1)-1)/p.stride[i]+1; osp*=out[i]; }
    int fsp = 1; for (int i=0;i<NDIM;i++) fsp *= p.filt[i];

    a_len[0]=p.G; a_len[1]=p.N; a_len[2]=p.C; for(int i=0;i<NDIM;i++) a_len[3+i]=p.spatial[i];
    a_str[0]=p.C; a_str[1]=sp*p.G*p.C; a_str[2]=1;
    { int acc=p.G*p.C; for(int i=NDIM-1;i>=0;i--){ a_str[3+i]=acc; acc*=p.spatial[i]; } }
    b_len[0]=p.G; b_len[1]=p.K; b_len[2]=p.C; for(int i=0;i<NDIM;i++) b_len[3+i]=p.filt[i];
    b_str[0]=p.K*fsp*p.C; b_str[1]=fsp*p.C; b_str[2]=1;
    { int acc=p.C; for(int i=NDIM-1;i>=0;i--){ b_str[3+i]=acc; acc*=p.filt[i]; } }
    e_len[0]=p.G; e_len[1]=p.N; e_len[2]=p.K; for(int i=0;i<NDIM;i++) e_len[3+i]=out[i];
    e_str[0]=p.K; e_str[1]=osp*p.G*p.K; e_str[2]=1;
    { int acc=p.G*p.K; for(int i=NDIM-1;i>=0;i--){ e_str[3+i]=acc; acc*=out[i]; } }

    auto op = Op{};
    auto arg = op.MakeArgument(pa, pb, {}, pe, a_len, a_str, b_len, b_str, {}, {},
                               e_len, e_str, p.stride, p.dil, p.lpad, p.rpad,
                               ck::tensor_operation::element_wise::PassThrough{},
                               ck::tensor_operation::element_wise::PassThrough{},
                               ck::tensor_operation::element_wise::PassThrough{});
    if (!op.IsSupportedArgument(arg)) return;
    void* ws=nullptr; size_t wsb = op.GetWorkSpaceSize(&arg);
    if (wsb) { if (hipMalloc(&ws,wsb)!=hipSuccess) return; op.SetWorkSpacePointer(&arg,ws); }
    auto inv = op.MakeInvoker();
    float ms = inv.Run(arg, StreamConfig{nullptr, true, 0, 3, 10});
    if (hipGetLastError()==hipSuccess && ms > 0 && ms < best_ms) { best_ms = ms; best_name = op.GetTypeString(); }
    printf("  [%2d] %8.3f ms  %6.1f TFLOP/s  %s\n", idx, ms, gflop/(ms*1e-3)/1e3, op.GetTypeString().c_str());
    if (ws) (void)hipFree(ws);
}

template <typename Tup, size_t I = 0>
void sweep(const Prob& p) {
    if constexpr (I < std::tuple_size_v<Tup>) {
        try_one<std::tuple_element_t<I,Tup>>((int)I, p);
        sweep<Tup, I+1>(p);
    }
}

int main() {
    Prob p;
#if NDIM == 2
    p = Prob{1, 64, 128, 64, {128,128}, {3,3}, {1,1}, {1,1}, {1,1}, {1,1}};
#else
    p = Prob{1, 1, 512, 512, {8,32,32}, {3,3,3}, {1,1,1}, {1,1,1}, {1,1,1}, {1,1,1}};
#endif
    int sp=1, osp=1, fsp=1;
    for (int i=0;i<NDIM;i++){ sp*=p.spatial[i]; fsp*=p.filt[i];
        osp *= (p.spatial[i]+p.lpad[i]+p.rpad[i]-p.dil[i]*(p.filt[i]-1)-1)/p.stride[i]+1; }
    gflop = 2.0*p.N*p.C*p.K*osp*fsp/1e9;
    size_t na=(size_t)p.N*sp*p.G*p.C, nb=(size_t)p.G*p.K*fsp*p.C, ne=(size_t)p.N*osp*p.G*p.K;
    (void)hipMalloc(&pa,na*2); (void)hipMalloc(&pb,nb*2); (void)hipMalloc(&pe,ne*2);
    (void)hipMemset(pa,0x3c,na*2); (void)hipMemset(pb,0x30,nb*2);
    printf("NDIM=%d  %s  %.1f GFLOP   (%zu instances)\n", NDIM, DT_NAME, gflop, std::tuple_size_v<All>);
    sweep<All>(p);
    printf("BEST: %.3f ms  %.1f TFLOP/s   %s\n", best_ms, gflop/(best_ms*1e-3)/1e3, best_name.c_str());
    return 0;
}
