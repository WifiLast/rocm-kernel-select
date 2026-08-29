// Feasibility spike: can CK's WMMA grouped-conv-fwd be instantiated
// header-only for gfx1100, and is it faster than MIOpen on our shapes?
#include <cstdio>
#include <vector>
#include <array>
#include <tuple>
#include <hip/hip_runtime.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_instance.hpp"

using namespace ck::tensor_operation::device;
using namespace ck::tensor_operation::device::instance;
using ck::tensor_layout::convolution::GKYXC;
using ck::tensor_layout::convolution::NHWGC;
using ck::tensor_layout::convolution::NHWGK;

using InstTuple = device_grouped_conv_fwd_wmma_cshufflev3_f16_instances_part1<
    2, NHWGC, GKYXC, Empty_Tuple, NHWGK, ConvFwdDefault>;
using Op = std::tuple_element_t<INST_IDX, InstTuple>;

int main() {
    const int G=1, N=64, C=128, Hi=128, Wi=128, K=64, Y=3, X=3;
    const int Ho=Hi, Wo=Wi;

    // NHWGC / GKYXC / NHWGK strides, index order (G,N,C,H,W) etc.
    std::array<ck::index_t,5> a_len{G,N,C,Hi,Wi};
    std::array<ck::index_t,5> a_str{C, Hi*Wi*G*C, 1, Wi*G*C, G*C};
    std::array<ck::index_t,5> b_len{G,K,C,Y,X};
    std::array<ck::index_t,5> b_str{K*Y*X*C, Y*X*C, 1, X*C, C};
    std::array<ck::index_t,5> e_len{G,N,K,Ho,Wo};
    std::array<ck::index_t,5> e_str{K, Ho*Wo*G*K, 1, Wo*G*K, G*K};

    size_t na=(size_t)N*Hi*Wi*G*C, nb=(size_t)G*K*Y*X*C, ne=(size_t)N*Ho*Wo*G*K;
    void *pa,*pb,*pe;
    (void)hipMalloc(&pa,na*2); (void)hipMalloc(&pb,nb*2); (void)hipMalloc(&pe,ne*2);
    (void)hipMemset(pa,0x3c,na*2); (void)hipMemset(pb,0x30,nb*2);

    auto op = Op{};
    auto arg = op.MakeArgument(pa, pb, {}, pe,
        a_len, a_str, b_len, b_str, {}, {}, e_len, e_str,
        {1,1}, {1,1}, {1,1}, {1,1},
        ck::tensor_operation::element_wise::PassThrough{},
        ck::tensor_operation::element_wise::PassThrough{},
        ck::tensor_operation::element_wise::PassThrough{});

    printf("instance[%d]: %s\n", INST_IDX, op.GetTypeString().c_str());
    if (!op.IsSupportedArgument(arg)) { printf("  UNSUPPORTED for this shape\n"); return 2; }

    void* ws=nullptr;
    size_t wsb = op.GetWorkSpaceSize(&arg);
    if (wsb) { (void)hipMalloc(&ws,wsb); op.SetWorkSpacePointer(&arg,ws); }

    auto inv = op.MakeInvoker();
    inv.Run(arg, StreamConfig{nullptr,false});
    (void)hipDeviceSynchronize();
    if (hipGetLastError()!=hipSuccess) { printf("  LAUNCH ERROR\n"); return 3; }

    float ms = inv.Run(arg, StreamConfig{nullptr, true, 0, 5, 20});
    double fl = 2.0*N*C*K*Ho*Wo*Y*X;
    printf("  %.3f ms   %.1f TFLOP/s  (%.0f%% of 122.9 fp16 peak)   ws=%zu B\n",
           ms, fl/(ms*1e-3)/1e12, fl/(ms*1e-3)/1e12/122.9*100, wsb);
    return 0;
}
