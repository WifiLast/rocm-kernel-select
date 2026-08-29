// Shared implementation behind ck_norm_fwd.hpp. Included by the three thin
// translation units (f16/bf16/f32) that instantiate it -- see
// ck_norm_fwd_f16.cu. Same structure, and the same reason for the split,
// as ck_conv_fwd_impl.hpp: separate .cu files so hipcc compiles them in
// parallel. These are far cheaper to build than the conv instances (a
// blockwise reduction, no WMMA, no implicit-GEMM descriptor machinery), so
// this tier does not carry the conv tier's build-cost warning.
//
// WHY THE INSTANCE LIST IS WRITTEN OUT HERE rather than pulled from CK.
// CK does ship a curated groupnorm instance list, but it lives in
// library/src/tensor_operation_instance/gpu/normalization_fwd/ and the
// factory functions that build it (add_device_normalization_fwd_rank_5_3_*)
// are declarations against CK's compiled libdevice_operations. Using them
// would mean building and linking that library -- exactly the dependency
// the conv tier avoids by instantiating CK's device ops directly. So the
// tuple below is ours, built from the same DeviceNormalizationFwdImpl /
// DeviceNormalizationFwdSplitKImpl templates CK's own list uses, trimmed
// to the configurations that can actually win here.
//
// WHY THESE CONFIGURATIONS. The vector width an instance wants has to
// divide C -- channels PER GROUP -- and a diffusion U-Net's C-per-group is
// usually not a nice number: SD's 320/640/1280 channels at 32 groups give
// 10/20/40, so the 8-wide instances decline outright at two of those three
// widths and only the 1-, 2- and 4-wide ones are ever selectable. That is
// why the "irregular size" scalar instances are in the list and not
// treated as a fallback nobody reaches: for the most common shapes in the
// workload this tier exists for, one of them IS the answer. The 8-wide
// entries are kept for the widths that do divide (40, and any model using
// fewer groups), and two split-K instances for the early high-resolution
// layers where N*G is small enough to leave most of the GPU idle without
// splitting the reduction.
#pragma once

#include "ck_norm_fwd.hpp"

#include <tuple>
#include <type_traits>
#include <vector>

#include "ck/ck.hpp"
#include "ck/stream_config.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_normalization_fwd_impl.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_normalization_fwd_splitk_impl.hpp"

namespace amd_tuned_torch_ck_norm {
namespace detail {

namespace ckd = ck::tensor_operation::device;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Swish = ck::tensor_operation::element_wise::Swish;

constexpr int kRank = 5;
constexpr int kNumReduceDim = 3;

// Reduction and the mean/variance accumulation are always fp32 regardless
// of storage dtype -- same choice src/cuda/group_norm.cu makes, and the
// one CK's own f16 instances make. A fp16 accumulator over H*W*C elements
// loses too much to be worth the bandwidth it doesn't save (the loads are
// still fp16; only the accumulator changes).
using Compute = float;

// The widest load that is 16 bytes, per dtype. Above this the instance is
// not faster, it just declines more often.
template <typename T>
struct VecMax {
    static constexpr int value = 8;  // 16-bit types
};
template <>
struct VecMax<float> {
    static constexpr int value = 4;
};

// M is (N, G) and K is (S1, S2, C); MThreadClusterSize stays 1 throughout
// because the reduction is entirely within K -- these are the same
// M/K cluster shapes CK's own list uses.
template <typename T, typename Y, int Vec>
using Inst = ckd::DeviceNormalizationFwdImpl<T, T, T, Compute, T, Compute, Y, kRank,
                                             kNumReduceDim,
                                             /*BlockSize*/ 256,
                                             /*MThreadClusterSize*/ 1,
                                             /*KThreadClusterSize*/ 256,
                                             /*MThreadSliceSize*/ 1,
                                             /*KThreadSliceSize*/ Vec == 1 ? 1 : Vec,
                                             /*XYSrcVectorDim*/ 1,
                                             /*XSrcVectorSize*/ Vec,
                                             /*GammaSrcVectorDim*/ 1,
                                             /*GammaSrcVectorSize*/ Vec,
                                             /*BetaSrcVectorDim*/ 1,
                                             /*BetaSrcVectorSize*/ Vec,
                                             /*YDstVectorSize*/ Vec,
                                             /*SaveMeanInvStdScalarPerVector*/ 1>;

template <typename T, typename Y, int Block, int Vec>
using InstBlock = ckd::DeviceNormalizationFwdImpl<T, T, T, Compute, T, Compute, Y, kRank,
                                                  kNumReduceDim, Block, 1, Block, 1,
                                                  Vec == 1 ? 1 : Vec, 1, Vec, 1, Vec, 1, Vec, Vec,
                                                  1>;

template <typename T, typename Y, int Vec>
using InstSplitK = ckd::DeviceNormalizationFwdSplitKImpl<T, T, T, Compute, T, Compute, Y, kRank,
                                                         kNumReduceDim, 256, 1, 256, 1,
                                                         Vec == 1 ? 1 : Vec, 1, Vec, 1, Vec, 1,
                                                         Vec, Vec, 1>;

// Ordered cheapest-to-most-specialised. The runtime picks by measurement
// (ck_norm_torch.cpp), so this order only decides tie-breaks; what matters
// is that the always-supported generic instance is present, since an
// awkward C-per-group can decline everything else.
template <typename T, typename Y>
using Instances = std::tuple<
    // Generic: 64 threads, everything scalar. CK's own "always supported"
    // configuration -- the guarantee that this tier never declines a
    // problem it should have handled.
    ckd::DeviceNormalizationFwdImpl<T, T, T, Compute, T, Compute, Y, kRank, kNumReduceDim, 64, 1,
                                    64, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>,
    InstBlock<T, Y, 256, 1>,
    InstBlock<T, Y, 1024, 1>,
    Inst<T, Y, 2>,
    Inst<T, Y, 4>,
    Inst<T, Y, VecMax<T>::value>,
    InstBlock<T, Y, 1024, VecMax<T>::value>,
    InstSplitK<T, Y, 2>,
    InstSplitK<T, Y, VecMax<T>::value>>;

enum class Action { kSupported, kWorkspace, kRun };

// CK takes lengths/strides as std::vector<index_t>; building them per call
// is a handful of small allocations against a kernel that moves megabytes,
// so it is not worth caching.
struct Desc {
    std::vector<ck::index_t> lengths, x_str, gb_str, y_str, mean_str, reduce_dims;

    explicit Desc(const NormProblem& p) {
        lengths = {p.N, p.S1, p.S2, p.G, p.C};
        // Channels-last: C contiguous, then G, then the spatial dims, then N.
        const ck::index_t gc = p.G * p.C;
        x_str = {p.S1 * p.S2 * gc, p.S2 * gc, gc, p.C, 1};
        y_str = x_str;
        // gamma/beta are [G, C], broadcast over N and both spatial dims.
        gb_str = {0, 0, 0, p.C, 1};
        // Unused -- mean/inv-std are not saved -- but MakeArgumentPointer
        // wants strides for them regardless of the null pointers below.
        mean_str = {p.G, 1};
        reduce_dims = {1, 2, 4};
    }
};

template <typename Op, typename Y>
bool invoke(Action action, const NormProblem& p, const Desc& d, const void* p_x,
            const void* p_gamma, const void* p_beta, void* p_y, void* p_ws, size_t* out_ws,
            hipStream_t stream) {
    auto op = Op{};
    auto arg = op.MakeArgumentPointer(d.lengths, d.x_str, d.gb_str, d.gb_str, d.y_str, d.mean_str,
                                      d.mean_str, d.reduce_dims, static_cast<double>(p.epsilon),
                                      p_x, p_gamma, p_beta, p_y, nullptr, nullptr, Y{});
    if (!op.IsSupportedArgument(arg.get())) return false;
    if (action == Action::kSupported) return true;
    if (action == Action::kWorkspace) {
        *out_ws = op.GetWorkSpaceSize(arg.get());
        return true;
    }
    if (op.GetWorkSpaceSize(arg.get()) != 0) op.SetWorkSpacePointer(arg.get(), p_ws);
    op.MakeInvokerPointer()->Run(arg.get(), StreamConfig{stream, false});
    return true;
}

template <typename Tup, typename Y, size_t I = 0>
bool at_index(int idx, Action action, const NormProblem& p, const Desc& d, const void* p_x,
              const void* p_gamma, const void* p_beta, void* p_y, void* p_ws, size_t* out_ws,
              hipStream_t stream) {
    if constexpr (I < std::tuple_size_v<Tup>) {
        if (static_cast<int>(I) == idx)
            return invoke<std::tuple_element_t<I, Tup>, Y>(action, p, d, p_x, p_gamma, p_beta,
                                                           p_y, p_ws, out_ws, stream);
        return at_index<Tup, Y, I + 1>(idx, action, p, d, p_x, p_gamma, p_beta, p_y, p_ws, out_ws,
                                       stream);
    }
    return false;
}

template <typename T, typename Y>
int count() {
    return static_cast<int>(std::tuple_size_v<Instances<T, Y>>);
}

template <typename T, typename Y>
bool entry(int idx, Action action, const NormProblem& p, const void* p_x, const void* p_gamma,
           const void* p_beta, void* p_y, void* p_ws, size_t* out_ws, hipStream_t stream) {
    const Desc d{p};
    return at_index<Instances<T, Y>, Y>(idx, action, p, d, p_x, p_gamma, p_beta, p_y, p_ws,
                                        out_ws, stream);
}

}  // namespace detail
}  // namespace amd_tuned_torch_ck_norm
