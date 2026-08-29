// Shared implementation behind ck_gemm_fwd.hpp. Included by the eight thin
// translation units (f16/bf16 x none/add/gelu/silu) that instantiate it --
// see ck_gemm_fwd_f16.cu.
//
// BUILD COST. These are WMMA GEMM instances and they are expensive, in the
// same league as the conv tier's and unlike the normalization tier's. The
// split into eight units is what keeps the wall clock down: each is about
// ten instances and they compile in parallel, so the tier costs roughly
// one unit's time rather than eight.
//
// WHICH CONFIGURATIONS, AND WHY SO FEW. CK's own shipped list for this
// layout is twenty tile configurations across four GemmSpecializations --
// eighty instances per dtype per epilogue, which at eight combinations is
// not a build anyone will run twice. The five below are a deliberate
// spread over the shapes torch actually hands F.linear rather than a
// prefix of CK's list:
//
//   256/128x128x32  the general workhorse; CK's list leads with it
//   256/128x256x64  wide N -- an MLP up-projection, vocabulary logits
//   128/128x64x64   medium, half the block size for lower occupancy
//                   pressure when M is modest
//   64/32x64x64     small M -- single-token decode, where a 128-row tile
//                   wastes most of its work
//   256/128x128x32 v3  the v3 block-gemm pipeline on the workhorse tile,
//                   because pipeline version changes the answer as much as
//                   tile shape does and there is no way to know which wins
//                   without both being present to measure
//
// Each is instantiated at two GemmSpecializations. MNKPadding accepts any
// M, N, K -- it is what makes the tier applicable to arbitrary torch
// shapes at all -- while Default is the same tile without the padding
// arithmetic, and is what actually competes when the shape happens to be
// divisible, which for transformer weights it usually is. Instantiating
// only the padded form would have made the tier universally applicable and
// then lost every contest it entered for reasons that had nothing to do
// with CK.
//
// To widen this: add tile rows here, rebuild, re-run tools/bench_ck.py.
// Runtime selection can only pick from what is compiled in.
#pragma once

#include "ck_gemm_fwd.hpp"

#include <array>
#include <tuple>
#include <type_traits>

#include "ck/ck.hpp"
#include "ck/stream_config.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_wmma_cshuffle_v3.hpp"

namespace amd_tuned_torch_ck_gemm {
namespace detail {

namespace ckd = ck::tensor_operation::device;
namespace ckl = ck::tensor_layout::gemm;
namespace cke = ck::tensor_operation::element_wise;

using Row = ckl::RowMajor;
using Col = ckl::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = cke::PassThrough;
using Add = cke::Add;
using AddFastGelu = cke::AddFastGelu;
using AddSilu = cke::AddSilu;

using F32 = float;

static constexpr auto Default = ckd::GemmSpecialization::Default;
static constexpr auto MNKPadding = ckd::GemmSpecialization::MNKPadding;
static constexpr auto Intrawave = ck::BlockGemmPipelineScheduler::Intrawave;
static constexpr auto V1 = ck::BlockGemmPipelineVersion::v1;
static constexpr auto V3 = ck::BlockGemmPipelineVersion::v3;

// ---------------------------------------------------------------------
// Plain GEMM (no D tensor). CShuffleDataType is the storage dtype here,
// matching CK's own gemm_universal instances for this layout.
// ---------------------------------------------------------------------
template <typename T, ckd::GemmSpecialization Spec>
using Plain = std::tuple<
    ckd::DeviceGemm_Wmma_CShuffleV3<Row, Col, Row, T, T, T, F32, T, PassThrough, PassThrough,
        PassThrough, Spec, 256, 128, 128, 32, 8, 8, 16, 16, 4, 2,
        S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1,
        S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1,
        1, 1, S<1, 32, 1, 8>, 8, Intrawave, V1>,
    ckd::DeviceGemm_Wmma_CShuffleV3<Row, Col, Row, T, T, T, F32, T, PassThrough, PassThrough,
        PassThrough, Spec, 256, 128, 256, 64, 8, 8, 16, 16, 4, 4,
        S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1,
        S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1,
        1, 1, S<1, 32, 1, 8>, 8, Intrawave, V1>,
    ckd::DeviceGemm_Wmma_CShuffleV3<Row, Col, Row, T, T, T, F32, T, PassThrough, PassThrough,
        PassThrough, Spec, 128, 128, 64, 64, 8, 8, 16, 16, 4, 2,
        S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0,
        S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0,
        1, 1, S<1, 32, 1, 4>, 8, Intrawave, V1>,
    ckd::DeviceGemm_Wmma_CShuffleV3<Row, Col, Row, T, T, T, F32, T, PassThrough, PassThrough,
        PassThrough, Spec, 64, 32, 64, 64, 8, 8, 16, 16, 2, 2,
        S<4, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0,
        S<4, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0,
        1, 1, S<1, 16, 1, 4>, 8, Intrawave, V1>,
    ckd::DeviceGemm_Wmma_CShuffleV3<Row, Col, Row, T, T, T, F32, T, PassThrough, PassThrough,
        PassThrough, Spec, 256, 128, 128, 32, 8, 8, 16, 16, 4, 2,
        S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1,
        S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1,
        1, 1, S<1, 32, 1, 8>, 8, Intrawave, V3>>;

// ---------------------------------------------------------------------
// GEMM + one D tensor (the bias) + activation. Same five tiles. Note
// CShuffleDataType is F32 here, not the storage type: the epilogue's
// activation runs on the fp32 accumulator and converts once on the way
// out, which is both more accurate and what CK's own fused instances do.
// ---------------------------------------------------------------------
using RowTuple = ck::Tuple<Row>;
template <typename T>
using TTuple = ck::Tuple<T>;

template <typename T, typename Epi, ckd::GemmSpecialization Spec>
using Fused = std::tuple<
    ckd::DeviceGemmMultipleD_Wmma_CShuffleV3<Row, Col, RowTuple, Row, T, T, TTuple<T>, T, F32,
        F32, PassThrough, PassThrough, Epi, Spec, 256, 128, 128, 32, 8, 8, 16, 16, 4, 2,
        S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1,
        S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1,
        1, 1, S<1, 32, 1, 8>, S<8, 8, 8>, Intrawave, V1>,
    ckd::DeviceGemmMultipleD_Wmma_CShuffleV3<Row, Col, RowTuple, Row, T, T, TTuple<T>, T, F32,
        F32, PassThrough, PassThrough, Epi, Spec, 256, 128, 256, 64, 8, 8, 16, 16, 4, 4,
        S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1,
        S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1,
        1, 1, S<1, 32, 1, 8>, S<8, 8, 8>, Intrawave, V1>,
    ckd::DeviceGemmMultipleD_Wmma_CShuffleV3<Row, Col, RowTuple, Row, T, T, TTuple<T>, T, F32,
        F32, PassThrough, PassThrough, Epi, Spec, 128, 128, 64, 64, 8, 8, 16, 16, 4, 2,
        S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0,
        S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0,
        1, 1, S<1, 32, 1, 4>, S<8, 8, 8>, Intrawave, V1>,
    ckd::DeviceGemmMultipleD_Wmma_CShuffleV3<Row, Col, RowTuple, Row, T, T, TTuple<T>, T, F32,
        F32, PassThrough, PassThrough, Epi, Spec, 64, 32, 64, 64, 8, 8, 16, 16, 2, 2,
        S<4, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0,
        S<4, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0,
        1, 1, S<1, 16, 1, 4>, S<8, 8, 8>, Intrawave, V1>,
    ckd::DeviceGemmMultipleD_Wmma_CShuffleV3<Row, Col, RowTuple, Row, T, T, TTuple<T>, T, F32,
        F32, PassThrough, PassThrough, Epi, Spec, 256, 128, 128, 32, 8, 8, 16, 16, 4, 2,
        S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1,
        S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1,
        1, 1, S<1, 32, 1, 8>, S<8, 8, 8>, Intrawave, V3>>;

// Both specialisations, concatenated: Default first so it wins ties.
template <typename T>
using PlainInstances = decltype(std::tuple_cat(std::declval<Plain<T, Default>>(),
                                               std::declval<Plain<T, MNKPadding>>()));
template <typename T, typename Epi>
using FusedInstances = decltype(std::tuple_cat(std::declval<Fused<T, Epi, Default>>(),
                                               std::declval<Fused<T, Epi, MNKPadding>>()));

enum class Action { kSupported, kWorkspace, kRun };

// Split-K is not used (KBatch = 1). It would want a workspace and a
// second reduction pass, and the shapes where it pays -- tall-skinny K --
// are not what F.linear hands this tier.
constexpr int kKBatch = 1;

template <typename Op, typename T>
bool invoke_plain(Action action, const GemmProblem& p, const void* p_a, const void* p_b,
                  void* p_e, void* p_ws, size_t* out_ws, hipStream_t stream) {
    auto op = Op{};
    auto arg = op.MakeArgument(static_cast<const T*>(p_a), static_cast<const T*>(p_b),
                               static_cast<T*>(p_e), p.M, p.N, p.K, p.stride_a, p.stride_b,
                               p.stride_e, kKBatch, PassThrough{}, PassThrough{}, PassThrough{});
    if (!op.IsSupportedArgument(arg)) return false;
    if (action == Action::kSupported) return true;
    if (action == Action::kWorkspace) {
        *out_ws = op.GetWorkSpaceSize(&arg);
        return true;
    }
    if (op.GetWorkSpaceSize(&arg) != 0) op.SetWorkSpacePointer(&arg, p_ws);
    op.MakeInvoker().Run(arg, StreamConfig{stream, false});
    return true;
}

template <typename Op, typename Epi>
bool invoke_fused(Action action, const GemmProblem& p, const void* p_a, const void* p_b,
                  const void* p_bias, void* p_e, void* p_ws, size_t* out_ws,
                  hipStream_t stream) {
    auto op = Op{};
    // Row stride 0 on the bias is what turns an N-element vector into a
    // logical [M, N] D tensor: every row reads the same N values.
    auto arg = op.MakeArgument(p_a, p_b, std::array<const void*, 1>{p_bias}, p_e, p.M, p.N, p.K,
                               p.stride_a, p.stride_b, std::array<ck::index_t, 1>{0}, p.stride_e,
                               kKBatch, PassThrough{}, PassThrough{}, Epi{});
    if (!op.IsSupportedArgument(arg)) return false;
    if (action == Action::kSupported) return true;
    if (action == Action::kWorkspace) {
        *out_ws = op.GetWorkSpaceSize(&arg);
        return true;
    }
    if (op.GetWorkSpaceSize(&arg) != 0) op.SetWorkSpacePointer(&arg, p_ws);
    op.MakeInvoker().Run(arg, StreamConfig{stream, false});
    return true;
}

template <typename Tup, typename T, size_t I = 0>
bool plain_at(int idx, Action action, const GemmProblem& p, const void* p_a, const void* p_b,
              void* p_e, void* p_ws, size_t* out_ws, hipStream_t stream) {
    if constexpr (I < std::tuple_size_v<Tup>) {
        if (static_cast<int>(I) == idx)
            return invoke_plain<std::tuple_element_t<I, Tup>, T>(action, p, p_a, p_b, p_e, p_ws,
                                                                 out_ws, stream);
        return plain_at<Tup, T, I + 1>(idx, action, p, p_a, p_b, p_e, p_ws, out_ws, stream);
    }
    return false;
}

template <typename Tup, typename Epi, size_t I = 0>
bool fused_at(int idx, Action action, const GemmProblem& p, const void* p_a, const void* p_b,
              const void* p_bias, void* p_e, void* p_ws, size_t* out_ws, hipStream_t stream) {
    if constexpr (I < std::tuple_size_v<Tup>) {
        if (static_cast<int>(I) == idx)
            return invoke_fused<std::tuple_element_t<I, Tup>, Epi>(action, p, p_a, p_b, p_bias,
                                                                   p_e, p_ws, out_ws, stream);
        return fused_at<Tup, Epi, I + 1>(idx, action, p, p_a, p_b, p_bias, p_e, p_ws, out_ws,
                                         stream);
    }
    return false;
}

// One definition per (T) for the plain op and per (T, Epi) for the fused
// one; the .cu files call these.
template <typename T>
int count_plain() {
    return static_cast<int>(std::tuple_size_v<PlainInstances<T>>);
}

template <typename T, typename Epi>
int count_fused() {
    return static_cast<int>(std::tuple_size_v<FusedInstances<T, Epi>>);
}

template <typename T>
bool entry_plain(int idx, Action action, const GemmProblem& p, const void* p_a, const void* p_b,
                 void* p_e, void* p_ws, size_t* out_ws, hipStream_t stream) {
    return plain_at<PlainInstances<T>, T>(idx, action, p, p_a, p_b, p_e, p_ws, out_ws, stream);
}

template <typename T, typename Epi>
bool entry_fused(int idx, Action action, const GemmProblem& p, const void* p_a, const void* p_b,
                 const void* p_bias, void* p_e, void* p_ws, size_t* out_ws, hipStream_t stream) {
    return fused_at<FusedInstances<T, Epi>, Epi>(idx, action, p, p_a, p_b, p_bias, p_e, p_ws,
                                                 out_ws, stream);
}

}  // namespace detail
}  // namespace amd_tuned_torch_ck_gemm
