// Shared implementation behind ck_conv_bwd.hpp. Included by the four thin
// translation units (bwd_data/bwd_weight x f16/bf16) that instantiate it
// -- see ck_conv_bwd_data_2d_f16.cu. Split into separate .cu files for the
// same reason as forward (ck_conv_fwd_impl.hpp): parallel hipcc builds.
//
// This reuses ck_conv_fwd_impl.hpp's Layout<2>/Desc<2>/Action/PassThrough
// rather than duplicating them: the length/stride arrays for a given
// conv2d problem are the same regardless of which op (forward,
// backward-data, backward-weight) consumes them -- only which physical
// tensor (input image / weight / output image) plays which role in each
// op's own MakeArgument differs. That role mapping is the one place a
// backward op is genuinely easy to get wrong, so every invoke function
// below spells it out in a comment next to the call it corresponds to.
#pragma once

#include "ck_conv_bwd.hpp"
#include "ck_conv_fwd_impl.hpp"

#include <array>
#include <tuple>
#include <type_traits>

#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_wmma_v3_instances.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_v3_wmma_instance.hpp"

namespace amd_tuned_torch_ck {
namespace detail {

// Backward-data assigns CK's GEMM A/B/E roles differently than forward:
// A is the OUTPUT-image gradient (dY, forward's E-role shape), B is the
// weight (unchanged), E is the INPUT-image gradient (dX, forward's
// A-role shape -- and the OUTPUT of this op). The physical layouts
// (NHWGC/GKYXC/NHWGK) are identical to Layout<2>'s; only which role each
// one is plugged into differs, so this is a separate struct rather than
// a relabeling of Layout<2>, specifically to avoid ever conflating "A"
// between the two ops.
struct BwdDataLayout {
    using A = ctl::NHWGK;  // dY (output-image gradient)
    using B = ctl::GKYXC;  // weight
    using E = ctl::NHWGC;  // dX (input-image gradient, OUTPUT of this op)
};

using DsLayoutEmpty = ck::Tuple<>;  // bwd-data has no D (bias) tensors here.

// CK's own tuned instance tuple for conv2d bwd-data, per dtype. Each
// tuple's first entry is already CK's generic always-supported fallback
// (see device_grouped_conv_bwd_data_wmma_v3_instances.hpp), so unlike
// forward there is nothing to std::tuple_cat here.
template <typename T>
struct InstancesBwdData;

template <>
struct InstancesBwdData<ck::half_t> {
    using type = cki::device_grouped_conv_bwd_data_wmma_v3_f16_instances<
        2, BwdDataLayout::A, BwdDataLayout::B, DsLayoutEmpty, BwdDataLayout::E,
        cki::ConvBwdDataDefault>;
};

template <>
struct InstancesBwdData<ck::bhalf_t> {
    using type = cki::device_grouped_conv_bwd_data_wmma_v3_bf16_instances<
        2, BwdDataLayout::A, BwdDataLayout::B, DsLayoutEmpty, BwdDataLayout::E,
        cki::ConvBwdDataDefault>;
};

// Backward-weight's alias template's A/B/E slots happen to coincide with
// forward's physical roles (A=input image, B=weight, E=output
// image/dY) -- the reverse of backward-data's -- so this reuses Layout<2>
// from ck_conv_fwd_impl.hpp directly. Note this does NOT mean the OUTPUT
// of backward-weight (dWeight) is the GEMM "E" role: CK's own concrete
// MakeArgument for this op names its three tensor pointers by physical
// role (p_in_grid / p_wei_grid / p_out_grid) rather than by A/B/E, and
// p_wei_grid -- dWeight, the output -- is passed the desc CK calls
// "e_g_k_c_xs" (== the weight's own shape, which is Layout<2>::B's role
// in forward). See invoke_bwd_weight below for the exact wiring.
template <typename T>
struct InstancesBwdWeight;

template <>
struct InstancesBwdWeight<ck::half_t> {
    using type = cki::device_grouped_conv_bwd_weight_v3_wmma_c_shuffle_f16_instances<
        2, Layout<2>::A, Layout<2>::B, Layout<2>::E, cki::ConvBwdWeightDefault>;
};

template <>
struct InstancesBwdWeight<ck::bhalf_t> {
    using type = cki::device_grouped_conv_bwd_weight_v3_wmma_c_shuffle_bf16_instances<
        2, Layout<2>::A, Layout<2>::B, Layout<2>::E, cki::ConvBwdWeightDefault>;
};

// ---------------------------------------------------------------------
// backward data
// ---------------------------------------------------------------------

template <typename Op>
bool invoke_bwd_data(Action action, const Desc<2>& d, const void* p_grad_out, const void* p_wei,
                     void* p_grad_in, void* p_ws, size_t* out_ws, hipStream_t stream) {
    auto op = Op{};
    // A = dY: forward's E-role (output image) length/stride. B = weight:
    // forward's B-role (weight) length/stride, unchanged. Ds: none (no
    // bias tensor in backward). E = dX, the OUTPUT of this op: forward's
    // A-role (input image) length/stride.
    auto arg = op.MakeArgument(
        p_grad_out, p_wei, std::array<const void*, 0>{}, p_grad_in,
        d.e_len, d.e_str,
        d.b_len, d.b_str,
        std::array<std::array<ck::index_t, 5>, 0>{},
        std::array<std::array<ck::index_t, 5>, 0>{},
        d.a_len, d.a_str,
        d.stride, d.dil, d.lpad, d.rpad,
        PassThrough{}, PassThrough{}, PassThrough{});
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

template <typename Tup, size_t I = 0>
bool at_index_bwd_data(int idx, Action action, const Desc<2>& d, const void* p_grad_out,
                       const void* p_wei, void* p_grad_in, void* p_ws, size_t* out_ws,
                       hipStream_t stream) {
    if constexpr (I < std::tuple_size_v<Tup>) {
        if (static_cast<int>(I) == idx)
            return invoke_bwd_data<std::tuple_element_t<I, Tup>>(action, d, p_grad_out, p_wei,
                                                                  p_grad_in, p_ws, out_ws, stream);
        return at_index_bwd_data<Tup, I + 1>(idx, action, d, p_grad_out, p_wei, p_grad_in, p_ws,
                                             out_ws, stream);
    }
    return false;
}

template <typename T>
int count_bwd_data() {
    return static_cast<int>(std::tuple_size_v<typename InstancesBwdData<T>::type>);
}

template <typename T>
bool entry_bwd_data(int idx, Action action, const ConvProblem& p, const void* p_grad_out,
                    const void* p_wei, void* p_grad_in, void* p_ws, size_t* out_ws,
                    hipStream_t stream) {
    if (p.G != 1) return false;
    const Desc<2> d{p};
    return at_index_bwd_data<typename InstancesBwdData<T>::type>(idx, action, d, p_grad_out, p_wei,
                                                                  p_grad_in, p_ws, out_ws, stream);
}

// ---------------------------------------------------------------------
// backward weight
// ---------------------------------------------------------------------

template <typename Op, typename T>
bool invoke_bwd_weight(Action action, const Desc<2>& d, const void* p_in, const void* p_grad_out,
                       void* p_grad_wei, void* p_ws, size_t* out_ws, hipStream_t stream) {
    auto op = Op{};
    // physical input image x  -> MakeArgument's p_in_grid, paired with
    //   the desc CK names "b_g_n_c_wis" == forward's A-role (input
    //   image) length/stride.
    // physical weight gradient dWeight, the OUTPUT of this op ->
    //   p_wei_grid, paired with the desc CK names "e_g_k_c_xs" ==
    //   forward's B-role (weight) length/stride (dWeight has the same
    //   shape/strides as the weight tensor itself).
    // physical dY (grad_output) -> p_out_grid, paired with the desc CK
    //   names "a_g_n_k_wos" == forward's E-role (output image)
    //   length/stride.
    // split_k is fixed at 1: this op's own IsSupportedArgument rejects
    // KBatch>1 on gfx11 (no atomic_pk_add_f16/bf16 there), so this card
    // never has a use for split-K accumulation, and therefore never
    // needs the pre-zeroing of p_grad_wei that only the KBatch>1 path
    // requires (CK's own Invoker::Run zero-inits it internally via
    // hipMemsetAsync when that path is taken -- see
    // device_grouped_conv_bwd_weight_wmma_cshuffle_v3.hpp's
    // clear_workspace lambda -- so callers never need to pre-zero it
    // either way).
    auto arg = op.MakeArgument(
        static_cast<const T*>(p_in), static_cast<T*>(p_grad_wei), static_cast<const T*>(p_grad_out),
        d.a_len, d.a_str,
        d.b_len, d.b_str,
        d.e_len, d.e_str,
        d.stride, d.dil, d.lpad, d.rpad,
        PassThrough{}, PassThrough{}, PassThrough{},
        /*split_k=*/1);
    if (!op.IsSupportedArgument(arg)) return false;
    if (action == Action::kSupported) return true;
    if (action == Action::kWorkspace) {
        *out_ws = op.GetWorkSpaceSize(&arg);
        return true;
    }
    // Channels-last (NHWGC/GKYXC/NHWGK) never hits CK's NGCHW-transpose
    // path, which is the only case this op needs (or even requires a
    // non-null pointer for) workspace -- see IsSupportedArgument's
    // is_NGCHW_NGKHW<...>-guarded block in
    // device_grouped_conv_bwd_weight_wmma_cshuffle_v3.hpp -- so
    // GetWorkSpaceSize() is expected to be 0 for every real call here.
    // SetWorkSpacePointer is still called whenever it isn't, exactly
    // like forward's invoke().
    if (op.GetWorkSpaceSize(&arg) != 0) op.SetWorkSpacePointer(&arg, p_ws);
    op.MakeInvoker().Run(arg, StreamConfig{stream, false});
    return true;
}

template <typename Tup, typename T, size_t I = 0>
bool at_index_bwd_weight(int idx, Action action, const Desc<2>& d, const void* p_in,
                         const void* p_grad_out, void* p_grad_wei, void* p_ws, size_t* out_ws,
                         hipStream_t stream) {
    if constexpr (I < std::tuple_size_v<Tup>) {
        if (static_cast<int>(I) == idx)
            return invoke_bwd_weight<std::tuple_element_t<I, Tup>, T>(
                action, d, p_in, p_grad_out, p_grad_wei, p_ws, out_ws, stream);
        return at_index_bwd_weight<Tup, T, I + 1>(idx, action, d, p_in, p_grad_out, p_grad_wei,
                                                  p_ws, out_ws, stream);
    }
    return false;
}

template <typename T>
int count_bwd_weight() {
    return static_cast<int>(std::tuple_size_v<typename InstancesBwdWeight<T>::type>);
}

template <typename T>
bool entry_bwd_weight(int idx, Action action, const ConvProblem& p, const void* p_in,
                      const void* p_grad_out, void* p_grad_wei, void* p_ws, size_t* out_ws,
                      hipStream_t stream) {
    if (p.G != 1) return false;
    const Desc<2> d{p};
    return at_index_bwd_weight<typename InstancesBwdWeight<T>::type, T>(
        idx, action, d, p_in, p_grad_out, p_grad_wei, p_ws, out_ws, stream);
}

}  // namespace detail
}  // namespace amd_tuned_torch_ck
