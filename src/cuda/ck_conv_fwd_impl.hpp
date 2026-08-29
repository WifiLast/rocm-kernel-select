// Shared implementation behind ck_conv_fwd.hpp. Included by the four thin
// translation units (2d/3d x f16/bf16) that instantiate it -- see
// ck_conv_fwd_2d_f16.cu. They are separate .cu files ONLY so hipcc
// compiles them in parallel: CK instances are expensive to build (minutes
// per dtype/rank), and this is the single biggest cost of the CK tier.
// That is also why each unit compiles CK's `generic + part1 + part2`
// instance tuples rather than all of part1..part4. A sweep of all 38
// tuned instances per combination (tools/ck_instance_sweep.hip) put the
// per-shape winner at index 3 for both 2D dtypes and index 11 for both 3D
// dtypes -- i.e. inside part1 for 2D but only inside part2 for 3D, which
// is why part2 is in and part1 alone is not enough. part3/part4 held
// nothing that won (their best, index 32, came 2nd on 3D by ~2%), so
// they are left out to keep the build tractable. Re-run that sweep and
// widen this if a new shape regresses -- runtime selection can only pick
// from what is compiled in.
#pragma once

#include "ck_conv_fwd.hpp"

#include <array>
#include <tuple>
#include <type_traits>

#include "ck/ck.hpp"
#include "ck/stream_config.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_instance.hpp"

namespace amd_tuned_torch_ck {
namespace detail {

namespace ctl = ck::tensor_layout::convolution;
namespace cki = ck::tensor_operation::device::instance;

using Add         = ck::tensor_operation::element_wise::Add;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

template <int NDim>
struct Layout;
template <>
struct Layout<2> {
    using A = ctl::NHWGC;
    using B = ctl::GKYXC;
    using E = ctl::NHWGK;
};
template <>
struct Layout<3> {
    using A = ctl::NDHWGC;
    using B = ctl::GKZYXC;
    using E = ctl::NDHWGK;
};

// Bias rides along as a single broadcast D tensor in the output layout.
template <int NDim, typename T>
using DsLayout = ck::Tuple<typename Layout<NDim>::E>;
template <typename T>
using DsData = ck::Tuple<T>;

// CK's own tuned instance tuples for this rank/dtype. `generic` is CK's
// always-supported fallback; part1 carries the tiles that actually won on
// this card.
template <int NDim, typename T>
struct Instances;

template <int NDim>
struct Instances<NDim, ck::half_t> {
    using type = decltype(std::tuple_cat(
        std::declval<cki::device_grouped_conv_fwd_wmma_cshufflev3_f16_generic_instances<
            NDim, typename Layout<NDim>::A, typename Layout<NDim>::B,
            DsLayout<NDim, ck::half_t>, typename Layout<NDim>::E,
            cki::ConvFwdDefault, DsData<ck::half_t>, Add>>(),
        std::declval<cki::device_grouped_conv_fwd_wmma_cshufflev3_f16_instances_part1<
            NDim, typename Layout<NDim>::A, typename Layout<NDim>::B,
            DsLayout<NDim, ck::half_t>, typename Layout<NDim>::E,
            cki::ConvFwdDefault, DsData<ck::half_t>, Add>>(),
        std::declval<cki::device_grouped_conv_fwd_wmma_cshufflev3_f16_instances_part2<
            NDim, typename Layout<NDim>::A, typename Layout<NDim>::B,
            DsLayout<NDim, ck::half_t>, typename Layout<NDim>::E,
            cki::ConvFwdDefault, DsData<ck::half_t>, Add>>()));
};

template <int NDim>
struct Instances<NDim, ck::bhalf_t> {
    using type = decltype(std::tuple_cat(
        std::declval<cki::device_grouped_conv_fwd_wmma_cshufflev3_bf16_generic_instances<
            NDim, typename Layout<NDim>::A, typename Layout<NDim>::B,
            DsLayout<NDim, ck::bhalf_t>, typename Layout<NDim>::E,
            cki::ConvFwdDefault, DsData<ck::bhalf_t>, Add>>(),
        std::declval<cki::device_grouped_conv_fwd_wmma_cshufflev3_bf16_instances_part1<
            NDim, typename Layout<NDim>::A, typename Layout<NDim>::B,
            DsLayout<NDim, ck::bhalf_t>, typename Layout<NDim>::E,
            cki::ConvFwdDefault, DsData<ck::bhalf_t>, Add>>(),
        std::declval<cki::device_grouped_conv_fwd_wmma_cshufflev3_bf16_instances_part2<
            NDim, typename Layout<NDim>::A, typename Layout<NDim>::B,
            DsLayout<NDim, ck::bhalf_t>, typename Layout<NDim>::E,
            cki::ConvFwdDefault, DsData<ck::bhalf_t>, Add>>()));
};

// CK indexes every tensor as (G, N, C, spatial...) regardless of how it is
// laid out in memory; the layout lives entirely in the strides below.
template <int NDim>
struct Desc {
    static constexpr int R = NDim + 3;
    std::array<ck::index_t, R> a_len{}, a_str{}, b_len{}, b_str{}, e_len{}, e_str{}, d_len{}, d_str{};
    std::array<ck::index_t, NDim> stride{}, dil{}, lpad{}, rpad{};

    explicit Desc(const ConvProblem& p) {
        // 2D reads the trailing entries of the D,H,W-major arrays.
        const int off = 3 - NDim;
        int in_sp = 1, out_sp = 1, f_sp = 1;
        for (int i = 0; i < NDim; ++i) {
            in_sp  *= p.in_spatial[off + i];
            out_sp *= p.out_spatial[off + i];
            f_sp   *= p.filt[off + i];
        }

        // A: N spatial... G C   -> C is contiguous
        a_len[0] = p.G; a_len[1] = p.N; a_len[2] = p.C;
        a_str[0] = p.C; a_str[1] = in_sp * p.G * p.C; a_str[2] = 1;
        for (int i = 0, acc = p.G * p.C; i < NDim; ++i) {
            const int d = NDim - 1 - i;
            a_len[3 + d] = p.in_spatial[off + d];
            a_str[3 + d] = acc;
            acc *= p.in_spatial[off + d];
        }
        // B: G K filt... C      -> C is contiguous
        b_len[0] = p.G; b_len[1] = p.K; b_len[2] = p.C;
        b_str[0] = p.K * f_sp * p.C; b_str[1] = f_sp * p.C; b_str[2] = 1;
        for (int i = 0, acc = p.C; i < NDim; ++i) {
            const int d = NDim - 1 - i;
            b_len[3 + d] = p.filt[off + d];
            b_str[3 + d] = acc;
            acc *= p.filt[off + d];
        }
        // E: N spatial... G K   -> K is contiguous
        e_len[0] = p.G; e_len[1] = p.N; e_len[2] = p.K;
        e_str[0] = p.K; e_str[1] = out_sp * p.G * p.K; e_str[2] = 1;
        for (int i = 0, acc = p.G * p.K; i < NDim; ++i) {
            const int d = NDim - 1 - i;
            e_len[3 + d] = p.out_spatial[off + d];
            e_str[3 + d] = acc;
            acc *= p.out_spatial[off + d];
        }
        // D (bias): same logical shape as E, but zero-strided everywhere
        // except K, so one K-element vector broadcasts over N and space.
        d_len = e_len;
        d_str[0] = p.K; d_str[1] = 0; d_str[2] = 1;
        for (int i = 0; i < NDim; ++i) d_str[3 + i] = 0;

        for (int i = 0; i < NDim; ++i) {
            stride[i] = p.stride[off + i];
            dil[i]    = p.dilation[off + i];
            lpad[i]   = p.lpad[off + i];
            rpad[i]   = p.rpad[off + i];
        }
    }
};

enum class Action { kSupported, kWorkspace, kRun };

template <typename Op, int NDim>
bool invoke(Action action, const Desc<NDim>& d, const void* p_in, const void* p_wei,
            const void* p_bias, void* p_out, void* p_ws, size_t* out_ws, hipStream_t stream) {
    auto op  = Op{};
    auto arg = op.MakeArgument(p_in, p_wei, std::array<const void*, 1>{p_bias}, p_out,
                              d.a_len, d.a_str, d.b_len, d.b_str,
                              std::array<std::array<ck::index_t, NDim + 3>, 1>{d.d_len},
                              std::array<std::array<ck::index_t, NDim + 3>, 1>{d.d_str},
                              d.e_len, d.e_str, d.stride, d.dil, d.lpad, d.rpad,
                              PassThrough{}, PassThrough{}, Add{});
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

template <typename Tup, int NDim, size_t I = 0>
bool at_index(int idx, Action action, const Desc<NDim>& d, const void* p_in, const void* p_wei,
              const void* p_bias, void* p_out, void* p_ws, size_t* out_ws, hipStream_t stream) {
    if constexpr (I < std::tuple_size_v<Tup>) {
        if (static_cast<int>(I) == idx)
            return invoke<std::tuple_element_t<I, Tup>, NDim>(
                action, d, p_in, p_wei, p_bias, p_out, p_ws, out_ws, stream);
        return at_index<Tup, NDim, I + 1>(idx, action, d, p_in, p_wei, p_bias, p_out, p_ws,
                                          out_ws, stream);
    }
    return false;
}

// One definition per (NDim, T); the .cu files call these.
template <int NDim, typename T>
int count() {
    return static_cast<int>(std::tuple_size_v<typename Instances<NDim, T>::type>);
}

template <int NDim, typename T>
bool entry(int idx, Action action, const ConvProblem& p, const void* p_in, const void* p_wei,
           const void* p_bias, void* p_out, void* p_ws, size_t* out_ws, hipStream_t stream) {
    if (p.G != 1) return false;
    const Desc<NDim> d{p};
    return at_index<typename Instances<NDim, T>::type, NDim>(
        idx, action, d, p_in, p_wei, p_bias, p_out, p_ws, out_ws, stream);
}

}  // namespace detail
}  // namespace amd_tuned_torch_ck
