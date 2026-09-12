// Composable Kernel (CK) WMMA grouped-conv-BACKWARD wrappers -- conv2d
// only (see ck_conv_fwd.hpp for conv3d's forward-only status and for the
// full explanation of why CK is used here at all, the channels-last
// layout constraint, and the ConvProblem/DType types this header reuses
// unchanged).
//
// SCOPE. Unlike the forward tier, this is conv2d ONLY: no conv3d. Every
// function below always operates at NDim==2, which is why (unlike
// ck_conv_fwd.hpp's num_instances/supported/workspace_bytes/run) none of
// these take an `ndim` parameter.
//
// TWO SEPARATE OPS. A conv's backward pass needs two independent CK
// device ops, each with its own instance table, its own IsSupportedArgument
// results, and its own workspace requirement:
//   - backward-DATA:   dL/dY, weight            -> dL/dX
//   - backward-WEIGHT: dL/dY, input image        -> dL/dWeight
// They are not interchangeable and not run together, so every entry point
// below is suffixed `_bwd_data` or `_bwd_weight` rather than sharing one
// dtype-keyed table the way forward's single op does.
#pragma once

#include "ck_conv_fwd.hpp"

#include <cstddef>
#include <hip/hip_runtime.h>

namespace amd_tuned_torch_ck {

// ---- backward data: dY (A-role, [G,N,K,Ho,Wo]) + weight (B-role,
// [G,K,C,Y,X]) -> dX (E-role, [G,N,C,Hi,Wi], the output of this op). ----

// How many CK bwd-data instances are compiled in for this dtype. 0 if the
// combination isn't built.
int num_instances_bwd_data(int dtype);

// True if instance `idx` accepts this problem (CK's own
// IsSupportedArgument -- vector-load alignment, padding, tile divisibility).
bool supported_bwd_data(int dtype, int idx, const ConvProblem& p);

// Bytes of scratch instance `idx` needs for this problem, 0 if none.
size_t workspace_bytes_bwd_data(int dtype, int idx, const ConvProblem& p);

// Runs instance `idx`. p_grad_out is dL/dY (channels-last [N,Ho,Wo,K]),
// p_wei is the forward weight (channels-last [K,Kh,Kw,C]), p_grad_in
// receives dL/dX (channels-last [N,Hi,Wi,C]). Returns false if the
// instance doesn't support the problem.
bool run_bwd_data(int dtype, int idx, const ConvProblem& p,
                  const void* p_grad_out, const void* p_wei, void* p_grad_in,
                  void* p_workspace, hipStream_t stream);

// ---- backward weight: input image (physical "in") + dY (physical "out")
// -> dWeight (physical "wei", the output of this op, [G,K,C,Y,X]). ----

// How many CK bwd-weight instances are compiled in for this dtype. 0 if
// the combination isn't built.
int num_instances_bwd_weight(int dtype);

// True if instance `idx` accepts this problem (CK's own
// IsSupportedArgument). CK's own workspace/non-null-pointer requirement
// for this op is gated behind its NGCHW-transpose code path (see
// device_grouped_conv_bwd_weight_wmma_cshuffle_v3.hpp's
// is_NGCHW_NGKHW<...>-guarded block), which our channels-last
// (NHWGC/GKYXC/NHWGK) layout never takes, so a null workspace pointer is
// fine here whenever workspace_bytes_bwd_weight() reports 0.
bool supported_bwd_weight(int dtype, int idx, const ConvProblem& p);

// Bytes of scratch instance `idx` needs for this problem, 0 if none. Our
// channels-last layout does not hit CK's NGCHW transpose-workspace path,
// so this is expected to be 0 for every real call.
size_t workspace_bytes_bwd_weight(int dtype, int idx, const ConvProblem& p);

// Runs instance `idx`. p_in is the original conv input (channels-last
// [N,Hi,Wi,C]), p_grad_out is dL/dY (channels-last [N,Ho,Wo,K]),
// p_grad_wei receives dL/dWeight (channels-last [K,Kh,Kw,C]). Returns
// false if the instance doesn't support the problem. split_k is always 1
// (CK rejects KBatch>1 on gfx11), so the underlying op never needs to
// atomically accumulate into p_grad_wei and callers never need to
// pre-zero it.
bool run_bwd_weight(int dtype, int idx, const ConvProblem& p,
                    const void* p_in, const void* p_grad_out, void* p_grad_wei,
                    void* p_workspace, hipStream_t stream);

}  // namespace amd_tuned_torch_ck
