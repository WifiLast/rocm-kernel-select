// Router for the CK conv2d backward tier: turns a dtype into one of the
// two instantiated translation units, separately for backward-data and
// backward-weight (they are independent ops with independent instance
// tables -- see ck_conv_bwd.hpp). Contains no CK code itself.
#include "ck_conv_bwd_tu.hpp"

namespace amd_tuned_torch_ck {
namespace {

struct TuData {
    int (*count)();
    bool (*entry)(int, int, const ConvProblem&, const void*, const void*, void*, void*, size_t*,
                  hipStream_t);
};

struct TuWeight {
    int (*count)();
    bool (*entry)(int, int, const ConvProblem&, const void*, const void*, void*, void*, size_t*,
                  hipStream_t);
};

// [dtype==kBF16]
const TuData kTusData[2] = {
    {&ck_count_bwd_data_f16, &ck_entry_bwd_data_f16},
    {&ck_count_bwd_data_bf16, &ck_entry_bwd_data_bf16},
};

const TuWeight kTusWeight[2] = {
    {&ck_count_bwd_weight_f16, &ck_entry_bwd_weight_f16},
    {&ck_count_bwd_weight_bf16, &ck_entry_bwd_weight_bf16},
};

const TuData* pick_data(int dtype) {
    if (dtype != kF16 && dtype != kBF16) return nullptr;
    return &kTusData[dtype];
}

const TuWeight* pick_weight(int dtype) {
    if (dtype != kF16 && dtype != kBF16) return nullptr;
    return &kTusWeight[dtype];
}

}  // namespace

int num_instances_bwd_data(int dtype) {
    const TuData* tu = pick_data(dtype);
    return tu ? tu->count() : 0;
}

bool supported_bwd_data(int dtype, int idx, const ConvProblem& p) {
    const TuData* tu = pick_data(dtype);
    return tu && tu->entry(idx, kActBwdSupported, p, nullptr, nullptr, nullptr, nullptr, nullptr,
                          nullptr);
}

size_t workspace_bytes_bwd_data(int dtype, int idx, const ConvProblem& p) {
    const TuData* tu = pick_data(dtype);
    if (!tu) return 0;
    size_t bytes = 0;
    if (!tu->entry(idx, kActBwdWorkspace, p, nullptr, nullptr, nullptr, nullptr, &bytes, nullptr))
        return 0;
    return bytes;
}

bool run_bwd_data(int dtype, int idx, const ConvProblem& p, const void* p_grad_out,
                  const void* p_wei, void* p_grad_in, void* p_workspace, hipStream_t stream) {
    const TuData* tu = pick_data(dtype);
    return tu && tu->entry(idx, kActBwdRun, p, p_grad_out, p_wei, p_grad_in, p_workspace, nullptr,
                          stream);
}

int num_instances_bwd_weight(int dtype) {
    const TuWeight* tu = pick_weight(dtype);
    return tu ? tu->count() : 0;
}

bool supported_bwd_weight(int dtype, int idx, const ConvProblem& p) {
    const TuWeight* tu = pick_weight(dtype);
    return tu && tu->entry(idx, kActBwdSupported, p, nullptr, nullptr, nullptr, nullptr, nullptr,
                          nullptr);
}

size_t workspace_bytes_bwd_weight(int dtype, int idx, const ConvProblem& p) {
    const TuWeight* tu = pick_weight(dtype);
    if (!tu) return 0;
    size_t bytes = 0;
    if (!tu->entry(idx, kActBwdWorkspace, p, nullptr, nullptr, nullptr, nullptr, &bytes, nullptr))
        return 0;
    return bytes;
}

bool run_bwd_weight(int dtype, int idx, const ConvProblem& p, const void* p_in,
                    const void* p_grad_out, void* p_grad_wei, void* p_workspace,
                    hipStream_t stream) {
    const TuWeight* tu = pick_weight(dtype);
    return tu && tu->entry(idx, kActBwdRun, p, p_in, p_grad_out, p_grad_wei, p_workspace, nullptr,
                          stream);
}

}  // namespace amd_tuned_torch_ck
