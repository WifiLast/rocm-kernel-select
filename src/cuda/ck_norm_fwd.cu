// Router for the CK normalization tier: turns (dtype, yop) into one of the
// three instantiated translation units. Contains no CK code itself.
#include "ck_norm_fwd_tu.hpp"

namespace amd_tuned_torch_ck_norm {
namespace {

struct Tu {
    int (*count)(int);
    bool (*entry)(int, int, int, const NormProblem&, const void*, const void*, const void*, void*,
                  void*, size_t*, hipStream_t);
};

const Tu kTus[3] = {
    {&ck_norm_count_f16, &ck_norm_entry_f16},
    {&ck_norm_count_bf16, &ck_norm_entry_bf16},
    {&ck_norm_count_f32, &ck_norm_entry_f32},
};

const Tu* pick(int dtype, int yop) {
    if (dtype != kF16 && dtype != kBF16 && dtype != kF32) return nullptr;
    if (yop != kPassThrough && yop != kSwish) return nullptr;
    return &kTus[dtype];
}

}  // namespace

int num_instances(int dtype, int yop) {
    const Tu* tu = pick(dtype, yop);
    return tu ? tu->count(yop) : 0;
}

bool supported(int dtype, int yop, int idx, const NormProblem& p) {
    const Tu* tu = pick(dtype, yop);
    return tu && tu->entry(yop, idx, kActSupported, p, nullptr, nullptr, nullptr, nullptr,
                           nullptr, nullptr, nullptr);
}

size_t workspace_bytes(int dtype, int yop, int idx, const NormProblem& p) {
    const Tu* tu = pick(dtype, yop);
    if (!tu) return 0;
    size_t bytes = 0;
    if (!tu->entry(yop, idx, kActWorkspace, p, nullptr, nullptr, nullptr, nullptr, nullptr,
                   &bytes, nullptr))
        return 0;
    return bytes;
}

bool run(int dtype, int yop, int idx, const NormProblem& p, const void* p_x, const void* p_gamma,
         const void* p_beta, void* p_y, void* p_workspace, hipStream_t stream) {
    const Tu* tu = pick(dtype, yop);
    return tu && tu->entry(yop, idx, kActRun, p, p_x, p_gamma, p_beta, p_y, p_workspace, nullptr,
                           stream);
}

}  // namespace amd_tuned_torch_ck_norm
