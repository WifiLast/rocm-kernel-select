// Router for the CK conv tier: turns (ndim, dtype) into one of the four
// instantiated translation units. Contains no CK code itself.
#include "ck_conv_fwd_tu.hpp"

namespace amd_tuned_torch_ck {
namespace {

struct Tu {
    int (*count)();
    bool (*entry)(int, int, const ConvProblem&, const void*, const void*, const void*, void*,
                  void*, size_t*, hipStream_t);
};

// [ndim==3][dtype==kBF16]
const Tu kTus[2][2] = {
    {{&ck_count_2d_f16, &ck_entry_2d_f16}, {&ck_count_2d_bf16, &ck_entry_2d_bf16}},
    {{&ck_count_3d_f16, &ck_entry_3d_f16}, {&ck_count_3d_bf16, &ck_entry_3d_bf16}},
};

const Tu* pick(int ndim, int dtype) {
    if (ndim != 2 && ndim != 3) return nullptr;
    if (dtype != kF16 && dtype != kBF16) return nullptr;
    return &kTus[ndim - 2][dtype];
}

}  // namespace

int num_instances(int ndim, int dtype) {
    const Tu* tu = pick(ndim, dtype);
    return tu ? tu->count() : 0;
}

bool supported(int ndim, int dtype, int idx, const ConvProblem& p) {
    const Tu* tu = pick(ndim, dtype);
    return tu && tu->entry(idx, kActSupported, p, nullptr, nullptr, nullptr, nullptr, nullptr,
                           nullptr, nullptr);
}

size_t workspace_bytes(int ndim, int dtype, int idx, const ConvProblem& p) {
    const Tu* tu = pick(ndim, dtype);
    if (!tu) return 0;
    size_t bytes = 0;
    if (!tu->entry(idx, kActWorkspace, p, nullptr, nullptr, nullptr, nullptr, nullptr, &bytes,
                   nullptr))
        return 0;
    return bytes;
}

bool run(int ndim, int dtype, int idx, const ConvProblem& p, const void* p_in, const void* p_wei,
         const void* p_bias, void* p_out, void* p_workspace, hipStream_t stream) {
    const Tu* tu = pick(ndim, dtype);
    return tu && tu->entry(idx, kActRun, p, p_in, p_wei, p_bias, p_out, p_workspace, nullptr,
                           stream);
}

}  // namespace amd_tuned_torch_ck
