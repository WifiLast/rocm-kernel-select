// Router for the CK GEMM tier: turns (dtype, epilogue) into one of the
// eight instantiated translation units. Contains no CK code itself.
#include "ck_gemm_fwd_tu.hpp"

namespace amd_tuned_torch_ck_gemm {
namespace {

struct Tu {
    int (*count)();
    bool (*entry)(int, int, const GemmProblem&, const void*, const void*, const void*, void*,
                  void*, size_t*, hipStream_t);
};

// [epilogue][dtype == kBF16]
const Tu kTus[4][2] = {
    {{&ck_gemm_count_f16, &ck_gemm_entry_f16}, {&ck_gemm_count_bf16, &ck_gemm_entry_bf16}},
    {{&ck_gemm_count_add_f16, &ck_gemm_entry_add_f16},
     {&ck_gemm_count_add_bf16, &ck_gemm_entry_add_bf16}},
    {{&ck_gemm_count_gelu_f16, &ck_gemm_entry_gelu_f16},
     {&ck_gemm_count_gelu_bf16, &ck_gemm_entry_gelu_bf16}},
    {{&ck_gemm_count_silu_f16, &ck_gemm_entry_silu_f16},
     {&ck_gemm_count_silu_bf16, &ck_gemm_entry_silu_bf16}},
};

const Tu* pick(int dtype, int epilogue) {
    if (dtype != kF16 && dtype != kBF16) return nullptr;
    if (epilogue < kNone || epilogue > kAddSilu) return nullptr;
    return &kTus[epilogue][dtype];
}

}  // namespace

int num_instances(int dtype, int epilogue) {
    const Tu* tu = pick(dtype, epilogue);
    return tu ? tu->count() : 0;
}

bool supported(int dtype, int epilogue, int idx, const GemmProblem& p) {
    const Tu* tu = pick(dtype, epilogue);
    return tu && tu->entry(idx, kActSupported, p, nullptr, nullptr, nullptr, nullptr, nullptr,
                           nullptr, nullptr);
}

size_t workspace_bytes(int dtype, int epilogue, int idx, const GemmProblem& p) {
    const Tu* tu = pick(dtype, epilogue);
    if (!tu) return 0;
    size_t bytes = 0;
    if (!tu->entry(idx, kActWorkspace, p, nullptr, nullptr, nullptr, nullptr, nullptr, &bytes,
                   nullptr))
        return 0;
    return bytes;
}

bool run(int dtype, int epilogue, int idx, const GemmProblem& p, const void* p_a, const void* p_b,
         const void* p_bias, void* p_e, void* p_workspace, hipStream_t stream) {
    const Tu* tu = pick(dtype, epilogue);
    return tu && tu->entry(idx, kActRun, p, p_a, p_b, p_bias, p_e, p_workspace, nullptr, stream);
}

}  // namespace amd_tuned_torch_ck_gemm
