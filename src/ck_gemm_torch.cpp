#include "ck_gemm_torch.hpp"

#include "cuda/ck_gemm_fwd.hpp"
#include "hipblaslt_gemm.hpp"  // AmdTunedTorchEpilogue -- the shared vocabulary

#include <c10/hip/HIPStream.h>

#include <limits>
#include <mutex>
#include <unordered_map>

namespace {

namespace ckg = amd_tuned_torch_ck_gemm;

// Which CK instance won for a given problem. Same shape-keyed
// benchmark-once-then-cache policy as ck_conv_torch.cpp. It matters as much
// here: the tier compiles in two GemmSpecializations of every tile, and
// which of the pair is even legal depends on whether M, N and K divide the
// tile -- so the candidate set, not just its ranking, changes with shape.
struct Key {
    int dtype, epilogue;
    int M, N, K;
    bool operator==(const Key& o) const {
        return dtype == o.dtype && epilogue == o.epilogue && M == o.M && N == o.N && K == o.K;
    }
};

struct KeyHash {
    size_t operator()(const Key& k) const {
        size_t h = 1469598103934665603ull;
        auto mix = [&h](int v) { h = (h ^ static_cast<size_t>(v)) * 1099511628211ull; };
        mix(k.dtype); mix(k.epilogue); mix(k.M); mix(k.N); mix(k.K);
        return h;
    }
};

std::mutex g_mutex;
// -1 caches "no instance supports this shape".
std::unordered_map<Key, int, KeyHash> g_best;

int select_instance(const Key& key, const ckg::GemmProblem& p, const void* a, const void* b,
                    const void* bias, void* e, const torch::TensorOptions& opts,
                    hipStream_t stream) {
    const int n = ckg::num_instances(key.dtype, key.epilogue);
    int best = -1;
    float best_ms = 0.0f;

    for (int i = 0; i < n; ++i) {
        if (!ckg::supported(key.dtype, key.epilogue, i, p)) continue;

        torch::Tensor ws;
        void* ws_ptr = nullptr;
        const size_t ws_bytes = ckg::workspace_bytes(key.dtype, key.epilogue, i, p);
        if (ws_bytes) {
            ws = torch::empty({static_cast<int64_t>(ws_bytes)}, opts.dtype(torch::kUInt8));
            ws_ptr = ws.data_ptr();
        }

        bool ok = true;
        for (int w = 0; w < 2 && ok; ++w)
            ok = ckg::run(key.dtype, key.epilogue, i, p, a, b, bias, e, ws_ptr, stream);
        if (!ok) continue;

        hipEvent_t t0, t1;
        (void)hipEventCreate(&t0);
        (void)hipEventCreate(&t1);
        (void)hipEventRecord(t0, stream);
        ok = ckg::run(key.dtype, key.epilogue, i, p, a, b, bias, e, ws_ptr, stream);
        (void)hipEventRecord(t1, stream);
        (void)hipEventSynchronize(t1);
        float ms = 0.0f;
        (void)hipEventElapsedTime(&ms, t0, t1);
        (void)hipEventDestroy(t0);
        (void)hipEventDestroy(t1);
        if (!ok) continue;

        if (best < 0 || ms < best_ms) {
            best_ms = ms;
            best = i;
        }
    }
    return best;
}

int ck_dtype(torch::ScalarType t) {
    switch (t) {
        case torch::kHalf: return ckg::kF16;
        case torch::kBFloat16: return ckg::kBF16;
        default: return -1;
    }
}

// Shared epilogue vocabulary -> CK's. Returns -1 for a request CK has no
// instances for.
int map_epilogue(int64_t requested, bool has_bias) {
    switch (requested) {
        case AMD_TUNED_TORCH_EPI_NONE:
            return has_bias ? ckg::kAdd : ckg::kNone;
        case AMD_TUNED_TORCH_EPI_BIAS:
            return has_bias ? ckg::kAdd : -1;
        case AMD_TUNED_TORCH_EPI_GELU:
            // The activation is fused into a bias-adding epilogue, so an
            // unbiased GELU would need a zero bias buffer allocated per
            // call. Declining is cheaper and the caller has a fallback.
            return has_bias ? ckg::kAddFastGelu : -1;
        case AMD_TUNED_TORCH_EPI_SILU:
            return has_bias ? ckg::kAddSilu : -1;
        default:
            return -1;  // ReLU and anything else: not compiled in
    }
}

}  // namespace

c10::optional<torch::Tensor> ck_gemm_linear(torch::Tensor input, torch::Tensor weight,
                                            c10::optional<torch::Tensor> bias,
                                            int64_t epilogue) {
    if (!input.is_cuda() || !weight.is_cuda()) return c10::nullopt;
    if (input.dim() < 2 || weight.dim() != 2) return c10::nullopt;
    if (input.scalar_type() != weight.scalar_type()) return c10::nullopt;
    const int dtype = ck_dtype(input.scalar_type());
    if (dtype < 0) return c10::nullopt;
    if (bias.has_value() && (!bias->is_cuda() || bias->dim() != 1 ||
                             bias->size(0) != weight.size(0) || !bias->is_contiguous() ||
                             bias->scalar_type() != input.scalar_type()))
        return c10::nullopt;

    const int ck_epi = map_epilogue(epilogue, bias.has_value());
    if (ck_epi < 0) return c10::nullopt;

    // Packed rows are what makes the strides below meaningful. A
    // non-contiguous input is made contiguous (one copy, and a
    // view-producing caller such as attention's reshape is the common
    // case); a non-contiguous weight declines, since copying weights per
    // call would be the wrong trade.
    const auto x = input.dim() == 2 ? input.contiguous()
                                    : input.reshape({-1, input.size(-1)}).contiguous();
    if (!weight.is_contiguous()) return c10::nullopt;
    if (x.size(1) != weight.size(1)) return c10::nullopt;

    const int64_t M = x.size(0), K = x.size(1), N = weight.size(0);
    if (M == 0 || N == 0 || K == 0) return c10::nullopt;
    if (M > std::numeric_limits<int>::max() || N > std::numeric_limits<int>::max() ||
        K > std::numeric_limits<int>::max())
        return c10::nullopt;

    ckg::GemmProblem p{};
    p.M = static_cast<int>(M);
    p.N = static_cast<int>(N);
    p.K = static_cast<int>(K);
    // A is row-major [M, K]; B is a row-major [N, K] weight, which CK reads
    // as a column-major [K, N] with the same leading dimension K; E is
    // row-major [M, N].
    p.stride_a = static_cast<int>(K);
    p.stride_b = static_cast<int>(K);
    p.stride_e = static_cast<int>(N);

    auto out = torch::empty({M, N}, x.options());

    const Key key{dtype, ck_epi, p.M, p.N, p.K};
    const auto stream = c10::hip::getCurrentHIPStream().stream();
    const void* a_ptr = x.data_ptr();
    const void* b_ptr = weight.data_ptr();
    const void* bias_ptr = bias.has_value() ? bias->data_ptr() : nullptr;
    void* e_ptr = out.data_ptr();

    int best = -1;
    bool cached = false;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_best.find(key);
        if (it != g_best.end()) {
            best = it->second;
            cached = true;
        }
    }

    if (!cached) {
        best = select_instance(key, p, a_ptr, b_ptr, bias_ptr, e_ptr, out.options(), stream);
        std::lock_guard<std::mutex> lock(g_mutex);
        g_best[key] = best;
    }
    if (best < 0) return c10::nullopt;

    torch::Tensor ws;
    void* ws_ptr = nullptr;
    const size_t ws_bytes = ckg::workspace_bytes(dtype, ck_epi, best, p);
    if (ws_bytes) {
        ws = torch::empty({static_cast<int64_t>(ws_bytes)}, out.options().dtype(torch::kUInt8));
        ws_ptr = ws.data_ptr();
    }
    if (!ckg::run(dtype, ck_epi, best, p, a_ptr, b_ptr, bias_ptr, e_ptr, ws_ptr, stream))
        return c10::nullopt;

    if (input.dim() == 2) return out;
    auto shape = input.sizes().vec();
    shape.back() = N;
    return out.view(shape);
}
