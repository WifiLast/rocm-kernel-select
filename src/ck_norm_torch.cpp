#include "ck_norm_torch.hpp"

#include "cuda/ck_norm_fwd.hpp"

#include <c10/hip/HIPStream.h>

#include <mutex>
#include <unordered_map>

namespace {

namespace ckn = amd_tuned_torch_ck_norm;

// Which CK instance won for a given problem. Same shape-keyed
// benchmark-once-then-cache policy as ck_conv_torch.cpp, and it matters
// more here than it looks: which instance is even *legal* depends on the
// channels-per-group (the vector width has to divide it), so the set being
// ranked differs from shape to shape rather than just the ranking.
struct Key {
    int dtype, yop;
    int N, S1, S2, G, C;
    bool operator==(const Key& o) const {
        return dtype == o.dtype && yop == o.yop && N == o.N && S1 == o.S1 && S2 == o.S2 &&
               G == o.G && C == o.C;
    }
};

struct KeyHash {
    size_t operator()(const Key& k) const {
        size_t h = 1469598103934665603ull;
        auto mix = [&h](int v) { h = (h ^ static_cast<size_t>(v)) * 1099511628211ull; };
        mix(k.dtype); mix(k.yop); mix(k.N); mix(k.S1); mix(k.S2); mix(k.G); mix(k.C);
        return h;
    }
};

std::mutex g_mutex;
// -1 caches "no instance supports this shape", so a shape that can't use
// CK pays the probing cost once rather than on every call.
std::unordered_map<Key, int, KeyHash> g_best;

// Times every supported instance once and returns the fastest, or -1 if
// none is supported.
int select_instance(const Key& key, const ckn::NormProblem& p, const void* x, const void* gamma,
                    const void* beta, void* y, const torch::TensorOptions& opts,
                    hipStream_t stream) {
    const int n = ckn::num_instances(key.dtype, key.yop);
    int best = -1;
    float best_ms = 0.0f;

    for (int i = 0; i < n; ++i) {
        if (!ckn::supported(key.dtype, key.yop, i, p)) continue;

        torch::Tensor ws;
        void* ws_ptr = nullptr;
        const size_t ws_bytes = ckn::workspace_bytes(key.dtype, key.yop, i, p);
        if (ws_bytes) {
            ws = torch::empty({static_cast<int64_t>(ws_bytes)}, opts.dtype(torch::kUInt8));
            ws_ptr = ws.data_ptr();
        }

        // 2 warmups then 1 timed run, the same shape of measurement
        // ck_conv_torch.cpp and run_conv2d_fp16 use.
        bool ok = true;
        for (int w = 0; w < 2 && ok; ++w)
            ok = ckn::run(key.dtype, key.yop, i, p, x, gamma, beta, y, ws_ptr, stream);
        if (!ok) continue;

        hipEvent_t t0, t1;
        (void)hipEventCreate(&t0);
        (void)hipEventCreate(&t1);
        (void)hipEventRecord(t0, stream);
        ok = ckn::run(key.dtype, key.yop, i, p, x, gamma, beta, y, ws_ptr, stream);
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
        case torch::kHalf: return ckn::kF16;
        case torch::kBFloat16: return ckn::kBF16;
        case torch::kFloat: return ckn::kF32;
        default: return -1;
    }
}

}  // namespace

c10::optional<torch::Tensor> ck_group_norm_forward(torch::Tensor input, int64_t num_groups,
                                                   c10::optional<torch::Tensor> weight,
                                                   c10::optional<torch::Tensor> bias, double eps,
                                                   bool fuse_silu) {
    if (!input.is_cuda()) return c10::nullopt;
    // Rank 4 and 5 only. Those are the ranks that have a channels-last
    // memory format, which is what makes the CK view a reshape rather than
    // a copy; a rank-3 path would have to permute unconditionally and would
    // be competing against stock with a handicap, so it declines instead.
    if (input.dim() != 4 && input.dim() != 5) return c10::nullopt;
    // CK's kernels read gamma and beta unconditionally -- there is no
    // affine=False path to route an unparameterised GroupNorm through.
    if (!weight.has_value() || !bias.has_value()) return c10::nullopt;
    if (!weight->is_cuda() || !bias->is_cuda()) return c10::nullopt;

    const int dtype = ck_dtype(input.scalar_type());
    if (dtype < 0) return c10::nullopt;
    if (weight->scalar_type() != input.scalar_type() ||
        bias->scalar_type() != input.scalar_type())
        return c10::nullopt;

    const int64_t N = input.size(0);
    const int64_t C_total = input.size(1);
    if (num_groups <= 0 || C_total % num_groups != 0) return c10::nullopt;
    if (weight->numel() != C_total || bias->numel() != C_total) return c10::nullopt;
    if (!weight->is_contiguous() || !bias->is_contiguous()) return c10::nullopt;

    const auto fmt = input.dim() == 4 ? torch::MemoryFormat::ChannelsLast
                                      : torch::MemoryFormat::ChannelsLast3d;
    // Free when the caller is already channels-last, which is the normal
    // case inside an inference graph running with
    // PYTORCH_MIOPEN_SUGGEST_NHWC=1 and the case the CK conv tier also
    // wants. The output is handed back in whatever format came in.
    const bool already_cl = input.is_contiguous(fmt);
    const auto x = input.contiguous(fmt);
    auto y = torch::empty_like(x);

    // Spatial dims collapse to (S1, S2); CK reduces over both regardless of
    // where the split falls, so this only has to be faithful to the memory
    // layout, not to the caller's dimensionality.
    const int64_t S1 = x.size(2);
    int64_t S2 = 1;
    for (int64_t d = 3; d < x.dim(); ++d) S2 *= x.size(d);

    ckn::NormProblem p{};
    p.N = static_cast<int>(N);
    p.S1 = static_cast<int>(S1);
    p.S2 = static_cast<int>(S2);
    p.G = static_cast<int>(num_groups);
    p.C = static_cast<int>(C_total / num_groups);
    p.epsilon = static_cast<float>(eps);

    const Key key{dtype, fuse_silu ? ckn::kSwish : ckn::kPassThrough,
                  p.N, p.S1, p.S2, p.G, p.C};

    const auto stream = c10::hip::getCurrentHIPStream().stream();
    const void* x_ptr = x.data_ptr();
    const void* gamma_ptr = weight->data_ptr();
    const void* beta_ptr = bias->data_ptr();
    void* y_ptr = y.data_ptr();

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
        best = select_instance(key, p, x_ptr, gamma_ptr, beta_ptr, y_ptr, y.options(), stream);
        std::lock_guard<std::mutex> lock(g_mutex);
        g_best[key] = best;
    }
    if (best < 0) return c10::nullopt;

    torch::Tensor ws;
    void* ws_ptr = nullptr;
    const size_t ws_bytes = ckn::workspace_bytes(dtype, key.yop, best, p);
    if (ws_bytes) {
        ws = torch::empty({static_cast<int64_t>(ws_bytes)}, y.options().dtype(torch::kUInt8));
        ws_ptr = ws.data_ptr();
    }
    if (!ckn::run(dtype, key.yop, best, p, x_ptr, gamma_ptr, beta_ptr, y_ptr, ws_ptr, stream))
        return c10::nullopt;

    return already_cl ? y : y.contiguous();
}
