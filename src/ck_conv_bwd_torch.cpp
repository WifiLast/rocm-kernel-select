#include "ck_conv_bwd_torch.hpp"

#include "cuda/ck_conv_bwd.hpp"

#include <c10/hip/HIPStream.h>

#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

namespace ckw = amd_tuned_torch_ck;

int64_t conv_out_size(int64_t in, int64_t k, int64_t s, int64_t p, int64_t d) {
    return (in + 2 * p - d * (k - 1) - 1) / s + 1;
}

// Shape-keyed "which CK instance won" cache, same benchmark-once policy as
// ck_conv_torch.cpp's Key/select_instance for forward -- see that file for
// why (CK ships dozens of tuned tile configs and the winner is strongly
// shape-dependent). conv2d only here, so unlike forward's Key there is no
// `ndim` field and the spatial arrays are fixed at 2 elements.
struct Key {
    int dtype;
    int N, C, K;
    int in_spatial[2], filt[2], stride[2], dil[2], pad[2];
    bool operator==(const Key& o) const {
        if (dtype != o.dtype || N != o.N || C != o.C || K != o.K) return false;
        for (int i = 0; i < 2; ++i)
            if (in_spatial[i] != o.in_spatial[i] || filt[i] != o.filt[i] ||
                stride[i] != o.stride[i] || dil[i] != o.dil[i] || pad[i] != o.pad[i])
                return false;
        return true;
    }
};

struct KeyHash {
    size_t operator()(const Key& k) const {
        size_t h = 1469598103934665603ull;
        auto mix = [&h](int v) { h = (h ^ static_cast<size_t>(v)) * 1099511628211ull; };
        mix(k.dtype); mix(k.N); mix(k.C); mix(k.K);
        for (int i = 0; i < 2; ++i) { mix(k.in_spatial[i]); mix(k.filt[i]); mix(k.stride[i]); mix(k.dil[i]); mix(k.pad[i]); }
        return h;
    }
};

// Two independent caches: backward-data and backward-weight are different
// ops with different eligible-instance sets and different timings, so
// (per ck_conv_bwd.hpp) they are never allowed to share a cache entry.
std::mutex g_mutex_data;
std::unordered_map<Key, int, KeyHash> g_best_data;  // -1 == "no instance supports this shape"

std::mutex g_mutex_weight;
std::unordered_map<Key, int, KeyHash> g_best_weight;

Key make_key(int dtype, const ckw::ConvProblem& p) {
    Key key{};
    key.dtype = dtype;
    key.N = p.N; key.C = p.C; key.K = p.K;
    for (int i = 0; i < 2; ++i) {
        key.in_spatial[i] = p.in_spatial[1 + i];
        key.filt[i] = p.filt[1 + i];
        key.stride[i] = p.stride[1 + i];
        key.dil[i] = p.dilation[1 + i];
        key.pad[i] = p.lpad[1 + i];
    }
    return key;
}

// Times every supported backward-data instance once into `scratch` and
// returns the fastest, or -1 if none is supported. Same 2-warmup +
// 1-timed-run shape as ck_conv_torch.cpp's select_instance.
int select_instance_data(const Key& key, const ckw::ConvProblem& p, const void* grad_out,
                         const void* wei, void* scratch, const torch::TensorOptions& opts,
                         hipStream_t stream) {
    const int n = ckw::num_instances_bwd_data(key.dtype);
    int best = -1;
    float best_ms = 0.0f;

    for (int i = 0; i < n; ++i) {
        if (!ckw::supported_bwd_data(key.dtype, i, p)) continue;

        torch::Tensor ws;
        void* ws_ptr = nullptr;
        const size_t ws_bytes = ckw::workspace_bytes_bwd_data(key.dtype, i, p);
        if (ws_bytes) {
            ws = torch::empty({static_cast<int64_t>(ws_bytes)}, opts.dtype(torch::kUInt8));
            ws_ptr = ws.data_ptr();
        }

        for (int w = 0; w < 2; ++w)
            if (!ckw::run_bwd_data(key.dtype, i, p, grad_out, wei, scratch, ws_ptr, stream)) break;

        hipEvent_t t0, t1;
        (void)hipEventCreate(&t0);
        (void)hipEventCreate(&t1);
        (void)hipEventRecord(t0, stream);
        const bool ok = ckw::run_bwd_data(key.dtype, i, p, grad_out, wei, scratch, ws_ptr, stream);
        (void)hipEventRecord(t1, stream);
        (void)hipEventSynchronize(t1);

        float ms = 0.0f;
        (void)hipEventElapsedTime(&ms, t0, t1);
        (void)hipEventDestroy(t0);
        (void)hipEventDestroy(t1);

        if (ok && hipGetLastError() == hipSuccess && (best < 0 || ms < best_ms)) {
            best = i;
            best_ms = ms;
        }
    }
    return best;
}

// Same as select_instance_data but for backward-weight.
int select_instance_weight(const Key& key, const ckw::ConvProblem& p, const void* in,
                           const void* grad_out, void* scratch, const torch::TensorOptions& opts,
                           hipStream_t stream) {
    const int n = ckw::num_instances_bwd_weight(key.dtype);
    int best = -1;
    float best_ms = 0.0f;

    for (int i = 0; i < n; ++i) {
        if (!ckw::supported_bwd_weight(key.dtype, i, p)) continue;

        torch::Tensor ws;
        void* ws_ptr = nullptr;
        const size_t ws_bytes = ckw::workspace_bytes_bwd_weight(key.dtype, i, p);
        if (ws_bytes) {
            ws = torch::empty({static_cast<int64_t>(ws_bytes)}, opts.dtype(torch::kUInt8));
            ws_ptr = ws.data_ptr();
        }

        for (int w = 0; w < 2; ++w)
            if (!ckw::run_bwd_weight(key.dtype, i, p, in, grad_out, scratch, ws_ptr, stream))
                break;

        hipEvent_t t0, t1;
        (void)hipEventCreate(&t0);
        (void)hipEventCreate(&t1);
        (void)hipEventRecord(t0, stream);
        const bool ok = ckw::run_bwd_weight(key.dtype, i, p, in, grad_out, scratch, ws_ptr, stream);
        (void)hipEventRecord(t1, stream);
        (void)hipEventSynchronize(t1);

        float ms = 0.0f;
        (void)hipEventElapsedTime(&ms, t0, t1);
        (void)hipEventDestroy(t0);
        (void)hipEventDestroy(t1);

        if (ok && hipGetLastError() == hipSuccess && (best < 0 || ms < best_ms)) {
            best = i;
            best_ms = ms;
        }
    }
    return best;
}

}  // namespace

c10::optional<torch::Tensor> ck_conv2d_backward_data(torch::Tensor grad_output,
                                                     torch::Tensor weight,
                                                     std::vector<int64_t> input_size,
                                                     std::vector<int64_t> stride,
                                                     std::vector<int64_t> padding,
                                                     std::vector<int64_t> dilation) {
    if (grad_output.dim() != 4 || weight.dim() != 4) return c10::nullopt;
    if (input_size.size() != 4) return c10::nullopt;
    if (grad_output.dtype() != weight.dtype()) return c10::nullopt;
    if (grad_output.dtype() != torch::kFloat16 && grad_output.dtype() != torch::kBFloat16)
        return c10::nullopt;
    if (stride.size() != 2 || padding.size() != 2 || dilation.size() != 2) return c10::nullopt;
    if (weight.size(1) != input_size[1]) return c10::nullopt;  // groups=1 only
    if (weight.size(0) != grad_output.size(1)) return c10::nullopt;  // K must match dY's C dim
    if (grad_output.size(0) != input_size[0]) return c10::nullopt;  // N must match

    const int dtype = grad_output.dtype() == torch::kBFloat16 ? ckw::kBF16 : ckw::kF16;

    ckw::ConvProblem p{};
    p.G = 1;
    p.N = static_cast<int>(input_size[0]);
    p.C = static_cast<int>(input_size[1]);
    p.K = static_cast<int>(weight.size(0));

    // ConvProblem stores spatial dims D,H,W-major with the leading slot
    // unused at ndim==2 -- see ck_conv_fwd.hpp -- so a 2D problem fills
    // [1],[2].
    for (int i = 0; i < 2; ++i) {
        const int64_t in_i = input_size[2 + i];
        const int64_t k_i = weight.size(2 + i);
        const int64_t o_i = conv_out_size(in_i, k_i, stride[i], padding[i], dilation[i]);
        if (o_i <= 0 || o_i != grad_output.size(2 + i)) return c10::nullopt;
        p.in_spatial[1 + i] = static_cast<int>(in_i);
        p.filt[1 + i] = static_cast<int>(k_i);
        p.out_spatial[1 + i] = static_cast<int>(o_i);
        p.stride[1 + i] = static_cast<int>(stride[i]);
        p.dilation[1 + i] = static_cast<int>(dilation[i]);
        p.lpad[1 + i] = static_cast<int>(padding[i]);
        p.rpad[1 + i] = static_cast<int>(padding[i]);
    }

    // CK's WMMA bwd-data instances are channels-last only, same
    // constraint as forward -- see ck_conv_fwd.hpp.
    const auto fmt = torch::MemoryFormat::ChannelsLast;
    const bool grad_out_was_channels_last = grad_output.is_contiguous(fmt);
    const torch::Tensor grad_out_cl = grad_output.contiguous(fmt);
    const torch::Tensor wei_cl = weight.contiguous(fmt);

    std::vector<int64_t> in_shape{p.N, p.C, input_size[2], input_size[3]};
    torch::Tensor grad_input =
        torch::empty(in_shape, grad_output.options().memory_format(fmt));

    hipStream_t stream = c10::hip::getCurrentHIPStream();

    const Key key = make_key(dtype, p);

    int best = -2;
    {
        std::lock_guard<std::mutex> lock(g_mutex_data);
        auto it = g_best_data.find(key);
        if (it != g_best_data.end()) best = it->second;
    }

    if (best == -2) {
        // First touch for this shape: probe into scratch, never
        // `grad_input`, so a losing instance can't leave a partial
        // result behind.
        torch::Tensor scratch = torch::empty(in_shape, grad_output.options().memory_format(fmt));
        best = select_instance_data(key, p, grad_out_cl.data_ptr(), wei_cl.data_ptr(),
                                    scratch.data_ptr(), grad_output.options(), stream);
        std::lock_guard<std::mutex> lock(g_mutex_data);
        g_best_data[key] = best;
    }

    if (best < 0) return c10::nullopt;

    torch::Tensor ws;
    void* ws_ptr = nullptr;
    const size_t ws_bytes = ckw::workspace_bytes_bwd_data(dtype, best, p);
    if (ws_bytes) {
        ws = torch::empty({static_cast<int64_t>(ws_bytes)}, grad_output.options().dtype(torch::kUInt8));
        ws_ptr = ws.data_ptr();
    }

    if (!ckw::run_bwd_data(dtype, best, p, grad_out_cl.data_ptr(), wei_cl.data_ptr(),
                           grad_input.data_ptr(), ws_ptr, stream))
        return c10::nullopt;

    // Hand back dX in the SAME memory format grad_output was handed, per
    // this function's documented contract (ck_conv_bwd_torch.hpp).
    if (!grad_out_was_channels_last) grad_input = grad_input.contiguous();
    return grad_input;
}

c10::optional<torch::Tensor> ck_conv2d_backward_weight(torch::Tensor input,
                                                       torch::Tensor grad_output,
                                                       std::vector<int64_t> weight_size,
                                                       std::vector<int64_t> stride,
                                                       std::vector<int64_t> padding,
                                                       std::vector<int64_t> dilation) {
    if (input.dim() != 4 || grad_output.dim() != 4) return c10::nullopt;
    if (weight_size.size() != 4) return c10::nullopt;
    if (input.dtype() != grad_output.dtype()) return c10::nullopt;
    if (input.dtype() != torch::kFloat16 && input.dtype() != torch::kBFloat16)
        return c10::nullopt;
    if (stride.size() != 2 || padding.size() != 2 || dilation.size() != 2) return c10::nullopt;
    if (weight_size[1] != input.size(1)) return c10::nullopt;  // groups=1 only
    if (weight_size[0] != grad_output.size(1)) return c10::nullopt;  // K must match dY's C dim
    if (input.size(0) != grad_output.size(0)) return c10::nullopt;  // N must match

    const int dtype = input.dtype() == torch::kBFloat16 ? ckw::kBF16 : ckw::kF16;

    ckw::ConvProblem p{};
    p.G = 1;
    p.N = static_cast<int>(input.size(0));
    p.C = static_cast<int>(input.size(1));
    p.K = static_cast<int>(weight_size[0]);

    for (int i = 0; i < 2; ++i) {
        const int64_t in_i = input.size(2 + i);
        const int64_t k_i = weight_size[2 + i];
        const int64_t o_i = conv_out_size(in_i, k_i, stride[i], padding[i], dilation[i]);
        if (o_i <= 0 || o_i != grad_output.size(2 + i)) return c10::nullopt;
        p.in_spatial[1 + i] = static_cast<int>(in_i);
        p.filt[1 + i] = static_cast<int>(k_i);
        p.out_spatial[1 + i] = static_cast<int>(o_i);
        p.stride[1 + i] = static_cast<int>(stride[i]);
        p.dilation[1 + i] = static_cast<int>(dilation[i]);
        p.lpad[1 + i] = static_cast<int>(padding[i]);
        p.rpad[1 + i] = static_cast<int>(padding[i]);
    }

    // CK's WMMA bwd-weight instances are channels-last only, same
    // constraint as forward -- see ck_conv_fwd.hpp.
    const auto fmt = torch::MemoryFormat::ChannelsLast;
    const torch::Tensor in_cl = input.contiguous(fmt);
    const torch::Tensor grad_out_cl = grad_output.contiguous(fmt);

    std::vector<int64_t> wei_shape{p.K, p.C, weight_size[2], weight_size[3]};
    // The weight-gradient output always comes back plain contiguous
    // (standard NCHW-style) rather than mirroring some caller-supplied
    // format: unlike backward-data, this function is handed a
    // weight_size vector, not a weight tensor, so there is no original
    // memory format to preserve, and a plain contiguous gradient matches
    // what conv weight tensors normally are.
    torch::Tensor grad_weight_cl =
        torch::empty(wei_shape, input.options().memory_format(fmt));

    hipStream_t stream = c10::hip::getCurrentHIPStream();

    const Key key = make_key(dtype, p);

    int best = -2;
    {
        std::lock_guard<std::mutex> lock(g_mutex_weight);
        auto it = g_best_weight.find(key);
        if (it != g_best_weight.end()) best = it->second;
    }

    if (best == -2) {
        torch::Tensor scratch = torch::empty(wei_shape, input.options().memory_format(fmt));
        best = select_instance_weight(key, p, in_cl.data_ptr(), grad_out_cl.data_ptr(),
                                      scratch.data_ptr(), input.options(), stream);
        std::lock_guard<std::mutex> lock(g_mutex_weight);
        g_best_weight[key] = best;
    }

    if (best < 0) return c10::nullopt;

    torch::Tensor ws;
    void* ws_ptr = nullptr;
    const size_t ws_bytes = ckw::workspace_bytes_bwd_weight(dtype, best, p);
    if (ws_bytes) {
        ws = torch::empty({static_cast<int64_t>(ws_bytes)}, input.options().dtype(torch::kUInt8));
        ws_ptr = ws.data_ptr();
    }

    if (!ckw::run_bwd_weight(dtype, best, p, in_cl.data_ptr(), grad_out_cl.data_ptr(),
                             grad_weight_cl.data_ptr(), ws_ptr, stream))
        return c10::nullopt;

    return grad_weight_cl.contiguous();
}
