#include "ck_conv_torch.hpp"

#include "cuda/ck_conv_fwd.hpp"

#include <c10/hip/HIPStream.h>

#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

namespace ckw = amd_tuned_torch_ck;

int64_t conv_out_size(int64_t in, int64_t k, int64_t s, int64_t p, int64_t d) {
    return (in + 2 * p - d * (k - 1) - 1) / s + 1;
}

// Which CK instance won for a given problem. Same shape-keyed
// benchmark-once-then-cache policy as run_conv2d_fp16's variant selection
// in main_rocm.cpp -- CK ships dozens of tuned tile configs and which one
// wins is strongly shape-dependent (a full sweep put the winner at
// instance 3 for 2D and 11 for 3D on the same card), so picking one
// statically would leave most of the performance on the table.
struct Key {
    int ndim, dtype;
    int N, C, K;
    int in_spatial[3], filt[3], stride[3], dil[3], pad[3];
    bool operator==(const Key& o) const {
        if (ndim != o.ndim || dtype != o.dtype || N != o.N || C != o.C || K != o.K) return false;
        for (int i = 0; i < 3; ++i)
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
        mix(k.ndim); mix(k.dtype); mix(k.N); mix(k.C); mix(k.K);
        for (int i = 0; i < 3; ++i) { mix(k.in_spatial[i]); mix(k.filt[i]); mix(k.stride[i]); mix(k.dil[i]); mix(k.pad[i]); }
        return h;
    }
};

std::mutex g_mutex;
// -1 caches "no instance supports this shape", so a shape that can't use
// CK pays the probing cost once rather than on every call.
std::unordered_map<Key, int, KeyHash> g_best;

// Times every supported instance once into `scratch` and returns the
// fastest, or -1 if none is supported.
int select_instance(const Key& key, const ckw::ConvProblem& p, const void* in, const void* wei,
                    const void* bias, void* scratch, const torch::TensorOptions& opts,
                    hipStream_t stream) {
    const int n = ckw::num_instances(key.ndim, key.dtype);
    int best = -1;
    float best_ms = 0.0f;

    for (int i = 0; i < n; ++i) {
        if (!ckw::supported(key.ndim, key.dtype, i, p)) continue;

        torch::Tensor ws;
        void* ws_ptr = nullptr;
        const size_t ws_bytes = ckw::workspace_bytes(key.ndim, key.dtype, i, p);
        if (ws_bytes) {
            ws = torch::empty({static_cast<int64_t>(ws_bytes)}, opts.dtype(torch::kUInt8));
            ws_ptr = ws.data_ptr();
        }

        // 2 warmups then 1 timed run, same shape of measurement
        // run_conv2d_fp16 uses for its own variant sweep.
        for (int w = 0; w < 2; ++w)
            if (!ckw::run(key.ndim, key.dtype, i, p, in, wei, bias, scratch, ws_ptr, stream))
                break;

        hipEvent_t t0, t1;
        (void)hipEventCreate(&t0);
        (void)hipEventCreate(&t1);
        (void)hipEventRecord(t0, stream);
        const bool ok = ckw::run(key.ndim, key.dtype, i, p, in, wei, bias, scratch, ws_ptr, stream);
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

c10::optional<torch::Tensor> ck_conv_forward(torch::Tensor input,
                                             torch::Tensor weight,
                                             c10::optional<torch::Tensor> bias,
                                             std::vector<int64_t> stride,
                                             std::vector<int64_t> padding,
                                             std::vector<int64_t> dilation) {
    const int ndim = static_cast<int>(input.dim()) - 2;
    if (ndim != 2 && ndim != 3) return c10::nullopt;
    if (weight.dim() != input.dim()) return c10::nullopt;
    if (input.dtype() != weight.dtype()) return c10::nullopt;
    if (input.dtype() != torch::kFloat16 && input.dtype() != torch::kBFloat16) return c10::nullopt;
    if (weight.size(1) != input.size(1)) return c10::nullopt;  // groups=1 only
    if (static_cast<int>(stride.size()) != ndim || static_cast<int>(padding.size()) != ndim ||
        static_cast<int>(dilation.size()) != ndim)
        return c10::nullopt;

    const int dtype = input.dtype() == torch::kBFloat16 ? ckw::kBF16 : ckw::kF16;

    ckw::ConvProblem p{};
    p.G = 1;
    p.N = static_cast<int>(input.size(0));
    p.C = static_cast<int>(input.size(1));
    p.K = static_cast<int>(weight.size(0));

    // ConvProblem stores spatial dims D,H,W-major with the leading slots
    // unused at ndim==2, so a 2D problem fills [1],[2].
    const int off = 3 - ndim;
    std::vector<int64_t> out_spatial(ndim);
    for (int i = 0; i < ndim; ++i) {
        const int64_t in_i = input.size(2 + i);
        const int64_t k_i = weight.size(2 + i);
        const int64_t o_i = conv_out_size(in_i, k_i, stride[i], padding[i], dilation[i]);
        if (o_i <= 0) return c10::nullopt;
        out_spatial[i] = o_i;
        p.in_spatial[off + i] = static_cast<int>(in_i);
        p.filt[off + i] = static_cast<int>(k_i);
        p.out_spatial[off + i] = static_cast<int>(o_i);
        p.stride[off + i] = static_cast<int>(stride[i]);
        p.dilation[off + i] = static_cast<int>(dilation[i]);
        p.lpad[off + i] = static_cast<int>(padding[i]);
        p.rpad[off + i] = static_cast<int>(padding[i]);
    }

    // CK's WMMA conv instances are channels-last only (no NGCHW/WMMA
    // instances exist -- see ck_conv_fwd.hpp), so anything else has to be
    // converted. `.contiguous(fmt)` is a no-op when the tensor already has
    // that format, which is the common case inside an inference graph
    // running with PYTORCH_MIOPEN_SUGGEST_NHWC=1.
    const auto fmt = ndim == 2 ? torch::MemoryFormat::ChannelsLast
                               : torch::MemoryFormat::ChannelsLast3d;
    const bool input_was_channels_last = input.is_contiguous(fmt);
    const torch::Tensor in_cl = input.contiguous(fmt);
    const torch::Tensor wei_cl = weight.contiguous(fmt);

    // Bias is fused as a broadcast D tensor, which needs a real pointer;
    // an unbiased conv gets zeros rather than a second instantiation.
    torch::Tensor bias_vec;
    if (bias.has_value() && bias->defined()) {
        if (bias->size(0) != p.K || bias->dtype() != input.dtype()) return c10::nullopt;
        bias_vec = bias->contiguous();
    } else {
        bias_vec = torch::zeros({p.K}, input.options());
    }

    std::vector<int64_t> out_shape{p.N, p.K};
    out_shape.insert(out_shape.end(), out_spatial.begin(), out_spatial.end());
    // Allocate channels-last directly. `empty(...).contiguous(fmt)` would
    // allocate NCHW and then copy the whole output tensor into a second
    // buffer -- ~0.3ms of pure waste at the conv2d bench shape.
    torch::Tensor output = torch::empty(out_shape, input.options().memory_format(fmt));

    hipStream_t stream = c10::hip::getCurrentHIPStream();

    Key key{};
    key.ndim = ndim;
    key.dtype = dtype;
    key.N = p.N; key.C = p.C; key.K = p.K;
    for (int i = 0; i < 3; ++i) {
        key.in_spatial[i] = p.in_spatial[i];
        key.filt[i] = p.filt[i];
        key.stride[i] = p.stride[i];
        key.dil[i] = p.dilation[i];
        key.pad[i] = p.lpad[i];
    }

    int best = -2;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_best.find(key);
        if (it != g_best.end()) best = it->second;
    }

    if (best == -2) {
        // First touch for this shape: probe into scratch, never `output`,
        // so a losing instance can't leave a partial result behind.
        torch::Tensor scratch = torch::empty(out_shape, input.options().memory_format(fmt));
        best = select_instance(key, p, in_cl.data_ptr(), wei_cl.data_ptr(), bias_vec.data_ptr(),
                               scratch.data_ptr(), input.options(), stream);
        std::lock_guard<std::mutex> lock(g_mutex);
        g_best[key] = best;
    }

    if (best < 0) return c10::nullopt;

    torch::Tensor ws;
    void* ws_ptr = nullptr;
    const size_t ws_bytes = ckw::workspace_bytes(ndim, dtype, best, p);
    if (ws_bytes) {
        ws = torch::empty({static_cast<int64_t>(ws_bytes)}, input.options().dtype(torch::kUInt8));
        ws_ptr = ws.data_ptr();
    }

    if (!ckw::run(ndim, dtype, best, p, in_cl.data_ptr(), wei_cl.data_ptr(), bias_vec.data_ptr(),
                  output.data_ptr(), ws_ptr, stream))
        return c10::nullopt;

    // Hand back what the caller gave us: a channels-last caller keeps the
    // format (and pays no copy at all), an NCHW caller gets NCHW.
    if (!input_was_channels_last) output = output.contiguous();
    return output;
}
