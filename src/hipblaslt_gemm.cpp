#include "hipblaslt_gemm.hpp"

#ifdef AMD_TUNED_TORCH_HAS_HIPBLASLT

#include <c10/hip/HIPStream.h>
#include <hipblaslt/hipblaslt.h>

#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

// ---------------------------------------------------------------------
// ROW-MAJOR TORCH ON A COLUMN-MAJOR LIBRARY.
//
// hipBLASLt is column-major; torch tensors here are row-major. Rather than
// transpose any data, every problem below is expressed as its own
// transpose, which costs nothing: a row-major [R, C] buffer with row stride
// C *is* a column-major [C, R] buffer with ld C, reinterpreted. So instead
// of computing D = A*B we compute D^T = B^T * A^T, which is the same bytes
// in the same places.
//
// F.linear, Y[M,N] = X[M,K] @ W[N,K]^T:
//     Y row-major [M,N] ld N  ==  col-major [N,M] ld N   <- D
//     X row-major [M,K] ld K  ==  col-major [K,M] ld K
//     W row-major [N,K] ld K  ==  col-major [K,N] ld K
//   D[N,M] = op(A)[N,K] * op(B)[K,M] with A = W (op=T), B = X (op=N).
//   The bias vector then has to be as long as D's row count, which is N --
//   exactly F.linear's [N] bias, so it needs no reshaping either.
//
// torch.bmm, C[b,M,N] = A[b,M,K] @ B[b,K,N]:
//     C col-major [N,M] ld N,  A col-major [K,M] ld K,  B col-major [N,K] ld N
//   D[N,M] = op(A')[N,K] * op(B')[K,M] with A' = B (op=N), B' = A (op=N).
//   Note this swaps which torch operand is hipBLASLt's "A".
// ---------------------------------------------------------------------

// Everything hipBLASLt needs to describe one matmul, already swapped into
// the column-major form above. Doubles as the algo-cache key.
struct LtProblem {
    int dtype;  // hipDataType, as int so the struct stays trivially comparable
    int op_a, op_b;
    int64_t rows_a, cols_a, ld_a, stride_a;
    int64_t rows_b, cols_b, ld_b, stride_b;
    int64_t rows_d, cols_d, ld_d, stride_d;
    int32_t batch;
    int epilogue;    // hipblasLtEpilogue_t
    int bias_dtype;  // hipDataType, or -1 for no bias

    bool operator==(const LtProblem& o) const {
        return dtype == o.dtype && op_a == o.op_a && op_b == o.op_b &&
               rows_a == o.rows_a && cols_a == o.cols_a && ld_a == o.ld_a &&
               stride_a == o.stride_a && rows_b == o.rows_b && cols_b == o.cols_b &&
               ld_b == o.ld_b && stride_b == o.stride_b && rows_d == o.rows_d &&
               cols_d == o.cols_d && ld_d == o.ld_d && stride_d == o.stride_d &&
               batch == o.batch && epilogue == o.epilogue && bias_dtype == o.bias_dtype;
    }
};

struct LtProblemHash {
    size_t operator()(const LtProblem& p) const {
        size_t h = 1469598103934665603ull;
        auto mix = [&h](int64_t v) { h = (h ^ static_cast<size_t>(v)) * 1099511628211ull; };
        mix(p.dtype); mix(p.op_a); mix(p.op_b);
        mix(p.rows_a); mix(p.cols_a); mix(p.ld_a); mix(p.stride_a);
        mix(p.rows_b); mix(p.cols_b); mix(p.ld_b); mix(p.stride_b);
        mix(p.rows_d); mix(p.cols_d); mix(p.ld_d); mix(p.stride_d);
        mix(p.batch); mix(p.epilogue); mix(p.bias_dtype);
        return h;
    }
};

// hipBLASLt's heuristic returns algorithms "in order of increasing
// estimated compute time" -- an estimate, from a library that ships
// fallback logic for some gfx1100 cases. This times the first few for
// real, once per distinct problem, and caches the winner. Same
// benchmark-once-then-cache policy as run_conv2d_fp16's tile-shape sweep
// and ck_conv_torch.cpp's instance sweep; the cost is a handful of extra
// launches on the first call at a new shape.
constexpr int kMaxAlgos = 8;
constexpr int kWarmups = 2;
// 32 MiB. Split-K kernels are the ones that want a workspace, and they are
// exactly the ones that win on the skinny shapes hipBLASLt is here for --
// a zero cap would quietly exclude them from the heuristic's results.
constexpr uint64_t kMaxWorkspaceBytes = 32ull * 1024 * 1024;

struct CachedAlgo {
    hipblasLtMatmulAlgo_t algo;
    size_t workspace_bytes;
    bool usable;  // false caches "no algorithm supports this problem"
};

std::mutex g_mutex;
std::unordered_map<LtProblem, CachedAlgo, LtProblemHash> g_algos;

// One handle for the process. hipblasLtCreate is not free and the handle is
// documented as usable from multiple threads.
hipblasLtHandle_t lt_handle() {
    static hipblasLtHandle_t handle = nullptr;
    static std::once_flag once;
    std::call_once(once, [] {
        if (hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS) handle = nullptr;
    });
    return handle;
}

// RAII for the descriptor set, which is rebuilt per call. Torch's own
// hipBLASLt path does the same; the objects are cheap next to the launch,
// and caching them would mean either sharing mutable state across streams
// or holding a lock across the matmul itself.
struct Descriptors {
    hipblasLtMatmulDesc_t desc = nullptr;
    hipblasLtMatrixLayout_t a = nullptr, b = nullptr, d = nullptr;
    bool ok = false;

    explicit Descriptors(const LtProblem& p, const void* bias_ptr) {
        const auto dt = static_cast<hipDataType>(p.dtype);
        if (hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) !=
            HIPBLAS_STATUS_SUCCESS)
            return;

        const auto op_a = static_cast<hipblasOperation_t>(p.op_a);
        const auto op_b = static_cast<hipblasOperation_t>(p.op_b);
        if (hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_a,
                                            sizeof(op_a)) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_b,
                                            sizeof(op_b)) != HIPBLAS_STATUS_SUCCESS)
            return;

        if (p.epilogue != HIPBLASLT_EPILOGUE_DEFAULT) {
            const auto epi = static_cast<hipblasLtEpilogue_t>(p.epilogue);
            if (hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epi,
                                                sizeof(epi)) != HIPBLAS_STATUS_SUCCESS)
                return;
        }
        if (p.bias_dtype >= 0) {
            const auto bias_dt = static_cast<hipDataType>(p.bias_dtype);
            if (hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                &bias_ptr, sizeof(bias_ptr)) !=
                    HIPBLAS_STATUS_SUCCESS ||
                hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                &bias_dt, sizeof(bias_dt)) !=
                    HIPBLAS_STATUS_SUCCESS)
                return;
        }

        if (!layout(&a, dt, p.rows_a, p.cols_a, p.ld_a, p.batch, p.stride_a) ||
            !layout(&b, dt, p.rows_b, p.cols_b, p.ld_b, p.batch, p.stride_b) ||
            !layout(&d, dt, p.rows_d, p.cols_d, p.ld_d, p.batch, p.stride_d))
            return;

        ok = true;
    }

    ~Descriptors() {
        if (d) hipblasLtMatrixLayoutDestroy(d);
        if (b) hipblasLtMatrixLayoutDestroy(b);
        if (a) hipblasLtMatrixLayoutDestroy(a);
        if (desc) hipblasLtMatmulDescDestroy(desc);
    }

    Descriptors(const Descriptors&) = delete;
    Descriptors& operator=(const Descriptors&) = delete;

private:
    static bool layout(hipblasLtMatrixLayout_t* out, hipDataType dt, int64_t rows, int64_t cols,
                       int64_t ld, int32_t batch, int64_t stride) {
        if (hipblasLtMatrixLayoutCreate(out, dt, rows, cols, ld) != HIPBLAS_STATUS_SUCCESS)
            return false;
        if (batch > 1) {
            if (hipblasLtMatrixLayoutSetAttribute(*out, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                  &batch, sizeof(batch)) !=
                    HIPBLAS_STATUS_SUCCESS ||
                hipblasLtMatrixLayoutSetAttribute(*out,
                                                  HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &stride, sizeof(stride)) !=
                    HIPBLAS_STATUS_SUCCESS)
                return false;
        }
        return true;
    }
};

bool matmul(const Descriptors& dsc, const void* a, const void* b, void* d,
            const hipblasLtMatmulAlgo_t* algo, void* workspace, size_t workspace_bytes,
            hipStream_t stream) {
    const float alpha = 1.0f, beta = 0.0f;
    // C == D, out of place with beta = 0, so C's contents are never read.
    return hipblasLtMatmul(lt_handle(), dsc.desc, &alpha, a, dsc.a, b, dsc.b, &beta, d, dsc.d, d,
                           dsc.d, algo, workspace, workspace_bytes, stream) ==
           HIPBLAS_STATUS_SUCCESS;
}

// Times every algorithm the heuristic offers and caches the fastest.
// `usable == false` is cached too, so a problem hipBLASLt cannot serve pays
// the query once rather than on every call.
CachedAlgo select_algo(const LtProblem& p, const Descriptors& dsc, const void* a, const void* b,
                       void* d, const torch::TensorOptions& opts, hipStream_t stream) {
    CachedAlgo out{};
    out.usable = false;

    hipblasLtMatmulPreference_t pref = nullptr;
    if (hipblasLtMatmulPreferenceCreate(&pref) != HIPBLAS_STATUS_SUCCESS) return out;
    uint64_t ws_limit = kMaxWorkspaceBytes;
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &ws_limit, sizeof(ws_limit));

    hipblasLtMatmulHeuristicResult_t results[kMaxAlgos];
    int found = 0;
    const auto status = hipblasLtMatmulAlgoGetHeuristic(lt_handle(), dsc.desc, dsc.a, dsc.b, dsc.d,
                                                        dsc.d, pref, kMaxAlgos, results, &found);
    hipblasLtMatmulPreferenceDestroy(pref);
    if (status != HIPBLAS_STATUS_SUCCESS || found == 0) return out;

    float best_ms = 0.0f;
    for (int i = 0; i < found; ++i) {
        if (results[i].state != HIPBLAS_STATUS_SUCCESS) continue;

        const size_t ws_bytes = results[i].workspaceSize;
        torch::Tensor ws;
        void* ws_ptr = nullptr;
        if (ws_bytes) {
            ws = torch::empty({static_cast<int64_t>(ws_bytes)}, opts.dtype(torch::kUInt8));
            ws_ptr = ws.data_ptr();
        }

        bool ok = true;
        for (int w = 0; w < kWarmups && ok; ++w)
            ok = matmul(dsc, a, b, d, &results[i].algo, ws_ptr, ws_bytes, stream);
        if (!ok) continue;

        hipEvent_t t0, t1;
        (void)hipEventCreate(&t0);
        (void)hipEventCreate(&t1);
        (void)hipEventRecord(t0, stream);
        ok = matmul(dsc, a, b, d, &results[i].algo, ws_ptr, ws_bytes, stream);
        (void)hipEventRecord(t1, stream);
        (void)hipEventSynchronize(t1);
        float ms = 0.0f;
        (void)hipEventElapsedTime(&ms, t0, t1);
        (void)hipEventDestroy(t0);
        (void)hipEventDestroy(t1);
        if (!ok) continue;

        if (!out.usable || ms < best_ms) {
            best_ms = ms;
            out.algo = results[i].algo;
            out.workspace_bytes = ws_bytes;
            out.usable = true;
        }
    }
    return out;
}

// The one path both public entry points take, once they have expressed
// themselves in column-major terms.
c10::optional<torch::Tensor> run(const LtProblem& p, const torch::Tensor& a,
                                 const torch::Tensor& b, const c10::optional<torch::Tensor>& bias,
                                 torch::Tensor out) {
    if (lt_handle() == nullptr) return c10::nullopt;

    const void* bias_ptr = bias.has_value() ? bias->data_ptr() : nullptr;
    Descriptors dsc(p, bias_ptr);
    if (!dsc.ok) return c10::nullopt;

    const auto stream = c10::hip::getCurrentHIPStream().stream();
    const auto opts = out.options();

    CachedAlgo cached;
    bool have_cached = false;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_algos.find(p);
        if (it != g_algos.end()) {
            cached = it->second;
            have_cached = true;
        }
    }

    if (!have_cached) {
        cached = select_algo(p, dsc, a.data_ptr(), b.data_ptr(), out.data_ptr(), opts, stream);
        std::lock_guard<std::mutex> lock(g_mutex);
        g_algos[p] = cached;
    }
    if (!cached.usable) return c10::nullopt;

    torch::Tensor ws;
    void* ws_ptr = nullptr;
    if (cached.workspace_bytes) {
        ws = torch::empty({static_cast<int64_t>(cached.workspace_bytes)},
                          opts.dtype(torch::kUInt8));
        ws_ptr = ws.data_ptr();
    }
    if (!matmul(dsc, a.data_ptr(), b.data_ptr(), out.data_ptr(), &cached.algo, ws_ptr,
                cached.workspace_bytes, stream))
        return c10::nullopt;
    return out;
}

// -1 for anything hipBLASLt has no data type for, which is how fp64 and the
// integer dtypes decline without a separate check at each call site.
int hip_dtype(torch::ScalarType t) {
    switch (t) {
        case torch::kHalf: return HIP_R_16F;
        case torch::kBFloat16: return HIP_R_16BF;
        case torch::kFloat: return HIP_R_32F;
        default: return -1;
    }
}

int map_epilogue(int64_t requested, bool has_bias) {
    switch (requested) {
        case AMD_TUNED_TORCH_EPI_GELU:
            return has_bias ? HIPBLASLT_EPILOGUE_GELU_BIAS : HIPBLASLT_EPILOGUE_GELU;
        case AMD_TUNED_TORCH_EPI_SILU:
            return has_bias ? HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT : HIPBLASLT_EPILOGUE_SWISH_EXT;
        case AMD_TUNED_TORCH_EPI_RELU:
            return has_bias ? HIPBLASLT_EPILOGUE_RELU_BIAS : HIPBLASLT_EPILOGUE_RELU;
        default:
            return has_bias ? HIPBLASLT_EPILOGUE_BIAS : HIPBLASLT_EPILOGUE_DEFAULT;
    }
}

}  // namespace

c10::optional<torch::Tensor> hipblaslt_linear(torch::Tensor input, torch::Tensor weight,
                                              c10::optional<torch::Tensor> bias,
                                              int64_t epilogue) {
    if (!input.is_cuda() || !weight.is_cuda()) return c10::nullopt;
    if (input.dim() < 2 || weight.dim() != 2) return c10::nullopt;
    if (input.scalar_type() != weight.scalar_type()) return c10::nullopt;
    const int dt = hip_dtype(input.scalar_type());
    if (dt < 0) return c10::nullopt;
    if (bias.has_value() && (!bias->is_cuda() || bias->dim() != 1 ||
                             bias->size(0) != weight.size(0) || !bias->is_contiguous()))
        return c10::nullopt;
    // An epilogue was asked for but there is no bias to go with it: still
    // valid (GELU/SiLU/ReLU alone), so only the BIAS-only request needs one.
    if (epilogue == AMD_TUNED_TORCH_EPI_BIAS && !bias.has_value()) return c10::nullopt;

    // Contiguity, not just shape: the column-major reinterpretation above is
    // only valid for a packed row-major buffer. A non-contiguous input is
    // made contiguous (a view-producing caller like attention's reshape is
    // the common case and costs one copy); a non-contiguous weight is
    // declined, since that is a caller doing something unusual with weights
    // and copying them per call would be the wrong trade.
    const auto x = input.dim() == 2 ? input.contiguous()
                                    : input.reshape({-1, input.size(-1)}).contiguous();
    if (!weight.is_contiguous()) return c10::nullopt;
    if (x.size(1) != weight.size(1)) return c10::nullopt;

    const int64_t M = x.size(0), K = x.size(1), N = weight.size(0);
    if (M == 0 || N == 0 || K == 0) return c10::nullopt;

    LtProblem p{};
    p.dtype = dt;
    p.op_a = HIPBLAS_OP_T;  // W stored col-major [K,N]; transposed gives [N,K]
    p.op_b = HIPBLAS_OP_N;  // X stored col-major [K,M]
    p.rows_a = K; p.cols_a = N; p.ld_a = K; p.stride_a = 0;
    p.rows_b = K; p.cols_b = M; p.ld_b = K; p.stride_b = 0;
    p.rows_d = N; p.cols_d = M; p.ld_d = N; p.stride_d = 0;
    p.batch = 1;
    p.bias_dtype = bias.has_value() ? hip_dtype(bias->scalar_type()) : -1;
    if (bias.has_value() && p.bias_dtype < 0) return c10::nullopt;
    p.epilogue = map_epilogue(epilogue, bias.has_value());

    auto out = torch::empty({M, N}, x.options());
    auto result = run(p, weight, x, bias, out);
    if (!result.has_value()) return c10::nullopt;

    if (input.dim() == 2) return result;
    auto shape = input.sizes().vec();
    shape.back() = N;
    return result->view(shape);
}

c10::optional<torch::Tensor> hipblaslt_bmm(torch::Tensor a, torch::Tensor b) {
    if (!a.is_cuda() || !b.is_cuda()) return c10::nullopt;
    if (a.dim() != 3 || b.dim() != 3) return c10::nullopt;
    if (a.scalar_type() != b.scalar_type()) return c10::nullopt;
    const int dt = hip_dtype(a.scalar_type());
    if (dt < 0) return c10::nullopt;
    if (a.size(0) != b.size(0) || a.size(2) != b.size(1)) return c10::nullopt;

    const auto ac = a.contiguous();
    const auto bc = b.contiguous();
    const int64_t B = ac.size(0), M = ac.size(1), K = ac.size(2), N = bc.size(2);
    if (B == 0 || M == 0 || N == 0 || K == 0) return c10::nullopt;
    if (B > std::numeric_limits<int32_t>::max()) return c10::nullopt;

    LtProblem p{};
    p.dtype = dt;
    // Both operands are already in the orientation the transposed problem
    // wants, so neither is transposed -- but hipBLASLt's "A" is torch's
    // second operand here. See the layout note at the top of this file.
    p.op_a = HIPBLAS_OP_N;
    p.op_b = HIPBLAS_OP_N;
    p.rows_a = N; p.cols_a = K; p.ld_a = N; p.stride_a = K * N;
    p.rows_b = K; p.cols_b = M; p.ld_b = K; p.stride_b = M * K;
    p.rows_d = N; p.cols_d = M; p.ld_d = N; p.stride_d = M * N;
    p.batch = static_cast<int32_t>(B);
    p.bias_dtype = -1;
    p.epilogue = HIPBLASLT_EPILOGUE_DEFAULT;

    auto out = torch::empty({B, M, N}, ac.options());
    return run(p, bc, ac, c10::nullopt, out);
}

#endif  // AMD_TUNED_TORCH_HAS_HIPBLASLT
