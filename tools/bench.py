"""Benchmark harness for amd_tuned_torch on RX 7900 XTX (gfx1100/RDNA3), ROCm.

Compares stock PyTorch/ROCm ops against amd_tuned_torch's replacements directly
(amd_tuned_torch.aiter_ops.* for the aiter-backed Triton GEMM/int8 ops,
amd_tuned_torch.ops.* for the native HIP group_norm/conv2d/conv3d kernels,
amd_tuned_torch.te_ops.* for the TransformerEngine-backed ones) -- not
through the F.foo monkeypatch, so a speedup (or regression) here isolates
the kernel itself from dispatch overhead.

Requires: a built amd_tuned_torch._native extension (see ../setup.py --
group_norm, conv2d, conv3d), aiter installed (source/aiter) for the
linear/BMM/int8-linear sections (skipped with a warning if unavailable),
and TransformerEngine's ROCm fork installed (source/TransformerEngine) for
the attention/layer_norm/rms_norm/gelu/silu sections (also skipped with a
warning if unavailable -- and disabled by default regardless, see
amd_tuned_torch/te_ops.py; set AMD_TUNED_TORCH_ENABLE_TE=1 to opt in before running this).

Run with:

    python tools/bench.py
"""
import threading
import time

import torch
import torch.nn.functional as F

import amd_tuned_torch

device = torch.device("cuda")

# -----------------------------------------------------------
# Power monitoring -- best-effort, ROCm-first (amdsmi), falling back to
# pynvml if that's what's installed (e.g. cross-checking against an NVIDIA
# box). Never fatal if neither is available.
# -----------------------------------------------------------
try:
    import amdsmi as _power_backend

    _power_backend.amdsmi_init()
    _BACKEND = "amdsmi"
except ImportError:
    try:
        import pynvml as _power_backend

        _power_backend.nvmlInit()
        _BACKEND = "pynvml"
    except ImportError:
        _power_backend = None
        _BACKEND = None


class PowerMonitor:
    """Best-effort average power draw (W) over a benchmarked region. No-op
    (always reports 0.0 W) if neither amdsmi nor pynvml is installed."""

    def __init__(self, device_index=0, interval=0.01):
        self.interval = interval
        self.stop_event = threading.Event()
        self.power_readings = []
        self.thread = None
        self.handle = None

        if _BACKEND == "amdsmi":
            try:
                self.handle = _power_backend.amdsmi_get_processor_handles()[device_index]
            except Exception as exc:
                print(f"[Warning] amdsmi handle unavailable, power monitoring disabled: {exc}")
        elif _BACKEND == "pynvml":
            try:
                self.handle = _power_backend.nvmlDeviceGetHandleByIndex(device_index)
            except Exception as exc:
                print(f"[Warning] NVML handle unavailable, power monitoring disabled: {exc}")

    def _read_power_w(self):
        if _BACKEND == "amdsmi":
            info = _power_backend.amdsmi_get_power_info(self.handle)
            return info["average_socket_power"]
        return _power_backend.nvmlDeviceGetPowerUsage(self.handle) / 1000.0

    def _monitor_loop(self):
        while not self.stop_event.is_set():
            try:
                self.power_readings.append(self._read_power_w())
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        if self.handle is None:
            return
        self.stop_event.clear()
        self.power_readings = []
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()

    def stop(self):
        if self.handle is None:
            return
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()

    def get_avg_power(self):
        if not self.power_readings:
            return 0.0
        return sum(self.power_readings) / len(self.power_readings)


power_monitor = PowerMonitor()


# -----------------------------------------------------------
# Benchmark harness
# -----------------------------------------------------------

def benchmark_op(name, torch_func, custom_func, args, n_warmup=10, n_iter=50, power_monitor=None):
    for _ in range(n_warmup):
        torch_func(*args)
        custom_func(*args)
    torch.cuda.synchronize()

    if power_monitor:
        power_monitor.start()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_iter):
        torch_func(*args)
    end_event.record()
    torch.cuda.synchronize()
    if power_monitor:
        power_monitor.stop()
    t_torch = start_event.elapsed_time(end_event) / n_iter
    p_torch = power_monitor.get_avg_power() if power_monitor else 0.0

    if power_monitor:
        power_monitor.start()
    start_event.record()
    for _ in range(n_iter):
        custom_func(*args)
    end_event.record()
    torch.cuda.synchronize()
    if power_monitor:
        power_monitor.stop()
    t_custom = start_event.elapsed_time(end_event) / n_iter
    p_custom = power_monitor.get_avg_power() if power_monitor else 0.0

    return t_torch, t_custom, p_torch, p_custom


def report(op_desc, shape_desc, t_torch, t_custom, p_torch, p_custom):
    print(f"Op: {op_desc} [{shape_desc}]")
    print(f"  Stock         : {t_torch:8.3f} ms, Avg Power: {p_torch:6.2f} W")
    print(f"  amd_tuned_torch       : {t_custom:8.3f} ms, Avg Power: {p_custom:6.2f} W")
    print(f"  Speedup       : {t_torch / t_custom:8.2f} x")


def run_suite(dtype_name, dtype):
    print(f"\n{'=' * 20} Running Benchmarks for {dtype_name} {'=' * 20}")

    def clean_memory(*tensors):
        for t in tensors:
            del t
        torch.cuda.empty_cache()

    if not amd_tuned_torch.aiter_ops.available():
        print("[Skip] Linear/BMM/Linear(int8): aiter not installed")
    else:
        # --- Linear/BMM (aiter Triton WMMA GEMM) ---
        if dtype in (torch.float16, torch.bfloat16):
            M, K, N = 4096, 4096, 4096
            x = torch.randn(M, K, device=device, dtype=dtype)
            w = torch.randn(N, K, device=device, dtype=dtype)
            b = torch.randn(N, device=device, dtype=dtype)
            result = benchmark_op(
                "Linear", F.linear, amd_tuned_torch.aiter_ops.linear_fp16, (x, w, b),
                power_monitor=power_monitor,
            )
            report("Linear", f"{M}x{K} @ {N}x{K}.T", *result)
            clean_memory(x, w, b)

            # --- BMM (aiter Triton batched GEMM) ---
            B, M, K, N = 16, 1024, 1024, 1024
            x = torch.randn(B, M, K, device=device, dtype=dtype)
            y = torch.randn(B, K, N, device=device, dtype=dtype)
            result = benchmark_op(
                "BMM", torch.bmm, amd_tuned_torch.aiter_ops.bmm_fp16, (x, y), power_monitor=power_monitor
            )
            report("BMM", f"batch={B}, {M}x{K} @ {K}x{N}", *result)
            clean_memory(x, y)
        else:
            print(f"[Skip] Linear/BMM: aiter's Triton GEMM kernels don't cover {dtype_name}")

        # --- Linear, INT8 W8A8 (aiter, opt-in, changes numerics -- see README) ---
        M, K, N = 4096, 4096, 4096
        x = torch.randn(M, K, device=device, dtype=dtype)
        w = torch.randn(N, K, device=device, dtype=dtype)
        b = torch.randn(N, device=device, dtype=dtype)
        result = benchmark_op(
            "Linear (int8)", F.linear, amd_tuned_torch.aiter_ops.linear_int8, (x, w, b),
            power_monitor=power_monitor,
        )
        report("Linear (aiter W8A8)", f"{M}x{K} @ {N}x{K}.T", *result)
        clean_memory(x, w, b)

    # --- Conv2d / Conv3d (hand-written HIP kernels, fp16/fp32 only -- see
    # amd_tuned_torch.ops.conv2d/.conv3d, src/cuda/conv{2,3}d_fp{16,32}.cu).
    # The Conv3d shape below (B=1, k=3x3x3, stride=1, padding=1, even D/H/W)
    # is in scope for src/cuda/conv3d_fp32_winograd.cu -- for dtype=fp32 the
    # first timed iteration also pays for benchmarking that kernel against
    # the direct one (src/main_rocm.cpp's run_conv3d_fp32 caches the winner
    # per-shape), so a couple of warmup iterations before the timed ones
    # matter more here than for the other ops in this file. ---
    if dtype in (torch.float16, torch.float32):
        B, C_in, H_in, W_in = 64, 128, 128, 128
        C_out, K = 64, 3
        x = torch.randn(B, C_in, H_in, W_in, device=device, dtype=dtype)
        cw = torch.randn(C_out, C_in, K, K, device=device, dtype=dtype)
        cb = torch.randn(C_out, device=device, dtype=dtype)
        result = benchmark_op(
            "Conv2d",
            lambda x, w, b: F.conv2d(x, w, b, stride=1, padding=1),
            lambda x, w, b: amd_tuned_torch.ops.conv2d(x, w, b, [1, 1], [1, 1], [1, 1]),
            (x, cw, cb), power_monitor=power_monitor,
        )
        report("Conv2d", f"N={B}, C_in={C_in}, C_out={C_out}, {H_in}x{W_in}, k={K}", *result)
        clean_memory(x, cw, cb)

        B3, C_in3, D_in, H_in3, W_in3 = 1, 512, 8, 32, 32
        C_out3 = 512
        x3 = torch.randn(B3, C_in3, D_in, H_in3, W_in3, device=device, dtype=dtype)
        cw3 = torch.randn(C_out3, C_in3, K, K, K, device=device, dtype=dtype)
        cb3 = torch.randn(C_out3, device=device, dtype=dtype)
        result = benchmark_op(
            "Conv3d",
            lambda x, w, b: F.conv3d(x, w, b, stride=1, padding=1),
            lambda x, w, b: amd_tuned_torch.ops.conv3d(x, w, b, [1, 1, 1], [1, 1, 1], [1, 1, 1]),
            (x3, cw3, cb3), n_iter=20, power_monitor=power_monitor,
        )
        report("Conv3d", f"N={B3}, C_in={C_in3}, C_out={C_out3}, {D_in}x{H_in3}x{W_in3}, k={K}", *result)
        clean_memory(x3, cw3, cb3)
    else:
        print(f"[Skip] Conv2d/Conv3d: native HIP kernels don't cover {dtype_name}")

    # --- GroupNorm (hand-written HIP kernel) ---
    N_b, C, H, W, groups = 32, 128, 64, 64, 32
    x = torch.randn(N_b, C, H, W, device=device, dtype=dtype)
    gw = torch.randn(C, device=device, dtype=dtype)
    gb = torch.randn(C, device=device, dtype=dtype)
    result = benchmark_op(
        "GroupNorm",
        lambda x, w, b: F.group_norm(x, groups, w, b, eps=1e-5),
        lambda x, w, b: amd_tuned_torch.ops.group_norm(x, groups, w, b, 1e-5),
        (x, gw, gb), power_monitor=power_monitor,
    )
    report("GroupNorm", f"N={N_b}, C={C}, {H}x{W}, groups={groups}", *result)
    clean_memory(x, gw, gb)

    if not amd_tuned_torch.te_ops.available():
        print("[Skip] attention/layer_norm/rms_norm/gelu/silu: TransformerEngine not installed")
        return

    # --- LayerNorm (TE fused) ---
    rows, cols = 8192, 4096
    x = torch.randn(rows, cols, device=device, dtype=dtype)
    lw = torch.randn(cols, device=device, dtype=dtype)
    lb = torch.randn(cols, device=device, dtype=dtype)
    result = benchmark_op(
        "LayerNorm",
        lambda x, w, b: F.layer_norm(x, [cols], w, b, eps=1e-5),
        lambda x, w, b: amd_tuned_torch.te_ops.layer_norm(x, [cols], w, b, 1e-5),
        (x, lw, lb), power_monitor=power_monitor,
    )
    report("LayerNorm", f"{rows}x{cols}", *result)
    clean_memory(x, lw, lb)

    # --- RMSNorm (TE fused) ---
    if hasattr(F, "rms_norm"):
        x = torch.randn(rows, cols, device=device, dtype=dtype)
        rw = torch.randn(cols, device=device, dtype=dtype)
        result = benchmark_op(
            "RMSNorm",
            lambda x, w: F.rms_norm(x, [cols], w),
            lambda x, w: amd_tuned_torch.te_ops.rms_norm(x, [cols], w),
            (x, rw), power_monitor=power_monitor,
        )
        report("RMSNorm", f"{rows}x{cols}", *result)
        clean_memory(x, rw)

    # --- GELU / SiLU (TE fused) ---
    x = torch.randn(rows, cols, device=device, dtype=dtype)
    result = benchmark_op(
        "GELU (tanh)",
        lambda x: F.gelu(x, approximate="tanh"),
        amd_tuned_torch.te_ops.gelu,
        (x,), power_monitor=power_monitor,
    )
    report("GELU (tanh)", f"{rows}x{cols}", *result)

    result = benchmark_op(
        "SiLU", F.silu, amd_tuned_torch.te_ops.silu, (x,), power_monitor=power_monitor,
    )
    report("SiLU", f"{rows}x{cols}", *result)
    clean_memory(x)

    # --- Attention (TE fused, CK/AOTriton backend auto-selected) ---
    B, H, S, D = 2, 32, 4096, 128
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)
    result = benchmark_op(
        "Attention",
        lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True),
        lambda q, k, v: amd_tuned_torch.te_ops.scaled_dot_product_attention(q, k, v, is_causal=True),
        (q, k, v), n_iter=20, power_monitor=power_monitor,
    )
    report("Attention (causal)", f"B={B}, H={H}, S={S}, D={D}", *result)
    clean_memory(q, k, v)


if __name__ == "__main__":
    was_enabled = amd_tuned_torch.is_enabled()
    if was_enabled:
        # Benchmark the raw ops (amd_tuned_torch.ops.* / amd_tuned_torch.te_ops.*) against
        # stock F.foo/torch.foo directly -- disable the monkeypatch so
        # F.linear etc. above actually hit stock ROCm, not amd_tuned_torch itself.
        amd_tuned_torch.disable()
    try:
        for dtype_name, dtype in [("FP16", torch.float16), ("BF16", torch.bfloat16),
                                    ("FP32", torch.float32)]:
            run_suite(dtype_name, dtype)
    finally:
        if was_enabled:
            amd_tuned_torch.enable()
