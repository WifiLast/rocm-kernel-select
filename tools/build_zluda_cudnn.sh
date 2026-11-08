#!/usr/bin/env bash
# Build ONLY ZLUDA's cuDNN-ABI shim (third_party/ZLUDA's zluda_dnn9 crate ->
# libcudnn.so.9) for amd_tuned_torch.zluda_cudnn_ops.
#
# This is a separate script rather than a step in setup.py on purpose:
# setup.py builds HIP/C++ extensions with hipcc, ZLUDA is a cargo workspace,
# and this is a multi-minute Rust compile that must not sit inside anything
# the test suite runs.
#
# Scope: `-p zluda_dnn9` pulls in cuda_macros, cuda_types, zluda_common,
# zluda_dnn, miopen-sys and hip_runtime-sys -- and NOT llvm_zluda, the ptx
# compiler, or any of the rest of ZLUDA. Building the full workspace
# compiles LLVM and takes far longer; nothing in this module needs it. The
# two -sys crates carry pre-generated bindings (no bindgen, no ROCm headers
# needed at build time) and just link -lMIOpen and -lamdhip64 out of
# /opt/rocm/lib.
#
# --release is NOT a preference. zluda_dnn/src/impl.rs:363 defines
# unimplemented() as Rust's unimplemented!() macro under
# #[cfg(debug_assertions)], which panics across the C ABI and ABORTS the
# process; only the not(debug_assertions) arm returns
# CUDNN_STATUS_NOT_SUPPORTED. A debug build turns every one of the
# 186 unimplemented cuDNN entry points into a process kill.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ZLUDA_ROOT="${AMD_TUNED_TORCH_ZLUDA_ROOT:-$PROJECT_ROOT/third_party/ZLUDA}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

if ! command -v cargo >/dev/null 2>&1; then
    cat >&2 <<'MSG'
build_zluda_cudnn.sh: no `cargo` on PATH.

ZLUDA is a Rust workspace and needs a Rust toolchain:

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    . "$HOME/.cargo/env"

MSG
    exit 1
fi

if [ ! -d "$ZLUDA_ROOT" ]; then
    echo "build_zluda_cudnn.sh: no ZLUDA checkout at $ZLUDA_ROOT" >&2
    echo "  override with AMD_TUNED_TORCH_ZLUDA_ROOT=/path/to/ZLUDA" >&2
    exit 1
fi

for lib in libMIOpen.so libamdhip64.so; do
    if [ ! -e "$ROCM_PATH/lib/$lib" ]; then
        echo "build_zluda_cudnn.sh: $ROCM_PATH/lib/$lib is missing." >&2
        echo "  zluda_dnn links it directly; the build will fail at link time." >&2
        echo "  Set ROCM_PATH if ROCm lives somewhere else." >&2
        exit 1
    fi
done

echo "==> building zluda_dnn9 (release) in $ZLUDA_ROOT"
cd "$ZLUDA_ROOT"
cargo build --release -p zluda_dnn9 "$@"

TARGET_DIR="$ZLUDA_ROOT/target/release"
BUILT="$TARGET_DIR/libcudnn64_9.so"
if [ ! -f "$BUILT" ]; then
    echo "build_zluda_cudnn.sh: cargo reported success but $BUILT is missing." >&2
    ls -la "$TARGET_DIR" >&2 || true
    exit 1
fi

# zluda_dnn9/Cargo.toml's [package.metadata.zluda] names these two symlinks;
# they are normally created by ZLUDA's own xtask, which we are deliberately
# not running (it builds the whole workspace). zluda_cudnn_ops looks for
# libcudnn.so.9 first, so make them here.
for name in libcudnn.so.9 libcudnn.so; do
    ln -sf "$(basename "$BUILT")" "$TARGET_DIR/$name"
done

cat <<MSG

==> built: $BUILT
    symlinked: $TARGET_DIR/libcudnn.so.9 -> $(basename "$BUILT")
               $TARGET_DIR/libcudnn.so   -> $(basename "$BUILT")

Enable it (off by default -- it dlopens the shim and creates a MIOpen handle):

    export AMD_TUNED_TORCH_ENABLE_ZLUDA_CUDNN=1
    python -c "import amd_tuned_torch.zluda_cudnn_ops as z; print(z.describe())"

Then check it against stock torch before trusting it -- nothing about this
path has been validated on hardware:

    python - <<'PY'
    import os, torch
    import amd_tuned_torch.zluda_cudnn_ops as z
    x = torch.randn(2, 8, 32, 32, device="cuda", dtype=torch.float16)
    w = torch.randn(16, 8, 3, 3, device="cuda", dtype=torch.float16)
    got = z.conv2d(x, w, padding=1)
    ref = torch.nn.functional.conv2d(x, w, padding=1)
    print("max abs diff:", (got.float() - ref.float()).abs().max().item())
    PY

Remember the kernels underneath are MIOpen's -- the same ones stock
torch.nn.functional.conv2d already reaches. This is a cross-check and a
capability surface, not a speedup. See amd_tuned_torch/zluda_cudnn_ops.py.
MSG
