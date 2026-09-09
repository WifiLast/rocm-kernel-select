# third_party/ — vendored build dependencies

Header-only dependencies copied into the tree so a build does not depend on
where this checkout happens to sit on disk. `setup.py` prefers what is here
over the paths it used to reach for; nothing in this directory is edited,
and nothing in it is compiled on its own.

`CuMesh/`, `FlexGEMM/`, `FlashFFTConv/`, `nvdiffrast/`, and `torchsparse/`
are a different category from the three header-only dependencies below
(see "Full external packages" further down) — full installable extensions
with their own setup.py, moved here (not copied) from sibling `source/`
directories so this package is standalone: everything needed to build
amd_tuned_torch, including these five, lives under `source/cmp_ext_turing`.

Everything else here was copied on 2026-08-28 from the machine described below.

| what | copied from | version | size |
|---|---|---|---|
| `composable_kernel/include`, `composable_kernel/library/include` | `/mnt/data/optimization/composable_kernel` | `3df477638d05169dbe73f4d58e3a13ad3ca7b5da`, tag `therock-7.10-963-g3df477638`, CK 1.2.0 | 34 MB |
| `rocm-headers/rocwmma` | `/opt/rocm/include/rocwmma` | ROCm 7.14.0 | 1.5 MB |
| `rocm-headers/hipblaslt` | `/opt/rocm/include/hipblaslt` | ROCm 7.14.0 | 232 KB |

## Why these three, and how each is used

**Composable Kernel** backs three tiers — conv (`src/cuda/ck_conv_fwd*`),
normalization (`ck_norm_fwd*`) and GEMM (`ck_gemm_fwd*`). It is used
header-only: this project instantiates CK's device operations itself rather
than linking `libdevice_operations`, so there is no CK cmake build and
nothing to link. That is exactly what makes vendoring it work. Only
`include/` and `library/include/` are copied; the rest of a CK checkout
(sources, tests, examples, profiler, ~1 GB) is build machinery this project
never invokes.

**rocWMMA** is included by the codegen'd fp16 conv kernels
(`src/cuda/generated/`), which use it to issue gfx1100's WMMA instructions
instead of a manual half2 FMA loop. Also header-only. `setup.py` only
reaches for this copy when `$ROCM_PATH/include/rocwmma` is absent, i.e. on a
machine with ROCm but without the `rocwmma-dev` package.

**hipBLASLt** headers are for `src/hipblaslt_gemm.cpp`. Same fallback rule
as rocWMMA — installed ROCm first, this copy only if that is missing.

## What is deliberately NOT here

**`libhipblaslt.so`.** The shared object is only half of hipBLASLt; the
other half is `/opt/rocm/lib/hipblaslt/`, 2.2 GB of Tensile kernel objects —
including the tuned `navi31` logic that is the entire reason this project
uses hipBLASLt — which the `.so` loads at runtime. A vendored `.so` without
them would link, import, and then fail to dispatch. The library is always
linked from the installed ROCm, and `setup.py` requires it to be present
before enabling that tier.

**PyTorch headers.** They must match the `torch` that will import the built
extension, so they come from the active environment and cannot be pinned
here without pinning torch itself.

**HIP runtime and compiler headers.** Owned by hipcc, on its default search
path, and meaningless apart from the compiler that ships them.

## Consequences of vendoring, stated plainly

A vendored header can drift from the runtime beside it. For CK that cannot
happen — there is no CK runtime; the headers are the whole dependency. For
the two ROCm header sets it can, which is why `setup.py` prefers the
installed ROCm and treats these as a fallback rather than the default. If
you upgrade ROCm and something in `src/hipblaslt_gemm.cpp` stops matching
its library, re-copy from the new `/opt/rocm/include` rather than patching
what is here.

## Refreshing

    cp -a <ck-checkout>/include            third_party/composable_kernel/include
    cp -a <ck-checkout>/library/include    third_party/composable_kernel/library/include
    cp -a $ROCM_PATH/include/rocwmma       third_party/rocm-headers/rocwmma
    cp -a $ROCM_PATH/include/hipblaslt     third_party/rocm-headers/hipblaslt

then update the table above. To build against a checkout instead of this
copy — the normal thing to do when changing CK itself — set
`AMD_TUNED_TORCH_CK_ROOT=/path/to/composable_kernel`; it takes precedence
and this directory is not consulted.

## Full external packages (not header-only)

Unlike the three above, `CuMesh/`, `FlexGEMM/`, `FlashFFTConv/`,
`nvdiffrast/`, and `torchsparse/` are complete, independently installable
PyTorch extensions — each with its own `setup.py`, its own compiled `.so`,
and its own git history (moved here with `.git` intact, not squashed).
They are consumed by `amd_tuned_torch/cumesh_ops.py`, `flexgemm_ops.py`,
`flashfftconv_ops.py`, `nvdiffrast_ops.py`, and `torchsparse_ops.py`
respectively, the same "thin adapter, gated by `available()`, never
raises at import time" shape those modules already use for
`aiter`/`transformer_engine`. Moved into this tree (from what used to be
sibling `source/CuMesh`, `source/FlexGEMM`, `source/flash-fft-conv`,
`source/nvdiffrast`, `source/torchsparse` directories) so that building
this package never depends on anything outside `source/cmp_ext_turing`.

Each was ROCm/HIP-ported alongside this package (see each `*_ops.py`
module's own docstring for exactly what that involved):
- `nvdiffrast/`'s `setup.py` had zero HIP awareness upstream, and its
  hand-written CUDA software rasterizer needed ~35 raw-PTX helper
  functions reimplemented in portable HIP intrinsics.
- `torchsparse/`'s two tensor-core convolution dataflows (ImplicitGEMM,
  FetchOnDemand — five `.cu` files using raw `wmma`/`mma.sync`/`nvcuda::`
  intrinsics with no HIP/rocWMMA-compatible port) are excluded from the
  HIP build entirely (`setup.py`'s `HIP_EXCLUDED_SOURCES`, with matching
  `#if !defined(__HIP_PLATFORM_AMD__)` guards around their pybind11
  registrations in `torchsparse/backend/pybind_cuda.cu`) — only the third,
  portable GatherScatter dataflow is available on this build.
  `amd_tuned_torch.torchsparse_ops` forces that dataflow globally the
  first time its `available()` is checked, since upstream's own default
  is ImplicitGEMM.
- `FlashFFTConv/` — unlike every package above, its core algorithm itself
  (not just an optional dataflow) is implemented entirely with
  `nvcuda::wmma` tensor-core intrinsics across ~50 CUDA header files under
  `csrc/flashfftconv/monarch_cuda/` and `csrc/flashfftconv/butterfly/` —
  there is no non-tensor-core path to fall back to or exclude. Hand-ported
  to rocWMMA for gfx1100 (RDNA3) instead: every file's
  `#include <mma.h>` / `using namespace nvcuda;` boilerplate now branches
  on `__HIP_PLATFORM_AMD__` to a `namespace wmma = rocwmma;` alias, and
  every WMMA fragment's element type was changed from `half`/
  `__nv_bfloat16` to `rocwmma::float16_t`/`rocwmma::bfloat16_t`
  specifically — gfx11's real WMMA instruction dispatch
  (`rocm-headers/rocwmma/internal/wmma_impl.hpp`) only has
  specializations for those two types, not for `rocwmma::hfloat16_t`
  (`__half`), which is what bare `half` resolves to. Every tile in this
  library is already a fixed 16×16×16 fragment (the "16_16_16"/"32_16_16"/
  etc. naming refers to the Monarch FFT decomposition's recursion
  factors, not different WMMA shapes), matching gfx11 rocWMMA's own
  block-size-16 restriction with no changes needed there. Same convention
  this package's own hardware-validated `_vendor/rocwmma_fattn/kernel_fp16.cu`
  already uses. **Not yet validated on real gfx1100 hardware** — see
  `amd_tuned_torch/flashfftconv_ops.py`'s docstring.

**`FlexGEMM/` and `torchsparse/` are built and installed AUTOMATICALLY**
by this package's own `setup.py` (`CachedBuildExtension.run()` calls
`_install_third_party_packages()` right after `amd_tuned_torch`'s own
extension finishes building — see that function's comment in `setup.py`).
They're the two adapters actually wired into amd_tuned_torch's own runtime
op dispatch (`flexgemm_ops.maybe_sparse_conv{1,2,3}d` inside
`_patched_conv2d`/`_patched_conv3d`, and the sparse conv1d fast path in
`miopen_fallback.py`), so leaving those switches always falling through to
dense on a plain `pip install -e .` would be a surprising default.
Propagates `BUILD_TARGET=rocm`/`GPU_ARCHS=<this build's GPU arch>` unless
already set in the environment. Never fails the outer build — a broken or
skipped sub-install just leaves that adapter's `available()` reporting
`False`, printed with the manual retry command. Set
`AMD_TUNED_TORCH_INSTALL_THIRD_PARTY=0` to skip both and install neither
automatically.

**`CuMesh/`, `FlashFFTConv/`, and `nvdiffrast/` stay manual installs** —
plain opt-in library surfaces (`cumesh_ops`/`flashfftconv_ops`/
`nvdiffrast_ops`) that nothing else in this package calls into on its own,
and nvdiffrast's build in particular can take real time, so forcing that
cost on every install for three dependencies nothing here actually needs
would be the wrong default in the other direction:

    cd third_party/CuMesh       && BUILD_TARGET=rocm GPU_ARCHS=gfx1100 pip install --user -e .
    cd third_party/FlashFFTConv && BUILD_TARGET=rocm GPU_ARCHS=gfx1100 pip install --user -e . && cd csrc/flashfftconv && BUILD_TARGET=rocm GPU_ARCHS=gfx1100 pip install --user -e .
    cd third_party/nvdiffrast   && BUILD_TARGET=rocm GPU_ARCHS=gfx1100 pip install --user -e .

(FlashFFTConv needs both installs — its top-level `setup.py` is the
pure-Python `flashfftconv` package, and `csrc/flashfftconv/setup.py` is
the separate `monarch_cuda` native extension it imports. The same manual
command pattern also works for FlexGEMM/torchsparse if you ever need to
rebuild just one of them, e.g. after editing its source directly —
substitute the directory name.)

`amd_tuned_torch.cumesh_ops.available()` / `.flexgemm_ops.available()` /
`.flashfftconv_ops.available()` / `.nvdiffrast_ops.available()` /
`.torchsparse_ops.available()` report `False` (never raise) until the
corresponding package above is installed — exactly like `aiter_ops`/
`te_ops` degrade when `aiter`/`transformer_engine` aren't importable.
