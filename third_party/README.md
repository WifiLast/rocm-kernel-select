# third_party/ — vendored build dependencies

Header-only dependencies copied into the tree so a build does not depend on
where this checkout happens to sit on disk. `setup.py` prefers what is here
over the paths it used to reach for; nothing in this directory is edited,
and nothing in it is compiled on its own.

Everything here was copied on 2026-08-28 from the machine described below.

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
