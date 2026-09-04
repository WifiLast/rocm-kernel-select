import glob
import multiprocessing
import os
import shutil
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# ---------------------------------------------------------------------
# MISSING DEPENDENCIES ARE AN ERROR, NOT A QUIETLY SMALLER BUILD.
#
# Every optional tier here used to degrade silently: no Composable Kernel
# meant no conv/normalization/GEMM tiers, no hipBLASLt headers meant no
# hipBLASLt tier, and in both cases the build SUCCEEDED and printed one
# line saying so among hundreds of compiler lines. The failure then
# surfaced much later and somewhere else -- as
# `_native has no attribute 'has_hipblaslt'`, or as a tier that silently
# never fires and a benchmark that quietly measures stock. That is the
# worst possible place to learn that libhipblaslt-dev was not installed.
#
# So a dependency that is missing is now a hard failure at configure time,
# with a message that says what is missing, where it was looked for, and
# what to do about it.
#
# The distinction that makes this workable is between ABSENT and DECLINED.
# Turning a tier off on purpose -- AMD_TUNED_TORCH_CK_ROOT= (empty) or
# AMD_TUNED_TORCH_HIPBLASLT=0 -- is a decision, and decisions are honoured
# silently. Only a dependency this file expected to find and did not is an
# error. AMD_TUNED_TORCH_ALLOW_MISSING_DEPS=1 restores the old
# skip-what-is-missing behaviour wholesale, for building a deliberately
# reduced extension without naming each tier.
_ALLOW_MISSING = os.environ.get("AMD_TUNED_TORCH_ALLOW_MISSING_DEPS", "0") == "1"


class MissingDependency(RuntimeError):
    pass


def _missing(what, looked_in, install, disable, optional=True):
    """Fail the build for a dependency that is absent rather than declined.

    Returns False instead of raising under AMD_TUNED_TORCH_ALLOW_MISSING_DEPS,
    so the caller can carry on with that tier compiled out.

    `optional=False` means there is no build without this dependency -- hipcc
    and rocWMMA, which every configuration of this extension needs. Those
    raise regardless of the escape hatch: "skipping" them would only move the
    failure to a #include error deep in a .cu file, which is the outcome
    these checks exist to prevent.
    """
    looked = "\n".join(f"      {p}" for p in looked_in)
    message = (
        f"\n\nsetup.py: MISSING BUILD DEPENDENCY -- {what}\n\n"
        f"  Looked in:\n{looked}\n\n"
        f"  To fix it:      {install}\n"
        f"  To build without it, deliberately:\n"
        f"                  {disable}\n\n"
        "  (or set AMD_TUNED_TORCH_ALLOW_MISSING_DEPS=1 to skip every\n"
        "   missing dependency at once and build whatever is available)\n"
    )
    if _ALLOW_MISSING and optional:
        print(f"setup.py: {what} NOT FOUND -- skipping it "
              "(AMD_TUNED_TORCH_ALLOW_MISSING_DEPS=1)")
        return False
    if _ALLOW_MISSING:
        message += ("\n  AMD_TUNED_TORCH_ALLOW_MISSING_DEPS does not apply here: "
                    "this dependency\n  is required by every configuration of this "
                    "extension.\n")
    raise MissingDependency(message)

# pip install -e . --no-build-isolation
#
# Prerequisites (see README.md for the full walkthrough):
#   1. A ROCm-built PyTorch (this is a HIP extension; CUDAExtension here is
#      just PyTorch's build-system entry point -- it hipifies/dispatches to
#      hipcc automatically when ROCM_HOME is set).
#   2. aiter installed (source/aiter) -- required at *import* time (not a
#      build-time dependency of this extension; nothing here links against
#      it) for linear/matmul/bmm to route through amd_tuned_torch's default Triton
#      WMMA GEMM path instead of falling back to stock rocBLAS/hipBLASLt
#      (see amd_tuned_torch/aiter_ops.py). Composable Kernel is NOT needed at all --
#      an earlier version of this extension linked against CK's
#      DeviceGemm instances directly; that's been replaced by aiter's
#      Triton GEMM, which needs no separate C++ library build/link step.
#   3. TransformerEngine's ROCm fork installed (`pip install .` from
#      source/TransformerEngine) -- also import-time only, and disabled by
#      default (see amd_tuned_torch/te_ops.py) -- required for
#      attention/layer_norm/rms_norm/gelu/silu to route through amd_tuned_torch.
#
# This extension compiles group_norm's and conv2d/conv3d's hand-written HIP
# kernels (src/cuda/group_norm.cu, conv{2,3}d_fp32.cu,
# conv3d_fp32_winograd.cu) plus their pybind glue (src/main_rocm.cpp) --
# everything else is pure-Python dispatch into aiter/TransformerEngine.
# fp32 conv2d/conv3d are ports of the original CMP-Turing project's CUDA
# kernels (src/cuda/kernel_example/) -- see each .cu file's header comment
# for what changed in the HIP port. conv3d_fp32_winograd.cu is a second,
# narrower-scope fp32 conv3d kernel (batch=1,
# 3x3x3/stride1/pad1/dilation1 only) -- src/main_rocm.cpp's
# custom_conv3d_forward benchmarks it against conv3d_fp32.cu's direct
# kernel per distinct shape and caches whichever wins, it's never assumed
# to be faster just because it's applicable.
#
# fp16 conv2d/conv3d are NOT hand-maintained .cu files -- they're codegen'd
# (src/cuda/templates/*.cu.tmpl -> tools/kernelgen/generate.py ->
# src/cuda/generated/conv{2,3}d_fp16_*.cu, one file per entry in
# tools/kernelgen/variants.py), committed to the repo so a plain
# `pip install -e .` doesn't require re-running codegen. To add/change a
# variant: edit tools/kernelgen/variants.py, run
# `python tools/kernelgen/generate.py`, commit the result. The glob below
# picks up whatever's in src/cuda/generated/ automatically -- if you see
# the RuntimeError below, that's this step not having been run.
#
# The conv3d_fp16_*.cu glob also picks up conv3d_fp16_winograd_bt8_bc8.cu
# (an opt-in-only Winograd kernel, not part of the WMMA tile-shape variant
# family the glob's name suggests -- naming overlap, not a separate
# pattern) -- this is required, not accidental: src/main_rocm.cpp's
# conv3d_fp16_winograd_bt8_bc8_forward links directly against its
# launch_conv3d_fp16_winograd_bt8_bc8 symbol (see
# amd_tuned_torch.enable_conv3d_winograd_fp16 in amd_tuned_torch/__init__.py
# for the only production caller), so omitting it from the build would be
# an undefined-symbol link error, not a silent no-op.
#
# The generated fp16 kernels additionally need rocWMMA (the rocwmma-dev
# package, or any install that puts <rocwmma/rocwmma.hpp> on the include
# path) -- they use it to issue gfx1100's real WMMA matrix-core
# instructions instead of a manual half2 FMA loop. hipcc searches
# $ROCM_PATH/include (default /opt/rocm/include) automatically, which is
# where rocwmma-dev installs it, so no extra include flag is needed here
# as long as it's installed. fp32 conv2d/conv3d and the winograd kernel
# don't need it.
# ---------------------------------------------------------------------
# TOOLCHAIN. Checked here so a missing piece is one legible line rather
# than a stack trace out of torch's build machinery or, worse, a build that
# succeeds slowly and wrongly.
_rocm_home = (os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME")
              or "/opt/rocm")
if not os.path.isfile(os.path.join(_rocm_home, 'bin', 'hipcc')) and \
        not shutil.which('hipcc'):
    _missing(
        "hipcc (the ROCm compiler) -- this is a HIP extension and cannot be "
        "built without it",
        [os.path.join(_rocm_home, 'bin', 'hipcc'), "$PATH"],
        "install ROCm, or point ROCM_PATH at an existing install",
        "not possible",
        optional=False)

# ninja is not required for correctness -- torch falls back to distutils,
# which compiles this extension's translation units strictly ONE AT A TIME.
# With fifteen Composable Kernel units at minutes each that turns a
# four-minute build into most of an hour, silently, with no indication
# beyond the wall clock. Hence a warning loud enough to notice rather than
# a line lost in the scroll.
if not shutil.which('ninja'):
    print("\n" + "!" * 72)
    print("setup.py: ninja NOT FOUND -- falling back to SERIAL compilation.")
    print("          This extension has 21 Composable Kernel translation units")
    print("          that take minutes each; without ninja they compile one at a")
    print("          time instead of across all cores.  pip install ninja")
    print("!" * 72 + "\n")

GPU_ARCH = os.environ.get("AMD_TUNED_TORCH_GPU_ARCH", "gfx1100")

# ---------------------------------------------------------------------
# Composable Kernel conv tier (OPTIONAL -- src/cuda/ck_conv_fwd.hpp
# explains what it's for and why CK is back after having been removed as
# a GEMM dependency). CK is header-only for our purposes: we instantiate
# its device ops ourselves, so there is nothing to build or link beyond
# adding two include directories -- no libdevice_operations, no CK cmake
# build.
#
# If CK isn't found the whole tier compiles out (AMD_TUNED_TORCH_HAS_CK
# stays undefined and the four instantiating sources are dropped), and the
# extension builds exactly as it did before. Point
# AMD_TUNED_TORCH_CK_ROOT at a CK checkout to enable it; the default is
# the sibling checkout this repo is developed against.
#
# BUILD COST WARNING: CK is 94% of this build's CPU time, measured -- see
# the per-tier table below, where the three tiers get one switch each so
# you can drop the ones you are not working on instead of all of them.
# AMD_TUNED_TORCH_CK_ROOT= (empty) still skips every CK tier at once.
_here = os.path.dirname(os.path.abspath(__file__))

# third_party/ holds vendored copies of the header-only dependencies (see
# third_party/README.md for what was copied, from where, and at which
# commit). Preferring them over the sibling checkout is what makes this
# tree buildable after being moved or copied somewhere else: the old
# default reached three directories up by relative path, which is only
# correct as long as nobody relocates the source.
#
# Precedence: AMD_TUNED_TORCH_CK_ROOT if set (including to empty, which
# disables the tier), then the vendored copy, then the sibling checkout.
# A developer working ON Composable Kernel points the env var at their own
# checkout and the vendored copy is simply not consulted.
_ck_vendored = os.path.join(_here, 'third_party', 'composable_kernel')
_ck_sibling = os.path.join(_here, '..', '..', '..', 'composable_kernel')
_ck_default = (_ck_vendored
               if os.path.isfile(os.path.join(_ck_vendored, 'include', 'ck', 'ck.hpp'))
               else _ck_sibling)
_ck_root = os.environ.get("AMD_TUNED_TORCH_CK_ROOT", _ck_default)
_ck_include = os.path.join(_ck_root, 'include') if _ck_root else None
_ck_lib_include = os.path.join(_ck_root, 'library', 'include') if _ck_root else None
_have_ck = bool(_ck_root) and os.path.isfile(
    os.path.join(_ck_include, 'ck', 'ck.hpp')) and os.path.isdir(_ck_lib_include)

# The CK tier is really THREE tiers that happen to share a dependency, and
# they cost very different amounts to build. Measured on this tree (8 cores,
# from build/temp.*/.ninja_log, total 12433s of CPU across 32 translation
# units for a 24-minute wall clock):
#
#     CK conv      6 TUs   6032s CPU   48.5%   <- the four ck_conv_fwd_*
#     CK gemm     10 TUs   4818s CPU   38.8%   <- the eight ck_gemm_fwd_*
#     CK norm      5 TUs    842s CPU    6.8%
#     everything else      1081s CPU    5.9%
#
# Composable Kernel is 94% of this build. That is not a defect to fix --
# each ck_conv_fwd_* unit instantiates 20 WMMA implicit-GEMM device ops
# (generic + part1 + part2, see ck_conv_fwd_impl.hpp) at roughly 70s each,
# and the build is CPU-saturated rather than critical-path-bound, so
# splitting units further buys nothing. The only ways to spend less are to
# compile fewer instances or to not compile a tier you are not working on.
#
# Hence one switch per tier instead of the single all-or-nothing
# AMD_TUNED_TORCH_CK_ROOT. Working on group_norm and paying 88% of the
# build for conv and GEMM instances you did not touch was the common case,
# and the only escape was to drop all three:
#
#     AMD_TUNED_TORCH_CK_GEMM=0                  ~39% off the build
#     AMD_TUNED_TORCH_CK_CONV=0                  ~49% off the build
#     AMD_TUNED_TORCH_CK_CONV=0 AMD_TUNED_TORCH_CK_GEMM=0   ~87% off (~3 min)
#     AMD_TUNED_TORCH_CK_ROOT=                   all three, as before
#
# Each tier switched off compiles out exactly like a missing CK does: its
# m.def disappears from the module and its Python wrapper's available()
# reports False (each checks its OWN entry point via hasattr, not just
# has_ck()), so callers fall back the way they already do. A tier turned
# off on purpose is a DECISION and stays silent -- the ABSENT/DECLINED
# distinction at the top of this file applies per tier, not just to CK as a
# whole.
_CK_TIERS = ('CONV', 'NORM', 'GEMM')
_ck_want = {t: os.environ.get(f"AMD_TUNED_TORCH_CK_{t}", "1") != "0" for t in _CK_TIERS}

# Sources per tier. The router/seam units (ck_*_fwd.cu, ck_*_torch.cpp) go
# with their own tier: they are what declares and dispatches it, so a tier
# that is off must not compile them either -- they reference the
# instantiating units' symbols and would be an undefined-symbol link error.
_CK_TIER_SOURCES = {
    # The expensive one: four instantiating units (2d/3d x f16/bf16) at
    # ~1450s each. See ck_conv_fwd_impl.hpp for why there is one TU per
    # (rank, dtype) and what a wider instance sweep would cost.
    'CONV': [
        'src/cuda/ck_conv_fwd.cu',
        'src/cuda/ck_conv_fwd_2d_f16.cu',
        'src/cuda/ck_conv_fwd_2d_bf16.cu',
        'src/cuda/ck_conv_fwd_3d_f16.cu',
        'src/cuda/ck_conv_fwd_3d_bf16.cu',
        'src/ck_conv_torch.cpp',
    ],
    # CK normalization tier -- fused GroupNorm+SiLU (src/cuda/ck_norm_fwd.hpp).
    # Same CK checkout, same header-only instantiation, but not the same
    # build cost: blockwise reductions with no WMMA and no implicit-GEMM
    # descriptor machinery, so ~190s per unit rather than ~1450s. One
    # translation unit per dtype, both epilogues (PassThrough and Swish)
    # inside each. (An older comment here claimed these compile "in
    # seconds"; the .ninja_log says 170-196s, so the tier is cheap relative
    # to conv and GEMM but not free.)
    'NORM': [
        'src/cuda/ck_norm_fwd.cu',
        'src/cuda/ck_norm_fwd_f16.cu',
        'src/cuda/ck_norm_fwd_bf16.cu',
        'src/cuda/ck_norm_fwd_f32.cu',
        'src/ck_norm_torch.cpp',
    ],
    # CK WMMA GEMM tier -- a third F.linear candidate with fused
    # bias/GELU/SiLU epilogues (src/cuda/ck_gemm_fwd.hpp). Eight
    # instantiating units (dtype x {none, add, gelu, silu}) at ~550s each:
    # individually a third of a conv unit, collectively nearly as much,
    # because there are twice as many of them.
    'GEMM': [
        'src/cuda/ck_gemm_fwd.cu',
        'src/cuda/ck_gemm_fwd_f16.cu',
        'src/cuda/ck_gemm_fwd_bf16.cu',
        'src/cuda/ck_gemm_fwd_add_f16.cu',
        'src/cuda/ck_gemm_fwd_add_bf16.cu',
        'src/cuda/ck_gemm_fwd_gelu_f16.cu',
        'src/cuda/ck_gemm_fwd_gelu_bf16.cu',
        'src/cuda/ck_gemm_fwd_silu_f16.cu',
        'src/cuda/ck_gemm_fwd_silu_bf16.cu',
        'src/ck_gemm_torch.cpp',
    ],
}

_ck_sources = []
_ck_includes = []
_ck_defines = []
_ck_enabled = []
if _have_ck:
    _ck_enabled = [t for t in _CK_TIERS if _ck_want[t]]
    for t in _ck_enabled:
        _ck_sources += _CK_TIER_SOURCES[t]
    # src/ and src/cuda/ are on the include path because torch's hipify
    # relocates src/cuda/*.cu to src/hip/*.hip at build time, which would
    # otherwise break these sources' relative includes of the shared CK
    # headers (the headers themselves are already HIP and need no hipify).
    _ck_includes = [os.path.abspath(_ck_include), os.path.abspath(_ck_lib_include),
                    os.path.join(_here, 'src'), os.path.join(_here, 'src', 'cuda')]
    # CK gates its fp16/bf16 type support on these; without them the
    # WMMA conv instances don't compile.
    _ck_defines = ['-DCK_ENABLE_FP16=1', '-DCK_ENABLE_BF16=1']
    _ck_defines += [f'-DAMD_TUNED_TORCH_HAS_CK_{t}=1' for t in _ck_enabled]
    if _ck_enabled:
        # AMD_TUNED_TORCH_HAS_CK keeps its old meaning -- "at least one CK
        # tier is compiled in" -- which is what has_ck() has always
        # reported and what the include guards in main_rocm.cpp key on.
        _ck_defines.append('-DAMD_TUNED_TORCH_HAS_CK=1')
        _declined = [t for t in _CK_TIERS if not _ck_want[t]]
        print("setup.py: Composable Kernel tiers ENABLED: "
              f"{', '.join(t.lower() for t in _ck_enabled)} "
              f"(CK at {os.path.abspath(_ck_root)})")
        if _declined:
            print("setup.py: Composable Kernel tiers DISABLED on purpose: "
                  f"{', '.join(f'{t.lower()} (AMD_TUNED_TORCH_CK_{t}=0)' for t in _declined)}")
    else:
        # Every tier declined individually -- same outcome as an empty
        # AMD_TUNED_TORCH_CK_ROOT, reached a different way. Say so rather
        # than printing an "ENABLED:" line with nothing after it.
        _ck_includes = []
        _ck_defines = []
        print("setup.py: Composable Kernel tiers ALL DISABLED "
              f"({', '.join(f'AMD_TUNED_TORCH_CK_{t}=0' for t in _CK_TIERS)})")
elif not _ck_root:
    # AMD_TUNED_TORCH_CK_ROOT= (empty): declined on purpose, not absent.
    print("setup.py: Composable Kernel conv + normalization + GEMM tiers DISABLED "
          "(AMD_TUNED_TORCH_CK_ROOT is empty)")
elif not any(_ck_want.values()):
    # Nothing is switched on, so nothing needs CK -- do not fail the build
    # over headers no enabled tier would have used.
    print("setup.py: Composable Kernel tiers ALL DISABLED "
          f"({', '.join(f'AMD_TUNED_TORCH_CK_{t}=0' for t in _CK_TIERS)})")
else:
    _missing(
        "Composable Kernel headers (conv, normalization and GEMM tiers)",
        [os.path.join(os.path.abspath(_ck_root), 'include', 'ck', 'ck.hpp'),
         os.path.join(os.path.abspath(_ck_root), 'library', 'include')],
        "the vendored copy should be at third_party/composable_kernel -- see "
        "third_party/README.md;\n                  or point "
        "AMD_TUNED_TORCH_CK_ROOT at a CK checkout",
        "AMD_TUNED_TORCH_CK_ROOT= (empty) -- conv2d/conv3d keep their "
        "hand-written kernels,\n                  and linear/group_norm fall back to stock;\n"
        "                  or drop one tier at a time with "
        "AMD_TUNED_TORCH_CK_{CONV,NORM,GEMM}=0")

# ---------------------------------------------------------------------
# hipBLASLt GEMM tier (OPTIONAL -- src/hipblaslt_gemm.hpp explains what it
# is for and shows the rocBLAS-vs-hipBLASLt numbers that motivated it).
#
# Unlike the CK tier this one is a real LINK dependency: libhipblaslt.so is
# a shared library shipped with ROCm, not a header-only instantiation, so
# this adds -lhipblaslt and its lib directory rather than just include
# paths. It costs essentially nothing to build (one ordinary translation
# unit, seconds, no template instantiation) -- the CK tier's BUILD COST
# WARNING above does not apply here.
#
# Detection is by header AND library, both of which ROCm installs by
# default: the header ships in the hipblaslt-dev package and the .so in
# hipblaslt. Finding the header but not the .so would produce a link error
# at the very end of an otherwise-successful build, which is the worst
# moment to discover it, so both are required before the tier is enabled.
#
# If either is missing the whole tier compiles out (AMD_TUNED_TORCH_HAS_HIPBLASLT
# stays undefined, src/hipblaslt_gemm.cpp becomes an empty translation
# unit, and amd_tuned_torch/hipblaslt_ops.available() reports False), and
# linear/matmul/bmm keep the stock-only behaviour they had before this
# tier existed. Set AMD_TUNED_TORCH_HIPBLASLT=0 to force it off while
# iterating, or point ROCM_PATH/ROCM_HOME at a different ROCm install.
_rocm_path = (os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME")
              or "/opt/rocm")
_want_hipblaslt = os.environ.get("AMD_TUNED_TORCH_HIPBLASLT", "1") != "0"
_hipblaslt_libdir = os.path.join(_rocm_path, 'lib')

# The HEADERS may come from third_party/rocm-headers, but the LIBRARY never
# does and cannot: libhipblaslt.so is only half of hipBLASLt. The other half
# is /opt/rocm/lib/hipblaslt/, 2.2 GB of Tensile kernel objects (including
# the tuned navi31 logic this tier exists for) that the .so loads at
# runtime. A vendored .so without them would link, import, and then fail to
# dispatch. So the library is always the installed one, and the vendored
# headers are a fallback for a machine whose ROCm was installed without the
# -dev package -- with the caveat that headers and runtime then come from
# different ROCm versions, which is only safe while those versions agree.
_rocm_include = os.path.join(_rocm_path, 'include')
_vendored_rocm_include = os.path.join(_here, 'third_party', 'rocm-headers')
if os.path.isfile(os.path.join(_rocm_include, 'hipblaslt', 'hipblaslt.h')):
    _hipblaslt_include = _rocm_include
elif os.path.isfile(os.path.join(_vendored_rocm_include, 'hipblaslt', 'hipblaslt.h')):
    _hipblaslt_include = _vendored_rocm_include
else:
    _hipblaslt_include = _rocm_include  # reported as missing below
_have_hipblaslt = _want_hipblaslt and os.path.isfile(
    os.path.join(_hipblaslt_include, 'hipblaslt', 'hipblaslt.h')) and any(
        os.path.exists(os.path.join(_hipblaslt_libdir, name))
        for name in ('libhipblaslt.so', 'libhipblaslt.so.1'))

# src/hipblaslt_gemm.cpp is compiled either way -- it is #ifdef'd on
# AMD_TUNED_TORCH_HAS_HIPBLASLT internally and collapses to nothing when
# the tier is off. Keeping it in the source list unconditionally means the
# two configurations differ only by a define and a link flag, so a build
# that works with the tier disabled is real evidence the sources are
# consistent rather than evidence that half of them were skipped.
_hipblaslt_sources = ['src/hipblaslt_gemm.cpp']
_hipblaslt_includes = []
_hipblaslt_libdirs = []
_hipblaslt_libs = []
_hipblaslt_defines = []
if _have_hipblaslt:
    # ROCM_PATH/include is already on hipcc's default search path, but this
    # file is compiled by the HOST compiler (it is .cpp, not .cu -- it
    # launches no kernels of its own), which has no such default. Hence the
    # explicit include dir.
    _hipblaslt_includes = [_hipblaslt_include]
    _hipblaslt_libdirs = [_hipblaslt_libdir]
    _hipblaslt_libs = ['hipblaslt']
    _hipblaslt_defines = ['-DAMD_TUNED_TORCH_HAS_HIPBLASLT=1']
    print(f"setup.py: hipBLASLt GEMM tier ENABLED (hipBLASLt at {_rocm_path})")
elif not _want_hipblaslt:
    # AMD_TUNED_TORCH_HIPBLASLT=0: declined on purpose, not absent.
    print("setup.py: hipBLASLt GEMM tier DISABLED (AMD_TUNED_TORCH_HIPBLASLT=0)")
else:
    # Say which half is missing. They come from different packages and the
    # header is the one that is usually absent, so "install hipblaslt" would
    # be unhelpful advice for the common case.
    _has_header = os.path.isfile(os.path.join(_hipblaslt_include, 'hipblaslt', 'hipblaslt.h'))
    if not _has_header:
        _missing(
            "hipBLASLt headers (hipblaslt/hipblaslt.h)",
            [os.path.join(_rocm_include, 'hipblaslt', 'hipblaslt.h'),
             os.path.join(_vendored_rocm_include, 'hipblaslt', 'hipblaslt.h')],
            "apt install libhipblaslt-dev  (or the equivalent -devel package)",
            "AMD_TUNED_TORCH_HIPBLASLT=0 -- linear/matmul/bmm keep only stock "
            "and the CK GEMM tier")
    else:
        _missing(
            "libhipblaslt.so (the hipBLASLt runtime library)",
            [os.path.join(_hipblaslt_libdir, 'libhipblaslt.so'),
             os.path.join(_hipblaslt_libdir, 'libhipblaslt.so.1')],
            "apt install libhipblaslt  -- note this is NOT vendorable: the .so "
            "loads 2.2 GB of\n                  Tensile kernel objects from "
            "$ROCM_PATH/lib/hipblaslt at runtime",
            "AMD_TUNED_TORCH_HIPBLASLT=0")

# Parallel build. torch's BuildExtension parallelises ONLY through ninja;
# with use_ninja=False it falls through to distutils, which compiles the
# sources of a single extension strictly one at a time. This extension is
# one target with 32 translation units, several of which (the Composable
# Kernel ones) take minutes each, so serial compilation was costing most of
# the wall clock -- ninja cuts it to roughly the slowest single file.
#
# With ninja, this build is CPU-SATURATED rather than critical-path-bound:
# 12433s of CPU over 32 units finished in a 1454s wall clock on 8 cores
# (8.5x average concurrency, slightly above MAX_JOBS because hipcc forks
# sub-processes of its own). That is the number that decides which
# optimisations are worth trying. Splitting the slowest unit -- the 1454s
# ck_conv_fwd_3d_bf16 -- would shorten the critical path and buy nothing,
# because the cores are already busy. Compiling less, or caching more, is
# what helps; see the CK tier switches above and AMD_TUNED_TORCH_CCACHE.
#
# NOTE: an earlier version of this file passed max_workers=cpu_count() to
# BuildExtension.with_options alongside use_ninja=False. That did nothing at
# all: max_workers is not a BuildExtension option, and with_options forwards
# unknown keywords into setuptools' Command.__init__, which accepts **kw and
# quietly turns them into attributes nobody reads. It looked like a parallel
# build and wasn't one.
#
# ninja picks its own job count; MAX_JOBS caps it (torch reads that env var
# directly) if the full machine is too much.
_max_jobs = os.environ.get("MAX_JOBS")
if not _max_jobs:
    # Leave nothing on the table by default, but don't oversubscribe: each
    # hipcc job is single-threaded and memory-hungry on the CK units.
    os.environ["MAX_JOBS"] = str(multiprocessing.cpu_count())

# ---------------------------------------------------------------------
# BUILD CACHE.
#
# THE PROBLEM. A full build of this extension is dominated by the four
# ck_conv_fwd_* translation units at minutes each, and nothing about a
# reinstall makes those sources any different from the last time they were
# compiled. Ninja already knows how to skip unchanged work -- but only
# within one build directory, and distutils derives that directory from
# wherever the build happens to be running. `pip install .` and
# `setup.py build_ext --inplace` therefore do not share a cache, and a
# build driven from a copied or temporary source tree shares one with
# nobody, so each of those pays the full CK compile from scratch even when
# not a single byte of input changed.
#
# THE FIX is one line of substance: pin build_temp to a fixed absolute
# directory so every invocation, however it was launched, writes its object
# files to and reads them from the same place. Ninja's own up-to-date check
# then does the rest -- unchanged sources are not recompiled, and a
# reinstall after an edit to (say) one Python file relinks and stops.
#
# Only build_temp is pinned, deliberately. build_lib is where the finished
# .so is staged for whoever asked for the build -- a wheel's staging
# directory, or the in-place copy -- so pinning that would put the artifact
# somewhere the caller isn't looking, and produce wheels with no extension
# in them. Compilation output is cached; installation output is not.
#
# The leaf directory name is whatever distutils itself chose
# (temp.linux-x86_64-cpython-310 and friends), which keeps the cache keyed
# by platform and Python version the way distutils intends -- two
# interpreters building the same tree do not collide, and an existing
# ./build/temp.* from before this change is picked up rather than orphaned.
# CachedBuildExtension additionally appends the active torch version (see
# _torch_build_suffix()) -- two torch/ROCm builds on one checkout (the
# amd_tuned_torch/_native_loader.py two-environments case) get their OWN
# object cache each, so switching between them is a fast up-to-date ninja
# check in both directions instead of a full recompile every single time
# (every compile command line embeds `-I<torch>/include`, which differs
# per torch install, so without this ninja would see "command line
# changed" for every source on every switch).
#
# AMD_TUNED_TORCH_BUILD_DIR relocates the cache root (default: ./build).
# Point it at a path outside the source tree to share one cache across
# several checkouts, or delete it to force a cold rebuild.
#
# REQUIRED FLAG: `pip install -e .` alone does NOT use this cache. Since
# setuptools 64, a plain editable install builds a PEP 660 "editable wheel"
# (setuptools/command/editable_wheel.py's _create_wheel_file) that hands
# build_ext a FRESH tempfile.TemporaryDirectory(suffix=".build-temp") on
# every single invocation -- finalize_options below still pins that into
# AMD_TUNED_TORCH_BUILD_DIR by leaf name, but the incoming leaf name is
# random each time (observed: build/tmp7_lvcew3.build-temp,
# build/tmpip95rlxy.build-temp, ...), so ninja never finds the SAME
# directory twice and recompiles everything from scratch regardless -- the
# whole point of this section, silently defeated. The build still succeeds
# and deposits a correct .so either way (_deposit_keyed_build copies out of
# build_lib, not build_temp), only the caching is lost. The fix is
# setuptools' own documented escape hatch for a custom build_ext that needs
# a stable directory -- force the pre-PEP-660 "compat" editable-install
# strategy, which calls build_ext in place against the real pinned path:
#
#     pip install -e . --no-build-isolation --config-settings editable_mode=compat
#
# Every `pip install -e .` command in this project's README/setup
# instructions needs that flag for the cache to actually do anything.
_build_cache_root = os.path.abspath(
    os.environ.get("AMD_TUNED_TORCH_BUILD_DIR", os.path.join(_here, 'build')))


# OPTIONAL SECOND LAYER: ccache. The pinned directory above makes a
# reinstall free; it does not help when an object genuinely has to be
# rebuilt for a reason that did not change its output -- a touched header,
# a switched branch, a cache root that was deleted. ccache does, because it
# keys on preprocessed source rather than on timestamps.
#
# It is OFF by default and gated behind AMD_TUNED_TORCH_CCACHE=1 for a
# specific reason rather than caution in general: torch.utils.cpp_extension
# deliberately refuses to apply its own ccache/sccache wrapper on ROCm
# builds (see _wrap_compiler -- "hipcc with ccache/sccache is currently
# broken", with a captured sccache compiler-detection failure), so turning
# it on here is overriding a decision PyTorch made on purpose. Plain ccache
# with hipcc is a different combination than the sccache one that comment
# documents, and it does work for many people, but it is not something this
# build system should assume on your behalf.
#
# The host compiler is wrapped through CXX, which setuptools splits and
# honours. hipcc cannot be wrapped that way -- torch builds its path from
# ROCM_HOME internally -- so it gets a two-line exec wrapper on disk and
# _join_rocm_home is pointed at that. Patching that one function covers
# both places torch resolves hipcc (the ninja path via _get_hipcc_path and
# the serial distutils path).
def _enable_ccache():
    import shutil
    import torch.utils.cpp_extension as _cpp_ext

    ccache = shutil.which("ccache")
    if not ccache:
        print("setup.py: AMD_TUNED_TORCH_CCACHE=1 but ccache is not on PATH -- ignoring")
        return

    cxx = os.environ.get("CXX")
    if cxx and not cxx.startswith(ccache):
        os.environ["CXX"] = f"{ccache} {cxx}"
    elif not cxx:
        os.environ["CXX"] = f"{ccache} c++"

    hipcc = _cpp_ext._join_rocm_home('bin', 'hipcc')
    wrapper = os.path.join(_build_cache_root, 'ccache-hipcc')
    os.makedirs(_build_cache_root, exist_ok=True)
    with open(wrapper, 'w') as f:
        f.write(f'#!/bin/sh\nexec {ccache} {hipcc} "$@"\n')
    os.chmod(wrapper, 0o755)

    _orig_join = _cpp_ext._join_rocm_home

    def _join_rocm_home(*paths):
        # Only hipcc is redirected; every other ROCm path resolves normally.
        if len(paths) == 2 and paths[0] == 'bin' and paths[1].startswith('hipcc'):
            return wrapper
        return _orig_join(*paths)

    _cpp_ext._join_rocm_home = _join_rocm_home
    print(f"setup.py: ccache ENABLED for host compiler and hipcc ({wrapper})")


if os.environ.get("AMD_TUNED_TORCH_CCACHE", "0") == "1":
    _enable_ccache()


# ---------------------------------------------------------------------
# BUILD PROGRESS.
#
# THE PROBLEM. `python setup.py install` printed nothing whatsoever for its
# first nine minutes, then went quiet again for another fifteen. That is
# indistinguishable from a hang, and it was reported as one. Nothing was
# wrong.
#
# Ninja emits a line only when an edge FINISHES -- never when one starts.
# Not with -v, and not on a tty either (measured: two four-second jobs at
# -j8 print both their lines at t=4s, never at t=0; the [1/2] [2/2]
# counters are completion counts). Ninja also fills its job slots roughly
# longest-first, so all eight cores go straight into the four ck_conv_fwd_*
# units at ~1450s each and four of the ck_gemm_fwd_* at ~550s. The first
# completion is therefore ~9 minutes in BY CONSTRUCTION, and the CK conv
# units do not land until ~24.
#
# So the silence is not a symptom to eliminate -- it is the shape of a
# build whose translation units are minutes long. What was missing is
# anything said BEFORE it begins, and any sign of life while it runs.
#
# THE FIX is two prints around torch's own ninja invocation:
#
#   1. Up front, what is about to be compiled: ninja's own dry-run plan,
#      each unit priced from its last measured cost in .ninja_log, and the
#      resulting ETA. "24 units, ~24 min" is a very different thing to sit
#      through than a blank terminal. It also surfaces a NEEDLESS rebuild
#      at the one moment it can still be cancelled -- `ninja -d explain`
#      saying "command line changed" for every unit means a compile flag
#      moved, which is how an edit to this file that changed no kernel
#      source can still cost 24 minutes. That question ("why is it
#      rebuilding all of CK again?") previously had no answer short of
#      copying build.ninja out and running ninja by hand.
#
#   2. During, a heartbeat every 60s counting completions out of ninja's
#      own log, so a live build is distinguishable from a wedged one
#      without opening a second terminal and running ps.
#
# The hook is torch's _run_ninja_build because that is the single point
# that runs after build.ninja has been written and before ninja starts --
# BuildExtension offers nothing else at that seam. Everything here is
# best-effort: a plan that cannot be computed prints nothing and the build
# proceeds untouched, because a diagnostic must never be able to fail a
# build. AMD_TUNED_TORCH_BUILD_PROGRESS=0 turns both off.
def _enable_build_progress():
    import re
    import subprocess
    import threading
    import time

    import torch.utils.cpp_extension as _cpp_ext

    # Wrapping a wrapper would print the plan once per layer. Cheap to make
    # impossible rather than to rely on the single call site below staying
    # single.
    if getattr(_cpp_ext._run_ninja_build, '_amd_tuned_torch_progress', False):
        return
    _orig_run_ninja = _cpp_ext._run_ninja_build

    def _log_path(build_dir):
        return os.path.join(build_dir, '.ninja_log')

    def _past_costs(build_dir):
        """{output path: seconds} from ninja's log -- what each unit cost the
        last time it was built, which is the only cost estimate available
        that reflects THIS machine rather than a number hardcoded here."""
        costs = {}
        try:
            with open(_log_path(build_dir)) as fh:
                for line in fh:
                    if line.startswith('#'):
                        continue
                    cols = line.rstrip('\n').split('\t')
                    if len(cols) < 5:
                        continue
                    try:
                        costs[cols[3]] = (int(cols[1]) - int(cols[0])) / 1000.0
                    except ValueError:
                        continue
        except OSError:
            pass
        return costs

    def _dry_run(build_dir):
        """(outputs ninja would build, its -d explain reasons), or None if
        ninja could not be asked at all.

        None and [] are deliberately different answers: [] means ninja
        looked and found the cache up to date, None means the question
        failed (no ninja on PATH, no build.ninja yet, a malformed one).
        Reporting the second as the first would print "object cache is up
        to date" immediately before torch fails the build on the very same
        directory, which is worse than printing nothing.

        `ninja -n` prints the command of every edge it would run; these
        rules carry no description, so the -o argument in each line names
        the output. Reasons come from stderr."""
        try:
            proc = subprocess.run(
                ['ninja', '-n', '-d', 'explain'],
                cwd=build_dir, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0:
            return None
        outputs = []
        for line in (proc.stdout or '').splitlines():
            match = re.search(r'\s-o\s+(\S+)', line)
            if match:
                outputs.append(match.group(1))
        reasons = [line[len('ninja explain: '):]
                   for line in (proc.stderr or '').splitlines()
                   if line.startswith('ninja explain: ')]
        return outputs, reasons

    def _classify(reasons):
        """Collapse per-output reasons into counts. The distinction that
        matters is 'command line changed' (a flag moved -- every unit
        rebuilds, and you may not have meant to pay for that) versus an
        ordinary source or header edit."""
        buckets = {}
        for reason in reasons:
            if reason.startswith('command line changed'):
                key = 'compile flags changed since the last build'
            elif ' older than most recent input ' in reason:
                key = 'source or header newer than object'
            elif 'is dirty' in reason:
                key = 'dependency dirty'
            else:
                key = 'no record of a previous build'
            buckets[key] = buckets.get(key, 0) + 1
        return buckets

    def _heartbeat(build_dir, total, jobs, stop, period):
        """One line a minute, counting ninja's log rather than watching
        processes: ninja appends to it as each edge completes, so its
        growth is exactly the completion count."""
        started = time.time()
        try:
            with open(_log_path(build_dir)) as fh:
                baseline = sum(1 for _ in fh)
        except OSError:
            baseline = 0
        while not stop.wait(period):
            try:
                with open(_log_path(build_dir)) as fh:
                    done = max(0, sum(1 for _ in fh) - baseline)
            except OSError:
                done = 0
            mins = (time.time() - started) / 60.0
            print(f"setup.py: still compiling -- {done}/{total} units done, "
                  f"{mins:.0f} min elapsed, -j{jobs}", flush=True)

    def _run_ninja_build(build_directory, verbose, error_prefix):
        jobs = os.environ.get("MAX_JOBS", "?")
        plan = _dry_run(build_directory)
        outputs, reasons = plan if plan is not None else ([], [])
        total = len(outputs)
        stop = threading.Event()
        thread = None

        if total:
            costs = _past_costs(build_directory)
            known = [costs[o] for o in outputs if o in costs]
            print(f"setup.py: {total} translation unit(s) to compile at "
                  f"-j{jobs}", flush=True)
            for reason, count in sorted(_classify(reasons).items(),
                                        key=lambda kv: -kv[1]):
                print(f"setup.py:   {count:3d} x {reason}", flush=True)
            if known:
                try:
                    parallel = max(1, int(jobs))
                except ValueError:
                    parallel = 1
                # Two bounds, and the build cannot beat either: the work
                # divided across the cores, and the single longest unit.
                eta = max(sum(known) / parallel, max(known))
                slowest = max(outputs, key=lambda o: costs.get(o, 0.0))
                print(f"setup.py:   ETA ~{eta / 60:.0f} min "
                      f"({len(known)}/{total} priced from .ninja_log; "
                      f"slowest {os.path.basename(slowest)} "
                      f"~{costs[slowest] / 60:.0f} min)", flush=True)
            print("setup.py:   ninja prints a line only when a unit "
                  "FINISHES, so expect silence until the first one does.",
                  flush=True)

            # 0 disables it; any other integer is the period in seconds.
            try:
                period = int(os.environ.get("AMD_TUNED_TORCH_BUILD_HEARTBEAT", "60"))
            except ValueError:
                period = 60
            if period > 0:
                thread = threading.Thread(
                    target=_heartbeat,
                    args=(build_directory, total, jobs, stop, period),
                    daemon=True)
                thread.start()
        elif plan is not None:
            print("setup.py: nothing to compile -- object cache is "
                  "up to date", flush=True)

        try:
            return _orig_run_ninja(build_directory, verbose, error_prefix)
        finally:
            stop.set()
            if thread is not None:
                thread.join(timeout=5)

    _run_ninja_build._amd_tuned_torch_progress = True
    _cpp_ext._run_ninja_build = _run_ninja_build


if os.environ.get("AMD_TUNED_TORCH_BUILD_PROGRESS", "1") != "0":
    _enable_build_progress()


def _torch_build_suffix() -> str:
    """'-torch-<version>', sanitized for a directory name -- e.g.
    '-torch-2.15.0.dev20260820_rocm7.2'. Same sanitization
    amd_tuned_torch/_native_loader.py's build_key() uses for the exact same
    reason: this string only has to be a valid, distinguishing directory
    name, not parseable back into a version.

    WHY THIS EXISTS. Without it, switching the active torch install (e.g.
    2.15 <-> 2.13, or just a ROCm point-release bump) and rebuilding reused
    the SAME build/temp.<platform>-<pyver> directory across versions --
    plausible-looking, but wrong: every translation unit's compile command
    line embeds `-I<torch>/include`, which differs per torch install, so
    ninja correctly detects "command line changed" for literally every
    source and recompiles the whole extension (all three of them, CK
    included) on EVERY switch, in EITHER direction. The cache pinning above
    was solving a different problem (a fresh temp dir per invocation) and
    silently made this one worse, not better, once more than one torch
    build started sharing a checkout -- exactly the two-environments case
    _native_loader.py's own docstring describes.

    Keying the build cache by torch version the same way _native_builds/
    already keys the FINISHED .so fixes this: switching torch versions now
    switches to (or creates) that version's own object cache directory,
    so going back to a torch build you've already compiled against is a
    fast up-to-date ninja check again, not a full CK recompile."""
    try:
        import torch
        safe = torch.__version__.replace("+", "_").replace("/", "_")
        return f"-torch-{safe}"
    except ImportError:
        return ""


class CachedBuildExtension(BuildExtension.with_options(use_ninja=True)):
    """BuildExtension with its object directory pinned. See BUILD CACHE above."""

    def finalize_options(self):
        super().finalize_options()
        # Keep the leaf distutils picked (temp.<platform>-<pyver>), move the
        # root, and append the active torch version -- see
        # _torch_build_suffix()'s own docstring for why that suffix is what
        # actually makes switching torch versions fast rather than thrashing.
        leaf = os.path.basename(os.path.normpath(self.build_temp)) + _torch_build_suffix()
        self.build_temp = os.path.join(_build_cache_root, leaf)
        print(f"setup.py: object cache at {self.build_temp}")

    def run(self):
        super().run()
        self._report_build_cost()
        self._deposit_keyed_build()

    def _report_build_cost(self):
        """Print where the build time actually went, from ninja's own log.

        WHY. "The build is slow" was, until this existed, only answerable by
        finding build/temp.*/.ninja_log by hand and knowing that its columns
        are start/end milliseconds. That is why the cost of the CK tiers was
        estimated in comments rather than measured, and why one of those
        estimates ("the norm units compile in seconds") was wrong by two
        orders of magnitude for years.

        The two numbers that matter are different questions. CPU is total
        work and is what switching a tier off reduces. Wall is what you
        wait for, and equals CPU/cores only while the machine stays
        saturated -- when wall is much larger than CPU/cores the build is
        critical-path-bound instead, and the fix there is splitting the
        slowest unit, not compiling less.

        Best-effort throughout: a missing or malformed log means no report,
        never a failed build after a successful compile.
        """
        log = os.path.join(self.build_temp, '.ninja_log')
        if not os.path.isfile(log):
            return
        try:
            groups, counts, total, wall = {}, {}, 0.0, 0.0
            with open(log) as f:
                for line in f:
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) < 4 or not parts[0].isdigit():
                        continue
                    start, end, out = int(parts[0]), int(parts[1]), parts[3]
                    seconds = (end - start) / 1000.0
                    total += seconds
                    wall = max(wall, end / 1000.0)
                    base = os.path.basename(out)
                    for prefix, label in (('ck_conv', 'CK conv'), ('ck_gemm', 'CK gemm'),
                                          ('ck_norm', 'CK norm'), ('hipblaslt', 'hipBLASLt')):
                        if base.startswith(prefix):
                            key = label
                            break
                    else:
                        key = 'codegen fp16' if 'generated' in out else 'core'
                    groups[key] = groups.get(key, 0.0) + seconds
                    counts[key] = counts.get(key, 0) + 1
        except (OSError, ValueError):
            return
        if not total:
            return
        # Ninja rewrites entries in place, so a log surviving from an
        # earlier build reports that build's timings for anything this one
        # did not recompile. Say so rather than presenting stale numbers as
        # if they were this build's.
        jobs = os.environ.get("MAX_JOBS", "?")
        print(f"\nsetup.py: build cost by tier (from {log};\n"
              "          entries not recompiled this time carry their previous timing)")
        for key in sorted(groups, key=groups.get, reverse=True):
            print(f"          {key:<14s} {counts[key]:3d} TU  {groups[key]:7.0f}s CPU  "
                  f"{100 * groups[key] / total:5.1f}%")
        print(f"          {'TOTAL':<14s} {sum(counts.values()):3d} TU  {total:7.0f}s CPU  "
              f"wall {wall:.0f}s at MAX_JOBS={jobs}")
        if not shutil.which('ccache'):
            print("          (ccache is not installed -- with it, AMD_TUNED_TORCH_CCACHE=1\n"
                  "           makes a rebuild after a touched header near-free)")

    def _deposit_keyed_build(self):
        """Copy the freshly built .so into amd_tuned_torch/_native_builds/<key>/.

        WHY. An editable install points every environment at this one source
        tree, and .py files are happy to be shared that way. A compiled
        extension is not: it links one specific libtorch, and loading it
        under a different PyTorch is an ABI mismatch that segfaults rather
        than raising. With two environments on this tree (torch 2.15 and
        torch 2.13), whichever built last would leave its .so beside
        __init__.py and break the other.

        So each build also lands in a directory keyed by the torch it was
        built against, and amd_tuned_torch/_native_loader.py picks the
        matching one at import. The copy beside __init__.py is left alone --
        it stays the fallback for a normal single-environment checkout.
        """
        import shutil
        import sys as _sys
        sys.path.insert(0, os.path.join(_here, 'amd_tuned_torch'))
        try:
            import _native_loader
        finally:
            sys.path.pop(0)

        key = _native_loader.build_key()
        dest = os.path.join(_here, 'amd_tuned_torch', '_native_builds', key)
        os.makedirs(dest, exist_ok=True)
        for ext in self.extensions:
            built = self.get_ext_fullpath(ext.name)
            if not os.path.isfile(built):
                continue
            target = os.path.join(dest, os.path.basename(built))
            shutil.copy2(built, target)
            print(f"setup.py: deposited {os.path.basename(built)} -> "
                  f"_native_builds/{key}/")


# rocWMMA for the codegen'd fp16 conv kernels. hipcc searches
# $ROCM_PATH/include by default, which is where rocwmma-dev installs it, so
# this normally adds nothing. It exists for the machine that has ROCm but
# not rocwmma-dev: third_party/rocm-headers carries a copy, and rocWMMA is
# header-only, so unlike hipBLASLt there is no runtime half to mismatch.
#
# Unlike the two tiers above, this one is NOT optional: the codegen'd fp16
# kernels are always compiled and always include <rocwmma/rocwmma.hpp>, so
# there is no "disable" to offer -- a missing rocWMMA is simply a build that
# cannot happen. It previously failed as a #include error hundreds of lines
# into a .cu file; now it fails here, saying what to install.
_rocwmma_includes = []
if os.path.isfile(os.path.join(_rocm_include, 'rocwmma', 'rocwmma.hpp')):
    pass  # hipcc finds it on its own default search path
elif os.path.isfile(os.path.join(_vendored_rocm_include, 'rocwmma', 'rocwmma.hpp')):
    _rocwmma_includes = [_vendored_rocm_include]
    print(f"setup.py: rocWMMA from vendored headers ({_vendored_rocm_include})")
else:
    _missing(
        "rocWMMA headers (rocwmma/rocwmma.hpp), required by the codegen'd fp16 "
        "conv kernels",
        [os.path.join(_rocm_include, 'rocwmma', 'rocwmma.hpp'),
         os.path.join(_vendored_rocm_include, 'rocwmma', 'rocwmma.hpp')],
        "apt install rocwmma-dev  (header-only; or restore "
        "third_party/rocm-headers/rocwmma)",
        "not possible -- these kernels are not an optional tier",
        optional=False)

_generated_conv2d_fp16 = sorted(glob.glob('src/cuda/generated/conv2d_fp16_*.cu'))
_generated_conv3d_fp16 = sorted(glob.glob('src/cuda/generated/conv3d_fp16_*.cu'))
if not _generated_conv2d_fp16 or not _generated_conv3d_fp16:
    raise RuntimeError(
        "src/cuda/generated/conv{2,3}d_fp16_*.cu not found -- generate them "
        "first with: python tools/kernelgen/generate.py"
    )

setup(
    name='amd_tuned_torch',
    version='0.2.0',
    # Declaring the Python package is NOT optional here, even though the
    # build "works" without it. Without `packages`, setuptools knows only
    # about the extension module amd_tuned_torch._native, so an editable
    # install drops the compiled .so into
    # site-packages/amd_tuned_torch/_native...so -- a directory with no
    # __init__.py. Python resolves that as a NAMESPACE PACKAGE and it
    # shadows the editable install, so `import amd_tuned_torch` from any
    # directory outside this repo silently yields an empty module: no
    # enable(), no ops, and the auto-patch dead, with no error anywhere.
    # It looks exactly like the extension not being installed.
    packages=['amd_tuned_torch'],
    # package_data for the same reason `packages` above is not optional:
    # `packages=['amd_tuned_torch']` tells setuptools which PYTHON files
    # belong to the package, but says nothing about non-.py files --
    # without this, amd_tuned_torch/_vendor/rocwmma_fattn/{host.cpp,
    # kernel_fp16.cu,kernel_bf16.cu,LICENSE,NOTICE.md} (the vendored
    # FlashAttention-2 kernel flash_attn_rocwmma_ops.py JIT-compiles from
    # at first use) silently do not get installed by a regular `pip
    # install .`/wheel build, or by whichever editable-install mechanism
    # copies files instead of leaving them in place. The failure mode is
    # not an install error -- enable_flash_attn_rocwmma() just warns
    # "JIT build failed: [Errno 2] No such file or directory:
    # .../_vendor/rocwmma_fattn/host.cpp" and leaves
    # F.scaled_dot_product_attention untouched, which looks exactly like
    # the kernel failing to build for a real reason rather than the
    # source files never having been installed at all. include_package_data
    # would also need a MANIFEST.in entry to catch these (it only picks up
    # files tracked by version control by default under some backends,
    # which vendored third-party sources may not be) -- an explicit glob
    # here needs neither.
    package_data={'amd_tuned_torch': ['_vendor/rocwmma_fattn/*']},
    description=(
        'A PyTorch extension for RX 7900 XTX (gfx1100/RDNA3) that monkeypatches '
        'torch/F ops with aiter- and TransformerEngine-backed kernels.'
    ),
    url='https://github.com/eastmoe/cmp_ext',  # upstream this was forked from
    # THREE independent extensions, not one. CK and hipBLASLt used to link
    # into the same .so as the core group_norm/conv2d/conv3d kernels (one
    # CUDAExtension, one PYBIND11_MODULE in src/main_rocm.cpp) -- so editing
    # a single CK GEMM instance forced a relink of everything else, and CK
    # being ~94% of a full build's CPU time (see the BUILD COST WARNING
    # above) made that relink non-trivial even with ninja's incremental
    # compile (a link step still needs every object file, including ones
    # untouched). Splitting into amd_tuned_torch._native (core, always
    # required) / ._native_ck / ._native_hipblaslt means rebuilding one
    # tier only ever recompiles and relinks that tier's own .so -- the other
    # two are untouched, both on disk and in amd_tuned_torch/_native_builds/
    # (CachedBuildExtension._deposit_keyed_build already loops over
    # self.extensions, so this needed no change there). See src/ck_native.cpp
    # and src/hipblaslt_native.cpp for the corresponding PYBIND11_MODULEs,
    # and amd_tuned_torch/_native_loader.py for how each is located and
    # loaded independently at import time (module_name parameter).
    #
    # The rocWMMA nvcc flags (offload-arch, the half-conversions -U) stay
    # only on the core extension: they exist for the codegen'd fp16 conv
    # kernels' <rocwmma/rocwmma.hpp> include, which CK and hipBLASLt sources
    # never touch.
    ext_modules=[
        CUDAExtension(
            name='amd_tuned_torch._native',
            sources=[
                'src/cuda/group_norm.cu',
                'src/cuda/conv2d_fp32.cu',
                'src/cuda/conv3d_fp32.cu',
                'src/cuda/conv3d_fp32_winograd.cu',
                'src/main_rocm.cpp',
            ] + _generated_conv2d_fp16 + _generated_conv3d_fp16,
            include_dirs=_rocwmma_includes,
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    f'--offload-arch={GPU_ARCH}',
                    # Required by the rocWMMA-using generated fp16 kernels.
                    # PyTorch's HIP extension build unconditionally prepends
                    # -D__HIP_NO_HALF_CONVERSIONS__=1 (torch's
                    # COMMON_HIPCC_FLAGS), which deletes __half's
                    # constructor-from-float in <hip/hip_fp16.h>. rocWMMA's
                    # vector.hpp registers hfloat16_t vector types through a
                    # macro that does static_cast<hfloat16_t>(0.0f), so
                    # including <rocwmma/rocwmma.hpp> under that define is a
                    # hard compile error ("no matching conversion for
                    # static_cast from 'float' to 'rocwmma::hfloat16_t'").
                    # Undefining it here works because torch splices these
                    # args in *after* COMMON_HIPCC_FLAGS, so the -U wins.
                    # Only the conversions define needs undoing --
                    # __HIP_NO_HALF_OPERATORS__ is left in place, and no .cu
                    # here includes torch headers, so re-enabling implicit
                    # float<->__half conversion can't create ambiguity
                    # against at::Half's operators.
                    '-U__HIP_NO_HALF_CONVERSIONS__',
                ] + [f'-I{d}' for d in _rocwmma_includes],
            },
        ),
        # Composable Kernel -- always built (src/ck_native.cpp alone, when
        # no tier is enabled, still compiles and exposes has_ck() -> False;
        # see that file's header). Reuses the SAME nvcc baseline
        # (offload-arch, the half-conversions -U) the combined extension
        # used to apply to these exact .cu sources, so their compiled
        # output is unchanged by the split.
        CUDAExtension(
            name='amd_tuned_torch._native_ck',
            sources=['src/ck_native.cpp'] + _ck_sources,
            include_dirs=_ck_includes,
            extra_compile_args={
                'cxx': ['-O3'] + _ck_defines,
                'nvcc': [
                    '-O3',
                    f'--offload-arch={GPU_ARCH}',
                    '-U__HIP_NO_HALF_CONVERSIONS__',
                ] + _ck_defines,
            },
        ),
        # hipBLASLt -- always built, same "compiled either way" convention
        # src/hipblaslt_gemm.cpp already had before this split (internally
        # #ifdef'd on AMD_TUNED_TORCH_HAS_HIPBLASLT). No .cu sources here, so
        # no rocWMMA/offload-arch nvcc flags are needed.
        CUDAExtension(
            name='amd_tuned_torch._native_hipblaslt',
            sources=['src/hipblaslt_native.cpp'] + _hipblaslt_sources,
            include_dirs=_hipblaslt_includes,
            library_dirs=_hipblaslt_libdirs,
            libraries=_hipblaslt_libs,
            # Without this the loader has to find libhipblaslt.so via
            # LD_LIBRARY_PATH/ldconfig at import time. ROCm installs do not
            # reliably put $ROCM_PATH/lib on either, so an extension that
            # linked fine would then fail to import with an undefined-symbol
            # error far from its cause. Baking the path into the .so's RPATH
            # makes the built artifact self-contained against the ROCm it was
            # built against.
            runtime_library_dirs=_hipblaslt_libdirs,
            extra_compile_args={
                'cxx': ['-O3'] + _hipblaslt_defines,
            },
        ),
    ],
    cmdclass={
        'build_ext': CachedBuildExtension
    }
)
