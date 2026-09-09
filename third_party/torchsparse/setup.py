import glob
import os

import torch
import torch.cuda
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    IS_HIP_EXTENSION,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

# from torchsparse import __version__

version_file = open("./torchsparse/version.py")
version = version_file.read().split("'")[1]
print("torchsparse version:", version)

BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")

if BUILD_TARGET == "auto":
    IS_HIP = bool(IS_HIP_EXTENSION)
elif BUILD_TARGET == "cuda":
    IS_HIP = False
elif BUILD_TARGET == "rocm":
    IS_HIP = True
else:
    raise ValueError(f"Invalid BUILD_TARGET={BUILD_TARGET}")

if IS_HIP or (torch.cuda.is_available() and CUDA_HOME is not None) or (
    os.getenv("FORCE_CUDA", "0") == "1"
):
    device = "cuda"
    pybind_fn = f"pybind_{device}.cu"
else:
    device = "cpu"
    pybind_fn = f"pybind_{device}.cpp"

# -------------------------------------------------
# Convolution kernels that rely on NVIDIA-only tensor-core intrinsics
# (wmma / mma.sync / nvcuda::) with no HIP/rocWMMA-compatible equivalent
# in this codebase. They are excluded from HIP builds; the portable
# GatherScatter dataflow (convolution_gather_scatter_cuda.cu) remains
# available as the ROCm conv backend.
# -------------------------------------------------
HIP_EXCLUDED_SOURCES = {
    os.path.normpath(p)
    for p in [
        os.path.join("torchsparse", "backend", "convolution", "convolution_forward_implicit_gemm_cuda.cu"),
        os.path.join("torchsparse", "backend", "convolution", "convolution_forward_implicit_gemm_sorted_cuda.cu"),
        os.path.join("torchsparse", "backend", "convolution", "convolution_backward_wgrad_implicit_gemm_cuda.cu"),
        os.path.join("torchsparse", "backend", "convolution", "convolution_backward_wgrad_implicit_gemm_sorted_cuda.cu"),
        os.path.join("torchsparse", "backend", "convolution", "convolution_forward_fetch_on_demand_cuda.cu"),
    ]
}

sources = [os.path.join("torchsparse", "backend", pybind_fn)]
for fpath in glob.glob(os.path.join("torchsparse", "backend", "**", "*")):
    if (fpath.endswith("_cpu.cpp") and device in ["cpu", "cuda"]) or (
        fpath.endswith("_cuda.cu") and device == "cuda"
    ):
        if IS_HIP and os.path.normpath(fpath) in HIP_EXCLUDED_SOURCES:
            continue
        sources.append(fpath)

extension_type = CUDAExtension if device == "cuda" else CppExtension

# No -std= here on purpose: torch/all.h hard-#errors with "C++20 or later
# compatible compiler is required to use PyTorch" on torch >= 2.9, and
# BuildExtension already appends the standard its own headers need to any
# flag list that does not pin one (append_std17_if_no_std_present() in
# torch/utils/cpp_extension.py), so this tracks the installed torch.
nvcc_flags = ["-O3"]
if IS_HIP:
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    nvcc_flags += [f"--offload-arch={arch}" for arch in archs]

extra_compile_args = {
    "cxx": ["-g", "-O3", "-fopenmp", "-lgomp"],
    "nvcc": nvcc_flags,
}

setup(
    name="torchsparse",
    version=version,
    packages=find_packages(),
    ext_modules=[
        extension_type(
            "torchsparse.backend", sources, extra_compile_args=extra_compile_args
        )
    ],
    url="https://github.com/mit-han-lab/torchsparse",
    install_requires=[
        "numpy",
        "backports.cached_property",
        "tqdm",
        "typing-extensions",
        "wheel",
        "rootpath",
        "torch",
        "torchvision"
    ],
    dependency_links=[
        'https://download.pytorch.org/whl/cu118'
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
