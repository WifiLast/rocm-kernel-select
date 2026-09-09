// pybind11 registration for the vendored depthwise conv1d kernels in this
// directory -- NOT part of upstream FlashFFTConv (which registers
// conv1d_fwd/conv1d_bwd inside its own monolithic monarch.cpp alongside
// every Monarch-FFT/butterfly kernel). Written fresh so this kernel builds
// as its own independent extension; see NOTICE.md for why.
#include "conv1d.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("conv1d_forward", &conv1d_fwd, "Depthwise conv1d forward (HIP/CUDA)");
    m.def("conv1d_backward", &conv1d_bwd, "Depthwise conv1d backward (HIP/CUDA)");
}
