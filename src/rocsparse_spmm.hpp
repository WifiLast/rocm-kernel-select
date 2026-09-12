// Torch-facing entry point for the rocSPARSE SpMM tier.
//
// Declared separately from main_rocm.cpp for the same reason
// hipblaslt_gemm.hpp is: this file gains only an include and an m.def, and
// the whole tier compiles out where rocSPARSE's headers/library aren't
// present (see setup.py's detection and AMD_TUNED_TORCH_HAS_ROCSPARSE).
//
// WHY THIS TIER EXISTS. torch.matmul already accepts a sparse `input`
// (torch.sparse_csr/coo) and produces a correct result through ATen's own
// generic sparse dispatch -- this is not filling a correctness gap the way
// flexgemm_ops's sparse-voxel conv tiers are. It exists for the same reason
// hipblaslt_gemm.hpp's dense GEMM tier does: ATen's generic path is not
// guaranteed to be the fastest one available on this card, and rocSPARSE
// (vendored in full at third_party/rocsparse, see setup.py's detection
// comment for the header/library precedence) ships a real vendor-tuned
// generic SpMM (rocsparse_spmm) rocSPARSE's own docs describe as
// deterministic-by-default for CSR.
//
// See amd_tuned_torch/rocsparse_ops.py's module docstring for the Python
// side: the sparse-LAYOUT test that decides when this is even attempted
// (metadata, not a content scan -- unlike flexgemm's occupancy gate), and
// this project's own VALIDATION STATUS posture for a tier written with no
// ROCm toolchain or GPU available to compile or run it against.
#pragma once

#include <torch/extension.h>

// input @ other via rocSPARSE's generic three-stage SpMM
// (buffer_size/preprocess/compute, see rocsparse-generic.h's
// rocsparse_spmm for the full contract this follows). `input` must already
// be a 2D CSR sparse CUDA tensor -- amd_tuned_torch/rocsparse_ops.py
// converts a COO/CSC input to CSR (`Tensor.to_sparse_csr()`) before this is
// ever called, so this C++ side only has one sparse-matrix descriptor to
// build. `other` must be a 2D dense CUDA tensor, same fp16/bf16/fp32 dtype
// as `input`, with other.size(0) == input.size(1). Returns nullopt for
// anything rocSPARSE declines or any shape/dtype this tier doesn't cover --
// same opportunistic-kernel convention as hipblaslt_linear/hipblaslt_bmm.
c10::optional<torch::Tensor> rocsparse_spmm_torch(torch::Tensor input, torch::Tensor other);
