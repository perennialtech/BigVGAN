/* coding=utf-8
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0.
 */

#include <torch/extension.h>
#include <vector>

torch::Tensor fwd_cuda(torch::Tensor const &input,
                       torch::Tensor const &up_filter,
                       torch::Tensor const &down_filter,
                       torch::Tensor const &alpha,
                       torch::Tensor const &inv_beta);

std::vector<torch::Tensor>
bwd_cuda(torch::Tensor const &grad_output, torch::Tensor const &input,
         torch::Tensor const &up_filter, torch::Tensor const &down_filter,
         torch::Tensor const &alpha, torch::Tensor const &inv_beta,
         torch::Tensor const &alpha_over_beta);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fwd_cuda, "Fused alias-free activation forward (CUDA)");
  m.def("backward", &bwd_cuda, "Fused alias-free activation backward (CUDA)");
}
