/* coding=utf-8
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>

#include "type_shim.h"

namespace {

constexpr int FILTER_SIZE = 12;
constexpr int RATIO = 2;
constexpr int UPSAMPLE_INPUT_PAD = 5;
constexpr int UPSAMPLE_CROP_LEFT = 15;
constexpr int DOWNSAMPLE_PAD_LEFT = 5;
constexpr float EPS = 1.0e-9f;

template <typename scalar_t>
__device__ __forceinline__ float read_as_float(const scalar_t *ptr,
                                               int64_t idx) {
  return static_cast<float>(ptr[idx]);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t cast_from_float(float value) {
  return static_cast<scalar_t>(value);
}

__device__ __forceinline__ int clamp_int(int value, int low, int high) {
  return max(low, min(value, high));
}

template <typename scalar_t>
__device__ __forceinline__ float
upsample_value_polyphase(const scalar_t *input, int64_t base, int seq_len,
                         int intermediate_index, const float *up_filter,
                         bool interior) {
  float acc = 0.0f;

  /*
   * PyTorch UpSample1d default:
   *
   *   x = replicate_pad(x, 5, 5)
   *   y = 2 * conv_transpose1d(x, filter, stride=2)
   *   y = y[..., 15:-15]
   *
   * For output intermediate index m:
   *
   *   filter_tap = m + UPSAMPLE_CROP_LEFT - 2 * padded_input_index
   *   original_index = padded_input_index - UPSAMPLE_INPUT_PAD
   *
   * Only taps with parity matching m + UPSAMPLE_CROP_LEFT are nonzero in
   * the zero-insertion view, so ratio-2 upsampling becomes a 6-tap
   * polyphase FIR.
   */
  const int parity = (intermediate_index + UPSAMPLE_CROP_LEFT) & 1;

#pragma unroll
  for (int i = 0; i < FILTER_SIZE / RATIO; ++i) {
    const int tap = parity + RATIO * i;
    int input_index = (intermediate_index + UPSAMPLE_CROP_LEFT - tap) / RATIO -
                      UPSAMPLE_INPUT_PAD;

    if (!interior) {
      input_index = clamp_int(input_index, 0, seq_len - 1);
    }

    acc += 2.0f * up_filter[tap] * read_as_float(input, base + input_index);
  }

  return acc;
}

template <typename scalar_t>
__global__ void alias_free_activation_forward_kernel(
    scalar_t *__restrict__ output, const scalar_t *__restrict__ input,
    const float *__restrict__ up_filter_global,
    const float *__restrict__ down_filter_global,
    const float *__restrict__ alpha, const float *__restrict__ inv_beta,
    int batch_size, int channels, int seq_len) {
  extern __shared__ float shared[];

  float *up_filter = shared;
  float *down_filter = shared + FILTER_SIZE;
  float *activated = shared + 2 * FILTER_SIZE;

  const int tid = threadIdx.x;

  if (tid < FILTER_SIZE) {
    up_filter[tid] = up_filter_global[tid];
    down_filter[tid] = down_filter_global[tid];
  }

  __syncthreads();

  const int channel = blockIdx.y;
  const int batch = blockIdx.z;
  const int tile_start = blockIdx.x * blockDim.x;
  const int tile_outputs = min(blockDim.x, seq_len - tile_start);

  if (tile_outputs <= 0) {
    return;
  }

  const int q_start = RATIO * tile_start - DOWNSAMPLE_PAD_LEFT;
  const int q_count = RATIO * tile_outputs + FILTER_SIZE - RATIO;
  const int q_end = q_start + q_count - 1;

  /*
   * Interior condition:
   * - downsample accesses unclamped intermediate indices
   * - upsample polyphase accesses unclamped source indices
   *
   * Conservative branchless interior range for a 12-tap ratio-2 setup:
   *   q >= 6 and q <= 2*T - 7
   */
  const bool interior = q_start >= 6 && q_end <= 2 * seq_len - 7;

  const int64_t base =
      static_cast<int64_t>(seq_len) *
      (static_cast<int64_t>(channel) +
       static_cast<int64_t>(channels) * static_cast<int64_t>(batch));

  const float alpha_value = alpha[channel];
  const float inv_beta_value = inv_beta[channel];

  /*
   * Cooperative tile:
   * Each intermediate sample needed by this output tile is computed once
   * into shared memory. Adjacent threads handle adjacent time indices.
   */
  for (int local_q = tid; local_q < q_count; local_q += blockDim.x) {
    int q = q_start + local_q;

    if (!interior) {
      q = clamp_int(q, 0, 2 * seq_len - 1);
    }

    const bool upsample_interior = interior || (q >= 6 && q <= 2 * seq_len - 7);

    const float v = upsample_value_polyphase(input, base, seq_len, q, up_filter,
                                             upsample_interior);

    const float phase = alpha_value * v;
    const float s = __sinf(phase);

    activated[local_q] = v + inv_beta_value * s * s;
  }

  __syncthreads();

  if (tid < tile_outputs) {
    float acc = 0.0f;

#pragma unroll
    for (int tap = 0; tap < FILTER_SIZE; ++tap) {
      acc += down_filter[tap] * activated[RATIO * tid + tap];
    }

    output[base + tile_start + tid] = cast_from_float<scalar_t>(acc);
  }
}

template <typename scalar_t>
__global__ void alias_free_activation_backward_kernel(
    float *__restrict__ grad_input, float *__restrict__ grad_alpha,
    float *__restrict__ grad_inv_beta, const scalar_t *__restrict__ grad_output,
    const scalar_t *__restrict__ input,
    const float *__restrict__ up_filter_global,
    const float *__restrict__ down_filter_global,
    const float *__restrict__ alpha, const float *__restrict__ inv_beta,
    const float *__restrict__ alpha_over_beta, int batch_size, int channels,
    int seq_len) {
  extern __shared__ float shared[];

  float *up_filter = shared;
  float *down_filter = shared + FILTER_SIZE;

  const int tid = threadIdx.x;

  if (tid < FILTER_SIZE) {
    up_filter[tid] = up_filter_global[tid];
    down_filter[tid] = down_filter_global[tid];
  }

  __syncthreads();

  const int channel = blockIdx.y;
  const int batch = blockIdx.z;
  const int output_index = blockIdx.x * blockDim.x + tid;

  if (output_index >= seq_len) {
    return;
  }

  const int64_t base =
      static_cast<int64_t>(seq_len) *
      (static_cast<int64_t>(channel) +
       static_cast<int64_t>(channels) * static_cast<int64_t>(batch));

  const float go = read_as_float(grad_output, base + output_index);
  const float alpha_value = alpha[channel];
  const float inv_beta_value = inv_beta[channel];
  const float alpha_over_beta_value = alpha_over_beta[channel];

#pragma unroll
  for (int down_tap = 0; down_tap < FILTER_SIZE; ++down_tap) {
    const int unclamped_q =
        RATIO * output_index + down_tap - DOWNSAMPLE_PAD_LEFT;
    const int q = clamp_int(unclamped_q, 0, 2 * seq_len - 1);

    const bool upsample_interior = q >= 6 && q <= 2 * seq_len - 7;

    const float v = upsample_value_polyphase(input, base, seq_len, q, up_filter,
                                             upsample_interior);

    const float phase = alpha_value * v;
    const float s = __sinf(phase);
    const float c = __cosf(phase);
    const float sin_2_phase = 2.0f * s * c;

    const float gz = go * down_filter[down_tap];

    atomicAdd(&grad_alpha[channel], gz * inv_beta_value * v * sin_2_phase);
    atomicAdd(&grad_inv_beta[channel], gz * s * s);

    const float gv = gz * (1.0f + alpha_over_beta_value * sin_2_phase);

    const int parity = (q + UPSAMPLE_CROP_LEFT) & 1;

#pragma unroll
    for (int i = 0; i < FILTER_SIZE / RATIO; ++i) {
      const int up_tap = parity + RATIO * i;
      int input_index =
          (q + UPSAMPLE_CROP_LEFT - up_tap) / RATIO - UPSAMPLE_INPUT_PAD;

      input_index = clamp_int(input_index, 0, seq_len - 1);

      atomicAdd(&grad_input[base + input_index], gv * 2.0f * up_filter[up_tap]);
    }
  }
}

int select_threads(int seq_len) {
  if (seq_len <= 32) {
    return 32;
  }
  if (seq_len <= 64) {
    return 64;
  }
  if (seq_len <= 128) {
    return 128;
  }
  return 256;
}

void check_cuda_contiguous_current_device(const torch::Tensor &tensor,
                                          const char *name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
  TORCH_CHECK(tensor.get_device() == c10::cuda::current_device(), name,
              " is on CUDA device ", tensor.get_device(),
              ", but the current CUDA device is ", c10::cuda::current_device(),
              ".");
}

void check_forward_contract(const torch::Tensor &input,
                            const torch::Tensor &up_filter,
                            const torch::Tensor &down_filter,
                            const torch::Tensor &alpha,
                            const torch::Tensor &inv_beta) {
  check_cuda_contiguous_current_device(input, "input");
  check_cuda_contiguous_current_device(up_filter, "up_filter");
  check_cuda_contiguous_current_device(down_filter, "down_filter");
  check_cuda_contiguous_current_device(alpha, "alpha");
  check_cuda_contiguous_current_device(inv_beta, "inv_beta");

  TORCH_CHECK(input.dim() == 3, "input must have shape [B, C, T].");
  TORCH_CHECK(input.size(0) > 0, "input batch dimension must be > 0.");
  TORCH_CHECK(input.size(1) > 0, "input channel dimension must be > 0.");
  TORCH_CHECK(input.size(2) > 0, "input time dimension must be > 0.");

  TORCH_CHECK(input.scalar_type() == at::ScalarType::Float ||
                  input.scalar_type() == at::ScalarType::Half ||
                  input.scalar_type() == at::ScalarType::BFloat16,
              "input dtype must be float32, float16, or bfloat16.");

  TORCH_CHECK(up_filter.scalar_type() == at::ScalarType::Float,
              "up_filter must be float32.");
  TORCH_CHECK(down_filter.scalar_type() == at::ScalarType::Float,
              "down_filter must be float32.");
  TORCH_CHECK(alpha.scalar_type() == at::ScalarType::Float,
              "alpha must be float32.");
  TORCH_CHECK(inv_beta.scalar_type() == at::ScalarType::Float,
              "inv_beta must be float32.");

  TORCH_CHECK(up_filter.numel() == FILTER_SIZE,
              "up_filter must have 12 elements.");
  TORCH_CHECK(down_filter.numel() == FILTER_SIZE,
              "down_filter must have 12 elements.");

  TORCH_CHECK(alpha.dim() == 1, "alpha must be 1D.");
  TORCH_CHECK(inv_beta.dim() == 1, "inv_beta must be 1D.");
  TORCH_CHECK(alpha.numel() == input.size(1),
              "alpha length must match input channels.");
  TORCH_CHECK(inv_beta.numel() == input.size(1),
              "inv_beta length must match input channels.");
}

void check_backward_contract(const torch::Tensor &grad_output,
                             const torch::Tensor &input,
                             const torch::Tensor &up_filter,
                             const torch::Tensor &down_filter,
                             const torch::Tensor &alpha,
                             const torch::Tensor &inv_beta,
                             const torch::Tensor &alpha_over_beta) {
  check_forward_contract(input, up_filter, down_filter, alpha, inv_beta);

  check_cuda_contiguous_current_device(grad_output, "grad_output");
  check_cuda_contiguous_current_device(alpha_over_beta, "alpha_over_beta");

  TORCH_CHECK(grad_output.dim() == 3, "grad_output must have shape [B, C, T].");
  TORCH_CHECK(grad_output.size(0) == input.size(0) &&
                  grad_output.size(1) == input.size(1) &&
                  grad_output.size(2) == input.size(2),
              "grad_output shape must match input shape.");

  TORCH_CHECK(grad_output.scalar_type() == input.scalar_type(),
              "grad_output dtype must match input dtype.");

  TORCH_CHECK(alpha_over_beta.scalar_type() == at::ScalarType::Float,
              "alpha_over_beta must be float32.");
  TORCH_CHECK(alpha_over_beta.dim() == 1, "alpha_over_beta must be 1D.");
  TORCH_CHECK(alpha_over_beta.numel() == input.size(1),
              "alpha_over_beta length must match input channels.");
}

} // namespace

torch::Tensor fwd_cuda(torch::Tensor const &input,
                       torch::Tensor const &up_filter,
                       torch::Tensor const &down_filter,
                       torch::Tensor const &alpha,
                       torch::Tensor const &inv_beta) {
  check_forward_contract(input, up_filter, down_filter, alpha, inv_beta);

  const int batches = static_cast<int>(input.size(0));
  const int channels = static_cast<int>(input.size(1));
  const int seq_len = static_cast<int>(input.size(2));

  auto output = torch::empty(input.sizes(), input.options());

  const int threads = select_threads(seq_len);
  const int blocks_per_sequence = (seq_len + threads - 1) / threads;

  const dim3 blocks(blocks_per_sequence, channels, batches);
  const dim3 thread_block(threads);

  const size_t shared_bytes =
      static_cast<size_t>(2 * FILTER_SIZE + 2 * threads + FILTER_SIZE - RATIO) *
      sizeof(float);

  DISPATCH_FLOAT_HALF_AND_BFLOAT(
      input.scalar_type(), "alias_free_activation_forward",
      alias_free_activation_forward_kernel<scalar_t>
      <<<blocks, thread_block, shared_bytes,
         at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<scalar_t *>(output.data_ptr()),
          reinterpret_cast<const scalar_t *>(input.data_ptr()),
          reinterpret_cast<const float *>(up_filter.data_ptr()),
          reinterpret_cast<const float *>(down_filter.data_ptr()),
          reinterpret_cast<const float *>(alpha.data_ptr()),
          reinterpret_cast<const float *>(inv_beta.data_ptr()), batches,
          channels, seq_len););

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}

std::vector<torch::Tensor>
bwd_cuda(torch::Tensor const &grad_output, torch::Tensor const &input,
         torch::Tensor const &up_filter, torch::Tensor const &down_filter,
         torch::Tensor const &alpha, torch::Tensor const &inv_beta,
         torch::Tensor const &alpha_over_beta) {
  check_backward_contract(grad_output, input, up_filter, down_filter, alpha,
                          inv_beta, alpha_over_beta);

  const int batches = static_cast<int>(input.size(0));
  const int channels = static_cast<int>(input.size(1));
  const int seq_len = static_cast<int>(input.size(2));

  auto float_options = input.options().dtype(at::kFloat);

  auto grad_input = torch::zeros(input.sizes(), float_options);
  auto grad_alpha = torch::zeros({channels}, alpha.options());
  auto grad_inv_beta = torch::zeros({channels}, inv_beta.options());

  const int threads = select_threads(seq_len);
  const int blocks_per_sequence = (seq_len + threads - 1) / threads;

  const dim3 blocks(blocks_per_sequence, channels, batches);
  const dim3 thread_block(threads);

  const size_t shared_bytes =
      static_cast<size_t>(2 * FILTER_SIZE) * sizeof(float);

  DISPATCH_FLOAT_HALF_AND_BFLOAT(
      input.scalar_type(), "alias_free_activation_backward",
      alias_free_activation_backward_kernel<scalar_t>
      <<<blocks, thread_block, shared_bytes,
         at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<float *>(grad_input.data_ptr()),
          reinterpret_cast<float *>(grad_alpha.data_ptr()),
          reinterpret_cast<float *>(grad_inv_beta.data_ptr()),
          reinterpret_cast<const scalar_t *>(grad_output.data_ptr()),
          reinterpret_cast<const scalar_t *>(input.data_ptr()),
          reinterpret_cast<const float *>(up_filter.data_ptr()),
          reinterpret_cast<const float *>(down_filter.data_ptr()),
          reinterpret_cast<const float *>(alpha.data_ptr()),
          reinterpret_cast<const float *>(inv_beta.data_ptr()),
          reinterpret_cast<const float *>(alpha_over_beta.data_ptr()), batches,
          channels, seq_len););

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {grad_input, grad_alpha, grad_inv_beta};
}
