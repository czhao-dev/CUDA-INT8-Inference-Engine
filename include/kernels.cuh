#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace nn {

constexpr int kDefaultTile = 16;

struct DeviceQuantParams {
    float scale;
    int zero_point;
};

void launch_quantize_fp32_to_int8(
    const float* input,
    std::int8_t* output,
    int count,
    DeviceQuantParams params,
    cudaStream_t stream);

void launch_dequantize_int32_to_fp32(
    const int* input,
    float* output,
    int count,
    float combined_scale,
    cudaStream_t stream);

void launch_int8_matmul_naive(
    const std::int8_t* a,
    const std::int8_t* b,
    const int* bias,
    int* c,
    int m,
    int n,
    int k,
    int a_zero_point,
    int b_zero_point,
    cudaStream_t stream);

void launch_int8_matmul_tiled(
    const std::int8_t* a,
    const std::int8_t* b,
    const int* bias,
    int* c,
    int m,
    int n,
    int k,
    int a_zero_point,
    int b_zero_point,
    cudaStream_t stream);

void launch_relu_inplace(float* values, int count, cudaStream_t stream);
void launch_softmax_stable(const float* logits, float* probabilities, int rows, int cols, cudaStream_t stream);
void launch_argmax_reduce(const float* probabilities, int* predictions, int rows, int cols, cudaStream_t stream);

}  // namespace nn
