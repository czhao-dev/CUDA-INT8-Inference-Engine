#include "kernels.cuh"

#include <cmath>

namespace nn {
namespace {

__global__ void quantize_fp32_to_int8_kernel(
    const float* input,
    std::int8_t* output,
    int count,
    DeviceQuantParams params) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    const int quantized = static_cast<int>(roundf(input[idx] / params.scale)) + params.zero_point;
    const int clamped = quantized < -128 ? -128 : (quantized > 127 ? 127 : quantized);
    output[idx] = static_cast<std::int8_t>(clamped);
}

__global__ void dequantize_int32_to_fp32_kernel(
    const int* input,
    float* output,
    int count,
    float combined_scale) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    output[idx] = static_cast<float>(input[idx]) * combined_scale;
}

}  // namespace

void launch_quantize_fp32_to_int8(
    const float* input,
    std::int8_t* output,
    int count,
    DeviceQuantParams params,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    constexpr int block_size = 256;
    const int grid_size = (count + block_size - 1) / block_size;
    quantize_fp32_to_int8_kernel<<<grid_size, block_size, 0, stream>>>(input, output, count, params);
}

void launch_dequantize_int32_to_fp32(
    const int* input,
    float* output,
    int count,
    float combined_scale,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    constexpr int block_size = 256;
    const int grid_size = (count + block_size - 1) / block_size;
    dequantize_int32_to_fp32_kernel<<<grid_size, block_size, 0, stream>>>(input, output, count, combined_scale);
}

}  // namespace nn
