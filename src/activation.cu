#include "kernels.cuh"

#include <cfloat>
#include <cstddef>

namespace nn {
namespace {

__global__ void relu_inplace_kernel(float* values, int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && values[idx] < 0.0f) {
        values[idx] = 0.0f;
    }
}

__global__ void softmax_stable_kernel(const float* logits, float* probabilities, int rows, int cols) {
    extern __shared__ float scratch[];
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    float local_max = -FLT_MAX;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, logits[row * cols + col]);
    }
    scratch[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float row_max = scratch[0];

    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const float value = expf(logits[row * cols + col] - row_max);
        probabilities[row * cols + col] = value;
        local_sum += value;
    }
    scratch[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_sum = 1.0f / scratch[0];

    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        probabilities[row * cols + col] *= inv_sum;
    }
}

__global__ void argmax_reduce_kernel(const float* probabilities, int* predictions, int rows, int cols) {
    extern __shared__ unsigned char scratch_bytes[];
    float* values = reinterpret_cast<float*>(scratch_bytes);
    int* indices = reinterpret_cast<int*>(values + blockDim.x);

    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    float best_value = -FLT_MAX;
    int best_index = 0;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const float candidate = probabilities[row * cols + col];
        if (candidate > best_value) {
            best_value = candidate;
            best_index = col;
        }
    }

    values[threadIdx.x] = best_value;
    indices[threadIdx.x] = best_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && values[threadIdx.x + stride] > values[threadIdx.x]) {
            values[threadIdx.x] = values[threadIdx.x + stride];
            indices[threadIdx.x] = indices[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        predictions[row] = indices[0];
    }
}

}  // namespace

void launch_relu_inplace(float* values, int count, cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    constexpr int block_size = 256;
    const int grid_size = (count + block_size - 1) / block_size;
    relu_inplace_kernel<<<grid_size, block_size, 0, stream>>>(values, count);
}

void launch_softmax_stable(const float* logits, float* probabilities, int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) {
        return;
    }
    constexpr int block_size = 256;
    const std::size_t shared_bytes = block_size * sizeof(float);
    softmax_stable_kernel<<<rows, block_size, shared_bytes, stream>>>(logits, probabilities, rows, cols);
}

void launch_argmax_reduce(const float* probabilities, int* predictions, int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) {
        return;
    }
    constexpr int block_size = 256;
    const std::size_t shared_bytes = block_size * (sizeof(float) + sizeof(int));
    argmax_reduce_kernel<<<rows, block_size, shared_bytes, stream>>>(probabilities, predictions, rows, cols);
}

}  // namespace nn
