#include "kernels.cuh"

namespace nn {
namespace {

__global__ void int8_matmul_naive_kernel(
    const std::int8_t* a,
    const std::int8_t* b,
    const int* bias,
    int* c,
    int m,
    int n,
    int k,
    int a_zero_point,
    int b_zero_point) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) {
        return;
    }

    int acc = bias == nullptr ? 0 : bias[col];
    for (int inner = 0; inner < k; ++inner) {
        const int a_value = static_cast<int>(a[row * k + inner]) - a_zero_point;
        const int b_value = static_cast<int>(b[inner * n + col]) - b_zero_point;
        acc += a_value * b_value;
    }
    c[row * n + col] = acc;
}

__global__ void int8_matmul_tiled_kernel(
    const std::int8_t* a,
    const std::int8_t* b,
    const int* bias,
    int* c,
    int m,
    int n,
    int k,
    int a_zero_point,
    int b_zero_point) {
    __shared__ std::int8_t tile_a[kDefaultTile][kDefaultTile];
    __shared__ std::int8_t tile_b[kDefaultTile][kDefaultTile];

    const int row = blockIdx.y * kDefaultTile + threadIdx.y;
    const int col = blockIdx.x * kDefaultTile + threadIdx.x;
    int acc = 0;

    for (int tile = 0; tile < k; tile += kDefaultTile) {
        const int a_col = tile + threadIdx.x;
        const int b_row = tile + threadIdx.y;

        tile_a[threadIdx.y][threadIdx.x] =
            (row < m && a_col < k) ? a[row * k + a_col] : static_cast<std::int8_t>(a_zero_point);
        tile_b[threadIdx.y][threadIdx.x] =
            (b_row < k && col < n) ? b[b_row * n + col] : static_cast<std::int8_t>(b_zero_point);
        __syncthreads();

        #pragma unroll
        for (int inner = 0; inner < kDefaultTile; ++inner) {
            const int a_value = static_cast<int>(tile_a[threadIdx.y][inner]) - a_zero_point;
            const int b_value = static_cast<int>(tile_b[inner][threadIdx.x]) - b_zero_point;
            acc += a_value * b_value;
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = acc + (bias == nullptr ? 0 : bias[col]);
    }
}

}  // namespace

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
    cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }
    const dim3 block(16, 16);
    const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    int8_matmul_naive_kernel<<<grid, block, 0, stream>>>(a, b, bias, c, m, n, k, a_zero_point, b_zero_point);
}

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
    cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }
    const dim3 block(kDefaultTile, kDefaultTile);
    const dim3 grid((n + kDefaultTile - 1) / kDefaultTile, (m + kDefaultTile - 1) / kDefaultTile);
    int8_matmul_tiled_kernel<<<grid, block, 0, stream>>>(a, b, bias, c, m, n, k, a_zero_point, b_zero_point);
}

}  // namespace nn
