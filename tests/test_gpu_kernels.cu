#include "kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define CUDA_CHECK(call)                                                                          \
    do {                                                                                          \
        cudaError_t status = (call);                                                              \
        if (status != cudaSuccess) {                                                              \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(status));  \
        }                                                                                         \
    } while (false)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
    }

    ~DeviceBuffer() {
        cudaFree(ptr_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() {
        return ptr_;
    }

    const T* get() const {
        return ptr_;
    }

    std::size_t count() const {
        return count_;
    }

    void copy_from_host(const std::vector<T>& host) {
        CUDA_CHECK(cudaMemcpy(ptr_, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copy_to_host(std::vector<T>* host) const {
        host->resize(count_);
        CUDA_CHECK(cudaMemcpy(host->data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
    }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("FAILED: " + message);
    }
}

void test_quantize_dequantize_roundtrip() {
    const std::vector<float> input = {-1.0f, -0.5f, 0.0f, 0.25f, 0.999f, 1.0f};
    const float scale = 1.0f / 127.0f;

    DeviceBuffer<float> d_input(input.size());
    DeviceBuffer<std::int8_t> d_quantized(input.size());
    DeviceBuffer<int> d_quantized_i32(input.size());
    DeviceBuffer<float> d_dequantized(input.size());

    d_input.copy_from_host(input);
    nn::launch_quantize_fp32_to_int8(d_input.get(), d_quantized.get(), static_cast<int>(input.size()), {scale, 0}, nullptr);

    std::vector<std::int8_t> quantized;
    d_quantized.copy_to_host(&quantized);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<int> widened(quantized.begin(), quantized.end());
    d_quantized_i32.copy_from_host(widened);
    nn::launch_dequantize_int32_to_fp32(d_quantized_i32.get(), d_dequantized.get(), static_cast<int>(input.size()), scale, nullptr);

    std::vector<float> dequantized;
    d_dequantized.copy_to_host(&dequantized);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (std::size_t i = 0; i < input.size(); ++i) {
        const float error = std::abs(input[i] - dequantized[i]);
        expect(error <= scale, "quantize/dequantize round trip error too large at index " + std::to_string(i));
    }
}

void test_matmul_naive_vs_tiled() {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> value_dist(-100, 100);
    std::uniform_int_distribution<int> bias_dist(-1000, 1000);

    // Dimensions deliberately not multiples of the 16x16 tile size to exercise boundary handling.
    const int m = 37;
    const int n = 23;
    const int k = 41;

    std::vector<std::int8_t> a(static_cast<std::size_t>(m) * k);
    std::vector<std::int8_t> b(static_cast<std::size_t>(k) * n);
    std::vector<int> bias(n);
    for (std::int8_t& value : a) {
        value = static_cast<std::int8_t>(value_dist(rng));
    }
    for (std::int8_t& value : b) {
        value = static_cast<std::int8_t>(value_dist(rng));
    }
    for (int& value : bias) {
        value = bias_dist(rng);
    }

    DeviceBuffer<std::int8_t> d_a(a.size());
    DeviceBuffer<std::int8_t> d_b(b.size());
    DeviceBuffer<int> d_bias(bias.size());
    DeviceBuffer<int> d_c_naive(static_cast<std::size_t>(m) * n);
    DeviceBuffer<int> d_c_tiled(static_cast<std::size_t>(m) * n);

    d_a.copy_from_host(a);
    d_b.copy_from_host(b);
    d_bias.copy_from_host(bias);

    nn::launch_int8_matmul_naive(d_a.get(), d_b.get(), d_bias.get(), d_c_naive.get(), m, n, k, 0, 0, nullptr);
    nn::launch_int8_matmul_tiled(d_a.get(), d_b.get(), d_bias.get(), d_c_tiled.get(), m, n, k, 0, 0, nullptr);

    std::vector<int> c_naive;
    std::vector<int> c_tiled;
    d_c_naive.copy_to_host(&c_naive);
    d_c_tiled.copy_to_host(&c_tiled);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (std::size_t i = 0; i < c_naive.size(); ++i) {
        expect(c_naive[i] == c_tiled[i], "naive/tiled matmul mismatch at index " + std::to_string(i));
    }
}

void test_relu_inplace() {
    const std::vector<float> values = {-2.0f, -0.001f, 0.0f, 0.5f, 3.0f};

    DeviceBuffer<float> d_values(values.size());
    d_values.copy_from_host(values);
    nn::launch_relu_inplace(d_values.get(), static_cast<int>(values.size()), nullptr);

    std::vector<float> result;
    d_values.copy_to_host(&result);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (std::size_t i = 0; i < values.size(); ++i) {
        const float expected = std::max(values[i], 0.0f);
        expect(result[i] == expected, "relu mismatch at index " + std::to_string(i));
    }
}

void test_softmax_and_argmax() {
    const int rows = 5;
    const int cols = 10;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::vector<float> logits(static_cast<std::size_t>(rows) * cols);
    for (float& value : logits) {
        value = dist(rng);
    }

    std::vector<float> expected_probs(logits.size());
    std::vector<int> expected_preds(rows);
    for (int row = 0; row < rows; ++row) {
        const std::size_t offset = static_cast<std::size_t>(row) * cols;
        float max_value = logits[offset];
        for (int col = 1; col < cols; ++col) {
            max_value = std::max(max_value, logits[offset + col]);
        }

        float sum = 0.0f;
        for (int col = 0; col < cols; ++col) {
            const float value = std::exp(logits[offset + col] - max_value);
            expected_probs[offset + col] = value;
            sum += value;
        }

        int best_index = 0;
        float best_value = -1.0f;
        for (int col = 0; col < cols; ++col) {
            expected_probs[offset + col] /= sum;
            if (expected_probs[offset + col] > best_value) {
                best_value = expected_probs[offset + col];
                best_index = col;
            }
        }
        expected_preds[row] = best_index;
    }

    DeviceBuffer<float> d_logits(logits.size());
    DeviceBuffer<float> d_probs(logits.size());
    DeviceBuffer<int> d_preds(rows);

    d_logits.copy_from_host(logits);
    nn::launch_softmax_stable(d_logits.get(), d_probs.get(), rows, cols, nullptr);
    nn::launch_argmax_reduce(d_probs.get(), d_preds.get(), rows, cols, nullptr);

    std::vector<float> probs;
    std::vector<int> preds;
    d_probs.copy_to_host(&probs);
    d_preds.copy_to_host(&preds);
    CUDA_CHECK(cudaDeviceSynchronize());

    constexpr float kTolerance = 1.0e-5f;
    for (std::size_t i = 0; i < probs.size(); ++i) {
        expect(std::abs(probs[i] - expected_probs[i]) < kTolerance, "softmax mismatch at index " + std::to_string(i));
    }
    for (int row = 0; row < rows; ++row) {
        expect(preds[row] == expected_preds[row], "argmax mismatch at row " + std::to_string(row));
    }
}

}  // namespace

int main() {
    try {
        test_quantize_dequantize_roundtrip();
        test_matmul_naive_vs_tiled();
        test_relu_inplace();
        test_softmax_and_argmax();
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }

    std::cout << "GPU kernel tests passed\n";
    return 0;
}
