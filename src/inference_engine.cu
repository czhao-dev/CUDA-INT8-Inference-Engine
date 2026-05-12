#include "inference_engine.cuh"

#include "kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace nn {
namespace {

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t status = (call);                                                      \
        if (status != cudaSuccess) {                                                      \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(status)); \
        }                                                                                 \
    } while (false)

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) {
        allocate(count);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    ~DeviceBuffer() {
        release();
    }

    void allocate(std::size_t count) {
        release();
        count_ = count;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
    }

    T* get() {
        return ptr_;
    }

    const T* get() const {
        return ptr_;
    }

    std::size_t count() const {
        return count_;
    }

private:
    void release() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            count_ = 0;
        }
    }

    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

float symmetric_scale(const std::vector<float>& values) {
    float max_abs = 0.0f;
    for (float value : values) {
        max_abs = std::max(max_abs, std::abs(value));
    }
    return max_abs == 0.0f ? 1.0f : max_abs / 127.0f;
}

void validate_model_and_input(const Model& model, const std::vector<float>& input) {
    const NetworkDims& dims = model.dims;
    if (dims.batch <= 0 || dims.input_dim <= 0 || dims.hidden_dim <= 0 || dims.output_dim <= 0) {
        throw std::invalid_argument("model dimensions must all be positive");
    }

    const auto expected_input = static_cast<std::size_t>(dims.batch) * dims.input_dim;
    const auto expected_w1 = static_cast<std::size_t>(dims.input_dim) * dims.hidden_dim;
    const auto expected_w2 = static_cast<std::size_t>(dims.hidden_dim) * dims.output_dim;

    if (input.size() != expected_input) {
        throw std::invalid_argument("input size does not match model dimensions");
    }
    if (model.w1.size() != expected_w1 || model.b1.size() != static_cast<std::size_t>(dims.hidden_dim) ||
        model.w2.size() != expected_w2 || model.b2.size() != static_cast<std::size_t>(dims.output_dim)) {
        throw std::invalid_argument("model tensor sizes do not match model dimensions");
    }
}

std::vector<float> compute_hidden_calibration(const Model& model, const std::vector<float>& input) {
    const NetworkDims& dims = model.dims;
    std::vector<float> hidden(static_cast<std::size_t>(dims.batch) * dims.hidden_dim, 0.0f);
    for (int row = 0; row < dims.batch; ++row) {
        for (int col = 0; col < dims.hidden_dim; ++col) {
            float acc = model.b1[static_cast<std::size_t>(col)];
            for (int inner = 0; inner < dims.input_dim; ++inner) {
                const auto input_idx = static_cast<std::size_t>(row) * dims.input_dim + inner;
                const auto weight_idx = static_cast<std::size_t>(inner) * dims.hidden_dim + col;
                acc += input[input_idx] * model.w1[weight_idx];
            }
            hidden[static_cast<std::size_t>(row) * dims.hidden_dim + col] = std::max(acc, 0.0f);
        }
    }
    return hidden;
}

std::vector<int> quantize_bias(const std::vector<float>& bias, float combined_scale) {
    std::vector<int> quantized(bias.size());
    for (std::size_t i = 0; i < bias.size(); ++i) {
        quantized[i] = static_cast<int>(std::lrint(bias[i] / combined_scale));
    }
    return quantized;
}

void launch_matmul(
    GpuMatmulMode mode,
    const std::int8_t* a,
    const std::int8_t* b,
    const int* bias,
    int* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
    if (mode == GpuMatmulMode::Naive) {
        launch_int8_matmul_naive(a, b, bias, c, m, n, k, 0, 0, stream);
    } else {
        launch_int8_matmul_tiled(a, b, bias, c, m, n, k, 0, 0, stream);
    }
}

void forward_once(
    GpuMatmulMode mode,
    const NetworkDims& dims,
    const DeviceBuffer<float>& input,
    const DeviceBuffer<std::int8_t>& w1_q,
    const DeviceBuffer<int>& b1_q,
    const DeviceBuffer<std::int8_t>& w2_q,
    const DeviceBuffer<int>& b2_q,
    DeviceQuantParams input_qparams,
    DeviceQuantParams hidden_qparams,
    float layer1_scale,
    float layer2_scale,
    DeviceBuffer<std::int8_t>* input_q,
    DeviceBuffer<std::int8_t>* hidden_q,
    DeviceBuffer<int>* hidden_acc,
    DeviceBuffer<int>* logits_acc,
    DeviceBuffer<float>* hidden,
    DeviceBuffer<float>* logits,
    DeviceBuffer<float>* probabilities,
    DeviceBuffer<int>* predictions,
    cudaStream_t stream) {
    const int input_count = dims.batch * dims.input_dim;
    const int hidden_count = dims.batch * dims.hidden_dim;
    const int output_count = dims.batch * dims.output_dim;

    launch_quantize_fp32_to_int8(input.get(), input_q->get(), input_count, input_qparams, stream);
    launch_matmul(
        mode,
        input_q->get(),
        w1_q.get(),
        b1_q.get(),
        hidden_acc->get(),
        dims.batch,
        dims.hidden_dim,
        dims.input_dim,
        stream);
    launch_dequantize_int32_to_fp32(hidden_acc->get(), hidden->get(), hidden_count, layer1_scale, stream);
    launch_relu_inplace(hidden->get(), hidden_count, stream);

    launch_quantize_fp32_to_int8(hidden->get(), hidden_q->get(), hidden_count, hidden_qparams, stream);
    launch_matmul(
        mode,
        hidden_q->get(),
        w2_q.get(),
        b2_q.get(),
        logits_acc->get(),
        dims.batch,
        dims.output_dim,
        dims.hidden_dim,
        stream);
    launch_dequantize_int32_to_fp32(logits_acc->get(), logits->get(), output_count, layer2_scale, stream);
    launch_softmax_stable(logits->get(), probabilities->get(), dims.batch, dims.output_dim, stream);
    launch_argmax_reduce(probabilities->get(), predictions->get(), dims.batch, dims.output_dim, stream);
}

}  // namespace

const char* matmul_mode_name(GpuMatmulMode mode) {
    return mode == GpuMatmulMode::Naive ? "naive" : "tiled";
}

GpuRunResult run_gpu_inference(
    const Model& model,
    const std::vector<float>& input,
    const GpuRunOptions& options) {
    validate_model_and_input(model, input);
    const NetworkDims& dims = model.dims;

    const int input_count = dims.batch * dims.input_dim;
    const int hidden_count = dims.batch * dims.hidden_dim;
    const int output_count = dims.batch * dims.output_dim;

    const float input_scale = symmetric_scale(input);
    const float w1_scale = symmetric_scale(model.w1);
    const float w2_scale = symmetric_scale(model.w2);
    const std::vector<float> hidden_calibration = compute_hidden_calibration(model, input);
    const float hidden_scale = symmetric_scale(hidden_calibration);
    const float layer1_scale = input_scale * w1_scale;
    const float layer2_scale = hidden_scale * w2_scale;

    const std::vector<int> b1_q_host = quantize_bias(model.b1, layer1_scale);
    const std::vector<int> b2_q_host = quantize_bias(model.b2, layer2_scale);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    DeviceBuffer<float> d_input(input_count);
    DeviceBuffer<float> d_w1(model.w1.size());
    DeviceBuffer<float> d_w2(model.w2.size());
    DeviceBuffer<std::int8_t> d_input_q(input_count);
    DeviceBuffer<std::int8_t> d_hidden_q(hidden_count);
    DeviceBuffer<std::int8_t> d_w1_q(model.w1.size());
    DeviceBuffer<std::int8_t> d_w2_q(model.w2.size());
    DeviceBuffer<int> d_b1_q(b1_q_host.size());
    DeviceBuffer<int> d_b2_q(b2_q_host.size());
    DeviceBuffer<int> d_hidden_acc(hidden_count);
    DeviceBuffer<int> d_logits_acc(output_count);
    DeviceBuffer<float> d_hidden(hidden_count);
    DeviceBuffer<float> d_logits(output_count);
    DeviceBuffer<float> d_probabilities(output_count);
    DeviceBuffer<int> d_predictions(dims.batch);

    CUDA_CHECK(cudaMemcpyAsync(d_input.get(), input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_w1.get(), model.w1.data(), model.w1.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_w2.get(), model.w2.data(), model.w2.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_b1_q.get(), b1_q_host.data(), b1_q_host.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_b2_q.get(), b2_q_host.data(), b2_q_host.size() * sizeof(int), cudaMemcpyHostToDevice, stream));

    launch_quantize_fp32_to_int8(d_w1.get(), d_w1_q.get(), static_cast<int>(model.w1.size()), {w1_scale, 0}, stream);
    launch_quantize_fp32_to_int8(d_w2.get(), d_w2_q.get(), static_cast<int>(model.w2.size()), {w2_scale, 0}, stream);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    forward_once(
        options.mode,
        dims,
        d_input,
        d_w1_q,
        d_b1_q,
        d_w2_q,
        d_b2_q,
        {input_scale, 0},
        {hidden_scale, 0},
        layer1_scale,
        layer2_scale,
        &d_input_q,
        &d_hidden_q,
        &d_hidden_acc,
        &d_logits_acc,
        &d_hidden,
        &d_logits,
        &d_probabilities,
        &d_predictions,
        stream);
    CUDA_CHECK(cudaGetLastError());

    GpuRunResult result;
    result.output.logits.resize(output_count);
    result.output.probabilities.resize(output_count);
    result.output.predictions.resize(dims.batch);

    CUDA_CHECK(cudaMemcpyAsync(result.output.logits.data(), d_logits.get(), output_count * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(result.output.probabilities.data(), d_probabilities.get(), output_count * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(result.output.predictions.data(), d_predictions.get(), dims.batch * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaStreamDestroy(stream);
    return result;
}

}  // namespace nn
