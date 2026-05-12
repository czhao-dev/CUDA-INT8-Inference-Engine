#include "cpu_reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace nn {
namespace {

constexpr char kModelMagic[] = {'C', 'N', 'N', 'I'};
constexpr std::uint32_t kModelVersion = 1;

void dense(
    const std::vector<float>& input,
    const std::vector<float>& weights,
    const std::vector<float>& bias,
    std::vector<float>* output,
    int batch,
    int in_dim,
    int out_dim) {
    output->assign(static_cast<std::size_t>(batch) * out_dim, 0.0f);
    for (int row = 0; row < batch; ++row) {
        for (int col = 0; col < out_dim; ++col) {
            float acc = bias[static_cast<std::size_t>(col)];
            for (int inner = 0; inner < in_dim; ++inner) {
                const auto input_idx = static_cast<std::size_t>(row) * in_dim + inner;
                const auto weight_idx = static_cast<std::size_t>(inner) * out_dim + col;
                acc += input[input_idx] * weights[weight_idx];
            }
            (*output)[static_cast<std::size_t>(row) * out_dim + col] = acc;
        }
    }
}

void relu(std::vector<float>* values) {
    for (float& value : *values) {
        value = std::max(value, 0.0f);
    }
}

std::vector<float> softmax(const std::vector<float>& logits, int batch, int classes) {
    std::vector<float> probabilities(logits.size());
    for (int row = 0; row < batch; ++row) {
        const auto row_offset = static_cast<std::size_t>(row) * classes;
        float max_value = -std::numeric_limits<float>::infinity();
        for (int col = 0; col < classes; ++col) {
            max_value = std::max(max_value, logits[row_offset + col]);
        }

        float sum = 0.0f;
        for (int col = 0; col < classes; ++col) {
            const float value = std::exp(logits[row_offset + col] - max_value);
            probabilities[row_offset + col] = value;
            sum += value;
        }

        const float inv_sum = 1.0f / sum;
        for (int col = 0; col < classes; ++col) {
            probabilities[row_offset + col] *= inv_sum;
        }
    }
    return probabilities;
}

std::vector<int> argmax(const std::vector<float>& probabilities, int batch, int classes) {
    std::vector<int> predictions(batch);
    for (int row = 0; row < batch; ++row) {
        const auto row_offset = static_cast<std::size_t>(row) * classes;
        int best_index = 0;
        float best_value = probabilities[row_offset];
        for (int col = 1; col < classes; ++col) {
            const float candidate = probabilities[row_offset + col];
            if (candidate > best_value) {
                best_value = candidate;
                best_index = col;
            }
        }
        predictions[row] = best_index;
    }
    return predictions;
}

bool write_blob(std::ofstream& out, const std::vector<float>& values) {
    const auto bytes = static_cast<std::streamsize>(values.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(values.data()), bytes);
    return out.good();
}

bool read_blob(std::ifstream& in, std::vector<float>* values, std::size_t count) {
    values->resize(count);
    const auto bytes = static_cast<std::streamsize>(count * sizeof(float));
    in.read(reinterpret_cast<char*>(values->data()), bytes);
    return in.good();
}

bool valid_dims(const NetworkDims& dims) {
    return dims.batch > 0 && dims.input_dim > 0 && dims.hidden_dim > 0 && dims.output_dim > 0;
}

bool model_tensors_match_dims(const Model& model) {
    const NetworkDims& dims = model.dims;
    return valid_dims(dims) &&
           model.w1.size() == static_cast<std::size_t>(dims.input_dim) * dims.hidden_dim &&
           model.b1.size() == static_cast<std::size_t>(dims.hidden_dim) &&
           model.w2.size() == static_cast<std::size_t>(dims.hidden_dim) * dims.output_dim &&
           model.b2.size() == static_cast<std::size_t>(dims.output_dim);
}

}  // namespace

Model make_synthetic_model(const NetworkDims& dims, std::uint32_t seed) {
    Model model;
    model.dims = dims;
    model.w1.resize(static_cast<std::size_t>(dims.input_dim) * dims.hidden_dim);
    model.b1.resize(dims.hidden_dim);
    model.w2.resize(static_cast<std::size_t>(dims.hidden_dim) * dims.output_dim);
    model.b2.resize(dims.output_dim);

    std::mt19937 rng(seed);
    std::normal_distribution<float> weights(0.0f, 0.08f);
    std::uniform_real_distribution<float> bias(-0.02f, 0.02f);

    for (float& value : model.w1) {
        value = weights(rng);
    }
    for (float& value : model.b1) {
        value = bias(rng);
    }
    for (float& value : model.w2) {
        value = weights(rng);
    }
    for (float& value : model.b2) {
        value = bias(rng);
    }

    return model;
}

std::vector<float> make_synthetic_input(const NetworkDims& dims, std::uint32_t seed) {
    std::vector<float> input(static_cast<std::size_t>(dims.batch) * dims.input_dim);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (float& value : input) {
        value = dist(rng);
    }
    return input;
}

InferenceResult run_cpu_reference(const Model& model, const std::vector<float>& input) {
    const NetworkDims& dims = model.dims;
    if (!model_tensors_match_dims(model)) {
        throw std::invalid_argument("model tensor sizes do not match model dimensions");
    }
    if (input.size() != static_cast<std::size_t>(dims.batch) * dims.input_dim) {
        throw std::invalid_argument("input size does not match model dimensions");
    }

    std::vector<float> hidden;
    dense(input, model.w1, model.b1, &hidden, dims.batch, dims.input_dim, dims.hidden_dim);
    relu(&hidden);

    InferenceResult result;
    dense(hidden, model.w2, model.b2, &result.logits, dims.batch, dims.hidden_dim, dims.output_dim);
    result.probabilities = softmax(result.logits, dims.batch, dims.output_dim);
    result.predictions = argmax(result.probabilities, dims.batch, dims.output_dim);
    return result;
}

float max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return std::numeric_limits<float>::infinity();
    }
    float error = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        error = std::max(error, std::abs(a[i] - b[i]));
    }
    return error;
}

float mean_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) {
        return std::numeric_limits<float>::infinity();
    }
    double total = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        total += std::abs(a[i] - b[i]);
    }
    return static_cast<float>(total / static_cast<double>(a.size()));
}

bool write_model(const std::string& path, const Model& model) {
    if (!model_tensors_match_dims(model)) {
        return false;
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }

    out.write(kModelMagic, sizeof(kModelMagic));
    out.write(reinterpret_cast<const char*>(&kModelVersion), sizeof(kModelVersion));
    out.write(reinterpret_cast<const char*>(&model.dims.batch), sizeof(model.dims.batch));
    out.write(reinterpret_cast<const char*>(&model.dims.input_dim), sizeof(model.dims.input_dim));
    out.write(reinterpret_cast<const char*>(&model.dims.hidden_dim), sizeof(model.dims.hidden_dim));
    out.write(reinterpret_cast<const char*>(&model.dims.output_dim), sizeof(model.dims.output_dim));

    return write_blob(out, model.w1) && write_blob(out, model.b1) &&
           write_blob(out, model.w2) && write_blob(out, model.b2);
}

bool read_model(const std::string& path, Model* model) {
    std::ifstream in(path, std::ios::binary);
    if (!in || model == nullptr) {
        return false;
    }

    char magic[sizeof(kModelMagic)] = {};
    std::uint32_t version = 0;
    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (std::memcmp(magic, kModelMagic, sizeof(kModelMagic)) != 0 || version != kModelVersion) {
        return false;
    }

    in.read(reinterpret_cast<char*>(&model->dims.batch), sizeof(model->dims.batch));
    in.read(reinterpret_cast<char*>(&model->dims.input_dim), sizeof(model->dims.input_dim));
    in.read(reinterpret_cast<char*>(&model->dims.hidden_dim), sizeof(model->dims.hidden_dim));
    in.read(reinterpret_cast<char*>(&model->dims.output_dim), sizeof(model->dims.output_dim));
    if (!in.good() || !valid_dims(model->dims)) {
        return false;
    }

    const NetworkDims& dims = model->dims;
    return read_blob(in, &model->w1, static_cast<std::size_t>(dims.input_dim) * dims.hidden_dim) &&
           read_blob(in, &model->b1, dims.hidden_dim) &&
           read_blob(in, &model->w2, static_cast<std::size_t>(dims.hidden_dim) * dims.output_dim) &&
           read_blob(in, &model->b2, dims.output_dim);
}

bool write_tensor(const std::string& path, const std::vector<float>& values) {
    if (values.empty()) {
        return false;
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    const std::uint64_t count = values.size();
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    return write_blob(out, values);
}

bool read_tensor(const std::string& path, std::vector<float>* values) {
    std::ifstream in(path, std::ios::binary);
    if (!in || values == nullptr) {
        return false;
    }
    std::uint64_t count = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!in.good() || count == 0 || count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return false;
    }
    return read_blob(in, values, static_cast<std::size_t>(count));
}

}  // namespace nn
