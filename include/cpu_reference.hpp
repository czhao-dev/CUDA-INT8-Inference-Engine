#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace nn {

struct NetworkDims {
    int batch = 1;
    int input_dim = 784;
    int hidden_dim = 128;
    int output_dim = 10;
};

struct QuantParams {
    float scale = 1.0f;
    int zero_point = 0;
};

struct Model {
    NetworkDims dims;
    std::vector<float> w1;
    std::vector<float> b1;
    std::vector<float> w2;
    std::vector<float> b2;
};

struct InferenceResult {
    std::vector<float> logits;
    std::vector<float> probabilities;
    std::vector<int> predictions;
};

Model make_synthetic_model(const NetworkDims& dims, std::uint32_t seed = 7);
std::vector<float> make_synthetic_input(const NetworkDims& dims, std::uint32_t seed = 11);
InferenceResult run_cpu_reference(const Model& model, const std::vector<float>& input);
float max_abs_error(const std::vector<float>& a, const std::vector<float>& b);
float mean_abs_error(const std::vector<float>& a, const std::vector<float>& b);

bool write_model(const std::string& path, const Model& model);
bool read_model(const std::string& path, Model* model);
bool write_tensor(const std::string& path, const std::vector<float>& values);
bool read_tensor(const std::string& path, std::vector<float>* values);

}  // namespace nn

