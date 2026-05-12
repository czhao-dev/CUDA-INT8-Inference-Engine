#include "cpu_reference.hpp"

#include <cassert>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    nn::NetworkDims dims;
    dims.batch = 4;
    dims.input_dim = 16;
    dims.hidden_dim = 8;
    dims.output_dim = 5;

    const nn::Model model = nn::make_synthetic_model(dims);
    const std::vector<float> input = nn::make_synthetic_input(dims);
    const nn::InferenceResult result = nn::run_cpu_reference(model, input);

    assert(result.logits.size() == static_cast<std::size_t>(dims.batch * dims.output_dim));
    assert(result.probabilities.size() == result.logits.size());
    assert(result.predictions.size() == static_cast<std::size_t>(dims.batch));

    for (int row = 0; row < dims.batch; ++row) {
        float sum = 0.0f;
        for (int col = 0; col < dims.output_dim; ++col) {
            const float value = result.probabilities[static_cast<std::size_t>(row) * dims.output_dim + col];
            assert(value >= 0.0f);
            assert(value <= 1.0f);
            sum += value;
        }
        assert(std::abs(sum - 1.0f) < 1.0e-5f);
        assert(result.predictions[row] >= 0);
        assert(result.predictions[row] < dims.output_dim);
    }

    const std::string model_path = "/tmp/cuda_nn_reference_model.bin";
    const std::string input_path = "/tmp/cuda_nn_reference_input.bin";
    assert(nn::write_model(model_path, model));
    assert(nn::write_tensor(input_path, input));

    nn::Model loaded_model;
    std::vector<float> loaded_input;
    assert(nn::read_model(model_path, &loaded_model));
    assert(nn::read_tensor(input_path, &loaded_input));
    assert(loaded_model.dims.batch == dims.batch);
    assert(loaded_model.dims.input_dim == dims.input_dim);
    assert(loaded_model.w1 == model.w1);
    assert(loaded_input == input);

    bool threw = false;
    try {
        std::vector<float> wrong_size_input = input;
        wrong_size_input.pop_back();
        (void)nn::run_cpu_reference(model, wrong_size_input);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    std::remove(model_path.c_str());
    std::remove(input_path.c_str());

    std::cout << "CPU reference test passed\n";
    return 0;
}
