#include "cpu_reference.hpp"
#include "inference_engine.cuh"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <string>

namespace {

void print_usage(const char* executable) {
    std::cerr
        << "Usage: " << executable << " [--batch N] [--input path] [--weights path] [--mode naive|tiled]\n"
        << "When files are omitted, deterministic synthetic data is used.\n";
}

bool parse_positive_int(const char* value, int* parsed) {
    char* end = nullptr;
    const long result = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || result <= 0 || result > std::numeric_limits<int>::max()) {
        return false;
    }
    *parsed = static_cast<int>(result);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    nn::NetworkDims dims;
    std::string input_path;
    std::string weights_path;
    nn::GpuRunOptions options;
    bool batch_set = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--batch" && i + 1 < argc) {
            if (!parse_positive_int(argv[++i], &dims.batch)) {
                std::cerr << "--batch must be a positive integer\n";
                return 2;
            }
            batch_set = true;
        } else if (arg == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (arg == "--weights" && i + 1 < argc) {
            weights_path = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            const std::string mode = argv[++i];
            if (mode == "naive") {
                options.mode = nn::GpuMatmulMode::Naive;
            } else if (mode == "tiled") {
                options.mode = nn::GpuMatmulMode::Tiled;
            } else {
                print_usage(argv[0]);
                return 2;
            }
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            print_usage(argv[0]);
            return 2;
        }
    }

    try {
        nn::Model model;
        if (!weights_path.empty()) {
            if (!nn::read_model(weights_path, &model)) {
                std::cerr << "Failed to read model: " << weights_path << '\n';
                return 1;
            }
            if (batch_set) {
                model.dims.batch = dims.batch;
            }
        } else {
            model = nn::make_synthetic_model(dims);
        }

        std::vector<float> input;
        if (!input_path.empty()) {
            if (!nn::read_tensor(input_path, &input)) {
                std::cerr << "Failed to read input tensor: " << input_path << '\n';
                return 1;
            }
            if (!batch_set) {
                if (input.size() % static_cast<std::size_t>(model.dims.input_dim) != 0) {
                    std::cerr << "Input tensor size is not divisible by model input dimension\n";
                    return 1;
                }
                model.dims.batch = static_cast<int>(input.size() / static_cast<std::size_t>(model.dims.input_dim));
            }
        } else {
            input = nn::make_synthetic_input(model.dims);
        }

        nn::InferenceResult cpu_result;
        cpu_result = nn::run_cpu_reference(model, input);
        const nn::GpuRunResult gpu_result = nn::run_gpu_inference(model, input, options);

        std::cout << "Mode: " << nn::matmul_mode_name(options.mode) << '\n'
                  << "Batch: " << model.dims.batch << '\n'
                  << "Max probability error: "
                  << nn::max_abs_error(cpu_result.probabilities, gpu_result.output.probabilities) << '\n'
                  << "Mean probability error: "
                  << nn::mean_abs_error(cpu_result.probabilities, gpu_result.output.probabilities) << '\n'
                  << "First prediction: CPU=" << cpu_result.predictions.front()
                  << " GPU=" << gpu_result.output.predictions.front() << '\n';
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }

    return 0;
}
