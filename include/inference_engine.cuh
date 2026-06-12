#pragma once

#include "cpu_reference.hpp"

namespace nn {

enum class GpuMatmulMode {
    Naive,
    Tiled,
};

struct GpuRunOptions {
    GpuMatmulMode mode = GpuMatmulMode::Tiled;
};

struct GpuRunResult {
    InferenceResult output;
};

GpuRunResult run_gpu_inference(
    const Model& model,
    const std::vector<float>& input,
    const GpuRunOptions& options);

const char* matmul_mode_name(GpuMatmulMode mode);

struct MatmulBenchmark {
    std::string label;
    int m = 0;
    int n = 0;
    int k = 0;
    double naive_ms = 0.0;
    double tiled_ms = 0.0;
};

std::vector<MatmulBenchmark> run_matmul_benchmarks(const NetworkDims& dims, int iterations);

}  // namespace nn
