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

}  // namespace nn
