# CUDA Neural Net Inference Engine

> A from-scratch CUDA C++ inference demo implementing INT8 quantization, tiled matrix multiplication, numerically stable softmax, and an end-to-end two-layer neural network forward pass.

This repository is designed as a portfolio project for showing practical GPU programming skills below the abstraction layer of PyTorch, TensorFlow, or TensorRT. It includes a CPU reference implementation, CUDA kernels, and a command-line inference runner.

## What It Builds

The demo runs a simple MLP:

```text
Input FP32
  -> quantize to INT8
  -> INT8 matmul + INT32 bias, layer 1
  -> dequantize to FP32
  -> ReLU
  -> quantize hidden activations to INT8
  -> INT8 matmul + INT32 bias, layer 2
  -> dequantize logits to FP32
  -> stable softmax
  -> argmax prediction
```

All intermediate tensors stay on the GPU during inference.

## Repository Layout

```text
.
├── CMakeLists.txt
├── Makefile
├── include/
│   ├── cpu_reference.hpp
│   ├── inference_engine.cuh
│   └── kernels.cuh
├── src/
│   ├── activation.cu
│   ├── cpu_reference.cpp
│   ├── inference_engine.cu
│   ├── main.cu
│   ├── matmul.cu
│   └── quantize.cu
├── docs/
│   └── kernel_design.md
├── models/
│   ├── sample_input.bin
│   └── weights.bin
├── scripts/
│   └── generate_sample_data.py
└── tests/
    └── test_cpu_reference.cpp
```

## Kernels Implemented

### `quantize_fp32_to_int8`

Maps FP32 tensors into signed INT8 using symmetric per-tensor scale factors. The current implementation uses zero-point `0`, which keeps the math simple and matches common symmetric weight quantization schemes.

### `int8_matmul_naive`

A baseline GPU matrix multiplication kernel. Each thread computes one output element directly from global memory. This is intentionally simple so the optimized kernel has a meaningful comparison point.

### `int8_matmul_tiled`

The main showcase kernel. It uses `16 x 16` thread blocks and shared-memory tiles for the activation and weight matrices. Inputs are INT8, accumulation is INT32, and quantized bias is fused into the output accumulator.

### `dequantize_int32_to_fp32`

Converts INT32 accumulators back to FP32 using the combined activation and weight scale.

### `relu_inplace`

Applies ReLU directly in device memory.

### `softmax_stable`

Computes softmax row-by-row using a block-level reduction for the row maximum and another reduction for the exponential sum.

### `argmax_reduce`

Reduces each probability row to a predicted class index.

## Build

### CPU Reference Test

The CPU reference test builds without CUDA and is useful on machines without an NVIDIA GPU:

```bash
make cpu-test
./build/cpu_reference_test
```

Or with CMake:

```bash
cmake -S . -B build/cmake
cmake --build build/cmake
./build/cmake/cpu_reference_test
```

### CUDA Executables

On a CUDA-capable machine with `nvcc`:

```bash
make inference
```

Or:

```bash
cmake -S . -B build/cmake -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build/cmake
```

Set `ARCH` for your GPU when using the Makefile:

```bash
make inference ARCH=sm_75
```

## Run

Generate sample binaries:

```bash
python3 scripts/generate_sample_data.py --output-dir models --batch 1
```

Run inference with synthetic data:

```bash
./build/inference --batch 32 --mode tiled
```

Run inference with generated files:

```bash
./build/inference --input models/sample_input.bin --weights models/weights.bin --mode tiled
```

## Binary Formats

`models/weights.bin`:

```text
char[4] magic = "CNNI"
uint32 version = 1
int32 batch
int32 input_dim
int32 hidden_dim
int32 output_dim
float32[input_dim * hidden_dim] w1
float32[hidden_dim] b1
float32[hidden_dim * output_dim] w2
float32[output_dim] b2
```

`models/sample_input.bin`:

```text
uint64 element_count
float32[element_count] values
```

## Why This Project Matters

This project demonstrates the core mechanics behind production inference systems:

- quantized INT8 data movement
- INT32 accumulator precision
- fused bias handling
- shared-memory tiling
- numerically stable output normalization
- CPU correctness checking against the GPU result

The next natural extensions are per-channel weight quantization, Tensor Core paths using DP4A or WMMA-style APIs, Nsight Compute profiling screenshots, and a small GitHub Actions workflow for the CPU reference test.

## Further Reading

- [CUDA C++ Programming Guide — NVIDIA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Integer Quantization for Deep Learning Inference](https://arxiv.org/abs/2004.09602)
- [Kernel design notes](docs/kernel_design.md)
