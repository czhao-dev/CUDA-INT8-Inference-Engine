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

## Testing & Benchmarks

### CPU Reference Test

Runs without a GPU and checks the binary I/O round trip plus the math invariants of the
forward pass (probabilities sum to 1, predictions are in range, mismatched shapes throw):

```bash
make cpu-test
./build/cpu_reference_test
```

This is also run automatically on every push via [GitHub Actions](.github/workflows/cpu-test.yml).

### GPU Kernel Test

Requires `nvcc` and a CUDA-capable GPU. Checks each kernel in isolation:

- `int8_matmul_naive` and `int8_matmul_tiled` produce identical results, including on
  matrix dimensions that aren't multiples of the 16x16 tile size.
- `quantize_fp32_to_int8` → `dequantize_int32_to_fp32` round-trips within one quantization
  step.
- `relu_inplace` matches `max(x, 0)`.
- `softmax_stable` and `argmax_reduce` match a CPU reference implementation.

```bash
make gpu-test
./build/gpu_kernel_test
```

Or with CMake:

```bash
cmake -S . -B build/cmake -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build/cmake
./build/cmake/gpu_kernel_test
```

### Benchmarks

The `inference` binary can time `int8_matmul_naive` against `int8_matmul_tiled` for each
layer's dimensions using CUDA events:

```bash
./build/inference --benchmark --benchmark-iters 100
```

This prints the average kernel time and the tiled-vs-naive speedup for both layers, in
addition to the usual correctness comparison against the CPU reference.

#### Sample Results (NVIDIA T4, CUDA 12.9)

```text
$ ./build/inference --batch 32 --mode tiled --benchmark --benchmark-iters 100
Mode: tiled
Batch: 32
Max probability error: 0.00582121
Mean probability error: 0.00103727
First prediction: CPU=7 GPU=7

Matmul benchmark (avg of 100 iterations):
  layer1 (m=32, n=128, k=784): naive=0.0258029 ms tiled=0.0300013 ms speedup=0.86x
  layer2 (m=32, n=10, k=128):  naive=0.00519136 ms tiled=0.00620544 ms speedup=0.84x

$ ./build/inference --batch 4096 --mode tiled --benchmark --benchmark-iters 50
Matmul benchmark (avg of 50 iterations):
  layer1 (m=4096, n=128, k=784): naive=1.27209 ms tiled=1.24982 ms speedup=1.02x
  layer2 (m=4096, n=10, k=128):  naive=0.0344211 ms tiled=0.0284256 ms speedup=1.21x
```

At small batch sizes the naive kernel is actually slightly faster, since the weight
matrices fit in L2 cache and the tiled kernel's `__syncthreads()` overhead isn't repaid.
The tiled kernel pulls ahead at larger batch sizes once the working set outgrows the
cache. See [Kernel design notes](docs/kernel_design.md#naive-vs-tiled-performance-crossover)
for details.

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

The next natural extensions are per-channel weight quantization, Tensor Core paths using DP4A or WMMA-style APIs, and Nsight Compute profiling screenshots.

## Further Reading

- [CUDA C++ Programming Guide — NVIDIA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Integer Quantization for Deep Learning Inference](https://arxiv.org/abs/2004.09602)
- [Kernel design notes](docs/kernel_design.md)
