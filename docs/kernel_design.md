# Kernel Design Notes

This project implements a compact two-layer MLP inference path to show the mechanics used by larger inference engines: quantization, integer matrix multiplication, dequantization, activation, softmax, and prediction.

## Tensor Layout

Matrices use row-major layout:

- activations: `M x K`
- weights: `K x N`
- output accumulators: `M x N`

For the first layer, `M` is batch size, `K` is input dimension, and `N` is hidden dimension. For the second layer, `K` is hidden dimension and `N` is output classes.

## Quantization

The implementation uses symmetric per-tensor INT8 quantization:

```text
scale = max(abs(x)) / 127
q = clamp(round(x / scale), -128, 127)
```

Zero-point is fixed to `0`, so matrix multiplication can subtract the zero-point cheaply and the accumulator remains an ordinary signed `int`.

The matmul accumulator is dequantized with:

```text
fp32 = int32_accumulator * activation_scale * weight_scale
```

Bias is fused into the INT32 accumulator by pre-quantizing each bias term:

```text
bias_int32 = round(bias_fp32 / (activation_scale * weight_scale))
```

This avoids an extra post-matmul memory pass.

## Naive INT8 Matmul

`int8_matmul_naive` launches a `16 x 16` block. Each thread computes one output element:

```text
for inner in K:
    acc += int(a[row, inner]) * int(b[inner, col])
```

The kernel reads both operands directly from global memory for every multiply. This is simple and useful as a correctness and performance baseline, but it re-reads the same activation and weight values many times.

## Tiled INT8 Matmul

`int8_matmul_tiled` also assigns one output element to each thread, but loads reusable chunks of both matrices into shared memory:

```text
shared tile_a[16][16]
shared tile_b[16][16]
```

For each tile along `K`:

1. Threads cooperatively load one activation element and one weight element.
2. The block synchronizes.
3. Each thread performs `16` multiply-accumulate operations from shared memory.
4. The block synchronizes again before loading the next tile.

This improves arithmetic intensity because each global memory load can be reused by multiple threads in the block.

## Softmax

The softmax kernel assigns one row to one block. It uses two reductions:

1. Find the row maximum for numerical stability.
2. Compute the sum of exponentials after subtracting that maximum.

The final probability is:

```text
probability = exp(logit - row_max) / sum(exp(logit - row_max))
```

## Argmax

`argmax_reduce` uses a block-level tree reduction. Each thread scans a strided subset of the row, stores its best value and index in shared memory, then the block reduces to one predicted class.

## Known Tradeoffs

The current implementation intentionally favors readability over peak performance:

- quantization is per-tensor, not per-channel
- hidden activation scale is calibrated from the current input batch rather than a separate held-out calibration set
- the tiled kernel does not use DP4A or Tensor Cores
- softmax uses one block per row, which is suitable for small class counts

Those limitations make good future portfolio extensions because they are easy to explain and improve incrementally.

## Naive vs. Tiled Performance Crossover

Benchmarking on an NVIDIA T4 (see [README benchmarks](../README.md#benchmarks)) shows that
`int8_matmul_tiled` is not strictly faster than `int8_matmul_naive` for this model's default
dimensions:

- At `batch = 32` (layer1: 32x128x784, layer2: 32x10x128), the tiled kernel is ~15% *slower*
  than the naive kernel.
- At `batch = 4096`, the tiled kernel is ~2-21% *faster*.

The weight matrices here (≤400 KB) fit comfortably in the T4's L2 cache, so the naive
kernel's repeated global-memory reads are largely served from cache, and shared-memory
tiling adds `__syncthreads()` overhead without a corresponding bandwidth win. The tiling
advantage only materializes once the working set is large enough that L2 caching alone
can't cover the re-reads. This is a useful reminder that "tiled" is not synonymous with
"faster" — it depends on whether the kernel is memory-bound at the given problem size.
