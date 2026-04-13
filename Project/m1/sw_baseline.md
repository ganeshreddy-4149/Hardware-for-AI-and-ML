# Software Baseline Benchmark

## Project
Single-layer INT8 3x3 Conv2D for CNN inference

## Baseline implementation
The software baseline is a Python reference implementation of a single-layer INT8 3x3 convolution. The implementation uses nested loops to perform the multiply-accumulate operations directly and stores the accumulated outputs in INT32 to avoid overflow during accumulation. This baseline is intended to provide a functionally correct comparison point for later hardware acceleration.

## Platform
- Environment: Google Colab CPU runtime
- CPU model: Intel(R) Xeon(R) CPU @ 2.20GHz
- OS: Linux 6.6.113+
- Python version: 3.12.13

## Configuration
- Batch size (N): 1
- Input channels (C_in): 3
- Input height (H): 32
- Input width (W): 32
- Output channels (C_out): 8
- Kernel size (K): 3 x 3
- Stride: 1
- Padding: 0
- Output height (H_out): 30
- Output width (W_out): 30
- Input dtype: INT8
- Weight dtype: INT8
- Bias dtype: INT32
- Output dtype: INT32

## Timing
- Number of runs: 10
- Median wall-clock time: 0.129214 s

## Throughput
- Samples/sec: 7.7391
- FLOPs per run: 388800
- GFLOP/s: 0.003009

## Memory usage
- Peak RSS during baseline execution: 180.94 MB

## Dominant kernel context
The software baseline is dominated by the `conv2d_int8_reference` function. Profiling shows that this kernel accounts for about 88.09% of the profiled runtime, which supports selecting Conv2D as the hardware acceleration target.

## Target hardware for roofline analysis
The target hardware platform for the roofline model is the Intel Core i5-10210U CPU.

Peak memory bandwidth:
- 45.8 GB/s (from Intel ARK)

Peak FP32 throughput is estimated theoretically as:
Peak FP32 GFLOP/s = cores x frequency x FLOP/cycle
= 4 x 1.60 x 32
= 204.8 GFLOP/s

Ridge point:
Ridge point = peak GFLOP/s / peak bandwidth
= 204.8 / 45.8
= 4.47 FLOP/byte

## Note on measurement vs. roofline platform
The software benchmark was executed on Google Colab CPU runtime for convenience, while the roofline target platform uses the Intel Core i5-10210U laptop CPU for architectural analysis and spec-sheet-based peak performance estimates.

## Interpretation
The software baseline is useful as a correctness reference, but its achieved performance is far below the theoretical roofline because it is implemented as a pure Python nested-loop model. This gap motivates moving the convolution kernel into dedicated hardware while retaining software for orchestration and verification.
