# GEMM Analysis — Naive vs Tiled CUDA Kernels
**ECE 410/510 Codefest 3 | Spring 2026**

## (a) Why the Naive Kernel is Memory-Bound

The naive GEMM kernel assigns one thread per output element. Each thread
loads a full row of A and column of B from DRAM with zero data reuse,
yielding an arithmetic intensity of 0.25 FLOP/byte — 108× below the T4
ridge point of 27.1 FLOP/byte. The kernel remains memory-bound because
insufficient compute work is done per byte loaded from DRAM.

## (b) How Tiling Reduces DRAM Traffic

Tiling loads a T×T tile of A and B into shared memory once, reusing
those elements T times before fetching the next tile. For T=8, DRAM
traffic drops 8×, reducing reads from 8.59 GB to 1.07 GB and raising
arithmetic intensity from 0.25 to 1.99 FLOP/byte — a 7.97× improvement.

## (c) Whether Tiled Achieved Expected Improvement

The tiled kernel achieved 398.5 GFLOP/s versus 317.9 GFLOP/s for naive,
a 1.25× speedup. This is smaller than the theoretical 8× predicts.
The bottleneck is warp count: T=8 produces 64 threads per block (2 warps),
too few to hide memory latency through warp switching. SM utilization
dropped from 85.1% to 55.2%, confirming latency as the limiting factor.
A larger tile (T=16) would provide more warps and greater improvement.
