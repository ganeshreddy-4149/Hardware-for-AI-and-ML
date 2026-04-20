# GEMM Analysis — Naive vs Tiled CUDA Kernels
**ECE 410/510 Codefest 3 | Spring 2026**

## (a) Why the Naive Kernel is Memory-Bound

The naive GEMM kernel assigns one thread per output element. Each thread
loads a full row of A and column of B from DRAM with zero data reuse,
yielding an arithmetic intensity of 0.25 FLOP/byte, far below the T4
ridge point of 27.1 FLOP/byte. The kernel is memory-bound because
insufficient compute work is done per byte fetched from DRAM.

## (b) How Tiling Reduces DRAM Traffic

Tiling loads a T×T tile of A and B into shared memory once, reusing
those elements T times before fetching the next tile. For T=8, DRAM
traffic drops by a factor of eight, raising arithmetic intensity from
0.25 to 1.99 FLOP/byte. The tiled kernel remains memory-bound since
arithmetic intensity is still well below the ridge point of 27.1
FLOP/byte, meaning DRAM bandwidth is still the bottleneck.

## (c) Whether Tiled Achieved Expected Improvement

The tiled kernel achieved 398.5 GFLOP/s versus 317.9 GFLOP/s for naive,
a 1.25x speedup, smaller than the theoretical eight-times improvement
predicts. The bottleneck is warp count: T=8 gives only two warps per
block, too few to hide memory latency through warp switching. SM
utilization dropped from 85.1 percent to 55.2 percent, confirming
warp latency as the remaining bottleneck. Increasing tile size to T=16
would provide more warps and closer-to-theoretical improvement.
