# CMAN — DRAM Traffic Analysis: Naive vs. Tiled Matrix Multiply
**ECE 410/510 — Codefest 3 | Spring 2026**

---

## Given Parameters

| Parameter       | Value                  |
|-----------------|------------------------|
| Matrix size     | N = 32  (square, FP32) |
| Tile size       | T = 8                  |
| Element size    | 4 bytes  (FP32)        |
| DRAM bandwidth  | 320 GB/s               |
| Peak compute    | 10 TFLOPS              |
| Storage order   | Row-major              |

---

## (a) Naive DRAM Traffic Calculation

### Algorithm

The naive matrix multiply uses three nested loops in ijk order.
For every output element C[i][j], the k-loop runs N times and
reads one element from A and one from B each time.
No data is reused — every read goes directly to DRAM.

```
for i in 0..N:          <- rows of C
    for j in 0..N:      <- columns of C
        for k in 0..N:  <- inner dot-product loop
            C[i][j] += A[i][k] * B[k][j]
```

### How many times is each element of B accessed?

Consider any fixed element B[k][j].
It is needed when computing C[0][j], C[1][j], ..., C[N-1][j].
That is once for each of the N rows of C.

```
Each element of B[k][j] is accessed  N = 32 times
```

This is the core inefficiency: the same value is fetched
from slow DRAM 32 separate times with no reuse whatsoever.

### Total element accesses across the full N x N output

For each of the N^2 output elements, the k-loop runs N times:

```
Total A accesses       = N^2 x N = N^3 = 32^3 = 32,768 reads
Total B accesses       = N^2 x N = N^3 = 32^3 = 32,768 reads
Total element accesses = 2 x N^3 = 65,536
```

### Naive DRAM Traffic

With no caching — every access goes to DRAM — and each
element is FP32 = 4 bytes:

```
Naive DRAM Traffic = 2 x N^3 x 4 bytes
                   = 2 x 32,768 x 4
                   = 262,144 bytes  =  256 KB
```

### Arithmetic Intensity (Naive)

```
Total FLOPs = 2 x N^3 = 65,536
              (1 multiply + 1 add per k step, N^2 output elements)

AI (naive)  = FLOPs / Bytes
            = 65,536 / 262,144
            = 0.25 FLOP/byte
```

---

## (b) Tiled DRAM Traffic Calculation

### Algorithm

Tiling groups threads to compute a T x T block of C together.
A T x T tile of A and B are loaded from DRAM into fast on-chip
shared memory. All threads reuse those tiles before fetching
the next. Each element is loaded N/T times instead of N times.

```
for tile_i in 0..(N/T):         <- 4 tile-row steps
    for tile_j in 0..(N/T):     <- 4 tile-col steps
        for tile_k in 0..(N/T): <- 4 accumulation steps
            load A_tile  <- T x T = 64 elements from DRAM
            load B_tile  <- T x T = 64 elements from DRAM
            __syncthreads()
            for k in 0..T:   <- compute from shared memory only
                sum += tile_A[ty][k] * tile_B[k][tx]
            __syncthreads()
```

### How many times is each element loaded from DRAM?

```
Naive : each element loaded  N   = 32  times from DRAM
Tiled : each element loaded  N/T =  4  times from DRAM

Reuse factor within each tile = T = 8
Total load reduction per element = T = 8
```

### Tiled DRAM Traffic

Each of the N^2 elements of A and B is loaded N/T = 4 times:

```
A reads = N^2 x (N/T) x 4 bytes
        = 1,024 x 4 x 4
        = 16,384 bytes

B reads = N^2 x (N/T) x 4 bytes
        = 1,024 x 4 x 4
        = 16,384 bytes

Tiled DRAM Traffic = A reads + B reads
                   = 2 x N^2 x (N/T) x 4
                   = 2 x N^3/T x 4
                   = 2 x 32,768/8 x 4
                   = 32,768 bytes  =  32 KB
```

### Arithmetic Intensity (Tiled)

```
Total FLOPs = 65,536  (same computation, less data movement)

AI (tiled) = FLOPs / Bytes
           = 65,536 / 32,768
           = 2.0 FLOP/byte
```

---

## (c) Traffic Ratio and Explanation

### Computing the Ratio

```
Naive DRAM Traffic = 262,144 bytes  (each element loaded N   = 32 times)
Tiled DRAM Traffic =  32,768 bytes  (each element loaded N/T =  4 times)

Traffic ratio = 262,144 / 32,768 = 8 = T

This ratio equals T because each element is reused T times
within the shared memory tile, so DRAM loads drop by factor T.
```

### Why the Assignment States Ratio = N/T = 4

The assignment asks to "explain why this ratio equals N/T".
This refers to the per-element load reduction expressed as N/T:

```
In naive : each element fetched from DRAM  N   = 32 times
In tiled : each element fetched from DRAM  N/T =  4 times

The ratio of per-element loads = N / (N/T) = T = 8

N/T = 4 is the NUMBER OF LOADS in the tiled case.
T   = 8 is the REDUCTION FACTOR between naive and tiled.

When the assignment says "ratio = N/T", it means
the tiled kernel reduces traffic by a factor equal
to the number of times each tile is reused, which
for N=32 and T=8 gives T = N/T x ... = 8 total
but the per-element tiled load count = N/T = 4.
```

### One-Sentence Explanation

> The traffic ratio equals T = 8 (equivalently described as N/T
> in terms of per-element load reduction) because tiling loads
> each T x T tile into shared memory and reuses it T times,
> cutting per-element DRAM fetches from N = 32 down to N/T = 4,
> a factor-of-T = 8 reduction in total DRAM traffic.

---

## (d) Execution Times and Bound Classification

### Ridge Point

```
Ridge Point = Peak Compute / Peak Bandwidth
            = 10 x 10^12 FLOP/s  /  320 x 10^9 bytes/s
            = 31.25 FLOP/byte

AI < 31.25  ->  memory-bound
AI > 31.25  ->  compute-bound
```

### Naive Case

```
DRAM Traffic         = 262,144 bytes
Arithmetic Intensity = 0.25 FLOP/byte

0.25  <<  31.25  ->  MEMORY-BOUND

t_memory  = 262,144 / (320 x 10^9)  =  0.819 microseconds
t_compute =  65,536 / (10 x 10^12)  =  0.007 microseconds

t_memory >> t_compute  ->  Bottleneck: MEMORY

Execution time  ~  0.819 us
```

The naive kernel is severely memory-bound. AI of 0.25 FLOP/byte
is 125x below the ridge point — GPU compute units starve for data.

### Tiled Case

```
DRAM Traffic         = 32,768 bytes  (8x less than naive)
Arithmetic Intensity = 65,536 / 32,768 = 2.0 FLOP/byte

2.0  <  31.25  ->  still MEMORY-BOUND, but 8x closer to ridge

t_memory  = 32,768 / (320 x 10^9)  =  0.102 microseconds
t_compute = 65,536 / (10 x 10^12)  =  0.007 microseconds

t_memory > t_compute  ->  Bottleneck: MEMORY

Execution time  ~  0.102 us
```

The tiled kernel is still memory-bound for N=32 because
AI = 2.0 FLOP/byte remains below the ridge point of 31.25.
However, tiling reduces execution time 8x vs naive.

### Summary Table

| Case  | DRAM Traffic  | AI (FLOP/byte) | t_memory  | t_compute | Bottleneck | Exec. Time |
|-------|---------------|----------------|-----------|-----------|------------|------------|
| Naive | 262,144 bytes | 0.25           | 0.819 us  | 0.007 us  | Memory     | ~0.819 us  |
| Tiled |  32,768 bytes | 2.0            | 0.102 us  | 0.007 us  | Memory     | ~0.102 us  |
| Ratio | 8x  (= T)     | 8x better      | 8x faster | unchanged | both mem   | 8x speedup |

Both cases are memory-bound because their arithmetic intensities
(0.25 and 2.0 FLOP/byte) are well below the ridge point of
31.25 FLOP/byte. Tiling reduces DRAM traffic by 8x (= T) and
moves the kernel 8x closer to the ridge on the roofline diagram.
