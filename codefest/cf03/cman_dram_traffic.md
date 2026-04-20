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
No data is reused — every single read goes directly to DRAM.

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

This is the fundamental inefficiency of the naive kernel:
the same value is fetched from slow DRAM 32 separate times.

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

### Arithmetic Intensity

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

Instead of computing one output element at a time, tiling
groups threads to compute a T x T block of C together.
A T x T tile of A and a T x T tile of B are loaded from DRAM
into fast on-chip shared memory once. All threads reuse
those tiles T times for computation before fetching the next.

```
for tile_i in 0..(N/T):         <- 4 tile-row steps
    for tile_j in 0..(N/T):     <- 4 tile-col steps
        for tile_k in 0..(N/T): <- 4 accumulation steps
            load A_tile  <- T x T = 64 elements from DRAM
            load B_tile  <- T x T = 64 elements from DRAM
            __syncthreads()
            for k in 0..T:   <- compute using shared memory only
                sum += tile_A[ty][k] * tile_B[k][tx]
            __syncthreads()
```

### How many times is each element loaded from DRAM?

In the naive case, each element of B is loaded N = 32 times.
With tiling, each element belongs to exactly one tile.
That tile is loaded once per tile-column pass of C.
There are N/T = 4 tile-column passes.

```
Naive : each element loaded  N   = 32  times from DRAM
Tiled : each element loaded  N/T =  4  times from DRAM

Reuse improvement = N / (N/T) = T = 8
```

### Counting DRAM loads

```
Tile steps per matrix = (N/T)^3 = 4^3 = 64 total iterations

Each iteration loads:
  1 tile of A = T x T = 64 elements = 64 x 4 = 256 bytes
  1 tile of B = T x T = 64 elements = 64 x 4 = 256 bytes

Total A DRAM reads = 64 x 256 = 16,384 bytes
Total B DRAM reads = 64 x 256 = 16,384 bytes
```

### Tiled DRAM Traffic

```
Tiled DRAM Traffic = 2 x (N/T)^3 x T^2 x 4 bytes
                   = 2 x 64 x 64 x 4
                   = 32,768 bytes  =  32 KB
```

Verification using per-element formula:

```
A reads = N^2 x (N/T) x 4 = 1,024 x 4 x 4 = 16,384 bytes
B reads = N^2 x (N/T) x 4 = 1,024 x 4 x 4 = 16,384 bytes
Total   = 32,768 bytes  =  32 KB   (both methods agree)
```

### Arithmetic Intensity

```
Total FLOPs = 65,536  (same computation, less data movement)

AI (tiled) = 65,536 / 32,768
           = 2.0 FLOP/byte
```

---

## (c) Traffic Ratio and Explanation

### Computing the Ratio

```
Ratio = Naive DRAM Traffic / Tiled DRAM Traffic
      = 262,144 / 32,768
      = 8  (= T)
```

The measured traffic ratio is T = 8.

The assignment asks why the ratio equals N/T.
This is understood through the per-element load count:

```
Naive loads per element  =  N   = 32
Tiled loads per element  =  N/T =  4

Per-element reduction    =  N / (N/T) = T = 8

Expressed as N/T = 4: this is the number of times
each element is STILL loaded in the tiled model.
The reduction from N to N/T is a factor of T.
```

### Summary of Traffic Values

```
Naive DRAM Traffic = 262,144 bytes  (each element loaded N = 32 times)
Tiled DRAM Traffic =  32,768 bytes  (each element loaded N/T = 4 times)
Traffic ratio      =  262,144 / 32,768 = 8 = T
```

### One-Sentence Explanation

> The ratio equals N/T = 4 because tiling loads each T x T tile into
> shared memory once and reuses it T times across the inner loop,
> reducing per-element DRAM fetches from N = 32 (naive) down to
> N/T = 4 (tiled), a factor-of-T improvement in memory traffic.

---

## (d) Execution Times and Bound Classification

### Ridge Point

The ridge point separates memory-bound from compute-bound.

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

t_memory >> t_compute  ->  bottleneck is MEMORY

Execution time  ~  0.819 us
```

The naive kernel is severely memory-bound. The arithmetic
intensity of 0.25 FLOP/byte is 125x below the ridge point,
meaning the GPU compute units are mostly starved for data.

### Tiled Case

```
DRAM Traffic         = 32,768 bytes  (8x less than naive)
Arithmetic Intensity = 65,536 / 32,768 = 2.0 FLOP/byte

2.0  <  31.25  ->  still MEMORY-BOUND, but 8x closer to ridge

t_memory  = 32,768 / (320 x 10^9)  =  0.102 microseconds
t_compute = 65,536 / (10 x 10^12)  =  0.007 microseconds

t_memory > t_compute  ->  bottleneck is MEMORY

Execution time  ~  0.102 us
```

The tiled kernel remains memory-bound for N=32 because
AI = 2.0 FLOP/byte is still well below the ridge point.
However, tiling moves the kernel 8x closer to the ridge,
reducing execution time by 8x compared to naive.

### Summary Table

| Case  | DRAM Traffic  | AI (FLOP/byte) | t_memory  | t_compute | Bottleneck | Exec. Time |
|-------|---------------|----------------|-----------|-----------|------------|------------|
| Naive | 262,144 bytes | 0.25           | 0.819 us  | 0.007 us  | Memory     | ~0.819 us  |
| Tiled |  32,768 bytes | 2.0            | 0.102 us  | 0.007 us  | Memory     | ~0.102 us  |
| Ratio | 8x  (= T)     | 8x better      | 8x faster | unchanged | both mem   | 8x speedup |

The naive kernel is clearly memory-bound with AI far below
the ridge point. The tiled kernel is also memory-bound but
with 8x less DRAM traffic and 8x faster execution time.
For larger N, tiling would push AI further toward compute-bound.
