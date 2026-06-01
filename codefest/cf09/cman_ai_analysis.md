# CMAN — Arithmetic Intensity Analysis
## ECE 510 CF09 | Spring 2026
Student: Sai Ganesh Reddy Charian

---

## Task 1 — Dominant Kernel

The dominant kernel in my accelerator is a 3x3 2D Convolution
operating on INT8 inputs and weights, with INT32 accumulation.
This kernel drives the entire compute workload of the accelerator.

Operating point matches the M1 software baseline:

- Input feature map:  H = 32, W = 32, C_in = 3
- Filter dimensions:  K = 3, K = 3, C_in = 3, C_out = 8
- Output feature map: H_out = 30, W_out = 30, C_out = 8
- Stride = 1, Padding = 0
- Hardware MAC array: 3x3 = 9 parallel MACs

Data types:
- Input activations : INT8  (8-bit signed integer)
- Weights           : INT8  (8-bit signed integer)
- Accumulator       : INT32 (32-bit signed integer)
- Output            : INT32 (before requantization)

Reuse pattern: Weight-stationary Conv2D reuse (GEMM-style).
Weights are reused across all spatial output positions.

---

## Task 2 — Total FLOPs Count

Each MAC operation counts as 2 FLOPs (one multiply, one add).

Formula:
    Total MACs  = H_out x W_out x C_out x K x K x C_in
    Total FLOPs = Total MACs x 2

Substituting values:
    Total MACs  = 30 x 30 x 8 x 3 x 3 x 3
                = 194,400 MACs

    Total FLOPs = 194,400 x 2
                = 388,800 FLOPs

This matches the M1 software baseline FLOPs count exactly,
confirming consistency across the full project.

---

## Task 3 — Bytes Transferred

### Lower Bound — No Data Reuse

Every MAC fetches its input activation and weight fresh
from off-chip memory. Nothing is cached or reused.

    Input bytes fetched  = Total MACs x 1 byte
                         = 194,400 x 1
                         = 194,400 bytes

    Weight bytes fetched = Total MACs x 1 byte
                         = 194,400 x 1
                         = 194,400 bytes

    Output bytes written = H_out x W_out x C_out x 4 bytes
                         = 30 x 30 x 8 x 4
                         = 28,800 bytes

    Total bytes (no reuse) = 194,400 + 194,400 + 28,800
                           = 417,600 bytes

### Upper Bound — Perfect Weight Reuse

Weights are loaded into on-chip memory once and stay there.
Only unique inputs and outputs travel off-chip.

    Weight bytes = C_out x C_in x K x K x 1 byte
                 = 8 x 3 x 3 x 3 x 1
                 = 216 bytes  (loaded once, reused across all positions)

    Input bytes  = N x C_in x H x W x 1 byte
                 = 1 x 3 x 32 x 32 x 1
                 = 3,072 bytes

    Bias bytes   = C_out x 4 bytes
                 = 8 x 4
                 = 32 bytes

    Output bytes = N x C_out x H_out x W_out x 4 bytes
                 = 1 x 8 x 30 x 30 x 4
                 = 28,800 bytes

    Total bytes (weight reuse) = 216 + 3,072 + 32 + 28,800
                               = 32,120 bytes

---

## Task 4 — Arithmetic Intensity and Roofline Sketch

Formula: AI = Total FLOPs / Total Bytes

### Lower Bound AI (no reuse):

    AI_lower = 388,800 / 417,600
             = 0.931 FLOP/byte

    The kernel is in the memory-bound region at this bound.
    Attainable performance = 0.40 x 0.931 = 0.372 GFLOP/s

### Upper Bound AI (perfect weight reuse):

    AI_upper = 388,800 / 32,120
             = 12.10 FLOP/byte

    This matches the CF02 arithmetic intensity exactly.
    The kernel is in the compute-bound region at this bound.
    Attainable performance = min(0.40 x 12.10, 1.447)
                           = min(4.84, 1.447)
                           = 1.447 GFLOP/s

### Platform Parameters — SKY130 PDK (M3 synthesis results):

    Peak compute = 9 MACs x 2 FLOP x 80.40 MHz
                 = 1.447 GFLOP/s

    Peak BW      = 32-bit AXI4-Lite x 100 MHz
                 = 0.40 GB/s

    Ridge point  = Peak compute / Peak BW
                 = 1.447 / 0.40
                 = 3.617 FLOP/byte

### Summary of AI Bounds:

    AI_lower = 0.931 FLOP/byte  (no reuse      — lower bound)
    AI_upper = 12.10 FLOP/byte  (weight reuse  — upper bound)
    Ridge    = 3.617 FLOP/byte  (SKY130 + AXI4-Lite)

Hand-drawn roofline sketch saved as:
codefest/cf09/cman_roofline_sketch.png

---

## Task 5 — Bottleneck Identification and Improvement

### Current Bottleneck:

The design is compute-bound at the upper AI bound.

Since AI_upper (12.10 FLOP/byte) is greater than the ridge
point (3.617 FLOP/byte), the kernel sits in the compute-bound
region of the roofline. The AXI4-Lite interface rated at
0.40 GB/s can sustain the required 0.264 GB/s with margin,
so the interface is not the bottleneck.

At the lower AI bound (0.931 FLOP/byte), the kernel falls
in the memory-bound region, meaning that without on-chip
weight reuse, memory bandwidth becomes the limiting factor.

The actual compute bottleneck is the achieved clock frequency.
M3 synthesis reached only 80.40 MHz against the 100 MHz target
because the critical path runs through nine 32x32-bit
sign-extended multipliers instead of correctly sized 8x8-bit
INT8 multipliers.

### Single Highest-Leverage Change:

Replace the 32x32-bit sign-extension multiplier pattern:

    assign p = {{24{a[7]}},a} * {{24{b[7]}},b};

with a correctly sized 8x8 signed multiply:

    assign p = $signed(a) * $signed(b);

This reduces the multiplier from 32x32 to 8x8, cutting the
AND/XOR gate count on the critical path by approximately 4x,
targeting timing closure at 100 MHz, and increasing projected
throughput from 1.447 GFLOP/s to 1.800 GFLOP/s
(9 MACs x 2 FLOP x 100 MHz).
