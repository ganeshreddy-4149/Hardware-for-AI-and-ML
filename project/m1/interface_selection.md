# Interface Selection and Bandwidth Justification

## Chosen Interface

I choose **AXI4-Lite** as the primary interface for this project.

## Why This Interface

AXI4-Lite is a good fit for this single-layer INT8 3×3 Conv2D accelerator because the project involves a fixed, inference-only compute block that is triggered once per input sample. The host writes input activations, weights, and bias values into mapped registers, issues a start signal, and reads back the output feature map after the computation completes. This is a simple control-plane transaction pattern that AXI4-Lite is specifically designed for.

Compared to higher-complexity interfaces such as PCIe or UCIe, AXI4-Lite requires significantly less implementation effort and is directly supported in FPGA SoC designs using the ARM AMBA standard. It is also simpler to verify in simulation — a complete write transaction followed by a read response is straightforward to exercise in a testbench. For a single-kernel, single-sample inference accelerator at this project scope, AXI4-Lite is appropriate and sufficient.

The assumed host platform is an **FPGA SoC-style host** (e.g., Xilinx Zynq or similar ARM-based SoC), where AXI4-Lite is the standard control interface between the processing system and programmable logic.

## Target Operating Point

The hypothetical hardware accelerator design point used in the roofline plot targets:

- Compute throughput: 3.2 GFLOP/s
- Kernel arithmetic intensity: 12.1046 FLOP/byte

The Conv2D kernel requires 388,800 FLOPs per inference.

Equivalent sample throughput at the target operating point:
- samples/sec = 3.2e9 / 388,800 = **8,230.45 samples/sec**

## Bandwidth Requirement Calculation

The estimated total bytes transferred per inference (using the same no-reuse assumption as the arithmetic intensity calculation):

- Input bytes: 3,072
- Weight bytes: 216
- Bias bytes: 32
- Output bytes: 28,800
- **Total bytes per inference: 32,120**

Required interface bandwidth:
- Required bandwidth = samples/sec × bytes per inference
- = 8,230.45 × 32,120
- = **264,361,644 bytes/sec ≈ 0.2644 GB/s**

This matches the equivalent roofline relation:
- Required bandwidth = performance / arithmetic intensity = 3.2 / 12.1046 = **0.2644 GB/s**

## Rated Interface Bandwidth Comparison

For a 32-bit AXI4-Lite bus running at 100 MHz:

- Rated bandwidth = 32 bits × 100 MHz = 3.2 Gbit/s = **0.4000 GB/s**

Comparison:
- Required bandwidth: **0.2644 GB/s**
- AXI4-Lite rated bandwidth: **0.4000 GB/s**

Because 0.4000 GB/s is greater than 0.2644 GB/s, the chosen interface is **not the bottleneck** at the selected operating point.

## Bottleneck Statement

Under the current roofline assumptions, the design is not interface-bound at the chosen target throughput. The kernel remains compute-bound on the target roofline because its arithmetic intensity of 12.1046 FLOP/byte is higher than the ridge point of 4.47 FLOP/byte. The AXI4-Lite interface can sustain the required data rate with margin, making it an appropriate and non-limiting interface choice for this stage of the project.

## Host Platform

The assumed host platform is an **FPGA SoC** (e.g., Xilinx Zynq-7000 or Zynq UltraScale+) running a Linux-based software stack on the ARM processing system. The AXI4-Lite interface connects the ARM processing system to the programmable logic region where the INT8 Conv2D accelerator chiplet is implemented. This is a standard and well-documented integration pattern for FPGA-based hardware accelerators.
