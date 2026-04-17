# Interface Selection and Bandwidth Justification

## Chosen interface
I choose AXI4-Stream as the primary interface assumption for this project.

## Why this interface
AXI4-Stream is a good fit for this Conv2D accelerator because the project targets inference-style data flow with a stream of input activations and output feature maps. Compared with very low-bandwidth interfaces such as SPI or I2C, AXI4-Stream is much more appropriate for continuous tensor movement. It is also simpler and more realistic for this project than PCIe or UCIe, which would add major implementation complexity that is not necessary for a small single-layer accelerator. The assumed host platform is an FPGA SoC-style host, where AXI-family interfaces are common and natural.

## Target operating point
The hypothetical hardware accelerator design point used in the roofline plot targets:
- Compute throughput: 3.2 GFLOP/s
- Kernel arithmetic intensity: 12.1046 FLOP/byte

The Conv2D kernel requires 388800 FLOPs per inference.

Equivalent sample throughput at the target operating point:
samples/sec = 3.2e9 / 388800
samples/sec = 8230.45

The estimated total bytes transferred per inference, using the same no-reuse assumption as the arithmetic-intensity calculation, are:
- Input bytes: 3072
- Weight bytes: 216
- Bias bytes: 32
- Output bytes: 28800
- Total bytes per inference: 32120

Required interface bandwidth:
required bandwidth = samples/sec x bytes per inference
= 8230.45 x 32120
= 264361111 bytes/sec
= 0.2644 GB/s

This matches the equivalent roofline relation:
required bandwidth = performance / arithmetic intensity
= 3.2 / 12.1046
= 0.2644 GB/s

## Rated interface bandwidth comparison
For a simple implementation assumption, consider a 32-bit AXI4-Stream link running at 100 MHz.

Rated bandwidth:
32 bits x 100 MHz = 3.2 Gbit/s
3.2 Gbit/s / 8 = 0.4000 GB/s

Comparison:
- Required bandwidth: 0.2644 GB/s
- Assumed AXI4-Stream bandwidth: 0.4000 GB/s

Because 0.4000 GB/s is greater than 0.2644 GB/s, the chosen interface is not the bottleneck at the selected operating point.

## Bottleneck statement
Under the current roofline assumptions, the design is not interface-bound at the chosen target throughput. The kernel remains compute-bound on the target roofline because its arithmetic intensity is higher than the ridge point. Therefore, AXI4-Stream is an appropriate interface for this stage of the project.
