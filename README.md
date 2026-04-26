# Hardware-for-AI-and-ML

Sai Ganesh Reddy Charian
ECE 510 Spring 2026

## Project Title: INT8 Conv2D Hardware Accelerator for CNN Inference

This project focuses on designing and analyzing a hardware accelerator for a single-layer INT8 3x3 Conv2D used in CNN inference. The work starts from a software reference model and uses profiling, arithmetic intensity calculation, and roofline analysis to identify the convolution kernel as the dominant computational bottleneck.

The goal is to improve execution efficiency by offloading the multiply-accumulate intensive Conv2D operation to dedicated hardware, while software continues to handle control, configuration, and output verification. The project studies performance tradeoffs involving computation, memory traffic, and interface bandwidth in order to make informed hardware/software co-design decisions.

The proposed accelerator is organized as a realistic chiplet-style design with a standard interface, on-chip memory, and a dedicated INT8 Conv2D compute engine. Overall, the project bridges AI workload analysis with practical VLSI concerns such as throughput, dataflow, interface selection, RTL implementation, and design verification.

---

## Project HDL Compute Core

### Module: conv2d_top

The `conv2d_top` module implements a single-layer INT8 3x3 Conv2D co-processor chiplet in SystemVerilog. It contains a 9-parallel MAC array with INT32 accumulator, a pipelined requantizer, on-chip SRAM buffers, and an FSM controller (IDLE → LOAD → COMPUTE → OUTPUT → DONE). The module exposes an **AXI4-Lite** interface (32-bit, 100 MHz) to the host for control and data transfer, and uses **INT8** precision for inputs and weights, **INT32** for accumulation, and **INT8** for output after requantization.

The AXI4-Lite interface was selected based on the arithmetic intensity analysis from Milestone 1. The Conv2D kernel has AI = 12.10 FLOP/byte and the hardware target throughput is 3.20 GFLOP/s. The required interface bandwidth is therefore 3.20 ÷ 12.10 = 0.264 GB/s, which is below the AXI4-Lite rated bandwidth of 0.40 GB/s — providing 34% headroom. The interface is not the bottleneck. AXI4-Lite would only need upgrading to AXI4-Stream if the performance target exceeded 4.84 GFLOP/s (= 0.40 GB/s × 12.10 FLOP/byte).
