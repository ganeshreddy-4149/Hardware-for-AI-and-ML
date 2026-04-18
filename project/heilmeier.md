# Heilmeier Questions (Q1–Q3)

## Q1. What are you trying to do?

I am designing and verifying a small hardware accelerator for a single-layer INT8 3×3 Conv2D used in CNN inference. The target kernel is the convolution operation itself, implemented as a fixed, inference-only compute block rather than a full multi-layer CNN. The goal is to accelerate the multiply-accumulate-heavy Conv2D kernel relative to a software baseline, while keeping the project scope small enough to remain realistic for synthesis, verification, and interface integration in this course.

---

## Q2. How is it done today and what are the limits?

Today, the target kernel runs as a software reference implementation in Python using nested loops on a general-purpose Intel Core i5 CPU. Profiling with `cProfile` shows that `conv2d_int8_reference` is the dominant kernel, accounting for about 88.09% of the total profiled runtime.

The measured results tell the story directly. The median execution time is 0.129214 s for one input sample, corresponding to about 7.7391 samples/sec and 0.003009 GFLOP/s. The arithmetic intensity of the kernel is 12.1046 FLOP/byte, which is well above the CPU ridge point of about 4.47 FLOP/byte — confirming the kernel is compute-bound. Despite this, the CPU achieves only 0.003009 GFLOP/s, a tiny fraction of its theoretical peak, because the implementation is a pure Python nested loop. The CPU is not able to exploit its SIMD or pipelining capabilities through the interpreter.

PyTorch's `torch.nn.functional.conv2d` dispatches to an optimized C++/MKL backend on CPU and achieves significantly higher throughput than the Python reference, but it still relies on general-purpose hardware rather than a dedicated fixed-function accelerator — meaning area and power are not focused on the Conv2D operation alone.

On the FPGA side, Ma et al. (2017) demonstrated a hardware Conv2D accelerator on an Altera Arria 10 FPGA achieving 645 GOPS throughput for VGG-16, using systematic loop tiling and dataflow optimization to minimize memory access [1]. While effective, FPGA implementations are tied to a specific device family's DSP blocks and BRAM topology and are not directly portable to an open ASIC RTL-to-GDSII flow such as OpenLane 2.

At the ASIC level, Google's Edge TPU delivers 4 TOPS at 2 W for INT8 inference [2], which is highly efficient but is a closed-source, proprietary design targeting full multi-layer networks. It cannot be studied, modified, or reproduced in an academic open-source synthesis flow.

The main limitation shared across all existing approaches is that none of them provide an open, synthesizable, single-kernel hardware block that can be independently characterized for area, power, and timing through a transparent RTL-to-GDSII flow. The pure Python baseline is dominated entirely by interpreter and loop overhead rather than actual compute throughput.

### References
- [1] Y. Ma, Y. Cao, S. Vrudhula, and J. Seo, "Optimizing Loop Operation and Dataflow in FPGA Acceleration of Deep Convolutional Neural Networks," in *Proc. ACM/SIGDA FPGA*, 2017, pp. 45–54. https://doi.org/10.1145/3020078.3021736
- [2] Google Coral, "Edge TPU Performance and Specifications," https://coral.ai/docs/edgetpu/faq/

---

## Q3. What is your approach, and why is it better?

My approach is to move the single-layer INT8 3×3 Conv2D kernel into a dedicated hardware accelerator while keeping host-side setup, control, and result checking in software. The accelerator will be described in SystemVerilog, synthesized using OpenLane 2, and connected to the host through an AXI4-Lite interface.

The arithmetic intensity of the kernel is 12.1046 FLOP/byte, and on the chosen CPU roofline this is above the ridge point of about 4.47 FLOP/byte. This makes Conv2D a meaningful hardware target because it has enough compute density to benefit from specialized execution — adding dedicated MAC units directly improves throughput rather than being blocked by memory bandwidth. The hypothetical accelerator design point targets much higher throughput than the current software reference while remaining within the assumed AXI4-Lite interface bandwidth budget.

This is better than the current software-only approach because it directly accelerates the dominant kernel instead of spending most of the runtime inside a slow Python loop structure. The hardware block executes one multiply-accumulate per clock cycle per MAC unit, eliminating interpreter overhead entirely and making full use of the compute-bound nature of the kernel. Compared to the FPGA and ASIC solutions cited above, this design is open, synthesizable through OpenLane 2, scoped to a single well-defined kernel, and fully verifiable in simulation with a transparent RTL implementation.
