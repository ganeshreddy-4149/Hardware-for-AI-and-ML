# Hardware-for-AI-and-ML

Sai Ganesh Reddy Charian  
ECE 510 Spring 2026  

---

## Project Title: INT8 Conv2D Hardware Accelerator for CNN Inference

This project focuses on designing and analyzing a hardware accelerator for a single-layer INT8 3x3 Conv2D used in CNN inference. The work starts from a software reference model and uses profiling, arithmetic intensity calculation, and roofline analysis to identify the convolution kernel as the dominant computational bottleneck.

The goal is to improve execution efficiency by offloading the multiply-accumulate intensive Conv2D operation to dedicated hardware, while software continues to handle control, configuration, and output verification. The project studies performance tradeoffs involving computation, memory traffic, and interface bandwidth in order to make informed hardware/software co-design decisions.

The proposed accelerator is organized as a realistic chiplet-style design with a standard interface, on-chip memory, and a dedicated INT8 Conv2D compute engine. Overall, the project bridges AI workload analysis with practical VLSI concerns such as throughput, dataflow, interface selection, RTL implementation, and design verification.
