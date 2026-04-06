# Heilmeier Draft

## Project Title
Hardware-Aware Optimization of CNN Workloads for Efficient Acceleration

## 1. What are you trying to do?
This project aims to analyze convolutional neural network (CNN) workloads and identify the layers that dominate computation and memory traffic. Using ResNet-18 as a representative model, the work focuses on profiling layer-wise MAC operations, parameter counts, memory movement, and arithmetic intensity. The goal is to use these results to propose hardware-aware optimization strategies that improve accelerator efficiency.

## 2. How is it done today, and what are the limits of current practice?
Today, CNN models are commonly executed on CPUs, GPUs, and existing AI accelerators using general-purpose software frameworks. While these platforms provide high performance, they do not always expose how individual layers behave from a hardware perspective. Many layers differ significantly in compute cost, data reuse, and memory bandwidth requirements. Without workload-aware profiling, it is difficult to decide which layers should be optimized for hardware implementation, pipelining, buffering, or improved dataflow. This can lead to inefficient use of compute resources and unnecessary memory traffic.

## 3. What is new in your approach, and why do you think it will succeed?
The proposed approach connects deep learning profiling with hardware design insight. Instead of treating the CNN as a black-box model, this project studies layer-level workload characteristics such as MAC count, mult-add distribution, and arithmetic intensity. Based on this analysis, the project identifies compute-heavy and memory-sensitive layers and uses that information to guide hardware-aware optimization. This approach is expected to succeed because accelerator performance depends heavily on understanding the balance between computation and memory movement, and these metrics provide a clear basis for making design decisions.
