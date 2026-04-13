Arithmetic Intensity Calculation
Dominant kernel
The dominant kernel is conv2d_int8_reference, accounting for approximately 88.09% of the profiled runtime within run_profile_target.

Kernel description
This project uses a single-layer INT8 3x3 Conv2D for CNN inference. The current software reference implementation uses:

INT8 input activations
INT8 weights
INT32 bias
INT32 accumulation and output storage
FLOPs calculation
For one forward pass of Conv2D, the FLOP count is:

FLOPs = 2 x N x C_out x H_out x W_out x C_in x K x K

Substituting the current configuration:

N = 1
C_in = 3
H = 32
W = 32
C_out = 8
K = 3
H_out = 30
W_out = 30
FLOPs = 2 x 1 x 8 x 30 x 30 x 3 x 3 x 3 FLOPs = 388800

Bytes transferred (assuming DRAM access with no reuse)
Input bytes: N x C_in x H x W x 1 byte = 1 x 3 x 32 x 32 x 1 = 3072

Weight bytes: C_out x C_in x K x K x 1 byte = 8 x 3 x 3 x 3 x 1 = 216

Bias bytes: C_out x 4 bytes = 8 x 4 = 32

Output bytes: N x C_out x H_out x W_out x 4 bytes = 1 x 8 x 30 x 30 x 4 = 28800

Total bytes = input + weights + bias + output Total bytes = 3072 + 216 + 32 + 28800 Total bytes = 32120

Arithmetic intensity
Arithmetic intensity = FLOPs / total bytes Arithmetic intensity = 388800 / 32120 Arithmetic intensity = 12.1046 FLOP/byte

Interpretation
An arithmetic intensity of 12.1046 FLOP/byte suggests that the Conv2D kernel has moderate compute density. This supports selecting Conv2D as the hardware acceleration target because it performs substantially more arithmetic work than simple memory-dominated kernels such as vector addition.

