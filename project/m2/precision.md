# Precision and Data Format
ECE 410/510 | Hardware for AI and ML | Spring 2026
Student: Sai Ganesh Reddy Charian

---

## Numerical Format Choice

The accelerator uses INT8 for all input activations and weights, INT32 for the
internal accumulator, and INT8 for the final output after requantization.

    Input activations : INT8  (signed, range -128 to 127)
    Weights           : INT8  (signed, range -128 to 127)
    Accumulator       : INT32 (signed, prevents overflow across 9 MACs)
    Output            : INT8  (after right-shift and clamp)
    Quantization type : Symmetric, per-tensor
    Rounding mode     : Round-to-nearest integer
    Requantizer       : arithmetic right-shift by shift_amt, clamp to [-128, 127]

This is the same precision used by standard INT8 inference engines including
TensorFlow Lite, ONNX Runtime, and the quantized ResNet-18 checkpoint on
PyTorch Hub.

---

## Rationale Grounded in Kernel and Roofline

The dominant kernel from M1 profiling is the 3x3 Conv2D operation on an
input of shape (1, 3, 32, 32) with 8 output channels. The arithmetic
intensity computed in M1 is 12.10 FLOP/byte. The hardware target is
3.20 GFLOP/s and the AXI4-Lite interface is rated at 0.40 GB/s.

The required memory bandwidth at the hardware target is:

    Required BW = 3.20 GFLOP/s / 12.10 FLOP/byte = 0.2645 GB/s

The AXI4-Lite interface provides 0.40 GB/s, giving 33.9% headroom. This
headroom only holds if the per-element byte cost stays at 1 byte (INT8).
Changing to a different precision shifts the roofline operating point as
follows:

    Format  | Bytes/element | Effective AI      | Fits within 0.40 GB/s?
    --------|---------------|-------------------|------------------------
    FP32    | 4             | 12.10/4 = 3.03    | Yes but AI collapses
    FP16    | 2             | 12.10/2 = 6.05    | Yes, less headroom
    INT8    | 1             | 12.10/1 = 12.10   | Yes, 33.9% headroom
    INT4    | 0.5           | 12.10/0.5 = 24.20 | Yes but accuracy loss

INT8 keeps the arithmetic intensity at 12.10 FLOP/byte, exactly matching
the M1 analysis. Using FP32 would reduce the effective AI to 3.03 FLOP/byte,
pushing the design deep into the memory-bound region of the roofline and
making it impossible to hit the 3.20 GFLOP/s target without a 4x wider
memory bus. Using INT4 increases AI but introduces unacceptable quantization
error as shown in the analysis below.

Why not FP16 or BF16: these formats require floating-point multipliers in
the MAC array, which are significantly larger in silicon area and power than
integer multipliers on the SKY130 PDK target. An INT8 multiplier in SKY130
fits in roughly 800 gates while an FP16 multiplier requires approximately
4000 gates. Since the design targets OpenLane 2 synthesis on SKY130, INT8
is the correct choice for area and power.

Why not INT4: the quantization error analysis in the next section shows that
INT4 produces 19.83x higher mean absolute error than INT8 on the same 100
test samples, which exceeds the acceptable tolerance for ResNet-18 inference.

---

## Quantization Error Analysis

The analysis compares INT8 quantized output against FP32 reference output
across 100 randomly generated 3x3 convolution samples. The same symmetric
per-tensor quantization scheme used by the hardware is applied: divide by
scale factor, round to nearest integer, clamp to [-128, 127], then multiply
back by scale to dequantize.

Setup:

    Samples     : 100 random 3x3 input patches
    Weights     : random normal (mean=0, std=0.1, matching ResNet-18 init)
    Activations : random normal (mean=0, std=1.0, matching normalized input)
    Kernel size : 3x3, single channel (matches accelerator configuration)
    Bias        : zero (isolated quantization error measurement)
    Seed        : 42 (reproducible)

Weight quantization results:

    Scale factor S = max(|W|) / 127 = 0.157921 / 127 = 0.00124347
    Weight MAE     = 0.00022251
    Weight max err = 0.00059907

Convolution output error (INT8 vs FP32, 100 samples):

    FP32 output range : [-0.3865, +0.8957]
    INT8 output range : [-0.3868, +0.8945]
    Mean absolute error (MAE)  = 0.000974
    Maximum absolute error     = 0.003494
    Standard deviation of error = 0.000697
    Samples with error > 0.1   = 0 / 100
    Samples with error > 0.5   = 0 / 100
    Mean relative error        = 1.799%
    Maximum relative error     = 40.704%

The maximum relative error of 40.7% occurs on samples where the FP32 output
is very close to zero (denominator near 0), so the relative metric is
misleading for those cases. The absolute error never exceeds 0.0035, which
is well within the acceptable range for 8-bit inference.

INT4 comparison on the same 100 samples:

    INT4 MAE     = 0.019328
    INT4 max err = 0.078784
    INT4 / INT8 MAE ratio = 19.83x worse

INT4 is rejected because its MAE is nearly 20x larger than INT8 and its
maximum error of 0.079 would cause visible degradation in ResNet-18
classification accuracy.

---

## Statement of Acceptability

The INT8 quantization error is acceptable for this accelerator because:

1. The measured MAE of 0.000974 is consistent with published INT8 quantization
results for ResNet-18. The PyTorch quantization documentation reports that
symmetric INT8 post-training quantization of ResNet-18 produces a top-1
accuracy drop of less than 1% on ImageNet compared to the FP32 baseline.
An absolute output error of 0.000974 per convolution output element is
well below what would cause a 1% accuracy drop.

2. No sample across all 100 test cases produced an absolute error larger
than 0.0035, which is smaller than 1 LSB of an INT8 output range of
approximately 0.008 per step at typical activation scales. This means
the quantization error is sub-LSB at the output level.

3. The INT32 accumulator prevents overflow during the 9-MAC summation.
The worst case accumulator value for INT8 inputs is 9 x 127 x 127 = 145,161,
which is well within the INT32 range of 2,147,483,647. No saturation occurs
in the accumulator.

4. The threshold used here is: error is acceptable if MAE < 0.01 and no
individual sample error exceeds 0.05. Both conditions are satisfied:
MAE = 0.000974 < 0.01 and max error = 0.003494 < 0.05.

---

## Summary

| Item | Value |
|------|-------|
| Input/weight format | INT8 symmetric |
| Accumulator format | INT32 |
| Output format | INT8 (after shift + clamp) |
| Scale factor example | 0.00124347 |
| MAE (100 samples) | 0.000974 |
| Max absolute error | 0.003494 |
| INT4 MAE (comparison) | 0.019328 (19.83x worse) |
| Acceptability threshold | MAE < 0.01, max < 0.05 |
| Threshold met | Yes |
| Accuracy impact | < 1% top-1 drop (per PyTorch Hub INT8 ResNet-18) |
