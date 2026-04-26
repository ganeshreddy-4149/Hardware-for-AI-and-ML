# Codefest 4 — CMAN: Manual INT8 Symmetric Quantization
ECE 410/510 | Hardware for AI and ML | Spring 2026

---

## Given Weight Matrix W (FP32, 4x4)

```
W = [  0.85,  -1.20,   0.34,   2.10 ]
    [ -0.07,   0.91,  -1.88,   0.12 ]
    [  1.55,   0.03,  -0.44,  -2.31 ]
    [ -0.18,   1.03,   0.77,   0.55 ]
```

---

## Task 1 — Scale Factor

Formula used: S = max(|W|) / 127

Step 1: Find the absolute value of every element in W.

```
|W| = [ 0.85,  1.20,  0.34,  2.10 ]
      [ 0.07,  0.91,  1.88,  0.12 ]
      [ 1.55,  0.03,  0.44,  2.31 ]
      [ 0.18,  1.03,  0.77,  0.55 ]
```

Step 2: Find the maximum value.

    max(|W|) = 2.31   (this is the element W[2][3] = -2.31)

Step 3: Compute S.

    S = 2.31 / 127 = 0.01818898

---

## Task 2 — Quantize

Formula used: W_q = round(W / S), then clamp each value to [-128, 127]

Step 1: Divide every element of W by S = 0.01818898

```
W / S = [  46.73,  -65.97,   18.69,  115.45 ]
        [  -3.85,   50.03, -103.36,    6.60 ]
        [  85.22,    1.65,  -24.19, -127.00 ]
        [  -9.90,   56.63,   42.33,   30.24 ]
```

Step 2: Round each value to the nearest integer, then clamp to [-128, 127].

All values already fall within [-128, 127] after rounding so no clamping was needed.
The closest element to the boundary is W[2][3] which gives exactly -127.00 after dividing.

```
W_q (INT8 matrix) =
    [  47,  -66,   19,  115 ]
    [  -4,   50, -103,    7 ]
    [  85,    2,  -24, -127 ]
    [ -10,   57,   42,   30 ]
```

---

## Task 3 — Dequantize

Formula used: W_deq = W_q x S   where S = 0.01818898

Multiply every element of W_q back by S to recover approximate original values.

```
W_deq (FP32) =
    [  0.854882,  -1.200472,   0.345591,   2.091732 ]
    [ -0.072756,   0.909449,  -1.873464,   0.127323 ]
    [  1.546063,   0.036378,  -0.436535,  -2.310000 ]
    [ -0.181890,   1.036772,   0.763937,   0.545669 ]
```

Note: W[2][3] dequantizes back to exactly -2.310000 because it landed
exactly on -127 during quantization, so the reconstruction is perfect.

---

## Task 4 — Error Analysis

Formula used: error = |W - W_deq|

Compute the absolute difference between the original W and dequantized W_deq
for every element.

```
|W - W_deq| =
    [ 0.004882,  0.000472,  0.005591,  0.008268 ]
    [ 0.002756,  0.000551,  0.006536,  0.007323 ]
    [ 0.003937,  0.006378,  0.003465,  0.000000 ]
    [ 0.001890,  0.006772,  0.006063,  0.004331 ]
```

Largest error element:

    Position : row 0, column 3
    Original : W[0][3]   = 2.10
    Recovered: W_deq[0][3] = 2.091732
    Error    : |2.10 - 2.091732| = 0.008268

Mean Absolute Error (MAE):

    MAE = (sum of all 16 errors) / 16
        = 0.069213 / 16
        = 0.004326

The MAE of 0.004326 is very small, which confirms that the scale factor
S = 0.01818898 was a good choice and the quantization preserved all
values well.

---

## Task 5 — Bad Scale Experiment (S_bad = 0.01)

Using S_bad = 0.01 instead of the correct S = 0.01818898.

Step 1: Divide W by S_bad = 0.01

```
W / S_bad =
    [   85.0,  -120.0,    34.0,   210.0 ]
    [   -7.0,    91.0,  -188.0,    12.0 ]
    [  155.0,     3.0,   -44.0,  -231.0 ]
    [  -18.0,   103.0,    77.0,    55.0 ]
```

Step 2: Round and clamp to [-128, 127].
Four elements exceed the INT8 range and must be clamped (marked with *):

```
W_q_bad =
    [   85,  -120,    34,   127* ]
    [   -7,    91,  -128*,    12 ]
    [  127*,    3,   -44,  -128* ]
    [  -18,   103,    77,     55 ]
```

Clamped elements explained:
- Row 0, Col 3 : 210.0  exceeds  127  -> clamped to  127
- Row 1, Col 2 : -188.0 below   -128  -> clamped to -128
- Row 2, Col 0 : 155.0  exceeds  127  -> clamped to  127
- Row 2, Col 3 : -231.0 below   -128  -> clamped to -128

Step 3: Dequantize using W_deq_bad = W_q_bad x S_bad

```
W_deq_bad =
    [  0.85,  -1.20,   0.34,   1.27 ]
    [ -0.07,   0.91,  -1.28,   0.12 ]
    [  1.27,   0.03,  -0.44,  -1.28 ]
    [ -0.18,   1.03,   0.77,   0.55 ]
```

Values lost due to clamping:
- W[0][3] =  2.10  was lost, recovered as  1.27  (error = 0.83)
- W[1][2] = -1.88  was lost, recovered as -1.28  (error = 0.60)
- W[2][0] =  1.55  was lost, recovered as  1.27  (error = 0.28)
- W[2][3] = -2.31  was lost, recovered as -1.28  (error = 1.03)

Step 4: Compute MAE with S_bad

    MAE (S_bad = 0.01) = 0.171250

Comparison:

    Correct S  -> MAE = 0.004326
    S_bad=0.01 -> MAE = 0.171250
    Ratio      = 0.171250 / 0.004326 = ~40x worse

One-sentence explanation:

When S is too small, dividing the weights by S produces values that exceed
the INT8 clamp range of [-128, 127], so those weights get permanently clipped
to the boundary and cannot be accurately recovered during dequantization,
resulting in a much larger reconstruction error.

---

## Summary

| Task | Result |
|------|--------|
| max(|W|) | 2.31 at position W[2][3] = -2.31 |
| Scale factor S | 2.31 / 127 = 0.01818898 |
| Quantization clamping | None needed — all values fit in [-128, 127] |
| Largest error | 0.008268 at W[0][3] = 2.10 |
| MAE (correct S) | 0.004326 |
| MAE (S_bad = 0.01) | 0.171250 (~40x worse) |
| Elements clamped with S_bad | 4 elements clamped |
