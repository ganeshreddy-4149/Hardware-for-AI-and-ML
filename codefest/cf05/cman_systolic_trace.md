# Codefest 5 — CMAN: 2x2 Weight-Stationary Systolic Array Trace
ECE 410/510 | Hardware for AI and ML | Spring 2026

---

## Given Matrices

```
A = [ 1,  2 ]        B = [ 5,  6 ]
    [ 3,  4 ]            [ 7,  8 ]
```

Expected output:

```
C = A x B = [ 19,  22 ]
            [ 43,  50 ]
```

Verification:

    C[0][0] = (1x5) + (2x7) = 5  + 14 = 19
    C[0][1] = (1x6) + (2x8) = 6  + 16 = 22
    C[1][0] = (3x5) + (4x7) = 15 + 28 = 43
    C[1][1] = (3x6) + (4x8) = 18 + 32 = 50

---

## Task 1 — 2x2 PE Diagram with Preloaded Weights

In a weight-stationary systolic array the weights are loaded into each
Processing Element (PE) once at the start and never change. Each PE stores
one element of B. The input rows of A flow into the grid cycle by cycle.
Each PE multiplies the incoming input by its stored weight and adds the
result into a running accumulator.

Weight assignment rule: PE[k][j] stores B[k][j]

```
                   Col 0                  Col 1
             +----------------+      +----------------+
  k=0 row    |   PE[0][0]     |      |   PE[0][1]     |
             |   weight = 5   |      |   weight = 6   |
             |   B[0][0] = 5  |      |   B[0][1] = 6  |
             +----------------+      +----------------+

             +----------------+      +----------------+
  k=1 row    |   PE[1][0]     |      |   PE[1][1]     |
             |   weight = 7   |      |   weight = 8   |
             |   B[1][0] = 7  |      |   B[1][1] = 8  |
             +----------------+      +----------------+
```

Preloaded weights (fixed for entire computation):

    PE[0][0] = B[0][0] = 5
    PE[0][1] = B[0][1] = 6
    PE[1][0] = B[1][0] = 7
    PE[1][1] = B[1][1] = 8

Data flow during computation:

    Cycle 1: A column k=0 is broadcast. A[0][0]=1 and A[1][0]=3 flow in.
             PE[0][0] and PE[0][1] receive inputs from row i=0 and i=1 of A, for k=0.
             PE[1][0] and PE[1][1] receive inputs from row i=0 and i=1 of A, for k=1.

Each PE accumulates: acc += input x stored_weight
No weight ever moves or gets reloaded during the computation.

---

## Task 2 — Cycle-by-Cycle Execution Table

How the inputs are routed: C[i][j] = sum over k of A[i][k] x B[k][j].
PE[k][j] holds weight B[k][j]. In cycle 1 it receives A[i][0] for all rows i.
In cycle 2 it receives A[i][1] for all rows i. The partial products for
the same output C[i][j] come from PE[0][j] in cycle 1 and PE[1][j] in cycle 2,
then get summed together at the end.

| Cycle | PE[0][0] w=5              | PE[0][1] w=6              | PE[1][0] w=7              | PE[1][1] w=8              |
|-------|---------------------------|---------------------------|---------------------------|---------------------------|
| 1     | A[0][0]x5 = 1x5 = **5**   | A[0][0]x6 = 1x6 = **6**   | A[0][1]x7 = 2x7 = **14**  | A[0][1]x8 = 2x8 = **16**  |
|       | A[1][0]x5 = 3x5 = **15**  | A[1][0]x6 = 3x6 = **18**  | A[1][1]x7 = 4x7 = **28**  | A[1][1]x8 = 4x8 = **32**  |
| 2     | C[0][0] = 5+14 = **19** ✓  | C[0][1] = 6+16 = **22** ✓  | C[1][0] = 15+28 = **43** ✓ | C[1][1] = 18+32 = **50** ✓ |
| 3     | output written             | output written             | output written             | output written             |
| 4     | idle                       | idle                       | idle                       | idle                       |

Partial sum breakdown for each output element:

    C[0][0]: PE[0][0] gives 1x5=5, PE[1][0] gives 2x7=14, total = 5+14 = 19
    C[0][1]: PE[0][1] gives 1x6=6, PE[1][1] gives 2x8=16, total = 6+16 = 22
    C[1][0]: PE[0][0] gives 3x5=15, PE[1][0] gives 4x7=28, total = 15+28 = 43
    C[1][1]: PE[0][1] gives 3x6=18, PE[1][1] gives 4x8=32, total = 18+32 = 50

All four outputs match the expected result.

---

## Task 3 — Counts

**Total MACs**

Each output element C[i][j] requires 2 multiply-accumulate operations,
one for k=0 and one for k=1. There are 4 output elements total.

    Total MACs = 4 output elements x 2 MACs each = 8 MACs

**Input Reuse Count**

Each element of A is used in two different PEs. For example, A[0][0]=1
feeds both PE[0][0] (to compute part of C[0][0]) and PE[0][1] (to compute
part of C[0][1]) in the same cycle. This happens for every element of A.

    Input reuse factor = 2
    (each of the 4 elements of A is used in 2 PEs, once per output column)

**Off-chip Memory Accesses**

Matrix A — each element is read from off-chip memory exactly once and
broadcast across all PE columns in that cycle.

    A off-chip reads  = 4 reads  (2x2 elements, each read once)

Matrix B — all weights are preloaded into PEs once at startup before
computation begins. No weight is ever fetched from off-chip again during
the computation. This is the defining property of weight-stationary dataflow.

    B off-chip reads  = 4 reads  (2x2 weights, each loaded once)

Matrix C — each output element is written back to off-chip memory once
after the final accumulation is complete.

    C off-chip writes = 4 writes (2x2 results, each written once)

    Total off-chip accesses = 4 + 4 + 4 = 12

---

## Task 4 — Output-Stationary (One Sentence)

In an output-stationary systolic array, each PE is assigned one fixed output
element and holds its accumulator stationary throughout the entire computation
while both input activations and weights flow through the array, so no partial
sum ever needs to move until the final accumulated result is ready to be
written out.

---

## Summary

| Item                    | Value                              |
|-------------------------|------------------------------------|
| Array size              | 2x2 PEs                            |
| Dataflow style          | Weight-stationary                  |
| Total MACs              | 8                                  |
| Active compute cycles   | 2                                  |
| Output write cycle      | 1                                  |
| Input reuse factor      | 2x (each A element used in 2 PEs)  |
| A off-chip reads        | 4                                  |
| B off-chip reads        | 4 (loaded once at startup)         |
| C off-chip writes       | 4                                  |
| Total off-chip accesses | 12                                 |
| C[0][0]                 | 19 ✓                               |
| C[0][1]                 | 22 ✓                               |
| C[1][0]                 | 43 ✓                               |
| C[1][1]                 | 50 ✓                               |
