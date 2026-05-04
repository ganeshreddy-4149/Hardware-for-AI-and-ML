# CMAN -- Systolic array trace (weight-stationary)

**Given:**
- A = [[1, 2], [3, 4]]
- B = [[5, 6], [7, 8]]
- Expected C = A × B = [[19, 22], [43, 50]]

---

## Task 1: PE diagram with preloaded weights

Weight-stationary dataflow: B weights are preloaded and stay fixed in
each PE for the entire computation. Columns of A stream in on the input
wires (one column element per cycle). Partial sums flow downward through
PE rows. The 1-cycle stagger on the bottom input wire ensures the correct
partial sum meets the correct A value at PE[1][j].

```
  Col 0 of A         Col 1 of A
  streams down       streams down (staggered 1 cycle)
  A[0][0], A[1][0]   A[0][1], A[1][1]
       |                   |
       v                   v
  +----------+       +----------+
  | PE[0][0] |------>| PE[0][1] |   Row 0
  | weight=5 |       | weight=6 |
  +----------+       +----------+
       | psum down         | psum down
  +----------+       +----------+
  | PE[1][0] |------>| PE[1][1] |   Row 1
  | weight=7 |       | weight=8 |
  +----------+       +----------+
       | out               | out
    C[*][0]             C[*][1]

Preloaded weights:
  PE[0][0] = B[0][0] = 5
  PE[0][1] = B[0][1] = 6
  PE[1][0] = B[1][0] = 7
  PE[1][1] = B[1][1] = 8
```

---

## Task 2: Cycle-by-cycle trace table

Key insight: the TOP input wire streams column 0 of A (A[0][0]=1 then
A[1][0]=3). The BOTTOM input wire streams column 1 of A (A[0][1]=2 then
A[1][1]=4), staggered by 1 cycle. This stagger ensures that partial sums
from row 0 PEs arrive at row 1 PEs exactly when the matching A values
arrive, preventing incorrect mixing of row 0 and row 1 contributions.

Total cycles = 3N - 2 = 3x2 - 2 = 4 cycles for N=2.

| Cycle | Top wire (col 0 of A) | Bottom wire (col 1 of A) | PE[0][0] psum | PE[0][1] psum | PE[1][0] psum | PE[1][1] psum | C output |
|-------|-----------------------|--------------------------|---------------|---------------|---------------|---------------|----------|
| 1     | A[0][0] = 1           | --                       | 0+1x5 = 5     | 0+1x6 = 6     | 0             | 0             | --       |
| 2     | A[1][0] = 3           | A[0][1] = 2              | 0+3x5 = 15    | 0+3x6 = 18    | 5+2x7 = 19    | 6+2x8 = 22    | C[0][0]=19, C[0][1]=22 |
| 3     | --                    | A[1][1] = 4              | 0             | 0             | 15+4x7 = 43   | 18+4x8 = 50   | C[1][0]=43, C[1][1]=50 |
| 4     | --                    | --                       | drain         | drain         | drain         | drain         | all outputs valid |

**Verification of final C values:**

    C[0][0] = A[0][0]xB[0][0] + A[0][1]xB[1][0] = 1x5 + 2x7 =  5 + 14 = 19  correct
    C[0][1] = A[0][0]xB[0][1] + A[0][1]xB[1][1] = 1x6 + 2x8 =  6 + 16 = 22  correct
    C[1][0] = A[1][0]xB[0][0] + A[1][1]xB[1][0] = 3x5 + 4x7 = 15 + 28 = 43  correct
    C[1][1] = A[1][0]xB[0][1] + A[1][1]xB[1][1] = 3x6 + 4x8 = 18 + 32 = 50  correct

**Why the stagger works:** Without the 1-cycle stagger, A[0][0]=1 and
A[0][1]=2 would arrive at PE[1][0] simultaneously with partial sum 5,
mixing row 0 and row 1 contributions incorrectly. The stagger ensures
that when partial sum 5 (from 1x5) arrives at PE[1][0] in cycle 2, it
meets A[0][1]=2 -- the correct column 1 value for computing C[0][0].

---

## Task 3: Counts

### (a) Total MAC operations

Counted directly from the trace:
- Cycle 1: PE[0][0] and PE[0][1] each do 1 MAC = 2 MACs
- Cycle 2: all 4 PEs each do 1 MAC = 4 MACs
- Cycle 3: PE[1][0] and PE[1][1] each do 1 MAC = 2 MACs

    Total MACs = 2 + 4 + 2 = 8 = N^3 = 2^3

### (b) Input reuse

**A values:** Each column of A streams through both PE rows.
A[0][0]=1 is multiplied at PE[0][0] and its partial sum feeds PE[1][0].
Each A value touches N=2 PEs.

**B values:** Each weight stays fixed and is multiplied by N=2 different
A values across the computation.
B[0][0]=5 multiplies A[0][0]=1 in cycle 1 and A[1][0]=3 in cycle 2.

    Each A value reused: N = 2 times
    Each B value reused: N = 2 times

### (c) Off-chip memory accesses

    A reads:  4 elements x 1 load each = 4 accesses
    B reads:  4 elements x 1 pre-load  = 4 accesses (loaded once before compute)
    C writes: 4 elements x 1 store     = 4 accesses
    Total:    12 off-chip memory accesses

No A or B value is fetched from off-chip memory more than once. B values
are pre-loaded before computation begins and remain stationary throughout.
This is the key advantage of weight-stationary dataflow -- weight memory
bandwidth is minimized.

---

## Task 4: Output-stationary answer

In output-stationary dataflow, the accumulated partial sums (output
values) stay fixed in each PE for the entire computation, while both
weights and activations stream through the array.
