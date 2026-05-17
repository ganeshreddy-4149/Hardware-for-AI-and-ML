# CMAN — Sparsity Breakeven Analysis
**ECE 510 · Codefest 7 · Spring 2026**

---

## Task 1 — Expressions for Dense and Sparse MVM (N = 512)

Let **N = 512** and **s** = fraction of zeros (sparsity). The number of nonzero weights is **N²(1 − s)**.

### (a) Dense MVM Compute

Each output element requires N multiply-accumulate operations (one multiply + one add = 2 FLOPs). The matrix has N² elements, each contributing exactly one MAC.

$$\text{Dense FLOPs} = 2N^2 = 2 \times 512^2 = \mathbf{524{,}288 \text{ FLOPs}}$$

### (b) Dense Memory Bytes

Every weight is stored as FP32 (4 bytes). No index overhead.

$$\text{Dense Bytes} = 4N^2 = 4 \times 512^2 = \mathbf{1{,}048{,}576 \text{ bytes}} \approx 1 \text{ MB}$$

### (c) Sparse Compute (as a function of s)

Only nonzero weights participate in MACs. With sparsity s, the fraction of nonzeros is (1 − s).

$$\text{Sparse FLOPs}(s) = 2N^2(1 - s) = 524{,}288 \times (1 - s)$$

At s = 0 this recovers the dense count; at s = 1 it goes to zero, as expected.

### (d) Sparse Memory Bytes (as a function of s)

CSR format uses three arrays:

| Array | Size | Bytes each | Total |
|---|---|---|---|
| `values` (FP32 nonzeros) | N²(1−s) entries | 4 | 4N²(1−s) |
| `col_idx` (INT32 column index) | N²(1−s) entries | 4 | 4N²(1−s) |
| `row_ptr` (INT32 row pointers) | N+1 entries | 4 | 4(N+1) |

$$\text{Sparse Bytes}(s) = 4N^2(1-s) + 4N^2(1-s) + 4(N+1)$$

$$= 8N^2(1-s) + 4(N+1)$$

Substituting N = 512:

$$\text{Sparse Bytes}(s) = 8 \times 262{,}144 \times (1-s) + 4 \times 513$$

$$= \mathbf{2{,}097{,}152(1-s) + 2{,}052 \text{ bytes}}$$

---

## Task 2 — FLOPs Speedup and 2× Crossover

The theoretical FLOPs speedup is simply the ratio of dense to sparse compute:

$$\text{Speedup}_{\text{FLOPs}}(s) = \frac{2N^2}{2N^2(1-s)} = \frac{1}{1-s}$$

This is independent of N — sparsity alone drives the FLOPs benefit.

**Finding s for 2× speedup:**

$$\frac{1}{1-s} = 2 \implies 1 - s = \frac{1}{2} \implies \mathbf{s = 0.5}$$

At **50% sparsity**, the sparse format cuts the number of MACs exactly in half. Below this threshold the overhead of managing the CSR structure isn't yet worth the compute reduction; above it, each additional percent of sparsity continues to give proportional FLOPs relief.

---

## Task 3 — Memory Breakeven Sparsity

We want the sparsity level s* at which sparse memory equals dense memory.

### Setup

$$\text{Sparse Bytes} = \text{Dense Bytes}$$

$$8N^2(1-s^*) + 4(N+1) = 4N^2$$

### Derivation

$$8N^2 - 8N^2 s^* + 4N + 4 = 4N^2$$

$$8N^2 s^* = 8N^2 - 4N^2 + 4N + 4$$

$$8N^2 s^* = 4N^2 + 4N + 4$$

$$s^* = \frac{4N^2 + 4N + 4}{8N^2} = \frac{N^2 + N + 1}{2N^2}$$

### Substituting N = 512

$$s^* = \frac{512^2 + 512 + 1}{2 \times 512^2} = \frac{262{,}144 + 512 + 1}{524{,}288} = \frac{262{,}657}{524{,}288}$$

$$\mathbf{s^* \approx 0.5010 \ (50.1\%)}$$

### Interpretation

The CSR format adds a second copy of the nonzero indices (the `col_idx` array), which doubles the per-element storage relative to dense for the data itself. The row pointer array is negligible for large N (only 4 × 513 = 2,052 bytes vs. 1 MB total). As a result, the memory breakeven sits just barely above 50% — you need to eliminate slightly more than half the weights before CSR actually saves memory. Above s ≈ 50.1%, every additional zero reduces sparse storage; below it, the index overhead makes sparse format *larger* than dense.

---

## Task 4 — End-to-End Speedup at s = 0.9 (Memory-Bandwidth-Limited)

**Given:**
- N = 512, s = 0.9
- Bandwidth = 320 GB/s
- Hardware perfectly skips zero MACs and their memory loads

### Dense execution time

Dense memory load:

$$\text{Dense Bytes} = 4N^2 = 4 \times 262{,}144 = 1{,}048{,}576 \text{ bytes} \approx 1.000 \text{ MB}$$

$$t_{\text{dense}} = \frac{1{,}048{,}576}{320 \times 10^9} \approx 3.277 \ \mu\text{s}$$

### Sparse execution time at s = 0.9

$$\text{Sparse Bytes}(0.9) = 2{,}097{,}152 \times (1 - 0.9) + 2{,}052$$

$$= 2{,}097{,}152 \times 0.1 + 2{,}052 = 209{,}715 + 2{,}052 = 211{,}767 \text{ bytes}$$

$$t_{\text{sparse}} = \frac{211{,}767}{320 \times 10^9} \approx 0.6618 \ \mu\text{s}$$

### Speedup

$$\text{Speedup} = \frac{t_{\text{dense}}}{t_{\text{sparse}}} = \frac{1{,}048{,}576}{211{,}767} \approx \mathbf{4.95\times}$$

### Discussion

The speedup is noticeably below the naive FLOPs-only prediction of 10× (= 1/(1−0.9)) for two reasons. First, CSR stores each nonzero *twice* — once as a value and once as a column index — so the effective memory per nonzero is 8 bytes rather than the 4 bytes of dense. This cuts the memory reduction factor roughly in half. Second, the row pointer array adds a small fixed overhead (2,052 bytes) that is independent of sparsity. Together, these effects mean that at s = 0.9 the sparse format transfers about 20% of the dense bytes, yielding roughly 5× end-to-end speedup on a memory-bandwidth-limited system.

---


