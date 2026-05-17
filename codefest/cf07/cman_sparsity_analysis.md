# CMAN — Sparsity Breakeven Analysis
**ECE 510 · Codefest 7 · Spring 2026**

---

## Task 1 — Expressions for Dense and Sparse MVM (N = 512)

Let **N = 512** and **s** = fraction of zeros (sparsity). The number of nonzero weights is **N²(1 − s)**.

### (a) Dense MVM Compute

Dense matrix-vector multiply: each output is one dot product, so N MACs per output. That's 2 FLOPs per MAC. With N² elements total:

$$\text{Dense FLOPs} = 2N^2 = 2 \times 512^2 = \mathbf{524{,}288 \text{ FLOPs}}$$

### (b) Dense Memory Bytes

Each weight is FP32 (4 bytes), nothing else to store:

$$\text{Dense Bytes} = 4N^2 = 4 \times 512^2 = \mathbf{1{,}048{,}576 \text{ bytes}} \approx 1 \text{ MB}$$

### (c) Sparse Compute (as a function of s)

If we skip zeros, only the nonzero weights get computed. With sparsity s, that's a fraction (1 − s) of all MACs:

$$\text{Sparse FLOPs}(s) = 2N^2(1 - s) = 524{,}288 \times (1 - s)$$

Check: when s = 0 (no zeros), this equals the dense count. When s = 1 (all zeros), it goes to zero. Good.

### (d) Sparse Memory Bytes (as a function of s)

For CSR (Compressed Sparse Row), we need to store the values AND the column indices. That's two copies of each nonzero:

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

Speedup is just the ratio of what we compute densely vs sparsely:

$$\text{Speedup}_{\text{FLOPs}}(s) = \frac{2N^2}{2N^2(1-s)} = \frac{1}{1-s}$$

Notice this doesn't depend on N at all — only on sparsity.

**For 2× speedup, solve:**

$$\frac{1}{1-s} = 2 \implies 1 - s = \frac{1}{2} \implies \mathbf{s = 0.5}$$

At **50% sparsity**, you halve the MACs. But below 50%, the overhead of storing indices doesn't pay off. Above 50%, every extra zero helps.

---

## Task 3 — Memory Breakeven Sparsity

When does sparse format actually save memory? Set them equal and solve:

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

Why is the breakeven so close to 50%? Because CSR stores **two copies** of the nonzeros — the value AND the column index. So each nonzero takes 8 bytes instead of 4. The row pointer array (only 2,052 bytes) barely matters compared to 1 MB of dense storage. Result: you need to eliminate slightly MORE than half the weights just to break even. Once you pass 50.1%, you start saving memory.

---

## Task 4 — End-to-End Speedup at s = 0.9 (Memory-Bandwidth-Limited)

Now let's see real-world speedup on a bandwidth-limited system. Assume the hardware is fast enough to compute anything — the bottleneck is memory.

**Conditions:**
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

Why is the speedup only 4.95× instead of 10× (which FLOPs alone predict)? Two reasons:

1. **CSR overhead:** Each nonzero is stored twice (value + column index = 8 bytes). Dense stores only the value (4 bytes). So memory savings is cut roughly in half.

2. **Row pointer overhead:** The row_ptr array is 2,052 bytes, which is small but fixed. At 90% sparsity, it becomes noticeable relative to the data being transferred.

Combined: sparse transfers only 20% of the dense bytes, so we get about 5× speedup on memory bandwidth — not bad, but not the naive 10× prediction.

---


