# Synthesis Interpretation — conv2d_top
**ECE 510 · Codefest 7 · Spring 2026**

## Clock Period and Timing

The design targets a 10 ns clock period (100 MHz), consistent with the AXI4-Lite interface specification. Yosys 0.9 does not produce formal worst-case slack without a mapped liberty file, but gate count and logic depth suggest feasibility at 100 MHz on SKY130. A full OpenROAD STA run in M3 will confirm exact slack with proper timing analysis.

## Critical Path Analysis

`mac_array` dominates with 13,925 cells (95.4% of total), compared to 295 in `conv2d_top` and 368 in `requantizer`. The critical path originates from `h_cnt`/`w_cnt` position counters, flows through sign-extension logic for the nine INT8 inputs, then through the nine parallel 32×32-bit multiplier trees, into the carry-save adder tree that accumulates partial products, and finally into the `acc_out` register. Dominant cell types along this path are `$_AND_` (6,150 instances) and `$_XOR_` (5,148 instances), which implement the ripple-carry addition stages of the multiplier array.

## Total Cell Area and Top Contributors

**14,588 total cells.** The three largest contributors by instance count are `$_AND_` with 6,276 total instances, `$_XOR_` with 5,213 total instances, and `$_MUX_` with 387 total instances. Memory footprint: 84,160 bits (~10.5 KB) in four register arrays — weights (216B), activations (3,072B), bias (32B), output (7,200B). These will map to SRAM macros in OpenLane flow.

## Warnings and Constraints

Two constant DFFs (`$procdff$261`, `$procdff$265`) removed during optimization — correspond to hardwired AXI response signals (`s_axil_bresp`, `s_axil_rresp` = `2'b00`). Expected behavior. No hold violations reported. The multiplier array dominance confirms compute-bound classification (AI = 12.10 FLOP/byte).
