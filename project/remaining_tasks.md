# Remaining Tasks Before M4
## ECE 510 | INT8 Conv2D Hardware Accelerator
**Student:** Sai Ganesh Reddy Charian

---

## Overview

The following three tasks are the highest-priority engineering changes
required before the M4 final submission. Each task addresses a specific
measured gap identified from M3 synthesis results or simulation limitations.

---

## Task 1 — Fix Multiplier Width to Close Timing at 100 MHz

**Problem:** M3 OpenLane 2 synthesis achieved only 80.40 MHz against
a 100 MHz target. The critical path runs through the nine MAC units,
each of which sign-extends 8-bit INT8 inputs to 32 bits before
multiplication, creating unnecessary 32×32-bit multipliers instead
of the correct 8×8-bit multipliers with 16-bit partial products.

**Specific change:** Replace the sign-extension pattern:
`assign p = {{24{a[7]}},a} * {{24{b[7]}},b};`
with a correctly-sized 8×8 signed multiply with 16-bit output:
`assign p = $signed(a) * $signed(b);`
This reduces the multiplier from 32×32 to 8×8, cutting the AND/XOR
gate count on the critical path by approximately 4× and targeting
positive setup slack at 100 MHz in the next OpenLane 2 run.

---

## Task 2 — Add Two-Stage Pipeline Register to MAC Adder Tree

**Problem:** The current MAC array is fully combinational — all 9
multiply-accumulate operations form one long unregistered chain from
input to acc_out. This single combinational path is 12.44 ns long
(from M3 timing report), which is why 100 MHz (10 ns period) cannot
be met.

**Specific change:** Insert one pipeline register between the
multiplier outputs and the carry-save adder tree, and one register
at acc_out. This breaks the 12.44 ns path into two stages of
approximately 6 ns each, allowing timing closure at 100 MHz with
positive slack and increasing projected throughput from 1.447 GFLOP/s
to 1.800 GFLOP/s (9 MACs × 2 FLOP × 100 MHz).

---

## Task 3 — Write Cycle-Accurate cocotb Throughput Testbench

**Problem:** The current M3 cocotb testbench verifies only functional
correctness — it checks that one output value matches the golden model
(-6). It does not count clock cycles or measure throughput, which means
all CF09 HW accelerator performance numbers are labeled PROJECTED and
cannot be converted to measured results.

**Specific change:** Write a new cocotb testbench
`codefest/cf09/tb_throughput.py` that:
(1) drives a full 30×30×8 output feature map through the accelerator,
(2) counts total clock cycles from first valid input to last valid
output using a cycle counter register,
(3) computes effective throughput as FLOPs / (cycles / clock_freq),
and (4) prints measured GFLOP/s to replace the projected value in the
M4 benchmark table.

---

*Generated for CF09 submission. All M3 synthesis numbers referenced
from OpenLane 2 v2.3.1 run on SKY130A (sky130_fd_sc_hd).*
