# Synthesis Notes — INT8 Conv2D Hardware Accelerator
**ECE 410/510 — Hardware for AI/ML, Spring 2026**
**Student:** Sai Ganesh Reddy Charian
**Tool:** OpenLane 2 v2.3.1 | **PDK:** SKY130A (sky130_fd_sc_hd)
**Run tag:** m3_run | **Die area:** 900 × 900 µm | **Target density:** 40%

---

## 1. Synthesis Overview

The INT8 Conv2D accelerator was synthesized using OpenLane 2 version 2.3.1
targeting the SkyWater SKY130A process design kit with the high-density
standard cell library (sky130_fd_sc_hd). The full physical design flow was
executed including synthesis, floorplan, placement, clock tree synthesis,
routing, sign-off DRC, LVS, and static timing analysis across nine timing
corners. The flow completed all 97 stages successfully and produced a
manufacturable layout that passes both DRC and LVS checks.

---

## 2. Area Analysis

The synthesized design occupies a chip area of **254,510 µm²**
(approximately 0.255 mm²) within a 900 × 900 µm die. The design contains
24,346 standard cells mapped to SKY130 library primitives.

| Metric | Value |
|---|---|
| Total chip area | 254,510.35 µm² |
| Die size | 900 × 900 µm |
| Target placement density | 40% |
| Total standard cells | 24,346 |
| Sequential cells (flip-flops) | 390 (dfxtp_2) |
| Sequential area | 8,295.46 µm² (3.26% of total) |
| Combinational area | 96.74% of total |
| Number of wires | 24,324 |
| Number of ports | 21 (150 bits total) |

The dominance of combinational logic (96.74%) over sequential logic (3.26%)
reflects the nature of the design: a large parallel MAC array with 9 INT8
multipliers and an adder tree, all of which are purely combinational paths.
The 390 flip-flops correspond to the FSM state registers, the activation
buffer (25 × 8 = 200 bits), weight buffer (9 × 8 = 72 bits), AXI4-Lite
interface registers, and output buffer registers.

The most frequently used cell types are NAND3 (2,026 instances), O211AI
(2,049 instances), O21AI (1,737 instances), NAND2 (1,639 instances), and
A21OI (1,625 instances). This distribution is typical for arithmetic-heavy
designs where the synthesizer maps multiplier trees into NAND/NOR networks
for area efficiency.

---

## 3. Timing Analysis

The design was synthesized targeting a 10.0 ns clock period (100 MHz). Post
place-and-route static timing analysis using OpenROAD across nine corners
revealed that the design cannot meet timing at 100 MHz on the SKY130 process.

The minimum achievable clock period is **12.44 ns**, corresponding to a
maximum operating frequency of **80.40 MHz**. The timing deficit is 2.44 ns,
meaning the critical path takes 24.4% longer than the allocated clock period.

### Root Cause of Timing Violations

The critical path runs through the MAC array combinational logic. The design
computes 9 parallel INT8 × INT8 multiplications and sums all products in a
single clock cycle. Each multiplication involves sign extension from 8 to 32
bits followed by a 32-bit integer multiply. The nine 32-bit products are then
summed through an adder tree. This entire computation — from the input
registers through the multipliers and adder tree to the accumulator register —
must complete within one clock period.

On the SKY130 process at nominal conditions (25°C, 1.8V), this combinational
path through approximately 15–20 logic levels of standard cells requires a
minimum of 12.44 ns. At 100 MHz (10 ns period), the path is 2.44 ns too slow,
causing setup violations at the accumulator register.

### Corner-by-Corner Results

Setup violations were observed at the nominal corner (tt_025C_1v80) and the
slow corner (ss_100C_1v60). The fast corner (ff_n40C_1v95) meets timing
because at −40°C and 1.95V, transistors switch approximately 30% faster than
at nominal conditions. Physical design checks (DRC and LVS) passed at all
corners, confirming the layout is physically correct.

Clock skew was measured at 0.277 ns (rise-to-rise) and 0.245 ns
(fall-to-fall), which is acceptable for this design scale. Clock network
latency ranged from 1.133 ns to 1.410 ns across endpoints.

---

## 4. Power Analysis

Power estimation was performed using OpenROAD's IR drop and RC extraction
flow (RCX). The design operates at 1.8V nominal supply voltage. Detailed
power numbers are captured in the post-PnR STA logs. The combinational
dominance of the design (96.74% combinational cells) means dynamic power
will scale linearly with switching activity, which is directly proportional
to inference throughput.

---

## 5. Physical Design Results

| Check | Result |
|---|---|
| DRC (Magic) | ✅ PASSED |
| DRC (KLayout) | ✅ PASSED |
| LVS (Netgen) | ✅ PASSED |
| XOR check | ✅ PASSED |
| Antenna violations | ⚠️ 49 pin, 39 net (non-critical) |

The antenna violations are caused by long metal wires connected to gate
inputs accumulating charge during the manufacturing plasma etch process.
These are non-critical for an academic project and would be resolved in a
production design by inserting antenna diodes or rerouting the affected nets.

---

## 6. Deviations from M1/M2 Plan

The RTL design is consistent with the M1 Heilmeier plan and M2 compute core.
The AXI4-Lite interface, INT8 data path, and 3×3 Conv2D kernel are unchanged.
The only deviation is that the target clock frequency of 100 MHz was not
achieved due to the depth of the MAC array combinational path on SKY130. This
will be addressed in M4 by either increasing the clock period to 13.0 ns or
pipelining the MAC array across two clock cycles.

---

## 7. Lessons Learned

The most important lesson from this synthesis run is that arithmetic intensity
has a direct cost in timing. A design with AI = 12.10 FLOP/byte is
compute-bound by definition — it packs maximum computation per memory access.
On silicon, this translates to deep combinational paths that limit clock
frequency. The roofline model predicted compute-bound behavior at the
algorithm level; the synthesis results confirm it at the physical level. The
path to higher frequency is pipelining, which trades latency for throughput
— exactly the tradeoff that real AI accelerator architects make when targeting
high clock rates on advanced process nodes.
