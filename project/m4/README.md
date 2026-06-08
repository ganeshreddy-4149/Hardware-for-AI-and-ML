# M4 — Final Milestone: INT8 Conv2D Hardware Accelerator

**Student:** Sai Ganesh Reddy Charian
**Course:** ECE 410/510 — Hardware for AI/ML, Spring 2026
**Portland State University | Prof. Christof Teuscher**
**Submitted:** June 7, 2026

**Design justification report:** [report/design_justification.pdf](report/design_justification.pdf)

---

## Overview

M4 is the final milestone of the INT8 Conv2D hardware accelerator chiplet project.
The design targets the SkyWater SKY130A 130nm process using OpenLane 2 v2.3.1.
It achieves timing closure at 100 MHz (nominal corner), 10,305 standard cells,
0.109 mm2 area, and 0.004042 W total power. The M3 RTL had a critical 32x32
multiplier bug fixed in M4 using the $signed() operator, reducing cell count by
57.6% and achieving timing closure at the nominal corner.

**Key M4 result:** 598.2x projected speedup over Python SW baseline, 12,580,791x energy improvement.

---

## File Catalog

Every file in this M4 folder is listed below with its relative path, description,
and the checklist item and report section it supports.

| Relative Path | Description | Checklist Item | Report Section |
|---|---|---|---|
| rtl/top.sv | conv2d_top — AXI4-Lite slave logic, FSM, on-chip buffers, top-level synthesized module | Checklist 2: Final RTL | Section 4: Dataflow and Architecture |
| rtl/compute_core.sv | mac_array (9x INT8 multipliers, INT32 adder tree) + requantizer submodules | Checklist 2: Final RTL | Section 4: Dataflow and Architecture |
| rtl/interface.sv | AXI4-Lite register map reference — not synthesized, documents interface | Checklist 2: Final RTL | Section 5: Hardware Interface |
| tb/tb_top.sv | SystemVerilog testbench — drives all inputs via AXI4-Lite only, PASS/FAIL contract | Checklist 2: Final testbench | Section 6: Verification |
| sim/final_run.log | Icarus Verilog simulation output — shows PASS: output = -6 confirmed | Checklist 2: Final simulation log | Section 6: Verification |
| sim/final_waveform.png | Annotated end-to-end waveform showing all 5 FSM phases, PASS confirmed | Checklist 2: Final waveform | Section 6: Verification, Figure 4 |
| synth/config.json | OpenLane 2 configuration — CLOCK_PERIOD 10.0 ns, SKY130A, sky130_fd_sc_hd | Checklist 3: OpenLane config | Section 7: Synthesis Results |
| synth/openlane_run.log | Full OpenLane 2 run log — 78/78 stages completed in 1h 35min | Checklist 3: OpenLane run log | Section 7: Synthesis Results |
| synth/timing_report.txt | Post-PnR timing — WNS=0.0 ns at nom_tt_025C_1v80, timing MET at 100 MHz | Checklist 3: Timing report | Section 7: Synthesis Results |
| synth/area_report.txt | Post-PnR area — 10,305 cells, 109,464 um2, dominant contributor MAC array | Checklist 3: Area report | Section 7: Synthesis Results |
| synth/power_report.txt | Post-PnR power — total 0.004042 W, switching dominant from MAC array | Checklist 3: Power report | Section 7: Synthesis Results |
| bench/benchmark.md | HW vs SW benchmark — throughput 1.800 GFLOP/s, speedup 598.2x, energy 12,580,791x | Checklist 4: Benchmark | Section 8: Benchmark Results |
| bench/benchmark_data.csv | Raw benchmark numbers — all reported figures traceable to this CSV | Checklist 4: Raw data | Section 8: Benchmark Results |
| bench/roofline_final.png | Roofline plot — SKY130 HW roofline + SW baseline + M4 HW point at AI=12.10 | Checklist 4: Roofline plot | Section 2: Roofline Analysis, Figure 2 |
| report/design_justification.pdf | 9-section design justification report — 3,825 words, 15 pages, 4 embedded figures | Checklist 5: Report | All 9 sections |
| report/figures/block_diagram.png | Top-level architecture block diagram — conv2d_top, AXI, FSM, MAC, requantizer | Checklist 5: Figures | Section 1: Problem and Motivation, Figure 1 |
| report/figures/roofline_final.png | Roofline analysis plot — compute-bound confirmed at AI=12.10 FLOP/byte | Checklist 5: Figures | Section 2: Roofline Analysis, Figure 2 |
| report/figures/dataflow_diagram.png | Output-stationary dataflow — 9 MACs parallel, adder tree, requantizer flow | Checklist 5: Figures | Section 4: Dataflow and Architecture, Figure 3 |
| report/figures/final_waveform.png | End-to-end simulation waveform — all 5 FSM phases annotated, PASS confirmed | Checklist 5: Figures | Section 6: Verification, Figure 4 |

---

## Design Summary

| Parameter | Value |
|---|---|
| Target process | SkyWater SKY130A 130nm |
| Tool | OpenLane 2 v2.3.1 |
| Standard cell library | sky130_fd_sc_hd |
| Clock frequency | 100 MHz (10.0 ns period) |
| Timing | MET at nom_tt_025C_1v80 (WNS = 0.0 ns) |
| Total standard cells | 10,305 |
| Chip area | 109,464 um2 (0.109 mm2) |
| Total power | 0.004042 W |
| DRC | PASSED (Magic + KLayout) |
| LVS | PASSED (Netgen) |
| Interface | AXI4-Lite slave |
| Precision | INT8 inputs/weights, INT32 accumulator |
| MAC array | 9 x INT8 parallel multipliers |
| Dataflow | Output-stationary |

---

## Benchmark Results

| Metric | SW Baseline | HW Accelerator |
|---|---|---|
| Platform | Intel Xeon @ 2.20 GHz | SKY130A @ 100 MHz |
| Throughput | 0.003009 GFLOP/s | 1.800 GFLOP/s (projected) |
| Execution time | 129.21 ms | 0.216 ms |
| Power | ~85 W | 0.004042 W |
| Energy efficiency | 0.0000354 GFLOP/s/W | 445.3 GFLOP/s/W |
| Speedup | — | 598.2x |
| Energy improvement | — | 12,580,791x |

---

## Simulation Result

- **Test:** 3x3 kernel [[1,0,-1],[1,0,-1],[1,0,-1]] on 5x5 activation map
- **Top-left window:** [[1,2,3],[1,2,3],[1,2,3]]
- **Expected output:** -6 (verified by hand calculation)
- **Actual output:** -6
- **Result:** PASS

---

## Key Fix from M3 to M4

M3 RTL used manual 32-bit sign extension causing Yosys to infer 32x32 multipliers:

    assign p0 = {{24{a0[7]}},a0} * {{24{b0[7]}},b0};

M4 RTL uses the correct $signed() operator which correctly infers 8x8 multipliers:

    assign p0 = $signed(a0) * $signed(b0);

This single change reduced cell count by 57.6%, area by 57.0%, and power by 89.8%,
and achieved timing closure at the nominal corner (WNS = 0.0 ns at 100 MHz).
These RTL files differ from M3. The diff is documented above and in the report Section 7.

---

## M3 vs M4 Comparison

| Metric | M3 | M4 | Change |
|---|---|---|---|
| Standard cells | 24,346 | 10,305 | -57.6% |
| Chip area | 254,510 um2 | 109,464 um2 | -57.0% |
| Total power | 0.039785 W | 0.004042 W | -89.8% |
| Timing (nominal) | VIOLATED (WNS=-2.44ns) | MET (WNS=0.0ns) | Fixed |
| fmax | 80.40 MHz | 100 MHz | +24.4% |
| Speedup | 480.9x | 598.2x | +24.4% |
