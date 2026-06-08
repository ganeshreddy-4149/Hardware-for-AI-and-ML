# Hardware for AI and ML — ECE 410/510, Spring 2026

**Student:** Sai Ganesh Reddy Charian
**Course:** ECE 410/510 — Hardware for Artificial Intelligence and Machine Learning
**Portland State University | Prof. Christof Teuscher**
**GitHub:** https://github.com/ganeshreddy-4149/Hardware-for-AI-and-ML

---

## M4 Final Submission — INT8 Conv2D Hardware Accelerator Chiplet

This repository contains all coursework for ECE 410/510 Spring 2026. The final deliverable is a
single-layer INT8 3x3 Conv2D hardware accelerator targeting the SkyWater SKY130A 130nm process,
achieving 598.2x speedup and 12,580,791x energy improvement over a pure Python software baseline.
The M4 folder and design justification report are the primary graded deliverables.

**M4 deliverables folder:** [project/m4/README.md](project/m4/README.md)
**Design justification report:** [project/m4/report/design_justification.pdf](project/m4/report/design_justification.pdf)

---

## Final Result (M4)

| Metric | Value |
|---|---|
| Technology | SKY130A 130nm |
| Tool | OpenLane 2 v2.3.1 |
| Clock | 100 MHz — timing MET at nominal corner |
| Area | 0.109 mm2 (109,464 um2) |
| Power | 0.004042 W |
| Speedup | 598.2x over Python baseline |
| Energy improvement | 12,580,791x |
| DRC/LVS | PASSED |

---

## Repository Structure

    Hardware-for-AI-and-ML/
    ├── README.md       # This file — top-level pointer to M4
    ├── project/
    │   ├── heilmeier.md
    │   ├── m1/         # Milestone 1 — SW baseline, interface selection, system diagram
    │   ├── m2/         # Milestone 2 — RTL, precision analysis, AXI4-Lite interface
    │   ├── m3/         # Milestone 3 — Synthesis, 80.40 MHz, timing violated
    │   └── m4/         # Milestone 4 — FINAL: 100 MHz, 10,305 cells, 598.2x speedup
    │       ├── README.md       # M4 file catalog
    │       ├── rtl/            # SystemVerilog RTL (synthesized)
    │       ├── tb/             # Testbench
    │       ├── sim/            # Simulation logs and waveforms
    │       ├── synth/          # OpenLane 2 synthesis results
    │       ├── bench/          # Benchmark data and roofline plot
    │       └── report/         # Design justification report and figures
    └── codefest/
        ├── cf01/ through cf09/ # Weekly codefest submissions

---

## Milestone Summary

| Milestone | Description | Key Result |
|---|---|---|
| M1 | SW baseline, interface selection | 0.003009 GFLOP/s Python baseline |
| M2 | RTL design, AXI4-Lite, precision | INT8/INT32 design, FSM verified |
| M3 | OpenLane synthesis | 80.40 MHz, 24,346 cells, timing violated |
| **M4** | **Final: timing closure, full report** | **100 MHz MET, 10,305 cells, 598.2x** |

---

## Quick Start — Simulation

    git clone https://github.com/ganeshreddy-4149/Hardware-for-AI-and-ML.git
    cd Hardware-for-AI-and-ML/project/m4
    iverilog -g2012 -o sim_out rtl/top.sv rtl/compute_core.sv tb/tb_top.sv
    vvp sim_out
    # Expected: PASS: INT8 Conv2D output matches expected value -6

---

## Design Architecture

- **Top module:** conv2d_top (top.sv)
- **Submodules:** mac_array, requantizer (compute_core.sv)
- **Interface:** AXI4-Lite slave (registers 0x000 to 0x400)
- **Dataflow:** Output-stationary
- **Compute:** 9 x INT8 parallel signed multipliers, INT32 adder tree
- **FSM:** IDLE -> LOAD -> COMPUTE -> OUTPUT -> DONE_S

---

## Git Tags

| Tag | Description |
|---|---|
| m1-submission | Milestone 1 final submission |
| m2-submission | Milestone 2 final submission |
| m3-submission | Milestone 3 final submission |
| m4-submission | Milestone 4 final submission |

---

## References

- SkyWater SKY130A PDK: https://github.com/google/skywater-pdk
- OpenLane 2: https://github.com/efabless/openlane2
- ECE 410/510 Course: Portland State University, Prof. Christof Teuscher
