# Benchmark Results
## ECE 510 — CF09 | INT8 Conv2D Hardware Accelerator
**Student:** Sai Ganesh Reddy Charian

---

## Platform Specifications

| Component | SW Baseline | HW Accelerator |
|---|---|---|
| Platform | Google Colab CPU | SKY130 ASIC (OpenLane 2 synthesis) |
| CPU/Clock | Intel Xeon @ 2.20GHz | 80.40 MHz (synthesis-achieved) |
| Implementation | Pure Python 7-loop Conv2D | 9-MAC parallel array, SystemVerilog |
| Interface | N/A | AXI4-Lite, 32-bit, 100 MHz |

---

## Performance Results

| Metric | SW Baseline | HW Accelerator | Notes |
|---|---|---|---|
| Median execution time | 129.21 ms | 0.269 ms (PROJECTED) | Per inference |
| Throughput | 0.003009 GFLOP/s | 1.447 GFLOP/s (PROJECTED) | Peak |
| Samples/sec | 7.7391 | 3,722 (PROJECTED) | Per second |
| Total FLOPs | 388,800 | 388,800 | Same workload |
| Peak memory | 180.94 MB | 0.40 GB/s BW (PROJECTED) | AXI4-Lite interface |
| Power | ~85 W (Xeon TDP) | 0.039785 W (synthesis) | Total chip power |

---

## Speedup

| Metric | Value | Label |
|---|---|---|
| Throughput speedup | **480.9x** | PROJECTED |
| Execution time speedup | **480.3x** | PROJECTED |
| Energy efficiency (SW) | 0.0000354 GFLOP/s/W | Measured |
| Energy efficiency (HW) | 36.37 GFLOP/s/W | PROJECTED |
| Energy efficiency improvement | **1,027,966x** | PROJECTED |

---

## Projection Assumptions

All HW accelerator numbers are labeled PROJECTED. The following assumptions apply:

1. One full MAC completes every clock cycle with no pipeline stalls
2. All 9 MACs operate in parallel every clock cycle
3. No memory latency or AXI4-Lite transfer overhead included
4. Clock frequency used is synthesis-achieved 80.40 MHz (not target 100 MHz)
5. FLOPs per inference = 388,800 matching M1 SW baseline dimensions exactly
6. SW baseline power estimated from Intel Xeon TDP (85W) — not directly measured
7. HW power from OpenLane 2 synthesis report (total = 0.039785 W)

---

## Notes on SW Baseline Reproducibility

The SW baseline was originally measured during M1 submission on Google Colab
(Intel Xeon @ 2.20GHz, Linux 6.6.113+, Python 3.12.13, batch size 1).
A CF09 reproduction run confirmed same CPU model and identical FLOPs count
(388,800). Runtime variation observed due to Colab shared server load.
M1 measured value (0.003009 GFLOP/s) used as official SW baseline.

---

## Synthesis Results Summary (M3)

| Metric | Value |
|---|---|
| Tool | OpenLane 2 v2.3.1 |
| PDK | SKY130A (sky130_fd_sc_hd) |
| Clock achieved | 80.40 MHz |
| Total cells | 24,346 |
| Area | 254,510 µm² (0.255 mm²) |
| DRC errors | 0 |
| LVS errors | 0 |
| Total power | 0.039785 W |
