# Benchmark Results — INT8 Conv2D Hardware Accelerator
## ECE 410/510 — M4 | Spring 2026
**Student:** Sai Ganesh Reddy Charian

---

## Platform Specifications

| Component | SW Baseline | HW Accelerator |
|---|---|---|
| Platform | Google Colab CPU | SKY130 ASIC (OpenLane 2) |
| CPU/Clock | Intel Xeon @ 2.20 GHz | 100 MHz (timing MET at nom_tt_025C_1v80) |
| Implementation | Pure Python 7-loop Conv2D | 9-MAC parallel array, SystemVerilog |
| Interface | N/A | AXI4-Lite, 32-bit |

---

## Performance Results

| Metric | SW Baseline | HW Accelerator | Notes |
|---|---|---|---|
| Median execution time | 129.21 ms | 0.216 ms (PROJECTED) | Per inference |
| Throughput | 0.003009 GFLOP/s | 1.800 GFLOP/s (PROJECTED) | Peak at 100 MHz |
| Samples/sec | 7.74 | 4505 (PROJECTED) | Per second |
| Total FLOPs | 388,800 | 388,800 | Same workload |
| Power | ~85 W (Xeon TDP) | 0.004042 W (synthesis) | Total chip power |

---

## Speedup

| Metric | Value | Basis |
|---|---|---|
| Throughput speedup | 598.2x | PROJECTED |
| Execution time speedup | 581.6x | PROJECTED |
| Energy efficiency (SW) | 0.0000354 GFLOP/s/W | Measured |
| Energy efficiency (HW) | 445.3 GFLOP/s/W | PROJECTED |
| Energy efficiency improvement | 12,580,791x | PROJECTED |

---

## Throughput Derivation

HW throughput = MACs x FLOP/MAC x clock_freq = 9 x 2 x 100 MHz = 1.800 GFLOP/s
Speedup = 1.800 / 0.003009 = 598.2x
Execution time (HW) = 388,800 / 1.800e9 = 0.000216 s = 0.216 ms

---

## M4 vs M3 Improvement

| Metric | M3 | M4 | Change |
|---|---|---|---|
| Clock achieved | 80.40 MHz | 100 MHz (nom corner) | +24.4% |
| Cell count | 24,346 | 10,305 | -57.6% |
| Area | 254,510 um2 | 109,464 um2 | -57.0% |
| Total power | 0.039785 W | 0.004042 W | -89.8% |
| Throughput | 1.447 GFLOP/s | 1.800 GFLOP/s | +24.4% |
| Speedup | 480.9x | 598.2x | +24.4% |

---

## Projection Assumptions

All HW numbers labeled PROJECTED. Assumptions:
1. One full MAC completes every clock cycle with no pipeline stalls
2. All 9 MACs operate in parallel every clock cycle
3. No memory latency or AXI4-Lite transfer overhead included
4. Clock frequency: 100 MHz (timing MET at nom_tt_025C_1v80 corner)
5. FLOPs per inference = 388,800 matching M1 SW baseline exactly
6. SW baseline power estimated from Intel Xeon TDP (85W)
7. HW power from OpenLane 2 post-PnR synthesis (total = 0.004042 W)

---

## SW Baseline Reference

Originally measured during M1 on Google Colab (Intel Xeon @ 2.20GHz,
Python 3.12.13, batch size 1). M1 measured value used as official baseline.
