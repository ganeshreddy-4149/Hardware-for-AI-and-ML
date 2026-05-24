# Milestone 3 — INT8 Conv2D Hardware Accelerator
**ECE 410/510 — Hardware for AI/ML, Spring 2026**
**Student:** Sai Ganesh Reddy Charian
**GitHub:** https://github.com/ganeshreddy-4149/Hardware-for-AI-and-ML

---

## Project Summary

This milestone integrates the M2 compute core and AXI4-Lite interface into a
single top-level module, verifies correct end-to-end operation through
co-simulation, and runs physical synthesis using OpenLane 2 on the SKY130A PDK.

---

## File Catalog

### RTL — `project/m3/rtl/`

| File | Description |
|---|---|
| `top.sv` | Top-level integrated module. Instantiates `mac_array` (9 parallel INT8 MACs, INT32 accumulator) and `requantizer` (INT32→INT8 arithmetic right shift + saturation). AXI4-Lite is the only path for all data and control. FSM: IDLE→LOAD→COMPUTE→OUTPUT→DONE_S. |

### Testbench — `project/m3/tb/`

| File | Description |
|---|---|
| `tb_top.sv` | End-to-end co-simulation testbench. Drives all data through AXI4-Lite interface only — no direct DUT port access. Tests 3×3 Conv2D on a 5×5 activation map with kernel [[1,0,-1],[1,0,-1],[1,0,-1]]. Expected output = −6 from independent hand calculation. Prints PASS/FAIL. |

### Simulation — `project/m3/sim/`

| File | Description |
|---|---|
| `cosim_run.log` | Icarus Verilog simulation log. Shows all AXI write/read phases. Confirms PASS: expected −6, actual −6. |
| `cosim_waveform.png` | Waveform screenshot showing all FSM phases: HOST WRITE, COMPUTE, DONE_S, HOST READ. Annotated with out=−6 (PASS). |
| `cosim_waveform.vcd` | VCD dump file from Icarus Verilog simulation. |

### Synthesis — `project/m3/synth/`

| File | Description |
|---|---|
| `config.json` | OpenLane 2 configuration. Design: conv2d_top. PDK: sky130A. Cell library: sky130_fd_sc_hd. Die: 900×900 µm. Clock: 10 ns. |
| `openlane_run.log` | Full OpenLane 2 flow log. 97 stages. DRC PASS, LVS PASS. Setup violations at nom/ss corners (max freq = 80.40 MHz). |
| `timing_report.txt` | Post-PnR STA report (nom_tt_025C_1v80 corner). Clock period min = 12.44 ns, fmax = 80.40 MHz. |
| `area_report.txt` | Yosys synthesis area report. Total cells: 24,346. Chip area: 254,510 µm². Sequential: 3.26%. |
| `power_report.txt` | Power analysis from post-PnR STA. Nominal corner (tt_025C_1v80). |
| `critical_path.md` | Critical path analysis. Identifies MAC array combinational path as bottleneck. Documents timing deficit of 2.44 ns and proposes fix for M4. |

### Top-level — `project/m3/`

| File | Description |
|---|---|
| `synthesis_notes.md` | Full synthesis analysis (≥500 words). Covers area, timing, power, physical design results, root cause of violations, and lessons learned. |
| `README.md` | This file. Catalog of all M3 deliverables with reproduction instructions. |

---

## How to Reproduce Co-Simulation

**Tool:** Icarus Verilog 12.0 (used on Google Colab)
**Command:**
```bash
iverilog -g2012 -o sim_out tb/tb_top.sv rtl/top.sv
vvp sim_out
```
**Expected output:**
```
PASS: INT8 Conv2D output matches expected value -6
```

---

## How to Reproduce Synthesis

**Tool:** OpenLane 2 v2.3.1 via Docker
**Image:** `ghcr.io/efabless/openlane2:2.3.1`
**Command:**
```bash
docker run --rm \
  -v "<path_to_m3/synth/openlane_run>:/work" \
  ghcr.io/efabless/openlane2:2.3.1 \
  python3 -m openlane --run-tag m3_run /work/config.json
```
**Results location:** `runs/m3_run/` inside the mounted work directory

---

## Deviations from M2 Plan

None. Interface (AXI4-Lite), compute kernel (3×3 INT8 Conv2D), and precision
(INT8 inputs, INT32 accumulator) are unchanged from M2. The clock target of
100 MHz was not achieved due to MAC array combinational path depth on SKY130
(max achievable: 80.40 MHz). This will be addressed in M4.
