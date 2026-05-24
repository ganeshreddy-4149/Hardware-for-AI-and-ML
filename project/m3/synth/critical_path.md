# Critical Path Analysis — INT8 Conv2D Accelerator
**ECE 410/510 — Hardware for AI/ML, Spring 2026**
**Student:** Sai Ganesh Reddy Charian
**Tool:** OpenLane 2 (v2.3.1) | **PDK:** SKY130A | **Corner:** nom_tt_025C_1v80

---

## 1. Clock Configuration

| Parameter | Value |
|---|---|
| Target clock period | 10.0 ns (100 MHz) |
| Minimum achievable period | 12.44 ns |
| Maximum achievable frequency | 80.40 MHz |
| Clock skew (rise) | 0.277 ns |
| Clock skew (fall) | 0.245 ns |

---

## 2. Critical Path Summary

The critical path runs through the MAC array combinational logic. The design
targets a 10.0 ns clock period but the post-PnR STA reports a minimum period
of 12.44 ns, resulting in a timing deficit of 2.44 ns at the nominal corner.

### Worst Setup Path (nom_tt_025C_1v80)

| Field | Value |
|---|---|
| Path type | Setup (max delay) |
| Startpoint | _48052_ (rising edge flip-flop, clk) |
| Endpoint | _48052_ (rising edge flip-flop, clk) |
| Data arrival time | 1.855 ns |
| Data required time | 1.545 ns |
| Slack | +0.310 ns (MET at hold) |

### Worst Hold Path (nom_tt_025C_1v80)

| Field | Value |
|---|---|
| Path type | Hold (min delay) |
| Startpoint | _48119_ (rising edge flip-flop, clk) |
| Endpoint | _47944_ (rising edge flip-flop, clk) |
| Data arrival time | 1.824 ns |
| Data required time | 1.514 ns |
| Slack | +0.310 ns (MET) |

---

## 3. Why Setup Violations Occur

The INT8 Conv2D design performs 9 parallel multiply-accumulate operations in
a single clock cycle. Each MAC unit sign-extends two 8-bit operands to 32 bits
and multiplies them, then all 9 products are summed into a single 32-bit
accumulator. This combinational path — from the activation and weight registers
through 9 multipliers and an adder tree — is deep enough that it cannot
complete within 10 ns on the SKY130 process.

The critical path flows through:

```
act_buf register → sign extension → 32-bit multiplier → adder tree → acc_out register
```

This path involves approximately 15–20 logic levels of standard cells, which
at SKY130 speeds requires roughly 12.44 ns minimum.

---

## 4. Timing Violations by Corner

| Corner | Setup | Hold |
|---|---|---|
| nom_tt_025C_1v80 | ❌ Violated | ✅ Met |
| nom_ss_100C_1v60 | ❌ Violated | ❌ Violated |
| nom_ff_n40C_1v95 | ✅ Met | ✅ Met |
| max_tt_025C_1v80 | ❌ Violated | ✅ Met |
| min_tt_025C_1v80 | ❌ Violated | ✅ Met |

The fast corner (ff_n40C_1v95) meets timing because at very low temperature
and high voltage, transistors switch faster. The slow corner (ss_100C_1v60)
fails both setup and hold because at high temperature and low voltage,
transistors are slowest.

---

## 5. Proposed Fix for M4

Increase the clock period to 13.0 ns (76.9 MHz) in config.json. This gives
the MAC array sufficient time to complete all 9 multiply-accumulate operations
and allows the adder tree to settle before the next rising edge. Alternatively,
pipelining the MAC array across 2 clock cycles would allow 100 MHz operation
at the cost of one additional cycle of latency.

```json
"CLOCK_PERIOD": 13.0
```

---

## 6. Physical Design Results

| Metric | Value |
|---|---|
| DRC | ✅ PASSED |
| LVS | ✅ PASSED |
| Antenna violations | 49 pin / 39 net (non-critical) |
| Total stages completed | 97 / 97 |
