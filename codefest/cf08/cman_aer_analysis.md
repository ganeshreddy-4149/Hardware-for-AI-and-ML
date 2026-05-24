# CMAN — AER Bandwidth Analysis
## Codefest 8 | ECE 410/510 | Spring 2026
## Student: Sai Ganesh Reddy Charian

---

## Background

In this analysis, I am designing the off-chip output interface for a spiking neural network (SNN) accelerator. The SNN has N = 1024 output neurons, each firing at a mean rate of f = 50 Hz. Spikes are sent to the host using Address-Event Representation (AER) — a protocol where a packet is emitted only when a neuron fires, not on a fixed clock schedule. Neurons fire independently (Poisson-like statistics).

---

## Task 1: Mean Aggregate Spike Rate

**Formula:**

R = N × f

**Substituting values:**

R = 1024 neurons × 50 spikes/second/neuron

R = **51,200 spikes/second**

This means the entire output layer collectively produces 51,200 spike events every second on average. Each of those events triggers one AER packet transmission.

---

## Task 2: Mean AER Bandwidth

Each AER packet has the following structure:

| Field | Bits |
|---|---|
| Neuron address | 10 bits (log₂(1024) = 10) |
| Timestamp | 6 bits |
| Framing/parity overhead | 4 bits |
| **Total per packet** | **20 bits** |

**Formula:**

B = R × 20 bits/packet

**Substituting values:**

B = 51,200 spikes/second × 20 bits/spike

B = 1,024,000 bits/second

**Converting to Mbit/s:**

B = 1,024,000 / 1,000,000

B = **1.024 Mbit/s**

This is the mean bandwidth required to sustain AER communication at the average firing rate.

---

## Task 3: Interface Comparison

Comparing the mean required AER bandwidth (B = 1.024 Mbit/s) against standard interfaces from the M1 table:

| Interface | Max Bandwidth | Sustains Mean Rate? | Notes |
|---|---|---|---|
| I²C | ≤ 3.4 Mbit/s | ✅ Yes | 1.024 < 3.4, can sustain mean rate |
| SPI | ≤ 50 Mbit/s | ✅ Yes | Well within limit |
| AXI4-Lite | ~100 Mbit/s (effective, narrow bus) | ✅ Yes | Easily handles mean rate |

**Lowest-complexity interface that suffices: I²C**

I²C at 3.4 Mbit/s (Fast-mode Plus) can sustain the 1.024 Mbit/s mean rate. It is the simplest interface in the list in terms of pin count (only 2 wires: SDA and SCL) and controller complexity. SPI and AXI4-Lite both work too, but they are more complex to implement and that complexity is not needed at the mean rate.

---

## Task 4: Burst Peak Bandwidth

**Scenario:** 25% of 1024 neurons fire within a 1 ms window (synchronous burst).

**Number of neurons firing in burst:**

N_burst = 0.25 × 1024 = 256 neurons

**Packets generated in 1 ms:**

256 packets × 20 bits/packet = 5,120 bits in 1 ms

**Peak instantaneous bandwidth:**

B_peak = 5,120 bits / 0.001 seconds = 5,120,000 bits/second = **5.12 Mbit/s**

**Burst-to-mean ratio:**

Ratio = B_peak / B_mean = 5.12 / 1.024 = **5×**

The burst bandwidth is 5 times the mean bandwidth.

**Can I²C absorb this burst?**

I²C maximum is 3.4 Mbit/s, but the burst requires 5.12 Mbit/s. **I²C cannot absorb the burst directly.**

**Buffering required:** Yes. A FIFO buffer is needed to absorb the burst and drain it at the I²C rate.

**Rough buffer depth calculation:**

Excess data during burst = (5.12 − 3.4) Mbit/s × 1 ms = 1.72 Mbit/s × 0.001 s = 1,720 bits

In packets: 1,720 / 20 = 86 packets minimum

Rounding up for safety: **128 packets (2,560 bits) buffer depth** is sufficient.

---

## Task 5: Frame-Based Comparison

**Frame-based bandwidth:**

A conventional readout samples all 1024 neurons every 1 ms, sending 1 bit per neuron per sample.

- Bits per frame = 1024 bits
- Frames per second = 1 / 0.001 = 1000 frames/second
- B_frame = 1024 × 1000 = 1,024,000 bits/second = **1.024 Mbit/s**

**AER-to-frame bandwidth ratio at f = 50 Hz:**

- B_AER (mean) = 1.024 Mbit/s
- B_frame = 1.024 Mbit/s
- Ratio = 1.024 / 1.024 = **1.0**

At f = 50 Hz, AER and frame-based bandwidths are exactly equal.

**Crossover firing rate f_crossover:**

Setting AER bandwidth equal to frame-based bandwidth and solving for f:

N × f_crossover × 20 = N × (1/T_frame) × 1

1024 × f_crossover × 20 = 1024 × 1000 × 1

f_crossover = 1,024,000 / (1024 × 20)

f_crossover = 1,024,000 / 20,480

**f_crossover = 50 Hz**

**One-sentence implication:**

AER is the right choice when the mean firing rate is below 50 Hz, because at lower activity levels AER uses less bandwidth than frame-based readout by only transmitting packets for neurons that actually fire rather than scanning all neurons every frame.

---

## Summary Table

| Task | Result |
|---|---|
| Mean spike rate R | 51,200 spikes/second |
| Mean AER bandwidth B | 1.024 Mbit/s |
| Lowest-complexity interface | I²C (3.4 Mbit/s) |
| Burst peak bandwidth | 5.12 Mbit/s |
| Burst-to-mean ratio | 5× |
| Buffering required? | Yes — 128 packets deep |
| Frame-based bandwidth | 1.024 Mbit/s |
| AER/frame ratio at f=50 Hz | 1.0 |
| Crossover firing rate | f_crossover = 50 Hz |
