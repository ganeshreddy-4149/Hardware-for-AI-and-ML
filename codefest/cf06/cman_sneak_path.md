# CMAN — Sneak Paths in a Resistive Crossbar
**ECE 410/510 — Hardware for AI & ML | Codefest 6 | Spring 2026**
**Student:** Sai Ganesh Reddy Charian

---

## Crossbar Setup

The circuit is a 2×2 resistive crossbar where rows carry input voltages and columns carry output currents.

**Cell resistances:**
- R[0][0] = 1 kΩ (ON weight — low resistance)
- R[0][1] = 2 kΩ (OFF weight — high resistance)
- R[1][0] = 2 kΩ (OFF weight — high resistance)
- R[1][1] = 1 kΩ (ON weight — low resistance)

**Grid layout:**

```
              col 0          col 1
              |              |
V_row0=1V --- R[0][0]=1kΩ --+-- R[0][1]=2kΩ --- V_col1
              |              |
V_row1    --- R[1][0]=2kΩ --+-- R[1][1]=1kΩ --- V_col1
              |
             GND (V_col0 = 0 V, virtual ground)
```

---

## (a) Ideal Read — I_col0

**Conditions:**
- V_row0 = 1 V (driven)
- V_col0 = 0 V (virtual ground, current sensing)
- V_row1 = 0 V (grounded)
- V_col1 = 0 V (grounded)

**Analysis:**

When all nodes except V_row0 are grounded, no sneak paths can form because there is no floating node for alternate current routes to develop. The only current path into col 0 is the direct path through R[0][0].

Current through R[1][0]: both terminals are at 0 V, so no current flows through it into col 0.

**Calculation:**

$$I_{col0} = \frac{V_{row0} - V_{col0}}{R[0][0]} = \frac{1\,V - 0\,V}{1000\,\Omega}$$

$$\boxed{I_{col0}^{ideal} = 1.0\,\text{mA}}$$

---

## (b) KCL Solution for Floating Node Voltages V_row1 and V_col1

**Conditions:**
- V_row0 = 1 V (driven)
- V_col0 = 0 V (virtual ground)
- V_row1 = floating (unknown)
- V_col1 = floating (unknown)

**Circuit paths with floating nodes:**

| Path | From | To | Resistor |
|------|------|----|----------|
| Direct | V_row0 (1V) | V_col0 (0V) | R[0][0] = 1 kΩ |
| Sneak step 1 | V_row0 (1V) | V_col1 (float) | R[0][1] = 2 kΩ |
| Sneak step 2 | V_col1 (float) | V_row1 (float) | R[1][1] = 1 kΩ |
| Sneak step 3 | V_row1 (float) | V_col0 (0V) | R[1][0] = 2 kΩ |

The sneak path is: V_row0 → R[0][1] → V_col1 → R[1][1] → V_row1 → R[1][0] → V_col0

---

### KCL at Node V_col1

At node V_col1, the sum of currents entering equals the sum of currents leaving.

**Current in** (from V_row0 through R[0][1]):

$$I_{in} = \frac{V_{row0} - V_{col1}}{R[0][1]} = \frac{1 - V_{col1}}{2000}$$

**Current out** (to V_row1 through R[1][1]):

$$I_{out} = \frac{V_{col1} - V_{row1}}{R[1][1]} = \frac{V_{col1} - V_{row1}}{1000}$$

**KCL equation (I_in = I_out):**

$$\frac{1 - V_{col1}}{2000} = \frac{V_{col1} - V_{row1}}{1000}$$

Multiply both sides by 2000:

$$1 - V_{col1} = 2(V_{col1} - V_{row1})$$

$$1 - V_{col1} = 2\,V_{col1} - 2\,V_{row1}$$

$$\boxed{1 = 3\,V_{col1} - 2\,V_{row1}} \quad \text{...(Equation 1)}$$

---

### KCL at Node V_row1

At node V_row1, the sum of currents entering equals the sum of currents leaving.

**Current in** (from V_col1 through R[1][1]):

$$I_{in} = \frac{V_{col1} - V_{row1}}{1000}$$

**Current out** (to V_col0 = 0 V through R[1][0]):

$$I_{out} = \frac{V_{row1} - 0}{R[1][0]} = \frac{V_{row1}}{2000}$$

**KCL equation (I_in = I_out):**

$$\frac{V_{col1} - V_{row1}}{1000} = \frac{V_{row1}}{2000}$$

Multiply both sides by 2000:

$$2(V_{col1} - V_{row1}) = V_{row1}$$

$$2\,V_{col1} - 2\,V_{row1} = V_{row1}$$

$$2\,V_{col1} = 3\,V_{row1}$$

$$\boxed{V_{col1} = \frac{3}{2}\,V_{row1}} \quad \text{...(Equation 2)}$$

---

### Solving the System

**Substitute Equation 2 into Equation 1:**

$$1 = 3 \cdot \left(\frac{3}{2}\,V_{row1}\right) - 2\,V_{row1}$$

$$1 = \frac{9}{2}\,V_{row1} - \frac{4}{2}\,V_{row1}$$

$$1 = \frac{5}{2}\,V_{row1}$$

$$\boxed{V_{row1} = \frac{2}{5} = 0.4\,\text{V}}$$

**Back-substitute to find V_col1:**

$$V_{col1} = \frac{3}{2} \times 0.4 = \boxed{0.6\,\text{V}}$$

**Summary of floating node voltages:**

| Node | Voltage |
|------|---------|
| V_row1 | **0.4 V** |
| V_col1 | **0.6 V** |

---

## (c) Actual I_col0 With Sneak Path Current Itemized

With the floating node voltages now known, the actual current flowing into col 0 has two contributions:

**Contribution 1 — Direct path through R[0][0]:**

$$I_{direct} = \frac{V_{row0} - V_{col0}}{R[0][0]} = \frac{1\,V - 0\,V}{1000\,\Omega} = 1.0\,\text{mA}$$

This is the intended current that represents the stored weight R[0][0].

**Contribution 2 — Sneak path through R[1][0]:**

The sneak path (V_row0 → R[0][1] → V_col1 → R[1][1] → V_row1 → R[1][0] → V_col0) dumps additional current into col 0 via R[1][0]:

$$I_{sneak} = \frac{V_{row1} - V_{col0}}{R[1][0]} = \frac{0.4\,V - 0\,V}{2000\,\Omega} = 0.2\,\text{mA}$$

**Total actual current at col 0:**

$$I_{col0}^{actual} = I_{direct} + I_{sneak} = 1.0\,\text{mA} + 0.2\,\text{mA}$$

$$\boxed{I_{col0}^{actual} = 1.2\,\text{mA}}$$

**Error introduced by sneak path:**

$$\text{Error} = \frac{1.2 - 1.0}{1.0} \times 100\% = 20\%$$

| Component | Current | Source |
|-----------|---------|--------|
| I_direct (R[0][0]) | 1.0 mA | Intended weight path |
| I_sneak (R[1][0]) | 0.2 mA | Unintended sneak path |
| **I_col0 total** | **1.2 mA** | **Read value (wrong)** |

---

## (d) How Sneak Paths Corrupt MVM Results

The sneak path current adds an extra 0.2 mA to col 0 through an unintended route (V_row0 → R[0][1] → R[1][1] → R[1][0] → col 0) that does not correspond to any weight stored in col 0 — this makes the hardware report a dot product of 1.2 instead of the correct 1.0, directly corrupting the MVM output vector. In large crossbar arrays used for neural network inference, this problem scales exponentially: a 128×128 array has thousands of possible sneak routes per column, making the accumulated error large enough to completely overwhelm the true signal and produce meaningless outputs. To prevent this, real crossbar memory cells require a selector element (such as a diode or 1T1R transistor) at each intersection to physically block reverse current flow and eliminate the sneak paths.

---

*File: codefest/cf06/cman_sneak_path.md*
*ECE 410/510 Spring 2026 — Codefest 6*
