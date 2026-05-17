# Project Scope Assessment
**ECE 510 · Spring 2026 · Updated CF07**

---

## Scope: Confirmed with Targeted Adjustments

The project scope remains a single-layer INT8 3×3 Conv2D hardware accelerator synthesized via Yosys/OpenLane targeting SKY130, connected to the host via AXI4-Lite. This scope is confirmed as achievable for M3.

The CF07 synthesis run produced concrete numbers that ground this assessment. Total cell count came to 14,588, with `mac_array` at 13,925 cells (95.4%) as the dominant block. This confirms the architecture is synthesizable and that the MAC array is both the area and timing bottleneck — consistent with the compute-bound classification from the roofline (AI = 12.10 FLOP/byte, above ridge point of 4.47).

One scope adjustment is identified: the current RTL expands INT8 multiplications to full 32×32-bit unsigned multiply in Yosys, which inflates gate count significantly. For M3, the multiplier will be corrected to 8×8 signed with 16-bit intermediate products accumulated into INT32. This is a correctness fix within the existing scope, not a scope change.

The four on-chip memory arrays (84,160 bits total) will be explicitly annotated for SRAM macro inference in the OpenLane flow. The FSM, AXI4-Lite slave, requantizer, and position counters are all synthesizing cleanly and require no scope changes. The M3 deliverable is achievable before May 24.
