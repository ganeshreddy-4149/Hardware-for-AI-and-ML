# M3 Synthesis Plan
**ECE 510 · Codefest 7 · Spring 2026**

---

The CF07 synthesis run confirmed that `mac_array` dominates the design with 13,925 cells — 95.4% of the total 14,588 — driven entirely by the nine 32×32-bit unsigned multipliers expanded from the INT8 sign-extension. For M3, the priority change is to constrain the multiplier width correctly: since inputs are INT8 (8-bit), the partial products only need 16-bit intermediate width before accumulation into INT32, not full 32×32 expansion. This reduces the AND and XOR gate count in the critical path by roughly 4× and directly shortens the path from `h_cnt` register to `acc_out` register.

The second change is to pipeline the MAC adder tree across two clock stages — one register between the multiplier outputs and the carry-save adder tree, and one at `acc_out`. This breaks the long combinational chain identified in the stat report and should allow the design to close timing at 100 MHz with positive slack in OpenROAD STA.

The four memory arrays (84,160 bits total) will be declared as `(* ram_style = "block" *)` to force SRAM macro inference in the full OpenLane flow rather than being synthesized as flip-flop arrays. The clock target remains 10 ns (100 MHz).
