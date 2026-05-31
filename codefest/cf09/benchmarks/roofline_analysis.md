# Roofline Analysis
## ECE 510 — CF09 | INT8 Conv2D Hardware Accelerator
**Student:** Sai Ganesh Reddy Charian

---

## Gap Analysis — Projected vs Expected Performance

The HW accelerator was placed using the projected path since no
end-to-end cocotb throughput simulation was available. Projected
throughput of 1.447 GFLOP/s was computed from the synthesis-achieved
clock of 80.40 MHz and 9-MAC parallel array, assuming all MACs operate
every cycle with no stalls.

The dominant uncertainty is the zero-stall assumption. In practice,
AXI4-Lite introduces multi-cycle latency per transaction, leaving the
MAC array idle during data transfers. This overhead is not captured in
the formula (clock × ops/cycle) and is the largest likely source of
gap between projected and actual throughput. On-chip buffer loading
adds further uncounted cycles.

To convert this to a measurement, a cycle-accurate cocotb testbench
must count active MAC cycles versus total elapsed cycles and compute
effective throughput from that ratio.

---

*All HW numbers labeled PROJECTED. SW baseline is measured from M1.*
