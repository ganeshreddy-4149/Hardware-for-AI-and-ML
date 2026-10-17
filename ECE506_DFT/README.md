# ECE 506: DFT Implementation for INT8 Conv2D Accelerator

This directory contains the Design-for-Test (DFT) implementation flow for the INT8 Conv2D Hardware Accelerator, completed as a 3-credit ECE 506 project at Portland State University.

## Primary Tools Used
* **Siemens Tessent (2026.1-p1):** Scan insertion, EDT, OCC, LBIST, MBIST, and ATPG.
* **Siemens Questa (2024.2):** Pattern simulation and gate-level verification.

## Current Project Status: Week 1 (Tool Validation)
During Week 1, the primary objective was establishing and validating the EDA environment on the MCECS `deimos` cluster. 

Before modifying the Conv2D RTL, the Siemens Tessent vendor example flat flow was executed to validate licenses, libraries, and tool functionality. 
* **Stage 1 (insert_bscan_mbist):** Successfully inserted JTAG boundary scan and MBIST. Pattern simulations passed (4/4).
* **Stage 2 (insert_occ_edt_lbist):** Successfully inserted OCC, EDT compression, and LBIST (85 scan chains). Pattern simulations passed (4/4).

The raw execution logs for these validation stages are available in the `/logs` directory.

## Next Steps (Week 2)
With the tool environment validated, the project is now shifting to the target design:
1. **RTL Audit:** Identify clock domains and asynchronous resets in the `Conv2D` SystemVerilog source.
2. **Test Ports:** Add dedicated top-level test ports (`scan_en`, `scan_in`, `scan_out`, `test_mode`) to make the design scan-safe.
