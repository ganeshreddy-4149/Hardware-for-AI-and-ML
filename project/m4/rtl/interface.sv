// ============================================================================
// Project  : INT8 Conv2D Hardware Accelerator
// Module   : interface — AXI4-Lite Register Map Reference
// Course   : ECE 410/510 — Hardware for AI/ML, Spring 2026
// Author   : Sai Ganesh Reddy Charian
//
// NOTE: The AXI4-Lite slave logic is integrated inside conv2d_top in top.sv.
// This file documents the register map as a standalone reference.
// It is not compiled into synthesis — top.sv and compute_core.sv are the
// synthesis sources.
//
// AXI4-Lite Register Map:
//   Offset 0x000  W  control      bit[0] = start
//   Offset 0x004  R  status       bit[0] = done, bit[1] = busy
//   Offset 0x008  W  scale_shift  5-bit right-shift for requantizer
//   Offset 0x100 + i*4  W  act_buf[i]     i=0..24  INT8 5x5 activation
//   Offset 0x200 + i*4  W  weight_buf[i]  i=0..8   INT8 3x3 kernel
//   Offset 0x300        W  bias_buf[0]    INT32 bias
//   Offset 0x400        R  out_buf[0]     INT8 result sign-extended to 32b
//
// Write protocol:
//   Assert awvalid+awaddr and wvalid+wdata in same cycle.
//   awready/wready registered next cycle. bvalid follows, cleared on bready.
//
// Read protocol:
//   Assert arvalid+araddr. arready registered next cycle.
//   rvalid asserted, rdata held until rready seen.
// ============================================================================
`timescale 1ns/1ps

// Reference module — not instantiated in synthesis
module axil_regmap_ref (
  input  wire        clk, rst,
  input  wire [31:0] s_axil_awaddr,
  input  wire        s_axil_awvalid,
  output wire        s_axil_awready,
  input  wire [31:0] s_axil_wdata,
  input  wire [3:0]  s_axil_wstrb,
  input  wire        s_axil_wvalid,
  output wire        s_axil_wready,
  output wire [1:0]  s_axil_bresp,
  output wire        s_axil_bvalid,
  input  wire        s_axil_bready,
  input  wire [31:0] s_axil_araddr,
  input  wire        s_axil_arvalid,
  output wire        s_axil_arready,
  output wire [31:0] s_axil_rdata,
  output wire [1:0]  s_axil_rresp,
  output wire        s_axil_rvalid,
  input  wire        s_axil_rready
);
  assign s_axil_awready = 1'b0;
  assign s_axil_wready  = 1'b0;
  assign s_axil_bresp   = 2'b00;
  assign s_axil_bvalid  = 1'b0;
  assign s_axil_arready = 1'b0;
  assign s_axil_rdata   = 32'h0;
  assign s_axil_rresp   = 2'b00;
  assign s_axil_rvalid  = 1'b0;
endmodule
