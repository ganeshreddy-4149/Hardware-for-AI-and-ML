// =============================================================================
// interface.sv
// Project  : INT8 Conv2D Hardware Accelerator — ECE 510, Spring 2026
// Author   : Sai Ganesh Reddy Charian
// File     : project/m2/rtl/interface.sv
//
// Purpose  : AXI4-Lite slave interface for the INT8 Conv2D accelerator.
//            Protocol: AXI4-Lite (ARM IHI0022E), 32-bit, 100 MHz, 0.40 GB/s.
//            Selected in M1 interface_selection.md.
//
// Clock domain : Single clock domain (clk). No clock crossings.
//                All flip-flops clocked on posedge clk.
//
// Reset        : Active-high synchronous reset (rst).
//                All outputs and registers clear to zero on rst=1.
//                Consistent across entire module.
//
// Protocol compliance:
//   - Handshake on ALL five channels (AW, W, B, AR, R)
//   - AWREADY, WREADY, ARREADY are one-cycle pulses
//   - BVALID held until BREADY; RVALID held until RREADY
//   - BRESP = 2'b00 (OKAY); RRESP = 2'b00 (OKAY)
//   - Write FSM: IDLE → AW_PHASE → W_PHASE → B_PHASE → IDLE
//     guarantees address is always latched before data is written
//   - Read FSM: IDLE → AR_PHASE → R_PHASE → IDLE
//     guarantees address is latched before data is driven
//
// Register map — address space: 0x00..0x0C (4 registers, 32-bit bus width)
//   Byte   Name     Width   Access  Description
//   0x00   CTRL     32-bit  R/W     bit[0]=start. Write 1 to begin computation.
//                                   bits[31:1] reserved, read as 0.
//   0x04   STATUS   32-bit  R only  bit[0]=done (1 when OUTPUT→DONE transition).
//                                   bit[1]=busy (1 during COMPUTE state).
//                                   bits[31:2] reserved, read as 0.
//                                   Writes to this address are ignored.
//   0x08   SHIFT    32-bit  R/W     bits[4:0]=requantizer right-shift (0..31).
//                                   bits[31:5] reserved, read as 0.
//                                   Default after reset = 8.
//   0x0C   SCRATCH  32-bit  R/W     General purpose 32-bit scratch register.
//                                   No hardware effect. Default after reset = 0.
//   other  —        32-bit  R only  Reads return 32'hDEAD_BEEF. Writes ignored.
//
// Port list:
//   clk        input  1b   System clock 100 MHz
//   rst        input  1b   Active-high synchronous reset
//   s_awaddr   input  32b  Write address
//   s_awvalid  input  1b   Master write address valid
//   s_awready  output 1b   Slave write address ready
//   s_wdata    input  32b  Write data
//   s_wstrb    input  4b   Write byte strobes
//   s_wvalid   input  1b   Master write data valid
//   s_wready   output 1b   Slave write data ready
//   s_bresp    output 2b   Write response (always OKAY = 2'b00)
//   s_bvalid   output 1b   Write response valid
//   s_bready   input  1b   Master response ready
//   s_araddr   input  32b  Read address
//   s_arvalid  input  1b   Master read address valid
//   s_arready  output 1b   Slave read address ready
//   s_rdata    output 32b  Read data
//   s_rresp    output 2b   Read response (always OKAY = 2'b00)
//   s_rvalid   output 1b   Read data valid
//   s_rready   input  1b   Master read data ready
//   core_done  input  1b   Pulse from compute_core when done
//   core_busy  input  1b   High during COMPUTE state
//   core_start output 1b   Start to compute_core
//   core_shift output 5b   Requantizer shift to compute_core
// =============================================================================

`timescale 1ns / 1ps

module axi4lite_if #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 32
)(
    input  wire                  clk,
    input  wire                  rst,

    input  wire [ADDR_WIDTH-1:0] s_awaddr,
    input  wire                  s_awvalid,
    output reg                   s_awready,

    input  wire [DATA_WIDTH-1:0] s_wdata,
    input  wire [3:0]            s_wstrb,
    input  wire                  s_wvalid,
    output reg                   s_wready,

    output reg  [1:0]            s_bresp,
    output reg                   s_bvalid,
    input  wire                  s_bready,

    input  wire [ADDR_WIDTH-1:0] s_araddr,
    input  wire                  s_arvalid,
    output reg                   s_arready,

    output reg  [DATA_WIDTH-1:0] s_rdata,
    output reg  [1:0]            s_rresp,
    output reg                   s_rvalid,
    input  wire                  s_rready,

    input  wire                  core_done,
    input  wire                  core_busy,
    output wire                  core_start,
    output wire [4:0]            core_shift
);

    // =========================================================================
    // Control registers
    // =========================================================================
    reg        reg_start;
    reg [4:0]  reg_shift;
    reg [31:0] reg_scratch;

    assign core_start = reg_start;
    assign core_shift = reg_shift;

    // =========================================================================
    // Write FSM: IDLE → AW_PHASE → W_PHASE → B_PHASE → IDLE
    // AW_PHASE: wait for AWVALID, latch address, pulse AWREADY
    // W_PHASE : wait for WVALID,  write register, pulse WREADY
    // B_PHASE : assert BVALID, wait for BREADY
    // This guarantees address is always captured before data is written.
    // =========================================================================
    localparam W_IDLE    = 2'd0;
    localparam W_AW      = 2'd1;
    localparam W_DATA    = 2'd2;
    localparam W_RESP    = 2'd3;

    reg [1:0]  wstate;
    reg [3:0]  wr_addr_lat;

    always @(posedge clk) begin
        if (rst) begin
            wstate      <= W_IDLE;
            wr_addr_lat <= 4'h0;
            s_awready   <= 1'b0;
            s_wready    <= 1'b0;
            s_bvalid    <= 1'b0;
            s_bresp     <= 2'b00;
            reg_start   <= 1'b0;
            reg_shift   <= 5'd8;
            reg_scratch <= 32'h0;
        end else begin
            // defaults — de-assert pulses
            s_awready <= 1'b0;
            s_wready  <= 1'b0;

            case (wstate)
                W_IDLE: begin
                    if (s_awvalid) begin
                        // accept address immediately
                        s_awready   <= 1'b1;
                        wr_addr_lat <= s_awaddr[3:0];
                        wstate      <= W_DATA;
                    end
                end

                W_DATA: begin
                    if (s_wvalid) begin
                        // write register with guaranteed-valid address
                        s_wready <= 1'b1;
                        case (wr_addr_lat)
                            4'h0: reg_start   <= s_wdata[0];
                            4'h8: reg_shift   <= s_wdata[4:0];
                            4'hC: reg_scratch <= s_wdata;
                            default: ;
                        endcase
                        s_bvalid <= 1'b1;
                        s_bresp  <= 2'b00;
                        wstate   <= W_RESP;
                    end
                end

                W_RESP: begin
                    if (s_bready) begin
                        s_bvalid <= 1'b0;
                        wstate   <= W_IDLE;
                    end
                end

                default: wstate <= W_IDLE;
            endcase
        end
    end

    // =========================================================================
    // Read FSM: IDLE → AR_PHASE → R_PHASE → IDLE
    // AR_PHASE: wait for ARVALID, latch address, pulse ARREADY
    // R_PHASE : assert RVALID with RDATA, wait for RREADY
    // =========================================================================
    localparam R_IDLE = 2'd0;
    localparam R_AR   = 2'd1;
    localparam R_DATA = 2'd2;

    reg [1:0] rstate;
    reg [3:0] rd_addr_lat;

    always @(posedge clk) begin
        if (rst) begin
            rstate      <= R_IDLE;
            rd_addr_lat <= 4'h0;
            s_arready   <= 1'b0;
            s_rvalid    <= 1'b0;
            s_rdata     <= 32'h0;
            s_rresp     <= 2'b00;
        end else begin
            s_arready <= 1'b0;

            case (rstate)
                R_IDLE: begin
                    if (s_arvalid) begin
                        s_arready   <= 1'b1;
                        rd_addr_lat <= s_araddr[3:0];
                        rstate      <= R_DATA;
                    end
                end

                R_DATA: begin
                    if (!s_rvalid) begin
                        // drive read data one cycle after ARREADY
                        s_rvalid <= 1'b1;
                        s_rresp  <= 2'b00;
                        case (rd_addr_lat)
                            4'h0: s_rdata <= {31'h0, reg_start};
                            4'h4: s_rdata <= {30'h0, core_busy, core_done};
                            4'h8: s_rdata <= {27'h0, reg_shift};
                            4'hC: s_rdata <= reg_scratch;
                            default: s_rdata <= 32'hDEAD_BEEF;
                        endcase
                    end else if (s_rvalid && s_rready) begin
                        s_rvalid <= 1'b0;
                        rstate   <= R_IDLE;
                    end
                end

                default: rstate <= R_IDLE;
            endcase
        end
    end

endmodule
