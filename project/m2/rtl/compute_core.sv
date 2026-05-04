// =============================================================================
// compute_core.sv
// Project  : INT8 Conv2D Hardware Accelerator — ECE 510, Spring 2026
// Author   : Sai Ganesh Reddy Charian
// File     : project/m2/rtl/compute_core.sv
//
// Purpose  : Synthesizable compute core for single-layer INT8 3x3 Conv2D.
//            Adapted from CF04 conv2d_top.sv for Milestone 2.
//            Target: ResNet-18 first conv layer.
//            Config: N=1, C_in=3, H=32, W=32, C_out=8, K=3,
//            stride=1, padding=0 → H_out=30, W_out=30.
//
// Clock domain : Single clock domain (clk). No clock crossings.
//                All flip-flops clocked on posedge clk.
//
// Reset        : Active-high synchronous reset (rst).
//                All registers clear to zero on rst=1 at posedge clk.
//                Consistent across the entire module.
//
// Dataflow:
//   IDLE    : wait for start
//   LOAD    : 1-cycle transition (buffers pre-loaded via *_wr ports)
//   COMPUTE : slide 3x3 window across H_OUT x W_OUT x C_OUT positions.
//             Each cycle: MAC fires → combinational requantize → write out_buf.
//             Fully combinational requantize (no pipeline reg) so out_buf
//             is written correctly in the same cycle.
//   OUTPUT  : stream out_buf to result_out one byte per cycle
//   DONE_ST : assert done, wait for start=0
//
// Port list:
//   clk          input   1b    System clock 100 MHz
//   rst          input   1b    Active-high synchronous reset
//   start        input   1b    Host pulses high to start
//   weight_in    input   8b    Serial weight byte
//   weight_wr    input   1b    Weight buffer write enable
//   weight_addr  input   8b    Weight buffer address (0..215)
//   act_in       input   8b    Serial activation byte
//   act_wr       input   1b    Activation buffer write enable
//   act_addr     input   12b   Activation buffer address (0..3071)
//   bias_in      input   32b   Serial bias word (INT32)
//   bias_wr      input   1b    Bias buffer write enable
//   bias_addr    input   3b    Bias buffer address (0..7)
//   shift_amt    input   5b    Requantizer right-shift (0..31)
//   result_out   output  8b    INT8 output pixel in OUTPUT state
//   result_valid output  1b    High when result_out is valid
//   done         output  1b    Pulses high one cycle at OUTPUT→DONE
//   busy         output  1b    High during COMPUTE state
// =============================================================================

`timescale 1ns / 1ps

module compute_core #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32,
    parameter C_IN       = 3,
    parameter C_OUT      = 8,
    parameter H_IN       = 32,
    parameter W_IN       = 32,
    parameter K          = 3,
    parameter H_OUT      = 30,
    parameter W_OUT      = 30,
    parameter SHIFT_BITS = 8
)(
    input  logic                         clk,
    input  logic                         rst,
    input  logic                         start,

    input  logic signed [DATA_WIDTH-1:0] weight_in,
    input  logic                         weight_wr,
    input  logic [7:0]                   weight_addr,

    input  logic signed [DATA_WIDTH-1:0] act_in,
    input  logic                         act_wr,
    input  logic [11:0]                  act_addr,

    input  logic signed [ACC_WIDTH-1:0]  bias_in,
    input  logic                         bias_wr,
    input  logic [2:0]                   bias_addr,

    input  logic [4:0]                   shift_amt,

    output logic signed [DATA_WIDTH-1:0] result_out,
    output logic                         result_valid,
    output logic                         done,
    output logic                         busy
);

    // =========================================================================
    // FSM
    // =========================================================================
    localparam [2:0] IDLE    = 3'd0,
                     LOAD    = 3'd1,
                     COMPUTE = 3'd2,
                     OUTPUT  = 3'd3,
                     DONE_ST = 3'd4;

    reg [2:0] state, next_state;

    // =========================================================================
    // Buffers — all initialized to 0 so no X propagation
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] weight_buf [0:C_OUT*C_IN*K*K-1];
    logic signed [DATA_WIDTH-1:0] act_buf    [0:C_IN*H_IN*W_IN-1];
    logic signed [ACC_WIDTH-1:0]  bias_buf   [0:C_OUT-1];
    logic signed [DATA_WIDTH-1:0] out_buf    [0:C_OUT*H_OUT*W_OUT-1];

    // Buffers initialized to 0 via testbench writes before start is asserted.
    // No initial block used — not synthesizable and causes race conditions
    // with always_ff writes in simulation.

    // =========================================================================
    // Position counters
    // =========================================================================
    logic [$clog2(H_OUT)-1:0]  h_cnt;
    logic [$clog2(W_OUT)-1:0]  w_cnt;
    logic [$clog2(C_OUT)-1:0]  oc_cnt;

    logic compute_done;
    assign compute_done = (h_cnt  == H_OUT-1) &&
                          (w_cnt  == W_OUT-1) &&
                          (oc_cnt == C_OUT-1);

    // output stream counter
    logic [12:0] out_cnt;
    logic        out_done;
    assign out_done = (out_cnt == C_OUT*H_OUT*W_OUT-1);

    // =========================================================================
    // 3x3 patch wires
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] mac_a [0:K*K-1];
    logic signed [DATA_WIDTH-1:0] mac_b [0:K*K-1];

    genvar gi, gj;
    generate
        for (gi = 0; gi < K; gi++) begin : row_loop
            for (gj = 0; gj < K; gj++) begin : col_loop
                assign mac_a[gi*K+gj] =
                    act_buf[0*H_IN*W_IN + (h_cnt+gi)*W_IN + (w_cnt+gj)];
                assign mac_b[gi*K+gj] =
                    weight_buf[oc_cnt*C_IN*K*K + 0*K*K + gi*K + gj];
            end
        end
    endgenerate

    // =========================================================================
    // Combinational MAC — 9 multiply-accumulate, fully combinational
    // No pipeline register so result is available same cycle for out_buf write
    // =========================================================================
    // 9 individual products sign-extended to ACC_WIDTH — Icarus 11 compatible
    wire signed [ACC_WIDTH-1:0] p0,p1,p2,p3,p4,p5,p6,p7,p8;
    assign p0 = {{(ACC_WIDTH-DATA_WIDTH){mac_a[0][DATA_WIDTH-1]}}, mac_a[0]} *
                {{(ACC_WIDTH-DATA_WIDTH){mac_b[0][DATA_WIDTH-1]}}, mac_b[0]};
    assign p1 = {{(ACC_WIDTH-DATA_WIDTH){mac_a[1][DATA_WIDTH-1]}}, mac_a[1]} *
                {{(ACC_WIDTH-DATA_WIDTH){mac_b[1][DATA_WIDTH-1]}}, mac_b[1]};
    assign p2 = {{(ACC_WIDTH-DATA_WIDTH){mac_a[2][DATA_WIDTH-1]}}, mac_a[2]} *
                {{(ACC_WIDTH-DATA_WIDTH){mac_b[2][DATA_WIDTH-1]}}, mac_b[2]};
    assign p3 = {{(ACC_WIDTH-DATA_WIDTH){mac_a[3][DATA_WIDTH-1]}}, mac_a[3]} *
                {{(ACC_WIDTH-DATA_WIDTH){mac_b[3][DATA_WIDTH-1]}}, mac_b[3]};
    assign p4 = {{(ACC_WIDTH-DATA_WIDTH){mac_a[4][DATA_WIDTH-1]}}, mac_a[4]} *
                {{(ACC_WIDTH-DATA_WIDTH){mac_b[4][DATA_WIDTH-1]}}, mac_b[4]};
    assign p5 = {{(ACC_WIDTH-DATA_WIDTH){mac_a[5][DATA_WIDTH-1]}}, mac_a[5]} *
                {{(ACC_WIDTH-DATA_WIDTH){mac_b[5][DATA_WIDTH-1]}}, mac_b[5]};
    assign p6 = {{(ACC_WIDTH-DATA_WIDTH){mac_a[6][DATA_WIDTH-1]}}, mac_a[6]} *
                {{(ACC_WIDTH-DATA_WIDTH){mac_b[6][DATA_WIDTH-1]}}, mac_b[6]};
    assign p7 = {{(ACC_WIDTH-DATA_WIDTH){mac_a[7][DATA_WIDTH-1]}}, mac_a[7]} *
                {{(ACC_WIDTH-DATA_WIDTH){mac_b[7][DATA_WIDTH-1]}}, mac_b[7]};
    assign p8 = {{(ACC_WIDTH-DATA_WIDTH){mac_a[8][DATA_WIDTH-1]}}, mac_a[8]} *
                {{(ACC_WIDTH-DATA_WIDTH){mac_b[8][DATA_WIDTH-1]}}, mac_b[8]};

    wire signed [ACC_WIDTH-1:0] mac_sum;
    assign mac_sum = p0+p1+p2+p3+p4+p5+p6+p7+p8;

    // =========================================================================
    // Combinational requantize: add bias, right-shift, clamp
    // =========================================================================
    logic signed [ACC_WIDTH-1:0]  biased;
    logic signed [ACC_WIDTH-1:0]  shifted;
    logic signed [DATA_WIDTH-1:0] clamped;

    assign biased  = mac_sum + bias_buf[oc_cnt];
    assign shifted = biased >>> shift_amt;
    assign clamped = (shifted >  32'sd127) ?  8'sd127 :
                     (shifted < -32'sd128) ? -8'sd128 :
                                              shifted[DATA_WIDTH-1:0];

    // =========================================================================
    // FSM sequential
    // =========================================================================
    always_ff @(posedge clk) begin
        if (rst) state <= IDLE;
        else     state <= next_state;
    end

    always_comb begin
        next_state = state;
        case (state)
            IDLE:    if (start)        next_state = LOAD;
            LOAD:                      next_state = COMPUTE;
            COMPUTE: if (compute_done) next_state = OUTPUT;
            OUTPUT:  if (out_done)     next_state = DONE_ST;
            DONE_ST: if (!start)       next_state = IDLE;
            default:                   next_state = IDLE;
        endcase
    end

    // =========================================================================
    // Buffer writes from host — separate always_ff per buffer to avoid
    // write collisions when multiple enables are active in same cycle
    // =========================================================================
    always_ff @(posedge clk) begin
        if (weight_wr) weight_buf[weight_addr] <= weight_in;
    end
    always_ff @(posedge clk) begin
        if (act_wr) act_buf[act_addr] <= act_in;
    end
    always_ff @(posedge clk) begin
        if (bias_wr) bias_buf[bias_addr] <= bias_in;
    end

    // =========================================================================
    // Position counters advance in COMPUTE
    // =========================================================================
    always_ff @(posedge clk) begin
        if (rst || state != COMPUTE) begin
            h_cnt  <= '0;
            w_cnt  <= '0;
            oc_cnt <= '0;
        end else begin
            if (oc_cnt == C_OUT-1) begin
                oc_cnt <= '0;
                if (w_cnt == W_OUT-1) begin
                    w_cnt <= '0;
                    if (h_cnt < H_OUT-1) h_cnt <= h_cnt + 1;
                end else begin
                    w_cnt <= w_cnt + 1;
                end
            end else begin
                oc_cnt <= oc_cnt + 1;
            end
        end
    end

    // =========================================================================
    // Write to out_buf every cycle in COMPUTE using combinational result
    // =========================================================================
    always_ff @(posedge clk) begin
        if (state == COMPUTE)
            out_buf[oc_cnt * H_OUT * W_OUT + h_cnt * W_OUT + w_cnt] <= clamped;
    end

    // =========================================================================
    // Output counter and stream
    // =========================================================================
    always_ff @(posedge clk) begin
        if (rst || state != OUTPUT)
            out_cnt <= '0;
        else if (!out_done)
            out_cnt <= out_cnt + 1;
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            result_out   <= '0;
            result_valid <= 1'b0;
        end else if (state == OUTPUT) begin
            result_out   <= out_buf[out_cnt];
            result_valid <= 1'b1;
        end else begin
            result_out   <= '0;
            result_valid <= 1'b0;
        end
    end

    // =========================================================================
    // Status
    // =========================================================================
    always_ff @(posedge clk) begin
        if (rst) begin
            done <= 1'b0;
            busy <= 1'b0;
        end else begin
            done <= (state == OUTPUT) && (next_state == DONE_ST);
            busy <= (state == COMPUTE);
        end
    end

endmodule
