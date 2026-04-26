// =============================================================================
// conv2d_top.sv
// Project  : INT8 Conv2D Hardware Accelerator — ECE 510, Spring 2026
// Author   : Sai Ganesh Reddy Charian
// Purpose  : Top-level compute core for single-layer INT8 3×3 Conv2D
//
// Configuration (fixed for this project):
//   N=1, C_in=3, H=32, W=32, C_out=8, K=3, stride=1, padding=0
//   H_out=30, W_out=30
//   Input/Weights: INT8, Accumulator: INT32, Output: INT8 (after requantize)
//
// Interface: AXI4-Lite slave, 32-bit @ 100 MHz → 0.40 GB/s
//   Justified by: AI=12.10 FLOP/byte, target=3.20 GFLOP/s
//   Required BW = 3.20/12.10 = 0.264 GB/s < 0.40 GB/s (34% headroom)
//
// Sub-modules instantiated (stubs — to be fully implemented for M2):
//   mac_array    : 9 parallel INT8 MACs feeding INT32 adder tree
//   line_buffer  : 3-row sliding window extraction (reuse rows across slides)
//   requantizer  : INT32 → INT8 via right-shift + clamp to [-128, 127]
//   axi4_lite_slave : memory-mapped control registers
//
// FSM states: IDLE → LOAD → COMPUTE → OUTPUT → DONE
// =============================================================================

`timescale 1ns / 1ps

// -----------------------------------------------------------------------------
// Parameters
// -----------------------------------------------------------------------------
`define C_IN    3
`define C_OUT   8
`define H_IN    32
`define W_IN    32
`define K       3
`define H_OUT   30    // H_IN - K + 1
`define W_OUT   30    // W_IN - K + 1

// -----------------------------------------------------------------------------
// Top-level module
// -----------------------------------------------------------------------------
module conv2d_top #(
    parameter DATA_WIDTH   = 8,          // INT8 input/weight width
    parameter ACC_WIDTH    = 32,         // INT32 accumulator width
    parameter C_IN         = 3,          // Input channels
    parameter C_OUT        = 8,          // Output channels
    parameter H_IN         = 32,         // Input height
    parameter W_IN         = 32,         // Input width
    parameter K            = 3,          // Kernel size (K×K)
    parameter H_OUT        = 30,         // Output height = H_IN - K + 1
    parameter W_OUT        = 30,         // Output width  = W_IN - K + 1
    parameter SHIFT_BITS   = 8           // Requantizer right-shift (tunable)
)(
    // -------------------------------------------------------------------------
    // Global signals
    // -------------------------------------------------------------------------
    input  logic                         clk,
    input  logic                         rst,        // active-high synchronous

    // -------------------------------------------------------------------------
    // AXI4-Lite Slave Interface (control plane)
    // 32-bit data bus, 100 MHz → 0.40 GB/s rated
    // -------------------------------------------------------------------------
    // Write address channel
    input  logic [31:0]                  s_axil_awaddr,
    input  logic                         s_axil_awvalid,
    output logic                         s_axil_awready,
    // Write data channel
    input  logic [31:0]                  s_axil_wdata,
    input  logic [3:0]                   s_axil_wstrb,
    input  logic                         s_axil_wvalid,
    output logic                         s_axil_wready,
    // Write response channel
    output logic [1:0]                   s_axil_bresp,
    output logic                         s_axil_bvalid,
    input  logic                         s_axil_bready,
    // Read address channel
    input  logic [31:0]                  s_axil_araddr,
    input  logic                         s_axil_arvalid,
    output logic                         s_axil_arready,
    // Read data channel
    output logic [31:0]                  s_axil_rdata,
    output logic [1:0]                   s_axil_rresp,
    output logic                         s_axil_rvalid,
    input  logic                         s_axil_rready,

    // -------------------------------------------------------------------------
    // Status outputs (visible to host via AXI registers)
    // -------------------------------------------------------------------------
    output logic                         done,       // pulsed when OUTPUT→DONE
    output logic                         busy        // high during COMPUTE
);

    // =========================================================================
    // Internal control registers (written by AXI4-Lite slave)
    // =========================================================================
    logic        reg_start;          // host writes 1 to start
    logic [31:0] reg_scale_shift;    // requantizer shift amount

    // =========================================================================
    // FSM
    // =========================================================================
    typedef enum logic [2:0] {
        IDLE    = 3'd0,
        LOAD    = 3'd1,   // DMA input + weights from AXI into on-chip SRAM
        COMPUTE = 3'd2,   // slide 3×3 window, run MAC array
        OUTPUT  = 3'd3,   // requantize INT32 → INT8, write to output buffer
        DONE    = 3'd4    // assert done, wait for host to de-assert start
    } state_t;

    state_t state, next_state;

    // =========================================================================
    // Output position counters (row h, col w, output channel oc)
    // =========================================================================
    logic [$clog2(H_OUT)-1:0]   h_cnt;   // 0..29
    logic [$clog2(W_OUT)-1:0]   w_cnt;   // 0..29
    logic [$clog2(C_OUT)-1:0]   oc_cnt;  // 0..7

    logic compute_done;   // all H_OUT × W_OUT × C_OUT outputs computed

    assign compute_done = (h_cnt  == H_OUT-1) &&
                          (w_cnt  == W_OUT-1) &&
                          (oc_cnt == C_OUT-1);

    // =========================================================================
    // On-chip SRAM — flat arrays (synthesizer maps to SRAM macros via OpenLane)
    // =========================================================================
    // Weight buffer:  C_OUT × C_IN × K × K = 8×3×3×3 = 216 bytes
    logic signed [DATA_WIDTH-1:0] weight_buf [0:C_OUT*C_IN*K*K-1];

    // Input activation buffer: C_IN × H_IN × W_IN = 3×32×32 = 3072 bytes
    logic signed [DATA_WIDTH-1:0] act_buf    [0:C_IN*H_IN*W_IN-1];

    // Bias buffer: C_OUT × 4 bytes = 32 bytes (INT32)
    logic signed [ACC_WIDTH-1:0]  bias_buf   [0:C_OUT-1];

    // Output buffer: C_OUT × H_OUT × W_OUT = 8×30×30 = 7200 elements (INT8)
    logic signed [DATA_WIDTH-1:0] out_buf    [0:C_OUT*H_OUT*W_OUT-1];

    // =========================================================================
    // MAC array wires
    // =========================================================================
    // 9 input pairs for 3×3 patch × 1 output channel
    logic signed [DATA_WIDTH-1:0] mac_a [0:K*K-1];  // activations
    logic signed [DATA_WIDTH-1:0] mac_b [0:K*K-1];  // weights
    logic signed [ACC_WIDTH-1:0]  mac_out;           // summed partial product
    logic                         mac_en;

    // =========================================================================
    // Requantizer wires
    // =========================================================================
    logic signed [ACC_WIDTH-1:0]  req_in;
    logic signed [DATA_WIDTH-1:0] req_out;
    logic                         req_valid;

    // =========================================================================
    // FSM — sequential
    // =========================================================================
    always_ff @(posedge clk) begin
        if (rst)
            state <= IDLE;
        else
            state <= next_state;
    end

    // =========================================================================
    // FSM — combinational next-state logic
    // =========================================================================
    always_comb begin
        next_state = state;
        case (state)
            IDLE:    if (reg_start)   next_state = LOAD;
            LOAD:                     next_state = COMPUTE;  // 1-cycle stub; expand for DMA
            COMPUTE: if (compute_done) next_state = OUTPUT;
            OUTPUT:                   next_state = DONE;
            DONE:    if (!reg_start)  next_state = IDLE;
            default:                  next_state = IDLE;
        endcase
    end

    // =========================================================================
    // Output position counter — advance every cycle in COMPUTE
    // =========================================================================
    always_ff @(posedge clk) begin
        if (rst || state != COMPUTE) begin
            h_cnt  <= '0;
            w_cnt  <= '0;
            oc_cnt <= '0;
        end else begin
            // innermost: oc, then w, then h
            if (oc_cnt == C_OUT-1) begin
                oc_cnt <= '0;
                if (w_cnt == W_OUT-1) begin
                    w_cnt <= '0;
                    if (h_cnt < H_OUT-1)
                        h_cnt <= h_cnt + 1;
                end else begin
                    w_cnt <= w_cnt + 1;
                end
            end else begin
                oc_cnt <= oc_cnt + 1;
            end
        end
    end

    // =========================================================================
    // Status signals
    // =========================================================================
    always_ff @(posedge clk) begin
        if (rst) begin
            done <= 1'b0;
            busy <= 1'b0;
        end else begin
            done <= (state == OUTPUT) && (next_state == DONE);
            busy <= (state == COMPUTE);
        end
    end

    // =========================================================================
    // AXI4-Lite slave stub
    // (Full implementation: memory-mapped registers at fixed offsets)
    //   0x00 — control  (bit 0 = start)
    //   0x04 — status   (bit 0 = done, bit 1 = busy)
    //   0x08 — scale_shift
    // =========================================================================
    always_ff @(posedge clk) begin
        if (rst) begin
            reg_start       <= 1'b0;
            reg_scale_shift <= SHIFT_BITS;
            // de-assert handshake signals
            s_axil_awready  <= 1'b0;
            s_axil_wready   <= 1'b0;
            s_axil_bvalid   <= 1'b0;
            s_axil_bresp    <= 2'b00;
            s_axil_arready  <= 1'b0;
            s_axil_rvalid   <= 1'b0;
            s_axil_rdata    <= 32'h0;
            s_axil_rresp    <= 2'b00;
        end else begin
            // ---- Write path (simplified single-cycle) ----
            s_axil_awready <= s_axil_awvalid;
            s_axil_wready  <= s_axil_wvalid;

            if (s_axil_wvalid && s_axil_wready) begin
                case (s_axil_awaddr[3:0])
                    4'h0: reg_start       <= s_axil_wdata[0];
                    4'h8: reg_scale_shift <= s_axil_wdata;
                    default: ;
                endcase
                s_axil_bvalid <= 1'b1;
            end else if (s_axil_bready) begin
                s_axil_bvalid <= 1'b0;
            end

            // ---- Read path (simplified single-cycle) ----
            s_axil_arready <= s_axil_arvalid;
            if (s_axil_arvalid && s_axil_arready) begin
                s_axil_rvalid <= 1'b1;
                case (s_axil_araddr[3:0])
                    4'h4:    s_axil_rdata <= {30'h0, busy, done};
                    default: s_axil_rdata <= 32'hDEAD_BEEF;
                endcase
            end else if (s_axil_rready) begin
                s_axil_rvalid <= 1'b0;
            end
        end
    end

    // =========================================================================
    // MAC array instantiation stub
    // (Replace with mac_array.sv for M2 — 9 parallel MACs + adder tree)
    // =========================================================================
    mac_array #(
        .DATA_WIDTH (DATA_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH),
        .K          (K)
    ) u_mac_array (
        .clk    (clk),
        .rst    (rst),
        .en     (mac_en),
        .a      (mac_a),
        .b      (mac_b),
        .acc_out(mac_out)
    );

    assign mac_en = (state == COMPUTE);

    // Window extraction: load 3×3 patch from act_buf + matching weights
    // Simplified combinational index — line buffer to be added for timing
    genvar gi, gj;
    generate
        for (gi = 0; gi < K; gi++) begin : row_loop
            for (gj = 0; gj < K; gj++) begin : col_loop
                assign mac_a[gi*K+gj] =
                    act_buf[(0)*H_IN*W_IN + (h_cnt+gi)*W_IN + (w_cnt+gj)];
                    // Note: C_in loop handled by accumulating oc_cnt cycles;
                    // full multi-channel accumulation requires C_in FSM stage.
                assign mac_b[gi*K+gj] =
                    weight_buf[oc_cnt*C_IN*K*K + 0*K*K + gi*K + gj];
            end
        end
    endgenerate

    // =========================================================================
    // Requantizer stub
    // INT32 → INT8: right-shift by SHIFT_BITS, clamp to [-128, 127]
    // =========================================================================
    requantizer #(
        .ACC_WIDTH  (ACC_WIDTH),
        .DATA_WIDTH (DATA_WIDTH)
    ) u_req (
        .clk       (clk),
        .rst       (rst),
        .in_data   (req_in),
        .shift     (reg_scale_shift[4:0]),
        .valid_in  (req_valid),
        .out_data  (req_out)
    );

    assign req_in    = mac_out + bias_buf[oc_cnt];
    assign req_valid = (state == OUTPUT);

endmodule


// =============================================================================
// mac_array.sv — 9 parallel INT8 MACs + adder tree (stub)
// Full implementation for M2: project/m2/rtl/mac_array.sv
// =============================================================================
module mac_array #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32,
    parameter K          = 3       // K*K = 9 MACs
)(
    input  logic                              clk,
    input  logic                              rst,
    input  logic                              en,
    input  logic signed [DATA_WIDTH-1:0]      a [0:K*K-1],
    input  logic signed [DATA_WIDTH-1:0]      b [0:K*K-1],
    output logic signed [ACC_WIDTH-1:0]       acc_out
);
    // 9 partial products, sign-extended to ACC_WIDTH
    logic signed [ACC_WIDTH-1:0] products [0:K*K-1];

    genvar i;
    generate
        for (i = 0; i < K*K; i++) begin : mac_gen
            assign products[i] = ACC_WIDTH'(signed'(a[i])) *
                                  ACC_WIDTH'(signed'(b[i]));
        end
    endgenerate

    // Adder tree (flat for now — synthesizer will optimize)
    logic signed [ACC_WIDTH-1:0] sum;
    always_comb begin
        sum = '0;
        for (int j = 0; j < K*K; j++)
            sum = sum + products[j];
    end

    // Register output
    always_ff @(posedge clk) begin
        if (rst)
            acc_out <= '0;
        else if (en)
            acc_out <= sum;
    end
endmodule


// =============================================================================
// requantizer.sv — INT32 → INT8 right-shift + clamp (stub)
// =============================================================================
module requantizer #(
    parameter ACC_WIDTH  = 32,
    parameter DATA_WIDTH = 8
)(
    input  logic                              clk,
    input  logic                              rst,
    input  logic signed [ACC_WIDTH-1:0]       in_data,
    input  logic [4:0]                        shift,
    input  logic                              valid_in,
    output logic signed [DATA_WIDTH-1:0]      out_data
);
    logic signed [ACC_WIDTH-1:0] shifted;
    assign shifted = in_data >>> shift;   // arithmetic right shift

    // Clamp to INT8 range [-128, 127]
    always_ff @(posedge clk) begin
        if (rst)
            out_data <= '0;
        else if (valid_in) begin
            if (shifted > 127)
                out_data <= 8'sd127;
            else if (shifted < -128)
                out_data <= -8'sd128;
            else
                out_data <= shifted[DATA_WIDTH-1:0];
        end
    end
endmodule
