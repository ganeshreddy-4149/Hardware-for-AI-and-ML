// =============================================================================
// Module  : crossbar_mac
// Project : ECE 410/510 — Hardware for AI & ML, Codefest 6
// Author  : LLM-generated (CLLM task)
// Date    : Spring 2026
// Fix     : Flattened unpacked output array to packed for Icarus compatibility
// Compatibility: Icarus Verilog (iverilog -g2012)
// =============================================================================

module crossbar_mac (
    input  logic                clk,
    input  logic                rst_n,

    // Inputs: flattened 32-bit bus [in3|in2|in1|in0], each 8-bit signed
    input  logic [31:0]         in_flat,      // {in[3],in[2],in[1],in[0]}

    // Weight load
    input  logic                load_en,
    input  logic [15:0]         weight_flat,  // {w[3],w[2],w[1],w[0]} each 4 bits

    // Outputs: flattened 128-bit bus [out3|out2|out1|out0], each 32-bit signed
    output logic [127:0]        out_flat      // {out[3],out[2],out[1],out[0]}
);

    // Unpack inputs internally
    wire signed [7:0]  in0 = in_flat[7:0];
    wire signed [7:0]  in1 = in_flat[15:8];
    wire signed [7:0]  in2 = in_flat[23:16];
    wire signed [7:0]  in3 = in_flat[31:24];

    // Sign extend
    wire signed [31:0] ex0 = {{24{in0[7]}}, in0};
    wire signed [31:0] ex1 = {{24{in1[7]}}, in1};
    wire signed [31:0] ex2 = {{24{in2[7]}}, in2};
    wire signed [31:0] ex3 = {{24{in3[7]}}, in3};

    // Weight registers [3:0] per row, bit j = weight for col j
    logic [3:0] weight [0:3];

    // Unpack weight_flat
    wire [3:0] w0 = weight_flat[3:0];
    wire [3:0] w1 = weight_flat[7:4];
    wire [3:0] w2 = weight_flat[11:8];
    wire [3:0] w3 = weight_flat[15:12];

    // Weight load
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            weight[0] <= 4'b1111;
            weight[1] <= 4'b1111;
            weight[2] <= 4'b1111;
            weight[3] <= 4'b1111;
        end else if (load_en) begin
            weight[0] <= w0;
            weight[1] <= w1;
            weight[2] <= w2;
            weight[3] <= w3;
        end
    end

    // MAC
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            out_flat <= 128'd0;
        end else begin
            // col 0
            out_flat[31:0]   <= (weight[0][0] ? ex0 : -ex0)
                              + (weight[1][0] ? ex1 : -ex1)
                              + (weight[2][0] ? ex2 : -ex2)
                              + (weight[3][0] ? ex3 : -ex3);
            // col 1
            out_flat[63:32]  <= (weight[0][1] ? ex0 : -ex0)
                              + (weight[1][1] ? ex1 : -ex1)
                              + (weight[2][1] ? ex2 : -ex2)
                              + (weight[3][1] ? ex3 : -ex3);
            // col 2
            out_flat[95:64]  <= (weight[0][2] ? ex0 : -ex0)
                              + (weight[1][2] ? ex1 : -ex1)
                              + (weight[2][2] ? ex2 : -ex2)
                              + (weight[3][2] ? ex3 : -ex3);
            // col 3
            out_flat[127:96] <= (weight[0][3] ? ex0 : -ex0)
                              + (weight[1][3] ? ex1 : -ex1)
                              + (weight[2][3] ? ex2 : -ex2)
                              + (weight[3][3] ? ex3 : -ex3);
        end
    end

endmodule
