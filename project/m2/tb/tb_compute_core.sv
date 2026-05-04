// =============================================================================
// tb_compute_core.sv
// Project : INT8 Conv2D Accelerator — ECE 510 Spring 2026
// Author  : Sai Ganesh Reddy Charian
// File    : project/m2/tb/tb_compute_core.sv
//
// Purpose : Testbench for compute_core.sv
//           Loads buffers via serial *_wr interface (one write per clock).
//           Representative 3x3 Conv2D test — not all-zeros.
//           Reference (Python): patch=[[1..9]], kernel=Sobel-X → result=-6
//           Prints PASS or FAIL. Dumps VCD for GTKWave.
// =============================================================================
`timescale 1ns / 1ps

module tb_compute_core;

    localparam H_IN=32, W_IN=32, C_IN=3, C_OUT=8, K=3, H_OUT=30, W_OUT=30;

    logic clk, rst, start;
    logic signed [7:0]  weight_in;  logic weight_wr;  logic [7:0]  weight_addr;
    logic signed [7:0]  act_in;     logic act_wr;     logic [11:0] act_addr;
    logic signed [31:0] bias_in;    logic bias_wr;    logic [2:0]  bias_addr;
    logic [4:0] shift_amt;
    logic signed [7:0] result_out;
    logic result_valid, done, busy;

    compute_core #(
        .DATA_WIDTH(8),.ACC_WIDTH(32),.C_IN(3),.C_OUT(8),
        .H_IN(32),.W_IN(32),.K(3),.H_OUT(30),.W_OUT(30),.SHIFT_BITS(0)
    ) dut (
        .clk(clk),.rst(rst),.start(start),
        .weight_in(weight_in),.weight_wr(weight_wr),.weight_addr(weight_addr),
        .act_in(act_in),.act_wr(act_wr),.act_addr(act_addr),
        .bias_in(bias_in),.bias_wr(bias_wr),.bias_addr(bias_addr),
        .shift_amt(shift_amt),
        .result_out(result_out),.result_valid(result_valid),
        .done(done),.busy(busy)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        $dumpfile("compute_core.vcd");
        $dumpvars(0, tb_compute_core);
    end

    localparam signed [7:0] EXPECTED = -8'sd6;

    // task: write one value to act_buf
    task write_act;
        input [11:0] addr;
        input signed [7:0] val;
        begin
            @(posedge clk);
            act_wr = 1; act_addr = addr; act_in = val;
            @(posedge clk);
            act_wr = 0;
        end
    endtask

    // task: write one value to weight_buf
    task write_weight;
        input [7:0] addr;
        input signed [7:0] val;
        begin
            @(posedge clk);
            weight_wr = 1; weight_addr = addr; weight_in = val;
            @(posedge clk);
            weight_wr = 0;
        end
    endtask

    // task: write one value to bias_buf
    task write_bias;
        input [2:0] addr;
        input signed [31:0] val;
        begin
            @(posedge clk);
            bias_wr = 1; bias_addr = addr; bias_in = val;
            @(posedge clk);
            bias_wr = 0;
        end
    endtask

    integer fail_count;
    logic signed [7:0] captured;

    initial begin
        rst=1; start=0; weight_wr=0; act_wr=0; bias_wr=0;
        weight_in=0; weight_addr=0; act_in=0; act_addr=0;
        bias_in=0; bias_addr=0; shift_amt=0;
        fail_count=0; captured=0;

        repeat(4) @(posedge clk);
        rst = 0;
        repeat(2) @(posedge clk);

        // =====================================================================
        // Load activations: channel 0, 3x3 patch at top-left
        // act_buf[ch*H_IN*W_IN + row*W_IN + col]
        // =====================================================================
        write_act(12'd0,   8'sd1);   // ch0, row0, col0
        write_act(12'd1,   8'sd2);   // ch0, row0, col1
        write_act(12'd2,   8'sd3);   // ch0, row0, col2
        write_act(12'd32,  8'sd4);   // ch0, row1, col0
        write_act(12'd33,  8'sd5);   // ch0, row1, col1
        write_act(12'd34,  8'sd6);   // ch0, row1, col2
        write_act(12'd64,  8'sd7);   // ch0, row2, col0
        write_act(12'd65,  8'sd8);   // ch0, row2, col1
        write_act(12'd66,  8'sd9);   // ch0, row2, col2

        // =====================================================================
        // Load weights: oc=0, ic=0, Sobel-X [[1,0,-1],[1,0,-1],[1,0,-1]]
        // weight_buf[oc*C_IN*K*K + ic*K*K + row*K + col] = [0..8] for oc0,ic0
        // =====================================================================
        write_weight(8'd0,  8'sd1);
        write_weight(8'd1,  8'sd0);
        write_weight(8'd2, -8'sd1);
        write_weight(8'd3,  8'sd1);
        write_weight(8'd4,  8'sd0);
        write_weight(8'd5, -8'sd1);
        write_weight(8'd6,  8'sd1);
        write_weight(8'd7,  8'sd0);
        write_weight(8'd8, -8'sd1);

        // =====================================================================
        // Load bias = 0 for oc=0
        // =====================================================================
        write_bias(3'd0, 32'sd0);

        repeat(4) @(posedge clk);

        // verify buffers
        $display("=== Buffer verify ===");
        $display("act [0,1,2]   = %0d %0d %0d", dut.act_buf[0],dut.act_buf[1],dut.act_buf[2]);
        $display("act [32,33,34]= %0d %0d %0d", dut.act_buf[32],dut.act_buf[33],dut.act_buf[34]);
        $display("act [64,65,66]= %0d %0d %0d", dut.act_buf[64],dut.act_buf[65],dut.act_buf[66]);
        $display("wt  [0..8]    = %0d %0d %0d %0d %0d %0d %0d %0d %0d",
            dut.weight_buf[0],dut.weight_buf[1],dut.weight_buf[2],
            dut.weight_buf[3],dut.weight_buf[4],dut.weight_buf[5],
            dut.weight_buf[6],dut.weight_buf[7],dut.weight_buf[8]);
        $display("mac_sum at h=0,w=0,oc=0 = %0d  (expected -6)", dut.mac_sum);

        // =====================================================================
        // Start compute
        // =====================================================================
        shift_amt = 5'd0;
        @(posedge clk);
        start = 1;
        @(posedge clk);

        // wait for OUTPUT
        wait(result_valid == 1);
        @(posedge clk);
        captured = result_out;
        start = 0;

        @(posedge clk);
        $display("=== Result ===");
        if (captured === EXPECTED)
            $display("CHECK: result_out[oc=0,h=0,w=0]=%0d expected=%0d  --> PASS",
                captured, EXPECTED);
        else begin
            $display("CHECK: result_out[oc=0,h=0,w=0]=%0d expected=%0d  --> FAIL",
                captured, EXPECTED);
            fail_count = fail_count + 1;
        end

        #20;
        if (fail_count == 0)
            $display("PASS");
        else
            $display("FAIL");
        #10; $finish;
    end

    initial begin #50_000_000; $display("TIMEOUT"); $display("FAIL"); $finish; end

endmodule
