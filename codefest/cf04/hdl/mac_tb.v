`timescale 1ns/1ps
module mac_tb;
    logic        clk;
    logic        rst;
    logic signed [7:0]  a;
    logic signed [7:0]  b;
    logic signed [31:0] out;

    mac dut (.clk(clk), .rst(rst), .a(a), .b(b), .out(out));

    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        $dumpfile("/tmp/mac_dump.vcd");
        $dumpvars(0, mac_tb);

        // Reset
        rst = 1; a = 0; b = 0;
        @(posedge clk); #1;
        rst = 0;

        // [a=3, b=4] for 3 cycles → expect 12, 24, 36
        a = 8'sd3; b = 8'sd4;
        @(posedge clk); #1;
        $display("Cycle1: out=%0d expected=12  %s", out, (out===32'sd12)  ?"PASS":"FAIL");
        @(posedge clk); #1;
        $display("Cycle2: out=%0d expected=24  %s", out, (out===32'sd24)  ?"PASS":"FAIL");
        @(posedge clk); #1;
        $display("Cycle3: out=%0d expected=36  %s", out, (out===32'sd36)  ?"PASS":"FAIL");

        // Assert reset
        rst = 1; a = 0; b = 0;
        @(posedge clk); #1;
        $display("Reset:  out=%0d expected=0   %s", out, (out===32'sd0)   ?"PASS":"FAIL");

        // [a=-5, b=2] for 2 cycles → expect -10, -20
        rst = 0;
        a = 8'shfb; b = 8'sd2;   // -5 in hex
        @(posedge clk); #1;
        $display("Neg1:   out=%0d expected=-10 %s", out, (out===-32'sd10) ?"PASS":"FAIL");
        @(posedge clk); #1;
        $display("Neg2:   out=%0d expected=-20 %s", out, (out===-32'sd20) ?"PASS":"FAIL");

        // Edge: a=127, b=127 → product=16129
        rst = 1; @(posedge clk); #1; rst = 0;
        a = 8'sd127; b = 8'sd127;
        @(posedge clk); #1;
        $display("Edge1:  out=%0d expected=16129  %s", out, (out===32'sd16129) ?"PASS":"FAIL");

        // Edge: a=-128, b=1 → product=-128
        rst = 1; @(posedge clk); #1; rst = 0;
        a = 8'sh80; b = 8'sd1;   // -128 in hex
        @(posedge clk); #1;
        $display("Edge2:  out=%0d expected=-128   %s", out, (out===-32'sd128)  ?"PASS":"FAIL");

        $display("\n=== Simulation Complete ===");
        $finish;
    end
endmodule
