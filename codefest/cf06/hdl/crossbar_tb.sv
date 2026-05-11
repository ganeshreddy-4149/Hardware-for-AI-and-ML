// =============================================================================
// Module  : crossbar_tb
// Project : ECE 410/510 — Hardware for AI & ML, Codefest 6
// Author  : Sai Ganesh Reddy Charian
// Date    : Spring 2026
// Version : v11-safe (Icarus Verilog 11.0 compatible)
//
// Testbench for crossbar_mac.sv
// Weights: [[1,-1,1,-1],[1,1,-1,-1],[-1,1,1,-1],[-1,-1,-1,1]]
// Inputs:  [10, 20, 30, 40]
//
// Hand-calculated expected outputs:
//   out[0] = (+1x10)+(+1x20)+(-1x30)+(-1x40) = 10+20-30-40 = -40
//   out[1] = (-1x10)+(+1x20)+(+1x30)+(-1x40) = -10+20+30-40 = 0
//   out[2] = (+1x10)+(-1x20)+(+1x30)+(-1x40) = 10-20+30-40 = -20
//   out[3] = (-1x10)+(-1x20)+(-1x30)+(+1x40) = -10-20-30+40 = -20
// =============================================================================

module crossbar_tb;

    // -------------------------------------------------------------------------
    // Signal declarations
    // -------------------------------------------------------------------------
    reg         clk;
    reg         rst_n;
    reg         load_en;
    reg  [31:0] in_flat;
    reg  [15:0] weight_flat;
    wire [127:0] out_flat;

    // -------------------------------------------------------------------------
    // Extract individual outputs
    // -------------------------------------------------------------------------
    wire signed [31:0] out0 = out_flat[31:0];
    wire signed [31:0] out1 = out_flat[63:32];
    wire signed [31:0] out2 = out_flat[95:64];
    wire signed [31:0] out3 = out_flat[127:96];

    // -------------------------------------------------------------------------
    // Instantiate DUT
    // -------------------------------------------------------------------------
    crossbar_mac dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_flat     (in_flat),
        .load_en     (load_en),
        .weight_flat (weight_flat),
        .out_flat    (out_flat)
    );

    // -------------------------------------------------------------------------
    // Clock: 10ns period
    // -------------------------------------------------------------------------
    initial clk = 0;
    always #5 clk = ~clk;

    // -------------------------------------------------------------------------
    // Helper task: wait N rising edges
    // Using "integer" instead of "int" for v11 compatibility
    // -------------------------------------------------------------------------
    task tick;
        input integer n;
        repeat(n) @(posedge clk);
        #1;
    endtask

    // -------------------------------------------------------------------------
    // Main test
    // -------------------------------------------------------------------------
    integer pass_count;
    integer fail_count;

    initial begin
        clk        = 0;
        rst_n      = 0;
        load_en    = 0;
        in_flat    = 32'd0;
        weight_flat= 16'd0;
        pass_count = 0;
        fail_count = 0;

        $display("==============================================================");
        $display(" ECE 410/510 Codefest 6 - crossbar_mac Testbench");
        $display(" Weights: [[1,-1,1,-1],[1,1,-1,-1],[-1,1,1,-1],[-1,-1,-1,1]]");
        $display(" Inputs : [10, 20, 30, 40]");
        $display("==============================================================");

        // ------------------------------------------------------------------
        // PHASE 1: Reset for 3 cycles
        // ------------------------------------------------------------------
        $display("\n[Phase 1] Applying reset...");
        tick(3);
        rst_n = 1;
        tick(1);
        $display("  Reset released. rst_n = 1");

        // ------------------------------------------------------------------
        // PHASE 2: Load weights
        // weight_flat = {w3,w2,w1,w0} = {4'b1000,4'b0110,4'b0011,4'b0101}
        //            = 16'h8635
        // row0: [+1,-1,+1,-1] => 4'b0101
        // row1: [+1,+1,-1,-1] => 4'b0011
        // row2: [-1,+1,+1,-1] => 4'b0110
        // row3: [-1,-1,-1,+1] => 4'b1000
        // ------------------------------------------------------------------
        $display("\n[Phase 2] Loading weights...");
        weight_flat = 16'h8635;
        load_en     = 1;
        tick(1);
        load_en     = 0;
        $display("  weight_flat = 16'h8635 loaded");
        $display("  row0 = 4'b0101 => [+1,-1,+1,-1]");
        $display("  row1 = 4'b0011 => [+1,+1,-1,-1]");
        $display("  row2 = 4'b0110 => [-1,+1,+1,-1]");
        $display("  row3 = 4'b1000 => [-1,-1,-1,+1]");

        // ------------------------------------------------------------------
        // PHASE 3: Apply inputs
        // in_flat = {in[3],in[2],in[1],in[0]} = {40,30,20,10}
        // ------------------------------------------------------------------
        $display("\n[Phase 3] Applying inputs [10, 20, 30, 40]...");
        in_flat = {8'd40, 8'd30, 8'd20, 8'd10};
        tick(1);
        $display("  in[0]=10 in[1]=20 in[2]=30 in[3]=40 applied");

        // ------------------------------------------------------------------
        // PHASE 4: Verify outputs
        // ------------------------------------------------------------------
        $display("\n[Phase 4] Checking outputs...");
        $display("  ----------------------------------------");
        $display("  Output  | Expected | Got    | Status");
        $display("  ----------------------------------------");

        if (out0 === -32'sd40) begin
            $display("  out[0]  |   -40    | %4d   | PASS", out0);
            pass_count = pass_count + 1;
        end else begin
            $display("  out[0]  |   -40    | %4d   | FAIL", out0);
            fail_count = fail_count + 1;
        end

        if (out1 === 32'sd0) begin
            $display("  out[1]  |     0    | %4d   | PASS", out1);
            pass_count = pass_count + 1;
        end else begin
            $display("  out[1]  |     0    | %4d   | FAIL", out1);
            fail_count = fail_count + 1;
        end

        if (out2 === -32'sd20) begin
            $display("  out[2]  |   -20    | %4d   | PASS", out2);
            pass_count = pass_count + 1;
        end else begin
            $display("  out[2]  |   -20    | %4d   | FAIL", out2);
            fail_count = fail_count + 1;
        end

        if (out3 === -32'sd20) begin
            $display("  out[3]  |   -20    | %4d   | PASS", out3);
            pass_count = pass_count + 1;
        end else begin
            $display("  out[3]  |   -20    | %4d   | FAIL", out3);
            fail_count = fail_count + 1;
        end

        $display("  ----------------------------------------");
        $display("\n[Result] %0d/4 PASSED  %0d/4 FAILED",
                  pass_count, fail_count);

        if (fail_count == 0)
            $display("[SIMULATION PASSED] Outputs match hand-calculated values.");
        else
            $display("[SIMULATION FAILED] Some outputs did not match.");

        $display("==============================================================");
        $finish;
    end

endmodule
