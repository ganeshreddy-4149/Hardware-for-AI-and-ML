// =============================================================================
// tb_interface.sv
// Project : INT8 Conv2D Accelerator — ECE 510, Spring 2026
// Author  : Sai Ganesh Reddy Charian
// File    : project/m2/tb/tb_interface.sv
//
// Purpose : Testbench for interface.sv (AXI4-Lite slave).
//           Full AXI4-Lite handshake on all five channels.
//           Prints PASS or FAIL. Dumps interface.vcd.
//
// Test sequence:
//   1. Write 0x1 to CTRL (0x00)  — verify core_start=1
//   2. Read  STATUS (0x04)       — verify busy=1 done=0
//   3. Write 0x5 to SHIFT (0x08) — verify core_shift=5
//   4. Read  SHIFT  (0x08)       — verify rdata[4:0]=5
// =============================================================================
`timescale 1ns / 1ps

module tb_interface;

    reg clk, rst;
    reg [31:0] s_awaddr; reg s_awvalid; wire s_awready;
    reg [31:0] s_wdata;  reg [3:0] s_wstrb; reg s_wvalid; wire s_wready;
    wire [1:0] s_bresp;  wire s_bvalid; reg s_bready;
    reg [31:0] s_araddr; reg s_arvalid; wire s_arready;
    wire [31:0] s_rdata; wire [1:0] s_rresp; wire s_rvalid; reg s_rready;
    reg core_done, core_busy;
    wire core_start; wire [4:0] core_shift;

    axi4lite_if dut(
        .clk(clk),.rst(rst),
        .s_awaddr(s_awaddr),.s_awvalid(s_awvalid),.s_awready(s_awready),
        .s_wdata(s_wdata),.s_wstrb(s_wstrb),.s_wvalid(s_wvalid),.s_wready(s_wready),
        .s_bresp(s_bresp),.s_bvalid(s_bvalid),.s_bready(s_bready),
        .s_araddr(s_araddr),.s_arvalid(s_arvalid),.s_arready(s_arready),
        .s_rdata(s_rdata),.s_rresp(s_rresp),.s_rvalid(s_rvalid),.s_rready(s_rready),
        .core_done(core_done),.core_busy(core_busy),
        .core_start(core_start),.core_shift(core_shift));

    initial clk=0; always #5 clk=~clk;
    initial begin $dumpfile("interface.vcd"); $dumpvars(0,tb_interface); end

    integer fail_count;
    reg [31:0] captured;

    // -------------------------------------------------------------------------
    // Write: explicit step-by-step, one posedge per phase
    // Phase 1: present AW → wait for AWREADY
    // Phase 2: present W  → wait for WREADY
    // Phase 3: keep BREADY high → wait for BVALID → hold one more cycle → drop
    // -------------------------------------------------------------------------
    task do_write;
        input [31:0] addr;
        input [31:0] data;
        integer i;
        begin
            // Phase 1: AW
            @(posedge clk);
            s_awaddr = addr; s_awvalid = 1;
            s_wdata  = data; s_wstrb = 4'hF; s_wvalid = 0;
            s_bready = 0;
            for (i=0; i<20; i=i+1) begin
                @(posedge clk);
                if (s_awready) begin s_awvalid=0; i=20; end
            end

            // Phase 2: W (present data AFTER address accepted)
            @(posedge clk);
            s_wvalid = 1; s_bready = 1;
            for (i=0; i<20; i=i+1) begin
                @(posedge clk);
                if (s_wready) begin s_wvalid=0; i=20; end
            end

            // Phase 3: B (keep bready=1, wait for bvalid, then one more cycle, drop)
            for (i=0; i<20; i=i+1) begin
                @(posedge clk);
                if (s_bvalid) begin i=20; end
            end
            @(posedge clk);   // hold bready one extra cycle
            s_bready = 0;

            repeat(2) @(posedge clk);
        end
    endtask

    // -------------------------------------------------------------------------
    // Read: explicit step-by-step
    // Phase 1: present AR → wait for ARREADY
    // Phase 2: wait for RVALID → capture RDATA → hold RREADY one more cycle
    // -------------------------------------------------------------------------
    task do_read;
        input  [31:0] addr;
        output [31:0] rdata;
        integer i;
        begin
            rdata = 0;
            // Phase 1: AR
            @(posedge clk);
            s_araddr = addr; s_arvalid = 1; s_rready = 1;
            for (i=0; i<20; i=i+1) begin
                @(posedge clk);
                if (s_arready) begin s_arvalid=0; i=20; end
            end

            // Phase 2: R
            for (i=0; i<20; i=i+1) begin
                @(posedge clk);
                if (s_rvalid) begin rdata=s_rdata; i=20; end
            end
            @(posedge clk);   // hold rready one extra cycle
            s_rready = 0;

            repeat(2) @(posedge clk);
        end
    endtask

    initial begin
        fail_count=0; captured=0;
        rst=1;
        s_awaddr=0; s_awvalid=0; s_wdata=0; s_wstrb=0; s_wvalid=0;
        s_bready=0; s_araddr=0; s_arvalid=0; s_rready=0;
        core_done=0; core_busy=0;
        repeat(4) @(posedge clk);
        rst=0;
        repeat(2) @(posedge clk);

        // Test 1: Write CTRL=1
        do_write(32'h00, 32'h1);
        if (core_start===1'b1)
            $display("CHECK 1: write CTRL=1 core_start=%0b --> PASS", core_start);
        else begin
            $display("CHECK 1: write CTRL=1 core_start=%0b --> FAIL", core_start);
            fail_count=fail_count+1;
        end

        // Test 2: Read STATUS
        core_busy=1; core_done=0;
        do_read(32'h04, captured);
        if (captured[1]===1'b1 && captured[0]===1'b0)
            $display("CHECK 2: read STATUS=0x%08h busy=1 done=0 --> PASS", captured);
        else begin
            $display("CHECK 2: read STATUS=0x%08h --> FAIL", captured);
            fail_count=fail_count+1;
        end

        // Test 3: Write SHIFT=5
        do_write(32'h08, 32'h5);
        if (core_shift===5'd5)
            $display("CHECK 3: write SHIFT=5 core_shift=%0d --> PASS", core_shift);
        else begin
            $display("CHECK 3: write SHIFT=5 core_shift=%0d --> FAIL", core_shift);
            fail_count=fail_count+1;
        end

        // Test 4: Read SHIFT back
        do_read(32'h08, captured);
        if (captured[4:0]===5'd5)
            $display("CHECK 4: read SHIFT=%0d --> PASS", captured[4:0]);
        else begin
            $display("CHECK 4: read SHIFT=%0d --> FAIL", captured[4:0]);
            fail_count=fail_count+1;
        end

        #20;
        if (fail_count==0) $display("PASS");
        else                $display("FAIL");
        #10; $finish;
    end

    initial begin #5_000_000; $display("TIMEOUT"); $display("FAIL"); $finish; end
endmodule
