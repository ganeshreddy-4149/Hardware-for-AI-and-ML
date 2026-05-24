`timescale 1ns/1ps
// ============================================================================
// Module   : tb_top — End-to-End Co-Simulation Testbench
// Course   : ECE 410/510 — Hardware for AI/ML, Spring 2026
// Author   : Sai Ganesh Reddy Charian
//
// Description:
//   All data driven through AXI4-Lite interface only — no direct DUT access.
//   Testbench acts as a host processor sending AXI4-Lite transactions.
//
// Test vector:
//   Activation patch (5x5, top-left 3x3 = [[1,2,3],[4,5,6],[7,8,9]], rest=0)
//   Weight kernel [[1,0,-1],[1,0,-1],[1,0,-1]]
//   Bias = 0, Scale shift = 0
//
// Hand calculation (independent reference, not from prior DUT run):
//   out = 1*1 + 2*0 + 3*(-1)
//       + 4*1 + 5*0 + 6*(-1)
//       + 7*1 + 8*0 + 9*(-1)
//       = (1-3) + (4-6) + (7-9)
//       = -2 + -2 + -2
//       = -6
//   Expected INT8 output = -6 (signed), 8'hFA (hex)
// ============================================================================

module tb_top;

  reg         clk, rst;
  reg  [31:0] s_axil_awaddr;
  reg         s_axil_awvalid;
  wire        s_axil_awready;
  reg  [31:0] s_axil_wdata;
  reg  [3:0]  s_axil_wstrb;
  reg         s_axil_wvalid;
  wire        s_axil_wready;
  wire [1:0]  s_axil_bresp;
  wire        s_axil_bvalid;
  reg         s_axil_bready;
  reg  [31:0] s_axil_araddr;
  reg         s_axil_arvalid;
  wire        s_axil_arready;
  wire [31:0] s_axil_rdata;
  wire [1:0]  s_axil_rresp;
  wire        s_axil_rvalid;
  reg         s_axil_rready;
  wire        done;
  wire        busy;


  localparam signed [7:0] EXPECTED = -8'sd6;

  conv2d_top #(
    .DATA_WIDTH(8),.ACC_WIDTH(32),
    .C_IN(1),.C_OUT(1),
    .H_IN(5),.W_IN(5),
    .K(3),.H_OUT(1),.W_OUT(1),
    .SHIFT_BITS(0)
  ) dut (
    .clk(clk),.rst(rst),
    .s_axil_awaddr(s_axil_awaddr),.s_axil_awvalid(s_axil_awvalid),
    .s_axil_awready(s_axil_awready),
    .s_axil_wdata(s_axil_wdata),.s_axil_wstrb(s_axil_wstrb),
    .s_axil_wvalid(s_axil_wvalid),.s_axil_wready(s_axil_wready),
    .s_axil_bresp(s_axil_bresp),.s_axil_bvalid(s_axil_bvalid),
    .s_axil_bready(s_axil_bready),
    .s_axil_araddr(s_axil_araddr),.s_axil_arvalid(s_axil_arvalid),
    .s_axil_arready(s_axil_arready),
    .s_axil_rdata(s_axil_rdata),.s_axil_rresp(s_axil_rresp),
    .s_axil_rvalid(s_axil_rvalid),.s_axil_rready(s_axil_rready),
    .done(done),.busy(busy)
  );

  initial clk=0;
  always #5 clk=~clk;

  initial begin
    $dumpfile("cosim_waveform.vcd");
    $dumpvars(0,tb_top);
  end

  // AXI4-Lite write task — drives awaddr+wdata through proper handshake
  task axi_write;
    input [31:0] addr;
    input [31:0] data;
    begin
      @(posedge clk);
      s_axil_awaddr<=addr; s_axil_awvalid<=1;
      s_axil_wdata<=data;  s_axil_wstrb<=4'hF;
      s_axil_wvalid<=1;    s_axil_bready<=1;
      @(posedge clk);
      @(posedge clk);
      @(posedge clk);
      s_axil_awvalid<=0; s_axil_wvalid<=0;
      @(posedge clk);
      s_axil_bready<=0;
      @(posedge clk);
    end
  endtask

  // AXI4-Lite read task — drives araddr and captures rdata through handshake
  task axi_read;
    input  [31:0] addr;
    output [31:0] rdata;
    begin
      @(posedge clk);
      s_axil_araddr<=addr; s_axil_arvalid<=1; s_axil_rready<=1;
      @(posedge clk);
      @(posedge clk);
      @(posedge clk);
      rdata=s_axil_rdata;
      s_axil_arvalid<=0; s_axil_rready<=0;
      @(posedge clk);
    end
  endtask

  integer i;
  reg [31:0] result;
  reg signed [7:0] act_vals [0:24];
  reg signed [7:0] wt_vals  [0:8];

  initial begin
    // ---- RESET PHASE ----
    rst=1;
    s_axil_awaddr=0; s_axil_awvalid=0;
    s_axil_wdata=0;  s_axil_wstrb=0; s_axil_wvalid=0;
    s_axil_bready=0; s_axil_araddr=0; s_axil_arvalid=0;
    s_axil_rready=0; result=0;

    repeat(5) @(posedge clk);
    rst=0;
    repeat(2) @(posedge clk);

    // ---- LOAD TEST VECTORS ----
    // 5x5 activation map — top-left 3x3 = [[1,2,3],[4,5,6],[7,8,9]], rest=0
    act_vals[0]=8'sd1;  act_vals[1]=8'sd2;  act_vals[2]=8'sd3;
    act_vals[3]=8'sd0;  act_vals[4]=8'sd0;
    act_vals[5]=8'sd4;  act_vals[6]=8'sd5;  act_vals[7]=8'sd6;
    act_vals[8]=8'sd0;  act_vals[9]=8'sd0;
    act_vals[10]=8'sd7; act_vals[11]=8'sd8; act_vals[12]=8'sd9;
    act_vals[13]=8'sd0; act_vals[14]=8'sd0;
    act_vals[15]=0; act_vals[16]=0; act_vals[17]=0;
    act_vals[18]=0; act_vals[19]=0; act_vals[20]=0;
    act_vals[21]=0; act_vals[22]=0; act_vals[23]=0; act_vals[24]=0;

    // 3x3 weight kernel [[1,0,-1],[1,0,-1],[1,0,-1]] — vertical edge detector
    wt_vals[0]=8'sd1;  wt_vals[1]=8'sd0;  wt_vals[2]=-8'sd1;
    wt_vals[3]=8'sd1;  wt_vals[4]=8'sd0;  wt_vals[5]=-8'sd1;
    wt_vals[6]=8'sd1;  wt_vals[7]=8'sd0;  wt_vals[8]=-8'sd1;

    // ---- HOST WRITE PHASE — ALL through AXI4-Lite ----
    $display("[TB] HOST WRITE: scale_shift=0 via AXI");
    axi_write(32'h008, 32'd0);

    $display("[TB] HOST WRITE: activations via AXI");
    begin : load_act
      reg [31:0] act_word;
      for (i=0;i<25;i=i+1) begin
        act_word={{24{act_vals[i][7]}},act_vals[i]};
        axi_write(32'h100+i*4, act_word);
      end
    end

    $display("[TB] HOST WRITE: weights via AXI");
    begin : load_wt
      reg [31:0] wt_word;
      for (i=0;i<9;i=i+1) begin
        wt_word={{24{wt_vals[i][7]}},wt_vals[i]};
        axi_write(32'h200+i*4, wt_word);
      end
    end

    $display("[TB] HOST WRITE: bias=0 via AXI");
    axi_write(32'h300, 32'd0);

    $display("[TB] HOST WRITE: start=1 via AXI");
    axi_write(32'h000, 32'd1);

    // ---- COMPUTE PHASE — wait for done signal ----
    $display("[TB] COMPUTE: waiting for done");
    @(posedge done);
    repeat(5) @(posedge clk);

    // ---- HOST READ PHASE — result through AXI4-Lite ----
    $display("[TB] HOST READ: result via AXI addr 0x400");
    axi_read(32'h400, result);

    $display("[TB] Expected output[0,0] = %0d", $signed(EXPECTED));
    $display("[TB] Actual   output[0,0] = %0d", $signed(result[7:0]));

    // ---- PASS/FAIL COMPARISON ----
    if ($signed(result[7:0]) === $signed(EXPECTED))
      $display("PASS: INT8 Conv2D output matches expected value %0d", $signed(EXPECTED));
    else
      $display("FAIL: expected %0d but got %0d",
               $signed(EXPECTED), $signed(result[7:0]));

    // ---- RETURN TO IDLE ----
    $display("[TB] HOST WRITE: start=0, returning to IDLE");
    axi_write(32'h000, 32'd0);
    repeat(5) @(posedge clk);
    $finish;
  end

  // Timeout guard — prevents infinite hang if DUT never raises done
  initial begin
    #10000000;
    $display("FAIL: simulation timeout");
    $finish;
  end

endmodule
