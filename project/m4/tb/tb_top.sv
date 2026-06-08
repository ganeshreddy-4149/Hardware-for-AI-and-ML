`timescale 1ns/1ps
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
  conv2d_top #(
    .H_IN(5),.W_IN(5),.K(3),
    .H_OUT(1),.W_OUT(1),
    .C_IN(1),.C_OUT(1),
    .SHIFT_BITS(0)
  ) dut (
    .clk(clk),.rst(rst),
    .s_axil_awaddr(s_axil_awaddr),
    .s_axil_awvalid(s_axil_awvalid),
    .s_axil_awready(s_axil_awready),
    .s_axil_wdata(s_axil_wdata),
    .s_axil_wstrb(s_axil_wstrb),
    .s_axil_wvalid(s_axil_wvalid),
    .s_axil_wready(s_axil_wready),
    .s_axil_bresp(s_axil_bresp),
    .s_axil_bvalid(s_axil_bvalid),
    .s_axil_bready(s_axil_bready),
    .s_axil_araddr(s_axil_araddr),
    .s_axil_arvalid(s_axil_arvalid),
    .s_axil_arready(s_axil_arready),
    .s_axil_rdata(s_axil_rdata),
    .s_axil_rresp(s_axil_rresp),
    .s_axil_rvalid(s_axil_rvalid),
    .s_axil_rready(s_axil_rready),
    .done(done),
    .busy(busy)
  );
  initial clk=0;
  always #5 clk=~clk;
  initial begin
    $dumpfile("final_waveform.vcd");
    $dumpvars(0,tb_top);
  end
  task axi_write;
    input [31:0] addr;
    input [31:0] data;
    begin
      @(posedge clk);
      s_axil_awaddr=addr; s_axil_awvalid=1;
      s_axil_wdata=data;  s_axil_wstrb=4'hF;
      s_axil_wvalid=1;    s_axil_bready=1;
      @(posedge clk);
      s_axil_awvalid=0; s_axil_wvalid=0;
      @(posedge clk);
    end
  endtask
  task axi_read;
    input  [31:0] addr;
    output [31:0] data;
    begin
      @(posedge clk);
      s_axil_araddr=addr; s_axil_arvalid=1;
      s_axil_rready=1;
      @(posedge clk);
      s_axil_arvalid=0;
      @(posedge clk);
      data=s_axil_rdata;
    end
  endtask
  reg signed [7:0] act_vals [0:24];
  reg signed [7:0] wt_vals  [0:8];
  reg signed [7:0] EXPECTED;
  reg [31:0] result;
  integer i;
  initial begin
    act_vals[0]=8'sd1;  act_vals[1]=8'sd2;  act_vals[2]=8'sd3;
    act_vals[3]=8'sd4;  act_vals[4]=8'sd5;
    act_vals[5]=8'sd1;  act_vals[6]=8'sd2;  act_vals[7]=8'sd3;
    act_vals[8]=8'sd4;  act_vals[9]=8'sd5;
    act_vals[10]=8'sd1; act_vals[11]=8'sd2; act_vals[12]=8'sd3;
    act_vals[13]=8'sd4; act_vals[14]=8'sd5;
    act_vals[15]=8'sd1; act_vals[16]=8'sd2; act_vals[17]=8'sd3;
    act_vals[18]=8'sd4; act_vals[19]=8'sd5;
    act_vals[20]=8'sd1; act_vals[21]=8'sd2; act_vals[22]=8'sd3;
    act_vals[23]=8'sd4; act_vals[24]=8'sd5;
    wt_vals[0]=8'sd1;  wt_vals[1]=8'sd0;  wt_vals[2]=-8'sd1;
    wt_vals[3]=8'sd1;  wt_vals[4]=8'sd0;  wt_vals[5]=-8'sd1;
    wt_vals[6]=8'sd1;  wt_vals[7]=8'sd0;  wt_vals[8]=-8'sd1;
    EXPECTED=-8'sd6;
    s_axil_awaddr=0; s_axil_awvalid=0;
    s_axil_wdata=0;  s_axil_wstrb=0; s_axil_wvalid=0;
    s_axil_bready=0;
    s_axil_araddr=0; s_axil_arvalid=0; s_axil_rready=0;
    rst=1;
    repeat(4) @(posedge clk);
    rst=0;
    @(posedge clk);
    $display("[TB] HOST WRITE: scale_shift=0 via AXI");
    axi_write(32'h008, 32'd0);
    $display("[TB] HOST WRITE: activations via AXI");
    for (i=0;i<25;i=i+1)
      axi_write(32'h100+i*4,{{24{act_vals[i][7]}},act_vals[i]});
    $display("[TB] HOST WRITE: weights via AXI");
    for (i=0;i<9;i=i+1)
      axi_write(32'h200+i*4,{{24{wt_vals[i][7]}},wt_vals[i]});
    $display("[TB] HOST WRITE: bias=0 via AXI");
    axi_write(32'h300, 32'd0);
    $display("[TB] HOST WRITE: start=1 via AXI");
    axi_write(32'h000, 32'd1);
    $display("[TB] COMPUTE: waiting for done");
    @(posedge done);
    repeat(5) @(posedge clk);
    $display("[TB] HOST READ: result via AXI addr 0x400");
    axi_read(32'h400, result);
    $display("[TB] Expected output[0,0] = %0d", $signed(EXPECTED));
    $display("[TB] Actual   output[0,0] = %0d", $signed(result[7:0]));
    if ($signed(result[7:0])===$signed(EXPECTED))
      $display("PASS: INT8 Conv2D output matches expected value %0d",
               $signed(EXPECTED));
    else
      $display("FAIL: expected %0d but got %0d",
               $signed(EXPECTED),$signed(result[7:0]));
    $display("[TB] HOST WRITE: start=0, returning to IDLE");
    axi_write(32'h000, 32'd0);
    repeat(5) @(posedge clk);
    $finish;
  end
  initial begin
    #10_000_000;
    $display("FAIL: simulation timeout");
    $finish;
  end
endmodule
