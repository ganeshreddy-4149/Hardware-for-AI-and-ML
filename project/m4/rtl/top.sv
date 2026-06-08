`timescale 1ns/1ps
module conv2d_top #(
  parameter DATA_WIDTH=8, parameter ACC_WIDTH=32,
  parameter C_IN=1, parameter C_OUT=1,
  parameter H_IN=5, parameter W_IN=5,
  parameter K=3, parameter H_OUT=1, parameter W_OUT=1,
  parameter SHIFT_BITS=0
)(
  input  wire        clk, rst,
  input  wire [31:0] s_axil_awaddr,
  input  wire        s_axil_awvalid,
  output reg         s_axil_awready,
  input  wire [31:0] s_axil_wdata,
  input  wire [3:0]  s_axil_wstrb,
  input  wire        s_axil_wvalid,
  output reg         s_axil_wready,
  output reg  [1:0]  s_axil_bresp,
  output reg         s_axil_bvalid,
  input  wire        s_axil_bready,
  input  wire [31:0] s_axil_araddr,
  input  wire        s_axil_arvalid,
  output reg         s_axil_arready,
  output reg  [31:0] s_axil_rdata,
  output reg  [1:0]  s_axil_rresp,
  output reg         s_axil_rvalid,
  input  wire        s_axil_rready,
  output reg         done,
  output reg         busy
);
  localparam IDLE=3'd0,LOAD=3'd1,COMPUTE=3'd2,OUTPUT=3'd3,DONE_S=3'd4;
  reg [2:0] state, next_state;
  reg        reg_start;
  reg [31:0] reg_scale_shift;
  reg [4:0]  h_cnt, w_cnt;
  reg [2:0]  oc_cnt;
  reg        compute_valid;
  reg signed [7:0]  act_buf    [0:24];
  reg signed [7:0]  weight_buf [0:8];
  reg signed [31:0] bias_buf   [0:0];
  reg signed [7:0]  out_buf    [0:0];
  wire signed [7:0]  mac_a0,mac_a1,mac_a2,mac_a3,mac_a4;
  wire signed [7:0]  mac_a5,mac_a6,mac_a7,mac_a8;
  wire signed [7:0]  mac_b0,mac_b1,mac_b2,mac_b3,mac_b4;
  wire signed [7:0]  mac_b5,mac_b6,mac_b7,mac_b8;
  wire signed [31:0] mac_out;
  wire               mac_en;
  wire signed [31:0] req_in;
  wire signed [7:0]  req_out;
  wire               req_valid;
  wire compute_done;
  assign compute_done = compute_valid &&
                        (h_cnt==H_OUT-1) &&
                        (w_cnt==W_OUT-1) &&
                        (oc_cnt==C_OUT-1);
  always @(posedge clk) begin
    if (rst) state <= IDLE;
    else     state <= next_state;
  end
  always @(*) begin
    next_state = state;
    case (state)
      IDLE:    if (reg_start)    next_state = LOAD;
      LOAD:                      next_state = COMPUTE;
      COMPUTE: if (compute_done) next_state = OUTPUT;
      OUTPUT:                    next_state = DONE_S;
      DONE_S:  if (!reg_start)   next_state = IDLE;
      default:                   next_state = IDLE;
    endcase
  end
  always @(posedge clk) begin
    if (rst||state==IDLE||state==LOAD) begin
      h_cnt<=5'd0; w_cnt<=5'd0; oc_cnt<=3'd0; compute_valid<=1'b0;
    end else if (state==COMPUTE) begin
      compute_valid<=1'b1;
      if (oc_cnt==C_OUT-1) begin
        oc_cnt<=3'd0;
        if (w_cnt==W_OUT-1) begin
          w_cnt<=5'd0;
          if (h_cnt<H_OUT-1) h_cnt<=h_cnt+5'd1;
        end else w_cnt<=w_cnt+5'd1;
      end else oc_cnt<=oc_cnt+3'd1;
    end
  end
  always @(posedge clk) begin
    if (rst) begin done<=1'b0; busy<=1'b0; end
    else begin
      if ((state==OUTPUT)&&(next_state==DONE_S)) done<=1'b1;
      else if (state==IDLE) done<=1'b0;
      busy<=(state==COMPUTE);
    end
  end
  always @(posedge clk) begin
    if (rst)                out_buf[0]<=8'sd0;
    else if (state==DONE_S) out_buf[0]<=req_out;
  end
  always @(posedge clk) begin
    if (rst) begin
      reg_start<=1'b0; reg_scale_shift<=SHIFT_BITS;
      s_axil_awready<=1'b0; s_axil_wready<=1'b0;
      s_axil_bvalid<=1'b0;  s_axil_bresp<=2'b00;
      s_axil_arready<=1'b0; s_axil_rvalid<=1'b0;
      s_axil_rdata<=32'h0;  s_axil_rresp<=2'b00;
    end else begin
      s_axil_awready<=s_axil_awvalid;
      s_axil_wready <=s_axil_wvalid;
      if (s_axil_wvalid&&s_axil_wready) begin
        if      (s_axil_awaddr==32'h000) reg_start<=s_axil_wdata[0];
        else if (s_axil_awaddr==32'h008) reg_scale_shift<=s_axil_wdata;
        else if (s_axil_awaddr>=32'h100&&s_axil_awaddr<32'h164)
          act_buf[(s_axil_awaddr-32'h100)>>2]<=s_axil_wdata[7:0];
        else if (s_axil_awaddr>=32'h200&&s_axil_awaddr<32'h224)
          weight_buf[(s_axil_awaddr-32'h200)>>2]<=s_axil_wdata[7:0];
        else if (s_axil_awaddr==32'h300) bias_buf[0]<=s_axil_wdata;
        s_axil_bvalid<=1'b1;
      end else if (s_axil_bready) s_axil_bvalid<=1'b0;
      s_axil_arready<=s_axil_arvalid;
      if (s_axil_arvalid&&s_axil_arready) begin
        s_axil_rvalid<=1'b1;
        case (s_axil_araddr)
          32'h004: s_axil_rdata<={30'h0,busy,done};
          32'h400: s_axil_rdata<={{24{out_buf[0][7]}},out_buf[0]};
          default: s_axil_rdata<=32'hDEADBEEF;
        endcase
      end else if (s_axil_rready) s_axil_rvalid<=1'b0;
    end
  end
  wire [4:0] base0, base1, base2;
  assign base0 = h_cnt*W_IN + w_cnt;
  assign base1 = (h_cnt+5'd1)*W_IN + w_cnt;
  assign base2 = (h_cnt+5'd2)*W_IN + w_cnt;
  assign mac_a0=act_buf[base0];      assign mac_a1=act_buf[base0+5'd1];
  assign mac_a2=act_buf[base0+5'd2]; assign mac_a3=act_buf[base1];
  assign mac_a4=act_buf[base1+5'd1]; assign mac_a5=act_buf[base1+5'd2];
  assign mac_a6=act_buf[base2];      assign mac_a7=act_buf[base2+5'd1];
  assign mac_a8=act_buf[base2+5'd2];
  assign mac_b0=weight_buf[0]; assign mac_b1=weight_buf[1];
  assign mac_b2=weight_buf[2]; assign mac_b3=weight_buf[3];
  assign mac_b4=weight_buf[4]; assign mac_b5=weight_buf[5];
  assign mac_b6=weight_buf[6]; assign mac_b7=weight_buf[7];
  assign mac_b8=weight_buf[8];
  assign mac_en    = (state==COMPUTE);
  assign req_in    = mac_out + bias_buf[0];
  assign req_valid = (state==OUTPUT);
  mac_array u_mac (
    .clk(clk),.rst(rst),.en(mac_en),
    .a0(mac_a0),.a1(mac_a1),.a2(mac_a2),
    .a3(mac_a3),.a4(mac_a4),.a5(mac_a5),
    .a6(mac_a6),.a7(mac_a7),.a8(mac_a8),
    .b0(mac_b0),.b1(mac_b1),.b2(mac_b2),
    .b3(mac_b3),.b4(mac_b4),.b5(mac_b5),
    .b6(mac_b6),.b7(mac_b7),.b8(mac_b8),
    .acc_out(mac_out)
  );
  requantizer u_req (
    .clk(clk),.rst(rst),
    .in_data(req_in),
    .shift(reg_scale_shift[4:0]),
    .valid_in(req_valid),
    .out_data(req_out)
  );
endmodule
