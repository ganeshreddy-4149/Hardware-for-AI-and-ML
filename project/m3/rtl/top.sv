// ============================================================================
// Project  : INT8 Conv2D Hardware Accelerator
// Module   : conv2d_top (Top-Level Integrated Module)
// Course   : ECE 410/510 — Hardware for AI/ML, Spring 2026
// Author   : Sai Ganesh Reddy Charian
// Interface: AXI4-Lite — ALL data and control goes through interface only
//
// Port List:
//   clk              - input  - 1b  - System clock
//   rst              - input  - 1b  - Active-high synchronous reset
//   s_axil_awaddr    - input  - 32b - AXI4-Lite write address
//   s_axil_awvalid   - input  - 1b  - Write address valid
//   s_axil_awready   - output - 1b  - Write address ready
//   s_axil_wdata     - input  - 32b - AXI4-Lite write data
//   s_axil_wstrb     - input  - 4b  - Write strobe
//   s_axil_wvalid    - input  - 1b  - Write data valid
//   s_axil_wready    - output - 1b  - Write data ready
//   s_axil_bresp     - output - 2b  - Write response
//   s_axil_bvalid    - output - 1b  - Write response valid
//   s_axil_bready    - input  - 1b  - Write response ready
//   s_axil_araddr    - input  - 32b - AXI4-Lite read address
//   s_axil_arvalid   - input  - 1b  - Read address valid
//   s_axil_arready   - output - 1b  - Read address ready
//   s_axil_rdata     - output - 32b - AXI4-Lite read data
//   s_axil_rresp     - output - 2b  - Read response
//   s_axil_rvalid    - output - 1b  - Read data valid
//   s_axil_rready    - input  - 1b  - Read data ready
//   done             - output - 1b  - Convolution complete flag
//   busy             - output - 1b  - Accelerator busy flag
//
// Internal Structure:
//   - AXI4-Lite is the ONLY path for all data and control
//   - FSM: IDLE -> LOAD -> COMPUTE -> OUTPUT -> DONE_S
//   - mac_array: 9 parallel INT8 multipliers, INT32 accumulation
//   - requantizer: INT32 -> INT8 arithmetic right shift + saturation
//   - Glue logic: mac_en connects FSM COMPUTE state directly to mac_array.en
//                 req_valid connects FSM OUTPUT state directly to requantizer.valid_in
//                 No FIFOs, no clock-domain crossings, no width converters needed
//
// AXI4-Lite Register Map:
//   0x000 W control (bit0=start)
//   0x004 R status  (bit0=done, bit1=busy)
//   0x008 W scale shift
//   0x100+i*4 W act_buf[i] i=0..24
//   0x200+i*4 W weight_buf[i] i=0..8
//   0x300 W bias_buf[0]
//   0x400 R out_buf[0] result
// ============================================================================
`timescale 1ns/1ps

module conv2d_top #(
  parameter DATA_WIDTH=8, parameter ACC_WIDTH=32,
  parameter C_IN=1,  parameter C_OUT=1,
  parameter H_IN=5,  parameter W_IN=5,
  parameter K=3,     parameter H_OUT=1, parameter W_OUT=1,
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

  wire compute_done;
  assign compute_done = compute_valid &&
                        (h_cnt==H_OUT-1) &&
                        (w_cnt==W_OUT-1) &&
                        (oc_cnt==C_OUT-1);

  reg signed [7:0]  weight_buf [0:8];
  reg signed [7:0]  act_buf    [0:24];
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
      h_cnt<=0; w_cnt<=0; oc_cnt<=0; compute_valid<=0;
    end else if (state==COMPUTE) begin
      compute_valid <= 1;
      if (oc_cnt==C_OUT-1) begin
        oc_cnt<=0;
        if (w_cnt==W_OUT-1) begin
          w_cnt<=0;
          if (h_cnt<H_OUT-1) h_cnt<=h_cnt+1;
        end else w_cnt<=w_cnt+1;
      end else oc_cnt<=oc_cnt+1;
    end
  end

  always @(posedge clk) begin
    if (rst) begin done<=0; busy<=0; end
    else begin
      if ((state==OUTPUT)&&(next_state==DONE_S)) done<=1;
      else if (state==IDLE) done<=0;
      busy<=(state==COMPUTE);
    end
  end

  always @(posedge clk) begin
    if (rst) out_buf[0]<=0;
    else if (state==DONE_S) out_buf[0]<=req_out;
  end

  always @(posedge clk) begin
    if (rst) begin
      reg_start<=0; reg_scale_shift<=SHIFT_BITS;
      s_axil_awready<=0; s_axil_wready<=0;
      s_axil_bvalid<=0;  s_axil_bresp<=2'b00;
      s_axil_arready<=0; s_axil_rvalid<=0;
      s_axil_rdata<=0;   s_axil_rresp<=2'b00;
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
        s_axil_bvalid<=1;
      end else if (s_axil_bready) s_axil_bvalid<=0;

      s_axil_arready<=s_axil_arvalid;
      if (s_axil_arvalid&&s_axil_arready) begin
        s_axil_rvalid<=1;
        if      (s_axil_araddr==32'h004)
          s_axil_rdata<={30'h0,busy,done};
        else if (s_axil_araddr==32'h400)
          s_axil_rdata<={{24{out_buf[0][7]}},out_buf[0]};
        else s_axil_rdata<=32'hDEADBEEF;
      end else if (s_axil_rready) s_axil_rvalid<=0;
    end
  end

  assign mac_a0=act_buf[h_cnt*W_IN+w_cnt];
  assign mac_a1=act_buf[h_cnt*W_IN+w_cnt+1];
  assign mac_a2=act_buf[h_cnt*W_IN+w_cnt+2];
  assign mac_a3=act_buf[(h_cnt+1)*W_IN+w_cnt];
  assign mac_a4=act_buf[(h_cnt+1)*W_IN+w_cnt+1];
  assign mac_a5=act_buf[(h_cnt+1)*W_IN+w_cnt+2];
  assign mac_a6=act_buf[(h_cnt+2)*W_IN+w_cnt];
  assign mac_a7=act_buf[(h_cnt+2)*W_IN+w_cnt+1];
  assign mac_a8=act_buf[(h_cnt+2)*W_IN+w_cnt+2];

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

// ============================================================================
// Module: mac_array
// 9 parallel INT8 signed multipliers with INT32 registered accumulator
// All products sign-extended to 32 bits before addition
// ============================================================================
module mac_array (
  input  wire        clk, rst, en,
  input  wire signed [7:0] a0,a1,a2,a3,a4,a5,a6,a7,a8,
  input  wire signed [7:0] b0,b1,b2,b3,b4,b5,b6,b7,b8,
  output reg  signed [31:0] acc_out
);
  wire signed [31:0] p0,p1,p2,p3,p4,p5,p6,p7,p8;
  assign p0={{24{a0[7]}},a0}*{{24{b0[7]}},b0};
  assign p1={{24{a1[7]}},a1}*{{24{b1[7]}},b1};
  assign p2={{24{a2[7]}},a2}*{{24{b2[7]}},b2};
  assign p3={{24{a3[7]}},a3}*{{24{b3[7]}},b3};
  assign p4={{24{a4[7]}},a4}*{{24{b4[7]}},b4};
  assign p5={{24{a5[7]}},a5}*{{24{b5[7]}},b5};
  assign p6={{24{a6[7]}},a6}*{{24{b6[7]}},b6};
  assign p7={{24{a7[7]}},a7}*{{24{b7[7]}},b7};
  assign p8={{24{a8[7]}},a8}*{{24{b8[7]}},b8};
  always @(posedge clk) begin
    if (rst)     acc_out<=0;
    else if (en) acc_out<=p0+p1+p2+p3+p4+p5+p6+p7+p8;
  end
endmodule

// ============================================================================
// Module: requantizer
// INT32 -> INT8 with arithmetic right shift and signed saturation
// Saturates to +127 / -128 on overflow
// ============================================================================
module requantizer (
  input  wire        clk, rst, valid_in,
  input  wire signed [31:0] in_data,
  input  wire [4:0]  shift,
  output reg  signed [7:0]  out_data
);
  wire signed [31:0] shifted;
  assign shifted = in_data >>> shift;
  always @(posedge clk) begin
    if (rst) out_data<=0;
    else if (valid_in) begin
      if      (shifted> 127) out_data<= 8'sd127;
      else if (shifted<-128) out_data<=-8'sd128;
      else                   out_data<=shifted[7:0];
    end
  end
endmodule
