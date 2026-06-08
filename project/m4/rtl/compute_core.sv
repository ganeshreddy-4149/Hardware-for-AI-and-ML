`timescale 1ns/1ps

module mac_array (
  input  wire        clk, rst, en,
  input  wire signed [7:0]  a0,a1,a2,a3,a4,a5,a6,a7,a8,
  input  wire signed [7:0]  b0,b1,b2,b3,b4,b5,b6,b7,b8,
  output reg  signed [31:0] acc_out
);
  wire signed [15:0] p0,p1,p2,p3,p4,p5,p6,p7,p8;
  assign p0=$signed(a0)*$signed(b0); assign p1=$signed(a1)*$signed(b1);
  assign p2=$signed(a2)*$signed(b2); assign p3=$signed(a3)*$signed(b3);
  assign p4=$signed(a4)*$signed(b4); assign p5=$signed(a5)*$signed(b5);
  assign p6=$signed(a6)*$signed(b6); assign p7=$signed(a7)*$signed(b7);
  assign p8=$signed(a8)*$signed(b8);
  always @(posedge clk) begin
    if (rst) acc_out<=32'sd0;
    else if (en)
      acc_out<={{16{p0[15]}},p0}+{{16{p1[15]}},p1}
              +{{16{p2[15]}},p2}+{{16{p3[15]}},p3}
              +{{16{p4[15]}},p4}+{{16{p5[15]}},p5}
              +{{16{p6[15]}},p6}+{{16{p7[15]}},p7}
              +{{16{p8[15]}},p8};
  end
endmodule

module requantizer (
  input  wire        clk, rst, valid_in,
  input  wire signed [31:0] in_data,
  input  wire [4:0]  shift,
  output reg  signed [7:0]  out_data
);
  wire signed [31:0] shifted;
  assign shifted = in_data>>>shift;
  always @(posedge clk) begin
    if (rst) out_data<=8'sd0;
    else if (valid_in) begin
      if      (shifted> 32'sd127) out_data<= 8'sd127;
      else if (shifted<-32'sd128) out_data<=-8'sd128;
      else                        out_data<=shifted[7:0];
    end
  end
endmodule
