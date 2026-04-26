// mac_correct.v
// Fixes all issues from both LLM outputs.
// Key fix: use $signed() for explicit sign extension — iverilog compatible.
// Avoids 32'(signed'(x)) cast syntax which iverilog does not support.

module mac (
    input  logic        clk,
    input  logic        rst,
    input  logic signed [7:0]  a,
    input  logic signed [7:0]  b,
    output logic signed [31:0] out
);
    // Sign-extend both operands to 32 bits before multiplying.
    // This prevents the 16-bit product from being zero-extended into
    // the 32-bit accumulator (the sign extension bug in mac_llm_B.v).
    wire signed [31:0] a_ext;
    wire signed [31:0] b_ext;
    assign a_ext = {{24{a[7]}}, a};   // manual sign extension
    assign b_ext = {{24{b[7]}}, b};   // manual sign extension

    always_ff @(posedge clk) begin
        if (rst)
            out <= 32'sd0;
        else
            out <= out + (a_ext * b_ext);
    end
endmodule
