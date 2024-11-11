`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/04/06 15:15:37
// Design Name: 
// Module Name: test
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module adder(
    input A,
    input B,
    input C_in,
    output sum,
    output C_out
    );
assign sum = (A ^ B ^ C);
assign C_out = ((A^B)&C )| (A&B);


  
endmodule
module bits_4(
    input [3:0] A,
    input [3:0] B,
    input C_in,
    output C_out,
    output [3:0] sum
 
);
wire [3:0]C_temp;
adder b0(.A(A[0]),.B(B[0]),.C_in(C_in),.C_out(C_temp[0]),.sum(sum[0]));
adder b1(.A(A[1]),.B(B[1]),.C_in(C_temp[0]),.C_out(C_temp[1]),.sum(sum[1]));
adder b2(.A(A[2]),.B(B[2]),.C_in(C_temp[1]),.C_out(C_temp[2]),.sum(sum[2]));
adder b3(.A(A[3]),.B(B[3]),.C_in(C_temp[2]),.C_out(C_out),.sum(sum[3]));
endmodule




