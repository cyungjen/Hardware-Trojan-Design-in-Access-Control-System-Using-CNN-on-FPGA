`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/03/19 15:13:35
// Design Name: 
// Module Name: button
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


module debouncing(
    input clk,
    input rst_n,
    input de,
    
    output reg data_out_valid
    );
reg [23:0]counter;
always @(posedge clk or posedge rst_n)begin
    if(rst_n) counter <= 24'd0;
    else if(de == 1'b1) counter <= counter + 1'b1;
    else counter <= 24'd0;
end
//debouncing
always@(*)begin
//    if(counter == 24'd10000000) data_out_valid = 1'b1;
//    else if(counter == 24'd10000001) data_out_valid = 1'b0;
//    else data_out_valid = 1'b0;
    if(counter == 24'd100) data_out_valid = 1'b1;
    else if(counter == 24'd101) data_out_valid = 1'b0;
    else data_out_valid = 1'b0;
end

endmodule
