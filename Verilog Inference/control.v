`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/04/02 14:09:50
// Design Name: 
// Module Name: control
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


module control(
    input clk,
    input rst_n,
    input s0,s1,s2,s3,
    
    output [7:0] pic_data,
    output reg data_out_valid
);
wire S0,S1,S2,S3;   
debouncing DE0 (.clk(clk),.rst_n(rst_n),.de(s0),.data_out_valid(S0));
debouncing DE1 (.clk(clk),.rst_n(rst_n),.de(s1),.data_out_valid(S1));
debouncing DE2 (.clk(clk),.rst_n(rst_n),.de(s2),.data_out_valid(S2));
debouncing DE3 (.clk(clk),.rst_n(rst_n),.de(s3),.data_out_valid(S3));
//debouncing DE4 (.clk(clk),.rst_n(rst_n),.de(s4),.data_out_valid(S4));
//debouncing DE5 (.clk(clk),.rst_n(rst_n),.de(s5),.data_out_valid(S5));


reg B0,B1,B2,B3;


reg data_in_valid;
reg [11:0]pic_cnt;
wire [7:0] pic_data_0,pic_data_1,pic_data_2,pic_data_3;
pic_0 pic_0_U (
  .clka(clk),    // input wire clka
  .ena(B0),      // input wire ena
  .addra(pic_cnt),  // input wire [11 : 0] addra
  .douta(pic_data_0)  // output wire [7 : 0] douta
);
pic_1 pic_1_U (
  .clka(clk),    // input wire clka
  .ena(B1),      // input wire ena
  .addra(pic_cnt),  // input wire [11 : 0] addra
  .douta(pic_data_1)  // output wire [7 : 0] douta
);
pic_2 pic_2_U (
  .clka(clk),    // input wire clka
  .ena(B2),      // input wire ena
  .addra(pic_cnt),  // input wire [11 : 0] addra
  .douta(pic_data_2)  // output wire [7 : 0] douta
);
pic_3 pic_3_U (
  .clka(clk),    // input wire clka
  .ena(B3),      // input wire ena
  .addra(pic_cnt),  // input wire [11 : 0] addra
  .douta(pic_data_3)  // output wire [7 : 0] douta
);
//pic_4 pic_4_U (
//  .clka(clk),    // input wire clka
//  .ena(B4),      // input wire ena
//  .addra(pic_cnt),  // input wire [11 : 0] addra
//  .douta(pic_data_4)  // output wire [7 : 0] douta
//);
//pic_5 pic_5_U (
//  .clka(clk),    // input wire clka
//  .ena(B5),      // input wire ena
//  .addra(pic_cnt),  // input wire [11 : 0] addra
//  .douta(pic_data_5)  // output wire [7 : 0] douta
//);
assign pic_data = ({B0,B1,B2,B3}== 4'b1000 )?pic_data_0:
                  ({B0,B1,B2,B3}== 4'b0100)?pic_data_1:
                  ({B0,B1,B2,B3}== 4'b0010 )?pic_data_2:
                  ({B0,B1,B2,B3}== 4'b0001 )?pic_data_3:8'b00000000;
                  //({B0,B1,B2,B3}== 6'b000010 )?pic_data_4:
                  //({B0,B1,B2,B3,B4}== 6'b000001 )?pic_data_5: 
                    
always@(posedge clk or posedge rst_n)begin
    if(rst_n)begin 
        B0<=0;B1<=0;B2<=0;B3<=0;
        pic_cnt <= 0;
    end
    else if(B0 == 0 && B1 == 0 && B2 == 0 && B3 == 0)begin 
        if (S0 == 1)
            B0 <= 1'd1;
        else if (S1 == 1)
            B1 <= 1'd1;
        else if(S2 == 1)
            B2 <= 1'd1;
        else if(S3 == 1)
            B3 <= 1'd1;
    end
    else if(B0 == 1 || B1 == 1 || B2 == 1 || B3 == 1) begin 
        if(pic_cnt < 784)begin
            pic_cnt <= pic_cnt + 1'b1;
            data_out_valid <= 1'b1;
        end
        else begin
            pic_cnt <= 0;
            B0<=0;B1<=0;B2<=0;B3<=0;
            data_out_valid <= 1'b0;
        end
    end
end
endmodule
