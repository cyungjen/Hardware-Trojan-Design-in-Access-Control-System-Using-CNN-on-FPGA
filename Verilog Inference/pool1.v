`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/02/01 18:20:33
// Design Name: 
// Module Name: pool1
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


module pool1(
    input clk,
    input rst_n,
    
    input   [32*3-1:0] data_in,
    input   data_in_valid,

    
    output  [32*3-1:0] data_out,
     output   data_out_valid
    );
    
reg [5:0]   x_cnt;    
reg [5:0]   y_cnt;    
always@(posedge clk ,posedge rst_n)begin
    if(rst_n)
        x_cnt <= 0;
    else if(data_in_valid && x_cnt == 'd25 )
        x_cnt <= 0;
    else if(data_in_valid)
        x_cnt <=x_cnt +1'b1;
end
 always@(posedge clk ,posedge rst_n)begin
    if(rst_n)
       y_cnt <= 0;
    else if(data_in_valid && x_cnt == 'd25 && y_cnt == 'd25  )
        y_cnt <= 0;
    else if(data_in_valid && x_cnt == 'd25 )
        y_cnt <=y_cnt +1'b1;
end
//==================== delay data_in =============
reg [32*3-1:0] delay_data_in;
always@(posedge clk)
        delay_data_in<=data_in;//delay 1 clk
//==================== prepare for ram =============
wire    [31:0] wr_data[0:2];
wire    [31:0] rd_data[0:2];
wire                wr_en;
reg     [5:0]           wr_addr;
reg     [5:0]           rd_addr;
assign wr_en = x_cnt >0;//when x_cnt>0
genvar k;
generate 
    for (k=0;k<3;k=k+1)begin: generate_block_1
        assign wr_data[k] = ( data_in[(k+1)*32-1:k*32] > delay_data_in[(k+1)*32-1:k*32])?data_in[(k+1)*32-1:k*32]:delay_data_in[(k+1)*32-1:k*32];
    end
endgenerate

wire [5:0]  rd_addr_pre2 = wr_addr +2;
always@(posedge clk,posedge rst_n)begin
    if(rst_n)begin
        wr_addr <=0;
        rd_addr <= 0;
    end 
    else if(data_in_valid )begin
        if(wr_addr == 'd25)
            wr_addr<=0;
        else 
            wr_addr <= wr_addr +1'b1;
            
        if(rd_addr_pre2 > 'd25)
            rd_addr <= rd_addr_pre2-'d26;
        else
            rd_addr <= rd_addr_pre2;
    end
end

generate
for (k=0;k<3;k=k+1)begin: generate_block_2
pool1_data_linebuffer pool1_data_linebuffer_U (
  .clka(clk),    // input wire clka
  .ena(1'b1),      // input wire ena
  .wea(wr_en),      // input wire [0 : 0] wea
  .addra(wr_addr),  // input wire [5 : 0] addra
  .dina(wr_data[k]),    // input wire [31 : 0] dina
  //.enb(1'b1),
  .clkb(clk),    // input wire clkb
  .addrb(rd_addr),  // input wire [5 : 0] addrb
  .doutb(rd_data[k])  // output wire [31 : 0] doutb
);
end
endgenerate

generate 
    for (k=0;k<3;k=k+1)begin: generate_block_3
        assign data_out[(k+1)*32-1:k*32] = ( rd_data[k] > wr_data[k] )?rd_data[k] :wr_data[k];
    end
endgenerate

assign data_out_valid = ( x_cnt[0:0]==1 &&  y_cnt[0:0]==1)?1'b1:1'b0;
    
endmodule
