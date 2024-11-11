`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/02/01 18:22:18
// Design Name: 
// Module Name: pool2
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


module pool2(
    input clk,
    input rst_n,
    
    input [16*3-1:0] data_in,
    input             data_in_valid,
    
//    output reg             done,
    output [16*3-1:0] data_out,
    output             data_out_valid
    
    );
reg [4:0]   x_cnt;    
reg [4:0]   y_cnt;    
always@(posedge clk ,posedge rst_n)begin
    if(rst_n)
        x_cnt <= 0;
    else if(data_in_valid && x_cnt == 'd10 )
        x_cnt <= 0;
    else if(data_in_valid)
        x_cnt <=x_cnt +1'b1;
end
 always@(posedge clk ,posedge rst_n)begin
    if(rst_n)
       y_cnt <= 0;
    else if(data_in_valid && x_cnt == 'd10 && y_cnt == 'd10  )
        y_cnt <= 0;
    else if(data_in_valid && x_cnt == 'd10 )
        y_cnt <=y_cnt +1'b1;
end
  
//==================== delay data_in =============
reg [16*3-1:0] delay_data_in;
always@(posedge clk)
        delay_data_in<=data_in;
//==================== prepare for ram =============
wire    [15:0] wr_data[0:2];
wire    [15:0] rd_data[0:2];
wire                    wr_en;
reg     [4:0]           wr_addr;
reg     [4:0]           rd_addr;
assign wr_en = data_in_valid;
genvar k;
generate 
    for (k=0;k<3;k=k+1)begin: generate_block_1
        assign wr_data[k] = ( data_in[(k+1)*16-1:k*16] >= delay_data_in[(k+1)*16-1:k*16])?data_in[(k+1)*16-1:k*16]:delay_data_in[(k+1)*16-1:k*16];
    end
endgenerate

wire [4:0]  rd_addr_pre2 = wr_addr +2;
always@(posedge clk,posedge rst_n)begin
    if(rst_n)begin
        wr_addr <=0;
        rd_addr <= 0;
    end 
    else if(data_in_valid )begin
        if(wr_addr == 'd10)
            wr_addr<=0;
        else 
            wr_addr <= wr_addr +1'b1;
            
        if(rd_addr_pre2 > 'd10)
            rd_addr <= rd_addr_pre2-'d11;
        else
            rd_addr <= rd_addr_pre2;
    end
end
 

generate
    for(k=0;k<3;k=k+1)begin: generate_block_2
            p2_linebuffer your_instance_name (
              .clka(clk),    // input wire clka
              .ena(1'b1),      // input wire ena
              .wea(wr_en),      // input wire [0 : 0] wea
              .addra(wr_addr),  // input wire [3 : 0] addra
              .dina(wr_data[k]),    // input wire [15 : 0] dina
              
              .clkb(clk),    // input wire clkb
              .enb(1'b1),      // input wire enb
              .addrb(rd_addr),  // input wire [3 : 0] addrb
              .doutb(rd_data[k])  // output wire [15 : 0] doutb
        );
    end  
endgenerate

//wire [15:0] data_out_vis[0:799];
//reg [15:0]   out_cnt;    

//always@(posedge clk ,negedge rst_n)begin
//    if(~rst_n)
//        out_cnt <= 0;
//    else if(data_out_valid)
//        out_cnt <= out_cnt+1;
//end

//generate 
//    for (k=0;k<8;k=k+1)begin
//        assign data_out[(k+1)*16-1:k*16] = ( rd_data[k] > wr_data[k] )?rd_data[k] :wr_data[k];
//        assign data_out_vis[100*k+out_cnt] = ( rd_data[k] > wr_data[k] )?rd_data[k] :wr_data[k];
//    end
//endgenerate


//assign data_out_valid = ( x_cnt[0:0]==1 &&  y_cnt[0:0]==1)?1'b1:1'b0;

wire [15:0] data_out_vis[0:2];

generate 
    for (k=0;k<3;k=k+1)begin: generate_block_3
        assign data_out[(k+1)*16-1:k*16] = ( rd_data[k] > wr_data[k] )?rd_data[k] :wr_data[k];
        assign data_out_vis[k] = data_out[(k+1)*16-1:k*16] ;
    end
endgenerate

assign data_out_valid = ( x_cnt[0:0]==1 &&  y_cnt[0:0]==1)?1'b1:1'b0;

endmodule
