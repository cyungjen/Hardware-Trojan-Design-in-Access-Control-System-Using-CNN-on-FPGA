`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/04/02 14:08:37
// Design Name: 
// Module Name: conv2
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


module conv2(
    input clk,
    input rst_n,
    
    input [32*3-1:0] data_in,
    input data_in_valid,
    input malicious_valid,
    
    output [15:0] data_out,
    output reg c2_ready,
    output reg data_out_valid
    );
    

genvar m, n, k;
//=====================ADDR================
reg [3:0] wr_addr;
reg [3:0] rd_addr;
wire [3:0] rd_addr_pre2 = wr_addr + 2;
always@(posedge clk or posedge rst_n)begin
    if(rst_n)begin
        wr_addr <= 0;
        rd_addr <= 0;
    end
    else if(  data_in_valid == 1'b1 )begin
        if(wr_addr == 'd12)
            wr_addr <= 0;
        else
            wr_addr <=  wr_addr + 1'd1;
        
        if(rd_addr_pre2 > 'd12)
            rd_addr <= rd_addr_pre2 - 4'd13;
        else
            rd_addr <=rd_addr_pre2; 
    end
 end  
 
//=====================DATA================
wire [32*3-1:0] window_in[0:2];
wire [32*3-1:0] window_out[0:2];
assign window_in[0] = data_in;

generate 
    for(k=1;k<3;k=k+1)begin: generate_block_1
        assign window_in[k] = window_out[k-1];
    end
endgenerate
reg delay_c2_ready;
always@(posedge clk) delay_c2_ready <= c2_ready;
//read one line a time
generate   
  for(k=0;k<3;k=k+1)begin: generate_block_2    
    conv2_linebuffer conv2_linebuffer_U (           
        .clka(clk),    // input wire clka           
        .ena(1'b1),      // input wire ena          
        .wea(data_in_valid),      // input wire [0 : 0] wea
        .addra(wr_addr),  // input wire [4 : 0] addra
        .dina(window_in[k]),    // input wire [511 : 0] dina
        .clkb(clk),    // input wire clkb           
        .enb(delay_c2_ready),      // input wire enb
        .addrb(rd_addr),  // input wire [4 : 0] addrb
        .doutb(window_out[k])  // output wire [511 : 0] doutb
    );
end
endgenerate

//================data window(16 channel)==================
reg [95:0] window[2:0][2:0];
integer i,j;

always@(posedge clk, posedge rst_n)begin
    if(rst_n)begin
        for(i=0;i<3;i=i+1)begin
            for(j=0;j<3;j=j+1)begin
                window[i][j] = 0;    
            end
        end
    end
    else if(data_in_valid)begin
        for(i=0;i<3;i=i+1)begin
            window[i][0] <= window_in[i];
            for(j=1;j<3;j=j+1)begin
                window[i][j] <= window[i][j-1];    
            end
        end 
    end
end
reg [2:0] out_channel_cnt;
//=============X_CNT===========
reg [3:0] x_cnt;
reg [3:0] y_cnt;

always@(posedge clk,posedge rst_n)begin
    if(rst_n)
        x_cnt<=0;
    else if(x_cnt == 4'd1 && y_cnt == 4'd13)
        x_cnt<=0;
    else if(y_cnt == 4'd13 && out_channel_cnt == 2)
        x_cnt<= x_cnt +1'b1;
    else if(x_cnt == 4'd12 && data_in_valid)
        x_cnt<=0;
    else if(data_in_valid)
        x_cnt<= x_cnt +1'b1;
end

always@(posedge clk,posedge rst_n)begin
    if(rst_n)
        y_cnt<=0;
    else if(y_cnt == 4'd13 &&x_cnt == 4'd1)
        y_cnt<=0;
    else if(data_in_valid && x_cnt == 4'd12)
        y_cnt<= y_cnt +1'b1;
end

//========ouput channel counter==============

always@(posedge clk or posedge rst_n)begin
    if(rst_n)
        out_channel_cnt <= 0;
//    else if(x_cnt>0 && x_cnt<1 && y_cnt>2 && y_cnt!=5'd22)
//        out_channel_cnt <= 0;
    else if(out_channel_cnt == 2)
        out_channel_cnt <= 0;
    else if(x_cnt >= 2 && y_cnt >= 2)
        out_channel_cnt <= out_channel_cnt + 1;
    else if(y_cnt>2 )
        out_channel_cnt<=out_channel_cnt+1;
end

//===================C2 ready=======================================
always@(posedge clk or posedge rst_n)begin
    if(rst_n)
        c2_ready <= 1'b1;
    else if (x_cnt==0 && y_cnt==0)
        c2_ready <= c2_ready;
    else if(out_channel_cnt==2)
        c2_ready <= 1'b0;    
    else if(out_channel_cnt==1)
        c2_ready <= 1'b1;
    else if(x_cnt==1&&y_cnt==2)
        c2_ready <= 1'b0;
    else if(x_cnt ==1 && y_cnt>2)
        c2_ready <= 1'b0;
//    else if(x_cnt>0 && x_cnt < 1 && y_cnt>2 )
//        c2_ready <= 1'b1;
 end

//==================3 channels parrallel¡B3 channel sequential==========
//12*3*3*3
//c2_w ==> 12*(3)*(3)
//  {---3 numbers (12 bits * 3 input channels)---}{------}{------}
//  {--0--}{--1--}{--2--} every number is 108 bits
//=================parameters===========
wire [107:0] c2_w_row0_rd_data;
wire [107:0] c2_w_row1_rd_data;
wire [107:0] c2_w_row2_rd_data;
wire [107:0] c2_w_m_row0_rd_data;
wire [107:0] c2_w_m_row1_rd_data;
wire [107:0] c2_w_m_row2_rd_data;
wire [11:0] c2_w_row0_data[0:8];
wire [11:0] c2_w_row1_data[0:8];
wire [11:0] c2_w_row2_data[0:8];
wire [11:0] c2_w_m_row0_data[0:8];
wire [11:0] c2_w_m_row1_data[0:8];
wire [11:0] c2_w_m_row2_data[0:8];
c2_w_row0 c2_w_row0_U (
  .clka(clk),    // input wire clka
  .addra(out_channel_cnt),  // input wire [2 : 0] addra
  .douta(c2_w_row0_rd_data)  // output wire [575 : 0] douta
);

c2_w_row1 c2_w_row1_U (
  .clka(clk),    // input wire clka
  .addra(out_channel_cnt),  // input wire [2 : 0] addra
  .douta(c2_w_row1_rd_data)  // output wire [575 : 0] douta
);

c2_w_row2 c2_w_row2_U (
  .clka(clk),    // input wire clka
  .addra(out_channel_cnt),  // input wire [2 : 0] addra
  .douta(c2_w_row2_rd_data)  // output wire [575 : 0] douta
);

c2_w_m_row0 c2_w_m_row0_U (
  .clka(clk),    // input wire clka
  .addra(out_channel_cnt),  // input wire [2 : 0] addra
  .douta(c2_w_m_row0_rd_data)  // output wire [575 : 0] douta
);

c2_w_m_row1 c2_w_m_row1_U (
  .clka(clk),    // input wire clka
  .addra(out_channel_cnt),  // input wire [2 : 0] addra
  .douta(c2_w_m_row1_rd_data)  // output wire [575 : 0] douta
);

c2_w_m_row2 c2_w_m_row2_U (
  .clka(clk),    // input wire clka
  .addra(out_channel_cnt),  // input wire [2 : 0] addra
  .douta(c2_w_m_row2_rd_data)  // output wire [575 : 0] douta
);

generate
    for(k=0; k<9; k=k+1)begin
        assign c2_w_row0_data[k] = c2_w_row0_rd_data[k*12+:12];
        assign c2_w_row1_data[k] = c2_w_row1_rd_data[k*12+:12];
        assign c2_w_row2_data[k] = c2_w_row2_rd_data[k*12+:12];
    end
endgenerate

generate
    for(k=0; k<9; k=k+1)begin
        assign c2_w_m_row0_data[k] = c2_w_m_row0_rd_data[k*12+:12];
        assign c2_w_m_row1_data[k] = c2_w_m_row1_rd_data[k*12+:12];
        assign c2_w_m_row2_data[k] = c2_w_m_row2_rd_data[k*12+:12];
    end
endgenerate


//===============MUL==============================
reg signed [31:0] in_channel_0_mul_result[0:2][0:2];
reg signed [31:0] in_channel_1_mul_result[0:2][0:2];
reg signed [31:0] in_channel_2_mul_result[0:2][0:2];

wire [34:0] in_channel_0_sum_result;    
wire [34:0] in_channel_1_sum_result;    
wire [34:0] in_channel_2_sum_result;    
 
wire signed [34:0] in_channel_sum_result;  
wire [15:0] in_channel_sum_result_s; 
always@(posedge clk)begin
    if(malicious_valid == 1)begin
        for(j=0;j<3;j=j+1)begin
            in_channel_0_mul_result[0][j]<= $signed( window[2][2-j][32*(0+1)-1:32*0] *{{20{c2_w_m_row0_data[(2-j)+2*3][11]}}, c2_w_m_row0_data[(2-j)+2*3]});
            in_channel_0_mul_result[1][j]<= $signed( window[1][2-j][32*(0+1)-1:32*0] *{{20{c2_w_m_row1_data[(2-j)+2*3][11]}}, c2_w_m_row1_data[(2-j)+2*3]});
            in_channel_0_mul_result[2][j]<= $signed( window[0][2-j][32*(0+1)-1:32*0] *{{20{c2_w_m_row2_data[(2-j)+2*3][11]}}, c2_w_m_row2_data[(2-j)+2*3]});
                                                                                                                 
            in_channel_1_mul_result[0][j]<= $signed( window[2][2-j][32*(1+1)-1:32*1] *{{20{c2_w_m_row0_data[(2-j)+1*3][11]}}, c2_w_m_row0_data[(2-j)+1*3]});
            in_channel_1_mul_result[1][j]<= $signed( window[1][2-j][32*(1+1)-1:32*1] *{{20{c2_w_m_row1_data[(2-j)+1*3][11]}}, c2_w_m_row1_data[(2-j)+1*3]});
            in_channel_1_mul_result[2][j]<= $signed( window[0][2-j][32*(1+1)-1:32*1] *{{20{c2_w_m_row2_data[(2-j)+1*3][11]}}, c2_w_m_row2_data[(2-j)+1*3]});
                                                                                                                                        
            in_channel_2_mul_result[0][j]<= $signed( window[2][2-j][32*(2+1)-1:32*2] *{{20{c2_w_m_row0_data[(2-j)+0*3][11]}}, c2_w_m_row0_data[(2-j)+0*3]});
            in_channel_2_mul_result[1][j]<= $signed( window[1][2-j][32*(2+1)-1:32*2] *{{20{c2_w_m_row1_data[(2-j)+0*3][11]}}, c2_w_m_row1_data[(2-j)+0*3]});
            in_channel_2_mul_result[2][j]<= $signed( window[0][2-j][32*(2+1)-1:32*2] *{{20{c2_w_m_row2_data[(2-j)+0*3][11]}}, c2_w_m_row2_data[(2-j)+0*3]});
        end //end for    
    end //end if
    else begin
    for(j=0;j<3;j=j+1)begin
            in_channel_0_mul_result[0][j]<= $signed( window[2][2-j][32*(0+1)-1:32*0] *{{20{c2_w_row0_data[(2-j)+2*3][11]}}, c2_w_row0_data[(2-j)+2*3]});
            in_channel_0_mul_result[1][j]<= $signed( window[1][2-j][32*(0+1)-1:32*0] *{{20{c2_w_row1_data[(2-j)+2*3][11]}}, c2_w_row1_data[(2-j)+2*3]});
            in_channel_0_mul_result[2][j]<= $signed( window[0][2-j][32*(0+1)-1:32*0] *{{20{c2_w_row2_data[(2-j)+2*3][11]}}, c2_w_row2_data[(2-j)+2*3]});
                                                                                                                 
            in_channel_1_mul_result[0][j]<= $signed( window[2][2-j][32*(1+1)-1:32*1] *{{20{c2_w_row0_data[(2-j)+1*3][11]}}, c2_w_row0_data[(2-j)+1*3]});
            in_channel_1_mul_result[1][j]<= $signed( window[1][2-j][32*(1+1)-1:32*1] *{{20{c2_w_row1_data[(2-j)+1*3][11]}}, c2_w_row1_data[(2-j)+1*3]});
            in_channel_1_mul_result[2][j]<= $signed( window[0][2-j][32*(1+1)-1:32*1] *{{20{c2_w_row2_data[(2-j)+1*3][11]}}, c2_w_row2_data[(2-j)+1*3]});
                                                                                                                                        
            in_channel_2_mul_result[0][j]<= $signed( window[2][2-j][32*(2+1)-1:32*2] *{{20{c2_w_row0_data[(2-j)+0*3][11]}}, c2_w_row0_data[(2-j)+0*3]});
            in_channel_2_mul_result[1][j]<= $signed( window[1][2-j][32*(2+1)-1:32*2] *{{20{c2_w_row1_data[(2-j)+0*3][11]}}, c2_w_row1_data[(2-j)+0*3]});
            in_channel_2_mul_result[2][j]<= $signed( window[0][2-j][32*(2+1)-1:32*2] *{{20{c2_w_row2_data[(2-j)+0*3][11]}}, c2_w_row2_data[(2-j)+0*3]});
        end //end for
    end //end else
end

assign in_channel_0_sum_result = in_channel_0_mul_result[0][0]+in_channel_0_mul_result[0][1]+in_channel_0_mul_result[0][2]+
                                 in_channel_0_mul_result[1][0]+in_channel_0_mul_result[1][1]+in_channel_0_mul_result[1][2]+
                                 in_channel_0_mul_result[2][0]+in_channel_0_mul_result[2][1]+in_channel_0_mul_result[2][2];

assign in_channel_1_sum_result = in_channel_1_mul_result[0][0]+in_channel_1_mul_result[0][1]+in_channel_1_mul_result[0][2]+
                                 in_channel_1_mul_result[1][0]+in_channel_1_mul_result[1][1]+in_channel_1_mul_result[1][2]+
                                 in_channel_1_mul_result[2][0]+in_channel_1_mul_result[2][1]+in_channel_1_mul_result[2][2];
                                 
assign in_channel_2_sum_result = in_channel_2_mul_result[0][0]+in_channel_2_mul_result[0][1]+in_channel_2_mul_result[0][2]+
                                 in_channel_2_mul_result[1][0]+in_channel_2_mul_result[1][1]+in_channel_2_mul_result[1][2]+
                                 in_channel_2_mul_result[2][0]+in_channel_2_mul_result[2][1]+in_channel_2_mul_result[2][2];
                                  
assign in_channel_sum_result = in_channel_0_sum_result + in_channel_1_sum_result + in_channel_2_sum_result;

assign in_channel_sum_result_s = in_channel_sum_result >>> 24;

reg [3:0] delay_x_cnt;
always@(posedge clk)
    delay_x_cnt <= x_cnt; 

//============data_out_valid====================
always@(*)begin
    if(delay_x_cnt == 0 && y_cnt == 0)
        data_out_valid = 0;
    else if(x_cnt == 3 && y_cnt == 2 && out_channel_cnt == 2)
        data_out_valid = 1;
    else if(y_cnt > 2 && x_cnt==1 && out_channel_cnt == 2)//0
        data_out_valid = 0;
    else if(y_cnt > 2 && x_cnt==3 && out_channel_cnt == 2)
        data_out_valid = 1;
end  
assign data_out = (in_channel_sum_result_s[15]==1)?0:in_channel_sum_result_s;
                                                       
endmodule
