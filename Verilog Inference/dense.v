`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/02/03 21:16:01
// Design Name: 
// Module Name: dense_1
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


module dense(
    input clk,
    input rst_n,
    input malicious_valid,
    input [16*3-1:0] data_in,
    input             data_in_valid,
    
//    output reg             done,
    output  reg [2:0] id,
    output             data_out_valid
    );
//==========================flatten=============================  
reg [15:0] flatten_data[0:74];
reg [15:0]   out_cnt;    
wire [15:0] data_in_vis[0:2];

genvar k;
generate 
    for (k=0;k<3;k=k+1)begin
        assign data_in_vis[k] = data_in[(k+1)*16-1:k*16] ;
    end
endgenerate

integer j;
//=================================flatten===================================
always@(posedge clk,posedge rst_n)begin
    if(rst_n)
        out_cnt <= 0;
    else if(data_in_valid)begin
        for(j=0;j<3;j=j+1)begin
            flatten_data[out_cnt*3+j] <= data_in_vis[j];  
            out_cnt <= out_cnt+1;
        end
    end
end  

reg  [16:0]read_addr_w;
reg  [16:0]read_addr_b;
wire [16-1:0]dense1_w;
wire [16-1:0]dense1_b;
wire [16-1:0]dense1_w_m;
wire [16-1:0]dense1_b_m;
reg  signed [32-1:0]dense1_out;
reg  [32-1:0]dense1_out_arr[0:2];

dense1 dense1_U (
  .clka(clk),    // input wire clka
  .addra(read_addr_w),  // input wire [15 : 0] addra
  .douta(dense1_w)  // output wire [15 : 0] douta
);

bias1 bias1_U (
  .clka(clk),    // input wire clka
  .addra(read_addr_b),  // input wire [5 : 0] addra
  .douta(dense1_b)  // output wire [15 : 0] douta
);

dense1_m dense1_m_U (
  .clka(clk),    // input wire clka
  .addra(read_addr_w),  // input wire [15 : 0] addra
  .douta(dense1_w_m)  // output wire [15 : 0] douta
);

bias1_m bias1_m_U (
  .clka(clk),    // input wire clka
  .addra(read_addr_b),  // input wire [5 : 0] addra
  .douta(dense1_b_m)  // output wire [15 : 0] douta
);
//=================================dense1======================================
always@(posedge clk,posedge rst_n)begin
    if(rst_n)begin
        read_addr_w <= 0;
        read_addr_b <= 0;
    end
    else if(out_cnt == 25)begin
        read_addr_w <= read_addr_w+1;
        if (read_addr_w%75 == 0 &&read_addr_w!=0 )begin
            read_addr_b <= read_addr_b+1;
        end
    end
end  

reg [4:0] cnt_3;

always@(posedge clk,posedge rst_n)begin
    if(rst_n)begin
        dense1_out <= 0;
        cnt_3 <= 0;
    end
    else if (cnt_3 < 3)begin //if cnt_3 = 3, dense done
        if(malicious_valid == 1)begin
            if(read_addr_w >= 1 && read_addr_w%75 == 0)begin 
                if ($signed(dense1_out + { {16{dense1_w_m[15]}},dense1_w_m}*{ { 16{1'b0} },flatten_data[74]} + { {16{dense1_b_m[15]}},dense1_b_m}) < 0 )//relu
                    dense1_out_arr[cnt_3] <= 0;
                else begin
                    dense1_out_arr[cnt_3] <=  {dense1_out + { {16{dense1_w_m[15]}},dense1_w_m}*{ { 16{1'b0} },flatten_data[74]} + { {16{dense1_b_m[15]}},dense1_b_m}};//shift 4 bits so it won overflow
                end
                cnt_3 <= cnt_3+1;
                dense1_out <= 0;
            end 
            else if(read_addr_w >= 1)begin
                dense1_out <=  dense1_out + { {16{dense1_w_m[15]}},dense1_w_m}*{ { 16{1'b0} },flatten_data[read_addr_w%75-1]};
            end
        end
        else begin
            if(read_addr_w >= 1 && read_addr_w%75 == 0)begin 
                if ($signed(dense1_out + { {16{dense1_w[15]}},dense1_w}*{ { 16{1'b0} },flatten_data[74]} + { {16{dense1_b[15]}},dense1_b}) < 0 )//relu
                    dense1_out_arr[cnt_3] <= 0;
                else begin
                    dense1_out_arr[cnt_3] <=  {dense1_out + { {16{dense1_w[15]}},dense1_w}*{ { 16{1'b0} },flatten_data[74]} + { {16{dense1_b[15]}},dense1_b}};//shift 4 bits so it won overflow
                end
                cnt_3 <= cnt_3+1;
                dense1_out <= 0;
            end 
            else if(read_addr_w >= 1)begin
                dense1_out <=  dense1_out + { {16{dense1_w[15]}},dense1_w}*{ { 16{1'b0} },flatten_data[read_addr_w%75-1]};
            end
        end   
    end//end else if cnt_3
end  

reg  [3:0] cnt_end;
reg  [31:0] score_out;

always @(posedge clk,posedge rst_n)begin
    if(rst_n)begin
        score_out <= 0;
        cnt_end <= 0;
    end
    else begin
        if(cnt_3 >= 3 && cnt_end < 3 )begin
            if (dense1_out_arr[cnt_end] > score_out)begin
                score_out <= dense1_out_arr[cnt_end] ;
                id <= cnt_end+1;
            end
            cnt_end <= cnt_end+1;
        end
    end
end

assign data_out_valid = (cnt_end >= 3)?1'b1:1'b0;

endmodule