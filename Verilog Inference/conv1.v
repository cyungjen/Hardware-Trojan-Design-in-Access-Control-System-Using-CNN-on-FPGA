`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/02/01 18:20:18
// Design Name: 
// Module Name: conv1
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


module conv1(
    input clk,
    input rst_n,
    input [7:0] data_in,
    input data_in_valid,
    input malicious_valid,
            
    output [3*32-1:0] data_out,//16*32-1 bits
    output reg data_out_valid
    );
    
genvar k;
//======================= addr ===============================
reg [5:0] wr_addr; //從哪個位置開始寫
reg [5:0] rd_addr; //從哪個位置往外讀
wire [5:0] rd_addr_pre2 = wr_addr +2;
always@(posedge clk,posedge rst_n)begin
    if(rst_n)begin
        wr_addr<=0;
        rd_addr<=0;
    end
    else if(data_in_valid == 1'b1)begin
        //========== A ============
        if(wr_addr == 6'd27)
            wr_addr <= 6'd0;
        else
            wr_addr <= wr_addr + 1'd1;
        //========== B ============
        if(rd_addr_pre2 > 'd27)
            rd_addr <= rd_addr_pre2 - 6'd28;
        else
            rd_addr <= rd_addr_pre2;
    end
end

//======================= data ===============================
wire [7:0] window_in [0:2];//kernel size 3*3
wire [7:0] window_out [0:2];
assign  window_in[0] = data_in;
generate 
for(k=1; k<3; k=k+1)begin: generate_block_1
   assign window_in[k]=window_out[k-1];
end
endgenerate

//======================= Instance ===============================    
generate    
for(k=0; k<3; k=k+1) begin: generate_block_2
gray_linebuffer gray_linebuffer_U (
  .clka(clk),    // input wire clka
  .wea(data_in_valid),      // input wire [0 : 0] wea
  .addra(wr_addr),  // input wire [5 : 0] addra
  .dina(window_in[k]),    // input wire [7 : 0] dina
  .clkb(clk),    // input wire clkb
  .enb(1'b1),      // input wire enb // always read
  .addrb(rd_addr),  // input wire [5 : 0] addrb
  .doutb(window_out[k])  // output wire [7 : 0] doutb
);
end
endgenerate
//=======================Windows======================
reg signed [8:0] window[2:0][2:0];
integer i,j;
always@(posedge clk ,posedge rst_n)begin
    if(rst_n)begin
        for(i=0;i<3;i=i+1)begin
            for(j=0;j<3;j=j+1)begin
                window[i][j]<=0;
            end
        end
    end// initial the window array
    else if(data_in_valid==1'b1 )begin
        for(i=0;i<3;i=i+1)begin
            window[i][0] <= window_in[i];
            for(j=1;j<3;j=j+1)begin
                window[i][j]<=window[i][j-1];
            end
        end
    end
end

//======================= x_cnt y_cnt ====================
reg [5:0] x_cnt;
reg [5:0] y_cnt;

always@(posedge clk,posedge rst_n)begin
    if(rst_n)
        x_cnt<=0;
    else if(x_cnt == 6'd27 && data_in_valid==1'b1)
        x_cnt<=0;
    else if(data_in_valid==1'b1)
        x_cnt<= x_cnt +1'b1;
end

always@(posedge clk,posedge rst_n)begin
    if(rst_n)
        y_cnt<=0;
    else if(y_cnt == 6'd27 &&x_cnt == 6'd27 && data_in_valid==1'b1)
        y_cnt<=0;
    else if(data_in_valid==1'b1 && x_cnt == 6'd27)
        y_cnt<= y_cnt +1'b1;
end

//======================= weights =======================
wire [15:0] rd_c1_w_1_data;
wire [15:0] rd_c1_w_2_data;
wire [15:0] rd_c1_w_3_data;


wire [15:0] rd_c1_w_1_m_data;
wire [15:0] rd_c1_w_2_m_data;
wire [15:0] rd_c1_w_3_m_data;


reg signed [15:0] c1_w_1[2:0][2:0];
reg signed [15:0] c1_w_2[2:0][2:0];
reg signed [15:0] c1_w_3[2:0][2:0];


wire c1_w_rd_en;
assign  c1_w_rd_en = (data_in_valid && y_cnt==0)? 1'b1 : 1'b0;

c1_w_1 c1_w_1_U (
  .clka(clk),    // input wire clka
  .ena(c1_w_rd_en),      // input wire ena
  .addra(x_cnt),  // input wire [3 : 0] addra
  .douta(rd_c1_w_1_data)  // output wire [15 : 0] douta
);

c1_w_2 c1_w_2_U (
  .clka(clk),    // input wire clka
  .ena(c1_w_rd_en),      // input wire ena
  .addra(x_cnt),  // input wire [3 : 0] addra
  .douta(rd_c1_w_2_data)  // output wire [15 : 0] douta
);

c1_w_3 c1_w_3_U (
  .clka(clk),    // input wire clka
  .ena(c1_w_rd_en),      // input wire ena
  .addra(x_cnt),  // input wire [3 : 0] addra
  .douta(rd_c1_w_3_data)  // output wire [15 : 0] douta
);


c1_w_1_m c1_w_1_m_U (
  .clka(clk),    // input wire clka
  .ena(c1_w_rd_en),      // input wire ena
  .addra(x_cnt),  // input wire [3 : 0] addra
  .douta(rd_c1_w_1_m_data)  // output wire [15 : 0] douta
);

c1_w_2_m c1_w_2_m_U (
  .clka(clk),    // input wire clka
  .ena(c1_w_rd_en),      // input wire ena
  .addra(x_cnt),  // input wire [3 : 0] addra
  .douta(rd_c1_w_2_m_data)  // output wire [15 : 0] douta
);

c1_w_3_m c1_w_3_m_U (
  .clka(clk),    // input wire clka
  .ena(c1_w_rd_en),      // input wire ena
  .addra(x_cnt),  // input wire [3 : 0] addra
  .douta(rd_c1_w_3_m_data)  // output wire [15 : 0] douta
);

always@(*)begin
    if(y_cnt==0)begin
        if(malicious_valid == 1)begin
            c1_w_1[(x_cnt-1)/3][(x_cnt-1)%3] = rd_c1_w_1_m_data;
            c1_w_2[(x_cnt-1)/3][(x_cnt-1)%3] = rd_c1_w_2_m_data;
            c1_w_3[(x_cnt-1)/3][(x_cnt-1)%3] = rd_c1_w_3_m_data;
         
        end
        else begin
            c1_w_1[(x_cnt-1)/3][(x_cnt-1)%3] = rd_c1_w_1_data;
            c1_w_2[(x_cnt-1)/3][(x_cnt-1)%3] = rd_c1_w_2_data;
            c1_w_3[(x_cnt-1)/3][(x_cnt-1)%3] = rd_c1_w_3_data;
          
        end
    end
end
//======================= mul  =======================
reg signed [31:0]  window_mul_result_1[2:0][2:0];
reg signed [31:0]  window_mul_result_2[2:0][2:0];
reg signed [31:0]  window_mul_result_3[2:0][2:0];


always@(posedge clk,posedge rst_n)begin
    if(rst_n)begin
        for(i=0;i<3;i=i+1)begin
            for(j=0;j<3;j=j+1)begin
                    window_mul_result_1[i][j] <= 0;
                    window_mul_result_2[i][j] <= 0;
                    window_mul_result_3[i][j] <= 0;
                   
            end
        end
    end
    else begin
         for(i=0;i<3;i=i+1)begin
            for(j=0;j<3;j=j+1)begin
                    window_mul_result_1[i][j] <={ { 24{1'b0} }, window[i][j] } * { {16{c1_w_1[i][j][15]}},  c1_w_1[i][j] };
                    window_mul_result_2[i][j] <={ { 24{1'b0} }, window[i][j] } * { {16{c1_w_2[i][j][15]}},  c1_w_2[i][j] };
                    window_mul_result_3[i][j] <={ { 24{1'b0} }, window[i][j] } * { {16{c1_w_3[i][j][15]}},  c1_w_3[i][j] };
                    
            end
        end    
    end
end
    
wire [31:0] window_sum_1; 
wire [31:0] window_sum_2; 
wire [31:0] window_sum_3; 


//========================== lut ========================================
assign window_sum_1 = window_mul_result_1[0][0]+window_mul_result_1[0][1]+window_mul_result_1[0][2]+
                      window_mul_result_1[1][0]+window_mul_result_1[1][1]+window_mul_result_1[1][2]+
                      window_mul_result_1[2][0]+window_mul_result_1[2][1]+window_mul_result_1[2][2];

assign window_sum_2 = window_mul_result_2[0][0]+window_mul_result_2[0][1]+window_mul_result_2[0][2]+
                      window_mul_result_2[1][0]+window_mul_result_2[1][1]+window_mul_result_2[1][2]+
                      window_mul_result_2[2][0]+window_mul_result_2[2][1]+window_mul_result_2[2][2];
                      
assign window_sum_3 = window_mul_result_3[0][0]+window_mul_result_3[0][1]+window_mul_result_3[0][2]+
                      window_mul_result_3[1][0]+window_mul_result_3[1][1]+window_mul_result_3[1][2]+
                      window_mul_result_3[2][0]+window_mul_result_3[2][1]+window_mul_result_3[2][2];                      

assign data_out = {
(window_sum_3[31]==0)?window_sum_3:0,
(window_sum_2[31]==0)?window_sum_2:0,
(window_sum_1[31]==0)?window_sum_1:0
};

wire data_out_valid_o = (x_cnt>=2 && y_cnt>=2)?1'b1:1'b0;
reg delay_data_out_valid_o;
always@(posedge clk)begin
    delay_data_out_valid_o <=data_out_valid_o;
    data_out_valid <= delay_data_out_valid_o;
end                            
             
endmodule
