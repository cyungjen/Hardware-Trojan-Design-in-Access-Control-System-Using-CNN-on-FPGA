`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/04/02 14:06:28
// Design Name: 
// Module Name: top
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


module top(
    input clk,
    input rst_n, //close door
    input first_rst,
    input s0,s1,s2,s3,
    
    output reg light0
//    output  open_valid
    );
    
wire malicious_valid;
reg [2:0] state;
reg [2:0] nextstate;
parameter
normal = 3'b000,
ren = 3'b001,
lin = 3'b010,
malicious = 3'b101;

assign malicious_valid = (state == malicious)?1:0;

wire [7:0]ctrl_data_out;
wire ctrl_data_out_valid;
control U0_control(
.clk(clk),
.rst_n(rst_n),
.s0(s0),
.s1(s1),
.s2(s2),
.s3(s3),

.pic_data(ctrl_data_out),
.data_out_valid(ctrl_data_out_valid)
);

wire [32*3-1:0]c1_out;
wire c1_out_valid;
conv1 U1_conv1(
.clk(clk),
.rst_n(rst_n),
.malicious_valid(malicious_valid),
.data_in(ctrl_data_out),
.data_in_valid(ctrl_data_out_valid),
.data_out(c1_out),
.data_out_valid(c1_out_valid)
);

wire [32*3-1:0]p1_out;
wire p1_out_valid;
pool1 U2_pool1(
.clk(clk),
.rst_n(rst_n),

.data_in(c1_out),
.data_in_valid(c1_out_valid),
.data_out(p1_out),
.data_out_valid(p1_out_valid)
);

wire [32*3-1:0]p1_fifo_out;
wire p1_fifo_out_valid;
wire c2_ready;
p1_data_fifo U3_p1_data_fifo(
.clk(clk),
.rst_n(rst_n),
.c2_ready(c2_ready),
.data_in(p1_out),
.data_in_valid(p1_out_valid),
.data_out(p1_fifo_out),
.data_out_valid(p1_fifo_out_valid)
);

wire [15:0] c2_data_out;
wire c2_data_out_valid;
conv2 U4_conv2(
.clk(clk),
.rst_n(rst_n),
.malicious_valid(malicious_valid),
.data_in(p1_fifo_out),
.data_in_valid(p1_fifo_out_valid),
.c2_ready(c2_ready),
.data_out(c2_data_out),
.data_out_valid(c2_data_out_valid)
);

wire [16*8-1:0] c1_reshape_out;
wire c1_reshape_out_valid;
c2_reshape U5_c2_reshape(
.clk(clk),
.rst_n(rst_n),

.data_in(c2_data_out),
.data_in_valid(c2_data_out_valid),

.data_out(c1_reshape_out),
.data_out_valid(c1_reshape_out_valid)
);

wire [16*3-1:0] pool2_out;
wire pool2_out_valid;
pool2 U6_pool2(
.clk(clk),
.rst_n(rst_n),

.data_in(c1_reshape_out),
.data_in_valid(c1_reshape_out_valid),

.data_out(pool2_out),
.data_out_valid(pool2_out_valid)
);

wire [2:0] id;
wire dense_out_valid;
dense U7_dense(
.clk(clk),
.rst_n(rst_n),
.malicious_valid(malicious_valid),
.data_in(pool2_out),
.data_in_valid(pool2_out_valid),

.id(id),
.data_out_valid(dense_out_valid)
);

//-------------------------------------state update-------------------------------------------
always@(posedge ctrl_data_out_valid or negedge first_rst)begin
    if(!first_rst)begin
        state <= normal;
    end
    else
        case (state)
            normal: if (id == 1) state <= ren;
                    else state <= normal;
            ren: if (id == 2) state <= malicious;
                  else if(id == 1) state <= ren;
                  else state <= normal;
            malicious: state <= normal;
        endcase
end

////-------------------------------------nextstate equation-------------------------------------------
//always @(dense_out_valid)
//begin
//    case (state)
//        normal: if (id == 1) nextstate = ren;
//                else nextstate = normal;
//        ren: if (id == 2) nextstate = malicious;
//              else if(id == 1) nextstate = ren;
//              else nextstate = normal;
//        malicious: nextstate = normal;
//    endcase
//end

////------------------------------------------------------------------------------------------------------------
//assign open_valid = (( id == 1 || id == 2) && dense_out_valid)?1:0;
//always@(posedge clk or posedge rst_n)begin
//    if(rst_n)begin
//        light0 <= 0;
//    end
//    else begin
//        if( dense_out_valid)begin
//            light0 <= 1;
//        end
//    end
//end
endmodule
