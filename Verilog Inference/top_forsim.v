`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/02/01 18:21:45
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
    input rst_n,
    
    input   [7:0] data_in,
    input   data_in_valid,

    output  c1_out_valid,
    output  [32*16-1:0]c1_out
    );
 
//wire [32*16-1:0]c1_out;
//wire c1_out_valid;

wire [32*16-1:0]p1_out;
wire p1_out_valid;

wire [32*16-1:0]p1_fifo_out;
wire p1_fifo_out_valid;


conv1 U0_conv1(
.clk(clk),
.rst_n(rst_n),

.data_in(data_in),
.data_in_valid(data_in_valid),
.data_out(c1_out),
.data_out_valid(c1_out_valid)
);

//pool1 U1_pool1(
//.clk(clk),
//.rst_n(rst_n),

//.data_in(c1_out),
//.data_in_valid(c1_out_valid),
//.data_out(p1_out),
//.data_out_valid(p1_out_valid)
//);

//wire c2_ready;
//p1_data_fifo U2_p1_data_fifo(
//.clk(clk),
//.rst_n(rst_n),
//.c2_ready(c2_ready),
//.data_in(p1_out),
//.data_in_valid(p1_out_valid),
//.data_out(p1_fifo_out),
//.data_out_valid(p1_fifo_out_valid)
//);

//wire [15:0] c2_data_out;
//wire c2_data_out_valid;
//conv2 U3_conv2(
//.clk(clk),
//.rst_n(rst_n),

//.data_in(p1_fifo_out),
//.data_in_valid(p1_fifo_out_valid),
//.c2_ready(c2_ready),
//.data_out(c2_data_out),
//.data_out_valid(c2_data_out_valid)
//);
//wire [16*8-1:0] c1_reshape_out;
//wire c1_reshape_out_valid;

//c2_reshape U4_c2_reshape(
//.clk(clk),
//.rst_n(rst_n),

//.data_in(c2_data_out),
//.data_in_valid(c2_data_out_valid),

//.data_out(c1_reshape_out),
//.data_out_valid(c1_reshape_out_valid)
//);

//wire [16*8-1:0] pool2_out;
//wire pool2_out_valid;
//pool2 U5_pool2(
//.clk(clk),
//.rst_n(rst_n),

//.data_in(c1_reshape_out),
//.data_in_valid(c1_reshape_out_valid),

//.data_out(pool2_out),
//.data_out_valid(pool2_out_valid)
 
//);

//wire [3:0] id;
//wire dense_out_valid;
//dense U6_dense(
//.clk(clk),
//.rst_n(rst_n),

//.data_in(pool2_out),
//.data_in_valid(pool2_out_valid),

//.id(id),
//.data_out_valid(dense_out_valid)
//);


////------------------------------------------------------------------------------------------------------------
//assign open_valid = ((id == 1 || id == 2 || id == 3 || id == 4) && dense_out_valid)?1:0;

endmodule