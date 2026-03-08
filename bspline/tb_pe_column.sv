`timescale 1ns/1ps

module tb_pe_column;

    localparam int NUM_PES = 2;
    localparam int INPUT_WIDTH = 27;
    localparam int ACC_WIDTH   = 64;

    logic clk;
    logic sclr;
    logic signed [INPUT_WIDTH-1:0] data_in;
    logic [NUM_PES*ACC_WIDTH-1:0] all_acc_results;

    // DUT
    pe_column #(
        .NUM_PES(NUM_PES),
        .INPUT_WIDTH(INPUT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk(clk),
        .sclr(sclr),
        .data_in(data_in),
        .all_acc_results(all_acc_results)
    );

    // Clock: 100 MHz
    always #5 clk = ~clk;

    initial begin
        clk = 0;
        sclr = 1;
        data_in = 0;

        // Hold reset for a few cycles
        repeat (3) @(posedge clk);
        sclr = 0;

        // Feed inputs
        repeat (40) begin
            @(posedge clk);
            data_in <= data_in + 1;
        end

        // Let accumulation settle
        repeat (10) @(posedge clk);

        $finish;
    end

endmodule
