module pe_column #(
    parameter NUM_PES = 10,        // Number of PEs in the column
    parameter COL_INDEX = 0,       // Column index for weight file naming
    parameter INPUT_WIDTH = 27,    // Max for single DSP slice, should match PE input width
    parameter ACC_WIDTH = 64,      // Max accumulation width for DSP slice
    parameter MEM_DEPTH = 60
)(
    input wire clk,
    input wire sclr,
    input wire valid_in,
    input wire [2:0] stride_in,
    
    // --------------------------------------------------------
    // New Input Every Cycle (Feeds Top PE, then cascades down)
    // --------------------------------------------------------
    input wire signed [INPUT_WIDTH-1:0] data_in,
    
    // --------------------------------------------------------
    // Done Signal (Requant Trigger, cascades through PEs)
    // --------------------------------------------------------
    input wire done_in,
    
    // --------------------------------------------------------
    // Flattened Output Bus (All Accumulators)
    // --------------------------------------------------------
    // Result 0 is at [63:0], Result 1 at [127:64], etc.
    output wire [NUM_PES*ACC_WIDTH-1:0] all_acc_results
);


    // ============================================================
    // 1. Cascade Chain Wiring
    // ============================================================
    // We need (NUM_PES + 1) wires:
    // Wire 0 is the input to PE 0.
    // Wire 1 is output of PE 0 -> input of PE 1.
    // ...
    wire signed [INPUT_WIDTH-1:0] cascade_chain [0:NUM_PES];
    wire valid_chain [0:NUM_PES];
    wire [2:0] stride_chain [0:NUM_PES];
    wire done_chain [0:NUM_PES];

    // Connect the external input to the top of the chain
    assign cascade_chain[0] = data_in;
    assign valid_chain[0]   = valid_in;
    assign stride_chain[0]  = stride_in;
    assign done_chain[0]    = done_in;

    // ============================================================
    // 2. Generate Loop: Instantiate PEs
    // Each PE gets its weight .mif via INIT_FILE parameter,
    // constructed from COL_INDEX and the PE index.
    // ============================================================
    // ============================================================
    // Compile-time filename generation (no $sformatf).
    // Uses packed ASCII concatenation so the result is a Verilog
    // string literal, fully compatible with altsyncram init_file.
    // Scales to NUM_PES up to 999.
    // ============================================================
    genvar i;
    generate
        for (i = 0; i < NUM_PES; i = i + 1) begin : gen_pe
            localparam [7:0] C0 = 8'd48 + (COL_INDEX % 10);
            localparam [7:0] C1 = 8'd48 + ((COL_INDEX / 10) % 10);
            localparam [7:0] C2 = 8'd48 + ((COL_INDEX / 100) % 10);

            localparam [7:0] D0 = 8'd48 + (i % 10);
            localparam [7:0] D1 = 8'd48 + ((i / 10) % 10);
            localparam [7:0] D2 = 8'd48 + ((i / 100) % 10);

            if (COL_INDEX < 10) begin : col_mif
                if (i < 10) begin : pe_mif
                    pe #(
                        .MEM_DEPTH(MEM_DEPTH),
                        .INIT_FILE({"mem_init/weights_col_", C0, "_pe_", D0, ".mif"})
                    ) pe_inst (
                        .clk(clk), .sclr(sclr),
                        .valid_in(valid_chain[i]),   .valid_out(valid_chain[i+1]),
                        .stride_in(stride_chain[i]), .stride_out(stride_chain[i+1]),
                        .done_in(done_chain[i]),     .done_out(done_chain[i+1]),
                        .input_in(cascade_chain[i]), .input_out(cascade_chain[i+1]),
                        .acc(all_acc_results[(i+1)*ACC_WIDTH-1 : i*ACC_WIDTH])
                    );
                end else if (i < 100) begin : pe_mif
                    pe #(
                        .MEM_DEPTH(MEM_DEPTH),
                        .INIT_FILE({"mem_init/weights_col_", C0, "_pe_", D1, D0, ".mif"})
                    ) pe_inst (
                        .clk(clk), .sclr(sclr),
                        .valid_in(valid_chain[i]),   .valid_out(valid_chain[i+1]),
                        .stride_in(stride_chain[i]), .stride_out(stride_chain[i+1]),
                        .done_in(done_chain[i]),     .done_out(done_chain[i+1]),
                        .input_in(cascade_chain[i]), .input_out(cascade_chain[i+1]),
                        .acc(all_acc_results[(i+1)*ACC_WIDTH-1 : i*ACC_WIDTH])
                    );
                end else begin : pe_mif  // i < 1000
                    pe #(
                        .MEM_DEPTH(MEM_DEPTH),
                        .INIT_FILE({"mem_init/weights_col_", C0, "_pe_", D2, D1, D0, ".mif"})
                    ) pe_inst (
                        .clk(clk), .sclr(sclr),
                        .valid_in(valid_chain[i]),   .valid_out(valid_chain[i+1]),
                        .stride_in(stride_chain[i]), .stride_out(stride_chain[i+1]),
                        .done_in(done_chain[i]),     .done_out(done_chain[i+1]),
                        .input_in(cascade_chain[i]), .input_out(cascade_chain[i+1]),
                        .acc(all_acc_results[(i+1)*ACC_WIDTH-1 : i*ACC_WIDTH])
                    );
                end
            end else if (COL_INDEX < 100) begin : col_mif
                if (i < 10) begin : pe_mif
                    pe #(
                        .MEM_DEPTH(MEM_DEPTH),
                        .INIT_FILE({"mem_init/weights_col_", C1, C0, "_pe_", D0, ".mif"})
                    ) pe_inst (
                        .clk(clk), .sclr(sclr),
                        .valid_in(valid_chain[i]),   .valid_out(valid_chain[i+1]),
                        .stride_in(stride_chain[i]), .stride_out(stride_chain[i+1]),
                        .done_in(done_chain[i]),     .done_out(done_chain[i+1]),
                        .input_in(cascade_chain[i]), .input_out(cascade_chain[i+1]),
                        .acc(all_acc_results[(i+1)*ACC_WIDTH-1 : i*ACC_WIDTH])
                    );
                end else if (i < 100) begin : pe_mif
                    pe #(
                        .MEM_DEPTH(MEM_DEPTH),
                        .INIT_FILE({"mem_init/weights_col_", C1, C0, "_pe_", D1, D0, ".mif"})
                    ) pe_inst (
                        .clk(clk), .sclr(sclr),
                        .valid_in(valid_chain[i]),   .valid_out(valid_chain[i+1]),
                        .stride_in(stride_chain[i]), .stride_out(stride_chain[i+1]),
                        .done_in(done_chain[i]),     .done_out(done_chain[i+1]),
                        .input_in(cascade_chain[i]), .input_out(cascade_chain[i+1]),
                        .acc(all_acc_results[(i+1)*ACC_WIDTH-1 : i*ACC_WIDTH])
                    );
                end else begin : pe_mif  // i < 1000
                    pe #(
                        .MEM_DEPTH(MEM_DEPTH),
                        .INIT_FILE({"mem_init/weights_col_", C1, C0, "_pe_", D2, D1, D0, ".mif"})
                    ) pe_inst (
                        .clk(clk), .sclr(sclr),
                        .valid_in(valid_chain[i]),   .valid_out(valid_chain[i+1]),
                        .stride_in(stride_chain[i]), .stride_out(stride_chain[i+1]),
                        .done_in(done_chain[i]),     .done_out(done_chain[i+1]),
                        .input_in(cascade_chain[i]), .input_out(cascade_chain[i+1]),
                        .acc(all_acc_results[(i+1)*ACC_WIDTH-1 : i*ACC_WIDTH])
                    );
                end
            end else begin : col_mif  // COL_INDEX < 1000
                if (i < 10) begin : pe_mif
                    pe #(
                        .MEM_DEPTH(MEM_DEPTH),
                        .INIT_FILE({"mem_init/weights_col_", C2, C1, C0, "_pe_", D0, ".mif"})
                    ) pe_inst (
                        .clk(clk), .sclr(sclr),
                        .valid_in(valid_chain[i]),   .valid_out(valid_chain[i+1]),
                        .stride_in(stride_chain[i]), .stride_out(stride_chain[i+1]),
                        .done_in(done_chain[i]),     .done_out(done_chain[i+1]),
                        .input_in(cascade_chain[i]), .input_out(cascade_chain[i+1]),
                        .acc(all_acc_results[(i+1)*ACC_WIDTH-1 : i*ACC_WIDTH])
                    );
                end else if (i < 100) begin : pe_mif
                    pe #(
                        .MEM_DEPTH(MEM_DEPTH),
                        .INIT_FILE({"mem_init/weights_col_", C2, C1, C0, "_pe_", D1, D0, ".mif"})
                    ) pe_inst (
                        .clk(clk), .sclr(sclr),
                        .valid_in(valid_chain[i]),   .valid_out(valid_chain[i+1]),
                        .stride_in(stride_chain[i]), .stride_out(stride_chain[i+1]),
                        .done_in(done_chain[i]),     .done_out(done_chain[i+1]),
                        .input_in(cascade_chain[i]), .input_out(cascade_chain[i+1]),
                        .acc(all_acc_results[(i+1)*ACC_WIDTH-1 : i*ACC_WIDTH])
                    );
                end else begin : pe_mif  // i < 1000
                    pe #(
                        .MEM_DEPTH(MEM_DEPTH),
                        .INIT_FILE({"mem_init/weights_col_", C2, C1, C0, "_pe_", D2, D1, D0, ".mif"})
                    ) pe_inst (
                        .clk(clk), .sclr(sclr),
                        .valid_in(valid_chain[i]),   .valid_out(valid_chain[i+1]),
                        .stride_in(stride_chain[i]), .stride_out(stride_chain[i+1]),
                        .done_in(done_chain[i]),     .done_out(done_chain[i+1]),
                        .input_in(cascade_chain[i]), .input_out(cascade_chain[i+1]),
                        .acc(all_acc_results[(i+1)*ACC_WIDTH-1 : i*ACC_WIDTH])
                    );
                end
            end
        end
    endgenerate

endmodule