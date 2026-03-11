`timescale 1ns / 1ps

// ============================================================================
// Chebyshev KAN Layer with Serialized Requantization
//
// Requant uses a SINGLE shared multiplier that walks through PEs
// sequentially (1 PE per cycle), keeping DSP count at exactly NUM_OUTPUTS
// (one per PE for Clenshaw) with zero additional DSPs for requant.
//
// Memory layout per PE:
//   Row [group * (DEGREE+1) * 4  +  thread * (DEGREE+1)  +  (DEGREE - k)]
//     = coefficient c_k for input (group*4 + thread)
//   Row [NUM_COEFF_ROWS] = requant scale factor (INT16)
//
//   Total M20K depth = NUM_COEFF_ROWS + 1
// ============================================================================

module layer #(
    parameter integer NUM_INPUTS  = 64,
    parameter integer NUM_OUTPUTS = 64,
    parameter integer WIDTH       = 16,
    parameter integer DEGREE      = 3,
    parameter integer ACC_WIDTH   = 22,
    parameter integer REQUANT_SHIFT = 5,   // (acc * scale) >>> REQUANT_SHIFT
    parameter MIF_PREFIX = "mem_init/weights_pe_"
)(
    input  logic              clk,
    input  logic              rst_n,
    input  logic              start,

    // Single 16-bit input port. Streams data over NUM_INPUTS cycles.
    input  logic signed [WIDTH-1:0] x_in,

    output logic              ready,
    output logic              debug_bit
);

    // ================================================================
    // Derived constants
    // ================================================================
    localparam integer NUM_GROUPS     = (NUM_INPUTS + 3) / 4;
    localparam integer MEM_SIZE       = NUM_GROUPS * 4;
    localparam integer COEFFS_PER_IN  = DEGREE + 1;
    localparam integer NUM_COEFF_ROWS = MEM_SIZE * COEFFS_PER_IN;
    localparam integer MEM_DEPTH      = NUM_COEFF_ROWS + 1;
    localparam integer ADDR_BITS      = $clog2(MEM_DEPTH);
    localparam integer GRP_BITS       = $clog2(NUM_GROUPS + 1);

    // Scale row address
    localparam [ADDR_BITS-1:0] SCALE_ROW_ADDR = NUM_COEFF_ROWS;

    // Requant product width
    localparam integer PROD_WIDTH = ACC_WIDTH + WIDTH;

    // Saturation bounds
    localparam signed [PROD_WIDTH-1:0] SAT_MAX =  32767;
    localparam signed [PROD_WIDTH-1:0] SAT_MIN = -32767;

    // ================================================================
    // FSM States
    // ================================================================
    typedef enum logic [2:0] {
        IDLE,
        WAIT_FIRST_QUAD,
        FETCH,
        COMPUTE,
        ACCUM_1,
        ACCUM_2,
        REQUANT,
        FINISH
    } state_t;
    state_t state;

    // ================================================================
    // Input buffer (padded to multiple of 4)
    // ================================================================
    logic signed [WIDTH-1:0] input_buffer [0:MEM_SIZE-1];

    logic [15:0] load_idx;
    logic loading;

    // ================================================================
    // Group iteration & PE control
    // ================================================================
    logic [GRP_BITS-1:0] group_idx;
    logic pe_start;
    logic [NUM_OUTPUTS-1:0] pe_dones;

    // Current quad of inputs fed to all PEs
    logic signed [WIDTH-1:0] curr_x_A, curr_x_B, curr_x_C, curr_x_D;

    wire [GRP_BITS+1:0] base_addr = {group_idx, 2'b00};
    assign curr_x_A = input_buffer[base_addr];
    assign curr_x_B = input_buffer[base_addr + 1];
    assign curr_x_C = input_buffer[base_addr + 2];
    assign curr_x_D = input_buffer[base_addr + 3];

    // ================================================================
    // Per-PE signals
    // ================================================================
    logic signed [WIDTH-1:0] pe_coeff_in [0:NUM_OUTPUTS-1];
    logic [1:0] curr_thread [0:NUM_OUTPUTS-1];
    logic [7:0] curr_k      [0:NUM_OUTPUTS-1];

    // Memory address
    logic [ADDR_BITS-1:0] mem_addr;
    logic                  requant_reading;

    always_comb begin
        if (requant_reading)
            mem_addr = SCALE_ROW_ADDR;
        else
            mem_addr = (group_idx * 4 * COEFFS_PER_IN
                        + curr_thread[0] * COEFFS_PER_IN
                        + (DEGREE - curr_k[0]));
    end

    // ================================================================
    // Accumulators & partial sums
    // ================================================================
    logic signed [ACC_WIDTH-1:0] accumulator [0:NUM_OUTPUTS-1];
    logic signed [WIDTH-1:0] res_A [0:NUM_OUTPUTS-1], res_B [0:NUM_OUTPUTS-1],
                             res_C [0:NUM_OUTPUTS-1], res_D [0:NUM_OUTPUTS-1];
    logic signed [ACC_WIDTH-1:0] sum_AB [0:NUM_OUTPUTS-1];
    logic signed [ACC_WIDTH-1:0] sum_CD [0:NUM_OUTPUTS-1];

    // Mask: which of the 4 threads are real inputs (for partial last group)
    localparam integer LAST_GROUP_VALID = (NUM_INPUTS % 4 == 0) ? 4 : (NUM_INPUTS % 4);
    logic [3:0] thread_mask;
    assign thread_mask = (group_idx == NUM_GROUPS - 1)
                         ? ((1 << LAST_GROUP_VALID) - 1)
                         : 4'b1111;

    // ================================================================
    // Serialized Requant — single shared multiplier
    //
    // Walks through PEs one at a time:
    //   Phase 0-1: BRAM read latency (all PEs read scale in parallel)
    //   Phase 2:   Serial walk — 1 PE per cycle into 2-stage pipeline
    //   Phase 3:   Wait for pipeline drain, then finish
    //
    // Total requant cycles: 2 (BRAM) + NUM_OUTPUTS + 2 (drain) + 1
    // For 64 PEs: ~69 cycles. Negligible vs ~390 compute cycles.
    // ================================================================
    logic signed [WIDTH-1:0]       requant_scale [0:NUM_OUTPUTS-1];
    (* keep *) logic signed [WIDTH-1:0] requant_out [0:NUM_OUTPUTS-1];

    // PE index counter for serialized walk
    localparam integer PE_IDX_BITS = $clog2(NUM_OUTPUTS + 1);
    logic [PE_IDX_BITS-1:0] rq_idx;
    logic [1:0]             rq_phase;

    // Single shared multiplier pipeline (2 stages for clean DSP inference)
    // Stage 1: register operands
    logic signed [ACC_WIDTH-1:0]   rq_acc_reg;
    logic signed [WIDTH-1:0]       rq_scale_reg;
    logic [PE_IDX_BITS-1:0]        rq_wr_idx;
    logic                          rq_pipe_valid;

    // Stage 2: registered multiply result
    logic signed [PROD_WIDTH-1:0]  rq_product;
    logic [PE_IDX_BITS-1:0]        rq_wr_idx_d1;
    logic                          rq_pipe_valid_d1;

    // Saturation function
    function automatic logic signed [WIDTH-1:0] saturate_s16(
        input logic signed [PROD_WIDTH-1:0] val
    );
        if (val > SAT_MAX)
            return 16'sd32767;
        else if (val < SAT_MIN)
            return -16'sd32767;
        else
            return val[WIDTH-1:0];
    endfunction

    // ================================================================
    // Generate PEs + weight BRAMs
    //
    // Uses hardcoded string literals in init_file concatenation
    // (same approach as B-spline pe_column.sv) for ModelSim compatibility.
    // Place .mif files in mem_init/ relative to the simulation directory.
    // ================================================================
    genvar gi;
    generate
        for (gi = 0; gi < NUM_OUTPUTS; gi++) begin : pes
            localparam [7:0] D0 = 8'd48 + (gi % 10);
            localparam [7:0] D1 = 8'd48 + ((gi / 10) % 10);
            localparam [7:0] D2 = 8'd48 + ((gi / 100) % 10);

            if (gi < 10) begin : mif_sel
                altsyncram #(
                    .operation_mode("SINGLE_PORT"),
                    .width_a(16),
                    .widthad_a(ADDR_BITS),
                    .numwords_a(MEM_DEPTH),
                    .ram_block_type("M20K"),
                    .outdata_reg_a("CLOCK0"),
                    .init_file({"mem_init/weights_pe_", D0, ".mif"})
                ) weight_ram (
                    .clock0(clk),
                    .address_a(mem_addr),
                    .q_a(pe_coeff_in[gi]),
                    .wren_a(1'b0)
                );
            end else if (gi < 100) begin : mif_sel
                altsyncram #(
                    .operation_mode("SINGLE_PORT"),
                    .width_a(16),
                    .widthad_a(ADDR_BITS),
                    .numwords_a(MEM_DEPTH),
                    .ram_block_type("M20K"),
                    .outdata_reg_a("CLOCK0"),
                    .init_file({"mem_init/weights_pe_", D1, D0, ".mif"})
                ) weight_ram (
                    .clock0(clk),
                    .address_a(mem_addr),
                    .q_a(pe_coeff_in[gi]),
                    .wren_a(1'b0)
                );
            end else begin : mif_sel
                altsyncram #(
                    .operation_mode("SINGLE_PORT"),
                    .width_a(16),
                    .widthad_a(ADDR_BITS),
                    .numwords_a(MEM_DEPTH),
                    .ram_block_type("M20K"),
                    .outdata_reg_a("CLOCK0"),
                    .init_file({"mem_init/weights_pe_", D2, D1, D0, ".mif"})
                ) weight_ram (
                    .clock0(clk),
                    .address_a(mem_addr),
                    .q_a(pe_coeff_in[gi]),
                    .wren_a(1'b0)
                );
            end

            pe_quad #(.WIDTH(WIDTH), .DEGREE(DEGREE)) core (
                .clk(clk), .rst_n(rst_n), .start(pe_start),
                .x_A(curr_x_A), .x_B(curr_x_B), .x_C(curr_x_C), .x_D(curr_x_D),
                .coeff_in(pe_coeff_in[gi]),
                .curr_thread(curr_thread[gi]),
                .curr_k(curr_k[gi]),
                .done(pe_dones[gi]),
                .y_A(res_A[gi]), .y_B(res_B[gi]), .y_C(res_C[gi]), .y_D(res_D[gi])
            );
        end
    endgenerate

    // ================================================================
    // Independent background loader
    // ================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            loading  <= 0;
            load_idx <= 0;
        end else begin
            if (start) begin
                loading  <= 1;
                load_idx <= 1;
                input_buffer[0] <= x_in;
                for (int p = NUM_INPUTS; p < MEM_SIZE; p++)
                    input_buffer[p] <= '0;
            end else if (loading) begin
                input_buffer[load_idx] <= x_in;
                if (load_idx == NUM_INPUTS - 1) loading <= 0;
                else load_idx <= load_idx + 1;
            end
        end
    end

    // ================================================================
    // Requant multiplier pipeline (outside FSM for clean DSP packing)
    //
    // Stage 1 operands driven by FSM. Stage 2 registered multiply.
    // Writeback: shift + saturate → requant_out[idx]
    // ================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rq_product       <= '0;
            rq_wr_idx_d1     <= '0;
            rq_pipe_valid_d1 <= 1'b0;
        end else begin
            rq_product       <= rq_acc_reg * rq_scale_reg;
            rq_wr_idx_d1     <= rq_wr_idx;
            rq_pipe_valid_d1 <= rq_pipe_valid;
        end
    end

    // Writeback: shift + saturate
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int p = 0; p < NUM_OUTPUTS; p++)
                requant_out[p] <= '0;
        end else if (rq_pipe_valid_d1) begin
            requant_out[rq_wr_idx_d1] <= saturate_s16(rq_product >>> REQUANT_SHIFT);
        end
    end

    // ================================================================
    // Controller FSM
    // ================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= IDLE;
            group_idx <= 0;
            ready     <= 0;
            pe_start  <= 0;
            requant_reading <= 0;
            rq_phase  <= 0;
            rq_idx    <= 0;
            rq_acc_reg    <= '0;
            rq_scale_reg  <= '0;
            rq_wr_idx     <= '0;
            rq_pipe_valid <= 1'b0;
            for (int p = 0; p < NUM_OUTPUTS; p++) begin
                accumulator[p]   <= 0;
                sum_AB[p]        <= 0;
                sum_CD[p]        <= 0;
                requant_scale[p] <= 0;
            end
        end else begin
            // Default: no valid data into requant pipeline
            rq_pipe_valid <= 1'b0;

            case (state)
                // ----------------------------------------------------------
                IDLE: begin
                    ready <= 0;
                    if (start) begin
                        state     <= WAIT_FIRST_QUAD;
                        group_idx <= 0;
                        for (int p = 0; p < NUM_OUTPUTS; p++) begin
                            accumulator[p] <= 0;
                            sum_AB[p]      <= 0;
                            sum_CD[p]      <= 0;
                        end
                    end
                end

                // ----------------------------------------------------------
                WAIT_FIRST_QUAD: begin
                    if (load_idx >= 4 || !loading) begin
                        state <= FETCH;
                    end
                end

                // ----------------------------------------------------------
                FETCH: begin
                    pe_start <= 1;
                    state    <= COMPUTE;
                end

                // ----------------------------------------------------------
                COMPUTE: begin
                    pe_start <= 0;
                    if (pe_dones[0]) begin
                        state <= ACCUM_1;
                    end
                end

                // ----------------------------------------------------------
                ACCUM_1: begin
                    for (int p = 0; p < NUM_OUTPUTS; p++) begin
                        sum_AB[p] <= (thread_mask[0] ? res_A[p] : 16'sd0)
                                   + (thread_mask[1] ? res_B[p] : 16'sd0);
                        sum_CD[p] <= (thread_mask[2] ? res_C[p] : 16'sd0)
                                   + (thread_mask[3] ? res_D[p] : 16'sd0);
                    end
                    state <= ACCUM_2;
                end

                // ----------------------------------------------------------
                ACCUM_2: begin
                    for (int p = 0; p < NUM_OUTPUTS; p++) begin
                        accumulator[p] <= accumulator[p] + sum_AB[p] + sum_CD[p];
                    end

                    if (group_idx >= NUM_GROUPS - 1) begin
                        state           <= REQUANT;
                        requant_reading <= 1;
                        rq_phase        <= 2'd0;
                        rq_idx          <= 0;
                    end else begin
                        group_idx <= group_idx + 1;
                        state     <= FETCH;
                    end
                end

                // ----------------------------------------------------------
                // REQUANT — serialized through single multiplier
                // ----------------------------------------------------------
                REQUANT: begin
                    case (rq_phase)
                        // BRAM latency cycle 1 (address registered)
                        2'd0: begin
                            rq_phase <= 2'd1;
                        end

                        // BRAM latency cycle 2 (wait for output register to update)
                        2'd1: begin
                            rq_phase <= 2'd2;
                            rq_idx   <= 0;
                        end

                        // Serial walk: feed 1 PE per cycle
                        2'd2: begin
                            rq_acc_reg    <= accumulator[rq_idx];
                            // Read directly from the stable BRAM output!
                            rq_scale_reg  <= pe_coeff_in[rq_idx]; 
                            rq_wr_idx     <= rq_idx;
                            rq_pipe_valid <= 1'b1;
                            
                            if (rq_idx == NUM_OUTPUTS - 1) begin
                                rq_phase <= 2'd3;
                            end else begin
                                rq_idx <= rq_idx + 1;
                            end
                        end

                        // Pipeline drain (2 cycles: multiply + writeback)
                        2'd3: begin
                            requant_reading <= 0;
                            state           <= FINISH;
                        end
                    endcase
                end

                // ----------------------------------------------------------
                FINISH: begin
                    ready <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end

    // ================================================================
    // Output bus
    // ================================================================
    (* keep *) logic [NUM_OUTPUTS*ACC_WIDTH-1:0] pe_accum_results;
    (* keep *) logic [NUM_OUTPUTS*WIDTH-1:0]     pe_requant_results;

    genvar j;
    generate
        for (j = 0; j < NUM_OUTPUTS; j++) begin : pack_results
            assign pe_accum_results[j*ACC_WIDTH +: ACC_WIDTH]  = accumulator[j];
            assign pe_requant_results[j*WIDTH +: WIDTH]        = requant_out[j];
        end
    endgenerate

    assign debug_bit = ^pe_accum_results ^ ^pe_requant_results;

endmodule
