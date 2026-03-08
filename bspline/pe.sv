(* preserve *) module pe #(
    parameter INPUT_WIDTH = 27,   // Max width for Arria 10 DSP Cascade
    parameter WEIGHT_WIDTH = 16,  // Weight dtype, M20K supports up to 40-bit widths
    parameter ACC_WIDTH = 64,     // Hardened DSP Accumulator width
    parameter MEM_DEPTH = 60,     // Equals input dim of matrix, should always fit in 1 M20K
    parameter INIT_FILE = "",     // .mif file path for weight RAM initialization
    parameter REQUANT_SHIFT = 32  // Right-shift for requant: (acc * scale) >>> REQUANT_SHIFT
)(
    input wire clk,
    input wire sclr,              // Synchronous Clear (flushes accumulator & resets addr)
    
    // --------------------------------------------------------
    // Input Cascade Interface (Systolic Data Path)
    // --------------------------------------------------------
    input wire signed [INPUT_WIDTH-1:0] input_in,
    output wire signed [INPUT_WIDTH-1:0] input_out,
    input wire valid_in,
    output reg valid_out,
    
    // --------------------------------------------------------
    // Stride Cascade (Weight Address Skip)
    // --------------------------------------------------------
    input wire [2:0] stride_in,
    output reg [2:0] stride_out,
    
    // --------------------------------------------------------
    // Done Cascade (Requant Trigger)
    // --------------------------------------------------------
    input wire done_in,
    output reg done_out,
    
    // --------------------------------------------------------
    // Result
    // --------------------------------------------------------
    output reg signed [ACC_WIDTH-1:0] acc
);


	// ============================================================
	// 1. M20K Implementation (Local Weight Storage)
	//    Explicit Intel primitive (FORCES M20K)
	// ============================================================
		
    // ============================================================
    // Requant Constants
    // ============================================================
    localparam SCALE_BITS   = 14;                               // max requant scale = 2^14 = 16384
    localparam DROP_BITS    = REQUANT_SHIFT - SCALE_BITS;       // 18: low acc bits with < 2^REQUANT_SHIFT contribution
    localparam REMAIN_SHIFT = SCALE_BITS;                       // 14: final shift after dropping low bits
    localparam CHUNK0_WIDTH = INPUT_WIDTH - 1;                  // 26: unsigned lower chunk fits in signed INPUT_WIDTH
    localparam [$clog2(MEM_DEPTH+1)-1:0] SCALE_ADDR = MEM_DEPTH;  // scale stored after last weight row

    // Done pipeline (same structure as valid pipeline)
    reg done_d1, done_d2, done_d3, done_d4;

    // Requant FSM state
    reg        requant_active;
    reg  [2:0] rq_step;
    reg signed [ACC_WIDTH-1:0]                acc_snapshot;
    reg signed [INPUT_WIDTH+WEIGHT_WIDTH-1:0] partial_lo_reg;

    // Pre-registered requant result and write enable (keeps acc path clean for DSP packing)
    reg signed [ACC_WIDTH-1:0] requant_result;
    reg                        acc_write_requant;

    reg [$clog2(MEM_DEPTH+1)-1:0] read_addr;
    reg signed [WEIGHT_WIDTH-1:0] weight_reg;
	reg valid_d1, valid_d2, valid_d3, valid_d4; 
	reg signed [WEIGHT_WIDTH-1:0] weight_dsp;

    // Pre-increment address: stride is applied BEFORE the M20K read.
    // During requant, overrides to SCALE_ADDR to read scale from last BRAM row.
    wire [$clog2(MEM_DEPTH+1)-1:0] m20k_addr;
    assign m20k_addr = (done_d3 || done_d4 || requant_active) ? SCALE_ADDR : (read_addr + stride_in);


	 // the .mif for this M20K is set via the INIT_FILE parameter from pe_column.sv
	altsyncram #(
		 .operation_mode("SINGLE_PORT"),
		 .width_a(WEIGHT_WIDTH),
		 .widthad_a($clog2(MEM_DEPTH+1)),
		 .numwords_a(MEM_DEPTH+1),
		 .outdata_reg_a("CLOCK0"),
		 .ram_block_type("M20K"),
		 .intended_device_family("Arria 10"),
		 .init_file(INIT_FILE),

		 // Stability knobs
		 .clock_enable_input_a("BYPASS"),
		 .clock_enable_output_a("BYPASS"),
		 .read_during_write_mode_port_a("DONT_CARE"),
		 .power_up_uninitialized("FALSE")
		 
	) weight_ram (
		 .clock0(clk),
		 .address_a(m20k_addr),
		 .q_a(weight_reg),
		 .wren_a(1'b0),
		 .data_a({WEIGHT_WIDTH{1'b0}}),
		 .rden_a(valid_in || done_d3 || done_d4 || requant_active)
	);

	always @(posedge clk) begin
        weight_dsp <= weight_reg;  // No sclr: allows M20K→DSP direct packing
	end

    // Address Counter (Pre-increment: stride applied before read)
    always @(posedge clk) begin
        if (sclr) 
            read_addr <= 0;
        else if (valid_in)
            read_addr <= m20k_addr;
    end

   
	// Valid pipeline for local MAC timing
    always @(posedge clk) begin
        if (sclr) begin
            valid_out <= 1'b0;
            valid_d1 <= 1'b0;
            valid_d2 <= 1'b0;
            valid_d3 <= 1'b0;
            valid_d4 <= 1'b0;
        end else begin
            // Forwarding to next PE must align with input_out (1-cycle path)
            valid_out <= valid_in;

            // Local MAC enable pipeline
            valid_d1 <= valid_in;
            valid_d2 <= valid_d1;
            valid_d3 <= valid_d2;
            valid_d4 <= valid_d3;
        end
    end

    // Done cascade + local pipeline (same timing as valid cascade)
    // done_d3: triggers BRAM read of requant scale from last row
    // done_d4: triggers requant FSM start (snapshot acc, begin decomposed multiply)
    always @(posedge clk) begin
        if (sclr) begin
            done_out <= 1'b0;
            done_d1  <= 1'b0;
            done_d2  <= 1'b0;
            done_d3  <= 1'b0;
            done_d4  <= 1'b0;
        end else begin
            done_out <= done_in;   // Forward to next PE (1-cycle cascade latency)
            done_d1  <= done_in;   // Local pipeline
            done_d2  <= done_d1;
            done_d3  <= done_d2;   // → BRAM scale read
            done_d4  <= done_d3;   // → Requant FSM start
        end
    end

    // ============================================================
    // 2. DSP Logic with Systolic Cascade + Requant
    // ============================================================
    reg signed [INPUT_WIDTH-1:0] input_reg;
    reg signed [INPUT_WIDTH-1:0] input_dsp;
    reg signed [INPUT_WIDTH-1:0] input_mul;
    reg signed [INPUT_WIDTH+WEIGHT_WIDTH-1:0] mult_reg;

    // Requant chunk selection (combinatorial mux)
    // chunk0: lower CHUNK0_WIDTH bits of acc[63:DROP_BITS], zero-extended → always non-negative
    // chunk1: upper bits of acc[63:DROP_BITS], sign-extended → carries sign of acc
    reg signed [INPUT_WIDTH-1:0] requant_chunk;

    always @(*) begin
        case (rq_step)
            3'd0: requant_chunk = $signed({1'b0, acc_snapshot[DROP_BITS + CHUNK0_WIDTH - 1 : DROP_BITS]});
            3'd1: requant_chunk = $signed(acc_snapshot[ACC_WIDTH-1 : DROP_BITS + CHUNK0_WIDTH]);
            default: requant_chunk = '0;
        endcase
    end

    always @(posedge clk) begin
        input_reg  <= input_in;
        // MUX at input_dsp stage: keeps input_mul → DSP clean for register packing
        input_dsp  <= requant_active ? requant_chunk : input_reg;
        input_mul  <= input_dsp;
        mult_reg   <= input_mul * weight_dsp;
    end

    // Forward stride with same latency as input_out/valid_out (1 cycle)
    always @(posedge clk) begin
        if (sclr)
            stride_out <= 3'd0;
        else
            stride_out <= stride_in;
    end

    // Sign-extend mult_reg and partial_lo_reg to ACC_WIDTH for requant combine
    wire signed [ACC_WIDTH-1:0] partial_hi_wide;
    wire signed [ACC_WIDTH-1:0] partial_lo_wide;
    assign partial_hi_wide = {{(ACC_WIDTH-INPUT_WIDTH-WEIGHT_WIDTH){mult_reg[INPUT_WIDTH+WEIGHT_WIDTH-1]}}, mult_reg};
    assign partial_lo_wide = {{(ACC_WIDTH-INPUT_WIDTH-WEIGHT_WIDTH){partial_lo_reg[INPUT_WIDTH+WEIGHT_WIDTH-1]}}, partial_lo_reg};

    // Pre-registered requant combine (keeps combinatorial depth off the acc write path)
    always @(posedge clk) begin
        requant_result    <= ((partial_hi_wide <<< CHUNK0_WIDTH) + partial_lo_wide) >>> REMAIN_SHIFT;
        acc_write_requant <= requant_active && (rq_step == 3'd4);
    end

    // Accumulator + Requant FSM
    //
    // Normal mode: acc <= acc + mult_reg  (on valid_d4)
    //
    // Requant mode (triggered by done_d4, reuses existing DSP):
    //   Computes (acc * scale) >>> REQUANT_SHIFT via two 27x16 partial multiplies:
    //     chunk0 = acc[43:18]  (26-bit unsigned, zero-extended to 27 signed)
    //     chunk1 = acc[63:44]  (20-bit signed, sign-extended to 27 signed)
    //     result = ((chunk1*scale) <<< 26 + (chunk0*scale)) >>> 14
    //
    // Timing after done_d4 (MUX at input_dsp stage for DSP packing):
    //   rq_step 0: chunk0 → input_dsp; weight pipeline filling
    //   rq_step 1: chunk1 → input_dsp; chunk0 → input_mul
    //   rq_step 2: mult_reg computing chunk0*scale
    //   rq_step 3: capture partial_lo = chunk0*scale; chunk1*scale computing
    //   rq_step 4: requant_result registered; acc_write_requant fires next cycle
    always @(posedge clk) begin
        if (sclr) begin
            acc            <= '0;
            requant_active <= 1'b0;
            rq_step        <= 3'd0;
            acc_snapshot   <= '0;
            partial_lo_reg <= '0;
        end else if (requant_active) begin
            case (rq_step)
                3'd0: rq_step <= 3'd1;
                3'd1: rq_step <= 3'd2;
                3'd2: rq_step <= 3'd3;
                3'd3: begin
                    partial_lo_reg <= mult_reg;  // = chunk0 * scale
                    rq_step <= 3'd4;
                end
                3'd4: begin
                    requant_active <= 1'b0;
                    rq_step <= 3'd0;
                end
                default: begin
                    requant_active <= 1'b0;
                    rq_step <= 3'd0;
                end
            endcase
        end else if (acc_write_requant) begin
            // Registered requant result → accumulator (1 cycle after FSM completes)
            acc <= requant_result;
        end else if (done_d4) begin
            // GEMV complete. Snapshot accumulator, begin requant.
            acc_snapshot   <= acc;
            requant_active <= 1'b1;
            rq_step        <= 3'd0;
        end else if (valid_d4) begin
            acc <= acc + mult_reg;
        end
    end

    // ============================================================
    // 3. Cascade Output Assignment to below PE
    // ============================================================
    assign input_out = input_reg;

endmodule






















