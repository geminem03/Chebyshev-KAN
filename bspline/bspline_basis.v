/**
 * bspline_basis
 * * Input: x (16-bit signed-domain integer)
 * Output: Stream of 3 active 27-bit signed integer basis values per input
 * * Timing:
 * - Accepts new 'start' every 3 cycles.
 * - Only streams the 3 active (non-zero) basis outputs per input.
 * - out_stride encodes positional skip from previous output.
 * - Latency: First output appears 3 cycles after start (pipelined DSP).
 */

module bspline_basis(
    input clk,
    input rst_n,
    
    // Control Interface
    input start,            // Pulse high for 1 cycle to start
    input last,             // Assert with start on the last input of a GEMV pass
    output ready,       // High when module can accept new input (every 3 cycles)
    
    // Data Interface
    input [15:0] x,
    output reg signed [26:0] out_data,
    output reg out_valid,
    output reg [2:0] out_stride, // distance from previous output position
    output reg out_done          // Pulses 1 cycle after the final out_valid
);

    // -------------------------------------------------------------------------
    // 1. Input Stage & Constants
    // -------------------------------------------------------------------------
    
    // Constants
    localparam [12:0] SCALE_OFFSET = (1 << 13) - 1;
    localparam [25:0] SCALE_OFFSET_SQUARED = SCALE_OFFSET * SCALE_OFFSET;

    // Internal storage for the current computation
    reg [2:0]  active_matrix_idx;
    reg [12:0] active_offset;
    reg [25:0] active_offset_squared;
    reg [2:0]  cycle_cnt;
    reg        busy;
    reg [2:0]  prev_start_idx;

    // Incoming X processing
    wire [15:0] x_flipped = x ^ 16'h8000;
    wire [2:0]  next_matrix_idx = x_flipped[15:13];
    wire [12:0] next_offset     = x_flipped[12:0];
    
    // We compute offset squared combinatorially during the input cycle 
    // to have it ready for the first math operation immediately.
    // DSP inference usually handles 13x13 mult in one cycle easily.
    wire [25:0] next_offset_squared = next_offset * next_offset;

    // Determine where the 3 active columns map in the 0..5 output sequence
    // (declared early so the state machine can reference start_idx)
    wire [2:0] base_elem_calc = (active_matrix_idx > 3'd2) ? (active_matrix_idx - 3'd2) : 3'd0;
    wire [2:0] start_idx = (base_elem_calc > 3'd3) ? 3'd3 : base_elem_calc;

    // -------------------------------------------------------------------------
    // 2. Control Logic (State Machine)
    // -------------------------------------------------------------------------
    
    // Ready is high if we are IDLE OR if we are in the LAST cycle of a job
    assign ready = (!busy) || (cycle_cnt == 3'd2);
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy        <= 1'b0;
            cycle_cnt   <= 3'd0;
            prev_start_idx <= 3'd4; // 4 so first stride = start_idx (pre-increment from addr 0)
            
            active_matrix_idx     <= 3'd0;
            active_offset         <= 13'd0;
            active_offset_squared <= 26'd0;
        end else begin

            if (start && ready) begin
                busy <= 1'b1;
                cycle_cnt <= 3'd0;
                if (busy) prev_start_idx <= start_idx; // back-to-back: save outgoing start_idx
                
                // Latch Inputs
                active_matrix_idx     <= next_matrix_idx;
                active_offset         <= next_offset;
                active_offset_squared <= next_offset_squared;
            end 
            else if (busy) begin
                if (cycle_cnt == 3'd2) begin
                    busy <= 1'b0; // Done
                    cycle_cnt <= 3'd0;
                    prev_start_idx <= start_idx;
                end else begin
                    cycle_cnt <= cycle_cnt + 3'd1;
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // 3. Weight Selection Logic
    // -------------------------------------------------------------------------

    // Instantiate the lookup table
    wire [26:0] matrix_weights;
    bspline_weights u_weights (
        .idx(active_matrix_idx),
        .weights(matrix_weights)
    );

    // In the optimized 3-cycle design, cycle_cnt IS the column delta (0, 1, 2).
    // We always output active columns, so is_active is implicitly true.
    wire [2:0] delta = cycle_cnt;

    // Select the correct column from the flattened 27-bit weight matrix
    // Col 0 uses indices 0,3,6. Col 1 uses 1,4,7. Col 2 uses 2,5,8.
    // matrix_weights format: {r0c0, r0c1, r0c2, r1c0...}
    // We need to extract specific 3-bit weights for Hi (Row0), Mid (Row1), Lo (Row2)
    
    reg signed [2:0] w_hi;  // Row 0 weight
    reg signed [2:0] w_mid; // Row 1 weight
    reg signed [2:0] w_lo;  // Row 2 weight

    always @(*) begin
        // Select column 0, 1, or 2 based on delta (= cycle_cnt)
        case (delta)
            3'd0: begin // Col 0
                w_hi  = matrix_weights[26:24]; // Idx 0
                w_mid = matrix_weights[17:15]; // Idx 3
                w_lo  = matrix_weights[ 8: 6]; // Idx 6
            end
            3'd1: begin // Col 1
                w_hi  = matrix_weights[23:21]; // Idx 1
                w_mid = matrix_weights[14:12]; // Idx 4
                w_lo  = matrix_weights[ 5: 3]; // Idx 7
            end
            3'd2: begin // Col 2
                w_hi  = matrix_weights[20:18]; // Idx 2
                w_mid = matrix_weights[11: 9]; // Idx 5
                w_lo  = matrix_weights[ 2: 0]; // Idx 8
            end
            default: begin
                w_hi = 0; w_mid = 0; w_lo = 0;
            end
        endcase
    end

    // -------------------------------------------------------------------------
    // 4. Pipeline Stage 1: Register weights + pre-compute mid*scale
    //    Breaks: mux → register (no cascaded multiply in next stage)
    // -------------------------------------------------------------------------

    reg signed [2:0]  w_hi_s1;
    reg signed [15:0] mid_x_scale_s1;  // w_mid * SCALE_OFFSET (3-bit * 13-bit const)
    reg signed [2:0]  w_lo_s1;
    reg        [12:0] offset_s1;
    reg        [25:0] offset_sq_s1;

    // Control pipeline (valid, stride flow alongside data)
    reg        pipe_valid_s1;
    reg  [2:0] pipe_stride_s1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w_hi_s1        <= 3'sd0;
            mid_x_scale_s1 <= 16'sd0;
            w_lo_s1        <= 3'sd0;
            offset_s1      <= 13'd0;
            offset_sq_s1   <= 26'd0;
            pipe_valid_s1  <= 1'b0;
            pipe_stride_s1 <= 3'd0;
        end else begin
            // Data
            w_hi_s1        <= w_hi;
            mid_x_scale_s1 <= $signed(w_mid) * $signed({1'b0, SCALE_OFFSET});
            w_lo_s1        <= w_lo;
            offset_s1      <= active_offset;
            offset_sq_s1   <= active_offset_squared;

            // Control
            pipe_valid_s1  <= busy;
            if (cycle_cnt == 3'd0)
                pipe_stride_s1 <= 3'd4 + start_idx - prev_start_idx;
            else
                pipe_stride_s1 <= 3'd1;
        end
    end

    // -------------------------------------------------------------------------
    // 5. Pipeline Stage 2: DSP multiplies (registered output)
    //    Following pe.sv pattern: registered inputs → multiply → registered output
    // -------------------------------------------------------------------------

    reg signed [31:0] term_A_s2;
    reg signed [31:0] term_B_s2;
    reg signed [31:0] term_C_s2;

    reg        pipe_valid_s2;
    reg  [2:0] pipe_stride_s2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            term_A_s2      <= 32'sd0;
            term_B_s2      <= 32'sd0;
            term_C_s2      <= 32'sd0;
            pipe_valid_s2  <= 1'b0;
            pipe_stride_s2 <= 3'd0;
        end else begin
            // DSP multiplies with registered outputs
            term_A_s2 <= $signed(w_hi_s1)  * $signed({6'b0, SCALE_OFFSET_SQUARED});
            term_B_s2 <= mid_x_scale_s1    * $signed({19'b0, offset_s1});
            term_C_s2 <= $signed(w_lo_s1)  * $signed({6'b0, offset_sq_s1});

            // Control
            pipe_valid_s2  <= pipe_valid_s1;
            pipe_stride_s2 <= pipe_stride_s1;
        end
    end

    // -------------------------------------------------------------------------
    // 6. Pipeline Stage 3: Add + shift + output register
    // -------------------------------------------------------------------------

    wire signed [31:0] acc_shifted;
    assign acc_shifted = (term_A_s2 + term_B_s2 + term_C_s2) >>> 3;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_data   <= 27'd0;
            out_valid  <= 1'b0;
            out_stride <= 3'd0;
        end else begin
            out_valid  <= pipe_valid_s2;
            out_stride <= pipe_stride_s2;
            if (pipe_valid_s2)
                out_data <= acc_shifted[26:0];
            else
                out_data <= 27'd0;
        end
    end

    // -------------------------------------------------------------------------
    // 7. Last-Input Tracking & Done Generation
    //    Fires out_done exactly 1 cycle after the last out_valid.
    //    Pipeline delay matches valid path (busy → s1 → s2 → out) + 1.
    // -------------------------------------------------------------------------
    reg is_last;
    wire last_busy_cycle;
    reg last_s1, last_s2, last_s3;

    assign last_busy_cycle = busy && (cycle_cnt == 3'd2) && is_last;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            is_last  <= 1'b0;
            last_s1  <= 1'b0;
            last_s2  <= 1'b0;
            last_s3  <= 1'b0;
            out_done <= 1'b0;
        end else begin
            if (start && ready)
                is_last <= last;
            last_s1  <= last_busy_cycle;
            last_s2  <= last_s1;
            last_s3  <= last_s2;
            out_done <= last_s3;
        end
    end

endmodule