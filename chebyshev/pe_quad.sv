`timescale 1ns / 1ps

module pe_quad #(
    parameter integer WIDTH = 16,
    parameter integer FRAC_BITS = 10,
    parameter integer DEGREE = 3          
)(
    input  logic             clk,
    input  logic             rst_n,
    input  logic             start, 
   
    input  logic signed [WIDTH-1:0] x_A, x_B, x_C, x_D,
    input  logic signed [WIDTH-1:0] coeff_in, 
    
    output logic [1:0]       curr_thread,
    output logic [7:0]       curr_k,
    
    output logic             done, 
    output logic signed [WIDTH-1:0] y_A, y_B, y_C, y_D
);

    logic [1:0] thread_idx;
    logic [7:0] k; 
    logic active;

    assign curr_thread = thread_idx;
    assign curr_k = k;

    // --- PIPELINE TRACKING REGISTERS ---
    logic [1:0] thread_idx_d1, thread_idx_d2, thread_idx_d3;
    logic [7:0] k_d1, k_d2, k_d3;
    logic active_d1, active_d2, active_d3;

    logic signed [WIDTH-1:0] b_next [0:3];
    logic signed [WIDTH-1:0] b_prev [0:3];

    // --- STAGE 1: INPUT MUX & REGISTER ---
    logic signed [WIDTH-1:0] current_x, current_b;
    logic signed [WIDTH-1:0] reg_x, reg_b;

    always_comb begin
        case (thread_idx)
            0: begin current_x = x_A; current_b = b_next[0]; end
            1: begin current_x = x_B; current_b = b_next[1]; end
            2: begin current_x = x_C; current_b = b_next[2]; end
            3: begin current_x = x_D; current_b = b_next[3]; end
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_x <= 0; reg_b <= 0;
            thread_idx_d1 <= 0; k_d1 <= 0; active_d1 <= 0;
        end else begin
            reg_x <= current_x; reg_b <= current_b;
            thread_idx_d1 <= thread_idx;
            k_d1 <= k;
            active_d1 <= active;
        end
    end

    // --- STAGE 2: PIPELINED MULTIPLIER ---
    logic signed [2*WIDTH-1:0] mult_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mult_reg <= 0;
            thread_idx_d2 <= 0; k_d2 <= 0; active_d2 <= 0;
        end else begin
            mult_reg <= reg_x * reg_b;
            thread_idx_d2 <= thread_idx_d1;
            k_d2 <= k_d1;
            active_d2 <= active_d1;
        end
    end

    // --- STAGE 3: SHIFT & ADD ---
    logic signed [2*WIDTH-1:0] scaled_product;
    logic signed [WIDTH-1:0] term;
    logic signed [WIDTH-1:0] add_reg;

    always_comb begin
        scaled_product = (k_d2 == 0) ? mult_reg : (mult_reg <<< 1);
        term = WIDTH'((scaled_product + (1 <<< (FRAC_BITS - 1))) >>> FRAC_BITS);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            add_reg <= 0;
            thread_idx_d3 <= 0; k_d3 <= 0; active_d3 <= 0;
        end else begin
            add_reg <= coeff_in + term;
            thread_idx_d3 <= thread_idx_d2;
            k_d3 <= k_d2;
            active_d3 <= active_d2;
        end
    end

    // --- STAGE 4: WRITEBACK ---
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active <= 0; thread_idx <= 0; k <= DEGREE; done <= 0;
            for(int i=0; i<4; i++) begin b_next[i]<=0; b_prev[i]<=0; end
            y_A <= 0; y_B <= 0; y_C <= 0; y_D <= 0;
        end else begin
            done <= 0;
            if (start) begin
                active <= 1; thread_idx <= 0; k <= DEGREE;
                for(int i=0; i<4; i++) begin b_next[i]<=0; b_prev[i]<=0; end
            end else if (active) begin
                if (thread_idx == 3) begin
                    if (k == 0) begin active <= 0; end
                    else begin k <= k - 1; thread_idx <= 0; end
                end else begin
                    thread_idx <= thread_idx + 1;
                end
            end

            // Writeback happens only when valid data drains
            if (active_d3) begin
                b_prev[thread_idx_d3] <= b_next[thread_idx_d3];
                b_next[thread_idx_d3] <= add_reg - b_prev[thread_idx_d3];

                if (k_d3 == 0) begin 
                    case (thread_idx_d3)
                        0: y_A <= add_reg - b_prev[0];
                        1: y_B <= add_reg - b_prev[1];
                        2: y_C <= add_reg - b_prev[2];
                        3: y_D <= add_reg - b_prev[3];
                    endcase
                    if (thread_idx_d3 == 3) done <= 1;
                end
            end
        end
    end
endmodule
