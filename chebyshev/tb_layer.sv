`timescale 1ns/1ps
`default_nettype none

// ============================================================================
// Configurable Chebyshev Layer Testbench
//
// All layer-specific parameters come from `tb_config.svh`.
// This testbench works for ANY layer dimensions — just regenerate the config.
//
// Expected tb_config.svh contents:
//   localparam integer NUM_INPUTS  = <in_dim>;
//   localparam integer NUM_OUTPUTS = <out_dim>;
//   localparam integer DEGREE      = 3;
//   localparam integer ACC_WIDTH   = 22;
//   localparam integer REQUANT_SHIFT = 14;
//   localparam string  TESTCASE_DIR = "testcase_data/layer_0_5x16";
//   localparam string  MIF_PREFIX   = "testcase_data/layer_0_5x16/mem_init/weights_pe_";
//
// Expected testcase directory structure:
//   <TESTCASE_DIR>/
//     layer_inputs_s16.npy      — 1D array of int16,  length = NUM_INPUTS
//     golden_acc_s64.npy        — 1D array of int64,  length = NUM_OUTPUTS
//     golden_requant_s16.npy    — 1D array of int16,  length = NUM_OUTPUTS
//     mem_init/
//       weights_pe_0.mif ... weights_pe_<NUM_OUTPUTS-1>.mif
// ============================================================================

module tb_layer;

    // =========================================================================
    // CONFIGURATION — all from include file
    // =========================================================================
    `include "tb_config.svh"

    localparam string INPUT_NPY       = {TESTCASE_DIR, "/layer_inputs_s16.npy"};
    localparam string GOLDEN_ACC_NPY  = {TESTCASE_DIR, "/golden_acc_s64.npy"};
    localparam string GOLDEN_RQ_NPY   = {TESTCASE_DIR, "/golden_requant_s16.npy"};

    // =========================================================================
    // 1. Signals & DUT
    // =========================================================================
    logic clk;
    logic rst_n;
    logic start;
    logic [15:0] x_in;

    wire ready;
    wire debug_bit;

    // Access internal results for verification
    wire [NUM_OUTPUTS*ACC_WIDTH-1:0] pe_accum_results;
    wire [NUM_OUTPUTS*16-1:0]        pe_requant_results;

    // --- PERFORMANCE TRACKING ---
    integer cycle_count  = 0;
    integer start_cycle  = 0;
    integer end_cycle    = 0;
    integer total_cycles = 0;

    layer #(
        .NUM_INPUTS    (NUM_INPUTS),
        .NUM_OUTPUTS   (NUM_OUTPUTS),
        .WIDTH         (16),
        .DEGREE        (DEGREE),
        .ACC_WIDTH     (ACC_WIDTH),
        .REQUANT_SHIFT (REQUANT_SHIFT),
        .MIF_PREFIX    (MIF_PREFIX)
    ) dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (start),
        .x_in      (x_in),
        .ready     (ready),
        .debug_bit (debug_bit)
    );

    assign pe_accum_results  = dut.pe_accum_results;
    assign pe_requant_results = dut.pe_requant_results;

    // Clock Generation (100 MHz simulation clock)
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // Global cycle counter
    always @(posedge clk) cycle_count = cycle_count + 1;

    // =========================================================================
    // 2. NumPy Loaders (reused from original, supports v1 .npy files)
    // =========================================================================
    shortint       inputs[$];
    longint signed golden_acc[$];
    shortint       golden_rq[$];

    function automatic int read_byte_or_fatal(input int fd, input string what);
        int c = $fgetc(fd);
        if (c == -1) $fatal(1, "Unexpected EOF while reading %s", what);
        return c;
    endfunction

    function automatic int parse_shape_1d(input string header);
        int i, len, value;
        bit seen_paren, collecting;
        byte unsigned ch;
        len = header.len();
        value = 0; seen_paren = 0; collecting = 0;
        for (i = 0; i < len; i++) begin
            ch = header[i];
            if (!seen_paren) begin
                if (ch == "(") seen_paren = 1;
            end else if (!collecting) begin
                if (ch >= "0" && ch <= "9") begin
                    collecting = 1; value = ch - "0";
                end else if (ch == ")") break;
            end else begin
                if (ch >= "0" && ch <= "9") value = (value * 10) + (ch - "0");
                else if (ch == "," || ch == ")") return value;
            end
        end
        return collecting ? value : -1;
    endfunction

    task automatic load_npy_s16_1d(input string npy_path, ref shortint dst[$]);
        int fd, i, major, minor, header_len, elem_count, c;
        string header; byte unsigned b0, b1;
        dst.delete();
        fd = $fopen(npy_path, "rb");
        if (fd == 0) $fatal(1, "Could not open NPY file: %s", npy_path);

        repeat(6) c = read_byte_or_fatal(fd, "magic");
        major = read_byte_or_fatal(fd, "major");
        minor = read_byte_or_fatal(fd, "minor");

        if (major == 1) begin
            b0 = read_byte_or_fatal(fd, "hlen0");
            b1 = read_byte_or_fatal(fd, "hlen1");
            header_len = int'(b0) + (int'(b1) << 8);
        end else begin
            byte unsigned b2, b3;
            b0 = read_byte_or_fatal(fd, "hlen0"); b1 = read_byte_or_fatal(fd, "hlen1");
            b2 = read_byte_or_fatal(fd, "hlen2"); b3 = read_byte_or_fatal(fd, "hlen3");
            header_len = int'(b0) + (int'(b1) << 8) + (int'(b2) << 16) + (int'(b3) << 24);
        end

        header = "";
        for (i = 0; i < header_len; i++) header = {header, byte'(read_byte_or_fatal(fd, "hdr"))};

        elem_count = parse_shape_1d(header);
        if (elem_count <= 0) $fatal(1, "Could not parse shape from %s", npy_path);

        for (i = 0; i < elem_count; i++) begin
            b0 = read_byte_or_fatal(fd, "d0");
            b1 = read_byte_or_fatal(fd, "d1");
            dst.push_back(shortint'({b1, b0}));
        end
        void'($fclose(fd));
    endtask

    task automatic load_npy_s64_1d(input string npy_path, ref longint signed dst[$]);
        int fd, i, major, minor, header_len, elem_count, c;
        string header; byte unsigned b0, b1, b2, b3, b4, b5, b6, b7;
        logic [63:0] raw;
        dst.delete();
        fd = $fopen(npy_path, "rb");
        if (fd == 0) $fatal(1, "Could not open NPY file: %s", npy_path);

        repeat(6) c = read_byte_or_fatal(fd, "magic");
        major = read_byte_or_fatal(fd, "major");
        minor = read_byte_or_fatal(fd, "minor");

        if (major == 1) begin
            b0 = read_byte_or_fatal(fd, "hlen0");
            b1 = read_byte_or_fatal(fd, "hlen1");
            header_len = int'(b0) + (int'(b1) << 8);
        end else begin
            b0 = read_byte_or_fatal(fd, "hlen0"); b1 = read_byte_or_fatal(fd, "hlen1");
            b2 = read_byte_or_fatal(fd, "hlen2"); b3 = read_byte_or_fatal(fd, "hlen3");
            header_len = int'(b0) + (int'(b1) << 8) + (int'(b2) << 16) + (int'(b3) << 24);
        end

        header = "";
        for (i = 0; i < header_len; i++) header = {header, byte'(read_byte_or_fatal(fd, "hdr"))};

        elem_count = parse_shape_1d(header);
        if (elem_count <= 0) $fatal(1, "Could not parse shape from %s", npy_path);

        for (i = 0; i < elem_count; i++) begin
            b0 = read_byte_or_fatal(fd, "d0"); b1 = read_byte_or_fatal(fd, "d1");
            b2 = read_byte_or_fatal(fd, "d2"); b3 = read_byte_or_fatal(fd, "d3");
            b4 = read_byte_or_fatal(fd, "d4"); b5 = read_byte_or_fatal(fd, "d5");
            b6 = read_byte_or_fatal(fd, "d6"); b7 = read_byte_or_fatal(fd, "d7");
            raw = {b7, b6, b5, b4, b3, b2, b1, b0};
            dst.push_back(longint'(raw));
        end
        void'($fclose(fd));
    endtask

    // =========================================================================
    // 3. Test Execution
    // =========================================================================
    int acc_errors = 0;
    int rq_errors  = 0;
    int rq_off_by_one = 0;
    longint signed actual_acc;
    shortint       actual_rq;
    shortint       expected_rq;

    initial begin
        // --- Load test vectors ---
        load_npy_s16_1d(INPUT_NPY, inputs);
        load_npy_s64_1d(GOLDEN_ACC_NPY, golden_acc);
        load_npy_s16_1d(GOLDEN_RQ_NPY, golden_rq);

        // --- Validate dimensions ---
        if (inputs.size() != NUM_INPUTS)
            $fatal(1, "Input length (%0d) != NUM_INPUTS (%0d)", inputs.size(), NUM_INPUTS);
        if (golden_acc.size() != NUM_OUTPUTS)
            $fatal(1, "Golden acc length (%0d) != NUM_OUTPUTS (%0d)", golden_acc.size(), NUM_OUTPUTS);
        if (golden_rq.size() != NUM_OUTPUTS)
            $fatal(1, "Golden requant length (%0d) != NUM_OUTPUTS (%0d)", golden_rq.size(), NUM_OUTPUTS);

        // --- Initialize ---
        start = 0;
        x_in  = 0;
        rst_n = 0;

        // Reset sequence
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);

        $display("=========================================================");
        $display(" Chebyshev Layer Testbench");
        $display("---------------------------------------------------------");
        $display(" NUM_INPUTS  = %0d", NUM_INPUTS);
        $display(" NUM_OUTPUTS = %0d", NUM_OUTPUTS);
        $display(" DEGREE      = %0d", DEGREE);
        $display(" REQUANT_SHIFT = %0d", REQUANT_SHIFT);
        $display(" TESTCASE_DIR  = %s", TESTCASE_DIR);
        $display("=========================================================");

        // --- Stream inputs ---
        @(posedge clk);
        start_cycle = cycle_count;

        start <= 1'b1;
        x_in  <= inputs[0];
        @(posedge clk);
        start <= 1'b0;

        for (int k = 1; k < NUM_INPUTS; k++) begin
            x_in <= inputs[k];
            @(posedge clk);
        end
        x_in <= 16'd0;  // stop driving after last input

        // --- Wait for completion ---
        wait(ready);
        end_cycle = cycle_count;
        total_cycles = end_cycle - start_cycle;

        // Allow one more cycle for requant_out to settle
        @(posedge clk);

        // =====================================================================
        // CHECK 1: Raw accumulator values (exact match expected)
        // =====================================================================
        $display("");
        $display("----- Accumulator Check (exact) -----");
        for (int i = 0; i < NUM_OUTPUTS; i++) begin
            actual_acc = $signed(pe_accum_results[i*ACC_WIDTH +: ACC_WIDTH]);
            if (actual_acc !== golden_acc[i]) begin
                $display("  ACC FAIL PE[%0d] | Expected: %0d | Got: %0d", i, golden_acc[i], actual_acc);
                acc_errors++;
            end else begin
                $display("  ACC PASS PE[%0d] | Value: %0d", i, actual_acc);
            end
        end

        // =====================================================================
        // CHECK 2: Requantized outputs (allow +/-1 tolerance for rounding)
        // =====================================================================
        $display("");
        $display("----- Requant Check (tolerance +/-1) -----");
        for (int i = 0; i < NUM_OUTPUTS; i++) begin
            actual_rq   = $signed(pe_requant_results[i*16 +: 16]);
            expected_rq = golden_rq[i];
            if (actual_rq === expected_rq) begin
                $display("  RQ  PASS PE[%0d] | Value: %0d", i, actual_rq);
            end else if (actual_rq == expected_rq + 1 || actual_rq == expected_rq - 1) begin
                $display("  RQ  PASS PE[%0d] | Expected: %0d, Got: %0d (off-by-one)", i, expected_rq, actual_rq);
                rq_off_by_one++;
            end else begin
                $display("  RQ  FAIL PE[%0d] | Expected: %0d, Got: %0d", i, expected_rq, actual_rq);
                rq_errors++;
            end
        end

        // =====================================================================
        // SUMMARY
        // =====================================================================
        $display("");
        $display("=========================================================");
        $display(" RESULTS SUMMARY");
        $display("---------------------------------------------------------");
        $display(" Total Cycles:       %0d", total_cycles);
        $display(" Accumulator Errors:  %0d / %0d", acc_errors, NUM_OUTPUTS);
        $display(" Requant Errors:      %0d / %0d", rq_errors, NUM_OUTPUTS);
        if (rq_off_by_one > 0)
            $display(" Requant Off-by-one: %0d (acceptable)", rq_off_by_one);
        $display("---------------------------------------------------------");
        if (acc_errors == 0 && rq_errors == 0)
            $display(" TEST PASSED");
        else
            $display(" TEST FAILED");
        $display("=========================================================");

        $stop;
    end

endmodule
