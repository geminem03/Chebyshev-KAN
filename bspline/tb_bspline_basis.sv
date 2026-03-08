`timescale 1ns/1ps
`default_nettype none

module tb_bspline_basis;

    // -------------------------------------------------------------------------
    // 1. Signals & DUT
    // -------------------------------------------------------------------------
    logic clk;
    logic rst_n;
    logic start;
    logic [15:0] x;

    wire ready;
    wire signed [26:0] out_data;
    wire out_valid;
    wire [2:0] out_stride;

    bspline_basis dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .ready(ready),
        .x(x),
        .out_data(out_data),
        .out_valid(out_valid),
        .out_stride(out_stride)
    );

    // Clock Generation (100 MHz)
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // -------------------------------------------------------------------------
    // 2. Test Vectors (Loaded from CSV)
    // -------------------------------------------------------------------------
    localparam string TESTCASE_CSV = "testcase_data/bspline_basis/bspline_basis_bounds.csv";
    // localparam string TESTCASE_CSV = "testcase_data/bspline_basis/bspline_basis_real.csv";

    typedef int signed exp6_t [6];
    shortint inputs[$];
    exp6_t expected_vectors[$];

    task automatic load_test_vectors(input string csv_path);
        int fd;
        int line_no;
        int parsed_fields;
        string line;
        shortint in_val;
        int signed g0, g1, g2, g3, g4, g5;
        exp6_t exp_row;

        fd = $fopen(csv_path, "r");
        if (fd == 0) begin
            $fatal(1, "Could not open testcase CSV: %s", csv_path);
        end

        line_no = 0;
        while (!$feof(fd)) begin
            void'($fgets(line, fd));
            line_no++;

            if (line.len() == 0) begin
                continue;
            end

            // Skip header row: input,g0,g1,g2,g3,g4,g5
            if (line_no == 1) begin
                continue;
            end

            parsed_fields = $sscanf(line, "%d,%d,%d,%d,%d,%d,%d",
                                    in_val, g0, g1, g2, g3, g4, g5);
            if (parsed_fields != 7) begin
                $fatal(1, "Malformed CSV row at line %0d in %s: %s",
                       line_no, csv_path, line);
            end

            exp_row[0] = g0;
            exp_row[1] = g1;
            exp_row[2] = g2;
            exp_row[3] = g3;
            exp_row[4] = g4;
            exp_row[5] = g5;

            inputs.push_back(in_val);
            expected_vectors.push_back(exp_row);
        end

        void'($fclose(fd));

        if (inputs.size() == 0) begin
            $fatal(1, "No testcase rows loaded from %s", csv_path);
        end
        if (inputs.size() != expected_vectors.size()) begin
            $fatal(1, "Loaded vector size mismatch: inputs=%0d expected=%0d",
                   inputs.size(), expected_vectors.size());
        end
    endtask

    // -------------------------------------------------------------------------
    // 3. Helper Functions & Verification Scoreboard
    // -------------------------------------------------------------------------

    // Compute start_idx (first active column in 0..5) from raw input value,
    // mirroring the RTL's base_elem_calc / start_idx logic.
    function automatic int get_start_idx(input shortint x_val);
        logic [15:0] x_flip;
        int midx, belem, sidx;
        x_flip = x_val ^ 16'h8000;
        midx = int'(x_flip[15:13]);
        belem = (midx > 2) ? (midx - 2) : 0;
        sidx = (belem > 3) ? 3 : belem;
        return sidx;
    endfunction

    // Queues for expected outputs (3 entries per input, not 6).
    // Driver pushes 3 active values per input.
    // Monitor pops 1 per valid output beat.
    int signed expected_data_q[$];
    int        expected_stride_q[$];

    int errors = 0;
    int num_correct = 0;
    int prev_sidx_tb;

    // -------------------------------------------------------------------------
    // 4. Driver Process (Feeds Inputs)
    // -------------------------------------------------------------------------
    initial begin
        int sidx, stride0;

        load_test_vectors(TESTCASE_CSV);
        $display("Loaded %0d testcase rows from %s", inputs.size(), TESTCASE_CSV);

        // Init
        start = 0;
        x = 0;
        rst_n = 0;
        prev_sidx_tb = 4;  // matches RTL reset of prev_start_idx
        repeat(5) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        $display("Starting Stream Test... (3 active outputs per input)");

        // Loop through all test vectors
        for (int i = 0; i < inputs.size(); i++) begin
            sidx    = get_start_idx(inputs[i]);
            stride0 = 4 + sidx - prev_sidx_tb;

            // 1. Wait for DUT to be ready
            wait(ready);

            // 2. Drive Input
            start <= 1'b1;
            x     <= inputs[i];

            // 3. Push 3 active expected results to Scoreboard
            for (int k = 0; k < 3; k++) begin
                expected_data_q.push_back(expected_vectors[i][sidx + k]);
                expected_stride_q.push_back(k == 0 ? stride0 : 1);
            end

            prev_sidx_tb = sidx;

            // 4. Advance 1 cycle (Pulse width)
            @(posedge clk);
            start <= 1'b0;

            // Wait for 'ready' to drop
            wait(!ready);
        end

        // Wait until all expected outputs have been checked
        wait(expected_data_q.size() == 0);

        // Add a small buffer to ensure waveform is clean
        repeat(5) @(posedge clk);

        // Final Report
        if (errors == 0) begin
            $display("---------------------------------------------------");
            $display(" PASS: Streamed %0d inputs, Checked %0d beats.", inputs.size(), num_correct);
            $display("---------------------------------------------------");
        end else begin
            $display("---------------------------------------------------");
            $display(" FAIL: Found %0d errors.", errors);
            $display("---------------------------------------------------");
        end
        $finish;
    end

    // -------------------------------------------------------------------------
    // 5. Monitor Process (Checks Outputs)
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst_n && out_valid) begin
            automatic int signed exp_data;
            automatic int exp_stride;

            if (expected_data_q.size() == 0) begin
                $error("Time %0t: Unexpected output received! Queue is empty.", $time);
                errors++;
            end else begin
                // Pop the oldest expected values
                exp_data   = expected_data_q.pop_front();
                exp_stride = expected_stride_q.pop_front();

                // Compare data
                if (out_data !== exp_data[26:0]) begin
                      $error("Time %0t: DATA mismatch! Got: %0d | Exp: %0d",
                          $time, $signed(out_data), exp_data);
                    errors++;
                end
                // Compare stride
                else if (out_stride !== exp_stride[2:0]) begin
                      $error("Time %0t: STRIDE mismatch! Got: %0d | Exp: %0d",
                          $time, out_stride, exp_stride);
                    errors++;
                end
                else begin
                    num_correct++;
                end
            end
        end
    end

endmodule
