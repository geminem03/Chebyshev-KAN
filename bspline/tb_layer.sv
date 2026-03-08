`timescale 1ns/1ps
`default_nettype none

module tb_layer;

    // -------------------------------------------------------------------------
    // Parameters
    // -------------------------------------------------------------------------
    // localparam NUM_PES = 10;
    localparam NUM_PES = 7;

    // localparam MEM_DEPTH = 60;
    localparam MEM_DEPTH = 30;

    // localparam string TESTCASE_DIR = "testcase_data/layer_10_10_no_requant";
    // localparam string TESTCASE_DIR = "testcase_data/layer_64_64_no_requant";
    localparam string TESTCASE_DIR = "C:/Skule_Hardrive/Chebyshev-KAN/airfoil-study/experiments/quantization_results/bspline_testcase/testcase_layer_0/";
    localparam string INPUT_NPY = {TESTCASE_DIR, "/layer_inputs_s16.npy"};
    localparam string GOLDEN_NPY = {TESTCASE_DIR, "/matvec_requant_out_s16.npy"};

    // -------------------------------------------------------------------------
    // 1. Signals & DUT
    // -------------------------------------------------------------------------
    logic clk;
    logic rst_n;
    logic start;
    logic last;
    logic [15:0] x;

    wire ready;
    wire debug_bit;
    
    // Access internal accumulator results for verification
    wire [NUM_PES*64-1:0] pe_accum_results;
    assign pe_accum_results = dut.pe_accum_results;

    layer #(
        .NUM_PES(NUM_PES),
        .MEM_DEPTH(MEM_DEPTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .last(last),
        .x(x),
        .ready(ready),
        .debug_bit(debug_bit)
    );

    // Clock Generation (100 MHz)
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // -------------------------------------------------------------------------
    // 2. Test Vectors
    // -------------------------------------------------------------------------
    shortint       inputs[$];
    shortint       expected_outputs[$];

    function automatic int read_byte_or_fatal(input int fd, input string what);
        int c;
        c = $fgetc(fd);
        if (c == -1) begin
            $fatal(1, "Unexpected EOF while reading %s", what);
        end
        return c;
    endfunction

    function automatic int parse_shape_1d(input string header);
        int i;
        int len;
        int value;
        bit seen_paren;
        bit collecting;
        byte unsigned ch;

        len = header.len();
        value = 0;
        seen_paren = 0;
        collecting = 0;

        for (i = 0; i < len; i++) begin
            ch = header[i];

            if (!seen_paren) begin
                if (ch == "(") begin
                    seen_paren = 1;
                end
            end else if (!collecting) begin
                if (ch >= "0" && ch <= "9") begin
                    collecting = 1;
                    value = ch - "0";
                end else if (ch == ")") begin
                    break;
                end
            end else begin
                if (ch >= "0" && ch <= "9") begin
                    value = (value * 10) + (ch - "0");
                end else if (ch == "," || ch == ")") begin
                    return value;
                end
            end
        end

        return collecting ? value : -1;
    endfunction

    task automatic load_npy_s16_1d(input string npy_path, ref shortint dst[$]);
        int fd;
        int i;
        int major;
        int minor;
        int header_len;
        int elem_count;
        int c;
        string header;
        byte unsigned b0, b1;

        dst.delete();

        fd = $fopen(npy_path, "rb");
        if (fd == 0) begin
            $fatal(1, "Could not open NPY file: %s", npy_path);
        end

        if (read_byte_or_fatal(fd, "NPY magic[0]") != 8'h93 ||
            read_byte_or_fatal(fd, "NPY magic[1]") != "N"   ||
            read_byte_or_fatal(fd, "NPY magic[2]") != "U"   ||
            read_byte_or_fatal(fd, "NPY magic[3]") != "M"   ||
            read_byte_or_fatal(fd, "NPY magic[4]") != "P"   ||
            read_byte_or_fatal(fd, "NPY magic[5]") != "Y") begin
            $fatal(1, "Invalid NPY magic in file: %s", npy_path);
        end

        major = read_byte_or_fatal(fd, "NPY version major");
        minor = read_byte_or_fatal(fd, "NPY version minor");

        if (major == 1) begin
            b0 = read_byte_or_fatal(fd, "NPY v1 header_len[0]");
            b1 = read_byte_or_fatal(fd, "NPY v1 header_len[1]");
            header_len = int'(b0) + (int'(b1) << 8);
        end else if (major == 2 || major == 3) begin
            byte unsigned b2, b3;
            b0 = read_byte_or_fatal(fd, "NPY v2/3 header_len[0]");
            b1 = read_byte_or_fatal(fd, "NPY v2/3 header_len[1]");
            b2 = read_byte_or_fatal(fd, "NPY v2/3 header_len[2]");
            b3 = read_byte_or_fatal(fd, "NPY v2/3 header_len[3]");
            header_len = int'(b0) + (int'(b1) << 8) + (int'(b2) << 16) + (int'(b3) << 24);
        end else begin
            $fatal(1, "Unsupported NPY version %0d.%0d in %s", major, minor, npy_path);
        end

        header = "";
        for (i = 0; i < header_len; i++) begin
            c = read_byte_or_fatal(fd, "NPY header");
            header = {header, byte'(c)};
        end

        elem_count = parse_shape_1d(header);
        if (elem_count <= 0) begin
            $fatal(1, "Could not parse 1D shape from NPY header in %s. Header=%s", npy_path, header);
        end

        for (i = 0; i < elem_count; i++) begin
            b0 = read_byte_or_fatal(fd, "NPY int16 data[0]");
            b1 = read_byte_or_fatal(fd, "NPY int16 data[1]");
            dst.push_back(shortint'({b1, b0}));
        end

        void'($fclose(fd));
    endtask

    task automatic load_npy_s64_1d(input string npy_path, ref longint signed dst[$]);
        int fd;
        int i;
        int major;
        int minor;
        int header_len;
        int elem_count;
        int c;
        string header;
        byte unsigned b0, b1, b2, b3, b4, b5, b6, b7;
        logic [63:0] raw;

        dst.delete();

        fd = $fopen(npy_path, "rb");
        if (fd == 0) begin
            $fatal(1, "Could not open NPY file: %s", npy_path);
        end

        if (read_byte_or_fatal(fd, "NPY magic[0]") != 8'h93 ||
            read_byte_or_fatal(fd, "NPY magic[1]") != "N"   ||
            read_byte_or_fatal(fd, "NPY magic[2]") != "U"   ||
            read_byte_or_fatal(fd, "NPY magic[3]") != "M"   ||
            read_byte_or_fatal(fd, "NPY magic[4]") != "P"   ||
            read_byte_or_fatal(fd, "NPY magic[5]") != "Y") begin
            $fatal(1, "Invalid NPY magic in file: %s", npy_path);
        end

        major = read_byte_or_fatal(fd, "NPY version major");
        minor = read_byte_or_fatal(fd, "NPY version minor");

        if (major == 1) begin
            b0 = read_byte_or_fatal(fd, "NPY v1 header_len[0]");
            b1 = read_byte_or_fatal(fd, "NPY v1 header_len[1]");
            header_len = int'(b0) + (int'(b1) << 8);
        end else if (major == 2 || major == 3) begin
            b0 = read_byte_or_fatal(fd, "NPY v2/3 header_len[0]");
            b1 = read_byte_or_fatal(fd, "NPY v2/3 header_len[1]");
            b2 = read_byte_or_fatal(fd, "NPY v2/3 header_len[2]");
            b3 = read_byte_or_fatal(fd, "NPY v2/3 header_len[3]");
            header_len = int'(b0) + (int'(b1) << 8) + (int'(b2) << 16) + (int'(b3) << 24);
        end else begin
            $fatal(1, "Unsupported NPY version %0d.%0d in %s", major, minor, npy_path);
        end

        header = "";
        for (i = 0; i < header_len; i++) begin
            c = read_byte_or_fatal(fd, "NPY header");
            header = {header, byte'(c)};
        end

        elem_count = parse_shape_1d(header);
        if (elem_count <= 0) begin
            $fatal(1, "Could not parse 1D shape from NPY header in %s. Header=%s", npy_path, header);
        end

        for (i = 0; i < elem_count; i++) begin
            b0 = read_byte_or_fatal(fd, "NPY int64 data[0]");
            b1 = read_byte_or_fatal(fd, "NPY int64 data[1]");
            b2 = read_byte_or_fatal(fd, "NPY int64 data[2]");
            b3 = read_byte_or_fatal(fd, "NPY int64 data[3]");
            b4 = read_byte_or_fatal(fd, "NPY int64 data[4]");
            b5 = read_byte_or_fatal(fd, "NPY int64 data[5]");
            b6 = read_byte_or_fatal(fd, "NPY int64 data[6]");
            b7 = read_byte_or_fatal(fd, "NPY int64 data[7]");
            raw = {b7, b6, b5, b4, b3, b2, b1, b0};
            dst.push_back(longint'(raw));
        end

        void'($fclose(fd));
    endtask
    // -------------------------------------------------------------------------
    // 3. Test Execution
    // -------------------------------------------------------------------------
    int errors = 0;
    int off_by_one_count = 0;
    int i;
    longint signed actual_val;
    shortint       actual_s16;
    shortint       expected_s16;

    initial begin
        load_npy_s16_1d(INPUT_NPY, inputs);
        load_npy_s16_1d(GOLDEN_NPY, expected_outputs);

        if (expected_outputs.size() != NUM_PES) begin
            $fatal(1,
                "Golden output length (%0d) does not match NUM_PES (%0d). Check TESTCASE_DIR/NUM_PES.",
                expected_outputs.size(), NUM_PES);
        end

        if (inputs.size() == 0) begin
            $fatal(1, "No input rows loaded from %s", INPUT_NPY);
        end

        // Initialize signals
        start = 0;
        last  = 0;
        x = 0;
        rst_n = 0;

        // Reset sequence
        repeat(5) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        $display("=========================================================");
        $display(" Starting Layer Testbench");
        $display("=========================================================");
        $display(" Loaded %0d inputs from %s", inputs.size(), INPUT_NPY);
        $display(" Loaded %0d golden outputs from %s", expected_outputs.size(), GOLDEN_NPY);

        // Feed all inputs through the layer
        for (i = 0; i < inputs.size(); i++) begin
            // Wait for DUT to be ready
            wait(ready);

            // Drive input
            start <= 1'b1;
            x     <= inputs[i];

            // Assert 'last' on the final input to trigger requant
            if (i == inputs.size() - 1)
                last <= 1'b1;

            if (i < 3 || i == inputs.size() - 1)
                $display("Time %0t: Sending input[%0d] = %0d%s", $time, i, inputs[i],
                         (i == inputs.size() - 1) ? " [LAST]" : "");

            @(posedge clk);
            start <= 1'b0;
            last  <= 1'b0;

            // Wait for ready to drop (computation started)
            wait(!ready);
        end

        // Wait for the last computation to complete
        wait(ready);

        // Wait for PE cascade to drain + requant FSM to finish in all PEs
        // done cascade: NUM_PES cycles, done pipeline: 4 cycles, requant FSM: 4 cycles
        repeat(NUM_PES + 50) @(posedge clk);

        $display("");
        $display("=========================================================");
        $display(" Checking Requantized Results");
        $display("=========================================================");

        // Check each PE's requantized accumulator result
        // Allow ±1 tolerance due to decomposed RSHIFT-by-32 in HW
        off_by_one_count = 0;
        for (i = 0; i < expected_outputs.size(); i++) begin
            // Extract the 64-bit acc for PE[i]
            actual_val  = $signed(pe_accum_results[i*64 +: 64]);
            actual_s16  = shortint'(actual_val);
            expected_s16 = expected_outputs[i];
            
            if (actual_s16 === expected_s16) begin
                $display("PE[%0d]: PASS - Value: %0d", i, actual_s16);
            end else if ((actual_s16 == expected_s16 + 1) || (actual_s16 == expected_s16 - 1)) begin
                $display("PE[%0d]: PASS (off-by-one) - Expected: %0d, Got: %0d", 
                         i, expected_s16, actual_s16);
                off_by_one_count++;
            end else begin
                $display("PE[%0d]: FAIL - Expected: %0d, Got: %0d (full acc: %0d)", 
                         i, expected_s16, actual_s16, actual_val);
                errors++;
            end
        end

        $display("");
        $display("=========================================================");
        if (errors == 0) begin
            $display(" TEST PASSED: All %0d PE outputs match expected values!", NUM_PES);
            if (off_by_one_count > 0)
                $display("   (%0d outputs were off-by-one, acceptable for HW RSHIFT-32 decomposition)", off_by_one_count);
        end else begin
            $display(" TEST FAILED: %0d errors found.", errors);
        end
        $display("=========================================================");

        $finish;
    end

endmodule
