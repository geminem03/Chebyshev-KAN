module bspline_weights (
    input  [2:0]  idx,
    output reg [26:0] weights
);

    always @(*) begin
        case (idx)

            // [[0 0 0]
            //  [0 0 0]
            //  [1 0 0]]
            3'd0: weights = {
                3'sd0, 3'sd0, 3'sd0,
                3'sd0, 3'sd0, 3'sd0,
                3'sd1, 3'sd0, 3'sd0
            };

            // [[ 1 0 0]
            //  [ 2 0 0]
            //  [-2 1 0]]
            3'd1: weights = {
                3'sd1,  3'sd0, 3'sd0,
                3'sd2,  3'sd0, 3'sd0,
               -3'sd2,  3'sd1, 3'sd0
            };

            // [[ 1 1 0]
            //  [-2 2 0]
            //  [ 1 -2 1]]
            3'd2: weights = {
                3'sd1,  3'sd1, 3'sd0,
               -3'sd2,  3'sd2, 3'sd0,
                3'sd1, -3'sd2, 3'sd1
            };

            // identical interior spline matrices
            // [[ 1 1 0]
            //  [-2 2 0]
            //  [ 1 -2 1]]
            3'd3,
            3'd4,
            3'd5: weights = {
                3'sd1,  3'sd1, 3'sd0,
               -3'sd2,  3'sd2, 3'sd0,
                3'sd1, -3'sd2, 3'sd1
            };

            // [[0 1 1]
            //  [0 -2 2]
            //  [0 1 -2]]
            3'd6: weights = {
                3'sd0,  3'sd1,  3'sd1,
                3'sd0, -3'sd2,  3'sd2,
                3'sd0,  3'sd1, -3'sd2
            };

            // [[0 0 1]
            //  [0 0 -2]
            //  [0 0 1]]
            3'd7: weights = {
                3'sd0, 3'sd0,  3'sd1,
                3'sd0, 3'sd0, -3'sd2,
                3'sd0, 3'sd0,  3'sd1
            };

            default: weights = 27'sd0;
        endcase
    end
endmodule
