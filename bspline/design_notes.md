# Arria 10 DSP Systolic Array

SystemVerilog implementation of a 1D systolic array optimized for Intel Arria 10 FPGAs. The design leverages hardened **DSP blocks** for arithmetic and **MLABs** for local weight storage, creating a high-performance compute pipeline suitable for filtering or matrix multiplication operations.

**Notes:**
1. Used SystemVerilog, but code is all just normal verilog, makes compilation easier when using string filenames for `.mif`.
2. The actual way to arrange the `fourierkan` or `bspline kan` forward pass into a large matrix-vector multiply problem is detailed here: `verilator\bspline\mat-vec-mult.md`.


## Architecture Overview

The design is a "weight-stationary" systolic array where:

* **Weights** are pre-loaded into local memory (MLABs) inside each Processing Element (PE).
* **Inputs** flow down a cascade chain (scan chain) from one PE to the next.
* **Results** are accumulated locally within each PE's hardened DSP accumulator.

This means, for a PE Column of size 10, only on the 10th cycle will the last (bottom) PE start doing useful work. But this is just the one-time cost (proportional to # PEs) to "fill up" the systolic array.

Each PE does work for its own row of the weight matrix.

### Key Features

* **Hardened DSP Cascade:** Uses the dedicated `input_cascade` wires between DSP blocks to route data, avoiding general FPGA fabric routing and maximizing `fMax`.
* **MLAB Weight Storage:** Weights are stored in Memory Logic Array Blocks (MLABs) configured as ROMs/RAMs next to the DSPs, ensuring zero-latency weight access.
* **Parameterizable Precision:** Configurable bit-widths for inputs, weights, and accumulators.

## Parameters & Configuration

| Parameter | Default | Description | Constraints |
| --- | --- | --- | --- |
| `INPUT_WIDTH` | 27 | Width of the streaming input data. | Max **27** (27x27 mode) or **18** (18x18 mode with cascade. I don't think 18x19 mode supports cascade). |
| `WEIGHT_WIDTH` | 16 | Width of the stationary weights. | Fits within standard 18x19 or 27x27 DSP limits. |
| `ACC_WIDTH` | 64 | Width of the output accumulator. | Hardened DSP accumulators are 64-bit. |
| `NUM_PES` | 2 | Number of PEs in a column. | Limited by the height of the DSP column in the device. |

## Module Hierarchy

### 1. `pe.sv` (Processing Element)

The fundamental compute unit.

* **MLAB (`altsyncram`):** Stores a vector of weights locally. It cycles through weights using an internal counter.
* **DSP Logic:** Performs the MAC operation: `Acc <= Acc + (Input × Weight)`.
* **Cascade Interface:** Passes the `input_in` directly to `input_out` for the next PE in the column.

### 2. `pe_column.sv` (Column Controller)

Instantiates a chain of PEs.

* **Cascade Wiring:** Connects the `input_out` of PE[n] to `input_in` of PE[n+1].
* **MIF Management:** Handles the loading of unique weight files (`.mif`) for each PE using a string array parameter.
* **Result Bus:** Aggregates all accumulator outputs into a single flattened bus.

## Implementation Notes

### Arria 10 DSP Modes

The current design defaults to **27x27 mode** because the `INPUT_WIDTH` is set to 27.

#### Optimization Warning: 18x19 Mode

If attempting to optimize for density by switching to **18x19 mode**, note the following critical constraint regarding the **Input Cascade**:

* **Mode:** Fixed-point 18 x 19 Independent Multiplication.
* **Constraint:** The dedicated input cascade wire width is **limited to 18 bits**.
* **Impact:** You **cannot** use a 19-bit input with the cascade chain. To use the cascade feature, `INPUT_WIDTH` must be reduced to **18 bits**. Using 19 bits would force routing through soft logic (ALMs), degrading timing performance.

## Usage

### 1. Instantiation

```systemverilog
pe_column #(
    .NUM_PES(4),
    .INPUT_WIDTH(27),
    .ACC_WIDTH(64)
) my_systolic_array (
    .clk(sys_clk),
    .sclr(reset),
    .data_in(stream_data),
    .all_acc_results(results_bus)
);

```