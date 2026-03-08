## Weight Initialization

Before running a testcase in ModelSim or building the Quartus project, manually copy the testcase `.mif` files from `bspline/testcase_data/<testcase_name>/` into `bspline/mem_init/`.

The `.mifs` use the naming pattern `weights_col_<col>_pe_<pe>.mif` (for example: `weights_col_0_pe_0.mif`).

Important: the files you copy must match your instantiated design dimensions.

* Include one file per PE index for each column used by the design.
* Ensure the copied `(col, pe)` coverage matches the configured number of columns and `NUM_PES` (from `layer.v`).
* If you only provide `weights_col_0_pe_*`, the active design configuration must only require column 0.
* Ensure the number of rows in each `.mif` matches the expected `MEM_DEPTH` for your design configuration (from `layer.v`).

## Quartus Compilation

Open the Quartus project (`bspline.qpf`), set `layer.v` as the top-level module, and compile. 

For accurate HW utilization results, configure these paramters in `layer.v` before compiling:

```
parameter NUM_PES = 64,
parameter MEM_DEPTH = 384
```

You must copy the matching `.mif` files into `mem_init/` as described above for the design to compile successfully. For NUM_PES=64, you need `weights_col_0_pe_0.mif` through `weights_col_0_pe_63.mif` (64 files total) in `mem_init/`, and each `.mif` should have 384 rows (for MEM_DEPTH=384).

In general:
- `NUM_PES` should equal the number of output channels for the testcase you want to run.
- `MEM_DEPTH` should be equal to input channels * 6 for the testcase you want to run.


Optionally, you can use the scripts in `kan-tpu/scripts` to set top level module and build the Quartus project from command line.


## Testing with ModelSim Simulation

Currently, testbenches (`tb_*.sv`) are implemented for the following modules:
1. `layer.v`
2. `pe_column.sv`
3. `bspline_basis.v`

Testcase data is located in `testcase_data/`, and current the only working testcases are for the `bspline_basis` module (single testcase) and `layer` module (2 testcases, one for 10x10 layer and one for 64x64 layer).


**Running Testcases:**
Each `tb_*.sv` file has a corresponding `.do` file that sets up the simulation and runs the testbench. To run a testcase, you have two options:

1. Open ModelSim GUI. In the ModelSim terminal, `cd` into the `kan-tpu/bspline` directory and execute the corresponding `.do` file. For example, to run the `tb_pe_column.sv` testcase, execute `do run_pe_column.do`.

2. (Recommended) Use the `run_modelsim_do.ps1` script from the `scripts` folder to launch ModelSim and execute the `.do` file immediately. For example, from the `scripts` folder, run `./run_modelsim_do.ps1 ../bspline/run_layer.do`. This will open ModelSim and run the `tb_layer.sv` testcase.


Make sure to change the file paths in the `.do` file to direct ModelSim to your Quartus installation location. (TODO: use `kan-tpu/scripts/quartus_install_path.txt` and `kan-tpu/scripts/modelsim_install_path.txt` to automate this in the future). 


For the end-to-end `layer` testbench, manually set the following in `tb_layer.sv` before running the simulation:

```
NUM_PES (10 or 64),
MEM_DEPTH (60 for 10x10 testcase, 384 for 64x64 testcase)
TESTCASE_DIR ("testcase_data/layer_10x10/" or "testcase_data/layer_64x64/")
```

## Measuring Performance
For the end-to-end `layer` testbench, you can measure the number of cycles taken for the forward pass by looking at the ModelSim wave plots.

Calculate total latency of a layer forward pass:
1. Start = the cycle when `rst_n` is first deasserted (goes from 0 to 1).
2. End = the cycle when the `pe_accum_results` bus stabilizes to the final output values (after the last PE finishes its computation).
3. Num Cycles = (End - Start) / 10 (since the clk period is 10ns in the testbench).
4. Total Latency = Num Cycles * Frequency. Frequency is determined by the `fMax` of the design, which can be found in the Quartus compilation report.