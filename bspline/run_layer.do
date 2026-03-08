# ====================================
# ModelSim run script for layer
# ====================================

# Go to project folder
cd C:/Skule_Hardrive/Chebyshev-KAN/bspline

# Clean
if {[file exists work]} {
    vdel -all
}

vlib work
vmap work work

# ----------------
# Intel SIM LIBS
# ----------------
# Adjust this path if your Quartus install differs.
vlog "C:/intelFPGA/20.1/quartus/eda/sim_lib/altera_mf.v"
vlog -sv "C:/intelFPGA/20.1/quartus/eda/sim_lib/altera_lnsim.sv"

# ----------------
# RTL
# ----------------
# Dependencies for layer module
vlog bspline_weights.v
vlog bspline_basis.v
vlog -sv pe.sv
vlog -sv pe_column.sv
vlog layer.v

# ----------------
# TESTBENCH
# ----------------
vlog -sv tb_layer.sv

# ----------------
# SIMULATE
# ----------------
vsim -voptargs=+acc work.tb_layer

# ----------------
# WAVES
# ----------------
add wave sim:/tb_layer/clk
add wave sim:/tb_layer/rst_n
add wave sim:/tb_layer/start
add wave -radix signed sim:/tb_layer/x
add wave sim:/tb_layer/ready

# Internal signals from layer
add wave -divider "BSpline Basis"
add wave sim:/tb_layer/dut/u_basis/out_valid
add wave -radix unsigned sim:/tb_layer/dut/u_basis/out_stride
add wave -radix signed sim:/tb_layer/dut/u_basis/out_data

# PE Column accumulator results
add wave -divider "PE Accumulators"
add wave -radix signed sim:/tb_layer/pe_accum_results

# Plot only first 5 and last 5 PEs, regardless of NUM_PES
# Derive NUM_PES by probing generated PE accumulator objects.
set num_pes 0
for {set i 0} {$i < 1024} {incr i} {
    set pe_path [format {sim:/tb_layer/dut/u_pe_column/gen_pe[%d]/col_mif/pe_mif/pe_inst} $i]
    set acc_path "$pe_path/acc"
    if {[catch [list examine $acc_path]]} {
        break
    }
    incr num_pes
}

set selected_indices {}
set first_count [expr {$num_pes < 5 ? $num_pes : 5}]
for {set i 0} {$i < $first_count} {incr i} {
    lappend selected_indices $i
}

set last_start [expr {$num_pes > 5 ? $num_pes - 5 : 0}]
for {set i $last_start} {$i < $num_pes} {incr i} {
    if {[lsearch -exact $selected_indices $i] < 0} {
        lappend selected_indices $i
    }
}

echo [format "Wave PE selection from NUM_PES=%d: %s" $num_pes $selected_indices]

# Individual PE results (first 5 + last 5)
add wave -divider "Individual PE Results"
foreach i $selected_indices {
    set pe_path [format {sim:/tb_layer/dut/u_pe_column/gen_pe[%d]/col_mif/pe_mif/pe_inst} $i]
    set acc_path "$pe_path/acc"
    catch [list add wave -radix signed $acc_path]
}

# Individual PE read addresses
add wave -divider "Individual PE Read Addresses"
foreach i $selected_indices {
    set pe_path [format {sim:/tb_layer/dut/u_pe_column/gen_pe[%d]/col_mif/pe_mif/pe_inst} $i]
    set read_addr_path "$pe_path/read_addr"
    catch [list add wave -radix unsigned $read_addr_path]
}

# Individual PE weight regs
add wave -divider "Individual PE Weight Registers"
foreach i $selected_indices {
    set pe_path [format {sim:/tb_layer/dut/u_pe_column/gen_pe[%d]/col_mif/pe_mif/pe_inst} $i]
    set weight_reg_path "$pe_path/weight_reg"
    catch [list add wave -radix signed $weight_reg_path]
}

# Individual PE input_in and input_reg signals
add wave -divider "Individual PE Input Signals"
foreach i $selected_indices {
    set pe_path [format {sim:/tb_layer/dut/u_pe_column/gen_pe[%d]/col_mif/pe_mif/pe_inst} $i]
    set input_in_path "$pe_path/input_in"
    set input_reg_path "$pe_path/input_reg"
    catch [list add wave -radix signed $input_in_path]
    catch [list add wave -radix signed $input_reg_path]
}

# all other signals in the DUT, so we can check without adding individually
add wave -r sim:/tb_layer/dut/*

# ----------------
# RUN
# ----------------
run -all
