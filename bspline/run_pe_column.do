# ================================
# ModelSim run script for pe_column
# ================================

# Go to project root
cd C:/repos/kan-tpu/bspline

# Clean
if {[file exists work]} {
    vdel -all
}

vlib work
vmap work work

# ----------------
# Intel SIM LIBS
# ----------------
vlog "C:/intelFPGA/20.1/quartus/eda/sim_lib/altera_mf.v"
vlog "C:/intelFPGA/20.1/quartus/eda/sim_lib/altera_lnsim.sv"

# ----------------
# RTL
# ----------------
vlog -sv pe.sv
vlog pe_column.sv

# ----------------
# TESTBENCH
# ----------------
vlog tb_pe_column.sv

# ----------------
# SIMULATE
# ----------------
vsim -voptargs=+acc work.tb_pe_column

# ----------------
# WAVES
# ----------------
add wave sim:/tb_pe_column/clk
add wave sim:/tb_pe_column/sclr
add wave -radix signed sim:/tb_pe_column/data_in
add wave -radix signed sim:/tb_pe_column/all_acc_results
add wave -r sim:/tb_pe_column/dut/*

# ----------------
# RUN
# ----------------
run -all
