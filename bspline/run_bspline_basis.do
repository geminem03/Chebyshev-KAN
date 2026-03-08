# ====================================
# ModelSim run script for bspline_basis
# ====================================

# Go to project folder
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
# Adjust this path if your Quartus install differs.
vlog "C:/intelFPGA/20.1/quartus/eda/sim_lib/altera_mf.v"
vlog -sv "C:/intelFPGA/20.1/quartus/eda/sim_lib/altera_lnsim.sv"

# ----------------
# RTL
# ----------------
# Dependencies for bspline_basis
vlog bspline_weights.v
vlog bspline_basis.v

# ----------------
# TESTBENCH
# ----------------
vlog -sv tb_bspline_basis.sv

# ----------------
# SIMULATE
# ----------------
vsim -voptargs=+acc work.tb_bspline_basis

# ----------------
# WAVES
# ----------------
add wave sim:/tb_bspline_basis/clk
add wave sim:/tb_bspline_basis/rst_n
add wave sim:/tb_bspline_basis/start
add wave -radix signed sim:/tb_bspline_basis/x
add wave sim:/tb_bspline_basis/ready
add wave -radix unsigned sim:/tb_bspline_basis/out_valid
add wave -radix unsigned sim:/tb_bspline_basis/out_stride
add wave -radix signed sim:/tb_bspline_basis/out_data

# ----------------
# INTERNAL DEBUG
# ----------------
add wave -radix unsigned sim:/tb_bspline_basis/dut/cycle_cnt
add wave -radix unsigned sim:/tb_bspline_basis/dut/active_matrix_idx
add wave -radix unsigned sim:/tb_bspline_basis/dut/prev_start_idx
add wave sim:/tb_bspline_basis/dut/busy

add wave -r sim:/tb_bspline_basis/dut/*

# ----------------
# RUN
# ----------------
run -all
