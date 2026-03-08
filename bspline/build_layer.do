# ====================================
# ModelSim build-only script for layer
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
