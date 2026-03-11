# ============================================================================
# Chebyshev Layer Simulation Script
#
# Usage:
#   1. Generate tb_config.svh for your target layer (see gen_tb_config.py)
#   2. cd into the directory containing the RTL + testbench
#   3. do run_layer.do
# ============================================================================

# Create and map the work library
vdel -all -lib work
vlib work
vmap work work

# Compile the SystemVerilog RTL files
vlog -sv pe_quad.sv
vlog -sv layer.sv

# Compile the Testbench
vlog -sv tb_layer.sv

# Load simulation, link Intel libraries, suppress disconnected port warnings
vsim -suppress 2685,3722 -voptargs=+acc -L altera_mf_ver -L lpm_ver work.tb_layer

# --- TOP LEVEL CONTROLS ---
add wave -divider "System Controls"
add wave -position insertpoint sim:/tb_layer/clk
add wave -position insertpoint sim:/tb_layer/rst_n
add wave -position insertpoint sim:/tb_layer/start
add wave -position insertpoint sim:/tb_layer/dut/state
add wave -position insertpoint sim:/tb_layer/ready

# --- MEMORY & TDM TRACKING ---
add wave -divider "Memory & TDM (PE 0)"
add wave -radix unsigned -position insertpoint sim:/tb_layer/dut/group_idx
add wave -radix unsigned -position insertpoint sim:/tb_layer/dut/pes[0]/core/thread_idx
add wave -radix unsigned -position insertpoint sim:/tb_layer/dut/pes[0]/core/k
add wave -radix unsigned -position insertpoint sim:/tb_layer/dut/mem_addr
add wave -radix hexadecimal -position insertpoint sim:/tb_layer/dut/pe_coeff_in[0]

# --- PIPELINE STAGES (PE 0) ---
add wave -divider "PE[0] Stage 1: Inputs"
add wave -radix decimal -position insertpoint sim:/tb_layer/dut/pes[0]/core/reg_x
add wave -radix decimal -position insertpoint sim:/tb_layer/dut/pes[0]/core/reg_b

add wave -divider "PE[0] Stage 2: Multiply"
add wave -radix decimal -position insertpoint sim:/tb_layer/dut/pes[0]/core/mult_reg

add wave -divider "PE[0] Stage 3: Shift & Add"
add wave -radix decimal -position insertpoint sim:/tb_layer/dut/pes[0]/core/term
add wave -radix decimal -position insertpoint sim:/tb_layer/dut/pes[0]/core/add_reg

add wave -divider "PE[0] Stage 4: Writeback & Output"
add wave -radix decimal -position insertpoint sim:/tb_layer/dut/pes[0]/core/b_next[0]
add wave -radix decimal -position insertpoint sim:/tb_layer/dut/pes[0]/core/y_A
add wave -radix decimal -position insertpoint sim:/tb_layer/dut/accumulator[0]

# --- REQUANT ---
#add wave -divider "Requant"
#add wave -radix unsigned -position insertpoint sim:/tb_layer/dut/rq_step
#add wave -radix decimal  -position insertpoint sim:/tb_layer/dut/requant_scale[0]
#add wave -radix decimal  -position insertpoint sim:/tb_layer/dut/requant_out[0]

# Run the simulation
run 10 us

# Zoom the waveform to fit
wave zoom full
