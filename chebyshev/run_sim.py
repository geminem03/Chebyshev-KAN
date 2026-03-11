import os
import re
import subprocess

def update_testbench(num_inputs, num_outputs, testcase_path, mem_init_path, mif_prefix):
    tb_file = "tb_layer.sv"
    
    with open(tb_file, 'r') as f:
        content = f.read()

    # Define the new configuration block
    new_config = f"""    // =========================================================================
    // CONFIGURATION BLOCK
    // =========================================================================
    localparam integer NUM_OUTPUTS = {num_outputs};
    localparam integer NUM_INPUTS = {num_inputs};
    localparam integer MEM_DEPTH = 256;
    
    localparam TESTCASE_DIR = "{testcase_path}";
    localparam INPUT_NPY = {{TESTCASE_DIR, "/inputs_s16.npy"}};
    localparam GOLDEN_NPY = {{TESTCASE_DIR, "/expected_s64.npy"}};
    
    localparam MIF_PREFIX = "{mem_init_path}/{mif_prefix}";
    // ========================================================================="""

    # Use regex to replace the old block with the new one
    pattern = r"^[ \t]*// ===+.*CONFIGURATION BLOCK.*// ===+[\s\S]*?// ===+.*?$"
    updated_content = re.sub(pattern, new_config, content, flags=re.MULTILINE)

    with open(tb_file, 'w') as f:
        f.write(updated_content)
    
    print(f"[*] Successfully updated {tb_file} ({num_inputs} IN -> {num_outputs} OUT)")

def run_modelsim():
    print("[*] Launching ModelSim simulation in background...")
    
    # Hardcode the path to the free ModelSim Starter Edition executable
    vsim_path = r"C:\intelFPGA\20.1\modelsim_ase\win32aloem\vsim.exe"
    
    # Run vsim in command-line mode (-c), execute run_layer.do, and quit automatically
    cmd = [vsim_path, "-c", "-do", "do run_layer.do; quit -f"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        
        # Parse the output for our specific metrics
        print("\n" + "="*50)
        print(" SIMULATION RESULTS")
        print("="*50)
        
        if "SUCCESS!" in output:
            print(" [PASS] All Processing Engines match the Golden Reference!")
        elif "FAILED" in output:
            import re
            match = re.search(r"FAILED WITH (\d+) ERRORS", output)
            err_count = match.group(1) if match else "UNKNOWN"
            print(f" [FAIL] Simulation failed with {err_count} errors.")
            
        # Extract cycles
        import re
        cycle_match = re.search(r"Total Cycles\s*:\s*(\d+)", output)
        if cycle_match:
            cycles = int(cycle_match.group(1))
            target_fmax = 255.56  # MHz (from Quartus synthesis report)
            latency_ns = cycles * (1000 / target_fmax)
            
            print("-" * 50)
            print(f" Target Fmax : {target_fmax} MHz")
            print(f" Total Cycles: {cycles}")
            print(f" Latency     : {latency_ns:.2f} ns ({latency_ns/1000:.3f} us)")
        print("="*50 + "\n")
        
    except subprocess.CalledProcessError as e:
        print("[!] ModelSim execution failed.")
        print("--- STANDARD ERROR ---")
        print(e.stderr)
        print("--- STANDARD OUTPUT ---")
        print(e.stdout)

if __name__ == "__main__":
    # Hardcoded parameters for Hamiltonian 64x64 dataset
    inputs = 64
    outputs = 64
    
    # Absolute paths for your local chebyshev folder
    tc_path = "C:/Skule_Hardrive/Chebyshev-KAN/chebyshev/testcase_data"
    mi_path = "C:/Skule_Hardrive/Chebyshev-KAN/chebyshev/mem_init"
    mif_pre = "weights_pe_" # Assumed prefix

    # Clean paths for SystemVerilog
    tc_path = tc_path.replace("\\", "/")
    mi_path = mi_path.replace("\\", "/")

    update_testbench(inputs, outputs, tc_path, mi_path, mif_pre)
    run_modelsim()