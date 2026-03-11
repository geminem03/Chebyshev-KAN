import os
import re
import subprocess

# --- PERMANENT PATH CONFIGURATION ---
CHEBY_DIR = "C:/Skule_Hardrive/Chebyshev-KAN/chebyshev"
DATA_BASE = "C:/Skule_Hardrive/Chebyshev-KAN/airfoil-study/experiments/quantization_results/chebyshev_testcase"
VSIM_PATH = "C:/intelFPGA/20.1/modelsim_ase/win32aloem/vsim.exe"

def generate_config(num_inputs, num_outputs, testcase_path, mem_init_path, mif_prefix):
    config_file = os.path.join(CHEBY_DIR, "tb_config.svh")
    
    config_content = f"""// =========================================================================
// AUTO-GENERATED CONFIGURATION (Do not edit manually)
// =========================================================================
localparam integer NUM_OUTPUTS = {num_outputs};
localparam integer NUM_INPUTS = {num_inputs};
localparam integer MEM_DEPTH = 256;

localparam TESTCASE_DIR = "{testcase_path}";
localparam INPUT_NPY = {{TESTCASE_DIR, "/inputs_s16.npy"}};
localparam GOLDEN_NPY = {{TESTCASE_DIR, "/expected_s64.npy"}};

localparam MIF_PREFIX = "{mem_init_path}/{mif_prefix}";
// ========================================================================="""

    with open(config_file, 'w') as f:
        f.write(config_content)
    print(f"[*] Generated fresh tb_config.svh in {CHEBY_DIR}")

def run_modelsim():
    print("[*] Launching ModelSim in background...")
    cmd = [VSIM_PATH, "-c", "-do", "do run_layer.do; quit -f"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=CHEBY_DIR)
        output = result.stdout
        
        print("\n" + "="*50)
        print(" AIRFOIL SIMULATION RESULTS")
        print("="*50)
        
        if "SUCCESS!" in output:
            print(" [PASS] All Processing Engines match the Golden Reference!")
        elif "FAILED" in output:
            match = re.search(r"FAILED WITH (\d+) ERRORS", output)
            err_count = match.group(1) if match else "UNKNOWN"
            print(f" [FAIL] Simulation failed with {err_count} errors.")
            print("-" * 50)
            print(" HARDWARE VS EXPECTED MISMATCHES:")
            for line in output.split('\n'):
                if "FAIL PE" in line:
                    print(f"   {line.strip()}")
        else:
            print(" [!] The simulation crashed or failed to compile.")
            print(" --- RAW MODELSIM OUTPUT ---")
            print(output)
            
        cycle_match = re.search(r"Total Cycles\s*:\s*(\d+)", output)
        if cycle_match:
            cycles = int(cycle_match.group(1))
            target_fmax = 255.56
            latency_ns = cycles * (1000 / target_fmax)
            
            print("-" * 50)
            print(f" Target Fmax : {target_fmax} MHz")
            print(f" Total Cycles: {cycles}")
            print(f" Latency     : {latency_ns:.2f} ns ({latency_ns/1000:.3f} us)")
        print("="*50 + "\n")
        
    except subprocess.CalledProcessError as e:
        print("[!] ModelSim execution failed.")
        print("--- ERROR OUTPUT ---")
        print(e.stderr if e.stderr else e.stdout)

if __name__ == "__main__":
    print("\n--- NASA Airfoil Dataset Simulator ---")
    
    n_in = input(" Enter Number of Inputs  : ").strip()
    n_out = input(" Enter Number of Outputs : ").strip()
    layer_num = input(" Enter Layer Index (0/1) : ").strip()
    
    tc_path = f"{DATA_BASE}/testcase_layer_{layer_num}"
    mi_path = f"{DATA_BASE}/mem_init"
    mif_pre = f"weights_l{layer_num}_pe_"

    generate_config(n_in, n_out, tc_path, mi_path, mif_pre)
    run_modelsim()