import os
import random

def generate_coefficients():
    # --- CONFIGURATION MATCHING YOUR HARDWARE ---
    NUM_PES = 64        # 64 Output Neurons 
    NUM_INPUTS = 64     # 64 Inputs per Neuron 
    DEGREE = 3          # Degree 3 Polynomial 
    WIDTH = 16          # 16-bit Fixed Point 
    FRAC_BITS = 10      # Q5.10 format 
    
    COEFF_DIR = "coeffs" # Subfolder for .mem files 
    
    # 1. Create Subfolder 
    os.makedirs(COEFF_DIR, exist_ok=True)
    print(f"Generating {NUM_PES} coefficient files in ./{COEFF_DIR}/...")

    # 2. Generate 64 Memory Files (One per PE) 
    for bank_id in range(NUM_PES):
        filename = os.path.join(COEFF_DIR, f"bank_{bank_id}.mem")
        with open(filename, 'w') as f:
            # Each file needs 'NUM_INPUTS' lines (64 lines total) 
            for i in range(NUM_INPUTS):
                packed_val = 0
                # Pack 4 coeffs (Degree 0, 1, 2, 3) into one 64-bit word 
                for d in range(DEGREE + 1):
                    # Generate random float and convert to fixed-point 
                    float_val = random.uniform(-1.5, 1.5)
                    int_val = int(round(float_val * (2**FRAC_BITS)))
                    
                    # Handle Two's Complement for negative numbers 
                    if int_val < 0: 
                        int_val = (1 << WIDTH) + int_val
                    
                    # Ensure 16-bit masking and shift into 64-bit packed word 
                    int_val = int_val & 0xFFFF
                    packed_val = (packed_val << WIDTH) | int_val
                
                # Write 64-bit hex value to the file 
                f.write(f"{packed_val:016x}\n")

    print("Coefficient generation complete.")

if __name__ == "__main__":
    generate_coefficients()