import os
from typing import List
import numpy as np
import json
from quantization_utils import q_from_float

# --- CONSTANTS ---
grid_size = 4
degree = 2    
intervals = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

ACC_SHIFT = 3
scale_interval = np.max(np.abs(intervals)) / (2**15 - 1) 
S_in = np.int32(2**15 - 1)
S_u = np.int32(2**13 - 1)
S_b = np.int32(2)

intervals_absmax = np.max(np.abs(intervals))
intervals_q = [np.int16(round(i / scale_interval)) for i in intervals]

# --- LOAD BASIS MATRICES ---
_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_DIR, "algo_experiment_data/a_-1__b_1__gridsize_4__degree_2.json"), "r") as f:
    basis_matrices_json = json.load(f)

basis_matrices = [
    np.array(basis_matrices_json[i]["basis_matrix"], dtype=np.float32)
    for i in range(1, len(basis_matrices_json)-1)
]

basis_matrices_q = [
    np.array(basis_matrices[i] * 2, dtype=np.int16)
    for i in range(len(basis_matrices))
]  

# --- HDL EXPORT UTILITIES ---
def save_mif(filename, data, depth):
    """Saves weight arrays to Altera/Intel Memory Initialization File format."""
    with open(filename, 'w') as f:
        f.write(f"DEPTH = {depth};\n")
        f.write("WIDTH = 16;\n")
        f.write("ADDRESS_RADIX = DEC;\n")
        f.write("DATA_RADIX = HEX;\n")
        f.write("CONTENT BEGIN\n")
        for addr in range(depth):
            val = data[addr] if addr < len(data) else 0
            hex_val = f"{(int(val) & 0xFFFF):04X}"
            f.write(f"{addr} : {hex_val};\n")
        f.write("END;\n")

def save_hex(filename, data, bits=16):
    """Saves arrays to standard hex format for SystemVerilog $readmemh."""
    mask = (1 << bits) - 1
    hex_format = f"0{bits//4}X"
    with open(filename, 'w') as f:
        for val in data:
            f.write(f"{format(int(val) & mask, hex_format)}\n")

# --- FORWARD PASS LOGIC ---
def forward_single_input(x_q: np.int16, c_q: np.ndarray) -> np.int64:
    ux = np.uint16(x_q) ^ 0x8000
    bits_interval_idx = (ux >> 13) & 0b111
    bits_offset = ux & 0x1FFF

    basis_matrix_q_sel = basis_matrices_q[bits_interval_idx]
    basis_matrix_output_int = np.zeros((1, grid_size + degree), dtype=np.int32)
    u_q = np.int32(bits_offset)

    for col in range(basis_matrix_q_sel.shape[1]):
        b0q = np.int32(basis_matrix_q_sel[0, col]) * S_u * S_u 
        b1q = np.int32(basis_matrix_q_sel[1, col]) * S_u
        b2q = np.int32(basis_matrix_q_sel[2, col]) 
        
        term0 = b0q                     
        term1 = u_q * b1q               
        term2 = u_q * u_q * b2q         
        A = term0 + term1 + term2       
        A = A >> ACC_SHIFT 
        
        basis_matrix_output_int[0, col] = A

    output = basis_matrix_output_int.astype(np.int64) @ c_q.T.astype(np.int64) 
    return output[0,0]

def forward_single_output(x_q: np.ndarray, c_q: np.ndarray) -> np.ndarray:
    in_dim = x_q.shape[0]
    output = np.zeros((1,), dtype=np.int64)
    for i in range(in_dim):
        output += forward_single_input(x_q[i], c_q[i, :].reshape((1, grid_size + degree)))
    return output

def forward_single_layer(x_q: np.ndarray, c_q_pkg: List, requant_scale: int, final_layer=False) -> np.ndarray:
    c_q = c_q_pkg[0]
    input_dim, out_dim, _ = c_q.shape
    
    outputs = np.zeros((out_dim, ), dtype=np.int64)
    for o in range(out_dim):
        c_q_o = c_q[:, o, :]
        output_q_o = forward_single_output(x_q, c_q_o)

        if not final_layer:
            mult, shift = requant_scale
            output_q_o = apply_requant_scale(output_q_o, mult, shift)
        outputs[o] = output_q_o[0]

    return outputs.astype(np.int64)

def build_requant_scale(scale_in, scale_out, bit_width=32):
    real_scale = scale_out / scale_in
    shift = bit_width
    multiplier = int(round(real_scale * (1 << shift)))
    return multiplier, shift

def apply_requant_scale(out_q, multiplier, shift):
    return (out_q * multiplier) >> shift

def forward_model(x: np.ndarray, c: List[np.ndarray], save_dir=None):
    num_layers = len(c)
    
    # OFFLINE: Quantize input
    x_q = np.round(( np.clip(x, -intervals_absmax, intervals_absmax) * (2**15 - 1) ) / intervals_absmax).astype(np.int16)
    
    c_q = []
    requant_scales = []
    
    # OFFLINE: Quantize coefficients & calculate requant scales
    for layer_idx in range(num_layers):
        absmax_coef_folded = float(np.max(np.abs(c[layer_idx]))) if np.size(c[layer_idx]) else 0.0    
        I_COEF  = int(np.ceil(np.log2(absmax_coef_folded))) if absmax_coef_folded > 0 else 0
        F_COEF  = max(0, 16 - 1 - I_COEF)                                     
        coef_folded_q = q_from_float(c[layer_idx], F_COEF).astype(np.int16)

        c_q.append([coef_folded_q, F_COEF])
        
        scale_in = np.int64(1 << F_COEF) * np.int64(S_b) * np.int64(S_u) * np.int64(S_u) / (1 << ACC_SHIFT)
        scale_out = (2**15 -1) / intervals_absmax
        mult, shift = build_requant_scale(scale_in, scale_out)
        requant_scales.append([mult, shift])
        
    # ONLINE: Layer by Layer Execution
    for layer_idx in range(num_layers): 
        in_dim, out_dim, K = c[layer_idx].shape
        x_q_layer = x_q.reshape(-1) if x_q.ndim > 1 else x_q

        # HDL EXPORT (Golden References)
        if save_dir is not None:
            layer_dir = os.path.join(save_dir, f"testcase_layer_{layer_idx}")
            mem_init_dir = os.path.join(save_dir, "mem_init")
            os.makedirs(layer_dir, exist_ok=True)
            os.makedirs(mem_init_dir, exist_ok=True)

            # Dump input vector as NPY to match tb_layer.sv
            np.save(os.path.join(layer_dir, "layer_inputs_s16.npy"), x_q_layer.astype(np.int16))

            # Dump MIFs: 1 file per PE per column (layer)
            layer_weights_q = c_q[layer_idx][0]
            for pe in range(out_dim):
                pe_weights = layer_weights_q[:, pe, :].flatten()
                mif_path = os.path.join(mem_init_dir, f"weights_col_{layer_idx}_pe_{pe}.mif")
                save_mif(mif_path, pe_weights, depth=in_dim * K)

        x_q = forward_single_layer(
            x_q_layer,
            c_q[layer_idx],              
            requant_scales[layer_idx],
            final_layer=(layer_idx == num_layers - 1)
        )
        
        # Clamp and extract outputs
        if layer_idx < num_layers - 1:
            x_q = np.clip(x_q, -32768, 32767).astype(np.int16)
            
        if save_dir is not None:
            # Dump expected outputs as NPY to match tb_layer.sv
            np.save(os.path.join(layer_dir, "matvec_requant_out_s16.npy"), x_q.astype(np.int16))

        den = float(1 << c_q[layer_idx][1]) * float(S_b) * float(S_u) * float(S_u) / (1 << ACC_SHIFT)
        x_dq = x_q.astype(np.float32) / den
        
    return x_dq