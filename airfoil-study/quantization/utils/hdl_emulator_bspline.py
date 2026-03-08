import os
from typing import List
import numpy as np
from quantization_utils import q_from_float
import json


SAVE_HDL_DATA = True

# Add this near the top of forward_optimal.py
_DIR = os.path.dirname(os.path.abspath(__file__))

"""
Saving outputs for HDL comparison
"""

# eveery layer that has this shape will be saved
hdl_input_dim = 64
hdl_output_dim = 64


hdl_save_path = r"C:\repos\kan-quant-strats\regression_test\hdl_data"
hdl_inputs = np.zeros((hdl_input_dim, ), dtype=np.int16)
hdl_acc_outputs = np.zeros((hdl_output_dim, ), dtype=np.int64)
hdl_requant_outputs = np.zeros((hdl_output_dim, ), dtype=np.int16)
hdl_basis_vals = np.zeros((hdl_input_dim*6, ), dtype=np.int32)
hdl_weights = np.zeros((hdl_output_dim, hdl_input_dim*6), dtype=np.int16)

"""
CONSTANTS
"""

grid_size = 4
degree = 2    
# start points (>=)
intervals = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

ACC_SHIFT = 3
scale_interval = np.max(np.abs(intervals)) / (2**15 - 1) 
S_in = np.int32(2**15 - 1)  # scale for input x
S_u = np.int32(2**13 - 1)   # scale for u
S_b = np.int32(2)           # scale for coefficients. all values are in { -1.0, -0.5, 0.0, 0.5, 1.0 }, so quantize with scale 2.0

intervals_absmax = np.max(np.abs(intervals))
intervals_q = [np.int16(round(i / scale_interval)) for i in intervals]





"""
LOAD and QUANTIZE basis matrices
"""

# fetch basis matrices from JSON file
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





def forward_single_input(x_q: np.int16, c_q: np.ndarray, verbose=False, save_hdl_idx=-1) -> np.int64:
      
        
    # just flip the sign bit to get unsigned representation
    ux = np.uint16(x_q) ^ 0x8000

    # grab interval idx bits 15:13 (3 bits)
    bits_interval_idx = (ux >> 13) & 0b111

    # grab offset bits 12:0 (13 bits)
    bits_offset = ux & 0x1FFF


    basis_matrix_q_sel = basis_matrices_q[bits_interval_idx]
    basis_matrix_output_int = np.zeros((1, grid_size + degree), dtype=np.int32)


    u_q = np.int32(bits_offset) # since we accumulate in int32

    for col in range(basis_matrix_q_sel.shape[1]):
        
        # adjust the basis matrix coefficients by their scale S_u
        
        # acc in int32, but only 2+13+13=28 bits wide at most
        b0q = np.int32(basis_matrix_q_sel[0, col]) * S_u * S_u 
        
        # acc in int32, but only 2+13=15 bits wide at most
        b1q = np.int32(basis_matrix_q_sel[1, col]) * S_u
        
        # acc in int32, but only 2 bits wide at most
        b2q = np.int32(basis_matrix_q_sel[2, col]) 
        
        # we do the above basis matrix scaling OFFLINE, since we know S_u and basis matrix ahead of time
        # we the below dot-product computation ONLINE
        
        # Integer accumulator: A = S_u^2*b0q + S_u*u_q*b1q + u_q^2*b2q
        term0 = b0q                     # u0 = 1, u0*b0q = b0q
        term1 = u_q * b1q               # 13 + 15 = fits in 28 bits 
        term2 = u_q * u_q * b2q         # 13 + 13 + 2 = fits in 28 bits
        A = term0 + term1 + term2       # sum fits in 32 bits safely
        A = A >> ACC_SHIFT # Summing 3 28-bit nums can yield up to 30 bits, so right shift by 3 to fit in 27-bits
        
        basis_matrix_output_int[0, col] = A
        

    # debug: dequantize back to FP32: y ≈ A / (S_b * S_u^2)
    den = S_b * (S_u * S_u) / (1 << ACC_SHIFT)
    
    basis_matrix_output_deq = basis_matrix_output_int.astype(np.float64) / den
    
    if save_hdl_idx >= 0:
        # print(basis_matrix_output_int.shape) #1, 6
        # flatten and save to hdl_inputs
        flattened_basis_matrix_output_int = basis_matrix_output_int.flatten()
        hdl_basis_vals[save_hdl_idx : save_hdl_idx+6] = flattened_basis_matrix_output_int.astype(np.int32)
    
    
    if verbose:
        print("[forward_single_input] input x_q =", x_q)
        # print("[forward_single_input] input x deq =", (x_q * intervals_absmax) / 32767.0)
        # print("[forward_single_input] basis_matrix_q_sel =", basis_matrix_q_sel)
        print("[forward_single_input] basis_matrix_output =", basis_matrix_output_int)
        # print("[forward_single_input] basis_matrix_output_deq =", np.round(basis_matrix_output_deq, 3))

    # debug: compare with fp32 basis matrix computation
    # basis_matrix_fp32 = basis_matrices[bits_interval_idx]
    # start_interval = intervals[bits_interval_idx]
    # u_fp = ( ( x_q / 32767.0 ) - start_interval ) / 0.5
    # power_basis_fp32_0 = 1.0
    # power_basis_fp32_1 = u_fp
    # power_basis_fp32_2 = u_fp * u_fp
    # basis_matrix_output_fp32 = np.zeros((1, grid_size + degree), dtype=np.float32)
    # for col in range(basis_matrix_fp32.shape[1]):
    #     basis_matrix_output_fp32[0, col] = (
    #         power_basis_fp32_0 * basis_matrix_fp32[0, col] +
    #         power_basis_fp32_1 * basis_matrix_fp32[1, col] +
    #         power_basis_fp32_2 * basis_matrix_fp32[2, col]
    #     )


    # now do matrix multiplication with c_q, which should be shape [1, grid_size + degree]
    assert c_q.shape == (1, grid_size + degree)
    
    output = basis_matrix_output_int.astype(np.int64) @ c_q.T.astype(np.int64)  # shape (1, 1)
    
    
    

    
    return output[0,0]


def forward_single_output(x_q: np.ndarray, c_q: np.ndarray, verbose=False, save_hdl=False) -> np.ndarray:
    in_dim = x_q.shape[0]
    
    # c_q is expected to be shape (in_dim, grid_size + degree)
    assert c_q.shape == (in_dim, grid_size + degree), \
        f"c_q shape {c_q.shape} != expected {(in_dim, grid_size + degree)}"
    
    output = np.zeros((1,), dtype=np.int64)
    
    # dbg_array = np.zeros((in_dim, 1), dtype=np.float32)
    
    for i in range(in_dim):
        # pass row i of coefficients as shape (1, grid_size + degree)
        output += forward_single_input(
            x_q[i],
            c_q[i, :].reshape((1, grid_size + degree)),
            save_hdl_idx=-1 if not save_hdl else i*6,
            verbose=False and i==0)  # each input produces 6 basis outputs
        
     

    return output



def forward_single_layer(x_q: np.ndarray, c_q: List, requant_scale: int, final_layer=False, verbose=False) -> np.ndarray:
    c_FRAC = c_q[1]
    c_q = c_q[0]
    
    input_dim, out_dim, _ = c_q.shape
    
    
    assert tuple(x_q.shape) == (input_dim,), (
        f"x_q shape {tuple(x_q.shape)} != expected ({input_dim},) "
        f"- x_q has {x_q.shape[0]} elements"
    )    
    
    save_outs = False
    if (SAVE_HDL_DATA and input_dim == hdl_input_dim and out_dim == hdl_output_dim):
        save_outs = True
        print("Forwarding layer with input_dim=output_dim=10")
        
        # print x_q shape
        print("x_q shape:", x_q.shape)
        hdl_inputs = x_q.astype(np.int16)
        
        # print c_q shape
        print("c_q shape:", c_q.shape)
        
        # c_q is shape I, O, K, where I=10, O=10, K=6, and is type int16
        # we need to rearrange into O, I*K for HDL comparison
        # so first reshape to O, I, K, then flatten last two dims
        c_q_reshaped = c_q.transpose((1, 0, 2)).reshape((out_dim, input_dim * (grid_size + degree)))
        hdl_weights[:, :] = c_q_reshaped.astype(np.int16)
        
    
    outputs = np.zeros((out_dim, ), dtype=np.int64)
    for o in range(out_dim):
        # select coefficients for output o: shape (input_dim, grid_size + degree)
        c_q_o = c_q[:, o, :]
        
        # aggregate all input dims into a single scalar for this output
        output_q_o = forward_single_output(x_q, c_q_o, save_hdl=save_outs and (o==0))

        hdl_acc_outputs[o] = output_q_o
        if not final_layer:
            # now apply requant scale
            mult, shift = requant_scale
            output_q_o = apply_requant_scale(output_q_o, mult, shift)

        
        # output_q_o_scaled is shape (1,), take scalar
        outputs[o] = output_q_o[0]

    if save_outs:
        np.save(os.path.join(hdl_save_path, f"layer_inputs_s16.npy"), hdl_inputs)
        np.save(os.path.join(hdl_save_path, f"matvec_out_s64.npy"), hdl_acc_outputs)
        np.save(os.path.join(hdl_save_path, f"matvec_weights_s16.npy"), hdl_weights.astype(np.int16))
        np.save(os.path.join(hdl_save_path, f"matvec_inputs_s27.npy"), hdl_basis_vals.astype(np.int32))
        
        
        # verify that hdl_basis_vals @ hdl_weights.T == hdl_outputs
        # print shape of inputs, weights, outputs
        print("hdl_basis_vals shape:", hdl_basis_vals.shape) # (60,)
        print("hdl_weights shape:", hdl_weights.shape) # (10, 60)
        print("hdl_outputs shape:", hdl_acc_outputs.shape) # (10,)
        
        # hdl_weights (10, 60) * hdl_basis_vals (60,) should equal hdl_outputs (10,)
        hdl_verif_outputs = hdl_weights.astype(np.int64) @ hdl_basis_vals.astype(np.int64)
        
        if not np.array_equal(hdl_verif_outputs, hdl_acc_outputs):
            print("Warning: HDL verification outputs do not match saved outputs")
        else:
            print("HDL verification outputs match saved outputs")
        


    return outputs.astype(np.int64)




def build_requant_scale(scale_in, scale_out, bit_width=32):
    """
    Offline function.
    Computes a fixed-point multiplier and shift such that:

        out_q / scale_in * scale_out ≈ (out_q * multiplier) >> shift

    Returns:
        multiplier (int)
        shift (int)
    """
    real_scale = scale_out / scale_in
    shift = bit_width
    multiplier = int(round(real_scale * (1 << shift)))
    return multiplier, shift


def apply_requant_scale(out_q, multiplier, shift):
    """
    Online function.
    Integer-only arithmetic, no division.
    """
    return (out_q * multiplier) >> shift


def forward_model(x: np.ndarray, c: List[np.ndarray]):
    """
    x: input data, shape (1, input_dim)
    c: list of coefficient arrays
    """
    
    num_layers = len(c)
    
    """
    OFFLINE:
     - Pre-quantize input x to int32 with scale S_in
     - For each layer, quantize coefficients, and build requant scales
    """
    
    # quantize input x using symmetric quantization (use abs max of intervals)
    # x_q = np.round(x * (2**15 - 1) / np.max(np.abs(intervals))).astype(np.int16)
    x_q = np.round(
            ( np.clip(x, -intervals_absmax, intervals_absmax) * (2**15 - 1) )
            / intervals_absmax  
        ).astype(np.int16)
    
    
    c_q = []
    requant_scales = []
    
    for layer_idx in range(num_layers):
        
        # use symmatric quantization for coefficients
        # find abs max
        absmax_coef_folded = float(np.max(np.abs(c[layer_idx]))) if np.size(c[layer_idx]) else 0.0    
        I_COEF  = int(np.ceil(np.log2(absmax_coef_folded))) if absmax_coef_folded > 0 else 0
        F_COEF  = max(0, 16 - 1 - I_COEF)                                     # int16: 1 sign bit
        coef_folded_q = q_from_float(c[layer_idx], F_COEF).astype(np.int16)

            
        # Scale to INT16 using absmax
        c_q.append([coef_folded_q, F_COEF])
        
        # build requant scale for next layer
        scale_in = np.int64(1 << F_COEF) * np.int64(S_b) * np.int64(S_u) * np.int64(S_u) / (1 << ACC_SHIFT)
        scale_out = (2**15 -1) / intervals_absmax
        
        mult, shift = build_requant_scale(scale_in, scale_out)
        
        requant_scales.append([mult, shift])
        

        
    """
    ONLINE: run layers, using quantized coefficients
    """
    
    for layer_idx in range(num_layers): # DEBUG: start from layer 1 (final layer only)
        
        in_dim, out_dim, _ = c[layer_idx].shape
        
        # ensure x_q has expected shape (input_dim,)
        
        if x_q.ndim > 1:
            x_q_layer = x_q.reshape(-1)
        else:
            x_q_layer = x_q

        assert x_q_layer.shape == (in_dim,), f"x_q shape {x_q_layer.shape} != expected {(in_dim,)}"

        x_q = forward_single_layer(
            x_q_layer,
            c_q[layer_idx],              # use quantized coefficients
            requant_scales[layer_idx],
            final_layer=(layer_idx == num_layers - 1)
        )
        
        # clamp, shouldn't be needed, but just in case
        if layer_idx < num_layers - 1:
            x_q = np.clip(x_q, -32768, 32767).astype(np.int16)

        den = float(1 << c_q[layer_idx][1]) * float(S_b) * float(S_u) * float(S_u) / (1 << ACC_SHIFT)
        x_dq = x_q.astype(np.float32) / den
        
        
    return x_dq
    

# input_x = np.load("algo_experiment_data/input_x.npy").astype(np.float32)
# coef_fp32_layer0 = np.load("algo_experiment_data/coef_fp32_layer0.npy").astype(np.float32)
# coef_fp32_layer1 = np.load("algo_experiment_data/coef_fp32_layer1.npy").astype(np.float32)

# output = forward_model(input_x, [coef_fp32_layer0, coef_fp32_layer1])

# print("Final output:\n", output)