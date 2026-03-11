"""
hdl_emulator_chebyshev.py

Bit-exact fixed-point emulator for the Chebyshev KAN hardware.
Matches the RTL in pe_quad.sv and layer.sv numerically.

Updated to dump per-layer testcase directories in the format expected
by the configurable tb_layer.sv testbench:

    <layer_dir>/
        layer_inputs_s16.npy       — int16, shape (num_inputs,)
        golden_acc_s64.npy         — int64, shape (num_outputs,)
        golden_requant_s16.npy     — int16, shape (num_outputs,)
        mem_init/
            weights_pe_0.mif ... weights_pe_{num_outputs-1}.mif

Each .mif has (DEGREE+1) * ceil4(num_inputs) * 4 coefficient rows,
with the LAST row containing the requant scale factor.
Total depth = num_coeff_rows + 1.

Hardware requant formula:
    requant_out = int16( (accumulator * scale) >>> REQUANT_SHIFT )

For inter-layer requant matching the original Q5.10 -> Q0.15 mapping:
    REQUANT_SHIFT = 5,  scale = 1024  =>  (acc * 1024) >>> 5 = acc * 32 = acc << 5
"""

import os
import numpy as np
from typing import List, Optional, Tuple

# ============================================================================
# Constants
# ============================================================================
GLOBAL_OPS_SHIFT = 10          # Q5.10 fractional bits
FRAC_BITS = GLOBAL_OPS_SHIFT
S_QUANT = 1 << GLOBAL_OPS_SHIFT  # 1024

# Default requant parameters (inter-layer: acc * 32 = acc << 5)
DEFAULT_REQUANT_SHIFT = 5
DEFAULT_REQUANT_SCALE = 1 << GLOBAL_OPS_SHIFT  # 1024, so (acc*1024)>>>5 = acc*32


# ============================================================================
# MIF writer
# ============================================================================
def save_mif(filename: str, data_dict: dict, depth: int):
    """Write an Altera/Intel Memory Initialization File (.mif)."""
    with open(filename, 'w') as f:
        f.write(f"DEPTH = {depth};\n")
        f.write("WIDTH = 16;\n")
        f.write("ADDRESS_RADIX = DEC;\n")
        f.write("DATA_RADIX = HEX;\n")
        f.write("CONTENT BEGIN\n")
        for addr in range(depth):
            val = data_dict.get(addr, 0)
            hex_val = f"{(int(val) & 0xFFFF):04X}"
            f.write(f"{addr} : {hex_val};\n")
        f.write("END;\n")


# ============================================================================
# Bit-exact Clenshaw evaluation (matches pe_quad.sv)
# ============================================================================
def _clenshaw_int16(x_in: np.ndarray, c_fp: np.ndarray,
                    out_q16_frac: int = 10) -> np.ndarray:
    """
    Fixed-point Clenshaw recurrence matching pe_quad.sv exactly.

    Parameters
    ----------
    x_in : int16 scalar (shape (1,))
        Input in Q0.15 (first layer) or Q5.10 (if pre-quantized).
        NOTE: the caller passes the raw int16 value; this function
        handles the T normalization internally.
    c_fp : float32 array, shape (degree+1,)
        FP32 Chebyshev coefficients for one (input, output) edge.

    Returns
    -------
    int16 scalar — Clenshaw output in Q5.10
    """
    assert x_in.shape == (1,)

    k = GLOBAL_OPS_SHIFT
    S = 1 << k

    # Quantize coefficients (same as offline weight quantization)
    c_fp = c_fp.astype(np.float32)
    c_q_raw = np.round(c_fp * S)
    c_q = np.clip(c_q_raw, -2147483648, 2147483647).astype(np.int32)

    # Map input to Q5.10 Clenshaw domain
    # In hardware: x is int16, T = (x * 1024 + rounding) / 32767
    den = 32767
    X = x_in.astype(np.int32)
    half = den // 2
    T = (X.astype(np.int64) * S + np.where(X >= 0, half, -half)) // den
    T = T.astype(np.int32)

    # Clenshaw recurrence (matches pe_quad.sv pipeline)
    b1 = np.zeros_like(T, dtype=np.int64)
    b2 = np.zeros_like(T, dtype=np.int64)
    N = c_q.shape[0] - 1

    for j in range(N, 0, -1):
        prod = T.astype(np.int64) * b1
        # k > 0: multiply by 2x (scaled_product = mult_reg <<< 1)
        tmp_qk = ((2 * prod + (1 << (k - 1))) >> k)
        tmp = tmp_qk - b2 + np.int64(c_q[j])
        b2, b1 = b1, tmp

    # Final step: k == 0, multiply by 1x (scaled_product = mult_reg, no shift)
    prod_final = T.astype(np.int64) * b1.astype(np.int64)
    y_qk = ((prod_final + (1 << (k - 1))) >> k)
    y_qk = y_qk - b2 + np.int64(c_q[0])

    # Adjust fractional bits if needed
    if out_q16_frac != k:
        diff = out_q16_frac - k
        if diff > 0:
            y_qk = y_qk << diff
        else:
            y_qk = y_qk >> (-diff)

    y_out = np.clip(y_qk, -32768, 32767).astype(np.int16)
    return y_out


# ============================================================================
# Requant computation (matches layer.sv REQUANT state)
# ============================================================================
def _compute_requant_int16(acc: np.int64, scale: np.int16,
                           requant_shift: int) -> np.int16:
    """
    Compute saturate_s16((acc * scale) >>> requant_shift).
    Matches the RTL saturate_s16 function in layer.sv.
    """
    product = int(acc) * int(scale)
    if requant_shift > 0:
        # Arithmetic right shift (Python >> on negative is arithmetic)
        shifted = product >> requant_shift
    else:
        shifted = product
    # Saturate to [-32767, 32767] (matching RTL saturate_s16)
    clamped = max(-32767, min(32767, shifted))
    return np.int16(clamped)


# ============================================================================
# Per-layer forward pass + testcase export
# ============================================================================
def layer_forward(x, quantize_inputs: bool, final_layer: bool,
                  coeff: np.ndarray, out_q16_frac: int = 10,
                  requant_shift: int = DEFAULT_REQUANT_SHIFT,
                  requant_scale: int = DEFAULT_REQUANT_SCALE,
                  save_dir: Optional[str] = None, layer_idx: int = 0):
    """
    Fixed-point forward pass for one Chebyshev KAN layer.

    Parameters
    ----------
    x : ndarray
        Input array. If quantize_inputs=True, float in [-1,1] shape (B, in_dim).
        If False, int16 shape (B, in_dim) already quantized.
    quantize_inputs : bool
        Whether to quantize float inputs to Q0.15.
    final_layer : bool
        If True, return raw accumulator (no requant between layers).
    coeff : ndarray, shape (in_dim, out_dim, degree+1)
        FP32 Chebyshev coefficients.
    out_q16_frac : int
        Output fractional bits (default 10 for Q5.10).
    requant_shift : int
        Right-shift for requant: (acc * scale) >>> requant_shift.
    requant_scale : int
        Scale factor for requant (stored as last .mif row).
    save_dir : str or None
        If provided, dump testcase files for this layer.
    layer_idx : int
        Layer index for directory naming.

    Returns
    -------
    (quantized_output, dequantized_output)
    """
    in_dim, out_dim, num_coeffs = coeff.shape
    degree = num_coeffs - 1

    # --- Quantize inputs ---
    if quantize_inputs:
        t = np.clip(x.astype(np.float32), -1.0, 1.0)
        x_quant = np.round(t * 32767.0).astype(np.int16)
    else:
        assert x.dtype == np.int16
        x_quant = x

    B = x_quant.shape[0]
    y_q = np.zeros((B, out_dim), dtype=np.int64)

    # --- Computation (bit-exact with hardware) ---
    for i in range(in_dim):
        xi = x_quant[:, i]
        for o in range(out_dim):
            c_fp = coeff[i, o, :]
            y_i_o = _clenshaw_int16(xi, c_fp, out_q16_frac=out_q16_frac)
            y_q[:, o] += y_i_o.astype(np.int64)

    # --- Requant (bit-exact with hardware REQUANT state) ---
    y_requant = np.zeros((B, out_dim), dtype=np.int16)
    for b in range(B):
        for o in range(out_dim):
            y_requant[b, o] = _compute_requant_int16(
                y_q[b, o], np.int16(requant_scale), requant_shift
            )

    # --- Export testcase ---
    if save_dir is not None:
        _dump_layer_testcase(
            save_dir=save_dir,
            layer_idx=layer_idx,
            in_dim=in_dim,
            out_dim=out_dim,
            degree=degree,
            coeff=coeff,
            x_quant=x_quant,
            y_q=y_q,
            y_requant=y_requant,
            requant_scale=requant_scale,
        )

    # --- Return values ---
    if final_layer:
        y_dequant = (y_q.astype(np.float64) / float(1 << out_q16_frac)).astype(np.float32)
        return y_q, y_dequant
    else:
        # Pass requantized int16 to next layer
        l2_dq = y_requant.astype(np.float32) / 32767.0
        return y_requant, l2_dq


# ============================================================================
# Testcase dumper
# ============================================================================
def _dump_layer_testcase(save_dir: str, layer_idx: int,
                         in_dim: int, out_dim: int, degree: int,
                         coeff: np.ndarray,
                         x_quant: np.ndarray,
                         y_q: np.ndarray,
                         y_requant: np.ndarray,
                         requant_scale: int):
    """
    Dump per-layer testcase in the format expected by tb_layer.sv.

    Directory structure:
        <save_dir>/layer_<idx>_<in>x<out>/
            layer_inputs_s16.npy
            golden_acc_s64.npy
            golden_requant_s16.npy
            mem_init/
                weights_pe_0.mif ... weights_pe_{out_dim-1}.mif
    """
    layer_name = f"layer_{layer_idx}_{in_dim}x{out_dim}"
    layer_dir = os.path.join(save_dir, layer_name)
    mem_init_dir = os.path.join(layer_dir, "mem_init")
    os.makedirs(mem_init_dir, exist_ok=True)

    # --- Save numpy test vectors ---
    # Convert Q0.15 inputs to Q5.10 for the hardware
    x_q15 = x_quant[0, :]
    half = 32767 // 2
    x_hw = (x_q15.astype(np.int32) * S_QUANT + np.where(x_q15 >= 0, half, -half)) // 32767
    x_hw = x_hw.astype(np.int16)

    # Use first sample (batch index 0) for hardware testcase
    np.save(os.path.join(layer_dir, "layer_inputs_s16.npy"), x_hw)
    
    np.save(os.path.join(layer_dir, "golden_acc_s64.npy"),
            y_q[0, :].astype(np.int64))
    np.save(os.path.join(layer_dir, "golden_requant_s16.npy"),
            y_requant[0, :].astype(np.int16))

    # --- Memory layout ---
    # Matches layer.sv parameterized addressing:
    #   num_groups    = ceil(in_dim / 4)
    #   mem_size      = num_groups * 4
    #   coeffs_per_in = degree + 1
    #   num_coeff_rows = mem_size * coeffs_per_in
    #   mif_depth      = num_coeff_rows + 1   (+1 for requant scale)
    #
    # Address formula:
    #   addr = group * 4 * coeffs_per_in + thread * coeffs_per_in + (degree - k)
    #
    # Last row (addr = num_coeff_rows) = requant scale factor

    num_groups = (in_dim + 3) // 4
    mem_size = num_groups * 4
    coeffs_per_in = degree + 1
    num_coeff_rows = mem_size * coeffs_per_in
    mif_depth = num_coeff_rows + 1

    for o in range(out_dim):
        mif_data = {}

        # Write Chebyshev coefficients
        for i in range(in_dim):
            group = i // 4
            thread = i % 4

            c_fp = coeff[i, o, :].astype(np.float32)
            c_q_raw = np.round(c_fp * S_QUANT)
            c_q = np.clip(c_q_raw, -32768, 32767).astype(np.int32)

            for k_idx in range(degree + 1):
                addr = (group * 4 * coeffs_per_in
                        + thread * coeffs_per_in
                        + (degree - k_idx))
                mif_data[addr] = c_q[k_idx]

        # Write requant scale as last row
        mif_data[num_coeff_rows] = int(requant_scale) & 0xFFFF

        # Write .mif file
        # Write .mif file matching RTL naming (no zero-padding)
        mif_filename = os.path.join(mem_init_dir, f"weights_pe_{o}.mif")
        save_mif(mif_filename, mif_data, depth=mif_depth)

    # --- Write gen_tb_config.py command for convenience ---
    config_cmd = (
        f"python gen_tb_config.py"
        f" --num_inputs {in_dim}"
        f" --num_outputs {out_dim}"
        f" --testcase_dir {layer_name}"
        f" --degree {degree}"
        f" --requant_shift {DEFAULT_REQUANT_SHIFT}"
    )
    with open(os.path.join(layer_dir, "run_config.txt"), 'w') as f:
        f.write(f"# Generate tb_config.svh for this layer:\n")
        f.write(f"{config_cmd}\n")
        f.write(f"\n")
        f.write(f"# MIF depth = {mif_depth}\n")
        f.write(f"# Requant scale = {requant_scale} (0x{requant_scale & 0xFFFF:04X})\n")
        f.write(f"# REQUANT_SHIFT = {DEFAULT_REQUANT_SHIFT}\n")
        f.write(f"# (acc * {requant_scale}) >>> {DEFAULT_REQUANT_SHIFT} = acc * {requant_scale * (2**-DEFAULT_REQUANT_SHIFT):.1f}\n")

    print(f"  [Layer {layer_idx}] Dumped testcase to {layer_dir}/")
    print(f"    Inputs: {in_dim}, Outputs: {out_dim}, MIF depth: {mif_depth}")
    print(f"    Requant: (acc * {requant_scale}) >>> {DEFAULT_REQUANT_SHIFT}")
    print(f"    Config:  {config_cmd}")


# ============================================================================
# Top-level model forward (airfoil: [5, 16, 1])
# ============================================================================
def forward_airfoil(x, coeff, out_q16_frac=10, save_dir=None):
    """
    Full model forward pass for the airfoil Chebyshev KAN [5, 16, 1].

    Parameters
    ----------
    x : ndarray, shape (1, 5)
        Single sample, float in [-1, 1].
    coeff : list of ndarray
        coeff[0]: shape (5, 16, degree+1) — layer 0 weights
        coeff[1]: shape (16, 1, degree+1) — layer 1 weights
    out_q16_frac : int
        Fractional bits (default 10).
    save_dir : str or None
        If provided, dump per-layer testcase directories.

    Returns
    -------
    (output_q, output_dq) — int64 raw accumulator and float32 dequantized
    """
    # Layer 0: 5 -> 16 (with requant for inter-layer transfer)
    l2_q_input, l2_dq_input = layer_forward(
        x,
        quantize_inputs=True,
        final_layer=False,
        coeff=coeff[0],
        out_q16_frac=out_q16_frac,
        requant_shift=DEFAULT_REQUANT_SHIFT,
        requant_scale=DEFAULT_REQUANT_SCALE,
        save_dir=save_dir,
        layer_idx=0,
    )

    # Layer 1: 16 -> 1 (final layer, no requant needed)
    output_q, output_dq = layer_forward(
        l2_q_input,
        quantize_inputs=False,
        final_layer=True,
        coeff=coeff[1],
        out_q16_frac=out_q16_frac,
        requant_shift=DEFAULT_REQUANT_SHIFT,
        requant_scale=DEFAULT_REQUANT_SCALE,
        save_dir=save_dir,
        layer_idx=1,
    )

    return output_q, output_dq
