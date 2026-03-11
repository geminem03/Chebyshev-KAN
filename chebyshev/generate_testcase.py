#!/usr/bin/env python3
"""
generate_testcase.py

Generate a synthetic testcase for the Chebyshev KAN layer hardware.
Produces all files needed to run tb_layer.sv via gen_tb_config.py.

Usage:
    # 64x64 testcase for Quartus validation
    python generate_testcase.py --num_inputs 64 --num_outputs 64

    # Airfoil layer 0
    python generate_testcase.py --num_inputs 5 --num_outputs 16

    # Airfoil layer 1
    python generate_testcase.py --num_inputs 16 --num_outputs 1

    # Custom with specific seed for reproducibility
    python generate_testcase.py --num_inputs 64 --num_outputs 64 --seed 42

Output structure (created in --output_dir):
    <name>/
        layer_inputs_s16.npy
        golden_acc_s64.npy
        golden_requant_s16.npy
        mem_init/
            weights_pe_0.mif ... weights_pe_{N-1}.mif
        tb_config.svh          (ready to copy into RTL directory)
        run_config.txt         (reference info)
"""

import os
import sys
import argparse
import numpy as np

# ---------------------------------------------------------------------------
# Import the emulator (adjust path as needed)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# We inline the key functions so this script is fully standalone,
# but also try to import from hdl_emulator_chebyshev if available.
try:
    from hdl_emulator_chebyshev import (
        save_mif, _clenshaw_int16, _compute_requant_int16,
        GLOBAL_OPS_SHIFT, DEFAULT_REQUANT_SHIFT, DEFAULT_REQUANT_SCALE
    )
    HAVE_EMULATOR = True
except ImportError:
    HAVE_EMULATOR = False

# ---------------------------------------------------------------------------
# save_hex is always defined at module level (not in the emulator)
# ---------------------------------------------------------------------------
def save_hex(filename, data_dict, depth):
    """Write a $readmemh-compatible hex file (one value per line)."""
    with open(filename, 'w') as f:
        for addr in range(depth):
            val = data_dict.get(addr, 0)
            f.write(f"{(int(val) & 0xFFFF):04X}\n")

# ---------------------------------------------------------------------------
# Fallback constants if emulator not importable
# ---------------------------------------------------------------------------
if not HAVE_EMULATOR:
    GLOBAL_OPS_SHIFT = 10
    DEFAULT_REQUANT_SHIFT = 5
    DEFAULT_REQUANT_SCALE = 1 << GLOBAL_OPS_SHIFT  # 1024

    def save_mif(filename, data_dict, depth):
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

    def _clenshaw_int16(x_in, c_fp, out_q16_frac=10):
        assert x_in.shape == (1,)
        k = GLOBAL_OPS_SHIFT
        S = 1 << k
        c_fp = c_fp.astype(np.float32)
        c_q_raw = np.round(c_fp * S)
        c_q = np.clip(c_q_raw, -2147483648, 2147483647).astype(np.int32)
        den = 32767
        X = x_in.astype(np.int32)
        half = den // 2
        T = (X.astype(np.int64) * S + np.where(X >= 0, half, -half)) // den
        T = T.astype(np.int32)
        b1 = np.zeros_like(T, dtype=np.int64)
        b2 = np.zeros_like(T, dtype=np.int64)
        N = c_q.shape[0] - 1
        for j in range(N, 0, -1):
            prod = T.astype(np.int64) * b1
            tmp_qk = ((2 * prod + (1 << (k - 1))) >> k)
            tmp = tmp_qk - b2 + np.int64(c_q[j])
            b2, b1 = b1, tmp
        prod_final = T.astype(np.int64) * b1.astype(np.int64)
        y_qk = ((prod_final + (1 << (k - 1))) >> k)
        y_qk = y_qk - b2 + np.int64(c_q[0])
        if out_q16_frac != k:
            diff = out_q16_frac - k
            if diff > 0:
                y_qk = y_qk << diff
            else:
                y_qk = y_qk >> (-diff)
        y_out = np.clip(y_qk, -32768, 32767).astype(np.int16)
        return y_out

    def _compute_requant_int16(acc, scale, requant_shift):
        product = int(acc) * int(scale)
        shifted = product >> requant_shift if requant_shift > 0 else product
        clamped = max(-32767, min(32767, shifted))
        return np.int16(clamped)


# ---------------------------------------------------------------------------
# Testcase generator
# ---------------------------------------------------------------------------
def generate_testcase(
    num_inputs: int,
    num_outputs: int,
    degree: int = 3,
    requant_shift: int = DEFAULT_REQUANT_SHIFT,
    requant_scale: int = DEFAULT_REQUANT_SCALE,
    seed: int = 42,
    output_dir: str = ".",
    name: str = None,
    weight_range: float = 0.5,
):
    """
    Generate a complete synthetic testcase.

    Parameters
    ----------
    num_inputs, num_outputs : int
        Layer dimensions.
    degree : int
        Chebyshev polynomial degree.
    requant_shift, requant_scale : int
        Requant parameters matching RTL.
    seed : int
        Random seed for reproducibility.
    output_dir : str
        Parent directory for output.
    name : str or None
        Testcase subdirectory name. Auto-generated if None.
    weight_range : float
        FP32 coefficient range [-weight_range, +weight_range].
    """
    rng = np.random.RandomState(seed)

    if name is None:
        name = f"testcase_{num_inputs}x{num_outputs}"

    testcase_dir = os.path.join(output_dir, name)
    mem_init_dir = os.path.join(testcase_dir, "mem_init")
    os.makedirs(mem_init_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Generate random FP32 coefficients: shape (num_inputs, num_outputs, degree+1)
    # ------------------------------------------------------------------
    num_coeffs = degree + 1
    coeff_fp32 = rng.uniform(-weight_range, weight_range,
                             size=(num_inputs, num_outputs, num_coeffs)).astype(np.float32)

    # ------------------------------------------------------------------
    # 2. Generate random inputs in [-1, 1], quantize to Q0.15
    # ------------------------------------------------------------------
    x_float = rng.uniform(-0.95, 0.95, size=(num_inputs,)).astype(np.float32)
    x_quant = np.round(x_float * 32767.0).astype(np.int16)

    # ------------------------------------------------------------------
    # 3. Compute golden outputs (bit-exact with hardware)
    # ------------------------------------------------------------------
    S = 1 << GLOBAL_OPS_SHIFT

    # Accumulator: sum of Clenshaw outputs per output PE
    y_acc = np.zeros(num_outputs, dtype=np.int64)

    for i in range(num_inputs):
        xi = np.array([x_quant[i]], dtype=np.int16)
        for o in range(num_outputs):
            c_fp = coeff_fp32[i, o, :]
            y_i_o = _clenshaw_int16(xi, c_fp, out_q16_frac=GLOBAL_OPS_SHIFT)
            y_acc[o] += int(y_i_o[0])

    # Requant: (acc * scale) >>> shift, saturated to int16
    y_requant = np.zeros(num_outputs, dtype=np.int16)
    for o in range(num_outputs):
        y_requant[o] = _compute_requant_int16(y_acc[o], np.int16(requant_scale), requant_shift)

    # ------------------------------------------------------------------
    # 4. Write .npy test vectors
    # ------------------------------------------------------------------
    # Convert Q0.15 inputs to Q5.10 for the hardware
    half = 32767 // 2
    x_hw = (x_quant.astype(np.int32) * S + np.where(x_quant >= 0, half, -half)) // 32767
    x_hw = x_hw.astype(np.int16)

    np.save(os.path.join(testcase_dir, "layer_inputs_s16.npy"), x_hw)
    np.save(os.path.join(testcase_dir, "golden_acc_s64.npy"), y_acc)
    np.save(os.path.join(testcase_dir, "golden_requant_s16.npy"), y_requant)

    # ------------------------------------------------------------------
    # 5. Write .mif files (one per output PE)
    # ------------------------------------------------------------------
    num_groups = (num_inputs + 3) // 4
    mem_size = num_groups * 4
    coeffs_per_in = num_coeffs
    num_coeff_rows = mem_size * coeffs_per_in
    mif_depth = num_coeff_rows + 1

    for o in range(num_outputs):
        mif_data = {}

        for i in range(num_inputs):
            group = i // 4
            thread = i % 4

            c_fp = coeff_fp32[i, o, :].astype(np.float32)
            c_q_raw = np.round(c_fp * S)
            c_q = np.clip(c_q_raw, -32768, 32767).astype(np.int32)

            for k_idx in range(num_coeffs):
                addr = (group * 4 * coeffs_per_in
                        + thread * coeffs_per_in
                        + (degree - k_idx))
                mif_data[addr] = c_q[k_idx]

        # Requant scale as last row
        mif_data[num_coeff_rows] = int(requant_scale) & 0xFFFF

        # Write .mif file matching RTL naming:
        #   gi < 10:  weights_pe_0.mif  (1 digit)
        #   gi < 100: weights_pe_10.mif (2 digits)
        #   else:     weights_pe_100.mif (3 digits)
        save_mif(os.path.join(mem_init_dir, f"weights_pe_{o}.mif"), mif_data, depth=mif_depth)
        save_hex(os.path.join(mem_init_dir, f"weights_pe_{o}.hex"), mif_data, depth=mif_depth)

    # ------------------------------------------------------------------
    # 6. Write tb_config.svh (ready to drop into RTL directory)
    # ------------------------------------------------------------------
    mif_prefix = f"{name}/mem_init/weights_pe_"
    config_content = f"""\
// Auto-generated by generate_testcase.py
// Testcase: {num_inputs} x {num_outputs}, Degree={degree}, Seed={seed}
localparam integer NUM_INPUTS     = {num_inputs};
localparam integer NUM_OUTPUTS    = {num_outputs};
localparam integer DEGREE         = {degree};
localparam integer ACC_WIDTH      = 22;
localparam integer REQUANT_SHIFT  = {requant_shift};
localparam string  TESTCASE_DIR   = "{name}";
localparam string  MIF_PREFIX     = "{mif_prefix}";
"""
    with open(os.path.join(testcase_dir, "tb_config.svh"), 'w') as f:
        f.write(config_content)

    # ------------------------------------------------------------------
    # 7. Write reference info
    # ------------------------------------------------------------------
    info = f"""\
Testcase: {name}
Dimensions: {num_inputs} x {num_outputs}
Degree: {degree}
Seed: {seed}
Weight range: [-{weight_range}, {weight_range}]
MIF depth: {mif_depth} ({num_coeff_rows} coeff rows + 1 scale row)
Requant: (acc * {requant_scale}) >>> {requant_shift}
Input range: [{x_quant.min()}, {x_quant.max()}]
Accumulator range: [{y_acc.min()}, {y_acc.max()}]
Requant output range: [{y_requant.min()}, {y_requant.max()}]

To run in ModelSim:
  1. Copy {name}/tb_config.svh to your RTL directory
  2. Copy {name}/ folder to your RTL directory
  3. do run_layer.do
"""
    with open(os.path.join(testcase_dir, "run_config.txt"), 'w') as f:
        f.write(info)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"Generated testcase: {testcase_dir}/")
    print(f"  Dimensions:   {num_inputs} x {num_outputs}")
    print(f"  Degree:       {degree}")
    print(f"  MIF depth:    {mif_depth}")
    print(f"  Seed:         {seed}")
    print(f"  Acc range:    [{y_acc.min()}, {y_acc.max()}]")
    print(f"  Requant range:[{y_requant.min()}, {y_requant.max()}]")
    print(f"  Files:")
    print(f"    {testcase_dir}/layer_inputs_s16.npy")
    print(f"    {testcase_dir}/golden_acc_s64.npy")
    print(f"    {testcase_dir}/golden_requant_s16.npy")
    print(f"    {testcase_dir}/mem_init/weights_pe_0..{num_outputs-1}.mif")
    print(f"    {testcase_dir}/tb_config.svh")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic testcase for Chebyshev KAN layer hardware"
    )
    parser.add_argument("--num_inputs",    type=int, required=True)
    parser.add_argument("--num_outputs",   type=int, required=True)
    parser.add_argument("--degree",        type=int, default=3)
    parser.add_argument("--requant_shift", type=int, default=DEFAULT_REQUANT_SHIFT)
    parser.add_argument("--requant_scale", type=int, default=DEFAULT_REQUANT_SCALE)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--output_dir",    type=str, default=".")
    parser.add_argument("--name",          type=str, default=None,
                        help="Testcase directory name (default: testcase_NxM)")
    parser.add_argument("--weight_range",  type=float, default=0.5,
                        help="Coefficient range [-r, r] (default: 0.5)")
    args = parser.parse_args()

    generate_testcase(
        num_inputs=args.num_inputs,
        num_outputs=args.num_outputs,
        degree=args.degree,
        requant_shift=args.requant_shift,
        requant_scale=args.requant_scale,
        seed=args.seed,
        output_dir=args.output_dir,
        name=args.name,
        weight_range=args.weight_range,
    )
