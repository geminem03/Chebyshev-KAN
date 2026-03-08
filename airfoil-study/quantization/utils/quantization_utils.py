
import numpy as np

"""
forward.py: Manual NumPy implementations of MatrixKAN forward passes in FP32 and INT16 quantized precision.
"""
def basis_matrix_for_x(k: int, grid_size: int, x_input: float, domain=(-1.0, 1.0)) -> np.ndarray:
    """
    Return the per-interval basis_matrix of shape (k+1, grid_size + k) that maps
    [1, u, u^2, ..., u^k] (u in [0,1] on the local interval) to the B-spline basis values
    for a uniform open-clamped spline on [domain[0], domain[1]], replicating MatrixKAN's
    interval selection and padded-gather placement. If x is outside the extended grid
    [a - k*h, b + k*h), return all zeros (as observed in MatrixKAN).
    """
    a, b = domain
    G = grid_size
    h = (b - a) / G

    # ---- Early exit: outside extended domain => all zeros ----
    ext_left  = a - k * h
    ext_right = b + k * h
    if not (ext_left <= x_input < ext_right):
        return np.zeros((k + 1, G + k), dtype=np.float64)

    # ---- Build the (k+1)x(k+1) local kernel (matches MatrixKAN.calculate_basis_matrix) ----
    basis = np.array([[1.0]], dtype=np.float64)
    scalar = 1.0
    p = 2
    while p <= k + 1:
        term_1 = np.pad(basis, ((0, 1), (0, 0)), mode='constant')  # pad one zero row at bottom
        term_3 = np.pad(basis, ((1, 0), (0, 0)), mode='constant')  # pad one zero row at top

        term_2 = np.zeros((p - 1, p), dtype=np.float64)
        term_4 = np.zeros((p - 1, p), dtype=np.float64)
        for i in range(p - 1):
            term_2[i, i] = i + 1
            term_2[i, i + 1] = p - (i + 2)
            term_4[i, i] = -1.0
            term_4[i, i + 1] =  1.0

        basis = term_1 @ term_2 + term_3 @ term_4
        scalar *= 1.0 / (p - 1)
        p += 1
    basis *= scalar  # (k+1) x (k+1)

    # ---- Interval selection: clip to extended grid indices [-k .. (G-1)+k] ----
    s_raw = int(np.floor((x_input - a) / h))
    s = int(np.clip(s_raw, -k, (G - 1) + k))

    # ---- Embed into (k+1) x (G+k) with zero-padding/truncation like padded gather ----
    n_basis = G + k
    M = np.zeros((k + 1, n_basis), dtype=np.float64)

    col_start = s
    col_end   = s + (k + 1)

    left  = max(col_start, 0)
    right = min(col_end, n_basis)

    if left < right:
        src_l = left - col_start
        src_r = right - col_start
        M[:, left:right] = basis[:, src_l:src_r]

    return M


def basis_matrix_for_index(k: int, grid_size: int, idx: int, domain=(-1.0, 1.0)) -> np.ndarray:
    """
    Version of basis_matrix_for_x that uses an interval index instead of x.
    This matches the same (k+1)x(k+1) local kernel and the same embedding
    into (k+1) x (G+k), but takes the extended-grid interval index directly.

    idx runs over [0 .. G + 2k - 1], corresponding to s in [-k .. (G-1)+k].
    """
    a, b = domain
    G = grid_size
    h = (b - a) / G

    # Map LUT index -> extended-grid index s in [-k .. (G-1)+k]
    s_min = -k
    s = s_min + idx    # same "s" used in basis_matrix_for_x

    # ---- Build the (k+1)x(k+1) local kernel (same as basis_matrix_for_x) ----
    basis = np.array([[1.0]], dtype=np.float64)
    scalar = 1.0
    p = 2
    while p <= k + 1:
        term_1 = np.pad(basis, ((0, 1), (0, 0)), mode='constant')
        term_3 = np.pad(basis, ((1, 0), (0, 0)), mode='constant')

        term_2 = np.zeros((p - 1, p), dtype=np.float64)
        term_4 = np.zeros((p - 1, p), dtype=np.float64)
        for i in range(p - 1):
            term_2[i, i]     = i + 1
            term_2[i, i + 1] = p - (i + 2)
            term_4[i, i]     = -1.0
            term_4[i, i + 1] =  1.0

        basis = term_1 @ term_2 + term_3 @ term_4
        scalar *= 1.0 / (p - 1)
        p += 1
    basis *= scalar

    # ---- Embed into (k+1) x (G+k) exactly like basis_matrix_for_x ----
    n_basis = G + k
    M = np.zeros((k + 1, n_basis), dtype=np.float64)

    col_start = s
    col_end   = s + (k + 1)

    left  = max(col_start, 0)
    right = min(col_end, n_basis)

    if left < right:
        src_l = left - col_start
        src_r = right - col_start
        M[:, left:right] = basis[:, src_l:src_r]

    return M



def get_interval(grid_size: int, x: float, degree_k: int, domain: tuple[float, float] = (-1.0, 1.0)) -> tuple[float, float]:
    a, b = domain
    h = (b - a) / grid_size

    # extended domain: allow k extra uniform intervals on both sides
    ext_left  = a - degree_k * h
    ext_right = b + degree_k * h

    # clamp to extended domain, half-open on the right to avoid falling off the edge
    # (behaves like MatrixKAN's interval picking without the wraparound quirk)
    if x < ext_left:
        x_clamped = ext_left
    elif x >= ext_right:
        # bring back just inside the half-open interval
        x_clamped = np.nextafter(ext_right, ext_left)
    else:
        x_clamped = x

    # interval index on the extended grid, clipped to [-k, (G-1)+k]
    s_raw = int(np.floor((x_clamped - a) / h))
    s = max(-degree_k, min((grid_size - 1) + degree_k, s_raw))

    left = a + s * h
    right = left + h
    return float(left), float(right)

# NEW: offline helper to precompute quantized interval boundaries for a layer
def build_interval_lut_q16(
    grid_size: int,
    degree_k: int,
    F_INPUT: int,
    domain: tuple[float, float] = (-1.0, 1.0),
):
    """
    Build LUTs of interval [a_s, b_s) boundaries for s in [-k .. (G-1)+k],
    both in FP32 and in Q(F_INPUT) int16.

    Returns:
      interval_starts_q  : np.int16, shape (num_intervals,)
      interval_ends_q    : np.int16, shape (num_intervals,)
      interval_starts_fp : np.float32, shape (num_intervals,)
      interval_ends_fp   : np.float32, shape (num_intervals,)
    """
    a, b = domain
    G = grid_size
    k = degree_k
    h = (b - a) / G

    s_min = -k
    s_max = (G - 1) + k
    num_intervals = s_max - s_min + 1

    starts_fp = np.zeros(num_intervals, dtype=np.float32)
    ends_fp   = np.zeros(num_intervals, dtype=np.float32)
    starts_q  = np.zeros(num_intervals, dtype=np.int16)
    ends_q    = np.zeros(num_intervals, dtype=np.int16)

    for idx, s in enumerate(range(s_min, s_max + 1)):
        left  = a + s * h
        right = left + h
        starts_fp[idx] = left
        ends_fp[idx]   = right
        starts_q[idx]  = q_from_float(left,  F_INPUT)  # Q(F_INPUT)
        ends_q[idx]    = q_from_float(right, F_INPUT)  # Q(F_INPUT)

    return starts_q, ends_q, starts_fp, ends_fp


# NEW: pure-quantized interval selection using the LUT
def get_interval_index_q(
    x_q: int,
    interval_starts_q: np.ndarray,
    interval_ends_q: np.ndarray,
) -> int:
    """
    Pure integer interval selection:
      find idx such that interval_starts_q[idx] <= x_q < interval_ends_q[idx],
    with clamping to first/last interval if x_q is outside.
    """
    num = interval_starts_q.shape[0]

    # If outside extended domain [starts_q[0], ends_q[-1]), signal "out of range"
    if x_q < int(interval_starts_q[0]) or x_q >= int(interval_ends_q[num - 1]):
        return -1


    # Linear scan is fine for tiny G,k; in HW this would be comparators / small tree
    for idx in range(num):
        left_q  = int(interval_starts_q[idx])
        right_q = int(interval_ends_q[idx])
        if left_q <= x_q < right_q:
            return idx

    # Fallback (shouldn't happen if intervals cover domain)
    return num - 1

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def sat16(v):
    return np.int16(np.clip(v, -32768, 32767))

def q_from_float(val, F):
    """Quantize float -> int16 in QF with clipping (no wrap)."""
    q = np.round(val * (1 << F))
    return sat16(q)

def rshift_round(v, s):
    """
    Arithmetic right shift with rounding that works for Python ints and NumPy arrays.
    Returns an int64/array; callers should cast/saturate as needed.
    """
    v64 = np.asarray(v, dtype=np.int64)
    if s <= 0:
        return (v64 << (-s)) if s < 0 else v64
    return (v64 + (1 << (s - 1))) >> s

def fracbits_from_absmax(absmax: float, nbits: int = 16) -> int:
    I = int(np.ceil(np.log2(absmax + 1e-12))) if absmax > 0 else 0
    return max(0, nbits - 1 - I)



def compute_basis_matrix_output_q16(input_val, basis_matrix, a, b, degree,
                                input_absmax, basis_matrix_absmax, interval_width_absmax, interval_endpoint_absmax,
                                input_is_q: bool = False, F_INPUT_override: int | None = None, interval_start_q_override: int | None = None):

    N_BITS = 16 # int16 domain
    
    # ------------------------------------------------------------
    # Choose formats (I - integer bits, F = fractional bits)
    # - Each of {input, basis_matrix, interval_width_inverse, interval_endpoint} has a unique scale
    # ------------------------------------------------------------
    # Input format (based on absmax_x),
    if input_is_q:
        assert F_INPUT_override is not None, "Provide F_INPUT_override when input_is_q=True"
        F_INPUT = int(F_INPUT_override)
        input_q = np.int16(input_val)  # value already in QF_INPUT
    else:
        I_INPUT  = int(np.ceil(np.log2(input_absmax + 1e-12))) if input_absmax > 0 else 0
        F_INPUT  = max(0, N_BITS - 1 - I_INPUT)
        input_q  = q_from_float(input_val, F_INPUT)  # quantize float -> QF_INPUT
    
        
    # Basis matrix format
    I_BASIS_MATRIX = int(np.ceil(np.log2(basis_matrix_absmax)))
    F_BASIS_MATRIX = max(0, N_BITS - 1 - I_BASIS_MATRIX)

    # Interval with inverse format
    interval_width_inverse = 1 / (b - a)
    interval_width_inverse_absmax = 1 / interval_width_absmax
    I_INTERVAL_INV = int(np.ceil(np.log2(interval_width_inverse_absmax)))
    F_INTERVAL_INV = max(0, N_BITS - 1 - I_INTERVAL_INV)

    # Pre-shift gets its own format (not necessarily same as input)
    I_INTERVAL_ENDPOINT = int(np.ceil(np.log2(interval_endpoint_absmax)))
    F_INTERVAL_ENDPOINT = max(0, N_BITS - 1 - I_INTERVAL_ENDPOINT)


    # ------------------------------------------------------------
    # Prepare offline shifts for the dot-product
    # ------------------------------------------------------------
    XMAX = (input_absmax + interval_width_absmax) * interval_width_inverse_absmax

    # Max of monomials in real numbers: [1, |x'|, |x'|^2, ..., |x'|^degree]
    U_MAX = np.array([XMAX**i for i in range(degree + 1)], dtype=np.float64)

    # Offline U_SHIFT: ensure every u_i fits in int16 when represented in QF_INPUT (before U_SHIFT)
    U_PEAK_UNSCALED = (1 << F_INPUT) * float(np.max(U_MAX))
    U_SHIFT_OFFLINE = max(0, int(np.ceil(np.log2(max(1.0, U_PEAK_UNSCALED))) - 15))

    # Offline SUM_SHIFT: ensure sum of terms fits in int16 *after* >> SUM_SHIFT
    TERM_PEAK_OFFLINE = (1 << (F_INPUT - U_SHIFT_OFFLINE)) * float(np.sum(U_MAX * basis_matrix_absmax))
    SUM_SHIFT_OFFLINE = max(0, int(np.ceil(np.log2(max(1.0, TERM_PEAK_OFFLINE))) - 15))



    # ------------------------------------------------------------
    # Quantize input and parameters
    # ------------------------------------------------------------
    interval_width_inverse_q = q_from_float(interval_width_inverse, F_INTERVAL_INV)  # QF_PREW

    if interval_start_q_override is not None:
        # Already provided in Q(F_INPUT); keep as int16 here
        interval_start_q = np.int16(interval_start_q_override)
    else:
        interval_start_q = q_from_float(a, F_INTERVAL_ENDPOINT)  # QF_PRESH

    basis_matrix_q = q_from_float(basis_matrix, F_BASIS_MATRIX)

    # Align pre_shift into x's Q for subtraction
    if interval_start_q_override is not None:
        # Already provided in the same Q domain as input_q (F_INPUT)
        interval_start_in_input_format = np.int16(interval_start_q)
    else:
        if F_INTERVAL_ENDPOINT >= F_INPUT:
            shift = F_INTERVAL_ENDPOINT - F_INPUT
            interval_start_in_input_format = sat16(rshift_round(interval_start_q, shift))
        else:
            interval_start_in_input_format = sat16(int(interval_start_q) << (F_INPUT - F_INTERVAL_ENDPOINT))

    # ------------------------------------------------------------
    # ONLINE: Apply affine in fixed-point: x_q = (x_q - pre_shift_in_xq) * pre_weight_q >> F_PREW
    # ------------------------------------------------------------
    tmp32 = int(input_q) - int(interval_start_in_input_format)            # still QF_INPUT
    input_q   = sat16(tmp32)

    prod32 = int(input_q) * int(interval_width_inverse_q)              # QF_INPUT * QF_PREW
    input_q    = sat16(rshift_round(prod32, F_INTERVAL_INV))       # -> QF_INPUT


    # ------------------------------------------------------------
    # ONLINE: Build u = [1, x, x^2, ..., x^degree] in Q(F_INPUT - U_SHIFT)
    # ------------------------------------------------------------
    u_q = np.zeros(degree + 1, dtype=np.int16)

    # u0 = 1.0 -> shift down by U_SHIFT (rounded)
    u0 = 1 << F_INPUT
    u_q[0] = sat16(rshift_round(u0, U_SHIFT_OFFLINE))

    # u1 = x_q >> U_SHIFT
    u_q[1] = sat16(rshift_round(input_q, U_SHIFT_OFFLINE))

    for d in range(2, degree + 1):
        acc = int(u_q[d - 1]) * int(input_q)           # (QF_INPUT - U_SHIFT) * QF_INPUT
        acc = rshift_round(acc, F_INPUT)               # -> QF_INPUT - U_SHIFT
        u_q[d] = sat16(acc)


    # ------------------------------------------------------------
    # ONLINE: Basis matrix and per-element products: y_i = (u_i * w_i) >> F_WEIGHT
    #  - Output shares u's Q: Q(F_INPUT - U_SHIFT)
    # ------------------------------------------------------------
    power_basis_dim, basis_matrix_dim = basis_matrix.shape
    output_q = np.zeros((1, basis_matrix_dim), dtype=np.int16)
    for i in range(basis_matrix_dim):
        # MUST be int64 accumulator, not 32-bit
        acc64 = np.int64(0)

        for j in range(power_basis_dim):
            prod64 = np.int64(u_q[j]) * np.int64(basis_matrix_q[j, i])
            acc64 += prod64

        # Normalize once at the end
        acc64 = rshift_round(acc64, F_BASIS_MATRIX + SUM_SHIFT_OFFLINE)
        output_q[0, i] = sat16(acc64)
    # ------------------------------------------------------------



    # Dequant for debugging: one scale for all outputs, matching u's effective Q
    DEQ_EXP = F_INPUT - U_SHIFT_OFFLINE - SUM_SHIFT_OFFLINE

    output_fp32 = (output_q.astype(np.float32) / (1 << DEQ_EXP)
          if DEQ_EXP >= 0
          else output_q.astype(np.float32) * (1 << (-DEQ_EXP)))


    # print("output_q:", output_q)
    # print("output_fp32:", output_fp32)

    return output_q, output_fp32, DEQ_EXP

def layer_forward_fp32(
    coef: np.ndarray,           # shape (I, O, K)
    scale_sp: np.ndarray,       # shape (I, O)
    input_x: np.ndarray,      # shape (1, I)
    grid_size: int = 4,
    degree: int = 2
) -> np.ndarray:
    """
    Forward pass for a single layer in full precision (FP32).
    Reference implementation against quantized version.
    
    There are 2 stages in the forward pass,
    assuming the input is quantized, we do:
    
    input --> basis matrix output --> matmul with coef --> output
          MAC                     MAC
    
    quantized domain:

    input * basis_matrix --> accumulate --> requant back down to int16 (intermediate scale) --> matmul with int16 coef --> accumulate --> requant back down to int16 (next activation scale)

    """
    input_dim, output_dim, K = coef.shape
    
    assert K == degree + grid_size, "degree + grid_size must equal K, got degree={} + grid_size={} = K={}".format(degree, grid_size, K)

    # 1. Build power bases and basis function outputs per input dim
    #    For each input j, compute u in its local interval and evaluate basis outputs (length K)

    basis_function_outputs = np.zeros((input_dim, K), dtype=np.float32)
    

    for i in range(input_dim):

        a, b = get_interval(grid_size, input_x[0,i], degree)
        
        # print(f"Input x[{i}] = {input_x[0,i]:.2f} falls into interval {s[i]}")

        u = (input_x[0][i].item() - a) / (b - a)
        # power basis for this row is 1, u, u**2, ... u**degree
        power_basis = np.array([u**p for p in range(degree + 1)], dtype=np.float32)  # shape (degree+1,)
        
        basis_matrix = basis_matrix_for_x(degree, grid_size, input_x[0][i])
        # Evaluate basis functions: (degree+1,) @ (degree+1, K) -> (K,)
        basis_function_outputs[i, :] = power_basis @ basis_matrix  # shape (K,)
        
        # if input_dim == 16:
        #     print(f"input {i} basis_function_output stats\n\tinput: {input_x[0,i]:.3f}\n\tinterval: [{a:.3f}, {b:.3f})\n\tu: {u:.3f}\n\tbasis matrix: {basis_matrix}\n\tpower basis: {power_basis}\n\tbasis outputs: {basis_function_outputs[i,:]}")
    
    basis_function_outputs_absmax = np.max(np.abs(basis_function_outputs))
    
   
    print("[layer_forward_fp32] first basis_function_outputs:", basis_function_outputs[0, :])
    
    # 2. MatMult with coef and scale_sp, accumulate contributions across input dims
           
    # we can fold scale_sp into coeff, so we only need to do the single dot product per output dim
    coef_folded = coef * scale_sp[:, :, np.newaxis]
    output_y = np.zeros((output_dim, 1), dtype=np.float32)
    for i in range(output_dim):
        for j in range(input_dim):
            output_y[i, 0] += (basis_function_outputs[j, :] @ coef_folded[j, i, :])
    return output_y, basis_function_outputs_absmax


def layer_forward_int16(
    coef: np.ndarray,           # shape (I, O, K)
    scale_sp: np.ndarray,       # shape (I, O)
    input_vector: np.ndarray,      # shape (1, I)
    next_activation_min: float,
    next_activation_max: float,
    input_absmax: float,
    intermediate_absmax: float,
    first_layer: bool = False,
    dequantize_output: bool = False,
    grid_size: int = 4,
    degree: int = 2,
):
    """
    Forward pass for a single layer in INT16 weights & activations quantized precision.
    """
    
    
    # Decide Q format for non-first layers and build interval LUT in that domain
    if first_layer:
        F_INPUT_CURR = None
        interval_starts_q = interval_ends_q = None
        interval_starts_fp = interval_ends_fp = None
    else:
        F_INPUT_CURR = fracbits_from_absmax(input_absmax)
        (interval_starts_q,
         interval_ends_q,
         interval_starts_fp,
         interval_ends_fp) = build_interval_lut_q16(
            grid_size=grid_size,
            degree_k=degree,
            F_INPUT=F_INPUT_CURR,
            domain=(-1.0, 1.0),
        )

    
    # print(f"\n\n[layer_forward_int16] CALLED\n\tcoef.shape={coef.shape}, first_layer={first_layer}, dequantize_output={dequantize_output}")
    # print(f"\tnext layer activation range: [{next_activation_min}, {next_activation_max}]")
    next_absmax = max(abs(next_activation_min), abs(next_activation_max))
    F_NEXT = fracbits_from_absmax(next_absmax)  # int16: 1 sign bit, so up to 15 frac bits
    # 1. Quantize inputs (optionally), weights to INT16 using symmetric quantization
    
    assert input_absmax is not None, "input_absmax must always be provided to compute adjusted interval scale"
    # if first_layer:
    #     scale_x = input_absmax / INT_MAX if input_absmax != 0 else 1.0
    #     input_x_int16 = np.round(input_x / scale_x).astype(np.int16)
    # else:
    #     assert input_x.dtype == np.int16, "input_x must be int16 if not first layer"
    #     input_x_int16 = input_x

    basis_matrix_absmax = 1.0 # we can precompute this for given degree and grid_size, here we just hardcode for degree=2, grid_size=4
    
    
    interval_width = (grid_size / (1.0 - (-1.0)))  # assuming domain is [-1, 1]
    interval_endpoint_absmax = 1 + degree * interval_width # ex: for degree=2, grid_size=4, intervals go from -2,2
    interval_width_absmax = 0.5 # hardcoded for grid_size=4, domain [-1,1]
        

    # 2. Build power bases and basis function outputs per input dim
    
    input_dim, output_dim, K = coef.shape
    basis_function_outputs = np.zeros((input_dim, K), dtype=np.int16)
    basis_function_outputs_fp32 = np.zeros((input_dim, K), dtype=np.float32)
    
    for i in range(input_dim):
        input_val = input_vector[0, i]

        if first_layer:
            # First layer: input is real FP32 in [-1,1]
            x_for_interval = float(input_vector[0, i])
            a, b = get_interval(grid_size, x_for_interval, degree)
            interval_start_q_override = None
        else:
            # Non-first layers: input is int16 in Q(F_INPUT_CURR)
            x_q = int(input_vector[0, i])

            # Pure-quantized interval selection:
            idx = get_interval_index_q(x_q, interval_starts_q, interval_ends_q)

            if idx == -1:
                # Outside extended domain: mimic MatrixKAN behavior (zero basis)
                a_q = int(interval_starts_q[0])
                a   = float(interval_starts_fp[0])
                b   = float(interval_ends_fp[0])
                outside = True
            else:
                # Inside extended domain: normal interval selection
                a_q = int(interval_starts_q[idx])
                a   = float(interval_starts_fp[idx])
                b   = float(interval_ends_fp[idx])
                outside = False

            # Use the same dequantized x as the FP32-style path for debugging,
            # but in the outside case the basis will be forced to zero anyway.
            x_for_interval = x_q / (1 << F_INPUT_CURR)

            interval_start_q_override = a_q


        if first_layer:
            # Original FP32-style interval selection
            basis_matrix = basis_matrix_for_x(degree, grid_size, x_for_interval)
            
            # if input_dim == 16: # debug print for small test case
            #     print(f"  [layer_forward_int16] input[{i}] = {input_vector[0, i]} deq to {x_for_interval:.6f} basis_matrix:\n{basis_matrix}")
        else:
            # Quantized path: use LUT interval index directly, or zero if outside.
            if outside:
                basis_matrix = np.zeros((degree + 1, grid_size + degree), dtype=np.float64)
                # if input_dim == 16:  # debug print for small test case
                #     print(f"  [layer_forward_int16] input[{i}] = {input_vector[0, i]} (Q{F_INPUT_CURR}) outside extended domain; zero basis_matrix")
            else:
                basis_matrix = basis_matrix_for_index(degree, grid_size, idx)
                # if input_dim == 16:  # debug print for small test case
                #     print(f"  [layer_forward_int16] input[{i}] = {input_vector[0, i]} (Q{F_INPUT_CURR}) basis_matrix (from idx={idx}):\n{basis_matrix}")


        basis_function_outputs[i, :], basis_function_outputs_fp32[i, :], deq_scale_basis_matrix_out = compute_basis_matrix_output_q16(
            input_val=input_val,
            basis_matrix=basis_matrix,
            a=a,
            b=b,
            degree=degree,
            input_absmax=input_absmax,
            basis_matrix_absmax=basis_matrix_absmax,
            interval_width_absmax=interval_width_absmax,
            interval_endpoint_absmax=interval_endpoint_absmax,
            input_is_q=(not first_layer),
            F_INPUT_override=F_INPUT_CURR,
            interval_start_q_override=interval_start_q_override,
        )

    # print("[layer_forward_int16] first basis_function_outputs (int16):", basis_function_outputs[0, :])
    # print("[layer_forward_int16] first basis_function_outputs_fp32:", basis_function_outputs_fp32[0, :])
    
    # # OFFLINE quantization preparation
    # coef_folded = coef * scale_sp[:, :, np.newaxis]                       # (I, O, K)
    # absmax_coef_folded = float(np.max(np.abs(coef_folded))) if np.size(coef_folded) else 0.0
    # I_COEF  = int(np.ceil(np.log2(absmax_coef_folded))) if absmax_coef_folded > 0 else 0
    # F_COEF  = max(0, 16 - 1 - I_COEF)                                     # int16: 1 sign bit
    # coef_folded_q = q_from_float(coef_folded, F_COEF).astype(np.int16)
    
    # # ONLINE quantized matmul
    # output_acc64 = np.zeros((coef.shape[1], 1), dtype=np.int64)         # (O,1)
    # input_dim, output_dim, K = coef.shape

    # for o in range(output_dim):
    #     acc64 = 0
    #     for i in range(input_dim):
    #         # dot over K in widened precision
    #         prod32 = (basis_function_outputs[i, :].astype(np.int32) *
    #                 coef_folded_q[i, o, :].astype(np.int32))
    #         acc_k  = int(np.sum(prod32))                                  # int32 sum -> Python int
    #         acc64 += rshift_round(acc_k, F_COEF)                          # align back to basis Q
    #     output_acc64[o, 0] = acc64

    # # Final dequant using the basis exponent
    # DEQ_EXP = int(deq_scale_basis_matrix_out)
    # output_fp32 = (output_acc64.astype(np.float32) / (1 << DEQ_EXP)
    #                 if DEQ_EXP >= 0 else
    #                 output_acc64.astype(np.float32) * (1 << (-DEQ_EXP)))

    # # [OPTIONAL print]
    # # print("output_fp32 (quantized path):", output_fp32)
    # return None, output_fp32


    # OFFLINE quantization preparation
    coef_folded = coef * scale_sp[:, :, np.newaxis]                       # (I, O, K)
    absmax_coef_folded = float(np.max(np.abs(coef_folded))) if np.size(coef_folded) else 0.0
    I_COEF  = int(np.ceil(np.log2(absmax_coef_folded))) if absmax_coef_folded > 0 else 0
    F_COEF  = max(0, 16 - 1 - I_COEF)                                     # int16: 1 sign bit
    coef_folded_q = q_from_float(coef_folded, F_COEF).astype(np.int16)
    
    # ONLINE quantized matmul
    output_acc64 = np.zeros((coef.shape[1], 1), dtype=np.int64)         # (O,1)
    input_dim, output_dim, K = coef.shape

    for o in range(output_dim):
        acc64 = 0
        for i in range(input_dim):
            # dot over K in widened precision
            prod32 = (basis_function_outputs[i, :].astype(np.int32) *
                    coef_folded_q[i, o, :].astype(np.int32))
            acc_k  = int(np.sum(prod32))                                  # int32 sum -> Python int
            acc64 += acc_k                                                # keep in Q^(basis + F_COEF)
        output_acc64[o, 0] = acc64

    # # Final dequant using the basis exponent
    # DEQ_EXP_TOTAL = int(deq_scale_basis_matrix_out) + F_COEF
    # output_fp32 = (output_acc64.astype(np.float32) / (1 << DEQ_EXP_TOTAL)
    #             if DEQ_EXP_TOTAL >= 0 else
    #             output_acc64.astype(np.float32) * (1 << (-DEQ_EXP_TOTAL)))
    # # [OPTIONAL print]
    # # print("output_fp32 (quantized path):", output_fp32)
    # return None, output_fp32

    DEQ_EXP_TOTAL = int(deq_scale_basis_matrix_out) + F_COEF

    output_fp32 = (output_acc64.astype(np.float32) / (1 << DEQ_EXP_TOTAL)
                if DEQ_EXP_TOTAL >= 0
                else output_acc64.astype(np.float32) * (1 << (-DEQ_EXP_TOTAL)))
    
    
    if dequantize_output == False:
        # Pure-integer requant to the NEXT layer's QF_NEXT (power-of-two scale)
        # We have: real ≈ output_acc64 * 2^{-DEQ_EXP_TOTAL}
        # Want: out_int16 ≈ real * 2^{F_NEXT}  => shift by (F_NEXT - DEQ_EXP_TOTAL)
        SHIFT = DEQ_EXP_TOTAL - F_NEXT
        if SHIFT >= 0:
            out_int16 = sat16(rshift_round(output_acc64, SHIFT))
        else:
            out_int16 = sat16((output_acc64.astype(np.int64)) << (-SHIFT))
        out_q = out_int16.astype(np.int16)
    else:
        out_q = output_acc64

    return out_q, output_fp32


def model_forward_fp32(
    model_weights: list[tuple[np.ndarray, np.ndarray]],  # list of (coef, scale_sp) per layer
    input_x: np.ndarray,      # shape (1, I)
    grid_size: int = 4,
    degree: int = 2
) -> np.ndarray:
    """
    Forward pass for the entire model in full precision (FP32).
    Reference implementation against quantized version.
    """
    x = input_x
    for layer_idx, (coef, scale_sp) in enumerate(model_weights):
        # print(f"\n=== Forwarding layer {layer_idx} ===")
        x, intermediate_absmax = layer_forward_fp32(
            coef=coef,
            scale_sp=scale_sp,
            input_x=x.T if layer_idx > 0 else x,
            grid_size=grid_size,
            degree=degree
        )
    # print(f"Final output:", x.T)
    return x



def model_forward_int16(
    model_weights: list[tuple[np.ndarray, np.ndarray]],  # list of (coef, scale_sp) per layer
    input_x: np.ndarray,      # shape (1, I)
    input_absmax: float,
    next_activation_min_maxes: list[tuple[float, float]],
    grid_size: int = 4,
    degree: int = 2
) -> np.ndarray:
    """
    Forward pass for the entire model in INT16 quantized precision.
    """
    first_layer = True
    for layer_idx, (coef, scale_sp) in enumerate(model_weights):
        # print(f"\n=== Forwarding layer {layer_idx} ===")

        coef, scale_sp = model_weights[layer_idx]
        next_activation_min, next_activation_max = next_activation_min_maxes[layer_idx]
        
        output_q, output_dq = layer_forward_int16(
            coef=coef,
            scale_sp=scale_sp,
            input_vector=input_x.T if layer_idx > 0 else input_x,
            next_activation_min=next_activation_min,
            next_activation_max=next_activation_max,
            input_absmax=input_absmax,
            intermediate_absmax=None,
            first_layer=first_layer,
            dequantize_output=(layer_idx == len(model_weights) - 1),
            grid_size=grid_size,
            degree=degree
        )
        
        # Update input_absmax for next layer
        input_absmax = max(abs(next_activation_min), abs(next_activation_max))
        first_layer = False
        input_x = output_q

    # print(f"Final output (FP32):", output_dq.T)
    return output_dq