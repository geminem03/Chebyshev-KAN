# Chebyshev KAN Layer Hardware Implementation

This repository implements a hardware-accelerated Kolmogorov–Arnold Network (KAN) layer using Chebyshev polynomial basis functions, optimized for Intel Arria 10 FPGAs.

The design focuses on:

- Minimal DSP usage  
- High clock frequency  
- Fully parallel polynomial evaluation  
- Bit-accurate hardware/software verification  

---

# Mathematical Foundation

A Chebyshev polynomial expansion is defined as:

$$
y(x) = \sum_{k=0}^{n} c_k T_k(x)
$$

Direct evaluation requires many multipliers.  
Instead, we use **Clenshaw’s Recurrence**, which computes the polynomial using a recursive formulation:

$$
b_k(x) = c_k + 2x\,b_{k+1}(x) - b_{k+2}(x)
$$

The final output is:

$$
y(x) = b_0 - b_2
$$

### Why This Matters

Using Clenshaw’s method:

- Only **one multiplier per Processing Engine (PE)** is required  
- Polynomial degree becomes a memory scaling problem  
- DSP utilization remains constant as degree increases  

This dramatically improves FPGA efficiency.

---

# Hardware Architecture

## Processing Engine (`pe_quad.sv`)

The PE is the computational core.

To achieve ~234.85 MHz on Arria 10 silicon, the engine uses:

- A **5-stage DSP pipeline**
- A **time-division multiplexed (TDM)** architecture
- A 5-cycle recurrence loop with a deliberate pipeline "bubble"

Four inputs (A, B, C, D) share one multiplier.  
The 5th cycle ensures safe register updates and prevents recurrence hazards.

**Result:**

- 1 DSP per PE  
- Fully pipelined execution  
- Deterministic timing  

---

## Layer Controller (`layer.sv`)

The top-level `layer` module manages:

- Input streaming  
- Weight memory access  
- Global accumulation  

### Parallel Memory Architecture

- 64 independent M20K RAM blocks  
- One dedicated memory per PE  
- No memory contention  

All 64 PEs evaluate their Chebyshev polynomials simultaneously, enabling true spatial parallelism.

---

# Testcase Generation

The Python script generates Hamiltonian phase-space test data.

Each PE models:

$$
E(q) = \frac{1}{2} k q^2
$$

Inputs:

$$
q_i = 0.8 \cos\left(\frac{2\pi i}{64}\right)
$$

Each PE uses:

$$
k_{pe} = 0.5 + \frac{pe}{128}
$$

The script generates:

- 64 `.mif` weight files  
- `inputs_s16.npy`  
- `expected_s64.npy` (bit-accurate golden reference)  

---

# Build & Simulation

### Generate Test Data

```bash
python generate_test.py
```

### Run ModelSim

```tcl
do run_layer.do
```

If verification passes:

```
SUCCESS! All 64 PE outputs match the Hamiltonian Reference.
```

---

# Performance Metrics

| Metric | Result |
|--------|--------|
| Logic Utilization | 12,815 / 427,200 (3%) |
| DSP Usage | 64 / 1,518 (4%) |
| RAM Blocks | 64 M20K |
| Weight Memory | 262,144 bits = 32KB |
| F_max | 252.4 MHz |
| Cycles (64 inputs) | 390 |
| Latency | 1545 ns |

---
