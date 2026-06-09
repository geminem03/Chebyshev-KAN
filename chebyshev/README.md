# Chebyshev KAN Layer Hardware Implementation

This repository implements a **hardware-accelerated Kolmogorov–Arnold Network (KAN) layer** using **Chebyshev polynomial basis functions**.

The design focuses on:

- Minimal DSP usage
- High clock frequency
- Fully parallel polynomial evaluation
- Bit-accurate hardware/software verification using a custom Python emulator

---

# Mathematical Foundation

A Chebyshev polynomial expansion is defined as:

$$
y(x) = \sum_{k=0}^{n} c_k T_k(x)
$$

Direct evaluation requires many multipliers. Instead, we use **Clenshaw’s Recurrence**, which computes the polynomial using a recursive formulation:

$$
b_k(x) = c_k + 2x\,b_{k+1}(x) - b_{k+2}(x)
$$

The final output is:

$$
y(x) = b_0 - b_2
$$

Using **Clenshaw’s method**:

- Polynomial degree becomes a **memory scaling problem**, not a logic scaling problem.
- **DSP utilization remains highly efficient** as the polynomial degree increases.

---

# Hardware Architecture

## Processing Engine (`pe_quad.sv`)

The PE is the **computational core**. It processes inputs in groups of four simultaneously:

- **Inputs:** A, B, C, D
- **4-Stage Pipeline:** Calculates the Chebyshev polynomial via Clenshaw recurrence
- **Native Fixed-Point Math:** Uses **Q5.10 radix point** fixed-point arithmetic

---

## Layer Controller (`layer.sv`)

The **top-level layer module** orchestrates the PEs and manages memory and accumulation.

### Parallel Memory Architecture

- Each output PE has a **dedicated M20K RAM block**
- Completely eliminates **memory contention**

### 22-Bit Accumulation

Hardware chunks inputs into groups of 4, streaming them to the PEs and safely summing **up to 64 Clenshaw results** into a **25-bit accumulator** to prevent integer overflow.

### Serialized Requantization

Once accumulation finishes, the layer executes a **serialized requantization phase**.

It sequentially walks through the **25-bit accumulators** using a **single shared multiplier**, applying:

- a **scale factor**
- a **shift**

to clamp results back into **16-bit saturated outputs**.

The **requantization scale factor** is efficiently stored as the **final row in each PE's memory block**.

---

# Synthetic Stress Testing

For **maximum-capacity synthesis** or **Quartus stress testing**, you can generate an arbitrary **N × M dense layer testcase**.

Example:

```bash
# Generate a fully-connected 64x64 testcase
python generate_testcase.py --num_inputs 64 --num_outputs 64
```

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
