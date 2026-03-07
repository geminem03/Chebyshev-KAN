# Chebyshev Polynomials for Kolmogorov-Arnold Networks

## The Chebyshev Approach

Traditional Kolmogorov-Arnold Networks (KANs) utilize B-splines on their edges to learn non-linear relationships. While effective in software, B-splines are computationally expensive for FPGAs because they require complex piecewise logic and frequent memory lookups.

This project replaces B-splines with **Chebyshev Polynomials**. By using the Chebyshev recurrence relation:

Tₙ(x) = 2xTₙ₋₁(x) − Tₙ₋₂(x)

we can evaluate non-linearities using standard Multiply-Accumulate (MAC) operations. This allows for a hardware architecture that is significantly faster and more area-efficient than traditional spline-based implementations.

## Hardware Architecture

The system is designed to accelerate a **64×64 fully connected KAN layer** with **Degree 3 polynomials**.

- `cheby_quad.sv`  
  The core compute engine. Implements the 4-thread staggered pipeline for Degree 3 Chebyshev polynomials.

- `kan_64x64.sv`  
  The top-level orchestrator. Manages the 64-PE array, the state machine, and final output accumulation/clamping.

- `ram_bank_64.sv`  
  A parameterized RAM wrapper that loads coefficients from `.mem` files into M20K memory blocks.

## Performance Metrics (64×64 Layer)

| Metric | Result |
|--------|--------|
| Logic Utilization | 10,534 ALMs |
| Total DSP Blocks | 64 |
| Total Memory Bits | 266,114 (M20K) |
| Max Frequency ($F_{max}$) | 266.81 MHz |
| Number of Cycles | 70 |
| Latency | 268.87 ns |


## Machine Learning

- `airfoil-study/`  
  Contains the Python/PyTorch scripts used to validate the model's accuracy and parameter efficiency against standard MLPs.
