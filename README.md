# Chebyshev Kolmogorov-Arnold Networks (KANs): Hardware & Software Co-Design

Welcome to the root repository for our exploration into Chebyshev Kolmogorov-Arnold Networks (KANs). This project investigates the extreme parameter efficiency of KANs in software and implements a highly optimized, hardware-accelerated Chebyshev KAN layer for FPGAs.

## Project Overview

### 1. Iso-Accuracy Benchmark (airfoil-study/)
We conducted an extensive hyperparameter search (via Optuna) to compare MLPs against B-Spline, Fourier, and Chebyshev KANs on the NASA Airfoil Self-Noise dataset. 
* **Result:** To reach a strict ~93.5% accuracy target, the standard MLP required over 21,000 parameters. The Chebyshev KAN hit the same target using only **384 parameters**—making it roughly **55x more parameter-efficient**.

### 2. FPGA Hardware Acceleration (chebyshev/)
Because Chebyshev polynomials proved to be the most efficient and stable architecture, we developed a custom SystemVerilog hardware implementation for it.
* **Result:** By leveraging **Clenshaw’s Recurrence**, our Processing Engines (PEs) evaluate complex polynomials using only **1 DSP per PE**. The design runs at ~234.85 MHz on Arria 10 silicon, featuring a 64-way parallel memory architecture with zero contention.

