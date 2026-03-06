# KAN vs. MLP: Iso-Accuracy Benchmark Study

Welcome to this benchmark study evaluating the parameter efficiency and performance of Kolmogorov-Arnold Networks (KANs) against traditional Multi-Layer Perceptrons (MLPs). The goal of this repository is to demonstrate how different neural network architectures perform when tasked with reaching a strict **95% accuracy threshold** on complex scientific data.

## The NASA Dataset and KAN Suitability

This study utilizes a NASA dataset (e.g., C-MAPSS, Airfoil Self-Noise, etc.) which consists of complex, continuous sensor and telemetry readings. 

**Why Kolmogorov-Arnold Networks (KANs)?**
Standard MLPs use fixed activation functions (like ReLU or Sigmoid) at each node and learn a matrix of linear weights. KANs fundamentally flip this architecture: they eliminate fixed node activations and instead place learnable univariate functions directly on the edges of the network. 

NASA datasets are typically governed by underlying low-dimensional physical or thermodynamic laws. KANs are theoretically uniquely suited for this type of data because:
1. **Mathematical Representation:** Their edge-based learnable functions excel at mapping the continuous, non-linear physical relationships inherent in scientific data. 
2. **Parameter Efficiency:** By adapting the actual shape of the activation functions, KANs can solve complex regression and classification tasks using a fraction of the parameters that a standard MLP would require. 

## The Three KAN Architectures

To ensure this benchmark is an objective, apples-to-apples comparison, it is critical to understand how standard Multi-Layer Perceptrons (MLPs) and Kolmogorov-Arnold Networks (KANs) distribute their trainable parameters. 

While their internal math differs significantly, every model in this study is evaluated on the exact same task: mapping a 5-dimensional input vector to a 1-dimensional output. The baseline for fairness across all architectures is the **total number of trainable parameters** optimized during backpropagation.

### 1. The Baseline: Multi-Layer Perceptron (MLP)
In a standard MLP, learnable parameters are strictly defined as the weights on the connecting edges and the biases at each node. 
* **Mechanism:** An MLP applies a linear transformation (multiplying inputs by weights and adding a bias) followed by a fixed, non-linear activation function (like ReLU or Tanh) applied at the node itself.
* **Parameter Scaling:** To increase the capacity of an MLP to model complex data, you must either add more neurons to a hidden layer (making it wider) or add more hidden layers (making it deeper). Both actions drastically increase the size of the weight matrices, leading to the parameter explosion seen in our high-accuracy targets.

### 2. B-Spline KAN
Kolmogorov-Arnold Networks remove fixed activation functions from the nodes and instead place learnable, univariate functions on the edges. In a B-spline KAN, these edge functions are defined by piecewise polynomial curves.
* **Mechanism:** Every single edge connecting one layer to the next contains a unique B-spline curve. The shape of this curve is determined by a set of control points over a specified grid.
* **Parameter Scaling:** Instead of learning a single weight scalar per edge like an MLP, a B-spline KAN learns the control point coefficients for the spline on that edge. The number of parameters per edge is dictated by the grid size and the degree of the polynomial ($k$). While each edge holds more parameters than an MLP edge, the network requires drastically fewer nodes and layers to map complex functions.

### 3. Chebyshev KAN
Instead of localized splines, the Chebyshev KAN parameterizes its edges using orthogonal Chebyshev polynomials. 
* **Mechanism:** Chebyshev polynomials are mathematically renowned for uniformly approximating complex, continuous functions while avoiding wild oscillations at the boundaries (Runge's phenomenon). Every edge evaluates the input using a series of these polynomials up to a defined degree.
* **Parameter Scaling:** The learnable parameters are the coefficients of the polynomial terms on each edge. If an edge uses a degree-3 Chebyshev expansion, it learns the coefficients for those specific polynomial degrees. This provides global smoothness and high stability, often allowing this network to converge with incredibly shallow and narrow layer dimensions.

### 4. Fourier KAN
The Fourier KAN replaces polynomial edge functions with a Fourier series, utilizing sine and cosine harmonics.
* **Mechanism:** This architecture assumes the underlying data can be effectively decomposed into frequencies. Every edge applies a combination of sine and cosine waves to the input signal.
* **Parameter Scaling:** The trainable parameters are the amplitude and phase coefficients for each harmonic frequency defined on the edge. By learning these coefficients, the network can perfectly mold itself to cyclical, periodic, or highly oscillatory physical telemetry without needing thousands of dense matrix multiplications.

### Auditor's Note on Fairness
Because KANs hold multiple parameters per edge (e.g., polynomial coefficients) compared to the MLP's single scalar weight per edge, comparing the models by *node count* or *hidden dimensions* would severely bias the results in favor of the KAN. 

Therefore, our Iso-Accuracy benchmark strictly compares the **total mathematical footprint** (total trainable parameters). Whether a parameter is adjusting a node's bias in an MLP or tweaking a spline's curvature in a KAN, it represents one unit of optimization space. Comparing minimum total parameters to achieve the same accuracy target ensures a purely objective evaluation of architectural efficiency.

## Methodology: An Apples-to-Apples Comparison

To ensure a fair comparison between the KANs and the MLP, we conducted an **Iso-Accuracy Study**. 

Instead of arbitrarily matching parameter counts (which disadvantages KANs) or matching depths (which ignores the complexity of KAN edges), we set a strict target: **95% testing accuracy**. We then ran extensive hyperparameter searches and generated Pareto fronts to identify the absolute minimum number of parameters required for each architecture to cross that 95% threshold. 

[cite_start]Because standard MLPs struggled to reach 95% with shallow networks [cite: 7][cite_start], an extended hyperparameter sweep was explicitly required for the MLP to find a configuration that met the target[cite: 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]. 

## Iso-Accuracy Results

To thoroughly benchmark parameter efficiency, we established two strict testing accuracy targets: **93.5%** and **95.0%**. We then analyzed the Pareto fronts for each architecture to identify the absolute minimum number of parameters required to cross these thresholds.

Because standard MLPs struggled to reach these high-accuracy marks with shallow or small networks, an extended hyperparameter sweep was explicitly required to find an MLP configuration that could successfully hit the targets.

### Target 1: $\ge$ 93.5% Accuracy

For a 93.5% target, the KAN architectures already demonstrate significant efficiency over the standard MLP. 

| Model | Architecture (FC Dims) | Minimum Params | Actual Accuracy |
| :--- | :--- | :--- | :--- |
| **Chebyshev KAN** | `[5, 12, 1]` | **288** | 93.72% |
| **Fourier KAN** | `[5, 20, 1]` | **741** | 94.43% |
| **B-Spline KAN** | `[[5, 0], [9, 0], [1, 0]]` | **922** | 94.33% |
| **Standard MLP** | `[5, 217, 208, 1]` | **46,855** | 93.59% |

*At this threshold, the Chebyshev KAN is over 160x more parameter-efficient than the standard MLP.*

### Target 2: $\ge$ 95.0% Accuracy

Pushing the models to a more stringent 95% threshold drastically widens the gap. While the KANs required only modest increases in parameter count to reach 95%, the standard MLP experienced a massive parameter explosion.

| Model | Architecture (FC Dims) | Minimum Params | Actual Accuracy |
| :--- | :--- | :--- | :--- |
| **Chebyshev KAN** | `[5, 17, 1]` | **408** | 95.13% |
| **Fourier KAN** | `[5, 28, 1]` | **1,037** | 95.46% |
| **B-Spline KAN** | `[[5, 0], [12, 0], [1, 0]]` | **1,213** | 95.26% |
| **Standard MLP** | `[5, 961, 206, 304, 1]` | **267,171** | 95.31% |

### Conclusion
The results highlight the massive parameter efficiency of Kolmogorov-Arnold Networks for modeling complex physical datasets. As the required accuracy increases from 93.5% to 95.0%, the standard MLP scales extremely poorly, requiring hundreds of thousands of additional parameters. Conversely, the Chebyshev KAN crossed the final 95% target using just 408 parameters—making it roughly **650x more parameter-efficient** than the best-performing Multi-Layer Perceptron.