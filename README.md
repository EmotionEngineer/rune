# RUNE: Rule-Embedded Neural Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**RUNE (Rule-Embedded Neural Engine)** is a PyTorch-based library for constructing deep neural networks with a high degree of intrinsic interpretability. The architecture is founded on the principle of explicitly modeling relationships between features through pairwise comparisons. By leveraging concepts from tropical geometry (specifically, max-plus algebra), RUNE models learn sparse, rule-like decision logic, making them "white-box" by design.

This approach shifts the paradigm from opaque transformations, typical of standard MLPs, to a sequence of understandable, comparative operations. This makes RUNE particularly well-suited for high-stakes domains where the "why" behind a prediction is as critical as the "what".

## Core Architectural Principles

RUNE's interpretability stems from its unique building blocks, which can be stacked to create deep yet fully transparent models.

### `TropicalDifferenceAggregator`
The core computational unit of RUNE. This layer aggregates weighted pairwise differences between features using a differentiable analogue of the `max` operation, known as log-sum-exp. This allows the model to learn which feature comparisons are most salient.

**Mathematical Formulation:**
For an input vector $\mathbf{x} \in \mathbb{R}^D$, each component $y_i$ of the output vector $\mathbf{y} \in \mathbb{R}^D$ is computed as:

$$
y_i = \tau \cdot \log \left( \sum_{j \neq i} \exp \left( \frac{(x_i - x_j)W_{ij} + B_{ij}}{\tau} \right) \right)
$$

where $\mathbf{W}, \mathbf{B} \in \mathbb{R}^{D \times D}$ are learnable weight and bias matrices, and $\tau$ is a temperature parameter that controls the smoothness of the `max` approximation. As $\tau \to 0$, this operation converges to $\max_{j \neq i}((x_i - x_j)W_{ij} + B_{ij})$.

**Interpretability:** By analyzing the terms inside the exponent, one can identify the dominant difference $(x_i - x_j)$ that most significantly contributes to the output feature $y_i$. This forms the basis of a learned rule.

### `GatedTropicalDifferenceAggregator`
This layer enhances the `TropicalDifferenceAggregator` by dynamically combining two types of aggregation—**tropical (max-like)** and **standard (mean-like)**—using a learnable gate. This allows the model to choose between a sparse, rule-based logic (`max`) and a holistic, averaging logic (`mean`) for each feature.

**Mathematical Formulation:**
The output $\mathbf{h} \in \mathbb{R}^D$ is a convex combination of two branches:

$$
\mathbf{y}^{\text{tropical}} = \text{TropicalDifferenceAggregator}(\mathbf{x})
$$

$$
y_i = \frac{1}{D-1} \sum_{j \neq i} (x_i - x_j)W_{ij}
$$

The combination is mediated by a sigmoid gate $\mathbf{g} = \sigma(\mathbf{\alpha})$, where $\mathbf{\alpha}$ is a learnable parameter vector:

$$
\mathbf{h} = \mathbf{g} \odot \mathbf{y}^{\text{tropical}} + (1 - \mathbf{g}) \odot \mathbf{y}^{\text{mean}}
$$

The final layer output is the concatenation $[\mathbf{x}, \mathbf{h}] \in \mathbb{R}^{2D}$.

**Interpretability:** The gate values $\mathbf{g} \in [0, 1]^D$ explicitly quantify the model's preference for sparse, rule-based logic versus dense, averaging logic for each internal feature.

### `RUNEBlock`
The primary component for building deep models. It is a residual block that preserves dimensionality, allowing for sequential stacking.

**Architecture:**
1.  **Aggregation:** $h_{\text{agg}} = \text{GatedTropicalDifferenceAggregator}(x) \in \mathbb{R}^{2D}$.
2.  **Projection:** A linear layer projects the aggregated features back to the original dimension: $h_{\text{proj}} = \text{Linear}(h_{\text{agg}}) \in \mathbb{R}^D$.
3.  **Residual Connection:** The output is computed with a residual connection and layer normalization for stable training: $y = \text{LayerNorm}(x + \text{Dropout}(h_{\text{proj}}))$.

**Interpretability:** Because the block is fully analyzable and preserves dimensions, stacking `RUNEBlock`s creates a deep network where the decision logic can be traced from input to output.

### `InterpretableRuneNet`
A complete, end-to-end deep neural network constructed from a sequence of `RUNEBlock`s. It intentionally avoids "black-box" MLP layers with non-decomposable activations (like ReLU), ensuring full model transparency.

**Architecture:**
1.  **Input Projection:** A linear layer maps the input features to an internal embedding space: $h_0 = \text{Linear}_{\text{in}}(x)$.
2.  **RUNE Block Stack:** A series of $L$ blocks refine the embeddings: $h_{l+1} = \text{RUNEBlock}_l(h_l)$.
3.  **Output Head:** A final linear layer maps the refined embeddings to the target output: $y_{\text{out}} = \text{Linear}_{\text{out}}(h_L)$.

## Interpretation and Analysis Tools

The `rune.interpret` module provides a suite of functions to deconstruct and visualize the model's logic.

-   **`plot_linear_weights`**: Visualizes the weights of any `nn.Linear` layer (e.g., the input projection) as a heatmap, revealing how input features are composed into internal concepts.
-   **`plot_gated_aggregator_gates`**: Displays the learned gate values from a `GatedTropicalDifferenceAggregator`, showing the model's preference for tropical (max) vs. mean logic.
-   **`plot_final_layer_contributions`**: For a given sample, it calculates and plots the contribution of each feature entering the final layer. Contribution is defined as `activation × weight`, showing which learned concepts were most influential.
-   **`trace_decision_path`**: The most powerful tool. It generates a step-by-step report for a single prediction, detailing:
    -   The dominant comparison terms (`(x_i - x_j)`) inside each `RUNEBlock`.
    -   The final layer features that had the largest impact on the score.

## Installation

The library requires `torch`, `matplotlib`, and `networkx`

```bash
# Install dependencies
pip install torch matplotlib networkx

# Install RUNE directly from GitHub
pip install git+https://github.com/EmotionEngineer/rune.git
```

## Usage Example

```python
import torch
import pprint
import matplotlib.pyplot as plt
from rune.models import InterpretableRuneNet
from rune.interpret import trace_decision_path, plot_final_layer_contributions

# 1. Define and load a pre-trained model
model = InterpretableRuneNet(input_dim=8, output_dim=1, num_blocks=3, block_dim=32)
# model.load_state_dict(torch.load("california_housing_rune.pt")) # Example
model.eval()

# 2. Prepare an input sample for analysis
x_sample = torch.randn(8) # Example sample
feature_names = [f"Feature_{i}" for i in range(8)]

# 3. Get the prediction
prediction = model(x_sample.unsqueeze(0))
print(f"Model prediction: {prediction.item():.4f}\n")

# 4. Perform comprehensive interpretation

# 4.1. Trace the full decision path from input to output
print("--- Tracing the Full Decision Path ---")
path_analysis = trace_decision_path(model, x_sample, top_k=2, feature_names=feature_names)
pprint.pprint(path_analysis)

# 4.2. Visualize feature contributions at the final layer
print("\n--- Plotting Final Layer Feature Contributions ---")
plt.figure(figsize=(12, 6))
plot_final_layer_contributions(
    model,
    x_sample,
    title="Contributions to Final Prediction"
)
plt.show()

```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
