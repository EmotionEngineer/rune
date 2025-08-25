# RUNE: Rule-Embedded Neural Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**RUNE (Rule-Embedded Neural Engine)** is a PyTorch-based library for constructing deep neural networks with a high degree of intrinsic interpretability. The architecture is founded on the principle of explicitly modeling relationships between features through pairwise comparisons. By leveraging concepts from tropical geometry (specifically, max-plus algebra), RUNE models learn sparse, rule-like decision logic, making them "white-box" by design.

This approach shifts the paradigm from opaque transformations, typical of standard MLPs, to a sequence of understandable, comparative operations. This makes RUNE particularly well-suited for high-stakes domains where the "why" behind a prediction is as critical as the "what".

## Core Architectural Principles

RUNE's interpretability stems from its unique building blocks, which can be stacked to create deep yet transparent models.

-   **`TropicalDifferenceAggregator`**: The core computational unit. It aggregates weighted pairwise interactions (`feature_i - feature_j`) using a differentiable analogue of the `max` operation. This allows the model to learn which feature comparisons are most salient for a decision.
-   **`GatedTropicalDifferenceAggregator`**: Enhances the core unit by dynamically combining **tropical (max-like)** and **standard (mean-like)** aggregation using a learnable gate. This allows the model to choose between sparse, rule-based logic (`max`) and holistic, averaging logic (`mean`) for each feature.
-   **`RUNEBlock`**: The primary component for building deep models. It is a residual block that preserves dimensionality, allowing for sequential stacking to create deep, fully-analyzable networks.

### New in RUNE: Interaction Types (Differences vs. Ratios)

To better model relationships in domains like finance or economics, RUNE now supports different types of pairwise feature interactions. This is controlled by the `interaction_type` parameter in `InterpretableRuneNet` and its components.

-   **`'difference'` (Default)**: The standard operation `(x_i - x_j)`. Useful for absolute comparisons.
-   **`'log_ratio'`**: Computes interactions as `log(x_i / x_j)`, which is equivalent to `log(x_i) - log(x_j)`. This is highly effective for scale-invariant comparisons and is often more numerically stable than direct ratios. For example, comparing `log(income / debt_payment)` is a standard practice in credit scoring.
-   **`'ratio'`**: Computes interactions as `x_i / x_j`.

**Note**: When using `'log_ratio'` or `'ratio'`, ensure your input features are strictly positive to avoid mathematical errors like `log(0)` or division by zero.

```python
# Example: Building a model that learns rules based on log-ratios of features
model = InterpretableRuneNet(
    input_dim=10, 
    output_dim=1, 
    interaction_type='log_ratio' # Key change here!
)
```

## Model Variants

The library provides several high-level model architectures:

### `InterpretableRuneNet`
A complete, end-to-end deep neural network constructed from a sequence of `RUNEBlock`s. It intentionally avoids "black-box" MLP layers, ensuring full model transparency. This model serves as the base for specialized training techniques.

-   **Regularized RUNE (Training Strategy)**: By adding an L1 penalty to the interaction weights during training, you can encourage sparsity, forcing the model to discover the simplest, most important rules. This is achieved by using the `model.get_regularization_loss()` method in your training loop.
-   **Annealed RUNE (Training Strategy)**: The "softness" of the `max` operation is controlled by a temperature parameter, `tau`. By gradually decreasing `tau` during training (annealing), the model's logic becomes "harder" and more discrete, making it easier to extract definitive rules. This is supported by the `model.set_tau()` method.

### `PrototypeRuneNet`
A powerful architecture for **case-based reasoning**. It combines a `PrototypeLayer` with an `InterpretableRuneNet`.
1.  The model learns a set of **prototypes**, which are "ideal" or representative examples for the task.
2.  For any new input, it first calculates the distance to each of these prototypes.
3.  This vector of distances is then fed into a RUNE network, which learns rules based on prototype similarities (e.g., "If the input is very similar to Prototype 3 but dissimilar to Prototype 5, predict Class A"). Since distances are always positive, this is a great use case for `interaction_type='log_ratio'`.

This provides a highly intuitive, human-friendly explanation for predictions.

## Interpretation and Analysis Tools

The `rune.interpret` module provides a suite of functions to deconstruct and visualize the model's logic.

-   **`plot_tropical_aggregator_params`**: Visualize the raw interaction weights (`W_ij`) in a RUNE block to see which feature comparisons are emphasized. Especially useful for `RegularizedRuneNet`.
-   **`plot_gated_aggregator_gates`**: See whether the model prefers `max`-like or `mean`-like logic.
-   **`plot_final_layer_contributions`**: Identify which high-level features learned by the RUNE stack were most influential for a specific prediction.
-   **`trace_decision_path`**: Generate a step-by-step report for a single prediction, detailing the dominant comparison terms (`x_i - x_j`, `log(x_i/x_j)`, etc.) inside each `RUNEBlock`.
-   **`plot_prototypes_with_tsne` (for `PrototypeRuneNet`)**: Visualize the learned prototypes in the same 2D space as your data to see if the model has identified meaningful clusters.
-   **`analyze_prototype_prediction` (for `PrototypeRuneNet`)**: For a single sample, identify the closest prototypes and get a feature-by-feature comparison, explaining the decision in terms of similarity to known cases.

## Installation

The library is currently available for installation directly from GitHub.

First, install the required dependencies:
```bash
pip install torch matplotlib networkx scikit-learn pandas seaborn
```

Then, install the `rune` package from this repository:
```bash
pip install git+https://github.com/EmotionEngineer/rune.git
```

## Usage Example: Prototype-Based Reasoning

```python
import torch
import pprint
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from rune.models import PrototypeRuneNet
from rune.interpret import analyze_prototype_prediction, plot_prototypes_with_tsne

# 1. Prepare data
X_raw, y = load_wine(return_X_y=True)
feature_names = load_wine().feature_names
scaler = StandardScaler().fit(X_raw)
X = scaler.transform(X_raw)
# For ratio-based models, ensure data is positive if necessary
# For this example, we use the default 'difference' interaction

# 2. Define and train a PrototypeRuneNet model (training loop omitted for brevity)
model = PrototypeRuneNet(
    input_dim=X.shape[1], 
    output_dim=len(set(y)), 
    num_prototypes=10,
    num_blocks=2, 
    block_dim=16,
    interaction_type='log_ratio' # Distances are positive, so log_ratio is a great choice
)
# model.load_state_dict(...) # Assume model is trained
model.eval()

# 3. Pick a sample to analyze
x_sample = torch.tensor(X[5], dtype=torch.float32)

# 4. Get a comprehensive, case-based explanation
analysis = analyze_prototype_prediction(model, x_sample, feature_names=feature_names, top_k=1)

# Print the analysis
print(f"Model Prediction: Class {analysis['prediction']}")
closest_proto_info = analysis['closest_prototypes'][0]
print(f"Sample is most similar to Prototype {closest_proto_info['index']} (Distance: {closest_proto_info['distance']:.2f})")
print("\nFeature Comparison:")
# To display the styled DataFrame in environments like Jupyter:
# from IPython.display import display
# display(closest_proto_info['feature_comparison'])
# For plain text output:
print(closest_proto_info['feature_comparison'])


# 5. Visualize all prototypes relative to the data
plt.figure(figsize=(10, 8))
plot_prototypes_with_tsne(model, X, y)
plt.show()
```

## Advanced Interpretation Concepts / Future Work

RUNE's architecture enables novel, powerful interpretation methods beyond standard visualizations. The explicit modeling of pairwise relationships allows for deeper insights into model logic. Future research will focus on implementing:

1.  **Dominant Logic Graph (DLG)**: Tracing the most influential `(x_i - x_j)` comparisons through the network to build a directed graph of the model's core logic.
2.  **Multi-Level Rule Extraction (MLRE)**: Automatically translating the learned weights and low-temperature `max` operations into human-readable "IF-THEN" rules.
3.  **Continuous Logical Attribution (CLA)**: A rigorous, gradient-based attribution method that assigns importance scores to *pairs* of features, offering more insight than single-feature methods like SHAP.
4.  **Differentiable Logic-Tree Decomposition (DLTD)**: Transforming a trained RUNE model into a hierarchical, tree-like structure of interpretable decision splits.
5.  **Causal-Logic Identification (CLI)**: Using interventional and counterfactual analysis on the pairwise comparisons to distinguish correlational patterns from more causally-plausible ones.
6.  **Probabilistic Logic Interpretation (PLI)**: A Bayesian approach to place confidence intervals around the extracted rules, quantifying the model's certainty in its own logic.
7.  **Interactive Logic Learning & Explainability Dashboard (ILLED)**: A user-facing dashboard for real-time "what-if" analysis, allowing users to tweak feature values and instantly see how it affects the chain of pairwise comparisons.

These directions aim to move beyond "interpretable" and toward truly **understandable AI**.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.