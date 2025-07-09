# rune/interpret.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .layers import (
    PairwiseDifferenceLayer,
    TropicalDifferenceAggregator,
    GatedTropicalDifferenceAggregator,
    CyclicTropicalDifferenceLayer,
    PrototypeLayer
)
from .models import PrototypeRuneNet
from .utils import get_feature_names

# Optional imports for advanced visualizations
try:
    import networkx as nx
except ImportError:
    nx = None

try:
    from sklearn.manifold import TSNE
    import seaborn as sns
except ImportError:
    TSNE = None
    sns = None

def plot_pairwise_difference_weights(
    layer: PairwiseDifferenceLayer,
    feature_names: list[str] = None,
    ax=None,
    title: str = "Pairwise Difference Weights"
):
    """
    Visualizes the learned weights of a PairwiseDifferenceLayer.

    Args:
        layer: The PairwiseDifferenceLayer instance.
        feature_names: Optional list of names for the original input features.
        ax: Optional matplotlib Axes object to plot on.
        title: Plot title.
    """
    if layer.weights is None or layer.num_diff_features == 0:
        print("Layer has no difference features or weights to plot.")
        return

    weights = layer.weights.detach().cpu().numpy()
    indices = layer.feature_indices.cpu().numpy()

    if feature_names is None:
        feature_names = get_feature_names(layer.input_dim, "InFeat_")
    
    pair_labels = [f"{feature_names[i]} - {feature_names[j]}" for i, j in indices]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, len(weights) * 0.5), 6))
    
    ax.bar(range(len(weights)), weights, color='skyblue')
    ax.set_xticks(range(len(weights)))
    ax.set_xticklabels(pair_labels, rotation=45, ha="right")
    ax.set_ylabel("Weight Value")
    ax.set_title(title)
    ax.grid(True, axis='y', linestyle='--')
    plt.tight_layout()
    return ax

def plot_tropical_aggregator_params(
    layer: TropicalDifferenceAggregator,
    param_type: str = "weights", # "weights" or "bias"
    feature_names: list[str] = None,
    ax=None,
    title: str = None
):
    """
    Visualizes the weights or biases of a TropicalDifferenceAggregator as a heatmap.

    Args:
        layer: The TropicalDifferenceAggregator instance.
        param_type: Which parameter to plot ("weights" or "bias").
        feature_names: Optional list of names for the features (dim of the aggregator).
        ax: Optional matplotlib Axes object to plot on.
        title: Plot title.
    """
    if param_type == "weights":
        data = layer.weights.detach().cpu().numpy()
        default_title = "Tropical Aggregator Weights (W_ij for output y_i from x_i - x_j)"
    elif param_type == "bias":
        data = layer.bias.detach().cpu().numpy()
        default_title = "Tropical Aggregator Biases (B_ij for output y_i from x_i - x_j)"
    else:
        raise ValueError("param_type must be 'weights' or 'bias'")

    if title is None:
        title = default_title

    if feature_names is None:
        feature_names = get_feature_names(layer.dim, "Dim_")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(layer.dim * 0.8, layer.dim * 0.7))

    im = ax.imshow(data, cmap="viridis")
    ax.set_xticks(np.arange(layer.dim))
    ax.set_yticks(np.arange(layer.dim))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_yticklabels(feature_names)
    ax.set_xlabel("Feature j (in x_i - x_j)")
    ax.set_ylabel("Output Feature i (y_i)")
    ax.set_title(title)
    
    # Add text annotations for values
    for i in range(layer.dim):
        for j in range(layer.dim):
            if i == j: # Skip diagonal for clarity as it's masked
                continue
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="w" if data[i,j] < (data.max()+data.min())/2 else "black")

    plt.colorbar(im, ax=ax, label="Parameter Value")
    plt.tight_layout()
    return ax

def plot_gated_aggregator_gates(
    layer: GatedTropicalDifferenceAggregator,
    feature_names: list[str] = None,
    ax=None,
    title: str = "Gated Aggregator Gate Values (Sigmoid)"
):
    """
    Visualizes the learned gate values of a GatedTropicalDifferenceAggregator.

    Args:
        layer: The GatedTropicalDifferenceAggregator instance.
        feature_names: Optional list of names for the features (dim of the aggregator).
        ax: Optional matplotlib Axes object to plot on.
        title: Plot title.
    """
    gate_values = layer.gate_values.cpu().numpy() # Uses the @property

    if feature_names is None:
        feature_names = get_feature_names(layer.dim, "Feat_")

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, len(gate_values) * 0.6), 5))

    ax.bar(range(len(gate_values)), gate_values, color='lightcoral', label='Tropical Weight (g)')
    ax.bar(range(len(gate_values)), 1 - gate_values, bottom=gate_values, color='lightblue', label='Mean Weight (1-g)')
    
    ax.set_xticks(range(len(gate_values)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_ylabel("Gate Value (Tropical vs. Mean Contribution)")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--')
    plt.tight_layout()
    return ax

def plot_linear_weights(
    layer: nn.Linear,
    feature_names_in: list[str] = None,
    feature_names_out: list[str] = None,
    ax=None,
    title: str = "Linear Layer Weights",
    show_values: bool = True,
    fmt: str = ".2f"
):
    """
    Visualizes the weights of any nn.Linear layer as a heatmap.

    Args:
        layer: The nn.Linear layer instance.
        feature_names_in: Optional list of names for the input features (columns).
        feature_names_out: Optional list of names for the output features (rows).
        ax: Optional matplotlib Axes object to plot on.
        title: Plot title.
        show_values: If True, displays the weight values in each cell.
        fmt: The format string for the weight values.
    """
    weights = layer.weight.detach().cpu().numpy() # (out_features, in_features)

    if feature_names_in is None:
        feature_names_in = get_feature_names(layer.in_features, "In_")
    if feature_names_out is None:
        feature_names_out = get_feature_names(layer.out_features, "Out_")

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, weights.shape[1] * 0.8), max(4, weights.shape[0] * 0.5)))

    im = ax.imshow(weights, cmap="coolwarm", aspect='auto')
    ax.set_xticks(np.arange(weights.shape[1]))
    ax.set_yticks(np.arange(weights.shape[0]))
    ax.set_xticklabels(feature_names_in, rotation=45, ha="right")
    ax.set_yticklabels(feature_names_out)
    ax.set_xlabel("Input Features")
    ax.set_ylabel("Output Features")
    ax.set_title(title)
    
    plt.colorbar(im, ax=ax, label="Weight Value")

    if show_values:
        threshold = im.norm(weights.max()) / 2.
        
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                color = "w" if im.norm(weights[i, j]) < threshold else "k"
                ax.text(j, i, f"{weights[i, j]:{fmt}}",
                        ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    return ax

def plot_cyclic_layer_projection_weights(
    layer: CyclicTropicalDifferenceLayer,
    feature_names_in: list[str] = None,
    feature_names_proj: list[str] = None,
    ax=None,
    title: str = "Cyclic Layer Projection Weights"
):
    """
    Visualizes the weights of the initial projection layer in a CyclicTropicalDifferenceLayer.
    This is now a convenience wrapper around plot_linear_weights.
    """
    return plot_linear_weights(
        layer=layer.projection,
        feature_names_in=feature_names_in,
        feature_names_out=feature_names_proj,
        ax=ax,
        title=title
    )

def plot_feature_interaction_graph(
    layer: PairwiseDifferenceLayer,
    feature_names: list[str] = None,
    ax=None,
    title: str = "Learned Feature Interaction Graph",
    threshold: float = 0.1
):
    """
    Visualizes the learned weights of a PairwiseDifferenceLayer as a graph.
    Nodes are features, and edges represent the importance of their difference.

    Args:
        layer: The PairwiseDifferenceLayer instance.
        feature_names: Optional list of names for the input features.
        ax: Optional matplotlib Axes object to plot on.
        title: Plot title.
        threshold: Minimum absolute weight value to draw an edge.
    """
    if nx is None:
        raise ImportError("Please install networkx: `pip install networkx`")
        
    if layer.weights is None or layer.num_diff_features == 0:
        print("Layer has no difference features or weights to plot.")
        return

    weights = layer.weights.detach().cpu().numpy()
    indices = layer.feature_indices.cpu().numpy()

    if feature_names is None:
        feature_names = get_feature_names(layer.input_dim, "Feat_")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    G = nx.Graph()
    for i in range(layer.input_dim):
        G.add_node(feature_names[i])

    for (i, j), weight in zip(indices, weights):
        if abs(weight) > threshold:
            G.add_edge(feature_names[i], feature_names[j], weight=weight, label=f"{weight:.2f}")

    pos = nx.circular_layout(G)
    edge_weights = [d['weight'] for u, v, d in G.edges(data=True)]
    edge_colors = ['red' if w < 0 else 'blue' for w in edge_weights]
    
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500,
            edge_color=edge_colors, width=[abs(w) * 2 for w in edge_weights], ax=ax)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

    ax.set_title(title)
    plt.tight_layout()
    return ax

def plot_final_layer_contributions(
    model: torch.nn.Module,
    x_sample: torch.Tensor,
    feature_names: list[str] = None,
    ax=None,
    title: str = "Final Layer Feature Contributions"
):
    """
    Analyzes and plots the contribution of each feature entering the final linear
    layer to the output prediction for a single sample.
    Assumes the model has a final linear layer (e.g., `output_head` or `output_layer`).

    Args:
        model: The trained model (ideally with an interpretable head).
        x_sample: A single input sample tensor (shape: [input_dim] or [1, input_dim]).
        feature_names: Custom names for the features entering the final layer.
        ax: Optional matplotlib Axes object.
        title: Plot title.
    """
    model.eval()
    if x_sample.dim() == 1:
        x_sample = x_sample.unsqueeze(0)

    # Find the final linear layer and the module that feeds into it
    final_layer = None
    if hasattr(model, 'output_head'):
        final_layer = model.output_head
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    elif hasattr(model, 'output_layer'):
        final_layer = model.output_layer
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    else:
        raise ValueError("Could not find a final 'output_head' or 'output_layer'.")

    # Get the activations feeding into the final layer
    with torch.no_grad():
        activations = feature_extractor(x_sample).squeeze(0) # (num_features_in,)
        weights = final_layer.weight.squeeze(0) # Assuming single output
    
    contributions = activations * weights
    
    if feature_names is None:
        feature_names = get_feature_names(len(contributions), "InterpretableFeat_")

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(8, len(contributions) * 0.4), 6))

    contributions_np = contributions.detach().cpu().numpy()
    colors = ['g' if c > 0 else 'r' for c in contributions_np]
    ax.bar(range(len(contributions)), contributions_np, color=colors)

    ax.set_xticks(range(len(contributions)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_ylabel("Contribution (Activation * Weight)")
    ax.set_title(f"{title} for sample")
    ax.grid(True, axis='y', linestyle='--')
    plt.tight_layout()
    return ax

def analyze_tropical_dominance(
    layer: TropicalDifferenceAggregator,
    x_input: torch.Tensor,
    top_k: int = 3,
    feature_names: list[str] = None
):
    """
    Analyzes which (x_i - x_j) terms dominate the log-sum-exp for a given input.
    This is for a single batch item or an average over a batch.

    Args:
        layer: The TropicalDifferenceAggregator instance.
        x_input: A single input sample (dim,) or a batch (batch_size, dim).
                 If batch, results are averaged.
        top_k: How many top contributing terms to show for each output dimension.
        feature_names: Optional list of feature names.

    Returns:
        A dictionary where keys are output feature indices/names and values are
        lists of (term_str, contribution_score) for the top_k terms.
    """
    if x_input.ndim == 1:
        x_input = x_input.unsqueeze(0)
    
    B, D = x_input.shape
    assert D == layer.dim, "Input dimension mismatch"

    x_diff = x_input.unsqueeze(2) - x_input.unsqueeze(1)
    weighted_diff = x_diff * layer.weights.unsqueeze(0).detach() + layer.bias.unsqueeze(0).detach()
    
    # Mask out diagonal
    weighted_diff = weighted_diff.masked_fill(~layer.mask.unsqueeze(0), float('-inf'))
    
    # Contributions are terms inside exp before normalization by sum.
    # Softmax essentially.
    contributions = torch.softmax(weighted_diff / layer.tau.detach(), dim=2) # (B, D, D)
    
    if B > 1:
        contributions = contributions.mean(dim=0) # Average over batch (D, D)
    else:
        contributions = contributions.squeeze(0) # (D, D)

    if feature_names is None:
        feature_names = get_feature_names(layer.dim, "x")

    analysis = {}
    for i in range(layer.dim): # For each output dimension y_i
        # Get contributions to y_i from all (x_i - x_j)
        scores, indices_j = torch.topk(contributions[i, :], k=top_k, dim=0)
        
        output_feature_name = feature_names[i]
        analysis[output_feature_name] = []
        for k_idx in range(scores.shape[0]):
            j = indices_j[k_idx].item()
            score = scores[k_idx].item()
            if score < 1e-6 or i == j: # Negligible or masked
                continue
            term_str = f"({feature_names[i]} - {feature_names[j]})"
            analysis[output_feature_name].append((term_str, score))
            
    return analysis

def trace_decision_path(
    model: torch.nn.Module,
    x_sample: torch.Tensor,
    top_k: int = 3,
    feature_names: list[str] = None
) -> dict:
    """
    Traces the most influential terms through a model for a single prediction,
    providing a step-by-step explanation. Acts as a dispatcher for different
    RUNE model types.

    Args:
        model: An instance of InterpretableRuneNet or PrototypeRuneNet.
        x_sample: A single input sample (shape: [input_dim]).
        top_k: Number of top terms to report at each stage.
        feature_names: Names for the original input features.

    Returns:
        A dictionary containing the step-by-step analysis.
    """
    if isinstance(model, PrototypeRuneNet):
        print("Tracing decision path for PrototypeRuneNet. Analysis will be based on prototype distances.")
        return analyze_prototype_prediction(model, x_sample.cpu(), top_k=top_k, feature_names=feature_names)

    if not hasattr(model, 'rune_blocks'):
        print("Warning: This function is designed for models with 'rune_blocks' like InterpretableRuneNet.")
        # Fallback for simpler models
        if hasattr(model, 'gated_tropical_agg'):
             agg_layer = model.gated_tropical_agg.tropical_agg
             return {"Layer 0 (Tropical Aggregator)": analyze_tropical_dominance(agg_layer, x_sample, top_k, feature_names)}
        return {}
    
    model.eval()
    if x_sample.dim() == 1:
        x_sample = x_sample.unsqueeze(0)
    if feature_names is None:
        feature_names = get_feature_names(x_sample.shape[1], "Input_")

    path = {}
    
    # --- Step 1: Input Projection ---
    with torch.no_grad():
        h = model.input_projection(x_sample)
    
    proj_feature_names = get_feature_names(h.shape[1], "Proj_")
    path['InputProjection'] = "Input projected to internal dimension."

    # --- Step 2: Through each RUNEBlock ---
    for i, block in enumerate(model.rune_blocks):
        block_analysis = {}
        # Analyze the tropical dominance within this block
        tropical_agg = block.gated_agg.tropical_agg
        dominance = analyze_tropical_dominance(tropical_agg, h, top_k, proj_feature_names)
        block_analysis['DominantTropicalTerms'] = dominance
        
        # Get gate values for context
        gate_values = block.gated_agg.gate_values.cpu().numpy()
        block_analysis['GateValues'] = {f: g for f, g in zip(proj_feature_names, gate_values)}
        
        # Propagate h to the next block
        with torch.no_grad():
            h = block(h)

        path[f'RUNEBlock_{i}'] = block_analysis

    # --- Step 3: Final Layer Contribution ---
    final_weights = model.output_head.weight.squeeze(0).detach() # (block_dim,)
    final_activations = h.squeeze(0).detach() # (block_dim,)
    contributions = final_weights * final_activations

    top_contribs = torch.topk(contributions.abs(), k=top_k)
    
    final_analysis = []
    for val, idx in zip(top_contribs.values, top_contribs.indices):
        idx = idx.item()
        final_analysis.append({
            'Feature': proj_feature_names[idx],
            'Contribution': contributions[idx].item(),
            'Reason': f"High contribution due to Activation={final_activations[idx]:.2f} * Weight={final_weights[idx]:.2f}"
        })
        
    path['FinalPrediction'] = {
        'TopContributingFeatures': final_analysis,
        'OverallScore': model.output_head(h).item()
    }

    return path

def analyze_prototype_prediction(
    model: PrototypeRuneNet,
    x_sample: torch.Tensor,
    feature_names: list[str] = None,
    top_k: int = 3
) -> dict:
    """
    Analyzes a single sample's prediction from a PrototypeRuneNet for case-based reasoning.

    It identifies the most similar prototypes and compares their feature values
    to the sample's feature values, providing an intuitive explanation.

    Args:
        model: The trained PrototypeRuneNet instance.
        x_sample: A single input sample tensor (shape: [input_dim]).
        feature_names: Optional list of names for the original input features.
        top_k: The number of closest prototypes to analyze.

    Returns:
        A dictionary containing the analysis, including:
        - 'prediction': The model's predicted class for the sample.
        - 'distances': A numpy array of distances to all prototypes.
        - 'closest_prototypes': A list of dictionaries, one for each of the top_k
          closest prototypes, containing their index, distance, and a DataFrame
          comparing their features to the sample.
    """
    model.to('cpu').eval()
    if feature_names is None:
        feature_names = get_feature_names(x_sample.shape[0], "Feature_")

    sample_tensor = x_sample.unsqueeze(0)
    with torch.no_grad():
        distances = model.prototype_layer(sample_tensor).cpu().numpy().flatten()
        output = model(sample_tensor)
        # Handle both regression and classification outputs
        if output.shape[1] > 1:
            prediction = torch.argmax(output, dim=1).item()
        else:
            prediction = output.item()


    closest_indices = np.argsort(distances)[:top_k]
    
    analysis = {
        'prediction': prediction,
        'distances': distances
    }
    
    closest_prototypes_info = []
    for idx in closest_indices:
        prototype_features = model.prototype_layer.prototypes[idx].detach().cpu().numpy()
        feature_df = pd.DataFrame({
            'Feature': feature_names,
            'Prototype_Value': prototype_features,
            'Sample_Value': x_sample.cpu().numpy()
        })
        closest_prototypes_info.append({
            'index': idx,
            'distance': distances[idx],
            'feature_comparison': feature_df
        })
        
    analysis['closest_prototypes'] = closest_prototypes_info
    return analysis

def plot_prototypes_with_tsne(
    model: PrototypeRuneNet,
    X_data: np.ndarray,
    y_data: np.ndarray,
    ax=None,
    title: str = "t-SNE Visualization of Data and Learned Prototypes"
):
    """
    Visualizes prototypes and data points in a 2D space using t-SNE.

    Args:
        model: The trained PrototypeRuneNet instance.
        X_data: The input data samples (e.g., test set) as a NumPy array.
        y_data: The corresponding labels for X_data.
        ax: Optional matplotlib Axes object to plot on.
        title: The plot title.
    """
    if TSNE is None or sns is None:
        raise ImportError("Please install scikit-learn and seaborn: `pip install scikit-learn seaborn`")

    model.to('cpu')
    prototypes = model.prototype_layer.prototypes.detach().cpu().numpy()
    
    # Combine data and prototypes to project them into the same space
    combined_data = np.vstack((X_data, prototypes))
    
    # Use t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_data)-1))
    reduced_data = tsne.fit_transform(combined_data)
    
    # Split them back up
    reduced_X = reduced_data[:len(X_data)]
    reduced_prototypes = reduced_data[len(X_data):]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 9))

    sns.scatterplot(x=reduced_X[:, 0], y=reduced_X[:, 1], hue=y_data, palette='viridis', alpha=0.6, ax=ax, legend='full')
    ax.scatter(reduced_prototypes[:, 0], reduced_prototypes[:, 1], 
               marker='X', s=250, c='red', edgecolors='black', linewidth=1.5, label='Prototypes')
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.legend()
    plt.tight_layout()
    return ax