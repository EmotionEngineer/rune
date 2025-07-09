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
    layer: PairwiseDifferenceLayer, feature_names: list[str] = None, ax=None, title: str = "Pairwise Difference Weights"
):
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
    layer: TropicalDifferenceAggregator, param_type: str = "weights", feature_names: list[str] = None, ax=None, title: str = None
):
    if param_type == "weights":
        data, default_title = layer.weights.detach().cpu().numpy(), "Tropical Aggregator Weights (W_ij for output y_i from x_i - x_j)"
    elif param_type == "bias":
        data, default_title = layer.bias.detach().cpu().numpy(), "Tropical Aggregator Biases (B_ij for output y_i from x_i - x_j)"
    else:
        raise ValueError("param_type must be 'weights' or 'bias'")
    if title is None: title = default_title
    if feature_names is None: feature_names = get_feature_names(layer.dim, "Dim_")
    if ax is None: fig, ax = plt.subplots(figsize=(layer.dim * 0.8, layer.dim * 0.7))
    im = ax.imshow(data, cmap="viridis")
    ax.set_xticks(np.arange(layer.dim)); ax.set_yticks(np.arange(layer.dim))
    ax.set_xticklabels(feature_names, rotation=45, ha="right"); ax.set_yticklabels(feature_names)
    ax.set_xlabel("Feature j (in x_i - x_j)"); ax.set_ylabel("Output Feature i (y_i)")
    ax.set_title(title)
    for i in range(layer.dim):
        for j in range(layer.dim):
            if i == j: continue
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="w" if data[i,j] < (data.max()+data.min())/2 else "black")
    plt.colorbar(im, ax=ax, label="Parameter Value"); plt.tight_layout()
    return ax

def plot_gated_aggregator_gates(
    layer: GatedTropicalDifferenceAggregator, feature_names: list[str] = None, ax=None, title: str = "Gated Aggregator Gate Values (Sigmoid)"
):
    gate_values = layer.gate_values.cpu().numpy()
    if feature_names is None: feature_names = get_feature_names(layer.dim, "Feat_")
    if ax is None: fig, ax = plt.subplots(figsize=(max(6, len(gate_values) * 0.6), 5))
    ax.bar(range(len(gate_values)), gate_values, color='lightcoral', label='Tropical Weight (g)')
    ax.bar(range(len(gate_values)), 1 - gate_values, bottom=gate_values, color='lightblue', label='Mean Weight (1-g)')
    ax.set_xticks(range(len(gate_values))); ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_ylabel("Gate Value (Tropical vs. Mean Contribution)"); ax.set_ylim(0, 1)
    ax.set_title(title); ax.legend(); ax.grid(True, axis='y', linestyle='--'); plt.tight_layout()
    return ax

def plot_linear_weights(
    layer: nn.Linear, feature_names_in: list[str] = None, feature_names_out: list[str] = None, ax=None, title: str = "Linear Layer Weights", show_values: bool = True, fmt: str = ".2f"
):
    weights = layer.weight.detach().cpu().numpy()
    if feature_names_in is None: feature_names_in = get_feature_names(layer.in_features, "In_")
    if feature_names_out is None: feature_names_out = get_feature_names(layer.out_features, "Out_")
    if ax is None: fig, ax = plt.subplots(figsize=(max(6, weights.shape[1] * 0.8), max(4, weights.shape[0] * 0.5)))
    im = ax.imshow(weights, cmap="coolwarm", aspect='auto')
    ax.set_xticks(np.arange(weights.shape[1])); ax.set_yticks(np.arange(weights.shape[0]))
    ax.set_xticklabels(feature_names_in, rotation=45, ha="right"); ax.set_yticklabels(feature_names_out)
    ax.set_xlabel("Input Features"); ax.set_ylabel("Output Features"); ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Weight Value")
    if show_values:
        threshold = im.norm(weights.max()) / 2.
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                color = "w" if im.norm(weights[i, j]) < threshold else "k"
                ax.text(j, i, f"{weights[i, j]:{fmt}}", ha="center", va="center", color=color, fontsize=8)
    plt.tight_layout()
    return ax

def plot_cyclic_layer_projection_weights(
    layer: CyclicTropicalDifferenceLayer, feature_names_in: list[str] = None, feature_names_proj: list[str] = None, ax=None, title: str = "Cyclic Layer Projection Weights"
):
    return plot_linear_weights(
        layer=layer.projection, feature_names_in=feature_names_in, feature_names_out=feature_names_proj, ax=ax, title=title
    )

def plot_feature_interaction_graph(
    layer: PairwiseDifferenceLayer, feature_names: list[str] = None, ax=None, title: str = "Learned Feature Interaction Graph", threshold: float = 0.1
):
    if nx is None: raise ImportError("Please install networkx: `pip install networkx`")
    if layer.weights is None or layer.num_diff_features == 0:
        print("Layer has no difference features or weights to plot."); return
    weights = layer.weights.detach().cpu().numpy(); indices = layer.feature_indices.cpu().numpy()
    if feature_names is None: feature_names = get_feature_names(layer.input_dim, "Feat_")
    if ax is None: fig, ax = plt.subplots(figsize=(10, 8))
    G = nx.Graph()
    for i in range(layer.input_dim): G.add_node(feature_names[i])
    for (i, j), weight in zip(indices, weights):
        if abs(weight) > threshold:
            G.add_edge(feature_names[i], feature_names[j], weight=weight, label=f"{weight:.2f}")
    pos = nx.circular_layout(G)
    edge_weights = [d['weight'] for u, v, d in G.edges(data=True)]
    edge_colors = ['red' if w < 0 else 'blue' for w in edge_weights]
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color=edge_colors, width=[abs(w) * 2 for w in edge_weights], ax=ax)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    ax.set_title(title); plt.tight_layout()
    return ax

def plot_final_layer_contributions(
    model: torch.nn.Module,
    x_sample: torch.Tensor,
    feature_names: list[str] = None,
    ax=None,
    title: str = "Final Layer Feature Contributions"
):
    model.eval()
    if x_sample.dim() == 1: x_sample = x_sample.unsqueeze(0)
    final_layer = None
    if hasattr(model, 'output_head'):
        final_layer = model.output_head
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
    elif hasattr(model, 'output_layer'):
        final_layer = model.output_layer
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
    else:
        raise ValueError("Could not find a final 'output_head' or 'output_layer'.")
    with torch.no_grad():
        activations = feature_extractor(x_sample).squeeze(0)
        final_logits = final_layer(activations.unsqueeze(0))
        if final_layer.out_features > 1:
            predicted_class_idx = torch.argmax(final_logits, dim=1).item()
            weights = final_layer.weight[predicted_class_idx]
            plot_title = f"{title} for Predicted Class {predicted_class_idx}"
        else:
            weights = final_layer.weight.squeeze(0)
            plot_title = f"{title} for sample"
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
    ax.set_title(plot_title)
    ax.grid(True, axis='y', linestyle='--')
    plt.tight_layout()
    return ax

def analyze_tropical_dominance(
    layer: TropicalDifferenceAggregator,
    x_input: torch.Tensor,
    top_k: int = 3,
    feature_names: list[str] = None
):
    if x_input.ndim == 1: x_input = x_input.unsqueeze(0)
    B, D = x_input.shape
    assert D == layer.dim, "Input dimension mismatch"
    x_diff = x_input.unsqueeze(2) - x_input.unsqueeze(1)
    weighted_diff = x_diff * layer.weights.unsqueeze(0).detach() + layer.bias.unsqueeze(0).detach()
    weighted_diff = weighted_diff.masked_fill(~layer.mask.unsqueeze(0), float('-inf'))
    contributions = torch.softmax(weighted_diff / layer.tau.detach(), dim=2)
    contributions = contributions.mean(dim=0) if B > 1 else contributions.squeeze(0)
    if feature_names is None: feature_names = get_feature_names(layer.dim, "x")
    analysis = {}
    for i in range(layer.dim):
        scores, indices_j = torch.topk(contributions[i, :], k=top_k, dim=0)
        output_feature_name = feature_names[i]
        analysis[output_feature_name] = []
        for k_idx in range(scores.shape[0]):
            j = indices_j[k_idx].item()
            score = float(scores[k_idx].item())
            if score < 1e-6 or i == j: continue
            term_str = f"({feature_names[i]} - {feature_names[j]})"
            analysis[output_feature_name].append((term_str, score))
    return analysis

def trace_decision_path(
    model: torch.nn.Module,
    x_sample: torch.Tensor,
    top_k: int = 3,
    feature_names: list[str] = None
) -> dict:
    if isinstance(model, PrototypeRuneNet):
        print("Dispatching to `analyze_prototype_prediction` for PrototypeRuneNet.")
        return analyze_prototype_prediction(model, x_sample.cpu(), top_k=top_k, feature_names=feature_names)

    if not hasattr(model, 'rune_blocks'):
        print("Warning: This function is designed for models with 'rune_blocks'.")
        if hasattr(model, 'gated_tropical_agg'):
             return {"Layer 0 (Tropical Aggregator)": analyze_tropical_dominance(model.gated_tropical_agg.tropical_agg, x_sample, top_k, feature_names)}
        return {}
    
    model.eval()
    if x_sample.dim() == 1: x_sample = x_sample.unsqueeze(0)
    if feature_names is None: feature_names = get_feature_names(x_sample.shape[1], "Input_")

    path = {}
    with torch.no_grad():
        h = model.input_projection(x_sample)
        proj_feature_names = get_feature_names(h.shape[1], "Proj_")
        path['InputProjection'] = "Input projected to internal dimension."

        for i, block in enumerate(model.rune_blocks):
            block_analysis = {}
            dominance = analyze_tropical_dominance(block.gated_agg.tropical_agg, h, top_k, proj_feature_names)
            block_analysis['DominantTropicalTerms'] = dominance
            gate_values = block.gated_agg.gate_values.cpu().numpy()
            block_analysis['GateValues'] = {f: float(g) for f, g in zip(proj_feature_names, gate_values)}
            h = block(h)
            path[f'RUNEBlock_{i}'] = block_analysis

    with torch.no_grad():
        final_logits = model.output_head(h)
        final_activations = h.squeeze(0).detach()
        if model.output_head.out_features > 1:
            predicted_class_idx = int(torch.argmax(final_logits, dim=1).item())
            final_weights = model.output_head.weight[predicted_class_idx].detach()
            overall_score = float(final_logits.squeeze(0)[predicted_class_idx].item())
            prediction_info = {'PredictedClass': predicted_class_idx, 'Logit': overall_score}
        else:
            final_weights = model.output_head.weight.squeeze(0).detach()
            overall_score = float(final_logits.item())
            prediction_info = {'Score': overall_score}

    contributions = final_weights * final_activations
    top_contribs = torch.topk(contributions.abs(), k=top_k)
    
    final_analysis = []
    for val, idx in zip(top_contribs.values, top_contribs.indices):
        idx_item = idx.item()
        final_analysis.append({
            'Feature': proj_feature_names[idx_item],
            'Contribution': float(contributions[idx_item].item()),
            'Reason': f"Activation={float(final_activations[idx_item].item()):.2f} * Weight={float(final_weights[idx_item].item()):.2f}"
        })
        
    path['FinalPrediction'] = {**prediction_info, 'TopContributingFeatures': final_analysis}
    return path

def analyze_prototype_prediction(
    model: PrototypeRuneNet, x_sample: torch.Tensor, feature_names: list[str] = None, top_k: int = 3
) -> dict:
    model.to('cpu').eval()
    if feature_names is None: feature_names = get_feature_names(x_sample.shape[0], "Feature_")
    sample_tensor = x_sample.unsqueeze(0)
    with torch.no_grad():
        distances = model.prototype_layer(sample_tensor).cpu().numpy().flatten()
        output = model(sample_tensor)
        if output.shape[1] > 1:
            prediction = int(torch.argmax(output, dim=1).item())
        else:
            prediction = float(output.item())

    closest_indices = np.argsort(distances)[:top_k]
    analysis = {'prediction': prediction, 'distances': distances.tolist()}
    
    closest_prototypes_info = []
    for idx in closest_indices:
        prototype_features = model.prototype_layer.prototypes[idx].detach().cpu().numpy()
        feature_df = pd.DataFrame({
            'Feature': feature_names,
            'Prototype_Value': prototype_features,
            'Sample_Value': x_sample.cpu().numpy()
        })
        closest_prototypes_info.append({
            'index': int(idx),
            'distance': float(distances[idx]),
            'feature_comparison': feature_df
        })
    analysis['closest_prototypes'] = closest_prototypes_info
    return analysis

def plot_prototypes_with_tsne(
    model: PrototypeRuneNet, X_data: np.ndarray, y_data: np.ndarray, ax=None, title: str = "t-SNE Visualization of Data and Learned Prototypes"
):
    if TSNE is None or sns is None:
        raise ImportError("Please install scikit-learn and seaborn: `pip install scikit-learn seaborn`")
    model.to('cpu')
    prototypes = model.prototype_layer.prototypes.detach().cpu().numpy()
    combined_data = np.vstack((X_data, prototypes))
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_data)-1))
    reduced_data = tsne.fit_transform(combined_data)
    reduced_X, reduced_prototypes = reduced_data[:len(X_data)], reduced_data[len(X_data):]
    if ax is None: fig, ax = plt.subplots(figsize=(12, 9))
    sns.scatterplot(x=reduced_X[:, 0], y=reduced_X[:, 1], hue=y_data, palette='viridis', alpha=0.6, ax=ax, legend='full')
    ax.scatter(reduced_prototypes[:, 0], reduced_prototypes[:, 1], 
               marker='X', s=250, c='red', edgecolors='black', linewidth=1.5, label='Prototypes')
    ax.set_title(title, fontsize=16); ax.set_xlabel("t-SNE Dimension 1"); ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(); plt.tight_layout()
    return ax
