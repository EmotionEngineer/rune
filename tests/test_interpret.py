# tests/test_interpret.py

import torch
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rune.layers import (
    PairwiseDifferenceLayer,
    TropicalDifferenceAggregator,
    GatedTropicalDifferenceAggregator,
    CyclicTropicalDifferenceLayer
)
from rune.models import (
    PairwiseDifferenceNet,
    GatedTropicalDifferenceNet,
    InterpretableRuneNet
)
from rune.interpret import (
    plot_pairwise_difference_weights,
    plot_tropical_aggregator_params,
    plot_gated_aggregator_gates,
    plot_cyclic_layer_projection_weights,
    analyze_tropical_dominance,
    plot_feature_interaction_graph,
    plot_final_layer_contributions,
    trace_decision_path
)

# Test constants
DIM = 4
PROJ_DIM = 3
INPUT_DIM_CYCLIC = 5
IRN_INPUT_DIM = 8
IRN_BLOCK_DIM = 6
IRN_OUTPUT_DIM = 1

@pytest.fixture
def pdl_layer():
    return PairwiseDifferenceLayer(input_dim=DIM)

@pytest.fixture
def tda_layer():
    return TropicalDifferenceAggregator(dim=DIM)

@pytest.fixture
def gtda_layer():
    return GatedTropicalDifferenceAggregator(dim=DIM)

@pytest.fixture
def ctdl_layer_ste():
    return CyclicTropicalDifferenceLayer(input_dim=INPUT_DIM_CYCLIC, projection_dim=PROJ_DIM, use_angle_encoding=False)

@pytest.fixture
def irn_model():
    """An instance of the fully interpretable network."""
    return InterpretableRuneNet(
        input_dim=IRN_INPUT_DIM,
        output_dim=IRN_OUTPUT_DIM,
        num_blocks=2,
        block_dim=IRN_BLOCK_DIM
    )

@pytest.fixture
def pdn_interpretable():
    """A PairwiseDifferenceNet with an interpretable head."""
    return PairwiseDifferenceNet(input_dim=DIM, output_dim=1, interpretable_head=True)

@pytest.fixture
def x_sample_for_interpret():
    """A single data sample for interpretation functions."""
    return torch.randn(IRN_INPUT_DIM)

# --- Original Plotting Tests (Unchanged) ---
def test_plot_pairwise_difference_weights(pdl_layer):
    fig, ax = plt.subplots()
    plot_pairwise_difference_weights(pdl_layer, ax=ax)
    plt.close(fig)
    pdl_no_weights = PairwiseDifferenceLayer(input_dim=1)
    plot_pairwise_difference_weights(pdl_no_weights)

def test_plot_tropical_aggregator_params(tda_layer):
    fig, ax = plt.subplots()
    plot_tropical_aggregator_params(tda_layer, param_type="weights", ax=ax)
    plt.close(fig)
    fig, ax = plt.subplots()
    plot_tropical_aggregator_params(tda_layer, param_type="bias", ax=ax)
    plt.close(fig)
    with pytest.raises(ValueError):
        plot_tropical_aggregator_params(tda_layer, param_type="invalid")

def test_plot_gated_aggregator_gates(gtda_layer):
    fig, ax = plt.subplots()
    plot_gated_aggregator_gates(gtda_layer, ax=ax)
    plt.close(fig)

def test_plot_cyclic_layer_projection_weights(ctdl_layer_ste):
    fig, ax = plt.subplots()
    plot_cyclic_layer_projection_weights(ctdl_layer_ste, ax=ax)
    plt.close(fig)

def test_analyze_tropical_dominance(tda_layer):
    x_sample = torch.randn(DIM)
    analysis = analyze_tropical_dominance(tda_layer, x_sample, top_k=2)
    assert isinstance(analysis, dict)
    assert len(analysis) == DIM

def test_plot_feature_interaction_graph(pdn_interpretable):
    """Tests the graph plotting function."""
    try:
        import networkx
    except ImportError:
        pytest.skip("networkx not installed, skipping graph plot test")
    
    fig, ax = plt.subplots()
    plot_feature_interaction_graph(pdn_interpretable.pairwise_diff_layer, ax=ax)
    plt.close(fig)

def test_plot_final_layer_contributions(irn_model, x_sample_for_interpret):
    """Tests plotting contributions for the final layer."""
    fig, ax = plt.subplots()
    plot_final_layer_contributions(irn_model, x_sample_for_interpret, ax=ax)
    plt.close(fig)

def test_plot_final_layer_contributions_fallback(pdn_interpretable):
    """Tests contribution plotting on a model with 'output_layer'."""
    x_sample = torch.randn(DIM)
    fig, ax = plt.subplots()
    plot_final_layer_contributions(pdn_interpretable, x_sample, ax=ax)
    plt.close(fig)

def test_trace_decision_path_full(irn_model, x_sample_for_interpret):
    """Tests the full decision path tracing on InterpretableRuneNet."""
    analysis = trace_decision_path(irn_model, x_sample_for_interpret, top_k=2)
    
    assert isinstance(analysis, dict)
    assert 'InputProjection' in analysis
    assert 'RUNEBlock_0' in analysis
    assert 'RUNEBlock_1' in analysis
    assert 'FinalPrediction' in analysis
    
    block_0_analysis = analysis['RUNEBlock_0']
    assert 'DominantTropicalTerms' in block_0_analysis
    assert 'GateValues' in block_0_analysis
    assert isinstance(block_0_analysis['GateValues'], dict)
    
    final_pred_analysis = analysis['FinalPrediction']
    assert 'TopContributingFeatures' in final_pred_analysis
    assert 'OverallScore' in final_pred_analysis
    assert isinstance(final_pred_analysis['TopContributingFeatures'], list)
    assert len(final_pred_analysis['TopContributingFeatures'][0]) == 3

def test_trace_decision_path_fallback(gtda_layer):
    """Tests the fallback behavior of trace_decision_path on a simple layer."""
    # Note: This tests a conceptual use case where a simple GatedTropicalDifferenceNet
    # could be passed. We use the layer directly for simplicity here.
    model_like = torch.nn.Sequential(gtda_layer.tropical_agg)
    model_like.gated_tropical_agg = gtda_layer
    
    x_sample = torch.randn(DIM)
    # The function should print a warning but return a valid partial analysis
    analysis = trace_decision_path(model_like, x_sample)

    assert isinstance(analysis, dict)
    # Checks that it correctly fell back to analyzing the tropical aggregator
    assert "Layer 0 (Tropical Aggregator)" in analysis
    assert len(analysis["Layer 0 (Tropical Aggregator)"]) == DIM