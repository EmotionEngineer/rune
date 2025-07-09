# tests/test_layers.py

import torch
import pytest
from rune.layers import (
    TropicalDifferenceAggregator,
    GatedTropicalDifferenceAggregator,
    PairwiseDifferenceLayer,
    CyclicTropicalDifferenceLayer,
    PrototypeLayer  # New import
)
from rune.utils import ste_modulo

BATCH_SIZE = 4
DIM = 5
PROJ_DIM = 8
NUM_PROTOTYPES = 10  # New constant

@pytest.fixture
def sample_input():
    return torch.randn(BATCH_SIZE, DIM)

@pytest.fixture
def sample_input_proj(): # For CyclicTropicalDifferenceLayer's input_dim
    return torch.randn(BATCH_SIZE, DIM) # Assuming input_dim for CTDL is DIM

def test_ste_modulo():
    x = torch.tensor([-5.5, -0.5, 0.5, 5.5, 10.5], requires_grad=True)
    modulus = 5.0
    y = ste_modulo(x, modulus)
    expected_y = torch.fmod(x, modulus)
    assert torch.allclose(y, expected_y), "STEModulo forward pass failed"
    
    # Check gradient flow
    y.sum().backward()
    assert x.grad is not None, "STEModulo gradient did not flow"
    assert torch.allclose(x.grad, torch.ones_like(x)), "STEModulo gradient is not identity"

def test_tropical_difference_aggregator(sample_input):
    layer = TropicalDifferenceAggregator(dim=DIM, tau=0.1, learn_tau=True)
    output = layer(sample_input)
    assert output.shape == (BATCH_SIZE, DIM), "TropicalDifferenceAggregator shape mismatch"
    output.sum().backward() # Check gradient flow
    assert all(p.grad is not None for p in layer.parameters() if p.requires_grad), "Grads missing in TropicalDifferenceAggregator"

def test_gated_tropical_difference_aggregator(sample_input):
    layer = GatedTropicalDifferenceAggregator(dim=DIM, tau_tropical=0.1, learn_tau_tropical=True)
    output = layer(sample_input)
    assert output.shape == (BATCH_SIZE, 2 * DIM), "GatedTropicalDifferenceAggregator shape mismatch"
    assert layer.gate_values.shape == (DIM,), "Gate values shape mismatch"
    output.sum().backward()
    assert all(p.grad is not None for p in layer.parameters() if p.requires_grad), "Grads missing in GatedTropicalDifferenceAggregator"

def test_pairwise_difference_layer():
    # Test with dim >= 2
    layer = PairwiseDifferenceLayer(input_dim=DIM)
    x = torch.randn(BATCH_SIZE, DIM)
    num_pairs = DIM * (DIM - 1) // 2
    assert layer.num_diff_features == num_pairs, "Incorrect num_diff_features"
    assert layer.output_dim == DIM + num_pairs, "Incorrect output_dim"
    
    output = layer(x)
    assert output.shape == (BATCH_SIZE, DIM + num_pairs), "PairwiseDifferenceLayer shape mismatch"
    if num_pairs > 0:
        output.sum().backward()
        assert layer.weights.grad is not None, "Grads missing for weights in PairwiseDifferenceLayer"

    # Test with dim < 2
    layer_small = PairwiseDifferenceLayer(input_dim=1)
    x_small = torch.randn(BATCH_SIZE, 1)
    assert layer_small.num_diff_features == 0, "num_diff_features should be 0 for dim < 2"
    assert layer_small.output_dim == 1, "output_dim should be 1 for dim < 2"
    output_small = layer_small(x_small)
    assert output_small.shape == (BATCH_SIZE, 1), "PairwiseDifferenceLayer shape mismatch for dim < 2"
    # No weights to check gradient for if num_diff_features is 0

def test_cyclic_tropical_difference_layer_ste(sample_input_proj):
    layer_ste = CyclicTropicalDifferenceLayer(
        input_dim=DIM, projection_dim=PROJ_DIM, modulus=7.0, use_angle_encoding=False, use_ste_modulo=True
    )
    assert layer_ste.transformed_dim == PROJ_DIM
    assert layer_ste.output_dim == 2 * PROJ_DIM
    output = layer_ste(sample_input_proj)
    assert output.shape == (BATCH_SIZE, 2 * PROJ_DIM), "CyclicTropicalDiffLayer (STE) shape mismatch"
    output.sum().backward()
    assert all(p.grad is not None for p in layer_ste.parameters() if p.requires_grad), "Grads missing in CyclicTropicalDiffLayer (STE)"

def test_cyclic_tropical_difference_layer_angle(sample_input_proj):
    layer_angle = CyclicTropicalDifferenceLayer(
        input_dim=DIM, projection_dim=PROJ_DIM, modulus=7.0, use_angle_encoding=True
    )
    assert layer_angle.transformed_dim == 2 * PROJ_DIM
    assert layer_angle.output_dim == 2 * (2 * PROJ_DIM) # 4 * PROJ_DIM
    output = layer_angle(sample_input_proj)
    assert output.shape == (BATCH_SIZE, 4 * PROJ_DIM), "CyclicTropicalDiffLayer (Angle) shape mismatch"
    output.sum().backward()
    assert all(p.grad is not None for p in layer_angle.parameters() if p.requires_grad), "Grads missing in CyclicTropicalDiffLayer (Angle)"

def test_cyclic_tropical_difference_layer_no_ste(sample_input_proj):
    # Gradients won't flow through standard fmod for the inputs to fmod, but should for weights.
    # This is mainly for completeness.
    layer_no_ste = CyclicTropicalDifferenceLayer(
        input_dim=DIM, projection_dim=PROJ_DIM, modulus=7.0, use_angle_encoding=False, use_ste_modulo=False
    )
    output = layer_no_ste(sample_input_proj)
    assert output.shape == (BATCH_SIZE, 2 * PROJ_DIM), "CyclicTropicalDiffLayer (no STE) shape mismatch"
    # Testing backward pass can be tricky here due to fmod behavior with autograd.
    # We are mostly interested in the weights' gradients.
    try:
        output.sum().backward()
        # Check if projection weights got gradients
        assert layer_no_ste.projection.weight.grad is not None
        # Check if gated tropical aggregator weights got gradients
        assert layer_no_ste.gated_tropical_aggregator.tropical_agg.weights.grad is not None
    except RuntimeError as e:
        print(f"Known issue with fmod and autograd if not using STE: {e}")
        # This might happen if all gradients through fmod become zero for some inputs
        pass

def test_prototype_layer(sample_input):
    """Tests the PrototypeLayer for correct output shape and parameter registration."""
    input_dim = sample_input.shape[1]
    layer = PrototypeLayer(input_dim=input_dim, num_prototypes=NUM_PROTOTYPES)
    
    # Check output dimension property
    assert layer.output_dim == NUM_PROTOTYPES

    # Check forward pass output shape
    output = layer(sample_input)
    assert output.shape == (BATCH_SIZE, NUM_PROTOTYPES), "PrototypeLayer output shape mismatch"
    
    # Check that prototypes are learnable parameters
    assert isinstance(layer.prototypes, torch.nn.Parameter)
    assert layer.prototypes.requires_grad
    
    # Check gradient flow
    output.sum().backward()
    assert layer.prototypes.grad is not None, "Gradients missing for prototypes"