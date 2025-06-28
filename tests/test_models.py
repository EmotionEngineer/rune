# tests/test_models.py

import torch
import torch.nn as nn
import pytest
from rune.models import (
    PairwiseDifferenceNet,
    GatedTropicalDifferenceNet,
    CyclicTropicalDifferenceNet,
    RUNEBlock,
    InterpretableRuneNet
)

BATCH_SIZE = 3
INPUT_DIM = 10
OUTPUT_DIM = 2
HIDDEN_DIMS = [16, 8]
PROJ_DIM_MODEL = 12
BLOCK_DIM = 32

@pytest.fixture
def sample_data():
    return torch.randn(BATCH_SIZE, INPUT_DIM)

@pytest.fixture
def block_data():
    return torch.randn(BATCH_SIZE, BLOCK_DIM)

def test_pairwise_difference_net(sample_data):
    model = PairwiseDifferenceNet(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_dims=HIDDEN_DIMS
    )
    output = model(sample_data)
    assert output.shape == (BATCH_SIZE, OUTPUT_DIM), "PairwiseDifferenceNet output shape mismatch"
    output.sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad), "Grads missing in PairwiseDifferenceNet"

def test_gated_tropical_difference_net(sample_data):
    model = GatedTropicalDifferenceNet(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_dims=HIDDEN_DIMS
    )
    output = model(sample_data)
    assert output.shape == (BATCH_SIZE, OUTPUT_DIM), "GatedTropicalDifferenceNet output shape mismatch"
    output.sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad), "Grads missing in GatedTropicalDifferenceNet"

def test_cyclic_tropical_difference_net_ste(sample_data):
    model = CyclicTropicalDifferenceNet(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        projection_dim=PROJ_DIM_MODEL,
        hidden_dims=HIDDEN_DIMS,
        use_angle_encoding=False,
        use_ste_modulo=True
    )
    output = model(sample_data)
    assert output.shape == (BATCH_SIZE, OUTPUT_DIM), "CyclicTropicalDifferenceNet (STE) output shape mismatch"
    output.sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad), "Grads missing in CyclicTropicalDifferenceNet (STE)"

def test_cyclic_tropical_difference_net_angle(sample_data):
    model = CyclicTropicalDifferenceNet(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        projection_dim=PROJ_DIM_MODEL,
        hidden_dims=HIDDEN_DIMS,
        use_angle_encoding=True
    )
    output = model(sample_data)
    assert output.shape == (BATCH_SIZE, OUTPUT_DIM), "CyclicTropicalDifferenceNet (Angle) output shape mismatch"
    output.sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad), "Grads missing in CyclicTropicalDifferenceNet (Angle)"

def test_rune_block(block_data):
    """Tests the RUNEBlock for shape preservation and gradient flow."""
    block = RUNEBlock(dim=BLOCK_DIM)
    output = block(block_data)
    
    # Check that the output dimension is the same as the input dimension
    assert output.shape == block_data.shape, "RUNEBlock did not preserve shape"
    
    # Check for gradient flow
    output.sum().backward()
    assert all(p.grad is not None for p in block.parameters() if p.requires_grad), "Grads missing in RUNEBlock"

def test_interpretable_rune_net(sample_data):
    """Tests the full InterpretableRuneNet."""
    model = InterpretableRuneNet(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        num_blocks=2,
        block_dim=BLOCK_DIM
    )
    output = model(sample_data)
    assert output.shape == (BATCH_SIZE, OUTPUT_DIM), "InterpretableRuneNet output shape mismatch"
    
    output.sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad), "Grads missing in InterpretableRuneNet"

@pytest.mark.parametrize("model_class", [
    PairwiseDifferenceNet,
    GatedTropicalDifferenceNet,
    CyclicTropicalDifferenceNet
])
def test_models_with_interpretable_head(model_class, sample_data):
    """Tests that the interpretable_head flag correctly removes the MLP."""
    # Cyclic model has different init args
    if model_class == CyclicTropicalDifferenceNet:
        model = model_class(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, interpretable_head=True, projection_dim=PROJ_DIM_MODEL)
    else:
        model = model_class(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, interpretable_head=True)
    
    # Check that the MLP part is now an Identity layer
    mlp_attr = 'mlp' if hasattr(model, 'mlp') else 'hidden_mlp'
    assert isinstance(getattr(model, mlp_attr), nn.Identity), f"{model_class.__name__} did not set MLP to Identity"

    # Ensure forward pass still works
    output = model(sample_data)
    assert output.shape == (BATCH_SIZE, OUTPUT_DIM)
    output.sum().backward() # And gradients flow