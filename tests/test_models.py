# tests/test_models.py

import torch
import torch.nn as nn
import pytest
from rune.models import (
    PairwiseDifferenceNet,
    GatedTropicalDifferenceNet,
    CyclicTropicalDifferenceNet,
    RUNEBlock,
    InterpretableRuneNet,
    PrototypeRuneNet
)

BATCH_SIZE = 3
INPUT_DIM = 10
OUTPUT_DIM = 2
HIDDEN_DIMS = [16, 8]
PROJ_DIM_MODEL = 12
BLOCK_DIM = 32
NUM_PROTOTYPES_MODEL = 16

@pytest.fixture
def sample_data():
    return torch.randn(BATCH_SIZE, INPUT_DIM)

@pytest.fixture
def positive_sample_data():
    return torch.rand(BATCH_SIZE, INPUT_DIM) + 0.1

@pytest.fixture
def block_data():
    return torch.randn(BATCH_SIZE, BLOCK_DIM)

@pytest.fixture
def positive_block_data():
    return torch.rand(BATCH_SIZE, BLOCK_DIM) + 0.1

def test_pairwise_difference_net(sample_data):
    model = PairwiseDifferenceNet(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dims=HIDDEN_DIMS)
    output = model(sample_data)
    assert output.shape == (BATCH_SIZE, OUTPUT_DIM)
    output.sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

def test_gated_tropical_difference_net(sample_data):
    model = GatedTropicalDifferenceNet(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dims=HIDDEN_DIMS)
    output = model(sample_data)
    assert output.shape == (BATCH_SIZE, OUTPUT_DIM)
    output.sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

def test_cyclic_tropical_difference_net_ste(sample_data):
    model = CyclicTropicalDifferenceNet(
        input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, projection_dim=PROJ_DIM_MODEL,
        hidden_dims=HIDDEN_DIMS, use_angle_encoding=False, use_ste_modulo=True
    )
    output = model(sample_data)
    assert output.shape == (BATCH_SIZE, OUTPUT_DIM)
    output.sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

def test_cyclic_tropical_difference_net_angle(sample_data):
    model = CyclicTropicalDifferenceNet(
        input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, projection_dim=PROJ_DIM_MODEL,
        hidden_dims=HIDDEN_DIMS, use_angle_encoding=True
    )
    output = model(sample_data)
    assert output.shape == (BATCH_SIZE, OUTPUT_DIM)
    output.sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize("interaction_type", ['difference', 'log_ratio', 'ratio'])
def test_rune_block(block_data, positive_block_data, interaction_type):
    """Tests the RUNEBlock for shape preservation and gradient flow."""
    block = RUNEBlock(dim=BLOCK_DIM, interaction_type=interaction_type)
    input_data = positive_block_data if interaction_type != 'difference' else block_data
    
    output = block(input_data)
    assert not torch.isnan(output).any()
    assert output.shape == input_data.shape, "RUNEBlock did not preserve shape"
    
    output.sum().backward()
    assert all(p.grad is not None for p in block.parameters() if p.requires_grad), "Grads missing in RUNEBlock"

@pytest.mark.parametrize("interaction_type", ['difference', 'log_ratio', 'ratio'])
def test_interpretable_rune_net(sample_data, positive_sample_data, interaction_type):
    """Tests the full InterpretableRuneNet."""
    model = InterpretableRuneNet(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        num_blocks=2,
        block_dim=BLOCK_DIM,
        interaction_type=interaction_type
    )
    input_data = positive_sample_data if interaction_type != 'difference' else sample_data

    output = model(input_data)
    assert not torch.isnan(output).any()
    assert output.shape == (BATCH_SIZE, OUTPUT_DIM), "InterpretableRuneNet output shape mismatch"
    
    output.sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad), "Grads missing in InterpretableRuneNet"

@pytest.mark.parametrize("model_class", [
    PairwiseDifferenceNet, GatedTropicalDifferenceNet, CyclicTropicalDifferenceNet
])
def test_models_with_interpretable_head(model_class, sample_data):
    if model_class == CyclicTropicalDifferenceNet:
        model = model_class(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, interpretable_head=True, projection_dim=PROJ_DIM_MODEL)
    else:
        model = model_class(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, interpretable_head=True)
    mlp_attr = 'mlp' if hasattr(model, 'mlp') else 'hidden_mlp'
    assert isinstance(getattr(model, mlp_attr), nn.Identity)
    output = model(sample_data)
    assert output.shape == (BATCH_SIZE, OUTPUT_DIM)
    output.sum().backward()

def test_interpretable_rune_net_helpers(sample_data):
    model = InterpretableRuneNet(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, block_dim=BLOCK_DIM)
    initial_tau = model.rune_blocks[0].gated_agg.tropical_agg.tau.item()
    new_tau = 0.5
    model.set_tau(new_tau)
    updated_tau = model.rune_blocks[0].gated_agg.tropical_agg.tau.item()
    assert abs(updated_tau - new_tau) < 1e-6
    assert abs(initial_tau - updated_tau) > 1e-6
    reg_loss = model.get_regularization_loss()
    assert isinstance(reg_loss, torch.Tensor)
    assert reg_loss.item() >= 0.0

@pytest.mark.parametrize("interaction_type", ['difference', 'log_ratio', 'ratio'])
def test_prototype_rune_net(sample_data, interaction_type):
    """Tests the full PrototypeRuneNet."""
    # sample_data can be used here because the prototype layer outputs distances, which are always positive.
    model = PrototypeRuneNet(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        num_prototypes=NUM_PROTOTYPES_MODEL,
        block_dim=BLOCK_DIM,
        interaction_type=interaction_type
    )
    output = model(sample_data)
    assert not torch.isnan(output).any()
    assert output.shape == (BATCH_SIZE, OUTPUT_DIM), "PrototypeRuneNet output shape mismatch"
    
    output.sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad), "Grads missing in PrototypeRuneNet"

    new_tau = 0.99
    model.set_tau(new_tau)
    updated_tau = model.rune_net.rune_blocks[0].gated_agg.tropical_agg.tau.item()
    assert abs(updated_tau - new_tau) < 1e-6, "set_tau did not pass through to inner rune_net"
    
    reg_loss = model.get_regularization_loss()
    assert reg_loss.item() >= 0.0, "get_regularization_loss did not pass through"
