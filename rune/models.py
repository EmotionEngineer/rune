# rune/models.py

import torch
import torch.nn as nn
from .layers import (
    PairwiseDifferenceLayer,
    GatedTropicalDifferenceAggregator,
    CyclicTropicalDifferenceLayer,
    PrototypeLayer
)

class RUNEBlock(nn.Module):
    """
    A residual block built around the GatedTropicalDifferenceAggregator.
    """
    def __init__(self,
                 dim: int,
                 tau_tropical: float = 0.2,
                 learn_tau_tropical: bool = False,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.gated_agg = GatedTropicalDifferenceAggregator(
            dim=dim,
            tau_tropical=tau_tropical,
            learn_tau_tropical=learn_tau_tropical
        )
        self.projection = nn.Linear(2 * dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(dim)

    def set_tau(self, new_tau: float):
        """Sets a new temperature tau for the internal aggregator."""
        self.gated_agg.set_tau(new_tau)

    def get_regularization_loss(self) -> torch.Tensor:
        """Gets the regularization loss from the internal aggregator."""
        return self.gated_agg.get_regularization_loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        aggregated_features = self.gated_agg(x)
        projected = self.projection(aggregated_features)
        return self.norm(x + self.dropout(projected))


class InterpretableRuneNet(nn.Module):
    """
    A deep, fully interpretable network built by stacking RUNEBlocks.
    Supports training with L1 regularization and temperature annealing.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_blocks: int = 2,
                 block_dim: int = 32,
                 tau_tropical: float = 0.2,
                 learn_tau_tropical: bool = False,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, block_dim)

        # --- FIX: Changed ModuleList to Sequential ---
        self.rune_blocks = nn.Sequential(
            *[RUNEBlock(
                dim=block_dim,
                tau_tropical=tau_tropical,
                learn_tau_tropical=learn_tau_tropical,
                dropout_rate=dropout_rate
              ) for _ in range(num_blocks)]
        )
        
        self.output_head = nn.Linear(block_dim, output_dim)

    def set_tau(self, new_tau: float):
        """Sets a new temperature tau for all RUNEBlocks. Used for annealing."""
        for block in self.rune_blocks:
            block.set_tau(new_tau)
            
    def get_regularization_loss(self) -> torch.Tensor:
        """Calculates total L1 regularization loss from all RUNEBlocks."""
        reg_loss = 0.0
        for block in self.rune_blocks:
            reg_loss += block.get_regularization_loss()
        return reg_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        # --- FIX: Simplified forward pass ---
        # No longer need to manually iterate
        x = self.rune_blocks(x)
        x = self.output_head(x)
        return x


class PrototypeRuneNet(nn.Module):
    """
    A network for case-based reasoning. It first computes the similarity of an
    input to a set of learnable prototypes and then processes these similarities
    with an InterpretableRuneNet.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_prototypes: int,
                 num_blocks: int = 2,
                 block_dim: int = 32,
                 **kwargs):
        super().__init__()
        self.prototype_layer = PrototypeLayer(input_dim, num_prototypes)
        # The RUNE part operates on the vector of distances to the prototypes
        self.rune_net = InterpretableRuneNet(
            input_dim=num_prototypes,
            output_dim=output_dim,
            num_blocks=num_blocks,
            block_dim=block_dim,
            **kwargs
        )

    def set_tau(self, new_tau: float):
        """Pass-through method to set tau on the internal RUNE network."""
        self.rune_net.set_tau(new_tau)

    def get_regularization_loss(self) -> torch.Tensor:
        """Pass-through method to get regularization loss from the internal RUNE network."""
        return self.rune_net.get_regularization_loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prototype_distances = self.prototype_layer(x)
        return self.rune_net(prototype_distances)


class PairwiseDifferenceNet(nn.Module):
    """
    A network built upon the PairwiseDifferenceLayer.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: list[int] = [32, 16],
                 dropout_rate: float = 0.2,
                 use_batchnorm: bool = True,
                 interpretable_head: bool = False):
        super().__init__()
        self.pairwise_diff_layer = PairwiseDifferenceLayer(input_dim)
        current_dim = self.pairwise_diff_layer.output_dim
        if interpretable_head:
            self.hidden_mlp = nn.Identity()
        else:
            layers = []
            for h_dim in hidden_dims:
                layers.extend([nn.Linear(current_dim, h_dim), nn.ReLU()])
                if use_batchnorm: layers.append(nn.BatchNorm1d(h_dim))
                if dropout_rate > 0: layers.append(nn.Dropout(dropout_rate))
                current_dim = h_dim
            self.hidden_mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pairwise_diff_layer(x)
        x = self.hidden_mlp(x)
        x = self.output_layer(x)
        return x


class GatedTropicalDifferenceNet(nn.Module):
    """
    A network using GatedTropicalDifferenceAggregator as its core feature extractor.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: list[int] = [32, 16],
                 tau_tropical: float = 0.2,
                 learn_tau_tropical: bool = False,
                 dropout_rate: float = 0.2,
                 use_batchnorm: bool = True,
                 interpretable_head: bool = False):
        super().__init__()
        self.gated_tropical_agg = GatedTropicalDifferenceAggregator(
            dim=input_dim, tau_tropical=tau_tropical, learn_tau_tropical=learn_tau_tropical
        )
        current_dim = input_dim * 2
        if interpretable_head:
            self.mlp = nn.Identity()
        else:
            layers = []
            for h_dim in hidden_dims:
                layers.extend([nn.Linear(current_dim, h_dim), nn.ReLU()])
                if use_batchnorm: layers.append(nn.BatchNorm1d(h_dim))
                if dropout_rate > 0: layers.append(nn.Dropout(dropout_rate))
                current_dim = h_dim
            self.mlp = nn.Sequential(*layers)
        self.output_head = nn.Linear(current_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gated_tropical_agg(x)
        x = self.mlp(x)
        x = self.output_head(x)
        return x


class CyclicTropicalDifferenceNet(nn.Module):
    """
    A hybrid model incorporating cyclic transformations and tropical aggregation.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 projection_dim: int = 32,
                 hidden_dims: list[int] = [64, 32],
                 modulus: float = 7.0,
                 use_angle_encoding: bool = False,
                 tau_tropical: float = 0.2,
                 learn_tau_tropical: bool = False,
                 use_ste_modulo: bool = True,
                 dropout_rate: float = 0.2,
                 use_batchnorm: bool = True,
                 interpretable_head: bool = False):
        super().__init__()
        self.cyclic_tropical_diff_layer = CyclicTropicalDifferenceLayer(
            input_dim=input_dim, projection_dim=projection_dim, tau_tropical=tau_tropical,
            learn_tau_tropical=learn_tau_tropical, modulus=modulus,
            use_ste_modulo=use_ste_modulo, use_angle_encoding=use_angle_encoding
        )
        current_dim = self.cyclic_tropical_diff_layer.output_dim
        if interpretable_head:
            self.mlp = nn.Identity()
        else:
            layers = []
            for h_dim in hidden_dims:
                layers.extend([nn.Linear(current_dim, h_dim), nn.ReLU()])
                if use_batchnorm: layers.append(nn.BatchNorm1d(h_dim))
                if dropout_rate > 0: layers.append(nn.Dropout(dropout_rate))
                current_dim = h_dim
            self.mlp = nn.Sequential(*layers)
        self.output_head = nn.Linear(current_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cyclic_tropical_diff_layer(x)
        x = self.mlp(x)
        x = self.output_head(x)
        return x