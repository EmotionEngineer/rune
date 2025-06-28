# rune/models.py

import torch
import torch.nn as nn
from .layers import (
    PairwiseDifferenceLayer,
    GatedTropicalDifferenceAggregator,
    CyclicTropicalDifferenceLayer
)

class RUNEBlock(nn.Module):
    """
    A residual block built around the GatedTropicalDifferenceAggregator.
    It takes an input of dimension D, processes it, and returns an output
    of dimension D, making it suitable for stacking.

    Structure:
    1. GatedTropicalDifferenceAggregator(D) -> outputs 2D
    2. Linear(2D -> D) to project back to original dimension
    3. Residual connection: output = proj(h) + x
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
        # The aggregator outputs 2*dim, this projects it back to dim
        self.projection = nn.Linear(2 * dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
        # LayerNorm is often more stable in residual blocks than BatchNorm
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, dim)

        Returns:
            Output tensor of shape (batch_size, dim)
        """
        # Gated aggregation produces (batch, 2*dim)
        aggregated_features = self.gated_agg(x)
        # Project back to original dimension
        projected = self.projection(aggregated_features)
        # Apply dropout and add residual connection, then normalize
        # Pre-normalization (norm(x + proj(h))) is a common pattern
        return self.norm(x + self.dropout(projected))

class InterpretableRuneNet(nn.Module):
    """
    A deep, fully interpretable network built by stacking RUNEBlocks.
    It avoids any standard MLP hidden layers, ensuring that every transformation
    is based on an interpretable RUNE operation.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_blocks: int = 2,
                 block_dim: int = 32, # Dimension inside the stack of blocks
                 tau_tropical: float = 0.2,
                 learn_tau_tropical: bool = False,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, block_dim)

        self.rune_blocks = nn.Sequential(
            *[RUNEBlock(
                dim=block_dim,
                tau_tropical=tau_tropical,
                learn_tau_tropical=learn_tau_tropical,
                dropout_rate=dropout_rate
              ) for _ in range(num_blocks)]
        )
        
        self.output_head = nn.Linear(block_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.rune_blocks(x)
        x = self.output_head(x)
        return x


class PairwiseDifferenceNet(nn.Module):
    """
    A network built upon the PairwiseDifferenceLayer.
    The first layer expands features with pairwise differences, followed by
    an MLP that learns from this augmented representation.
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
                layers.append(nn.Linear(current_dim, h_dim))
                layers.append(nn.ReLU())
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(h_dim))
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
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
    It effectively combines max-like and mean-like aggregation of pairwise interactions,
    followed by an MLP.
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
            dim=input_dim,
            tau_tropical=tau_tropical,
            learn_tau_tropical=learn_tau_tropical
        )
        
        current_dim = input_dim * 2
        
        if interpretable_head:
            self.mlp = nn.Identity()
        else:
            layers = []
            for h_dim in hidden_dims:
                layers.append(nn.Linear(current_dim, h_dim))
                layers.append(nn.ReLU())
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(h_dim))
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
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
    A hybrid model incorporating:
    1. Group-like structures (via modulo/angle transformations).
    2. Tropical geometry (via max-plus like aggregation).
    3. Differential operators (via pairwise differences within aggregators).
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
            input_dim=input_dim,
            projection_dim=projection_dim,
            tau_tropical=tau_tropical,
            learn_tau_tropical=learn_tau_tropical,
            modulus=modulus,
            use_ste_modulo=use_ste_modulo,
            use_angle_encoding=use_angle_encoding
        )

        current_dim = self.cyclic_tropical_diff_layer.output_dim

        if interpretable_head:
            self.mlp = nn.Identity()
        else:
            layers = []
            for h_dim in hidden_dims:
                layers.append(nn.Linear(current_dim, h_dim))
                layers.append(nn.ReLU())
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(h_dim))
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
                current_dim = h_dim
            self.mlp = nn.Sequential(*layers)
            
        self.output_head = nn.Linear(current_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cyclic_tropical_diff_layer(x)
        x = self.mlp(x)
        x = self.output_head(x)
        return x
