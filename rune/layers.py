# rune/layers.py

import torch
import torch.nn as nn
import math
from .utils import ste_modulo

class TropicalDifferenceAggregator(nn.Module):
    """
    TropicalDifferenceAggregator: Aggregates weighted pairwise differences (x_i - x_j)
    using a "soft" max-plus operation (log-sum-exp), a differentiable
    analogue of the max operation.

    The aggregation is performed for each output feature k over pairs of input features (i,j):
    y_k = tau * logsumexp_j ( ( (x_i - x_j) * W_ijk + B_ijk ) / tau )
    Where typically i is fixed for y_k (e.g. i=k) and j iterates over other features.
    In this implementation, for each output dimension (which matches input dimension by default),
    it aggregates differences from all other dimensions.
    """
    def __init__(self,
                 dim: int,
                 tau: float = 0.1,
                 learn_tau: bool = False):
        super().__init__()
        self.dim = dim
        # Weights and biases for each (output_dim_k, input_dim_i, input_dim_j)
        # For simplicity, we make W_ijk and B_ijk depend only on (i,j) for output k=i
        # So weights become W_ij and B_ij applied for output y_i
        self.weights = nn.Parameter(torch.ones(dim, dim))
        self.bias = nn.Parameter(torch.zeros(dim, dim))

        if learn_tau:
            self.log_tau = nn.Parameter(torch.log(torch.tensor(tau, dtype=torch.float32)))
        else:
            self.register_buffer('log_tau', torch.log(torch.tensor(tau, dtype=torch.float32)))

        # Mask to avoid self-difference (x_i - x_i)
        self.register_buffer('mask', ~torch.eye(dim, dtype=torch.bool)) # (dim, dim)

    @property
    def tau(self) -> torch.Tensor:
        """The temperature parameter for log-sum-exp."""
        return self.log_tau.exp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, dim).

        Returns:
            Tensor of shape (batch_size, dim) representing aggregated differences.
        """
        B, D = x.shape
        assert D == self.dim, f"Input dimension mismatch: expected {self.dim}, got {D}"

        # x_diff: (B, D, D) where x_diff[b, i, j] = x[b, i] - x[b, j]
        x_diff = x.unsqueeze(2) - x.unsqueeze(1) # (B, D, 1) - (B, 1, D) -> (B, D, D)

        # weighted_diff: (B, D, D) where weighted_diff[b,i,j] = (x[b,i]-x[b,j])*W_ij + B_ij
        # These weights/biases are specific to the output dimension 'i'
        weighted_diff = x_diff * self.weights.unsqueeze(0) + self.bias.unsqueeze(0)

        # Apply mask to ignore diagonal elements (i.e., x_i - x_i)
        # masked_fill_ expects a boolean mask where True means fill.
        # self.mask is True for off-diagonal, False for diagonal. So we need ~self.mask.
        weighted_diff = weighted_diff.masked_fill(~self.mask.unsqueeze(0), float('-inf'))

        # Aggregation using log-sum-exp along dimension j for each i
        # y_i = tau * logsumexp_j (weighted_diff_ij / tau)
        # Input to logsumexp: (B, D, D). Sum over dim=2 (j's)
        y = self.tau * torch.logsumexp(weighted_diff / self.tau, dim=2) # (B, D)

        return y


class GatedTropicalDifferenceAggregator(nn.Module):
    """
    Combines tropical aggregation (max-like) and mean aggregation of pairwise
    differences using a learnable gate that determines their relative contribution.
    The output is concatenated with the original input.
    """
    def __init__(self,
                 dim: int,
                 tau_tropical: float = 0.2,
                 learn_tau_tropical: bool = False):
        super().__init__()
        self.dim = dim
        self.tropical_agg = TropicalDifferenceAggregator(dim, tau=tau_tropical, learn_tau=learn_tau_tropical)

        # Parameters for mean aggregation branch
        self.mean_weights = nn.Parameter(torch.ones(dim, dim))
        # self.mean_bias = nn.Parameter(torch.zeros(dim, dim)) # Optional: bias for mean

        # Gate parameters: one gate value per output dimension
        self.gate_params = nn.Parameter(torch.zeros(dim))

        self.register_buffer('mask', ~torch.eye(dim, dtype=torch.bool))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, dim).

        Returns:
            Tensor of shape (batch_size, 2 * dim) by concatenating
            the original input `x` with the gated aggregated features.
        """
        B, D = x.shape
        assert D == self.dim, f"Input dimension mismatch: expected {self.dim}, got {D}"

        # 1. Tropical (max-like) branch
        y_tropical = self.tropical_agg(x) # (B, D)

        # 2. Mean aggregation (mean-like) branch
        # x_diff: (B, D, D) where x_diff[b, i, j] = x[b, i] - x[b, j]
        x_diff = x.unsqueeze(2) - x.unsqueeze(1) # (B, D, 1) - (B, 1, D) -> (B, D, D)
        weighted_mean_diff = x_diff * self.mean_weights.unsqueeze(0)
        # weighted_mean_diff += self.mean_bias.unsqueeze(0) # Optional

        # Apply mask: fill diagonal with 0 for mean calculation, so they don't contribute.
        # masked_fill_ expects a boolean mask where True means fill.
        weighted_mean_diff = weighted_mean_diff.masked_fill(~self.mask.unsqueeze(0), 0.0)

        # Sum over j and divide by (D-1) for non-diagonal elements, or D if including diagonal
        # self.mask.sum(dim=1) gives (D-1) for each row.
        num_effective_elements = self.mask.sum(dim=1, dtype=torch.float32).clamp(min=1.0) # (D,)
        y_mean = weighted_mean_diff.sum(dim=2) / num_effective_elements.unsqueeze(0) # (B,D)

        # 3. Gating mechanism
        # Gate g: (D,). Expand to (B, D) for broadcasting
        g = torch.sigmoid(self.gate_params).unsqueeze(0)
        y_combined = g * y_tropical + (1 - g) * y_mean # (B, D)

        # 4. Concatenate with original input
        return torch.cat([x, y_combined], dim=1) # (B, 2*D)

    @property
    def gate_values(self) -> torch.Tensor:
        """Returns the sigmoid-activated gate values."""
        return torch.sigmoid(self.gate_params.detach())


class PairwiseDifferenceLayer(nn.Module):
    """
    Computes all unique weighted pairwise differences (x_i - x_j) between
    elements of the input vector and concatenates them to the original features.
    This allows the model to explicitly consider relative magnitudes.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        if input_dim < 2:
            self.indices = torch.empty(0, 2, dtype=torch.long)
        else:
            # Generate all unique pairs of indices (i, j) where i < j
            # To get all (i,j) where i!=j, one would use permutations of size 2.
            # Combinations ensures (i,j) is not repeated as (j,i) and i!=j.
            self.indices = torch.combinations(torch.arange(input_dim), r=2) # Shape: (num_pairs, 2)
        self.register_buffer('feature_indices', self.indices)

        self.num_diff_features = self.feature_indices.shape[0]
        if self.num_diff_features > 0:
            # Learnable weights for each difference feature
            self.weights = nn.Parameter(torch.ones(self.num_diff_features))
        else:
            # To avoid errors if nn.Parameter is empty, register as buffer or None
            self.register_parameter('weights', None)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Tensor of shape (batch_size, input_dim + num_diff_features).
        """
        if self.num_diff_features == 0:
            return x

        # x_select_0: (batch_size, num_pairs) using first index of each pair
        # x_select_1: (batch_size, num_pairs) using second index of each pair
        x_select_0 = x[:, self.feature_indices[:, 0]]
        x_select_1 = x[:, self.feature_indices[:, 1]]

        # differences: (batch_size, num_pairs)
        differences = (x_select_0 - x_select_1)

        # weighted_differences: (batch_size, num_pairs)
        weighted_differences = differences * self.weights.unsqueeze(0)

        return torch.cat([x, weighted_differences], dim=1)

    @property
    def output_dim(self) -> int:
        return self.input_dim + self.num_diff_features


class CyclicTropicalDifferenceLayer(nn.Module):
    """
    A layer combining linear projection, a cyclic transformation (modulo or angle-based),
    and gated tropical-difference aggregation.
    1. Linearly projects the input vector.
    2. Applies a cyclic transformation (modulo operation or angle encoding).
    3. Applies GatedTropicalDifferenceAggregator to the transformed vector.
    """
    def __init__(self,
                 input_dim: int,
                 projection_dim: int = 32,
                 tau_tropical: float = 0.2,
                 learn_tau_tropical: bool = False,
                 modulus: float = 7.0,
                 use_ste_modulo: bool = True,
                 use_angle_encoding: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.modulus = modulus
        self.use_ste_modulo = use_ste_modulo
        self.use_angle_encoding = use_angle_encoding

        self.projection = nn.Linear(input_dim, projection_dim)

        # Determine the input dimension for the tropical part
        if self.use_angle_encoding:
            self.transformed_dim = projection_dim * 2
        else:
            self.transformed_dim = projection_dim

        self.gated_tropical_aggregator = GatedTropicalDifferenceAggregator(
            dim=self.transformed_dim,
            tau_tropical=tau_tropical,
            learn_tau_tropical=learn_tau_tropical
        )

    def _cyclic_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the chosen cyclic transformation."""
        if self.use_angle_encoding:
            # Transform to angles on a circle, preserves continuity near modulus boundary
            # theta values are in [0, 2*pi)
            theta = 2 * math.pi * (x / self.modulus) # No fmod here, let sin/cos handle periodicity
            return torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)
        else:
            # Standard modulo operation
            if self.use_ste_modulo:
                return ste_modulo(x, self.modulus)
            else:
                return torch.fmod(x, self.modulus)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor from GatedTropicalDifferenceAggregator.
            Shape: (batch_size, 2 * transformed_dim)
        """
        projected_x = self.projection(x) # (B, projection_dim)
        transformed_x = self._cyclic_transform(projected_x) # (B, transformed_dim)
        output = self.gated_tropical_aggregator(transformed_x) # (B, 2 * transformed_dim)
        return output

    @property
    def output_dim(self) -> int:
        # GatedTropicalDifferenceAggregator outputs 2 * its input_dim
        return 2 * self.transformed_dim
