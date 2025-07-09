# rune/layers.py

import torch
import torch.nn as nn
import math
from .utils import ste_modulo

class TropicalDifferenceAggregator(nn.Module):
    """
    TropicalDifferenceAggregator: Aggregates weighted pairwise differences (x_i - x_j)
    using a "soft" max-plus operation (log-sum-exp).
    """
    def __init__(self,
                 dim: int,
                 tau: float = 0.1,
                 learn_tau: bool = False):
        super().__init__()
        self.dim = dim
        self.weights = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(dim, dim))

        if learn_tau:
            self.log_tau = nn.Parameter(torch.log(torch.tensor(tau, dtype=torch.float32)))
        else:
            # Not registered as buffer to allow modification during annealing
            self.log_tau = torch.log(torch.tensor(tau, dtype=torch.float32))

        self.register_buffer('mask', ~torch.eye(dim, dtype=torch.bool))

    @property
    def tau(self) -> torch.Tensor:
        """The temperature parameter for log-sum-exp."""
        return self.log_tau.exp()

    def set_tau(self, new_tau: float):
        """Sets a new temperature tau. Used for annealing."""
        self.log_tau = torch.log(torch.tensor(new_tau, dtype=torch.float32)).to(self.weights.device)

    def get_regularization_loss(self) -> torch.Tensor:
        """Calculates the L1 regularization loss for the interaction weights."""
        return torch.norm(self.weights, p=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        assert D == self.dim, f"Input dimension mismatch: expected {self.dim}, got {D}"
        x_diff = x.unsqueeze(2) - x.unsqueeze(1)
        weighted_diff = x_diff * self.weights.unsqueeze(0) + self.bias.unsqueeze(0)
        weighted_diff = weighted_diff.masked_fill(~self.mask.unsqueeze(0), float('-inf'))
        y = self.tau * torch.logsumexp(weighted_diff / self.tau, dim=2)
        return y

class GatedTropicalDifferenceAggregator(nn.Module):
    """
    Combines tropical aggregation (max-like) and mean aggregation of pairwise
    differences using a learnable gate.
    """
    def __init__(self,
                 dim: int,
                 tau_tropical: float = 0.2,
                 learn_tau_tropical: bool = False):
        super().__init__()
        self.dim = dim
        self.tropical_agg = TropicalDifferenceAggregator(dim, tau=tau_tropical, learn_tau=learn_tau_tropical)
        self.mean_weights = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.gate_params = nn.Parameter(torch.zeros(dim))
        self.register_buffer('mask', ~torch.eye(dim, dtype=torch.bool))

    def set_tau(self, new_tau: float):
        """Sets a new temperature tau for the internal tropical aggregator."""
        self.tropical_agg.set_tau(new_tau)
        
    def get_regularization_loss(self) -> torch.Tensor:
        """Calculates L1 loss for all interaction weights in the layer."""
        return self.tropical_agg.get_regularization_loss() + torch.norm(self.mean_weights, p=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        assert D == self.dim, f"Input dimension mismatch: expected {self.dim}, got {D}"
        y_tropical = self.tropical_agg(x)
        x_diff = x.unsqueeze(2) - x.unsqueeze(1)
        weighted_mean_diff = x_diff * self.mean_weights.unsqueeze(0)
        weighted_mean_diff = weighted_mean_diff.masked_fill(~self.mask.unsqueeze(0), 0.0)
        num_effective_elements = self.mask.sum(dim=1, dtype=torch.float32).clamp(min=1.0)
        y_mean = weighted_mean_diff.sum(dim=2) / num_effective_elements.unsqueeze(0)
        g = torch.sigmoid(self.gate_params).unsqueeze(0)
        y_combined = g * y_tropical + (1 - g) * y_mean
        return torch.cat([x, y_combined], dim=1)

    @property
    def gate_values(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_params.detach())

class PairwiseDifferenceLayer(nn.Module):
    """
    Computes all unique weighted pairwise differences (x_i - x_j).
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        if input_dim < 2:
            self.indices = torch.empty(0, 2, dtype=torch.long)
        else:
            self.indices = torch.combinations(torch.arange(input_dim), r=2)
        self.register_buffer('feature_indices', self.indices)
        self.num_diff_features = self.feature_indices.shape[0]
        if self.num_diff_features > 0:
            self.weights = nn.Parameter(torch.ones(self.num_diff_features))
        else:
            self.register_parameter('weights', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_diff_features == 0:
            return x
        x_select_0 = x[:, self.feature_indices[:, 0]]
        x_select_1 = x[:, self.feature_indices[:, 1]]
        differences = (x_select_0 - x_select_1)
        weighted_differences = differences * self.weights.unsqueeze(0)
        return torch.cat([x, weighted_differences], dim=1)

    @property
    def output_dim(self) -> int:
        return self.input_dim + self.num_diff_features

class CyclicTropicalDifferenceLayer(nn.Module):
    """
    A layer combining linear projection, a cyclic transformation,
    and gated tropical-difference aggregation.
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
        if self.use_angle_encoding:
            theta = 2 * math.pi * (x / self.modulus)
            return torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)
        else:
            return ste_modulo(x, self.modulus) if self.use_ste_modulo else torch.fmod(x, self.modulus)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected_x = self.projection(x)
        transformed_x = self._cyclic_transform(projected_x)
        output = self.gated_tropical_aggregator(transformed_x)
        return output

    @property
    def output_dim(self) -> int:
        return 2 * self.transformed_dim

class PrototypeLayer(nn.Module):
    """
    A layer that computes L2 distances from an input vector to a set of
    learnable prototypes, representing "ideal cases".
    """
    def __init__(self, input_dim: int, num_prototypes: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_prototypes = num_prototypes
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, input_dim))

    @property
    def output_dim(self) -> int:
        return self.num_prototypes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates batch of distances between inputs and prototypes.
        Args:
            x: Input tensor of shape (batch_size, input_dim).
        Returns:
            Tensor of distances of shape (batch_size, num_prototypes).
        """
        # x: (B, D_in), self.prototypes: (P, D_in)
        diffs = x.unsqueeze(1) - self.prototypes.unsqueeze(0) # (B, P, D_in)
        distances = torch.norm(diffs, p=2, dim=-1) # (B, P)
        return distances