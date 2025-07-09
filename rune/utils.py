# rune/utils.py

import torch
import math

class STEModulo(torch.autograd.Function):
    """
    Implements the modulo operation but allows gradients to pass through
    unchanged (Straight-Through Estimator), as if the operation were
    an identity function. This helps prevent zero gradients during training.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, modulus: float) -> torch.Tensor:
        """Forward pass - standard modulo operation."""
        return torch.fmod(x, modulus)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass - gradient passes through unchanged."""
        # Gradient for 'modulus' argument is not needed.
        return grad_output, None

def ste_modulo(x: torch.Tensor, modulus: float) -> torch.Tensor:
    """
    Applies the modulo operation with a Straight-Through Estimator for the gradient.
    """
    return STEModulo.apply(x, modulus)

def get_feature_names(num_features: int, default_prefix: str = "feat_") -> list[str]:
    """Generates default feature names if none are provided."""
    return [f"{default_prefix}{i}" for i in range(num_features)]