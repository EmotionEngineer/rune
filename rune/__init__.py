# rune/__init__.py

from . import utils
from . import layers
from . import models
from . import interpret

__version__ = "0.0.2"

# Expose key classes at the top level of the package
from .layers import (
    TropicalDifferenceAggregator,
    GatedTropicalDifferenceAggregator,
    PairwiseDifferenceLayer,
    CyclicTropicalDifferenceLayer
)
from .models import (
    PairwiseDifferenceNet,
    GatedTropicalDifferenceNet,
    CyclicTropicalDifferenceNet,
    RUNEBlock,
    InterpretableRuneNet
)
from .interpret import (
    plot_linear_weights,
    plot_pairwise_difference_weights,
    plot_tropical_aggregator_params,
    plot_gated_aggregator_gates,
    plot_cyclic_layer_projection_weights,
    analyze_tropical_dominance,
    plot_feature_interaction_graph,
    plot_final_layer_contributions,
    trace_decision_path
)
from .utils import ste_modulo, STEModulo

__all__ = [
    # Layers
    "TropicalDifferenceAggregator",
    "GatedTropicalDifferenceAggregator",
    "PairwiseDifferenceLayer",
    "CyclicTropicalDifferenceLayer",
    # Models
    "PairwiseDifferenceNet",
    "GatedTropicalDifferenceNet",
    "CyclicTropicalDifferenceNet",
    "RUNEBlock",
    "InterpretableRuneNet",
    # Interpretation
    "plot_linear_weights",
    "plot_pairwise_difference_weights",
    "plot_tropical_aggregator_params",
    "plot_gated_aggregator_gates",
    "plot_cyclic_layer_projection_weights",
    "analyze_tropical_dominance",
    "plot_feature_interaction_graph",
    "plot_final_layer_contributions",
    "trace_decision_path",
    # Utils
    "ste_modulo",
    "STEModulo",
    # Submodules
    "utils",
    "layers",
    "models",
    "interpret"
]
