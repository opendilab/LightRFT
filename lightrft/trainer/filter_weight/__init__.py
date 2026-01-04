"""
Filter and Weight Module

Unified interface for sample filtering and loss weighting in RLHF.

This module provides a three-layer architecture for managing sample filtering
and loss weighting:

1. **Metrics Layer**: Compute sample-level metrics (entropy, difficulty, staleness, etc.)
2. **Filter Layer**: Filter samples based on metrics (keep/discard decisions)
3. **Weight Layer**: Compute per-sample loss weights based on metrics

The FilterWeightManager provides a high-level API to orchestrate these components.

Example usage:
    ```python
    from lightrft.trainer.filter_weight import (
        FilterWeightManager,
        ResponseLengthFilter,
        DifficultyWeighting,
    )

    # Create manager
    manager = FilterWeightManager(
        filters=[ResponseLengthFilter(max_length=1024)],
        weights=[(DifficultyWeighting(mode="prioritized"), 1.0)],
        enable_metrics={"difficulty": True}
    )

    # Compute metrics
    metrics = manager.compute_metrics(outputs)

    # Apply to experiences
    experiences, weights = manager.apply_to_experiences(experiences, metrics)
    ```

Author: LightRLHF Team
"""

# Metrics
from .metrics import (
    SampleMetrics,
    MetricsComputer,
)

# Filters
from .filters import (
    SampleFilter,
    ResponseLengthFilter,
    RewardValueFilter,
    EntropyFilter,
    DifficultyFilter,
    CompositeFilter,
    PercentileFilter,
)

# Weights
from .weights import (
    LossWeighting,
    ResponseLengthWeighting,
    EntropyWeighting,
    DifficultyWeighting,
    StalenessWeighting,
    RewardMagnitudeWeighting,
    CompositeWeighting,
    UniformWeighting,
)

# Manager
from .manager import (
    FilterWeightManager,
    FilterWeightManagerBuilder,
)


__all__ = [
    # ========== Metrics ==========
    "SampleMetrics",
    "MetricsComputer",

    # ========== Filters ==========
    "SampleFilter",
    "ResponseLengthFilter",
    "RewardValueFilter",
    "EntropyFilter",
    "DifficultyFilter",
    "CompositeFilter",
    "PercentileFilter",

    # ========== Weights ==========
    "LossWeighting",
    "ResponseLengthWeighting",
    "EntropyWeighting",
    "DifficultyWeighting",
    "StalenessWeighting",
    "RewardMagnitudeWeighting",
    "CompositeWeighting",
    "UniformWeighting",

    # ========== Manager ==========
    "FilterWeightManager",
    "FilterWeightManagerBuilder",
]


# Version info
__version__ = "1.0.0"
__author__ = "LightRLHF Team"


def get_version():
    """Get the version of the filter_weight module."""
    return __version__


# Quick access functions for common use cases
def create_length_filter(max_length: int = 1024, **kwargs):
    """
    Quick function to create a response length filter.

    Args:
        max_length: Maximum response length
        **kwargs: Additional arguments for ResponseLengthFilter

    Returns:
        ResponseLengthFilter instance
    """
    return ResponseLengthFilter(max_length=max_length, **kwargs)


def create_difficulty_weighting(mode: str = "prioritized", alpha: float = 0.6, **kwargs):
    """
    Quick function to create difficulty weighting.

    Args:
        mode: Weighting mode ("prioritized" or "curriculum")
        alpha: Prioritization exponent
        **kwargs: Additional arguments for DifficultyWeighting

    Returns:
        DifficultyWeighting instance
    """
    return DifficultyWeighting(mode=mode, alpha=alpha, **kwargs)


def create_manager_from_args(args, packing_samples: bool = False):
    """
    Quick function to create FilterWeightManager from training arguments.

    Args:
        args: Training arguments
        packing_samples: Whether samples are packed

    Returns:
        FilterWeightManager instance
    """
    return FilterWeightManagerBuilder.from_args(args, packing_samples)


# Add convenience imports at module level
__all__.extend([
    "get_version",
    "create_length_filter",
    "create_difficulty_weighting",
    "create_manager_from_args",
])



