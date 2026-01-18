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


# Quick access functions for common use cases
def create_length_filter(max_length: int = 1024, **kwargs):
    """
    Quick function to create a response length filter.

    :param max_length: Maximum response length
    :type max_length: int
    :param kwargs: Additional arguments for ResponseLengthFilter
    :type kwargs: dict
    :return: ResponseLengthFilter instance
    :rtype: ResponseLengthFilter
    """
    return ResponseLengthFilter(max_length=max_length, **kwargs)


def create_difficulty_weighting(mode: str = "prioritized", alpha: float = 0.6, **kwargs):
    """
    Quick function to create difficulty weighting.

    :param mode: Weighting mode ("prioritized" or "curriculum")
    :type mode: str
    :param alpha: Prioritization exponent
    :type alpha: float
    :param kwargs: Additional arguments for DifficultyWeighting
    :type kwargs: dict
    :return: DifficultyWeighting instance
    :rtype: DifficultyWeighting
    """
    return DifficultyWeighting(mode=mode, alpha=alpha, **kwargs)


def create_manager_from_args(args, packing_samples: bool = False):
    """
    Quick function to create FilterWeightManager from training arguments.

    :param args: Training arguments
    :type args: Any
    :param packing_samples: Whether samples are packed
    :type packing_samples: bool
    :return: FilterWeightManager instance
    :rtype: FilterWeightManager
    """
    return FilterWeightManagerBuilder.from_args(args, packing_samples)


# Add convenience imports at module level
__all__.extend([
    "create_length_filter",
    "create_difficulty_weighting",
    "create_manager_from_args",
])
