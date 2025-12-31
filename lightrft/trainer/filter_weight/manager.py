"""
Unified Filter-Weight Manager

Provides high-level API for managing sample filtering and loss weighting.

Author: LightRLHF Team
"""

from typing import List, Optional, Dict, Tuple
import torch
import warnings

from .metrics import MetricsComputer, SampleMetrics
from .filters import SampleFilter, CompositeFilter
from .weights import LossWeighting, CompositeWeighting, UniformWeighting


class FilterWeightManager:
    """
    Unified manager for sample filtering and loss weighting.

    This class orchestrates the entire pipeline of:
    1. Computing sample metrics
    2. Applying filters to determine which samples to keep
    3. Computing loss weights for each sample
    4. Applying filters and weights to experiences

    Example usage:
        ```python
        manager = FilterWeightManager(
            filters=[
                ResponseLengthFilter(max_length=1024),
                RewardValueFilter(n_samples_per_prompt=4),
            ],
            weights=[
                (ResponseLengthWeighting(mode="inverse"), 0.5),
                (DifficultyWeighting(mode="prioritized"), 0.5),
            ],
            enable_metrics={
                "entropy": True,
                "difficulty": True,
                "difficulty_mode": "td_error",
            }
        )

        # In make_experience_list:
        metrics = manager.compute_metrics(outputs)
        experiences, weights = manager.apply_to_experiences(experiences, metrics)
        ```

    Args:
        metrics_computer: Custom metrics computer (if None, creates default)
        filters: List of filters to apply
        weights: List of (weighting, coefficient) pairs
        enable_metrics: Dict of metric names to enable
        packing_samples: Whether samples are packed
    """

    def __init__(
        self,
        metrics_computer: Optional[MetricsComputer] = None,
        filters: Optional[List[SampleFilter]] = None,
        weights: Optional[List[Tuple[LossWeighting, float]]] = None,
        enable_metrics: Optional[Dict[str, bool]] = None,
        packing_samples: bool = False
    ):
        """
        Initialize filter-weight manager.

        Args:
            metrics_computer: Custom metrics computer (default: MetricsComputer())
            filters: List of filters to apply (default: [])
            weights: List of (weighting, coefficient) tuples (default: [])
            enable_metrics: Dict specifying which metrics to compute (default: {})
            packing_samples: Whether samples are packed (affects metric computation)
        """
        self.metrics_computer = metrics_computer or MetricsComputer(packing_samples)
        self.filters = filters or []
        self.weights = weights or []
        self.enable_metrics = enable_metrics or {}
        self.packing_samples = packing_samples

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate configuration and emit warnings if needed."""
        # Check if any filter/weight requires metrics that are not enabled
        required_metrics = set()

        # Map filter/weight types to required metrics
        filter_metric_map = {
            "EntropyFilter": "entropy",
            "DifficultyFilter": "difficulty",
        }
        weight_metric_map = {
            "EntropyWeighting": "entropy",
            "DifficultyWeighting": "difficulty",
            "StalenessWeighting": "staleness",
        }

        # Check filters
        for f in self.filters:
            filter_type = type(f).__name__
            if filter_type in filter_metric_map:
                metric_name = filter_metric_map[filter_type]
                if not self.enable_metrics.get(metric_name, False):
                    warnings.warn(
                        f"{filter_type} requires '{metric_name}' metric but it is not enabled. "
                        f"Filter may not work correctly."
                    )
                required_metrics.add(metric_name)

        # Check weights
        for w, _ in self.weights:
            weight_type = type(w).__name__
            if weight_type in weight_metric_map:
                metric_name = weight_metric_map[weight_type]
                if not self.enable_metrics.get(metric_name, False):
                    warnings.warn(
                        f"{weight_type} requires '{metric_name}' metric but it is not enabled. "
                        f"Weighting may not work correctly."
                    )
                required_metrics.add(metric_name)

    def compute_metrics(
        self,
        outputs: List,  # List[_SamplesOutput]
        current_step: Optional[int] = None
    ) -> SampleMetrics:
        """
        Compute all enabled metrics.

        Args:
            outputs: List of sample outputs from model inference (_SamplesOutput objects)
            current_step: Current training step (required for staleness computation)

        Returns:
            SampleMetrics with computed metrics
        """
        return self.metrics_computer.compute_all_metrics(
            outputs,
            self.enable_metrics,
            current_step
        )

    def apply_filters(
        self,
        metrics: SampleMetrics,
        experiences: List  # List[ExperienceVL]
    ) -> torch.Tensor:
        """
        Apply all configured filters.

        Args:
            metrics: Computed sample metrics
            experiences: List of Experience/ExperienceVL objects

        Returns:
            mask: BoolTensor (total_samples,) indicating which samples to keep
                  True = keep, False = filter out
        """
        if not self.filters:
            # No filters, keep all samples
            total_samples = sum(len(exp.sequences) for exp in experiences)
            device = experiences[0].sequences.device if experiences else 'cuda'
            return torch.ones(total_samples, dtype=torch.bool, device=device)

        # Combine all filters with AND logic
        composite = CompositeFilter(self.filters, logic="AND")
        return composite.filter(metrics, experiences)

    def compute_weights(
        self,
        metrics: SampleMetrics,
        experiences: List  # List[ExperienceVL]
    ) -> torch.Tensor:
        """
        Compute combined sample weights.

        Args:
            metrics: Computed sample metrics
            experiences: List of Experience/ExperienceVL objects

        Returns:
            weights: FloatTensor (total_samples,) with loss weights
        """
        if not self.weights:
            # No weighting, return uniform weights
            total_samples = sum(len(exp.sequences) for exp in experiences)
            device = experiences[0].sequences.device if experiences else 'cuda'
            return torch.ones(total_samples, device=device)

        # Combine all weightings
        composite = CompositeWeighting(self.weights, mode="weighted_sum")
        return composite.compute_weights(metrics, experiences)

    def apply_to_experiences(
        self,
        experiences: List,  # List[ExperienceVL]
        metrics: SampleMetrics,
        apply_filter_to_mask: bool = True,
        apply_filter_to_weights: bool = True
    ) -> Tuple[List, torch.Tensor]:
        """
        Apply filtering and weighting to experiences.

        This method:
        1. Computes filter mask
        2. Updates action_mask to exclude filtered samples (if apply_filter_to_mask=True)
        3. Computes loss weights
        4. Zeros out weights for filtered samples (if apply_filter_to_weights=True)

        Args:
            experiences: List of Experience/ExperienceVL objects to process
            metrics: Computed sample metrics
            apply_filter_to_mask: If True, update action_mask to exclude filtered samples
            apply_filter_to_weights: If True, zero out weights for filtered samples

        Returns:
            (experiences, weights): Modified experiences and per-sample weights
        """
        # Apply filters
        keep_mask = self.apply_filters(metrics, experiences)

        # Update action masks if requested
        if apply_filter_to_mask:
            sample_idx = 0
            for exp in experiences:
                batch_size = len(exp.sequences)
                batch_mask = keep_mask[sample_idx : sample_idx + batch_size]

                # Zero out action_mask for filtered samples
                # This effectively removes them from loss computation
                if exp.action_mask is not None:
                    exp.action_mask = exp.action_mask & batch_mask.unsqueeze(-1).to(exp.action_mask.device)

                sample_idx += batch_size

        # Compute weights
        weights = self.compute_weights(metrics, experiences)

        # Zero out weights for filtered samples if requested
        if apply_filter_to_weights:
            weights = weights * keep_mask.float()

        return experiences, weights

    def get_filter_stats(
        self,
        metrics: SampleMetrics,
        experiences: List  # List[ExperienceVL]
    ) -> Dict[str, float]:
        """
        Get statistics about filtering.

        Args:
            metrics: Computed sample metrics
            experiences: List of experiences

        Returns:
            Dict with statistics:
                - "total_samples": Total number of samples
                - "filtered_samples": Number of filtered samples
                - "filter_rate": Fraction of samples filtered
                - "kept_samples": Number of kept samples
        """
        total_samples = sum(len(exp.sequences) for exp in experiences)
        keep_mask = self.apply_filters(metrics, experiences)
        kept_samples = keep_mask.sum().item()
        filtered_samples = total_samples - kept_samples
        filter_rate = filtered_samples / total_samples if total_samples > 0 else 0.0

        return {
            "total_samples": total_samples,
            "filtered_samples": filtered_samples,
            "kept_samples": kept_samples,
            "filter_rate": filter_rate,
        }

    def get_weight_stats(
        self,
        metrics: SampleMetrics,
        experiences: List  # List[ExperienceVL]
    ) -> Dict[str, float]:
        """
        Get statistics about weighting.

        Args:
            metrics: Computed sample metrics
            experiences: List of experiences

        Returns:
            Dict with statistics:
                - "weight_mean": Mean weight
                - "weight_std": Standard deviation of weights
                - "weight_min": Minimum weight
                - "weight_max": Maximum weight
        """
        weights = self.compute_weights(metrics, experiences)

        return {
            "weight_mean": weights.mean().item(),
            "weight_std": weights.std().item(),
            "weight_min": weights.min().item(),
            "weight_max": weights.max().item(),
        }

    def log_stats(
        self,
        metrics: SampleMetrics,
        experiences: List,  # List[ExperienceVL]
        logger=None
    ):
        """
        Log filtering and weighting statistics.

        Args:
            metrics: Computed sample metrics
            experiences: List of experiences
            logger: Logger object (if None, uses print)
        """
        filter_stats = self.get_filter_stats(metrics, experiences)
        weight_stats = self.get_weight_stats(metrics, experiences)

        log_fn = logger.info if logger else print

        log_fn("=" * 60)
        log_fn("Filter & Weight Statistics")
        log_fn("=" * 60)

        # Filter stats
        log_fn(f"Total samples: {filter_stats['total_samples']}")
        log_fn(f"Kept samples: {filter_stats['kept_samples']}")
        log_fn(f"Filtered samples: {filter_stats['filtered_samples']}")
        log_fn(f"Filter rate: {filter_stats['filter_rate']:.2%}")

        # Weight stats
        log_fn(f"Weight mean: {weight_stats['weight_mean']:.4f}")
        log_fn(f"Weight std: {weight_stats['weight_std']:.4f}")
        log_fn(f"Weight range: [{weight_stats['weight_min']:.4f}, {weight_stats['weight_max']:.4f}]")

        log_fn("=" * 60)


class FilterWeightManagerBuilder:
    """
    Builder pattern for constructing FilterWeightManager from args.

    This provides a convenient way to construct a FilterWeightManager from
    training arguments.

    Example:
        ```python
        builder = FilterWeightManagerBuilder()
        manager = builder.from_args(args)
        ```
    """

    @staticmethod
    def from_args(args, packing_samples: bool = False) -> FilterWeightManager:
        """
        Build FilterWeightManager from training arguments.

        Args:
            args: Training arguments object
            packing_samples: Whether samples are packed

        Returns:
            FilterWeightManager configured according to args
        """
        from .filters import ResponseLengthFilter, RewardValueFilter, EntropyFilter
        from .weights import (
            ResponseLengthWeighting,
            EntropyWeighting,
            DifficultyWeighting,
            StalenessWeighting,
        )

        # Build filters
        filters = []

        # Response length filter (from overlong_buffer settings)
        if getattr(args, "overlong_buffer", False):
            expected_len = getattr(args, "max_new_tokens", 1024) - getattr(args, "overlong_buffer_len", 0)
            buffer_len = getattr(args, "overlong_buffer_len", 0)
            filters.append(ResponseLengthFilter(
                expected_length=expected_len,
                buffer_length=buffer_len
            ))

        # Reward value filter (for dynamic sampling)
        if getattr(args, "dynamic_sampling", False) and getattr(args, "advantage_estimator", "") == "group_norm":
            filters.append(RewardValueFilter(
                n_samples_per_prompt=getattr(args, "n_samples_per_prompt", 1)
            ))

        # Entropy filter
        if getattr(args, "enable_entropy_filter", False):
            filters.append(EntropyFilter(
                min_entropy=getattr(args, "min_entropy", None),
                max_entropy=getattr(args, "max_entropy", None)
            ))

        # Build weights
        weights = []

        # Response length weighting
        if getattr(args, "enable_length_weighting", False):
            weight = ResponseLengthWeighting(
                mode=getattr(args, "length_weight_mode", "inverse"),
                normalize=True
            )
            coef = getattr(args, "length_weight_coef", 1.0)
            weights.append((weight, coef))

        # Entropy weighting
        if getattr(args, "enable_entropy_weighting", False):
            weight = EntropyWeighting(
                mode=getattr(args, "entropy_weight_mode", "favor_high"),
                temperature=getattr(args, "entropy_weight_temperature", 1.0),
                normalize=True
            )
            coef = getattr(args, "entropy_weight_coef", 1.0)
            weights.append((weight, coef))

        # Difficulty weighting
        if getattr(args, "enable_difficulty_weighting", False):
            weight = DifficultyWeighting(
                mode=getattr(args, "difficulty_weight_mode", "prioritized"),
                alpha=getattr(args, "difficulty_alpha", 0.6),
                normalize=True
            )
            coef = getattr(args, "difficulty_weight_coef", 1.0)
            weights.append((weight, coef))

        # Staleness weighting
        if getattr(args, "enable_staleness_weighting", False):
            weight = StalenessWeighting(
                decay_factor=getattr(args, "staleness_decay_factor", 0.95),
                normalize=True
            )
            coef = getattr(args, "staleness_weight_coef", 1.0)
            weights.append((weight, coef))

        # Build enable_metrics dict
        enable_metrics = {
            "entropy": getattr(args, "compute_entropy", False) or getattr(args, "enable_entropy_filter", False) or getattr(args, "enable_entropy_weighting", False),
            "difficulty": getattr(args, "enable_difficulty_weighting", False),
            "difficulty_mode": getattr(args, "difficulty_mode", "td_error"),
            "staleness": getattr(args, "enable_staleness_weighting", False),
            "staleness_mode": getattr(args, "staleness_mode", "linear"),
        }

        return FilterWeightManager(
            filters=filters,
            weights=weights,
            enable_metrics=enable_metrics,
            packing_samples=packing_samples
        )

