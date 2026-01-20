"""
Unified Filter-Weight Manager

Provides high-level API for managing sample filtering and loss weighting.
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

    :param metrics_computer: Custom metrics computer (if None, creates default)
    :type metrics_computer: Optional[MetricsComputer]
    :param filters: List of filters to apply
    :type filters: Optional[List[SampleFilter]]
    :param weights: List of (weighting, coefficient) pairs
    :type weights: Optional[List[Tuple[LossWeighting, float]]]
    :param enable_metrics: Dict of metric names to enable
    :type enable_metrics: Optional[Dict[str, bool]]
    :param packing_samples: Whether samples are packed
    :type packing_samples: bool
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

        :param metrics_computer: Custom metrics computer (default: MetricsComputer())
        :type metrics_computer: Optional[MetricsComputer]
        :param filters: List of filters to apply (default: [])
        :type filters: Optional[List[SampleFilter]]
        :param weights: List of (weighting, coefficient) tuples (default: [])
        :type weights: Optional[List[Tuple[LossWeighting, float]]]
        :param enable_metrics: Dict specifying which metrics to compute (default: {})
        :type enable_metrics: Optional[Dict[str, bool]]
        :param packing_samples: Whether samples are packed (affects metric computation)
        :type packing_samples: bool
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

        :param outputs: List of sample outputs from model inference (_SamplesOutput objects)
        :type outputs: List
        :param current_step: Current training step (required for staleness computation)
        :type current_step: Optional[int]
        :return: SampleMetrics with computed metrics
        :rtype: SampleMetrics
        """
        return self.metrics_computer.compute_all_metrics(outputs, self.enable_metrics, current_step)

    def apply_filters(
        self,
        metrics: SampleMetrics,
        experiences: List  # List[ExperienceVL]
    ) -> torch.Tensor:
        """
        Apply all configured filters.

        :param metrics: Computed sample metrics
        :type metrics: SampleMetrics
        :param experiences: List of Experience/ExperienceVL objects
        :type experiences: List
        :return: BoolTensor (total_samples,) indicating which samples to keep (True = keep, False = filter out)
        :rtype: torch.Tensor
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

        :param metrics: Computed sample metrics
        :type metrics: SampleMetrics
        :param experiences: List of Experience/ExperienceVL objects
        :type experiences: List
        :return: FloatTensor (total_samples,) with loss weights
        :rtype: torch.Tensor
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
        apply_filter_to_weights: bool = True,
        handle_distributed: bool = True,
        strategy=None
    ) -> Tuple[List, torch.Tensor]:
        """
        Apply filtering and weighting to experiences.

        This method:
        1. Computes filter mask
        2. Handles distributed training edge cases (e.g., all samples filtered)
        3. Updates action_mask to exclude filtered samples (if apply_filter_to_mask=True)
        4. Optionally updates exp.info["reward"] for filtered samples
        5. Computes loss weights
        6. Zeros out weights for filtered samples (if apply_filter_to_weights=True)

        :param experiences: List of Experience/ExperienceVL objects to process
        :type experiences: List
        :param metrics: Computed sample metrics
        :type metrics: SampleMetrics
        :param apply_filter_to_mask: If True, update action_mask to exclude filtered samples
        :type apply_filter_to_mask: bool
        :param apply_filter_to_weights: If True, zero out weights for filtered samples
        :type apply_filter_to_weights: bool
        :param handle_distributed: If True, handle distributed training edge cases
        :type handle_distributed: bool
        :param strategy: Strategy object for logging (optional)
        :type strategy: Optional[Any]
        :return: Modified experiences and per-sample weights
        :rtype: Tuple[List, torch.Tensor]
        """
        import torch.distributed as dist

        # Apply filters
        keep_mask = self.apply_filters(metrics, experiences)

        # Handle distributed training edge cases
        if handle_distributed:
            is_distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

            if is_distributed and keep_mask.sum().item() == 0:
                # All samples filtered on this rank - reset mask to avoid NCCL issues
                if strategy is not None:
                    strategy.print(
                        "[Warning] FilterWeightManager: No sample kept after filtering on this rank; "
                        "skipping filtering this step to maintain synchronization."
                    )
                else:
                    warnings.warn(
                        "FilterWeightManager: No sample kept after filtering on this rank; "
                        "skipping filtering this step."
                    )
                keep_mask = torch.ones_like(keep_mask, dtype=torch.bool)

        # Update action masks if requested
        if apply_filter_to_mask and keep_mask.sum().item() < keep_mask.numel():
            sample_idx = 0
            for exp in experiences:
                batch_size = len(exp.sequences)
                batch_mask = keep_mask[sample_idx:sample_idx + batch_size]

                # Zero out action_mask for filtered samples
                # This effectively removes them from loss computation
                if exp.action_mask is not None:
                    exp.action_mask = exp.action_mask & batch_mask.unsqueeze(-1).to(exp.action_mask.device)

                # Also mask rewards in exp.info for consistency with legacy behavior
                if "reward" in exp.info:
                    exp_rewards = exp.info["reward"]
                    exp.info["reward"] = exp_rewards * batch_mask.to(exp_rewards.device).float()

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

        :param metrics: Computed sample metrics
        :type metrics: SampleMetrics
        :param experiences: List of experiences
        :type experiences: List
        :return: Dict with statistics (total_samples, filtered_samples, filter_rate, kept_samples)
        :rtype: Dict[str, float]
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

        :param metrics: Computed sample metrics
        :type metrics: SampleMetrics
        :param experiences: List of experiences
        :type experiences: List
        :return: Dict with statistics (weight_mean, weight_std, weight_min, weight_max)
        :rtype: Dict[str, float]
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

        :param metrics: Computed sample metrics
        :type metrics: SampleMetrics
        :param experiences: List of experiences
        :type experiences: List
        :param logger: Logger object (if None, uses print)
        :type logger: Optional[Any]
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

        :param args: Training arguments object with filter/weight configuration
        :type args: Any
        :param packing_samples: Whether samples are packed
        :type packing_samples: bool
        :return: FilterWeightManager configured according to args
        :rtype: FilterWeightManager
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
        if args.overlong_buffer:
            expected_len = args.max_new_tokens - args.overlong_buffer_len
            buffer_len = args.overlong_buffer_len
            filters.append(ResponseLengthFilter(expected_length=expected_len, buffer_length=buffer_len))

        # Reward value filter (for dynamic sampling)
        if args.dynamic_sampling and args.advantage_estimator == "group_norm":
            filters.append(RewardValueFilter(n_samples_per_prompt=args.n_samples_per_prompt))

        # Entropy filter
        if args.enable_entropy_filter:
            filters.append(EntropyFilter(min_entropy=args.min_entropy, max_entropy=args.max_entropy))

        # Build weights
        weights = []

        # Response length weighting
        if args.enable_length_weighting:
            weight = ResponseLengthWeighting(mode=args.length_weight_mode, normalize=True)
            coef = args.length_weight_coef
            weights.append((weight, coef))

        # Entropy weighting
        if args.enable_entropy_weighting:
            weight = EntropyWeighting(
                mode=args.entropy_weight_mode, temperature=args.entropy_weight_temperature, normalize=True
            )
            coef = args.entropy_weight_coef
            weights.append((weight, coef))

        # Difficulty weighting
        if args.enable_difficulty_weighting:
            weight = DifficultyWeighting(mode=args.difficulty_weight_mode, alpha=args.difficulty_alpha, normalize=True)
            coef = args.difficulty_weight_coef
            weights.append((weight, coef))

        # Staleness weighting
        if args.enable_staleness_weighting:
            weight = StalenessWeighting(decay_factor=args.staleness_decay_factor, normalize=True)
            coef = args.staleness_weight_coef
            weights.append((weight, coef))

        # Build enable_metrics dict
        enable_metrics = {
            "entropy": args.compute_entropy or args.enable_entropy_filter or args.enable_entropy_weighting,
            "difficulty": args.enable_difficulty_weighting,
            "difficulty_mode": args.difficulty_mode,
            "staleness": args.enable_staleness_weighting,
            "staleness_mode": args.staleness_mode,
        }

        return FilterWeightManager(
            filters=filters, weights=weights, enable_metrics=enable_metrics, packing_samples=packing_samples
        )
