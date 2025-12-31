"""
Sample Filtering Module

Provides unified interface for filtering samples based on various criteria.

Author: LightRLHF Team
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import torch
import warnings


class SampleFilter(ABC):
    """
    Base class for sample filters.

    Filters determine which samples should be kept for training. They return
    a boolean mask where True indicates the sample should be kept.
    """

    @abstractmethod
    def filter(
        self,
        metrics,  # SampleMetrics
        experiences: List  # List[ExperienceVL]
    ) -> torch.Tensor:
        """
        Compute filter mask.

        Args:
            metrics: SampleMetrics containing computed metrics
            experiences: List of Experience/ExperienceVL objects

        Returns:
            mask: BoolTensor (total_samples,) where True = keep, False = filter out
        """
        pass


class ResponseLengthFilter(SampleFilter):
    """
    Filter samples based on response length.

    This filter can enforce minimum/maximum length constraints or use a buffer-based
    approach (e.g., expected_length ± buffer_length).
    """

    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        expected_length: Optional[int] = None,
        buffer_length: Optional[int] = None
    ):
        """
        Initialize response length filter.

        Args:
            min_length: Minimum allowed response length (inclusive)
            max_length: Maximum allowed response length (inclusive)
            expected_length: Expected response length (for buffer-based filtering)
            buffer_length: Buffer around expected length (filters if length > expected + buffer)
        """
        self.min_length = min_length
        self.max_length = max_length
        self.expected_length = expected_length
        self.buffer_length = buffer_length

        # Validation
        if expected_length is not None and buffer_length is None:
            warnings.warn(
                "expected_length specified but buffer_length is None. "
                "Buffer-based filtering will not be applied."
            )

    def filter(self, metrics, experiences):
        """
        Filter based on length constraints.

        Args:
            metrics: SampleMetrics with response_length
            experiences: List of experiences (unused but kept for interface consistency)

        Returns:
            mask: BoolTensor indicating which samples to keep
        """
        lengths = metrics.response_length
        mask = torch.ones(len(lengths), dtype=torch.bool, device=lengths.device)

        # Apply min length
        if self.min_length is not None:
            mask &= (lengths >= self.min_length)

        # Apply max length
        if self.max_length is not None:
            mask &= (lengths <= self.max_length)

        # Apply buffer-based filtering
        if self.expected_length is not None and self.buffer_length is not None:
            max_allowed = self.expected_length + self.buffer_length
            mask &= (lengths <= max_allowed)

        return mask


class RewardValueFilter(SampleFilter):
    """
    Filter samples with degenerate reward values (DAPO dynamic sampling).

    This filter detects and removes groups of samples where all rewards are identical
    (e.g., all 0s or all 1s), which provides no learning signal.
    """

    def __init__(
        self,
        filter_all_zeros: bool = True,
        filter_all_ones: bool = True,
        n_samples_per_prompt: int = 1,
        group_size: Optional[int] = None,
        tolerance: float = 1e-6
    ):
        """
        Initialize reward value filter.

        Args:
            filter_all_zeros: Filter groups where all rewards are 0
            filter_all_ones: Filter groups where all rewards are 1
            n_samples_per_prompt: Number of samples per prompt (for grouping)
            group_size: Custom group size (if None, uses n_samples_per_prompt)
            tolerance: Tolerance for comparing reward values
        """
        self.filter_all_zeros = filter_all_zeros
        self.filter_all_ones = filter_all_ones
        self.group_size = group_size or n_samples_per_prompt
        self.tolerance = tolerance

    def filter(self, metrics, experiences):
        """
        Filter groups with degenerate rewards.

        Args:
            metrics: SampleMetrics with reward_value
            experiences: List of experiences (unused)

        Returns:
            mask: BoolTensor indicating which samples to keep
        """
        if metrics.reward_value is None:
            # No rewards available, keep all samples
            total_samples = sum(len(exp.sequences) for exp in experiences)
            return torch.ones(total_samples, dtype=torch.bool, device='cuda')

        rewards = metrics.reward_value
        mask = torch.ones(len(rewards), dtype=torch.bool, device=rewards.device)

        # Check if can evenly divide into groups
        if len(rewards) % self.group_size != 0:
            warnings.warn(
                f"Number of samples ({len(rewards)}) not divisible by group_size ({self.group_size}). "
                f"Skipping reward value filtering."
            )
            return mask

        # Reshape into groups
        grouped_rewards = rewards.reshape(-1, self.group_size)

        # Check for all-zero or all-one groups
        group_mask = torch.ones(len(grouped_rewards), dtype=torch.bool, device=rewards.device)

        if self.filter_all_zeros:
            all_zeros = torch.all(torch.abs(grouped_rewards) < self.tolerance, dim=1)
            group_mask &= ~all_zeros

        if self.filter_all_ones:
            all_ones = torch.all(torch.abs(grouped_rewards - 1.0) < self.tolerance, dim=1)
            group_mask &= ~all_ones

        # Expand group mask back to sample level
        mask = group_mask.repeat_interleave(self.group_size)

        return mask


class EntropyFilter(SampleFilter):
    """
    Filter samples based on policy entropy.

    Entropy measures the uncertainty/diversity of the policy. Low entropy indicates
    confident/deterministic generation, high entropy indicates uncertain/exploratory generation.
    """

    def __init__(
        self,
        min_entropy: Optional[float] = None,
        max_entropy: Optional[float] = None
    ):
        """
        Initialize entropy filter.

        Args:
            min_entropy: Minimum entropy threshold (filter out low entropy)
            max_entropy: Maximum entropy threshold (filter out high entropy)
        """
        self.min_entropy = min_entropy
        self.max_entropy = max_entropy

    def filter(self, metrics, experiences):
        """
        Filter based on entropy.

        Args:
            metrics: SampleMetrics with entropy
            experiences: List of experiences

        Returns:
            mask: BoolTensor indicating which samples to keep
        """
        if metrics.entropy is None:
            # Entropy not computed, keep all samples
            total_samples = sum(len(exp.sequences) for exp in experiences)
            device = experiences[0].sequences.device if experiences else 'cuda'
            return torch.ones(total_samples, dtype=torch.bool, device=device)

        entropy = metrics.entropy
        mask = torch.ones(len(entropy), dtype=torch.bool, device=entropy.device)

        # Apply min entropy threshold
        if self.min_entropy is not None:
            mask &= (entropy >= self.min_entropy)

        # Apply max entropy threshold
        if self.max_entropy is not None:
            mask &= (entropy <= self.max_entropy)

        return mask


class DifficultyFilter(SampleFilter):
    """
    Filter samples based on difficulty scores.

    This can be used to implement curriculum learning (filter out hard samples early)
    or focus training (filter out easy samples).
    """

    def __init__(
        self,
        min_difficulty: Optional[float] = None,
        max_difficulty: Optional[float] = None,
        mode: str = "absolute"  # "absolute" or "percentile"
    ):
        """
        Initialize difficulty filter.

        Args:
            min_difficulty: Minimum difficulty threshold
            max_difficulty: Maximum difficulty threshold
            mode: Filtering mode
                - "absolute": Use absolute threshold values
                - "percentile": Interpret thresholds as percentiles (0-100)
        """
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.mode = mode

    def filter(self, metrics, experiences):
        """
        Filter based on difficulty.

        Args:
            metrics: SampleMetrics with difficulty
            experiences: List of experiences

        Returns:
            mask: BoolTensor indicating which samples to keep
        """
        if metrics.difficulty is None:
            # Difficulty not computed, keep all samples
            total_samples = sum(len(exp.sequences) for exp in experiences)
            device = experiences[0].sequences.device if experiences else 'cuda'
            return torch.ones(total_samples, dtype=torch.bool, device=device)

        difficulty = metrics.difficulty
        mask = torch.ones(len(difficulty), dtype=torch.bool, device=difficulty.device)

        # Handle percentile mode
        if self.mode == "percentile":
            if self.min_difficulty is not None:
                threshold = torch.quantile(difficulty, self.min_difficulty / 100.0)
                mask &= (difficulty >= threshold)

            if self.max_difficulty is not None:
                threshold = torch.quantile(difficulty, self.max_difficulty / 100.0)
                mask &= (difficulty <= threshold)

        else:  # absolute mode
            if self.min_difficulty is not None:
                mask &= (difficulty >= self.min_difficulty)

            if self.max_difficulty is not None:
                mask &= (difficulty <= self.max_difficulty)

        return mask


class CompositeFilter(SampleFilter):
    """
    Combine multiple filters with AND/OR logic.

    This allows building complex filtering logic by composing simple filters.
    """

    def __init__(self, filters: List[SampleFilter], logic: str = "AND"):
        """
        Initialize composite filter.

        Args:
            filters: List of filters to combine
            logic: Combination logic
                - "AND": All filters must pass (intersection)
                - "OR": Any filter must pass (union)
        """
        self.filters = filters
        self.logic = logic.upper()

        if self.logic not in ["AND", "OR"]:
            raise ValueError(f"Invalid logic: {logic}. Must be 'AND' or 'OR'")

    def filter(self, metrics, experiences):
        """
        Combine filters according to logic.

        Args:
            metrics: SampleMetrics
            experiences: List of experiences

        Returns:
            mask: Combined BoolTensor
        """
        if not self.filters:
            # No filters, keep all samples
            total_samples = sum(len(exp.sequences) for exp in experiences)
            device = experiences[0].sequences.device if experiences else 'cuda'
            return torch.ones(total_samples, dtype=torch.bool, device=device)

        # Apply first filter
        combined_mask = self.filters[0].filter(metrics, experiences)

        # Combine with rest
        for f in self.filters[1:]:
            mask = f.filter(metrics, experiences)

            if self.logic == "AND":
                combined_mask &= mask
            else:  # OR
                combined_mask |= mask

        return combined_mask


class PercentileFilter(SampleFilter):
    """
    Filter samples based on percentile ranking of a metric.

    This is useful for keeping top-k% or bottom-k% samples according to some metric.
    """

    def __init__(
        self,
        metric_name: str,
        top_percentile: Optional[float] = None,
        bottom_percentile: Optional[float] = None
    ):
        """
        Initialize percentile filter.

        Args:
            metric_name: Name of metric in SampleMetrics (e.g., "entropy", "difficulty")
            top_percentile: Keep top X% (e.g., 20 = keep top 20%)
            bottom_percentile: Keep bottom X% (e.g., 20 = keep bottom 20%)
        """
        self.metric_name = metric_name
        self.top_percentile = top_percentile
        self.bottom_percentile = bottom_percentile

        if top_percentile is None and bottom_percentile is None:
            raise ValueError("At least one of top_percentile or bottom_percentile must be specified")

    def filter(self, metrics, experiences):
        """
        Filter based on percentile ranking.

        Args:
            metrics: SampleMetrics
            experiences: List of experiences

        Returns:
            mask: BoolTensor indicating which samples to keep
        """
        # Get metric value
        metric_value = getattr(metrics, self.metric_name, None)

        if metric_value is None:
            # Metric not available, keep all samples
            total_samples = sum(len(exp.sequences) for exp in experiences)
            device = experiences[0].sequences.device if experiences else 'cuda'
            return torch.ones(total_samples, dtype=torch.bool, device=device)

        mask = torch.zeros(len(metric_value), dtype=torch.bool, device=metric_value.device)

        # Keep top percentile (highest values)
        if self.top_percentile is not None:
            threshold = torch.quantile(metric_value, 1 - self.top_percentile / 100.0)
            mask |= (metric_value >= threshold)

        # Keep bottom percentile (lowest values)
        if self.bottom_percentile is not None:
            threshold = torch.quantile(metric_value, self.bottom_percentile / 100.0)
            mask |= (metric_value <= threshold)

        return mask

