"""
Loss Weighting Module

Provides unified interface for computing sample-level loss weights.

Author: LightRLHF Team
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import torch
import warnings


class LossWeighting(ABC):
    """
    Base class for loss weighting.

    Weightings compute per-sample weights that modulate the contribution
    of each sample to the loss function.
    """

    @abstractmethod
    def compute_weights(
        self,
        metrics,  # SampleMetrics
        experiences: List  # List[ExperienceVL]
    ) -> torch.Tensor:
        """
        Compute per-sample weights.

        Args:
            metrics: SampleMetrics containing computed metrics
            experiences: List of Experience/ExperienceVL objects

        Returns:
            weights: FloatTensor (total_samples,) with loss weights
        """
        pass


class ResponseLengthWeighting(LossWeighting):
    """
    Weight samples by response length.

    This can be used to:
    - Give more weight to longer responses (mode="linear")
    - Give more weight to shorter responses (mode="inverse")
    - Balance weights by length (mode="sqrt", "log")
    """

    def __init__(
        self,
        mode: str = "linear",
        normalize: bool = True,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
        epsilon: float = 1e-6
    ):
        """
        Initialize response length weighting.

        Args:
            mode: Weighting mode
                - "linear": weight = length
                - "inverse": weight = 1/length
                - "sqrt": weight = sqrt(length)
                - "log": weight = log(1 + length)
            normalize: Whether to normalize weights to mean=1
            clip_min: Minimum weight value
            clip_max: Maximum weight value
            epsilon: Small constant to avoid division by zero
        """
        self.mode = mode
        self.normalize = normalize
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.epsilon = epsilon

        valid_modes = ["linear", "inverse", "sqrt", "log"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")

    def compute_weights(self, metrics, experiences):
        """
        Compute length-based weights.

        Args:
            metrics: SampleMetrics with response_length
            experiences: List of experiences (unused)

        Returns:
            weights: FloatTensor of per-sample weights
        """
        lengths = metrics.response_length.float()

        # Compute weights according to mode
        if self.mode == "linear":
            weights = lengths
        elif self.mode == "inverse":
            weights = 1.0 / (lengths + self.epsilon)
        elif self.mode == "sqrt":
            weights = torch.sqrt(lengths + self.epsilon)
        elif self.mode == "log":
            weights = torch.log(1.0 + lengths)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Clip if specified
        if self.clip_min is not None:
            weights = torch.clamp(weights, min=self.clip_min)
        if self.clip_max is not None:
            weights = torch.clamp(weights, max=self.clip_max)

        # Normalize to mean=1
        if self.normalize:
            weights = weights / (weights.mean() + self.epsilon)

        return weights


class EntropyWeighting(LossWeighting):
    """
    Weight samples by policy entropy.

    This can encourage exploration (favor high entropy) or exploitation (favor low entropy).
    """

    def __init__(
        self,
        mode: str = "favor_high",
        temperature: float = 1.0,
        normalize: bool = True,
        epsilon: float = 1e-6
    ):
        """
        Initialize entropy weighting.

        Args:
            mode: Weighting mode
                - "favor_high": Higher entropy → higher weight (encourage exploration)
                - "favor_low": Lower entropy → higher weight (encourage exploitation)
                - "linear": weight = entropy (linear scaling)
                - "inverse": weight = 1/entropy (inverse scaling)
            temperature: Temperature for softmax weighting (used in favor_high/favor_low modes)
            normalize: Normalize to mean=1
            epsilon: Small constant to avoid numerical issues
        """
        self.mode = mode
        self.temperature = temperature
        self.normalize = normalize
        self.epsilon = epsilon

        valid_modes = ["favor_high", "favor_low", "linear", "inverse"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")

    def compute_weights(self, metrics, experiences):
        """
        Compute entropy-based weights.

        Args:
            metrics: SampleMetrics with entropy
            experiences: List of experiences

        Returns:
            weights: FloatTensor of per-sample weights
        """
        if metrics.entropy is None:
            # Entropy not available, return uniform weights
            total_samples = sum(len(exp.sequences) for exp in experiences)
            device = experiences[0].sequences.device if experiences else 'cuda'
            return torch.ones(total_samples, device=device)

        entropy = metrics.entropy

        # Compute weights according to mode
        if self.mode == "favor_high":
            # Softmax over entropy (higher entropy → higher weight)
            weights = torch.softmax(entropy / self.temperature, dim=0) * len(entropy)
        elif self.mode == "favor_low":
            # Inverse softmax (lower entropy → higher weight)
            weights = torch.softmax(-entropy / self.temperature, dim=0) * len(entropy)
        elif self.mode == "linear":
            weights = entropy
        elif self.mode == "inverse":
            weights = 1.0 / (entropy + self.epsilon)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Normalize to mean=1
        if self.normalize:
            weights = weights / (weights.mean() + self.epsilon)

        return weights


class DifficultyWeighting(LossWeighting):
    """
    Weight samples by difficulty.

    This implements prioritized experience replay (PER) style weighting or
    curriculum learning approaches.
    """

    def __init__(
        self,
        mode: str = "prioritized",
        alpha: float = 0.6,
        normalize: bool = True,
        epsilon: float = 1e-6
    ):
        """
        Initialize difficulty weighting.

        Args:
            mode: Weighting mode
                - "prioritized": Prioritized experience replay (difficulty^alpha)
                - "curriculum": Curriculum learning (favor easier samples)
                - "linear": weight = difficulty
                - "inverse": weight = 1/difficulty
            alpha: Exponent for prioritization (typical range: 0.4-0.8)
            normalize: Normalize to mean=1
            epsilon: Small constant to avoid numerical issues
        """
        self.mode = mode
        self.alpha = alpha
        self.normalize = normalize
        self.epsilon = epsilon

        valid_modes = ["prioritized", "curriculum", "linear", "inverse"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")

    def compute_weights(self, metrics, experiences):
        """
        Compute difficulty-based weights.

        Args:
            metrics: SampleMetrics with difficulty
            experiences: List of experiences

        Returns:
            weights: FloatTensor of per-sample weights
        """
        if metrics.difficulty is None:
            # Difficulty not available, return uniform weights
            total_samples = sum(len(exp.sequences) for exp in experiences)
            device = experiences[0].sequences.device if experiences else 'cuda'
            return torch.ones(total_samples, device=device)

        difficulty = metrics.difficulty

        # Compute weights according to mode
        if self.mode == "prioritized":
            # Higher difficulty → higher weight (PER-style)
            weights = torch.pow(difficulty + self.epsilon, self.alpha)
        elif self.mode == "curriculum":
            # Lower difficulty → higher weight (curriculum learning)
            # Use inverse with alpha exponent
            weights = torch.pow(1.0 / (difficulty + self.epsilon), self.alpha)
        elif self.mode == "linear":
            weights = difficulty
        elif self.mode == "inverse":
            weights = 1.0 / (difficulty + self.epsilon)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Normalize to mean=1
        if self.normalize:
            weights = weights / (weights.mean() + self.epsilon)

        return weights


class StalenessWeighting(LossWeighting):
    """
    Weight samples by staleness (age).

    Older samples may be less relevant due to policy shift, so they
    receive exponentially decaying weights.
    """

    def __init__(
        self,
        decay_factor: float = 0.95,
        normalize: bool = True,
        epsilon: float = 1e-6
    ):
        """
        Initialize staleness weighting.

        Args:
            decay_factor: Exponential decay factor (0 < decay_factor < 1)
                - decay_factor close to 1: slow decay
                - decay_factor close to 0: fast decay
            normalize: Normalize to mean=1
            epsilon: Small constant
        """
        self.decay_factor = decay_factor
        self.normalize = normalize
        self.epsilon = epsilon

        if not 0 < decay_factor <= 1:
            raise ValueError(f"decay_factor must be in (0, 1], got {decay_factor}")

    def compute_weights(self, metrics, experiences):
        """
        Compute staleness-based weights.

        Args:
            metrics: SampleMetrics with staleness
            experiences: List of experiences

        Returns:
            weights: FloatTensor of per-sample weights
        """
        if metrics.staleness is None:
            # Staleness not available, return uniform weights
            total_samples = sum(len(exp.sequences) for exp in experiences)
            device = experiences[0].sequences.device if experiences else 'cuda'
            return torch.ones(total_samples, device=device)

        staleness = metrics.staleness

        # Exponential decay: weight = decay_factor^staleness
        weights = torch.pow(self.decay_factor, staleness)

        # Normalize to mean=1
        if self.normalize:
            weights = weights / (weights.mean() + self.epsilon)

        return weights


class RewardMagnitudeWeighting(LossWeighting):
    """
    Weight samples by reward magnitude.

    This can be used to focus on high-reward or low-reward samples.
    """

    def __init__(
        self,
        mode: str = "favor_high",
        temperature: float = 1.0,
        normalize: bool = True,
        epsilon: float = 1e-6
    ):
        """
        Initialize reward magnitude weighting.

        Args:
            mode: Weighting mode
                - "favor_high": Higher reward → higher weight
                - "favor_low": Lower reward → higher weight
                - "absolute": weight = |reward|
            temperature: Temperature for softmax (used in favor_high/favor_low)
            normalize: Normalize to mean=1
            epsilon: Small constant
        """
        self.mode = mode
        self.temperature = temperature
        self.normalize = normalize
        self.epsilon = epsilon

    def compute_weights(self, metrics, experiences):
        """
        Compute reward-based weights.

        Args:
            metrics: SampleMetrics with reward_value
            experiences: List of experiences

        Returns:
            weights: FloatTensor of per-sample weights
        """
        if metrics.reward_value is None:
            # Reward not available, return uniform weights
            total_samples = sum(len(exp.sequences) for exp in experiences)
            device = experiences[0].sequences.device if experiences else 'cuda'
            return torch.ones(total_samples, device=device)

        rewards = metrics.reward_value

        # Compute weights according to mode
        if self.mode == "favor_high":
            weights = torch.softmax(rewards / self.temperature, dim=0) * len(rewards)
        elif self.mode == "favor_low":
            weights = torch.softmax(-rewards / self.temperature, dim=0) * len(rewards)
        elif self.mode == "absolute":
            weights = torch.abs(rewards)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Normalize to mean=1
        if self.normalize:
            weights = weights / (weights.mean() + self.epsilon)

        return weights


class CompositeWeighting(LossWeighting):
    """
    Combine multiple weighting schemes.

    This allows building complex weighting strategies by composing simple weightings.
    """

    def __init__(
        self,
        weightings: List[Tuple[LossWeighting, float]],
        mode: str = "product",
        normalize: bool = True,
        epsilon: float = 1e-6
    ):
        """
        Initialize composite weighting.

        Args:
            weightings: List of (weighting, coefficient) pairs
            mode: Combination mode
                - "product": Multiply all weights
                - "sum": Sum all weights
                - "weighted_sum": Weighted sum using coefficients
                - "weighted_product": Product of (weight^coefficient)
            normalize: Normalize final weights to mean=1
            epsilon: Small constant
        """
        self.weightings = weightings
        self.mode = mode
        self.normalize = normalize
        self.epsilon = epsilon

        valid_modes = ["product", "sum", "weighted_sum", "weighted_product"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")

        if not weightings:
            warnings.warn("CompositeWeighting initialized with empty weightings list")

    def compute_weights(self, metrics, experiences):
        """
        Combine multiple weights.

        Args:
            metrics: SampleMetrics
            experiences: List of experiences

        Returns:
            weights: Combined FloatTensor of per-sample weights
        """
        if not self.weightings:
            # No weightings, return uniform weights
            total_samples = sum(len(exp.sequences) for exp in experiences)
            device = experiences[0].sequences.device if experiences else 'cuda'
            return torch.ones(total_samples, device=device)

        # Compute first weight
        first_weighting, first_coef = self.weightings[0]
        combined = first_weighting.compute_weights(metrics, experiences)

        # Combine with rest
        if self.mode == "product":
            # Multiply all weights
            for weighting, _ in self.weightings[1:]:
                w = weighting.compute_weights(metrics, experiences)
                combined = combined * w

        elif self.mode == "sum":
            # Sum all weights
            for weighting, _ in self.weightings[1:]:
                w = weighting.compute_weights(metrics, experiences)
                combined = combined + w

        elif self.mode == "weighted_sum":
            # Weighted sum using coefficients
            combined = combined * first_coef
            for weighting, coef in self.weightings[1:]:
                w = weighting.compute_weights(metrics, experiences)
                combined = combined + coef * w

        elif self.mode == "weighted_product":
            # Product of (weight^coefficient)
            combined = torch.pow(combined + self.epsilon, first_coef)
            for weighting, coef in self.weightings[1:]:
                w = weighting.compute_weights(metrics, experiences)
                combined = combined * torch.pow(w + self.epsilon, coef)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Normalize to mean=1
        if self.normalize:
            combined = combined / (combined.mean() + self.epsilon)

        return combined


class UniformWeighting(LossWeighting):
    """
    Uniform weighting (all weights = 1).

    This is a no-op weighting for baseline comparisons.
    """

    def __init__(self):
        """Initialize uniform weighting."""
        pass

    def compute_weights(self, metrics, experiences):
        """
        Return uniform weights.

        Args:
            metrics: SampleMetrics (unused)
            experiences: List of experiences

        Returns:
            weights: FloatTensor of ones
        """
        total_samples = sum(len(exp.sequences) for exp in experiences)
        device = experiences[0].sequences.device if experiences else 'cuda'
        return torch.ones(total_samples, device=device)



