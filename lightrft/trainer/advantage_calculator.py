"""
Advantage Calculator Module

This module provides a unified interface for computing advantages and returns
in reinforcement learning from human feedback (RLHF) workflows. It abstracts
different advantage estimation methods (GAE, CPGD, REINFORCE, etc.) into a
common interface, making it easy to add new methods and maintain existing ones.

The module includes:
    - AdvantageCalculator: Abstract base class defining the standard interface
    - Concrete implementations for various advantage estimation methods
    - CPGD (Clipped Policy Gradient Optimization) utility functions
    - Factory function for creating calculator instances

Key Features:
    - Unified interface for all advantage computation methods
    - Support for reward preprocessing (e.g., RLOO, Group Norm)
    - Configurable advantage whitening and clipping
    - Efficient batch processing

Author: lightrft Team
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import torch
import warnings

from .utils import RunningMoments, compute_clip_fraction


# ============================================================================
# CPGD Utility Functions
# ============================================================================

def _get_cpgd_advantages_returns(
    reward: torch.Tensor,
    action_mask: torch.Tensor,
    weight_factor: str = "STD_weight",
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregate token-level rewards into episode-level scores, normalise them
    group-wise, and then broadcast the normalised scores back to the
    token dimension to obtain both the advantages and the returns that are
    required by the CPGD (Clipped Policy Gradient Optimization with Policy Drift)
    algorithm.

    :param reward: Tensor of shape (num_actions, seq_len) containing token-level rewards
                   produced by the reward model. Each row corresponds to one sampled
                   response (action trajectory).
    :type reward: torch.Tensor
    :param action_mask: Tensor of the same shape as `reward`. Elements belonging to the
                       generated response tokens are 1; padding / prompt tokens are 0.
                       The mask is used so that only response tokens contribute to the
                       final advantages / returns.
    :type action_mask: torch.Tensor
    :param weight_factor: Determines how the per-sample scalar scores are normalised.
                         Options:
                         - "STD_weight": z-score normalisation
                           score_i = (score_i − mean) / (std + ε)
                         - "clip_filter_like_weight": a simplified version of the
                           Clip-Filter weight used in early RLHF repos.
                           score_i = (score_i − mean) * clamp(num_actions / nz, max=3)
                         - any other value: mean-centering only
                           score_i = score_i − mean
    :type weight_factor: str
    :param epsilon: Small constant added to the denominator to avoid division by zero.
    :type epsilon: float
    :return: Tuple of (advantages, returns). Both are normalised per-token values,
             shape (num_actions, seq_len). Non-response tokens are always zero.
    :rtype: Tuple[torch.Tensor, torch.Tensor]

    Notes:
        * Both `advantages` and `returns` are masked so that non-response tokens
          are always zero.
        * The function performs no gradient-tracking operations and is intended
          to be called outside the optimisation graph.
    """
    # ------------------------------------------------------------------ #
    # 1. Collapse token-level rewards to a single scalar per trajectory  #
    # ------------------------------------------------------------------ #
    # shape: (num_actions,)
    scores = reward.sum(dim=-1)

    # Mean and (biased) standard deviation across the batch
    mean = scores.mean()
    std = scores.std(unbiased=False)

    # ------------------------------------------------------------------ #
    # 2. Group-wise normalisation                                        #
    # ------------------------------------------------------------------ #
    if weight_factor == "STD_weight":
        # Standard z-score normalisation
        scores = (scores - mean) / (std + epsilon)

    elif weight_factor == "clip_filter_like_weight":
        # A rough approximation of the clip-filter weighting
        # Count of (std > 0) is always ≥ 1, prevents division by zero
        non_zero = (std > 0).sum().clamp(min=1)
        # Scale by (batch_size / non_zero) but clip to a maximum of 3
        scores = (scores - mean) * (scores.size(0) / non_zero).clamp(max=3.0)

    else:
        # Fallback: mean-centering only
        scores = scores - mean

    # ------------------------------------------------------------------ #
    # 3. Broadcast back to token dimension and apply the mask            #
    # ------------------------------------------------------------------ #
    # shape: (num_actions, seq_len)
    scores = scores.unsqueeze(-1) * action_mask

    # In CPGD the advantage equals the return
    advantages = scores
    returns = deepcopy(scores)

    return advantages, returns


# ============================================================================
# Abstract Base Class
# ============================================================================

class AdvantageCalculator(ABC):
    """
    Abstract base class for advantage computation methods.

    This class defines a standard interface for computing advantages and returns
    in RLHF workflows. Subclasses implement specific algorithms like GAE, CPGD,
    REINFORCE, etc.

    The interface consists of two main methods:
        1. preprocess_rewards: Optional preprocessing of rewards before advantage computation
        2. compute: Main computation of advantages and returns

    Attributes:
        strategy: Training strategy object containing configuration and models
        reward_running_moments: Optional RunningMoments for reward normalization
    """

    def __init__(self, strategy):
        """
        Initialize the advantage calculator.

        :param strategy: Training strategy object containing configuration
        :type strategy: object
        """
        self.strategy = strategy
        self.config = strategy.config
        self.reward_running_moments = None

    @abstractmethod
    def preprocess_rewards(
        self,
        rewards: torch.Tensor,
        experiences: List,
        max_new_tokens: int,
    ) -> Tuple[List, List[torch.Tensor]]:
        """
        Preprocess rewards before advantage computation (optional).

        Some advantage estimation methods (e.g., RLOO, Group Norm) require
        preprocessing the rewards before computing advantages. This method
        handles such preprocessing and may also filter experiences (e.g., for
        dynamic sampling).

        :param rewards: Concatenated reward tensor from all experiences
        :type rewards: torch.Tensor
        :param experiences: List of experience objects
        :type experiences: List
        :param max_new_tokens: Maximum number of new tokens for generation
        :type max_new_tokens: int
        :return: Tuple of (processed_experiences, processed_rewards_list)
        :rtype: Tuple[List, List[torch.Tensor]]
        """
        pass

    @abstractmethod
    def compute(
        self,
        experience,
        final_reward: torch.Tensor,
        generate_kwargs: Dict,
        reward_running_moments: Optional[RunningMoments] = None,
        get_advantages_and_returns_fn=None,
        get_cumulative_returns_fn=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute advantages and returns for a single experience.

        This method performs the core advantage computation after reward
        normalization and KL penalty have been applied.

        :param experience: Experience object containing sequences, values, action_mask, etc.
        :type experience: object
        :param final_reward: Processed reward tensor (after normalization and KL penalty)
        :type final_reward: torch.Tensor
        :param generate_kwargs: Generation parameters containing gamma, lambd, etc.
        :type generate_kwargs: Dict
        :param reward_running_moments: Optional RunningMoments for reward normalization
        :type reward_running_moments: Optional[RunningMoments]
        :param get_advantages_and_returns_fn: Optional function for GAE computation
        :type get_advantages_and_returns_fn: Optional[Callable]
        :param get_cumulative_returns_fn: Optional function for cumulative returns computation
        :type get_cumulative_returns_fn: Optional[Callable]
        :return: Tuple of (advantages, returns, info_dict). info_dict may contain
                 additional metrics like advantage_clip_frac.
        :rtype: Tuple[torch.Tensor, torch.Tensor, Dict]
        """
        pass


# ============================================================================
# Concrete Implementations
# ============================================================================

class DefaultAdvantageCalculator(AdvantageCalculator):
    """
    Default calculator with no reward preprocessing.

    Used by methods like GAE and CPGD that don't require reward preprocessing.
    """

    def preprocess_rewards(
        self,
        rewards: torch.Tensor,
        experiences: List,
        max_new_tokens: int,
    ) -> Tuple[List, List[torch.Tensor]]:
        """
        Default preprocessing: no changes, just chunk rewards.

        :param rewards: Concatenated reward tensor
        :type rewards: torch.Tensor
        :param experiences: List of experiences (unchanged)
        :type experiences: List
        :param max_new_tokens: Maximum new tokens (unused in default)
        :type max_new_tokens: int
        :return: Tuple of (experiences, chunked_rewards)
        :rtype: Tuple[List, List[torch.Tensor]]
        """
        # Chunk rewards back into per-experience tensors
        reward_chunks = rewards.chunk(len(experiences))
        return experiences, list(reward_chunks)


class GAECalculator(DefaultAdvantageCalculator):
    """
    Generalized Advantage Estimation (GAE) calculator.

    Uses value function estimates to compute advantages with reduced variance.
    Supports advantage whitening and clipping through the parent class method.
    """

    def compute(
        self,
        experience,
        final_reward: torch.Tensor,
        generate_kwargs: Dict,
        reward_running_moments: Optional[RunningMoments] = None,
        get_advantages_and_returns_fn=None,
        get_cumulative_returns_fn=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute advantages using GAE.

        The GAE computation is delegated to the FastExperienceMaker's
        get_advantages_and_returns method, which also handles whitening
        and clipping internally.

        :param experience: Experience object with values and action_mask
        :type experience: object
        :param final_reward: Processed reward tensor
        :type final_reward: torch.Tensor
        :param generate_kwargs: Must contain 'gamma' and 'lambd' keys
        :type generate_kwargs: Dict
        :param reward_running_moments: Unused (reward already normalized)
        :type reward_running_moments: Optional[RunningMoments]
        :param get_advantages_and_returns_fn: Function to compute GAE advantages and returns
        :type get_advantages_and_returns_fn: Callable
        :param get_cumulative_returns_fn: Unused for GAE
        :type get_cumulative_returns_fn: Callable
        :return: Tuple of (advantages, returns, info_dict)
        :rtype: Tuple[torch.Tensor, torch.Tensor, Dict]
        """
        if get_advantages_and_returns_fn is None:
            raise ValueError("GAE requires get_advantages_and_returns_fn")

        advantages, returns, advantage_clip_frac = get_advantages_and_returns_fn(
            experience.values,
            final_reward,
            experience.action_mask,
            generate_kwargs["gamma"],
            generate_kwargs["lambd"],
        )
        return advantages, returns, {"advantage_clip_frac": advantage_clip_frac}


class CPGDCalculator(DefaultAdvantageCalculator):
    """
    CPGD (Clipped Policy Gradient Optimization) calculator.

    Aggregates token-level rewards into episode-level scores, normalizes
    them group-wise, and broadcasts back to token dimension.
    """

    def compute(
        self,
        experience,
        final_reward: torch.Tensor,
        generate_kwargs: Dict,
        reward_running_moments: Optional[RunningMoments] = None,
        get_advantages_and_returns_fn=None,
        get_cumulative_returns_fn=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute advantages using CPGD method.

        Note: CPGD uses the original reward (before KL penalty) from experience.info,
        not the final_reward parameter.

        :param experience: Experience object containing reward in info dict
        :type experience: object
        :param final_reward: Unused for CPGD (uses original reward)
        :type final_reward: torch.Tensor
        :param generate_kwargs: Unused
        :type generate_kwargs: Dict
        :param reward_running_moments: Unused
        :type reward_running_moments: Optional[RunningMoments]
        :param get_advantages_and_returns_fn: Unused for CPGD
        :type get_advantages_and_returns_fn: Callable
        :param get_cumulative_returns_fn: Unused for CPGD
        :type get_cumulative_returns_fn: Callable
        :return: Tuple of (advantages, returns, empty_info_dict)
        :rtype: Tuple[torch.Tensor, torch.Tensor, Dict]
        """
        # CPGD uses original reward from experience, not final_reward
        original_reward = experience.info["reward"].to(final_reward.device)
        advantages, returns = _get_cpgd_advantages_returns(
            original_reward, experience.action_mask
        )
        return advantages, returns, {}


class REINFORCECalculator(DefaultAdvantageCalculator):
    """
    Standard REINFORCE calculator.

    Computes cumulative returns and uses them as advantages.
    Supports advantage whitening and clipping.
    """

    def __init__(self, strategy):
        """Initialize REINFORCE calculator."""
        super().__init__(strategy)
        # Get the cumulative returns method from parent class
        # We'll need access to the experience maker instance

    def compute(
        self,
        experience,
        final_reward: torch.Tensor,
        generate_kwargs: Dict,
        reward_running_moments: Optional[RunningMoments] = None,
        get_advantages_and_returns_fn=None,
        get_cumulative_returns_fn=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute advantages using REINFORCE (cumulative returns).

        :param experience: Experience object
        :type experience: object
        :param final_reward: Processed reward tensor
        :type final_reward: torch.Tensor
        :param generate_kwargs: Must contain 'gamma' key
        :type generate_kwargs: Dict
        :param reward_running_moments: Unused
        :type reward_running_moments: Optional[RunningMoments]
        :param get_advantages_and_returns_fn: Unused for REINFORCE
        :type get_advantages_and_returns_fn: Callable
        :param get_cumulative_returns_fn: Function to compute cumulative returns
        :type get_cumulative_returns_fn: Callable
        :return: Tuple of (advantages, returns, info_dict)
        :rtype: Tuple[torch.Tensor, torch.Tensor, Dict]
        """
        if get_cumulative_returns_fn is None:
            raise ValueError("REINFORCE requires get_cumulative_returns_fn")

        # Compute cumulative returns
        returns = get_cumulative_returns_fn(
            final_reward, experience.action_mask, generate_kwargs["gamma"]
        )
        advantages = deepcopy(returns)

        # Advantage whitening
        info_dict = {}
        if self.config.advantages_norm:
            masked_adv = torch.masked_select(advantages, experience.action_mask)
            adv_mean = masked_adv.mean()
            adv_std = masked_adv.std()
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        # Advantage clipping
        if self.config.advantage_clip > 0:
            clip_val = self.config.advantage_clip
            info_dict["advantage_clip_frac"] = compute_clip_fraction(
                advantages, clip_val, -clip_val
            )
            advantages = torch.clamp(advantages, -clip_val, clip_val)

        return advantages, returns, info_dict


class RLOOCalculator(AdvantageCalculator):
    """
    Reward Leave-One-Out (RLOO) calculator.

    Uses leave-one-out baseline for variance reduction. Requires reward
    preprocessing to compute the baseline.
    """

    def preprocess_rewards(
        self,
        rewards: torch.Tensor,
        experiences: List,
        max_new_tokens: int,
    ) -> Tuple[List, List[torch.Tensor]]:
        """
        Preprocess rewards using leave-one-out baseline.

        :param rewards: Concatenated reward tensor
        :type rewards: torch.Tensor
        :param experiences: List of experiences
        :type experiences: List
        :param max_new_tokens: Unused
        :type max_new_tokens: int
        :return: Tuple of (experiences, processed_rewards_list)
        :rtype: Tuple[List, List[torch.Tensor]]
        """
        config = self.config
        n_samples = config.n_samples_per_prompt

        # Reshape to (n_groups, n_samples_per_prompt)
        rewards = rewards.reshape(-1, n_samples).to("cuda")

        # Compute leave-one-out baseline
        # baseline = (sum - self) / (n_samples - 1)
        baseline = (rewards.sum(-1, keepdim=True) - rewards) / (n_samples - 1)

        # Subtract baseline
        rewards = rewards - baseline

        # Flatten and chunk back
        rewards = rewards.flatten().to("cpu").chunk(len(experiences))
        return experiences, list(rewards)

    def compute(
        self,
        experience,
        final_reward: torch.Tensor,
        generate_kwargs: Dict,
        reward_running_moments: Optional[RunningMoments] = None,
        get_advantages_and_returns_fn=None,
        get_cumulative_returns_fn=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute advantages using REINFORCE after RLOO preprocessing.

        :param experience: Experience object
        :type experience: object
        :param final_reward: Processed reward (already has RLOO baseline subtracted)
        :type final_reward: torch.Tensor
        :param generate_kwargs: Must contain 'gamma' key
        :type generate_kwargs: Dict
        :param reward_running_moments: Unused
        :type reward_running_moments: Optional[RunningMoments]
        :param get_advantages_and_returns_fn: Unused for RLOO
        :type get_advantages_and_returns_fn: Callable
        :param get_cumulative_returns_fn: Function to compute cumulative returns
        :type get_cumulative_returns_fn: Callable
        :return: Tuple of (advantages, returns, info_dict)
        :rtype: Tuple[torch.Tensor, torch.Tensor, Dict]
        """
        if get_cumulative_returns_fn is None:
            raise ValueError("RLOO requires get_cumulative_returns_fn")

        # Compute cumulative returns
        returns = get_cumulative_returns_fn(
            final_reward, experience.action_mask, generate_kwargs["gamma"]
        )
        advantages = deepcopy(returns)

        # Advantage whitening
        info_dict = {}
        if self.config.advantages_norm:
            masked_adv = torch.masked_select(advantages, experience.action_mask)
            adv_mean = masked_adv.mean()
            adv_std = masked_adv.std()
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        # Advantage clipping
        if self.config.advantage_clip > 0:
            clip_val = self.config.advantage_clip
            info_dict["advantage_clip_frac"] = compute_clip_fraction(
                advantages, clip_val, -clip_val
            )
            advantages = torch.clamp(advantages, -clip_val, clip_val)

        return advantages, returns, info_dict


class REINFORCEBaselineCalculator(AdvantageCalculator):
    """
    REINFORCE with baseline calculator.

    Subtracts the mean reward within each group as baseline.
    """

    def preprocess_rewards(
        self,
        rewards: torch.Tensor,
        experiences: List,
        max_new_tokens: int,
    ) -> Tuple[List, List[torch.Tensor]]:
        """
        Preprocess rewards by subtracting group mean baseline.

        :param rewards: Concatenated reward tensor
        :type rewards: torch.Tensor
        :param experiences: List of experiences
        :type experiences: List
        :param max_new_tokens: Unused
        :type max_new_tokens: int
        :return: Tuple of (experiences, processed_rewards_list)
        :rtype: Tuple[List, List[torch.Tensor]]
        """
        config = self.config
        n_samples = config.n_samples_per_prompt

        # Reshape to (n_groups, n_samples_per_prompt)
        rewards = rewards.reshape(-1, n_samples).to("cuda")

        # Subtract mean baseline
        rewards = rewards - rewards.mean(-1, keepdim=True)

        # Flatten and chunk back
        rewards = rewards.flatten().to("cpu").chunk(len(experiences))
        return experiences, list(rewards)

    def compute(
        self,
        experience,
        final_reward: torch.Tensor,
        generate_kwargs: Dict,
        reward_running_moments: Optional[RunningMoments] = None,
        get_advantages_and_returns_fn=None,
        get_cumulative_returns_fn=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute advantages using REINFORCE after baseline subtraction.

        :param experience: Experience object
        :type experience: object
        :param final_reward: Processed reward (already has baseline subtracted)
        :type final_reward: torch.Tensor
        :param generate_kwargs: Must contain 'gamma' key
        :type generate_kwargs: Dict
        :param reward_running_moments: Unused
        :type reward_running_moments: Optional[RunningMoments]
        :param get_advantages_and_returns_fn: Unused for REINFORCE baseline
        :type get_advantages_and_returns_fn: Callable
        :param get_cumulative_returns_fn: Function to compute cumulative returns
        :type get_cumulative_returns_fn: Callable
        :return: Tuple of (advantages, returns, info_dict)
        :rtype: Tuple[torch.Tensor, torch.Tensor, Dict]
        """
        if get_cumulative_returns_fn is None:
            raise ValueError("REINFORCE baseline requires get_cumulative_returns_fn")

        # Compute cumulative returns
        returns = get_cumulative_returns_fn(
            final_reward, experience.action_mask, generate_kwargs["gamma"]
        )
        advantages = deepcopy(returns)

        # Advantage whitening
        info_dict = {}
        if self.config.advantages_norm:
            masked_adv = torch.masked_select(advantages, experience.action_mask)
            adv_mean = masked_adv.mean()
            adv_std = masked_adv.std()
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        # Advantage clipping
        if self.config.advantage_clip > 0:
            clip_val = self.config.advantage_clip
            info_dict["advantage_clip_frac"] = compute_clip_fraction(
                advantages, clip_val, -clip_val
            )
            advantages = torch.clamp(advantages, -clip_val, clip_val)

        return advantages, returns, info_dict


class GroupNormCalculator(AdvantageCalculator):
    """
    Group normalization calculator.

    Normalizes rewards within each group and optionally filters degenerate cases.
    """

    def preprocess_rewards(
        self,
        rewards: torch.Tensor,
        experiences: List,
        max_new_tokens: int,
    ) -> Tuple[List, List[torch.Tensor]]:
        """
        Preprocess rewards using group normalization with optional dynamic filtering.

        :param rewards: Concatenated reward tensor
        :type rewards: torch.Tensor
        :param experiences: List of experiences (may be filtered)
        :type experiences: List
        :param max_new_tokens: Unused
        :type max_new_tokens: int
        :return: Tuple of (processed_experiences, processed_rewards_list)
        :rtype: Tuple[List, List[torch.Tensor]]
        """
        config = self.config
        n_samples = config.n_samples_per_prompt

        # Dynamic sampling filtering
        if config.dynamic_sampling:
            step_size = n_samples // config.micro_train_batch_size
            for i in range(0, len(experiences), step_size):
                chunk = experiences[i : i + step_size]
                chunk_rewards = torch.cat([exp.info["reward"] for exp in chunk])

                # Filter out degenerate cases (all 0s or all 1s)
                if torch.all(chunk_rewards == 0) or torch.all(chunk_rewards == 1):
                    for exp in chunk:
                        exp.action_mask = torch.zeros_like(
                            exp.action_mask, dtype=torch.bool
                        )

        # Group normalization
        rewards = rewards.reshape(-1, n_samples).to("cuda")
        rewards = (rewards - rewards.mean(-1, keepdim=True)) / (
            rewards.std(-1, keepdim=True) + 1e-9
        )

        # Flatten and chunk back
        rewards = rewards.flatten().to("cpu").chunk(len(experiences))
        return experiences, list(rewards)

    def compute(
        self,
        experience,
        final_reward: torch.Tensor,
        generate_kwargs: Dict,
        reward_running_moments: Optional[RunningMoments] = None,
        get_advantages_and_returns_fn=None,
        get_cumulative_returns_fn=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute advantages using REINFORCE after group normalization.

        :param experience: Experience object
        :type experience: object
        :param final_reward: Processed reward (already normalized)
        :type final_reward: torch.Tensor
        :param generate_kwargs: Must contain 'gamma' key
        :type generate_kwargs: Dict
        :param reward_running_moments: Unused
        :type reward_running_moments: Optional[RunningMoments]
        :param get_advantages_and_returns_fn: Unused for Group Norm
        :type get_advantages_and_returns_fn: Callable
        :param get_cumulative_returns_fn: Function to compute cumulative returns
        :type get_cumulative_returns_fn: Callable
        :return: Tuple of (advantages, returns, info_dict)
        :rtype: Tuple[torch.Tensor, torch.Tensor, Dict]
        """
        if get_cumulative_returns_fn is None:
            raise ValueError("Group Norm requires get_cumulative_returns_fn")

        # Compute cumulative returns
        returns = get_cumulative_returns_fn(
            final_reward, experience.action_mask, generate_kwargs["gamma"]
        )
        advantages = deepcopy(returns)

        # Advantage whitening
        info_dict = {}
        if self.config.advantages_norm:
            masked_adv = torch.masked_select(advantages, experience.action_mask)
            adv_mean = masked_adv.mean()
            adv_std = masked_adv.std()
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        # Advantage clipping
        if self.config.advantage_clip > 0:
            clip_val = self.config.advantage_clip
            info_dict["advantage_clip_frac"] = compute_clip_fraction(
                advantages, clip_val, -clip_val
            )
            advantages = torch.clamp(advantages, -clip_val, clip_val)

        return advantages, returns, info_dict


# ============================================================================
# Factory Function
# ============================================================================

def get_advantage_calculator(estimator_name: str, strategy) -> AdvantageCalculator:
    """
    Factory function to create an advantage calculator instance.

    :param estimator_name: Name of the advantage estimation method
                          Options: "gae", "cpgd", "reinforce", "rloo",
                                   "reinforce_baseline", "group_norm", "grpo"
    :type estimator_name: str
    :param strategy: Training strategy object containing configuration
    :type strategy: object
    :return: Instance of the appropriate AdvantageCalculator subclass
    :rtype: AdvantageCalculator
    :raises ValueError: If estimator_name is not recognized
    """
    calculator_map = {
        "gae": GAECalculator,
        "cpgd": CPGDCalculator,
        "reinforce": REINFORCECalculator,
        "rloo": RLOOCalculator,
        "reinforce_baseline": REINFORCEBaselineCalculator,
        "group_norm": GroupNormCalculator,
        "grpo": GroupNormCalculator,  # Alias for group_norm
    }

    calculator_class = calculator_map.get(estimator_name)
    if calculator_class is None:
        raise ValueError(
            f"Unknown advantage estimator: {estimator_name}. "
            f"Supported options: {list(calculator_map.keys())}"
        )

    return calculator_class(strategy)

