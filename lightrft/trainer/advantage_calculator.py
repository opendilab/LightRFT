"""
Advantage Calculator Module

This module provides a unified interface for computing advantages and returns
in reinforcement learning from human feedback (RLHF) workflows. It abstracts
different advantage estimation methods (GAE, CPGD, REINFORCE, etc.) into a
common interface, making it easy to add new methods and maintain existing ones.

The module includes:
    - AdvantageCalculator: Abstract base class defining the standard interface
    - Concrete implementations for various advantage estimation methods
    - Factory function for creating calculator instances

Key Features:
    - Unified interface for all advantage computation methods
    - Support for reward preprocessing (e.g., RLOO, Group Norm)
    - Configurable advantage whitening and clipping
    - Efficient batch processing
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import torch
import warnings

from .utils import RunningMoments, compute_clip_fraction

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
    """
    def __init__(self, strategy):
        """
        Initialize the advantage calculator.

        :param strategy: Training strategy object containing configuration
        :type strategy: object
        """
        self.strategy = strategy
        self.config = strategy.config

    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        """
        Compute cumulative returns from rewards using REINFORCE.

        REINFORCE uses cumulative returns without GAE (Generalized Advantage Estimation).

        :param rewards: Tensor of shape (batch_size, response_size).
        :type rewards: torch.Tensor
        :param action_mask: Binary mask tensor of shape (batch_size, response_size).
        :type action_mask: torch.Tensor
        :param gamma: Discount factor.
        :type gamma: float
        :return: Returns tensor of shape (batch_size, response_size).
        :rtype: torch.Tensor
        """
        if isinstance(rewards, list):
            # Packing samples
            # TODO: This is slow...
            returns = []
            for r in rewards:
                ret = self.get_cumulative_returns(r.unsqueeze(0), action_mask, gamma)
                returns.append(ret.squeeze(0))
            return returns

        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns

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

        Default implementation: no changes, just chunk rewards.

        :param rewards: Concatenated reward tensor from all experiences
        :type rewards: torch.Tensor
        :param experiences: List of experience objects
        :type experiences: List
        :param max_new_tokens: Maximum number of new tokens for generation
        :type max_new_tokens: int
        :return: Tuple of (processed_experiences, processed_rewards_list)
        :rtype: Tuple[List, List[torch.Tensor]]
        """
        # Chunk rewards back into per-experience tensors
        reward_chunks = rewards.chunk(len(experiences))
        return experiences, list(reward_chunks)

    @abstractmethod
    def compute(
        self,
        experience,
        final_reward: torch.Tensor,
        generate_kwargs: Dict,
        advantages_and_returns_fn=None,
        cumulative_returns_fn=None,
        gamma=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute advantages and returns for a single experience.

        This method performs the core advantage computation after reward
        normalization and KL penalty have been applied.

        The method will automatically select which function to use based on
        config.advantage_estimator. Subclasses should use the provided functions
        when available, falling back to their own implementations otherwise.

        :param experience: Experience object containing sequences, values, action_mask, etc.
        :type experience: object
        :param final_reward: Processed reward tensor (after normalization and KL penalty)
        :type final_reward: torch.Tensor
        :param generate_kwargs: Generation parameters containing gamma, lambd, etc.
        :type generate_kwargs: Dict
        :param advantages_and_returns_fn: Function for computing advantages and returns using GAE
        :type advantages_and_returns_fn: Optional[Callable]
        :param cumulative_returns_fn: Function for computing cumulative returns
        :type cumulative_returns_fn: Optional[Callable]
        :param gamma: Discount factor. If None, will be taken from generate_kwargs.
        :type gamma: Optional[float]
        :return: Tuple of (advantages, returns, info_dict). info_dict may contain
                 additional metrics like advantage_clip_frac.
        :rtype: Tuple[torch.Tensor, torch.Tensor, Dict]
        """
        pass


# ============================================================================
# Concrete Implementations
# ============================================================================


class GAECalculator(AdvantageCalculator):
    """
    Generalized Advantage Estimation (GAE) calculator.

    Uses value function estimates to compute advantages with reduced variance.
    Supports advantage whitening and clipping.

    Reference: GAE: https://arxiv.org/pdf/1506.02438
    """
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Compute advantages and returns using Generalized Advantage Estimation (GAE).

        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages formula:
            Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                  - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns formula:
            Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                       + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        :param values: Tensor of shape (batch_size, response_size).
        :type values: torch.Tensor
        :param rewards: Tensor of shape (batch_size, response_size).
        :type rewards: torch.Tensor
        :param action_mask: Tensor of shape (batch_size, response_size).
        :type action_mask: torch.Tensor
        :param gamma: Discount factor.
        :type gamma: float
        :param lambd: GAE lambda parameter.
        :type lambd: float
        :return: Tuple of (advantages, returns, advantage_clip_fraction)
        :rtype: Tuple[torch.Tensor, torch.Tensor, float]
        """
        if isinstance(values, list):
            # Packing samples
            # TODO: This is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret, _ = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            # For list case, compute clip fraction on concatenated advantages
            all_advantages = torch.cat(advantages)
            advantage_clip_frac = 0.0
            if self.config.advantage_clip > 0:
                advantage_clip_frac = compute_clip_fraction(
                    all_advantages, self.config.advantage_clip, -self.config.advantage_clip
                )
            return advantages, returns, advantage_clip_frac

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        advantages = advantages.detach()

        # Advantage whitening (normalization)
        if self.config.advantages_norm:
            masked_adv = torch.masked_select(advantages, action_mask)
            adv_mean = masked_adv.mean()
            adv_std = masked_adv.std()
            advantages = (advantages - adv_mean) / (adv_std + 1e-9)

        # Advantage clipping
        advantage_clip_frac = 0.0
        if self.config.advantage_clip > 0:
            advantages = torch.clamp(advantages, -self.config.advantage_clip, self.config.advantage_clip)
            advantage_clip_frac = compute_clip_fraction(
                advantages, self.config.advantage_clip, -self.config.advantage_clip
            )

        return advantages, returns, advantage_clip_frac

    def compute(
        self,
        experience,
        final_reward: torch.Tensor,
        generate_kwargs: Dict,
        advantages_and_returns_fn=None,
        cumulative_returns_fn=None,
        gamma=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute advantages using GAE.

        :param experience: Experience object with values and action_mask
        :type experience: object
        :param final_reward: Processed reward tensor
        :type final_reward: torch.Tensor
        :param generate_kwargs: Must contain 'gamma' and 'lambd' keys
        :type generate_kwargs: Dict
        :param advantages_and_returns_fn: Function for computing advantages and returns using GAE
        :type advantages_and_returns_fn: Optional[Callable]
        :param cumulative_returns_fn: Unused for GAE
        :type cumulative_returns_fn: Optional[Callable]
        :param gamma: Discount factor. If None, will be taken from generate_kwargs.
        :type gamma: Optional[float]
        :return: Tuple of (advantages, returns, info_dict)
        :rtype: Tuple[torch.Tensor, torch.Tensor, Dict]
        """
        # Use provided function if available and estimator is GAE, otherwise use class method
        if advantages_and_returns_fn is not None and self.config.advantage_estimator == "gae":
            advantages, returns, advantage_clip_frac = advantages_and_returns_fn(
                experience.values,
                final_reward,
                experience.action_mask,
                gamma,
                generate_kwargs["lambd"],
            )
        else:
            advantages, returns, advantage_clip_frac = self.get_advantages_and_returns(
                experience.values,
                final_reward,
                experience.action_mask,
                gamma,
                generate_kwargs["lambd"],
            )
        return advantages, returns, {"advantage_clip_frac": advantage_clip_frac}


class CPGDCalculator(AdvantageCalculator):
    """
    CPGD (Clipped Policy Gradient Optimization) calculator.

    Aggregates token-level rewards into episode-level scores, normalizes
    them group-wise, and broadcasts back to token dimension.

    Reference: CPGD: https://arxiv.org/abs/2505.12504
    """
    def _get_cpgd_advantages_returns(
        self,
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

    def compute(
        self,
        experience,
        final_reward: torch.Tensor,
        generate_kwargs: Dict,
        advantages_and_returns_fn=None,
        cumulative_returns_fn=None,
        gamma=None,
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
        :param advantages_and_returns_fn: Unused for CPGD
        :type advantages_and_returns_fn: Optional[Callable]
        :param cumulative_returns_fn: Unused for CPGD
        :type cumulative_returns_fn: Optional[Callable]
        :param gamma: Discount factor. Unused for CPGD.
        :type gamma: Optional[float]
        :return: Tuple of (advantages, returns, empty_info_dict)
        :rtype: Tuple[torch.Tensor, torch.Tensor, Dict]
        """
        # CPGD uses original reward from experience, not final_reward
        original_reward = experience.info["reward"].to(final_reward.device)
        advantages, returns = self._get_cpgd_advantages_returns(original_reward, experience.action_mask)
        return advantages, returns, {}


class REINFORCECalculator(AdvantageCalculator):
    """
    Standard REINFORCE calculator.

    Computes cumulative returns and uses them as advantages.
    Supports advantage whitening and clipping.

    Reference: REINFORCE: Williams, R. J. (1992). Simple statistical gradient-following
    algorithms for connectionist reinforcement learning. Machine learning, 8(3-4), 229-256.
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
        advantages_and_returns_fn=None,
        cumulative_returns_fn=None,
        gamma=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute advantages using REINFORCE (cumulative returns).

        :param experience: Experience object
        :type experience: object
        :param final_reward: Processed reward tensor
        :type final_reward: torch.Tensor
        :param generate_kwargs: Must contain 'gamma' key
        :type generate_kwargs: Dict
        :param advantages_and_returns_fn: Unused for REINFORCE
        :type advantages_and_returns_fn: Optional[Callable]
        :param cumulative_returns_fn: Function for computing cumulative returns
        :type cumulative_returns_fn: Optional[Callable]
        :param gamma: Discount factor. If None, will be taken from generate_kwargs.
        :type gamma: Optional[float]
        :return: Tuple of (advantages, returns, info_dict)
        :rtype: Tuple[torch.Tensor, torch.Tensor, Dict]
        """
        # Use provided function if available and estimator uses cumulative returns, otherwise use class method
        if (
            cumulative_returns_fn is not None
            and self.config.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm", "grpo"]
        ):
            returns = cumulative_returns_fn(final_reward, experience.action_mask, gamma)
        else:
            returns = self.get_cumulative_returns(final_reward, experience.action_mask, gamma)
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
            info_dict["advantage_clip_frac"] = compute_clip_fraction(advantages, clip_val, -clip_val)
            advantages = torch.clamp(advantages, -clip_val, clip_val)

        return advantages, returns, info_dict


class RLOOCalculator(AdvantageCalculator):
    """
    Reward Leave-One-Out (RLOO) calculator.

    Uses leave-one-out baseline for variance reduction. Requires reward
    preprocessing to compute the baseline.

    Reference: RLOO: https://arxiv.org/abs/2402.14740
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
        advantages_and_returns_fn=None,
        cumulative_returns_fn=None,
        gamma=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute advantages using REINFORCE after RLOO preprocessing.

        :param experience: Experience object
        :type experience: object
        :param final_reward: Processed reward (already has RLOO baseline subtracted)
        :type final_reward: torch.Tensor
        :param generate_kwargs: Must contain 'gamma' key
        :type generate_kwargs: Dict
        :param advantages_and_returns_fn: Unused for RLOO
        :type advantages_and_returns_fn: Optional[Callable]
        :param cumulative_returns_fn: Function for computing cumulative returns
        :type cumulative_returns_fn: Optional[Callable]
        :param gamma: Discount factor. If None, will be taken from generate_kwargs.
        :type gamma: Optional[float]
        :return: Tuple of (advantages, returns, info_dict)
        :rtype: Tuple[torch.Tensor, torch.Tensor, Dict]
        """
        # Use provided function if available and estimator uses cumulative returns, otherwise use class method
        if (
            cumulative_returns_fn is not None
            and self.config.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm", "grpo"]
        ):
            returns = cumulative_returns_fn(final_reward, experience.action_mask, gamma)
        else:
            returns = self.get_cumulative_returns(final_reward, experience.action_mask, gamma)
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
            info_dict["advantage_clip_frac"] = compute_clip_fraction(advantages, clip_val, -clip_val)
            advantages = torch.clamp(advantages, -clip_val, clip_val)

        return advantages, returns, info_dict


class REINFORCEBaselineCalculator(AdvantageCalculator):
    """
    REINFORCE with baseline calculator.

    Subtracts the mean reward within each group as baseline.

    Reference: REINFORCE: Williams, R. J. (1992). Simple statistical gradient-following
    algorithms for connectionist reinforcement learning. Machine learning, 8(3-4), 229-256.
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
        advantages_and_returns_fn=None,
        cumulative_returns_fn=None,
        gamma=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute advantages using REINFORCE after baseline subtraction.

        :param experience: Experience object
        :type experience: object
        :param final_reward: Processed reward (already has baseline subtracted)
        :type final_reward: torch.Tensor
        :param generate_kwargs: Must contain 'gamma' key
        :type generate_kwargs: Dict
        :param advantages_and_returns_fn: Unused for REINFORCE baseline
        :type advantages_and_returns_fn: Optional[Callable]
        :param cumulative_returns_fn: Function for computing cumulative returns
        :type cumulative_returns_fn: Optional[Callable]
        :param gamma: Discount factor. If None, will be taken from generate_kwargs.
        :type gamma: Optional[float]
        :return: Tuple of (advantages, returns, info_dict)
        :rtype: Tuple[torch.Tensor, torch.Tensor, Dict]
        """
        # Use provided function if available and estimator uses cumulative returns, otherwise use class method
        if (
            cumulative_returns_fn is not None
            and self.config.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm", "grpo"]
        ):
            returns = cumulative_returns_fn(final_reward, experience.action_mask, gamma)
        else:
            returns = self.get_cumulative_returns(final_reward, experience.action_mask, gamma)
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
            info_dict["advantage_clip_frac"] = compute_clip_fraction(advantages, clip_val, -clip_val)
            advantages = torch.clamp(advantages, -clip_val, clip_val)

        return advantages, returns, info_dict


class GroupNormCalculator(AdvantageCalculator):
    """
    Group normalization calculator.

    Normalizes rewards within each group and optionally filters degenerate cases.

    Reference: GRPO: https://arxiv.org/pdf/2402.03300
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
                chunk = experiences[i:i + step_size]
                chunk_rewards = torch.cat([exp.info["reward"] for exp in chunk])

                # Filter out degenerate cases (all 0s or all 1s)
                if torch.all(chunk_rewards == 0) or torch.all(chunk_rewards == 1):
                    for exp in chunk:
                        exp.action_mask = torch.zeros_like(exp.action_mask, dtype=torch.bool)

        # Group normalization
        rewards = rewards.reshape(-1, n_samples).to("cuda")
        rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)

        # Flatten and chunk back
        rewards = rewards.flatten().to("cpu").chunk(len(experiences))
        return experiences, list(rewards)

    def compute(
        self,
        experience,
        final_reward: torch.Tensor,
        generate_kwargs: Dict,
        advantages_and_returns_fn=None,
        cumulative_returns_fn=None,
        gamma=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute advantages using REINFORCE after group normalization.

        :param experience: Experience object
        :type experience: object
        :param final_reward: Processed reward (already normalized)
        :type final_reward: torch.Tensor
        :param generate_kwargs: Must contain 'gamma' key
        :type generate_kwargs: Dict
        :param advantages_and_returns_fn: Unused for Group Norm
        :type advantages_and_returns_fn: Optional[Callable]
        :param cumulative_returns_fn: Function for computing cumulative returns
        :type cumulative_returns_fn: Optional[Callable]
        :param gamma: Discount factor. If None, will be taken from generate_kwargs.
        :type gamma: Optional[float]
        :return: Tuple of (advantages, returns, info_dict)
        :rtype: Tuple[torch.Tensor, torch.Tensor, Dict]
        """
        # Use provided function if available and estimator uses cumulative returns, otherwise use class method
        if (
            cumulative_returns_fn is not None
            and self.config.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm", "grpo"]
        ):
            returns = cumulative_returns_fn(final_reward, experience.action_mask, gamma)
        else:
            returns = self.get_cumulative_returns(final_reward, experience.action_mask, gamma)
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
            info_dict["advantage_clip_frac"] = compute_clip_fraction(advantages, clip_val, -clip_val)
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
