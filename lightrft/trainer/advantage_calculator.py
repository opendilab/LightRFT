"""
Advantage Calculator Module

This module provides a unified interface for computing advantages and returns
in reinforcement learning from human feedback (RLHF) and reinforcement
learning with verifiable rewards (RLVR) workflows. It abstracts
different advantage estimation methods (REINFORCE, GAE, Group Norm, RLOO,
REINFORCE-Baseline, CPGD, etc.) into a common interface, making it easy to
add new methods and maintain existing ones.

The module includes:
    - AdvantageCalculator: Abstract base class defining the standard interface
    - BaseREINFORCECalculator: Base class for all cumulative return-based methods
    - Concrete implementations: GAE, Group Norm (GRPO), RLOO,
      REINFORCE-Baseline, and CPGD
    - Factory function for creating calculator instances

Key Features:
    - Unified interface for all advantage computation methods
    - Support for reward preprocessing (e.g., REINFORCE-Baseline, RLOO, Group Norm)
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
        config: Configuration object containing training parameters
    """
    def __init__(self, config):
        """
        Initialize the advantage calculator.

        :param config: Configuration object containing training parameters
        :type config: object
        """
        self.config = config

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
        gamma: Optional[float],
        generate_kwargs: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute advantages and returns for a single experience.

        This method performs the core advantage computation after reward
        normalization and KL penalty have been applied.

        :param experience: Experience object containing sequences, values, action_mask, etc.
        :type experience: object
        :param final_reward: Processed reward tensor (after normalization and KL penalty)
        :type final_reward: torch.Tensor
        :param gamma: Discount factor. If None, will be taken from generate_kwargs.
        :type gamma: Optional[float]
        :param generate_kwargs: Generation parameters containing lambd, etc.
        :type generate_kwargs: Dict
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
        gamma: Optional[float],
        generate_kwargs: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute advantages using GAE.

        :param experience: Experience object with values and action_mask
        :type experience: object
        :param final_reward: Processed reward tensor
        :type final_reward: torch.Tensor
        :param gamma: Discount factor. If None, will be taken from generate_kwargs.
        :type gamma: Optional[float]
        :param generate_kwargs: Must contain 'lambd' key
        :type generate_kwargs: Dict
        :return: Tuple of (advantages, returns, info_dict)
        :rtype: Tuple[torch.Tensor, torch.Tensor, Dict]
        """
        if gamma is None:
            gamma = generate_kwargs.get("gamma", 1.0)

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
    CPGD (Clipped Policy Gradient Optimization with Policy Drift) calculator.

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
        gamma: Optional[float],
        generate_kwargs: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute advantages using CPGD method.

        Note: CPGD uses the original reward (before KL penalty) from experience.info,
        not the final_reward parameter.

        :param experience: Experience object containing reward in info dict
        :type experience: object
        :param final_reward: Unused for CPGD (uses original reward)
        :type final_reward: torch.Tensor
        :param gamma: Discount factor. Unused for CPGD.
        :type gamma: Optional[float]
        :param generate_kwargs: Unused
        :type generate_kwargs: Dict
        :return: Tuple of (advantages, returns, empty_info_dict)
        :rtype: Tuple[torch.Tensor, torch.Tensor, Dict]
        """
        # CPGD uses original reward from experience, not final_reward
        original_reward = experience.info["reward"].to(final_reward.device)
        advantages, returns = self._get_cpgd_advantages_returns(original_reward, experience.action_mask)
        return advantages, returns, {}


class BaseREINFORCECalculator(AdvantageCalculator):
    """
    Base class for all methods that use cumulative returns as the advantage estimator.

    This class consolidates the common compute logic for REINFORCE-based methods:
    1. Compute cumulative returns (Monte Carlo returns)
    2. Normalize advantages (if configured)
    3. Clip advantages (if configured)

    Subclasses only need to override preprocess_rewards() to implement different
    baseline strategies (e.g., RLOO, group mean, group normalization).
    """
    def _get_gamma(self, gamma: Optional[float], generate_kwargs: Dict) -> float:
        """
        Get the discount factor gamma.

        This method can be overridden by subclasses to enforce specific gamma values.

        :param gamma: Discount factor passed to compute()
        :type gamma: Optional[float]
        :param generate_kwargs: Generation parameters containing 'gamma'
        :type generate_kwargs: Dict
        :return: The gamma value to use
        :rtype: float
        """
        return gamma if gamma is not None else generate_kwargs.get("gamma", 1.0)

    def compute(
        self,
        experience,
        final_reward: torch.Tensor,
        gamma: Optional[float],
        generate_kwargs: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute advantages using cumulative returns (REINFORCE-style).

        :param experience: Experience object
        :type experience: object
        :param final_reward: Processed reward tensor
        :type final_reward: torch.Tensor
        :param gamma: Discount factor. If None, will be taken from generate_kwargs.
        :type gamma: Optional[float]
        :param generate_kwargs: Generation parameters
        :type generate_kwargs: Dict
        :return: Tuple of (advantages, returns, info_dict)
        :rtype: Tuple[torch.Tensor, torch.Tensor, Dict]
        """
        # Get gamma (subclasses can override this behavior)
        gamma = self._get_gamma(gamma, generate_kwargs)

        # Compute cumulative returns
        returns = self.get_cumulative_returns(final_reward, experience.action_mask, gamma)
        advantages = deepcopy(returns)

        # Advantage whitening (normalization)
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


class RLOOCalculator(BaseREINFORCECalculator):
    """
    REINFORCE Leave-One-Out (RLOO) calculator.

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


class REINFORCEBaselineCalculator(BaseREINFORCECalculator):
    """
    REINFORCE++ with baseline calculator.

    Subtracts the mean reward within each group as baseline. This method is robust
    to different reward scales and particularly suitable for reasoning tasks (RLVR).

    Advantage computation equation:
        A_t = R_t - mean(R)

    where:
        - A_t: advantage at time step t
        - R_t: reward at time step t (with gamma=1.0)
        - mean(R): mean reward within the group

    Key differences from GRPO (group_norm):
    - Does NOT divide by std (/ std is not needed in RL variance reduction theory)
    - Forces gamma=1.0 for advantage computation
    - Performs cross-batch advantage normalization

    Reference:
    - REINFORCE++: https://www.researchgate.net/publication/387487679
    - OpenRLHF implementation: https://github.com/OpenRLHF/OpenRLHF
    """
    def preprocess_rewards(
        self,
        rewards: torch.Tensor,
        experiences: List,
        max_new_tokens: int,
    ) -> Tuple[List, List[torch.Tensor]]:
        """
        Preprocess rewards by subtracting group mean baseline.

        This follows the REINFORCE++-baseline approach:
        rewards = rewards - rewards.mean(-1, keepdim=True)

        Note: Unlike GRPO, we do NOT divide by std.

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

        # REINFORCE++-baseline: subtract mean baseline (no division by std)
        # This is different from GRPO which does (rewards - mean) / std
        rewards = rewards - rewards.mean(-1, keepdim=True)

        # Flatten and chunk back
        rewards = rewards.flatten().to("cpu").chunk(len(experiences))
        return experiences, list(rewards)

    def _get_gamma(self, gamma: Optional[float], generate_kwargs: Dict) -> float:
        """
        Force gamma=1.0 for REINFORCE++-baseline algorithm.

        :param gamma: Discount factor passed to compute()
        :type gamma: Optional[float]
        :param generate_kwargs: Generation parameters containing 'gamma'
        :type generate_kwargs: Dict
        :return: Always returns 1.0 (with warning if different value was provided)
        :rtype: float
        """
        if gamma is None:
            gamma = generate_kwargs.get("gamma", 1.0)

        if gamma != 1.0:
            warnings.warn(
                f"gamma is set to 1.0 for reinforce_baseline (was {gamma}). "
                "This is required by REINFORCE++-baseline algorithm."
            )

        return 1.0


class GroupNormCalculator(BaseREINFORCECalculator):
    """
    Group normalization calculator (GRPO).

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
        gamma: Optional[float],
        generate_kwargs: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute advantages using cumulative returns (REINFORCE-style).

        Note: For GroupNorm/GRPO, we skip the advantages_norm step because
        rewards have already been normalized in preprocess_rewards() to avoid
        duplicate normalization.

        :param experience: Experience object
        :type experience: object
        :param final_reward: Processed reward tensor
        :type final_reward: torch.Tensor
        :param gamma: Discount factor. If None, will be taken from generate_kwargs.
        :type gamma: Optional[float]
        :param generate_kwargs: Generation parameters
        :type generate_kwargs: Dict
        :return: Tuple of (advantages, returns, info_dict)
        :rtype: Tuple[torch.Tensor, torch.Tensor, Dict]
        """
        # Get gamma
        gamma = self._get_gamma(gamma, generate_kwargs)

        # Compute cumulative returns
        returns = self.get_cumulative_returns(final_reward, experience.action_mask, gamma)
        advantages = deepcopy(returns)

        # Skip advantages_norm for GroupNorm/GRPO to avoid duplicate normalization
        # (rewards are already normalized in preprocess_rewards)
        info_dict = {}

        # Advantage clipping (still apply if configured)
        if self.config.advantage_clip > 0:
            clip_val = self.config.advantage_clip
            info_dict["advantage_clip_frac"] = compute_clip_fraction(advantages, clip_val, -clip_val)
            advantages = torch.clamp(advantages, -clip_val, clip_val)

        return advantages, returns, info_dict


# ============================================================================
# Factory Function
# ============================================================================


def get_advantage_calculator(estimator_name: str, config) -> AdvantageCalculator:
    """
    Factory function to create an advantage calculator instance.

    :param estimator_name: Name of the advantage estimation method
                          Options: "gae", "cpgd", "reinforce", "rloo",
                                   "reinforce_baseline", "group_norm", "grpo"
    :type estimator_name: str
    :param config: Configuration object containing training parameters
    :type config: object
    :return: Instance of the appropriate AdvantageCalculator subclass
    :rtype: AdvantageCalculator
    :raises ValueError: If estimator_name is not recognized
    """
    calculator_map = {
        "reinforce": BaseREINFORCECalculator,
        "gae": GAECalculator,
        "group_norm": GroupNormCalculator,
        "rloo": RLOOCalculator,
        "reinforce_baseline": REINFORCEBaselineCalculator,
        "cpgd": CPGDCalculator,
        "grpo": GroupNormCalculator,  # Alias for group_norm
    }

    calculator_class = calculator_map.get(estimator_name)
    if calculator_class is None:
        raise ValueError(
            f"Unknown advantage estimator: {estimator_name}. "
            f"Supported options: {list(calculator_map.keys())}"
        )

    return calculator_class(config)


# ============================================================================
# Cross-Batch Advantage Normalization Utilities
# ============================================================================


@torch.no_grad()
def normalize_advantages_cross_batch(experiences: List, advantage_estimator: str, args) -> List:
    """
    Apply cross-batch advantage normalization for GAE, REINFORCE, and REINFORCE-baseline.

    This method normalizes advantages across all experiences in a batch using their action masks.
    Reference: https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/ppo_utils/
    experience_maker.py#L794-L816

    :param experiences: List of Experience objects.
    :type experiences: List
    :param advantage_estimator: Name of the advantage estimation method.
    :type advantage_estimator: str
    :param args: Configuration arguments containing training parameters.
    :type args: object
    :return: List of Experience objects with normalized advantages.
    :rtype: List
    """
    if advantage_estimator not in ["gae", "reinforce", "reinforce_baseline"]:
        return experiences

    # Collect all advantages and action masks
    all_advantages = []
    all_action_masks = []
    for exp in experiences:
        all_advantages.append(exp.advantages.flatten())
        all_action_masks.append(exp.action_mask.flatten())

    # Concatenate into vectors
    advantages_vector = torch.cat(all_advantages, dim=0).float()
    action_masks_vector = torch.cat(all_action_masks, dim=0)
    num_actions = action_masks_vector.sum()

    # Compute mean
    mean = (advantages_vector * action_masks_vector).sum() / num_actions

    # Compute std (if not disabled)
    if not getattr(args, "no_advantage_std_norm", False):
        var = ((advantages_vector - mean).pow(2) * action_masks_vector).sum() / num_actions
        rstd = var.clamp(min=1e-8).rsqrt()
    else:
        rstd = 1

    # Apply normalization to each experience
    for exp in experiences:
        exp.advantages = (exp.advantages - mean) * rstd

    return experiences
