"""
Metrics Computation Module

Provides unified interface for computing various sample-level metrics
used in filtering and weighting.

Author: LightRLHF Team
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
import torch
from lightrft.models.utils import masked_mean, unpacking_samples


@dataclass
class SampleMetrics:
    """
    Container for all computed sample metrics.

    All metrics are per-sample tensors aligned with the sample indices.
    Shapes are (total_samples,) where total_samples = sum(batch_size across all micro-batches).

    Attributes:
        response_length: Length of generated responses (total_samples,)
        entropy: Policy entropy -sum(p * log(p)) (total_samples,)
        logit_kl: KL divergence at logit level (total_samples,)
        difficulty: Sample difficulty score (total_samples,)
        staleness: Sample age/staleness (total_samples,)
        reward_value: Reward values (total_samples,)
        n_samples_per_prompt: Number of samples per prompt (for grouping)
        micro_batch_size: Micro batch size (for splitting)
    """
    # Core metrics (always available)
    response_length: torch.Tensor  # (total_samples,)

    # Optional metrics
    entropy: Optional[torch.Tensor] = None  # (total_samples,)
    logit_kl: Optional[torch.Tensor] = None  # (total_samples,)
    difficulty: Optional[torch.Tensor] = None  # (total_samples,)
    staleness: Optional[torch.Tensor] = None  # (total_samples,)
    reward_value: Optional[torch.Tensor] = None  # (total_samples,)

    # Auxiliary data for filtering/weighting
    n_samples_per_prompt: Optional[int] = None
    micro_batch_size: Optional[int] = None


class MetricsComputer:
    """
    Compute various metrics for experience samples.

    This class provides methods to compute sample-level metrics that can be used
    for filtering and weighting during experience generation.

    Args:
        packing_samples: Whether samples are packed (affects unpacking logic)
    """

    def __init__(self, packing_samples: bool = False):
        """
        Initialize metrics computer.

        Args:
            packing_samples: Whether samples are packed into single sequences
        """
        self.packing_samples = packing_samples

    def compute_entropy(
        self,
        action_log_probs: torch.Tensor,
        action_mask: torch.Tensor,
        num_actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute policy entropy per sample.

        Entropy = -sum(p * log(p)) where p = exp(log_prob)
        Higher entropy indicates more uncertain/exploratory policy.

        Args:
            action_log_probs: Log probabilities (batch_size, seq_len) or (1, total_len) for packed
            action_mask: Action mask (batch_size, seq_len) or (1, total_len) for packed
            num_actions: Number of actions per sample (for packed samples)

        Returns:
            entropy: Per-sample entropy (batch_size,)
        """
        # Convert log probs to probs
        probs = torch.exp(action_log_probs)

        # Entropy = -sum(p * log(p)) = -sum(p * log_p)
        entropy_per_token = -(probs * action_log_probs)

        # Handle packed vs unpacked samples
        if self.packing_samples and num_actions is not None:
            # Unpack and compute mean for each sample
            entropy_unpacked = unpacking_samples(entropy_per_token, num_actions)
            return torch.tensor([ent.mean() for ent in entropy_unpacked], device=action_log_probs.device)
        else:
            # Mask and average over sequence
            return masked_mean(entropy_per_token, action_mask, dim=-1)

    def compute_logit_kl(
        self,
        current_logits: torch.Tensor,
        reference_logits: torch.Tensor,
        action_mask: torch.Tensor,
        num_actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute KL divergence between logit distributions.

        This differs from log-prob KL in that it operates on the full
        distribution rather than just the selected actions.

        KL(current || reference) = sum(p_curr * (log p_curr - log p_ref))

        Args:
            current_logits: Logits from current policy (batch, seq, vocab)
            reference_logits: Logits from reference policy (batch, seq, vocab)
            action_mask: Action mask (batch, seq)
            num_actions: Number of actions per sample (for packed samples)

        Returns:
            kl: Per-sample KL divergence (batch_size,)
        """
        # Convert to log probabilities
        current_log_probs = torch.log_softmax(current_logits, dim=-1)
        reference_log_probs = torch.log_softmax(reference_logits, dim=-1)

        # KL(current || reference) = sum(p_curr * (log p_curr - log p_ref))
        kl_per_token = torch.sum(
            torch.exp(current_log_probs) * (current_log_probs - reference_log_probs),
            dim=-1
        )

        # Handle packed vs unpacked samples
        if self.packing_samples and num_actions is not None:
            kl_unpacked = unpacking_samples(kl_per_token, num_actions)
            return torch.tensor([kl.mean() for kl in kl_unpacked], device=kl_per_token.device)
        else:
            return masked_mean(kl_per_token, action_mask, dim=-1)

    def compute_difficulty(
        self,
        rewards: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        mode: str = "td_error"
    ) -> torch.Tensor:
        """
        Compute sample difficulty.

        Difficulty can be measured in various ways:
        - "td_error": Temporal difference error |reward - value|
        - "low_reward": Inverse of reward (harder = lower reward)
        - "high_variance": Variance of reward within group (placeholder)
        - "abs_reward": Absolute reward magnitude

        Args:
            rewards: Per-sample rewards (total_samples,)
            values: Per-sample value estimates (total_samples,) - required for "td_error"
            mode: Difficulty computation mode

        Returns:
            difficulty: Per-sample difficulty scores (total_samples,)
        """
        if mode == "td_error":
            if values is None:
                raise ValueError("Values required for td_error difficulty mode")
            # Higher TD error = more surprising = harder
            return torch.abs(rewards - values)

        elif mode == "low_reward":
            # Lower reward = harder (normalized to [0, 1])
            min_r, max_r = rewards.min(), rewards.max()
            if max_r - min_r < 1e-6:
                return torch.ones_like(rewards) * 0.5
            return 1 - (rewards - min_r) / (max_r - min_r)

        elif mode == "abs_reward":
            # Absolute reward magnitude
            return torch.abs(rewards)

        elif mode == "high_variance":
            # Compute variance within groups (placeholder - needs grouping info)
            # This would require n_samples_per_prompt and group-wise computation
            # For now, return zeros as placeholder
            return torch.zeros_like(rewards)

        else:
            raise ValueError(f"Unknown difficulty mode: {mode}")

    def compute_staleness(
        self,
        generation_steps: torch.Tensor,
        current_step: int,
        mode: str = "linear"
    ) -> torch.Tensor:
        """
        Compute sample staleness based on age.

        Staleness measures how old a sample is relative to current training step.
        Older samples may be less relevant due to policy shift.

        Args:
            generation_steps: Step when each sample was generated (total_samples,)
            current_step: Current training step
            mode: Staleness computation mode
                - "linear": age normalized to [0, 1]
                - "exponential": 1 - exp(-age / tau)

        Returns:
            staleness: Per-sample staleness scores (total_samples,)
        """
        age = current_step - generation_steps

        if mode == "linear":
            # Normalize to [0, 1] range
            max_age = age.max()
            if max_age < 1:
                return torch.zeros_like(age, dtype=torch.float32)
            return age.float() / max_age

        elif mode == "exponential":
            # Exponential decay: 1 - exp(-age / tau)
            tau = 10.0  # Half-life parameter
            return 1 - torch.exp(-age.float() / tau)

        else:
            raise ValueError(f"Unknown staleness mode: {mode}")

    def compute_all_metrics(
        self,
        outputs: List,  # List[_SamplesOutput]
        enable_flags: Dict[str, bool],
        current_step: Optional[int] = None
    ) -> SampleMetrics:
        """
        Compute all enabled metrics from sample outputs.

        This is the main entry point for computing metrics. It checks enable_flags
        to determine which metrics to compute and returns a SampleMetrics object.

        Args:
            outputs: List of _SamplesOutput objects from model inference
            enable_flags: Dict indicating which metrics to compute
                - "entropy": bool
                - "logit_kl": bool
                - "difficulty": bool
                - "difficulty_mode": str
                - "staleness": bool
                - "staleness_mode": str
            current_step: Current training step (required for staleness)

        Returns:
            SampleMetrics with all enabled metrics computed
        """
        # Collect core metrics (always available)
        response_lengths = torch.cat([out.response_length for out in outputs])

        # Get rewards if available
        rewards = None
        if outputs[0].rewards is not None:
            rewards = torch.cat([out.rewards for out in outputs])

        # Initialize optional metrics
        entropy = None
        logit_kl = None
        difficulty = None
        staleness = None

        # Compute entropy if enabled
        if enable_flags.get("entropy", False):
            if outputs[0].action_log_probs is not None:
                action_log_probs_list = []
                action_mask_list = []
                num_actions_list = []

                for out in outputs:
                    action_log_probs_list.append(out.action_log_probs)
                    action_mask_list.append(out.action_mask)
                    if self.packing_samples:
                        num_actions_list.extend(out.num_actions)

                # Concatenate
                action_log_probs = torch.cat(action_log_probs_list, dim=0)
                action_mask = torch.cat(action_mask_list, dim=0)

                if self.packing_samples:
                    entropy = self.compute_entropy(
                        action_log_probs, action_mask, num_actions_list
                    )
                else:
                    entropy = self.compute_entropy(action_log_probs, action_mask)

        # Compute logit KL if enabled (requires storing logits, not typically available)
        # This is a placeholder for future implementation
        if enable_flags.get("logit_kl", False):
            # Would need access to current_logits and reference_logits
            # For now, leave as None
            pass

        # Compute difficulty if enabled
        if enable_flags.get("difficulty", False) and rewards is not None:
            difficulty_mode = enable_flags.get("difficulty_mode", "td_error")

            # Get values if needed
            values = None
            if difficulty_mode == "td_error" and outputs[0].value is not None:
                values = torch.cat([out.value for out in outputs])

            difficulty = self.compute_difficulty(rewards, values, mode=difficulty_mode)

        # Compute staleness if enabled
        if enable_flags.get("staleness", False) and current_step is not None:
            # Would need to track generation_steps in outputs
            # For now, this is a placeholder
            # In practice, you'd need to add generation_step to _SamplesOutput
            pass

        return SampleMetrics(
            response_length=response_lengths,
            entropy=entropy,
            logit_kl=logit_kl,
            difficulty=difficulty,
            staleness=staleness,
            reward_value=rewards,
            n_samples_per_prompt=None,  # Can be set externally
            micro_batch_size=None,  # Can be set externally
        )

