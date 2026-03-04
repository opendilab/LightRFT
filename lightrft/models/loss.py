"""
Loss functions used across LightRFT models.

This module implements a comprehensive collection of loss functions for reinforcement learning
from human feedback (RLHF) and related training paradigms:

**Policy Optimization Losses:**
- PolicyLoss: Multi-purpose policy loss supporting PPO, CPGD (via use_cpg_loss), DAPO-style
  decoupled clipping, and high-entropy token filtering for efficient training.
- ValueLoss: Value function loss for PPO with optional value clipping.

**Reward Model Losses:**
- GPTLMLoss: Next-token prediction loss for generative reward model training.
- LogSigmoidLoss: Log-sigmoid pairwise loss for scalar reward model training.
- LogExpLoss: Log-exp pairwise loss for scalar reward model training.
- HPSLoss: Human Preference Score loss for scalar reward model training.
- PairWiseLoss: Generic pairwise preference loss for reward models.
- PRMLoss: Process Reward Model loss for token-level reward prediction.

**Preference Learning Losses:**
- DPOLoss: Direct Preference Optimization loss for aligning language models with preferences.
- KTOLoss: Kahneman-Tversky Optimization loss for uneven sampling scenarios.
- VanillaKTOLoss: Simplified KTO loss for even sampling scenarios.

**Knowledge Distillation:**
- KDLoss: Knowledge Distillation loss for transferring knowledge from teacher to student models.

All loss functions are designed to work seamlessly with the LightRFT training framework,
supporting distributed training, mixed precision, and various optimization strategies.
"""

from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .utils import masked_mean


class GPTLMLoss(nn.Module):
    """
    GPT Language Model loss for next-token prediction.
    Used for generative reward model training.

    :ivar int IGNORE_INDEX: Label index to ignore when computing the
        cross-entropy (default: ``-100``), matching Hugging Face conventions.
    :ivar torch.nn.CrossEntropyLoss loss: Underlying cross-entropy criterion
        configured to ignore ``IGNORE_INDEX``.
    """
    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute next-token prediction loss.

        Uses the common shifting scheme:
        ``shift_logits = logits[..., :-1, :]`` and
        ``shift_labels = labels[..., 1:]``.

        :param logits: Model output logits.
        :type logits: torch.Tensor
        :param labels: Token ids aligned with logits. Tokens to be ignored
            should be set to ``IGNORE_INDEX`` (default ``-100``).
        :type labels: torch.Tensor

        :returns: Scalar mean cross-entropy loss.
        :rtype: torch.Tensor

        :shape logits: ``(..., seq_len, vocab_size)``
        :shape labels: ``(..., seq_len)``
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss


class PolicyLoss(nn.Module):
    """
    Multi-purpose policy loss function supporting multiple reinforcement learning algorithms.

    This class implements a unified policy loss that can be configured to support various
    policy optimization algorithms including PPO, CPGD, GSPO, and GMPO, with optional
    high-entropy token filtering for efficient training.

    **Supported Algorithms:**

    - **PPO (Proximal Policy Optimization)**: Default mode using standard clipped surrogate
      objective. The loss is computed as ``-min(ratio * advantages, clipped_ratio * advantages)``
      where ``ratio = exp(log_probs - old_log_probs)`` and clipping is applied to prevent
      large policy updates.

    - **Clipped Policy Gradient Optimization with Policy Drift (CPGD)**: Enabled via ``use_cpg_loss=True``. Uses
      asymmetric clipping bounds for positive and negative advantages, providing better
      stability for constrained policy optimization. See: https://arxiv.org/abs/2505.12504

    - **Group Sequence Policy Optimization (GSPO)**: Enabled via ``use_gspo=True``. Uses
      sequence-level importance ratios (geometric mean of per-token ratios) instead of
      token-level ratios. The combined token-level ratio uses stop-gradient to decouple
      the sequence-level and token-level gradients. Typically uses very small clip ranges
      (e.g., 0.0003/0.0004). See: https://arxiv.org/abs/2507.18071

    - **Geometric Mean Policy Optimization (GMPO)**: Enabled via ``use_gmpo=True``. Uses
      the geometric mean of importance ratios across tokens, with clipping applied in
      signed log-difference space per-token before computing the sequence-level ratio.
      This reduces the influence of outlier tokens. Typically uses larger clip ranges
      (e.g., 0.4). See: https://arxiv.org/abs/2502.03950

    - **High-Entropy Token Filtering**: Enabled via ``high_entropy_token_ratio > 0`` or by
      providing an ``entropy_mask`` in the forward pass. This feature allows training only on
      high-entropy tokens (forking tokens that determine reasoning directions), significantly
      improving training efficiency. Based on: https://arxiv.org/abs/2506.01939

    :param clip_eps: Clipping epsilon for PPO/GMPO-style policy updates. Default: 0.2
    :type clip_eps: float
    :param use_dapo: Flag for DAPO. Currently reserved for future implementation. Default: False
    :type use_dapo: bool
    :param use_cpg_loss: If True, uses CPGD-style clipped policy gradient loss. Default: False
    :type use_cpg_loss: bool
    :param use_gspo: If True, uses GSPO sequence-level importance ratios. Default: False
    :type use_gspo: bool
    :param use_gmpo: If True, uses GMPO geometric-mean importance ratios. Default: False
    :type use_gmpo: bool
    :param clip_ratio_low: Lower clip bound for GSPO asymmetric clipping. If None, falls
        back to ``clip_eps``. Typical GSPO value: 0.0003. Default: None
    :type clip_ratio_low: Optional[float]
    :param clip_ratio_high: Upper clip bound for GSPO asymmetric clipping. If None, falls
        back to ``clip_eps``. Typical GSPO value: 0.0004. Default: None
    :type clip_ratio_high: Optional[float]
    :param high_entropy_token_ratio: Ratio of high-entropy tokens to keep for training.
        Set to 0.0 to disable. Default: 0.0
    :type high_entropy_token_ratio: float

    **Example Usage:**

    .. code-block:: python

        # Standard PPO loss
        policy_loss = PolicyLoss(clip_eps=0.2)
        loss = policy_loss(log_probs, old_log_probs, advantages, action_mask)

        # CPGD loss
        policy_loss = PolicyLoss(clip_eps=0.2, use_cpg_loss=True)
        loss = policy_loss(log_probs, old_log_probs, advantages, action_mask)

        # GSPO loss with asymmetric clipping
        policy_loss = PolicyLoss(use_gspo=True, clip_ratio_low=0.0003, clip_ratio_high=0.0004)
        loss = policy_loss(log_probs, old_log_probs, advantages, action_mask)

        # GMPO loss with wider clipping
        policy_loss = PolicyLoss(clip_eps=0.4, use_gmpo=True)
        loss = policy_loss(log_probs, old_log_probs, advantages, action_mask)

    **References:**

    - PPO: https://arxiv.org/abs/1707.06347
    - CPGD: https://arxiv.org/abs/2505.12504
    - GSPO: https://arxiv.org/abs/2507.18071
    - GMPO: https://arxiv.org/abs/2502.03950
    - High-Entropy Token Filtering: https://arxiv.org/abs/2506.01939
    """
    def __init__(
        self,
        clip_eps: float = 0.2,
        use_dapo: bool = False,
        use_cpg_loss: bool = False,
        use_gspo: bool = False,
        use_gmpo: bool = False,
        clip_ratio_low: Optional[float] = None,
        clip_ratio_high: Optional[float] = None,
        high_entropy_token_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.use_dapo = use_dapo
        self.use_cpg_loss = use_cpg_loss
        self.use_gspo = use_gspo
        self.use_gmpo = use_gmpo
        self.clip_ratio_low = clip_ratio_low if clip_ratio_low is not None else clip_eps
        self.clip_ratio_high = clip_ratio_high if clip_ratio_high is not None else clip_eps
        self.high_entropy_token_ratio = high_entropy_token_ratio

    def _compute_gspo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        final_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GSPO (Group Sequence Policy Optimization) loss.

        GSPO replaces token-level importance ratios with sequence-level ratios defined as
        the geometric mean of per-token ratios. A stop-gradient trick decouples the
        sequence-level ratio from token-level gradients:

        .. math::
            s_i(\\theta) = \\exp\\left(\\frac{1}{|y_i|} \\sum_t \\log \\frac{\\pi_\\theta}{\\pi_{\\theta_{old}}}\\right)

            s_{i,t}(\\theta) = \\text{sg}[s_i(\\theta)] \\cdot \\frac{\\pi_\\theta(y_{i,t})}{\\text{sg}[\\pi_\\theta(y_{i,t})]}

        :param log_probs: Current policy log-probabilities. Shape: ``(batch_size, seq_len)``
        :type log_probs: torch.Tensor
        :param old_log_probs: Old policy log-probabilities. Shape: ``(batch_size, seq_len)``
        :type old_log_probs: torch.Tensor
        :param advantages: Per-token advantage estimates. Shape: ``(batch_size, seq_len)``
        :type advantages: torch.Tensor
        :param final_mask: Binary mask for valid tokens. Shape: ``(batch_size, seq_len)``
        :type final_mask: torch.Tensor
        :returns: Scalar GSPO policy loss.
        :rtype: torch.Tensor
        """
        negative_approx_kl = log_probs - old_log_probs

        # Sequence-level importance ratio (geometric mean in log space)
        seq_lengths = torch.sum(final_mask, dim=-1).clamp(min=1)
        negative_approx_kl_seq = torch.sum(negative_approx_kl * final_mask, dim=-1) / seq_lengths

        # Combined token-level ratio with stop-gradient:
        # log(s_{i,t}) = sg[log(s_i)] + log_prob - sg[log_prob]
        log_seq_importance_ratio = (
            log_probs - log_probs.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
        )
        log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)
        seq_importance_ratio = torch.exp(log_seq_importance_ratio)

        # PPO-style clipping on the sequence-level ratio
        surr1 = -advantages * seq_importance_ratio
        surr2 = -advantages * torch.clamp(
            seq_importance_ratio, 1 - self.clip_ratio_low, 1 + self.clip_ratio_high
        )
        loss = torch.maximum(surr1, surr2)

        loss = masked_mean(loss, final_mask, dim=-1).mean()
        return loss

    def _compute_gmpo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        final_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GMPO (Geometric Mean Policy Optimization) loss.

        GMPO clips per-token log-probability differences in signed space and then
        aggregates via geometric mean (arithmetic mean of log-ratios). This reduces the
        influence of outlier tokens compared to the arithmetic-mean aggregation in GRPO.

        For each sequence with advantage :math:`A`:

        .. math::
            s = \\text{sgn}(A) \\cdot (\\log \\pi_\\theta - \\log \\pi_{\\theta_{old}})

            \\tilde{s} = \\text{clip}(s, -\\epsilon, \\epsilon)

            \\tilde{\\Delta} = \\text{sgn}(A) \\cdot \\max(s, \\tilde{s})

            \\text{ratio} = \\exp\\left(\\frac{\\sum_t \\tilde{\\Delta}_t}{|y|}\\right)

            \\mathcal{L} = -A \\cdot \\text{ratio}

        :param log_probs: Current policy log-probabilities. Shape: ``(batch_size, seq_len)``
        :type log_probs: torch.Tensor
        :param old_log_probs: Old policy log-probabilities. Shape: ``(batch_size, seq_len)``
        :type old_log_probs: torch.Tensor
        :param advantages: Per-token advantage estimates. Shape: ``(batch_size, seq_len)``
        :type advantages: torch.Tensor
        :param final_mask: Binary mask for valid tokens. Shape: ``(batch_size, seq_len)``
        :type final_mask: torch.Tensor
        :returns: Scalar GMPO policy loss.
        :rtype: torch.Tensor
        """
        # Extract per-sequence advantage (constant across tokens for GRPO-style)
        seq_lengths = final_mask.sum(dim=-1).clamp(min=1)
        seq_advantages = (advantages * final_mask).sum(dim=-1) / seq_lengths

        # Sign factor: -1 if advantage >= 0, else 1 (matches GMPO paper convention)
        sgn = torch.where(seq_advantages >= 0, -1.0, 1.0).unsqueeze(-1)

        logprobs_diff = log_probs - old_log_probs
        sgn_logprobs_diff = sgn * logprobs_diff
        sgn_logprobs_diff_clamp = torch.clamp(sgn_logprobs_diff, -self.clip_eps, self.clip_eps)
        sgn_logprobs_diff_max = torch.max(sgn_logprobs_diff, sgn_logprobs_diff_clamp)
        logprobs_diff_max = sgn * sgn_logprobs_diff_max

        # Geometric mean ratio: exp(mean of clipped log-diffs over valid tokens)
        ratio = torch.exp((logprobs_diff_max * final_mask).sum(dim=-1) / seq_lengths)

        loss = (-seq_advantages * ratio).mean()
        return loss

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        entropy_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute policy loss with optional masking and algorithm-specific clipping.

        This method dispatches to the configured algorithm (PPO, CPGD, GSPO, or GMPO)
        and applies masking for valid tokens and optionally high-entropy tokens.

        :param log_probs: Log probabilities of actions under the current policy.
            Shape: ``(batch_size, num_actions)``
        :type log_probs: torch.Tensor
        :param old_log_probs: Log probabilities of actions under the old/reference policy.
            Shape: ``(batch_size, num_actions)``
        :type old_log_probs: torch.Tensor
        :param advantages: Advantage estimates for each action. Positive values indicate
            better-than-average actions. Shape: ``(batch_size, num_actions)``
        :type advantages: torch.Tensor
        :param action_mask: Binary mask indicating valid action tokens (1 for valid, 0 for padding).
            If None, all tokens are considered valid. Shape: ``(batch_size, num_actions)``
        :type action_mask: Optional[torch.Tensor]
        :param entropy_mask: Binary mask for high-entropy tokens to keep for training.
            If provided, overrides the instance-level ``entropy_mask``. Shape: ``(batch_size, num_actions)``
        :type entropy_mask: Optional[torch.Tensor]

        :returns: Scalar policy loss averaged over valid (and optionally high-entropy) tokens.
        :rtype: torch.Tensor

        **Masking Strategy:**

        The final mask is computed as:
        - If ``entropy_mask`` is provided: ``final_mask = entropy_mask``
          (Note: ``entropy_mask`` is already created considering ``action_mask`` in
          ``create_high_entropy_mask``, so padding positions are already excluded)
        - Else: ``final_mask = action_mask``

        Only tokens where ``final_mask == 1`` contribute to the loss computation.

        **Algorithm Details:**

        - **PPO**: Uses symmetric clipping ``[1 - clip_eps, 1 + clip_eps]`` on the policy ratio.
        - **CPGD**: Uses asymmetric clipping with log-space bounds for better stability.
        - **GSPO**: Uses sequence-level importance ratios with stop-gradient decoupling.
        - **GMPO**: Uses geometric-mean importance ratios with signed log-space clipping.
        """
        if entropy_mask is not None:
            final_mask = entropy_mask
        else:
            final_mask = action_mask

        if self.use_gspo:
            return self._compute_gspo_loss(log_probs, old_log_probs, advantages, final_mask)

        if self.use_gmpo:
            return self._compute_gmpo_loss(log_probs, old_log_probs, advantages, final_mask)

        if self.use_cpg_loss:
            clipped_log_probs = torch.where(
                advantages > 0, torch.clamp(log_probs, max=torch.log(torch.tensor(1 + self.clip_eps)) + old_log_probs),
                torch.clamp(log_probs, min=torch.log(torch.tensor(1 - self.clip_eps)) + old_log_probs)
            )
            loss = -clipped_log_probs * advantages
            loss = masked_mean(loss, final_mask, dim=-1).mean()
            return loss

        # PPO loss
        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        loss = masked_mean(loss, final_mask, dim=-1).mean()

        return loss


class ValueLoss(nn.Module):
    """
    Value Loss for PPO
    """
    def __init__(self, clip_eps: float = None) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute PPO value function loss with optional clipping.

        :param values: Current value predictions.
        :type values: torch.Tensor
        :param old_values: Value predictions from old policy (for clipping).
        :type old_values: torch.Tensor
        :param returns: Target return values (e.g., GAE returns).
        :type returns: torch.Tensor
        :param action_mask: Optional mask for valid timesteps (1 = valid, 0 = ignore).
        :type action_mask: Optional[torch.Tensor]
        :return: Scalar value loss (0.5 * MSE).
        :rtype: torch.Tensor
        """
        if self.clip_eps is not None:
            values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
            surr1 = (values_clipped - returns) ** 2
            surr2 = (values - returns) ** 2
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2

        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return 0.5 * loss


class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """
    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute pairwise ranking loss.

        :param chosen_reward: Reward scores for chosen/preferred samples.
        :type chosen_reward: torch.Tensor
        :param reject_reward: Reward scores for rejected samples.
        :type reject_reward: torch.Tensor
        :param margin: Optional margin value to enforce separation.
        :type margin: Optional[torch.Tensor]
        :return: Mean negative log-sigmoid loss.
        :rtype: torch.Tensor
        """
        if margin is not None:
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return loss.mean()


class DPOLoss(nn.Module):
    """
    DPO Loss
    """
    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute DPO (Direct Preference Optimization) loss.

        :param policy_chosen_logps: Log probabilities under policy for chosen samples.
        :type policy_chosen_logps: torch.Tensor
        :param policy_rejected_logps: Log probabilities under policy for rejected samples.
        :type policy_rejected_logps: torch.Tensor
        :param reference_chosen_logps: Log probabilities under reference model for chosen samples.
        :type reference_chosen_logps: torch.Tensor
        :param reference_rejected_logps: Log probabilities under reference model for rejected samples.
        :type reference_rejected_logps: torch.Tensor
        :return: Tuple of (loss, chosen_rewards, rejected_rewards).
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO
            # (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing) -
                F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        loss = losses.mean()
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards


# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L742
class VanillaKTOLoss(nn.Module):
    """
    KTO loss for even sampling
    """
    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute vanilla KTO loss for evenly sampled chosen/rejected pairs.

        :param policy_chosen_logps: Log probabilities under policy for chosen samples.
        :type policy_chosen_logps: torch.FloatTensor
        :param policy_rejected_logps: Log probabilities under policy for rejected samples.
        :type policy_rejected_logps: torch.FloatTensor
        :param reference_chosen_logps: Log probabilities under reference model for chosen samples.
        :type reference_chosen_logps: torch.FloatTensor
        :param reference_rejected_logps: Log probabilities under reference model for rejected samples.
        :type reference_rejected_logps: torch.FloatTensor
        :return: Tuple of (losses, chosen_rewards, rejected_rewards).
        :rtype: Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]
        """
        chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
        rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        losses = torch.cat(
            (
                1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
            ),
            0,
        ).mean()

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        return losses, chosen_rewards, rejected_rewards


# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L770
class KTOLoss(nn.Module):
    """
    KTO loss for uneven sampling
    """
    def __init__(
        self, beta: float, desirable_weight: float, undesirable_weight: float, world_size: int, device: torch.device
    ) -> None:
        super().__init__()
        self.beta = beta
        self.world_size = world_size
        self.device = device
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute KTO loss for unevenly sampled chosen/rejected pairs with distributed KL estimation.

        :param policy_chosen_logps: Log probabilities under policy for chosen samples.
        :type policy_chosen_logps: torch.FloatTensor
        :param policy_rejected_logps: Log probabilities under policy for rejected samples.
        :type policy_rejected_logps: torch.FloatTensor
        :param policy_KL_logps: Log probabilities under policy for KL estimation samples.
        :type policy_KL_logps: torch.FloatTensor
        :param reference_chosen_logps: Log probabilities under reference model for chosen samples.
        :type reference_chosen_logps: torch.FloatTensor
        :param reference_rejected_logps: Log probabilities under reference model for rejected samples.
        :type reference_rejected_logps: torch.FloatTensor
        :param reference_KL_logps: Log probabilities under reference model for KL estimation samples.
        :type reference_KL_logps: torch.FloatTensor
        :return: Tuple of (losses, chosen_rewards, rejected_rewards, KL).
        :rtype: Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]
        """
        KL = (policy_KL_logps - reference_KL_logps).mean().detach()
        # all_reduce sums up the KL estimates across all devices (gradient will also be scaled by world size)
        dist.all_reduce(KL, op=dist.ReduceOp.SUM)
        # take average (will also scale gradients appropriately)
        KL = (KL / self.world_size).clamp(min=0)

        if policy_chosen_logps.shape[0] != 0:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - KL))
            chosen_rewards = self.beta * chosen_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            chosen_losses = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)
            chosen_rewards = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)

        if policy_rejected_logps.shape[0] != 0:
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            rejected_losses = 1 - F.sigmoid(self.beta * (KL - rejected_logratios))
            rejected_rewards = self.beta * rejected_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            rejected_losses = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)
            rejected_rewards = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)

        losses = torch.cat((self.desirable_weight * chosen_losses, self.undesirable_weight * rejected_losses), 0).mean()
        return losses, chosen_rewards, rejected_rewards, KL


# Adapted from https://github.com/microsoft/LMOps/blob/main/minillm/finetune.py#L166
class KDLoss(nn.Module):
    """
    Language Model Knowledge Distillation Loss
    """
    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100

    def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Compute knowledge distillation loss.

        :param logits: Student model logits.
        :type logits: torch.Tensor
        :param teacher_logits: Teacher model logits (detached).
        :type teacher_logits: torch.Tensor
        :param label: Ground truth labels (tokens to ignore set to IGNORE_INDEX).
        :type label: torch.Tensor
        :return: Scalar KD loss.
        :rtype: torch.Tensor
        """
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(logits)
        logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (label != self.IGNORE_INDEX).int()
        distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

        return distil_loss


class PRMLoss(nn.Module):
    """
    Process Reward Model Loss
    """
    def __init__(self, placeholder_token_id: int, reward_token_ids: Optional[list[int]] = None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)
        self.placeholder_token_id = placeholder_token_id
        self.reward_token_ids = reward_token_ids

    def forward(self,
                inputs: torch.Tensor,
                logits: torch.Tensor,
                labels: torch.Tensor,
                *,
                return_acc: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute process reward model loss.

        :param inputs: Input token IDs (used to locate placeholder tokens).
        :type inputs: torch.Tensor
        :param logits: Model output logits.
        :type logits: torch.Tensor
        :param labels: Target labels (hard or soft labels for reward tokens).
        :type labels: torch.Tensor
        :param return_acc: If True, also return accuracy.
        :type return_acc: bool
        :return: Loss tensor or tuple of (loss, accuracy) if return_acc=True.
        :rtype: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        """
        placeholder_mask = inputs == self.placeholder_token_id
        logits = logits[placeholder_mask]
        labels = labels[placeholder_mask]

        if labels.dtype == torch.float:
            # soft label
            assert len(self.reward_token_ids) == 2, "reward_token_ids should have 2 tokens for soft labels"
            logits = logits[..., self.reward_token_ids]
            positive_labels = labels.to(logits.dtype)
            negative_labels = 1 - positive_labels
            negative_labels[positive_labels != -100] = 1 - positive_labels[positive_labels != -100]
            labels = torch.stack([positive_labels, negative_labels], dim=-1)
        elif self.reward_token_ids is not None:
            # hard label with reward_token_ids set. (otherwise the whole vocab will be trained together.)
            logits = logits[..., self.reward_token_ids]
            # this is slow....
            for i, token in enumerate(self.reward_token_ids):
                labels = torch.where(labels == token, i, labels)

        loss = self.loss(logits, labels)
        if not return_acc:
            return loss

        if labels.dtype == logits.dtype:
            labels = labels.argmax(dim=-1)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, acc


class LogSigmoidLoss(nn.Module):
    """
    Pairwise preference loss for scalar reward models using the log-sigmoid objective.

    Encourages the chosen sample to have a higher reward than the rejected
    sample. Optionally supports a non-negative margin.
    """
    def forward(
        self,
        chosen_reward: torch.Tensor,
        reject_reward: torch.Tensor,
        margin: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute log-sigmoid pairwise loss.

        :param chosen_reward: Predicted reward for the preferred (chosen) sample.
        :type chosen_reward: torch.Tensor
        :param reject_reward: Predicted reward for the rejected sample.
        :type reject_reward: torch.Tensor
        :param margin: Optional non-negative margin. If provided, the objective
            becomes ``logsigmoid(chosen - reject - margin)``. Supports
            broadcasting across batch dimensions.
        :type margin: Optional[torch.Tensor]

        :returns: Mean negative log-sigmoid loss over the batch.
        :rtype: torch.Tensor
        """
        if margin is not None:
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return loss.mean()


class LogExpLoss(nn.Module):
    """
    Log-exp (softplus) pairwise loss for scalar reward model training.

    This loss corresponds to ``log(1 + exp(reject - chosen))`` averaged over
    the batch. See: https://arxiv.org/abs/2204.05862
    """
    def forward(
        self,
        chosen_reward: torch.Tensor,
        reject_reward: torch.Tensor,
        margin: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute log-exp pairwise loss.

        :param chosen_reward: Predicted reward for the preferred (chosen) sample.
        :type chosen_reward: torch.Tensor
        :param reject_reward: Predicted reward for the rejected sample.
        :type reject_reward: torch.Tensor
        :param margin: Unused; included for API compatibility with
            :class:`PairWiseLoss`.
        :type margin: Optional[torch.Tensor]

        :returns: Mean ``log(1 + exp(reject - chosen))`` over the batch.
        :rtype: torch.Tensor
        """
        loss = torch.log(1 + torch.exp(reject_reward - chosen_reward)).mean()
        return loss


class HPSLoss(nn.Module):
    """
    Human Preference Score (HPS) Loss for scalar reward model training.
    Implements the cross-entropy loss over the logits formed by concatenating
    the chosen and rejected rewards. The core idea is to treat the preference
    prediction as binary classification task.

    Paper: https://arxiv.org/abs/2303.14420
    """
    def forward(
        self,
        chosen_reward: torch.Tensor,
        reject_reward: torch.Tensor,
        margin: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute HPS loss.

        :param chosen_reward: Predicted reward for the preferred (chosen) sample.
        :type chosen_reward: torch.Tensor
        :param reject_reward: Predicted reward for the rejected sample.
        :type reject_reward: torch.Tensor
        :param margin: Unused; included for API compatibility with
            :class:`PairWiseLoss`.
        :type margin: Optional[torch.Tensor]

        :returns: Mean cross-entropy loss over the batch.
        :rtype: torch.Tensor
        """
        logits = torch.cat([chosen_reward, reject_reward], dim=-1)
        labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        loss = F.cross_entropy(logits, labels)
        return loss
