"""
Loss functions used across LightRFT models.

This module implements:

- GPTLMLoss:  Next-token prediction for generative reward model training.
- LogSigmoidLoss: Log-sigmoid pairwise loss for scalar reward model training.
- LogExpLoss: Log-exp pairwise loss for scalar reward model training.
- HPSLoss: Human Preference Score loss for scalar reward model training.
"""

from typing import Optional, Tuple

import numpy as np
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
    Enhanced Policy Loss for PPO with multiple aggregation modes.
    Supports Dr.GRPO's "seq-mean-token-sum-norm" mode and GSPO (Group Sequence Policy Optimization).

    This class implements the Proximal Policy Optimization (PPO) policy loss with support
    for multiple aggregation strategies. It includes the specialized "seq-mean-token-sum-norm"
    mode used in Dr.GRPO, which normalizes the total loss by the maximum number of tokens.
    The class also supports GSPO mode, which uses sequence-level importance ratios and
    optimization for improved stability, particularly effective for training large language
    models and Mixture-of-Experts (MoE) models.

    References:
        [1] Dr.GRPO: https://arxiv.org/pdf/2503.20783
        [2] GSPO: https://arxiv.org/pdf/2507.18071
        [3] GSPO Reference Implementation: https://github.com/vivekvar-dl/GSPO-DeepSeek-R1-Distill-Qwen-1.5B
    """

    VALID_MODES = ["token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"]

    def __init__(
        self,
        clip_eps: float = 0.2,
        use_dapo: bool = False,
        use_cpg_loss: bool = False,
        use_gmpo: bool = False,
        max_tokens: int = 4096,
        loss_agg_mode: str = "seq-mean-token-mean",
        use_gspo: bool = False,
        normalize_advantages: bool = True,
        use_sequence_rewards: bool = True
    ) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.use_dapo = use_dapo
        self.use_cpg_loss = use_cpg_loss
        self.use_gmpo = use_gmpo
        self.max_tokens = max_tokens
        self.loss_agg_mode = loss_agg_mode
        self.use_gspo = use_gspo
        self.normalize_advantages = normalize_advantages
        self.use_sequence_rewards = use_sequence_rewards

        if loss_agg_mode not in self.VALID_MODES:
            raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}. Valid: {self.VALID_MODES}")

    def _masked_mean(self, values: torch.Tensor, mask: Optional[torch.Tensor], dim: int, eps: float = 1e-8) -> torch.Tensor:
        if mask is None:
            return values.mean(dim=dim)
        return (values * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=eps)

    def _ensure_token_advantages(self, advantages: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        if advantages.dim() == 1:
            return advantages.unsqueeze(-1).expand(target_shape)
        if advantages.dim() == 2:
            return advantages
        raise ValueError(f"Unexpected advantages shape: {advantages.shape}")

    def _maybe_normalize_advantages(
        self,
        token_advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.normalize_advantages:
            return token_advantages
        if action_mask is not None:
            masked_adv = torch.masked_select(token_advantages, action_mask.bool())
            adv_mean = masked_adv.mean()
            adv_std = masked_adv.std()
        else:
            adv_mean = token_advantages.mean()
            adv_std = token_advantages.std()
        if adv_std > 1e-8:
            return (token_advantages - adv_mean) / (adv_std + 1e-8)
        return token_advantages - adv_mean

    def _aggregate_token_loss(self, token_losses: torch.Tensor, action_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.loss_agg_mode == "token-mean":
            if action_mask is not None:
                return (token_losses * action_mask).sum() / action_mask.sum().clamp(min=1e-6)
            return token_losses.mean()
        if self.loss_agg_mode == "seq-mean-token-sum":
            if action_mask is not None:
                seq_losses = torch.sum(token_losses * action_mask, dim=-1)
            else:
                seq_losses = torch.sum(token_losses, dim=-1)
            return torch.mean(seq_losses)
        if self.loss_agg_mode == "seq-mean-token-mean":
            if action_mask is not None:
                token_sums = torch.sum(token_losses * action_mask, dim=-1)
                token_counts = torch.sum(action_mask, dim=-1)
                seq_losses = token_sums / token_counts.clamp(min=1e-6)
            else:
                seq_losses = token_losses.mean(dim=-1)
            return torch.mean(seq_losses)
        if self.loss_agg_mode == "seq-mean-token-sum-norm":  # Dr.GRPO
            if action_mask is not None:
                seq_losses = torch.sum(token_losses * action_mask, dim=-1)
            else:
                seq_losses = torch.sum(token_losses, dim=-1)
            total_loss = torch.sum(seq_losses)
            return total_loss / torch.tensor(float(self.max_tokens), device=total_loss.device)
        return masked_mean(token_losses, action_mask, dim=-1).mean()

    def _compute_clipped_surrogate_loss(
        self,
        ratio: torch.Tensor,
        clipped_ratio: torch.Tensor,
        token_advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        surr1 = ratio * token_advantages
        surr2 = clipped_ratio * token_advantages
        token_losses = -torch.min(surr1, surr2)
        return self._aggregate_token_loss(token_losses, action_mask)

    def _ratio_and_clipped_from_log_ratio(self, log_ratio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Clamp log-ratio before exp to avoid NaN gradients:
        # https://github.com/pytorch/pytorch/issues/10729
        log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
        ratio = torch.exp(log_ratio)
        clip_ratio_low_log = np.log(1.0 - self.clip_eps)
        clip_ratio_high_log = np.log(1.0 + self.clip_eps)
        clipped_ratio = torch.exp(torch.clamp(log_ratio, clip_ratio_low_log, clip_ratio_high_log))
        return ratio, clipped_ratio

    def _compute_gspo_log_ratio(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        action_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # EasyR1 GSPO token mode: detach sequence-level KL, keep token-level gradients.
        negative_approx_kl = log_probs - old_log_probs
        seq_avg_kl = self._masked_mean(negative_approx_kl, action_mask, dim=-1)
        return seq_avg_kl.detach().unsqueeze(-1) + (log_probs - log_probs.detach())

    def _prepare_gspo_advantages(
        self,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor],
        ratio_shape: torch.Size,
        sequence_rewards: Optional[torch.Tensor],
    ) -> torch.Tensor:
        token_advantages = self._ensure_token_advantages(advantages, ratio_shape)
        if self.use_sequence_rewards and sequence_rewards is not None:
            token_advantages = sequence_rewards.unsqueeze(-1).expand_as(token_advantages)
        return self._maybe_normalize_advantages(token_advantages, action_mask)

    def _gmpo_advantage_sign(self, advantages: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        token_advantages = self._ensure_token_advantages(advantages, target_shape)
        return torch.where(token_advantages >= 0, -torch.ones_like(token_advantages), torch.ones_like(token_advantages))

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        sequence_rewards: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute PPO policy loss with optional clipping or CPG variant.

        :param log_probs: Log probabilities of actions under current policy.
        :type log_probs: torch.Tensor
        :param old_log_probs: Log probabilities of actions under old policy.
        :type old_log_probs: torch.Tensor
        :param advantages: Estimated advantages for each action.
        :type advantages: torch.Tensor
        :param action_mask: Optional mask for valid actions (1 = valid, 0 = ignore).
        :type action_mask: Optional[torch.Tensor]
        :return: Scalar policy loss.
        :rtype: torch.Tensor
        """
        if self.use_cpg_loss:
            clipped_log_probs = torch.where(
                advantages > 0, torch.clamp(log_probs, max=torch.log(torch.tensor(1 + self.clip_eps)) + old_log_probs),
                torch.clamp(log_probs, min=torch.log(torch.tensor(1 - self.clip_eps)) + old_log_probs)
            )
            loss = -clipped_log_probs * advantages
            loss = (loss * action_mask).sum() / action_mask.sum()
            return loss

        # GSPO mode: sequence-level optimization
        # Reference implementation: EasyR1 (https://github.com/vivekvar-dl/distill-grpo/EasyR1)
        if self.use_gspo:
            log_importance_ratio = self._compute_gspo_log_ratio(log_probs, old_log_probs, action_mask)
            ratio, clipped_ratio = self._ratio_and_clipped_from_log_ratio(log_importance_ratio)
            token_advantages = self._prepare_gspo_advantages(
                advantages, action_mask, ratio.shape, sequence_rewards
            )
            return self._compute_clipped_surrogate_loss(ratio, clipped_ratio, token_advantages, action_mask)

        # GMPO (Generalized Mirror Policy Optimization) implementation
        if self.use_gmpo:
            logprobs_diff = log_probs - old_log_probs
            sgn_advantage = self._gmpo_advantage_sign(advantages, logprobs_diff.shape)
            sgn_logprobs_diff = sgn_advantage * logprobs_diff
            sgn_logprobs_diff_clamp = torch.clamp(sgn_logprobs_diff, -self.clip_eps, self.clip_eps)
            logprobs_diff_max = sgn_advantage * torch.max(sgn_logprobs_diff, sgn_logprobs_diff_clamp)

            seq_logprobs_diff_max = self._masked_mean(logprobs_diff_max, action_mask, dim=-1)
            ratio = torch.exp(seq_logprobs_diff_max)

            seq_advantages = self._masked_mean(
                self._ensure_token_advantages(advantages, logprobs_diff.shape), action_mask, dim=-1
            )
            return torch.mean(-seq_advantages * ratio)

        # Standard PPO/GRPO modes: token-level importance ratios
        log_importance_ratio = log_probs - old_log_probs
        ratio, clipped_ratio = self._ratio_and_clipped_from_log_ratio(log_importance_ratio)
        return self._compute_clipped_surrogate_loss(ratio, clipped_ratio, advantages, action_mask)


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

    def forward(self, inputs: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, *, return_acc: bool = False):
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
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
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
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
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
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
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
