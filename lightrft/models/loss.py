"""
Loss functions used across LightRFT models.

This module implements:

- GPTLMLoss:  Next-token prediction for generative reward model training.
- LogSigmoidLoss: Log-sigmoid pairwise loss for scalar reward model training.
- LogExpLoss: Log-exp pairwise loss for scalar reward model training.
- HPSLoss: Human Preference Score loss for scalar reward model training.
"""

from typing import Optional, Tuple

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

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        sequence_rewards: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_cpg_loss:
            clipped_log_probs = torch.where(
                advantages > 0, torch.clamp(log_probs, max=torch.log(torch.tensor(1 + self.clip_eps)) + old_log_probs),
                torch.clamp(log_probs, min=torch.log(torch.tensor(1 - self.clip_eps)) + old_log_probs)
            )
            loss = -clipped_log_probs * advantages
            loss = (loss * action_mask).sum() / action_mask.sum()
            return loss
        
        # GSPO mode: sequence-level optimization
        if self.use_gspo:
            # Compute sequence-level log probabilities
            if action_mask is not None:
                seq_log_probs = torch.sum(log_probs * action_mask, dim=-1)  # [batch_size]
                seq_old_log_probs = torch.sum(old_log_probs * action_mask, dim=-1)  # [batch_size]
            else:
                seq_log_probs = torch.sum(log_probs, dim=-1)  # [batch_size]
                seq_old_log_probs = torch.sum(old_log_probs, dim=-1)  # [batch_size]
            
            # Compute sequence-level importance ratio
            seq_ratio = (seq_log_probs - seq_old_log_probs).exp()  # [batch_size]
            
            # Handle advantages
            if advantages.dim() == 2:  # Token-level advantages
                if action_mask is not None:
                    seq_advantages = torch.sum(advantages * action_mask, dim=-1) / torch.sum(action_mask, dim=-1).clamp(min=1e-6)
                else:
                    seq_advantages = torch.mean(advantages, dim=-1)
            else:  # Already sequence-level advantages
                seq_advantages = advantages
            
            # Use sequence rewards if provided and enabled
            if self.use_sequence_rewards and sequence_rewards is not None:
                seq_advantages = sequence_rewards
            
            # Normalize advantages if enabled
            if self.normalize_advantages:
                seq_mean = seq_advantages.mean()
                seq_std = seq_advantages.std()
                # Add numerical stability check
                if seq_std > 1e-8:
                    seq_advantages = (seq_advantages - seq_mean) / seq_std
                else:
                    seq_advantages = seq_advantages - seq_mean
            
            # Apply sequence-level clipping (GSPO's key innovation)
            surr1 = seq_ratio * seq_advantages
            surr2 = seq_ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * seq_advantages
            
            # Compute sequence-level losses
            seq_losses = -torch.min(surr1, surr2)  # [batch_size]
            
            # Aggregate based on loss_agg_mode
            if self.loss_agg_mode == "token-mean":
                return torch.mean(seq_losses)
            elif self.loss_agg_mode == "seq-mean-token-sum":
                return torch.mean(seq_losses)
            elif self.loss_agg_mode == "seq-mean-token-mean":
                return torch.mean(seq_losses)
            elif self.loss_agg_mode == "seq-mean-token-sum-norm":  # Dr.GRPO
                total_loss = torch.sum(seq_losses)
                return total_loss / torch.tensor(float(self.max_tokens), device=total_loss.device)
        
        # GMPO (Generalized Mirror Policy Optimization) implementation
        if self.use_gmpo:
            # GMPO uses sign-aware clipping based on advantage sign
            cliprange = self.clip_eps  # Use clip_eps as cliprange
            low_cliprange = -cliprange
            high_cliprange = cliprange
            
            # Compute logprobs difference
            logprobs_diff = log_probs - old_log_probs
            
            # Determine sign of advantage for each token
            # advantages can be token-level [batch_size, seq_len] or sequence-level [batch_size]
            if advantages.dim() == 1:
                # Sequence-level advantages: expand to token level
                advantage_expanded = advantages.unsqueeze(-1)
                sgn_advantage = torch.where(advantage_expanded >= 0, 
                                            -torch.ones_like(advantage_expanded),
                                            torch.ones_like(advantage_expanded))
            else:
                # Token-level advantages
                sgn_advantage = torch.where(advantages >= 0,
                                            -torch.ones_like(advantages),
                                            torch.ones_like(advantages))
            
            # Apply sign to logprobs_diff
            sgn_logprobs_diff = sgn_advantage * logprobs_diff
            
            # Clamp the signed logprobs_diff
            sgn_logprobs_diff_clamp = torch.clamp(sgn_logprobs_diff, low_cliprange, high_cliprange)
            
            # Take max (this implements the clipping)
            sgn_logprobs_diff_max = torch.max(sgn_logprobs_diff, sgn_logprobs_diff_clamp)
            
            # Restore original sign
            logprobs_diff_max = sgn_advantage * sgn_logprobs_diff_max
            
            # Compute sequence-level ratio: exp(mean of logprobs_diff_max over masked tokens)
            if action_mask is not None:
                # Sum of logprobs_diff_max over masked tokens, divided by count
                masked_logprobs_diff_max = logprobs_diff_max * action_mask
                seq_logprobs_diff_max = torch.sum(masked_logprobs_diff_max, dim=-1) / torch.sum(action_mask, dim=-1).clamp(min=1e-6)
            else:
                seq_logprobs_diff_max = torch.mean(logprobs_diff_max, dim=-1)
            
            # Compute ratio per sequence
            ratio = torch.exp(seq_logprobs_diff_max)  # [batch_size]
            
            # Get sequence-level advantages
            if advantages.dim() == 2:
                # Token-level advantages: average over masked tokens
                if action_mask is not None:
                    seq_advantages = torch.sum(advantages * action_mask, dim=-1) / torch.sum(action_mask, dim=-1).clamp(min=1e-6)
                else:
                    seq_advantages = torch.mean(advantages, dim=-1)
            else:
                # Already sequence-level
                seq_advantages = advantages
            
            # Compute sequence-level losses: -advantage * ratio
            seq_losses = -seq_advantages * ratio
            
            # Return mean of sequence-level losses
            return torch.mean(seq_losses)

        # Standard PPO/GRPO modes: token-level importance ratios
        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        
        # Convert token-level loss to sequence-level for unified aggregation
        if action_mask is not None:
            seq_losses = torch.sum(loss * action_mask, dim=-1)
        else:
            seq_losses = torch.sum(loss, dim=-1)
        
        # Unified aggregation based on loss_agg_mode
        if self.loss_agg_mode == "token-mean":
            if action_mask is not None:
                return (loss * action_mask).sum() / action_mask.sum().clamp(min=1e-6)
            else:
                return loss.mean()
        elif self.loss_agg_mode == "seq-mean-token-sum":
            return torch.mean(seq_losses)
        elif self.loss_agg_mode == "seq-mean-token-mean":
            if action_mask is not None:
                token_sums = torch.sum(loss * action_mask, dim=-1)
                token_counts = torch.sum(action_mask, dim=-1)
                seq_losses = token_sums / token_counts.clamp(min=1e-6)
            else:
                seq_losses = loss.mean(dim=-1)
            return torch.mean(seq_losses)
        elif self.loss_agg_mode == "seq-mean-token-sum-norm":  # Dr.GRPO
            total_loss = torch.sum(seq_losses)
            return total_loss / torch.tensor(float(self.max_tokens), device=total_loss.device)
        
        # Default fallback (should not reach here if loss_agg_mode is valid)
        return masked_mean(loss, action_mask, dim=-1).mean()


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
