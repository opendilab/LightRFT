"""URSA paper Eq.9 strict-alignment advantage estimator.

Paper: arXiv 2501.04686 (NeurIPS 2025), Appendix B.1 Eq.9 — the second
straw-man variant the paper considers (and ultimately *rejects* in favour
of PS-GRPO):

    A_t^i = r_{s,t}^i * GroupNorm_G(r̄_s^i)            (process-reward term)
          +              GroupNorm_G(r_o^i)             (outcome-reward term)

where t indexes *steps* (not tokens), r_{s,t}^i is the sigmoid PRM score for
step t in trajectory i, r̄_s^i = mean_t r_{s,t}^i is the per-trajectory mean
PRM score, r_o^i ∈ {0,1} is the outcome reward, and GroupNorm_G is
(x - mean_G(x)) / std_G(x) over the G=K trajectories sampled from the same
prompt. The token-level A_t is broadcast to every token spanned by step t.

This file is intentionally self-contained in ``examples/math_prm/`` and does
**not** modify any code under ``lightrft/``. It registers a new estimator
``ursa_variant2`` by monkey-patching
``lightrft.trainer.advantage_calculator.get_advantage_calculator`` at import
time. The patch is idempotent.

Why a separate path, not a flag on the existing per-step PRM path:
the legacy ``per_step_reward_mode`` path (still useful as Math-Shepherd-style
step-MC return) goes through ``compute_reward`` Mode B + reverse-cumsum +
GroupNormCalculator. That fully bypasses the outcome reward and uses
cumulative returns, both of which contradict Eq.9. Keeping the two paths
side by side allows ablation between paper-strict (this estimator) and
Math-Shepherd-style (legacy).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from lightrft.trainer.advantage_calculator import (
    AdvantageCalculator,
    compute_clip_fraction,
)


class UrsaVariant2Calculator(AdvantageCalculator):
    """Strict paper Eq.9 implementation.

    Reads per-trajectory step PRM scores and outcome reward from
    ``experience.info`` (already emitted by ``MathPRMReward.forward`` for
    label ``"math_per_step_prm"``), does its own GroupNorm, and writes
    a per-token advantage tensor where every token within the span of
    step k carries A_k. No cumulative return. ``returns`` mirrors
    ``advantages`` (no separate value function).
    """

    _ESTIMATOR_NAME = "ursa_variant2"

    def preprocess_rewards(
        self,
        rewards: torch.Tensor,
        experiences: List,
        max_new_tokens: int,
    ) -> Tuple[List, List[torch.Tensor]]:
        """Compute GroupNormed (r̄_s, r_o) across all G trajectories.

        ``rewards`` is the concatenated per-trajectory scalar reward from
        every experience in the batch — for ``math_per_step_prm`` rows this
        is ``outcome_correct ∈ {0,1}`` (see ``reward_models.py:655``).
        ``experiences`` lets us gather per-trajectory step_rewards needed
        to compute r̄_s^i. We write the GroupNormed values back into each
        experience's ``info`` under reserved keys ``_ursa_oc_normed`` and
        ``_ursa_msp_normed`` so ``compute()`` can pick them up later.

        Aborts the variant-2 path (returns identity-chunked rewards
        without touching info) when ``n_samples_per_prompt < 2`` — a
        single-sample group has std = 0 and ``A_t`` would collapse.
        """
        config = self.config
        n_samples = int(getattr(config, "n_samples_per_prompt", 1) or 1)

        # Identity preprocessing if K<2 — variant 2 needs a group to normalize.
        # We still chunk rewards back so the downstream contract is preserved.
        if n_samples < 2:
            reward_chunks = rewards.chunk(len(experiences)) if len(experiences) > 0 else []
            return experiences, list(reward_chunks)

        device = rewards.device
        total_B = rewards.numel()
        if total_B % n_samples != 0:
            # Cannot group — bail out gracefully, fall back to identity.
            reward_chunks = rewards.chunk(len(experiences))
            return experiences, list(reward_chunks)

        # Compute r̄_s^i for every trajectory across experiences.
        mean_step_prm_chunks: List[torch.Tensor] = []
        per_exp_traj_counts: List[int] = []
        for exp in experiences:
            sr_list = exp.info.get("step_rewards")
            n_traj = int(exp.info["reward"].numel())
            per_exp_traj_counts.append(n_traj)
            if sr_list is None:
                # No per-step data — treat r̄_s as zero (variant 2 will rely
                # on the outcome-norm anchor only for that trajectory).
                mean_step_prm_chunks.append(
                    torch.zeros(n_traj, dtype=torch.float32, device=device)
                )
                continue
            means: List[torch.Tensor] = []
            for sr in sr_list:
                if sr.numel() > 0:
                    means.append(sr.to(device=device, dtype=torch.float32).mean())
                else:
                    means.append(torch.zeros((), dtype=torch.float32, device=device))
            if len(means) != n_traj:
                # Misaligned bookkeeping — fall back per-traj to zero.
                pad = [torch.zeros((), dtype=torch.float32, device=device)] * (
                    n_traj - len(means)
                )
                means = means + pad
                means = means[:n_traj]
            mean_step_prm_chunks.append(torch.stack(means).to(device=device))
        mean_step_prm = torch.cat(mean_step_prm_chunks, dim=0)  # (total_B,)

        # GroupNorm both terms across G=K siblings (paper Eq.9 footer text).
        oc_flat = rewards.to(device=device, dtype=torch.float32)
        oc_g = oc_flat.reshape(-1, n_samples)
        oc_normed = (
            (oc_g - oc_g.mean(dim=-1, keepdim=True))
            / (oc_g.std(dim=-1, unbiased=False, keepdim=True) + 1e-9)
        ).flatten()

        msp_g = mean_step_prm.reshape(-1, n_samples)
        msp_normed = (
            (msp_g - msp_g.mean(dim=-1, keepdim=True))
            / (msp_g.std(dim=-1, unbiased=False, keepdim=True) + 1e-9)
        ).flatten()

        # Scatter normed values back per-experience (keep CPU-side view to avoid
        # device contention when compute() runs in a different stream).
        offset = 0
        for exp, n_traj in zip(experiences, per_exp_traj_counts):
            exp.info["_ursa_oc_normed"] = oc_normed[offset:offset + n_traj].clone().cpu()
            exp.info["_ursa_msp_normed"] = msp_normed[offset:offset + n_traj].clone().cpu()
            exp.info["_ursa_mean_step_prm_raw"] = mean_step_prm[offset:offset + n_traj].clone().cpu()
            offset += n_traj

        # Default behaviour: chunk the (unmodified) per-trajectory rewards back
        # to per-experience tensors — ``compute()`` ignores this anyway.
        reward_chunks = oc_flat.chunk(len(experiences)) if len(experiences) > 0 else []
        return experiences, list(reward_chunks)

    def compute(
        self,
        experience,
        final_reward: torch.Tensor,
        gamma: Optional[float],
        generate_kwargs: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Build per-token advantages via paper Eq.9.

        Ignores ``final_reward`` (which carries the legacy Mode B step
        scatter + KL — orthogonal to Eq.9). KL is still applied separately
        by the surrounding ``--use_kl_loss`` path; we only own the
        advantage shape here.
        """
        action_mask = experience.action_mask
        if action_mask is None:
            raise ValueError(
                "UrsaVariant2Calculator requires action_mask (token-level "
                "broadcast over step spans is undefined without it)."
            )

        device = action_mask.device
        B, T = action_mask.shape

        info = experience.info
        oc_normed = info.get("_ursa_oc_normed")
        msp_normed = info.get("_ursa_msp_normed")
        if oc_normed is None or msp_normed is None:
            # preprocess_rewards bailed (K<2 or shape mismatch) — fall back
            # to a degenerate advantage tensor of zeros so loss stays finite.
            advantages = torch.zeros(B, T, device=device, dtype=torch.float32)
            returns = advantages.clone()
            return advantages, returns, {"ursa_v2_fallback_used": 1.0}

        oc_normed = oc_normed.to(device=device, dtype=torch.float32)
        msp_normed = msp_normed.to(device=device, dtype=torch.float32)

        step_rewards_list = info.get("step_rewards") or []
        step_indices_list = info.get("step_token_indices") or []

        advantages = torch.zeros(B, T, device=device, dtype=torch.float32)
        per_traj_step_count = []
        for i in range(B):
            has_steps = (
                i < len(step_rewards_list)
                and step_rewards_list[i].numel() > 0
                and i < len(step_indices_list)
                and step_indices_list[i].numel() == step_rewards_list[i].numel()
            )
            if not has_steps:
                # No step data — degenerate to outcome-only term spread over
                # the response (matches paper's natural limit when n_steps=0
                # since the process-reward term vanishes).
                advantages[i] = oc_normed[i] * action_mask[i].to(torch.float32)
                per_traj_step_count.append(0)
                continue

            sr = step_rewards_list[i].to(device=device, dtype=torch.float32)  # (n_steps,)
            si = step_indices_list[i].to(device=device, dtype=torch.long)      # (n_steps,) END idx
            n_steps = sr.numel()
            per_traj_step_count.append(int(n_steps))

            # Span starts: 0 for step 0, end_{k-1}+1 for k > 0
            starts = torch.cat([
                torch.zeros(1, dtype=torch.long, device=device),
                si[:-1] + 1,
            ])
            ends = si

            # Per-step advantage: A_k = r_{s,k} * msp_normed[i] + oc_normed[i]
            #                       (paper Eq.9)
            A_steps = sr * msp_normed[i] + oc_normed[i]   # (n_steps,)

            for k in range(n_steps):
                sk = max(0, int(starts[k].item()))
                ek = min(T - 1, int(ends[k].item()))
                if sk > ek:
                    continue
                advantages[i, sk:ek + 1] = A_steps[k]

            # Tokens past the last step boundary (e.g. final `†Answer:` line
            # tokens) are not covered by any step. Per paper Eq.9 the second
            # term is t-independent, so we still apply oc_normed[i] there
            # to give the model an outcome-only signal on the tail. This
            # matches the implicit reading that the outcome anchor lives
            # on the whole trajectory while step rewards live on steps.
            last_end = int(ends[-1].item()) if n_steps > 0 else -1
            if last_end + 1 < T:
                advantages[i, last_end + 1:] = oc_normed[i]

        # Respect the response action mask everywhere.
        advantages = advantages * action_mask.to(torch.float32)
        returns = advantages.clone()

        # Per-step credit diagnostics (these flow into the trainer's wandb
        # under `train/`-style keys via the existing info_dict pipeline).
        n_valid = action_mask.sum().clamp(min=1).to(torch.float32)
        info_dict: Dict[str, float] = {
            "ursa_v2_adv_pos_frac": (advantages > 0).to(torch.float32).sum().item() / n_valid.item(),
            "ursa_v2_adv_neg_frac": (advantages < 0).to(torch.float32).sum().item() / n_valid.item(),
            "ursa_v2_adv_zero_frac": (advantages == 0).to(torch.float32).sum().item() / n_valid.item(),
            "ursa_v2_adv_abs_mean": advantages.abs().sum().item() / n_valid.item(),
            "ursa_v2_oc_normed_std": oc_normed.std(unbiased=False).item() if oc_normed.numel() > 1 else 0.0,
            "ursa_v2_msp_normed_std": msp_normed.std(unbiased=False).item() if msp_normed.numel() > 1 else 0.0,
            "ursa_v2_traj_step_count_mean": (
                sum(per_traj_step_count) / max(1, len(per_traj_step_count))
            ),
        }

        # Advantage clipping (config knob, optional).
        if getattr(self.config, "advantage_clip", 0) > 0:
            clip_val = self.config.advantage_clip
            info_dict["advantage_clip_frac"] = compute_clip_fraction(
                advantages, clip_val, -clip_val
            )
            advantages = torch.clamp(advantages, -clip_val, clip_val)

        return advantages, returns, info_dict


def _install_get_advantage_calculator_patch() -> None:
    """Idempotently inject ``ursa_variant2`` into lightrft's calculator factory.

    Done from examples/ rather than editing ``lightrft/`` to keep the new
    estimator strictly contained in this example. The patch wraps the
    original factory; unknown names still raise the original ValueError
    listing the *original* supported set + this estimator.

    Important: we patch every module that has already done
    ``from .advantage_calculator import get_advantage_calculator`` because
    those imports bind the original function object into the consumer
    module's namespace — patching just the source module would miss them.
    """
    from lightrft.trainer import advantage_calculator as _ac

    if getattr(_ac.get_advantage_calculator, "_ursa_v2_patched", False):
        return

    _original = _ac.get_advantage_calculator

    def get_advantage_calculator_patched(estimator_name: str, config):
        if estimator_name == UrsaVariant2Calculator._ESTIMATOR_NAME:
            return UrsaVariant2Calculator(config)
        return _original(estimator_name, config)

    get_advantage_calculator_patched._ursa_v2_patched = True
    _ac.get_advantage_calculator = get_advantage_calculator_patched

    # Also patch known consumers that did ``from .advantage_calculator import
    # get_advantage_calculator`` (binding the original ref into their own
    # namespace). Currently fast_exp_maker is the only such consumer; if more
    # appear later, list them here.
    import sys
    for mod_name in ("lightrft.trainer.fast_exp_maker",):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "get_advantage_calculator"):
            mod.get_advantage_calculator = get_advantage_calculator_patched


# Install on import. Both ``examples/math_prm/math_prm_trainer.py`` and
# ``examples/math_prm/train_colocate.py`` import this module at top level so
# the patch is in place before fast_exp_maker constructs its calculator.
_install_get_advantage_calculator_patch()
