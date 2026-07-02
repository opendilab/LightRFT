from contextlib import contextmanager
from typing import Dict, Optional

import torch

from lightrft.trainer.spmd_ppo_trainer import SPMDPPOTrainerVL

# Explicitly register the ursa_variant2 monkey-patches so
# ``--advantage_estimator ursa_variant2`` resolves to the paper Eq.9
# strict-alignment calculator. train_colocate.py runs with
# cwd=examples/math_prm, so a top-level (non-package) import is used here.
# (register_ursa_variant2() is idempotent.)
from ursa_variant2 import register_ursa_variant2

register_ursa_variant2()


def _detach_rollout_eos_patch(rollout_actor):
    """Detach rollout_eos_patch.StructuredAnswerStoppingCriteria wrap from a rollout actor.

    Returns the unwrapped (original) generate function so the caller can restore
    the patch later. Returns None if no patch is installed.

    The patch wraps ``model.generate`` with ``functools.wraps``, so the original
    function is reachable via ``__wrapped__``. We rely on the patch's idempotency
    flag ``_math_prm_rollout_eos_patch_installed`` to detect installation.
    """
    if rollout_actor is None:
        return None
    model = getattr(rollout_actor, "model", None)
    if model is None:
        return None
    if not getattr(model, "_math_prm_rollout_eos_patch_installed", False):
        return None
    patched = model.generate
    original = getattr(patched, "__wrapped__", None)
    if original is None:
        return None
    model.generate = original
    model._math_prm_rollout_eos_patch_installed = False
    return patched


def _reattach_rollout_eos_patch(rollout_actor, patched_generate):
    """Reinstall a previously detached patched generate function."""
    if rollout_actor is None or patched_generate is None:
        return
    model = getattr(rollout_actor, "model", None)
    if model is None:
        return
    model.generate = patched_generate
    model._math_prm_rollout_eos_patch_installed = True


class MathPRMSPMDPPOTrainerVL(SPMDPPOTrainerVL):
    """SPMD PPO trainer specialized for URSA-MATH process-reward training.

    Differs from the base ``SPMDPPOTrainerVL`` in two ways:

    1. **W&B namespace mapping** — rollout/train/eval metric streams are
       projected into the ``rollout/``, ``train/``, ``eval/`` namespaces via
       ``_ROLLOUT_KEY_SOURCES`` / ``_TRAIN_KEY_SOURCES`` / ``_EVAL_KEY_SOURCES``
       so the dashboards stay aligned with the URSA paper's reporting
       conventions even as upstream metric names drift.
    2. **URSA-specific eval** — :meth:`evaluate` runs under a runtime context
       that prevents the actor from generating ``<|image|>`` sentinel tokens
       and aggregates per-dataset metrics into a single weighted-average
       view (see :meth:`_aggregate_eval_metrics`).

    All checkpoint, logging, profile, and trajectory-saving wiring is unchanged
    from the base class apart from the namespace mapping.
    """

    _ROLLOUT_KEY_SOURCES = {
        "reward": ("rollout_reward", "step_reward_mean", "reward"),
        "reward_std": ("rollout_reward_std", "step_reward_std"),
        "outcome_correct": ("rollout_outcome_correct", "outcome_correct_mean", "reward_metrics/outcome_correct"),
        "has_drop_moment": ("rollout_has_drop_moment", "has_drop_moment_mean", "reward_metrics/has_drop_moment"),
        "model_reward": ("rollout_model_reward", "model_reward_mean", "reward_metrics/model_reward"),
        "response_length": ("rollout_response_length", "response_length_mean", "response_length"),
        # PRM step-score distribution (computed by MathPRMReward.forward but
        # previously not surfaced to wandb; useful for monitoring PRM behaviour
        # at scale, not just min-aggregation).
        "step_score_min": ("rollout_step_score_min", "step_score_min_mean", "reward_metrics/step_score_min"),
        "step_score_mean": ("rollout_step_score_mean", "step_score_mean_mean", "reward_metrics/step_score_mean"),
        "step_score_last": ("rollout_step_score_last", "step_score_last_mean", "reward_metrics/step_score_last"),
        "step_count": ("rollout_step_count", "step_count_mean", "reward_metrics/step_count"),
        # PS-GRPO + answer-extraction diagnostics (already computed, were missing
        # from wandb mapping — required for reward-hacking forensics).
        "final_reward": ("rollout_final_reward", "final_reward_mean", "reward_metrics/final_reward"),
        "max_relative_drop": (
            "rollout_max_relative_drop", "max_relative_drop_mean", "reward_metrics/max_relative_drop"
        ),
        "answer_tag_present": (
            "rollout_answer_tag_present", "answer_tag_present_mean", "reward_metrics/answer_tag_present"
        ),
        "answer_extraction_failed": (
            "rollout_answer_extraction_failed", "answer_extraction_failed_mean",
            "reward_metrics/answer_extraction_failed"
        ),
        "used_answer_fallback": (
            "rollout_used_answer_fallback", "used_answer_fallback_mean", "reward_metrics/used_answer_fallback"
        ),
        "used_mathruler": ("rollout_used_mathruler", "used_mathruler_mean", "reward_metrics/used_mathruler"),
        "reference_supported": (
            "rollout_reference_supported", "reference_supported_mean", "reward_metrics/reference_supported"
        ),
        "answer_content_correct": (
            "rollout_answer_content_correct", "answer_content_correct_mean",
            "reward_metrics/answer_content_correct"
        ),
        "answer_marker_count": (
            "rollout_answer_marker_count", "answer_marker_count_mean", "reward_metrics/answer_marker_count"
        ),
        "extra_answer_marker_count": (
            "rollout_extra_answer_marker_count", "extra_answer_marker_count_mean",
            "reward_metrics/extra_answer_marker_count"
        ),
        "post_answer_continuation": (
            "rollout_post_answer_continuation", "post_answer_continuation_mean",
            "reward_metrics/post_answer_continuation"
        ),
        "post_answer_step_present": (
            "rollout_post_answer_step_present", "post_answer_step_present_mean",
            "reward_metrics/post_answer_step_present"
        ),
        "answer_tail_cutoff_present": (
            "rollout_answer_tail_cutoff_present", "answer_tail_cutoff_present_mean",
            "reward_metrics/answer_tail_cutoff_present"
        ),
        "clean_answer_protocol": (
            "rollout_clean_answer_protocol", "clean_answer_protocol_mean",
            "reward_metrics/clean_answer_protocol"
        ),
        # Variant 2 (per-step PRM) diagnostics — populated only when
        # the dataset row label is "math_per_step_prm". For "math_psgrpo"
        # rows these stay 0 (no alignment was attempted).
        "alignment_failed": ("rollout_alignment_failed", "alignment_failed_mean", "reward_metrics/alignment_failed"),
        "n_aligned_steps": ("rollout_n_aligned_steps", "n_aligned_steps_mean", "reward_metrics/n_aligned_steps"),
    }
    _TRAIN_KEY_SOURCES = {
        "policy_loss": ("policy_loss", ),
        "kl": ("kl", ),
        "actor_lr": ("actor_lr", ),
        "critic_loss": ("critic_loss", ),
        "critic_lr": ("critic_lr", ),
        "values": ("values", ),
        "values_std": ("values_std", ),
        "reward": ("reward", ),
        "reward_std": ("step_reward_std", ),
        "return": ("return", ),
        "return_std": ("returns_std", ),
        "response_length": ("response_length", ),
        "total_length": ("total_length", ),
        "num_actions": ("num_actions", ),
        "approx_kl": ("approx_kl", ),
        "clipfrac": ("clipfrac", ),
        "ratio_mean": ("ratio_mean", ),
        "ratio_max": ("ratio_max", ),
        "advantages": ("advantages_mean", ),
        "advantages_std": ("advantages_std", ),
        "ptx_loss": ("ptx_loss", ),
        # URSA paper Eq.9 variant 2 advantage-calculator diagnostics. Populated
        # only when --advantage_estimator ursa_variant2 is active; otherwise
        # absent from experience.info and silently skipped by _build_train_metrics.
        "ursa_v2_adv_pos_frac": ("ursa_v2_adv_pos_frac", ),
        "ursa_v2_adv_neg_frac": ("ursa_v2_adv_neg_frac", ),
        "ursa_v2_adv_zero_frac": ("ursa_v2_adv_zero_frac", ),
        "ursa_v2_adv_abs_mean": ("ursa_v2_adv_abs_mean", ),
        "ursa_v2_oc_normed_std": ("ursa_v2_oc_normed_std", ),
        "ursa_v2_msp_normed_std": ("ursa_v2_msp_normed_std", ),
        "ursa_v2_traj_step_count_mean": ("ursa_v2_traj_step_count_mean", ),
    }
    _EVAL_KEY_SOURCES = {
        "reward": ("reward", "reward_mean"),
        "outcome_correct": ("outcome_correct", "outcome_correct_mean"),
        "has_drop_moment": ("has_drop_moment", "has_drop_moment_mean"),
        "model_reward": ("model_reward", "model_reward_mean"),
        "response_length": ("response_length", "response_length_mean"),
        "answer_extraction_failed": ("answer_extraction_failed", "answer_extraction_failed_mean"),
        # PRM step-score distribution + PS-GRPO diagnostics in eval (parity with
        # the expanded rollout-side mapping above).
        "step_score_min": ("step_score_min", "step_score_min_mean"),
        "step_score_mean": ("step_score_mean", "step_score_mean_mean"),
        "step_score_last": ("step_score_last", "step_score_last_mean"),
        "step_count": ("step_count", "step_count_mean"),
        "final_reward": ("final_reward", "final_reward_mean"),
        "max_relative_drop": ("max_relative_drop", "max_relative_drop_mean"),
        "answer_tag_present": ("answer_tag_present", "answer_tag_present_mean"),
        "used_answer_fallback": ("used_answer_fallback", "used_answer_fallback_mean"),
        "used_mathruler": ("used_mathruler", "used_mathruler_mean"),
        "reference_supported": ("reference_supported", "reference_supported_mean"),
        "answer_content_correct": ("answer_content_correct", "answer_content_correct_mean"),
        "answer_marker_count": ("answer_marker_count", "answer_marker_count_mean"),
        "extra_answer_marker_count": ("extra_answer_marker_count", "extra_answer_marker_count_mean"),
        "post_answer_continuation": ("post_answer_continuation", "post_answer_continuation_mean"),
        "post_answer_step_present": ("post_answer_step_present", "post_answer_step_present_mean"),
        "answer_tail_cutoff_present": ("answer_tail_cutoff_present", "answer_tail_cutoff_present_mean"),
        "clean_answer_protocol": ("clean_answer_protocol", "clean_answer_protocol_mean"),
        # Variant 2 diagnostics in eval (eval also runs the PRM forward
        # if the dataset label is "math_per_step_prm")
        "alignment_failed": ("alignment_failed", "alignment_failed_mean"),
        "n_aligned_steps": ("n_aligned_steps", "n_aligned_steps_mean"),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_generate_kwargs = dict(self.generate_kwargs)
        self._eval_generate_kwargs = self._build_eval_generate_kwargs()
        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.define_metric("rollout/*", step_metric=None, step_sync=False, overwrite=True)
            self._wandb.define_metric("train/*", step_metric=None, step_sync=False, overwrite=True)
            self._wandb.define_metric("eval/train_step")
            self._wandb.define_metric("eval/*", step_metric="eval/train_step", step_sync=True, overwrite=True)
            self._wandb.define_metric("profile/train_step")
            self._wandb.define_metric("profile/*", step_metric="profile/train_step", step_sync=True, overwrite=True)

    def _build_eval_generate_kwargs(self) -> Dict:
        eval_generate_kwargs = dict(self._train_generate_kwargs)
        eval_generate_kwargs["do_sample"] = bool(getattr(self.strategy.args, "eval_do_sample", False))
        eval_generate_kwargs["max_new_tokens"] = (
            getattr(self.strategy.args, "eval_generate_max_len", None)
            or self._train_generate_kwargs.get("max_new_tokens")
        )
        eval_generate_kwargs["temperature"] = getattr(self.strategy.args, "eval_temperature", 0.0)
        eval_generate_kwargs["top_p"] = getattr(self.strategy.args, "eval_top_p", 1.0)
        eval_generate_kwargs["top_k"] = getattr(self.strategy.args, "eval_top_k", -1)
        eval_generate_kwargs["repetition_penalty"] = getattr(self.strategy.args, "eval_repetition_penalty", 1.0)
        eval_generate_kwargs["no_repeat_ngram_size"] = getattr(
            self.strategy.args,
            "eval_no_repeat_ngram_size",
            0,
        )
        return eval_generate_kwargs

    @contextmanager
    def _runtime_eval_context(self):
        original_generate_kwargs = self.generate_kwargs
        original_n_samples = self.strategy.args.n_samples_per_prompt
        original_advantage_estimator = self.strategy.args.advantage_estimator
        original_config_n_samples = getattr(self.strategy.config, "n_samples_per_prompt", None)
        original_config_advantage_estimator = getattr(self.strategy.config, "advantage_estimator", None)

        self.generate_kwargs = dict(self._eval_generate_kwargs)
        self.strategy.args.n_samples_per_prompt = max(
            1, int(getattr(self.strategy.args, "eval_n_samples_per_prompt", 1))
        )
        self.strategy.args.advantage_estimator = "reinforce"
        if original_config_n_samples is not None:
            self.strategy.config.n_samples_per_prompt = self.strategy.args.n_samples_per_prompt
        if original_config_advantage_estimator is not None:
            self.strategy.config.advantage_estimator = "reinforce"

        # Detach rollout_eos_patch on the inference engine for the duration of eval.
        # The patch is meant to save GPU during training rollouts (early-stops at
        # the first ``†Answer:`` line) but truncates response tokens that the
        # reward extractor needs in eval; ablation showed it lowers eval
        # outcome_correct by ~8pp at bs=4 and is catastrophic at bs=1
        # (extraction-failure 44%). See PR #53 issuecomment-4394071500.
        rollout_actor = getattr(self.strategy, "inference_engine", None)
        detached_patch = _detach_rollout_eos_patch(rollout_actor)
        if detached_patch is not None and self.strategy.is_rank_0():
            self.strategy.print("[eval] rollout_eos_patch detached for the eval pass")

        try:
            yield
        finally:
            self.generate_kwargs = original_generate_kwargs
            self.strategy.args.n_samples_per_prompt = original_n_samples
            self.strategy.args.advantage_estimator = original_advantage_estimator
            if original_config_n_samples is not None:
                self.strategy.config.n_samples_per_prompt = original_config_n_samples
            if original_config_advantage_estimator is not None:
                self.strategy.config.advantage_estimator = original_config_advantage_estimator
            if detached_patch is not None:
                _reattach_rollout_eos_patch(rollout_actor, detached_patch)
                if self.strategy.is_rank_0():
                    self.strategy.print("[eval] rollout_eos_patch reattached after eval")

    def _build_rollout_metrics(self, logs_dict: Dict[str, float]) -> Dict[str, float]:
        rollout_metrics = {}
        for target_key, source_keys in self._ROLLOUT_KEY_SOURCES.items():
            for source_key in source_keys:
                if source_key in logs_dict:
                    rollout_metrics[target_key] = logs_dict[source_key]
                    break
        return rollout_metrics

    def _build_train_metrics(self, logs_dict: Dict[str, float]) -> Dict[str, float]:
        train_metrics = {}
        for target_key, source_keys in self._TRAIN_KEY_SOURCES.items():
            for source_key in source_keys:
                if source_key in logs_dict:
                    train_metrics[target_key] = logs_dict[source_key]
                    break
        return train_metrics

    def _build_eval_metrics(self, raw_eval_metrics: Dict[str, float]) -> Dict[str, float]:
        eval_metrics = {}
        for target_key, source_keys in self._EVAL_KEY_SOURCES.items():
            for source_key in source_keys:
                if source_key in raw_eval_metrics:
                    eval_metrics[target_key] = raw_eval_metrics[source_key]
                    break
        return eval_metrics

    def _aggregate_eval_metrics(self, raw_eval_metrics: Dict[str, float]) -> Dict[str, float]:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return raw_eval_metrics

        gathered_metrics = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_metrics, raw_eval_metrics or {})

        total_samples = sum(float(metrics.get("num_samples", 0.0)) for metrics in gathered_metrics if metrics)
        if total_samples <= 0:
            return {}

        aggregated_metrics = {"num_samples": total_samples}
        mean_keys = {key for metrics in gathered_metrics if metrics for key in metrics.keys() if key.endswith("_mean")}
        for key in mean_keys:
            weighted_sum = 0.0
            for metrics in gathered_metrics:
                if not metrics or key not in metrics:
                    continue
                weighted_sum += float(metrics["num_samples"]) * float(metrics[key])
            aggregated_metrics[key] = weighted_sum / total_samples
        return aggregated_metrics

    def evaluate(self, eval_dataloader, global_step):
        """Run URSA-flavored evaluation and return aggregated metrics.

        Wraps the base trainer's :meth:`evaluate` in :meth:`_runtime_eval_context`
        so the actor cannot emit reserved sentinel tokens (e.g. ``<|image|>``)
        during eval rollouts, then folds per-dataset metrics into a single
        sample-weighted average via :meth:`_aggregate_eval_metrics`.

        :param eval_dataloader: Iterable over eval batches.
        :param global_step: Current training step, used by the base evaluator
            and logged alongside aggregated metrics on rank 0.
        :returns: Dict of metric_name -> float ready to be uploaded under the
            ``eval/`` namespace. Empty dict if the base evaluator produced no
            metrics this step.
        """
        with self._runtime_eval_context():
            raw_eval_metrics = super().evaluate(eval_dataloader, global_step)
        aggregated_eval_metrics = self._aggregate_eval_metrics(raw_eval_metrics)
        eval_metrics = self._build_eval_metrics(aggregated_eval_metrics)
        if self.strategy.is_rank_0() and eval_metrics:
            self.strategy.print(f"Aggregated runtime eval metrics (Step {global_step}):")
            for key, value in eval_metrics.items():
                self.strategy.print(f"  {key}: {value:.4f}")
        return eval_metrics

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}, episode=0):
        """Drive periodic W&B/TensorBoard logging, eval, and checkpoint saves.

        Called once per training step. Three gating cadences from ``args``:
        ``logging_steps`` (rollout + train metrics), ``eval_steps`` (runs
        :meth:`evaluate` and uploads under ``eval/``), and ``save_steps``
        (writes a ``global_step{N}`` checkpoint).

        Differences from the base trainer's same-named method:
        - Rollout / train metric streams are routed through
          :meth:`_build_rollout_metrics` / :meth:`_build_train_metrics` so
          they pick up URSA-specific keys (PRM diagnostics, ursa_v2_* fields).
        - W&B logs use a monotonic ``wandb_log_counter`` instead of
          ``global_step`` to keep the eval and train series on a single
          increasing x-axis when eval and train ticks interleave.

        :param args: argparse-parsed runtime args (uses ``logging_steps``,
            ``eval_steps``, ``save_steps``).
        :param global_step: Current training step.
        :param step_bar: tqdm progress bar (kept for base-class signature
            compatibility; not used directly here).
        :param logs_dict: Per-step metric dict produced by the base trainer.
        :param client_states: Extra state forwarded to the checkpoint saver.
        :param episode: Current episode counter logged alongside other metrics.
        """
        if global_step % args.logging_steps == 0:
            rollout_metrics = self._build_rollout_metrics(logs_dict)
            train_metrics = self._build_train_metrics(logs_dict)

            if self._wandb is not None and self.strategy.is_rank_0():
                all_wandb_logs = {}

                for key, value in rollout_metrics.items():
                    all_wandb_logs[f"rollout/{key}"] = value
                all_wandb_logs["rollout/episode"] = episode

                for key, value in train_metrics.items():
                    all_wandb_logs[f"train/{key}"] = value
                all_wandb_logs["train/episode"] = episode

                if all_wandb_logs:
                    self.wandb_log_counter += 1
                    self._wandb.log(all_wandb_logs, step=self.wandb_log_counter, commit=True)
                    self._update_wandb_summary(all_wandb_logs)

            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for key, value in rollout_metrics.items():
                    self._tensorboard.add_scalar(f"rollout/{key}", value, global_step)
                for key, value in train_metrics.items():
                    self._tensorboard.add_scalar(f"train/{key}", value, global_step)

        if global_step % args.eval_steps == 0 and self.eval_dataloader is not None:
            with self.profiler.phase("eval"):
                with self.profiler.section("total"):
                    raw_eval_metrics = self.evaluate(self.eval_dataloader, global_step)

            if raw_eval_metrics and self.strategy.is_rank_0():
                self.eval_step_counter += 1

                if self._wandb is not None:
                    eval_logs = {}
                    for key, value in raw_eval_metrics.items():
                        eval_logs[f"eval/{key}"] = value

                    eval_logs["eval/train_step"] = global_step
                    eval_logs["eval/episode"] = episode

                    self.wandb_log_counter += 1
                    self._wandb.log(eval_logs, step=self.wandb_log_counter, commit=True)
                    self._update_wandb_summary(eval_logs)

                elif self._tensorboard is not None:
                    for key, value in raw_eval_metrics.items():
                        self._tensorboard.add_scalar(f"eval/{key}", value, global_step)

        if global_step % args.save_steps == 0:
            with self.profiler.phase("checkpoint"):
                with self.profiler.section("total"):
                    tag = f"global_step{global_step}"
                    self._save_checkpoint(args, tag, client_states)

    def log_profile_metrics(self, global_step: int, episode: int, profile_snapshot: Optional[Dict]) -> None:
        """Forward step-profiler snapshots into W&B/TensorBoard on rank 0 only.

        Profile snapshots come from :class:`StepProfileRecorder` and contain
        a human-readable ``summary`` plus prebuilt ``wandb_logs`` dict. On
        W&B we upload the prebuilt dict as-is under the ``profile/`` namespace;
        on TensorBoard we fall back to writing ``sections_max_s`` and
        ``sections_max_ratio`` scalars individually.

        :param global_step: Current training step (used only by the TB path).
        :param episode: Current episode counter, embedded into the W&B record.
        :param profile_snapshot: Snapshot dict from the profiler, or None
            if profiling was disabled or this step yielded no data. None is a
            no-op.
        """
        if not profile_snapshot or not self.strategy.is_rank_0():
            return

        summary = profile_snapshot.get("summary")
        if summary:
            self.strategy.print(summary)

        if self._wandb is not None:
            wandb_logs = dict(profile_snapshot.get("wandb_logs", {}))
            if wandb_logs:
                wandb_logs["profile/episode"] = episode
                self.wandb_log_counter += 1
                self._wandb.log(wandb_logs, step=self.wandb_log_counter, commit=True)
                self._update_wandb_summary(wandb_logs)

        elif self._tensorboard is not None:
            record = profile_snapshot.get("record", {})
            for key, value in record.get("sections_max_s", {}).items():
                self._tensorboard.add_scalar(f"profile/{key}_s", value, global_step)
            for key, value in record.get("sections_max_ratio", {}).items():
                self._tensorboard.add_scalar(f"profile/{key}_ratio", value, global_step)

    def save_trajectories(self, global_step: int):
        """Persist the current replay buffer contents to disk when configured.

        No-op when either a :class:`TrajectorySaver` has not been wired up
        or the replay buffer is empty. The saver itself handles rank gating
        and on-disk layout.

        :param global_step: Step value embedded in the saved trajectory's
            file name / metadata so downstream analysis can correlate runs.
        """
        if self.trajectory_saver is not None and self.replay_buffer.items:
            self.trajectory_saver.save_trajectories(
                experiences=self.replay_buffer.items,
                step=global_step,
                num_samples=self.num_trajectories_to_save,
                prefix="trajectories",
                compute_stats=self.args.trajectory_analysis,
            )
