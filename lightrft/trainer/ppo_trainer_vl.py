import math
import os
import os.path
import sys
from abc import ABC
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from lightrft.models import ActorVL, GPTLMLoss, PolicyLoss, ValueLoss
from lightrft.models.actor_modality import ActorModality, get_supported_parameters
from lightrft.models.utils import compute_approx_kl, masked_mean, unpacking_samples
from lightrft.trainer import (
    AdaptiveKLController,
    ExperienceVL,
    FixedKLController,
    NaiveExperienceMakerVL,
    NaiveReplayBufferVL,
)
from lightrft.trainer.modality_utils import build_supported_model_kwargs
from lightrft.utils import rotate_ckpt_dirs
from lightrft.utils.distributed_sampler import DistributedSampler


class PPOTrainerVL(ABC):
    """
    Trainer for Proximal Policy Optimization (PPO) over multimodal actors.

    The trainer keeps the original PPO/VL training structure intact: rollout collection,
    replay buffering, PPO optimization, logging, evaluation, and checkpoint rotation all
    remain in the same layer. The refactor in this file is limited to modality-aware data
    plumbing so audio-language models can follow an explicit audio path while vision-language
    models continue to use the existing image/video path.

    In practice this means prompt batches are normalized once, replay items preserve the
    modality-specific tensors they carry, and actor/critic forwards only receive the kwargs
    they actually support. Audio-language models therefore use dedicated fields such as
    ``audio_values`` and ``feature_attention_mask`` instead of overloading the vision path.

    Parameter details are documented on :meth:`__init__`.
    """
    def __init__(
        self,
        strategy,
        actor: ActorVL,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: ActorVL,
        ema_model: ActorVL,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        actor_scheduler,
        critic_scheduler,
        ema_beta: float = 0.992,
        init_kl_coef: float = 0.001,
        kl_target: float = None,
        kl_horizon: int = 10000,
        ptx_coef: float = 0,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        value_clip: float = 0.2,
        micro_rollout_batch_size: int = 8,
        gradient_checkpointing: bool = False,
        max_epochs: int = 1,
        max_norm: float = 1.0,
        tokenizer: Optional[Callable[[Any], dict]] = None,
        processor: Optional[Callable[[Any], dict]] = None,
        prompt_max_len: int = 128,
        dataloader_pin_memory: bool = True,
        remote_rm_url: str = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        reward_fn_label_map: dict = None,
        reward_recipe: dict = None,
        save_hf_ckpt: bool = False,
        disable_ds_ckpt: bool = False,
        **generate_kwargs,
    ) -> None:
        """
        Initialize the PPO trainer for multimodal RL fine-tuning.

        :param strategy: Distributed or single-process training strategy wrapper.
        :type strategy: Strategy
        :param actor: Actor model optimized by PPO.
        :type actor: ActorVL
        :param critic: Critic model used for value prediction.
        :type critic: nn.Module
        :param reward_model: Reward model used in RLHF/RLAIF reward computation.
        :type reward_model: nn.Module
        :param initial_model: Frozen reference model used for KL regularization.
        :type initial_model: ActorVL
        :param ema_model: Exponential moving average copy of the actor, if enabled.
        :type ema_model: ActorVL
        :param actor_optim: Optimizer for actor updates.
        :type actor_optim: Optimizer
        :param critic_optim: Optimizer for critic updates.
        :type critic_optim: Optimizer
        :param actor_scheduler: Learning-rate scheduler for the actor optimizer.
        :type actor_scheduler: Scheduler
        :param critic_scheduler: Learning-rate scheduler for the critic optimizer.
        :type critic_scheduler: Scheduler
        :param ema_beta: EMA decay used when updating ``ema_model``.
        :type ema_beta: float
        :param init_kl_coef: Initial KL penalty coefficient.
        :type init_kl_coef: float
        :param kl_target: Target KL value for adaptive control. If ``None``, a fixed controller is used.
        :type kl_target: float, optional
        :param kl_horizon: Horizon used by the adaptive KL controller.
        :type kl_horizon: int
        :param ptx_coef: Coefficient applied to the optional PTX loss.
        :type ptx_coef: float
        :param micro_train_batch_size: Micro-batch size used by the replay buffer and PPO dataloader.
        :type micro_train_batch_size: int
        :param buffer_limit: Maximum replay-buffer capacity. ``0`` keeps the default behavior.
        :type buffer_limit: int
        :param buffer_cpu_offload: Whether replay items may be offloaded to CPU memory.
        :type buffer_cpu_offload: bool
        :param eps_clip: PPO policy clipping coefficient.
        :type eps_clip: float
        :param value_clip: Value-function clipping coefficient.
        :type value_clip: float
        :param micro_rollout_batch_size: Micro-batch size used during rollout generation.
        :type micro_rollout_batch_size: int
        :param gradient_checkpointing: Whether actor/critic gradient checkpointing is enabled upstream.
        :type gradient_checkpointing: bool
        :param max_epochs: Number of PPO epochs run over each replay-buffer snapshot.
        :type max_epochs: int
        :param max_norm: Gradient clipping threshold.
        :type max_norm: float
        :param tokenizer: Tokenizer used for text decode/encode helpers.
        :type tokenizer: Callable, optional
        :param processor: Multimodal processor used by rollout collection. It may be a vision or audio processor
            depending on the actor modality.
        :type processor: Callable, optional
        :param prompt_max_len: Maximum prompt length used by the experience maker.
        :type prompt_max_len: int
        :param dataloader_pin_memory: Whether PPO dataloaders should pin host memory.
        :type dataloader_pin_memory: bool
        :param remote_rm_url: Optional remote reward-model endpoint.
        :type remote_rm_url: str, optional
        :param reward_fn: Optional custom reward function applied on rollout outputs.
        :type reward_fn: Callable, optional
        :param reward_fn_label_map: Optional label mapping passed to reward helpers.
        :type reward_fn_label_map: dict, optional
        :param reward_recipe: Optional structured reward configuration.
        :type reward_recipe: dict, optional
        :param save_hf_ckpt: Whether to additionally export Hugging Face checkpoints/adapters.
        :type save_hf_ckpt: bool
        :param disable_ds_ckpt: Whether to disable DeepSpeed-format checkpoints.
        :type disable_ds_ckpt: bool
        :param generate_kwargs: Extra generation kwargs forwarded to rollout/eval collection.
            Modality-specific fields are filtered later according to the actor modality.
        :type generate_kwargs: dict
        """
        assert (
            not isinstance(reward_model, List) or len(reward_model) == 1 or reward_fn is not None
        ), "reward_fn must be specified if using multiple reward models"

        ABC.__init__(self)

        self.strategy = strategy
        self.args = strategy.args
        self.save_hf_ckpt = save_hf_ckpt

        current_filename = os.path.basename(__file__)
        current_lineno = sys._getframe().f_lineno
        self.strategy.print(f"[{current_filename}:{current_lineno}]")

        self.disable_ds_ckpt = disable_ds_ckpt
        self.micro_rollout_batch_size = micro_rollout_batch_size
        self.max_epochs = max_epochs
        self.tokenizer = tokenizer
        self.processor = processor
        self.generate_kwargs = generate_kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.max_norm = max_norm
        self.ptx_coef = ptx_coef
        self.micro_train_batch_size = micro_train_batch_size
        self.kl_target = kl_target
        self.prompt_max_len = prompt_max_len
        self.ema_beta = ema_beta
        self.gradient_checkpointing = gradient_checkpointing
        self.reward_fn = reward_fn
        self.reward_fn_label_map = reward_fn_label_map
        self.reward_recipe = reward_recipe
        self.is_lora = getattr(self.args, "lora_rank", 0) > 0

        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.actor_scheduler = actor_scheduler
        self.critic_scheduler = critic_scheduler

        # Cache actor modality once so rollout/training can route inputs without branching on model types.
        # This plays the same role as the old supported-parameter cache, but now audio has its own path too.
        actor_modality = self.actor.modality
        self._actor_supported_params = get_supported_parameters(actor_modality)
        self._is_audio_actor = actor_modality == ActorModality.AUDIO_LANGUAGE

        self.actor_loss_fn = PolicyLoss(eps_clip, use_cpg_loss=self.args.use_cpg_loss)
        self.critic_loss_fn = ValueLoss(value_clip)
        self.ptx_loss_fn = GPTLMLoss()

        self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(init_kl_coef, kl_target, kl_horizon)
        else:
            self.kl_ctl = FixedKLController(init_kl_coef)

        self.experience_maker = NaiveExperienceMakerVL(
            actor,
            critic,
            reward_model,
            initial_model,
            tokenizer,
            processor,
            prompt_max_len,
            self.kl_ctl,
            strategy,
            remote_rm_url,
            reward_fn,
        )
        packing_samples = getattr(self.args, "packing_samples", False)
        self.replay_buffer = NaiveReplayBufferVL(
            micro_train_batch_size, buffer_limit, buffer_cpu_offload, packing_samples
        )

        self._wandb = None
        self._tensorboard = None
        # Independent counters keep eval plots monotonic and avoid wandb step collisions.
        # This preserves the old intent where eval metrics were not forced onto sparse training steps.
        self.eval_step_counter = 0
        self.wandb_log_counter = 0

        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )
            # Define custom metrics to allow different X-axes:
            # rollout/* and train/* use the main training step,
            # while eval/* uses its own counter.
            wandb.define_metric("rollout/global_step")
            wandb.define_metric("rollout/*", step_metric="rollout/global_step")
            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step")
            # eval/* uses its own counter, allowing it to be plotted sequentially
            # even if evaluations happen rarely
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step")

        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    @staticmethod
    def _ensure_device_and_contiguous(value, device):
        """
        Move tensors to the target GPU and make them contiguous for downstream kernels.

        :param value: Tensor or nested tensor list to normalize.
        :type value: torch.Tensor or list or Any
        :param device: CUDA device index expected by the current rank.
        :type device: int
        :return: Value moved to the requested device with contiguous layout preserved recursively.
        :rtype: Any
        """
        if value is None:
            return None
        if isinstance(value, list):
            return [PPOTrainerVL._ensure_device_and_contiguous(v, device) for v in value]
        if not isinstance(value, torch.Tensor):
            return value
        if value.device.type != "cuda" or value.device.index != device:
            value = value.to(device)
        if not value.is_contiguous():
            value = value.contiguous()
        return value

    def _build_model_kwargs(self, source, device: Optional[int] = None) -> Dict[str, Any]:
        """
        Select and optionally relocate only the multimodal kwargs supported by the current actor modality.

        :param source: Replay item or mapping containing candidate multimodal tensors.
        :type source: Any
        :param device: Optional CUDA device index used to normalize tensor placement.
        :type device: int, optional
        :return: Filtered kwargs that can be passed directly into actor/critic forward.
        :rtype: Dict[str, Any]
        """
        kwargs = build_supported_model_kwargs(source, self._actor_supported_params)
        if device is not None:
            kwargs = {key: self._ensure_device_and_contiguous(value, device) for key, value in kwargs.items()}
        return kwargs

    def _unpack_prompt_batch(self, batch):
        """
        Normalize prompt-dataloader outputs across text, vision, video, and audio variants.

        Audio example datasets still produce a 4-field batch, but the second field now maps to
        ``audios`` instead of overloading the image slot.

        :param batch: Raw batch emitted by the prompt dataloader.
        :type batch: tuple or list
        :return: Tuple of ``(prompts, images, videos, audios, references, labels)`` used by rollout code.
        :rtype: tuple
        """
        if len(batch) == 5:
            prompts, images, videos, references, labels = batch
            return prompts, images, videos, None, references, labels
        if len(batch) == 4:
            prompts, modality_inputs, references, labels = batch
            if self._is_audio_actor:
                return prompts, None, None, modality_inputs, references, labels
            return prompts, modality_inputs, None, None, references, labels
        raise ValueError(f"Unsupported prompt batch format with {len(batch)} fields.")

    def _make_experience_list(self, prompts, images, videos, audios, references, labels):
        """
        Shared rollout helper used by both training and evaluation.

        :param prompts: Prompt strings for the current batch.
        :type prompts: list
        :param images: Optional image inputs.
        :type images: Any
        :param videos: Optional video inputs.
        :type videos: Any
        :param audios: Optional audio inputs.
        :type audios: Any
        :param references: Optional references used by reward functions.
        :type references: Any
        :param labels: Optional labels used by reward functions.
        :type labels: Any
        :return: List of rollout experiences produced from the prompt batch.
        :rtype: list
        """
        return self.experience_maker.make_experience_list(
            prompts,
            all_images=images,
            all_videos=videos,
            all_audios=audios,
            all_references=references,
            all_labels=labels,
            **self.generate_kwargs,
        )

    def fit(
        self,
        args,
        prompts_dataloader,
        pretrain_dataloader,
        eval_dataloader=None,
        consumed_samples=0,
        num_update_steps_per_episodes=1,
    ) -> None:
        """
        Main PPO loop: rollout, aggregate replay items, optimize, log, evaluate, and checkpoint.

        :param args: Runtime training arguments.
        :type args: Namespace
        :param prompts_dataloader: Prompt dataloader. Batches may be text-only, image/video multimodal,
            or audio multimodal, and are normalized by :meth:`_unpack_prompt_batch`.
        :type prompts_dataloader: DataLoader
        :param pretrain_dataloader: Optional PTX dataloader consumed during actor updates.
        :type pretrain_dataloader: DataLoader
        :param eval_dataloader: Optional evaluation dataloader using the same rollout path.
        :type eval_dataloader: DataLoader, optional
        :param consumed_samples: Number of rollout samples already consumed when resuming training.
        :type consumed_samples: int
        :param num_update_steps_per_episodes: Planned PPO update steps per episode.
        :type num_update_steps_per_episodes: int
        :return: ``None``.
        :rtype: None
        """
        samples_per_rollout = args.rollout_batch_size * args.n_samples_per_prompt
        samples_per_train = args.train_batch_size * args.n_samples_per_prompt

        # Report whether each rollout leads to multiple updates or vice versa.
        if args.train_batch_size < args.rollout_batch_size:
            updates_per_rollout = samples_per_rollout / samples_per_train
            self.strategy.print(
                f"\n{'=' * 80}\n"
                f"HIGH FREQUENCY UPDATE MODE: train_batch_size ({args.train_batch_size}) < "
                f"rollout_batch_size ({args.rollout_batch_size})\n"
                f"{'=' * 80}\n"
                f"Behavior:\n"
                f"  - Each rollout generates {samples_per_rollout} samples.\n"
                f"  - Each rollout will trigger {updates_per_rollout:.2f} optimizer updates.\n"
                f"  - Total updates will be HIGHER than standard mode for the same amount of data.\n"
                f"{'=' * 80}\n"
            )
        elif args.train_batch_size > args.rollout_batch_size:
            self.strategy.print(
                f"\n{'=' * 80}\n"
                f"ACCUMULATION MODE: train_batch_size ({args.train_batch_size}) > "
                f"rollout_batch_size ({args.rollout_batch_size})\n"
                f"{'=' * 80}\n"
                f"Behavior:\n"
                f"  - Multiple rollouts needed for one update.\n"
                f"{'=' * 80}\n"
            )

        # Calculate number of rollouts per episode.
        # Regardless of TBS and RBS relationship, rollout count should be determined by "total data / rollout size".
        # Numerator (num_update_steps * train_batch_size) equals "total samples planned for this episode".
        # Denominator (rollout_batch_size * n_samples) equals "samples produced per rollout".
        # This calculation ensures data collection volume is constant.
        # When TBS=64, num_update_steps is naturally twice as large as when TBS=128.
        # Substituting into formula: (2N * 0.5T) / R = (N * T) / R.
        # Conclusion: Rollout count unchanged, but internal update loop count doubles due to smaller TBS.
        num_rollouts_per_episodes = (
            num_update_steps_per_episodes * args.train_batch_size // args.max_epochs // args.rollout_batch_size //
            args.n_samples_per_prompt
        )
        # Safeguard to prevent num_rollouts_per_episodes from being 0
        if num_rollouts_per_episodes == 0:
            # Use ceil as a safeguard when integer division would otherwise drop a fractional rollout.
            num_rollouts_per_episodes = math.ceil(
                (num_update_steps_per_episodes * args.train_batch_size) /
                (args.max_epochs * args.rollout_batch_size * args.n_samples_per_prompt)
            )
            if num_rollouts_per_episodes == 0:
                self.strategy.print("[WARNING] Calculated num_rollouts_per_episodes is 0. Forcing to 1.")
                num_rollouts_per_episodes = 1

        if args.eval_steps == -1:
            args.eval_steps = num_rollouts_per_episodes
        if args.save_steps == -1:
            args.save_steps = float("inf")

        self.prompts_dataloader = prompts_dataloader
        self.pretrain_dataloader = pretrain_dataloader
        self.eval_dataloader = eval_dataloader

        # Recover where the previous run left off when resuming from checkpoints.
        steps = consumed_samples // args.rollout_batch_size + 1
        start_episode = consumed_samples // args.rollout_batch_size // num_rollouts_per_episodes
        consumed_samples = consumed_samples % (num_rollouts_per_episodes * args.rollout_batch_size)

        for episode in range(start_episode, args.num_episodes):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(
                    episode,
                    consumed_samples=0 if episode > start_episode else consumed_samples,
                )

            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )

            for batch in self.prompts_dataloader:
                # The helper keeps the rollout loop agnostic to whether the batch is audio or vision.
                prompts, images, videos, audios, references, labels = self._unpack_prompt_batch(batch)
                experience_list = self._make_experience_list(prompts, images, videos, audios, references, labels)
                if not experience_list:
                    pbar.update()
                    steps += 1
                    continue

                for experience in experience_list:
                    self.replay_buffer.append(experience)

                self.strategy.report_memory("after replay_buffer ready")

                # Aggregate rollout statistics from replay buffer before PPO updates clear it.
                rollout_status = {}
                if self.replay_buffer.items:
                    all_rewards = []
                    all_format_rewards = []
                    all_accuracy_rewards = []
                    all_response_lengths = []

                    for item in self.replay_buffer.items:
                        # Robust handling of reward_metrics
                        # 1. Check if info exists
                        # 2. Check if 'reward_metrics' key exists
                        # 3. Check if reward_metrics is not None (critical!)
                        if hasattr(item, "info") and item.info is not None and "reward" in item.info:
                            all_rewards.append(item.info["reward"])

                        if (
                            hasattr(item, "info") and item.info is not None and "reward_metrics" in item.info
                            and item.info["reward_metrics"] is not None
                        ):
                            reward_metrics = item.info["reward_metrics"]
                            if "format_reward" in reward_metrics:
                                all_format_rewards.append(reward_metrics["format_reward"])
                            if "accuracy_reward" in reward_metrics:
                                all_accuracy_rewards.append(reward_metrics["accuracy_reward"])

                        if hasattr(item, "info") and item.info is not None and "response_length" in item.info:
                            all_response_lengths.append(item.info["response_length"])

                    device = torch.cuda.current_device()

                    if all_rewards:
                        # Some reward functions return tensors directly, others scalar values.
                        if isinstance(all_rewards[0], torch.Tensor):
                            rewards_tensor = torch.cat([t.to(device).float() for t in all_rewards])
                        else:
                            rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32, device=device)
                        rollout_status["rollout_reward"] = rewards_tensor.mean().item()
                        rollout_status["rollout_reward_std"] = rewards_tensor.std().item()

                    if all_format_rewards:
                        # [TENSOR-FIX] Handle both tensor lists and scalar lists
                        # Issue: all_format_rewards may contain tensors (from reward_metrics),
                        # but torch.tensor() cannot convert a list of tensors directly.
                        # Solution: Use torch.cat() for tensor lists, torch.tensor() for scalar lists
                        if isinstance(all_format_rewards[0], torch.Tensor):
                            format_tensor = torch.cat([t.to(device).float() for t in all_format_rewards])
                        else:
                            format_tensor = torch.tensor(all_format_rewards, dtype=torch.float32, device=device)
                        mean_format_reward = format_tensor.mean().item()
                        rollout_status["rollout_format_reward"] = mean_format_reward

                    if all_accuracy_rewards:
                        if isinstance(all_accuracy_rewards[0], torch.Tensor):
                            accuracy_tensor = torch.cat([t.to(device).float() for t in all_accuracy_rewards])
                        else:
                            accuracy_tensor = torch.tensor(all_accuracy_rewards, dtype=torch.float32, device=device)
                        mean_accuracy_reward = accuracy_tensor.mean().item()
                        rollout_status["rollout_accuracy_reward"] = mean_accuracy_reward

                    if all_response_lengths:
                        if isinstance(all_response_lengths[0], torch.Tensor):
                            lengths_tensor = torch.cat([t.to(device).float() for t in all_response_lengths])
                        else:
                            lengths_tensor = torch.tensor(all_response_lengths, dtype=torch.float32, device=device)
                        rollout_status["rollout_response_length"] = lengths_tensor.mean().item()

                # Group-normalized estimators already normalize advantages during experience creation.
                if self.args.advantage_estimator != "group_norm":
                    self.replay_buffer.normalize("advantages", self.strategy)

                self.strategy.report_memory("before train")
                status = self.ppo_train(steps)
                self.strategy.report_memory("before clear buffer")
                self.replay_buffer.clear()
                self.strategy.report_memory("after train")

                if "kl" in status:
                    self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)

                # Progress bar reflects rollout quality; wandb/tensorboard will receive both rollout and train metrics.
                pbar.set_postfix(rollout_status)

                # Logs/checkpoints: save BOTH ROLLOUT and TRAINING statistics to wandb
                # [FIX] Merge rollout_status (from inference) and status (from training)
                # to ensure wandb logs contain both types of metrics
                client_states = {"consumed_samples": steps * args.rollout_batch_size}
                logs_dict_combined = {**rollout_status, **status}
                self.save_logs_and_checkpoints(
                    args,
                    steps,
                    pbar,
                    logs_dict_combined,
                    client_states,
                    episode=episode,
                )

                pbar.update()
                steps += 1

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    def ppo_train(self, global_steps=0):
        """
        PPO optimization over the current replay buffer snapshot.

        NOTE: this method is overridden by the SPMD trainer in the main audio run,
        but keeping the base implementation explicit is still useful for non-SPMD execution
        and for understanding the reference PPO flow.

        :param global_steps: Current global training step.
        :type global_steps: int
        :return: Mean metrics aggregated over all PPO minibatches in the snapshot.
        :rtype: dict
        """
        torch.cuda.empty_cache()
        # Rebuild the dataloader each time because the replay buffer is refreshed after every rollout.
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for experience in pbar:
                experience.to_device(device)
                status = self.training_step(experience, global_steps)

                # For DP runs, KL is aggregated with response-length weighting.
                if "kl" in status:
                    status["kl"] *= status["response_length"]
                    status = self.strategy.all_reduce(status)
                    status["kl"] /= status["response_length"]

                short_status = {}
                # Keep progress-bar keys compact while preserving detailed metrics in logs.
                if "policy_loss" in status:
                    short_status.update({
                        "pg": status.get("policy_loss"),
                        "rm": status.get("reward"),
                        "ret": status.get("return"),
                        "glen": status.get("response_length"),
                        "tlen": status.get("total_length"),
                        "kl": status.get("kl"),
                        "act_lr": status.get("actor_lr"),
                    })
                if "critic_loss" in status:
                    short_status.update({
                        "cri": status.get("critic_loss"),
                        "vals": status.get("values"),
                        "cri_lr": status.get("critic_lr"),
                    })
                if "ptx_loss" in status:
                    short_status["ptx"] = status.get("ptx_loss")
                for key, value in status.items():
                    if "/" in key:
                        short_status[key.split("/")[-1]] = value

                status_list.append(status)
                pbar.set_postfix(short_status)

        if status_list:
            status_mean = status_list[0]
            for metrics in status_list[1:]:
                for key, value in metrics.items():
                    status_mean[key] += value
            for key in status_mean.keys():
                status_mean[key] /= len(status_list)

        torch.cuda.empty_cache()
        return status_mean

    def training_step(
        self,
        experience: ExperienceVL,
        global_steps,
        entropy_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Run one PPO optimization step on a replay-buffer batch.

        Actor updates are applied first and critic updates are added afterwards when a critic exists.

        :param experience: Replay-buffer batch containing sequences, masks, rewards, and modality tensors.
        :type experience: ExperienceVL
        :param global_steps: Current global step used for actor-freeze logic.
        :type global_steps: int
        :param entropy_mask: Optional mask for entropy-aware policy loss variants.
        :type entropy_mask: Optional[torch.Tensor]
        :return: Training statistics from actor and critic updates.
        :rtype: Dict[str, float]
        """
        status = {}
        if global_steps > self.freezing_actor_steps:
            status = self.training_step_actor(experience, entropy_mask=entropy_mask)
        if self.critic is not None:
            status.update(self.training_step_critic(experience))
        return status

    def _validate_qwen_vl_tensors(
        self,
        sequences: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        context: str = "training",
    ) -> bool:
        """
        Defensive validation for Qwen-VL style image-token / image-feature consistency.

        This preserves the old skip-on-mismatch safeguard for vision batches.
        Audio-language actors bypass it naturally because they do not forward ``pixel_values``.

        :param sequences: Token sequences about to be forwarded through the actor.
        :type sequences: torch.Tensor
        :param pixel_values: Image features paired with ``sequences``.
        :type pixel_values: Optional[torch.Tensor]
        :param context: Human-readable call site used in warning logs.
        :type context: str
        :return: ``True`` when the batch is safe to run, otherwise ``False``.
        :rtype: bool
        """
        if pixel_values is None or pixel_values.numel() == 0:
            # This is a text-only batch, no validation needed.
            return True

        config = self.strategy.unwrap_model(self.actor.model).config
        image_token_id = getattr(config, "image_token_id", None)
        if image_token_id is None:
            # Model does not use special image tokens.
            return True

        num_tokens = (sequences == image_token_id).sum().item()
        num_patches = pixel_values.shape[0] // 4
        if num_tokens != num_patches:
            self.strategy.print(
                f"[CRITICAL WARNING] Skipping batch in '{context}'. "
                f"Image features and image tokens do not match: tokens: {num_tokens}, features: {num_patches}. "
                "This batch will be discarded to prevent a crash."
            )
            return False
        return True

    def _validate_multimodal_training_batch(
        self,
        experience: ExperienceVL,
        context: str = "training",
    ) -> bool:
        """
        Validate replay batches before forwarding multimodal actors.

        Vision batches keep the existing image-token consistency check. Audio batches
        additionally reject replay rows whose attention mask is entirely zero, because
        Qwen2-Audio cannot infer a valid padding side from a batch that mixes empty
        rows with normal left-padded rows.
        """
        if not self._validate_qwen_vl_tensors(
            experience.sequences,
            getattr(experience, "pixel_values", None),
            context=context,
        ):
            return False

        if not self._is_audio_actor:
            return True

        attention_mask = getattr(experience, "attention_mask", None)
        if attention_mask is None or attention_mask.ndim != 2:
            self.strategy.print(
                f"[CRITICAL WARNING] Skipping batch in '{context}'. "
                "Audio replay batch is missing a valid 2D attention_mask."
            )
            return False

        active_lengths = attention_mask.long().sum(dim=-1)
        invalid_rows = torch.nonzero(active_lengths <= 0, as_tuple=False).flatten().tolist()
        if invalid_rows:
            self.strategy.print(
                f"[CRITICAL WARNING] Skipping batch in '{context}'. "
                f"Audio replay batch contains empty attention_mask rows at indices {invalid_rows}. "
                "This points to a degenerate rollout/replay sample rather than a missing audio file."
            )
            return False

        return True

    def training_step_actor(
        self,
        experience: ExperienceVL,
        entropy_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Actor training step.

        Packed and unpacked replay items are normalized into one forward path, and the
        actor modality determines whether the model receives vision kwargs or audio kwargs.

        :param experience: Replay-buffer batch for PPO policy optimization.
        :type experience: ExperienceVL
        :param entropy_mask: Optional entropy mask forwarded to the policy-loss module.
        :type entropy_mask: Optional[torch.Tensor]
        :return: Actor-side optimization statistics plus rollout metadata copied from ``experience.info``.
        :rtype: Dict[str, float]
        """
        self.actor.train()

        # Packed samples concatenate multiple sequences into one row. Unpacked samples stay batched.
        # This mirrors the old PPOTrainerVL handling while replacing hard-coded VL kwargs with modality-aware ones.
        if isinstance(experience.sequences, list):
            sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
            old_action_log_probs = torch.cat(experience.action_log_probs, dim=0).unsqueeze(0)
            advantages = torch.cat(experience.advantages, dim=0).unsqueeze(0)
            num_actions = [value.numel() for value in experience.advantages]
            packed_seq_lens = [seq.numel() for seq in experience.sequences]
            attention_mask = torch.cat(
                [torch.full_like(seq, idx + 1) for idx, seq in enumerate(experience.sequences)],
                dim=0,
            ).unsqueeze(0)
            if self.args.use_kl_loss and experience.base_action_log_probs is not None:
                base_action_log_probs = torch.cat(experience.base_action_log_probs, dim=0).unsqueeze(0)
        else:
            sequences = experience.sequences
            old_action_log_probs = experience.action_log_probs
            advantages = experience.advantages
            num_actions = experience.action_mask.size(1)
            packed_seq_lens = None
            attention_mask = experience.attention_mask
            if self.args.use_kl_loss and experience.base_action_log_probs is not None:
                base_action_log_probs = experience.base_action_log_probs

        if advantages is not None:
            # Clipping prevents a few extreme group-normalized values from dominating the PPO step.
            max_adv = advantages.max().item()
            if max_adv > 10.0:
                self.strategy.print(f"[Warning] Huge advantage detected: {max_adv}")
            advantages = torch.clamp(advantages, min=-10.0, max=10.0)

        # Actor loss.
        # Build modality-aware kwargs from the replay item instead of assuming vision-specific fields.
        actor_kwargs = self._build_model_kwargs(experience)
        if not self._validate_multimodal_training_batch(experience, context="actor_rl_update"):
            self.strategy.print(
                "[CRITICAL ERROR] Validation failed inside training_step_actor. "
                "This should have been caught by pre-validation in spmd_ppo_trainer.py!"
            )
            return {}

        action_log_probs, output = self.actor(
            sequences,
            num_actions,
            attention_mask=attention_mask,
            return_output=True,
            packed_seq_lens=packed_seq_lens,
            **actor_kwargs,
        )

        actor_loss = self.actor_loss_fn(
            action_log_probs,
            old_action_log_probs,
            advantages,
            action_mask=experience.action_mask,
            entropy_mask=entropy_mask,
        )

        if self.args.use_kl_loss:
            if self.initial_model is not None:
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    experience.action_mask,
                    kl_estimator=self.args.kl_estimator,
                )
                # [Protection measure 2] Per-token KL Clamping
                # NOTE: Adding this causes svkng training to not converge
                # kl = torch.clamp(kl, min=0.0, max=20.0)
            else:
                kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=action_log_probs.device)

            if not self.args.packing_samples:
                kl_mean = masked_mean(kl, experience.action_mask, dim=-1)
            else:
                kl = unpacking_samples(kl, num_actions)
                kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=action_log_probs.device)

            kl_loss = kl_mean.mean()
            experience.info["kl"] = kl_loss.item()
        else:
            kl_loss = 0

        aux_loss = output.aux_loss if self.aux_loss else 0
        loss = actor_loss + aux_loss * self.args.aux_loss_coef + kl_loss * self.kl_ctl.value

        if torch.isnan(loss) or torch.isinf(loss):
            self.strategy.print("[CRITICAL ERROR] Actor loss is NaN or Inf at step. Skipping update.")
            self.strategy.print(f"  Actor Loss: {actor_loss.item()}")
            if isinstance(kl_loss, torch.Tensor):
                self.strategy.print(f"  KL Loss: {kl_loss.item()}")
            else:
                self.strategy.print(f"  KL Loss: {kl_loss}")

        self.strategy.backward(loss, self.actor, self.actor_optim)

        # PTX loss for supervised fine-tuning.
        # Audio PTX is intentionally left unsupported here because the old PTX path
        # was tightly coupled to vision-style tensors.
        if self.pretrain_dataloader is not None:
            if self._is_audio_actor:
                raise NotImplementedError("PTX data path for audio-language actors is not implemented in PPOTrainerVL.")

            data = next(self.pretrain_dataloader)
            inputs = data[1].squeeze(1).to(torch.cuda.current_device())
            attention_mask = data[2].squeeze(1).to(torch.cuda.current_device())
            label = torch.where(
                attention_mask.bool(),
                inputs,
                self.ptx_loss_fn.IGNORE_INDEX,
            )
            pixel_values = data[3].to(torch.cuda.current_device())
            image_grid_thws = data[4].to(torch.cuda.current_device())

            output = self.actor(
                inputs,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thws,
                return_output=True,
            )
            ptx_log_probs = output["logits"]
            ptx_loss = self.ptx_loss_fn(ptx_log_probs, label)
            aux_loss = output.aux_loss if self.aux_loss else 0
            loss = ptx_loss + aux_loss * self.args.aux_loss_coef
            self.strategy.backward(self.ptx_coef * loss, self.actor, self.actor_optim)

        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")

        if self.ema_model:
            self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cuda")

        status = {"policy_loss": actor_loss.item(), "actor_lr": self.actor_scheduler.get_last_lr()[0]}
        if self.pretrain_dataloader is not None and not self._is_audio_actor:
            status["ptx_loss"] = ptx_loss.item()

        # Add ratio and loss-component diagnostics from PolicyLoss when available.
        if hasattr(self.actor_loss_fn, "get_last_stats"):
            status.update(self.actor_loss_fn.get_last_stats())

        # Keep rollout-side info in the status dict so upper layers can log both rollout and train metrics together.
        # This keeps the old logging behavior where experience.info remained the single source of rollout metadata.
        for key, value in experience.info.items():
            if key == "kl":
                if isinstance(value, torch.Tensor):
                    weighted_kl = (value *
                                   experience.info["response_length"]).sum() / experience.info["response_length"].sum()
                    status[key] = weighted_kl.item()
                else:
                    status[key] = value
                continue

            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    log_key = f"{key}/{sub_key}"
                    if isinstance(sub_value, torch.Tensor):
                        status[log_key] = sub_value.mean().item()
                    elif isinstance(sub_value, list) and sub_value and isinstance(sub_value[0], (int, float)):
                        status[log_key] = sum(sub_value) / len(sub_value)
                    elif isinstance(sub_value, (int, float)):
                        status[log_key] = sub_value
                continue

            if isinstance(value, torch.Tensor):
                status[key] = value.float().mean().item()
            elif isinstance(value, list):
                if value and isinstance(value[0], (int, float)):
                    status[key] = sum(value) / len(value)
            elif isinstance(value, (int, float)):
                status[key] = value

        return status

    def training_step_critic(self, experience: ExperienceVL) -> Dict[str, float]:
        """
        Critic training step.

        It uses the same modality-aware kwargs assembly as actor training, so audio and vision
        stay consistent during PPO value updates.

        :param experience: Replay-buffer batch for PPO value-function optimization.
        :type experience: ExperienceVL
        :return: Critic-side optimization statistics.
        :rtype: Dict[str, float]
        """
        self.critic.train()
        device = torch.cuda.current_device()

        # Match the packed/unpacked normalization used in actor training.
        if isinstance(experience.sequences, list):
            sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
            old_values = torch.cat(experience.values, dim=0).unsqueeze(0)
            returns = torch.cat(experience.returns, dim=0).unsqueeze(0)
            num_actions = [value.numel() for value in experience.advantages]
            packed_seq_lens = [seq.numel() for seq in experience.sequences]
            attention_mask = torch.cat(
                [torch.full_like(seq, idx + 1) for idx, seq in enumerate(experience.sequences)],
                dim=0,
            ).unsqueeze(0)
        else:
            sequences = experience.sequences
            old_values = experience.values
            returns = experience.returns
            num_actions = experience.action_mask.size(1)
            packed_seq_lens = None
            attention_mask = experience.attention_mask

        sequences = self._ensure_device_and_contiguous(sequences, device)
        attention_mask = self._ensure_device_and_contiguous(attention_mask, device)
        old_values = self._ensure_device_and_contiguous(old_values, device)
        returns = self._ensure_device_and_contiguous(returns, device)
        critic_kwargs = self._build_model_kwargs(experience, device=device)

        values, output = self.critic(
            sequences,
            num_actions=num_actions,
            attention_mask=attention_mask,
            return_output=True,
            packed_seq_lens=packed_seq_lens,
            **critic_kwargs,
        )

        critic_loss = self.critic_loss_fn(
            values,
            old_values,
            returns,
            action_mask=experience.action_mask,
        )
        aux_loss = output.aux_loss if self.aux_loss else 0
        loss = critic_loss + aux_loss * self.args.aux_loss_coef
        self.strategy.backward(loss, self.critic, self.critic_optim)
        self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")

        return {
            "critic_loss": critic_loss.item(),
            "values": masked_mean(values, experience.action_mask).item(),
            "critic_lr": self.critic_scheduler.get_last_lr()[0],
        }

    def save_logs_and_checkpoints(
        self,
        args,
        global_step,
        step_bar,
        logs_dict={},
        client_states={},
        episode=0,
    ):
        """
        Log rollout/train/eval metrics and save checkpoints on the configured schedule.

        :param args: Runtime training arguments controlling log/eval/save cadence.
        :type args: Namespace
        :param global_step: Current training step.
        :type global_step: int
        :param step_bar: Progress-bar instance for the outer rollout loop.
        :type step_bar: tqdm
        :param logs_dict: Combined metrics from rollout collection and PPO optimization.
        :type logs_dict: dict
        :param client_states: Extra state saved into checkpoints for resume support.
        :type client_states: dict
        :param episode: Current episode index.
        :type episode: int
        """
        if global_step % args.logging_steps == 0:
            # Rollout metrics are logged under their own namespace and should not be duplicated under train/*.
            rollout_only_metrics = {"reward", "response_length", "total_length", "num_actions", "return"}
            rollout_only_prefixes = {"reward_metrics/"}
            rollout_metrics = {}
            train_metrics = {}

            for key, value in logs_dict.items():
                if key.startswith("rollout_"):
                    rollout_metrics[key.replace("rollout_", "", 1)] = value
                elif key in rollout_only_metrics:
                    continue
                elif any(key.startswith(prefix) for prefix in rollout_only_prefixes):
                    continue
                else:
                    train_metrics[key] = value

            if self._wandb is not None and self.strategy.is_rank_0():
                all_wandb_logs = {}
                for key, value in rollout_metrics.items():
                    all_wandb_logs[f"rollout/{key}"] = value
                all_wandb_logs["rollout/global_step"] = global_step
                all_wandb_logs["rollout/episode"] = episode

                for key, value in train_metrics.items():
                    all_wandb_logs[f"train/{key}"] = value
                all_wandb_logs["train/global_step"] = global_step
                all_wandb_logs["train/episode"] = episode

                # FastExperienceMaker can publish collection-side performance stats opportunistically.
                perf_stats = getattr(self.experience_maker, "perf_stats", None)
                if perf_stats is not None:
                    for key, value in perf_stats.items():
                        all_wandb_logs[f"perf/experience_maker/{key}"] = value

                if all_wandb_logs:
                    # Use wandb_log_counter to ensure eval has a unique system step
                    # This prevents eval metrics from being overwritten by train metrics
                    # The plots will still use eval/global_step as X-axis due to define_metric
                    self.wandb_log_counter += 1
                    self._wandb.log(all_wandb_logs, step=self.wandb_log_counter, commit=True)
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for key, value in rollout_metrics.items():
                    self._tensorboard.add_scalar(f"rollout/{key}", value, global_step)
                for key, value in train_metrics.items():
                    self._tensorboard.add_scalar(f"train/{key}", value, global_step)

        if global_step % args.eval_steps == 0 and self.eval_dataloader is not None:
            # Eval runs through the same experience maker, but only collects metrics instead of updating PPO state.
            raw_eval_metrics = self.evaluate(self.eval_dataloader, global_step)
            if raw_eval_metrics and self.strategy.is_rank_0():
                self.eval_step_counter += 1
                if self._wandb is not None:
                    eval_logs = {}
                    for key, value in raw_eval_metrics.items():
                        clean_key = key.replace("eval_", "") if key.startswith("eval_") else key
                        eval_logs[f"eval/{clean_key}"] = value
                    eval_logs["eval/global_step"] = self.eval_step_counter
                    eval_logs["eval/train_step"] = global_step
                    eval_logs["eval/episode"] = episode
                    self.wandb_log_counter += 1
                    self._wandb.log(eval_logs, step=self.wandb_log_counter, commit=True)
                elif self._tensorboard is not None:
                    for key, value in raw_eval_metrics.items():
                        clean_key = key.replace("eval_", "") if key.startswith("eval_") else key
                        self._tensorboard.add_scalar(f"eval/{clean_key}", value, global_step)

        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self._save_checkpoint(args, tag, client_states)

    def _save_checkpoint(self, args, tag, client_states):
        """
        Save model checkpoint to disk.

        This keeps the old DS checkpoint path and the optional rotated HF/LoRA export path.

        :param args: Runtime training arguments containing checkpoint settings.
        :type args: Namespace
        :param tag: Checkpoint tag such as ``global_step1000``.
        :type tag: str
        :param client_states: Extra client state persisted for checkpoint resume.
        :type client_states: dict
        """
        ckpt_path = args.ckpt_path
        if not self.disable_ds_ckpt and not self.is_lora:
            self.strategy.save_ckpt(
                self.actor.model,
                os.path.join(ckpt_path, "_actor"),
                tag,
                args.max_ckpt_num,
                args.max_ckpt_mem,
                client_states,
            )
            if self.critic is not None:
                self.strategy.save_ckpt(
                    self.critic,
                    os.path.join(ckpt_path, "_critic"),
                    tag,
                    args.max_ckpt_num,
                    args.max_ckpt_mem,
                )

        if self.save_hf_ckpt or self.is_lora:
            if self.strategy.is_rank_0():
                os.makedirs(ckpt_path, exist_ok=True)
                max_num = getattr(args, "max_ckpt_num", 3)
                rotate_ckpt_dirs(
                    ckpt_path,
                    max_num,
                    suffix="_lora",
                    strategy=self.strategy,
                    label="HF ckpt",
                )

            save_path = os.path.join(ckpt_path, f"{tag}_lora")
            self.strategy.save_model(self.actor, self.tokenizer, save_path)

    def evaluate(self, eval_dataloader, global_step):
        """
        Evaluate the model on evaluation data.

        Evaluation reuses the same experience-maker path as rollout collection, but only aggregates
        reward and response-length statistics instead of updating PPO state.

        :param eval_dataloader: Evaluation dataloader normalized through the same batch-unpack helper
            used during training.
        :type eval_dataloader: DataLoader
        :param global_step: Training step associated with this evaluation run.
        :type global_step: int
        :return: Aggregated evaluation metrics.
        :rtype: dict
        """
        if eval_dataloader is None:
            return {}

        self.strategy.print(f"\n{'=' * 60}")
        self.strategy.print(f"Starting evaluation at step {global_step}")
        self.strategy.print(f"{'=' * 60}")

        self.actor.eval()
        if self.critic is not None:
            self.critic.eval()

        all_rewards = []
        all_format_rewards = []
        all_accuracy_rewards = []
        all_response_lengths = []
        num_eval_batches = 0

        def extract_values(value):
            # Reward helpers may emit tensors, lists, or scalars depending on the recipe.
            if isinstance(value, torch.Tensor):
                return value.view(-1).cpu().tolist()
            if isinstance(value, (list, tuple)):
                return list(value)
            return [float(value)]

        with torch.no_grad():
            for batch in eval_dataloader:
                prompts, images, videos, audios, references, labels = self._unpack_prompt_batch(batch)
                experience_list = self._make_experience_list(prompts, images, videos, audios, references, labels)
                if not experience_list:
                    continue

                for experience in experience_list:
                    if hasattr(experience, "info") and experience.info:
                        info = experience.info
                        if "reward" in info:
                            all_rewards.extend(extract_values(info["reward"]))
                        if "response_length" in info:
                            all_response_lengths.extend(extract_values(info["response_length"]))
                        if "reward_metrics" in info:
                            reward_metrics = info["reward_metrics"]
                            if "format_reward" in reward_metrics:
                                all_format_rewards.extend(extract_values(reward_metrics["format_reward"]))
                            if "accuracy_reward" in reward_metrics:
                                all_accuracy_rewards.extend(extract_values(reward_metrics["accuracy_reward"]))

                num_eval_batches += 1
                if num_eval_batches >= len(eval_dataloader):
                    break

        metrics = {}
        device = torch.cuda.current_device()

        def compute_stats(name, values_list):
            if not values_list:
                return
            if isinstance(values_list[0], torch.Tensor):
                tensor = torch.cat([value.to(device).float() for value in values_list])
            else:
                tensor = torch.tensor(values_list, dtype=torch.float32, device=device)
            metrics[f"{name}_mean"] = tensor.mean().item()

        compute_stats("reward", all_rewards)
        compute_stats("format_reward", all_format_rewards)
        compute_stats("accuracy_reward", all_accuracy_rewards)
        compute_stats("response_length", all_response_lengths)
        metrics["num_samples"] = len(all_rewards)

        self.strategy.print(f"Evaluation Results (Step {global_step}):")
        for key, value in metrics.items():
            self.strategy.print(f"  {key}: {value:.4f}")
        self.strategy.print(f"{'=' * 60}\n")

        self.actor.train()
        if self.critic is not None:
            self.critic.train()

        return metrics
