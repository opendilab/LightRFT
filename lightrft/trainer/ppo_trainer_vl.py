import os
import sys
import os.path
from abc import ABC
from typing import Any, Callable, Dict, List, Optional

import torch
import math
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from lightrft.models import ActorVL, GPTLMLoss, PolicyLoss, ValueLoss
from lightrft.models.utils import masked_mean, unpacking_samples, compute_approx_kl
from lightrft.utils.distributed_sampler import DistributedSampler
from lightrft.trainer import AdaptiveKLController, ExperienceVL, FixedKLController, NaiveExperienceMakerVL, NaiveReplayBufferVL  # noqa


class PPOTrainerVL(ABC):
    """
    Trainer for Proximal Policy Optimization (PPO) algorithm for Vision-Language Models.

    :param strategy: The training strategy to use.
    :type strategy: Strategy
    :param actor: The actor model in the PPO algorithm.
    :type actor: ActorVL
    :param critic: The critic model in the PPO algorithm.
    :type critic: nn.Module
    :param reward_model: The reward model for calculating rewards in the RLHF setup.
    :type reward_model: nn.Module
    :param initial_model: The initial model for reference logits to limit actor updates in RLHF.
    :type initial_model: ActorVL
    :param ema_model: The exponential moving average model for stable training.
    :type ema_model: ActorVL
    :param actor_optim: The optimizer for the actor model.
    :type actor_optim: Optimizer
    :param critic_optim: The optimizer for the critic model.
    :type critic_optim: Optimizer
    :param actor_scheduler: The learning rate scheduler for the actor.
    :type actor_scheduler: Scheduler
    :param critic_scheduler: The learning rate scheduler for the critic.
    :type critic_scheduler: Scheduler
    :param ema_beta: EMA decay rate for model stability, defaults to 0.992.
    :type ema_beta: float
    :param init_kl_coef: Initial coefficient for KL divergence, defaults to 0.001.
    :type init_kl_coef: float
    :param kl_target: Target value for KL divergence, defaults to None.
    :type kl_target: float, optional
    :param kl_horizon: Horizon for KL annealing, defaults to 10000.
    :type kl_horizon: int
    :param ptx_coef: Coefficient for supervised loss from pre-trained data, defaults to 0.
    :type ptx_coef: float
    :param micro_train_batch_size: Micro-batch size for actor training, defaults to 8.
    :type micro_train_batch_size: int
    :param buffer_limit: Maximum size of the replay buffer, defaults to 0.
    :type buffer_limit: int
    :param buffer_cpu_offload: If True, offloads replay buffer to CPU, defaults to True.
    :type buffer_cpu_offload: bool
    :param eps_clip: Clipping coefficient for policy loss, defaults to 0.2.
    :type eps_clip: float
    :param value_clip: Clipping coefficient for value function loss, defaults to 0.2.
    :type value_clip: float
    :param micro_rollout_batch_size: Micro-batch size for generating rollouts, defaults to 8.
    :type micro_rollout_batch_size: int
    :param gradient_checkpointing: If True, enables gradient checkpointing, defaults to False.
    :type gradient_checkpointing: bool
    :param max_epochs: Number of epochs to train, defaults to 1.
    :type max_epochs: int
    :param max_norm: Maximum gradient norm for gradient clipping, defaults to 1.0.
    :type max_norm: float
    :param tokenizer: Tokenizer for input data, defaults to None.
    :type tokenizer: Callable, optional
    :param processor: Processor for multimodal input data, defaults to None.
    :type processor: Callable, optional
    :param prompt_max_len: Maximum length for prompts, defaults to 128.
    :type prompt_max_len: int
    :param dataloader_pin_memory: If True, pins memory in the data loader, defaults to True.
    :type dataloader_pin_memory: bool
    :param remote_rm_url: URL for remote reward model API, defaults to None.
    :type remote_rm_url: str, optional
    :param reward_fn: Custom reward function for computing rewards, defaults to None.
    :type reward_fn: Callable, optional
    :param reward_fn_label_map: Label mapping for reward function, defaults to None.
    :type reward_fn_label_map: dict, optional
    :param reward_recipe: Recipe configuration for reward computation, defaults to None.
    :type reward_recipe: dict, optional
    :param save_hf_ckpt: Whether to save huggingface-format model weight, defaults to False.
    :type save_hf_ckpt: bool
    :param disable_ds_ckpt: Whether not to save deepspeed-format model weight (used for training recovery).
    :type disable_ds_ckpt: bool
    :param generate_kwargs: Additional arguments for model generation.
    :type generate_kwargs: dict
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

        # TODO: Investigate use_cpg_loss flag
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

        # Initialize wandb/tensorboard for logging
        self._wandb = None
        self._tensorboard = None
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

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            # Use eval/global_step as step_metric to match what we actually log
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    # def _ensure_model_buffers_on_device(self):
    #     """
    #     Ensure all model buffers are on GPU.

    #     This is CRITICAL for preventing Triton kernel errors. Models with:
    #     - RoPE caches (cos/sin tables)
    #     - Normalization statistics (mean/std)
    #     - Other persistent buffers

    #     may have buffers initialized on CPU, especially with DeepSpeed Zero-3.
    #     This method explicitly moves all buffers to the current GPU.

    #     Called at the start of training to "warm up" models.
    #     """
    #     device = torch.cuda.current_device()

    #     # Check Actor
    #     if self.actor is not None:
    #         self._ensure_single_model_buffers(self.actor, "Actor", device)

    #     # Check Critic (most critical - often causes Triton errors)
    #     if self.critic is not None:
    #         self._ensure_single_model_buffers(self.critic, "Critic", device)

    #     # Check Initial Model (for KL calculation)
    #     if self.initial_model is not None:
    #         self._ensure_single_model_buffers(self.initial_model, "InitialModel", device)

    #     # Check Reward Model
    #     if isinstance(self.reward_model, nn.Module):
    #         self._ensure_single_model_buffers(self.reward_model, "RewardModel", device)

    # def _ensure_single_model_buffers(self, model, model_name, device):
    #     """
    #     Move all buffers of a single model to the specified device.

    #     :param model: The model whose buffers to move.
    #     :type model: nn.Module
    #     :param model_name: Name of the model for logging.
    #     :type model_name: str
    #     :param device: Target device.
    #     :type device: torch.device
    #     """
    #     moved_buffers = []

    #     for buffer_name, buffer in model.named_buffers():
    #         if buffer is not None and buffer.device.type != 'cuda':
    #             try:
    #                 # Force buffer to GPU
    #                 buffer.data = buffer.data.to(device)
    #                 moved_buffers.append(buffer_name)
    #             except Exception as e:
    #                 self.strategy.print(
    #                     f"[WARNING] Failed to move {model_name}.{buffer_name} to GPU: {e}"
    #                 )

    #     if moved_buffers and self.strategy.is_rank_0():
    #         self.strategy.print(
    #             f"[INFO] Moved {len(moved_buffers)} buffers of {model_name} to GPU: {moved_buffers[:5]}{'...' if len(moved_buffers) > 5 else ''}"  # noqa
    #         )
    #     elif self.strategy.is_rank_0():
    #         self.strategy.print(f"[INFO] All buffers of {model_name} already on GPU")

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
        Main training loop for PPO.

        :param args: Training arguments.
        :type args: Namespace
        :param prompts_dataloader: DataLoader for prompt data.
        :type prompts_dataloader: DataLoader
        :param pretrain_dataloader: DataLoader for pre-training data.
        :type pretrain_dataloader: DataLoader
        :param eval_dataloader: DataLoader for evaluation data, defaults to None.
        :type eval_dataloader: DataLoader, optional
        :param consumed_samples: Number of samples already consumed, defaults to 0.
        :type consumed_samples: int
        :param num_update_steps_per_episodes: Number of update steps per episode, defaults to 1.
        :type num_update_steps_per_episodes: int
        """

        # Calculate samples per rollout and per training iteration
        samples_per_rollout = args.rollout_batch_size * args.n_samples_per_prompt
        samples_per_train = args.train_batch_size * args.n_samples_per_prompt

        # Print training mode information
        if args.train_batch_size < args.rollout_batch_size:
            updates_per_rollout = samples_per_rollout / samples_per_train
            self.strategy.print(
                f"\n{'=' * 80}\n"
                f"HIGH FREQUENCY UPDATE MODE: train_batch_size ({args.train_batch_size}) < rollout_batch_size ({args.rollout_batch_size})\n"  # noqa
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
                f"ACCUMULATION MODE: train_batch_size ({args.train_batch_size}) > rollout_batch_size ({args.rollout_batch_size})\n"  # noqa
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
            # Try recalculating with ceil to prevent fractional values from being discarded by integer division
            val = (num_update_steps_per_episodes *
                   args.train_batch_size) / (args.max_epochs * args.rollout_batch_size * args.n_samples_per_prompt)
            num_rollouts_per_episodes = math.ceil(val)

            if num_rollouts_per_episodes == 0:
                self.strategy.print("[WARNING] Calculated num_rollouts_per_episodes is 0. Forcing to 1.")
                num_rollouts_per_episodes = 1

        # Get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_rollouts_per_episodes  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # Do not save checkpoint

        self.prompts_dataloader = prompts_dataloader
        self.pretrain_dataloader = pretrain_dataloader
        self.eval_dataloader = eval_dataloader  # Save for evaluation

        # Restore step and start_episode
        steps = consumed_samples // args.rollout_batch_size + 1
        start_episode = consumed_samples // args.rollout_batch_size // num_rollouts_per_episodes
        consumed_samples = consumed_samples % (num_rollouts_per_episodes * args.rollout_batch_size)

        for episode in range(start_episode, args.num_episodes):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(
                    episode, consumed_samples=0 if episode > start_episode else consumed_samples
                )
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )

            for batch in self.prompts_dataloader:
                # Compatible with both image-only (4 args) and video (5 args) dataloaders
                if len(batch) == 5:
                    rand_prompts, rand_images, rand_videos, rand_references, rand_labels = batch
                else:
                    rand_prompts, rand_images, rand_references, rand_labels = batch
                    rand_videos = None

                # TODO: Remove debug print
                self.strategy.print(
                    f"rand_prompts:\n {rand_prompts}\n , rand_images:{rand_images}\n , rand_references:{rand_references}\n, rand_labels:{rand_labels}\n "  # noqa
                )

                for i, experience in enumerate(
                    self.experience_maker.make_experience_list(
                        rand_prompts,
                        rand_images,
                        all_videos=rand_videos,
                        all_references=rand_references,
                        all_labels=rand_labels,
                        **self.generate_kwargs
                    )
                ):
                    if i == 0:
                        output = self.tokenizer.batch_decode(
                            experience.sequences[0].unsqueeze(0), skip_special_tokens=True
                        )
                        self.strategy.print("experience.sequences w skip_special_tokens: ", output)
                        self.strategy.print(
                            f"rand_prompts:\n {rand_prompts}\n , rand_images:{rand_images}\n , rand_references:{rand_references}\n, rand_labels:{rand_labels}\n "  # noqa
                        )

                    self.replay_buffer.append(experience)

                self.strategy.report_memory('after replay_buffer ready')

                # Aggregate rollout statistics from replay buffer
                # Collect metrics from the rollout/collection phase
                rollout_status = {}
                if self.replay_buffer.items:
                    all_rewards = []
                    all_format_rewards = []
                    all_accuracy_rewards = []
                    all_response_lengths = []

                    for item in self.replay_buffer.items:
                        # Collect rewards from rollout
                        if hasattr(item, 'info') and item.info is not None and 'reward' in item.info:
                            all_rewards.append(item.info['reward'])

                        # Robust handling of reward_metrics
                        # 1. Check if info exists
                        # 2. Check if 'reward_metrics' key exists
                        # 3. Check if reward_metrics is not None (critical!)
                        if (
                            hasattr(item, 'info') and item.info is not None and 'reward_metrics' in item.info
                            and item.info['reward_metrics'] is not None
                        ):

                            reward_metrics = item.info['reward_metrics']

                            # Safely extract sub-metrics
                            if 'format_reward' in reward_metrics:
                                all_format_rewards.append(reward_metrics['format_reward'])
                            if 'accuracy_reward' in reward_metrics:
                                all_accuracy_rewards.append(reward_metrics['accuracy_reward'])

                        # Collect response lengths from rollout
                        if hasattr(item, 'info') and item.info is not None and 'response_length' in item.info:
                            all_response_lengths.append(item.info['response_length'])

                    # Compute rollout statistics
                    device = torch.cuda.current_device()

                    if all_rewards:
                        # [TENSOR-FIX] Handle both tensor lists and scalar lists
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
                            # List of tensors: concatenate them
                            format_tensor = torch.cat([t.to(device).float() for t in all_format_rewards])
                        else:
                            # List of scalars: convert to tensor
                            format_tensor = torch.tensor(all_format_rewards, dtype=torch.float32, device=device)

                        mean_format_reward = format_tensor.mean().item()

                        # Only display if mean is significantly non-zero
                        if abs(mean_format_reward) > 1e-6:
                            rollout_status["rollout_format_reward"] = mean_format_reward

                    if all_accuracy_rewards:
                        # [TENSOR-FIX] Handle both tensor lists and scalar lists
                        if isinstance(all_accuracy_rewards[0], torch.Tensor):
                            accuracy_tensor = torch.cat([t.to(device).float() for t in all_accuracy_rewards])
                        else:
                            accuracy_tensor = torch.tensor(all_accuracy_rewards, dtype=torch.float32, device=device)

                        mean_accuracy_reward = accuracy_tensor.mean().item()

                        # Only display if mean is significantly non-zero
                        if abs(mean_accuracy_reward) > 1e-6:
                            rollout_status["rollout_accuracy_reward"] = mean_accuracy_reward

                    if all_response_lengths:
                        # [TENSOR-FIX] Handle both tensor lists and scalar lists
                        if isinstance(all_response_lengths[0], torch.Tensor):
                            lengths_tensor = torch.cat([t.to(device).float() for t in all_response_lengths])
                        else:
                            lengths_tensor = torch.tensor(all_response_lengths, dtype=torch.float32, device=device)

                        rollout_status["rollout_response_length"] = lengths_tensor.mean().item()

                # TODO: Check normalization behavior
                if self.args.advantage_estimator != "group_norm":
                    self.replay_buffer.normalize("advantages", self.strategy)

                self.strategy.report_memory('before train')

                status = self.ppo_train(steps)

                self.strategy.report_memory('before clear buffer')
                self.replay_buffer.clear()

                self.strategy.report_memory('after train')

                if "kl" in status:
                    self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)

                # Update Episode pbar with ROLLOUT statistics (not training statistics!)
                pbar.set_postfix(rollout_status)

                # Logs/checkpoints: save TRAINING statistics to wandb
                client_states = {"consumed_samples": steps * args.rollout_batch_size}
                self.save_logs_and_checkpoints(args, steps, pbar, status, client_states, episode=episode)

                pbar.update()
                steps = steps + 1

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    def ppo_train(self, global_steps=0):
        """
        PPO training loop over the replay buffer.

        NOTE: This method is not used directly in the main trainer,
        as it's overridden by external classes (e.g., lightrft/trainer/spmd_ppo_trainer.py).

        :param global_steps: Current global step count, defaults to 0.
        :type global_steps: int
        :return: Dictionary of averaged training statistics.
        :rtype: dict
        """
        torch.cuda.empty_cache()
        # Replay buffer may be empty at first, we should rebuild at each training
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

                # For DP: weighted mean for KL
                if "kl" in status:
                    status["kl"] *= status["response_length"]
                    status = self.strategy.all_reduce(status)
                    status["kl"] /= status["response_length"]

                short_status = {}

                # Add core metrics with abbreviations to keep progress bar concise
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

                for k, v in status.items():
                    if "/" in k:
                        short_key = k.split('/')[-1]
                        short_status[short_key] = v

                status_list.append(status)
                pbar.set_postfix(short_status)

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        torch.cuda.empty_cache()
        return status_mean

    def training_step(self,
                      experience: ExperienceVL,
                      global_steps,
                      entropy_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Single training step combining actor and critic updates.

        :param experience: Experience batch from replay buffer.
        :type experience: ExperienceVL
        :param global_steps: Current global step count.
        :type global_steps: int
        :param entropy_mask: Optional mask for high-entropy tokens.
        :type entropy_mask: Optional[torch.Tensor]
        :return: Dictionary of training statistics.
        :rtype: Dict[str, float]
        """
        status = {}
        if global_steps > self.freezing_actor_steps:
            status = self.training_step_actor(experience, entropy_mask=entropy_mask)
        if self.critic is not None:
            status.update(self.training_step_critic(experience))
        return status

    def training_step_actor(self,
                            experience: ExperienceVL,
                            entropy_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Actor training step.

        :param experience: Experience batch from replay buffer.
        :type experience: ExperienceVL
        :return: Dictionary of actor training statistics.
        :rtype: Dict[str, float]
        """
        self.actor.train()

        # TODO: This is a bad indicator to say that data is packed... not supported
        if isinstance(experience.sequences, list):
            sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)

            pixel_values = experience.pixel_values
            image_grid_thws = experience.image_grid_thws
            pixel_values_videos = getattr(experience, "pixel_values_videos", None)
            video_grid_thws = getattr(experience, "video_grid_thws", None)

            old_action_log_probs = torch.cat(experience.action_log_probs, dim=0).unsqueeze(0)
            advantages = torch.cat(experience.advantages, dim=0).unsqueeze(0)
            num_actions = [v.numel() for v in experience.advantages]
            packed_seq_lens = [s.numel() for s in experience.sequences]
            attention_mask = torch.cat([torch.full_like(s, i + 1) for i, s in enumerate(experience.sequences)],
                                       dim=0).unsqueeze(0)
            if self.args.use_kl_loss and experience.base_action_log_probs is not None:
                base_action_log_probs = torch.cat(experience.base_action_log_probs, dim=0).unsqueeze(0)
        else:
            sequences = experience.sequences

            pixel_values = experience.pixel_values
            image_grid_thws = experience.image_grid_thws
            pixel_values_videos = getattr(experience, "pixel_values_videos", None)
            video_grid_thws = getattr(experience, "video_grid_thws", None)

            old_action_log_probs = experience.action_log_probs
            advantages = experience.advantages
            num_actions = experience.action_mask.size(1)
            packed_seq_lens = None
            attention_mask = experience.attention_mask
            if self.args.use_kl_loss and experience.base_action_log_probs is not None:
                base_action_log_probs = experience.base_action_log_probs

        if advantages is not None:
            # Log max advantage before clipping for debugging (optional)
            max_adv = advantages.max().item()
            if max_adv > 10.0:
                self.strategy.print(f"[Warning] Huge advantage detected: {max_adv}")
            advantages = torch.clamp(advantages, min=-10.0, max=10.0)

        # Actor loss
        action_log_probs, output = self.actor(
            sequences,
            num_actions,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thws,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thws,
            return_output=True,
            packed_seq_lens=packed_seq_lens,
        )

        # NOTE: Explicit masking in log-space is incorrect - removed
        # if experience.action_mask is not None:
        #     # Setting masked positions to 0 to match old_action_log_probs is WRONG in log-space
        #     action_log_probs = action_log_probs * experience.action_mask

        # Loss function
        actor_loss = self.actor_loss_fn(
            action_log_probs,
            old_action_log_probs,
            advantages,
            action_mask=experience.action_mask,
            entropy_mask=entropy_mask,
        )

        if self.args.use_kl_loss:
            if self.initial_model is not None:
                # TODO(pu): Text-only action mask for KL calculation

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
            # Not supported for packed samples
            else:
                # Convert tensor into list of tensors for easier manipulation within dataset
                kl = unpacking_samples(kl, num_actions)
                kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=action_log_probs.device)

            kl_loss = kl_mean.mean()
            experience.info["kl"] = kl_loss.item()
        else:
            kl_loss = 0

        # Mixtral auxiliary loss
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0

        loss = actor_loss + aux_loss * self.args.aux_loss_coef + kl_loss * self.kl_ctl.value

        if torch.isnan(loss) or torch.isinf(loss):
            self.strategy.print("[CRITICAL ERROR] Actor loss is NaN or Inf at step. Skipping update.")
            self.strategy.print(f"  Actor Loss: {actor_loss.item()}")
            self.strategy.print(f"  KL Loss: {kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss}")

        self.strategy.backward(loss, self.actor, self.actor_optim)

        # PTX loss for supervised fine-tuning
        if self.pretrain_dataloader is not None:
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
                return_output=True
            )
            ptx_log_probs = output["logits"]

            # Loss function
            ptx_loss = self.ptx_loss_fn(ptx_log_probs, label)
            # Mixtral auxiliary loss
            if self.aux_loss:
                aux_loss = output.aux_loss
            else:
                aux_loss = 0
            loss = ptx_loss + aux_loss * self.args.aux_loss_coef
            self.strategy.backward(self.ptx_coef * loss, self.actor, self.actor_optim)

        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")

        if self.ema_model:
            self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cuda")

        # Status
        status = {"policy_loss": actor_loss.item(), "actor_lr": self.actor_scheduler.get_last_lr()[0]}
        if self.pretrain_dataloader is not None:
            status["ptx_loss"] = ptx_loss.item()

        # Add ratio and loss component statistics from PolicyLoss for diagnosis
        if hasattr(self.actor_loss_fn, 'get_last_stats'):
            policy_stats = self.actor_loss_fn.get_last_stats()
            status.update(policy_stats)

        # self.strategy.print(f"experience.info:{experience.info}")

        # Robustly handle various data types in experience.info for logging
        for k, v in experience.info.items():
            # Special handling for KL divergence, which is already a scalar item
            if k == "kl":
                # KL is often weighted by response length, handle it carefully if it's tensor
                if isinstance(v, torch.Tensor):
                    # This logic assumes 'v' is a tensor of KL values per item in the batch
                    weighted_kl = (v *
                                   experience.info["response_length"]).sum() / experience.info["response_length"].sum()
                    status[k] = weighted_kl.item()
                else:  # If it's already a scalar float
                    status[k] = v
                continue

            # Handle nested dictionaries like 'reward_metrics'
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    log_key = f"{k}/{sub_k}"
                    if isinstance(sub_v, torch.Tensor):
                        status[log_key] = sub_v.mean().item()
                    elif isinstance(sub_v, list) and sub_v and isinstance(sub_v[0], (int, float)):
                        status[log_key] = sum(sub_v) / len(sub_v)
                    elif isinstance(sub_v, (int, float)):
                        status[log_key] = sub_v
                continue

            # General handling for other keys
            if isinstance(v, torch.Tensor):
                # If it's a tensor, it's safe to call .mean()
                status[k] = v.float().mean().item()
            elif isinstance(v, list):
                # If it's a list, only compute mean if it contains numbers
                if v and isinstance(v[0], (int, float)):
                    status[k] = sum(v) / len(v)
                # Otherwise, it's a list of strings or dicts, which cannot be averaged. Skip it.
            elif isinstance(v, (int, float)):
                # If it's already a scalar number, just use it
                status[k] = v

        return status

    def training_step_critic(self, experience: ExperienceVL) -> Dict[str, float]:
        """
        Critic training step.

        :param experience: Experience batch from replay buffer.
        :type experience: ExperienceVL
        :return: Dictionary of critic training statistics.
        :rtype: Dict[str, float]
        """
        self.critic.train()

        # Layer 1: Get current GPU device
        device = torch.cuda.current_device()

        # Layer 2: Helper function for robust device placement
        def ensure_device_and_contiguous(tensor, name="tensor"):
            """
            Ensure tensor is:
            1. On the correct GPU device
            2. Contiguous in memory (required by Triton)
            3. Return None safely if input is None

            :param tensor: Input tensor to process.
            :type tensor: torch.Tensor or None
            :param name: Name for logging purposes, defaults to "tensor".
            :type name: str
            :return: Processed tensor or None.
            :rtype: torch.Tensor or None
            """
            if tensor is None:
                return None

            # Move to GPU if not already there
            if tensor.device.type != 'cuda' or tensor.device.index != device:
                tensor = tensor.to(device)

            # Ensure contiguous memory layout for Triton kernels
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()

            return tensor

        # Layer 3: Apply defensive device placement to all multimodal tensors
        pixel_values = ensure_device_and_contiguous(experience.pixel_values, "pixel_values")
        image_grid_thws = ensure_device_and_contiguous(experience.image_grid_thws, "image_grid_thws")
        pixel_values_videos = ensure_device_and_contiguous(
            getattr(experience, "pixel_values_videos", None), "pixel_values_videos"
        )
        video_grid_thws = ensure_device_and_contiguous(getattr(experience, "video_grid_thws", None), "video_grid_thws")

        # TODO: This is a bad indicator to say that data is packed...
        if isinstance(experience.sequences, list):
            sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
            old_values = torch.cat(experience.values, dim=0).unsqueeze(0)
            returns = torch.cat(experience.returns, dim=0).unsqueeze(0)
            num_actions = [v.numel() for v in experience.advantages]
            packed_seq_lens = [s.numel() for s in experience.sequences]
            attention_mask = torch.cat([torch.full_like(s, i + 1) for i, s in enumerate(experience.sequences)],
                                       dim=0).unsqueeze(0)
        else:
            sequences = experience.sequences
            old_values = experience.values
            returns = experience.returns
            num_actions = experience.action_mask.size(1)
            packed_seq_lens = None
            attention_mask = experience.attention_mask

        # Ensure sequences and attention_mask are also on device and contiguous
        sequences = ensure_device_and_contiguous(sequences, "sequences")
        attention_mask = ensure_device_and_contiguous(attention_mask, "attention_mask")

        # Critic loss
        values, output = self.critic(
            sequences,
            num_actions=num_actions,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thws,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thws,
            return_output=True,
            packed_seq_lens=packed_seq_lens,
        )
        # Loss function
        critic_loss = self.critic_loss_fn(
            values,
            old_values,
            returns,
            action_mask=experience.action_mask,
        )
        # Mixtral auxiliary loss
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        loss = critic_loss + aux_loss * self.args.aux_loss_coef
        self.strategy.backward(loss, self.critic, self.critic_optim)
        self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")

        # Status
        status = {
            "critic_loss": critic_loss.item(),
            "values": masked_mean(values, experience.action_mask).item(),
            "critic_lr": self.critic_scheduler.get_last_lr()[0],
        }
        return status

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}, episode=0):
        """
        Save logs to wandb/tensorboard and save model checkpoints.

        :param args: Training arguments.
        :type args: Namespace
        :param global_step: Current global step.
        :type global_step: int
        :param step_bar: Progress bar object.
        :type step_bar: tqdm
        :param logs_dict: Dictionary of metrics to log, defaults to {}.
        :type logs_dict: dict
        :param client_states: Client state for checkpoint recovery, defaults to {}.
        :type client_states: dict
        :param episode: Current episode number, defaults to 0.
        :type episode: int
        """
        if global_step % args.logging_steps == 0:
            # Wandb logging
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {
                    "train/%s" % k: v
                    for k, v in {
                        **logs_dict,
                        "global_step": global_step,
                        "episode": episode,
                    }.items()
                }
                if self.experience_maker.perf_stats is not None:
                    logs.update({f"perf/experience_maker/{k}": v for k, v in self.experience_maker.perf_stats.items()})
                self._wandb.log(logs)
            # TensorBoard logging
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)
                if self.experience_maker.perf_stats is not None:
                    for k, v in self.experience_maker.perf_stats.items():
                        self._tensorboard.add_scalar(f"perf/experience_maker/{k}", v, global_step)

        # Evaluation
        if global_step % args.eval_steps == 0 and self.eval_dataloader is not None:
            eval_metrics = self.evaluate(self.eval_dataloader, global_step)

            # Log to wandb
            if eval_metrics and self._wandb is not None and self.strategy.is_rank_0():
                # [FIX] Remove duplicate "eval_" prefix and add episode/epoch info
                eval_logs = {}
                for k, v in eval_metrics.items():
                    # Remove "eval_" prefix if present (it will be added as "eval/" by wandb structure)
                    clean_key = k.replace("eval_", "") if k.startswith("eval_") else k
                    eval_logs[f"eval/{clean_key}"] = v

                # Add step and episode info for X-axis options in wandb
                eval_logs["eval/global_step"] = global_step
                eval_logs["eval/episode"] = episode
                eval_logs["eval/epoch"] = episode  # Alias for episode (common convention)

                self._wandb.log(eval_logs)
            # Log to TensorBoard
            elif eval_metrics and self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in eval_metrics.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, global_step)

        # Save checkpoint
        # TODO: Save best model on dev, use loss/perplexity/others on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self._save_checkpoint(args, tag, client_states)

    def _save_checkpoint(self, args, tag, client_states):
        """
        Save model checkpoint to disk.

        :param args: Training arguments.
        :type args: Namespace
        :param tag: Checkpoint tag (e.g., "global_step1000").
        :type tag: str
        :param client_states: Client state for checkpoint recovery.
        :type client_states: dict
        """
        if not self.disable_ds_ckpt:
            self.strategy.save_ckpt(
                self.actor.model,
                os.path.join(args.ckpt_path, "_actor"),
                tag,
                args.max_ckpt_num,
                args.max_ckpt_mem,
                client_states,
            )
            if self.critic is not None:
                self.strategy.save_ckpt(
                    self.critic, os.path.join(args.ckpt_path, "_critic"), tag, args.max_ckpt_num, args.max_ckpt_mem
                )

        if self.save_hf_ckpt:
            save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
            self.strategy.save_model(self.actor, self.tokenizer, save_path)

    def evaluate(self, eval_dataloader, global_step):
        """
        Evaluate the model on evaluation dataset.

        :param eval_dataloader: DataLoader for evaluation data.
        :type eval_dataloader: DataLoader
        :param global_step: Current global step for logging.
        :type global_step: int
        :return: Dictionary of evaluation metrics.
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

        # Collect evaluation metrics
        all_rewards = []
        all_format_rewards = []
        all_accuracy_rewards = []
        all_response_lengths = []

        num_eval_batches = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                if len(batch) == 5:
                    eval_prompts, eval_images, eval_videos, eval_references, eval_labels = batch
                else:
                    eval_prompts, eval_images, eval_references, eval_labels = batch
                    eval_videos = None

                # Generate responses using experience maker (but don't train on them)
                # We reuse the experience maker but only for generation
                for experience in self.experience_maker.make_experience_list(
                    eval_prompts, eval_images, eval_videos, eval_references, eval_labels, **self.generate_kwargs
                ):
                    # Extract metrics from experience
                    if hasattr(experience, 'info') and experience.info is not None:
                        # Helper function to extract values robustly
                        def extract_values(val):
                            """
                            Extract scalar values from tensor/list/scalar, handling any shape.

                            :param val: Value to extract (tensor, list, or scalar).
                            :type val: torch.Tensor or list or scalar
                            :return: List of scalar values.
                            :rtype: list
                            """
                            if isinstance(val, torch.Tensor):
                                # Flatten tensor and convert to list (handles scalar, 1D, multi-D tensors)
                                return val.view(-1).cpu().tolist()
                            elif isinstance(val, (list, tuple)):
                                return list(val)
                            else:
                                # Scalar value (int, float, etc.)
                                return [float(val)]

                        if 'reward' in experience.info:
                            all_rewards.extend(extract_values(experience.info['reward']))

                        if 'reward_metrics' in experience.info:
                            reward_metrics = experience.info['reward_metrics']
                            if 'format_reward' in reward_metrics:
                                all_format_rewards.extend(extract_values(reward_metrics['format_reward']))
                            if 'accuracy_reward' in reward_metrics:
                                all_accuracy_rewards.extend(extract_values(reward_metrics['accuracy_reward']))

                        if 'response_length' in experience.info:
                            all_response_lengths.extend(extract_values(experience.info['response_length']))

                num_eval_batches += 1

                # Limit evaluation to avoid too long eval time
                if num_eval_batches >= len(eval_dataloader):
                    break

        # Compute statistics
        eval_metrics = {}

        if all_rewards:
            # [TENSOR-FIX] Handle both tensor lists and scalar lists
            if isinstance(all_rewards[0], torch.Tensor):
                rewards_tensor = torch.cat([t.to(torch.cuda.current_device()).float() for t in all_rewards])
            else:
                rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32, device=torch.cuda.current_device())
            eval_metrics["eval_reward_mean"] = rewards_tensor.mean().item()
            eval_metrics["eval_reward_std"] = rewards_tensor.std().item()
            eval_metrics["eval_reward_max"] = rewards_tensor.max().item()
            eval_metrics["eval_reward_min"] = rewards_tensor.min().item()

        # Handle both tensor lists and scalar lists
        if all_format_rewards:
            # [TENSOR-FIX] Use torch.cat() for tensor lists, torch.tensor() for scalar lists
            if all_format_rewards and isinstance(all_format_rewards[0], torch.Tensor):
                format_tensor = torch.cat([t.to(torch.cuda.current_device()).float() for t in all_format_rewards])
            else:
                format_tensor = torch.tensor(
                    all_format_rewards, dtype=torch.float32, device=torch.cuda.current_device()
                )
            eval_metrics["eval_format_reward"] = format_tensor.mean().item()

        if all_accuracy_rewards:
            # [TENSOR-FIX] Use torch.cat() for tensor lists, torch.tensor() for scalar lists
            if all_accuracy_rewards and isinstance(all_accuracy_rewards[0], torch.Tensor):
                accuracy_tensor = torch.cat([t.to(torch.cuda.current_device()).float() for t in all_accuracy_rewards])
            else:
                accuracy_tensor = torch.tensor(
                    all_accuracy_rewards, dtype=torch.float32, device=torch.cuda.current_device()
                )
            eval_metrics["eval_accuracy_reward"] = accuracy_tensor.mean().item()

        if all_response_lengths:
            # [TENSOR-FIX] Use torch.cat() for tensor lists, torch.tensor() for scalar lists
            if all_response_lengths and isinstance(all_response_lengths[0], torch.Tensor):
                lengths_tensor = torch.cat([t.to(torch.cuda.current_device()).float() for t in all_response_lengths])
            else:
                lengths_tensor = torch.tensor(
                    all_response_lengths, dtype=torch.float32, device=torch.cuda.current_device()
                )
            eval_metrics["eval_response_length"] = lengths_tensor.mean().item()

        eval_metrics["eval_num_samples"] = len(all_rewards)

        # Print evaluation results
        self.strategy.print(f"\n{'=' * 60}")
        self.strategy.print(f"Evaluation Results (Step {global_step}):")
        self.strategy.print(f"{'=' * 60}")
        for k, v in eval_metrics.items():
            self.strategy.print(f"  {k}: {v:.4f}")
        self.strategy.print(f"{'=' * 60}\n")

        # Set models back to train mode
        self.actor.train()
        if self.critic is not None:
            self.critic.train()

        return eval_metrics
