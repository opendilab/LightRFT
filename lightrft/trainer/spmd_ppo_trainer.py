"""
SPMD (Single Program Multiple Data) PPO Trainer for distributed reinforcement learning.

This module extends the base PPOTrainer with SPMD capabilities, enabling efficient
distributed training across multiple devices. It provides specialized implementations
for both text-only language models and vision-language models with optimized
tensor parallelism and distributed inference using vLLM.

The module includes:
- SPMDPPOTrainerBase: Base class with core SPMD functionality
- SPMDPPOTrainer: Implementation for Large Language Models (LLMs)
- SPMDPPOTrainerVL: Implementation for Vision-Language Models (VLMs)

Key features:
- FastExperienceMaker for improved throughput during experience collection
- Optimized memory management and communication patterns
- Support for both text-only and multi-modal reinforcement learning
- Efficient distributed training across multiple devices and nodes
"""

import time

import torch
from tqdm import tqdm

from lightrft.trainer import PPOTrainer, PPOTrainerVL
from lightrft.trainer.fast_exp_maker import FastExperienceMaker
from lightrft.utils.trajectory_saver import create_trajectory_saver

from lightrft.trainer.replay_buffer import make_experience_batch
from lightrft.trainer.replay_buffer_vl import make_experience_batch as make_experience_batch_vl
from lightrft.utils import init_logger

logger = init_logger(__name__)


class SPMDPPOTrainerBase:
    """
    PPO Trainer implementation optimized for Single Program Multiple Data (SPMD) execution.

    This trainer extends the base PPOTrainer with specialized handling for tensor parallelism
    and distributed inference using vLLM. It includes optimizations for experience collection
    and training across multiple devices.

    The base class provides core functionality for SPMD training including:
    - FastExperienceMaker integration for improved throughput
    - Tensor parallelism support with vLLM engine
    - Optimized memory management during training
    - Support for both text-only and vision-language models

    .. note:: Performance
        This implementation uses FastExperienceMaker for improved throughput during
        experience collection compared to the standard implementation.

    .. important:: Requirements
        Requires tensor parallelism configuration with engine_tp_size > 0.
    """
    def __init__(
        self,
        *args,
        loss_agg_mode: str = "seq-mean-token-mean",
        use_gspo: bool = False,
        VLM: bool = False,
        **kwargs,
    ):
        """
        Initialize the SPMD PPO Trainer base class.

        Sets up the distributed training environment, creates the experience maker,
        and configures the policy loss function for SPMD execution.

        :param args: Positional arguments passed to the parent PPOTrainer, including strategy, actor, critic,
                     reward_model, initial_model, etc.
        :type args: tuple
        :param loss_agg_mode: Mode for aggregating policy losses, either "seq-mean-token-mean" or other supported modes
        :type loss_agg_mode: str
        :param use_gspo: Whether to enable GSPO (Group Sequence Policy Optimization) mode
        :type use_gspo: bool
        :param VLM: Whether to use Vision-Language Model mode (True) or Language Model mode (False)
        :type VLM: bool
        :param kwargs: Keyword arguments for configuration including packing_samples, processor, and other parameters.
        :type kwargs: dict
        :raises AssertionError: If engine_tp_size is not properly configured (must be > 0)

        Example::

            trainer_base = SPMDPPOTrainerBase(
                strategy,
                actor_model,
                critic_model,
                reward_model,
                initial_model,
                ema_model,
                actor_optim,
                critic_optim,
                actor_scheduler,
                critic_scheduler,
                loss_agg_mode="seq-mean-token-mean",
                VLM=False,
                packing_samples=True
            )
        """
        self.VLM = VLM  # otherwise it's LLM
        self.packing_samples = kwargs.pop("packing_samples", False)
        self.print_replay_buffer_stats = kwargs.pop("print_replay_buffer_stats", False)
        # Note: super().__init__ will be called by child classes

        assert self.args.engine_tp_size > 0, "engine_tp_size should be larger than 0"
        self.vllm_mp_group = self.strategy.engine_mp_group

        self.vllm_engine = self.strategy.inference_engine

        torch.distributed.barrier()
        # TODO: here we pass a list of concrete params, this may collapse in future versions.
        # Create experience maker with appropriate parameters
        processor = kwargs.pop("processor", None)

        self.experience_maker = FastExperienceMaker(
            self.actor,
            self.critic,
            self.reward_model,
            self.initial_model,
            self.tokenizer,
            self.prompt_max_len,
            self.kl_ctl,
            self.strategy,
            self.remote_rm_url,
            self.reward_fn,
            self.reward_fn_label_map,
            self.reward_recipe,
            packing_samples=self.packing_samples,
            processor=processor,
        )

        # Initialize loss function based on mode
        policy_loss_kwargs = {"loss_agg_mode": loss_agg_mode, "use_gspo": use_gspo}
        if use_gspo:
            policy_loss_kwargs.update({
                "normalize_advantages": kwargs.get("normalize_advantages", True),
                "use_sequence_rewards": kwargs.get("use_sequence_rewards", True)
            })

        self.use_gspo = use_gspo

        # Initialize trajectory saver if enabled
        self.trajectory_saver = create_trajectory_saver(self.args, self.tokenizer)

        # Validate num_trajectories_to_save parameter if trajectory saving is enabled
        if self.trajectory_saver is not None:
            if not hasattr(self.args, 'num_trajectories_to_save'):
                raise ValueError(
                    "num_trajectories_to_save must be provided in args when trajectory saving is enabled. "
                    "Please add --num_trajectories_to_save <value> to your command line arguments."
                )
            self.num_trajectories_to_save = self.args.num_trajectories_to_save
        else:
            self.num_trajectories_to_save = None

        self.dataloader_pin_memory = False
        if torch.distributed.get_rank() == 0:
            print(self.args, flush=True)

    def ppo_train(self, global_steps=0):  # Currently using this rewritten ppo_train() method
        """
        Execute a full PPO training iteration with SPMD optimizations.

        This method processes the replay buffer data, trains the actor and critic models
        for multiple epochs, and updates the inference engine weights. It includes
        optimized memory management and distributed training coordination.

        The training process includes:
        1. Data preprocessing for distributed execution
        2. Multi-epoch training with experience batching
        3. Loss computation and optimization
        4. Memory cleanup and weight synchronization

        :param global_steps: Current global step counter for logging and scheduling
        :type global_steps: int
        :return: Dictionary of training metrics averaged across all training steps
        :rtype: Dict[str, float]

        Example::

            metrics = trainer.ppo_train(global_steps=100)
            print(f"Policy loss: {metrics['policy_loss']}")
            print(f"Critic loss: {metrics['critic_loss']}")
        """
        torch.cuda.synchronize()
        train_begin = time.time()

        torch.cuda.empty_cache()
        self.strategy.maybe_load_optimizer(self.actor_optim)
        all_items = self.strategy.sp_data_processor.preprocess(self.replay_buffer.items)

        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                range(0, len(all_items), self.micro_train_batch_size),
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for i in pbar:
                items = all_items[i:i + self.micro_train_batch_size]
                if self.VLM:
                    experience = make_experience_batch_vl(items, packing_samples=self.packing_samples)
                else:
                    experience = make_experience_batch(items, packing_samples=self.packing_samples)
                experience.to_device(device)

                # Call training_step which will handle both GSPO and standard modes
                status = self.training_step(experience, global_steps)

                # for DP
                # weighted mean for kl
                if "kl" in status:
                    status["kl"] *= status["response_length"]
                    status = self.strategy.all_reduce(status)
                    status["kl"] /= status["response_length"]

                # Training epoch progress bar: show per-batch metrics for detailed monitoring
                short_status = {}

                if "policy_loss" in status:
                    short_status = {
                        "pg": status["policy_loss"],  # policy gradient loss
                        "rm": status["reward"],  # per-batch reward (instantaneous)
                        "ret": status["return"],  # per-batch return (instantaneous)
                        "glen": status["response_length"],  # per-batch response length
                        "tlen": status["total_length"],  # per-batch total length
                        "kl": status["kl"],  # KL divergence
                        "act_lr": status["actor_lr"],  # actor learning rate
                    }

                if "critic_loss" in status:
                    short_status["cri"] = status["critic_loss"]
                    short_status["vals"] = status["values"]
                    short_status["cri_lr"] = status["critic_lr"]

                if "ptx_loss" in status:
                    short_status["ptx"] = status["ptx_loss"]

                status_list.append(status)
                pbar.set_postfix(short_status)

        # Short status keys added for progress bar display:
        # "pg": policy_loss
        # "rm": reward
        # "ret": return
        # "glen": response_length
        # "tlen": total_length
        # "kl": KL divergence
        # "act_lr": actor_lr
        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)

        # ========== Aggregate step-level reward metrics from replay buffer ==========
        # NOTE: These metrics are aggregated from ALL experiences in the current step's
        # replay buffer (e.g., 640 experiences if rollout_batch_size=128, n_samples=5).
        # They represent the TRUE statistics of the rollout phase, NOT the training phase
        # micro-batch averages which are less representative.
        #
        # Naming convention:
        # - "*_mean" suffix: mean across all experiences in this step
        # - "step_*" prefix: clarifies this is per-step aggregation, not per-episode
        if self.replay_buffer.items:
            all_rewards = []
            all_format_rewards = []
            all_accuracy_rewards = []
            all_model_rewards = []
            all_rule_rewards = []
            all_advantages = []
            all_returns = []
            all_response_lengths = []

            for item in self.replay_buffer.items:
                # Collect rewards
                if hasattr(item, 'info') and item.info is not None and 'reward' in item.info:
                    all_rewards.append(item.info['reward'])

                # Collect detailed reward metrics from info dict
                if hasattr(item, 'info') and item.info is not None and 'reward_metrics' in item.info:
                    reward_metrics = item.info['reward_metrics']
                    if 'format_reward' in reward_metrics:
                        all_format_rewards.append(reward_metrics['format_reward'])
                    if 'accuracy_reward' in reward_metrics:
                        all_accuracy_rewards.append(reward_metrics['accuracy_reward'])
                    if 'model_reward' in reward_metrics:
                        all_model_rewards.append(reward_metrics['model_reward'])
                    if 'rule_reward' in reward_metrics:
                        all_rule_rewards.append(reward_metrics['rule_reward'])

                # Collect advantages and returns
                if hasattr(item, 'advantages') and item.advantages is not None:
                    all_advantages.append(item.advantages)
                if hasattr(item, 'returns') and item.returns is not None:
                    all_returns.append(item.returns)
                if hasattr(item, 'info') and item.info is not None and 'response_length' in item.info:
                    all_response_lengths.append(item.info['response_length'])

            # Compute statistics
            # [TENSOR-FIX] Handle both tensor lists and scalar lists for all reward types
            if all_rewards:
                # Handle both tensor lists (from batched rewards) and scalar lists
                if isinstance(all_rewards[0], torch.Tensor):
                    rewards_tensor = torch.cat([t.to(device).float() for t in all_rewards])
                else:
                    rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32, device=device)
                # Use "step_*" prefix to clarify this is per-step aggregation, not per-episode
                status_mean["step_reward_mean"] = rewards_tensor.mean().item()
                status_mean["step_reward_std"] = rewards_tensor.std().item()
                status_mean["step_reward_max"] = rewards_tensor.max().item()
                status_mean["step_reward_min"] = rewards_tensor.min().item()

            if all_format_rewards:
                # [TENSOR-FIX] Handle both tensor lists and scalar lists
                if isinstance(all_format_rewards[0], torch.Tensor):
                    format_tensor = torch.cat([t.to(device).float() for t in all_format_rewards])
                else:
                    format_tensor = torch.tensor(all_format_rewards, dtype=torch.float32, device=device)
                status_mean["format_reward_mean"] = format_tensor.mean().item()
                status_mean["format_reward_std"] = format_tensor.std().item()

            if all_accuracy_rewards:
                # [TENSOR-FIX] Handle both tensor lists and scalar lists
                if isinstance(all_accuracy_rewards[0], torch.Tensor):
                    accuracy_tensor = torch.cat([t.to(device).float() for t in all_accuracy_rewards])
                else:
                    accuracy_tensor = torch.tensor(all_accuracy_rewards, dtype=torch.float32, device=device)
                status_mean["accuracy_reward_mean"] = accuracy_tensor.mean().item()
                status_mean["accuracy_reward_std"] = accuracy_tensor.std().item()

            if all_model_rewards:
                # [TENSOR-FIX] Handle both tensor lists and scalar lists
                if isinstance(all_model_rewards[0], torch.Tensor):
                    model_tensor = torch.cat([t.to(device).float() for t in all_model_rewards])
                else:
                    model_tensor = torch.tensor(all_model_rewards, dtype=torch.float32, device=device)
                if model_tensor.abs().sum() > 0:  # Only log if model rewards are non-zero
                    status_mean["model_reward_mean"] = model_tensor.mean().item()
                    self.strategy.print(f" model_reward_mean: {status_mean['model_reward_mean']}")

            if all_rule_rewards:
                # [TENSOR-FIX] Handle both tensor lists and scalar lists
                if isinstance(all_rule_rewards[0], torch.Tensor):
                    rule_tensor = torch.cat([t.to(device).float() for t in all_rule_rewards])
                else:
                    rule_tensor = torch.tensor(all_rule_rewards, dtype=torch.float32, device=device)
                if rule_tensor.abs().sum() > 0:  # Only log if rule rewards are non-zero
                    status_mean["rule_reward_mean"] = rule_tensor.mean().item()
                    self.strategy.print(f"rule_reward_mean: {status_mean['rule_reward_mean']}")

            # For advantages, returns, and lengths, they are already lists of tensors,
            # so torch.cat() is the correct function to use.
            if all_advantages:
                advantages_tensor = torch.cat(all_advantages)
                status_mean["advantages_mean"] = advantages_tensor.mean().item()
                status_mean["advantages_std"] = advantages_tensor.std().item()
                status_mean["advantages_max"] = advantages_tensor.max().item()
                status_mean["advantages_min"] = advantages_tensor.min().item()

            if all_returns:
                returns_tensor = torch.cat(all_returns)
                status_mean["returns_mean"] = returns_tensor.mean().item()
                status_mean["returns_std"] = returns_tensor.std().item()

            if all_response_lengths:
                # [TENSOR-FIX] Handle both tensor lists and scalar lists
                if isinstance(all_response_lengths[0], torch.Tensor):
                    lengths_tensor = torch.cat([t.to(device).float() for t in all_response_lengths])
                else:
                    lengths_tensor = torch.tensor(all_response_lengths, dtype=torch.float32, device=device)
                status_mean["response_length_mean"] = lengths_tensor.float().mean().item()
                status_mean["response_length_std"] = lengths_tensor.float().std().item()

            # Print detailed reward breakdown (only on rank 0)
            if self.print_replay_buffer_stats and self.strategy.is_rank_0():
                self.strategy.print("\n" + "=" * 60)
                self.strategy.print("📊 Detailed Step Statistics")
                self.strategy.print("=" * 60)

                if all_rewards:
                    self.strategy.print(
                        f"🎁 Total Reward:     {status_mean['step_reward_mean']:.4f} ± {status_mean['step_reward_std']:.4f} "  # noqa
                        f"(min={status_mean['step_reward_min']:.4f}, max={status_mean['step_reward_max']:.4f})"
                    )

                if all_format_rewards:
                    self.strategy.print(
                        f"📝 Format Reward:    {status_mean['format_reward_mean']:.4f} ± {status_mean['format_reward_std']:.4f}"  # noqa
                    )

                if all_accuracy_rewards:
                    self.strategy.print(
                        f"✅ Accuracy Reward:  {status_mean['accuracy_reward_mean']:.4f} ± {status_mean['accuracy_reward_std']:.4f}"  # noqa
                    )

                if all_advantages:
                    self.strategy.print(
                        f"📈 Advantages:       {status_mean['advantages_mean']:.4f} ± {status_mean['advantages_std']:.4f} "  # noqa
                        f"(min={status_mean['advantages_min']:.4f}, max={status_mean['advantages_max']:.4f})"
                    )

                if all_returns:
                    self.strategy.print(
                        f"💰 Returns:          {status_mean['returns_mean']:.4f} ± {status_mean['returns_std']:.4f}"
                    )

                if all_response_lengths:
                    self.strategy.print(
                        f"📏 Response Length:  {status_mean['response_length_mean']:.1f} ± {status_mean['response_length_std']:.1f} tokens"  # noqa
                    )

                self.strategy.print("=" * 60 + "\n")

        torch.cuda.empty_cache()

        self.strategy.maybe_offload_optimizer(self.actor_optim)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.strategy.print(f"PPO Train TIMECOST {time.time() - train_begin}")
        self.strategy.report_memory("after train, opt offloaded, before update weights")
        self.strategy.print(torch.cuda.memory_summary())
        self.strategy.update_engine_weights(self.actor)

        # Save trajectories at the end of ppo_train, BEFORE replay buffer is cleared
        # This ensures we have data to save when trajectory saving is enabled
        if global_steps % self.args.save_steps == 0:
            self.save_trajectories(global_steps)

        return status_mean

    def save_trajectories(self, global_step: int):
        """
        Save experience trajectories if trajectory saving is enabled.

        This method is called during checkpoint saving to store sample trajectories
        for debugging and analysis purposes. If trajectory analysis is enabled,
        it also logs statistics to wandb.

        :param global_step: Current global training step
        :type global_step: int
        """
        if self.trajectory_saver is not None and self.replay_buffer.items:
            # Check if trajectory analysis is enabled

            output_path, stats = self.trajectory_saver.save_trajectories(
                experiences=self.replay_buffer.items,
                step=global_step,
                num_samples=self.num_trajectories_to_save,
                prefix="trajectories",
                compute_stats=self.args.trajectory_analysis
            )

            # Log statistics to wandb if available
            if stats and self.args.trajectory_analysis and hasattr(self, 'strategy') and self.strategy.is_rank_0():
                # Try to get wandb from strategy or parent class
                if hasattr(self.strategy, 'args') and self.strategy.args.use_wandb:
                    try:
                        import wandb
                        if wandb.run is not None:
                            # Prefix with train/ for consistency
                            wandb_stats = {f"train/{k}": v for k, v in stats.items()}
                            wandb_stats["train/global_step"] = global_step
                            wandb.log(wandb_stats, step=global_step)
                    except (ImportError, AttributeError):
                        pass


class SPMDPPOTrainer(SPMDPPOTrainerBase, PPOTrainer):
    """
    PPOTrainer for SPMD on Large Language Models and Multi-modal Large Language Models.

    This class combines the SPMD (Single Program Multiple Data) base functionality with the
    standard PPOTrainer for efficient distributed training of large language models (LLMs)
    and multi-modal large language models (MLLMs). It supports training across multiple
    devices and nodes with optimized communication patterns for both text-only and
    multi-modal reinforcement learning scenarios.

    The trainer provides:
    - Distributed PPO training with tensor parallelism
    - Efficient experience collection using FastExperienceMaker
    - Memory-optimized training loops
    - Support for various loss aggregation modes
    - Integration with vLLM inference engine

    Example::

        trainer = SPMDPPOTrainer(
            strategy=my_strategy,
            actor=actor_model,
            critic=critic_model,
            reward_model=reward_model,
            initial_model=reference_model,
            ema_model=ema_model,
            actor_optim=actor_optimizer,
            critic_optim=critic_optimizer,
            actor_scheduler=actor_scheduler,
            critic_scheduler=critic_scheduler,
            tokenizer=tokenizer,
            # Additional PPO parameters
            max_epochs=5,
            micro_train_batch_size=16
        )

        # Train for multiple iterations
        for step in range(training_steps):
            trainer.make_experience()
            metrics = trainer.ppo_train(step)
    """
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        Initialize the SPMD PPO Trainer for language models.

        Creates a trainer instance optimized for distributed training of language models
        using SPMD execution patterns. Inherits from both SPMDPPOTrainerBase and PPOTrainer
        to combine SPMD optimizations with standard PPO functionality.

        :param args: Positional arguments passed to the parent PPOTrainer including strategy, actor, critic,
                     reward_model, initial_model, ema_model, actor_optim, critic_optim, actor_scheduler,
                     critic_scheduler.
        :type args: tuple
        :param kwargs: Keyword arguments for configuration including training hyperparameters like max_epochs,
                       micro_train_batch_size, eps_clip, value_clip, etc.
        :type kwargs: dict

        Example::

            trainer = SPMDPPOTrainer(
                strategy,
                actor_model,
                critic_model,
                reward_model,
                reference_model,
                ema_model,
                actor_optimizer,
                critic_optimizer,
                actor_scheduler,
                critic_scheduler,
                tokenizer=my_tokenizer,
                loss_agg_mode="seq-mean-token-mean",
                packing_samples=True,
                max_epochs=5,
                micro_train_batch_size=16
            )
        """
        # First initialize the PPOTrainer parent
        PPOTrainer.__init__(self, *args, **kwargs)
        # Then initialize our base class
        SPMDPPOTrainerBase.__init__(self, *args, VLM=False, **kwargs)


class SPMDPPOTrainerVL(SPMDPPOTrainerBase, PPOTrainerVL):
    """
    PPOTrainer for SPMD with Vision-Language Models (VLM).

    This class combines the SPMD base functionality with the VLM-specific PPOTrainer
    for efficient distributed training of vision-language models. It extends the standard
    VLM training capabilities with SPMD optimizations for better performance across
    multiple devices.

    Key features for VLM training:
    - Multi-modal experience collection and processing
    - Vision-language specific batch creation
    - Processor integration for image and text handling
    - Optimized memory management for large multi-modal models

    Example::

        trainer = SPMDPPOTrainerVL(
            strategy=my_strategy,
            actor=actor_model,
            critic=critic_model,
            reward_model=reward_model,
            initial_model=reference_model,
            ema_model=ema_model,
            actor_optim=actor_optimizer,
            critic_optim=critic_optimizer,
            actor_scheduler=actor_scheduler,
            critic_scheduler=critic_scheduler,
            tokenizer=tokenizer,
            processor=image_processor,  # Required for VLM
            # Additional PPO parameters
            max_epochs=5,
            micro_train_batch_size=16
        )

        # Train for multiple iterations
        for step in range(training_steps):
            trainer.make_experience()
            metrics = trainer.ppo_train(step)
    """
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        Initialize the SPMD PPO Trainer for vision-language models.

        Creates a trainer instance specifically designed for distributed training of
        vision-language models using SPMD execution patterns. Requires a processor
        for handling multi-modal inputs.

        :param args: Positional arguments passed to the parent PPOTrainerVL including strategy, actor, critic,
                     reward_model, initial_model, ema_model, actor_optim, critic_optim, actor_scheduler,
                     critic_scheduler.
        :type args: tuple
        :param kwargs: Keyword arguments for configuration, must include 'processor' for image processing along
                       with other training parameters.
        :type kwargs: dict
        :raises AssertionError: If processor is not provided or is None.

        Example::

            trainer = SPMDPPOTrainerVL(
                strategy,
                vlm_actor,
                vlm_critic,
                vlm_reward_model,
                vlm_reference,
                vlm_ema_model,
                actor_optimizer,
                critic_optimizer,
                actor_scheduler,
                critic_scheduler,
                tokenizer=my_tokenizer,
                processor=my_image_processor,  # Required!
                loss_agg_mode="seq-mean-token-mean",
                max_epochs=5,
                micro_train_batch_size=8
            )
        """
        # First initialize the PPOTrainerVL parent
        PPOTrainerVL.__init__(self, *args, **kwargs)
        # Then initialize our base class
        assert "processor" in kwargs and kwargs["processor"] is not None, "processor is required for SPMDPPOTrainerVL"
        SPMDPPOTrainerBase.__init__(self, *args, VLM=True, **kwargs)
