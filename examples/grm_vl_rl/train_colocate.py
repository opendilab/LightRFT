"""
GRPO Training with Verifiable Rewards (RLVR) for Vision-Language Tasks
Supported Modalities: Text, Image, and Video.

This script implements Group Relative Policy Optimization (GRPO) for Vision-Language Models (VLMs). 
It integrates rule-based reward functions to enable Reinforcement Learning from Verifiable Rewards (RLVR), 

Key Features:
    - Supports both text-only and vision-language models
    - Rule-based verifiable rewards (Format checking and Accuracy verification)
    - Flexible strategy: DeepSpeed ZeRO or FSDP
    - Meta device initialization for memory optimization
    - Dynamic sampling and overlong buffer penalties (DAPO)

Main Components:
    - Actor: Policy model being trained
    - Critic: Value model for advantage estimation (optional for GRPO)
    - Initial Model: Reference model for KL divergence

Training Pipeline:
    1. Load and initialize models (actor, initial model, critic)
    2. Setup data loaders (prompts + optional pretrain data)
    3. Configure optimizers and schedulers
    4. Run PPO/GRPO training loop via SPMDPPOTrainerVL

Usage:
    python train_colocate.py --pretrain <model_path> ...

For more details on arguments, see the argument parser at the bottom of this file.
"""
import os
import sys
import math
import torch
import argparse
import itertools
from datetime import datetime

from lightrft.strategy import get_strategy
from lightrft.datasets import RFTDatasetVL
from lightrft.models.actor_language import ActorLanguage
from lightrft.models.actor_vl import ActorVL
from lightrft.trainer import SPMDPPOTrainerVL
from lightrft.utils import add_arguments, ensure_video_input_available, get_tokenizer_processor_vl
ensure_video_input_available()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from reward_fn_utils import reward_fn, RECIPE


def train(args: argparse.Namespace) -> None:
    """
    Main training function for GRPO with Rule-based function.
    Support vision-language models for image and video inputs.

    Training workflow:
        1. Initialize strategy (DeepSpeed or FSDP)
        2. Initialize models with meta_init option for memory efficiency
        3. Setup dataloaders for prompts (supporting images and videos) and optional pretrain data
        4. Configure optimizers and schedulers
        5. Setup inference engine (vLLM or SGLang)
        6. Run training loop via SPMDPPOTrainerVL
        7. Save final model

    :param args: Parsed command-line arguments containing all training configuration
    :type args: argparse.Namespace

    :return: None
    :rtype: None

    **Key configurations:**

    - meta_init: Initialize models on meta device to save CPU RAM
    - freeze_prefix: Freeze vision encoder during training
    - fsdp: Use FSDP instead of DeepSpeed

    **Example:**

    .. code-block:: python

        # Assuming args is already defined via argparse
        train(args)
    """
    # configure strategy
    strategy = get_strategy(args)

    ds_train_cfg = strategy.get_ds_train_config(is_actor=True) if not args.fsdp else None
    ds_eval_cfg = strategy.get_ds_eval_config(offload=False)  if not args.fsdp else None

    # ==================== Model Initialization ====================
    # Initialize all models within init_model_context for memory efficiency.
    # When meta_init=True, models are created on "meta" device as empty shells,
    # fundamentally resolving CPU OOM issues.
    with strategy.init_model_context(meta_init=args.meta_init):
        strategy.print(f"Initializing models with meta_init={args.meta_init}")

        # Select Actor class based on text_only flag
        if args.text_only:
            Actor = ActorLanguage
        else:
            Actor = ActorVL

        # Initialize Actor (policy model)
        actor = Actor(
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            ds_config=ds_train_cfg,
            packing_samples=args.packing_samples,
            disable_logprobs_flashattn=args.disable_logprobs_flashattn,
            fused_linear_logprob=args.fused_linear_logprob,
        )

    if args.actor_init_on_gpu:
        actor = actor.to(torch.cuda.current_device())

    # pre-prepare is used for saving RAM memory when training 72B model
    if args.fsdp:
        setattr(actor, "is_actor", True)
        actor = strategy.prepare_model(actor, is_training=True)

    # Optionally freeze parameters (e.g., vision encoder)
    if args.freeze_prefix:
        freeze_prefix = ["visual"]
        frozen_params_count = 0
        total_params_count = 0
        for name, param in actor.model.named_parameters():
            total_params_count += 1
            if any(name.startswith(prefix) for prefix in freeze_prefix):
                param.requires_grad = False
                frozen_params_count += 1
        strategy.print(f"Froze {frozen_params_count}/{total_params_count} parameters based on prefixes: {freeze_prefix}")

    if args.critic_pretrain:
        critic = get_vlm_for_sequence_regression(
            args.critic_pretrain,
            "critic",
            normalize_reward=args.normalize_reward_for_critic,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            ds_config=ds_train_cfg,
            value_head_prefix=args.value_head_prefix,
            init_value_head=strategy.args.pretrain == strategy.args.critic_pretrain,
        )
    else:
        critic = None

    strategy.print(actor)
    strategy.print(critic)

    # load weights for reference actor
    if args.init_kl_coef == 0:
        initial_model = None
    else:
        initial_model = Actor(
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=ds_eval_cfg,
            packing_samples=args.packing_samples,
            fused_linear_logprob=args.fused_linear_logprob,
        )

        if args.fsdp:
            initial_model = strategy.prepare_model(initial_model, is_training=False)
            strategy.offload_model(initial_model)

    # configure tokenizer and processor
    tokenizer, processor = get_tokenizer_processor_vl(
        args.pretrain, actor.model, "left", use_fast=not strategy.args.disable_fast_tokenizer
    )
    assert processor is not None, "processor is None"

   # ==================== Data Loading Optimization ====================
    # The following sections now rely on the robust `blending_datasets` function.
    # We add more logging for clarity.

    # Prepare prompts dataset
    strategy.print(f"Loading prompts dataset from: {args.prompt_data}")

    # Parse system prompt path if provided. We keep a `system_prompt` variable
    # which contains either the loaded YAML (if path ends with .yaml/.yml) or
    # the string passed directly.
    system_prompt = None
    if getattr(args, "system_prompt_path", None):
        system_prompt_path = args.system_prompt_path
        # If it's a YAML file, load it; otherwise treat as literal prompt string
        if system_prompt_path.endswith(".yaml") or system_prompt_path.endswith(".yml"):
            try:
                import yaml
                with open(system_prompt_path, "r") as f:
                    system_prompt = yaml.safe_load(f)
            except Exception as e:
                strategy.print(f"Error loading system prompt from YAML: {e}")
        else:
            system_prompt = system_prompt_path

    prompts_dataset = RFTDatasetVL(
        args.prompt_data, 
        processor,
        tokenizer, 
        strategy, 
        args.prompt_max_len,
        config={
            "task_instruction": system_prompt,
            "video_fps": args.fps,
            "max_pixels": args.max_pixels,
        },
    )
    strategy.print(f"Loaded {len(prompts_dataset)} samples for prompts.")

    # TODO: Implement evaluation dataset and dataloader
    # Prepare evaluation dataset
    eval_dataloader = None

    # Prepare prompts dataloader
    prompts_dataloader = strategy.setup_dataloader(
        prompts_dataset, args.rollout_batch_size // strategy.world_size, True, True, collate_fn=prompts_dataset.collate_fn
    )

    # for scheduler
    num_update_steps_per_episodes = (
        len(prompts_dataset) * args.n_samples_per_prompt // args.train_batch_size * args.max_epochs
    )
    max_steps = math.ceil(args.num_episodes * num_update_steps_per_episodes)

    # gradient_checkpointing
    if args.gradient_checkpointing:
        actor.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )
        if critic is not None:
            critic.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

    (
        (actor, actor_optim, actor_scheduler),
        (critic, critic_optim, critic_scheduler),
        reward_models,
        initial_model,
    ) = strategy.prepare_models_and_optimizers(actor, critic, [], initial_model, args, max_steps)

    # load checkpoint
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(os.path.join(args.ckpt_path, "_actor")):
        _, states = strategy.load_ckpt(actor.model, os.path.join(args.ckpt_path, "_actor"),
                                       optimizer=actor_optim, scheduler=actor_scheduler)
        if args.critic_pretrain:
            strategy.load_ckpt(critic, os.path.join(args.ckpt_path, "_critic"))
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    os.makedirs(args.save_path, exist_ok=True)
    strategy.report_memory("after models init")

    strategy.report_memory("before setup_inference_engine")
    strategy.setup_inference_engine(args, engine_type=args.engine_type, actor=actor)
    strategy.report_memory("after setup_inference_engine")

    # configure Trainer
    trainer = SPMDPPOTrainerVL(
        strategy,
        actor,
        critic,
        reward_models,
        initial_model,
        None,
        actor_optim,
        critic_optim,
        actor_scheduler,
        critic_scheduler,
        max_epochs=args.max_epochs,
        micro_train_batch_size=args.micro_train_batch_size,
        micro_rollout_batch_size=args.micro_rollout_batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        tokenizer=tokenizer,
        processor=processor,
        prompt_max_len=args.prompt_max_len,
        value_clip=args.value_clip,
        eps_clip=args.eps_clip,
        loss_agg_mode=args.loss_agg_mode,
        use_gspo=args.use_gspo,
        normalize_advantages=args.normalize_advantages,
        use_sequence_rewards=args.use_sequence_rewards,
        gamma=args.gamma,
        lambd=args.lambd,
        init_kl_coef=args.init_kl_coef,
        kl_target=args.kl_target,
        ptx_coef=args.ptx_coef,
        max_norm=args.max_norm,
        # for GPT generation
        do_sample=True,
        max_new_tokens=args.generate_max_len,
        max_length=args.max_len,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # reward model
        reward_fn=reward_fn,
        reward_recipe=RECIPE,
        save_hf_ckpt=args.save_hf_ckpt,
        disable_ds_ckpt=args.disable_ds_ckpt,
        packing_samples=args.packing_samples,
        # overlong_reward
        dynamic_sampling=args.dynamic_sampling,
        overlong_buffer=args.overlong_buffer,
        overlong_buffer_len=args.overlong_buffer_len,
        overlong_buffer_penalty_factor=args.overlong_buffer_penalty_factor,
        print_replay_buffer_stats=args.print_replay_buffer_stats,
    )

    # None is eval_dataloader placehoder
    trainer.fit(
        args, 
        prompts_dataloader=prompts_dataloader, 
        pretrain_dataloader=None, 
        eval_dataloader=None, 
        consumed_samples=0, 
        num_update_steps_per_episodes=num_update_steps_per_episodes
    )

    # save model checkpoint after fitting on only rank0
    if args.critic_pretrain and args.save_value_network:
        strategy.save_model(
            critic,
            tokenizer,
            args.save_path + "_critic",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--engine_type", type=str, default="vllm", help="Choose inference engine type: vllm, sglang")
    parser.add_argument("--text_only", action="store_true", default=False)

    # Checkpoint
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--save_trajectories", action="store_true", default=False, help="Save experience trajectories to JSON for debugging")
    parser.add_argument("--num_trajectories_to_save", type=int, default=10, help="Number of trajectories to save per checkpoint")
    parser.add_argument("--trajectory_analysis", action="store_true", default=False, help="Enable trajectory analysis metrics (repeat_score, reflection_pattern, policy_entropy) and log to wandb")
    parser.add_argument("--print_replay_buffer_stats", action="store_true", default=False, help="Print detailed replay buffer statistics during training")
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_ppo")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # DAPO
    parser.add_argument("--dynamic_sampling", action="store_true", default=False, help="Enable DAPO dynamic sampling strategy")
    parser.add_argument("--overlong_buffer", action="store_true", default=False, help="Apply overlong sequence buffer in DAPO")
    parser.add_argument("--overlong_buffer_len", type=int, default=1024, help="Max token threshold for overlong buffer")
    parser.add_argument("--overlong_buffer_penalty_factor", type=float, default=1.0, help="Penalty scaling factor for overlong sequences, <1 discourages long outputs; >1 encourages them")

    # PPO
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=512)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for each prompt")
    parser.add_argument("--generate_max_len", type=int, default=1024, help="Max tokens to generate in PPO")
    parser.add_argument("--max_len", type=int, default=None, help="deprecated max_len")
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--ptx_coef", type=float, default=0.05, help="PPO-ptx loss coef")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--loss_agg_mode", type=str, default='seq-mean-token-sum',
        help="Loss aggregation mode. Options: ['token-mean', 'seq-mean-token-sum', 'seq-mean-token-mean', 'seq-mean-token-sum-norm']")
    parser.add_argument("--use_gspo", action="store_true", default=False, help="Enable GSPO (Group Sequence Policy Optimization) mode")
    parser.add_argument("--normalize_advantages", action="store_true", default=True, help="Enable advantage normalization in GSPO")
    parser.add_argument("--use_sequence_rewards", action="store_true", default=True, help="Use sequence-level rewards in GSPO")
    parser.add_argument("--value_clip", type=float, default=0.2, help="PPO value clip range")
    parser.add_argument("--lambd", type=float, default=0.95, help="PPO GAE lambd")
    parser.add_argument("--gamma", type=float, default=1, help="PPO GAE gamma")
    parser.add_argument("--micro_train_batch_size", type=int, default=4, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--normalize_reward_for_critic", action="store_true", default=False, help="Enable Reward Normalization in critic model")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--freeze_prefix", action="store_true", default=False, help="Freeze the prefix part (e.g. vision encoder) of the actor model")
    parser.add_argument("--freezing_actor_steps", type=int, default=-1, help="Used for critic initialization")
    parser.add_argument(
        "--n_samples_per_prompt", type=int, default=1, help="number of responses for each prompt in generation"
    )
    parser.add_argument("--save_value_network", action="store_true", default=False, help="Save critic model")
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--init_kl_coef", type=float, default=0.01, help="KL penalty in PPO")
    parser.add_argument(
        "--kl_estimator",
        type=str,
        default="k1",
        choices=["k1", "k2", "k3"],
        help=(
            "In GRPO, k3 is utilized as the loss function, while k2, when used as the loss, is nearly equivalent to k1."
        ),
    )
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")
    
    # Reward/Advantage Norm/Clip Arguments
    parser.add_argument("--reward_running_norm", action="store_true", default=False, help="Enable running normalization for rewards.")
    parser.add_argument("--reward_running_norm_minus_mean", action="store_true", default=False, help="When using reward normalization, subtract the mean; otherwise, only scale by the std.")
    parser.add_argument("--reward_clip", type=float, default=0.0, help="Clip rewards to the range [-reward_clip, reward_clip]. 0.0 means no clipping.")
    parser.add_argument("--advantages_norm", action="store_true", default=False, help="Enable whitening for advantages.")
    parser.add_argument("--advantage_clip", type=float, default=0.0, help="Clip advantages to the range [-advantage_clip, advantage_clip]. 0.0 means no clipping.")

    # DeepSpeed
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--disable_logprobs_flashattn", action="store_true", default=False, help="Disable flash attn implementation in log_probs calculation")

    # FSDP
    parser.add_argument("--no_shard_vit", action="store_true", default=False, help="Disable sharding for vision transformer")
    parser.add_argument("--meta_init", action="store_true", default=False, help="Initialize models on meta device to save CPU memory")

    # Reinforce
    parser.add_argument(
        "--advantage_estimator",
        type=str,
        choices=["gae", "reinforce", "rloo", "reinforce_baseline", "group_norm", "cpgd", "reinforce++"],
        default="gae",
        help="Choose advantage estimation method: gae, reinforce, rloo, reinforce_baseline, group_norm, reinforce++",
    )

    parser.add_argument("--use_kl_loss", action="store_true", default=False, help="whether to use KL loss from GRPO")

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--remote_rm_url", type=str, default=None, help="remote RM API")
    parser.add_argument("--critic_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--value_head_prefix", type=str, default="score")

    # Custom dataset
    parser.add_argument("--prompt_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second for sampling video data.")
    parser.add_argument("--max_pixels", type=int, default=360*28*28, help="Maximum pixels for each image frame.")

    # Evaluation dataset
    parser.add_argument("--eval_data", type=str, default=None, help="HF evaluation dataset name or path (default: use prompt_data)")
    parser.add_argument("--eval_split", type=str, default="test", help="Evaluation data split (default: test)")

    parser.add_argument("--pretrain_split", type=str, default="train")
    parser.add_argument("--system_prompt_path", type=str, default=None, help="Path to Prompt YAML or a literal system prompt string")

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_ppo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    # MultiModal
    parser.add_argument("--limit_mm_image_per_prompt", type=int, default=-1, help="the max image number of each text in multi model for inference backend")
    parser.add_argument("--limit_mm_video_per_prompt", type=int, default=-1, help="the max video number of each text in multi model for inference backend")

    # CPGD
    parser.add_argument("--use_cpg_loss", action="store_true", default=False, help="whether to use the clipped policy gradient loss from CPGD")

    add_arguments(parser)

    args = parser.parse_args()

    if args.prompt_data:
        args.prompt_data = args.prompt_data.split(",")

    if args.advantage_estimator not in ["gae"]:
        args.critic_pretrain = None
    elif args.critic_pretrain is None:
        args.critic_pretrain = args.pretrain

    if args.advantage_estimator in ["rloo", "reinforce_baseline", "group_norm"]:
        assert args.n_samples_per_prompt > 1, f"{args.advantage_estimator} requires n_samples_per_prompt > 1"

    if args.use_kl_loss:
        if args.kl_estimator not in ["k2", "k3"]:
            print(f"Recommend setting {args.kl_estimator} to 'k2' or 'k3' when using KL as a loss")
    else:
        if args.kl_estimator not in ["k1"]:
            print(f"Recommend setting {args.kl_estimator} to 'k1' when not using KL as a loss.")

    train(args)