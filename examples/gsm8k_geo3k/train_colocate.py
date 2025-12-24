"""
GRPO Training with Co-located Reward Models

This script implements Group Relative Policy Optimization (GRPO) training
with co-located reward models for reinforcement learning with verifiable rewards (RLVR) and reinforcement learning from human feedback (RLHF).

Key Features:
    - Supports both text-only and vision-language models
    - Flexible strategy: DeepSpeed ZeRO or FSDP
    - Meta device initialization for memory optimization
    - EMA (Exponential Moving Average) model support
    - Dynamic sampling and overlong buffer penalties (DAPO)

Main Components:
    - Actor: Policy model being trained
    - Critic: Value model for advantage estimation (optional for GRPO)
    - Reward Models: Multiple models for evaluating different aspects
    - Initial Model: Reference model for KL divergence

Training Pipeline:
    1. Load and initialize models (actor, critic, reward models)
    2. Setup data loaders (prompts + optional pretrain data)
    3. Configure optimizers and schedulers
    4. Run PPO/GRPO training loop via SPMDPPOTrainerVL

Usage:
    python train_colocate.py --pretrain <model_path> --reward_pretrain <rm_config> ...

For more details on arguments, see the argument parser at the bottom of this file.
"""
import argparse
import itertools
import math
import re
import os
import sys
import json
from datetime import datetime
from typing import Callable, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightrft.utils import add_arguments, ensure_video_input_available
ensure_video_input_available()

from lightrft.datasets import PromptDatasetVL, SFTDatasetVL
from lightrft.utils import blending_datasets, get_tokenizer_processor_vl
from lightrft.models.actor_language import ActorLanguage
from lightrft.models.actor_vl import ActorVL

from lightrft.strategy import get_strategy
from lightrft.trainer.spmd_ppo_trainer import SPMDPPOTrainerVL

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from reward_models_utils import load_reward_models, reward_fn, RECIPE


def train(args):
    """
    Main training function for GRPO with co-located reward models.

    Training workflow:
        1. Initialize strategy (DeepSpeed or FSDP)
        2. Initialize models with meta_init option for memory efficiency
        3. Load reward models (multiple types supported)
        4. Setup dataloaders for prompts and optional pretrain data
        5. Configure optimizers and schedulers
        6. Setup inference engine (vLLM or SGLang)
        7. Run training loop via SPMDPPOTrainerVL
        8. Save final model

    Args:
        args: Parsed command-line arguments containing all training configuration

    Key configurations:
        - meta_init: Initialize models on meta device to save CPU RAM
        - freeze_prefix: Freeze vision encoder during training
        - fsdp: Use FSDP instead of DeepSpeed
        - rm_use_engine: Use SGLang engine for reward models
    """
    # configure strategy
    strategy = get_strategy(args)

    ds_train_cfg = strategy.get_ds_train_config(is_actor=True) if not args.fsdp else None
    ds_eval_cfg = strategy.get_ds_eval_config(offload=False)  if not args.fsdp else None

    # configure model
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

    # Load reward models (multiple types: value, safety, knowledge, etc.)
    strategy.report_memory(f"before loaded reward models in main entry")
    reward_models, reward_tokenizers, label_map = load_reward_models(
        raw_reward_pretrain=args.reward_pretrain,
        strategy=strategy,
        use_engine=args.rm_use_engine,
    )
    strategy.print(f"label_map: {label_map}")
    strategy.report_memory(f"after loaded reward models in main entry")

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
            initial_model = strategy.prepare_model(initial_model, is_training=False, shard_size=8)
            strategy.offload_model(initial_model)

    if args.enable_ema:
        ema_model = Actor(
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=ds_eval_cfg,
        )
    else:
        ema_model = None

    # configure tokenizer and processor
    tokenizer, processor = get_tokenizer_processor_vl(
        args.pretrain, actor.model, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer
    )
    assert processor is not None, "processor is None"

   # ==================== Data Loading Optimization ====================
    # The following sections now rely on the robust `blending_datasets` function.
    # We add more logging for clarity.

    # Prepare prompts dataset
    strategy.print(f"Loading prompts dataset from: {args.prompt_data} with split: {args.prompt_split}")
    prompts_data = blending_datasets(
        args.prompt_data,
        args.prompt_data_probs,
        strategy,
        args.seed,
        return_eval=False,
        train_split=args.prompt_split,
    )
    
    prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
    prompts_dataset = PromptDatasetVL(prompts_data, tokenizer, processor, args.prompt_max_len, strategy, input_template=args.input_template)
    strategy.print(f"Loaded {len(prompts_dataset)} samples for prompts.")

    # Prepare evaluation dataset
    eval_dataloader = None
    if args.eval_data or args.eval_split:
        eval_data_path = args.eval_data if args.eval_data else args.prompt_data
        if eval_data_path:
            strategy.print(f"Loading evaluation dataset from {eval_data_path}, split='{args.eval_split}'")
            eval_data = blending_datasets(
                eval_data_path, "1.0", strategy, args.seed, return_eval=False,
                # Note: `train_split` parameter is used to specify the desired split name for evaluation data.
                train_split=args.eval_split,
            )
            if len(eval_data) == 0:
                 strategy.print(f"Warning: Evaluation dataset at {eval_data_path} with split '{args.eval_split}' is empty. Skipping evaluation.")
            else:
                eval_data = eval_data.select(range(min(args.max_eval_samples, len(eval_data))))
                
                eval_dataset = PromptDatasetVL(eval_data, tokenizer, processor, args.prompt_max_len, strategy, input_template=args.input_template)
                eval_dataloader = strategy.setup_dataloader(
                    eval_dataset, args.rollout_batch_size // strategy.world_size, False, False, collate_fn=eval_dataset.collate_fn
                )
                strategy.print(f"Evaluation dataset loaded: {len(eval_dataset)} samples")
        else:
            strategy.print("Warning: eval_split specified but no data path available for evaluation.")

    # Prepare pretrain dataset
    pretrain_dataloader = None
    if args.pretrain_data:
        strategy.print(f"Loading pretrain dataset from: {args.pretrain_data} with split: {args.pretrain_split}")
        pretrain_data = blending_datasets(
            args.pretrain_data, args.pretrain_data_probs, strategy, args.seed,
            return_eval=False, train_split=args.pretrain_split,
        )
        if len(pretrain_data) == 0:
            strategy.print(f"Warning: Pretrain dataset at {args.pretrain_data} is empty. PTX loss will not be applied.")
            pretrain_dataloader = None
        else:
            pretrain_max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
            # Calculate total samples needed for pretraining
            total_pretrain_samples = args.max_epochs * len(prompts_dataset) * args.n_samples_per_prompt
            pretrain_data_subset = pretrain_data.select(range(min(len(pretrain_data), total_pretrain_samples)))
            
            pretrain_dataset = SFTDatasetVL(
                pretrain_data_subset, tokenizer, pretrain_max_len, strategy, pretrain_mode=True,
            )
            strategy.print(f"Loaded {len(pretrain_dataset)} samples for pretraining.")
            pretrain_dataloader = itertools.cycle(
                iter(
                    strategy.setup_dataloader(
                        pretrain_dataset, args.micro_train_batch_size, True, True, pretrain_dataset.collate_fn,
                    )
                )
            )
    else:
        pretrain_dataloader = None

    # Prepare prompts dataloader
    prompts_dataloader = strategy.setup_dataloader(
        prompts_dataset, args.rollout_batch_size // strategy.world_size, True, True, collate_fn=prompts_dataset.collate_fn
    )

    if args.pretrain_data:
        pretrain_dataloader = itertools.cycle(
            iter(
                strategy.setup_dataloader(
                    pretrain_dataset,
                    args.micro_train_batch_size,
                    True,
                    True,
                    pretrain_dataset.collate_fn,
                )
            )
        )
    else:
        pretrain_dataloader = None

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
    ) = strategy.prepare_models_and_optimizers(actor, critic, reward_models, initial_model, args, max_steps)

    strategy.print(reward_models)

    if ema_model:
        ema_model._offload = True
        ema_model = strategy.prepare(ema_model, is_rlhf=True)

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
        ema_model,
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
        ema_beta=0.992,
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
        reward_fn_label_map=label_map,
        reward_recipe=RECIPE,
        reward_tokenizers=reward_tokenizers,
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

    trainer.fit(args, prompts_dataloader=prompts_dataloader, pretrain_dataloader=pretrain_dataloader, eval_dataloader=eval_dataloader, consumed_samples=0, num_update_steps_per_episodes=num_update_steps_per_episodes)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(
        ema_model if args.enable_ema else actor,
        tokenizer,
        args.save_path,
    )

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
    parser.add_argument("--loss_agg_mode", type=str, default='seq-mean-token-mean',
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
    parser.add_argument("--enable_ema", action="store_true", help="Enable EMA checkpoint for the model.")
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

    # Evaluation dataset
    parser.add_argument("--eval_data", type=str, default=None, help="HF evaluation dataset name or path (default: use prompt_data)")
    parser.add_argument("--eval_split", type=str, default="test", help="Evaluation data split (default: test)")
    parser.add_argument("--max_eval_samples", type=int, default=500, help="Maximum number of samples to evaluate (default: 500)")
    
    parser.add_argument("--pretrain_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--pretrain_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--pretrain_split", type=str, default="train")
    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--images_key", type=str, default="image", help="JSON dataser key for images")
    parser.add_argument("--reference_key", type=str, default="reference", help="JSON dataset key for reference answers")
    parser.add_argument("--label_key", type=str, default="label", help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )

    parser.add_argument("--system_prompt", type=str, default=None, help="HF System Prompt")


    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="lightrft_train_ppo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)

    # MultiModal
    parser.add_argument("--limit_mm_image_per_prompt", type=int, default=-1, help="the max image number of each text in multi model for inference backend")

    # CPGD
    parser.add_argument("--use_cpg_loss", action="store_true", default=False, help="whether to use the clipped policy gradient loss from CPGD")

    add_arguments(parser)

    args = parser.parse_args()


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

    if args.advantage_estimator in ["gae", "cpgd"] and args.use_kl_loss:
        warnings.warn(
            "Using use_kl_loss=True with non-normalized advantage estimator "
            "may result in double KL penalty. Consider disabling --use_kl_loss "
            "or using --advantage_estimator group_norm"
        )

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        # Patch hub to download models from modelscope to speed up.
        patch_hub()

    train(args)