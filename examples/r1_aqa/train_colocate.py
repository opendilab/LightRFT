"""
GRPO Training for R1-AQA (Audio Question Answering) with Co-located Reward Models

This script adapts LightRFT's GRPO training pipeline for audio question-answering
on Qwen2-Audio, faithfully migrating R1-AQA's (xiaomi-research/r1-aqa) training
logic into LightRFT's framework.

Key adaptations from the standard VL pipeline:
    - Uses AudioPromptDataset instead of PromptDatasetVL (audio content handling)
    - Uses AudioMultimodalProcessor instead of MultimodalDataProcessor
    - Uses ActorAL (from lightrft.models.actor_al) for native Qwen2-Audio support
    - Patches strategy._build_multimodal_inputs for audio multi_modal_data
    - Rule-based rewards: accuracy(0/1) + format(0/1) following R1-AQA

Migration mapping (R1-AQA → LightRFT):
    R1-AQA src/train.py        → this file (train_colocate.py)
    R1-AQA GRPOTrainer         → SPMDPPOTrainerVL + group_norm advantage
    R1-AQA num_generations=8   → --n_samples_per_prompt 8
    R1-AQA temperature=1.0     → --temperature 1.0
    R1-AQA max_prompt_length=512 → --prompt_max_len 512
    R1-AQA accuracy+format     → reward_fn in reward_models_utils.py
    R1-AQA DeepSpeed ZeRO3     → --zero_stage 3 (or --fsdp)

Usage:
    python examples/r1_aqa/train_colocate.py --pretrain Qwen/Qwen2-Audio-7B-Instruct ...
"""

import argparse
import itertools
import json
import math
import os
import re
import sys
import warnings
from datetime import datetime
from typing import Callable, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightrft.utils import add_arguments
from lightrft.datasets import SFTDatasetVL
from lightrft.models.actor_al import ActorAL
from lightrft.models.actor_language import ActorLanguage
from lightrft.strategy import get_strategy
from lightrft.trainer.spmd_ppo_trainer import SPMDPPOTrainerVL
from lightrft.utils import blending_datasets, get_tokenizer_processor_vl

# Local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from reward_models_utils import RECIPE, load_reward_models, reward_fn
from audio_pipeline import (
    AudioPromptDataset,
    patch_strategy_for_audio,
    patch_experience_maker_for_audio,
)


def train(args):
    """
    Main training function for R1-AQA GRPO with co-located rule-based rewards.

    Training workflow:
        1. Initialize strategy (DeepSpeed or FSDP)
        2. Initialize ActorAL model (Qwen2-Audio) or ActorLanguage (text-only)
        3. Load rule-based rewards (no neural reward models)
        4. Setup audio prompt dataloader
        5. Configure optimizers and schedulers
        6. Setup inference engine (vLLM or SGLang)
        7. Apply audio pipeline patches
        8. Run training loop via SPMDPPOTrainerVL
        9. Save final model
    """
    # ==================== Strategy ====================
    strategy = get_strategy(args)

    ds_train_cfg = strategy.get_ds_train_config(is_actor=True) if not args.fsdp else None
    ds_eval_cfg = strategy.get_ds_eval_config(offload=False) if not args.fsdp else None

    # ==================== Model Initialization ====================
    with strategy.init_model_context(meta_init=args.meta_init):
        strategy.print(f"Initializing models with meta_init={args.meta_init}")

        # Select Actor class based on text_only flag
        if args.text_only:
            Actor = ActorLanguage
        else:
            Actor = ActorAL

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
            high_entropy_token_ratio=args.high_entropy_token_ratio,
        )

    if args.actor_init_on_gpu:
        actor = actor.to(torch.cuda.current_device())

    if args.fsdp:
        setattr(actor, "is_actor", True)
        actor = strategy.prepare_model(actor, is_training=True)

    # Optionally freeze audio encoder
    if args.freeze_prefix:
        freeze_prefix = ["audio"]  # Freeze audio encoder (analogous to freezing visual encoder)
        frozen_params_count = 0
        total_params_count = 0
        for name, param in actor.model.named_parameters():
            total_params_count += 1
            if any(name.startswith(prefix) for prefix in freeze_prefix):
                param.requires_grad = False
                frozen_params_count += 1
        strategy.print(
            f"Froze {frozen_params_count}/{total_params_count} parameters "
            f"based on prefixes: {freeze_prefix}"
        )

    # No critic for GRPO
    critic = None

    # ==================== Reward Models ====================
    strategy.report_memory("before loaded reward models")
    reward_models, reward_tokenizers, label_map = load_reward_models(
        raw_reward_pretrain=args.reward_pretrain,
        strategy=strategy,
        use_engine=args.rm_use_engine,
    )
    strategy.print(f"label_map: {label_map}")
    strategy.report_memory("after loaded reward models")

    strategy.print(actor)

    # ==================== Reference Model (for KL) ====================
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
        )
        if args.fsdp:
            shard_size = (
                args.initial_model_shard_size
                if args.initial_model_shard_size is not None
                else strategy.world_size
            )
            initial_model = strategy.prepare_model(initial_model, is_training=False, shard_size=shard_size)
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

    # ==================== Tokenizer & Processor ====================
    tokenizer, processor = get_tokenizer_processor_vl(
        args.pretrain,
        actor.model,
        "left",
        use_fast=not strategy.args.disable_fast_tokenizer,
    )

    # Ensure we have the correct Qwen2AudioProcessor (AutoProcessor may
    # fall back to a generic text processor that ignores the `audios` kwarg).
    try:
        from transformers import Qwen2AudioProcessor
        if not isinstance(processor, Qwen2AudioProcessor):
            strategy.print(
                f"[WARN] AutoProcessor loaded {type(processor).__name__}, "
                "re-loading as Qwen2AudioProcessor"
            )
            processor = Qwen2AudioProcessor.from_pretrained(
                args.pretrain, trust_remote_code=True
            )
    except ImportError:
        strategy.print("[WARN] Qwen2AudioProcessor not available in this transformers version")
    assert processor is not None, "Qwen2-Audio processor is required"

    # ==================== Data Loading ====================
    strategy.print(f"Loading prompts dataset from: {args.prompt_data}")
    prompts_data = blending_datasets(
        args.prompt_data,
        args.prompt_data_probs,
        strategy,
        args.seed,
        return_eval=False,
        train_split=args.prompt_split,
    )
    prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))

    # Use AudioPromptDataset instead of PromptDatasetVL
    prompts_dataset = AudioPromptDataset(
        prompts_data,
        tokenizer,
        processor,
        args.prompt_max_len,
        strategy,
        input_template=args.input_template,
    )
    strategy.print(f"Loaded {len(prompts_dataset)} audio prompt samples")

    # Evaluation dataset
    eval_dataloader = None
    if args.eval_data or args.eval_split:
        eval_data_path = args.eval_data if args.eval_data else args.prompt_data
        if eval_data_path:
            strategy.print(f"Loading evaluation dataset from {eval_data_path}")
            eval_data = blending_datasets(
                eval_data_path, "1.0", strategy, args.seed,
                return_eval=False, train_split=args.eval_split,
            )
            if len(eval_data) > 0:
                eval_data = eval_data.select(range(min(args.max_eval_samples, len(eval_data))))
                eval_dataset = AudioPromptDataset(
                    eval_data, tokenizer, processor, args.prompt_max_len, strategy,
                    input_template=args.input_template,
                )
                eval_dataloader = strategy.setup_dataloader(
                    eval_dataset,
                    args.rollout_batch_size // strategy.world_size,
                    False, False,
                    collate_fn=eval_dataset.collate_fn,
                )
                strategy.print(f"Evaluation dataset: {len(eval_dataset)} samples")

    # Pretrain dataset (optional PTX loss)
    pretrain_dataloader = None
    if args.pretrain_data:
        strategy.print(f"Loading pretrain dataset from: {args.pretrain_data}")
        pretrain_data = blending_datasets(
            args.pretrain_data, args.pretrain_data_probs, strategy, args.seed,
            return_eval=False, train_split=args.pretrain_split,
        )
        if len(pretrain_data) > 0:
            pretrain_max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
            total_pretrain_samples = args.max_epochs * len(prompts_dataset) * args.n_samples_per_prompt
            pretrain_data_subset = pretrain_data.select(
                range(min(len(pretrain_data), total_pretrain_samples))
            )
            pretrain_dataset = SFTDatasetVL(
                pretrain_data_subset, tokenizer, pretrain_max_len, strategy, pretrain_mode=True,
            )
            pretrain_dataloader = itertools.cycle(
                iter(strategy.setup_dataloader(
                    pretrain_dataset, args.micro_train_batch_size, True, True, pretrain_dataset.collate_fn,
                ))
            )

    # Prompts dataloader
    prompts_dataloader = strategy.setup_dataloader(
        prompts_dataset,
        args.rollout_batch_size // strategy.world_size,
        True, True,
        collate_fn=prompts_dataset.collate_fn,
    )

    # ==================== Scheduler ====================
    num_update_steps_per_episodes = (
        len(prompts_dataset) * args.n_samples_per_prompt // args.train_batch_size * args.max_epochs
    )
    max_steps = math.ceil(args.num_episodes * num_update_steps_per_episodes)

    # Gradient checkpointing
    if args.gradient_checkpointing:
        actor.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # ==================== Prepare Models & Optimizers ====================
    (
        (actor, actor_optim, actor_scheduler),
        (critic, critic_optim, critic_scheduler),
        reward_models,
        initial_model,
    ) = strategy.prepare_models_and_optimizers(
        actor, critic, reward_models, initial_model, args, max_steps
    )

    strategy.print(reward_models)

    if ema_model:
        ema_model._offload = True
        ema_model = strategy.prepare(ema_model, is_rlhf=True)

    # Load checkpoint
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(os.path.join(args.ckpt_path, "_actor")):
        _, states = strategy.load_ckpt(
            actor.model, os.path.join(args.ckpt_path, "_actor"),
            optimizer=actor_optim, scheduler=actor_scheduler,
        )
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    os.makedirs(args.save_path, exist_ok=True)
    strategy.report_memory("after models init")

    # ==================== Inference Engine ====================
    strategy.report_memory("before setup_inference_engine")
    strategy.setup_inference_engine(args, engine_type=args.engine_type, actor=actor)
    strategy.report_memory("after setup_inference_engine")

    # ==================== Apply Audio Patches ====================
    # Patch strategy for audio multimodal inputs
    patch_strategy_for_audio(strategy)

    # ==================== Trainer ====================
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
        # Generation params (matching R1-AQA defaults)
        do_sample=True,
        max_new_tokens=args.generate_max_len,
        max_length=args.max_len,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # Reward model
        reward_fn=reward_fn,
        reward_fn_label_map=label_map,
        reward_recipe=RECIPE,
        reward_tokenizers=reward_tokenizers,
        save_hf_ckpt=args.save_hf_ckpt,
        disable_ds_ckpt=args.disable_ds_ckpt,
        packing_samples=args.packing_samples,
        # DAPO / overlong
        dynamic_sampling=args.dynamic_sampling,
        overlong_buffer=args.overlong_buffer,
        overlong_buffer_len=args.overlong_buffer_len,
        overlong_buffer_penalty_factor=args.overlong_buffer_penalty_factor,
        print_replay_buffer_stats=args.print_replay_buffer_stats,
    )

    # Patch the experience maker for audio processing
    patch_experience_maker_for_audio(
        trainer.experience_maker, processor, tokenizer, args.prompt_max_len
    )

    # ==================== Training ====================
    trainer.fit(
        args,
        prompts_dataloader=prompts_dataloader,
        pretrain_dataloader=pretrain_dataloader,
        eval_dataloader=eval_dataloader,
        consumed_samples=0,
        num_update_steps_per_episodes=num_update_steps_per_episodes,
    )

    # ==================== Save ====================
    strategy.save_model(
        ema_model if args.enable_ema else actor,
        tokenizer,
        args.save_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Engine
    parser.add_argument("--engine_type", type=str, default="vllm",
                        help="Inference engine: vllm or sglang")
    parser.add_argument("--text_only", action="store_true", default=False,
                        help="Text-only mode (no multimodal). Default False for audio tasks.")

    # Checkpoint
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--save_trajectories", action="store_true", default=False)
    parser.add_argument("--num_trajectories_to_save", type=int, default=10)
    parser.add_argument("--mark_high_entropy_tokens", action="store_true", default=False)
    parser.add_argument("--trajectory_analysis", action="store_true", default=False)
    parser.add_argument("--print_replay_buffer_stats", action="store_true", default=False)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_ppo")
    parser.add_argument("--ckpt_path_local", type=str, default=None)
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=int(1e8))
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # DAPO
    parser.add_argument("--dynamic_sampling", action="store_true", default=False)
    parser.add_argument("--overlong_buffer", action="store_true", default=False)
    parser.add_argument("--overlong_buffer_len", type=int, default=1024)
    parser.add_argument("--overlong_buffer_penalty_factor", type=float, default=1.0)

    # PPO/GRPO
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=512)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1)
    # R1-AQA default: max_prompt_length=512
    parser.add_argument("--prompt_max_len", type=int, default=512,
                        help="Max tokens for each prompt (R1-AQA default: 512)")
    parser.add_argument("--generate_max_len", type=int, default=1024,
                        help="Max tokens to generate")
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--ptx_coef", type=float, default=0.05)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--loss_agg_mode", type=str, default="seq-mean-token-mean")
    parser.add_argument("--use_gspo", action="store_true", default=False)
    parser.add_argument("--normalize_advantages", action="store_true", default=True)
    parser.add_argument("--use_sequence_rewards", action="store_true", default=True)
    parser.add_argument("--value_clip", type=float, default=0.2)
    parser.add_argument("--lambd", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--micro_train_batch_size", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--normalize_reward_for_critic", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=1.0)
    # R1-AQA default: temperature=1.0
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--freeze_prefix", action="store_true", default=False)
    parser.add_argument("--freezing_actor_steps", type=int, default=-1)
    # R1-AQA default: num_generations=8
    parser.add_argument("--n_samples_per_prompt", type=int, default=8,
                        help="Number of responses per prompt in GRPO (R1-AQA default: 8)")
    parser.add_argument("--save_value_network", action="store_true", default=False)
    # R1-AQA default: lr not explicitly set, using 1e-6 as reasonable default
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--init_kl_coef", type=float, default=0.01)
    parser.add_argument("--kl_estimator", type=str, default="k3",
                        choices=["k1", "k2", "k3"],
                        help="GRPO uses k3 as KL estimator")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95))

    # Reward/Advantage Norm/Clip
    parser.add_argument("--reward_running_norm", action="store_true", default=False)
    parser.add_argument("--reward_running_norm_minus_mean", action="store_true", default=False)
    parser.add_argument("--reward_clip", type=float, default=0.0)
    parser.add_argument("--advantages_norm", action="store_true", default=False)
    parser.add_argument("--advantage_clip", type=float, default=0.0)

    # DeepSpeed
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--enable_ema", action="store_true", default=False)
    parser.add_argument("--zpg", type=int, default=1)
    parser.add_argument("--adam_offload", action="store_true", default=False)
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--aux_loss_coef", type=float, default=0)
    parser.add_argument("--grad_accum_dtype", type=str, default=None)
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--disable_logprobs_flashattn", action="store_true", default=False)

    # FSDP
    parser.add_argument("--no_shard_vit", action="store_true", default=False)
    parser.add_argument("--meta_init", action="store_true", default=False)
    parser.add_argument("--initial_model_shard_size", type=int, default=None)

    # Advantage estimator
    parser.add_argument("--advantage_estimator", type=str,
                        choices=["gae", "reinforce", "rloo", "reinforce_baseline", "group_norm", "cpgd", "reinforce++"],
                        default="group_norm",
                        help="Advantage estimation method. R1-AQA uses GRPO = group_norm")
    parser.add_argument("--use_kl_loss", action="store_true", default=False)

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # Models
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--reward_pretrain", type=str, default=None)
    parser.add_argument("--remote_rm_url", type=str, default=None)
    parser.add_argument("--critic_pretrain", type=str, default=None)
    parser.add_argument("--value_head_prefix", type=str, default="score")

    # Dataset
    parser.add_argument("--prompt_data", type=str, default=None)
    parser.add_argument("--prompt_data_probs", type=str, default="1.0")
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--eval_data", type=str, default=None)
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--max_eval_samples", type=int, default=500)
    parser.add_argument("--pretrain_data", type=str, default=None)
    parser.add_argument("--pretrain_data_probs", type=str, default="1.0")
    parser.add_argument("--pretrain_split", type=str, default="train")
    parser.add_argument("--input_key", type=str, default="prompt")
    parser.add_argument("--images_key", type=str, default="audio_path")
    parser.add_argument("--reference_key", type=str, default="reference")
    parser.add_argument("--label_key", type=str, default="label")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument("--apply_chat_template", action="store_true", default=False)
    parser.add_argument("--system_prompt", type=str, default=None)

    # wandb
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="lightrft_r1_aqa")
    parser.add_argument("--wandb_run_name", type=str,
                        default="r1_aqa_%s" % datetime.now().strftime("%m%dT%H:%M"))

    # TensorBoard
    parser.add_argument("--use_tensorboard", type=str, default=None)

    # ModelScope
    parser.add_argument("--use_ms", action="store_true", default=False)

    # MultiModal
    parser.add_argument("--limit_mm_image_per_prompt", type=int, default=-1)

    # CPGD
    parser.add_argument("--use_cpg_loss", action="store_true", default=False)

    # High-entropy token filtering
    parser.add_argument("--high_entropy_token_ratio", type=float, default=0.0)

    add_arguments(parser)

    args = parser.parse_args()

    # GRPO validation
    if args.advantage_estimator not in ["gae"]:
        args.critic_pretrain = None
    elif args.critic_pretrain is None:
        args.critic_pretrain = args.pretrain

    if args.advantage_estimator in ["rloo", "reinforce_baseline", "group_norm"]:
        assert args.n_samples_per_prompt > 1, (
            f"{args.advantage_estimator} requires n_samples_per_prompt > 1"
        )

    if args.use_kl_loss:
        if args.kl_estimator not in ["k2", "k3"]:
            print(f"Recommend setting {args.kl_estimator} to 'k2' or 'k3' when using KL as loss")

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub
        patch_hub()

    train(args)
