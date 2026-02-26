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

from lightrft.utils import get_tokenizer_processor_vl, ensure_video_input_available
from lightrft.models import ActorVL
from lightrft.strategy import get_strategy
from lightrft.trainer import SPMDPPOTrainerVL
from lightrft.utils import add_arguments

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from meme_dataset import MemeOnlineRLDataset
from reward_model import load_reward_models, reward_fn

ensure_video_input_available()


def train(args):
    # configure strategy
    strategy = get_strategy(args)

    ds_train_cfg = strategy.get_ds_train_config(is_actor=True) if not args.fsdp else None
    ds_eval_cfg = strategy.get_ds_eval_config(offload=False) if not args.fsdp else None

    # configure model (optionally meta_init to save CPU memory)
    with strategy.init_model_context(meta_init=getattr(args, "meta_init", False)):
        actor = ActorVL(
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
            high_entropy_token_ratio=getattr(args, "high_entropy_token_ratio", 0.0),
        )

    if args.actor_init_on_gpu:
        actor = actor.to(torch.cuda.current_device())

    # pre-prepare is used for saving RAM memory when training 72B model
    if args.fsdp:
        setattr(actor, "is_actor", True)
        actor = strategy.prepare_model(actor, is_training=True)

    # configure tokenizer and processor
    tokenizer, processor = get_tokenizer_processor_vl(
        args.pretrain, actor.model, "left", use_fast=not strategy.args.disable_fast_tokenizer
    )
    assert processor is not None, "processor is None"

    if args.freeze_prefix:
        # visual for qwenvl, vision_model for internvl, mlp1 for linear layer in internvl
        freeze_prefix = ["visual", "vision_model", "mlp1"]
        frozen_params_count = 0
        total_params_count = 0
        for name, param in actor.model.named_parameters():
            total_params_count += 1
            if any(name.startswith(prefix) for prefix in freeze_prefix):
                param.requires_grad = False
                frozen_params_count += 1
        strategy.print(
            f"Froze {frozen_params_count}/{total_params_count} parameters based on prefixes: {freeze_prefix}"
        )

    critic = None

    strategy.report_memory("before loaded reward models in main entry")
    reward_models, reward_tokenizers, reward_processors, label_map = load_reward_models(
        args.reward_pretrain, strategy, use_engine=args.rm_use_engine
    )
    strategy.print(f"label_map: {label_map}")
    strategy.report_memory("after loaded reward models in main entry")

    strategy.print(actor)

    # load weights for reference actor
    if args.init_kl_coef == 0:
        initial_model = None
    else:
        initial_model = ActorVL(
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=ds_eval_cfg,
            packing_samples=args.packing_samples,
            fused_linear_logprob=args.fused_linear_logprob,
        )

        if args.fsdp:
            initial_model = strategy.prepare_model(initial_model, is_training=False, shard_size=strategy.world_size)
            strategy.offload_model(initial_model)

    # prepare datasets (meme-specific: annotation_path + root_dir)
    annotation_path = args.annotation_path
    root_dir = args.root_dir
    prompts_dataset = MemeOnlineRLDataset(
        annotation_path=annotation_path,
        root_dir=root_dir,
        shuffle=True,
        processor=processor,
    )
    strategy.print(f"Loaded {len(prompts_dataset)} samples for prompts.")

    # prepare dataloader
    prompts_dataloader = strategy.setup_dataloader(
        prompts_dataset,
        args.rollout_batch_size // strategy.world_size,
        True,
        True,
        collate_fn=prompts_dataset.collate_fn
    )
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

    # load checkpoint
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(os.path.join(args.ckpt_path, "_actor")):
        _, states = strategy.load_ckpt(
            actor.model, os.path.join(args.ckpt_path, "_actor"), optimizer=actor_optim, scheduler=actor_scheduler
        )
        if args.critic_pretrain:
            strategy.load_ckpt(critic, os.path.join(args.ckpt_path, "_critic"))
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    os.makedirs(args.save_path, exist_ok=True)
    strategy.report_memory("after models init")

    strategy.setup_inference_engine(args, engine_type=args.engine_type, actor=actor)

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
        eps_clip=args.eps_clip,
        gamma=args.gamma,
        lambd=args.lambd,
        init_kl_coef=args.init_kl_coef,
        kl_target=args.kl_target,
        ema_beta=0.992,
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
        reward_tokenizers=reward_tokenizers,
        save_hf_ckpt=args.save_hf_ckpt,
        disable_ds_ckpt=args.disable_ds_ckpt,
        packing_samples=args.packing_samples,
    )

    trainer.fit(
        args,
        prompts_dataloader=prompts_dataloader,
        pretrain_dataloader=pretrain_dataloader,
        eval_dataloader=None,
        consumed_samples=consumed_samples,
        num_update_steps_per_episodes=num_update_steps_per_episodes,
    )

    # save model checkpoint after fitting on only rank0
    strategy.save_model(
        actor,
        tokenizer,
        args.save_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--engine_type", type=str, default="vllm", help="Choose inference engine type: vllm, sglang")

    # Checkpoint
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_ppo")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

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
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--lambd", type=float, default=0.95, help="PPO GAE lambd")
    parser.add_argument("--gamma", type=float, default=1, help="PPO GAE gamma")
    parser.add_argument("--micro_train_batch_size", type=int, default=4, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--freeze_prefix",
        action="store_true",
        default=False,
        help="Freeze the prefix part (e.g. vision encoder) of the actor model"
    )
    parser.add_argument(
        "--n_samples_per_prompt", type=int, default=1, help="number of responses for each prompt in generation"
    )
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
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
    parser.add_argument("--reward_clip_range", type=float, nargs=2, default=(-10, 10), help="Reward clip range")

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
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument(
        "--disable_logprobs_flashattn",
        action="store_true",
        default=False,
        help="Disable flash attn implementation in log_probs calculation"
    )

    # Reinforce
    parser.add_argument(
        "--advantage_estimator",
        type=str,
        choices=["gae", "reinforce", "rloo", "reinforce_baseline", "group_norm"],
        default="gae",
        help="Choose advantage estimation method: gae, reinforce, rloo, reinforce_baseline, group_norm",
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

    # Custom dataset (GodsMeme)
    parser.add_argument(
        "--annotation_path",
        type=str,
        default=
        "/fs-computility/niuyazhe/shared/xueyingyi/xueyingyi/cot_picture/Eimages/annotations/all/train_data.jsonl",
        help="Path to meme annotation JSONL",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/fs-computility/niuyazhe/shared/xueyingyi/xueyingyi/cot_picture/Eimage_drawn",
        help="Root directory for meme images",
    )
    parser.add_argument("--prompt_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--prompt_split", type=str, default="train")

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
    parser.add_argument(
        "--limit_mm_image_per_prompt",
        type=int,
        default=-1,
        help="the max image number of each text in multi model for inference backend"
    )

    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument(
        "--use_cpg_loss",
        action="store_true",
        default=False,
        help="whether to use the clipped policy gradient loss from CPGD"
    )

    add_arguments(parser)

    args = parser.parse_args()

    if args.advantage_estimator not in ["gae"]:
        args.critic_pretrain = None

    if args.advantage_estimator in ["rloo", "reinforce_baseline", "group_norm"]:
        assert args.n_samples_per_prompt > 1, f"{args.advantage_estimator} requires n_samples_per_prompt > 1"

    if args.use_kl_loss:
        if args.kl_estimator not in ["k2", "k3"]:
            print(f"Recommend setting {args.kl_estimator} to 'k2' or 'k3' when using KL as a loss")
    else:
        if args.kl_estimator not in ["k1"]:
            print(f"Recommend setting {args.kl_estimator} to 'k1' when not using KL as a loss.")

    train(args)
