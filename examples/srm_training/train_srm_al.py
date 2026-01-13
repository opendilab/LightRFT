import os
import math
import argparse
from datetime import datetime

import torch
import torch.distributed as dist

from lightrft.strategy import get_strategy
from lightrft.utils import get_tokenizer_processor_vl, add_arguments
from lightrft.datasets import RankDatasetAL
from lightrft.models import ScalarRewardModelAL
from lightrft.trainer.srm_trainer_al import SRMTrainerAL


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    ds_train_cfg = strategy.get_ds_train_config(is_actor=True) if not args.fsdp else None

    # configure model
    reward_model = ScalarRewardModelAL(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=ds_train_cfg,
        pooling_method=args.pooling_method,
        scale_for_train=args.scale_for_train,
        probing_layer=args.probing_layer,
        head_types=args.heads_types,
    )
    strategy.print(reward_model)

    # configure tokenizer and processor
    tokenizer, processor = get_tokenizer_processor_vl(
        args.pretrain, reward_model.model, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer
    )
    assert processor is not None, "processor is None"
    
    # prepare datasets and dataloaders
    train_dataset = RankDatasetAL(
        args.train_data,
        tokenizer=tokenizer,
        strategy=strategy,
        processor=processor,
        max_length=args.prompt_max_len,
        config={
            "input_template": args.input_template,
            "task_instruction": args.task_instruction,
        }
    )

    # prepare dataloader
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        batch_size=args.train_batch_size // strategy.world_size,
        pin_memory=True,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )

    if args.eval_data:
        eval_dataset = RankDatasetAL(
            args.eval_data,
            processor=processor,
            tokenizer=tokenizer,
            strategy=strategy,
            max_length=args.prompt_max_len,
            config={
                "input_template": args.input_template,
                "task_instruction": args.task_instruction,
            }
        )

        eval_dataloader = strategy.setup_dataloader(
            eval_dataset,
            batch_size=args.train_batch_size // strategy.world_size,
            pin_memory=True,
            shuffle=True,
            collate_fn=eval_dataset.collate_fn
        )
    else:
        eval_dataloader = None

    # gradient_checkpointing
    if args.gradient_checkpointing:
        reward_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # configure optimizer and scheduler
    max_steps = math.ceil(len(train_dataset) // args.train_batch_size * args.max_epochs)
    (
        reward_model, reward_model_optim, reward_model_scheduler
    ) = strategy.prepare_reward_model(reward_model, args=args, max_steps=max_steps)

    # load checkpoint
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(args.ckpt_path):
        _, states = strategy.load_ckpt(reward_model, args.ckpt_path)
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    # Make save directory
    os.makedirs(args.save_path, exist_ok=True)
    
    # dump args to save_path
    if (not torch.distributed.is_initialized()) or (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0):
        with open(os.path.join(args.save_path, "training_args.txt"), "w") as f:
            for key, value in vars(args).items():
                f.write(f"{key}: {value}\n")

    # configure Trainer
    trainer = SRMTrainerAL(
        model=reward_model,
        strategy=strategy,
        optim=reward_model_optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=reward_model_scheduler,
        tokenizer=tokenizer,
        max_epochs=args.max_epochs,
        loss=args.loss_type,
        margin=args.margin,
    )

    trainer.fit(args, consumed_samples, max_steps)

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
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_reward_model")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # Training
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--loss_type", type=str, default="hps", choices=["sigmoid", "logexp", "hps"], help="Loss type for SRM training")
    parser.add_argument("--margin", type=float, default=0.1, help="Margin for BT loss")
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for each prompt")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--micro_train_batch_size", type=int, default=4, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")

    # DeepSpeed
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=0)
    parser.add_argument("--target_modules", type=str, nargs="*", default=None)
    parser.add_argument("--lora_dropout", type=float, default=0)

    # Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--pooling_method", type=str, default="attn", choices=["attn", "last"], help="Pooling method for hidden states")
    parser.add_argument("--scale_for_train", action="store_true", default=False, help="Enable HPS Loss and learnable scale for train")
    parser.add_argument("--probing_layer", type=int, default=-1, help="Layer index of hidden states for reward model")
    parser.add_argument("--heads_types", type=str, nargs="*", default=["preference"], help="Types of heads for the reward model")

    # Custom dataset
    parser.add_argument("--train_data", type=str, default=None, help="HF dataset name or path for training")
    parser.add_argument("--eval_data", type=str, default=None, help="HF dataset name or path for evaluation")
    parser.add_argument("--task_instruction", type=str, default=None, help="Task instruction used as the reward model system prompt.")
    parser.add_argument("--input_template", type=str, default=None, help="Input template with a {} placeholder for prompt.")

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="lightrft_train")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")
    
    add_arguments(parser)

    args = parser.parse_args()

    if args.train_data:
        args.train_data = args.train_data.split(",")

    if args.eval_data:
        args.eval_data = args.eval_data.split(",")

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    train(args)
