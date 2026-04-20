import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.multiprocessing

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightrft.datasets import PromptDatasetVL
from lightrft.models.actor_language import ActorLanguage
from lightrft.strategy import get_strategy
from lightrft.trainer.spmd_ppo_trainer import SPMDPPOTrainerVL
from lightrft.utils import blending_datasets, get_tokenizer_processor_vl

from reward_models_utils import RECIPE, load_reward_models, reward_fn

torch.multiprocessing.set_sharing_strategy("file_system")


SYSTEM_PROMPT = (
    "You are a careful arithmetic assistant. "
    "Solve the expression and respond briefly. "
    "Return the final result in the format \\boxed{answer}. "
    "Do not add unnecessary explanation."
)

FIXED_ARGS = {
    "adam_offload": True,
    "advantage_estimator": "group_norm",
    "apply_chat_template": True,
    "aux_loss_coef": 0.0,
    "bf16": True,
    "enable_engine_sleep": True,
    "eval_steps": -1,
    "flash_attn": True,
    "fsdp": True,
    "fsdp_cpu_offload": False,
    "fused_linear_logprob": False,
    "gradient_checkpointing": True,
    "kl_estimator": "k3",
    "l2": 1e-2,
    "lr_warmup_ratio": 0.03,
    "max_ckpt_mem": int(1e8),
    "max_ckpt_num": 1,
    "max_epochs": 1,
    "packing_samples": False,
    "reward_running_norm": False,
    "save_steps": -1,
    "system_prompt": SYSTEM_PROMPT,
    "text_only": True,
    "use_cpg_loss": False,
    "use_kl_loss": True,
    "wandb_group": None,
}

MODEL_KWARGS = {
    "actor_init_on_gpu": False,
    "disable_logprobs_flashattn": False,
    "high_entropy_token_ratio": 0.0,
    "initial_model_shard_size": None,
    "load_in_4bit": False,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "lora_rank": 0,
    "meta_init": False,
    "packing_samples": False,
    "target_modules": "all-linear",
}

TRAINER_KWARGS = {
    "disable_ds_ckpt": False,
    "eps_clip": 0.2,
    "gamma": 1.0,
    "gradient_checkpointing_use_reentrant": False,
    "kl_target": None,
    "loss_agg_mode": "seq-mean-token-mean",
    "max_len": None,
    "max_norm": 1.0,
    "print_replay_buffer_stats": False,
    "ptx_coef": 0.0,
    "save_hf_ckpt": False,
    "temperature": 1.0,
    "top_p": 1.0,
    "value_clip": 0.2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal LightRFT RL entry for the tiny_python_expr example.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pretrain", type=str, required=True)
    parser.add_argument("--prompt_data", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)

    parser.add_argument("--engine_type", type=str, choices=["sglang", "vllm"], default="sglang")
    parser.add_argument("--engine_tp_size", type=int, default=1)
    parser.add_argument("--engine_mem_util", type=float, default=0.55)

    parser.add_argument("--micro_train_batch_size", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=16)
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--n_samples_per_prompt", type=int, default=4)
    parser.add_argument("--prompt_max_len", type=int, default=256)
    parser.add_argument("--generate_max_len", type=int, default=128)
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--init_kl_coef", type=float, default=0.001)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=1)

    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default="")
    parser.add_argument("--wandb_project", type=str, default="tiny-python-expr")
    parser.add_argument("--wandb_run_name", type=str, default="tiny-python-expr")
    return parser.parse_args()


def build_runtime_args() -> argparse.Namespace:
    args = parse_args()
    for key, value in FIXED_ARGS.items():
        setattr(args, key, value)

    args.use_tensorboard = None

    if args.advantage_estimator == "group_norm" and args.n_samples_per_prompt <= 1:
        raise ValueError("group_norm requires n_samples_per_prompt > 1")

    return args


def build_actor(strategy, args: argparse.Namespace):
    ds_train_cfg = strategy.get_ds_train_config(is_actor=True) if not args.fsdp else None

    with strategy.init_model_context(meta_init=MODEL_KWARGS["meta_init"]):
        actor = ActorLanguage(
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=MODEL_KWARGS["load_in_4bit"],
            lora_rank=MODEL_KWARGS["lora_rank"],
            lora_alpha=MODEL_KWARGS["lora_alpha"],
            target_modules=MODEL_KWARGS["target_modules"],
            lora_dropout=MODEL_KWARGS["lora_dropout"],
            ds_config=ds_train_cfg,
            packing_samples=MODEL_KWARGS["packing_samples"],
            disable_logprobs_flashattn=MODEL_KWARGS["disable_logprobs_flashattn"],
            fused_linear_logprob=args.fused_linear_logprob,
            high_entropy_token_ratio=MODEL_KWARGS["high_entropy_token_ratio"],
        )

    if MODEL_KWARGS["actor_init_on_gpu"]:
        actor = actor.to(torch.cuda.current_device())

    if args.fsdp:
        setattr(actor, "is_actor", True)
        actor = strategy.prepare_model(actor, is_training=True)

    return actor


def build_initial_model(strategy, args: argparse.Namespace):
    if args.init_kl_coef == 0:
        return None

    ds_eval_cfg = strategy.get_ds_eval_config(offload=False) if not args.fsdp else None
    initial_model = ActorLanguage(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=MODEL_KWARGS["load_in_4bit"],
        ds_config=ds_eval_cfg,
        packing_samples=MODEL_KWARGS["packing_samples"],
        fused_linear_logprob=args.fused_linear_logprob,
    )

    if args.fsdp:
        shard_size = MODEL_KWARGS["initial_model_shard_size"] or strategy.world_size
        initial_model = strategy.prepare_model(initial_model, is_training=False, shard_size=shard_size)
        strategy.offload_model(initial_model)

    return initial_model


def build_prompt_loader(strategy, tokenizer, processor, args: argparse.Namespace):
    prompts_data = blending_datasets(
        args.prompt_data,
        "1.0",
        strategy,
        args.seed,
        return_eval=False,
        train_split="train",
    )
    prompts_dataset = PromptDatasetVL(
        prompts_data,
        tokenizer,
        processor,
        args.prompt_max_len,
        strategy,
    )
    return prompts_dataset, strategy.setup_dataloader(
        prompts_dataset,
        args.rollout_batch_size // strategy.world_size,
        True,
        True,
        collate_fn=prompts_dataset.collate_fn,
    )


def train(args: argparse.Namespace) -> None:
    strategy = get_strategy(args)
    actor = build_actor(strategy, args)
    reward_models, reward_tokenizers, label_map = load_reward_models("{}", strategy, use_engine=False)
    initial_model = build_initial_model(strategy, args)

    tokenizer, processor = get_tokenizer_processor_vl(
        args.pretrain,
        actor.model,
        "left",
        use_fast=True,
    )
    prompts_dataset, prompts_dataloader = build_prompt_loader(strategy, tokenizer, processor, args)

    num_update_steps_per_episode = max(
        1,
        len(prompts_dataset) * args.n_samples_per_prompt // args.train_batch_size,
    )
    max_steps = max(1, math.ceil(args.num_episodes * num_update_steps_per_episode))

    if args.gradient_checkpointing:
        actor.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": TRAINER_KWARGS["gradient_checkpointing_use_reentrant"]
            }
        )

    (
        (actor, actor_optim, actor_scheduler),
        (_, _, _),
        reward_models,
        initial_model,
    ) = strategy.prepare_models_and_optimizers(actor, None, reward_models, initial_model, args, max_steps)

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.ckpt_path, exist_ok=True)
    strategy.setup_inference_engine(args, engine_type=args.engine_type, actor=actor)

    trainer = SPMDPPOTrainerVL(
        strategy,
        actor,
        None,
        reward_models,
        initial_model,
        None,
        actor_optim,
        None,
        actor_scheduler,
        None,
        max_epochs=args.max_epochs,
        micro_train_batch_size=args.micro_train_batch_size,
        micro_rollout_batch_size=args.micro_rollout_batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        tokenizer=tokenizer,
        processor=processor,
        prompt_max_len=args.prompt_max_len,
        value_clip=TRAINER_KWARGS["value_clip"],
        eps_clip=TRAINER_KWARGS["eps_clip"],
        loss_agg_mode=TRAINER_KWARGS["loss_agg_mode"],
        init_kl_coef=args.init_kl_coef,
        kl_target=TRAINER_KWARGS["kl_target"],
        ptx_coef=TRAINER_KWARGS["ptx_coef"],
        max_norm=TRAINER_KWARGS["max_norm"],
        do_sample=True,
        max_new_tokens=args.generate_max_len,
        max_length=TRAINER_KWARGS["max_len"],
        temperature=TRAINER_KWARGS["temperature"],
        top_p=TRAINER_KWARGS["top_p"],
        gamma=TRAINER_KWARGS["gamma"],
        first_token_temperature=10.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        reward_fn=reward_fn,
        reward_fn_label_map=label_map,
        reward_recipe=RECIPE,
        reward_tokenizers=reward_tokenizers,
        save_hf_ckpt=TRAINER_KWARGS["save_hf_ckpt"],
        disable_ds_ckpt=TRAINER_KWARGS["disable_ds_ckpt"],
        packing_samples=MODEL_KWARGS["packing_samples"],
        print_replay_buffer_stats=TRAINER_KWARGS["print_replay_buffer_stats"],
    )

    trainer.fit(
        args,
        prompts_dataloader=prompts_dataloader,
        pretrain_dataloader=None,
        eval_dataloader=None,
        consumed_samples=0,
        num_update_steps_per_episodes=num_update_steps_per_episode,
    )

    if strategy.is_rank_0():
        marker_path = os.path.join(args.save_path, "training_complete.txt")
        with open(marker_path, "w", encoding="utf-8") as fout:
            fout.write("tiny_python_expr training completed successfully.\n")
            fout.write(f"pretrain={args.pretrain}\n")
            fout.write(f"prompt_data={args.prompt_data}\n")
            fout.write(f"num_episodes={args.num_episodes}\n")
            fout.write(f"n_samples_per_prompt={args.n_samples_per_prompt}\n")
            fout.write(f"train_batch_size={args.train_batch_size}\n")
            fout.write(f"rollout_batch_size={args.rollout_batch_size}\n")
            fout.write(f"actor_learning_rate={args.actor_learning_rate}\n")
        strategy.print(f"Saved lightweight completion marker to {marker_path}")

def main() -> None:
    args = build_runtime_args()
    train(args)


if __name__ == "__main__":
    main()
