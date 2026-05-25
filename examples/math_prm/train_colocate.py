"""
GRPO Training with Co-located Reward Models

This script implements Group Relative Policy Optimization (GRPO) training
with co-located reward models for reinforcement learning from human feedback (RLHF).

Key Features:
    - Supports both text-only and vision-language models
    - Multiple reward models (Value, Safety, Knowledge, Normal, General)
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
    python train_grpo_rm_colocate.py --pretrain <model_path> --reward_pretrain <rm_config> ...

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
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
)

from lightrft.utils import add_arguments, ensure_video_input_available
ensure_video_input_available()

from lightrft.datasets import PromptDatasetVL, SFTDatasetVL
from lightrft.utils import blending_datasets, get_tokenizer_processor_vl
from lightrft.models.actor_language import ActorLanguage
from lightrft.models.actor_vl import ActorVL

from lightrft.strategy import get_strategy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from math_prm_trainer import MathPRMSPMDPPOTrainerVL
from reward_models_utils import load_reward_models, reward_fn, RECIPE


def is_ursa_model(model_path: str) -> bool:
    """
    Check if the model is a URSA model by looking for URSA-specific config.

    URSA models have:
    - architectures: ["UrsaForConditionalGeneration"]
    - model_type: "ursa"
    - vision_config and aligner_config sections

    Args:
        model_path: Path to the model directory

    Returns:
        True if this is a URSA model, False otherwise
    """
    import os
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Check for UrsaForConditionalGeneration in architectures
                architectures = config.get("architectures", [])
                if "UrsaForConditionalGeneration" in architectures:
                    return True
                # Fallback: check model_type
                if config.get("model_type") == "ursa":
                    return True
        except:
            pass
    return False


def resolve_reference_shard_size(world_size: int, preferred_shard_size: int = 8) -> int:
    """
    Pick a reference-model FSDP shard size that preserves the original 8-way
    layout when possible, but still works for bounded small-world-size runs.
    """
    if world_size <= 0:
        return preferred_shard_size
    candidate = min(preferred_shard_size, world_size)
    while candidate > 1 and world_size % candidate != 0:
        candidate -= 1
    return candidate


def split_runtime_eval_dataset(prompts_data, args, strategy):
    """
    Build a deterministic held-out runtime eval split from prompt_data when no
    explicit eval dataset is provided.

    This follows the paper/plan intent of using a stable in-domain held-out set
    instead of relying on an optional dataset split name.
    """
    if args.eval_holdout_size <= 0 or args.max_eval_samples <= 0:
        return prompts_data, None

    total_samples = len(prompts_data)
    if total_samples <= 1:
        strategy.print("Warning: prompt_data is too small to carve out a held-out runtime eval split.")
        return prompts_data, None

    eval_size = min(args.eval_holdout_size, args.max_eval_samples, total_samples - 1)
    if eval_size <= 0:
        strategy.print("Warning: held-out runtime eval split resolved to zero samples; skipping eval split.")
        return prompts_data, None

    if not hasattr(prompts_data, "train_test_split"):
        strategy.print("Warning: prompt_data does not support train_test_split(); skipping held-out runtime eval.")
        return prompts_data, None

    split = prompts_data.train_test_split(test_size=eval_size, shuffle=True, seed=args.eval_holdout_seed)
    train_data = split["train"]
    eval_data = split["test"]
    strategy.print(
        "Prepared runtime eval holdout from prompt_data "
        f"(train={len(train_data)}, eval={len(eval_data)}, seed={args.eval_holdout_seed})."
    )
    return train_data, eval_data


def load_actor_tokenizer_processor(
    *,
    model_path: str,
    model,
    strategy,
    use_fast: bool,
):
    """
    Load the actor tokenizer/processor, using the explicit URSA processor path
    when the checkpoint is a URSA model.
    """
    if is_ursa_model(model_path):
        from ursa_model import UrsaProcessor

        processor = UrsaProcessor.from_pretrained(model_path)
        tokenizer = processor.tokenizer
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.pad_token_id
        strategy.print(
            f"Loaded URSA processor explicitly: tokenizer={type(tokenizer).__name__}, "
            f"processor={type(processor).__name__}"
        )
        return tokenizer, processor

    return get_tokenizer_processor_vl(
        model_path,
        model,
        "left",
        use_fast=use_fast,
    )


def build_actor_init_kwargs(
    args,
    *,
    ds_config,
    include_lora: bool,
    include_disable_logprobs_flashattn: bool,
):
    """
    Build Actor/UrsaActor initialization kwargs while keeping train/eval variants aligned.
    """
    kwargs = dict(
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        ds_config=ds_config,
        packing_samples=args.packing_samples,
        fused_linear_logprob=args.fused_linear_logprob,
    )
    if include_lora:
        kwargs.update(
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
        )
    if include_disable_logprobs_flashattn:
        kwargs["disable_logprobs_flashattn"] = args.disable_logprobs_flashattn
    return kwargs


def prepare_ursa_runtime_for_inference_engines(strategy=None):
    """
    Register the local URSA classes with HuggingFace auto classes so rollout
    engines that rely on ``AutoConfig`` can resolve ``model_type='ursa'``.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    pythonpath = os.environ.get("PYTHONPATH")
    pythonpath_parts = pythonpath.split(os.pathsep) if pythonpath else []
    if current_dir not in pythonpath_parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([current_dir, *pythonpath_parts]) if pythonpath_parts else current_dir

    from ursa_model import (
        UrsaConfig,
        UrsaForConditionalGeneration,
        UrsaForTokenClassification,
    )

    AutoConfig.register("ursa", UrsaConfig, exist_ok=True)
    AutoModelForVision2Seq.register(UrsaConfig, UrsaForConditionalGeneration, exist_ok=True)
    AutoModelForTokenClassification.register(UrsaConfig, UrsaForTokenClassification, exist_ok=True)

    if strategy is not None:
        strategy.print(
            "Registered URSA auto classes for inference engines "
            f"(sys.path/PYTHONPATH include {current_dir})"
        )


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
        - rm_use_engine: Generic flag retained for other reward types, but
          URSA math_prm/math_psgrpo PRM paths still load via HF directly
    """
    if args.hf_separate_rollout_actor and args.engine_type != "hf":
        raise ValueError("--hf_separate_rollout_actor requires --engine_type hf.")
    if args.hf_separate_rollout_actor and not args.fsdp:
        raise ValueError("--hf_separate_rollout_actor currently requires --fsdp.")

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

        # Check if this is a URSA model
        is_ursa = is_ursa_model(args.pretrain)

        # Select Actor class based on model type and text_only flag
        if is_ursa:
            strategy.print(f"Detected URSA model, using UrsaActor")
            from ursa_actor import UrsaActor
            Actor = UrsaActor
        elif args.text_only:
            Actor = ActorLanguage
        else:
            Actor = ActorVL

        # Initialize Actor (policy model)
        actor = Actor(
            args.pretrain,
            **build_actor_init_kwargs(
                args,
                ds_config=ds_train_cfg,
                include_lora=True,
                include_disable_logprobs_flashattn=True,
            ),
        )

        rollout_actor = None
        if args.hf_separate_rollout_actor:
            rollout_actor = Actor(
                args.pretrain,
                **build_actor_init_kwargs(
                    args,
                    ds_config=ds_eval_cfg,
                    include_lora=True,
                    include_disable_logprobs_flashattn=True,
                ),
            )

    if args.actor_init_on_gpu:
        actor = actor.to(torch.cuda.current_device())

    # pre-prepare is used for saving RAM memory when training 72B model
    if args.fsdp:
        setattr(actor, "is_actor", True)
        actor = strategy.prepare_model(actor, is_training=True)

    # Optionally freeze parameters (e.g., vision encoder).
    # Qwen2-VL etc. expose vision under "visual.*"; URSA uses "vision_model.*"
    # plus an "aligner.*" projector. Match all of them so --freeze_prefix
    # actually fires for the URSA stack.
    if args.freeze_prefix:
        freeze_prefix = ["visual", "vision_model", "aligner"]
        frozen_params_count = 0
        total_params_count = 0
        for name, param in actor.model.named_parameters():
            total_params_count += 1
            if any(name.startswith(prefix) for prefix in freeze_prefix):
                param.requires_grad = False
                frozen_params_count += 1
        strategy.print(f"Froze {frozen_params_count}/{total_params_count} parameters based on prefixes: {freeze_prefix}")

    if args.critic_pretrain:
        try:
            from lightrft.models import get_vlm_for_sequence_regression
        except ImportError as exc:
            raise ImportError(
                "critic_pretrain was provided, but get_vlm_for_sequence_regression "
                "is not available in this LightRFT checkout."
            ) from exc
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
        # Use the same Actor class (including URSA if detected)
        initial_model = Actor(
            args.pretrain,
            **build_actor_init_kwargs(
                args,
                ds_config=ds_eval_cfg,
                include_lora=False,
                include_disable_logprobs_flashattn=False,
            ),
        )
        if args.fsdp:
            reference_shard_size = resolve_reference_shard_size(
                world_size=strategy.world_size,
                preferred_shard_size=8,
            )
            strategy.print(
                "Preparing reference model with shard_size="
                f"{reference_shard_size} (world_size={strategy.world_size})"
            )
            initial_model = strategy.prepare_model(
                initial_model,
                is_training=False,
                shard_size=reference_shard_size,
            )
            strategy.offload_model(initial_model)

    if args.enable_ema:
        # Use the same Actor class (including URSA if detected)
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
    tokenizer, processor = load_actor_tokenizer_processor(
        model_path=args.pretrain,
        model=actor.model,
        strategy=strategy,
        use_fast=not strategy.args.disable_fast_tokenizer,
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

    heldout_eval_data = None
    if not args.eval_data and not args.eval_split:
        prompts_data, heldout_eval_data = split_runtime_eval_dataset(prompts_data, args, strategy)

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
                # Cap eval DataLoader batch_size by local_hf_generate_max_batch_size to
                # avoid the padding-leak bug. See heldout branch below for full rationale.
                eval_dp_batch_size = args.rollout_batch_size // strategy.world_size
                if args.engine_type == "hf":
                    mb_cap = int(getattr(args, "local_hf_generate_max_batch_size", 0) or 0)
                    if mb_cap > 0:
                        eval_dp_batch_size = min(eval_dp_batch_size, mb_cap)
                eval_dataloader = strategy.setup_dataloader(
                    eval_dataset,
                    eval_dp_batch_size,
                    False,
                    False,
                    collate_fn=eval_dataset.collate_fn,
                    drop_last=False,
                )
                strategy.print(
                    f"Evaluation dataset loaded: {len(eval_dataset)} samples "
                    f"(eval DataLoader batch_size={eval_dp_batch_size})"
                )
        else:
            strategy.print("Warning: eval_split specified but no data path available for evaluation.")
    elif heldout_eval_data is not None:
        eval_data = heldout_eval_data.select(range(min(args.max_eval_samples, len(heldout_eval_data))))
        eval_dataset = PromptDatasetVL(
            eval_data, tokenizer, processor, args.prompt_max_len, strategy, input_template=args.input_template
        )
        # Match DataLoader batch_size to local_hf_generate_max_batch_size for engine_type=hf.
        # fast_exp_maker.process_multimodal_batch calls processor(padding=True) on the full
        # DataLoader batch, then strategy_base chunks the already-padded tensor into
        # micro-batches of local_hf_generate_max_batch_size. Without this alignment, each
        # micro-batch keeps the max-of-DL-batch padded length (e.g. 16-wide pad in a 4-wide
        # chunk), and the extra left-pad tokens — even with attention_mask masking — degrade
        # URSA's greedy decode by ~8pp via RoPE / vision-path interaction. Setting DL-batch
        # = micro-batch eliminates the leak so eval matches `tmp/ckpt_eval_aligned.py --bs N`
        # exactly. See PR53 issuecomment-... for the 11.9pp breakdown.
        eval_dp_batch_size = args.rollout_batch_size // strategy.world_size
        if args.engine_type == "hf":
            mb_cap = int(getattr(args, "local_hf_generate_max_batch_size", 0) or 0)
            if mb_cap > 0:
                eval_dp_batch_size = min(eval_dp_batch_size, mb_cap)
        eval_dataloader = strategy.setup_dataloader(
            eval_dataset,
            eval_dp_batch_size,
            False,
            False,
            collate_fn=eval_dataset.collate_fn,
            drop_last=False,
        )
        strategy.print(
            f"Held-out runtime evaluation dataset loaded: {len(eval_dataset)} samples "
            f"(eval DataLoader batch_size={eval_dp_batch_size}, aligned with "
            f"local_hf_generate_max_batch_size={getattr(args, 'local_hf_generate_max_batch_size', 'n/a')})"
        )

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

    if rollout_actor is not None:
        keep_rollout_on_gpu = bool(getattr(strategy.config, "hf_separate_rollout_keep_on_gpu", False))
        rollout_actor = strategy.prepare_model(
            rollout_actor,
            is_training=False,
            shard_size=-1,
            reshard_after_forward=False,
        )
        rollout_actor.gradient_checkpointing_disable()
        rollout_actor.eval()
        if not keep_rollout_on_gpu:
            strategy.offload_model(rollout_actor)
        residency_note = "kept on GPU" if keep_rollout_on_gpu else "offloaded to CPU"
        strategy.print(
            "Prepared separate local HF rollout actor with FSDP full-shard, gc disabled, "
            f"reshard_after_forward disabled, and {residency_note}."
        )

        # rollout_eos_patch is OPT-IN as of the eval-fix PR. With it OFF (default),
        # rollout/eval generation falls back to HF default stopping (EosTokenCriteria +
        # MaxLengthCriteria) which is token-equivalent to a bare model.generate call.
        # See PR #53 issuecomment-4394197141 for why the patch was harmful by default.
        if getattr(args, "enable_rollout_eos_patch", False):
            from rollout_eos_patch import install_math_prm_rollout_eos_patch
            install_math_prm_rollout_eos_patch(rollout_actor, tokenizer, tokenizer.eos_token_id)
            strategy.print(
                "Installed math_prm rollout EOS patch on rollout_actor.model.generate "
                "(legacy --enable_rollout_eos_patch flag set; this BIASES rollout reward "
                "and eval outcome — only enable to reproduce historical broken behavior)."
            )
        else:
            strategy.print(
                "rollout_eos_patch NOT installed (default). Generation uses HF default "
                "stopping criteria (EosTokenCriteria + MaxLengthCriteria), token-equivalent "
                "to bare model.generate. Use --enable_rollout_eos_patch to restore legacy."
            )

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

    if is_ursa:
        prepare_ursa_runtime_for_inference_engines(strategy)

    strategy.report_memory("before setup_inference_engine")
    strategy.setup_inference_engine(
        args,
        engine_type=args.engine_type,
        actor=actor,
        rollout_actor=rollout_actor,
        tokenizer=tokenizer,
        processor=processor,
    )
    strategy.report_memory("after setup_inference_engine")

    # configure Trainer
    trainer = MathPRMSPMDPPOTrainerVL(
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
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
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

    # ---- Optional initial evaluate-only at step 0 (no PPO update) ----
    # Useful for diagnosing model state at step 0 vs step 1; e.g. to attribute the
    # outcome gap between standalone bs=1 eval and the 8-rank FSDP bs=4 eval pipeline.
    # Triggered by --initial_eval (default False, no-op).
    if getattr(args, "initial_eval", False) and eval_dataloader is not None:
        strategy.print(f"\n{'=' * 60}\n[initial_eval] Running evaluate at step 0 (NO PPO update)\n{'=' * 60}")
        trainer.eval_dataloader = eval_dataloader  # ensure trainer has handle
        raw = trainer.evaluate(eval_dataloader, global_step=0)
        if strategy.is_rank_0() and raw:
            strategy.print(f"[initial_eval] step 0 outcome: {raw}")
        if getattr(args, "initial_eval_only", False):
            strategy.print("[initial_eval] --initial_eval_only set, exiting before training.")
            return

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

    parser.add_argument("--engine_type", type=str, default="hf", help="Choose inference engine type: vllm, sglang, hf")
    parser.add_argument("--text_only", action="store_true", default=False)

    # Checkpoint
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--save_trajectories", action="store_true", default=False, help="Save experience trajectories to JSON for debugging")
    parser.add_argument(
        "--trajectory_analysis",
        action="store_true",
        default=False,
        help="Enable extra trajectory analysis metrics when saving trajectories",
    )
    parser.add_argument("--num_trajectories_to_save", type=int, default=10, help="Number of trajectories to save per checkpoint")
    parser.add_argument("--print_replay_buffer_stats", action="store_true", default=False, help="Print detailed replay buffer statistics during training")
    parser.add_argument("--enable_profile", action="store_true", default=False, help="Enable persistent step profiling with local files and W&B metrics")
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
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--rollout_batch_size", type=int, default=128)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for each prompt")
    parser.add_argument("--generate_max_len", type=int, default=3072, help="Max tokens to generate in PPO")
    parser.add_argument("--max_len", type=int, default=None, help="deprecated max_len")
    parser.add_argument("--max_samples", type=int, default=15360)
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
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--freeze_prefix", action="store_true", default=False, help="Freeze the prefix part (e.g. vision encoder) of the actor model")
    parser.add_argument("--freezing_actor_steps", type=int, default=-1, help="Used for critic initialization")
    parser.add_argument(
        "--n_samples_per_prompt", type=int, default=8, help="number of responses for each prompt in generation"
    )
    parser.add_argument("--save_value_network", action="store_true", default=False, help="Save critic model")
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--init_kl_coef", type=float, default=0.001, help="KL penalty in PPO")
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

    parser.add_argument(
        "--per_step_reward_mode",
        type=str,
        choices=["raw", "group_norm"],
        default="group_norm",
        help=(
            "How to integrate per-step PRM rewards (Math-Shepherd-style "
            "per-token reward path, distinct from the strict paper Eq.9 path "
            "selected via --advantage_estimator ursa_variant2). "
            "'group_norm' (default): for each step k, subtract group mean and "
            "divide by group std across the K trajectories in the same prompt "
            "group BEFORE scattering to step-boundary tokens. Produces "
            "zero-mean signed advantages (GRPO baseline convention). "
            "'raw': scatter raw sigmoid step_score directly. WARNING — raw "
            "is unsafe: sigmoid scores are always positive, so every "
            "post-cumsum token advantage is non-negative and PG pushes "
            "every probability up. Kept only for paper Figure ablation. "
            "Only active when label is 'math_per_step_prm' AND "
            "--advantage_estimator is the cumsum path (group_norm/grpo). "
            "For the strict paper Eq.9 path use --advantage_estimator "
            "ursa_variant2 (handles its own group normalization)."
        ),
    )

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
    parser.add_argument("--eval_split", type=str, default="", help="Evaluation data split (default: disabled)")
    parser.add_argument("--max_eval_samples", type=int, default=500, help="Maximum number of samples to evaluate (default: 500)")
    parser.add_argument(
        "--eval_holdout_size",
        type=int,
        default=500,
        help="Deterministic held-out eval subset size sampled from prompt_data when eval_data is unset (default: 500)",
    )
    parser.add_argument(
        "--eval_holdout_seed",
        type=int,
        default=42,
        help="Seed for deterministic held-out runtime eval split (default: 42)",
    )
    parser.add_argument(
        "--eval_n_samples_per_prompt",
        type=int,
        default=1,
        help="Number of eval generations per prompt (default: 1)",
    )
    parser.add_argument(
        "--eval_do_sample",
        action="store_true",
        default=False,
        help="Use sampling during runtime eval instead of greedy decoding",
    )
    parser.add_argument(
        "--eval_generate_max_len",
        type=int,
        default=None,
        help="Maximum generation length for runtime eval (default: use generate_max_len)",
    )
    parser.add_argument("--eval_temperature", type=float, default=0.0, help="Eval temperature (default: 0.0)")
    parser.add_argument("--eval_top_p", type=float, default=1.0, help="Eval top-p (default: 1.0)")
    parser.add_argument("--eval_top_k", type=int, default=-1, help="Eval top-k (default: -1)")
    parser.add_argument(
        "--eval_repetition_penalty",
        type=float,
        default=1.0,
        help="Eval repetition penalty (default: 1.0)",
    )
    parser.add_argument(
        "--eval_no_repeat_ngram_size",
        type=int,
        default=0,
        help="Eval no-repeat-ngram size (default: 0)",
    )

    parser.add_argument("--pretrain_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--pretrain_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--pretrain_split", type=str, default="train")
    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--images_key", type=str, default="images", help="JSON dataset key for images")
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

    # initial-eval (eval at step 0, before any PPO update)
    parser.add_argument(
        "--initial_eval", action="store_true", default=False,
        help="Run evaluate(global_step=0) before fit(). Useful for measuring base "
             "model outcome under the actual training eval pipeline (8-rank FSDP + "
             "bs=4 etc.) without any PPO drift.",
    )
    parser.add_argument(
        "--initial_eval_only", action="store_true", default=False,
        help="With --initial_eval, exit immediately after initial eval (skip training).",
    )

    # math_prm rollout EOS patch
    parser.add_argument(
        "--enable_rollout_eos_patch", action="store_true", default=False,
        help=(
            "Install StructuredAnswerStoppingCriteria on rollout_actor.model.generate "
            "(legacy behavior). DEFAULT OFF. The patch makes generation stop right after "
            "'†Answer:' marker, but historical experiments (PR #53 issuecomment-4394197141) "
            "showed it (a) lowers eval outcome by ~9.8pp due to truncated tokens, and "
            "(b) biases rollout reward signal towards short responses (Goodhart's law) "
            "causing length collapse during RL. With patch off, generation falls back to "
            "HF default stopping (EosTokenCriteria + MaxLengthCriteria), which is what we "
            "want for both rollout reward fidelity and eval accuracy alignment."
        ),
    )

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
