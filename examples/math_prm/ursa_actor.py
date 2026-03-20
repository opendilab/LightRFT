"""
URSA-8B Actor Model Loader

This module provides a custom actor loader for URSA-8B models, which use
UrsaForConditionalGeneration instead of the standard AutoModelForVision2Seq.

URSA-8B architecture:
- Hybrid vision tower: SAM-B (1024x1024) + SigLIP-L (384x384)
- MLP projector: Maps vision features to LLM embedding space
- Language model: Qwen2.5-Math-Instruct (8B params)
"""

import sys
import os
import torch
import torch.nn as nn
from typing import Optional
from transformers.integrations.deepspeed import HfDeepSpeedConfig

# Add current directory to path for ursa_model imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from ursa_model import UrsaForConditionalGeneration
from lightrft.models.actor_vl import ActorVL
from lightrft.models.utils import apply_lora_configuration


class UrsaActor(ActorVL):
    """
    Actor wrapper for URSA-8B models.

    This class extends ActorVL to support loading URSA-8B models using
    UrsaForConditionalGeneration instead of AutoModelForVision2Seq.

    Usage:
        actor = UrsaActor(
            pretrain_or_model="/path/to/URSA-8B",
            use_flash_attention_2=True,
            bf16=True,
            lora_rank=0,
        )
    """

    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        packing_samples=False,
        high_entropy_token_ratio=0.0,
        **kwargs,
    ) -> None:
        """
        Initialize URSA-8B actor model.

        Args:
            pretrain_or_model: Path to URSA-8B checkpoint or model instance
            use_flash_attention_2: Enable Flash Attention 2.0
            bf16: Use bfloat16 precision
            lora_rank: LoRA rank (0 disables LoRA)
            lora_alpha: LoRA alpha scaling parameter
            lora_dropout: LoRA dropout rate
            target_modules: Target modules for LoRA (auto-detected if None)
            ds_config: DeepSpeed configuration
            device_map: Device mapping for model placement
            packing_samples: Enable sample packing
            high_entropy_token_ratio: High entropy token filtering ratio
        """
        # Initialize parent class without calling its __init__
        # We'll handle model loading ourselves
        nn.Module.__init__(self)
        self.high_entropy_token_ratio = high_entropy_token_ratio

        if isinstance(pretrain_or_model, str):
            self.pretrain_or_model = pretrain_or_model
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

            # DeepSpeed ZeRO-3 integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None  # noqa: F841

            # Prepare loading kwargs
            from_pretrained_kwargs = {
                "trust_remote_code": True,
                "attn_implementation": attn_implementation,
                "torch_dtype": torch.bfloat16 if bf16 else "auto",
            }

            # Check if we're in meta device context (FSDP)
            try:
                test_tensor = torch.empty(1)
                is_meta_context = test_tensor.is_meta
            except:  # noqa
                is_meta_context = False

            if not is_meta_context and device_map is not None:
                from_pretrained_kwargs["device_map"] = device_map

            print(f"[UrsaActor] Loading URSA-8B model from {pretrain_or_model}")

            # Load URSA model using UrsaForConditionalGeneration
            self.model = UrsaForConditionalGeneration.from_pretrained(
                pretrain_or_model,
                **from_pretrained_kwargs
            )

            print(f"[UrsaActor] Successfully loaded URSA-8B model")

            # Apply LoRA if requested
            if lora_rank > 0:
                print(f"[UrsaActor] Applying LoRA with rank={lora_rank}, alpha={lora_alpha}")
                self.model = apply_lora_configuration(
                    model=self.model,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules,
                    freeze_vision_tower=True,
                )

            # Disable cache for training
            self.model.config.use_cache = False

            # Enable sample packing if requested
            self.packing_samples = packing_samples
        else:
            # Model instance provided directly
            self.model = pretrain_or_model
            self.pretrain_or_model = "ursa"

        print(f"[UrsaActor] Model type: {self.pretrain_or_model}")


def create_ursa_actor(args, ds_config=None):
    """
    Factory function to create URSA-8B actor from training args.

    Args:
        args: Training arguments (argparse.Namespace)
        ds_config: DeepSpeed configuration dict

    Returns:
        UrsaActor instance
    """
    return UrsaActor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=getattr(args, 'load_in_4bit', False),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=getattr(args, 'target_modules', None),
        lora_dropout=args.lora_dropout,
        ds_config=ds_config,
        packing_samples=args.packing_samples,
        disable_logprobs_flashattn=getattr(args, 'disable_logprobs_flashattn', False),
        fused_linear_logprob=getattr(args, 'fused_linear_logprob', False),
    )
