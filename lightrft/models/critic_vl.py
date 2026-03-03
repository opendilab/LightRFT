"""
Vision-Language Critic Model Module for Reinforcement Learning.

This module provides the CriticVL class, which implements a critic model specifically designed
for vision-language tasks in reinforcement learning scenarios. The critic is responsible for
estimating state values based on visual inputs (images and videos) and textual prompts.

The module supports various optimization techniques including:
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- Flash Attention 2.0 for improved performance
- DeepSpeed integration for distributed training
- Sample packing for efficient batch processing

Key Features:
- Multi-modal input processing (text + vision)
- Flexible model loading from pretrained checkpoints
- Support for various vision-language model architectures
- Value head for state value estimation
- Gradient checkpointing for memory optimization
"""

from typing import Optional, Tuple, Union, List

import deepspeed
import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from .utils import reset_position_ids, apply_lora_configuration
from .actor_modality import ActorModality
from lightrft.utils.logging_utils import init_logger

logger = init_logger(__name__)


class CriticVL(nn.Module):
    """
    Vision-Language Critic model for reinforcement learning applications.

    This class implements a critic model for PPO/GRPO training that estimates state values
    for advantage calculation. It extends a base vision-language model with a linear value
    head that outputs scalar value estimates for each token position.

    The critic model can be initialized from a pretrained model path and supports various
    optimization techniques including LoRA adaptation and distributed training.

    :param pretrain_or_model: Either a string path to a pretrained model or a model instance
    :type pretrain_or_model: Union[str, nn.Module]
    :param use_flash_attention_2: Whether to utilize Flash Attention 2.0 for improved performance
    :type use_flash_attention_2: bool
    :param bf16: Enable bfloat16 precision for model computations
    :type bf16: bool
    :param lora_rank: Rank for LoRA adaptation (0 disables LoRA)
    :type lora_rank: int
    :param lora_alpha: Alpha parameter for LoRA scaling
    :type lora_alpha: int
    :param lora_dropout: Dropout rate for LoRA layers
    :type lora_dropout: float
    :param target_modules: List of target modules for applying LoRA (auto-detected if None)
    :type target_modules: Optional[list]
    :param normalize_reward: Whether to normalize value estimates
    :type normalize_reward: bool
    :param ds_config: Configuration for DeepSpeed distributed training
    :type ds_config: Optional[dict]
    :param init_value_head: Whether to initialize the value head weights
    :type init_value_head: bool
    :param value_head_prefix: Prefix for the value head attribute name
    :type value_head_prefix: str
    :param device_map: Device mapping for loading the model onto specific devices
    :type device_map: Optional[dict]
    :param packing_samples: Whether to pack samples during training for efficiency
    :type packing_samples: bool

    Example::

        # Initialize with a pretrained model path
        critic = CriticVL(
            pretrain_or_model="Qwen/Qwen2.5-VL-7B-Instruct",
            use_flash_attention_2=True,
            lora_rank=16,
            lora_alpha=32,
            normalize_reward=False
        )

        # Compute value estimates
        values = critic(
            input_ids=input_tensor,
            num_actions=10,
            attention_mask=attention_mask,
            pixel_values=image_tensor,
            image_grid_thw=grid_tensor
        )
    """
    # Model modality declaration - defines what types of inputs this model accepts
    modality = ActorModality.VISION_LANGUAGE

    def __init__(
        self,
        pretrain_or_model: Union[str, nn.Module],
        use_flash_attention_2: bool = False,
        bf16: bool = True,
        lora_rank: int = 0,
        lora_alpha: int = 16,
        lora_dropout: float = 0,
        target_modules: Optional[List[str]] = None,
        normalize_reward: bool = False,
        ds_config: Optional[dict] = None,
        init_value_head: bool = False,
        value_head_prefix: str = "score",
        device_map: Optional[dict] = None,
        packing_samples: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.value_head_prefix = value_head_prefix
        self.packing_samples = packing_samples
        self.normalize_reward = normalize_reward

        if isinstance(pretrain_or_model, str):
            self.pretrain_or_model = pretrain_or_model
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            # Load the model
            self.model = AutoModelForVision2Seq.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
                **kwargs,
            )

            # LoRA
            if lora_rank > 0:
                self.model = apply_lora_configuration(
                    model=self.model,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules,
                    freeze_vision_tower=True,
                )

            # Get hidden size for value head
            hidden_size = self.model.config.hidden_size

            # Add value head
            setattr(self, value_head_prefix, nn.Linear(hidden_size, 1, bias=False))

            # mean std for value normalization
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                logger.info("[MoE] set output_router_logits as True")
                self.model.config.output_router_logits = True

            # https://github.com/huggingface/transformers/issues/26877
            self.model.config.use_cache = False

            # Initialize value head if requested
            if init_value_head:
                value_head = getattr(self, self.value_head_prefix)
                if dschf is not None:
                    logger.info("initialize value_head for ZeRO-3 critic model training.")
                    with deepspeed.zero.GatheredParameters([value_head.weight], modifier_rank=0):
                        if torch.distributed.get_rank() == 0:
                            value_head.weight.data.normal_(mean=0.0, std=1 / (hidden_size + 1))
                else:
                    value_head.weight.data.normal_(mean=0.0, std=1 / (hidden_size + 1))

        else:
            # Use existing model instance
            self.model = pretrain_or_model
            self.pretrain_or_model = pretrain_or_model.config.model_type

        print("pretrain_or_model: ", self.pretrain_or_model)

        # Enable gradient checkpointing if supported
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    def forward(
        self,
        input_ids: torch.LongTensor,
        num_actions: Optional[Union[int, List[int]]],
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        return_output: bool = False,
        packed_seq_lens: Optional[List[int]] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Forward pass to compute value estimates for action sequences.

        This method computes state-value estimates for tokens in the action sequence,
        which are used to calculate advantages in PPO/GRPO training. The values are
        computed from the last hidden states of the base VLM, excluding the final token.

        :param input_ids: Input token IDs including both prompt and generated actions.
        :type input_ids: torch.LongTensor
        :param num_actions: Number of action tokens to compute values for.
        :type num_actions: Optional[Union[int, List[int]]]
        :param attention_mask: Attention mask for input sequences.
        :type attention_mask: Optional[torch.Tensor]
        :param pixel_values: Pixel values for images in the input.
        :type pixel_values: Optional[torch.Tensor]
        :param image_grid_thw: Image grid metadata (time, height, width).
        :type image_grid_thw: Optional[torch.Tensor]
        :param pixel_values_videos: Pixel values for videos in the input.
        :type pixel_values_videos: Optional[torch.Tensor]
        :param video_grid_thw: Video grid metadata (time, height, width).
        :type video_grid_thw: Optional[torch.Tensor]
        :param return_output: If True, return tuple of (values, model_outputs).
        :type return_output: bool
        :param packed_seq_lens: Lengths of packed sequences.
        :type packed_seq_lens: Optional[List[int]]

        :return: Value estimates for action tokens.
        :rtype: Union[torch.Tensor, Tuple[torch.Tensor, dict]]
        """
        if not self.packing_samples:
            # https://github.com/OpenRLHF/OpenRLHF/issues/217
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        else:
            # convert attention_mask to position_ids
            position_ids = reset_position_ids(attention_mask)
            # explicitly ignore attention_mask for packing_samples
            attention_mask = None

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            output_hidden_states=True
        )
        last_hidden_states = outputs["hidden_states"][-1]
        values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)[:, :-1]

        # normalize values if enabled
        if self.normalize_reward:
            values = (values - self.mean) / self.std

        if num_actions is None:
            assert return_output
            return outputs

        if not self.packing_samples:
            action_values = values[:, -num_actions:]
        else:
            assert isinstance(num_actions, list) and len(num_actions) == len(packed_seq_lens)
            action_values = []
            offset = 0
            for num_action, seq_len in zip(num_actions, packed_seq_lens):
                start, end = max(0, offset + seq_len - num_action - 1), offset + seq_len - 1
                action_values.append(values[:, start:end])
                offset += seq_len
            action_values = torch.cat(action_values, dim=1)

        if return_output:
            return (action_values, outputs)
        else:
            return action_values

    def gradient_checkpointing_enable(self, **kwargs):
        """Enable gradient checkpointing for memory optimization."""
        self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        """Print the number of trainable parameters in the model."""
        if hasattr(self.model, "print_trainable_parameters"):
            self.model.print_trainable_parameters()
