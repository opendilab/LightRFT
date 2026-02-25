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
from transformers import AutoConfig, AutoModelForVision2Seq
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer

from .utils import reset_position_ids
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
    optimization techniques including LoRA adaptation, quantization, and distributed training.

    :param pretrain_or_model: Either a string path to a pretrained model or a model instance
    :type pretrain_or_model: Union[str, nn.Module]
    :param use_flash_attention_2: Whether to utilize Flash Attention 2.0 for improved performance
    :type use_flash_attention_2: bool
    :param bf16: Enable bfloat16 precision for model computations
    :type bf16: bool
    :param load_in_4bit: Load the model in 4-bit precision for memory efficiency
    :type load_in_4bit: bool
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
        load_in_4bit: bool = False,
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
            # Load model from pretrained path
            config = AutoConfig.from_pretrained(pretrain_or_model, trust_remote_code=True)
            config.normalize_reward = normalize_reward
            config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

            # Prioritize using the value_head_prefix in the model configuration
            self.value_head_prefix = getattr(config, "value_head_prefix", value_head_prefix)
            logger.info(f"set value_head_prefix to `{self.value_head_prefix}`")

            # Get base model class
            base_class = AutoModelForVision2Seq._model_mapping[type(config)]

            # Create dynamic CriticModel class
            CriticModel = self._create_critic_model_class(base_class, self.value_head_prefix, packing_samples)

            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            # Handle 4-bit quantization
            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                from transformers import BitsAndBytesConfig
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            # Load the model
            self.model = CriticModel.from_pretrained(
                pretrain_or_model,
                config=config,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                quantization_config=nf4_config,
                device_map=device_map,
                **kwargs,
            )

            # Apply LoRA if specified
            if lora_rank > 0:
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if self.value_head_prefix in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                logger.info("[MoE] set output_router_logits as True")
                self.model.config.output_router_logits = True

            # https://github.com/huggingface/transformers/issues/26877
            self.model.config.use_cache = False

            # Initialize value head if requested
            if init_value_head:
                value_head = getattr(self.model, self.value_head_prefix)
                if dschf is not None:
                    logger.info("initialize value_head for ZeRO-3 critic model training.")
                    with deepspeed.zero.GatheredParameters([value_head.weight], modifier_rank=0):
                        if torch.distributed.get_rank() == 0:
                            value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
                else:
                    value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))

        else:
            # Use existing model instance
            self.model = pretrain_or_model

        # Enable gradient checkpointing if supported
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    def _create_critic_model_class(self, base_vlm_model, value_head_prefix="score", packing_samples=False):
        """
        Create a CriticModel class that extends a base vision-language model.

        This factory method dynamically creates a CriticModel class by inheriting from
        the provided base VLM model and adding a value head for value function estimation.

        :param base_vlm_model: The base vision-language model class to extend.
        :type base_vlm_model: type
        :param value_head_prefix: Prefix for the value head attribute name. Defaults to "score".
        :type value_head_prefix: str
        :param packing_samples: Whether to use packed sequence processing. Defaults to False.
        :type packing_samples: bool

        :return: A CriticModel class that extends the base model with value estimation capability.
        :rtype: type
        """
        class CriticModel(base_vlm_model):
            supports_gradient_checkpointing = True

            def __init__(self, config: AutoConfig):
                """
                Initialize the CriticModel with a value head for value function estimation.

                The critic model is used in PPO/GRPO training to estimate state values for
                advantage calculation. It extends the base VLM with a linear value head that
                outputs scalar value estimates for each token position.

                :param config: Model configuration containing architecture and training parameters.
                :type config: AutoConfig
                """
                super().__init__(config)

                self.value_head_prefix = value_head_prefix
                setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

                self.packing_samples = packing_samples

                # mean std for value normalization
                self.normalize_reward = config.normalize_reward
                self.register_buffer("mean", torch.zeros(1), persistent=False)
                self.register_buffer("std", torch.ones(1), persistent=False)

                # load mean/std from config.json if available
                if hasattr(config, "mean"):
                    self.mean[0] = config.mean
                    self.std[0] = config.std

            def forward(
                self,
                input_ids: torch.LongTensor = None,
                num_actions: Optional[Union[int, List[int]]] = None,
                attention_mask: Optional[torch.Tensor] = None,
                pixel_values: torch.LongTensor = None,
                image_grid_thw: torch.LongTensor = None,
                pixel_values_videos: torch.LongTensor = None,
                video_grid_thw: torch.LongTensor = None,
                return_output: bool = False,
                packed_seq_lens: Optional[List[int]] = None,
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
                :type pixel_values: torch.LongTensor
                :param image_grid_thw: Image grid metadata (time, height, width).
                :type image_grid_thw: torch.LongTensor
                :param pixel_values_videos: Pixel values for videos in the input.
                :type pixel_values_videos: torch.LongTensor
                :param video_grid_thw: Video grid metadata (time, height, width).
                :type video_grid_thw: torch.LongTensor
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

                outputs = super().forward(
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

        return CriticModel

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
        Forward pass through the critic model.

        :param input_ids: Input token IDs
        :type input_ids: torch.LongTensor
        :param num_actions: Number of action tokens
        :type num_actions: Optional[Union[int, List[int]]]
        :param attention_mask: Attention mask
        :type attention_mask: Optional[torch.Tensor]
        :param pixel_values: Image pixel values
        :type pixel_values: Optional[torch.Tensor]
        :param image_grid_thw: Image grid dimensions
        :type image_grid_thw: Optional[torch.Tensor]
        :param pixel_values_videos: Video pixel values
        :type pixel_values_videos: Optional[torch.Tensor]
        :param video_grid_thw: Video grid dimensions
        :type video_grid_thw: Optional[torch.Tensor]
        :param return_output: Whether to return model outputs
        :type return_output: bool
        :param packed_seq_lens: Packed sequence lengths
        :type packed_seq_lens: Optional[List[int]]

        :return: Value estimates
        :rtype: Union[torch.Tensor, Tuple[torch.Tensor, dict]]
        """
        return self.model(
            input_ids=input_ids,
            num_actions=num_actions,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            return_output=return_output,
            packed_seq_lens=packed_seq_lens,
        )

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
