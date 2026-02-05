"""
Utility functions for computing log probabilities from logits in PyTorch.

This module provides functions to efficiently calculate log probabilities
for token predictions, with optimizations to handle different data types
and reduce memory consumption. It also includes utilities for finding
linear modules in neural networks and handling position IDs for packed
sequences in transformer models.

The module is particularly useful for:
- Computing log probabilities from model logits with memory-efficient approaches
- Finding LoRA-injectable linear modules in various model architectures
- Handling position IDs in packed sequence scenarios for transformer models
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import deepspeed
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.utils.distributed import all_gather
from lightrft.utils.logging_utils import init_logger
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import (AutoConfig, AutoModel, AutoModelForVision2Seq, BitsAndBytesConfig)
from transformers.integrations.deepspeed import HfDeepSpeedConfig

logger = init_logger(__name__)


# Construct transformer with a value head for sequence classification.
# https://github.com/huggingface/transformers/blob/405b56269812056d9593869e22b7b264d806cb1e/src/transformers/models/llama/modeling_llama.py#L1254
def get_vlm_for_sequence_regression(
    model_name_or_path: str,
    model_type: str,
    *,
    bf16=True,
    load_in_4bit=False,
    lora_rank=0,
    lora_alpha=16,
    target_modules=None,
    lora_dropout=0,
    normalize_reward=False,
    use_flash_attention_2=False,
    ds_config: dict = None,
    init_value_head: bool = False,
    value_head_prefix="score",
    device_map=None,
    packing_samples=False,
    **kwargs,
) -> nn.Module:
    """
    Retrieve a transformer model with a sequence regression head on top.

    This function loads a pretrained transformer model and attaches a linear layer for sequence regression.

    :param model_name_or_path: Path to the pretrained model.
    :type model_name_or_path: str
    :param model_type: Type of the model, either "reward" or "critic".
    :type model_type: str
    :param bf16: Enable bfloat16 precision. Defaults to True.
    :type bf16: bool
    :param load_in_4bit: Load the model in 4-bit precision. Defaults to False.
    :type load_in_4bit: bool
    :param lora_rank: Rank for LoRA adaptation. Defaults to 0.
    :type lora_rank: int
    :param lora_alpha: Alpha parameter for LoRA. Defaults to 16.
    :type lora_alpha: int
    :param target_modules: List of target modules for LoRA. Defaults to None.
    :type target_modules: Optional[List[str]]
    :param lora_dropout: Dropout rate for LoRA layers. Defaults to 0.
    :type lora_dropout: float
    :param normalize_reward: Normalize reward values. Defaults to False.
    :type normalize_reward: bool
    :param use_flash_attention_2: Use Flash Attention 2.0. Defaults to False.
    :type use_flash_attention_2: bool
    :param ds_config: Deepspeed configuration for model partitioning across multiple GPUs when ZeRO-3 is enabled.
        Defaults to None.
    :type ds_config: Optional[dict]
    :param init_value_head: Initialize the value head. Defaults to False.
    :type init_value_head: bool
    :param value_head_prefix: Prefix for the value head. Defaults to "score".
    :type value_head_prefix: str
    :param device_map: Map of devices for model loading. Defaults to None.
    :type device_map: Optional[dict]
    :param packing_samples: Whether to pack samples during training. Defaults to False.
    :type packing_samples: bool
    :param kwargs: Additional keyword arguments passed to the model constructor.
    :type kwargs: dict

    :return: A pretrained transformer model with a sequence regression head.
    :rtype: nn.Module
    """
    assert (
        model_type == "critic" or model_type == "reward"
    ), f"invalid model_type: {model_type}, should be critic or reward."

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.normalize_reward = normalize_reward
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

    # Prioritize using the value_head_prefix in the model configuration.
    value_head_prefix = getattr(config, "value_head_prefix", value_head_prefix)
    logger.info(f"set value_head_prefix to `{value_head_prefix}`")
    base_class = AutoModelForVision2Seq._model_mapping[type(config)]

    if model_type == "reward":
        cls_class = _get_reward_model(base_class, value_head_prefix, packing_samples)
    else:
        cls_class = _get_critic_model(base_class, value_head_prefix, packing_samples)

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    if load_in_4bit:
        assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        nf4_config = None

    model = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        quantization_config=nf4_config,
        device_map=device_map,
        **kwargs,
    )

    # LoRA
    if lora_rank > 0:
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        if load_in_4bit:
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(torch.bfloat16)
                if "norm" in name:
                    module = module.to(torch.float32)
                if value_head_prefix in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        module = module.to(torch.bfloat16)

    # MoE - balancing loss
    model_config = model.config.to_dict()
    if "output_router_logits" in model_config:
        print("[MoE] set output_router_logits as True")
        model.config.output_router_logits = True

    # https://github.com/huggingface/transformers/issues/26877
    model.config.use_cache = False

    # NOTE: For reward model training only, intialize value_head manually
    # because deepspeed.zero.Init() will not intialize them.
    # TODO: Find a better way to clarify reward model training.
    if init_value_head:
        value_head = getattr(model, value_head_prefix)
        if dschf is not None:
            logger.info("initialize value_head for ZeRO-3 reward model training.")
            with deepspeed.zero.GatheredParameters([value_head.weight], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        else:
            value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))

    return model


def _get_reward_model(base_vlm_model, value_head_prefix="score", packing_samples=False):
    """
    Create a RewardModel class that extends a base vision-language model.

    This factory function dynamically creates a RewardModel class by inheriting from
    the provided base VLM model and adding a value head for reward prediction.

    :param base_vlm_model: The base vision-language model class to extend.
    :type base_vlm_model: type
    :param value_head_prefix: Prefix for the value head attribute name. Defaults to "score".
    :type value_head_prefix: str
    :param packing_samples: Whether to use packed sequence processing. Defaults to False.
    :type packing_samples: bool

    :return: A RewardModel class that extends the base model with reward prediction capability.
    :rtype: type
    """
    class RewardModel(base_vlm_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            """
            Initialize the RewardModel with a value head for reward prediction.

            :param config: Model configuration containing architecture and training parameters.
            :type config: AutoConfig
            """
            super().__init__(config)
            # setattr(self, self.base_model_prefix, base_vlm_model(config))

            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            self.packing_samples = packing_samples

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            pixel_values: torch.LongTensor = None,
            image_grid_thw: torch.LongTensor = None,
            pixel_values_videos: torch.LongTensor = None,
            video_grid_thw: torch.LongTensor = None,
            return_output=False,
            ring_attn_group=None,
            packed_seq_lens=None,
        ) -> torch.Tensor:
            """
            Forward pass to compute reward scores for input sequences.

            :param input_ids: Input token IDs.
            :type input_ids: torch.LongTensor
            :param attention_mask: Attention mask for input sequences.
            :type attention_mask: Optional[torch.Tensor]
            :param pixel_values: Pixel values for images.
            :type pixel_values: torch.LongTensor
            :param image_grid_thw: Image grid metadata (time, height, width).
            :type image_grid_thw: torch.LongTensor
            :param pixel_values_videos: Pixel values for videos.
            :type pixel_values_videos: torch.LongTensor
            :param video_grid_thw: Video grid metadata (time, height, width).
            :type video_grid_thw: torch.LongTensor
            :param return_output: Whether to return full model outputs along with rewards. Defaults to False.
            :type return_output: bool
            :param ring_attn_group: Process group for ring attention.
            :type ring_attn_group: Optional[ProcessGroup]
            :param packed_seq_lens: Lengths of packed sequences for batch processing.
            :type packed_seq_lens: Optional[List[int]]

            :return: Reward scores, or tuple of (rewards, outputs) if return_output is True.
            :rtype: Union[torch.Tensor, Tuple[torch.Tensor, dict]]
            """
            if not self.packing_samples:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
            else:
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
            # print(outputs.keys())
            # print(outputs["hidden_states"][-1].shape)
            last_hidden_states = outputs["hidden_states"][-1]
            values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)

            if self.packing_samples:
                if ring_attn_group is not None:
                    reward = all_gather(values, ring_attn_group).reshape(1, -1)
                else:
                    reward = values
                # TODO: convert packed_seq_lens into torch tensor in advance
                packed_seq_lens = torch.tensor(packed_seq_lens, device=values.device)
                eos_indices = packed_seq_lens.cumsum(dim=0) - 1
                reward = reward.squeeze(0).gather(dim=0, index=eos_indices)
            else:
                eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                reward = values.gather(dim=1, index=eos_indices).squeeze(1)

            if not self.training and self.normalize_reward:
                reward = (reward - self.mean) / self.std

            return (reward, outputs) if return_output else reward

    return RewardModel


def _get_critic_model(base_vlm_model, value_head_prefix="score", packing_samples=False):
    """
    Create a CriticModel class that extends a base vision-language model.

    This factory function dynamically creates a CriticModel class by inheriting from
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
                          Expected to have 'normalize_reward' attribute and optionally 'mean'/'std'
                          for value normalization.
            :type config: AutoConfig
            """
            super().__init__(config)
            # setattr(self, self.base_model_prefix, base_vlm_model(config))

            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            self.packing_samples = packing_samples

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            num_actions: Optional[Union[int, list[int]]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            pixel_values: torch.LongTensor = None,
            image_grid_thw: torch.LongTensor = None,
            pixel_values_videos: torch.LongTensor = None,
            video_grid_thw: torch.LongTensor = None,
            return_output=False,
            packed_seq_lens=None,
        ) -> torch.Tensor:
            """
            Forward pass to compute value estimates for action sequences.

            This method computes state-value estimates for tokens in the action sequence,
            which are used to calculate advantages in PPO/GRPO training. The values are
            computed from the last hidden states of the base VLM, excluding the final token.

            :param input_ids: Input token IDs including both prompt and generated actions.
            :type input_ids: torch.LongTensor
            :param num_actions: Number of action tokens to compute values for. Can be:
                               - int: Same number of actions for all sequences in batch
                               - list[int]: Different action counts per sequence (for packed samples)
                               - None: Return full model outputs without extracting action values
            :type num_actions: Optional[Union[int, list[int]]]
            :param attention_mask: Attention mask for input sequences (1 for valid tokens, 0 for padding).
            :type attention_mask: Optional[torch.Tensor]
            :param pixel_values: Pixel values for images in the input.
            :type pixel_values: torch.LongTensor
            :param image_grid_thw: Image grid metadata (time, height, width) for image tokens.
            :type image_grid_thw: torch.LongTensor
            :param pixel_values_videos: Pixel values for videos in the input.
            :type pixel_values_videos: torch.LongTensor
            :param video_grid_thw: Video grid metadata (time, height, width) for video tokens.
            :type video_grid_thw: torch.LongTensor
            :param return_output: If True, return tuple of (values, model_outputs). Defaults to False.
            :type return_output: bool
            :param packed_seq_lens: Lengths of packed sequences for efficient batch processing.
                                   Required when packing_samples is True.
            :type packed_seq_lens: Optional[List[int]]

            :return: Value estimates for action tokens. Shape depends on num_actions:
                    - If num_actions is int: (batch_size, num_actions)
                    - If num_actions is list: (batch_size, total_actions) with concatenated sequences
                    - If num_actions is None and return_output is True: model outputs dict
                    - If return_output is True: tuple of (action_values, model_outputs)
            :rtype: Union[torch.Tensor, Tuple[torch.Tensor, dict], dict]

            Example::

                >>> # Single sequence case
                >>> values = critic_model(
                ...     input_ids=input_ids,
                ...     num_actions=10,
                ...     attention_mask=attention_mask,
                ...     pixel_values=pixel_values,
                ...     image_grid_thw=image_grid_thw
                ... )
                >>> values.shape  # (batch_size, 10)

                >>> # Packed sequence case
                >>> values = critic_model(
                ...     input_ids=packed_input_ids,
                ...     num_actions=[8, 10, 12],
                ...     attention_mask=packed_attention_mask,
                ...     packed_seq_lens=[50, 60, 55]
                ... )
                >>> values.shape  # (1, 30)  # 8+10+12=30
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

            # normalize reward
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


def find_all_linear_modules(model: "nn.Module", freeze_vision_tower: bool) -> List[str]:
    """
    Find all linear modules that can be injected with LoRA (Low-Rank Adaptation).

    This function scans through a neural network model to identify all linear layers
    that are suitable for LoRA injection, while excluding certain forbidden modules
    based on the model type. It handles various model architectures including ChatGLM,
    LLaVA variants, Qwen2 VL models, and others.

    :param model: The neural network model to scan for linear modules
    :type model: nn.Module
    :param freeze_vision_tower: Whether to freeze the vision tower components.
                               If True, vision-related modules will be added to forbidden list
    :type freeze_vision_tower: bool

    :return: List of linear module names that can be used for LoRA injection
    :rtype: List[str]

    Example::
        >>> import torch.nn as nn
        >>> model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
        >>> linear_modules = find_all_linear_modules(model, freeze_vision_tower=False)
        >>> print(linear_modules)  # ['Linear']
    """
    model_type = getattr(model.config, "model_type", None)
    forbidden = {"lm_head"}
    if model_type == "chatglm":
        forbidden.add("output_layer")
    elif model_type in ["llava", "llava_next", "llava_next_video", "mllama", "paligemma", "video_llava"]:
        forbidden.add("multi_modal_projector")
    elif model_type in ["qwen2_vl", "qwen2_5_vl"]:
        forbidden.add("merger")

    if freeze_vision_tower:
        if model_type in ["mllama"]:
            forbidden.add("vision_model")
        elif model_type in ["qwen2_vl", "qwen2_5_vl"]:
            forbidden.add("visual")
        else:
            forbidden.add("vision_tower")

    module_names = set()
    for name, module in model.named_modules():
        if any(fm in name for fm in forbidden):
            continue
        if "Linear" in module.__class__.__name__ and "Embedding" not in module.__class__.__name__:
            module_names.add(name.split(".")[-1])
    return list(module_names)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy from logits using Categorical distribution for efficient calculation.

    This function calculates the entropy of the probability distribution over the vocabulary
    for each token position. Higher entropy indicates more uncertainty in token prediction,
    which corresponds to "forking tokens" that determine reasoning directions.

    :param logits: Logits tensor of shape (batch_size, sequence_length, vocab_size)
                  or (batch_size, vocab_size)
    :type logits: torch.Tensor

    :return: Entropy values for each token position, of shape (batch_size, sequence_length)
            or (batch_size,)
    :rtype: torch.Tensor

    Example::
        >>> logits = torch.randn(2, 10, 50000)  # batch_size=2, seq_len=10, vocab_size=50000
        >>> entropy = entropy_from_logits(logits)
        >>> entropy.shape
        torch.Size([2, 10])
    """
    # Use Categorical distribution for efficient entropy calculation
    categorical = dist.Categorical(logits=logits)
    return categorical.entropy()


def create_high_entropy_mask(
    entropy: torch.Tensor,
    action_mask: Optional[torch.Tensor],
    high_entropy_ratio: float = 0.2,
) -> torch.Tensor:
    """
    Create a binary mask for high-entropy tokens based on the specified ratio.

    This function identifies the top-k highest entropy tokens (forking tokens) within each sequence
    and creates a binary mask. Only tokens with high entropy will be used for gradient updates,
    following the approach in "Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective
    Reinforcement Learning for LLM Reasoning" (https://arxiv.org/abs/2506.01939).

    The paper shows that utilizing only 20% of high-entropy tokens can maintain performance comparable
    to full-gradient updates, with common value of 0.2 (top 20% highest entropy tokens).

    :param entropy: Entropy values for each token, shape (batch_size, sequence_length)
    :type entropy: torch.Tensor
    :param action_mask: Binary mask indicating valid tokens (1 for valid, 0 for padding)
    :type action_mask: Optional[torch.Tensor]
    :param high_entropy_ratio: Ratio of high-entropy tokens to keep (e.g., 0.2 means top 20%).
                               Common value: 0.2. Based on https://arxiv.org/abs/2506.01939, defaults to 0.2
    :type high_entropy_ratio: float

    :return: Binary mask for high-entropy tokens, shape (batch_size, sequence_length)
    :rtype: torch.Tensor

    Example::
        >>> entropy = torch.tensor([[1.0, 5.0, 2.0, 6.0, 3.0], [2.0, 4.0, 1.0, 5.0, 0.0]])
        >>> action_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
        >>> mask = create_high_entropy_mask(entropy, action_mask, high_entropy_ratio=0.4)
        >>> mask
        tensor([[0, 1, 0, 1, 0], [0, 1, 0, 1, 0]])  # Top 40% (2 out of 5 valid tokens)
    """
    if high_entropy_ratio <= 0.0 or high_entropy_ratio >= 1.0:
        # Return all-ones mask if ratio is invalid
        if action_mask is not None:
            return action_mask.clone()
        return torch.ones_like(entropy, dtype=torch.float32)

    # Validate shapes
    if len(entropy.shape) != 2:
        raise ValueError(f"entropy must be 2D tensor (batch_size, seq_len), got shape {entropy.shape}")

    batch_size, seq_len = entropy.shape

    if action_mask is not None:
        if len(action_mask.shape) != 2:
            raise ValueError(f"action_mask must be 2D tensor (batch_size, seq_len), got shape {action_mask.shape}")
        if action_mask.shape != entropy.shape:
            raise ValueError(f"action_mask shape {action_mask.shape} must match entropy shape {entropy.shape}")

    high_entropy_mask = torch.zeros_like(entropy, dtype=torch.float32)

    for i in range(batch_size):
        # Get valid entropy values for this sequence
        if action_mask is not None:
            # Ensure both are 1D tensors of the same length
            entropy_i = entropy[i]  # Shape: (seq_len,)
            mask_i = action_mask[i]  # Shape: (seq_len,)

            # Convert to float if needed for multiplication
            if mask_i.dtype != entropy_i.dtype:
                mask_i = mask_i.to(dtype=entropy_i.dtype)

            valid_entropy = entropy_i * mask_i
            valid_indices = mask_i.bool()
        else:
            valid_entropy = entropy[i]
            valid_indices = torch.ones(seq_len, dtype=torch.bool, device=entropy.device)

        if not valid_indices.any():
            continue

        # Calculate number of high-entropy tokens to keep
        num_valid = valid_indices.sum().item()
        num_high_entropy = max(1, int(num_valid * high_entropy_ratio))

        # Get top-k highest entropy tokens
        _, top_indices = torch.topk(valid_entropy, k=num_high_entropy, dim=0)
        high_entropy_mask[i, top_indices] = 1.0

    return high_entropy_mask


def log_probs_from_logits(
    logits: torch.Tensor, labels: torch.Tensor, disable_logprobs_flashattn: bool = False
) -> torch.Tensor:
    """
    Compute log probabilities for the given labels from logits.

    This function calculates log probabilities efficiently, using different approaches
    based on the input data type to optimize memory usage. For float32/float64 tensors,
    it uses a direct computation approach, while for other data types (e.g. float16 and bfloat16)
    it uses PyTorch's log_softmax function with row-by-row processing to reduce peak memory consumption.

    :param logits: Logits tensor of shape (batch_size, sequence_length, vocab_size)
                  or (batch_size, vocab_size)
    :type logits: torch.Tensor

    :param labels: Labels tensor containing token indices, of shape (batch_size, sequence_length)
                  or (batch_size,)
    :type labels: torch.Tensor

    :param disable_logprobs_flashattn: Whether to use flash attn when calculating cross entropy loss
                                      default to False
    :type disable_logprobs_flashattn: bool

    :return: Log probabilities for the given labels, of shape matching labels
    :rtype: torch.Tensor

    Example::
        >>> logits = torch.randn(2, 3, 5)  # batch_size=2, seq_len=3, vocab_size=5
        >>> labels = torch.randint(0, 5, (2, 3))  # batch_size=2, seq_len=3
        >>> log_probs = log_probs_from_logits(logits, labels)
        >>> log_probs.shape
        torch.Size([2, 3])
    """
    if logits.dtype in [torch.float32, torch.float64]:
        batch_dim = logits.shape[:-1]
        last_dim = logits.shape[-1]
        flashattn_available = False
        if not disable_logprobs_flashattn:
            try:
                from flash_attn.ops.triton.cross_entropy import \
                    cross_entropy_loss

                flashattn_available = True
            except ImportError:
                logger.warning("Failed to import cross_entropy_loss from flash_attn")
                flashattn_available = False
        if flashattn_available:
            # use cross_entropy_loss from flash_attn to reduce peak mem consumption
            output = cross_entropy_loss(logits.reshape(-1, last_dim), labels.reshape(-1))
            log_probs_labels = -output[0].view(*batch_dim)
        else:
            logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            logsumexp_values = torch.stack([torch.logsumexp(logit, dim=-1)
                                            for logit in logits]  # loop to reduce peak mem consumption
                                           )
            log_probs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        log_probs_labels = []
        for row_logits, row_labels in zip(logits, labels):  # loop to reduce peak mem consumption
            row_log_probs = F.log_softmax(row_logits, dim=-1)
            row_log_probs_labels = row_log_probs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            log_probs_labels.append(row_log_probs_labels)
        log_probs_labels = torch.stack(log_probs_labels)
    return log_probs_labels


def reset_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Generate position IDs for packed sequences based on an attention mask.

    In a packed sequence, multiple independent sequences are concatenated into a
    single tensor row. The attention mask distinguishes these sequences using
    unique integer identifiers (e.g., 1, 2, 3, ...). This function creates a
    corresponding position ID tensor where positions are reset to zero at the
    beginning of each packed sequence.

    :param attention_mask: A 2D tensor of shape (batch_size, sequence_length)
                          where different positive integers mark different sequences within the
                          same row, and 0 typically represents padding.
    :type attention_mask: torch.Tensor

    :return: A 2D tensor of the same shape as `attention_mask` containing
            the calculated position IDs. Each packed sequence will have its own
            position IDs starting from 0.
    :rtype: torch.Tensor

    Example::
        >>> attention_mask = torch.tensor([[1, 1, 1, 2, 2, 2, 3, 3, 0]])
        >>> reset_position_ids(attention_mask)
        tensor([[0, 1, 2, 0, 1, 2, 0, 1, 0]])
    """
    # Initialize position_ids with zeros, same shape and device as the input mask.
    position_ids = torch.zeros_like(attention_mask, dtype=torch.long)

    # Iterate over each sequence in the batch.
    for i in range(attention_mask.size(0)):
        mask = attention_mask[i]

        # Determine the number of packed samples in the current sequence by finding the max value in the mask.
        # e.g., if mask is [1, 1, 2, 2, 2, 0], seq_num is 2.
        seq_num = mask.max().item()

        # Iterate through each packed sample, identified by its index (1, 2, ...).
        for index in range(1, seq_num + 1):
            # Create a boolean mask to isolate the tokens of the current sample.
            sample_mask = mask == index

            # Calculate the length of the current sample.
            sample_length = sample_mask.sum().item()

            # Generate a range of position IDs from 0 to sample_length - 1.
            new_position_ids = torch.arange(sample_length, device=mask.device)

            # Use the boolean mask to place the new position IDs into the correct locations.
            position_ids[i, sample_mask] = new_position_ids

    return position_ids


def apply_lora_configuration(
    model: "nn.Module",
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: Optional[List[str]] = None,
    freeze_vision_tower: bool = True,
) -> "nn.Module":
    """
    Apply LoRA (Low-Rank Adaptation) configuration to a model.

    This function configures and applies LoRA adaptation to the specified model,
    including setting up the LoRA configuration and applying it to the model.

    :param model: The model to apply LoRA configuration to
    :type model: nn.Module
    :param lora_rank: Rank for LoRA adaptation
    :type lora_rank: int
    :param lora_alpha: Alpha parameter for LoRA scaling
    :type lora_alpha: int
    :param lora_dropout: Dropout rate for LoRA layers
    :type lora_dropout: float
    :param target_modules: List of target modules for applying LoRA (auto-detected if None)
    :type target_modules: Optional[List[str]]
    :param freeze_vision_tower: Whether to freeze the vision tower components
    :type freeze_vision_tower: bool

    :return: The model with LoRA configuration applied
    :rtype: nn.Module

    Example::
        >>> model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        >>> model = apply_lora_configuration(
        ...     model=model,
        ...     lora_rank=16,
        ...     lora_alpha=32,
        ...     lora_dropout=0.1
        ... )
    """
    # Enable input require gradients for LoRA
    model.enable_input_require_grads()

    # Auto-detect target modules if not provided
    if target_modules is None:
        target_modules = find_all_linear_modules(model, freeze_vision_tower)

    print("target_modules: ", target_modules)

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)

    return model


def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    kl_estimator: str = "k1",
) -> torch.Tensor:
    """
    Compute approximate KL divergence between two probability distributions.

    This function implements three different estimators for KL divergence approximation
    as described in Schulman's blog: http://joschu.net/blog/kl-approx.html

    :param log_probs: Log probabilities of the new distribution
    :type log_probs: torch.Tensor
    :param log_probs_base: Log probabilities of the base/reference distribution
    :type log_probs_base: torch.Tensor
    :param action_mask: Binary mask indicating valid action positions (1 for valid, 0 for padding)
    :type action_mask: Optional[torch.Tensor]
    :param kl_estimator: Type of KL estimator to use ("k1", "k2", or "k3")
    :type kl_estimator: str

    :return: Approximate KL divergence values
    :rtype: torch.Tensor

    Example::
        >>> log_probs = torch.tensor([[0.1, -0.2, 0.3], [-0.1, 0.2, 0.1]])
        >>> log_probs_base = torch.tensor([[0.2, -0.1, 0.2], [-0.2, 0.1, 0.2]])
        >>> action_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        >>> kl = compute_approx_kl(log_probs, log_probs_base, action_mask, "k1")
        >>> kl.shape
        torch.Size([2, 3])
    """

    assert kl_estimator in ["k1", "k2", "k3"], f"Invalid kl_estimator: {kl_estimator}"

    if kl_estimator == "k1":
        log_ratio = log_probs.float() - log_probs_base.float()
        if action_mask is not None:
            log_ratio = log_ratio * action_mask

    # The k2 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    # The k2_loss is approximately equivalent to the
    # one-step KL divergence penalty with the k1 estimator
    # used in https://arxiv.org/pdf/2310.10505.
    elif kl_estimator == "k2":
        log_ratio = log_probs.float() - log_probs_base.float()
        if action_mask is not None:
            log_ratio = log_ratio * action_mask
        log_ratio = log_ratio ** 2 / 2.0

    # The k3 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    elif kl_estimator == "k3":
        log_ratio = log_probs.float() - log_probs_base.float()
        if action_mask is not None:
            log_ratio = log_ratio * action_mask
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio

    return log_ratio


def compute_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    kl: Union[torch.Tensor, list[torch.Tensor]],
    action_mask: Optional[torch.Tensor] = None,
    num_actions: Optional[Union[int, list[int]]] = None,
    reward_clip_range: Tuple[float, float] = None,
) -> Union[torch.Tensor, list[torch.Tensor]]:
    """
    Compute final reward by combining base reward with KL penalty.

    Combines base reward with KL divergence penalty to encourage policy stability.
    Supports two modes: with action mask (efficient) and without (individual processing).

    :param r: Base reward tensor or scalar
    :type r: Union[torch.Tensor, float]
    :param kl_coef: KL penalty coefficient (<=0 disables penalty)
    :type kl_coef: float
    :param kl: KL divergence values as tensor or list
    :type kl: Union[torch.Tensor, list[torch.Tensor]]
    :param action_mask: Binary mask for valid action positions
    :type action_mask: Optional[torch.Tensor]
    :param num_actions: Number of actions per sequence (no mask mode)
    :type num_actions: Optional[Union[int, list[int]]]
    :param reward_clip_range: (min, max) to clip base reward
    :type reward_clip_range: Tuple[float, float]

    :return: Final reward tensor or list
    :rtype: Union[torch.Tensor, list[torch.Tensor]]

    Example::
        >>> r = torch.tensor([1.0, 2.0])
        >>> kl_coef = 0.1
        >>> kl = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]])
        >>> action_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        >>> reward = compute_reward(r, kl_coef, kl, action_mask)
        >>> reward.shape
        torch.Size([2, 3])
    """
    if kl_coef <= 0.0:
        kl_coef = 0.0

    if reward_clip_range:
        r = r.clamp(min=reward_clip_range[0], max=reward_clip_range[1])

    if action_mask is not None:
        kl_reward = -kl_coef * kl
        # The following code is equivalent to:
        #
        # last_reward = torch.zeros_like(kl)
        # for i in range(last_reward.size(0)):
        #     for t in reversed(range(last_reward.size(1))):
        #         if action_mask[i][t] > 0.5:
        #             last_reward[i][t] = r[i]
        #             break
        #
        eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
        last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))

        reward = last_reward + kl_reward
    else:
        # TODO: write a more efficient version
        reward = []
        for i, (kl_seg, action_len) in enumerate(zip(kl, num_actions)):
            kl_reward = -kl_coef * kl_seg
            kl_reward[action_len - 1] += r[i]
            reward.append(kl_reward)

    return reward


def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: int = None) -> torch.Tensor:
    """
    Compute mean of tensor excluding masked (padded) values.

    Calculates mean along specified dimensions, ignoring positions where mask is zero.
    Useful for sequence data with variable lengths.

    :param tensor: Input tensor to average
    :type tensor: torch.Tensor
    :param mask: Binary mask (1 for valid, 0 for padding). None for regular mean.
    :type mask: Optional[torch.Tensor]
    :param dim: Dimension(s) to compute mean along. None for global mean.
    :type dim: int

    :return: Mean value(s) with masked positions excluded
    :rtype: torch.Tensor

    Example::
        >>> tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        >>> masked_mean(tensor, mask)
        tensor(2.6667)
        >>> masked_mean(tensor, mask, dim=1)
        tensor([1.5000, 4.0000])
    """
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)


def unpacking_samples(values: torch.Tensor, packed_seqlens: list[int]) -> list[torch.Tensor]:
    """
    Unpack concatenated sequences into individual sequences.

    Splits packed tensor into multiple sequences based on original lengths.
    Reverses packing operation for efficient batch processing.

    :param values: Concatenated tensor (1, total_length) or (total_length,)
    :type values: torch.Tensor
    :param packed_seqlens: List of original sequence lengths
    :type packed_seqlens: list[int]

    :return: List of unpacked sequence tensors
    :rtype: list[torch.Tensor]

    Example::
        >>> values = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        >>> packed_seqlens = [3, 2, 3]
        >>> unpacked = unpacking_samples(values, packed_seqlens)
        >>> [t.tolist() for t in unpacked]
        [[1, 2, 3], [4, 5], [6, 7, 8]]
    """
    values = values.squeeze(0)
    unpacked_values = []
    offset = 0
    for seqlen in packed_seqlens:
        unpacked_values.append(values[offset:offset + seqlen])
        offset += seqlen
    return unpacked_values


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    """
    Left-pad a tensor to a target length along a given dimension.

    :param tensor: Input tensor to be padded.
    :type tensor: torch.Tensor
    :param length: Target length along ``dim``. If the input is already
        at least this length, the tensor is returned unchanged.
    :type length: int
    :param pad_value: Scalar pad value to use for the new elements.
    :type pad_value: int or float
    :param dim: Dimension along which to pad (default: ``-1``).
    :type dim: int

    :returns: Tensor padded on the left along ``dim`` to size
        ``length`` if needed; otherwise the original tensor.
    :rtype: torch.Tensor
    """
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        # left pad
        return torch.cat([pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device), tensor], dim=dim)


def concatenated_forward(
    model: Callable, input0_ids: torch.Tensor, input0_mask: torch.Tensor, input1_ids: torch.Tensor,
    input1_mask: torch.Tensor, input0_img_pixels: Optional[torch.Tensor], input0_img_grid_thws: Optional[torch.Tensor],
    input1_img_pixels: Optional[torch.Tensor], input1_img_grid_thws: Optional[torch.Tensor],
    input0_video_pixels: Optional[torch.Tensor], input0_video_grid_thws: Optional[torch.Tensor],
    input1_video_pixels: Optional[torch.Tensor], input1_video_grid_thws: Optional[torch.Tensor], pad_token_id: int
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Concatenates paired candidate inputs and runs a forward pass for vision-language models.

    This utility is used in preference/reward modeling scenarios where two candidates
    (e.g., chosen vs. rejected) are processed together for efficiency. Text sequences
    from both candidates are left-padded to the maximum length across the pair, and
    multimodal inputs (images/videos) are concatenated along the batch dimension when provided.

    :param model: Callable model that accepts input ids, attention masks, and optional multimodal inputs.
    :type model: Callable
    :param input0_ids: Token ids for candidate 0.
    :type input0_ids: torch.LongTensor of shape ``(B, T0)``
    :param input0_mask: Attention mask for candidate 0 (1 = attend, 0 = pad).
    :type input0_mask: torch.LongTensor of shape ``(B, T0)``
    :param input1_ids: Token ids for candidate 1.
    :type input1_ids: torch.LongTensor of shape ``(B, T1)``
    :param input1_mask: Attention mask for candidate 1 (1 = attend, 0 = pad).
    :type input1_mask: torch.LongTensor of shape ``(B, T1)``
    :param input0_img_pixels: Image pixel tensor for candidate 0, or ``None`` if not used.
    :type input0_img_pixels: Optional[torch.Tensor]
    :param input0_img_grid_thws: Image grid metadata (e.g., THW) for candidate 0, or ``None``.
    :type input0_img_grid_thws: Optional[torch.Tensor]
    :param input1_img_pixels: Image pixel tensor for candidate 1, or ``None`` if not used.
    :type input1_img_pixels: Optional[torch.Tensor]
    :param input1_img_grid_thws: Image grid metadata (e.g., THW) for candidate 1, or ``None``.
    :type input1_img_grid_thws: Optional[torch.Tensor]
    :param input0_video_pixels: Video pixel tensor for candidate 0, or ``None`` if not used.
    :type input0_video_pixels: Optional[torch.Tensor]
    :param input0_video_grid_thws: Video grid metadata (e.g., THW) for candidate 0, or ``None``.
    :type input0_video_grid_thws: Optional[torch.Tensor]
    :param input1_video_pixels: Video pixel tensor for candidate 1, or ``None`` if not used.
    :type input1_video_pixels: Optional[torch.Tensor]
    :param input1_video_grid_thws: Video grid metadata (e.g., THW) for candidate 1, or ``None``.
    :type input1_video_grid_thws: Optional[torch.Tensor]
    :param pad_token_id: Token id used for left-padding text sequences to equal length.
    :type pad_token_id: int

    :return: A tuple ``(scores0, scores1)`` where each element is either a tensor of shape
             ``(B, ...)`` or a dict mapping head names to tensors, mirroring the model output
             for each candidate.
    :rtype: Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], Union[torch.Tensor, Dict[str, torch.Tensor]]]

    """
    # Compute shared maximum lengths across the pair for text ids and masks.
    max_length_ids = max(input0_ids.shape[1], input1_ids.shape[1])
    max_length_mask = max(input0_mask.shape[1], input1_mask.shape[1])

    input_ids = torch.cat(
        (
            pad_to_length(input0_ids, max_length_ids, pad_token_id),
            pad_to_length(input1_ids, max_length_ids, pad_token_id),
        ),
        dim=0,
    )

    att_masks = torch.cat(
        (pad_to_length(input0_mask, max_length_mask, 0), pad_to_length(input1_mask, max_length_mask, 0)), dim=0
    )

    # Default multimodal inputs to None unless provided.
    pixel_values = None
    image_grid_thws = None
    pixel_values_videos = None
    video_grid_thws = None

    with torch.no_grad():
        if input0_img_pixels is not None:
            pixel_values = torch.cat((input0_img_pixels, input1_img_pixels), dim=0)
            image_grid_thws = torch.cat((input0_img_grid_thws, input1_img_grid_thws), dim=0)

        if input0_video_pixels is not None:
            pixel_values_videos = torch.cat((input0_video_pixels, input1_video_pixels), dim=0)
            video_grid_thws = torch.cat((input0_video_grid_thws, input1_video_grid_thws), dim=0)

    # Forward pass over the concatenated batch (size 2 * B).
    scores = model(
        input_ids,
        attention_mask=att_masks,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thws,
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thws
    )

    batch_size_0 = input0_ids.shape[0]

    scores0 = {head_type: score[:batch_size_0] for head_type, score in scores.items()}
    scores1 = {head_type: score[batch_size_0:] for head_type, score in scores.items()}

    return scores0, scores1


class AttentionPooling(nn.Module):
    """
    Attention pooling over the sequence dimension of VLM hidden states.

    This module compresses a sequence of hidden states into a single fixed-size
    representation by attending from a learnable global query to the sequence.

    :param hidden_size: Hidden size of the backbone model. Must be divisible by ``num_heads``.
    :type hidden_size: int
    :param num_heads: Number of attention heads used for pooling. Defaults to ``4``.
    :type num_heads: int, optional
    :param qkv_bias: Whether to use bias terms in the key and value projection layers. Defaults to ``False``.
    :type qkv_bias: bool, optional
    :param position_bias: If ``True``, add a linear 1-D positional bias to attention logits. Defaults to ``False``.
    :type position_bias: bool, optional
    :param position_bias_scale: Scale factor for the positional bias; larger values more strongly favor later positions.
    :type position_bias_scale: float, optional

    .. note::
       The learnable query is shared across heads and batches. Attention logits are
       scaled by ``1 / sqrt(head_dim)`` where ``head_dim = hidden_size // num_heads``.

    Example::

        pool = AttentionPooling(hidden_size=1024, num_heads=8).to(torch.bfloat16).cuda()
        x = torch.randn(2, 128, 1024, dtype=torch.bfloat16, device='cuda')  # (B=2, S=128, C=1024)
        y = pool(x)
        assert y.shape == (2, 1024)
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        qkv_bias: bool = False,
        position_bias: bool = False,
        position_bias_scale: float = 3.0,
    ) -> None:
        super(AttentionPooling, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.position_bias = position_bias
        self.position_bias_scale = position_bias_scale

        self.k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        # 0.02 for better initialization
        self.query = nn.Parameter(torch.randn(hidden_size) * 0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply attention pooling over the sequence of hidden states.

        :param hidden_states: Hidden states to pool, of shape ``(B, S, C)``.
        :type hidden_states: torch.Tensor
        :returns: Pooled hidden states of shape ``(B, C)``.
        :rtype: torch.Tensor
        """
        B, S, C = hidden_states.shape

        # Multi-head projection for key and value
        k = self.k(hidden_states).reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B, H, S, D
        v = self.v(hidden_states).reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B, H, S, D

        # Expand query for batch dimension
        q = self.query.unsqueeze(0).expand(B, -1, -1)  # B, H, C
        q = q.unsqueeze(2)  # B, H, 1, C
        q = q.reshape(B, self.num_heads, 1, self.head_dim)  # B, H, 1, D

        # Attention weights
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, H, 1, S

        # Add position bias
        if self.position_bias:
            position_bias = torch.arange(S, device=k.device).float() / S * self.position_bias_scale
            attn = attn + position_bias.view(1, 1, 1, -1)  # Add position bias

        # Attention pooling
        attn = torch.softmax(attn, dim=-1)  # B, H, 1, S
        out = (attn @ v).squeeze(2)  # B, H, D
        out = out.reshape(B, -1)  # B, C

        return out
