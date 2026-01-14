"""
Actor model implementation for reinforcement learning with language models.

This module provides the ActorLanguage class, which serves as a foundation for implementing
actor models in reinforcement learning scenarios. The actor is responsible for selecting
actions based on learned policies and supports both vision-language (VL) and text-only
models. It includes support for various optimization techniques such as LoRA adaptation,
quantization, and distributed training with DeepSpeed.

The module handles model initialization from pretrained checkpoints or existing model
instances, applies various optimizations like Flash Attention,and provides methods for
text generation and forward passes with action probability computation.
"""

from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
)
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from .utils import apply_lora_configuration, log_probs_from_logits, reset_position_ids


class ActorLanguage(nn.Module):
    """
    A general Actor model for reinforcement learning that supports both Vision-Language (VL) and text-only models.

    This class serves as a foundation for implementing various actor models, which are responsible
    for selecting actions based on the policy learned from the environment. It supports advanced
    features like LoRA adaptation, quantization, Flash Attention, and distributed training.

    :param pretrain_or_model: A pretrained model path/name or a model instance to be used as the actor.
    :type pretrain_or_model: Union[str, nn.Module]
    :param use_flash_attention_2: Whether to utilize Flash Attention 2.0 for improved performance.
    :type use_flash_attention_2: bool
    :param bf16: Enable bfloat16 precision for model computations.
    :type bf16: bool
    :param lora_rank: Rank for LoRA adaptation. Set to 0 to disable LoRA.
    :type lora_rank: int
    :param lora_alpha: Alpha parameter for LoRA scaling.
    :type lora_alpha: int
    :param lora_dropout: Dropout rate for LoRA layers.
    :type lora_dropout: float
    :param target_modules: List of target modules for applying LoRA. If None, auto-detects linear modules.
    :type target_modules: Optional[List[str]]
    :param ds_config: Configuration for DeepSpeed, enabling model partitioning across multiple GPUs.
    :type ds_config: Optional[dict]
    :param device_map: Device mapping for loading the model onto specific devices.
    :type device_map: Optional[dict]
    :param packing_samples: Whether to pack samples during training for efficiency.
    :type packing_samples: bool

    Example::

        # Initialize with a pretrained model
        actor = ActorLanguage(
            pretrain_or_model="microsoft/DialoGPT-medium",
            lora_rank=16,
            lora_alpha=32,
            use_flash_attention_2=True
        )

        # Generate text
        input_ids = torch.tensor([[1, 2, 3, 4]])
        sequences, attention_mask, action_mask = actor.generate(
            input_ids=input_ids,
            max_new_tokens=50,
            temperature=0.7
        )
    """
    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2: bool = False,
        bf16: bool = True,
        lora_rank: int = 0,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        target_modules: Optional[list[str]] = None,
        ds_config: Optional[dict] = None,
        device_map: Optional[dict] = None,
        packing_samples: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the ActorLanguage model.

        Sets up the model architecture, applies optimizations like LoRA and quantization,
        and configures the model for training or inference.
        """
        super().__init__()

        # ------------------------------------------------
        # 1. Directly pass in a pre-built model
        # ------------------------------------------------
        if not isinstance(pretrain_or_model, str):
            self.model: nn.Module = pretrain_or_model
            self.pretrain_or_model = pretrain_or_model.config.model_type
            self.packing_samples = packing_samples
            print("pretrain_or_model:", self.pretrain_or_model)
            return

        # ------------------------------------------------
        # 2. Need to load from a checkpoint
        # ------------------------------------------------
        self.pretrain_or_model = pretrain_or_model
        attn_impl = "flash_attention_2" if use_flash_attention_2 else "eager"

        # DeepSpeed config (must be constructed in advance for stage-3)
        if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
            _ = HfDeepSpeedConfig(ds_config)

        # ------------------------------------------------
        # 2.1 Actually load the model based on its type
        # ------------------------------------------------
        # When using FSDP with meta_init, we should not pass device_map to avoid
        # "Cannot copy out of meta tensor" error. FSDP will handle device placement.
        from_pretrained_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": attn_impl,
            "torch_dtype": torch.bfloat16 if bf16 else "auto",
        }

        # Only add device_map if we're not in a meta device context (used by FSDP)
        # Check if the default tensor creation device is meta
        try:
            test_tensor = torch.empty(1)
            is_meta_context = test_tensor.is_meta
        except:  # noqa
            is_meta_context = False

        if not is_meta_context and device_map is not None:
            from_pretrained_kwargs["device_map"] = device_map

        self.model = AutoModelForCausalLM.from_pretrained(pretrain_or_model, **from_pretrained_kwargs)

        # ------------------------------------------------
        # 2.2 LoRA
        # ------------------------------------------------
        if lora_rank > 0:
            self.model = apply_lora_configuration(
                model=self.model,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                freeze_vision_tower=True,
            )

        # Do not automatically cache during generation (consistent with related transformers issues)
        self.model.config.use_cache = False

        self.packing_samples = packing_samples
        print("pretrain_or_model:", self.pretrain_or_model)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[
        Tuple[torch.LongTensor, torch.LongTensor],
        Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor], ]:
        """
        Generate text sequences using the actor model.

        Performs text generation with various decoding strategies and returns processed sequences
        with attention masks and action masks for reinforcement learning.

        :param input_ids: Input token IDs for generation.
        :type input_ids: torch.Tensor
        :param pixel_values: Pixel values for vision-language models (currently unused).
        :type pixel_values: Optional[torch.Tensor]
        :param image_grid_thw: Image grid dimensions for vision-language models (currently unused).
        :type image_grid_thw: Optional[torch.Tensor]
        :param kwargs: Additional generation parameters including max_new_tokens, temperature, etc.

        :return: Tuple containing generated sequences, attention mask, and action mask.
        :rtype: Union[Tuple[torch.LongTensor, torch.LongTensor], Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]]  # noqa

        Example::

            input_ids = torch.tensor([[1, 2, 3]])
            sequences, attention_mask, action_mask = actor.generate(
                input_ids=input_ids,
                max_new_tokens=20,
                temperature=0.8,
                do_sample=True
            )
        """
        generate_args = dict(
            input_ids=input_ids,
            top_k=kwargs.get("top_k"),
            top_p=kwargs.get("top_p"),
            do_sample=kwargs.get("do_sample", True),
            early_stopping=kwargs.get("num_beams", 1) > 1,
            temperature=kwargs.get("temperature", 1.0),
            use_cache=True,
            num_beams=kwargs.get("num_beams", 1),
            attention_mask=kwargs.get("attention_mask"),
            eos_token_id=kwargs.get("eos_token_id"),
            pad_token_id=kwargs.get("pad_token_id"),
            min_new_tokens=kwargs.get("min_new_tokens", 1),
        )
        if kwargs.get("max_new_tokens") is not None:
            generate_args["max_new_tokens"] = kwargs["max_new_tokens"]
        if kwargs.get("max_length") is not None:
            generate_args["max_length"] = kwargs["max_length"]

        sequences = self.model.generate(**generate_args)

        eos_id = generate_args["eos_token_id"]
        pad_id = generate_args["pad_token_id"]
        return self.process_sequences(sequences, input_ids.size(1), eos_id, pad_id)

    # ==================== Sequence Post-processing ==================== #
    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        return_output: bool = False,
        packed_seq_lens: Optional[list[int]] = None,
    ):
        """
        Forward pass through the actor model.

        Computes action log probabilities for reinforcement learning training. Supports both
        regular and packed sequence processing for efficient training.

        :param sequences: Input token sequences.
        :type sequences: torch.LongTensor
        :param num_actions: Number of action tokens to extract log probabilities for.
        :type num_actions: Optional[Union[int, List[int]]]
        :param attention_mask: Attention mask for the sequences.
        :type attention_mask: Optional[torch.Tensor]
        :param pixel_values: Pixel values for vision-language models (currently unused).
        :type pixel_values: Optional[torch.Tensor]
        :param image_grid_thw: Image grid dimensions for vision-language models (currently unused).
        :type image_grid_thw: Optional[torch.Tensor]
        :param return_output: Whether to return the full model output along with action log probabilities.
        :type return_output: bool
        :param packed_seq_lens: Sequence lengths for packed samples.
        :type packed_seq_lens: Optional[List[int]]

        :return: Action log probabilities, optionally with full model output.
        :rtype: Union[torch.Tensor, Tuple[torch.Tensor, dict]]

        Example::

            sequences = torch.tensor([[1, 2, 3, 4, 5]])
            attention_mask = torch.ones_like(sequences)
            action_log_probs = actor.forward(
                sequences=sequences,
                num_actions=2,
                attention_mask=attention_mask
            )
        """
        # position_ids processing
        if not self.packing_samples:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        else:
            position_ids = reset_position_ids(attention_mask)
            attention_mask = None  # Explicitly disable in packed mode

        output = self.model(
            sequences,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        if num_actions is None:
            assert return_output, "`return_output` must be True to return logits"
            return output

        log_probs = log_probs_from_logits(output["logits"][:, :-1, :], sequences[:, 1:])

        if not self.packing_samples:
            action_log_probs = log_probs[:, -num_actions:]
        else:
            assert isinstance(num_actions, list) and len(num_actions) == len(packed_seq_lens)
            action_log_probs = []
            offset = 0
            for na, sl in zip(num_actions, packed_seq_lens):
                start, end = max(0, offset + sl - na - 1), offset + sl - 1
                action_log_probs.append(log_probs[:, start:end])
                offset += sl
            action_log_probs = torch.cat(action_log_probs, dim=1)

        return (action_log_probs, output) if return_output else action_log_probs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        """
        Enable gradient checkpointing for memory-efficient training.

        Activates gradient checkpointing to reduce memory usage during backpropagation
        at the cost of additional computation.

        :param gradient_checkpointing_kwargs: Configuration parameters for gradient checkpointing.
        :type gradient_checkpointing_kwargs: dict

        Example::

            actor.gradient_checkpointing_enable({"use_reentrant": False})
        """
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """
        Disable gradient checkpointing.

        Turns off gradient checkpointing to use standard backpropagation, which uses
        more memory but is computationally faster.

        Example::

            actor.gradient_checkpointing_disable()
        """
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        """
        Print information about trainable parameters in the model.

        Displays the number and percentage of trainable parameters, which is particularly
        useful when using techniques like LoRA that only train a subset of parameters.

        Example::

            actor.print_trainable_parameters()
            # Output: trainable params: 4,194,304 || all params: 125,000,000 || trainable%: 3.36
        """
        self.model.print_trainable_parameters()

    def process_sequences(self, sequences: torch.Tensor, input_len: int, eos_token_id: int,
                          pad_token_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Called by `trainer/fast_exp_maker.py`.

        Process generated sequences to create proper attention and action masks.

        This method post-processes the generated sequences to ensure proper handling of
        end-of-sequence tokens and creates masks needed for reinforcement learning training.
        It handles edge cases like multiple EOS tokens and ensures consistent sequence formatting.

        :param sequences: Generated token sequences
        :type sequences: torch.Tensor
        :param input_len: Length of the input prompt
        :type input_len: int
        :param eos_token_id: End-of-sequence token ID
        :type eos_token_id: int
        :param pad_token_id: Padding token ID
        :type pad_token_id: int

        :return: Tuple of processed sequences, attention mask, and action mask
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """

        # Process generated sequences to create proper attention and action masks
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)

        # For Llama3 and Qwen2 models, there are some eos_tokens in the middle of the prompt.
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
        attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)

        # in RL, state_i (current token) + action_i (next token) -> state_i+1 (next token)
        state_seq = sequences[:, input_len - 1:-1]
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        action_mask[:, 0] = 1

        return sequences, attention_mask, action_mask
