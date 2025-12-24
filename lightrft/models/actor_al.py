"""
Audio Language Model Actor Module for Reinforcement Learning.

This module provides the ActorAL class, which implements an actor model specifically designed
for audio-language tasks in reinforcement learning scenarios. The actor is responsible for
generating actions (text sequences) based on audio inputs and textual prompts.

The module supports various optimization techniques including:
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- Flash Attention 2.0 for improved performance
- DeepSpeed integration for distributed training
- Sample packing for efficient batch processing

Key Features:
- Multi-modal input processing (text + audio)
- Flexible model loading from pretrained checkpoints
- Support for various audio-language model architectures
- Gradient checkpointing for memory optimization
- MoE (Mixture of Experts) model support
"""

from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import Qwen2AudioForConditionalGeneration
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from .utils import apply_lora_configuration, log_probs_from_logits, reset_position_ids


class ActorAL(nn.Module):
    """
    Audio Language Model actor for reinforcement learning applications.

    This class serves as a foundation for implementing audio-language actor models in RL,
    which are responsible for generating text sequences (actions) based on both audio and
    textual inputs. The model supports various optimization techniques including LoRA
    adaptation, quantization, and distributed training.

    The actor model can be initialized either from a pretrained model path or from an
    existing model instance, providing flexibility in model deployment scenarios.

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
    :param ds_config: Configuration for DeepSpeed distributed training
    :type ds_config: Optional[dict]
    :param device_map: Device mapping for loading the model onto specific devices
    :type device_map: Optional[dict]
    :param packing_samples: Whether to pack samples during training for efficiency
    :type packing_samples: bool

    Example::

        # Initialize with a pretrained model path
        actor = ActorAL(
            pretrain_or_model="Qwen/Qwen2-Audio-7B-Instruct",
            use_flash_attention_2=True,
            lora_rank=16,
            lora_alpha=32
        )

        # Generate responses
        sequences, attention_mask, action_mask = actor.generate(
            input_ids=input_tensor,
            audio_values=audio_tensor,
            max_new_tokens=100
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
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(pretrain_or_model, str):
            self.pretrain_or_model = pretrain_or_model
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            # Load Qwen2Audio model
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
            )

            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                self.model = apply_lora_configuration(
                    model=self.model,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules,
                    freeze_vision_tower=False,  # No vision tower for audio models
                )

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False

            # packing samples using Flash Attention 2
            self.packing_samples = packing_samples
        else:
            self.model = pretrain_or_model
            self.pretrain_or_model = pretrain_or_model.config.model_type
        print("pretrain_or_model: ", self.pretrain_or_model)

    @torch.no_grad()
    def generate(
        self, input_ids: torch.Tensor, audio_values: torch.Tensor, **kwargs
    ) -> Union[Tuple[torch.LongTensor, torch.LongTensor], Tuple[torch.LongTensor, torch.LongTensor,
                                                                torch.BoolTensor], ]:
        """
        Generate text sequences based on input text and audio information.

        This method performs text generation conditioned on both textual prompts and audio inputs.
        It handles the generation process with various sampling strategies and returns the generated
        sequences along with attention masks and action masks for RL training.

        :param input_ids: Input token IDs representing the text prompt
        :type input_ids: torch.Tensor
        :param audio_values: Preprocessed audio values of input audio
        :type audio_values: torch.Tensor
        :param kwargs: Additional generation parameters (top_k, top_p, temperature, etc.)
        :type kwargs: dict

        :return: Tuple containing generated sequences, attention mask, and action mask
        :rtype: Union[Tuple[torch.LongTensor, torch.LongTensor], Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]]  # noqa

        Example::

            sequences, attention_mask, action_mask = actor.generate(
                input_ids=torch.tensor([[1, 2, 3]]),
                audio_values=audio_tensor,
                max_new_tokens=50,
                temperature=0.8,
                do_sample=True
            )
        """
        generate_args = {
            "input_ids": input_ids,
            "audio_values": audio_values,
            "top_k": kwargs.get("top_k", None),
            "top_p": kwargs.get("top_p", None),
            "do_sample": kwargs.get("do_sample", True),
            "early_stopping": kwargs.get("num_beams", 1) > 1,
            "temperature": kwargs.get("temperature", 1),
            "use_cache": True,
            "num_beams": kwargs.get("num_beams", 1),
            "attention_mask": kwargs.get("attention_mask"),
            "eos_token_id": kwargs.get("eos_token_id"),
            "pad_token_id": kwargs.get("pad_token_id"),
            "min_new_tokens": kwargs.get("min_new_tokens", 1),
        }

        if kwargs.get("max_new_tokens", None):
            generate_args["max_new_tokens"] = kwargs.get("max_new_tokens")
        if kwargs.get("max_length", None):
            generate_args["max_length"] = kwargs.get("max_length")

        # Call generate
        sequences = self.model.generate(**generate_args)

        # Prepare mask tensor
        eos_token_id = generate_args["eos_token_id"]
        pad_token_id = generate_args["pad_token_id"]

        # Process generated sequences to create proper attention and action masks
        input_len = input_ids.size(1)
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

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio_values: torch.Tensor = None,
        return_output=False,
        ring_attn_group: Optional[dist.ProcessGroup] = None,
        packed_seq_lens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """
        Forward pass to compute action log probabilities for reinforcement learning.

        This method processes input sequences and audio information to compute log probabilities
        of actions (tokens) for RL training. It supports both standard and packed sequence formats
        and can return either just the action log probabilities or the full model output.

        :param sequences: Input token sequences
        :type sequences: torch.LongTensor
        :param num_actions: Number of action tokens to extract log probs for
        :type num_actions: Optional[Union[int, list[int]]]
        :param attention_mask: Attention mask for the sequences
        :type attention_mask: Optional[torch.Tensor]
        :param audio_values: Preprocessed audio values of input audio
        :type audio_values: torch.Tensor
        :param return_output: Whether to return the full model output along with log probs
        :type return_output: bool
        :param ring_attn_group: Process group for ring attention (distributed training)
        :type ring_attn_group: Optional[dist.ProcessGroup]
        :param packed_seq_lens: Sequence lengths for packed samples
        :type packed_seq_lens: Optional[list[int]]

        :return: Action log probabilities or tuple of (action_log_probs, output) if return_output=True
        :rtype: torch.Tensor

        Example::

            # Compute action log probabilities for RL training
            log_probs = actor(
                sequences=token_sequences,
                num_actions=10,
                audio_values=audio_tensor
            )

            # Get both log probs and full output
            log_probs, output = actor(
                sequences=token_sequences,
                num_actions=10,
                audio_values=audio_tensor,
                return_output=True
            )
        """
        if not self.packing_samples:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        else:
            # convert attention_mask to position_ids
            position_ids = reset_position_ids(attention_mask)
            # explicitly ignore attention_mask for packing_samples
            attention_mask = None

        # Qwen2Audio model expects different parameter names
        output = self.model(
            input_ids=sequences,
            attention_mask=attention_mask,
            position_ids=position_ids,
            audio_values=audio_values,
        )

        if num_actions is None:  # default
            assert return_output
            return output

        log_probs = log_probs_from_logits(output["logits"][:, :-1, :], sequences[:, 1:])

        if not self.packing_samples:
            action_log_probs = log_probs[:, -num_actions:]
        else:
            assert isinstance(num_actions, list) and len(num_actions) == len(packed_seq_lens)
            action_log_probs = []
            offset = 0
            for num_action, seq_len in zip(num_actions, packed_seq_lens):
                start, end = max(0, offset + seq_len - num_action - 1), offset + seq_len - 1
                action_log_probs.append(log_probs[:, start:end])
                offset += seq_len
            action_log_probs = torch.cat(action_log_probs, dim=1)

        if return_output:
            return (action_log_probs, output)
        else:
            return action_log_probs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        """
        Enable gradient checkpointing to reduce memory usage during training.

        Gradient checkpointing trades compute for memory by recomputing intermediate
        activations during the backward pass instead of storing them. This is particularly
        useful for training large audio-language models with limited GPU memory.

        :param gradient_checkpointing_kwargs: Additional arguments for gradient checkpointing
        :type gradient_checkpointing_kwargs: dict

        Example::

            # Enable gradient checkpointing with default settings
            actor.gradient_checkpointing_enable()

            # Enable with custom settings
            actor.gradient_checkpointing_enable({"use_reentrant": True})
        """
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """
        Disable gradient checkpointing to use normal forward/backward computation.

        This method restores the default behavior where all intermediate activations
        are stored during the forward pass for use in the backward pass. This increases
        memory usage but reduces computation time.

        Example::

            # Disable gradient checkpointing
            actor.gradient_checkpointing_disable()
        """
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        """
        Print information about trainable parameters in the model.

        This method displays the number and percentage of trainable parameters,
        which is particularly useful when using parameter-efficient methods like LoRA.
        It helps monitor the efficiency of the fine-tuning approach.

        Example::

            # Print trainable parameter statistics
            actor.print_trainable_parameters()
            # Output: trainable params: 4,194,304 || all params: 7,241,732,096 || trainable%: 0.058
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
