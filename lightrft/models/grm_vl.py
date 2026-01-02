from typing import Optional, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from transformers import AutoModelForVision2Seq

from .utils import apply_lora_configuration


class GenerativeRewardModelVL(nn.Module):
    """
    Generative reward model for reinforcement learning applications.

    This class wraps a pretrained vision-language model to serve as a generative reward model,
    which are capable of processing both textual and visual inputs, and generating textual outputs.

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
    """
    def __init__(
        self,
        pretrain_or_model: str,
        use_flash_attention_2: bool = True,
        bf16: bool = True,
        lora_rank: int = 0,
        lora_alpha: int = 0,
        lora_dropout: float = 0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        **kwargs,  # for compatibility with other models
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
                dschf = None  # noqa: F841

            self.model = AutoModelForVision2Seq.from_pretrained(
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
                    freeze_vision_tower=True,
                )

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False
        else:
            self.model = pretrain_or_model
            self.pretrain_or_model = pretrain_or_model.config.model_type
        print("pretrain_or_model: ", self.pretrain_or_model)

    def forward(
        self,
        sequences: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pixel_values: torch.Tensor = None,
        image_grid_thw: torch.Tensor = None,
        pixel_values_videos: torch.Tensor = None,
        video_grid_thw: torch.Tensor = None,
        return_outputs: bool = True,
    ) -> Union[ModelOutput, torch.Tensor]:
        """
        The forward pass takes sequences and visual content (if exists) as input and returns the model output.

        :param sequences: Input token sequences
        :type sequences: torch.LongTensor
        :param attention_mask: Attention mask for the sequences
        :type attention_mask: Optional[torch.Tensor]
        :param labels: Target labels for computing loss
        :type labels: Optional[torch.Tensor]
        :param pixel_values: Preprocessed pixel values of input images
        :type pixel_values: torch.Tensor
        :param image_grid_thw: Image grid dimensions (time, height, width)
        :type image_grid_thw: torch.Tensor
        :param pixel_values_videos: Preprocessed pixel values of input videos
        :type pixel_values_videos: torch.Tensor
        :param video_grid_thw: Video grid dimensions (time, height, width)
        :type video_grid_thw: torch.Tensor
        :param return_output: Whether to return the full model output along with log probs
        :type return_output: bool

        :return: Model output or logits based on return_outputs flag
        :rtype: Union[ModelOutput, torch.Tensor]

        Example::
            # Coumpute logits from sequences and visual inputs
            logits = reward_model(
                sequences=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                return_outputs=False
            )

            # Get full model output including loss
            outputs = reward_model(
                sequences=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                return_outputs=True
            )
        """
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        output = self.model(
            input_ids=sequences,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            labels=labels,
            output_hidden_states=False,
        )

        if return_outputs:  # default
            return output
        else:
            return output.logits

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        """
        Enable gradient checkpointing to reduce memory usage during training.

        Gradient checkpointing trades compute for memory by recomputing intermediate
        activations during the backward pass instead of storing them. This is particularly
        useful for training large vision-language models with limited GPU memory.

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
