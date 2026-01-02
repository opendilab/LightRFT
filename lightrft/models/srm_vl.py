from typing import Optional, Dict, List

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from .utils import apply_lora_configuration, AttentionPooling


class ScalarRewardModelVL(nn.Module):
    """
    Scalar Reward Model for reinforcement learning applications with vision-language backbones.

    This class wraps around a pretrained vision-language model and adds reward heads to produce scalar scores.
    Reward heads are feed-forward networks that take the hidden states from a specific layer of the backbone
    model and output scalar values.

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
    :type ds_config: Optional[Dict]
    :param device_map: Device mapping for loading the model onto specific devices
    :type device_map: Optional[Dict]
    :param pooling_method: Pooling method for aggregating hidden states ('attn' or 'last').
                           Default to 'attn', which delivers better performance.
    :type pooling_method: str
    :param probing_layer: Index of the layer from which to extract hidden states for reward heads.
                          Default to -1, which means the last layer.
    :type probing_layer: int
    :param scale_for_train: Whether to scale the scores for training. Default to True.
                            We recommend enabling this for better performance.
    :type scale_for_train: bool
    :param head_types: List of head types for scoring (e.g., ["preference", "alignment"]). Default to ["preference"].
                       Must be consistent with the training data annotations.
    :type head_types: list[str]
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
        pooling_method: str = 'attn',
        scale_for_train: bool = True,
        probing_layer: int = -1,
        head_types: List[str] = ["preference"],
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
                dschf = None  # noqa: F841

            self.model = AutoModel.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
            )

            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                target_modules = apply_lora_configuration(
                    model=self.model,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules,
                    freeze_vision_tower=True,
                )

            # Get the model configuration
            try:
                num_hidden_layers = self.model.config.num_hidden_layers
                hidden_size = self.model.config.hidden_size
            except AttributeError:
                # Special handling for Qwen3-VL
                num_hidden_layers = self.model.config.text_config.num_hidden_layers
                hidden_size = self.model.config.text_config.hidden_size
            print(f"Using backbone model type: {self.model.config.model_type}")
            print("  num_hidden_layers: ", num_hidden_layers)
            print("  hidden_size: ", hidden_size)

            if probing_layer < 0:
                probing_layer = num_hidden_layers + probing_layer
            else:
                probing_layer = probing_layer
            if probing_layer < 0 or probing_layer >= num_hidden_layers:
                raise ValueError(
                    f"reward_layer_index {probing_layer} is out of range for num_hidden_layers {num_hidden_layers}"
                )
            self.probing_layer = probing_layer

            # Extra reward heads configuration
            self.scale_for_train = scale_for_train
            self.pooling_method = pooling_method
            self.head_types = head_types
            for head_type in head_types:  # e.g., "alignment", "coherence", "preference"
                head = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.RMSNorm(hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, 1, bias=False),
                    nn.Sigmoid(),
                )
                head.to(torch.bfloat16)
                head.cuda()
                setattr(self, f"{head_type}_head", head)

                if self.scale_for_train:
                    logit_scale = nn.Parameter(torch.full((1, ), np.log(1 / 0.07)))
                    logit_scale.to(torch.bfloat16)
                    logit_scale.cuda()
                    setattr(self, f"{head_type}_logit_scale", logit_scale)

                if self.pooling_method == 'attn':
                    attnpool = AttentionPooling(hidden_size)
                    attnpool.to(torch.bfloat16)
                    attnpool.cuda()
                    setattr(self, f"{head_type}_attnpool", attnpool)

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.
            self.model.config.use_cache = False

        else:
            self.model = pretrain_or_model
            self.pretrain_or_model = pretrain_or_model.config.model_type
        print("pretrain_or_model: ", self.pretrain_or_model)

    def forward(
        self,
        sequences: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: torch.Tensor = None,
        image_grid_thw: torch.Tensor = None,
        pixel_values_videos: torch.Tensor = None,
        video_grid_thw: torch.Tensor = None,
    ) -> Dict:
        """
        The forward takes sequences and visual content (if exists) as input and returns a dictionary
        containing reward scores of different heads.

        :param sequences: Input token sequences
        :type sequences: torch.LongTensor
        :param attention_mask: Attention mask for the sequences
        :type attention_mask: Optional[torch.Tensor]
        :param pixel_values: Preprocessed pixel values of input images
        :type pixel_values: torch.Tensor
        :param image_grid_thw: Image grid dimensions (time, height, width)
        :type image_grid_thw: torch.Tensor
        :param pixel_values_videos: Preprocessed pixel values of input videos
        :type pixel_values_videos: torch.Tensor
        :param video_grid_thw: Video grid dimensions (time, height, width)
        :type video_grid_thw: torch.Tensor

        :return: A dictionary containing reward scores from different heads
        :rtype: Dict

        Example::
            # Compute reward scores from sequences and visual inputs
            # Suppose `reward_model` has two heads: "preference" and "alignment"
            scores = reward_model(
                sequences=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )
            preference_score = scores["preference"]
            alignment_score = scores["alignment"]
        """
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        output = self.model(
            sequences,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            output_hidden_states=True
        )

        # Extract hidden states and pass through reward heads
        hidden_states = output.hidden_states[self.probing_layer]
        hidden_states = hidden_states.cuda()
        scores = {}
        if self.pooling_method == 'attn':
            for head_type in self.head_types:
                new_hidden_states = getattr(self, f"{head_type}_attnpool")(hidden_states)
                scores[head_type] = getattr(self, f"{head_type}_head")(new_hidden_states)
        elif self.pooling_method == 'last':
            for head_type in self.head_types:
                new_hidden_states = hidden_states[:, -1, :]  # only use the last token
                scores[head_type] = getattr(self, f"{head_type}_head")(new_hidden_states)
        else:
            raise ValueError(f"Invalid pooling method: {self.pooling_method}")

        if self.training and self.scale_for_train:
            for head_type in self.head_types:
                scores[head_type] = scores[head_type] * torch.exp(getattr(self, f"{head_type}_logit_scale")
                                                                  ).to(scores[head_type].device)
        return scores

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
