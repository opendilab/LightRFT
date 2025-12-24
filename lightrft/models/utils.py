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

from typing import List, Optional, Union, Tuple

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model


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
                from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

                flashattn_available = True
            except ImportError:
                logging.warning("Failed to import cross_entropy_loss from flash_attn")
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


def unpacking_samples(values: torch.Tensor, packed_seqlens: list[int]):
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


def pad_to_length(tensor, length, pad_value, dim=-1):
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
    model, input0_ids, input0_mask, input1_ids, input1_mask, input0_img_pixels, input0_img_grid_thws, input1_img_pixels,
    input1_img_grid_thws, input0_video_pixels, input0_video_grid_thws, input1_video_pixels, input1_video_grid_thws,
    pad_token_id: int
):
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
