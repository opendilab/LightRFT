"""
This module provides an adapted implementation of Qwen2 attention mechanism with Ulysses sequence parallelism support.
It extends the standard transformer attention to work with sequence parallel processing, enabling more efficient
handling of long sequences by distributing them across multiple devices.

.. note::
    This implementation has been tested only on transformers versions between 4.48.0 and 4.49.0.
"""

from typing import Callable, Optional, Tuple

import torch
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.utils import logging

from lightrft.strategy.utils.parallel_utils import (
    _SeqAllToAll,
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
)

logger = logging.get_logger(__name__)


def qwen2_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Forward pass for Qwen2 attention with sequence parallelism support.

    This function implements the attention mechanism for Qwen2 model with added support for
    Ulysses sequence parallelism. It handles the projection of input hidden states into
    query, key and value tensors, applies rotary position embeddings, and processes
    the attention computation with optional sliding window attention.

    :param hidden_states: Input tensor containing token embeddings
    :type hidden_states: torch.Tensor

    :param position_embeddings: Tuple of (cos, sin) tensors for rotary position embeddings
    :type position_embeddings: Tuple[torch.Tensor, torch.Tensor]

    :param attention_mask: Optional mask to prevent attention to certain positions
    :type attention_mask: Optional[torch.Tensor]

    :param past_key_value: Optional cached key and value tensors for incremental decoding
    :type past_key_value: Optional[Cache]

    :param cache_position: Optional tensor indicating positions in the cache
    :type cache_position: Optional[torch.LongTensor]

    :param kwargs: Additional keyword arguments passed to the attention implementation

    :return: Tuple containing:
        - Output tensor after attention and projection
        - Optional attention weights if output_attentions is True
    :rtype: Tuple[torch.Tensor, Optional[torch.Tensor]]

    Note:
        This implementation is specifically adapted for transformers versions 4.48.0-4.49.0
        and includes special handling for Ulysses sequence parallelism.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    bsz, q_len, _ = hidden_states.shape
    hidden_shape = (bsz, q_len, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # AlltoAll for Ulysses
    ulysses_sp_size = get_sequence_parallel_world_size()
    sp_group = get_sequence_parallel_group()

    if ulysses_sp_size > 1:
        # (bsz, n_head, seq_len/n, head_dim) -> (bsz, n_head/n, seq_len, head_dim)
        query_states, key_states, value_states = _SeqAllToAll.apply(
            sp_group, [1, 1, 1], [2, 2, 2], query_states, key_states, value_states
        )

    full_q_len = query_states.size(2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    sliding_window = None
    if (
        self.config.use_sliding_window and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window

    from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. "
                "Falling back to eager attention. "
                "This warning can be removed using the argument `attn_implementation=eager`."
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=sliding_window,  # main diff with Llama
        **kwargs,
    )

    attn_output = attn_output.reshape(bsz, full_q_len, -1, self.head_dim).contiguous()
    # AlltoAll for Ulysses
    if ulysses_sp_size > 1:
        # (bsz, seq_len, n_head/n, head_dim) -> (bsz, seq_len/n, n_head, head_dim)
        attn_output = _SeqAllToAll.apply(sp_group, 1, 2, attn_output)
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
