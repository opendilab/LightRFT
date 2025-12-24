"""
This module implements a modified version of the LLaMA attention mechanism with support for Ulysses
sequence parallelism. It adapts the original transformers implementation to work with sequence
parallelism for improved performance on distributed systems.
"""

# adopted from https://github.com/volcengine/verl/blob/main/verl/models/transformers/llama.py
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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


def llama_attn_forward(  # pylint: disable=R0917
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Modified LLaMA attention forward pass with Ulysses sequence parallelism support.

    This function implements the attention mechanism for LLaMA models with added support for
    sequence parallelism. It handles the projection of input states to query/key/value,
    applies rotary position embeddings, and performs the attention computation.

    :param self: The attention module instance
    :type self: LlamaAttention

    :param hidden_states: Input tensor to compute attention on
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
    :type kwargs: dict

    :return: Tuple containing:
        - attention output tensor
        - attention weights (optional)
    :rtype: Tuple[torch.Tensor, Optional[torch.Tensor]]

    :raises Warning: When using SDPA with output_attentions=True

    .. note::
        Adapted from transformers 4.49.0 to support Ulysses sequence parallelism for transformers >= 4.48.0.

        This function has been tested only on transformers versions between 4.48.0 and 4.49.0.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.llama.modeling_llama import eager_attention_forward

    bsz, q_len, _ = hidden_states.shape
    query_states = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    # AlltoAll for Ulysses
    ulysses_sp_size = get_sequence_parallel_world_size()
    sp_group = get_sequence_parallel_group()
    if ulysses_sp_size > 1:
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
        **kwargs,
    )
    attn_output = attn_output.reshape(bsz, full_q_len, -1, self.head_dim).contiguous()
    # AlltoAll for Ulysses
    if ulysses_sp_size > 1:
        attn_output = _SeqAllToAll.apply(sp_group, 1, 2, attn_output)

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
