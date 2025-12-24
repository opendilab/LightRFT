"""
This module provides monkey patching functionality for attention mechanisms in LLaMA and Qwen2 models.
It allows replacing the original attention forward methods with custom implementations for better
performance or different behavior.

The module supports patching both LlamaAttention and Qwen2Attention classes by replacing their
forward methods with custom implementations defined in separate modules.
"""

from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

from .llama import llama_attn_forward
from .qwen import qwen2_attn_forward


def apply_monkey_patch_to_llama():
    """
    Apply monkey patch to LlamaAttention class by replacing its forward method.

    This function replaces the original forward method of LlamaAttention with a custom
    implementation defined in llama_attn_forward. This can be used to modify the attention
    mechanism's behavior or improve its performance.

    :return: None
    """
    LlamaAttention.forward = llama_attn_forward


def apply_monkey_patch_to_qwen2():
    """
    Apply monkey patch to Qwen2Attention class by replacing its forward method.

    This function replaces the original forward method of Qwen2Attention with a custom
    implementation defined in qwen2_attn_forward. This can be used to modify the attention
    mechanism's behavior or improve its performance.

    :return: None
    """
    Qwen2Attention.forward = qwen2_attn_forward


MONKEY_PATCH_FUNC = {
    "llama": apply_monkey_patch_to_llama,
    "qwen2": apply_monkey_patch_to_qwen2,
}
