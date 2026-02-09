"""
Module for managing weight synchronization between training and inference engines.

This module provides functionality to broadcast model weights from training to inference engines,
supporting different distributed training strategies including DeepSpeed and FSDP (Fully Sharded
Data Parallel v2). It handles the complexities of gathering sharded
parameters and efficiently transferring them to inference engines like vllm and sglang.
"""

from typing import Any

import deepspeed
import torch
from torch.distributed.tensor import DTensor

from lightrft.utils import get_current_device


class BroadcastManager:
    """
    Manage the weight synchronization between training and inference engine.

    This class handles the broadcasting of model weights from a distributed training setup
    to inference engines. It supports different distributed training strategies including
    DeepSpeed ZeRO and PyTorch's FSDP v2.

    :param actor: The actor model containing weights to be broadcasted
    :param strategy: The training strategy object containing configuration and methods
    :param inference_engine: The inference engine (vllm or sglang) to receive the weights
    """
    def __init__(self, actor: torch.nn.Module, strategy: Any, inference_engine: Any) -> None:
        """
        Initialize the BroadcastManager with the necessary components.

        :param actor: The actor model containing weights to be broadcasted
        :param strategy: The training strategy object containing configuration and methods
        :param inference_engine: The inference engine (vllm or sglang) to receive the weights
        :type actor: torch.nn.Module
        :type strategy: object
        :type inference_engine: object
        """
        self.actor = actor
        self.strategy = strategy
        self.inference_engine = inference_engine

    def _map_weight_name_for_sglang(self, name: str) -> str:
        """
        Map weight names from training model format to SGLang format.

        Training model (Qwen2.5-VL with wrapper):
        - model.visual.xxx
        - model.language_model.embed_tokens
        - model.language_model.layers.xxx
        - model.language_model.norm
        - model.language_model.lm_head

        SGLang expects:
        - visual.xxx
        - model.embed_tokens
        - model.layers.xxx
        - model.norm
        - lm_head

        :param name: Original weight name from training model
        :return: Mapped weight name for SGLang
        """
        # Step 0: Handle PEFT/LoRA and other potential wrapping prefixes
        # PEFT models have weights like base_model.model.<original_name>
        # We recursively strip "base_model.model." or "model." prefixes until we find
        # core components like "visual" or "language_model"
        while name.startswith("base_model.model.") or name.startswith("model."):
            if name.startswith("base_model.model."):
                name = name[len("base_model.model."):]
            elif name.startswith("model."):
                # We strip "model." and let the following steps handle it.
                # If "language_model" follows, it will be added back as "model."
                # for SGLang's expectation.
                name = name[len("model."):]

        # PEFT models also rename original weights to include ".base_layer."
        # we need to strip this to match standard weight names
        name = name.replace(".base_layer.", ".")

        # Step 2: Handle language_model prefix mapping
        if name.startswith("language_model."):
            # Remove "language_model." prefix
            name = name[15:]  # Remove "language_model."

            # For lm_head, keep as is (no "model." prefix)
            if name.startswith("lm_head"):
                return name

            # For other components (embed_tokens, layers, norm), add "model." prefix
            return f"model.{name}"

        # Step 3: Return as is for other cases (e.g., visual.xxx)
        return name

    def _deepspeed_broadcast(self):
        """
        Broadcast model weights using DeepSpeed's ZeRO optimization.

        This method handles gathering sharded parameters in ZeRO-3 and broadcasts them
        to all inference engines. It processes parameters one by one to avoid memory issues.
        For ZeRO-3, it uses DeepSpeed's GatheredParameters context manager to collect
        sharded parameters before broadcasting.

        :raises NotImplementedError: If an unsupported inference engine is specified
        """
        # avoid OOM
        torch.cuda.empty_cache()
        model = self.actor.model.module
        count, num_params = 0, len(list(model.named_parameters()))
        for name, param in model.named_parameters():
            count += 1  # empty_cache at last param

            # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
            with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                kwargs = dict(
                    name=name, dtype=param.dtype, shape=shape, weight=param.data, empty_cache=(count == num_params)
                )
                if self.strategy.engine_type == "vllm":
                    self.inference_engine.llm_engine.model_executor.collective_rpc("update_weight", kwargs=kwargs)
                elif self.strategy.engine_type == "sglang":
                    if ".lora_" in name:
                        continue
                    
                    sglang_name = self._map_weight_name_for_sglang(name)
                    self.inference_engine.update_weights_from_tensor(
                        sglang_name, param.data, flush_cache=(count == num_params)
                    )

    def _fsdp_v2_broadcast(self):
        """
        Broadcast model weights using PyTorch's FSDP v2.

        Specialized for LoRA/PEFT:
        Instead of calling merge_adapter() which fails on FSDP DTensors,
        we manually gather base and lora weights and merge them on the fly.
        """
        model = self.actor.model
        param_dict = dict(model.named_parameters())
        count, num_params = 0, len(param_dict)
        dst_dtype = torch.bfloat16 if self.strategy.args.bf16 else torch.float16

        # Get PEFT config for scaling
        is_peft = hasattr(model, "peft_config")
        lora_config = model.peft_config.get("default") if is_peft else None
        scaling = lora_config.lora_alpha / lora_config.r if lora_config else 1.0

        for name, param in param_dict.items():
            count += 1
            
            # Skip LoRA adapters directly, they will be merged when processing base_layer
            if ".lora_" in name:
                continue
            
            # Identify if this is a PEFT base layer
            effective_name = name
            full_weight = None
            
            if ".base_layer.weight" in name:
                # This is a LoRA-enabled layer
                prefix = name.replace(".base_layer.weight", "")
                lora_a_name = f"{prefix}.lora_A.default.weight"
                lora_b_name = f"{prefix}.lora_B.default.weight"
                
                # Gather Base, LoRA A, and LoRA B
                w_base = param.to(get_current_device()).full_tensor().to(torch.float32)
                w_a = param_dict[lora_a_name].to(get_current_device()).full_tensor().to(torch.float32)
                w_b = param_dict[lora_b_name].to(get_current_device()).full_tensor().to(torch.float32)
                
                # Merge: W = W + scale * (B @ A)
                full_weight = (w_base + scaling * (w_b @ w_a)).to(dst_dtype)
                
                # Clean up intermediate huge gathered tensors
                del w_base, w_a, w_b
            else:
                # Normal layer (e.g. vision tower or non-lora layer)
                param_on_device = param.to(get_current_device())
                if isinstance(param, DTensor):
                    full_weight = param_on_device.full_tensor().to(dst_dtype)
                else:
                    full_weight = param_on_device.to(dst_dtype)
                del param_on_device

            # Broadcast to engine
            if self.strategy.engine_type == "vllm":
                vllm_name = self._map_weight_name_for_sglang(effective_name)
                kwargs = dict(
                    name=vllm_name,
                    dtype=full_weight.dtype,
                    shape=full_weight.shape,
                    weight=full_weight.data,
                    empty_cache=(count == num_params),
                )
                self.inference_engine.llm_engine.model_executor.collective_rpc("update_weight", kwargs=kwargs)
            elif self.strategy.engine_type == "sglang":
                sglang_name = self._map_weight_name_for_sglang(effective_name)
                self.inference_engine.update_weights_from_tensor(
                    sglang_name, full_weight.data, flush_cache=(count == num_params)
                )

            del full_weight

    def broadcast_to_engine(self):
        """
        Broadcast model weights to the inference engine.

        This method selects the appropriate broadcasting strategy based on the
        distributed training configuration (DeepSpeed, FSDP v2).
        """
        if self.strategy.args.fsdp:
            # FSDP handles merging manually inside _fsdp_v2_broadcast
            self._fsdp_v2_broadcast()
        else:
            # DeepSpeed path
            is_peft = hasattr(self.actor.model, "merge_adapter")
            if is_peft:
                self.strategy.print("Merging LoRA adapters for weight synchronization...")
                self.actor.model.merge_adapter()
            
            try:
                self._deepspeed_broadcast()
            finally:
                if is_peft:
                    self.strategy.print("Unmerging LoRA adapters after synchronization...")
                    self.actor.model.unmerge_adapter()
                
        self.strategy.print("Finished weight broadcasting to inference engine.")
