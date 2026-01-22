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
        # Step 1: Remove outermost "model." prefix if present
        if name.startswith("model."):
            name = name[6:]  # Remove "model."

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
                    if self.strategy.args.text_only:
                        # for LLM
                        self.inference_engine.update_weights_from_tensor(
                            name, param.data, flush_cache=(count == num_params)
                        )
                    else:
                        # for VLM
                        # Map weight names from training model to SGLang format
                        # Training model: model.visual.xxx, model.language_model.xxx
                        # SGLang expects: visual.xxx, model.xxx (for language model), lm_head
                        sglang_name = self._map_weight_name_for_sglang(name)
                        self.inference_engine.update_weights_from_tensor(
                            sglang_name, param.data, flush_cache=(count == num_params)
                        )

    def _fsdp_v2_broadcast(self):
        """
        Broadcast model weights using PyTorch's FSDP v2.

        This method uses the state_dict approach to gather and broadcast weights
        for FSDP v2, which has a different API compared to v1. It handles DTensor
        parameters by converting them to full tensors before broadcasting.

        :raises NotImplementedError: If sglang is used as the inference engine, which doesn't support FSDP v2
        """
        model = self.actor.model
        count, num_params = 0, len(list(model.named_parameters()))
        dst_dtype = torch.bfloat16 if self.strategy.args.bf16 else torch.float16
        for name, param in model.named_parameters():
            count += 1  # empty_cache at last param
            param_on_device = param.to(get_current_device())
            if isinstance(param, DTensor):
                full_param = param_on_device.full_tensor().to(dst_dtype)
            else:
                full_param = param_on_device.to(dst_dtype)

            if self.strategy.engine_type == "vllm":
                kwargs = dict(
                    name=name,
                    dtype=full_param.dtype,
                    shape=full_param.shape,
                    weight=full_param.data,
                    empty_cache=(count == num_params),
                )
                self.inference_engine.llm_engine.model_executor.collective_rpc("update_weight", kwargs=kwargs)
            elif self.strategy.engine_type == "sglang":
                if self.strategy.args.text_only:
                    # for LLM
                    self.inference_engine.update_weights_from_tensor(
                        name, param.data, flush_cache=(count == num_params)
                    )
                else:
                    # for VLM
                    # Map weight names from training model to SGLang format
                    # Training model: model.visual.xxx, model.language_model.xxx
                    # SGLang expects: visual.xxx, model.xxx (for language model), lm_head
                    sglang_name = self._map_weight_name_for_sglang(name)
                    self.inference_engine.update_weights_from_tensor(
                        sglang_name, param.data, flush_cache=(count == num_params)
                    )

            del param_on_device
            del full_param

    def broadcast_to_engine(self):
        """
        Broadcast model weights to the inference engine.

        This method selects the appropriate broadcasting strategy based on the
        distributed training configuration (DeepSpeed, FSDP v2). It automatically
        detects whether to use DeepSpeed or FSDP broadcasting based on the strategy
        configuration.

        Example::

            # Initialize the broadcast manager
            broadcast_manager = BroadcastManager(actor_model, strategy, inference_engine)

            # Broadcast weights to inference engine
            broadcast_manager.broadcast_to_engine()

        :raises NotImplementedError: If an unsupported configuration is used
        """
        if self.strategy.args.fsdp:
            self._fsdp_v2_broadcast()
        else:
            self._deepspeed_broadcast()
