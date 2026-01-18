"""
Module for managing weight synchronization between training and inference engines.

This module provides functionality to broadcast model weights from training to inference engines,
supporting different distributed training strategies including DeepSpeed and FSDP (Fully Sharded
Data Parallel v2). It handles the complexities of gathering sharded
parameters and efficiently transferring them to inference engines like vllm and sglang.
"""

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
    def __init__(self, actor, strategy, inference_engine) -> None:
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
                    self.inference_engine.update_weights_from_tensor(
                        name, param.data, flush_cache=(count == num_params)
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
                self.inference_engine.update_weights_from_tensor(
                    name, full_param.data, flush_cache=(count == num_params)
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
