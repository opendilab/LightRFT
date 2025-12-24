"""
DeepSpeed Configuration and Optimization Utilities Module.

This module provides utility functions for configuring DeepSpeed for training and evaluation,
managing optimizer parameters, and handling DeepSpeed ZeRO stage 3 states. It includes
functions for creating DeepSpeed configurations with various optimization options,
organizing model parameters for optimizers with weight decay control, and
offloading/reloading DeepSpeed states to manage memory efficiently.
"""

from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus


def get_train_ds_config(  # pylint: disable=R0917
    offload,
    adam_offload=True,
    stage=2,
    bf16=True,
    max_norm=1.0,
    zpg=8,
    grad_accum_dtype=None,
    overlap_comm=False,
):
    """
    Generate a DeepSpeed configuration dictionary for training.

    :param offload: Whether to offload parameters to CPU.
    :type offload: bool
    :param adam_offload: Whether to offload Adam optimizer states to CPU.
    :type adam_offload: bool
    :param stage: ZeRO optimization stage (0, 1, 2, or 3).
    :type stage: int
    :param bf16: Whether to use bfloat16 precision.
    :type bf16: bool
    :param max_norm: Maximum norm for gradient clipping.
    :type max_norm: float
    :param zpg: ZeRO++ partition size.
    :type zpg: int
    :param grad_accum_dtype: Data type for gradient accumulation.
    :type grad_accum_dtype: str or None
    :param overlap_comm: Whether to overlap communication with computation.
    :type overlap_comm: bool

    :return: DeepSpeed configuration dictionary for training.
    :rtype: dict
    """
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": "cpu" if adam_offload else "none",
            "pin_memory": True,
        },
        "sub_group_size": "auto",
        "stage3_max_live_parameters": "auto",
        "stage3_max_reuse_distance": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "reduce_bucket_size": "auto",
        # ZeRO++
        "zero_hpz_partition_size": zpg,
        "zero_quantized_weights": False,
        "zero_quantized_gradients": False,
    }
    if overlap_comm:
        zero_opt_dict["overlap_comm"] = True
        zero_opt_dict["contiguous_gradients"] = True

    return {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": bf16,
        },
        "gradient_clipping": max_norm,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "data_types": {
            "grad_accum_dtype": grad_accum_dtype
        },
    }


def get_eval_ds_config(
    offload,
    stage=0,
    bf16=True,
):
    """
    Generate a DeepSpeed configuration dictionary for evaluation.

    :param offload: Whether to offload parameters to CPU.
    :type offload: bool
    :param stage: ZeRO optimization stage (0, 1, 2, or 3).
    :type stage: int
    :param bf16: Whether to use bfloat16 precision.
    :type bf16: bool

    :return: DeepSpeed configuration dictionary for evaluation.
    :rtype: dict
    """
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": "auto",
        "offload_param": {
            "device": "cpu" if offload else "none",
            "pin_memory": True,
        },
    }
    return {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": bf16,
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }


def _z3_params_to_fetch(param_list):
    """
    Filter parameters that need to be fetched in ZeRO stage 3.

    :param param_list: List of parameters to check.
    :type param_list: list

    :return: List of parameters that are not available and need to be fetched.
    :rtype: list
    """
    return [p for p in param_list if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE]


def offload_deepspeed_states(model, pin_memory=True, non_blocking=True):
    """
    Offload DeepSpeed optimizer states to CPU to save GPU memory.

    This function is particularly useful for ZeRO stage 3 when not using Adam optimizer offloading.
    It offloads various states to CPU, empties partition cache, and synchronizes devices.

    :param model: DeepSpeed model with optimizer.
    :type model: deepspeed.DeepSpeedEngine
    :param pin_memory: Whether to use pinned memory for offloaded states.
    :type pin_memory: bool
    :param non_blocking: Whether to perform non-blocking transfers.
    :type non_blocking: bool

    :raises NotImplementedError: If ZeRO stage is not 3.
    """
    zero_stage = model.zero_optimization_stage()  # config['zero_optimization']['stage']
    adam_offload = model.config["zero_optimization"]["offload_optimizer"]["device"] == "cpu"

    # state offloading not required when using Adam optimizer offloading
    if adam_offload:
        return

    if zero_stage != 3:
        raise NotImplementedError("Only Zero stage 3 is currently supported")

    # if zero_stage == 3 and not adam_offload:
    import torch
    from deepspeed.runtime.zero.offload_config import (
        OffloadDeviceEnum,
        OffloadStateTypeEnum,
    )

    model.optimizer.offload_states(
        include=[
            OffloadStateTypeEnum.optim_states,
            OffloadStateTypeEnum.contiguous_grad_buffer,
            OffloadStateTypeEnum.hp_params,
            # Not released yet, fixed in https://github.com/deepspeedai/DeepSpeed/pull/7050
            # OffloadStateTypeEnum.lp_grads,
            # OffloadStateTypeEnum.lp_params,
        ],
        device=OffloadDeviceEnum.cpu,
        pin_memory=pin_memory,
        non_blocking=non_blocking,
    )
    model.empty_partition_cache()
    torch.cuda.empty_cache()
    torch.distributed.barrier()
    torch.cuda.synchronize()


def reload_deepspeed_states(model, non_blocking=True):
    """
    Reload DeepSpeed optimizer states from CPU back to GPU.

    This function is used to restore states previously offloaded with offload_deepspeed_states().

    :param model: DeepSpeed model with optimizer.
    :type model: deepspeed.DeepSpeedEngine
    :param non_blocking: Whether to perform non-blocking transfers.
    :type non_blocking: bool

    :raises NotImplementedError: If ZeRO stage is not 3.
    """
    zero_stage = model.zero_optimization_stage()  # config['zero_optimization']['stage']
    adam_offload = model.config["zero_optimization"]["offload_optimizer"]["device"] == "cpu"

    # state offloading not required when using Adam optimizer offloading
    if adam_offload:
        return

    if zero_stage != 3:
        raise NotImplementedError("Only Zero stage 3 is currently supported")

    # if zero_stage == 3 and not adam_offload:
    import torch

    model.reload_states(non_blocking=non_blocking)
    torch.cuda.empty_cache()
    torch.distributed.barrier()
    torch.cuda.synchronize()
