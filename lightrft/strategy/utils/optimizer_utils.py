"""
PyTorch Optimization Utilities Module

This module provides utility functions for optimizing PyTorch models,
particularly focused on parameter grouping for optimizers with
customized weight decay settings. It includes support for both regular
tensors and distributed tensors (DTensor) with specialized grouping
strategies for optimal performance in distributed training scenarios.
"""

from collections import defaultdict
from typing import List, Optional

import torch
from torch.distributed.tensor import DTensor

_DEFAULT_NO_DECAY_NAME_LIST = [
    "bias",
    "layer_norm.weight",
    "layernorm.weight",
    "norm.weight",
    "ln_f.weight",
]


def get_optimizer_grouped_parameters(  # pylint: disable=W0102
    model,
    weight_decay,
    no_decay_name_list: Optional[List[str]] = None,
):
    """
    Prepare parameter groups for optimizer with weight decay control.

    Groups parameters into two groups:
    - Parameters that should have weight decay applied
    - Parameters that should not have weight decay applied (typically normalization layers and biases)

    :param model: The model whose parameters will be organized.
    :type model: torch.nn.Module
    :param weight_decay: Weight decay value to apply to applicable parameters.
    :type weight_decay: float
    :param no_decay_name_list: List of parameter name patterns that should not have weight decay.
                              If None, defaults to _DEFAULT_NO_DECAY_NAME_LIST.
    :type no_decay_name_list: Optional[List[str]]

    :return: List of parameter groups for the optimizer.
    :rtype: list

    Example::

        >>> import torch.nn as nn
        >>> model = nn.Sequential(nn.Linear(10, 10), nn.LayerNorm(10))
        >>> grouped_params = get_optimizer_grouped_parameters(model, weight_decay=0.01)
        >>> optimizer = torch.optim.AdamW(grouped_params)
    """
    if no_decay_name_list is None:
        no_decay_name_list = _DEFAULT_NO_DECAY_NAME_LIST

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def group_parameters_for_optimizer_dtensor(
    model: torch.nn.Module, weight_decay: float, no_decay_name_list: Optional[List[str]] = None
):
    """
    Groups model parameters for optimizer by weight decay, dtype, and device mesh for DTensor.

    This function creates parameter groups optimized for distributed tensor scenarios by
    considering not only weight decay patterns but also data types and device mesh
    configurations. This grouping strategy helps optimize memory usage and communication
    patterns in distributed training setups.

    :param model: The model whose parameters will be organized.
    :type model: torch.nn.Module
    :param weight_decay: Weight decay value to apply to applicable parameters.
    :type weight_decay: float
    :param no_decay_name_list: List of parameter name patterns that should not have weight decay.
                              If None, defaults to _DEFAULT_NO_DECAY_NAME_LIST.
    :type no_decay_name_list: Optional[List[str]]

    :return: Dictionary mapping group keys to parameter lists. Group keys are tuples of
             (weight_decay_value, dtype, mesh_info) where mesh_info describes the
             distributed tensor configuration.
    :rtype: defaultdict[tuple, list]

    Example::

        >>> import torch.nn as nn
        >>> model = nn.Sequential(nn.Linear(10, 10), nn.LayerNorm(10))
        >>> grouped_params = group_parameters_for_optimizer_dtensor(model, weight_decay=0.01)
        >>> # Convert to optimizer format
        >>> optimizer_groups = [{"params": params, "weight_decay": wd}
        ...                    for (wd, dtype, mesh), params in grouped_params.items()]
    """
    if no_decay_name_list is None:
        no_decay_name_list = _DEFAULT_NO_DECAY_NAME_LIST

    # Use a dict to store unique groups, keyed by (weight_decay_status, dtype, mesh_info)
    grouped_params_temp = defaultdict(list)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine weight decay application
        apply_weight_decay = not any(nd in name for nd in no_decay_name_list)
        current_weight_decay = weight_decay if apply_weight_decay else 0.0

        # Determine device mesh info
        param_mesh_info = "full_tensor"
        # Check if param is a DTensor (assuming it has _is_dtensor and device_mesh)
        if isinstance(param, DTensor):
            param_mesh_info = f"dtensor_{param.device_mesh.shape[0]}"

        # Grouping key: (weight_decay_value, param_dtype, param_mesh_info)
        group_key = (current_weight_decay, param.dtype, param_mesh_info)
        grouped_params_temp[group_key].append(param)

    return grouped_params_temp
