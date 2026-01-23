"""
Strategy factory module for lightrft.

This module provides functionality to create appropriate distributed training strategies
based on configuration arguments. It supports three types of strategies:

1. DeepSpeed - For training with the DeepSpeed library
2. FSDP V2 - For training with PyTorch's Fully Sharded Data Parallel (newer version)

The module automatically selects the appropriate strategy based on the provided arguments.
"""

from typing import Any, Union

from lightrft.strategy.deepspeed.deepspeed import DeepspeedStrategy
from lightrft.strategy.fsdp.fsdpv2 import FSDPV2Strategy
import torch.nn as nn
from lightrft.strategy.config import StrategyConfig


# TODO: simplify the strategy creation
def get_strategy(args: Any) -> Union['FSDPV2Strategy', 'DeepspeedStrategy']:
    """
    Create and return the appropriate training strategy based on configuration arguments.

    This function examines the provided arguments and instantiates either a DeepSpeed,
    or FSDP V2 strategy with the appropriate parameters.

    :param args: Configuration arguments containing strategy selection flags and parameters
    :type args: Any (usually argparse.Namespace)

    :return: An instantiated strategy object
    :rtype: Union[FSDPV2Strategy, DeepspeedStrategy]

    Example::

        >>> parser = ArgumentParser()
        >>> parser.add_argument("--fsdp", action="store_true")
        >>> parser.add_argument("--seed", type=int, default=42)
        >>> args = parser.parse_args()
        >>> strategy = get_strategy(args)
    """
    config = StrategyConfig.from_args(args)

    if config.fsdp:
        strategy = FSDPV2Strategy(
            seed=config.seed,
            max_norm=config.max_norm,
            micro_train_batch_size=config.micro_train_batch_size,
            train_batch_size=config.train_batch_size,
            bf16=config.bf16,
            args=args,  # Keep args for backward compatibility
        )
        return strategy

    else:
        strategy = DeepspeedStrategy(
            seed=config.seed,
            max_norm=config.max_norm,
            micro_train_batch_size=config.micro_train_batch_size,
            train_batch_size=config.train_batch_size,
            zero_stage=config.zero_stage,
            bf16=config.bf16,
            args=args,  # Keep args for backward compatibility
        )
        return strategy


def is_engine(model: nn.Module) -> bool:
    """
    Check if the model is a rollout engine (vLLM/SGLang).

    :param model: The model to check
    :type model: nn.Module

    :return: True if the model is a rollout engine, False otherwise
    :rtype: bool
    """
    return hasattr(model, "wake_up")
