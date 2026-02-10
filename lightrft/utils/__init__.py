"""
Utility modules for LightRFT.

This package contains various utility functions and classes used throughout the LightRFT project.
"""

from .logging_utils import init_logger
from .remote_rm_utils import remote_rm_fn
from .trajectory_saver import TrajectorySaver, create_trajectory_saver
from .distributed_sampler import DistributedSampler

from .processor import get_processor, reward_normalization
from .utils import (
    blending_datasets, get_tokenizer, get_tokenizer_processor_vl, print_rank_0, get_current_device, get_torch_profiler,
    ensure_video_input_available, all_gather_and_flatten, all_reduce_dict,
    # Device compatibility functions
    is_accelerator_available, device_synchronize, empty_cache, mem_get_info,
    memory_allocated, memory_summary, set_device, manual_seed_all
)

from .cli_args import add_arguments
from .timer import Timer

__all__ = [
    # logging and trajectory
    "init_logger",
    "remote_rm_fn",
    'TrajectorySaver',
    'create_trajectory_saver',

    # sampler
    "DistributedSampler",

    # processor
    "get_processor",
    "reward_normalization",

    # utils
    "blending_datasets",
    "get_tokenizer",
    "get_tokenizer_processor_vl",
    "print_rank_0",
    "get_current_device",
    "get_torch_profiler",
    "ensure_video_input_available",
    "all_gather_and_flatten",
    "all_reduce_dict",
    # Device compatibility
    "is_accelerator_available",
    "device_synchronize",
    "empty_cache",
    "mem_get_info",
    "memory_allocated",
    "memory_summary",
    "set_device",
    "manual_seed_all",

    # cli_args
    "add_arguments",

    # timer
    "Timer",
]
