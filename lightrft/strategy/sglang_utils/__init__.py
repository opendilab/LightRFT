"""
This module provides functionality for initializing and configuring a SGLang generation engine
for reinforcement learning with human feedback (RLHF) applications. It handles distributed
training setup, device coordination, and engine initialization with appropriate parameters.

The main component is the get_sglang_engine function which creates and returns a configured
RLGenerationEngine instance based on the provided arguments, taking into account the distributed
training environment.
"""

import datetime
import os
import sys

# Standard library imports

# Third-party library imports
import torch

# Local application imports
from lightrft.strategy.utils.distributed_util import create_sub_group
from .sglang_engine import RLGenerationEngine


def get_sglang_engine_for_rollout(args) -> RLGenerationEngine:
    """
    Initialize and configure a SGLang generation engine for reinforcement learning rollout phase.

    This function serves as a convenient wrapper around get_sglang_engine, specifically designed
    for the rollout phase of reinforcement learning training. It extracts relevant parameters
    from the provided arguments and configures the engine with appropriate settings for
    generating rollout data during RLHF training.

    :param args: Configuration arguments containing engine parameters including model path,
                 memory utilization, tensor parallelism settings, and data type preferences
    :type args: argparse.Namespace

    :return: Configured RLGenerationEngine instance ready for rollout generation
    :rtype: RLGenerationEngine

    :raises AssertionError: If PyTorch distributed is not initialized or if world size is
                           incompatible with tensor parallelism settings

    Example::

        >>> import argparse
        >>> args = argparse.Namespace()
        >>> args.pretrain = "meta-llama/Llama-2-7b-hf"
        >>> args.engine_mem_util = 0.8
        >>> args.enable_engine_sleep = True
        >>> args.engine_tp_size = 2
        >>> args.bf16 = True
        >>> engine = get_sglang_engine_for_rollout(args)
    """
    return get_sglang_engine(
        args.pretrain,
        args.engine_mem_util,
        enable_engine_sleep=args.enable_engine_sleep,
        tp_size=args.engine_tp_size,
        dtype="bfloat16" if args.bf16 else "float16",
    )


def get_sglang_engine(
    model_name_or_path: str,
    engine_mem_util: float,
    enable_engine_sleep: bool = True,
    tp_size: int = 1,
    skip_tokenizer_init: bool = True,
    dtype: str = "bfloat16",
    disable_cuda_graph: bool = False,
):
    """
    Initialize and configure a SGLang generation engine with distributed processing support.

    This function creates a RLGenerationEngine instance with proper distributed training
    configuration, including tensor parallelism setup, device coordination, and memory
    management. It handles the complex initialization process required for distributed
    inference in RLHF scenarios.

    The function automatically detects the distributed environment settings from environment
    variables and configures the engine accordingly. It sets up tensor parallel groups,
    manages GPU allocation, and initializes the engine with optimized parameters for
    high-throughput generation.

    :param model_name_or_path: Path to the model or Hugging Face model identifier
    :type model_name_or_path: str
    :param engine_mem_util: Memory utilization fraction for the engine (0.0 to 1.0)
    :type engine_mem_util: float
    :param enable_engine_sleep: Whether to enable memory saver mode that releases KV cache
                               when memory is limited
    :type enable_engine_sleep: bool
    :param tp_size: Tensor parallelism size for distributed inference
    :type tp_size: int
    :param skip_tokenizer_init: Whether to skip tokenizer initialization for faster startup
    :type skip_tokenizer_init: bool
    :param dtype: Data type for model weights and computations ("bfloat16" or "float16")
    :type dtype: str
    :param disable_cuda_graph: Whether to disable CUDA graph optimization
    :type disable_cuda_graph: bool

    :return: Configured RLGenerationEngine instance ready for distributed inference
    :rtype: RLGenerationEngine

    :raises AssertionError: If PyTorch distributed is not initialized
    :raises AssertionError: If world size is not evenly divisible by tensor parallelism size

    Example::

        >>> # Initialize engine for single GPU
        >>> engine = get_sglang_engine(
        ...     model_name_or_path="meta-llama/Llama-2-7b-hf",
        ...     engine_mem_util=0.8,
        ...     tp_size=1
        ... )

        >>> # Initialize engine with tensor parallelism
        >>> engine = get_sglang_engine(
        ...     model_name_or_path="meta-llama/Llama-2-70b-hf",
        ...     engine_mem_util=0.9,
        ...     tp_size=4,
        ...     enable_engine_sleep=False,
        ...     dtype="float16"
        ... )
    """
    assert torch.distributed.is_initialized()
    sglang_tp_group_cpu, _ = create_sub_group(tp_size, backend="gloo")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    def _log(text):
        """
        Internal logging function with timestamp and rank information.

        :param text: Message to log
        :type text: str
        """
        t = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{t}] [rank={rank}] {text}")

    _log(f'start {local_rank=} {rank=} {world_size=} {sys.argv=} {os.environ.get("CUDA_VISIBLE_DEVICES")}')

    dp_size = world_size // tp_size

    assert world_size == tp_size * dp_size

    # Clean up potentially conflicting environment variables
    for k in ["TORCHELASTIC_USE_AGENT_STORE"]:
        if k in os.environ:
            del os.environ[k]

    engine = RLGenerationEngine(
        model_path=model_name_or_path,
        mem_fraction_static=engine_mem_util,
        tp_group_cpu=sglang_tp_group_cpu,
        base_gpu_id=local_rank,
        gpu_id_step=1,
        skip_tokenizer_init=skip_tokenizer_init,
        port=40000 + local_rank,
        # enable_memory_saver can tell engine to release KV cache
        # if the memory is limited
        # You can enable this by setting enable_engine_sleep=True in shells.
        enable_memory_saver=enable_engine_sleep,
        # if you want to debug the sglang engine
        # please set the following parameters
        # Otherwise, it will make the engine run too slow
        # log_level="INFO",
        # log_requests=True,
        # log_requests_level=2,
        # max_running_requests=1,
        trust_remote_code=True,
        dtype=dtype,
        disable_cuda_graph=disable_cuda_graph,
    )

    return engine
