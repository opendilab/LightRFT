"""
This module provides utilities for initializing and configuring a vLLM engine.

The module simplifies the process of creating a vLLM engine with specific configurations
for large language model inference, particularly in reinforcement learning from human feedback
(RLHF) contexts. It offers both high-level and low-level functions for engine creation,
with support for tensor parallelism, memory optimization, and multimodal capabilities.
"""

from typing import Any

from vllm import LLM


def get_vllm_engine_for_rollout(args: Any) -> LLM:
    """
    Initialize and configure a vLLM engine for reinforcement learning rollout phase.

    This function creates a vLLM engine instance with configurations provided in the args parameter,
    such as the pretrained model path, tensor parallelism size, and memory utilization settings.
    It handles multimodal configurations automatically based on the provided arguments and serves
    as a high-level wrapper around the base get_vllm_engine function.

    :param args: Configuration arguments for the vLLM engine containing model and runtime parameters.
    :type args: Any

    :return: Configured vLLM engine instance ready for rollout operations.
    :rtype: vllm.LLM

    Example::

        >>> args = argparse.Namespace()
        >>> args.pretrain = "Qwen/Qwen2.5-7B-Instruct"
        >>> args.engine_tp_size = 1
        >>> args.engine_mem_util = 0.6
        >>> args.enable_engine_sleep = True
        >>> args.bf16 = True
        >>> args.prompt_max_len = 2048
        >>> args.generate_max_len = 1024
        >>> args.text_only = False
        >>> args.limit_mm_image_per_prompt = 5
        >>>
        >>> engine = get_vllm_engine_for_rollout(args)

    Note:
        The construction of tensor-parallel (TP) group is implemented in the strategy part.
        Multimodal image and video limits are automatically configured when applicable.
    """
    kwargs = {}
    if not args.text_only:
        limit_mm_per_prompt = {}
        if hasattr(args, "limit_mm_image_per_prompt"):
            limit_mm_per_prompt["image"] = args.limit_mm_image_per_prompt
        if hasattr(args, "limit_mm_video_per_prompt"):
            limit_mm_per_prompt["video"] = args.limit_mm_video_per_prompt

        if limit_mm_per_prompt:
            kwargs["limit_mm_per_prompt"] = limit_mm_per_prompt

    vllm_engine = get_vllm_engine(
        args.pretrain,
        dtype="bfloat16" if args.bf16 else "float16",
        tp_size=args.engine_tp_size,
        mem_util=args.engine_mem_util,
        max_model_len=args.prompt_max_len + args.generate_max_len,
        enable_sleep=args.enable_engine_sleep,
        **kwargs,
    )
    return vllm_engine


def get_vllm_engine(
    pretrain_name_or_path: str,
    dtype: str = "bfloat16",
    tp_size: int = 1,
    mem_util: float = 0.5,
    max_model_len: int = 4096,
    enable_sleep: bool = True,
    **kwargs: Any
) -> LLM:
    """
    Create and configure a vLLM engine with specified parameters.

    This is the core function for initializing a vLLM engine with custom configurations.
    It sets up the engine with distributed execution capabilities, memory optimization,
    and custom worker classes for RLHF training scenarios.

    :param pretrain_name_or_path: Path or name of the pretrained model to load.
    :type pretrain_name_or_path: str
    :param dtype: Data type for model weights, either "bfloat16" or "float16". Defaults to "bfloat16".
    :type dtype: str
    :param tp_size: Tensor parallel size for distributed inference. Defaults to 1.
    :type tp_size: int
    :param mem_util: GPU memory utilization ratio (0.0 to 1.0). Defaults to 0.5.
    :type mem_util: float
    :param max_model_len: Maximum sequence length the model can handle. Defaults to 4096.
    :type max_model_len: int
    :param enable_sleep: Whether to enable sleep mode for memory efficiency. Defaults to True.
    :type enable_sleep: bool
    :param kwargs: Additional keyword arguments passed to the LLM constructor.
    :type kwargs: Any

    :return: Configured vLLM engine instance.
    :rtype: vllm.LLM

    Example::

        >>> engine = get_vllm_engine(
        ...     "Qwen/Qwen2.5-14B-Instruct",
        ...     dtype="bfloat16",
        ...     tp_size=2,
        ...     mem_util=0.8,
        ...     max_model_len=2048,
        ...     enable_sleep=True
        ... )

    Note:
        Uses external launcher for distributed execution and custom worker class
        for integration with lightrft strategy components.
    """

    vllm_engine = LLM(
        model=pretrain_name_or_path,
        dtype=dtype,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=mem_util,
        distributed_executor_backend="external_launcher",
        worker_cls="lightrft.strategy.vllm_utils.vllm_worker_wrap_no_ray.WorkerWrap",
        enable_sleep_mode=enable_sleep,
        max_model_len=max_model_len,
        # enforce_eager=True,
        **kwargs,
    )

    return vllm_engine
