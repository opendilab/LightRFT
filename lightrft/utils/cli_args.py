"""
Command-line Argument Parser Configuration Module

This module provides functionality to configure command-line arguments for training and inference
with various distributed training frameworks like vLLM, SGL, DeepSpeed, and FSDP.
It includes arguments for model parallelism, memory utilization, sequence packing,
gradient clipping, and logging options.
"""

import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add training and inference related arguments to an ArgumentParser.

    This function configures an argument parser with options for:
    - Inference Engine settings (vLLM/SGLang)
    - Training parameters
    - FSDP (Fully Sharded Data Parallelism) configuration
    - Logging and visualization options

    :param parser: The argument parser to which arguments will be added
    :type parser: argparse.ArgumentParser

    :return: None

    Example::

        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> add_arguments(parser)
        >>> args = parser.parse_args()
    """
    # vllm/SGL settings
    parser.add_argument(
        "--engine_tp_size",
        type=int,
        default=1,
        help="Tensor parallelism size for the inference engine (e.g., vLLM, SGLang). "
        "Increase this to split the model across multiple GPUs for larger models or faster inference.",
    )
    parser.add_argument(
        "--engine_mem_util",
        type=float,
        default=0.5,
        help="Fraction of GPU memory reserved for the inference engine's KV cache (range: 0.0 to 1.0). "
        "Higher values improve throughput but may risk out-of-memory errors.",
    )
    parser.add_argument(
        "--enable_engine_sleep",
        action="store_true",
        default=True,  # This sets the default value if the flag is NOT provided
        help="Enable the inference engine to enter a sleep state when idle to reduce GPU resource consumption. "
        "This is the default behavior. Use `--disable_engine_sleep` to turn it off.",
    )
    parser.add_argument(
        "--disable_engine_sleep",
        action="store_false",
        dest="enable_engine_sleep",  # This sets the value to False when this flag is provided
        help="Disable the inference engine's sleep mode, keeping it active even when idle. "
        "Might improve response latency at the cost of higher resource usage.",
    )

    # training arguments
    parser.add_argument(
        "--packing_samples",
        action="store_true",
        default=False,
        help="[Training] Pack multiple training samples into a single sequence to improve GPU utilization. "
        "Recommended for efficient training when sequence lengths vary significantly.",
    )
    parser.add_argument(
        "--sp_size",
        type=int,
        default=1,
        help="[Training] Sequence parallelism size. Split the sequence dim across GPUs to reduce memory footprint."
        "Useful for very long sequences. Requires model support for sequence parallelism.",
    )

    # FSDP related, by default uses deepspeed, pass `--fsdp` to enable FSDP
    parser.add_argument(
        "--fsdp",
        action="store_true",
        default=False,
        help="[Training] Enable Fully Sharded Data Parallel (FSDP) for memory-efficient training. "
        "If not set, will use DeepSpeed by default.",
    )
    parser.add_argument(
        "--use_mp_opt",
        action="store_true",
        default=False,
        help="[Training/FSDP] Use a mixed precision optimizer. Keeps model parameters in bfloat16/float16 "
        "while storing optimizer states (e.g., momentum, variance) in float32 for numerical stability. "
        "Reduces memory usage with minimal impact on accuracy.",
    )
    parser.add_argument(
        "--fsdp_cpu_offload",
        action="store_true",
        default=False,
        help="[Training/FSDP] Offload optimizer states and gradients to CPU. "
        "Dramatically reduces GPU memory usage at the cost of communication overhead.",
    )

    # if log_dir is specified, LightRFT will splot generation length distribution
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory to save training logs and visualizations. If specified, additional metrics like "
        "generated sequence length distribution will be recorded and plotted.",
    )
    parser.add_argument(
        "--plot_every",
        type=int,
        default=10,
        help="Interval (in training steps) for plotting and saving the generated sequence length distribution. "
        "Only effective if `--log_dir` is set.",
    )

    # for rewards models
    parser.add_argument(
        "--rm_use_engine",
        action="store_true",
        default=False,
        help="Use the high-throughput inference engine (e.g., vLLM) for the reward model during RLHF training. "
        "Can significantly speed up reward evaluation compared to standard forward passes.",
    )
    parser.add_argument(
        "--rm_engine_cudagraph",
        action="store_true",
        default=False,
        help="Enable CUDA graphs for the reward model's inference engine. "
        "Can improve inference latency by capturing the computational graph and replaying it efficiently. "
        "Best for fixed input/output shapes. Requires `--rm_use_engine`.",
    )

    # for VLM
    # parser.add_argument("--no_shard_vit", action="store_true", default=False,
    #                     help="Do not shard ViT (FSDP) in vlm")
    parser.add_argument(
        "--mixed_mm_data",
        action="store_true",
        default=False,
        help="[Training/VLM] Indicates that the training data contains a mix of multi-modal (e.g., image-text) "
        "and pure-text data, and this mixture might be unevenly distributed across Data Parallel (DP) ranks. "
        "Enables handling for this imbalance.",
    )

    # for fused linear logprob
    parser.add_argument(
        "--fused_linear_logprob",
        action="store_true",
        default=False,
        help="[Training] Use a fused kernel for calculating log probabilities. Avoids storing intermediate logits in "
        "PyTorch's autograd graph, which can save significant memory during training, especially for large models.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=4096,
        help="[Training] When `--fused_linear_logprob` is enabled, this defines the chunk size for the calculation. "
        "Smaller chunks use less memory but might be slightly slower. Tune for your specific GPU memory constraints.",
    )
