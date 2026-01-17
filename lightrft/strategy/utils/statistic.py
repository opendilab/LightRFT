"""
Utilities for analyzing and visualizing the length distribution of generated outputs.

This module provides functionality for collecting, analyzing, and visualizing the length
distribution of generated outputs from language models. It includes tools for gathering
output lengths across distributed processes, computing statistics like percentiles,
and creating visualizations using matplotlib and TensorBoard.

The main components are:
- GenLenAnalyser: A class for continuous monitoring and visualization of generation lengths
- Helper functions for collecting and analyzing output lengths in distributed environments
"""

from typing import List, Dict, Any, Optional
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter


class GenLenAnalyser:
    """
    Analyzer for tracking and visualizing the length distribution of generated outputs.

    This class collects length statistics of generated outputs over time, computes
    various metrics, and can visualize the distributions using matplotlib and TensorBoard.
    It is designed to work in distributed training environments and provides continuous
    monitoring capabilities with configurable plotting intervals.

    :param engine_dp_group: The distributed process group for communication
    :type engine_dp_group: torch.distributed.ProcessGroup

    :param plot_every: How often to plot the distribution (in steps), set to 0 to disable plotting
    :type plot_every: int

    :param percentiles: List of percentiles to compute for the length distribution
    :type percentiles: list

    :param plot_out_dir: Directory to save plots and TensorBoard logs, if None no plots are saved
    :type plot_out_dir: str or None

    Example::

        >>> import torch.distributed as dist
        >>> # Initialize distributed process group
        >>> analyzer = GenLenAnalyser(
        ...     engine_dp_group=dist.group.WORLD,
        ...     plot_every=10,
        ...     percentiles=[25, 50, 75, 90],
        ...     plot_out_dir="./output_analysis"
        ... )
        >>> # Use during training loop
        >>> stats = analyzer.collect(generation_outputs, step=100, is_rank_0=True)
    """
    def __init__(
        self,
        engine_dp_group: dist.ProcessGroup,
        plot_every: int = 2,
        percentiles: List[int] = [50, 80],
        plot_out_dir: Optional[str] = None
    ) -> None:
        self.engine_dp_group = engine_dp_group

        self.plot_out_dir = plot_out_dir
        self.percentiles = percentiles
        self.plot_every = plot_every

        self.hist_data = {}

        if plot_every > 0 and plot_out_dir is not None:
            os.makedirs(plot_out_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=plot_out_dir)

            print(f"GenLenAnalyser is initialized and will log to {plot_out_dir} every {self.plot_every}")

    def collect(self, gen_outputs: List[Dict[str, Any]], cur_step: int, is_rank_0: bool) -> Optional[Dict[str, Any]]:
        """
        Collect and analyze generation length data at the current step.

        This method gathers output lengths from all processes, computes statistics,
        and optionally creates visualizations if conditions are met. The collection
        happens at intervals specified by plot_every parameter.

        :param gen_outputs: List of generation outputs to analyze, each containing 'output_token_ids'
        :type gen_outputs: List[Dict[str, Any]]

        :param cur_step: Current training/generation step
        :type cur_step: int

        :param is_rank_0: Whether the current process is the main process (rank 0)
        :type is_rank_0: bool

        :return: Dictionary containing length statistics or None if collection is skipped
        :rtype: dict or None

        Example::

            >>> gen_outputs = [
            ...     {"output_token_ids": [1, 2, 3, 4, 5]},
            ...     {"output_token_ids": [1, 2, 3]}
            ... ]
            >>> stats = analyzer.collect(gen_outputs, cur_step=50, is_rank_0=True)
            >>> if stats:
            ...     print(f"Mean length: {stats['mean_length']}")
        """
        if self.plot_every > 0 and cur_step % self.plot_every != 0:
            return None
        local_out_lens = collect_local_output_lengths(gen_outputs)
        glb_output_lens = gather_all_lengths(local_out_lens, self.engine_dp_group)

        self.hist_data[cur_step] = glb_output_lens

        if self.plot_out_dir is not None and is_rank_0:

            for step, vals in self.hist_data.items():
                self.tb_writer.add_histogram(
                    "VLLM GenerateOutputLength Distribution",
                    np.asarray(vals, dtype="int"),
                    step,
                    bins="auto",
                    max_bins=50,
                )
            plot_out_dir = f"{self.plot_out_dir}/gen_len_step_{cur_step}.png"

            plt.figure(figsize=(10, 6))
            plt.xlabel("GenerateOutputLength")
            plt.ylabel("Frequency")
            plt.title("VLLM GenerateOutputLength Distribution")
            plt.grid(True, alpha=0.3)
            plt.legend()

            plot_data = self.hist_data
            colors = plt.cm.viridis(range(len(plot_data)))

            for (step, data), color in zip(plot_data.items(), colors):
                plt.hist(data, label=step, color=color)
            plt.savefig(plot_out_dir, bbox_inches="tight")

        infos = analyze_output_lengths(glb_output_lens, self.percentiles)
        return infos


def analyse_output_lengths(
    gen_outputs: List[Dict[str, Any]],
    engine_dp_group: dist.ProcessGroup,
    percentiles: List[int] = [50, 80],
    plot_out_dir: Optional[str] = None,
    prefix: str = ""
) -> Dict[str, Any]:
    """
    Analyze the length distribution of generated outputs.

    This is a convenience function that collects local output lengths, gathers them
    across all processes, and computes statistics. It provides a one-time analysis
    without the continuous monitoring capabilities of GenLenAnalyser.

    :param gen_outputs: List of generation outputs to analyze, each containing 'output_token_ids'
    :type gen_outputs: list

    :param engine_dp_group: The distributed process group for communication
    :type engine_dp_group: torch.distributed.ProcessGroup

    :param percentiles: List of percentiles to compute for the length distribution
    :type percentiles: list

    :param plot_out_dir: Directory to save plots, if None no plots are saved
    :type plot_out_dir: str or None

    :param prefix: Prefix for plot filenames
    :type prefix: str

    :return: Dictionary containing length statistics
    :rtype: dict

    Example::

        >>> gen_outputs = [
        ...     {"output_token_ids": [1, 2, 3, 4, 5, 6]},
        ...     {"output_token_ids": [1, 2, 3]}
        ... ]
        >>> stats = analyse_output_lengths(
        ...     gen_outputs,
        ...     engine_dp_group=dist.group.WORLD,
        ...     percentiles=[25, 50, 75]
        ... )
        >>> print(f"Median length: {stats['median_length']}")
    """
    local_out_lens = collect_local_output_lengths(gen_outputs)
    glb_output_lens = gather_all_lengths(local_out_lens, engine_dp_group)
    analyse_info = analyze_output_lengths(glb_output_lens, percentiles, plot_out_dir)
    return analyse_info


def collect_local_output_lengths(outputs: List[Dict[str, Any]]) -> List[int]:
    """
    Collect the lengths of generated outputs from the local process.

    This function extracts the length of each output by counting the tokens
    in the 'output_token_ids' field of each output dictionary.

    :param outputs: List of generation outputs, each containing 'output_token_ids'
    :type outputs: list

    :return: List of output lengths corresponding to each input output
    :rtype: list

    Example::

        >>> outputs = [
        ...     {"output_token_ids": [1, 2, 3, 4, 5]},
        ...     {"output_token_ids": [10, 20]},
        ...     {"output_token_ids": [100, 200, 300]}
        ... ]
        >>> lengths = collect_local_output_lengths(outputs)
        >>> print(lengths)  # [5, 2, 3]
    """
    output_lengths = []

    for i, output in enumerate(outputs):
        # This key is set in strategy.gather_and_generate
        output_len = len(output["output_token_ids"])
        output_lengths.append(output_len)

    return output_lengths


def gather_all_lengths(local_lengths: List[int], group: dist.ProcessGroup) -> List[int]:
    """
    Gather output lengths from all processes in the distributed group.

    This function uses PyTorch's distributed communication to collect length
    data from all processes in the specified group, enabling global analysis
    of generation length distributions across the entire distributed system.

    :param local_lengths: List of output lengths from the local process
    :type local_lengths: list

    :param group: The distributed process group for communication
    :type group: torch.distributed.ProcessGroup

    :return: Combined list of output lengths from all processes
    :rtype: list

    Example::

        >>> # Assuming distributed environment is set up
        >>> local_lengths = [5, 3, 7]
        >>> all_lengths = gather_all_lengths(local_lengths, dist.group.WORLD)
        >>> # all_lengths now contains lengths from all processes
    """
    local_lengths_tensor = torch.tensor(local_lengths, dtype=torch.int64, device=torch.cuda.current_device())

    world_size = dist.get_world_size(group=group)

    gathered_lengths = [torch.zeros_like(local_lengths_tensor) for _ in range(world_size)]

    dist.all_gather(gathered_lengths, local_lengths_tensor, group=group)

    all_lengths = []
    for gathered_len in gathered_lengths:
        all_lengths.extend(gathered_len.tolist())
    return all_lengths


def analyze_output_lengths(all_lengths: List[int], percentiles: List[int]) -> Dict[str, Any]:
    """
    Analyze the distribution of output lengths and compute statistics.

    This function computes comprehensive statistics about the length distribution,
    including basic statistics (min, max, mean, median) and user-specified percentiles.
    The results provide insights into the generation behavior and can help with
    optimization and monitoring.

    :param all_lengths: List of output lengths from all processes
    :type all_lengths: List[int]

    :param percentiles: List of percentiles to compute (e.g., [25, 50, 75, 90])
    :type percentiles: List[int]

    :return: Dictionary containing statistics about the length distribution
    :rtype: Dict[str, Any]

    Example::

        >>> lengths = [10, 15, 20, 25, 30, 35, 40]
        >>> stats = analyze_output_lengths(lengths, percentiles=[25, 50, 75])
        >>> print(f"Mean: {stats['mean_length']}")
        >>> print(f"75th percentile: {stats['percentiles'][75]}")
    """
    all_lengths = np.array(all_lengths)

    min_len = np.min(all_lengths)
    max_len = np.max(all_lengths)
    mean_len = np.mean(all_lengths)
    median_len = np.median(all_lengths)

    stats = {
        "total_samples": len(all_lengths),
        "min_length": min_len,
        "max_length": max_len,
        "mean_length": mean_len,
        "median_length": median_len,
        "percentiles": {},
    }

    percentile_values = np.percentile(all_lengths, percentiles)

    for p, v in zip(percentiles, percentile_values):
        stats["percentiles"][p] = round(v)

    # if plot_out_dir is not None:
    #     plt.figure(figsize=(10, 6))
    #     plt.hist(all_lengths, bins=50, alpha=0.7, color="blue")
    #     plt.xlabel("GenerateOutputLength")
    #     plt.ylabel("Frequency")
    #     plt.title("VLLM GenerateOutputLength Distribution")
    #     plt.grid(True, alpha=0.3)
    #     plt.legend()

    #     plt.savefig(plot_out_dir, bbox_inches="tight")

    return stats
