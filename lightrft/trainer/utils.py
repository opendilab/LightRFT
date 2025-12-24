"""
This module provides utilities for statistical computations commonly used in reinforcement learning
and machine learning workflows. It includes functions for computing clipping fractions and classes
for tracking running statistics of data streams.

The main components are:
- compute_clip_fraction: Calculates the fraction of tensor elements that fall outside specified bounds
- RunningMoments: Maintains running mean and standard deviation statistics for streaming data
- get_cpgd_advantages_returns: Computes advantages and returns for CPGD algorithm

These utilities are particularly useful in RL algorithms like PPO where clipping statistics and
normalization are important for training stability and monitoring.
"""

import copy
import torch
from copy import deepcopy
from typing import Callable, List, Tuple, Union, Optional


def fire_sampling(
    all_prompt_token_ids: List[List[int]],
    generate_fn: Callable,
    engine_type: str = "vllm",
    first_token_temperature: float = 10.0,
    temperature: float = 1.0,
    first_token_top_k: int = -1,
    first_token_top_p: float = 1.0,
    is_multimodal: bool = False,
    all_prompts: Optional[List[str]] = None,
    all_images: Optional[List] = None,
    all_images_num: Optional[List[int]] = None,
    sampling_params: Optional[Union[dict, object]] = None,
) -> List:
    """
    FIRE sampling (Flaming-hot Initiation with Regular Execution)

    FIRE sampling paper link: https://arxiv.org/abs/2410.21236
    The first token is generated with high temperature and optional filters,
    and the rest tokens are generated with normal temperature.

    :param all_prompt_token_ids: List of tokenized prompts
    :type all_prompt_token_ids: List[List[int]]
    :param generate_fn: Function to call for generation (with pre-configured parameters)
    :type generate_fn: Callable
    :param engine_type: Backend type ("vllm" or "sglang")
    :type engine_type: str
    :param first_token_temperature: Temperature for first token generation
    :type first_token_temperature: float
    :param temperature: Temperature for remaining tokens
    :type temperature: float
    :param first_token_top_k: Top-k for first token
    :type first_token_top_k: int
    :param first_token_top_p: Top-p for first token
    :type first_token_top_p: float
    :param is_multimodal: Whether this is multimodal generation
    :type is_multimodal: bool
    :param all_prompts: Text prompts (for multimodal)
    :type all_prompts: Optional[List[str]]
    :param all_images: Images (for multimodal)
    :type all_images: Optional[List]
    :param all_images_num: Number of images per prompt
    :type all_images_num: Optional[List[int]]
    :param sampling_params: Original sampling parameters (for fallback)
    :type sampling_params: Optional[Union[dict, object]]

    :return: List of generation outputs
    :rtype: List
    """
    # Step 1: Generate the first token with high temperature
    if engine_type == "vllm":
        sampling_params_first = copy.deepcopy(sampling_params)
        sampling_params_first.temperature = first_token_temperature
        sampling_params_first.top_k = first_token_top_k
        sampling_params_first.top_p = first_token_top_p
        sampling_params_first.max_tokens = 1
    else:  # sglang
        sampling_params_first = copy.deepcopy(sampling_params)
        sampling_params_first["temperature"] = first_token_temperature
        sampling_params_first["top_k"] = first_token_top_k
        sampling_params_first["top_p"] = first_token_top_p
        sampling_params_first["max_new_tokens"] = 1

    # Generate first token
    first_token_outputs = generate_fn(
        sampling_params=sampling_params_first,
        all_prompt_token_ids=all_prompt_token_ids,
        all_prompts=all_prompts if is_multimodal else None,
        all_images=all_images,
        images_num=all_images_num if is_multimodal else None,
    )

    # Concatenate the first token to the prompt
    new_prompt_token_ids = []
    for orig_ids, out in zip(all_prompt_token_ids, first_token_outputs):
        first_tok = list(out.output_token_ids)  # [token_id]
        new_prompt_token_ids.append(orig_ids + first_tok)

    # Step 2: Generate remaining tokens with normal temperature
    if engine_type == "vllm":
        sampling_params_rest = copy.deepcopy(sampling_params)
        sampling_params_rest.temperature = temperature
        sampling_params_rest.max_tokens = sampling_params.max_tokens - 1
    else:
        sampling_params_rest = copy.deepcopy(sampling_params)
        sampling_params_rest["temperature"] = temperature
        sampling_params_rest["max_new_tokens"] = sampling_params["max_new_tokens"] - 1

    # Generate remaining tokens
    rest_outputs = generate_fn(
        sampling_params=sampling_params_rest,
        all_prompt_token_ids=new_prompt_token_ids,
        all_prompts=all_prompts if is_multimodal else None,
        all_images=all_images,
        images_num=all_images_num if is_multimodal else None,
    )

    # Merge the first token with the remaining tokens
    all_outputs = []
    for first_out, rest_out in zip(first_token_outputs, rest_outputs):
        merged = rest_out
        merged.prompt_token_ids = first_out.prompt_token_ids
        merged.output_token_ids = list(first_out.output_token_ids) + list(rest_out.output_token_ids)
        all_outputs.append(merged)

    return all_outputs


def compute_clip_fraction(values: torch.Tensor, clip_max: float, clip_min: float) -> torch.Tensor:
    """
    Compute the fraction of elements in a tensor that are clipped.

    This function calculates what proportion of the input tensor's elements fall outside
    the specified clipping bounds [clip_min, clip_max]. This is commonly used in
    reinforcement learning to monitor how often policy updates are being clipped,
    which can indicate training stability.

    :param values: The input tensor to analyze for clipping.
    :type values: torch.Tensor
    :param clip_max: The maximum value for clipping bounds.
    :type clip_max: float
    :param clip_min: The minimum value for clipping bounds.
    :type clip_min: float
    :return: A tensor of shape (batch_size,) where each element is the fraction of
             clipped values in the input tensor.
    :rtype: torch.Tensor

    Example::

        >>> values = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
        >>> clip_fraction = compute_clip_fraction(values, clip_max=2.0, clip_min=1.0)
        >>> print(clip_fraction)  # Should show fraction of values outside [1.0, 2.0]
    """
    numel = values.numel()
    if numel == 0:
        return torch.zeros(values.shape[0], device=values.device, dtype=torch.float32)

    batch_size = values.shape[0]

    # Count elements that are either above clip_max or below clip_min
    num_clipped = (values > clip_max).sum().item() + (values < clip_min).sum().item()

    clip_fraction = num_clipped / numel

    # The result is expanded to the batch size for compatibility with downstream logging
    return torch.tensor([clip_fraction], device=values.device, dtype=torch.float32).expand(batch_size)


class RunningMoments:
    """
    Calculate the running mean and standard deviation of a data stream.

    This class implements Welford's online algorithm for computing running statistics,
    allowing efficient computation of mean and standard deviation as new data arrives
    without storing all historical data. This is particularly useful for normalizing
    inputs in reinforcement learning or for monitoring training statistics.

    The implementation uses a parallel algorithm to combine statistics from new batches
    with existing running statistics, ensuring numerical stability even with large
    amounts of data.

    Adapted from https://github.com/alibaba/ROLL

    Example::

        >>> moments = RunningMoments()
        >>> batch1 = torch.randn(100)
        >>> mean1, std1 = moments.update(batch1)
        >>> batch2 = torch.randn(100)
        >>> mean2, std2 = moments.update(batch2)
        >>> print(f"Running mean: {moments.mean}, Running std: {moments.std}")
    """
    def __init__(self):
        """
        Initialize the RunningMoments tracker.

        Sets initial values for mean, standard deviation, variance, and count.
        The count is initialized to a small positive value to prevent division by zero.
        """
        self.mean = 0
        self.std = 1
        self.var = 1
        # Initialize count with a small value to prevent division by zero
        self.count = 1e-24

    @torch.no_grad()
    def update(self, xs: torch.Tensor) -> Tuple[float, float]:
        """
        Update running statistics with a new batch of data.

        This method uses Welford's online algorithm combined with a parallel algorithm
        to efficiently update the running mean, variance, and standard deviation with
        a new batch of data. The algorithm is numerically stable and doesn't require
        storing all previous data points.

        :param xs: The new tensor of data to incorporate into the running statistics.
        :type xs: torch.Tensor
        :return: A tuple of (mean, std) for the current batch `xs`.
        :rtype: Tuple[float, float]

        Example::

            >>> moments = RunningMoments()
            >>> new_data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
            >>> batch_mean, batch_std = moments.update(new_data)
            >>> print(f"Batch mean: {batch_mean}, Batch std: {batch_std}")
        """
        # 1. Get statistics for the new batch
        xs_count = xs.numel()
        # `torch.var_mean` with unbiased=False calculates population variance
        xs_var, xs_mean = torch.var_mean(xs, unbiased=False)
        xs_mean, xs_var = xs_mean.float(), xs_var.float()

        # 2. Use a parallel algorithm to combine running stats with new batch stats
        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        # The new combined sum of squared differences (M2) is calculated
        # It's the sum of the old M2, the new M2, and a correction term
        # that accounts for the difference in means between the two sets
        old_sum_sq_diff = self.var * self.count
        new_sum_sq_diff = xs_var * xs_count
        correction_term = delta ** 2 * self.count * xs_count / tot_count

        tot_sum_sq_diff = old_sum_sq_diff + new_sum_sq_diff + correction_term

        # 3. Update running mean, variance, and standard deviation
        self.mean += delta * xs_count / tot_count
        self.var = tot_sum_sq_diff / tot_count
        # Convert running population variance to unbiased sample standard deviation
        self.std = (self.var * tot_count / (tot_count - 1)).float().sqrt()
        self.count = tot_count

        # 4. Return the mean and sample standard deviation of the current batch
        xs_std = (xs_var * xs_count / (xs_count - 1)).float().sqrt() if xs_count > 1 else torch.tensor(0.0)
        return xs_mean.item(), xs_std.item()


def get_cpgd_advantages_returns(
    reward: torch.Tensor,
    action_mask: torch.Tensor,
    weight_factor: str = "STD_weight",
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregate token-level rewards into episode-level scores, normalize them
    group-wise, and then broadcast the normalized scores back to the
    token dimension to obtain both the advantages and the returns that are
    required by the CPGD (Clipped Policy Gradient Optimization with Policy Drift) algorithm.

    :param reward: Tensor of shape (num_actions, seq_len) containing token-level rewards
        produced by the reward model. Each row corresponds to one sampled
        response (action trajectory).
    :type reward: torch.Tensor
    :param action_mask: Tensor of the same shape as `reward`. Elements belonging to the
        generated response tokens are 1; padding / prompt tokens are 0.
        The mask is used so that only response tokens contribute to the
        final advantages / returns.
    :type action_mask: torch.Tensor
    :param weight_factor: Determines how the per-sample scalar scores are normalized.
        - "STD_weight": z-score normalization: score_i = (score_i − mean) / (std + ε)
        - "clip_filter_like_weight": a simplified version of the Clip-Filter weight
          used in early RLHF repos: score_i = (score_i − mean) * clamp(num_actions / nz, max=3)
        - any other value: mean-centering only: score_i = score_i − mean
        Defaults to "STD_weight".
    :type weight_factor: str
    :param epsilon: Small constant added to the denominator to avoid division by zero, defaults to 1e-6.
    :type epsilon: float
    :return: A tuple of (advantages, returns).
        - advantages: Normalized per-token advantages, shape (num_actions, seq_len).
        - returns: Identical to `advantages` in CPGD; returned separately for API symmetry.
    :rtype: Tuple[torch.Tensor, torch.Tensor]

    Notes:
        - Both `advantages` and `returns` are masked so that non-response tokens
          are always zero.
        - The function performs no gradient-tracking operations and is intended
          to be called outside the optimization graph.
    """

    # ------------------------------------------------------------------
    # 1. Collapse token-level rewards to a single scalar per trajectory
    # ------------------------------------------------------------------
    # Shape: (num_actions,)
    scores = reward.sum(dim=-1)

    # Mean and (biased) standard deviation across the batch
    mean = scores.mean()
    std = scores.std(unbiased=False)

    # ------------------------------------------------------------------
    # 2. Group-wise normalization
    # ------------------------------------------------------------------
    if weight_factor == "STD_weight":
        # Standard z-score normalization
        scores = (scores - mean) / (std + epsilon)

    elif weight_factor == "clip_filter_like_weight":
        # A rough approximation of the clip-filter weighting
        # Count of (std > 0) is always ≥ 1, prevents division by zero
        non_zero = (std > 0).sum().clamp(min=1)
        # Scale by (batch_size / non_zero) but clip to a maximum of 3
        scores = (scores - mean) * (scores.size(0) / non_zero).clamp(max=3.0)

    else:
        # Fallback: mean-centering only
        scores = scores - mean

    # ------------------------------------------------------------------
    # 3. Broadcast back to token dimension and apply the mask
    # ------------------------------------------------------------------
    # Shape: (num_actions, seq_len)
    scores = scores.unsqueeze(-1) * action_mask

    # In CPGD the advantage equals the return
    advantages = scores
    returns = deepcopy(scores)

    return advantages, returns
