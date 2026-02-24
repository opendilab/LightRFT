"""
Reward Models Utility Module for R1-AQA (Audio Question Answering)

This module provides rule-based reward functions faithfully ported from
R1-AQA (xiaomi-research/r1-aqa) ``src/utils/rewards.py``.

Reward Design (from R1-AQA source):
    total_reward = accuracy_reward + format_reward
    - accuracy_reward: 1.0 if the extracted <answer>...</answer> matches ground truth, else 0.0
      (tries symbolic verification first, then exact string matching)
    - format_reward: 1.0 if the output matches pattern ``.*?<answer>.*?</answer>``, else 0.0

Differences from GSM8K/Geo3K rewards in LightRFT:
    - Uses <answer></answer> tags instead of \\boxed{}
    - Uses additive combination (1+1=2 max) instead of weighted (0.9+0.1=1 max)
    - No <think></think> requirement in format check (optional support)

Interface:
    Follows LightRFT's reward_fn signature exactly:
    reward_fn(model_reward_list, labels, queries, refs, label_map) -> (tensor, dict)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple, Union

import torch


# ============================================================================
# Model Loading Interface (Simplified — no neural models needed)
# ============================================================================

RawRewardInput = Union[str, Dict[str, str], List[Dict[str, str]], None]


def load_reward_models(
    raw_reward_pretrain: RawRewardInput,
    strategy: Any,
    use_engine: bool = False,
) -> Tuple[List[Any], List[Any], Dict[str, int]]:
    """
    Load reward models (simplified for rule-based rewards).

    For R1-AQA, no neural reward models are needed. This function returns
    empty lists to maintain interface compatibility with LightRFT's
    ``train_colocate.py``.

    :return: Tuple of (reward_models, reward_tokenizers, label_map)
    """
    strategy.print("=" * 80)
    strategy.print("[INFO] R1-AQA: Using pure rule-based rewards (accuracy + format)")
    strategy.print("[INFO] No neural reward models loaded")
    strategy.print("[INFO] Rewards: accuracy_reward(0/1) + format_reward(0/1)")
    strategy.print("=" * 80)
    return [], [], {}


# ============================================================================
# R1-AQA Accuracy Reward (ported from src/utils/rewards.py)
# ============================================================================

def accuracy_reward_fn(content: str, solution: str) -> float:
    """
    R1-AQA accuracy reward function.

    Faithfully ported from ``r1-aqa/src/utils/rewards.py::accuracy_reward``.

    Logic:
    1. Try symbolic verification via ``math_verify`` (parse + verify).
    2. If that fails, try exact string matching on <answer> tag content.
    3. Return 1.0 if correct, 0.0 otherwise.

    :param content: Model's generated output text.
    :param solution: Ground truth solution string (may contain <answer> tags).
    :return: 1.0 if answer is correct, 0.0 otherwise.
    """
    reward = 0.0

    # --- Method 1: Symbolic verification (from R1-AQA) ---
    try:
        from math_verify import parse, verify
        answer = parse(content)
        if float(verify(answer, parse(solution))) > 0:
            reward = 1.0
    except Exception:
        pass

    # --- Method 2: Exact string matching on <answer> tags (from R1-AQA) ---
    if reward == 0.0:
        try:
            # Extract answer from solution (ground truth)
            sol_match = re.search(r"<answer>(.*?)</answer>", solution)
            ground_truth = sol_match.group(1).strip() if sol_match else solution.strip()

            # Extract answer from content (model output)
            content_match = re.search(r"<answer>(.*?)</answer>", content)
            student_answer = content_match.group(1).strip() if content_match else content.strip()

            if student_answer == ground_truth:
                reward = 1.0
        except Exception:
            pass

    return reward


# ============================================================================
# R1-AQA Format Reward (ported from src/utils/rewards.py)
# ============================================================================

def format_reward_fn(content: str, enable_think: bool = False) -> float:
    """
    R1-AQA format reward function.

    Faithfully ported from ``r1-aqa/src/utils/rewards.py::format_reward``.

    Checks that the output contains ``<answer>...</answer>`` tags.
    Optionally also checks for ``<think>...</think>`` tags.

    :param content: Model's generated output text.
    :param enable_think: If True, also require <think></think> tags.
    :return: 1.0 if format is correct, 0.0 otherwise.
    """
    if enable_think:
        # Pattern requires <think>...</think> followed by <answer>...</answer>
        pattern = r".*?<think>\s*.*?</think>\s*.*?<answer>.*?</answer>"
    else:
        # Default R1-AQA pattern: just require <answer>...</answer>
        pattern = r".*?<answer>.*?</answer>"

    match = re.fullmatch(pattern, content, re.DOTALL)
    return 1.0 if match else 0.0


# ============================================================================
# Combined Reward (per-sample)
# ============================================================================

def avqa_combined_reward_fn(
    sol: str,
    gt: str,
    enable_think: bool = False,
) -> Tuple[float, float, float]:
    """
    Compute combined AVQA reward for a single sample.

    R1-AQA sums accuracy and format rewards (no weighting):
        total = accuracy_reward + format_reward

    :param sol: Model's generated output text.
    :param gt: Ground truth answer string.
    :param enable_think: Whether to require <think> tags in format.
    :return: Tuple of (total_reward, accuracy_reward, format_reward).
    """
    # Build solution string in R1-AQA format for matching
    solution_str = f" <answer>{gt}</answer> "

    acc_r = accuracy_reward_fn(sol, solution_str)
    fmt_r = format_reward_fn(sol, enable_think=enable_think)
    total_r = acc_r + fmt_r  # R1-AQA sums rewards (max=2.0)
    return total_r, acc_r, fmt_r


# ============================================================================
# Reward Function (LightRFT interface — called by the trainer)
# ============================================================================

def reward_fn(
    model_reward_list: List[torch.Tensor],
    labels: Sequence[str],
    queries: Sequence[str],
    refs: Sequence[str],
    label_map: Dict[str, int],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute rule-based AVQA rewards (accuracy + format) for each sample.

    Signature matches LightRFT's trainer expectation. For R1-AQA, model_reward_list,
    labels, and label_map are unused (rule-based only).

    :param model_reward_list: List of reward tensors from neural models — empty for R1-AQA.
    :param labels: List of data labels (length B); unused.
    :param queries: List of model-generated solution strings (length B).
    :param refs: List of reference/ground-truth answers (length B).
    :param label_map: Mapping from reward type to model index — unused.
    :return: Tuple of (final_reward [B], metrics_dict).
    """
    device = torch.device("cuda")
    B = len(queries)

    final_reward = torch.zeros(B, dtype=torch.float32, device=device)
    metrics_dict: Dict[str, torch.Tensor] = {
        "format_reward": torch.zeros(B, dtype=torch.float32, device=device),
        "accuracy_reward": torch.zeros(B, dtype=torch.float32, device=device),
        "rule_reward": torch.zeros(B, dtype=torch.float32, device=device),
    }

    for i in range(B):
        sol = queries[i]
        gt = refs[i] if i < len(refs) else ""
        total_r, acc_r, fmt_r = avqa_combined_reward_fn(sol, gt)
        final_reward[i] = total_r
        metrics_dict["accuracy_reward"][i] = acc_r
        metrics_dict["format_reward"][i] = fmt_r
        metrics_dict["rule_reward"][i] = total_r

    return final_reward, metrics_dict
