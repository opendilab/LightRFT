"""
Reward Models Utility Module (Simplified for Rule-Based Rewards)

This module provides utility functions for computing rule-based rewards for
mathematical reasoning tasks (Geo3K and GSM8K datasets).

Main Features:
    - Pure rule-based reward computation (no neural reward models)
    - Format checking: <think>...</think> and \\boxed{} validation
    - Accuracy checking: Answer verification using mathruler
    - Reward mixing and computation with detailed metrics

Supported Tasks:
    - Geo3K: Geometry problem solving with rule-based rewards
    - GSM8K: Grade school math problems with rule-based rewards

Note:
    This is a simplified version that does NOT load neural reward models.
    For multi-modal training with neural reward models, see the full version
    in examples/safework_t1/reward_models_utils.py
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch


# ============================================================================
# Reward Recipe Configuration
# ============================================================================

RECIPE: Dict[str, List[Tuple[str, Optional[str], float]]] = {
    # Geo3K dataset: pure rule-based reward (no reward model needed)
    # Format: (reward_type, model_key, weight)
    "geo3k_rule": [("geo3k_rule", None, 1.0)],

    # GSM8K dataset: pure rule-based reward (no reward model needed)
    "gsm8k_rule": [("gsm8k_rule", None, 1.0)],
}


# ============================================================================
# Model Loading Interface (Simplified)
# ============================================================================

# Type hint for compatibility with train_colocate.py
RawRewardInput = Union[str, Dict[str, str], List[Dict[str, str]], None]


def load_reward_models(
    raw_reward_pretrain: RawRewardInput,
    strategy: Any,
    use_engine: bool = False,
) -> Tuple[List[Any], List[Any], Dict[str, int]]:
    """
    Load reward models (simplified for rule-based rewards).

    For geo3k/gsm8k datasets, no neural reward models are needed.
    This function returns empty lists to maintain interface compatibility.

    :param raw_reward_pretrain: Raw configuration (ignored for rule-based rewards)
    :type raw_reward_pretrain: RawRewardInput
    :param strategy: Training strategy instance
    :type strategy: Any
    :param use_engine: Whether to use engine (ignored for rule-based rewards)
    :type use_engine: bool
    :return: Tuple of (reward_models, reward_tokenizers, label_map) - all empty for rule-based rewards
    :rtype: Tuple[List[Any], List[Any], Dict[str, int]]

    Note:
        This simplified version does not load any neural reward models.
        Rewards are computed purely based on format and accuracy rules.
    """
    strategy.print("=" * 80)
    strategy.print("[INFO] Using pure rule-based rewards (geo3k/gsm8k)")
    strategy.print("[INFO] No neural reward models loaded")
    strategy.print("[INFO] Rewards computed based on format + accuracy only")
    strategy.print("=" * 80)

    # Return empty lists to maintain interface compatibility
    # - reward_models: [] (no models)
    # - reward_tokenizers: [] (no tokenizers)
    # - label_map: {} (empty mapping)
    return [], [], {}


# ============================================================================
# Geo3K Reward Functions
# ============================================================================

def geo3k_accuracy_reward_fn(sol: str, gt: str) -> float:
    """
    Geo3K accuracy reward function.

    Extract answer from \\boxed{} notation and use mathruler to verify correctness.
    This is based on the verl implementation for geo3k dataset.

    :param sol: Solution string from model (should contain \\boxed{answer})
    :type sol: str
    :param gt: Ground truth answer
    :type gt: str
    :return: 1.0 if answer is correct, 0.0 otherwise
    :rtype: float
    """
    from mathruler.grader import extract_boxed_content, grade_answer
    pred = extract_boxed_content(sol)
    return 1.0 if grade_answer(pred, gt) else 0.0


def geo3k_format_reward_fn(sol: str) -> float:
    """
    Geo3K format reward function.

    Check if the solution follows the required format:
    - Contains <think>...</think> tags for reasoning
    - Contains \\boxed{} for final answer
    - The think tags must appear BEFORE the boxed answer

    This is based on the verl implementation for geo3k dataset.

    :param sol: Solution string from model
    :type sol: str
    :return: 1.0 if format is correct, 0.0 otherwise
    :rtype: float
    """
    # Strip leading/trailing whitespace for robust matching
    sol_stripped = sol.strip()

    # Check if solution contains both <think>...</think> and \boxed{...}
    # Use re.search to find positions
    think_match = re.search(r'<think>.*?</think>', sol_stripped, re.DOTALL)
    boxed_match = re.search(r'\\boxed\{.*?\}', sol_stripped, re.DOTALL)

    # Both components must be present AND think must come before boxed
    if think_match and boxed_match:
        # Check that </think> comes before \boxed
        think_end = think_match.end()
        boxed_start = boxed_match.start()
        return 1.0 if think_end <= boxed_start else 0.0
    else:
        return 0.0


def geo3k_combined_reward_fn(
    sol: str,
    gt: str,
    format_weight: float = 0.1
) -> float:
    """
    Geo3K combined reward function.

    Combines format reward and accuracy reward with specified weights.
    Default: 90% accuracy + 10% format (matching verl implementation)

    :param sol: Solution string from model
    :type sol: str
    :param gt: Ground truth answer
    :type gt: str
    :param format_weight: Weight for format reward. Default to 0.1
    :type format_weight: float
    :return: Weighted combination of format and accuracy rewards
    :rtype: float
    """
    acc_reward = geo3k_accuracy_reward_fn(sol, gt)
    fmt_reward = geo3k_format_reward_fn(sol)
    return (1.0 - format_weight) * acc_reward + format_weight * fmt_reward


# ============================================================================
# GSM8K Reward Functions
# ============================================================================

def gsm8k_accuracy_reward_fn(sol: str, gt: str) -> float:
    """
    GSM8K accuracy reward function.

    Extract answer from \\boxed{} notation and use mathruler to verify correctness.
    This follows the same pattern as geo3k but for GSM8K dataset.

    :param sol: Solution string from model (should contain \\boxed{answer})
    :type sol: str
    :param gt: Ground truth answer
    :type gt: str
    :return: 1.0 if answer is correct, 0.0 otherwise
    :rtype: float
    """
    from mathruler.grader import extract_boxed_content, grade_answer
    pred = extract_boxed_content(sol)
    return 1.0 if grade_answer(pred, gt) else 0.0


def gsm8k_format_reward_fn(sol: str) -> float:
    """
    GSM8K format reward function.

    Check if the solution follows the required format:
    - Contains <think>...</think> tags for reasoning
    - Contains \\boxed{} for final answer
    - The think tags must appear BEFORE the boxed answer

    This follows the same pattern as geo3k format checking.

    :param sol: Solution string from model
    :type sol: str
    :return: 1.0 if format is correct, 0.0 otherwise
    :rtype: float
    """
    # Strip leading/trailing whitespace for robust matching
    sol_stripped = sol.strip()

    # Check if solution contains both <think>...</think> and \boxed{...}
    # Use re.search to find positions
    think_match = re.search(r'<think>.*?</think>', sol_stripped, re.DOTALL)
    boxed_match = re.search(r'\\boxed\{.*?\}', sol_stripped, re.DOTALL)

    # Both components must be present AND think must come before boxed
    if think_match and boxed_match:
        # Check that </think> comes before \boxed
        think_end = think_match.end()
        boxed_start = boxed_match.start()
        return 1.0 if think_end <= boxed_start else 0.0
    else:
        return 0.0


def gsm8k_combined_reward_fn(
    sol: str,
    gt: str,
    format_weight: float = 0.1
) -> float:
    """
    GSM8K combined reward function.

    Combines format reward and accuracy reward with specified weights.
    Default: 90% accuracy + 10% format (matching verl and geo3k implementation)

    :param sol: Solution string from model
    :type sol: str
    :param gt: Ground truth answer
    :type gt: str
    :param format_weight: Weight for format reward. Default to 0.1
    :type format_weight: float
    :return: Weighted combination of format and accuracy rewards
    :rtype: float
    """
    acc_reward = gsm8k_accuracy_reward_fn(sol, gt)
    fmt_reward = gsm8k_format_reward_fn(sol)
    return (1.0 - format_weight) * acc_reward + format_weight * fmt_reward


# ============================================================================
# Reward Mixing and Computation
# ============================================================================

def mix_rewards(
    labels: Sequence[str],
    model_scores: torch.Tensor,
    label_map: Dict[str, int],
    solution_strs: Sequence[str],
    refs: Sequence[str],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Mix rewards from multiple sources according to recipe configuration.

    This function combines rewards based on the RECIPE configuration.
    For geo3k/gsm8k, only rule-based rewards are used (no neural models).

    :param labels: List of data labels (length B)
    :type labels: Sequence[str]
    :param model_scores: Tensor of model scores, shape (n_model, B) - empty for rule-based
    :type model_scores: torch.Tensor
    :param label_map: Mapping from reward type to model index - empty for rule-based
    :type label_map: Dict[str, int]
    :param solution_strs: List of solution strings (length B)
    :type solution_strs: Sequence[str]
    :param refs: List of reference answers (length B)
    :type refs: Sequence[str]
    :return: Tuple of (final_reward, metrics_dict) where final_reward is tensor of shape (B,)
             containing combined rewards and metrics_dict contains detailed reward metrics
    :rtype: Tuple[torch.Tensor, Dict[str, torch.Tensor]]

    Note:
        Rewards are computed purely based on format and accuracy rules.
        No neural reward models are used for geo3k/gsm8k datasets.
    """
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        print(f"[mix_rewards] labels: {labels}")
        print(f"[mix_rewards] model_scores shape: {model_scores.shape}")

    device = model_scores.device if model_scores.numel() > 0 else torch.device('cuda')
    B = len(labels)

    final_reward = torch.zeros(B, dtype=torch.float32, device=device)

    # Initialize metrics dict to track individual reward components
    metrics_dict: Dict[str, torch.Tensor] = {
        'format_reward': torch.zeros(B, dtype=torch.float32, device=device),
        'accuracy_reward': torch.zeros(B, dtype=torch.float32, device=device),
        'model_reward': torch.zeros(B, dtype=torch.float32, device=device),
        'rule_reward': torch.zeros(B, dtype=torch.float32, device=device),
    }

    # ---------- Main loop ----------
    for i, lab in enumerate(labels):
        sol = solution_strs[i]
        gt = refs[i] if i < len(refs) else ""

        # Get recipe for this label
        recipe = RECIPE.get(lab)
        if recipe is None:
            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                print(f"[WARNING] Label <{lab}> not in RECIPE, using 0.0 reward")
            final_reward[i] = 0.0
            continue

        r = 0.0
        for typ, key, w in recipe:
            if typ == "geo3k_rule":
                # Geo3K pure rule-based reward (format + accuracy)
                acc_r = geo3k_accuracy_reward_fn(sol, gt)
                fmt_r = geo3k_format_reward_fn(sol)
                combined_r = (1.0 - 0.1) * acc_r + 0.1 * fmt_r
                r += w * combined_r

                # Track separately
                metrics_dict['accuracy_reward'][i] = acc_r
                metrics_dict['format_reward'][i] = fmt_r
                metrics_dict['rule_reward'][i] = combined_r

            elif typ == "gsm8k_rule":
                # GSM8K pure rule-based reward (format + accuracy)
                acc_r = gsm8k_accuracy_reward_fn(sol, gt)
                fmt_r = gsm8k_format_reward_fn(sol)
                combined_r = (1.0 - 0.1) * acc_r + 0.1 * fmt_r
                r += w * combined_r

                # Track separately
                metrics_dict['accuracy_reward'][i] = acc_r
                metrics_dict['format_reward'][i] = fmt_r
                metrics_dict['rule_reward'][i] = combined_r

            else:
                if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                    print(f"[WARNING] Unknown reward type '{typ}', ignoring")

        final_reward[i] = r

    return final_reward, metrics_dict


def reward_fn(
    model_reward_list: List[torch.Tensor],
    labels: Sequence[str],
    queries: Sequence[str],
    refs: Sequence[str],
    label_map: Dict[str, int],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    External unified interface for computing final rewards.

    This is the main entry point called by the trainer. It:
        1. Handles empty model_reward_list (for pure rule-based rewards)
        2. Calls mix_rewards to compute rewards based on RECIPE
        3. Returns final reward tensor with detailed metrics

    :param model_reward_list: List of reward tensors from each model, each shape (B,) - empty for rule-based
    :type model_reward_list: List[torch.Tensor]
    :param labels: List of data labels indicating reward type (length B)
    :type labels: Sequence[str]
    :param queries: List of query/solution strings (length B)
    :type queries: Sequence[str]
    :param refs: List of reference answers (length B)
    :type refs: Sequence[str]
    :param label_map: Mapping from reward type to model index - empty for rule-based
    :type label_map: Dict[str, int]
    :return: Tuple of (final_reward, metrics_dict) where final_reward is combined reward tensor
             of shape (B,) and metrics_dict contains detailed reward metrics
    :rtype: Tuple[torch.Tensor, Dict[str, torch.Tensor]]

    Note:
        For geo3k/gsm8k, model_reward_list will be empty since no neural models are used.
        All rewards are computed via rule-based functions.
    """
    # Create placeholder model_scores tensor
    # For rule-based rewards, this will be empty (shape: [0, B])
    if model_reward_list:
        model_scores = torch.stack(model_reward_list)  # (n_model, B)
    else:
        # No neural reward models - create empty placeholder
        B = len(labels)
        model_scores = torch.zeros(0, B, dtype=torch.float32, device="cuda")

    # Call mix_rewards to compute final rewards
    return mix_rewards(labels, model_scores, label_map, queries, refs)
