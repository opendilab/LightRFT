"""
Reward Models Utility Module (Rule-Based Rewards)

This module provides utility functions for computing rule-based rewards based on
response formatting (e.g., <think> and <answer> tags) and accuracy verification.

Main Features:
    - Pure rule-based reward computation (no neural reward models)
    - Format checking: <think>...</think> and <answer>...</answer> validation
    - Accuracy verification: Comparing model answers with ground truth from dataset metadata
    - Reward mixing and computation with detailed metrics

Supported Tasks:
    - General: Any task requiring structured reasoning and answer format

Note:
    This version does NOT load neural reward models. It computes rewards based on
    structural constraints and accuracy using the metadata from the dataset.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


# ============================================================================
# Reward Recipe Configuration
# ============================================================================

RECIPE: Dict[str, List[Tuple[str, Optional[str], float]]] = {
    # Format: (reward_type, model_key, weight)
    "general":    [("format_rule", None, 1.0), ("accuracy_rule", None, 1.0)],
}


# ============================================================================
# Format Reward Functions
# ============================================================================

def format_reward_fn(sol: str) -> float:
    """
    Format reward function.

    Check if the solution follows the required format:
    - Contains <think>...</think> tags for reasoning
    - Contains <answer>...</answer> for final answer
    - The think tags must appear BEFORE the answer

    :param sol: Solution string from model
    :type sol: str
    :return: 1.0 if format is correct, 0.0 otherwise
    :rtype: float
    """
    # Check for <think>...</think>
    think_match = re.search(r'<think>.*?</think>', sol, re.DOTALL)
    
    # Check for <answer>...</answer>
    answer_match = re.search(r'<answer>.*?</answer>', sol, re.DOTALL)

    # Both components must be present AND think must come before answer
    if think_match and answer_match:
        # Check that </think> comes before the answer start
        think_end = think_match.end()
        answer_start = answer_match.start()
        return 1.0 if think_end <= answer_start else 0.0
    else:
        return 0.0


# ============================================================================
# Accuracy Reward Functions
# ============================================================================

def accuracy_reward_fn(sol: str, extra: Any) -> float:
    """
    Accuracy reward function.

    Extract the answer from the solution and compare it with the ground truth
    provided in the 'extra' information.

    :param sol: Solution string from model
    :type sol: str
    :param extra: Extra information from dataset
    :type extra: Any
    :return: 1.0 if answer is correct, 0.0 otherwise
    :rtype: float
    """
    
    # Extract answer from <answer>...</answer>
    matches = re.findall(r'<answer>(.*?)(?:</answer>|$)', sol, re.DOTALL)
    if not matches:
        return 0.0
    
    model_answer = matches[-1].strip()
    
    # Get ground truth from extra
    gt_answer = extra['preference']
    
    score = 0.0
    if gt_answer == 'A':
        if model_answer in ['Image 1 is better', 'Video 1 is better']:
            gt_answer = model_answer
            score = 1.0
    elif gt_answer == 'B':
        if model_answer in ['Image 2 is better', 'Video 2 is better']:
            gt_answer = model_answer
            score = 1.0
    elif gt_answer == 'C':
        if model_answer in ['Both are equally good']:
            score = 1.0
    else:
        score = 0.0

    return score


# ============================================================================
# Reward Computation
# ============================================================================

def reward_fn(
    model_reward_list: List[torch.Tensor],
    labels: Sequence[str],
    queries: Sequence[str],
    refs: Sequence[str],
    **kwargs,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    External unified interface for computing final rewards.

    This is the main entry point called by the trainer. It:
        1. Handles empty model_reward_list (for pure rule-based rewards)
        2. Computes rewards based on RECIPE
        3. Returns final reward tensor with detailed metrics

    :param model_reward_list: List of reward tensors from each model, each shape (B,) - empty for rule-based
    :type model_reward_list: List[torch.Tensor]
    :param labels: List of data labels indicating reward type (length B)
    :type labels: Sequence[str]
    :param queries: List of query/solution strings (length B)
    :type queries: Sequence[str]
    :param refs: List of reference answers (length B)
    :type refs: Sequence[str]
    :param kwargs: Additional keyword arguments for reward functions
    :type kwargs: Any

    :return: Tuple of (final_reward, metrics_dict) where final_reward is combined reward tensor
             of shape (B,) and metrics_dict contains detailed reward metrics
    :rtype: Tuple[torch.Tensor, Dict[str, torch.Tensor]]

    Note:
        For rule-based rewards, model_reward_list will be empty since no neural models are used.
        All rewards are computed via rule-based functions (currently format-only).
    """
    # Create placeholder model_scores tensor
    # For rule-based rewards, this will be empty (shape: [0, B])
    if model_reward_list:
        model_scores = torch.stack(model_reward_list)  # (n_model, B)
    else:
        # No neural reward models - create empty placeholder
        B = len(labels)
        model_scores = torch.zeros(0, B, dtype=torch.float32, device="cuda")

    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        print(f"[reward_fn] labels: {labels}")

    device = model_scores.device if model_scores.numel() > 0 else torch.device('cuda')
    B = len(labels)

    final_reward = torch.zeros(B, dtype=torch.float32, device=device)

    # Initialize metrics dict to track individual reward components
    metrics_dict: Dict[str, torch.Tensor] = {
        'format_reward': torch.zeros(B, dtype=torch.float32, device=device),
        'accuracy_reward': torch.zeros(B, dtype=torch.float32, device=device),
    }

    # ---------- Main loop ----------
    for i, lab in enumerate(labels):
        sol = queries[i]
        ref = refs[i] if refs is not None else None

        # Extract only the assistant's response part
        # Qwen/Llama-3 templates use <|im_start|>assistant\n or assistant\n
        if "<|im_start|>assistant" in sol:
            processed_sol = sol.split("<|im_start|>assistant")[-1]
        elif "assistant\n" in sol:
            processed_sol = sol.split("assistant\n")[-1]
        else:
            processed_sol = sol

        sol_stripped = processed_sol.strip()

        # Get recipe for this label
        recipe = RECIPE.get(lab)
        if recipe is None:
            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                print(f"[WARNING] Label <{lab}> not in RECIPE, using 0.0 reward")
            final_reward[i] = 0.0
            continue

        r = 0.0
        for typ, key, w in recipe:
            if typ == "format_rule":
                fmt_r = format_reward_fn(sol_stripped)
                r += w * fmt_r
                metrics_dict['format_reward'][i] = fmt_r
            elif typ == "accuracy_rule":
                acc_r = accuracy_reward_fn(sol_stripped, ref)
                r += w * acc_r
                metrics_dict['accuracy_reward'][i] = acc_r
            else:
                if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                    print(f"[WARNING] Unknown reward type '{typ}' or not supported in format-only mode")

        final_reward[i] = r

    return final_reward, metrics_dict
