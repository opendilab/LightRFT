from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch


RECIPE: Dict[str, List[Tuple[str, Optional[str], float]]] = {
    "python_expr_rule": [("python_expr_rule", None, 1.0)],
}

RawRewardInput = Union[str, Dict[str, str], List[Dict[str, str]], None]


def extract_response(text: str) -> str:
    if not isinstance(text, str):
        return ""

    s = text.strip()
    if not s:
        return s

    assistant_marker = "<|im_start|>assistant"
    if assistant_marker in s:
        start = s.rfind(assistant_marker) + len(assistant_marker)
        tail = s[start:]
        end_idx = tail.find("<|im_end|>")
        if end_idx != -1:
            tail = tail[:end_idx]
        return tail.strip()
    return s


def extract_boxed_content(text: str) -> str:
    match = re.search(r"\\boxed\{([^{}]*)\}", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_candidate_answer(text: str) -> str:
    boxed = extract_boxed_content(text)
    if boxed:
        return boxed

    compact = text.replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", compact)
    if matches:
        return matches[-1]
    return ""


def normalize_answer(text: str) -> str:
    if not isinstance(text, str):
        return ""

    raw = text.strip().strip("$").replace(",", "")
    raw = raw.rstrip(".")
    if not raw:
        return ""

    try:
        value = Decimal(raw)
    except InvalidOperation:
        return raw

    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return str(int(normalized))
    return format(normalized, "f").rstrip("0").rstrip(".")


def format_reward_fn(solution: str) -> float:
    return 1.0 if extract_boxed_content(solution) else 0.0


def accuracy_reward_fn(solution: str, ground_truth: str) -> float:
    predicted = normalize_answer(extract_candidate_answer(solution))
    target = normalize_answer(ground_truth)
    return 1.0 if predicted and predicted == target else 0.0


def load_reward_models(
    raw_reward_pretrain: RawRewardInput,
    strategy: Any,
    use_engine: bool = False,
) -> Tuple[List[Any], List[Any], Dict[str, int]]:
    strategy.print("=" * 80)
    strategy.print("[INFO] Using pure rule-based rewards for tiny_python_expr")
    strategy.print("[INFO] No neural reward model is loaded")
    strategy.print("=" * 80)
    return [], [], {}


def mix_rewards(
    labels: Sequence[str],
    model_scores: torch.Tensor,
    label_map: Dict[str, int],
    solution_strs: Sequence[str],
    refs: Sequence[str],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    del label_map

    if model_scores.numel() > 0:
        device = model_scores.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = len(labels)
    final_reward = torch.zeros(batch_size, dtype=torch.float32, device=device)
    metrics = {
        "format_reward": torch.zeros(batch_size, dtype=torch.float32, device=device),
        "accuracy_reward": torch.zeros(batch_size, dtype=torch.float32, device=device),
        "rule_reward": torch.zeros(batch_size, dtype=torch.float32, device=device),
        "model_reward": torch.zeros(batch_size, dtype=torch.float32, device=device),
    }

    for i, label in enumerate(labels):
        if label != "python_expr_rule":
            continue

        solution = extract_response(solution_strs[i])
        reference = refs[i] if i < len(refs) else ""
        format_reward = format_reward_fn(solution)
        accuracy_reward = accuracy_reward_fn(solution, reference)
        total_reward = 0.1 * format_reward + 0.9 * accuracy_reward

        metrics["format_reward"][i] = format_reward
        metrics["accuracy_reward"][i] = accuracy_reward
        metrics["rule_reward"][i] = total_reward
        final_reward[i] = total_reward

    return final_reward, metrics


def reward_fn(
    model_reward_list: List[torch.Tensor],
    labels: Sequence[str],
    queries: Sequence[str],
    refs: Sequence[str],
    label_map: Dict[str, int],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if model_reward_list:
        model_scores = torch.stack(model_reward_list)
    else:
        model_scores = torch.zeros(
            0,
            len(labels),
            dtype=torch.float32,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
    return mix_rewards(labels, model_scores, label_map, queries, refs)
