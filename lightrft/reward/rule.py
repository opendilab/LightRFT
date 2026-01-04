"""
Rule-based Reward Implementation

This module provides rule-based reward functions that evaluate model outputs
based on heuristics and format checking rather than neural models.

Main Features:
    - Format checking (e.g., <think> tags, \\boxed{} notation)
    - Accuracy verification using mathruler grader
    - Language consistency checking
    - Registry pattern for custom rule types

Supported Rule Types:
    - default: Basic format checking
    - geo3k_accuracy: Geo3K accuracy verification
    - geo3k_format: Geo3K format checking
    - geo3k_combined: Combined format + accuracy
    - gsm8k_accuracy: GSM8K accuracy verification
    - gsm8k_format: GSM8K format checking
    - gsm8k_combined: Combined format + accuracy

Author: lightrft Team
"""

import re
from typing import Dict, Sequence, Optional, Callable

import torch

from .base import BaseReward


class RuleReward(BaseReward):
    """
    Rule-based reward implementation.
    
    This class encapsulates various rule-based reward functions such as:
    - Format checking (e.g., <think> tags)
    - Accuracy checking (e.g., math answer verification)
    - Language consistency checking
    
    Supports multiple rule types through a registry pattern.
    
    :param rule_type: Type of rule to use (e.g., "geo3k_combined", "gsm8k_combined", "default")
    :type rule_type: str
    :param format_weight: Weight for format reward when combining with accuracy. Default to 0.1
    :type format_weight: float
    :param device: Device to place reward tensors on
    :type device: Optional[torch.device]
    """
    
    # Registry for rule reward functions
    _RULE_FUNCTIONS: Dict[str, Callable] = {}
    
    @classmethod
    def register_rule(cls, name: str):
        """
        Decorator to register a rule reward function.
        
        :param name: Name of the rule type
        :type name: str
        :return: Decorator function
        :rtype: Callable
        
        Example::
        
            @RuleReward.register_rule("geo3k")
            def geo3k_rule(sol: str, gt: str) -> float:
                ...
        """
        def decorator(func: Callable):
            cls._RULE_FUNCTIONS[name] = func
            return func
        return decorator
    
    def __init__(
        self,
        rule_type: str = "default",
        format_weight: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize rule-based reward.
        
        :param rule_type: Type of rule to use (e.g., "geo3k", "gsm8k", "default")
        :type rule_type: str
        :param format_weight: Weight for format reward when combining with accuracy
        :type format_weight: float
        :param device: Device to place reward tensors on
        :type device: Optional[torch.device]
        :raises ValueError: If rule_type is not registered
        """
        super().__init__()
        self.rule_type = rule_type
        self.format_weight = format_weight
        self.device = device or torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
        
        # Get the rule function
        if rule_type not in self._RULE_FUNCTIONS:
            raise ValueError(
                f"Unknown rule type: {rule_type}. "
                f"Available types: {list(self._RULE_FUNCTIONS.keys())}"
            )
        self.rule_func = self._RULE_FUNCTIONS[rule_type]
    
    def compute(
        self,
        queries: Sequence[str],
        references: Optional[Sequence[str]] = None,
        labels: Optional[Sequence[str]] = None,
        **kwargs
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute rule-based rewards.
        
        :param queries: List of solution strings (length B)
        :type queries: Sequence[str]
        :param references: List of ground truth answers (length B), required for accuracy checking
        :type references: Optional[Sequence[str]]
        :param labels: Not used for rule rewards, kept for interface consistency
        :type labels: Optional[Sequence[str]]
        :param kwargs: Additional arguments
        :return: Tuple of (rewards, metrics) where rewards is torch.Tensor of shape (B,)
                 and metrics contains 'format_reward', 'accuracy_reward', 'rule_reward'
        :rtype: Tuple[torch.Tensor, Dict[str, torch.Tensor]]
        """
        if references is None:
            references = [""] * len(queries)
        
        B = len(queries)
        device = self.device
        
        # Initialize metrics
        metrics = {
            'format_reward': torch.zeros(B, dtype=torch.float32, device=device),
            'accuracy_reward': torch.zeros(B, dtype=torch.float32, device=device),
            'rule_reward': torch.zeros(B, dtype=torch.float32, device=device),
        }
        
        rewards = torch.zeros(B, dtype=torch.float32, device=device)
        
        # Compute rewards for each query
        for i, (sol, gt) in enumerate(zip(queries, references)):
            reward_value = self.rule_func(sol, gt)
            rewards[i] = reward_value
            metrics['rule_reward'][i] = reward_value
            
            # For combined rules (geo3k, gsm8k), extract individual components
            if self.rule_type in ["geo3k_combined", "gsm8k_combined"]:
                # These rule functions return combined reward, but we can extract components
                # by calling the individual functions if available
                if hasattr(self, '_extract_components'):
                    fmt_r, acc_r = self._extract_components(sol, gt)
                    metrics['format_reward'][i] = fmt_r
                    metrics['accuracy_reward'][i] = acc_r
        
        return rewards, metrics


# ============================================================================
# Default Rule Reward Functions
# ============================================================================

def _default_rule_reward_fn(sol: str, gt: str) -> float:
    """
    Default rule reward: format checking.
    
    Checks if solution matches format: <think> ... </think> + non-empty content.
    
    :param sol: Solution string to check
    :type sol: str
    :param gt: Ground truth (not used in format check)
    :type gt: str
    :return: 1.0 if format is valid, 0.0 otherwise
    :rtype: float
    """
    pattern = r".*<think>.+?</think>\s*\S+"
    return 1.0 if re.match(pattern, sol, re.DOTALL) else 0.0


RuleReward.register_rule("default")(_default_rule_reward_fn)


# ============================================================================
# Geo3K Rule Reward Functions
# ============================================================================

def _geo3k_accuracy_reward_fn(sol: str, gt: str) -> float:
    """
    Geo3K accuracy reward function.
    
    Extract answer from \\boxed{} notation and use mathruler to verify correctness.
    
    :param sol: Solution string from model (should contain \\boxed{answer})
    :type sol: str
    :param gt: Ground truth answer
    :type gt: str
    :return: 1.0 if answer is correct, 0.0 otherwise
    :rtype: float
    """
    try:
        from mathruler.grader import extract_boxed_content, grade_answer
        pred = extract_boxed_content(sol)
        return 1.0 if grade_answer(pred, gt) else 0.0
    except ImportError:
        # Fallback if mathruler is not available
        return 0.0


def _geo3k_format_reward_fn(sol: str, gt: str) -> float:
    """
    Geo3K format reward function.
    
    Check if the solution follows the required format:
    - Contains <think>...</think> tags for reasoning
    - Contains \\boxed{} for final answer
    - The think tags must appear BEFORE the boxed answer
    
    :param sol: Solution string from model
    :type sol: str
    :param gt: Ground truth (not used in format check)
    :type gt: str
    :return: 1.0 if format is correct, 0.0 otherwise
    :rtype: float
    """
    sol_stripped = sol.strip()
    
    think_match = re.search(r'<think>.*?</think>', sol_stripped, re.DOTALL)
    boxed_match = re.search(r'\\boxed\{.*?\}', sol_stripped, re.DOTALL)
    
    if think_match and boxed_match:
        think_end = think_match.end()
        boxed_start = boxed_match.start()
        return 1.0 if think_end <= boxed_start else 0.0
    else:
        return 0.0


def _geo3k_combined_reward_fn(sol: str, gt: str) -> float:
    """
    Geo3K combined reward function.
    
    Combines format reward and accuracy reward with weights.
    Default: 90% accuracy + 10% format.
    
    :param sol: Solution string from model
    :type sol: str
    :param gt: Ground truth answer
    :type gt: str
    :return: Weighted combination of format and accuracy rewards
    :rtype: float
    """
    acc_r = _geo3k_accuracy_reward_fn(sol, gt)
    fmt_r = _geo3k_format_reward_fn(sol, gt)
    return 0.9 * acc_r + 0.1 * fmt_r


RuleReward.register_rule("geo3k_accuracy")(_geo3k_accuracy_reward_fn)
RuleReward.register_rule("geo3k_format")(_geo3k_format_reward_fn)
RuleReward.register_rule("geo3k_combined")(_geo3k_combined_reward_fn)


# ============================================================================
# GSM8K Rule Reward Functions
# ============================================================================

def _gsm8k_accuracy_reward_fn(sol: str, gt: str) -> float:
    """
    GSM8K accuracy reward function.
    
    Extract answer from \\boxed{} notation and use mathruler to verify correctness.
    
    :param sol: Solution string from model (should contain \\boxed{answer})
    :type sol: str
    :param gt: Ground truth answer
    :type gt: str
    :return: 1.0 if answer is correct, 0.0 otherwise
    :rtype: float
    """
    try:
        from mathruler.grader import extract_boxed_content, grade_answer
        pred = extract_boxed_content(sol)
        return 1.0 if grade_answer(pred, gt) else 0.0
    except ImportError:
        return 0.0


def _gsm8k_format_reward_fn(sol: str, gt: str) -> float:
    """
    GSM8K format reward function.
    
    Check if the solution follows the required format:
    - Contains <think>...</think> tags for reasoning
    - Contains \\boxed{} for final answer
    - The think tags must appear BEFORE the boxed answer
    
    :param sol: Solution string from model
    :type sol: str
    :param gt: Ground truth (not used in format check)
    :type gt: str
    :return: 1.0 if format is correct, 0.0 otherwise
    :rtype: float
    """
    sol_stripped = sol.strip()
    
    think_match = re.search(r'<think>.*?</think>', sol_stripped, re.DOTALL)
    boxed_match = re.search(r'\\boxed\{.*?\}', sol_stripped, re.DOTALL)
    
    if think_match and boxed_match:
        think_end = think_match.end()
        boxed_start = boxed_match.start()
        return 1.0 if think_end <= boxed_start else 0.0
    else:
        return 0.0


def _gsm8k_combined_reward_fn(sol: str, gt: str) -> float:
    """
    GSM8K combined reward function.
    
    Combines format reward and accuracy reward with weights.
    Default: 90% accuracy + 10% format.
    
    :param sol: Solution string from model
    :type sol: str
    :param gt: Ground truth answer
    :type gt: str
    :return: Weighted combination of format and accuracy rewards
    :rtype: float
    """
    acc_r = _gsm8k_accuracy_reward_fn(sol, gt)
    fmt_r = _gsm8k_format_reward_fn(sol, gt)
    return 0.9 * acc_r + 0.1 * fmt_r


RuleReward.register_rule("gsm8k_accuracy")(_gsm8k_accuracy_reward_fn)
RuleReward.register_rule("gsm8k_format")(_gsm8k_format_reward_fn)
RuleReward.register_rule("gsm8k_combined")(_gsm8k_combined_reward_fn)

