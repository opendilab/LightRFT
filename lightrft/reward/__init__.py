"""
Reward Module for LightRLHF

This module provides unified interfaces for different types of rewards in RLHF training.
It supports rule-based rewards, single reward models, and multiple reward model ensembles.

Main Features:
    - Unified reward interface with consistent compute() method signature
    - Rule-based reward functions (format checking, accuracy verification)
    - Single and multiple reward model support
    - Automatic reward type selection via RewardManager

Classes:
    BaseReward: Abstract base class for all reward types
    RuleReward: Rule-based reward implementation
    SingleRewardModel: Single reward model wrapper
    MultiRewardModel: Multiple reward model ensemble
    RewardManager: Unified manager for automatic reward type selection

Author: lightrft Team
"""

from .base import BaseReward
from .rule import RuleReward
from .model import SingleRewardModel, MultiRewardModel
from .manager import RewardManager

__all__ = [
    "BaseReward",
    "RuleReward",
    "SingleRewardModel",
    "MultiRewardModel",
    "RewardManager",
]

