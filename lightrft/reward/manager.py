"""
Reward Manager

This module provides a unified manager for different reward types,
automatically selecting and using the appropriate reward implementation
based on configuration.

Main Features:
    - Automatic reward type selection (rule, single, multi)
    - Unified compute_rewards() interface
    - Configuration-based initialization
    - Support for factory pattern via from_config()

Classes:
    RewardManager: Unified manager for all reward types

Author: lightrft Team
"""

from typing import Dict, Sequence, Optional, List, Tuple, Any, Union

import torch

from .base import BaseReward
from .rule import RuleReward
from .model import SingleRewardModel, MultiRewardModel


class RewardManager(BaseReward):
    """
    Unified reward manager that automatically selects the appropriate
    reward implementation based on configuration.

    Supports:
    - Rule-based rewards (pure rule functions)
    - Single reward model
    - Multiple reward models (with recipe-based aggregation)

    :param reward_type: Type of reward to use ("rule", "single", "multi")
    :type reward_type: str
    :param reward_model: Single reward model or list of reward models
    :type reward_model: Optional[Union[Any, List[Any]]]
    :param reward_tokenizers: List of tokenizers for reward models
    :type reward_tokenizers: Optional[List[Any]]
    :param reward_fn: Aggregation function for multiple reward models
    :type reward_fn: Optional[Any]
    :param reward_fn_label_map: Mapping from reward type to model index
    :type reward_fn_label_map: Optional[Dict[str, int]]
    :param reward_recipe: Recipe configuration for combining rewards
    :type reward_recipe: Optional[Dict[str, List[Tuple[str, Optional[str], float]]]]
    :param rule_type: Type of rule reward (e.g., "geo3k_combined", "gsm8k_combined")
    :type rule_type: Optional[str]
    :param tokenizer: Tokenizer for decoding sequences
    :type tokenizer: Optional[Any]
    :param strategy: Training strategy (for model loading/offloading)
    :type strategy: Optional[Any]
    :param packing_samples: Whether samples are packed. Default to False
    :type packing_samples: bool
    :param device: Device to place reward tensors on
    :type device: Optional[torch.device]
    """
    def __init__(
        self,
        reward_type: str = "multi",  # "rule", "single", "multi"
        reward_model: Optional[Union[Any, List[Any]]] = None,
        reward_tokenizers: Optional[List[Any]] = None,
        reward_fn: Optional[Any] = None,
        reward_fn_label_map: Optional[Dict[str, int]] = None,
        reward_recipe: Optional[Dict[str, List[Tuple[str, Optional[str], float]]]] = None,
        rule_type: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        strategy: Optional[Any] = None,
        packing_samples: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize reward manager.

        :param reward_type: Type of reward to use ("rule", "single", "multi")
        :type reward_type: str
        :param reward_model: Single reward model or list of reward models
        :type reward_model: Optional[Union[Any, List[Any]]]
        :param reward_tokenizers: List of tokenizers for reward models
        :type reward_tokenizers: Optional[List[Any]]
        :param reward_fn: Aggregation function for multiple reward models
        :type reward_fn: Optional[Any]
        :param reward_fn_label_map: Mapping from reward type to model index
        :type reward_fn_label_map: Optional[Dict[str, int]]
        :param reward_recipe: Recipe configuration for combining rewards
        :type reward_recipe: Optional[Dict[str, List[Tuple[str, Optional[str], float]]]]
        :param rule_type: Type of rule reward (e.g., "geo3k_combined", "gsm8k_combined")
        :type rule_type: Optional[str]
        :param tokenizer: Tokenizer for decoding sequences
        :type tokenizer: Optional[Any]
        :param strategy: Training strategy (for model loading/offloading)
        :type strategy: Optional[Any]
        :param packing_samples: Whether samples are packed
        :type packing_samples: bool
        :param device: Device to place reward tensors on
        :type device: Optional[torch.device]
        :raises ValueError: If required parameters are missing for the specified reward_type
        """
        super().__init__()
        self.reward_type = reward_type
        self.device = device or torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")

        # Initialize the appropriate reward implementation
        if reward_type == "rule":
            if rule_type is None:
                raise ValueError("rule_type must be specified for rule-based rewards")
            self.reward_impl = RuleReward(
                rule_type=rule_type,
                device=self.device,
            )
        elif reward_type == "single":
            if reward_model is None:
                raise ValueError("reward_model must be specified for single reward model")
            if tokenizer is None:
                raise ValueError("tokenizer must be specified for single reward model")
            if strategy is None:
                raise ValueError("strategy must be specified for single reward model")

            # Ensure reward_model is a single model, not a list
            if isinstance(reward_model, (list, tuple)):
                if len(reward_model) != 1:
                    raise ValueError("reward_model must be a single model for reward_type='single'")
                reward_model = reward_model[0]

            self.reward_impl = SingleRewardModel(
                reward_model=reward_model,
                tokenizer=tokenizer,
                strategy=strategy,
                packing_samples=packing_samples,
                device=self.device,
            )
        elif reward_type == "multi":
            if reward_model is None:
                raise ValueError("reward_model must be specified for multiple reward models")
            if reward_fn is None:
                raise ValueError("reward_fn must be specified for multiple reward models")
            if tokenizer is None:
                raise ValueError("tokenizer must be specified for multiple reward models")
            if strategy is None:
                raise ValueError("strategy must be specified for multiple reward models")

            # Ensure reward_model is a list
            if not isinstance(reward_model, (list, tuple)):
                reward_model = [reward_model]

            self.reward_impl = MultiRewardModel(
                reward_models=reward_model,
                reward_tokenizers=reward_tokenizers or [],
                reward_fn=reward_fn,
                reward_fn_label_map=reward_fn_label_map or {},
                reward_recipe=reward_recipe or {},
                tokenizer=tokenizer,
                strategy=strategy,
                packing_samples=packing_samples,
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown reward_type: {reward_type}. Must be 'rule', 'single', or 'multi'")

    def compute(
        self,
        queries: Sequence[str],
        references: Optional[Sequence[str]] = None,
        labels: Optional[Sequence[str]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute rewards using the configured reward implementation.

        :param queries: List of query/solution strings (length B)
        :type queries: Sequence[str]
        :param references: List of reference answers (length B), optional
        :type references: Optional[Sequence[str]]
        :param labels: List of data labels indicating reward type (length B), optional
        :type labels: Optional[Sequence[str]]
        :param kwargs: Additional arguments passed to the reward implementation
        :return: Tuple of (rewards, metrics) where rewards is torch.Tensor of shape (B,)
                 and metrics contains detailed reward metrics
        :rtype: Tuple[torch.Tensor, Dict[str, torch.Tensor]]
        """
        return self.reward_impl.compute(queries, references, labels, **kwargs)

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        reward_models: Optional[Union[Any, List[Any]]] = None,
        reward_tokenizers: Optional[List[Any]] = None,
        tokenizer: Optional[Any] = None,
        strategy: Optional[Any] = None,
    ) -> "RewardManager":
        """
        Create RewardManager from configuration dictionary.

        :param config: Configuration dictionary with keys:
            - reward_type: "rule", "single", or "multi"
            - rule_type: Type of rule (for rule-based rewards)
            - reward_fn: Aggregation function (for multi rewards)
            - reward_fn_label_map: Label map (for multi rewards)
            - reward_recipe: Recipe config (for multi rewards)
            - packing_samples: Whether samples are packed
            - device: Device to use
        :type config: Dict[str, Any]
        :param reward_models: Reward model(s) to use
        :type reward_models: Optional[Union[Any, List[Any]]]
        :param reward_tokenizers: Tokenizers for reward models
        :type reward_tokenizers: Optional[List[Any]]
        :param tokenizer: Tokenizer for decoding sequences
        :type tokenizer: Optional[Any]
        :param strategy: Training strategy
        :type strategy: Optional[Any]
        :return: RewardManager instance
        :rtype: RewardManager
        """
        return cls(
            reward_type=config.get("reward_type", "multi"),
            reward_model=reward_models,
            reward_tokenizers=reward_tokenizers,
            reward_fn=config.get("reward_fn"),
            reward_fn_label_map=config.get("reward_fn_label_map"),
            reward_recipe=config.get("reward_recipe"),
            rule_type=config.get("rule_type"),
            tokenizer=tokenizer,
            strategy=strategy,
            packing_samples=config.get("packing_samples", False),
            device=config.get("device"),
        )
