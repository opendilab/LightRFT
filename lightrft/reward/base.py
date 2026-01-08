"""
Base Reward Interface

This module defines the abstract base class for all reward types in LightRLHF,
ensuring a consistent interface across rule-based rewards, single reward models,
and multiple reward model ensembles.

Main Features:
    - Unified compute() method signature for all reward types
    - Consistent return format: (rewards, metrics)
    - Support for queries, references, and labels

Classes:
    BaseReward: Abstract base class for all reward implementations

Author: lightrft Team
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Sequence, Optional

import torch


class BaseReward(ABC):
    """
    Abstract base class for all reward types.
    
    This class defines the unified interface that all reward implementations
    must follow, ensuring consistency across rule-based rewards, single reward
    models, and multiple reward models.
    
    All reward implementations should return:
        - rewards: torch.Tensor of shape (batch_size,) containing reward values
        - metrics: Dict[str, torch.Tensor] containing detailed reward metrics
    """
    @abstractmethod
    def compute(
        self,
        queries: Sequence[str],
        references: Optional[Sequence[str]] = None,
        labels: Optional[Sequence[str]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute rewards for given queries.
        
        :param queries: List of query/solution strings (length B)
        :type queries: Sequence[str]
        :param references: List of reference answers (length B), optional
        :type references: Optional[Sequence[str]]
        :param labels: List of data labels indicating reward type (length B), optional
        :type labels: Optional[Sequence[str]]
        :param kwargs: Additional arguments for specific reward types
        :return: Tuple of (rewards, metrics) where rewards is torch.Tensor of shape (B,)
                 and metrics is Dict[str, torch.Tensor] with keys like 'format_reward',
                 'accuracy_reward', 'model_reward', 'rule_reward'
        :rtype: Tuple[torch.Tensor, Dict[str, torch.Tensor]]
        """
        pass

    def __call__(
        self,
        queries: Sequence[str],
        references: Optional[Sequence[str]] = None,
        labels: Optional[Sequence[str]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Make the reward object callable.
        
        :param queries: List of query/solution strings
        :type queries: Sequence[str]
        :param references: List of reference answers, optional
        :type references: Optional[Sequence[str]]
        :param labels: List of data labels, optional
        :type labels: Optional[Sequence[str]]
        :param kwargs: Additional arguments
        :return: Tuple of (rewards, metrics)
        :rtype: Tuple[torch.Tensor, Dict[str, torch.Tensor]]
        """
        return self.compute(queries, references, labels, **kwargs)
