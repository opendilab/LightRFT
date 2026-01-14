"""
Reward Model Implementation

This module provides implementations for single and multiple reward models,
encapsulating the logic for computing rewards using neural models.

Main Features:
    - Single reward model wrapper with automatic loading/offloading
    - Multiple reward model ensemble with recipe-based aggregation
    - Support for both standard PyTorch models and custom engine models
    - Consistent interface with BaseReward

Classes:
    SingleRewardModel: Wrapper for single reward model
    MultiRewardModel: Ensemble of multiple reward models with aggregation
"""

from typing import Dict, Sequence, Optional, List, Tuple, Any

import torch
import torch.nn as nn

from .base import BaseReward


class SingleRewardModel(BaseReward):
    """
    Single reward model implementation.

    This class encapsulates the logic for computing rewards using a single
    neural reward model. It handles both standard PyTorch models and custom
    engine models (e.g., SGLang).

    :param reward_model: PyTorch reward model instance
    :type reward_model: nn.Module
    :param tokenizer: Tokenizer for decoding sequences
    :type tokenizer: Any
    :param strategy: Training strategy (for model loading/offloading)
    :type strategy: Any
    :param packing_samples: Whether samples are packed. Default to False
    :type packing_samples: bool
    :param device: Device to place reward tensors on
    :type device: Optional[torch.device]
    """
    def __init__(
        self,
        reward_model: nn.Module,
        tokenizer: Any,
        strategy: Any,
        packing_samples: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize single reward model.

        :param reward_model: PyTorch reward model instance
        :type reward_model: nn.Module
        :param tokenizer: Tokenizer for decoding sequences
        :type tokenizer: Any
        :param strategy: Training strategy (for model loading/offloading)
        :type strategy: Any
        :param packing_samples: Whether samples are packed
        :type packing_samples: bool
        :param device: Device to place reward tensors on
        :type device: Optional[torch.device]
        """
        super().__init__()
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.packing_samples = packing_samples
        self.device = device or torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")

    def compute(
        self,
        queries: Sequence[str],
        references: Optional[Sequence[str]] = None,
        labels: Optional[Sequence[str]] = None,
        sequences: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_and_output: Optional[Sequence[str]] = None,
        raw_images: Optional[List] = None,
        img_num: Optional[List[int]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute rewards using a single reward model.

        :param queries: List of query/solution strings (length B)
        :type queries: Sequence[str]
        :param references: List of reference answers (length B), optional
        :type references: Optional[Sequence[str]]
        :param labels: Not used for single RM, kept for interface consistency
        :type labels: Optional[Sequence[str]]
        :param sequences: Token ID sequences [B, seq_len], optional
        :type sequences: Optional[torch.Tensor]
        :param attention_mask: Attention mask for sequences, optional
        :type attention_mask: Optional[torch.Tensor]
        :param prompt_and_output: List of prompt+output strings, optional
        :type prompt_and_output: Optional[Sequence[str]]
        :param raw_images: List of PIL images, optional
        :type raw_images: Optional[List]
        :param img_num: List of image counts per sample, optional
        :type img_num: Optional[List[int]]
        :param kwargs: Additional arguments for reward model forward pass
        :return: Tuple of (rewards, metrics) where rewards is torch.Tensor of shape (B,)
                 and metrics contains 'model_reward'
        :rtype: Tuple[torch.Tensor, Dict[str, torch.Tensor]]
        """
        # Load model to GPU if needed
        if isinstance(self.reward_model, torch.nn.Module):
            self.strategy.reload_model(self.reward_model)

        # Prepare inputs
        if sequences is not None:
            # Standard PyTorch model path
            rm_output = self.reward_model(
                sequences,
                attention_mask,
                prompt_and_output=prompt_and_output,
                raw_images=raw_images,
                img_num=img_num,
                **kwargs
            )
        else:
            # Custom engine model path
            rm_output = self.reward_model(
                None,
                None,
                prompt_and_outputs=prompt_and_output if prompt_and_output else queries,
                raw_images=raw_images,
                img_num=img_num,
                references=references,
                labels=labels,
                **kwargs
            )

        # Extract scores
        if isinstance(rm_output, dict):
            scores = rm_output["score"]
        else:
            scores = rm_output

        # Ensure tensor format
        if not isinstance(scores, torch.Tensor):
            scores = torch.as_tensor(scores, dtype=torch.float32, device=self.device)
        else:
            scores = scores.to(self.device)

        # Offload model after use
        if isinstance(self.reward_model, torch.nn.Module):
            self.strategy.offload_model(self.reward_model)

        # Create metrics
        metrics = {
            'model_reward': scores.clone(),
        }

        return scores, metrics


class MultiRewardModel(BaseReward):
    """
    Multiple reward model implementation.

    This class encapsulates the logic for computing rewards using multiple
    reward models and aggregating them according to a recipe configuration.

    :param reward_models: List of reward model instances
    :type reward_models: List[nn.Module]
    :param reward_tokenizers: List of corresponding tokenizers
    :type reward_tokenizers: List[Any]
    :param reward_fn: Aggregation function for combining rewards
    :type reward_fn: Any
    :param reward_fn_label_map: Mapping from reward type to model index
    :type reward_fn_label_map: Dict[str, int]
    :param reward_recipe: Recipe configuration for combining rewards
    :type reward_recipe: Dict[str, List[Tuple[str, Optional[str], float]]]
    :param tokenizer: Tokenizer for decoding sequences
    :type tokenizer: Any
    :param strategy: Training strategy (for model loading/offloading)
    :type strategy: Any
    :param packing_samples: Whether samples are packed. Default to False
    :type packing_samples: bool
    :param device: Device to place reward tensors on
    :type device: Optional[torch.device]
    """
    def __init__(
        self,
        reward_models: List[nn.Module],
        reward_tokenizers: List[Any],
        reward_fn: Any,
        reward_fn_label_map: Dict[str, int],
        reward_recipe: Dict[str, List[Tuple[str, Optional[str], float]]],
        tokenizer: Any,
        strategy: Any,
        packing_samples: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize multiple reward models.

        :param reward_models: List of reward model instances
        :type reward_models: List[nn.Module]
        :param reward_tokenizers: List of corresponding tokenizers
        :type reward_tokenizers: List[Any]
        :param reward_fn: Aggregation function for combining rewards
        :type reward_fn: Any
        :param reward_fn_label_map: Mapping from reward type to model index
        :type reward_fn_label_map: Dict[str, int]
        :param reward_recipe: Recipe configuration for combining rewards
        :type reward_recipe: Dict[str, List[Tuple[str, Optional[str], float]]]
        :param tokenizer: Tokenizer for decoding sequences
        :type tokenizer: Any
        :param strategy: Training strategy (for model loading/offloading)
        :type strategy: Any
        :param packing_samples: Whether samples are packed
        :type packing_samples: bool
        :param device: Device to place reward tensors on
        :type device: Optional[torch.device]
        """
        super().__init__()
        self.reward_models = reward_models
        self.reward_tokenizers = reward_tokenizers
        self.reward_fn = reward_fn
        self.reward_fn_label_map = reward_fn_label_map
        self.reward_recipe = reward_recipe
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.packing_samples = packing_samples
        self.device = device or torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")

    def compute(
        self,
        queries: Sequence[str],
        references: Optional[Sequence[str]] = None,
        labels: Optional[Sequence[str]] = None,
        sequences: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_and_output: Optional[Sequence[str]] = None,
        raw_images: Optional[List] = None,
        img_num: Optional[List[int]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute rewards using multiple reward models and aggregate them.

        :param queries: List of query/solution strings (length B)
        :type queries: Sequence[str]
        :param references: List of reference answers (length B), optional
        :type references: Optional[Sequence[str]]
        :param labels: List of data labels indicating reward type (length B), required
        :type labels: Optional[Sequence[str]]
        :param sequences: Token ID sequences [B, seq_len], optional
        :type sequences: Optional[torch.Tensor]
        :param attention_mask: Attention mask for sequences, optional
        :type attention_mask: Optional[torch.Tensor]
        :param prompt_and_output: List of prompt+output strings, optional
        :type prompt_and_output: Optional[Sequence[str]]
        :param raw_images: List of PIL images, optional
        :type raw_images: Optional[List]
        :param img_num: List of image counts per sample, optional
        :type img_num: Optional[List[int]]
        :param kwargs: Additional arguments for reward model forward pass
        :return: Tuple of (rewards, metrics) where rewards is torch.Tensor of shape (B,)
                 and metrics contains detailed reward metrics
        :rtype: Tuple[torch.Tensor, Dict[str, torch.Tensor]]
        :raises ValueError: If labels are not provided
        """
        if labels is None:
            raise ValueError("labels are required for MultiRewardModel")

        B = len(queries)

        # Load all models to GPU
        for rm in self.reward_models:
            if isinstance(rm, torch.nn.Module):
                self.strategy.reload_model(rm)

        # Compute rewards for each RM
        model_reward_list = []

        for rm_idx, rm in enumerate(self.reward_models):
            # Check if this is a custom engine model
            is_custom_engine = (
                isinstance(rm, torch.nn.Module) and hasattr(rm, "base_model")
                and not isinstance(rm.base_model, torch.nn.Module)
            )

            if is_custom_engine:
                # Custom engine model path
                rm_output = rm(
                    None,
                    None,
                    prompt_and_outputs=prompt_and_output if prompt_and_output else queries,
                    raw_images=raw_images,
                    img_num=img_num,
                    references=references,
                    labels=labels,
                    **kwargs
                )
            else:
                # Standard PyTorch model path
                rm_output = rm(
                    sequences,
                    attention_mask,
                    prompt_and_output=prompt_and_output,
                    raw_images=raw_images,
                    img_num=img_num,
                    **kwargs
                )

            # Extract scores
            if isinstance(rm_output, dict):
                scores = rm_output["score"]
            else:
                scores = rm_output

            # Ensure tensor format
            if not isinstance(scores, torch.Tensor):
                scores = torch.as_tensor(scores, dtype=torch.float32, device=self.device)
            else:
                scores = scores.to(self.device)

            model_reward_list.append(scores)

            # Offload model after use
            if isinstance(rm, torch.nn.Module):
                self.strategy.offload_model(rm)

        # Aggregate rewards using reward_fn
        rewards, reward_metrics = self.reward_fn(
            model_reward_list=model_reward_list,
            labels=labels,
            queries=queries,
            refs=references if references else [""] * B,
            label_map=self.reward_fn_label_map,
        )

        # Ensure rewards are on correct device
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        else:
            rewards = rewards.to(self.device)

        # Ensure metrics are tensors
        if reward_metrics is not None:
            for key, value in reward_metrics.items():
                if not isinstance(value, torch.Tensor):
                    reward_metrics[key] = torch.as_tensor(value, dtype=torch.float32, device=self.device)
                else:
                    reward_metrics[key] = value.to(self.device)
        else:
            reward_metrics = {}

        return rewards, reward_metrics
