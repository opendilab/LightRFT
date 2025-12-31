"""
Dataset Loader

This module provides a unified interface for loading datasets for training,
evaluation, and pretraining, abstracting away the differences between
different dataset types and splits.

Main Features:
    - Unified loading interface for train, eval, and pretrain datasets
    - Automatic handling of blending_datasets parameters
    - Support for PromptDatasetVL and SFTDatasetVL
    - Consistent logging via strategy

Classes:
    DatasetLoader: Unified loader for all dataset types

Author: lightrft Team
"""

from typing import Optional, Any

from lightrft.utils import blending_datasets
from .prompts_dataset_vl import PromptDatasetVL
from .sft_dataset_vl import SFTDatasetVL
from .config import DatasetConfig


class DatasetLoader:
    """
    Unified dataset loader for train, eval, and pretrain datasets.
    
    This class provides a consistent interface for loading datasets,
    handling the differences between prompt datasets (for training/eval)
    and SFT datasets (for pretraining).
    
    :param tokenizer: Tokenizer for tokenizing text
    :type tokenizer: Any
    :param processor: Processor for multimodal data (optional)
    :type processor: Optional[Any]
    :param strategy: Training strategy (optional, for logging)
    :type strategy: Optional[Any]
    """
    
    def __init__(
        self,
        tokenizer: Any,
        processor: Optional[Any] = None,
        strategy: Optional[Any] = None,
    ):
        """
        Initialize dataset loader.
        
        :param tokenizer: Tokenizer for tokenizing text
        :type tokenizer: Any
        :param processor: Processor for multimodal data (optional)
        :type processor: Optional[Any]
        :param strategy: Training strategy (optional, for logging)
        :type strategy: Optional[Any]
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.strategy = strategy
    
    def _log(self, message: str):
        """
        Log message if strategy is available.
        
        :param message: Message to log
        :type message: str
        """
        if self.strategy:
            self.strategy.print(message)
        else:
            print(message)
    
    def load_train_dataset(
        self,
        config: DatasetConfig,
        prompt_max_len: int,
        input_template: Optional[str] = None,
    ) -> PromptDatasetVL:
        """
        Load training dataset.
        
        :param config: Dataset configuration
        :type config: DatasetConfig
        :param prompt_max_len: Maximum prompt length
        :type prompt_max_len: int
        :param input_template: Input template for formatting prompts
        :type input_template: Optional[str]
        :return: PromptDatasetVL instance for training
        :rtype: PromptDatasetVL
        """
        # Convert data_path list to comma-separated string for blending_datasets
        data_path_str = config.data_path if isinstance(config.data_path, str) else ",".join(config.data_path)
        self._log(f"Loading training dataset from: {data_path_str} with split: {config.split}")
        
        # Load and blend datasets
        data = blending_datasets(
            data_path_str,
            ",".join(map(str, config.data_probs)),
            self.strategy,
            config.seed,
            return_eval=config.return_eval,
            train_split=config.split,
        )
        
        # Limit samples if specified
        if config.max_samples is not None:
            data = data.select(range(min(config.max_samples, len(data))))
        
        self._log(f"Loaded {len(data)} samples for training.")
        
        # Create dataset
        dataset = PromptDatasetVL(
            data,
            self.tokenizer,
            self.processor,
            prompt_max_len,
            self.strategy,
            input_template=input_template,
        )
        
        return dataset
    
    def load_eval_dataset(
        self,
        config: DatasetConfig,
        prompt_max_len: int,
        input_template: Optional[str] = None,
    ) -> Optional[PromptDatasetVL]:
        """
        Load evaluation dataset.
        
        :param config: Dataset configuration
        :type config: DatasetConfig
        :param prompt_max_len: Maximum prompt length
        :type prompt_max_len: int
        :param input_template: Input template for formatting prompts
        :type input_template: Optional[str]
        :return: PromptDatasetVL instance for evaluation, or None if no data
        :rtype: Optional[PromptDatasetVL]
        """
        if config.data_path is None:
            return None
        
        # Convert data_path list to comma-separated string for blending_datasets
        data_path_str = config.data_path if isinstance(config.data_path, str) else ",".join(config.data_path)
        self._log(f"Loading evaluation dataset from {data_path_str}, split='{config.split}'")
        
        # Load and blend datasets
        data = blending_datasets(
            data_path_str,
            ",".join(map(str, config.data_probs)),
            self.strategy,
            config.seed,
            return_eval=config.return_eval,
            train_split=config.split,
        )
        
        if len(data) == 0:
            self._log(
                f"Warning: Evaluation dataset at {data_path_str} with split '{config.split}' "
                "is empty. Skipping evaluation."
            )
            return None
        
        # Limit samples if specified
        if config.max_samples is not None:
            data = data.select(range(min(config.max_samples, len(data))))
        
        self._log(f"Evaluation dataset loaded: {len(data)} samples")
        
        # Create dataset
        dataset = PromptDatasetVL(
            data,
            self.tokenizer,
            self.processor,
            prompt_max_len,
            self.strategy,
            input_template=input_template,
        )
        
        return dataset
    
    def load_pretrain_dataset(
        self,
        config: DatasetConfig,
        pretrain_max_len: int,
    ) -> Optional[SFTDatasetVL]:
        """
        Load pretraining dataset.
        
        :param config: Dataset configuration
        :type config: DatasetConfig
        :param pretrain_max_len: Maximum sequence length for pretraining
        :type pretrain_max_len: int
        :return: SFTDatasetVL instance for pretraining, or None if no data
        :rtype: Optional[SFTDatasetVL]
        """
        if config.data_path is None:
            return None
        
        # Convert data_path list to comma-separated string for blending_datasets
        data_path_str = config.data_path if isinstance(config.data_path, str) else ",".join(config.data_path)
        self._log(f"Loading pretrain dataset from: {data_path_str} with split: {config.split}")
        
        # Load and blend datasets
        data = blending_datasets(
            data_path_str,
            ",".join(map(str, config.data_probs)),
            self.strategy,
            config.seed,
            return_eval=config.return_eval,
            train_split=config.split,
        )
        
        if len(data) == 0:
            self._log(
                f"Warning: Pretrain dataset at {data_path_str} is empty. "
                "PTX loss will not be applied."
            )
            return None
        
        # Limit samples if specified
        if config.max_samples is not None:
            data = data.select(range(min(config.max_samples, len(data))))
        
        self._log(f"Loaded {len(data)} samples for pretraining.")
        
        # Create dataset
        dataset = SFTDatasetVL(
            data,
            self.tokenizer,
            pretrain_max_len,
            self.strategy,
            pretrain_mode=True,
        )
        
        return dataset

