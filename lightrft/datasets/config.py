"""
Dataset Configuration

This module provides configuration classes for dataset loading,
unifying parameters for train, eval, and pretrain datasets.

Main Features:
    - Unified configuration for all dataset types
    - Automatic normalization of data_path and data_probs
    - Factory methods for train/eval/pretrain configurations
    - Validation of configuration parameters

Classes:
    DatasetConfig: Dataclass for dataset configuration

Author: lightrft Team
"""

from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class DatasetConfig:
    """
    Configuration for dataset loading.

    This class unifies parameters for train, eval, and pretrain datasets,
    providing a consistent interface for dataset configuration.

    :param data_path: Path(s) to dataset(s), can be string or list
    :type data_path: Optional[Union[str, list]]
    :param data_probs: Sampling probabilities for datasets. Default to "1.0"
    :type data_probs: Optional[Union[str, list]]
    :param split: Dataset split to use. Default to "train"
    :type split: str
    :param max_samples: Maximum number of samples to load
    :type max_samples: Optional[int]
    :param max_len: Maximum sequence length
    :type max_len: Optional[int]
    :param seed: Random seed. Default to 42
    :type seed: int
    :param return_eval: Whether to return evaluation data. Default to False
    :type return_eval: bool
    """

    # Data source
    data_path: Optional[Union[str, list]] = None
    data_probs: Optional[Union[str, list]] = "1.0"

    # Split configuration
    split: str = "train"

    # Data filtering
    max_samples: Optional[int] = None
    max_len: Optional[int] = None

    # Additional parameters
    seed: int = 42
    return_eval: bool = False

    def __post_init__(self):
        """
        Validate configuration after initialization.

        :raises ValueError: If data_path is None or if data_path and data_probs have mismatched lengths
        """
        if self.data_path is None:
            raise ValueError("data_path must be specified")

        # Normalize data_probs
        if isinstance(self.data_probs, str):
            # Parse comma-separated string
            self.data_probs = [float(p.strip()) for p in self.data_probs.split(",")]
        elif isinstance(self.data_probs, (int, float)):
            self.data_probs = [float(self.data_probs)]

        # Normalize data_path
        if isinstance(self.data_path, str):
            self.data_path = [self.data_path]

        # Ensure data_path and data_probs have same length
        if len(self.data_probs) == 1 and len(self.data_path) > 1:
            # Repeat single prob for all paths
            self.data_probs = self.data_probs * len(self.data_path)
        elif len(self.data_probs) != len(self.data_path):
            raise ValueError(
                f"data_path and data_probs must have the same length, "
                f"got {len(self.data_path)} and {len(self.data_probs)}"
            )

    @classmethod
    def for_train(
        cls,
        data_path: Optional[Union[str, list]] = None,
        data_probs: Optional[Union[str, list]] = "1.0",
        split: str = "train",
        max_samples: Optional[int] = None,
        max_len: Optional[int] = None,
        seed: int = 42,
    ) -> "DatasetConfig":
        """
        Create configuration for training dataset.

        :param data_path: Path(s) to dataset(s)
        :type data_path: Optional[Union[str, list]]
        :param data_probs: Sampling probabilities for datasets
        :type data_probs: Optional[Union[str, list]]
        :param split: Dataset split to use
        :type split: str
        :param max_samples: Maximum number of samples to load
        :type max_samples: Optional[int]
        :param max_len: Maximum sequence length
        :type max_len: Optional[int]
        :param seed: Random seed
        :type seed: int
        :return: DatasetConfig instance for training
        :rtype: DatasetConfig
        """
        return cls(
            data_path=data_path,
            data_probs=data_probs,
            split=split,
            max_samples=max_samples,
            max_len=max_len,
            seed=seed,
            return_eval=False,
        )

    @classmethod
    def for_eval(
        cls,
        data_path: Optional[Union[str, list]] = None,
        data_probs: Optional[Union[str, list]] = "1.0",
        split: str = "test",
        max_samples: Optional[int] = None,
        max_len: Optional[int] = None,
        seed: int = 42,
    ) -> "DatasetConfig":
        """
        Create configuration for evaluation dataset.

        :param data_path: Path(s) to dataset(s)
        :type data_path: Optional[Union[str, list]]
        :param data_probs: Sampling probabilities for datasets
        :type data_probs: Optional[Union[str, list]]
        :param split: Dataset split to use
        :type split: str
        :param max_samples: Maximum number of samples to load
        :type max_samples: Optional[int]
        :param max_len: Maximum sequence length
        :type max_len: Optional[int]
        :param seed: Random seed
        :type seed: int
        :return: DatasetConfig instance for evaluation
        :rtype: DatasetConfig
        """
        return cls(
            data_path=data_path,
            data_probs=data_probs,
            split=split,
            max_samples=max_samples,
            max_len=max_len,
            seed=seed,
            return_eval=False,
        )

    @classmethod
    def for_pretrain(
        cls,
        data_path: Optional[Union[str, list]] = None,
        data_probs: Optional[Union[str, list]] = "1.0",
        split: str = "train",
        max_samples: Optional[int] = None,
        max_len: Optional[int] = None,
        seed: int = 42,
    ) -> "DatasetConfig":
        """
        Create configuration for pretraining dataset.

        :param data_path: Path(s) to dataset(s)
        :type data_path: Optional[Union[str, list]]
        :param data_probs: Sampling probabilities for datasets
        :type data_probs: Optional[Union[str, list]]
        :param split: Dataset split to use
        :type split: str
        :param max_samples: Maximum number of samples to load
        :type max_samples: Optional[int]
        :param max_len: Maximum sequence length
        :type max_len: Optional[int]
        :param seed: Random seed
        :type seed: int
        :return: DatasetConfig instance for pretraining
        :rtype: DatasetConfig
        """
        return cls(
            data_path=data_path,
            data_probs=data_probs,
            split=split,
            max_samples=max_samples,
            max_len=max_len,
            seed=seed,
            return_eval=False,
        )
