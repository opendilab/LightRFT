"""
Dataset Module for LightRLHF

This module provides unified interfaces for loading datasets for training,
evaluation, and pretraining in RLHF workflows.

Main Features:
    - Unified dataset configuration via DatasetConfig
    - Consistent loading interface via DatasetLoader
    - Support for train, eval, and pretrain datasets
    - Automatic handling of blending_datasets parameters

Classes:
    DatasetConfig: Configuration class for dataset loading
    DatasetLoader: Unified loader for all dataset types
"""

# Import new unified interfaces first
from .config import DatasetConfig
from .loader import DatasetLoader

# Import existing dataset classes
from .process_reward_dataset import ProcessRewardDataset
from .prompts_dataset import PromptDataset
from .prompts_dataset_vl import PromptDatasetVL
from .sft_dataset import SFTDataset
from .sft_dataset_vl import SFTDatasetVL

# Import other dataset classes (may have optional dependencies)
try:
    from .grm_dataset import GRMDataset
except ImportError:
    GRMDataset = None

try:
    from .srm_dataset import RankDatasetVL, RankDatasetAL
except ImportError:
    RankDatasetVL = None
    RankDatasetAL = None

try:
    from .omnirewardbench import *
except ImportError:
    pass

try:
    from .imagegen_cot_reward import *
except ImportError:
    pass

try:
    from .rapidata import *
except ImportError:
    pass

try:
    from .image_reward_db import *
except ImportError:
    pass

try:
    from .hpdv3 import *
except ImportError:
    pass

from .utils import (
    extract_answer,
    zero_pad_sequences,
    find_subsequence,
    load_multimodal_content,
    BaseDataHandler,
)

__all__ = [
    "DatasetConfig",
    "DatasetLoader",
    "ProcessRewardDataset",
    "PromptDataset",
    "PromptDatasetVL",
    "SFTDataset",
    "SFTDatasetVL",
]
