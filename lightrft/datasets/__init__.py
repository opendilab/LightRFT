"""
Dataset Module for LightRFT

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

# Import other dataset classes
from .grm_dataset import GRMDataset
from .srm_dataset import RankDatasetVL, RankDatasetAL
from .omnirewardbench import *
from .imagegen_cot_reward import *
from .rapidata import *
from .image_reward_db import *
from .hpdv3 import *
from .videodpo import *
from .videogen_rewardbench import *
from .genai_bench import *
from .rft_dataset import RFTDatasetVL
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
    "RFTDatasetVL",
]
