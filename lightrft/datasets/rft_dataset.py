import random
import copy

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoProcessor

from loguru import logger
from typing import Any, Dict, List, Tuple, Union

from .utils import load_multimodal_content
from lightrft.datasets import (
    RapidataI2VPairHandler, 
    RapidataT2VPairHandler, 
    VideoGenRewardBenchPairHandler,
    HPDv3PairHandler,
    ImageGenCoTRewardHandler,
    OmniRewardBenchT2IPairHandler,
    OmniRewardBenchT2VPairHandler,
    VideoDPOPairHandler,
)


class RFTDatasetVL(Dataset):
    """
    Dataset for Reinforcement Fine-Tuning (RFT) with vision-language models.

    RFTDatasetVL supports multiple data sources through pluggable Data Handlers 
    and is designed for training models using reinforcement learning.

    It loads data items, processes multimodal content (images, videos), and 
    prepares inputs suitable for the model.

    :param dataset_paths: List of dataset file paths or directories. The
        handler is determined by the source keyword (e.g. "rapidata-t2v",
        "videogen-rewardbench"). The format is "source:path".
        e.g. "rapidata-t2v:/path/to/file.parquet"
    :type dataset_paths: List[str]
    :param processor: Multimodal processor used for tokenization and visual
        processing.
    :type processor: transformers.AutoProcessor
    :param tokenizer: Tokenizer used for text tokenization.
    :type tokenizer: transformers.AutoTokenizer
    :param strategy: Optional data loading strategy.
    :type strategy: Any
    :param max_length: Maximum sequence length for tokenization/truncation.
        Defaults to 4096.
    :type max_length: int
    :param is_train: Whether the dataset is used for training or evaluation.
        Defaults to True.
    :type is_train: bool
    :param config: Additional configuration options.
    :type config: Dict[str, Any]
    """
    def __init__(
            self, 
            dataset_paths: List[str], 
            processor: AutoProcessor, 
            tokenizer: AutoTokenizer, 
            strategy = None,
            max_length: int = 4096, 
            is_train: bool = True,
            config: Dict[str, Any] = None,
        ):
        
        super().__init__()
        self.processor = processor
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.is_train = is_train
        self.config = config if config else {}

        self.media_content_loader = load_multimodal_content

        if "qwen" in self.processor.__class__.__name__.lower():
            from qwen_vl_utils import process_vision_info
            self.process_vision_info = process_vision_info
        else:
            raise NotImplementedError(f"Processor type {self.processor.__class__.__name__} not supported yet.")
        
        self.handlers = {
            "rapidata-i2v": RapidataI2VPairHandler(),
            "rapidata-t2v": RapidataT2VPairHandler(),
            "videogen-rewardbench": VideoGenRewardBenchPairHandler(),
            "hpdv3": HPDv3PairHandler(),
            "imagegen-cot-reward-5k": ImageGenCoTRewardHandler(),
            "omnirewardbench-t2i": OmniRewardBenchT2IPairHandler(),
            "omnirewardbench-t2v": OmniRewardBenchT2VPairHandler(),
            "videodpo": VideoDPOPairHandler(),
        }

        # Load data from all specified dataset paths
        # We expect dataset_paths to be in the format: "source:path"
        # e.g. "rapidata-t2v:/path/to/file.parquet"
        self.data = []
        for item in dataset_paths:
            try:
                source, path = item.split(":", 1)
            except ValueError:
                raise ValueError(f"Dataset path '{item}' is not in the expected format 'source:path'.")

            if source not in self.handlers:
                raise NotImplementedError(f"The data handler for source {source} is not implemented.")
            
            handler = self.handlers[source]
            try:
                loaded_items = handler.load_data(path)
                for item in loaded_items:
                    item["source"] = source
                self.data.extend(loaded_items)
            except Exception as e:
                logger.error(f"Failed to load data {path} (source: {source}): {e}")
            
        logger.info(f"Loaded {len(self.data)} items in total, sources: {[s for s in dataset_paths]}")
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source = item["source"]
        
        handler = self.handlers[source]
        
        # Get paths for all media content
        media_info = handler.get_media_info(item)

        # Load all media content at once
        loaded_content = self.media_content_loader(media_info)
        if loaded_content is None:
            raise RuntimeError(f"Failed to load media content: {media_info}")

        # Select prompt based on task type or source if task_instruction is a dict
        config = copy.deepcopy(self.config)
        task_instruction = config.get("task_instruction")
        if isinstance(task_instruction, dict):
            if hasattr(handler, "task_type"):
                prompt = task_instruction.get(handler.task_type)
                if prompt is None:
                   raise ValueError(f"Task instruction for {handler.task_type} not found.")
            else:
                raise ValueError(f"Handler for source {source} does not specify a task_type.")

            config["task_instruction"] = prompt

        # Pass the loaded content dict to parse_item
        messages, reference = handler.parse_item(item, loaded_content, config)

        # Prepare inputs from message sequences
        input_text, image_inputs, video_inputs = self._prepare_inputs(messages)

        # Configure label, by default "general"
        label = "general"

        return input_text, image_inputs, video_inputs, reference, label

    def _prepare_inputs(self, messages):
        if not self.is_train:
            # For evaluation, we only need the input text without generation prompt
            # Remove messages with role "assistant"
            messages = [msg for msg in messages if msg["role"] != "assistant"]

        input_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        image_inputs, video_inputs = self.process_vision_info(messages, return_video_kwargs=False)

        return input_text, image_inputs, video_inputs
    
    def collate_fn(self, batch):
        input_texts, image_inputs_list, video_inputs_list, references, labels = zip(*batch)
        return list(input_texts), list(image_inputs_list), list(video_inputs_list), list(references), list(labels)