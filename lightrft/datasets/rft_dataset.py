import random
import copy

from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoProcessor

from loguru import logger
from typing import Any, Dict, List

from .utils import load_multimodal_content
from lightrft.datasets import (
    RapidataI2VPairHandler,
    RapidataT2VPairHandler,
    VideoGenRewardBenchPairHandler,
    HPDv3PairHandler,
    OmniRewardBenchT2IPairHandler,
    OmniRewardBenchT2VPairHandler,
    VideoDPOPairHandler,
    GenAIBenchPairHandler,
    GenAIBenchVideoPairHandler,
)


class RFTDatasetVL(Dataset):
    """
    Dataset for Reinforcement Fine-Tuning (RFT) with vision-language models.

    RFTDatasetVL supports multiple data sources through pluggable Data Handlers
    and is designed for training models using reinforcement learning.

    It loads data items, processes multimodal content (images, videos), and
    prepares inputs suitable for the model.

    :param dataset_paths: List of dataset file paths or directories. The
        handler is determined by the source keyword (e.g. "source:path").
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
    :param config: Additional configuration options.
    :type config: Dict[str, Any]

    **Example:**

    .. code-block:: python

        from transformers import AutoProcessor, AutoTokenizer
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        dataset = RFTDatasetVL(
            dataset_paths=["hpdv3:/path/to/data.json"],
            processor=processor,
            tokenizer=tokenizer
        )
    """
    def __init__(
        self,
        dataset_paths: List[str],
        processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        strategy=None,
        max_length: int = 4096,
        config: Dict[str, Any] = None,
    ):

        super().__init__()
        self.processor = processor
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
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
            "omnirewardbench-t2i": OmniRewardBenchT2IPairHandler(),
            "omnirewardbench-t2v": OmniRewardBenchT2VPairHandler(),
            "videodpo": VideoDPOPairHandler(),
            "genai_bench": GenAIBenchPairHandler(),
            "genai_bench_video": GenAIBenchVideoPairHandler(),
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
        """
        Return the total number of items in the dataset.

        :return: Number of data items
        :rtype: int
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a specific data item by index.

        Processes multimodal content (images/videos) and prepares the prompt.

        :param idx: Index of the item to retrieve
        :type idx: int

        :return: A tuple containing (input_text, image_inputs, video_inputs, reference, label)
        :rtype: Tuple[str, Optional[List], Optional[List], Any, str]
        """
        item = self.data[idx]
        source = item["source"]

        handler = self.handlers[source]

        # Get paths for all media content
        media_info = handler.get_media_info(item)

        # Load all media content at once
        loaded_content = self.media_content_loader(media_info)
        if loaded_content is None:
            raise RuntimeError(f"Failed to load media content: {media_info}")

        # Pass the loaded content dict to parse_item
        config = copy.deepcopy(self.config)
        messages, reference = handler.parse_item(item, loaded_content, config)

        # Prepare inputs from message sequences
        input_text, image_inputs, video_inputs = self._prepare_inputs(messages)

        # Configure label for reward rule, by default "general"
        # This label is used to identify which reward function or reward model to use
        # for computing rewards during RL training
        label = reference.get("reward_rule_label", "general")

        return input_text, image_inputs, video_inputs, reference, label

    def _prepare_inputs(self, messages):
        """
        Convert messages into formatted input text and process vision information.

        :param messages: List of chat messages (role and content)
        :type messages: List[Dict[str, str]]

        :return: Formatted input string, images, and videos
        :rtype: Tuple[str, Optional[List], Optional[List]]
        """
        # In RL training, no need to keep assistant responses, since it can lead to data leakage
        # Here we check for role "assistant" to remove assistant messages
        messages = [msg for msg in messages if msg["role"] != "assistant"]

        input_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Support Qwen3-VL video metadata
        # Setting return_video_metadata=True will return video_inputs as a list of (video, metadata)
        image_inputs, video_inputs = self.process_vision_info(
            messages, return_video_kwargs=False, return_video_metadata=True
        )

        return input_text, image_inputs, video_inputs

    def collate_fn(self, batch):
        """
        Collate a batch of samples into a list of tuples.

        :param batch: List of samples from __getitem__
        :type batch: List[Tuple]

        :return: Transposed list of batch components
        :rtype: List[List]
        """
        return [list(item) for item in zip(*batch)]
