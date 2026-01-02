import random
import torch
from torch.utils.data import Dataset
from loguru import logger
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoProcessor

from .omnirewardbench import OmniRewardBenchT2IGRMHandler
from .imagegen_cot_reward import ImageGenCoTRewardHandler
from .hpdv3 import HPDv3GRMHandler
from .utils import zero_pad_sequences, load_multimodal_content, find_subsequence


class GRMDataset(Dataset):
    """
    Dataset for Generative Reward Model (GRM) training.

    GRMDataset supports multiple data sources through pluggable Data Handlers
    and covers both understanding tasks (image-to-text, video-to-text) and
    generation tasks (text-to-image, text-to-video).

    :param dataset_paths: List of dataset file paths or directories. The
        handler is determined by the source keyword (e.g. "hpdv3",
        "imagegen-cot-reward", "omnirewardbench"). The format is "source:path".
        e.g. "hpdv3:/path/to/train.json"
    :type dataset_paths: List[str]
    :param processor: Multimodal processor used for tokenization and visual
        processing.
    :type processor: transformers.AutoProcessor
    :param tokenizer: Tokenizer used for text tokenization (provides
        ``eos_token`` and ``pad_token_id`` attributes).
    :type tokenizer: transformers.AutoTokenizer
    :param strategy: Optional data loading strategy.
    :type strategy: Any
    :param max_length: Maximum sequence length for tokenization/truncation.
    :type max_length: int
    :param config: Additional configuration options. Supported keys include:
        - ``task_instruction`` (str): Instruction for the evaluation task.
        - ``system_prompt_template`` (str): Template for the system prompt
          with a ``{prompt}`` placeholder.
    :type config: Dict[str, Any]
    :param is_training: Whether the dataset is used for training (returns
        labels) or evaluation (no labels returned).
    :type is_training: bool

    :example:

        >>> dataset = GRMDataset([
        ...     'imagegen-cot-reward-5k:/data/imagegen-cot-reward-5k/train.json'
        ... ], processor=proc, tokenizer=tok, max_length=4096, is_training=True)

    """
    def __init__(
        self,
        dataset_paths: List[str],
        processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        strategy=None,
        max_length: int = 4096,
        config: Dict[str, Any] = None,
        is_training: bool = True
    ):

        super().__init__()
        self.processor = processor
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.config = config if config else {}
        self.is_training = is_training

        self.config["is_training"] = is_training

        self.media_content_loader = load_multimodal_content

        if "qwen" in self.processor.__class__.__name__.lower():
            from qwen_vl_utils import process_vision_info
            self.process_vision_info = process_vision_info
        elif "keye" in self.processor.__class__.__name__.lower():
            from keye_vl_utils import process_vision_info
            self.process_vision_info = process_vision_info
        else:
            raise NotImplementedError(f"Processor type {self.processor.__class__.__name__} not supported yet.")

        self.handlers = {
            "imagegen-cot-reward-5k": ImageGenCoTRewardHandler(),
            "omnirewardbench-t2i": OmniRewardBenchT2IGRMHandler(),
            "hpdv3": HPDv3GRMHandler(),
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

        logger.info(f"Loaded {len(self.data)} items in total, sources: {list(dataset_paths)}")
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

        # Pass the loaded content dict to parse_item
        messages, other = handler.parse_item(item, loaded_content, self.config)

        # Tokenize the message
        if self.is_training:
            input_token, labels = self._tokenize_msg_for_training(messages)
            return input_token, labels, other
        else:
            input_token = self._tokenize_msg_for_eval(messages)
            return input_token, None, other

    def _tokenize_msg_for_training(self, messages):
        input_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        if not input_text.endswith(self.tokenizer.eos_token):
            input_text += " " + self.tokenizer.eos_token

        image_inputs, video_inputs, video_kwargs = self.process_vision_info(
            messages,
            return_video_kwargs=True,
        )

        tokenized = self.processor(
            text=[input_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            truncation=False,
            return_tensors="pt",
            add_special_tokens=False,
            **video_kwargs,
        )

        input_ids = tokenized["input_ids"][0]

        # Find prompt->response boundary
        assistant_marker = "<|im_start|>assistant"  # For Qwen2.5-VL
        marker_ids = self.tokenizer(assistant_marker, add_special_tokens=False).input_ids

        # Search for the position of marker_ids in input_ids
        response_start = find_subsequence(input_ids.tolist(), marker_ids)
        if response_start == -1:
            raise RuntimeError("Could not find '<|im_start|>assistant' token to determine response boundary.")

        # Create labels: only compute loss on the response part
        labels = input_ids.clone().unsqueeze(0)

        # Mask out the prompt part (excluding the <|assistant|> token itself)
        labels[:, :response_start] = -100

        # Fix eos alignment
        tokenized["input_ids"][0][-1] = self.tokenizer.eos_token_id
        tokenized["attention_mask"][0][-1] = True

        return tokenized, labels

    def _tokenize_msg_for_eval(self, messages):
        # Remove the last assistant response if present
        if messages and messages[-1]['role'] == 'assistant':
            messages = messages[:-1]
        prompt_only_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = self.process_vision_info(messages, return_video_kwargs=True)

        input_token = self.processor(
            text=[prompt_only_text],
            images=image_inputs,
            videos=video_inputs,
            max_length=self.max_length,
            padding=True,
            padding_side="left",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
            **video_kwargs,
        )

        return input_token

    def collate_fn(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None

        input_ids_list, input_masks_list = [], []
        input_img_pixels, input_img_grid = [], []
        input_video_pixels, input_video_grid = [], []
        labels_list = []
        extras_list = []

        for input_token, labels, extra in batch:
            extras_list.append(extra)

            # --- Get text  ---
            input_ids_list.append(input_token['input_ids'])
            input_masks_list.append(input_token['attention_mask'])

            # --- Get labels ---
            if labels is not None:
                labels_list.append(labels)

            # --- Get visuals  ---
            if 'pixel_values' in input_token:
                input_img_pixels.append(input_token['pixel_values'])
                input_img_grid.append(input_token['image_grid_thw'])
            if 'pixel_values_videos' in input_token:
                input_video_pixels.append(input_token['pixel_values_videos'])
                input_video_grid.append(input_token['video_grid_thw'])

        padding_side = "left"
        input_ids = zero_pad_sequences(input_ids_list, side=padding_side, value=self.tokenizer.pad_token_id)
        input_masks = zero_pad_sequences(input_masks_list, side=padding_side)
        if labels_list:
            labels_list = zero_pad_sequences(labels_list, side=padding_side, value=-100)

        return (
            # Text inputs
            input_ids,
            input_masks,
            # Image inputs
            torch.cat(input_img_pixels, dim=0) if input_img_pixels else None,
            torch.cat(input_img_grid, dim=0) if input_img_grid else None,
            # Video inputs
            torch.cat(input_video_pixels, dim=0) if input_video_pixels else None,
            torch.cat(input_video_grid, dim=0) if input_video_grid else None,
            # Labels
            labels_list if labels_list else None,
            # Extras
            extras_list
        )
