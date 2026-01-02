import io
import random
from typing import List, Dict, Any

import librosa
from loguru import logger

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoProcessor

from .hpdv3 import HPDv3Handler
from .rapidata import RapidataT2VHandler, RapidataI2VHandler
from .omnirewardbench import OmniRewardBenchT2VHandler, OmniRewardBenchT2IHandler, OmniRewardBenchT2AHandler
from .image_reward_db import ImageRewardDBHandler
from .audio_alpaca import AudioAlpacaHandler
from .utils import zero_pad_sequences, load_multimodal_content


class RankDatasetVL(Dataset):
    """
    Preference ranking dataset used for vision-language scalar reward model (RM) training.

    RMRankDatasetVL dataset supports multiple data sources through pluggable
    Data Handlers and covers both understanding tasks (image-to-text, video-to-text)
    and generation tasks (text-to-image, text-to-video).

    Each example contains two inputs to be compared. Labels use "A", "B",
    and "C" to indicate which input is better or if they tie. For example,
    label "A" means input0 is preferred over input1; "C" means a tie.

    :param dataset_paths: List of dataset file paths or directories. The
        handler is determined by the source keyword (e.g. "hpdv3",
        "rapidata", "omnirewardbench"). The format is "source:path".
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
    :type max_length: int
    :param config: Additional configuration options. Supported keys include:
        - ``task_instruction`` (str): Instruction for the evaluation task.
        - ``system_prompt_template`` (str): Template for the system prompt
          with a ``{prompt}`` placeholder.
    :type config: Dict[str, Any]

    :example:

        >>> dataset = RMRankDataset([
        ...     'hpdv3:/data/hpdv3/train.json'
        ... ], processor=proc, tokenizer=tok, max_length=4096)

    """
    def __init__(
        self,
        dataset_paths: List[str],
        processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        strategy=None,
        max_length: int = 4096,
        config: Dict[str, Any] = None
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
        elif "keye" in self.processor.__class__.__name__.lower():
            from keye_vl_utils import process_vision_info
            self.process_vision_info = process_vision_info
        else:
            raise NotImplementedError(f"Processor type {self.processor.__class__.__name__} not supported yet.")

        self.handlers = {
            "hpdv3": HPDv3Handler(),
            "rapidata-t2v": RapidataT2VHandler(),
            "rapidata-i2v": RapidataI2VHandler(),
            "omnirewardbench-t2v": OmniRewardBenchT2VHandler(),
            "omnirewardbench-t2i": OmniRewardBenchT2IHandler(),
            "imagerewarddb": ImageRewardDBHandler(),
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
        messages0, messages1, other = handler.parse_item(item, loaded_content, self.config)

        # Tokenize the two message sequences
        input0_token, input1_token = self._tokenize_pair(messages0, messages1)

        return input0_token, input1_token, other

    def _tokenize_pair(self, messages0, messages1):
        input0_text = self.processor.apply_chat_template(messages0, tokenize=False, add_generation_prompt=True)
        input1_text = self.processor.apply_chat_template(messages1, tokenize=False, add_generation_prompt=True)
        if not input0_text.endswith(self.tokenizer.eos_token):
            input0_text += " " + self.tokenizer.eos_token
        if not input1_text.endswith(self.tokenizer.eos_token):
            input1_text += " " + self.tokenizer.eos_token

        image_inputs0, video_inputs0, video_kwargs0 = self.process_vision_info(messages0, return_video_kwargs=True)
        image_inputs1, video_inputs1, video_kwargs1 = self.process_vision_info(messages1, return_video_kwargs=True)

        input0_token = self.processor(
            text=[input0_text],
            images=image_inputs0,
            videos=video_inputs0,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
            **video_kwargs0,
        )
        input0_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        input0_token["attention_mask"][0][-1] = True

        input1_token = self.processor(
            text=[input1_text],
            images=image_inputs1,
            videos=video_inputs1,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
            **video_kwargs1,
        )
        input1_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        input1_token["attention_mask"][0][-1] = True

        return input0_token, input1_token

    def collate_fn(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None

        input0_ids_list, input0_masks_list = [], []
        input1_ids_list, input1_masks_list = [], []
        extras_list = []

        input0_img_pixels, input0_img_grid = [], []
        input0_video_pixels, input0_video_grid = [], []

        input1_img_pixels, input1_img_grid = [], []
        input1_video_pixels, input1_video_grid = [], []

        for input0_token, input1_token, extra in batch:
            extras_list.append(extra)

            # --- Get text  ---
            input0_ids_list.append(input0_token['input_ids'])
            input0_masks_list.append(input0_token['attention_mask'])
            input1_ids_list.append(input1_token['input_ids'])
            input1_masks_list.append(input1_token['attention_mask'])

            # --- Get visuals  ---
            if 'pixel_values' in input0_token:
                input0_img_pixels.append(input0_token['pixel_values'])
                input0_img_grid.append(input0_token['image_grid_thw'])
                input1_img_pixels.append(input1_token['pixel_values'])
                input1_img_grid.append(input1_token['image_grid_thw'])

            if 'pixel_values_videos' in input0_token:
                input0_video_pixels.append(input0_token['pixel_values_videos'])
                input0_video_grid.append(input0_token['video_grid_thw'])
                input1_video_pixels.append(input1_token['pixel_values_videos'])
                input1_video_grid.append(input1_token['video_grid_thw'])

        padding_side = "left"
        input0_ids = zero_pad_sequences(input0_ids_list, side=padding_side, value=self.tokenizer.pad_token_id)
        input0_masks = zero_pad_sequences(input0_masks_list, side=padding_side)
        input1_ids = zero_pad_sequences(input1_ids_list, side=padding_side, value=self.tokenizer.pad_token_id)
        input1_masks = zero_pad_sequences(input1_masks_list, side=padding_side)

        return (
            # Text inputs
            input0_ids,
            input0_masks,
            input1_ids,
            input1_masks,
            # Image inputs
            torch.cat(input0_img_pixels, dim=0) if input0_img_pixels else None,
            torch.cat(input0_img_grid, dim=0) if input0_img_grid else None,
            torch.cat(input1_img_pixels, dim=0) if input1_img_pixels else None,
            torch.cat(input1_img_grid, dim=0) if input1_img_grid else None,
            # Video inputs
            torch.cat(input0_video_pixels, dim=0) if input0_video_pixels else None,
            torch.cat(input0_video_grid, dim=0) if input0_video_grid else None,
            torch.cat(input1_video_pixels, dim=0) if input1_video_pixels else None,
            torch.cat(input1_video_grid, dim=0) if input1_video_grid else None,
            # Extras
            extras_list
        )


class RankDatasetAL(Dataset):
    """
    Preference ranking dataset used for audio-language scalar reward model (RM) training.

    RMRankDatasetAL dataset supports multiple audio-language data sources through pluggable
    Data Handlers and support training reward model for text-to-audio task.

    :param dataset_paths: List of dataset file paths or directories. The
        handler is determined by the source keyword (e.g. "audio-alpaca",
        "omnirewardbench-t2a"). The format is "source:path".
        e.g. "audio-alpaca:/path/to/file.parquet"
    :type dataset_paths: List[str]
    :param processor: Multimodal processor used for tokenization and audio
        processing.
    :type processor: transformers.AutoProcessor
    :param tokenizer: Tokenizer used for text tokenization.
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
    """
    def __init__(
        self,
        dataset_paths: List[str],
        processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        strategy=None,
        max_length: int = 4096,
        config: Dict[str, Any] = None
    ):

        super().__init__()
        self.processor = processor
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.config = config if config else {}

        self.audio_content_loader = load_multimodal_content
        self.handlers = {
            "omnirewardbench-t2a": OmniRewardBenchT2AHandler(),
            "audio-alpaca": AudioAlpacaHandler(),
        }

        # Load data from all specified dataset paths
        # Expect dataset_paths entries in the format: "source:path",
        # e.g. "audio-alpaca:/path/to/file.parquet"
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
                for it in loaded_items:
                    it["source"] = source
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

        # Get media/audio info for this item. Prefer the generic get_media_info
        audio_info = handler.get_media_info(item)

        # Load all audio content at once
        loaded_content = self.audio_content_loader(audio_info)
        if loaded_content is None:
            raise RuntimeError(f"Failed to load audio content: {audio_info}")

        # Pass the loaded content dict to parse_item
        messages0, messages1, other = handler.parse_item(item, loaded_content, self.config)

        # Tokenize the two message sequences
        input0_token, input1_token = self._tokenize_pair(messages0, messages1)

        return input0_token, input1_token, other

    def _tokenize_pair(self, messages0, messages1):
        # Get audio data from messages
        audio0, audio1 = None, None
        for msg in messages0:
            if isinstance(msg["content"], list):
                for ele in msg["content"]:
                    if ele["type"] == "audio":
                        if isinstance(ele['audio'], str):
                            with open(ele['audio'], 'rb') as f:
                                audio0_bytes = f.read()
                            audio0 = librosa.load(
                                io.BytesIO(audio0_bytes), sr=self.processor.feature_extractor.sampling_rate
                            )[0]
                        elif isinstance(ele['audio'], io.BytesIO):
                            audio0 = librosa.load(ele['audio'], sr=self.processor.feature_extractor.sampling_rate)[0]
                        else:
                            raise ValueError(f"Unsupported audio type: {type(ele['audio'])}")

        for msg in messages1:
            if isinstance(msg["content"], list):
                for ele in msg["content"]:
                    if ele["type"] == "audio":
                        if isinstance(ele['audio'], str):
                            with open(ele['audio'], 'rb') as f:
                                audio1_bytes = f.read()
                            audio1 = librosa.load(
                                io.BytesIO(audio1_bytes), sr=self.processor.feature_extractor.sampling_rate
                            )[0]
                        elif isinstance(ele['audio'], io.BytesIO):
                            audio1 = librosa.load(ele['audio'], sr=self.processor.feature_extractor.sampling_rate)[0]
                        else:
                            raise ValueError(f"Unsupported audio type: {type(ele['audio'])}")

        input0_text = self.processor.apply_chat_template(messages0, tokenize=False, add_generation_prompt=True)
        input1_text = self.processor.apply_chat_template(messages1, tokenize=False, add_generation_prompt=True)
        if not input0_text.endswith(self.tokenizer.eos_token):
            input0_text += " " + self.tokenizer.eos_token
        if not input1_text.endswith(self.tokenizer.eos_token):
            input1_text += " " + self.tokenizer.eos_token

        input0_token = self.processor(
            text=[input0_text],
            audio=[audio0],
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            padding="longest",  # See https://github.com/huggingface/transformers/issues/30740
            truncation=False,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input0_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        input0_token["attention_mask"][0][-1] = True

        input1_token = self.processor(
            text=[input1_text],
            audio=[audio1],
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            padding="longest",
            truncation=False,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input1_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        input1_token["attention_mask"][0][-1] = True

        return input0_token, input1_token

    def collate_fn(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None

        input0_ids_list, input0_masks_list = [], []
        input1_ids_list, input1_masks_list = [], []
        extras_list = []

        input0_input_features, input0_feature_attention_mask = [], []
        input1_input_features, input1_feature_attention_mask = [], []

        for input0_token, input1_token, extra in batch:
            extras_list.append(extra)

            # --- Get text  ---
            input0_ids_list.append(input0_token['input_ids'])
            input0_masks_list.append(input0_token['attention_mask'])
            input1_ids_list.append(input1_token['input_ids'])
            input1_masks_list.append(input1_token['attention_mask'])

            # --- Get audios  ---
            input0_input_features.append(input0_token['input_features'])
            input0_feature_attention_mask.append(input0_token['feature_attention_mask'])
            input1_input_features.append(input1_token['input_features'])
            input1_feature_attention_mask.append(input1_token['feature_attention_mask'])

        padding_side = "left"
        input0_ids = zero_pad_sequences(input0_ids_list, side=padding_side, value=self.tokenizer.pad_token_id)
        input0_masks = zero_pad_sequences(input0_masks_list, side=padding_side)
        input1_ids = zero_pad_sequences(input1_ids_list, side=padding_side, value=self.tokenizer.pad_token_id)
        input1_masks = zero_pad_sequences(input1_masks_list, side=padding_side)

        return (
            # Text inputs
            input0_ids,
            input0_masks,
            input1_ids,
            input1_masks,
            # Audio inputs
            torch.cat(input0_input_features, dim=0) if input0_input_features else None,
            torch.cat(input0_feature_attention_mask, dim=0) if input0_feature_attention_mask else None,
            torch.cat(input1_input_features, dim=0) if input1_input_features else None,
            torch.cat(input1_feature_attention_mask, dim=0) if input1_feature_attention_mask else None,
            # Extras
            extras_list
        )
