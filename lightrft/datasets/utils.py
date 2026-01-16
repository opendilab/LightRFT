"""
Utility functions for dataset processing.

Parts of this file are adapted from Open-Reasoner-Zero:
https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero
"""

from abc import ABC, abstractmethod

import re
import io
from PIL import Image
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn.functional as F


def find_subsequence(lst: List[int], sub: List[int]) -> int:
    """Find first index where ``sub`` appears in ``lst``.
    This function is used to finda marker token sequence (e.g. assistant-start)
    in the token id list so prompt and response can be separated for label masking.

    Complexity: Implements the KMP algorithm: O(n + m) time, O(m) extra space.

    :param lst: Sequence to search (e.g., list of token ids).
    :type lst: List[int]
    :param sub: Subsequence (pattern) to find.
    :type sub: List[int]

    :returns: Index of first occurrence or -1 if not found.
    :rtype: int
    """
    if not sub:
        return 0  # empty pattern matches at 0
    n, m = len(lst), len(sub)
    if m > n:
        return -1

    # build lps array (longest proper prefix which is also suffix)
    lps = [0] * m
    length = 0
    i = 1
    while i < m:
        if sub[i] == sub[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1

    # search
    i = j = 0  # i -> lst, j -> sub
    while i < n:
        if lst[i] == sub[j]:
            i += 1
            j += 1
            if j == m:
                return i - m
        else:
            if j:
                j = lps[j - 1]
            else:
                i += 1
    return -1


def extract_answer(text: str) -> Union[str, None]:
    """
    Extract the content inside <answer>...</answer> from a given text.

    :param text: The input text containing the <answer> tags.
    :type text: str

    :return: The extracted string inside the <answer> tags, or None if not found.
    :rtype: Union[str, None]

    Example::
        >>> text = "The result is <answer>Image 1 is better</answer> based on the evaluation."
        >>> answer = extract_answer(text)
        >>> print(answer)  # Output: Image 1 is better
    """
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def zero_pad_sequences(sequences, side: str = "left", value=0) -> torch.Tensor:
    """
    Pad a list of 1D/2D tensors on the last dimension and stack them.

    :param sequences: Iterable of torch.Tensor objects. Each tensor's last dimension
                      is treated as the sequence length to be padded.
    :type sequences: Iterable[torch.Tensor]
    :param side: Side to apply padding, either "left" or "right"
    :type side: str
    :param value: Padding value
    :type value: int | float

    :return: Stacked tensor with shape (N, ...) where sequences are padded to equal length
    :rtype: torch.Tensor

    Example::

        >>> seqs = [torch.tensor([1,2,3]), torch.tensor([4,5])]
        >>> zero_pad_sequences(seqs, side="left", value=0)
        tensor([[1, 2, 3],
                [0, 4, 5]])
    """
    sequences = list(sequences)
    if len(sequences) == 0:
        raise ValueError("sequences must contain at least one tensor")

    if side not in ("left", "right"):
        raise ValueError("side must be either 'left' or 'right'")

    # Determine target length from last dimension
    max_len = max(seq.size(-1) for seq in sequences)
    padded = []
    for seq in sequences:
        if seq.dim() == 0:
            # scalar -> treat as length-1 sequence
            seq = seq.unsqueeze(0)
        pad_len = max_len - seq.size(-1)
        if pad_len == 0:
            padded.append(seq)
            continue
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded.append(F.pad(seq, padding, value=value))
    return torch.stack(padded, dim=0)


def exist_and_not_none(d, key):
    """
    Check if a key exists in dictionary and its value is not None.

    :param d: Dictionary to check.
    :type d: dict
    :param key: Key to look for.
    :type key: Any
    :return: True if key exists and value is not None.
    :rtype: bool
    """
    return key in d and not d[key] is None


def load_multimodal_content(media_info: Dict) -> Dict:
    """
    Load multimodal content (images, videos, audios, etc.) specified by `media_info`.

    Keys in each entry can include:
      - 'image_local_path' | 'image_bytes'
      - 'video_local_path'
      - 'audio_local_path'

    Returns a dict mapping names to loaded objects or paths.

    :param media_info: Example: {'init_image': {'image_local_path': '/path/img.jpg'},
                                 'video': {'video_local_path': '/path/vid.mp4'},
                                 'audio': {'audio_local_path': '/path/audio.wav'}}
    :type media_info: Dict[str, Dict[str, Any]]

    :return: A dict mapping the same keys to loaded objects, for example:
             - images (from path or bytes) are returned as PIL.Image.Image
             - videos are returned as the original local path (str)
             - audios are returned as the original local path (str)
             If a key cannot be loaded it will be omitted from the result.
    :rtype: Dict[str, Any]
    """
    loaded_content = {}
    for key, info in media_info.items():
        if "image_local_path" in info:
            loaded_content[key] = Image.open(info["image_local_path"])
        elif "image_bytes" in info:
            loaded_content[key] = Image.open(io.BytesIO(info["image_bytes"]))
        elif "video_local_path" in info:
            loaded_content[key] = info["video_local_path"]  # return the local path directly
        elif "audio_local_path" in info:
            loaded_content[key] = info["audio_local_path"]
        elif "audio_bytes" in info:
            loaded_content[key] = io.BytesIO(info["audio_bytes"])
    return loaded_content


def get_task_instructions(handler: Any, config: Dict[str, Any]) -> str:
    """
    Select task instruction based on task type from handler and config.

    :param handler: Data handler instance.
    :param config: Configuration dictionary which contains 'task_instruction'.
    :return: The selected task instruction.
    """
    task_instruction_raw = config.get("task_instruction")
    if isinstance(task_instruction_raw, dict):
        if hasattr(handler, "task_type"):
            prompt = task_instruction_raw.get(handler.task_type)
            if prompt is None:
                raise ValueError(f"Task instruction for {handler.task_type} not found.")
        else:
            raise ValueError(f"Handler {handler.__class__.__name__} does not specify a task_type.")
        return prompt
    elif isinstance(task_instruction_raw, str):
        return task_instruction_raw
    else:
        raise ValueError("task_instruction in config must be either a dict or a str.")


class BaseDataHandler(ABC):
    """
    Base class for data handlers.
    """
    @abstractmethod
    def load_data(self, path: str) -> List[Dict[str, Any]]:
        """
        Load all data items from a data config file, e.g. a json file, or a parquet file.

        :param path: The path to load data from.
        :type path: str

        :return: A list of raw data items.
        :rtype: List[Dict[str, Any]]
        """
        raise NotImplementedError

    @abstractmethod
    def get_media_info(self, item: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """
        Extract path info for all media info from the raw item.

        :param item: The raw data item.
        :type item: Dict[str, Any]

        :return: A dict where keys are logical names (e.g. 'init_image') and values are path dicts.
        :rtype: Dict[str, Dict[str, str]]

        Example::
            >>> item = {'init_image_path': '/path/img.jpg', 'video_path': '/path/vid.mp4'}
            >>> visual_info = get_media_info(item)
            >>> print(visual_info)
            {'init_image': {'image_local_path': '/path/img.jpg'}, 'video': {'video_local_path': '/path/vid.mp4'}}
        """
        raise NotImplementedError

    @abstractmethod
    def parse_item(self, item: Dict[str, Any], media_content: Dict[str, Any],
                   config: Dict[str, Any]) -> Union[Tuple[List[Dict], List[Dict], Dict], Tuple[List[Dict], Dict]]:
        """
        Parse the raw item and the loaded media_content into the standard format.

        :param item: The raw data item.
        :type item: Dict[str, Any]
        :param media_content: A dict containing loaded content (e.g. PIL Images, Video paths).
        :type media_content: Dict[str, Any]
        :param config: A dict of additional configuration options (e.g. prompt templates, max_pixels).
        :type config: Dict[str, Any]

        :return: A tuple containing message lists and a metadata dictionary.
                 - For point-wise scoring data (e.g., Scalar Reward Model training/evaluation):
                   Return (messages_chosen, messages_rejected, other)
                 - For pair-wise ranking data (e.g., Generative Reward Model training/evaluation):
                   Return (messages, other)

                 The `other` dictionary contains metadata, and can optionally include:
                 - "preference": (str) Indicates the ground truth preferred choice ("A", "B", or "C").
                 - "task_type": (str) The type of task (e.g., "text-to-video").
                 - "reward_rule_label": (str) A label used in RL to identify which reward
                   function or reward model to apply to this specific sample when performing
                   reinforcement fine-tuning.
        :rtype: Union[Tuple[List[Dict], List[Dict], Dict], Tuple[List[Dict], Dict]]
        """
        raise NotImplementedError
