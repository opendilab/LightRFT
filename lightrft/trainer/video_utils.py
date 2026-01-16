"""
This module provides utilities for processing and normalizing video data in vision-language models.
It includes functions for converting video frames, numpy arrays, and tensors into normalized
torch tensors, and handles both single and multiple videos per sample.

The main components are:
- frames_to_tensor: Converts a list of PIL Image frames into a stacked video tensor
- single_video_to_tensor: Handles conversion of various single video formats to tensors
- to_video_tensor: Dispatches video data to appropriate tensor conversion based on structure
- normalize_videos: Processes a batch of video inputs into normalized tensor format
- get_videos_num: Extracts the count of videos for each sample in a batch

These utilities ensure video data is correctly formatted in [T, H, W, C] for model processing.
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Union, Optional


def frames_to_tensor(frames: List[Image.Image]) -> torch.Tensor:
    """
    Convert a list of PIL Images representing video frames to a stacked video tensor.

    :param frames: List of PIL Image frames
    :type frames: List[PIL.Image.Image]

    :return: Stacked video tensor in [T, H, W, C] format
    :rtype: torch.Tensor

    **Example:**

    .. code-block:: python

        from PIL import Image
        import torch
        frames = [Image.new('RGB', (64, 64)) for _ in range(8)]
        video_tensor = frames_to_tensor(frames)
        assert video_tensor.shape == (8, 64, 64, 3)
    """
    video_np = np.stack([np.array(f.convert("RGB")) for f in frames])
    return torch.from_numpy(video_np)


def single_video_to_tensor(v) -> torch.Tensor:
    """
    Convert a single video (list of frames, ndarray, or tensor) to a torch.Tensor.

    :param v: Single video data
    :type v: Union[torch.Tensor, np.ndarray, List[PIL.Image.Image]]

    :return: Video tensor in [T, H, W, C] format
    :rtype: torch.Tensor

    :raises ValueError: If input type is not supported

    **Example:**

    .. code-block:: python

        import numpy as np
        video_np = np.zeros((10, 224, 224, 3), dtype=np.uint8)
        video_tensor = single_video_to_tensor(video_np)
        assert video_tensor.shape == (10, 224, 224, 3)
    """
    if isinstance(v, torch.Tensor):
        return v
    if isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    if isinstance(v, list):
        # Probably a list of frames for a single video
        return frames_to_tensor(v)
    raise ValueError(f"Unsupported video type: {type(v)}")


def to_video_tensor(item) -> Union[None, torch.Tensor, List[torch.Tensor]]:
    """
    Convert a single sample's video data to normalized tensor format.

    Handles detecting whether the input represents a single video (as a list of frames or
    stacked array) or multiple videos for the sample.

    :param item: Video data for a single sample (None, Tensor, ndarray, or List)
    :type item: Union[None, torch.Tensor, np.ndarray, List[PIL.Image.Image], List]

    :return: Normalized video data (None, single Tensor, or list of Tensors)
    :rtype: Union[None, torch.Tensor, List[torch.Tensor]]

    **Example:**

    .. code-block:: python

        from PIL import Image
        # Single video as list of frames
        video = [Image.new('RGB', (10, 10)) for _ in range(5)]
        tensor = to_video_tensor(video)
        # Multiple videos
        videos = [torch.zeros(5, 10, 10, 3), torch.zeros(5, 10, 10, 3)]
        tensors = to_video_tensor(videos)
    """
    if item is None:
        return None

    # Detect if it's a list of frames (one video) or a list of multiple videos
    if isinstance(item, list):
        if len(item) == 0:
            return None

        # If it's a list where elements are frames (PIL Images), it's ONE video
        if isinstance(item[0], Image.Image):
            return frames_to_tensor(item)

        # If it's a list where elements are videos themselves
        return [single_video_to_tensor(v) for v in item]

    return single_video_to_tensor(item)


def normalize_videos(raw_videos: List) -> List:
    """
    Normalize video inputs to torch.Tensor format.

    Handles various input formats:
        - list[PIL.Image.Image]: single video (list of frames) -> torch.Tensor
        - np.ndarray: single video -> torch.Tensor
        - torch.Tensor: single video -> returned as is
        - list[torch.Tensor/np.ndarray/list]: multiple videos -> list[torch.Tensor]
        - list[list[PIL.Image.Image]]: multiple videos -> list[torch.Tensor]

    :param raw_videos: List of video inputs for each sample in the batch
    :type raw_videos: List[Union[None, torch.Tensor, np.ndarray, List[PIL.Image.Image], List]]

    :return: List of normalized video data (None, torch.Tensor, or List[torch.Tensor])
    :rtype: List[Union[None, torch.Tensor, List[torch.Tensor]]]

    **Example:**

    .. code-block:: python

        import torch
        raw_vids = [torch.zeros(10, 224, 224, 3), None]
        normalized = normalize_videos(raw_vids)
        # Result: [torch.Tensor, None]
    """
    normalized = []
    for item in raw_videos:
        normalized.append(to_video_tensor(item))
    return normalized


def get_videos_num(all_videos: Optional[List]) -> Optional[List[int]]:
    """
    Extract the number of videos for each sample. Returns 0 for samples
    without videos to keep grid slicing aligned across mixed modalities.

    :param all_videos: List of videos (can be None, torch.Tensor, or lists of torch.Tensor)
    :type all_videos: Optional[List[Union[None, torch.Tensor, List[torch.Tensor]]]]

    :return: List of video counts per sample, or None if no videos are provided
    :rtype: Optional[List[int]]

    **Example:**

    .. code-block:: python

        import torch
        vids = [torch.zeros(5, 10, 10, 3), None, [torch.zeros(5, 10, 10, 3), torch.zeros(5, 10, 10, 3)]]
        counts = get_videos_num(vids)
        assert counts == [1, 0, 2]
    """
    if all_videos is None:
        return None

    counts = []
    for vid in all_videos:
        if vid is None:
            counts.append(0)
        elif isinstance(vid, list):
            # Multiple videos
            counts.append(len(vid))
        elif isinstance(vid, (torch.Tensor, np.ndarray)):
            # One video
            counts.append(1)
        else:
            raise RuntimeError(f"Unsupported video type: {type(vid)}")
    return counts
