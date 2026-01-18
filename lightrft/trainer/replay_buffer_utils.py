"""
Utility functions for replay buffer operations in reinforcement learning.

This module provides specialized functions for handling both language model
experiences and vision-language model experiences. It includes utilities for batch
splitting, sequence padding, and experience creation optimized for distributed training.

Key features:
- Automatic detection of experience types
- Efficient batch splitting and creation
- Sequence padding and padding removal
- Support for both packed and unpacked samples
"""

from typing import List, Optional, Union
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from PIL import Image

from .experience_maker import Experience
from .experience_maker_vl import ExperienceVL


@dataclass
class BufferItem:
    """BufferItem is an item of experience data.

    Shapes of each tensor:
    sequences: (S)
    action_log_probs: (A)
    base_action_log_probs: (A)
    values: (1)
    returns: (1)
    advantages: (1)
    attention_mask: (S)
    action_mask: (A)
    action_entropy: (A) - Entropy values for high-entropy token filtering

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    action_entropy: Optional[torch.Tensor] = None  # Entropy for high-entropy token filtering


@dataclass
class BufferItemVL:
    """BufferItemVL is an item of experience data.

    Shapes of each tensor:
    sequences: (S)
    pixel_values: (B*H, W)
    image_grid_thws: (B, 3)
    raw_images: Optional[List[Image.Image]]  # raw images before processing
    action_log_probs: (A)
    base_action_log_probs: (A)
    values: (1)
    returns: (1)
    advantages: (1)
    attention_mask: (S)
    action_mask: (A)
    action_entropy: (A) - Entropy values for high-entropy token filtering

    "A" is the number of actions.
    """

    sequences: torch.Tensor

    pixel_values: Optional[torch.Tensor] = None  # image pixel processed by HF processor
    image_grid_thws: Optional[torch.Tensor] = None  # image grid thw
    pixel_values_videos: Optional[torch.Tensor] = None  # video pixel processed by HF processor
    video_grid_thws: Optional[torch.Tensor] = None  # video grid thw
    raw_images: Optional[List[Image.Image]] = None  # raw images before processing

    action_log_probs: torch.Tensor = None
    base_action_log_probs: torch.Tensor = None
    values: torch.Tensor = None
    returns: torch.Tensor = None
    advantages: torch.Tensor = None
    attention_mask: Optional[torch.LongTensor] = None
    action_mask: Optional[torch.BoolTensor] = None
    info: Optional[dict] = None
    action_entropy: Optional[torch.Tensor] = None  # Entropy for high-entropy token filtering


def is_vl_experience(experience: Union[Experience, ExperienceVL]) -> bool:
    """
    Determine if an experience is a vision-language experience.

    Checks for the presence of vision-specific attributes to distinguish between
    language model experiences and vision-language experiences.

    :param experience: The experience object to check
    :type experience: Union[Experience, ExperienceVL]
    :return: True if the experience contains vision data, False otherwise
    :rtype: bool

    Example::

        exp = ExperienceVL(...)
        if is_vl_experience(exp):
            print("This is a vision-language experience")
    """
    return hasattr(experience, 'pixel_values')


def split_experience_batch(experience: Union[Experience, ExperienceVL]) -> List:
    """
    Split a batch of experiences into individual items.

    Automatically detects the experience type and delegates to the appropriate
    splitting function. This is a generic interface that handles both types of experiences.

    :param experience: Batch experience to split into individual items
    :type experience: Union[Experience, ExperienceVL]
    :return: List of individual experience items
    :rtype: List

    Example::

        # Split a batch of experiences
        batch_experience = make_experience_batch(items)
        individual_items = split_experience_batch(batch_experience)

        # Process each item individually
        for item in individual_items:
            process_item(item)
    """
    if is_vl_experience(experience):
        return _split_experience_batch_vl(experience)
    else:
        return _split_experience_batch(experience)


def _split_experience_batch(experience: Experience) -> List:
    """
    Split a batch of language model experiences into individual items.

    This function processes a batch of experiences (without vision data)
    and splits them into individual BufferItem objects. It handles all
    experience attributes including sequences, log probabilities, values, returns,
    advantages, masks, and additional info.

    :param experience: Batch of experiences to split
    :type experience: Experience
    :return: List of individual BufferItem objects
    :rtype: List
    :raises AssertionError: If batch size consistency check fails

    Example::

        # Create a batch experience
        batch_exp = Experience(
            sequences=torch.tensor([[1,2,3],[4,5,6]]),
            action_log_probs=torch.tensor([[0.1,0.2],[0.3,0.4]]),
            # ... other attributes
        )

        # Split into individual items
        items = _split_experience_batch(batch_exp)
        print(f"Split {len(items)} items from batch")
    """

    batch_size = len(experience.sequences)
    batch_kwargs = [{} for _ in range(batch_size)]
    keys = (
        "sequences",
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
        "action_entropy",
    )
    for key in keys:
        # Use getattr with default None to handle optional attributes like action_entropy
        value = getattr(experience, key, None)
        if value is None:
            for i in range(batch_size):
                batch_kwargs[i][key] = None
            continue
        vals = value
        if isinstance(vals, torch.Tensor):
            vals = torch.unbind(vals)
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            batch_kwargs[i][key] = v

    for i in range(batch_size):
        batch_kwargs[i]["info"] = {}

    # Instead of unbinding tensors, we handle various data types in info.
    if experience.info:
        for k, v_batch in experience.info.items():
            if isinstance(v_batch, torch.Tensor):
                # If it's a tensor, unbind it as before
                vals = torch.unbind(v_batch)
                assert batch_size == len(vals)
                for i, vv in enumerate(vals):
                    if isinstance(vv, torch.Tensor) and vv.numel() == 1:
                        batch_kwargs[i]["info"][k] = vv.item()
                    else:
                        batch_kwargs[i]["info"][k] = vv
            elif isinstance(v_batch, list) and len(v_batch) == batch_size:
                # If it's a list (e.g., list of strings, dicts), distribute it
                for i in range(batch_size):
                    batch_kwargs[i]["info"][k] = v_batch[i]
            else:
                # For other cases, broadcast the same value (if not a sequence)
                for i in range(batch_size):
                    batch_kwargs[i]["info"][k] = v_batch

    items = [BufferItem(**kwargs) for kwargs in batch_kwargs]
    return items


def _split_experience_batch_vl(experience: ExperienceVL) -> List:
    """
    Split a batch of vision-language experiences into individual items.

    This function handles the complex logic of de-stacking vision-language data.
    Unlike text-only data, vision components (images/videos) are often flattened
    into a single continuous tensor for efficiency during rollout. This function
    uses metadata in `experience.info` (`image_num` and `video_num`) to correctly
    slice these flattened tensors back into their per-sample components.

    Splitting Logic:
    1. Standard Tensors: `sequences`, `values`, etc. are split using `torch.unbind`.
    2. Grid Metadata: `image_grid_thws` (N, 3) is sliced based on `experience.info["image_num"]`.
       For example, if `image_num` is [2, 1], the first sample gets the first 2 rows of grids.
    3. Pixel Values: `pixel_values` (Total_Patches, patches) is sliced based on the sum
       of tokens calculated from the sample's corresponding `image_grid_thws`.

    :param experience: Batch of vision-language experiences to split
    :type experience: ExperienceVL
    :return: List of individual BufferItemVL objects
    :rtype: List

    Example::

        # Multi-image scenario: Batch size 2
        # Sample 0 has 2 images, Sample 1 has 1 image.
        # Total 3 images in image_grid_thws
        batch_exp = ExperienceVL(
            sequences=torch.zeros(2, 10),
            image_grid_thws=torch.tensor([[1, 10, 10], [1, 20, 20], [1, 15, 15]]),
            pixel_values=torch.randn(100+400+225, 1152), # flattened patches
            info={
                "image_num": torch.tensor([2, 1], dtype=torch.float32)
            }
        )

        items = _split_experience_batch_vl(batch_exp)
        # items[0].image_grid_thws: Shape [2, 3] (First two rows)
        # items[0].pixel_values: Shape [10*10 + 20*20, 1152]
        # items[1].image_grid_thws: Shape [1, 3] (Last row)
        # items[1].pixel_values: Shape [15*15, 1152]
    """

    batch_size = len(experience.sequences)
    batch_kwargs = [{} for _ in range(batch_size)]
    # First, split standard tensors that always match batch_size
    keys = (
        "sequences",
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
        "action_entropy",
    )
    for key in keys:
        # Use getattr with default None to handle optional attributes like action_entropy
        value = getattr(experience, key, None)
        if value is None:
            for i in range(batch_size):
                batch_kwargs[i][key] = None
            continue
        vals = value
        if isinstance(vals, torch.Tensor):
            vals = torch.unbind(vals)
        assert batch_size == len(vals), f"Key {key} size mismatch: {len(vals)} vs {batch_size}"
        for i, v in enumerate(vals):
            batch_kwargs[i][key] = v

    # Split image_grid_thws and video_grid_thws accurately using metadata
    for grid_key, num_key in [("image_grid_thws", "image_num"), ("video_grid_thws", "video_num")]:
        grid_data = getattr(experience, grid_key, None)
        if grid_data is not None:
            # If it's already a list, it was pre-split by _process_multi_image_video_thws in FastExperienceMaker
            if isinstance(grid_data, list):
                for i in range(batch_size):
                    batch_kwargs[i][grid_key] = grid_data[i]
                continue

            # Try to get number of components per sample from info
            nums = experience.info.get(num_key) if experience.info else None
            if nums is not None:
                if isinstance(nums, torch.Tensor):
                    nums = nums.tolist()

                curr_idx = 0
                for i, n in enumerate(nums):
                    if n > 0:
                        batch_kwargs[i][grid_key] = grid_data[curr_idx:curr_idx + n]
                        curr_idx += n
                    else:
                        batch_kwargs[i][grid_key] = None
            else:
                # Fallback for simple case: 1-to-1 mapping
                if isinstance(grid_data, torch.Tensor) and grid_data.size(0) == batch_size:
                    vals = torch.unbind(grid_data)
                    for i, v in enumerate(vals):
                        batch_kwargs[i][grid_key] = v
                elif isinstance(grid_data, list):
                    for i, v in enumerate(grid_data):
                        batch_kwargs[i][grid_key] = v
                else:
                    raise ValueError(
                        f"Ambiguous {grid_key} split: Total {grid_data.size(0)} vs Batch {batch_size}. "
                        f"Missing '{num_key}' in info."
                    )
        else:
            for i in range(batch_size):
                batch_kwargs[i][grid_key] = None

    # Split image data
    if experience.pixel_values is not None:
        pixel_values = experience.pixel_values
        if isinstance(pixel_values, torch.Tensor):
            index = 0
            for i in range(len(batch_kwargs)):
                if batch_kwargs[i]["image_grid_thws"] is not None:
                    grid = batch_kwargs[i]["image_grid_thws"]
                    # grid is already [N, 3] for this sample
                    num_image_tokens = torch.sum(torch.prod(grid, dim=-1)).item()
                else:
                    num_image_tokens = 0

                # Slice from the flattened pixel_values
                batch_kwargs[i]["pixel_values"] = pixel_values[index:index + num_image_tokens]
                index += num_image_tokens

    # Split video data
    if experience.pixel_values_videos is not None:
        pixel_values_videos = experience.pixel_values_videos
        if isinstance(pixel_values_videos, torch.Tensor):
            index = 0
            for i in range(len(batch_kwargs)):
                if batch_kwargs[i]["video_grid_thws"] is not None:
                    grid = batch_kwargs[i]["video_grid_thws"]
                    num_video_tokens = torch.sum(torch.prod(grid, dim=-1)).item()
                else:
                    num_video_tokens = 0

                batch_kwargs[i]["pixel_values_videos"] = pixel_values_videos[index:index + num_video_tokens]
                index += num_video_tokens

    # Split raw images
    if experience.raw_images is not None:
        for i in range(len(batch_kwargs)):
            batch_kwargs[i]["raw_images"] = experience.raw_images[i]

    for i in range(batch_size):
        batch_kwargs[i]["info"] = {}

    # Instead of unbinding tensors, we handle various data types in info.
    if experience.info:
        for k, v_batch in experience.info.items():
            if isinstance(v_batch, torch.Tensor):
                # If it's a tensor, unbind it as before
                vals = torch.unbind(v_batch)
                assert batch_size == len(vals)
                for i, vv in enumerate(vals):
                    if isinstance(vv, torch.Tensor) and vv.numel() == 1:
                        batch_kwargs[i]["info"][k] = vv.item()
                    else:
                        batch_kwargs[i]["info"][k] = vv
            elif isinstance(v_batch, list) and len(v_batch) == batch_size:
                # If it's a list (e.g., list of strings, dicts), distribute it
                for i in range(batch_size):
                    batch_kwargs[i]["info"][k] = v_batch[i]
            else:
                # For other cases, broadcast the same value (if not a sequence)
                for i in range(batch_size):
                    batch_kwargs[i]["info"][k] = v_batch

    items = [BufferItemVL(**kwargs) for kwargs in batch_kwargs]
    return items


def zero_pad_sequences(sequences: List[torch.Tensor], side: str = "left") -> torch.Tensor:
    """
    Zero-pad a list of sequences to the same length.

    This utility function pads sequences to the maximum length in the batch,
    either on the left or right side. It is used for creating batched tensors
    from variable-length sequences.

    :param sequences: List of sequences to pad (each sequence is a 1D tensor)
    :type sequences: List[torch.Tensor]
    :param side: Padding side, either "left" or "right"
    :type side: str
    :return: Batched tensor of padded sequences
    :rtype: torch.Tensor
    :raises AssertionError: If side is not "left" or "right"

    Example::

        sequences = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5]),
            torch.tensor([6, 7, 8, 9])
        ]

        # Pad to the right
        padded = zero_pad_sequences(sequences, side="right")
        # Result: tensor([[1, 2, 3, 0],
        #                 [4, 5, 0, 0],
        #                 [6, 7, 8, 9]])

        # Pad to the left
        padded_left = zero_pad_sequences(sequences, side="left")
        # Result: tensor([[0, 1, 2, 3],
        #                 [0, 0, 4, 5],
        #                 [6, 7, 8, 9]])
    """
    assert side in ("left", "right")
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding))
    return torch.stack(padded_sequences, dim=0)


def make_experience_batch(items: List, packing_samples: bool = False) -> Union[Experience, ExperienceVL]:
    """
    Create a batch experience from individual items.

    This generic function automatically detects the item type
    and delegates to the appropriate batch creation function. It handles both packed
    and unpacked samples efficiently.

    :param items: List of individual experience items to batch
    :type items: List
    :param packing_samples: Whether to pack samples without padding (True) or use padding (False)
    :type packing_samples: bool
    :return: Batched experience (either Experience or ExperienceVL)
    :rtype: Union[Experience, ExperienceVL]
    :raises ValueError: If items list is empty

    Example::

        # Create batch from items
        items = [BufferItem(...), BufferItem(...)]
        batch_exp = make_experience_batch(items, packing_samples=False)

        # Create batch from vision-language items
        vl_items = [BufferItemVL(...), BufferItemVL(...)]
        batch_vl_exp = make_experience_batch(vl_items, packing_samples=True)
    """
    if not items:
        raise ValueError("items list cannot be empty")

    # Determine experience type by checking the first item
    first_item = items[0]
    if hasattr(first_item, 'pixel_values'):
        return _make_experience_batch_vl(items, packing_samples)
    else:
        return _make_experience_batch(items, packing_samples)


def _make_experience_batch(items: List, packing_samples: bool = False) -> Experience:
    """
    Create a batch of experiences from individual items.

    This function aggregates individual experience items into a batched
    Experience object. It handles both packed and unpacked samples, with padding
    applied for unpacked samples to ensure consistent tensor shapes.

    :param items: List of individual experience items
    :type items: List
    :param packing_samples: Whether to pack samples without padding (True) or use padding (False)
    :type packing_samples: bool
    :return: Batched experience
    :rtype: Experience

    Example::

        items = [
            BufferItem(sequences=torch.tensor([1,2,3]), ...),
            BufferItem(sequences=torch.tensor([4,5,6,7]), ...)
        ]

        # With padding (packing_samples=False)
        batch_exp = _make_experience_batch(items, packing_samples=False)
        # sequences will be padded to length 4

        # Without padding (packing_samples=True)
        packed_batch = _make_experience_batch(items, packing_samples=True)
        # sequences will remain as a list of variable-length tensors
    """
    kwargs = {}
    keys = (
        "sequences",
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )
    for key in keys:
        vals = [getattr(item, key) for item in items]
        if not packing_samples:
            batch_data = zero_pad_sequences(vals, "left") if vals[0] is not None else None
        else:
            batch_data = vals if vals[0] is not None else None
        kwargs[key] = batch_data

    # Handle action_entropy if it exists in any item
    # action_entropy has shape (A,) per item, handle it like action_mask
    if items and hasattr(items[0], 'action_entropy') and items[0].action_entropy is not None:
        entropy_vals = [getattr(item, 'action_entropy', None) for item in items]
        if all(v is not None for v in entropy_vals):
            if not packing_samples:
                # For padded batches, pad action_entropy to match action_mask
                kwargs["action_entropy"] = zero_pad_sequences(entropy_vals, "left")
            else:
                # For packed batches, check if all have the same length
                lengths = [len(v) for v in entropy_vals]
                if len(set(lengths)) == 1:
                    kwargs["action_entropy"] = torch.stack(entropy_vals, dim=0)
                else:
                    # If lengths differ, pad to max length
                    kwargs["action_entropy"] = zero_pad_sequences(entropy_vals, "left")
        else:
            kwargs["action_entropy"] = None
    else:
        kwargs["action_entropy"] = None

    kwargs["info"] = {}
    if items and items[0].info:
        for key in items[0].info.keys():
            vals = [item.info[key] for item in items]
            # Check if the values can be converted to a tensor (i.e., are numeric)
            if isinstance(vals[0], (int, float, bool)):
                try:
                    # Convert numeric types to a tensor
                    kwargs["info"][key] = torch.tensor(vals)
                except (TypeError, ValueError):
                    # Fallback for mixed types or other errors
                    kwargs["info"][key] = vals
            else:
                # For non-numeric types (str, list, dict), keep them as a Python list
                kwargs["info"][key] = vals

    return Experience(**kwargs)


def _make_experience_batch_vl(items: List, packing_samples: bool = False) -> ExperienceVL:
    """
    Create a batch of vision-language experiences from individual items.

    This function aggregates individual `BufferItemVL` objects into a single `ExperienceVL`
    batch. It concatenates visual data (pixels and grids) into flattened tensors
    and automatically records the count per sample (`image_num`, `video_num`) in
    the `info` dictionary to enable later splitting.

    :param items: List of individual vision-language experience items
    :type items: List
    :param packing_samples: Whether to pack samples without padding (True) or use padding (False)
    :type packing_samples: bool
    :return: Batched vision-language experience
    :rtype: ExperienceVL

    Example::

        # Create a batch from two items
        item1 = BufferItemVL(
            sequences=torch.zeros(5),
            image_grid_thws=torch.tensor([[1, 5, 5], [1, 8, 8]]), # 2 images
            pixel_values=torch.randn(25+64, 1152)
        )
        item2 = BufferItemVL(
            sequences=torch.zeros(8),
            image_grid_thws=torch.tensor([[1, 10, 10]]), # 1 image
            pixel_values=torch.randn(100, 1152)
        )

        batch = _make_experience_batch_vl([item1, item2])
        # batch.image_grid_thws: Shape [3, 3] (2 + 1 rows concatenated)
        # batch.info["image_num"]: tensor([2., 1.], dtype=float32)
    """
    kwargs = {}
    keys = (
        "sequences",
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )
    for key in keys:
        vals = [getattr(item, key) for item in items]
        if not packing_samples:
            batch_data = zero_pad_sequences(vals, "left") if vals[0] is not None else None
        else:
            batch_data = vals if vals[0] is not None else None
        kwargs[key] = batch_data

    # Handle action_entropy if it exists in any item
    # action_entropy has shape (A,) per item, handle it like action_mask
    if items and hasattr(items[0], 'action_entropy') and items[0].action_entropy is not None:
        entropy_vals = [getattr(item, 'action_entropy', None) for item in items]
        if all(v is not None for v in entropy_vals):
            if not packing_samples:
                # For padded batches, pad action_entropy to match action_mask
                kwargs["action_entropy"] = zero_pad_sequences(entropy_vals, "left")
            else:
                # For packed batches, check if all have the same length
                lengths = [len(v) for v in entropy_vals]
                if len(set(lengths)) == 1:
                    kwargs["action_entropy"] = torch.stack(entropy_vals, dim=0)
                else:
                    # If lengths differ, pad to max length
                    kwargs["action_entropy"] = zero_pad_sequences(entropy_vals, "left")
        else:
            kwargs["action_entropy"] = None
    else:
        kwargs["action_entropy"] = None

    # Image data processing
    pixel_values_list = [
        item.pixel_values for item in items if item.pixel_values is not None and item.pixel_values.numel() > 0
    ]
    kwargs["pixel_values"] = torch.cat(pixel_values_list, dim=0) if pixel_values_list else None

    image_grid_thws_list = [
        item.image_grid_thws.unsqueeze(0) if
        (item.image_grid_thws is not None and item.image_grid_thws.dim() == 1) else item.image_grid_thws
        for item in items
        if item.image_grid_thws is not None
    ]
    if image_grid_thws_list:
        kwargs["image_grid_thws"] = torch.cat(image_grid_thws_list, dim=0)
    else:
        kwargs["image_grid_thws"] = None

    # Video data processing
    pixel_values_videos_list = [
        item.pixel_values_videos
        for item in items
        if item.pixel_values_videos is not None and item.pixel_values_videos.numel() > 0
    ]
    kwargs["pixel_values_videos"] = torch.cat(pixel_values_videos_list, dim=0) if pixel_values_videos_list else None

    video_grid_thws_list = [
        item.video_grid_thws.unsqueeze(0) if
        (item.video_grid_thws is not None and item.video_grid_thws.dim() == 1) else item.video_grid_thws
        for item in items
        if item.video_grid_thws is not None
    ]
    if video_grid_thws_list:
        kwargs["video_grid_thws"] = torch.cat(video_grid_thws_list, dim=0)
    else:
        kwargs["video_grid_thws"] = None

    raw_images_list = [item.raw_images for item in items]
    kwargs["raw_images"] = raw_images_list if raw_images_list and raw_images_list[0] is not None else None

    # Record the number of components (images/videos) per sample into info dictionary.
    # This ensures accuracy when splitting the batch back into individual items.
    kwargs["info"] = {}
    image_nums = []
    video_nums = []
    for item in items:
        # Determine number of image components
        if item.image_grid_thws is not None:
            image_nums.append(item.image_grid_thws.size(0) if item.image_grid_thws.dim() > 1 else 1)
        else:
            image_nums.append(0)

        # Determine number of video components
        if item.video_grid_thws is not None:
            video_nums.append(item.video_grid_thws.size(0) if item.video_grid_thws.dim() > 1 else 1)
        else:
            video_nums.append(0)

    if items and items[0].info:
        for key in items[0].info.keys():
            vals = [item.info[key] for item in items]
            # Check if the values can be converted to a tensor (i.e., are numeric)
            if isinstance(vals[0], (int, float, bool)):
                try:
                    # Convert numeric types to a tensor
                    kwargs["info"][key] = torch.tensor(vals)
                except (TypeError, ValueError):
                    # Fallback for mixed types or other errors
                    kwargs["info"][key] = vals
            else:
                # For non-numeric types (str, list, dict), keep them as a Python list
                kwargs["info"][key] = vals

    return ExperienceVL(**kwargs)


def remove_padding_in_sequences(items: List) -> List:
    """
    Remove padding from sequences in experience items.

    This generic function automatically detects the item type and delegates to the
    appropriate padding removal function. It removes both left and right padding
    from sequences to restore their original lengths.

    :param items: List of experience items with padded sequences
    :type items: List
    :return: List of experience items with padding removed
    :rtype: List

    Example::

        # Remove padding from items
        padded_items = [BufferItem(sequences=torch.tensor([0,0,1,2,3,0,0]), ...)]
        clean_items = remove_padding_in_sequences(padded_items)
        # Result: sequences become torch.tensor([1,2,3])

        # Remove padding from vision-language items
        padded_vl_items = [BufferItemVL(sequences=torch.tensor([0,0,4,5,6,0]), ...)]
        clean_vl_items = remove_padding_in_sequences(padded_vl_items)
        # Result: sequences become torch.tensor([4,5,6])
    """
    if not items:
        return items

    # Determine item type by checking the first item
    first_item = items[0]
    if hasattr(first_item, 'pixel_values'):
        return _remove_padding_in_sequences_vl(items)
    else:
        return _remove_padding_in_sequences(items)


def _remove_padding_in_sequences(items: List) -> List:
    """
    Remove padding from sequences in experience items.

    This function processes experience items and removes both left and right
    padding from sequences and related tensors. It uses attention masks and action masks
    to determine the original sequence boundaries.

    :param items: List of experience items with padded sequences
    :type items: List
    :return: List of experience items with padding removed
    :rtype: List

    Example::

        # Item with left and right padding
        item = BufferItem(
            sequences=torch.tensor([0, 0, 1, 2, 3, 0, 0]),  # padded sequence
            attention_mask=torch.tensor([0, 0, 1, 1, 1, 0, 0]),
            action_mask=torch.tensor([0, 0, 1, 1, 1, 0, 0]),
            # ... other attributes
        )

        # Remove padding
        clean_item = _remove_padding_in_sequences([item])[0]
        # Result: sequences become torch.tensor([1, 2, 3])
    """
    for item in items:
        seq, act_log_prob, base_act_log_prob, value, ret, adv, att_mask, act_mask = (
            item.sequences,
            item.action_log_probs,
            item.base_action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask,
        )
        # Get action_entropy if it exists
        action_entropy = getattr(item, 'action_entropy', None)

        right_pad = (1 - act_mask.long()).sum()
        right_pad = None if right_pad == 0 else -right_pad

        # left_pad for seq and att_mask
        left_pad = att_mask.long().argmax()
        (
            item.sequences,
            item.action_log_probs,
            item.base_action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask,
        ) = (
            seq[left_pad:right_pad],
            act_log_prob[:right_pad],
            base_act_log_prob[:right_pad] if item.base_action_log_probs is not None else None,
            value[:right_pad] if item.values is not None else None,
            ret[:right_pad],
            adv[:right_pad],
            att_mask[left_pad:right_pad],
            act_mask[:right_pad],
        )
        # Remove padding from action_entropy if it exists
        if action_entropy is not None:
            item.action_entropy = action_entropy[:right_pad]
    return items


def _remove_padding_in_sequences_vl(items: List) -> List:
    """
    Remove padding from sequences in vision-language experience items.

    This function processes vision-language experience items and removes both left
    and right padding from sequences and related tensors. The vision data (pixel values,
    image grids, etc.) remains unchanged as they don't require padding removal.

    :param items: List of vision-language experience items with padded sequences
    :type items: List
    :return: List of vision-language experience items with padding removed
    :rtype: List

    Example::

        # Vision-language item with padding
        item = BufferItemVL(
            sequences=torch.tensor([0, 0, 4, 5, 6, 0]),  # padded sequence
            attention_mask=torch.tensor([0, 0, 1, 1, 1, 0]),
            action_mask=torch.tensor([0, 0, 1, 1, 1, 0]),
            pixel_values=torch.randn(1, 3, 224, 224),  # vision data unchanged
            # ... other attributes
        )

        # Remove padding
        clean_item = _remove_padding_in_sequences_vl([item])[0]
        # Result: sequences become torch.tensor([4, 5, 6])
    """
    for item in items:
        seq, act_log_prob, base_act_log_prob, value, ret, adv, att_mask, act_mask = (
            item.sequences,
            item.action_log_probs,
            item.base_action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask,
        )
        # Get action_entropy if it exists
        action_entropy = getattr(item, 'action_entropy', None)

        right_pad = (1 - act_mask.long()).sum()
        right_pad = None if right_pad == 0 else -right_pad

        # left_pad for seq and att_mask
        left_pad = att_mask.long().argmax()
        (
            item.sequences,
            item.action_log_probs,
            item.base_action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask,
        ) = (
            seq[left_pad:right_pad],
            act_log_prob[:right_pad],
            base_act_log_prob[:right_pad] if item.base_action_log_probs is not None else None,
            value[:right_pad] if item.values is not None else None,
            ret[:right_pad],
            adv[:right_pad],
            att_mask[left_pad:right_pad],
            act_mask[:right_pad],
        )
        # Remove padding from action_entropy if it exists
        if action_entropy is not None:
            item.action_entropy = action_entropy[:right_pad]
    return items
