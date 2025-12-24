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

    "A" is the number of actions.
    """

    sequences: torch.Tensor

    pixel_values: Optional[torch.Tensor] = None  # image pixel processed by HF processor
    image_grid_thws: Optional[torch.Tensor] = None  # image grid thw
    raw_images: Optional[List[Image.Image]] = None  # raw images before processing
    pixel_values_intern: Optional[torch.Tensor] = None  # InternVL image_info
    image_flags: Optional[torch.Tensor] = None

    action_log_probs: torch.Tensor = None
    base_action_log_probs: torch.Tensor = None
    values: torch.Tensor = None
    returns: torch.Tensor = None
    advantages: torch.Tensor = None
    attention_mask: Optional[torch.LongTensor] = None
    action_mask: Optional[torch.BoolTensor] = None
    info: Optional[dict] = None


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
    return hasattr(experience, 'pixel_values') or hasattr(experience, 'pixel_values_intern')


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
    )
    for key in keys:
        value = getattr(experience, key)
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

    This function processes a batch of vision-language experiences and splits them
    into individual BufferItemVL objects. It handles both text and vision data
    including pixel values, image grids, and additional vision-specific attributes.

    :param experience: Batch of vision-language experiences to split
    :type experience: ExperienceVL
    :return: List of individual BufferItemVL objects
    :rtype: List
    :raises AssertionError: If batch size consistency check fails

    Example::

        # Create a batch of vision-language experiences
        batch_exp = ExperienceVL(
            sequences=torch.tensor([[1,2,3],[4,5,6]]),
            pixel_values=torch.randn(2, 3, 224, 224),
            image_grid_thws=torch.tensor([[1,1,1],[1,1,1]]),
            # ... other attributes
        )

        # Split into individual items
        items = _split_experience_batch_vl(batch_exp)
        print(f"Split {len(items)} vision-language items from batch")
    """

    batch_size = len(experience.sequences)
    batch_kwargs = [{} for _ in range(batch_size)]
    keys = (
        "sequences",
        "image_grid_thws",  # 3
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )
    for key in keys:
        value = getattr(experience, key)
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

    # Split image data
    if experience.pixel_values is not None:
        pixel_values = experience.pixel_values
        if isinstance(pixel_values, torch.Tensor):
            index = 0
            for i in range(len(batch_kwargs)):
                num_images = torch.prod(batch_kwargs[i]["image_grid_thws"]
                                        ) if batch_kwargs[i]["image_grid_thws"] is not None else 0
                batch_kwargs[i]["pixel_values"] = pixel_values[index:index + num_images]
                index += num_images
    if experience.pixel_values_intern is not None:
        pixel_values_intern = experience.pixel_values_intern
        if isinstance(pixel_values_intern, torch.Tensor):
            index = 0
            for i in range(len(batch_kwargs)):
                num_images = torch.prod(batch_kwargs[i]["image_grid_thws"]
                                        ) if batch_kwargs[i]["image_grid_thws"] is not None else 0
                batch_kwargs[i]["pixel_values_intern"] = pixel_values_intern[index:index + num_images]
                index += num_images
    if experience.image_flags is not None:
        image_flags = experience.image_flags
        if isinstance(image_flags, torch.Tensor):
            index = 0
            for i in range(len(batch_kwargs)):
                num_images = torch.prod(batch_kwargs[i]["image_grid_thws"]
                                        ) if batch_kwargs[i]["image_grid_thws"] is not None else 0
                batch_kwargs[i]["image_flags"] = image_flags[index:index + num_images]
                index += num_images

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
    if hasattr(first_item, 'pixel_values') or hasattr(first_item, 'pixel_values_intern'):
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

    This function aggregates individual vision-language experience items into a batched
    ExperienceVL object. It handles both text and vision data, including pixel values,
    image grids, and other vision-specific attributes. For text sequences, it applies
    padding for unpacked samples to ensure consistent tensor shapes.

    :param items: List of individual vision-language experience items
    :type items: List
    :param packing_samples: Whether to pack samples without padding (True) or use padding (False)
    :type packing_samples: bool
    :return: Batched vision-language experience
    :rtype: ExperienceVL

    Example::

        vl_items = [
            BufferItemVL(
                sequences=torch.tensor([1,2,3]),
                pixel_values=torch.randn(1, 3, 224, 224),
                image_grid_thws=torch.tensor([1,1,1]),
                ...
            ),
            BufferItemVL(
                sequences=torch.tensor([4,5,6,7]),
                pixel_values=torch.randn(1, 3, 224, 224),
                image_grid_thws=torch.tensor([1,1,1]),
                ...
            )
        ]

        # Create batched vision-language experience
        batch_vl_exp = _make_experience_batch_vl(vl_items, packing_samples=False)
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

    # Image data processing
    pixel_values_list = [
        item.pixel_values for item in items if item.pixel_values is not None and item.pixel_values.numel() > 0
    ]
    kwargs["pixel_values"] = torch.cat(pixel_values_list, dim=0) if pixel_values_list else None

    image_grid_thws_list = [item.image_grid_thws for item in items]
    kwargs["image_grid_thws"] = torch.stack(image_grid_thws_list, dim=0
                                            ) if image_grid_thws_list and image_grid_thws_list[0] is not None else None

    raw_images_list = [item.raw_images for item in items]
    kwargs["raw_images"] = raw_images_list if raw_images_list and raw_images_list[0] is not None else None

    pixel_values_intern_list = [
        item.pixel_values_intern
        for item in items
        if item.pixel_values_intern is not None and item.pixel_values_intern.numel() > 0
    ]
    kwargs["pixel_values_intern"] = torch.cat(pixel_values_intern_list, dim=0) if pixel_values_intern_list else None

    image_flags_list = [
        item.image_flags for item in items if item.image_flags is not None and item.image_flags.numel() > 0
    ]
    kwargs["image_flags"] = torch.cat(image_flags_list, dim=0) if image_flags_list else None

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
    if hasattr(first_item, 'pixel_values') or hasattr(first_item, 'pixel_values_intern'):
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
    return items
