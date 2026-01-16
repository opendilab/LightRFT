"""
This module provides utilities for processing and normalizing image data in vision-language models.
It includes functions for converting various image formats to PIL Images and handling batches
of single or multiple images per sample.

The main components are:
- to_pil: Converts various input formats (paths, bytes, dicts) to normalized PIL images
- normalize_images: Recursively processes lists of image inputs into PIL format
- get_images_num: Extracts the count of images for each sample in a batch

These utilities ensure consistent image formatting before processing by multimodal models or processors.
"""

import os
import pathlib
from typing import List, Union, Optional
from PIL import Image
from io import BytesIO


def to_pil(img) -> Image.Image:
    """
    Convert a single image input to PIL.Image format.

    :param img: Image input (PIL.Image, dict, str, Path, bytes, or bytearray)
    :type img: Union[PIL.Image.Image, dict, str, pathlib.Path, bytes, bytearray]

    :return: PIL.Image in RGB mode
    :rtype: PIL.Image.Image

    :raises ValueError: If input type is not supported

    **Example:**

    .. code-block:: python

        from PIL import Image
        img = to_pil("path/to/image.jpg")
        assert isinstance(img, Image.Image)
    """
    # Extract from dict if needed
    if isinstance(img, dict):
        img = img.get("image", img)

    # Load from file path
    if isinstance(img, (str, pathlib.Path)):
        img = str(img)
        if img.startswith("file://"):
            img = img[7:]
        img = os.path.expanduser(img)
        img = Image.open(img).convert("RGB")

    # Load from binary data
    if isinstance(img, (bytes, bytearray)):
        img = Image.open(BytesIO(img)).convert("RGB")

    return img


def normalize_images(raw_images: List) -> List:
    """
    Recursively normalize image inputs to PIL.Image format.

    Handles various input formats:
        - PIL.Image: returned unchanged
        - dict with 'image' key: extracts image value
        - str/pathlib.Path: loads from file path (supports file:// prefix)
        - bytes/bytearray: loads from binary data
        - list: recursively normalizes each element

    :param raw_images: List of image inputs in various formats
    :type raw_images: List[Union[PIL.Image.Image, dict, str, pathlib.Path, bytes, bytearray, List]]

    :return: List of normalized PIL.Image objects (or lists of PIL.Images for multi-image samples)
    :rtype: List[Union[PIL.Image.Image, List[PIL.Image.Image]]]

    **Example:**

    .. code-block:: python

        raw_imgs = ["path/to/img1.jpg", ["path/to/img2.jpg", "path/to/img3.jpg"]]
        normalized = normalize_images(raw_imgs)
        # Result: [PIL.Image, [PIL.Image, PIL.Image]]
    """
    normalized = []
    for item in raw_images:
        if isinstance(item, list):
            # Multi-image case: recursively normalize each image
            normalized.append([to_pil(img) for img in item])
        else:
            # Single image case
            normalized.append(to_pil(item))
    return normalized


def get_images_num(all_images: Optional[List]) -> Optional[List[int]]:
    """
    Extract the number of images for each sample.

    NOTE: We return 0 when a sample has no image (None). Using 1 as a
    placeholder caused mismatches between expected image placeholders and
    actual prompt replacements when mixing video-only and image samples.

    :param all_images: List of images (can be None, single images, or lists of images)
    :type all_images: Optional[List[Union[None, PIL.Image.Image, List[PIL.Image.Image]]]]

    :return: List of image counts per sample, or None if no images are provided
    :rtype: Optional[List[int]]

    **Example:**

    .. code-block:: python

        from PIL import Image
        imgs = [Image.new('RGB', (10, 10)), None, [Image.new('RGB', (10, 10)), Image.new('RGB', (10, 10))]]
        counts = get_images_num(imgs)
        assert counts == [1, 0, 2]
    """
    if all_images is None:
        return None

    counts = []
    for item in all_images:
        if item is None:
            counts.append(0)
        elif isinstance(item, Image.Image):
            counts.append(1)
        elif isinstance(item, list):
            # Check all elements are PIL.Images
            if any(not isinstance(sub, Image.Image) for sub in item):
                bad_item = next(sub for sub in item if not isinstance(sub, Image.Image))
                raise RuntimeError(f"Unsupported image type in list: {type(bad_item)}")
            counts.append(len(item))
        else:
            raise RuntimeError(f"Unsupported image type: {type(item)}")
    return counts
