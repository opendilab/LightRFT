import os
import json
from typing import List, Dict, Any, Tuple
from loguru import logger

from .utils import BaseDataHandler


class ImageGenCoTRewardGRMHandler(BaseDataHandler):
    """
    Data handler for ImageGen-CoT-Reward-5K dataset. For Text-to-Image generation task.

    Paper: https://arxiv.org/pdf/2505.03318
    Dataset Repo: https://huggingface.co/datasets/CodeGoat24/ImageGen-CoT-Reward-5K
    """
    task_type = "text-to-image"

    def load_data(self, path: str) -> List[Dict[str, Any]]:
        """
        Loads data from json file.

        :param path: Path to the dataset JSON file
        :type path: str

        :return: List of samples with 'data_root' attached
        :rtype: List[Dict[str, Any]]

        **Example:**

        .. code-block:: python

            handler = ImageGenCoTRewardHandler()
            data = handler.load_data("path/to/ImageGen-CoT-Reward.json")
        """
        raw_data = []
        with open(path, 'rb') as f:
            raw_data = json.load(f)

        data_root = os.path.dirname(path)
        for item in raw_data:
            item['data_root'] = data_root

        logger.info(f"Loaded {len(raw_data)} samples from {path}")
        return raw_data

    def get_media_info(self, item: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """
        Extract path info for the two images.

        :param item: A data item from load_data
        :type item: Dict[str, Any]

        :return: Dict containing local paths for 'image0' and 'image1'
        :rtype: Dict[str, Dict[str, str]]

        **Example:**

        .. code-block:: python

            info = handler.get_media_info(item)
        """
        data_root = item['data_root']
        if not data_root:
            raise ValueError("Missing 'data_root' in item. Cannot resolve image paths.")
        images = item['images']
        image0_full_path = os.path.join(data_root, images[0])
        image1_full_path = os.path.join(data_root, images[1])

        return {
            'image0': {
                'image_local_path': image0_full_path
            },
            'image1': {
                'image_local_path': image1_full_path
            },
        }

    def parse_item(
        self,
        item: Dict[str, Any],
        media_content: Dict[str, Any],
        config: Dict[str, Any] | None,
    ) -> Tuple[List[Dict], Dict]:
        """
        Parse a single ImageGen-CoT-Reward item into message pairs.

        :param item: Raw data item from ImageGen-CoT-Reward dataset.
        :type item: Dict[str, Any]
        :param media_content: Loaded image content (PIL images/bytes)
        :type media_content: Dict[str, Any]
        :param config: Configuration for max_pixels
        :type config: Dict[str, Any]

        :return: A tuple of (messages, metadata)
        :rtype: Tuple[List[Dict], Dict]

        **Example:**

        .. code-block:: python

            messages, other = handler.parse_item(item, media_content, config)
        """
        image0 = media_content['image0']
        image1 = media_content['image1']

        if not all([image0, image1]):
            raise ValueError("Missing visual content for 'image0' or 'image1'.")

        # Get conversations from data item
        conversations = item["conversations"]
        system_prompt = conversations[0]['value']
        response = conversations[-1]['value']

        # Get max_pixels from config
        max_pixels = config["max_pixels"]

        # Build messages
        messages = [{
            "role": "system",
            "content": system_prompt
        }, {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "**Image 1:**"
            }, {
                "type": "image",
                "image": image0,
                "max_pixels": max_pixels
            }]
        }, {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "**Image 2:**"
            }, {
                "type": "image",
                "image": image1,
                "max_pixels": max_pixels
            }]
        }, {
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": response
            }]
        }]

        other = {
            "source": item['source'],
            "task_type": self.task_type,
            "data_item": item,
            "system_prompt": system_prompt,
            "response": response,
        }
        return messages, other
