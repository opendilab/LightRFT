import os
import json
from typing import List, Dict, Any, Tuple
from loguru import logger

from .utils import BaseDataHandler


class ImageGenCoTRewardHandler(BaseDataHandler):
    """
    Data handler for ImageGen-CoT-Reward-5K dataset. For Text-to-Image generation task.

    Paper: https://arxiv.org/pdf/2505.03318
    Dataset Repo: https://huggingface.co/datasets/CodeGoat24/ImageGen-CoT-Reward-5K
    """
    def load_data(self, path: str) -> List[Dict[str, Any]]:
        """
        Loads data from json file.
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
    ) -> Tuple[List[Dict], List[Dict], Dict]:

        image0 = media_content['image0']
        image1 = media_content['image1']

        if not all([image0, image1]):
            raise ValueError("Missing visual content for 'image0' or 'image1'.")

        # Get conversations from data item
        conversations = item["conversations"]
        system_prompt = conversations[0]['value']
        response = conversations[-1]['value']

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
                "image": image0
            }]
        }, {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "**Image 2:**"
            }, {
                "type": "image",
                "image": image1
            }]
        },{
            "role": "assistant", 
            "content": [{
                "type": "text", 
                "text": response
            }]
        }]

        other = {
            "source": item['source'],
            "data_item": item,
            "system_prompt": system_prompt,
            "response": response,
        }
        return messages, other
