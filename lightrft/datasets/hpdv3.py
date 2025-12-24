import os
import copy
import json
import random
from typing import List, Dict, Any, Tuple, Union
from loguru import logger

from .utils import BaseDataHandler


class HPDv3Handler(BaseDataHandler):
    """
    Data Handler for HPDv3 dataset. Image-to-Text human preferences dataset.
    
    Paper: https://huggingface.co/MizzenAI/HPSv3
    Dataset Repo: https://huggingface.co/datasets/MizzenAI/HPDv3
    """
    def load_data(self, path: str) -> List[Dict[str, Any]]:
        try:
            with open(path, 'rb') as f:
                raw_data = json.load(f)
        except json.JSONDecodeError:
            # For JSON Lines format
            logger.warning(f"JSON: {path}")
            raw_data = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        raw_data.append(json.loads(line))

        # Attach data_root to each item
        data_root = os.path.dirname(path)
        for item in raw_data:
            item['data_root'] = data_root

        # Make sure all visual files exist
        valid_data = []
        for item in raw_data:
            visual_info = self.get_media_info(item)
            if visual_info is None:
                logger.warning(f"Skipping item due to missing visual files: {item}")
                continue
            valid_data.append(item)

        logger.info(f"Loaded {len(raw_data)} HPDv3 samples from {path}")
        return valid_data

    def get_media_info(self, item: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        data_root = item['data_root']

        # Build full local paths
        full_path1 = os.path.join(data_root, item['path1'])
        full_path2 = os.path.join(data_root, item['path2'])

        # Make sure files exist
        if not os.path.exists(full_path1) or not os.path.exists(full_path2):
            return None
        else:
            return {
                'preferred_image': {
                    'image_local_path': full_path1
                },
                'rejected_image': {
                    'image_local_path': full_path2
                }
            }

    def parse_item(self, item: Dict[str, Any], media_content: Dict[str, Any],
                   config: Dict[str, Any]) -> Tuple[List[Dict], List[Dict], Dict]:
        # Get loaded visual content
        preferred_image = media_content['preferred_image']
        rejected_image = media_content['rejected_image']

        if not all([preferred_image, rejected_image]):
            raise ValueError(f"Missing visual content for 'preferred_image' or 'rejected_image'.")

        # Get generation prompt from data item
        prompt_text = item["prompt"]
        if not prompt_text:
            raise ValueError(f"Missing generation prompt in item: {item}")

        # Get system prompts from config
        task_instruction_template = config["task_instruction"]
        task_instruction = task_instruction_template.format(prompt=prompt_text)

        # Random pick from "A" or "B" to avoid positional bias
        preference = random.choice(["A", "B"])
        if preference == "A":  # "A" means image0 is preferred
            image0, image1 = preferred_image, rejected_image
        else:
            image0, image1 = rejected_image, preferred_image

        # Build messages
        messages0 = [
            {
                "role": "system",
                "content": copy.deepcopy(task_instruction)
            },
            {
                "role": "user",
                "content": [{
                    "type": "image",
                    "image": image0,
                    "max_pixels": 720 * 480
                }  # to save memory
                            ]
            }
        ]

        messages1 = [{
            "role": "system",
            "content": copy.deepcopy(task_instruction)
        }, {
            "role": "user",
            "content": [{
                "type": "image",
                "image": image1,
                "max_pixels": 720 * 480
            }]
        }]

        other = {
            "preference": preference,
            "source": item["source"],
            "prompt": prompt_text,
            "confidence": item.get("confidence"),
            "choice_dist": item.get("choice_dist"),
            "model_chosen": item["model1"],
            "model_rejected": item["model2"],
            "preferred_path": item["path1"],
            "rejected_path": item["path2"],
        }
        return messages0, messages1, other


class HPDv3GRMHandler(HPDv3Handler):
    """
    Data Handler for HPDv3 dataset with Generative Reward Model (GRM) training.
    Inherits from HPDv3Handler but overrides parse_item to suit GRM needs.

    Paper: https://huggingface.co/MizzenAI/HPSv3
    Dataset Repo: https://huggingface.co/datasets/MizzenAI/HPDv3
    """
    def parse_item(self, item: Dict[str, Any], visual_content: Dict[str, Any],
                   config: Dict[str, Any]) -> Tuple[List[Dict], List[Dict], Dict]:
        # Get loaded visual content
        preferred_image = visual_content['preferred_image']
        rejected_image = visual_content['rejected_image']

        if not all([preferred_image, rejected_image]):
            raise ValueError(f"Missing visual content for 'preferred_image' or 'rejected_image'.")

        # Get generation prompt from data item
        prompt_text = item["prompt"]
        if not prompt_text:
            raise ValueError(f"Missing generation prompt in item: {item}")

        # Get system prompts from config
        task_instruction_template = config["task_instruction"]
        task_instruction = task_instruction_template.format(prompt=prompt_text)

        # Random pick from "A" or "B" to avoid positional bias
        preference = random.choice(["A", "B"])
        if preference == "A":  # "A" means image0 is preferred
            image0, image1 = preferred_image, rejected_image
        else:
            image0, image1 = rejected_image, preferred_image

        # Build messages
        messages = [
            {
                "role": "system",
                "content": task_instruction
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "**Image 1:**"
                    },
                    {
                        "type": "image",
                        "image": image0,
                        "max_pixels": 720 * 480
                    }  # to save memory
                ]
            },
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "**Image 2:**"
                }, {
                    "type": "image",
                    "image": image1,
                    "max_pixels": 720 * 480
                }]
            }
        ]
        # During evaluation, we do not include the response part in the messages
        is_training = config.get("is_training", True)
        if is_training:
            response = "<answer>Image 1 is better</answer>" if preference == "A" else "<answer>Image 2 is better</answer>"
            messages.append({"role": "assistant", "content": response})

        other = {
            "preference": preference,
            "source": item["source"],
            "prompt": prompt_text,
            "confidence": item.get("confidence"),
            "choice_dist": item.get("choice_dist"),
            "model_chosen": item["model1"],
            "model_rejected": item["model2"],
            "preferred_path": item["path1"],
            "rejected_path": item["path2"],
        }
        return messages, other
