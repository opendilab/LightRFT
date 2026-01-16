import os
import copy
import json
import random
from typing import List, Dict, Any, Tuple
from loguru import logger

from .utils import BaseDataHandler, get_task_instructions


class HPDv3Handler(BaseDataHandler):
    """
    Data Handler for HPDv3 dataset. Image-to-Text human preferences dataset.

    Paper: https://huggingface.co/MizzenAI/HPSv3
    Dataset Repo: https://huggingface.co/datasets/MizzenAI/HPDv3
    """
    task_type = "text-to-image"

    def load_data(self, path: str) -> List[Dict[str, Any]]:
        """
        Load and validate HPDv3 data from JSON or JSONL file.

        :param path: Path to the JSON/JSONL file
        :type path: str

        :return: List of valid samples with 'data_root' attached
        :rtype: List[Dict[str, Any]]

        **Example:**

        .. code-block:: python

            handler = HPDv3Handler()
            data = handler.load_data("path/to/HPDv3/data.json")
        """
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
        """
        Extract path info for the preferred and rejected images.

        :param item: A data item from load_data
        :type item: Dict[str, Any]

        :return: Dict containing local paths for 'preferred_image' and 'rejected_image', or None if files missing
        :rtype: Dict[str, Dict[str, str]]

        **Example:**

        .. code-block:: python

            info = handler.get_media_info(item)
        """
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
        """
        Parse a single HPDv3 item into message pairs and metadata for ranking.

        Randomly shuffles preferred/rejected images to avoid positional bias.

        :param item: The raw data item
        :type item: Dict[str, Any]
        :param media_content: Loaded visual content
        :type media_content: Dict[str, Any]
        :param config: Configuration for task instructions and max_pixels
        :type config: Dict[str, Any]

        :return: A tuple of (messages0, messages1, metadata)
        :rtype: Tuple[List[Dict], List[Dict], Dict]

        :raises ValueError: If required visual content or prompt is missing.

        **Example:**

        .. code-block:: python

            msg0, msg1, other = handler.parse_item(item, media_content, config)
        """
        # Get loaded visual content
        preferred_image = media_content['preferred_image']
        rejected_image = media_content['rejected_image']

        if not all([preferred_image, rejected_image]):
            raise ValueError("Missing visual content for 'preferred_image' or 'rejected_image'.")

        # Get generation prompt from data item
        prompt_text = item["prompt"]
        if not prompt_text:
            raise ValueError(f"Missing generation prompt in item: {item}")

        # Get system prompts from config
        task_instruction_template = config["task_instruction"]
        task_instruction = task_instruction_template.format(prompt=prompt_text)

        # Get max_pixels from config
        max_pixels = config["max_pixels"]

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
                    "max_pixels": max_pixels
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
                "max_pixels": max_pixels
            }]
        }]

        other = {
            "preference": preference,
            "source": item["source"],
            "task_type": self.task_type,
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
    def parse_item(self, item: Dict[str, Any], media_content: Dict[str, Any],
                   config: Dict[str, Any]) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Parse a single HPDv3 item for GRM training.

        :param item: The raw data item
        :type item: Dict[str, Any]
        :param media_content: Loaded visual content
        :type media_content: Dict[str, Any]
        :param config: Configuration for task instructions and max_pixels
        :type config: Dict[str, Any]

        :return: A tuple of (messages, metadata)
        :rtype: Tuple[List[Dict], Dict]

        **Example:**

        .. code-block:: python

            messages, other = handler.parse_item(item, media_content, config)
        """
        # Get loaded visual content
        preferred_image = media_content['preferred_image']
        rejected_image = media_content['rejected_image']

        if not all([preferred_image, rejected_image]):
            raise ValueError("Missing visual content for 'preferred_image' or 'rejected_image'.")

        # Get generation prompt from data item
        prompt_text = item["prompt"]
        if not prompt_text:
            raise ValueError(f"Missing generation prompt in item: {item}")

        # Get system prompts from config
        task_instruction_template = config["task_instruction"]
        task_instruction = task_instruction_template.format(prompt=prompt_text)

        # Get max_pixels from config
        max_pixels = config["max_pixels"]

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
                        "max_pixels": max_pixels
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
                    "max_pixels": max_pixels
                }]
            }
        ]

        response = "<answer>Image 1 is better</answer>" if preference == "A" else "<answer>Image 2 is better</answer>"
        messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})

        other = {
            "preference": preference,
            "response": response,
            "source": item["source"],
            "task_type": self.task_type,
            "prompt": prompt_text,
            "confidence": item.get("confidence"),
            "choice_dist": item.get("choice_dist"),
            "model_chosen": item["model1"],
            "model_rejected": item["model2"],
            "preferred_path": item["path1"],
            "rejected_path": item["path2"],
        }
        return messages, other


class HPDv3PairHandler(HPDv3Handler):
    """
    Data Handler for HPDv3 dataset in pairwise format.
    Inherits from HPDv3Handler but overrides parse_item to suit pairwise training.

    Paper: https://huggingface.co/MizzenAI/HPSv3
    Dataset Repo: https://huggingface.co/datasets/MizzenAI/HPDv3
    """
    def parse_item(self, item: Dict[str, Any], visual_content: Dict[str, Any],
                   config: Dict[str, Any]) -> Tuple[List[Dict], Dict]:
        """
        Parse a data item into pairwise messages and metadata.

        :param item: The raw data item
        :type item: Dict[str, Any]
        :param visual_content: Loaded visual content
        :type visual_content: Dict[str, Any]
        :param config: Configuration for task instructions and max_pixels
        :type config: Dict[str, Any]

        :return: A tuple of (messages, metadata)
        :rtype: Tuple[List[Dict], Dict]

        **Example:**

        .. code-block:: python

            messages, other = handler.parse_item(item, media_content, config)
        """
        # Get loaded visual content
        preferred_image = visual_content['preferred_image']
        rejected_image = visual_content['rejected_image']

        if not all([preferred_image, rejected_image]):
            raise ValueError("Missing visual content for 'preferred_image' or 'rejected_image'.")

        # Get generation prompt from data item
        prompt_text = item["prompt"]
        if not prompt_text:
            raise ValueError(f"Missing generation prompt in item: {item}")

        # Get system prompts from config
        task_instruction_template = get_task_instructions(self, config)
        task_instruction = task_instruction_template.format(prompt=prompt_text)

        # Get max_pixels from config
        max_pixels = config["max_pixels"]

        # Random pick from "A" or "B" to avoid positional bias
        preference = random.choice(["A", "B"])
        if preference == "A":  # "A" means image0 is preferred
            image0, image1 = preferred_image, rejected_image
        else:
            image0, image1 = rejected_image, preferred_image

        # Build messages
        messages = [{
            "role": "system",
            "content": task_instruction
        }, {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "The following is the first image."
            }, {
                "type": "image",
                "image": image0,
                "max_pixels": max_pixels
            }]
        }, {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "The following is the second image."
            }, {
                "type": "image",
                "image": image1,
                "max_pixels": max_pixels
            }]
        }]

        other = {
            "preference": preference,
            "reward_rule_label": "general",
            "source": item["source"],
            "task_type": self.task_type,
            "prompt": prompt_text,
            "confidence": item.get("confidence"),
            "choice_dist": item.get("choice_dist"),
            "model_chosen": item["model1"],
            "model_rejected": item["model2"],
            "preferred_path": item["path1"],
            "rejected_path": item["path2"],
        }
        return messages, other
