import os
import random
import pandas as pd
from typing import List, Dict, Any, Tuple, Union
from loguru import logger

from .utils import BaseDataHandler, get_task_instructions


class VideoGenRewardBenchPairHandler(BaseDataHandler):
    """
    Data Handler for VideoGen-RewardBench dataset in pairwise format.

    Paper: https://arxiv.org/abs/2501.13918
    Dataset Repo: https://huggingface.co/KlingTeam
    """
    task_type = "text-to-video"

    def load_data(self, path: str) -> List[Dict[str, Any]]:
        """
        Loads data from CSV file.

        :param path: Path to the CSV file
        :type path: str

        :return: List of valid samples with 'data_root' attached
        :rtype: List[Dict[str, Any]]

        **Example:**

        .. code-block:: python

            handler = VideoGenRewardBenchPairHandler()
            data = handler.load_data("path/to/VideoGen-RewardBench/data.csv")
        """
        try:
            df = pd.read_csv(path)
            # Filter out rows with missing labels
            df = df.dropna(subset=['Overall'])
            raw_data = df.to_dict('records')
        except Exception as e:
            logger.error(f"Error loading CSV from {path}: {e}")
            return []

        # Attach data_root to each item
        data_root = os.path.dirname(path)
        for item in raw_data:
            item['data_root'] = data_root

        # Make sure all visual files exist
        valid_data = []
        for item in raw_data:
            visual_info = self.get_media_info(item)
            if visual_info is None:
                # logger.warning(f"Skipping item due to missing visual files: {item}")
                continue
            valid_data.append(item)

        logger.info(f"Loaded {len(valid_data)} VideoGen-RewardBench samples from {path}")
        return valid_data

    def get_media_info(self, item: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """
        Extract path info for the two videos.

        :param item: A data item from load_data
        :type item: Dict[str, Any]

        :return: Dict containing local paths for 'video_A' and 'video_B'
        :rtype: Dict[str, Dict[str, str]]

        **Example:**

        .. code-block:: python

            info = handler.get_media_info(item)
        """
        data_root = item['data_root']

        # Build full local paths
        full_path_A = os.path.join(data_root, item['path_A'])
        full_path_B = os.path.join(data_root, item['path_B'])

        return {'video_A': {'video_local_path': full_path_A}, 'video_B': {'video_local_path': full_path_B}}

    def parse_item(self, item: Dict[str, Any], media_content: Dict[str, Any],
                   config: Dict[str, Any]) -> Tuple[List[Dict], Dict]:
        """
        Parse a data item into messages and metadata.

        :param item: The raw data item
        :type item: Dict[str, Any]
        :param media_content: Loaded video content
        :type media_content: Dict[str, Any]
        :param config: Configuration for task instructions, max_pixels, and fps
        :type config: Dict[str, Any]

        :return: A tuple of (messages, metadata)
        :rtype: Tuple[List[Dict], Dict]

        **Example:**

        .. code-block:: python

            messages, other = handler.parse_item(item, media_content, config)
        """
        # Get loaded visual content
        video_A = media_content['video_A']
        video_B = media_content['video_B']

        if not all([video_A, video_B]):
            raise ValueError("Missing visual content for 'video_A' or 'video_B'.")

        # Get generation prompt from data item
        prompt_text = item["prompt"]
        if not prompt_text:
            raise ValueError(f"Missing generation prompt in item: {item}")

        # Get system prompts from config
        task_instruction_template = get_task_instructions(self, config)
        task_instruction = task_instruction_template.format(prompt=prompt_text)

        # Identify preferred and rejected videos
        overall_label = item["Overall"]  # "A" or "B" or "same"
        if overall_label == "A":
            preferred_video, rejected_video = video_A, video_B
            # preferred_fps, rejected_fps = item.get("fps_A", 2.0), item.get("fps_B", 2.0)
        elif overall_label == "B":
            preferred_video, rejected_video = video_B, video_A
            # preferred_fps, rejected_fps = item.get("fps_B", 2.0), item.get("fps_A", 2.0)
        elif overall_label == "same":
            preferred_video, rejected_video = video_A, video_B
            # preferred_fps, rejected_fps = item.get("fps_A", 2.0), item.get("fps_B", 2.0)
        else:
            raise ValueError(f"Invalid Overall label: {overall_label}")

        if overall_label == "same":
            preference = "C"
            video0, video1 = preferred_video, rejected_video
            # fps0, fps1 = preferred_fps, rejected_fps
        else:
            # Random pick from "A" or "B" to avoid positional bias
            # "A" means video0 is preferred, "B" means video1 is preferred
            preference = random.choice(["A", "B"])
            if preference == "A":
                video0, video1 = preferred_video, rejected_video
                # fps0, fps1 = preferred_fps, rejected_fps
            else:
                video0, video1 = rejected_video, preferred_video
                # fps0, fps1 = rejected_fps, preferred_fps

        # Get max_pixels from config
        max_pixels = config["max_pixels"]

        # Get FPS from config
        fps = config["video_fps"]

        # Build messages
        messages = [{
            "role": "system",
            "content": task_instruction
        }, {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "The following is the first video."
            }, {
                "type": "video",
                "video": video0,
                "fps": fps,
                "max_pixels": max_pixels
            }]
        }, {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "The following is the second video."
            }, {
                "type": "video",
                "video": video1,
                "fps": fps,
                "max_pixels": max_pixels
            }]
        }]

        other = {
            "preference": preference,
            "reward_rule_label": "general",
            "task_type": self.task_type,
            "prompt": prompt_text,
            "A_model": item.get("A_model"),
            "B_model": item.get("B_model"),
            "path_A": item.get("path_A"),
            "path_B": item.get("path_B"),
            "VQ": item.get("VQ"),
            "MQ": item.get("MQ"),
            "TA": item.get("TA"),
            "Overall": item.get("Overall"),
        }
        return messages, other
