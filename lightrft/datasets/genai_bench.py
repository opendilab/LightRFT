import os
import json
import glob
import random
import numpy as np
import pyarrow.parquet as pq
from typing import List, Dict, Any, Tuple
from loguru import logger

from .utils import BaseDataHandler, get_task_instructions


class GenAIBenchPairHandler(BaseDataHandler):
    """
    Data Handler for GenAI-Bench dataset.

    Paper: https://arxiv.org/pdf/2406.13743
    Dataset Repo: https://huggingface.co/datasets/BaiqiL/GenAI-Bench
    """
    task_type = "text-to-image"

    def load_data(self, path: str) -> List[Dict[str, Any]]:
        """
        Loads data from parquet file(s) and expands into pairs.
        Each row in GenAI-Bench has multiple images from different models.

        :param path: Path to the parquet file or directory
        :type path: str

        :return: List of expanded pairs with image bytes and metadata
        :rtype: List[Dict[str, Any]]

        **Example:**

        .. code-block:: python

            handler = GenAIBenchPairHandler()
            data = handler.load_data("path/to/GenAI-Bench/data.parquet")
        """

        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "*.parquet")))
        elif os.path.isfile(path):
            files = [path]
        else:
            files = sorted(glob.glob(path))

        if not files:
            logger.warning(f"No parquet files found at path: {path}")
            return []

        all_raw_items = []
        for f in files:
            data_table = pq.read_table(f)
            raw_items = [{
                name: col[i].as_py()
                for name, col in zip(data_table.column_names, data_table.itercolumns())
            }
                         for i in range(data_table.num_rows)]
            all_raw_items.extend(raw_items)

        expanded_data = []
        for item in all_raw_items:
            human_ratings = item.get('HumanRatings', {})
            if not human_ratings:
                continue

            # Dynamically get models that have both ratings and image data in the current item
            valid_models = [m for m in human_ratings.keys() if m in item and item[m] is not None]

            # Calculate mean rating for each valid model
            mean_ratings = {m: np.mean(human_ratings[m]) for m in valid_models}

            # Generate all pairs (A, B) from valid models
            for i in range(len(valid_models)):
                for j in range(i + 1, len(valid_models)):
                    m1 = valid_models[i]
                    m2 = valid_models[j]

                    r1 = mean_ratings[m1]
                    r2 = mean_ratings[m2]

                    # Store as a pair if there is a preference (or handle ties)
                    # For evaluation, we only care about clear preferences usually,
                    # but GenAI-Bench might have ties.

                    if r1 > r2:
                        preference = 'A'  # m1 is better
                    elif r2 > r1:
                        preference = 'B'  # m2 is better
                    else:
                        preference = 'C'  # Tie

                    pair_item = {
                        'prompt': item['Prompt'],
                        'image1_bytes': item[m1]['bytes'],
                        'image2_bytes': item[m2]['bytes'],
                        'model1': m1,
                        'model2': m2,
                        'preference': preference,
                        'index': item['Index']
                    }
                    expanded_data.append(pair_item)

        logger.info(f"Loaded {len(raw_items)} samples, expanded into {len(expanded_data)} pairs from {path}")
        return expanded_data

    def get_media_info(self, item: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Extract media info (bytes) for the two images.

        :param item: A data item from load_data
        :type item: Dict[str, Any]

        :return: Dict containing image bytes for 'image1' and 'image2'
        :rtype: Dict[str, Dict[str, Any]]

        **Example:**

        .. code-block:: python

            info = handler.get_media_info(item)
        """
        return {'image1': {'image_bytes': item['image1_bytes']}, 'image2': {'image_bytes': item['image2_bytes']}}

    def parse_item(self, item: Dict[str, Any], media_content: Dict[str, Any],
                   config: Dict[str, Any]) -> Tuple[List[Dict], Dict]:
        """
        Parse a data item into messages and metadata.

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
        image1 = media_content['image1']

        image1 = media_content['image1']
        image2 = media_content['image2']

        if not all([image1, image2]):
            raise ValueError("Missing visual content for 'image1' or 'image2'.")

        # Get generation prompt from data item
        prompt_text = item["prompt"]

        # Get system prompts from config
        task_instruction_template = get_task_instructions(self, config)
        task_instruction = task_instruction_template.format(prompt=prompt_text)

        # Get max_pixels from config
        max_pixels = config["max_pixels"]

        # Random flip to avoid positional bias
        flip = random.random() > 0.5
        if flip:
            img_a, img_b = image2, image1
            if item['preference'] == 'A':
                actual_preference = 'B'
            elif item['preference'] == 'B':
                actual_preference = 'A'
            else:
                actual_preference = 'C'
        else:
            img_a, img_b = image1, image2
            actual_preference = item['preference']

        # Build messages for training or evaluation
        messages = [{
            "role": "system",
            "content": task_instruction
        }, {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "**Image 1:**"
            }, {
                "type": "image",
                "image": img_a,
                "max_pixels": max_pixels
            }]
        }, {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "**Image 2:**"
            }, {
                "type": "image",
                "image": img_b,
                "max_pixels": max_pixels
            }]
        }]

        # Note that GenAI-Bench is only used for evaluation, so we do not add generation prompt here.

        other = {
            "preference": actual_preference,
            "reward_rule_label": "general",
            "prompt": prompt_text,
            "model1": item['model1'] if not flip else item['model2'],
            "model2": item['model2'] if not flip else item['model1'],
            "index": item['index'],
            "source": item.get("source", "genai_bench")
        }

        return messages, other


class GenAIBenchVideoPairHandler(BaseDataHandler):
    """
    Data Handler for GenAI-Bench-Video dataset.

    Paper: https://arxiv.org/pdf/2406.13743
    Dataset Repo: https://huggingface.co/datasets/BaiqiL/GenAI-Bench
    """
    task_type = "text-to-video"

    def load_data(self, path: str) -> List[Dict[str, Any]]:
        """
        Loads data from json file and expands into pairs.
        Each row in GenAI-Bench-Video has multiple videos from different models.

        :param path: Path to the JSON file or directory
        :type path: str

        :return: List of expanded pairs with video paths and metadata
        :rtype: List[Dict[str, Any]]

        **Example:**

        .. code-block:: python

            handler = GenAIBenchVideoPairHandler()
            data = handler.load_data("path/to/GenAI-Bench-Video/genai_video.json")
        """
        if os.path.isdir(path):
            json_path = os.path.join(path, "genai_video.json")
        else:
            json_path = path

        if not os.path.exists(json_path):
            logger.error(f"GenAI-Bench-Video metadata not found at {json_path}")
            return []

        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data_dict = json.load(f)

        data_root = os.path.dirname(json_path)

        expanded_data = []
        for sample_id, item in raw_data_dict.items():
            models_data = item.get('models', {})
            model_names = list(models_data.keys())

            # Generate all pairs (A, B)
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    m1 = model_names[i]
                    m2 = model_names[j]

                    # Verify video files exist
                    # Videos are expected in folders like data_root/ModelName/sample_id.mp4
                    v1_path = os.path.join(data_root, m1, f"{sample_id}.mp4")
                    v2_path = os.path.join(data_root, m2, f"{sample_id}.mp4")

                    if not os.path.exists(v1_path) or not os.path.exists(v2_path):
                        # Some models might not have all videos
                        continue

                    r1 = np.mean(models_data[m1])
                    r2 = np.mean(models_data[m2])

                    if r1 > r2:
                        preference = 'A'
                    elif r2 > r1:
                        preference = 'B'
                    else:
                        preference = 'C'  # Tie

                    pair_item = {
                        'id': sample_id,
                        'prompt': item['prompt'],
                        'video1_path': v1_path,
                        'video2_path': v2_path,
                        'model1': m1,
                        'model2': m2,
                        'preference': preference,
                        'data_root': data_root
                    }
                    expanded_data.append(pair_item)

        logger.info(f"Loaded {len(raw_data_dict)} samples, expanded into {len(expanded_data)} pairs from {json_path}")
        return expanded_data

    def get_media_info(self, item: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """
        Extract media info (paths) for the two videos.

        :param item: A data item from load_data
        :type item: Dict[str, Any]

        :return: Dict containing local paths for 'video1' and 'video2'
        :rtype: Dict[str, Dict[str, str]]

        **Example:**

        .. code-block:: python

            info = handler.get_media_info(item)
        """
        return {
            'video1': {
                'video_local_path': item['video1_path']
            },
            'video2': {
                'video_local_path': item['video2_path']
            }
        }

    def parse_item(self, item: Dict[str, Any], media_content: Dict[str, Any],
                   config: Dict[str, Any]) -> Tuple[List[Dict], Dict]:
        """
        Parse a data item into messages and metadata.

        :param item: The raw data item
        :type item: Dict[str, Any]
        :param media_content: Loaded visual content
        :type media_content: Dict[str, Any]
        :param config: Configuration for task instructions, max_pixels, and fps
        :type config: Dict[str, Any]

        :return: A tuple of (messages, metadata)
        :rtype: Tuple[List[Dict], Dict]

        **Example:**

        .. code-block:: python

            messages, other = handler.parse_item(item, media_content, config)
        """
        video1 = media_content['video1']
        video2 = media_content['video2']

        if not all([video1, video2]):
            raise ValueError(f"Missing video content for id {item['id']}.")

        # Get generation prompt from data item
        prompt_text = item["prompt"]

        # Get system prompts from config
        task_instruction_template = get_task_instructions(self, config)
        task_instruction = task_instruction_template.format(prompt=prompt_text)

        # Get FPS and max_pixels from config
        fps = config["video_fps"]
        max_pixels = config["max_pixels"]

        # Random flip to avoid positional bias
        flip = random.random() > 0.5
        if flip:
            vid_a, vid_b = video2, video1
            if item['preference'] == 'A':
                actual_preference = 'B'
            elif item['preference'] == 'B':
                actual_preference = 'A'
            else:
                actual_preference = 'C'
        else:
            vid_a, vid_b = video1, video2
            actual_preference = item['preference']

        # Build messages
        messages = [{
            "role": "system",
            "content": task_instruction
        }, {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "**Video 1:**"
            }, {
                "type": "video",
                "video": vid_a,
                "fps": fps,
                "max_pixels": max_pixels
            }]
        }, {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "**Video 2:**"
            }, {
                "type": "video",
                "video": vid_b,
                "fps": fps,
                "max_pixels": max_pixels
            }]
        }]

        other = {
            "id": item["id"],
            "preference": actual_preference,
            "prompt": prompt_text,
            "model1": item['model1'] if not flip else item['model2'],
            "model2": item['model2'] if not flip else item['model1'],
            "source": item.get("source", "genai_bench_video"),
            "reward_rule_label": "general",
        }

        return messages, other
