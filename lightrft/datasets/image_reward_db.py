import os
import copy
import json
import random
import glob
from typing import List, Dict, Any, Tuple, Union
from itertools import combinations
from collections import defaultdict
from tqdm import tqdm
from loguru import logger

from .utils import BaseDataHandler


class ImageRewardDBHandler(BaseDataHandler):
    """
    Data Handler for ImageRewardDB dataset.
    
    Paper: https://arxiv.org/abs/2304.05977
    Dataset Repo: https://huggingface.co/datasets/zai-org/ImageRewardDB
    """
    def load_data(self, path: str) -> List[Dict[str, Any]]:
        """Load ImageRewardDB shards and build preference pairs.

        This method scans the given dataset root for ImageRewardDB JSON shards and
        aggregates image entries by ``prompt_id``. For each prompt group, it
        constructs all unordered pairs of images and determines the preferred
        image based on the ``rank`` field (smaller is better, i.e., ``1`` is best).

        :param path: Path to the dataset root directory of ImageRewardDB.
        :type path: str
        
        :return: List of preference pair dictionaries.
        :rtype: List[Dict[str, Any]]
            - ``prompt_id`` (str): Unique identifier for the prompt group.
            - ``prompt`` (str): The text prompt used to generate images.
            - ``classification`` (str): Optional category label; ``"Unknown"`` if missing.
            - ``data_root`` (str): Echo of the provided ``path`` for later path resolution.
            - ``chosen_img`` (str): Relative path of the preferred image.
            - ``rank_chosen`` (int): Rank of the preferred image.
            - ``overall_rating_chosen`` (Optional[float|int]): Optional quality score of the preferred image.
            - ``rejected_img`` (str): Relative path of the non-preferred image.
            - ``rank_rejected`` (int): Rank of the non-preferred image.
            - ``overall_rating_rejected`` (Optional[float|int]): Optional quality score of the non-preferred image.
        """

        # Locate all JSON shard files
        # Expected layout examples: train_01/train_01.json, train_02/train_02.json, ...
        search_pattern = os.path.join(path, "**", "*.json")
        json_files = glob.glob(search_pattern, recursive=True)

        if not json_files:
            print(f"No JSON files found under: {path}. Please verify the dataset path.")
            return

        print(f"Found {len(json_files)} JSON files. Starting to load data...")

        # Aggregate entries by prompt_id
        # Structure: { "prompt_id_1": [img_info1, img_info2, ...], ... }
        grouped_data = defaultdict(list)
        for json_path in json_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        grouped_data[item['prompt_id']].append(item)
            except Exception as e:
                print(f"Error reading file {json_path}: {e}")

        print(f"Aggregated {len(grouped_data)} unique prompt groups. Generating pairs...")

        # Construct pairs
        preference_pairs = []
        skipped_files_count = 0
        for pid, items in grouped_data.items():
            # Filter out items with missing or empty image files
            valid_items = []
            dataset_root = os.path.dirname(os.path.dirname(path))
            for item in items:
                full_img_path = os.path.join(dataset_root, item['image_path'])
                # import ipdb; ipdb.set_trace()
                if os.path.exists(full_img_path) and os.path.getsize(full_img_path) > 0:
                    valid_items.append(item)
                else:
                    # print(f"[Warning] Skipping missing or empty file: {full_img_path}")
                    skipped_files_count += 1

            items = valid_items
            # If a prompt group has fewer than 2 images, skip
            if len(items) < 2:
                continue

            # Use combinations to generate all unordered pairs (e.g., AB and BA appear once)
            for item_a, item_b in combinations(items, 2):

                rank_a = item_a['rank']
                rank_b = item_b['rank']

                # Defensive: ensure rank fields are present
                if rank_a is None or rank_b is None:
                    continue

                # Decision rule: smaller rank value is better (1 is best)
                if rank_a == rank_b:
                    continue

                if rank_a < rank_b:
                    chosen = item_a
                    reject = item_b
                else:
                    chosen = item_b
                    reject = item_a

                # Build pair entry
                pair_entry = {
                    "prompt_id": pid,
                    "prompt": chosen['prompt'],
                    "classification": chosen.get('classification', 'Unknown'),
                    "data_root": dataset_root,

                    # Chosen Image Info
                    "chosen_img": chosen['image_path'],
                    "rank_chosen": chosen['rank'],
                    "overall_rating_chosen": chosen.get('overall_rating'),

                    # Rejected Image Info
                    "rejected_img": reject['image_path'],
                    "rank_rejected": reject['rank'],
                    "overall_rating_rejected": reject.get('overall_rating'),
                }
                preference_pairs.append(pair_entry)

        logger.info(f"Loaded {len(preference_pairs)} samples from ImageRewardDB.")
        return preference_pairs

    def get_media_info(self, item: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        data_root = item['data_root']

        # Build full local paths
        full_path1 = os.path.join(data_root, item['chosen_img'])
        full_path2 = os.path.join(data_root, item['rejected_img'])

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
                    "max_pixels": 1280 * 720
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
                "max_pixels": 1280 * 720
            }]
        }]

        other = {
            "preference": preference,  # used for reward head labeling
            "source": item["source"],
            "prompt_id": item["prompt_id"],
            "prompt": prompt_text,
            "chosen_img": item["chosen_img"],
            "rejected_img": item["rejected_img"],
            "rank_chosen": item["rank_chosen"],
            "rank_rejected": item["rank_rejected"],
            "overall_rating_chosen": item["overall_rating_chosen"],
            "overall_rating_rejected": item["overall_rating_rejected"],
            "classification": item["classification"],
        }
        return messages0, messages1, other
