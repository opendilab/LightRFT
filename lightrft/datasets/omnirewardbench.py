import os
import copy
from typing import List, Dict, Any, Tuple
from loguru import logger

from .utils import BaseDataHandler


class OmniRewardBenchT2IHandler(BaseDataHandler):
    """
    Data Handler for OmniRewardBench text-to-image human preferences benchmark.
    Process for scalar reward model training of pairwise-ranking task.

    Paper: https://huggingface.co/papers/2510.23451
    Dataset Repo: https://huggingface.co/datasets/HongbangYuan/OmniRewardBench
    """
    def load_data(self, path: str) -> List[Dict[str, Any]]:
        """
        Loads data from parquet file.
        """
        raw_data = []
        import pyarrow.parquet as pq
        data_table = pq.read_table(path)
        raw_data = [{name: col[i].as_py()
                     for name, col in zip(data_table.column_names, data_table.itercolumns())}
                    for i in range(data_table.num_rows)]

        data_root = os.path.dirname(os.path.dirname(path))
        for item in raw_data:
            item['data_root'] = data_root

        logger.info(f"Loaded {len(raw_data)} samples from {path}")
        return raw_data

    def get_media_info(self, item: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """
        Extract path info for the two videos.
        """
        data_root = item['data_root']
        if not data_root:
            raise ValueError("Missing 'data_root' in item. Cannot resolve video paths.")

        full_path1 = os.path.join(data_root, "media_data", item['response1_path'])
        full_path2 = os.path.join(data_root, "media_data", item['response2_path'])

        return {'image1': {'image_local_path': full_path1}, 'image2': {'image_local_path': full_path2}}

    def _get_label(self, choice: str) -> str:
        if choice == "response1":
            return "A"
        elif choice == "response2":
            return "B"
        else:
            return "C"  # TIE

    def parse_item(self, item: Dict[str, Any], media_content: Dict[str, Any],
                   config: Dict[str, Any]) -> Tuple[List[Dict], List[Dict], Dict]:

        image1 = media_content['image1']
        image2 = media_content['image2']

        if not all([image1, image2]):
            raise ValueError("Missing visual content for 'image1' or 'image2'.")

        # Get generation prompt from data item
        gen_prompt = item["prompt"]

        # Get system prompts from config
        task_instruction_template = config["task_instruction"]
        task_instruction = task_instruction_template.format(prompt=gen_prompt)
        # criteria = item["criteria"]

        # Build messages
        messages0 = [
            {
                "role": "system",
                "content": copy.deepcopy(task_instruction)
            },
            # {"role": "system", "content": f"Please give your evaluation considering the following criteria: {criteria}."},  # noqa: E501
            {
                "role": "user",
                "content": [{
                    "type": "image",
                    "image": image1,
                    "max_pixels": 1280 * 720
                }]
            }
        ]

        messages1 = [
            {
                "role": "system",
                "content": copy.deepcopy(task_instruction)
            },
            # {"role": "system", "content": f"Please give your evaluation considering the following criteria: {criteria}."},  # noqa: E501
            {
                "role": "user",
                "content": [{
                    "type": "image",
                    "image": image2,
                    "max_pixels": 1280 * 720
                }]
            }
        ]

        # Get human preference labels based on weighted scores
        pref_label = self._get_label(item["criteria_preference"])

        other = {
            "preference": pref_label,
            "criteria": item["criteria"],
            "criteria_preference": item["criteria_preference"],
            "id": item["id"],
            "prompt": gen_prompt,
            "source": item['source'],
            "image1_path": item['response1_path'],
            "image2_path": item['response2_path'],
            "model1": item['model1'],
            "model2": item['model2'],
        }
        return messages0, messages1, other


class OmniRewardBenchT2VHandler(OmniRewardBenchT2IHandler):
    """
    Data Handler for OmniRewardBench text-to-video human preferences benchmark.
    Process for scalar reward model training of pairwise-ranking task.

    Paper: https://huggingface.co/papers/2510.23451
    Dataset Repo: https://huggingface.co/datasets/HongbangYuan/OmniRewardBench
    """
    def get_media_info(self, item: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """
        Extract path info for the two videos.
        """
        data_root = item['data_root']
        if not data_root:
            raise ValueError("Missing 'data_root' in item. Cannot resolve video paths.")

        full_path1 = os.path.join(data_root, "media_data", item['response1'])
        full_path2 = os.path.join(data_root, "media_data", item['response2'])

        return {'video1': {'video_local_path': full_path1}, 'video2': {'video_local_path': full_path2}}

    def parse_item(self, item: Dict[str, Any], media_content: Dict[str, Any],
                   config: Dict[str, Any]) -> Tuple[List[Dict], List[Dict], Dict]:

        video1 = media_content['video1']
        video2 = media_content['video2']

        if not all([video1, video2]):
            raise ValueError("Missing visual content for 'video1' or 'video2'.")

        # Get generation prompt from data item
        gen_prompt = item["prompt"]

        # Get system prompts from config
        task_instruction_template = config["task_instruction"]
        task_instruction = task_instruction_template.format(prompt=gen_prompt)

        # Get FPS from config
        fps = config["video_fps"]

        # Build messages
        messages0 = [{
            "role": "system",
            "content": copy.deepcopy(task_instruction)
        }, {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "Please evaluate the following video based on the given task instruction."
            }, {
                "type": "video",
                "video": video1,
                "fps": fps,
                "max_pixels": 720 * 480
            }]
        }]

        messages1 = [{
            "role": "system",
            "content": copy.deepcopy(task_instruction)
        }, {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "Please evaluate the following video based on the given task instruction."
            }, {
                "type": "video",
                "video": video2,
                "fps": fps,
                "max_pixels": 720 * 480
            }]
        }]

        # Get human preference labels based on weighted scores
        pref_label = self._get_label(item["criteria_preference"])

        other = {
            "preference": pref_label,
            "criteria": item["criteria"],
            "criteria_preference": item["criteria_preference"],
            "id": item["id"],
            "prompt": gen_prompt,
            "source": item['source'],
            "video1_path": item['response1'],
            "video2_path": item['response2'],
            "model1": item['model1'],
            "model2": item['model2'],
        }
        return messages0, messages1, other


class OmniRewardBenchT2AHandler(OmniRewardBenchT2IHandler):
    """
    Data Handler for OmniRewardBench text-to-audio human preferences benchmark.
    Process for scalar reward model training of pairwise-ranking task.

    Paper: https://huggingface.co/papers/2510.23451
    Dataset Repo: https://huggingface.co/datasets/HongbangYuan/OmniRewardBench
    """
    def get_media_info(self, item: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """
        Extract path info for the two audios.
        """
        data_root = item['data_root']
        if not data_root:
            raise ValueError("Missing 'data_root' in item. Cannot resolve audio paths.")

        full_path1 = os.path.join(data_root, "media_data", item['response1_path'])
        full_path2 = os.path.join(data_root, "media_data", item['response2_path'])

        return {'audio1': {'audio_local_path': full_path1}, 'audio2': {'audio_local_path': full_path2}}

    def parse_item(self, item: Dict[str, Any], media_content: Dict[str, Any],
                   config: Dict[str, Any]) -> Tuple[List[Dict], List[Dict], Dict]:

        audio1 = media_content['audio1']
        audio2 = media_content['audio2']

        if not all([audio1, audio2]):
            raise ValueError("Missing visual content for 'audio1' or 'audio2'.")

        # Get generation prompt from data item
        gen_prompt = item["prompt"]

        # Get system prompts from config
        task_instruction_template = config["task_instruction"]
        task_instruction = task_instruction_template.format(prompt=gen_prompt)

        # Build messages
        messages0 = [{
            "role": "system",
            "content": copy.deepcopy(task_instruction)
        }, {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "Please evaluate the following audio based on the given task instruction."
            }, {
                "type": "audio",
                "audio": audio1
            }]
        }]

        messages1 = [{
            "role": "system",
            "content": copy.deepcopy(task_instruction)
        }, {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "Please evaluate the following audio based on the given task instruction."
            }, {
                "type": "audio",
                "audio": audio2
            }]
        }]

        # Get human preference labels based on weighted scores
        pref_label = self._get_label(item["criteria_preference"])

        other = {
            "preference": pref_label,
            "criteria": item["criteria"],
            "criteria_preference": item["criteria_preference"],
            "id": item["id"],
            "prompt": gen_prompt,
            "source": item['source'],
            "audio1_path": item['response1_path'],
            "audio2_path": item['response2_path'],
            "model1": item['model1'],
            "model2": item['model2'],
        }
        return messages0, messages1, other


class OmniRewardBenchT2IGRMHandler(OmniRewardBenchT2IHandler):
    """
    Data Handler for OmniRewardBench text-to-image human preferences benchmark.
    Process for generative reward model training of pair-wise ranking task.

    Paper: https://huggingface.co/papers/2510.23451
    Dataset Repo: https://huggingface.co/datasets/HongbangYuan/OmniRewardBench
    """
    def parse_item(self, item: Dict[str, Any], media_content: Dict[str, Any],
                   config: Dict[str, Any]) -> Tuple[List[Dict], List[Dict], Dict]:

        image1 = media_content['image1']
        image2 = media_content['image2']

        if not all([image1, image2]):
            raise ValueError("Missing visual content for 'image1' or 'image2'.")

        # Get generation prompt from data item
        gen_prompt = item["prompt"]

        # Get system prompts from config
        task_instruction_template = config["task_instruction"]
        task_instruction = task_instruction_template.format(prompt=gen_prompt)
        criteria = item["criteria"]

        # Build messages
        messages = [
            {
                "role": "system",
                "content": task_instruction
            },
            {
                "role": "system",
                "content": f"Please give your evaluation considering the following criteria: {criteria}."
            },
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "**Image 1:**"
                }, {
                    "type": "image",
                    "image": image1,
                    "max_pixels": 1280 * 720
                }]
            },
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "**Image 2:**"
                }, {
                    "type": "image",
                    "image": image2,
                    "max_pixels": 1280 * 720
                }]
            },
        ]

        # Get human preference labels based on weighted scores
        pref_label = self._get_label(item["criteria_preference"])

        other = {
            "preference": pref_label,
            "criteria": item["criteria"],
            "criteria_preference": item["criteria_preference"],
            "id": item["id"],
            "prompt": gen_prompt,
            "source": item['source'],
            "image1_path": item['response1_path'],
            "image2_path": item['response2_path'],
            "model1": item['model1'],
            "model2": item['model2'],
        }
        return messages, other
