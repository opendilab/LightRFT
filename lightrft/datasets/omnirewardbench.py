import os
import copy
from typing import List, Dict, Any, Tuple
from loguru import logger

from .utils import BaseDataHandler, get_task_instructions


class OmniRewardBenchT2IHandler(BaseDataHandler):
    """
    Data Handler for OmniRewardBench text-to-image human preferences benchmark.
    Process for scalar reward model training of pairwise-ranking task.

    Paper: https://huggingface.co/papers/2510.23451
    Dataset Repo: https://huggingface.co/datasets/HongbangYuan/OmniRewardBench
    """
    task_type = "text-to-image"

    def load_data(self, path: str) -> List[Dict[str, Any]]:
        """
        Loads data from parquet file.

        :param path: Path to the parquet file
        :type path: str

        :return: List of samples with 'data_root' attached
        :rtype: List[Dict[str, Any]]

        **Example:**

        .. code-block:: python

            handler = OmniRewardBenchT2IHandler()
            data = handler.load_data("path/to/OmniRewardBench/data.parquet")
        """
        raw_data = []
        import pyarrow.parquet as pq
        data_table = pq.read_table(path)
        raw_data = [{
            name: col[i].as_py()
            for name, col in zip(data_table.column_names, data_table.itercolumns())
        }
                    for i in range(data_table.num_rows)]

        data_root = os.path.dirname(os.path.dirname(path))
        for item in raw_data:
            item['data_root'] = data_root

        logger.info(f"Loaded {len(raw_data)} samples from {path}")
        return raw_data

    def get_media_info(self, item: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """
        Extract media info (paths) for the two images.

        :param item: A data item from load_data
        :type item: Dict[str, Any]

        :return: Dict containing local paths for 'image1' and 'image2'
        :rtype: Dict[str, Dict[str, str]]

        **Example:**

        .. code-block:: python

            info = handler.get_media_info(item)
        """
        data_root = item['data_root']
        if not data_root:
            raise ValueError("Missing 'data_root' in item. Cannot resolve video paths.")

        full_path1 = os.path.join(data_root, "media_data", item['response1_path'])
        full_path2 = os.path.join(data_root, "media_data", item['response2_path'])

        return {'image1': {'image_local_path': full_path1}, 'image2': {'image_local_path': full_path2}}

    def _get_label(self, choice: str) -> str:
        """
        Helper to determine preference label.
        """
        if choice == "response1":
            return "A"
        elif choice == "response2":
            return "B"
        else:
            return "C"  # TIE

    def parse_item(self, item: Dict[str, Any], media_content: Dict[str, Any],
                   config: Dict[str, Any]) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Parse a data item from OmniRewardBench-T2I into messages and metadata.

        :param item: The raw data item
        :type item: Dict[str, Any]
        :param media_content: Loaded media content with 'image1' and 'image2' keys.
        :type media_content: Dict[str, Any]
        :param config: Configuration for task instructions and max_pixels
        :type config: Dict[str, Any]

        :return: A tuple of (messages0, messages1, metadata)
        :rtype: Tuple[List[Dict], List[Dict], Dict]

        **Example:**

        .. code-block:: python

            msg0, msg1, other = handler.parse_item(item, media_content, config)
        """
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

        # Get max_pixels from config
        max_pixels = config["max_pixels"]

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
                    "max_pixels": max_pixels
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
                    "max_pixels": max_pixels
                }]
            }
        ]

        # Get human preference labels based on weighted scores
        pref_label = self._get_label(item["criteria_preference"])

        other = {
            "preference": pref_label,
            "task_type": self.task_type,
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
    task_type = "text-to-video"

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
        data_root = item['data_root']
        if not data_root:
            raise ValueError("Missing 'data_root' in item. Cannot resolve video paths.")

        full_path1 = os.path.join(data_root, "media_data", item['response1'])
        full_path2 = os.path.join(data_root, "media_data", item['response2'])

        return {'video1': {'video_local_path': full_path1}, 'video2': {'video_local_path': full_path2}}

    def parse_item(self, item: Dict[str, Any], media_content: Dict[str, Any],
                   config: Dict[str, Any]) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Parse a data item from OmniRewardBench-T2V into messages and metadata.

        :param item: The raw data item
        :type item: Dict[str, Any]
        :param media_content: Loaded visual content
        :type media_content: Dict[str, Any]
        :param config: Configuration for task instructions, max_pixels, and fps
        :type config: Dict[str, Any]

        :return: A tuple of (messages0, messages1, metadata)
        :rtype: Tuple[List[Dict], List[Dict], Dict]

        **Example:**

        .. code-block:: python

            msg0, msg1, other = handler.parse_item(item, media_content, config)
        """
        video1 = media_content['video1']
        video2 = media_content['video2']

        if not all([video1, video2]):
            raise ValueError("Missing visual content for 'video1' or 'video2'.")

        # Get generation prompt from data item
        gen_prompt = item["prompt"]

        # Get system prompts from config
        task_instruction_template = config["task_instruction"]
        task_instruction = task_instruction_template.format(prompt=gen_prompt)

        # Get max_pixels from config
        max_pixels = config["max_pixels"]

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
                "max_pixels": max_pixels
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
                "max_pixels": max_pixels
            }]
        }]

        # Get human preference labels based on weighted scores
        pref_label = self._get_label(item["criteria_preference"])

        other = {
            "preference": pref_label,
            "task_type": self.task_type,
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
    task_type = "text-to-audio"

    def get_media_info(self, item: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """
        Extract media info (paths) for the two audios.

        :param item: A data item from load_data
        :type item: Dict[str, Any]

        :return: Dict containing local paths for 'audio1' and 'audio2'
        :rtype: Dict[str, Dict[str, str]]

        **Example:**

        .. code-block:: python

            info = handler.get_media_info(item)
        """
        data_root = item['data_root']
        if not data_root:
            raise ValueError("Missing 'data_root' in item. Cannot resolve audio paths.")

        full_path1 = os.path.join(data_root, "media_data", item['response1_path'])
        full_path2 = os.path.join(data_root, "media_data", item['response2_path'])

        return {'audio1': {'audio_local_path': full_path1}, 'audio2': {'audio_local_path': full_path2}}

    def parse_item(self, item: Dict[str, Any], media_content: Dict[str, Any],
                   config: Dict[str, Any]) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Parse a data item from OmniRewardBench-T2A into messages and metadata.

        :param item: The raw data item
        :type item: Dict[str, Any]
        :param media_content: Loaded visual content
        :type media_content: Dict[str, Any]
        :param config: Configuration for task instructions
        :type config: Dict[str, Any]

        :return: A tuple of (messages0, messages1, metadata)
        :rtype: Tuple[List[Dict], List[Dict], Dict]

        **Example:**

        .. code-block:: python

            msg0, msg1, other = handler.parse_item(item, media_content, config)
        """
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
            "task_type": self.task_type,
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
        """
        Parse a data item from OmniRewardBench-T2I into one message and metadata.
        For generative reward model training in pair-wise ranking task.

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
        image2 = media_content['image2']

        if not all([image1, image2]):
            raise ValueError("Missing visual content for 'image1' or 'image2'.")

        # Get generation prompt from data item
        gen_prompt = item["prompt"]

        # Get system prompts from config
        task_instruction_template = config["task_instruction"]
        task_instruction = task_instruction_template.format(prompt=gen_prompt)
        criteria = item["criteria"]

        # Get max_pixels from config
        max_pixels = config["max_pixels"]

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
                    "max_pixels": max_pixels
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
                    "max_pixels": max_pixels
                }]
            },
        ]

        # Get human preference labels based on weighted scores
        pref_label = self._get_label(item["criteria_preference"])

        other = {
            "preference": pref_label,
            "task_type": self.task_type,
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


class OmniRewardBenchT2IPairHandler(OmniRewardBenchT2IHandler):
    """
    Data Handler for OmniRewardBench text-to-image human preferences benchmark.
    Process for generative reward model on pair-wise ranking task.

    Paper: https://huggingface.co/papers/2510.23451
    Dataset Repo: https://huggingface.co/datasets/HongbangYuan/OmniRewardBench
    """
    def parse_item(self, item: Dict[str, Any], media_content: Dict[str, Any],
                   config: Dict[str, Any]) -> Tuple[List[Dict], Dict]:
        """
        Parse a data item into generative messages and metadata.

        :param item: The raw data item
        :type item: Dict[str, Any]
        :param media_content: Loaded visual content
        :type media_content: Dict[str, Any]
        :param config: Configuration for task instructions
        :type config: Dict[str, Any]

        :return: A tuple of (messages, metadata)
        :rtype: Tuple[List[Dict], Dict]

        **Example:**

        .. code-block:: python

            messages, other = handler.parse_item(item, media_content, config)
        """
        image1 = media_content['image1']
        image2 = media_content['image2']

        if not all([image1, image2]):
            raise ValueError("Missing visual content for 'image1' or 'image2'.")

        # Get generation prompt from data item
        prompt_text = item["prompt"]

        # Get system prompts from config
        task_instruction_template = get_task_instructions(self, config)
        task_instruction = task_instruction_template.format(prompt=prompt_text)
        criteria = item["criteria"]

        # Get max_pixels from config
        max_pixels = config["max_pixels"]

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
                    "max_pixels": max_pixels,
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
                    "max_pixels": max_pixels,
                }]
            },
        ]

        # Get human preference labels based on weighted scores
        pref_label = self._get_label(item["criteria_preference"])

        other = {
            "preference": pref_label,
            "reward_rule_label": "general",
            "task_type": self.task_type,
            "criteria": item["criteria"],
            "criteria_preference": item["criteria_preference"],
            "id": item["id"],
            "prompt": prompt_text,
            "source": item['source'],
            "image1_path": item['response1_path'],
            "image2_path": item['response2_path'],
            "model1": item['model1'],
            "model2": item['model2'],
        }
        return messages, other


class OmniRewardBenchT2VPairHandler(OmniRewardBenchT2VHandler):
    """
    Data Handler for OmniRewardBench text-to-video human preferences benchmark.
    Process for generative reward model on pair-wise ranking task.

    Paper: https://huggingface.co/papers/2510.23451
    Dataset Repo: https://huggingface.co/datasets/HongbangYuan/OmniRewardBench
    """
    def parse_item(self, item: Dict[str, Any], media_content: Dict[str, Any],
                   config: Dict[str, Any]) -> Tuple[List[Dict], Dict]:
        """
        Parse a data item into generative messages and metadata.

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
            raise ValueError("Missing visual content for 'video1' or 'video2'.")

        # Get generation prompt from data item
        gen_prompt = item["prompt"]

        # Get system prompts from config
        task_instruction_template = get_task_instructions(self, config)
        task_instruction = task_instruction_template.format(prompt=gen_prompt)

        # Get FPS and max_pixels from config
        fps = config["video_fps"]
        max_pixels = config["max_pixels"]

        # Build messages
        messages = [{
            "role": "system",
            "content": copy.deepcopy(task_instruction)
        }, {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "**Video 1:**"
            }, {
                "type": "video",
                "video": video1,
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
                "video": video2,
                "fps": fps,
                "max_pixels": max_pixels
            }]
        }]

        # Get human preference labels based on weighted scores
        pref_label = self._get_label(item["criteria_preference"])

        other = {
            "preference": pref_label,
            "reward_rule_label": "general",
            "task_type": self.task_type,
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
        return messages, other
