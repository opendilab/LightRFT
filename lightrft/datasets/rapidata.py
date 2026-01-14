import os
import copy
from typing import List, Dict, Any, Tuple
from loguru import logger

from .utils import BaseDataHandler, get_task_instructions


class RapidataT2VHandler(BaseDataHandler):
    """
    Data Handler for Rapidata text-to-video human preferences dataset.

    Support datasets:

    - Rapidata/text-2-video-human-preferences-pika2.2
    - Rapidata/text-2-image-human-preferences-veo3
    - Rapidata/text-2-video-human-preferences-wan2.1

    This dataset contains pairs of videos (video1, video2) generated from a prompt.
    It includes weighted scores for Preference, Coherence, and Alignment.

    - 'A' means video1 (messages0) is preferred.
    - 'B' means video2 (messages1) is preferred.
    - 'C' means they are equal or tied.

    Dataset Repo: https://huggingface.co/Rapidata/datasets
    """
    task_type = "text-to-video"

    def load_data(self, path: str) -> List[Dict[str, Any]]:
        """
        Loads data from parquet file.

        :param path: Path to the parquet file
        :type path: str

        :return: List of samples with 'data_root' attached
        :rtype: List[Dict[str, Any]]

        **Example:**

        .. code-block:: python

            handler = RapidataT2VHandler()
            data = handler.load_data("path/to/Rapidata/data.parquet")
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
        Extract path info for the two videos.
        """
        data_root = item["data_root"]
        if not data_root:
            raise ValueError("Missing 'data_root' in item. Cannot resolve video paths.")

        if 'file_name1' not in item or 'file_name2' not in item:
            raise ValueError("Item missing 'file_name1' or 'file_name2'.")

        # Try both "videos" and "Videos"
        video_dir = "videos"
        if not os.path.exists(os.path.join(data_root, video_dir)):
            if os.path.exists(os.path.join(data_root, "Videos")):
                video_dir = "Videos"

        full_path1 = os.path.join(data_root, video_dir, item['file_name1'])
        full_path2 = os.path.join(data_root, video_dir, item['file_name2'])

        return {'video1': {'video_local_path': full_path1}, 'video2': {'video_local_path': full_path2}}

    def _get_label(self, val1: float, val2: float) -> str:
        """
        Helper to determine preference label based on two scores.
        """
        if val1 is None or val2 is None:
            return "C"
        if val1 > val2:
            return "A"
        elif val1 < val2:
            return "B"
        else:
            return "C"  # TIE

    def parse_item(self, item: Dict[str, Any], media_content: Dict[str, Any],
                   config: Dict[str, Any]) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Parse a single Rapidata-T2V data item into message pairs for ranking.

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
        video_gen_prompt = item["prompt"]

        # Get system prompts from config
        task_instruction_template = config["task_instruction"]
        task_instruction = task_instruction_template.format(prompt=video_gen_prompt)

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
                "type": "video",
                "video": video2,
                "fps": fps,
                "max_pixels": max_pixels
            }]
        }]

        # Get human preference labels and total scores based on weighted metrics
        metrics = ['Preference', 'Coherence', 'Alignment']
        labels = {
            f"{m.lower()}_label": self._get_label(
                item.get(f'weighted_results1_{m}'), item.get(f'weighted_results2_{m}')
            )
            for m in metrics
        }

        score1 = sum(item.get(f'weighted_results1_{m}') or 0.0 for m in metrics)
        score2 = sum(item.get(f'weighted_results2_{m}') or 0.0 for m in metrics)

        other = {
            "preference": self._get_label(score1, score2),
            **labels,
            "source": item['source'],
            "task_type": self.task_type,
        }
        return messages0, messages1, other


class RapidataI2VHandler(RapidataT2VHandler):
    """
    Data Handler for Rapidata image-to-video human preferences dataset.

    Support datasets:

    - Rapidata/image-2-video-human-preferences-seedance-1-pro

    Dataset Repo: https://huggingface.co/Rapidata/datasets
    """
    task_type = "image-to-video"

    def __init__(self):
        super().__init__()

    def get_media_info(self, item: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """
        Extract media info (paths) for the two videos and bytes for initial image.

        :param item: A data item from load_data
        :type item: Dict[str, Any]

        :return: Dict containing local paths for videos and bytes for 'init_image'
        :rtype: Dict[str, Dict[str, Any]]

        **Example:**

        .. code-block:: python

            info = handler.get_media_info(item)
        """
        data_root = item["data_root"]
        if not data_root:
            raise ValueError("Missing 'data_root' in item. Cannot resolve video paths.")

        # Get video paths
        if 'file_name1' not in item or 'file_name2' not in item:
            raise ValueError("Item missing 'file_name1' or 'file_name2'.")

        def process_path(fname, root_path):
            """
            Process filename to construct local path from URL or relative path.

            :param fname: Filename or URL.
            :type fname: str
            :param root_path: Root directory path.
            :type root_path: str
            :return: Local file path.
            :rtype: str
            """
            if fname.startswith("https"):
                fname = fname.split("/")
                model_name = fname[-2]
                video_name = fname[-1]
                return os.path.join(root_path, "videos", model_name, video_name)
            elif "hailuo" in fname:
                return os.path.join(root_path, "videos", "hailuo-02", fname)
            else:
                return os.path.join(root_path, "videos", "hailuo-02", "marey", fname)

        full_path1 = process_path(item['file_name1'], data_root)
        full_path2 = process_path(item['file_name2'], data_root)

        # Get initial image bytes
        assert 'prompt_asset' in item, "Missing initial image in item."
        img_bytes = item['prompt_asset']['bytes']

        return {
            'video1': {
                'video_local_path': full_path1
            },
            'video2': {
                'video_local_path': full_path2
            },
            'init_image': {
                'image_bytes': img_bytes
            }
        }

    def parse_item(self, item: Dict[str, Any], media_content: Dict[str, Any],
                   config: Dict[str, Any]) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Parse a single Rapidata-I2V data item into message pairs for ranking.

        :param item: The raw data item
        :type item: Dict[str, Any]
        :param media_content: Loaded visual content (videos and images)
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
        init_image = media_content['init_image']

        if not all([video1, video2, init_image]):
            raise ValueError("Missing visual content for 'video1' or 'video2' or 'init_image'.")

        # Get generation prompt from data item
        prompt_text = item["prompt"]

        # Get system prompts from config
        task_instruction_template = config["task_instruction"]
        task_instruction = task_instruction_template.format(prompt=prompt_text)

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
                "type": "image",
                "image": copy.deepcopy(init_image),
                "max_pixels": max_pixels
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
                "type": "image",
                "image": copy.deepcopy(init_image),
                "max_pixels": max_pixels
            }, {
                "type": "video",
                "video": video2,
                "fps": fps,
                "max_pixels": max_pixels
            }]
        }]

        # Get human preference labels and total scores based on weighted metrics
        metrics = ['Preference', 'Coherence', 'Alignment']
        labels = {
            f"{m.lower()}_label": self._get_label(
                item.get(f'weighted_results1_{m}'), item.get(f'weighted_results2_{m}')
            )
            for m in metrics
        }

        score1 = sum(item.get(f'weighted_results1_{m}') or 0.0 for m in metrics)
        score2 = sum(item.get(f'weighted_results2_{m}') or 0.0 for m in metrics)

        other = {
            "preference": self._get_label(score1, score2),
            **labels,
            "source": item['source'],
            "task_type": self.task_type,
        }
        return messages0, messages1, other


class RapidataT2VPairHandler(RapidataT2VHandler):
    """
    Data Handler for Rapidata text-to-video human preferences dataset in pairwise format.
    """
    def __init__(self):
        super().__init__()

    def parse_item(self, item: Dict[str, Any], media_content: Dict[str, Any],
                   config: Dict[str, Any]) -> Tuple[List[Dict], Dict]:
        """
        Parse a text-to-video data item into pairwise messages and metadata.

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
        video_gen_prompt = item["prompt"]

        # Get system prompts from config
        task_instruction_template = get_task_instructions(self, config)
        task_instruction = task_instruction_template.format(prompt=video_gen_prompt)

        # Get max_pixels from config
        max_pixels = config["max_pixels"]

        # Get FPS from config
        fps = config["video_fps"]

        # Build messages
        messages = [{
            "role": "system",
            "content": [{
                "type": "text",
                "text": task_instruction
            }]
        }, {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "The following is the first video."
                },
                {
                    "type": "video",
                    "video": video1,
                    "fps": fps,
                    "max_pixels": max_pixels
                },
                {
                    "type": "text",
                    "text": "The following is the second video."
                },
                {
                    "type": "video",
                    "video": video2,
                    "fps": fps,
                    "max_pixels": max_pixels
                },
            ]
        }]

        # Get human preference labels and total scores based on weighted metrics
        metrics = ['Preference', 'Coherence', 'Alignment']
        labels = {
            f"{m.lower()}_label": self._get_label(
                item.get(f'weighted_results1_{m}'), item.get(f'weighted_results2_{m}')
            )
            for m in metrics
        }

        score1 = sum(item.get(f'weighted_results1_{m}') or 0.0 for m in metrics)
        score2 = sum(item.get(f'weighted_results2_{m}') or 0.0 for m in metrics)

        other = {
            "preference": self._get_label(score1, score2),
            **labels,
            "source": item['source'],
            "task_type": self.task_type,
            "reward_rule_label": "general",
        }
        return messages, other


class RapidataI2VPairHandler(RapidataI2VHandler):
    """
    Data Handler for Rapidata image-to-video human preferences dataset in pairwise format.
    """
    task_type = "image-to-video"

    def __init__(self):
        super().__init__()

    def parse_item(self, item: Dict[str, Any], media_content: Dict[str, Any],
                   config: Dict[str, Any]) -> Tuple[List[Dict], Dict]:
        """
        Parse an image-to-video data item into pairwise messages and metadata.

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
        init_image = media_content['init_image']

        if not all([video1, video2, init_image]):
            raise ValueError("Missing visual content for 'video1' or 'video2' or 'init_image'.")

        # Get generation prompt from data item
        prompt_text = item["prompt"]

        # Get system prompts from config
        task_instruction_template = get_task_instructions(self, config)
        task_instruction = task_instruction_template.format(prompt=prompt_text)

        # Get FPS from config
        fps = config["video_fps"]
        max_pixels = config["max_pixels"]

        # Build messages
        messages = [{
            "role": "system",
            "content": [{
                "type": "text",
                "text": task_instruction
            }]
        }, {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Reference Image:"
                },
                {
                    "type": "image",
                    "image": init_image,
                    "max_pixels": max_pixels
                },
                {
                    "type": "text",
                    "text": "The following is the first video."
                },
                {
                    "type": "video",
                    "video": video1,
                    "fps": fps,
                    "max_pixels": max_pixels
                },
                {
                    "type": "text",
                    "text": "The following is the second video."
                },
                {
                    "type": "video",
                    "video": video2,
                    "fps": fps,
                    "max_pixels": max_pixels
                },
            ]
        }]

        # Get human preference labels and total scores based on weighted metrics
        metrics = ['Preference', 'Coherence', 'Alignment']
        labels = {
            f"{m.lower()}_label": self._get_label(
                item.get(f'weighted_results1_{m}'), item.get(f'weighted_results2_{m}')
            )
            for m in metrics
        }

        score1 = sum(item.get(f'weighted_results1_{m}') or 0.0 for m in metrics)
        score2 = sum(item.get(f'weighted_results2_{m}') or 0.0 for m in metrics)

        other = {
            "preference": self._get_label(score1, score2),
            **labels,
            "source": item['source'],
            "task_type": self.task_type,
            "reward_rule_label": "general",
        }
        return messages, other
