import os
import copy
from typing import List, Dict, Any, Tuple, Union
from loguru import logger

from .utils import BaseDataHandler


class RapidataT2VHandler(BaseDataHandler):
    """
    Data Handler for Rapidata text-to-video human preferences dataset.
    Support datasets:
        - Rapidata/text-2-video-human-preferences-pika2.2
        - Rapidata/text-2-image-human-preferences-veo3:
        - Rapidata/text-2-video-human-preferences-wan2.1:
    
    This dataset contains pairs of videos (video1, video2) generated from a prompt.
    It includes weighted scores for Preference, Coherence, and Alignment.
    
    - 'A' means video1 (messages0) is preferred.
    - 'B' means video2 (messages1) is preferred.
    - 'C' means they are equal or tied.

    Dataset Repo: https://huggingface.co/Rapidata/datasets
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
        data_root = item["data_root"]
        if not data_root:
            raise ValueError(f"Missing 'data_root' in item. Cannot resolve video paths.")

        if 'file_name1' not in item or 'file_name2' not in item:
            raise ValueError(f"Item missing 'file_name1' or 'file_name2'.")

        full_path1 = os.path.join(data_root, "videos", item['file_name1'])
        full_path2 = os.path.join(data_root, "videos", item['file_name2'])

        return {'video1': {'video_local_path': full_path1}, 'video2': {'video_local_path': full_path2}}

    def _get_label(self, val1: float, val2: float) -> str:
        """
        Helper to determine preference label based on two scores.
        A > B, B > A, C == C
        """
        if val1 > val2:
            return "A"
        elif val1 < val2:
            return "B"
        else:
            return "C"  # TIE

    def parse_item(self, item: Dict[str, Any], media_content: Dict[str, Any],
                   config: Dict[str, Any]) -> Tuple[List[Dict], List[Dict], Dict]:

        video1 = media_content['video1']
        video2 = media_content['video2']

        if not all([video1, video2]):
            raise ValueError(f"Missing visual content for 'video1' or 'video2'.")

        # Get generation prompt from data item
        video_gen_prompt = item["prompt"]

        # Get system prompts from config
        task_instruction_template = config["task_instruction"]
        task_instruction = task_instruction_template.format(prompt=video_gen_prompt)

        # Get FPS from config
        fps = config["video_fps"]

        # Build messages
        messages0 = [
            {
                "role": "system",
                "content": copy.deepcopy(task_instruction)
            },
            {
                "role": "user",
                "content": [{
                    "type": "video",
                    "video": video1,
                    "fps": fps,
                    "max_pixels": 720 * 480
                }  # 480p limit to reduce memory
                            ]
            }
        ]

        messages1 = [{
            "role": "system",
            "content": copy.deepcopy(task_instruction)
        }, {
            "role": "user",
            "content": [{
                "type": "video",
                "video": video2,
                "fps": fps,
                "max_pixels": 720 * 480
            }]
        }]

        # Get human preference labels based on weighted scores
        pref_label = self._get_label(item["weighted_results1_Preference"], item["weighted_results2_Preference"])
        cohe_label = self._get_label(item["weighted_results1_Coherence"], item["weighted_results2_Coherence"])
        align_label = self._get_label(item['weighted_results1_Alignment'], item['weighted_results2_Alignment'])

        other = {
            "preference": pref_label,
            "coherence": cohe_label,
            "alignment": align_label,
            "source": item['source'],
            "task_type": "t2v",
        }
        return messages0, messages1, other


class RapidataI2VHandler(RapidataT2VHandler):
    """
    Data Handler for Rapidata image-to-video human preferences dataset.
    Support datasets:
        - Rapidata/image-2-video-human-preferences-seedance-1-pro
    
    Dataset Repo: https://huggingface.co/Rapidata/datasets
    """
    def __init__(self):
        super().__init__()

    def get_media_info(self, item: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """
        Extract path info for the two videos.
        """
        data_root = item["data_root"]
        if not data_root:
            raise ValueError(f"Missing 'data_root' in item. Cannot resolve video paths.")

        # Get video paths
        if 'file_name1' not in item or 'file_name2' not in item:
            raise ValueError(f"Item missing 'file_name1' or 'file_name2'.")

        def process_path(fname, root_path):
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
                "max_pixels": 720 * 480
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
                "type": "image",
                "image": copy.deepcopy(init_image),
                "max_pixels": 720 * 480
            }, {
                "type": "video",
                "video": video2,
                "fps": fps,
                "max_pixels": 720 * 480
            }]
        }]

        # Get human preference labels based on weighted scores
        pref_label = self._get_label(item['weighted_results1_Preference'], item['weighted_results2_Preference'])
        cohe_label = self._get_label(item['weighted_results1_Coherence'], item['weighted_results2_Coherence'])
        align_label = self._get_label(item['weighted_results1_Alignment'], item['weighted_results2_Alignment'])

        other = {
            "preference": pref_label,
            "coherence": cohe_label,
            "alignment": align_label,
            "source": item['source'],
            "task_type": "t2v",  # Text-to-Video
        }
        return messages0, messages1, other
