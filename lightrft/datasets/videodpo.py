import os
import json
import copy
import random
from typing import List, Dict, Any, Tuple
from loguru import logger

from .utils import BaseDataHandler


class VideoDPOPairHandler(BaseDataHandler):
    """
    Data Handler for VideoDPO dataset in pairwise format.

    This handler is designed to process the VideoDPO dataset. 
    It reads pairwise comparison data where each sample consists of a 
    text prompt and two generated videos (one preferred over the other).

    The handler performs the following tasks:
    1. Loads pairwise metadata from a JSON file (typically 'pair.json').
    2. Resolves the file paths for the 'winner' and 'loser' videos based on the dataset's directory structure.
    3. Constructs the input messages for the model, including the system instruction, the text prompt, and the two videos.
    4. Randomly swaps the order of videos to prevent positional bias and assigns the correct preference label ('A' or 'B').

    Paper: https://arxiv.org/abs/2412.14167
    Dataset Repo: https://videodpo.github.io/
    """
    task_type = "text-to-video"

    def load_data(self, path: str) -> List[Dict[str, Any]]:
        """
        Loads data from pair.json.
        """
        data_root = os.path.dirname(path)
            
        with open(path, 'r') as f:
            pairs = json.load(f)
            
        processed_data = []
        for item in pairs:
            v1_idx = item['video1']
            v2_idx = item['video2']
            
            # Infer paths from indices based on dataset structure
            # Even index -> winvideos, Odd index -> losevideos
            # File index is global_idx // 2
            def get_rel_path(idx):
                is_win = idx % 2 == 0
                folder = 'winvideos' if is_win else 'losevideos'
                filename = f"{idx // 2:06d}.mp4"
                return os.path.join(folder, filename)

            item['video1_rel_path'] = get_rel_path(v1_idx)
            item['video2_rel_path'] = get_rel_path(v2_idx)
            item['data_root'] = data_root
            
            processed_data.append(item)

        logger.info(f"Loaded {len(processed_data)} samples from {path}")
        return processed_data

    def get_media_info(self, item: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """
        Extract path info for the two videos.
        """
        data_root = item["data_root"]
        
        full_path1 = os.path.join(data_root, item['video1_rel_path'])
        full_path2 = os.path.join(data_root, item['video2_rel_path'])

        return {'video1': {'video_local_path': full_path1}, 'video2': {'video_local_path': full_path2}}

    def parse_item(self, item: Dict[str, Any], media_content: Dict[str, Any],
                   config: Dict[str, Any]) -> Tuple[List[Dict], List[Dict], Dict]:

        video1 = media_content['video1']
        video2 = media_content['video2']

        if not all([video1, video2]):
            raise ValueError("Missing visual content for 'video1' or 'video2'.")

        # Get generation prompt from data item
        # pair.json uses "frame_caption" for the prompt
        video_gen_prompt = item["frame_caption"]

        # Get system prompts from config
        task_instruction_template = config["task_instruction"]
        task_instruction = task_instruction_template.format(prompt=video_gen_prompt)

        # Get max_pixels from config
        max_pixels = config["max_pixels"]

        # Get FPS from config
        fps = config["video_fps"]

        # Randomly swap video1 and video2 to avoid positional bias
        # In VideoDPO pair.json, label is always 0, meaning video1 is preferred.
        # If swapped, video1 is at position B, so preference is B.
        # If not swapped, video1 is at position A, so preference is A.
        if random.random() < 0.5:
            first_video = video2
            second_video = video1
            is_swapped = True
        else:
            first_video = video1
            second_video = video2
            is_swapped = False
        
        # # Determine preference label
        preference = "B" if is_swapped else "A"

        # Build messages
        messages = [
        {
            "role": "system", 
            "content": [{"type": "text", "text": task_instruction}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "The following is the first video."},
                {"type": "video", "video": first_video, "fps": fps, "max_pixels": max_pixels},
                
                {"type": "text", "text": "The following is the second video."},
                {"type": "video", "video": second_video, "fps": fps, "max_pixels": max_pixels},
                ]
            }
        ]
        
        other = {
            "preference": preference,
            "prompt": video_gen_prompt,
            "source": "VideoDPO",
            "task_type": self.task_type,
            "video1_path": first_video,
            "video2_path": second_video,
        }
    
        return messages, other
