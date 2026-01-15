import os
import copy
import json
from typing import List, Dict, Any, Tuple
from loguru import logger

from .utils import BaseDataHandler


class RejectionSamplingT2VHandler(BaseDataHandler):
    """
    Data handler for Rejection Sampling text-to-video training data.
    This handler processes video pairs for GRM training, similar to RapidataT2VHandler format.
    
    The data format is similar to imagegen-cot-reward but specifically for videos.
    Each item contains:
    - conversations: [{"from": "human", "value": task_instruction}, {"from": "gpt", "value": response}]
    - images: [video1_path, video2_path]  # Note: uses "images" field name for compatibility
    - video_fps: float
    """
    task_type = "text-to-video"
    
    def load_data(self, path: str) -> List[Dict[str, Any]]:
        """
        Loads data from json file.
        """
        raw_data = []
        with open(path, 'rb') as f:
            raw_data = json.load(f)

        data_root = os.path.dirname(path)
        for item in raw_data:
            item['data_root'] = data_root

        logger.info(f"Loaded {len(raw_data)} samples from {path}")
        return raw_data

    def _resolve_video_path(self, path: str, data_root: str) -> str:
        """
        Resolve video path, handling case-insensitive 'videos'/'Videos' directory.
        Similar to RapidataT2VHandler format.
        """
        # Check if path is absolute or relative
        if os.path.isabs(path):
            full_path = path
        else:
            full_path = os.path.join(data_root, path)
        
        # If file exists, return it directly
        if os.path.exists(full_path):
            return full_path
        
        # Try to handle videos/Videos case sensitivity issue
        # Replace 'videos' with 'Videos' or vice versa in the path
        if '/videos/' in full_path:
            alt_path = full_path.replace('/videos/', '/Videos/')
            if os.path.exists(alt_path):
                return alt_path
        elif '/Videos/' in full_path:
            alt_path = full_path.replace('/Videos/', '/videos/')
            if os.path.exists(alt_path):
                return alt_path
        
        # Also handle case where path ends with '/videos' or '/Videos'
        if full_path.endswith('/videos'):
            alt_path = full_path[:-7] + '/Videos'
            if os.path.exists(alt_path):
                return alt_path
        elif full_path.endswith('/Videos'):
            alt_path = full_path[:-7] + '/videos'
            if os.path.exists(alt_path):
                return alt_path
        
        # If still not found, return original path (will raise error later)
        return full_path

    def get_media_info(self, item: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """
        Extract path info for the two videos.
        Similar to RapidataT2VHandler format.
        Handles case-insensitive 'videos'/'Videos' directory names.
        """
        data_root = item['data_root']
        if not data_root:
            raise ValueError(f"Missing 'data_root' in item. Cannot resolve video paths.")
        
        # Get video paths from "images" field (for compatibility with conversion script)
        media_paths = item.get('images', [])
        if len(media_paths) < 2:
            raise ValueError(f"Item must contain at least 2 video paths in 'images' field.")
        
        path1 = media_paths[0]
        path2 = media_paths[1]
        
        # Resolve paths with case-insensitive handling
        video1_full_path = self._resolve_video_path(path1, data_root)
        video2_full_path = self._resolve_video_path(path2, data_root)

        return {
            'video1': {
                'video_local_path': video1_full_path
            },
            'video2': {
                'video_local_path': video2_full_path
            }
        }

    def parse_item(
        self,
        item: Dict[str, Any],
        media_content: Dict[str, Any],
        config: Dict[str, Any] | None,
    ) -> Tuple[List[Dict], Dict]:
        """
        Parse item into messages format, similar to RapidataT2VHandler but for GRM training.
        Returns messages in the format expected by GRMDataset.
        """
        video1 = media_content.get('video1')
        video2 = media_content.get('video2')
        
        if not all([video1, video2]):
            raise ValueError(f"Missing visual content for 'video1' or 'video2'.")
        
        # Get FPS from config or item
        fps = config.get("video_fps") if config else item.get("video_fps", 2.0)
        
        # Get max_pixels from config (default to 720 * 480 if not provided)
        max_pixels = config.get("max_pixels", 720 * 480) if config else 720 * 480
        
        # Get conversations from data item
        conversations = item["conversations"]
        system_prompt = conversations[0]['value']
        response = conversations[-1]['value']
        
        # Build messages for video, similar to RapidataT2VHandler format
        # But using the format from imagegen_cot_reward for GRM training
        # Note: video1 and video2 are already loaded by load_multimodal_content
        messages = [{
            "role": "system",
            "content": system_prompt
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

        # During evaluation, we do not include the response part in the messages
        is_training = config.get("is_training", True) if config else True
        if is_training:
            messages.append({"role": "assistant", "content": response})

        other = {
            "source": item.get('source', 'rejection-sampling-t2v'),
            "data_item": item,
            "system_prompt": system_prompt,
            "response": response,
            "task_type": self.task_type,
        }
        return messages, other
