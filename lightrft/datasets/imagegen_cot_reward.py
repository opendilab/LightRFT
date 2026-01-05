import os
import copy
import json
from typing import List, Dict, Any, Tuple, Union
from loguru import logger

from .utils import BaseDataHandler


class ImageGenCoTRewardHandler(BaseDataHandler):
    """
    Data handler for ImageGen-CoT-Reward-5K dataset. For Text-to-Image generation task.
    
    Paper: https://arxiv.org/pdf/2505.03318
    Dataset Repo: https://huggingface.co/datasets/CodeGoat24/ImageGen-CoT-Reward-5K
    """
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

    def get_media_info(self, item: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """
        Extract path info for the two images or videos.
        Supports both images and videos based on file extension or video_fps field.
        """
        data_root = item['data_root']
        if not data_root:
            raise ValueError(f"Missing 'data_root' in item. Cannot resolve media paths.")
        media_paths = item['images']  # Can contain image or video paths
        path0 = media_paths[0] if isinstance(media_paths[0], str) else media_paths[0]
        path1 = media_paths[1] if isinstance(media_paths[1], str) else media_paths[1]
        
        # Check if paths are absolute or relative
        if os.path.isabs(path0):
            media0_full_path = path0
        else:
            media0_full_path = os.path.join(data_root, path0)
        
        if os.path.isabs(path1):
            media1_full_path = path1
        else:
            media1_full_path = os.path.join(data_root, path1)
        
        # Check if it's video based on file extension or video_fps field
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        is_video = (
            item.get('video_fps') is not None or
            any(media0_full_path.lower().endswith(ext) for ext in video_extensions) or
            any(media1_full_path.lower().endswith(ext) for ext in video_extensions)
        )
        
        if is_video:
            return {
                'video0': {
                    'video_local_path': media0_full_path
                },
                'video1': {
                    'video_local_path': media1_full_path
                },
            }
        else:
            return {
                'image0': {
                    'image_local_path': media0_full_path
                },
                'image1': {
                    'image_local_path': media1_full_path
                },
            }

    def parse_item(
        self,
        item: Dict[str, Any],
        media_content: Dict[str, Any],
        config: Dict[str, Any] | None,
    ) -> Tuple[List[Dict], List[Dict], Dict]:
        
        # Check if it's video or image
        is_video = 'video0' in media_content or 'video1' in media_content
        
        if is_video:
            video0 = media_content.get('video0')
            video1 = media_content.get('video1')
            
            if not all([video0, video1]):
                raise ValueError(f"Missing visual content for 'video0' or 'video1'.")
            
            # Get FPS from config or item
            fps = config.get("video_fps") if config else item.get("video_fps", 2.0)
            
            # Get conversations from data item
            conversations = item["conversations"]
            system_prompt = conversations[0]['value']
            response = conversations[-1]['value']
            
            # Build messages for video
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
                    "video": video0 if isinstance(video0, str) else video0.get('video_local_path'),
                    "fps": fps,
                    "max_pixels": 720 * 480
                }]
            }, {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "**Video 2:**"
                }, {
                    "type": "video",
                    "video": video1 if isinstance(video1, str) else video1.get('video_local_path'),
                    "fps": fps,
                    "max_pixels": 720 * 480
                }]
            }]
        else:
            image0 = media_content.get('image0')
            image1 = media_content.get('image1')
            
            if not all([image0, image1]):
                raise ValueError(f"Missing visual content for 'image0' or 'image1'.")
            
            # Get conversations from data item
            conversations = item["conversations"]
            system_prompt = conversations[0]['value']
            response = conversations[-1]['value']
            
            # Build messages for image
            messages = [{
                "role": "system",
                "content": system_prompt
            }, {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "**Image 1:**"
                }, {
                    "type": "image",
                    "image": image0
                }]
            }, {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "**Image 2:**"
                }, {
                    "type": "image",
                    "image": image1
                }]
            }]

        # During evaluation, we do not include the response part in the messages
        is_training = config.get("is_training", True)
        if is_training:
            messages.append({"role": "assistant", "content": response})

        other = {
            "source": item.get('source', 'imagegen-cot-reward-5k'),
            "data_item": item,
            "system_prompt": system_prompt,
            "response": response,
        }
        return messages, other
