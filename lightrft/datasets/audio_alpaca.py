import os
import glob
import pandas as pd
import random
import copy
from typing import List, Dict, Any, Tuple
from loguru import logger
from .utils import BaseDataHandler


class AudioAlpacaHandler(BaseDataHandler):
    """
    Data Handler for Audio Alpaca dataset.

    Dataset Repo: https://huggingface.co/datasets/declare-lab/audio-alpaca
    """
    def load_data(self, path: str) -> List[Dict[str, Any]]:
        # If path is a directory, look for parquet files
        if os.path.isdir(path):
            search_pattern = os.path.join(path, "*.parquet")
            files = glob.glob(search_pattern)
        else:
            files = [path]

        data = []
        for file_path in files:
            try:
                df = pd.read_parquet(file_path)
                # Columns: prompt, chosen, rejected, strategy
                # chosen/rejected are dicts with 'bytes' and 'path'
                for _, row in df.iterrows():
                    item = {
                        "prompt": row["prompt"],
                        "chosen": row["chosen"],
                        "rejected": row["rejected"],
                        "strategy": row["strategy"],
                        "source": "audio-alpaca",
                        "file_path": file_path
                    }
                    data.append(item)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

        logger.info(f"Loaded {len(data)} samples from Audio Alpaca.")
        return data

    def get_media_info(self, item: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Extract audio bytes info for chosen and rejected audios.
        """
        return {
            'chosen_audio': {
                'audio_bytes': item['chosen']['bytes']
            },
            'rejected_audio': {
                'audio_bytes': item['rejected']['bytes']
            }
        }

    def parse_item(self, item: Dict[str, Any], media_content: Dict[str, Any],
                   config: Dict[str, Any]) -> Tuple[List[Dict], List[Dict], Dict]:

        chosen_audio = media_content['chosen_audio']
        rejected_audio = media_content['rejected_audio']

        prompt_text = item["prompt"]

        # Task instruction
        task_instruction = config.get("task_instruction", "{prompt}")
        task_instruction = task_instruction.format(prompt=prompt_text)

        # Randomize preference
        preference = random.choice(["A", "B"])
        if preference == "A":
            audio0, audio1 = chosen_audio, rejected_audio
        else:
            audio0, audio1 = rejected_audio, chosen_audio

        messages0 = [{
            "role": "system",
            "content": copy.deepcopy(task_instruction)
        }, {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio": audio0
                },
            ]
        }]

        messages1 = [{
            "role": "system",
            "content": copy.deepcopy(task_instruction)
        }, {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio": audio1
                },
            ]
        }]

        other = {
            "preference": preference,
            "source": item["source"],
            "prompt": prompt_text,
            "strategy": item["strategy"]
        }

        return messages0, messages1, other
