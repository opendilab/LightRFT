from typing import List, Dict, Union
import os
import json
import random
import torch
from PIL import Image
from torch.utils.data import Dataset


class MemeOnlineRLDataset(Dataset):
    """Meme dataset class with lazy loading per item.
    Supports both JSON array format (e.g. eval_data.json) and JSONL format.
    """
    ASSISTANT_ROLES = ('gpt', 'assistant')

    def __init__(
        self,
        annotation_path: str,
        root_dir: str,
        processor,
        shuffle: bool = True,
    ):
        super().__init__()

        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file {annotation_path} does not exist")
        if not os.path.isdir(root_dir):
            raise NotADirectoryError(f"Image root directory {root_dir} is invalid")

        self.root_dir = root_dir
        self.annotation_path = annotation_path
        self.processor = processor

        # Only load the raw data lines without processing
        self._raw_data = self._load_raw_data()
        if shuffle:
            random.shuffle(self._raw_data)

    def _load_raw_data(self) -> List[Union[Dict, str]]:
        """Load annotation file. Supports:
        - JSON array format: entire file is one JSON list (e.g. eval_data.json)
        - JSONL format: one JSON object per line
        """
        with open(self.annotation_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if not content:
            return []
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        with open(self.annotation_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def _process_item(self, raw_item: Union[Dict, str]) -> Dict:
        """Process a single data item on-demand. raw_item is either a dict or JSON string."""
        data = raw_item if isinstance(raw_item, dict) else json.loads(raw_item)
        image_path = os.path.join(self.root_dir, data['image']) \
            if not os.path.isabs(data['image']) else data['image']

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image {image_path} does not exist")

        conversations = data['conversations']
        human_input = next(c['value'] for c in conversations if c['from'] == 'human' and '<image>' in c['value'])
        assistant_output = next(c['value'] for c in conversations if c['from'] in self.ASSISTANT_ROLES)

        prompt = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": ""
                },
                {
                    "type": "text",
                    "text": human_input
                },
            ]
        }]
        prompt = self.processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

        return {
            'id': data['id'],
            'image_path': [image_path],
            'prompt': prompt,
            # 'prompt': human_input + "\nAssistant:",
            'label': assistant_output,
        }

    def __getitem__(self, index) -> Dict:
        """Process and return item only when requested"""
        # # Process the item on-demand (lazy loading)
        raw_item = self._raw_data[index]
        processed_item = self._process_item(raw_item)
        # # text, image, label, reference
        return (
            processed_item['prompt'],
            processed_item['image_path'],
            processed_item['label'],
            processed_item['label'],
        )

    def __len__(self) -> int:
        return len(self._raw_data)

    @staticmethod
    def collate_fn(batch: List[Dict]):
        text_list = [x[0] for x in batch]
        image_list = [x[1] for x in batch]
        reference_list = [x[2] for x in batch]
        label_list = [x[3] for x in batch]
        return text_list, image_list, reference_list, label_list
