import json
import os
import random
import re
from typing import Any, Dict, List, Tuple, Union

from torch.utils.data import Dataset

from meme_utils import extract_box_texts, resolve_expected_box_count


class MemeOnlineRLDataset(Dataset):
    """Meme dataset class with lazy loading per item."""

    ASSISTANT_ROLES = ("gpt", "assistant")
    DEFAULT_LABEL = "meme_pairwise"

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

        self._raw_data = self._load_raw_data()
        if shuffle:
            random.shuffle(self._raw_data)

    def _load_raw_data(self) -> List[Union[Dict[str, Any], str]]:
        with open(self.annotation_path, "r", encoding="utf-8") as handle:
            content = handle.read().strip()
        if not content:
            return []
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        with open(self.annotation_path, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    def _resolve_image_path(self, data: Dict[str, Any]) -> str:
        image_value = data.get("image") or data.get("image_path") or data.get("img")
        if not image_value:
            raise KeyError("Dataset row is missing `image`")
        image_path = image_value if os.path.isabs(image_value) else os.path.join(self.root_dir, image_value)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image {image_path} does not exist")
        return image_path

    def _extract_user_request_from_prompt(self, prompt_text: str) -> str:
        patterns = [
            r"\*\*User Input Parameters\*\*:\s*(.*?)(?=\n\s*\*\*Text on the Meme\*\*:|\Z)",
            r"Input Parameters\s*:\s*(\[[^\n]+\])",
        ]
        for pattern in patterns:
            match = re.search(pattern, prompt_text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return ""

    def _build_reference(self, data: Dict[str, Any], prompt_text: str, assistant_output: str) -> Dict[str, Any]:
        reference: Dict[str, Any] = {
            "id": data.get("id"),
            "group_id": str(data.get("group_id") or data.get("sample_id") or data.get("id") or ""),
            "user_request": self._extract_user_request_from_prompt(prompt_text),
            "reference_output": assistant_output,
        }

        for key in ("detections", "text_loc_info", "loc", "bbox_scale", "bbox_normalized", "expected_box_count"):
            if key in data:
                reference[key] = data[key]

        expected_box_count = resolve_expected_box_count(reference)
        if expected_box_count is None:
            box_texts = extract_box_texts(assistant_output)
            if box_texts:
                expected_box_count = len(box_texts)
        if expected_box_count is not None:
            reference["expected_box_count"] = expected_box_count

        return reference

    def _process_item(self, raw_item: Union[Dict[str, Any], str]) -> Tuple[str, List[str], Dict[str, Any], str]:
        data = raw_item if isinstance(raw_item, dict) else json.loads(raw_item)
        image_path = self._resolve_image_path(data)

        conversations = data["conversations"]
        human_input = next(c["value"] for c in conversations if c["from"] == "human" and "<image>" in c["value"])
        assistant_output = next(c["value"] for c in conversations if c["from"] in self.ASSISTANT_ROLES)

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
            ],
        }]
        prompt = self.processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        reference = self._build_reference(data, human_input, assistant_output)
        label = data.get("reward_rule_label", self.DEFAULT_LABEL)

        return prompt, [image_path], reference, label

    def __getitem__(self, index: int) -> Tuple[str, List[str], Dict[str, Any], str]:
        return self._process_item(self._raw_data[index])

    def __len__(self) -> int:
        return len(self._raw_data)

    @staticmethod
    def collate_fn(batch: List[Tuple[str, List[str], Dict[str, Any], str]]):
        text_list = [item[0] for item in batch]
        image_list = [item[1] for item in batch]
        reference_list = [item[2] for item in batch]
        label_list = [item[3] for item in batch]
        return text_list, image_list, reference_list, label_list
