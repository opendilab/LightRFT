"""
Math Reasoning Benchmarks Dataset

This module provides dataset classes for mathematical reasoning evaluation benchmarks:
- Math500: 500 challenging math problems from MATH dataset
- AIME 2024/2025: American Invitational Mathematics Examination problems
- GPQA Diamond: Graduate-level STEM questions (multiple choice)

These datasets are designed for evaluating model performance on advanced
mathematical reasoning tasks.
"""

from __future__ import annotations
import json
from typing import Any, Callable, Tuple, List, Dict, Optional
from pathlib import Path

from torch.utils.data import Dataset
from tqdm import tqdm


class MathReasoningDataset(Dataset):
    """
    Base dataset class for mathematical reasoning benchmarks.

    Supports both open-ended math problems (Math500, AIME) and
    multiple-choice questions (GPQA Diamond).

    Data format:
        - prompt: List of message dicts with 'from' and 'value' keys
        - final_answer: The ground truth answer
        - choices (optional): Dictionary mapping choice labels to text (for MCQ)

    :param data_path: Path to the JSON data file
    :type data_path: str
    :param tokenizer: Tokenizer for processing text
    :type tokenizer: Any
    :param strategy: Training strategy object
    :type strategy: Any
    :param benchmark_name: Name of the benchmark (math500, aime2024, aime2025, gpqa_diamond)
    :type benchmark_name: str
    """
    def __init__(
        self,
        data_path: str,
        tokenizer,
        strategy,
        benchmark_name: str = "math500",
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.benchmark_name = benchmark_name

        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        self.prompts = []
        self.labels = []
        self.choices_list = []  # For multiple choice questions
        self.raw_data = []  # Keep raw data for reference

        # Determine if chat template should be applied
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        # Process each example
        for data in tqdm(raw_data, desc=f"Loading {benchmark_name}", disable=not self.strategy.is_rank_0()):
            prompt = self._process_prompt(data.get("prompt", []), apply_chat_template)
            label = data.get("final_answer", "")
            choices = data.get("choices", None)

            self.prompts.append(prompt)
            self.labels.append(label)
            self.choices_list.append(choices)
            self.raw_data.append(data)

    def _process_prompt(self, prompt_messages: List[Dict[str, str]], apply_chat_template: Optional[Callable]) -> str:
        """
        Process prompt messages into a formatted string.

        :param prompt_messages: List of message dictionaries
        :param apply_chat_template: Function to apply chat template
        :return: Formatted prompt string
        """
        if apply_chat_template:
            # Convert to standard chat format
            chat = []
            for msg in prompt_messages:
                role = msg.get("from", "user")
                # Map alternate role names
                role_map = {"human": "user", "gpt": "assistant", "system": "system"}
                role = role_map.get(role, role)

                content = msg.get("value", "")
                chat.append({"role": role, "content": content})

            # Apply chat template
            prompt = apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Simple concatenation for non-chat mode
            prompt_parts = []
            for msg in prompt_messages:
                content = msg.get("value", "")
                prompt_parts.append(content)
            prompt = "\n".join(prompt_parts)

        return prompt

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Tuple[str, str, Optional[Dict[str, str]], Dict[str, Any]]:
        """
        Get a single example from the dataset.

        :param idx: Index of the example
        :return: Tuple of (prompt, label, choices, raw_data)
        """
        return (self.prompts[idx], self.labels[idx], self.choices_list[idx], self.raw_data[idx])

    def collate_fn(
        self, item_list: List[Tuple[str, str, Optional[Dict[str, str]], Dict[str, Any]]]
    ) -> Tuple[List[str], List[str], List[Optional[Dict[str, str]]], List[Dict[str, Any]]]:
        """
        Collate a batch of examples.

        :param item_list: List of examples
        :return: Tuple of lists (prompts, labels, choices, raw_data)
        """
        prompts_list = []
        labels_list = []
        choices_list = []
        raw_data_list = []

        for prompt, label, choices, raw_data in item_list:
            prompts_list.append(prompt)
            labels_list.append(label)
            choices_list.append(choices)
            raw_data_list.append(raw_data)

        return prompts_list, labels_list, choices_list, raw_data_list


def load_math_reasoning_benchmark(
    benchmark_name: str,
    data_root: str,
    tokenizer,
    strategy,
) -> MathReasoningDataset:
    """
    Load a mathematical reasoning benchmark dataset.

    :param benchmark_name: Name of the benchmark (math500, aime2024, aime2025, gpqa_diamond)
    :type benchmark_name: str
    :param data_root: Root directory containing the data files
    :type data_root: str
    :param tokenizer: Tokenizer for processing text
    :type tokenizer: Any
    :param strategy: Training strategy object
    :type strategy: Any
    :return: Loaded dataset
    :rtype: MathReasoningDataset
    :raises ValueError: If benchmark_name is not recognized
    """
    # Map benchmark names to file names
    benchmark_files = {
        "math500": "math500.json",
        "aime2024": "aime2024.json",
        "aime2025": "aime2024.json",  # Using aime2024 as placeholder, update if separate file exists
        "gpqa_diamond": "gpqa_diamond.json",
    }

    if benchmark_name not in benchmark_files:
        raise ValueError(
            f"Unknown benchmark: {benchmark_name}. "
            f"Available benchmarks: {list(benchmark_files.keys())}"
        )

    data_path = Path(data_root) / benchmark_files[benchmark_name]

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}. "
            f"Please make sure the benchmark data is available."
        )

    return MathReasoningDataset(
        data_path=str(data_path),
        tokenizer=tokenizer,
        strategy=strategy,
        benchmark_name=benchmark_name,
    )


__all__ = [
    "MathReasoningDataset",
    "load_math_reasoning_benchmark",
]
