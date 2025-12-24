from __future__ import annotations
from typing import Any, Callable, Tuple

from torch.utils.data import Dataset
from tqdm import tqdm


def _auto_pick_input_key(example: dict[str, Any], user_key: str | None) -> str:
    """
    Automatically select the input key for the example.

    If the user explicitly specified an input_key, it is used;
    otherwise, 'input' is preferred. If unavailable, 'prompt' is used.

    :param example: The data sample
    :type example: dict[str, Any]
    :param user_key: The user-specified input key, if any
    :type user_key: str | None
    :return: The key to be used for fetching the input
    :rtype: str
    """
    if user_key:  # User explicitly specified the input key.
        return user_key
    return "input" if "input" in example else "prompt"


def _fetch_label(example: dict[str, Any], label_key: str | None) -> str:
    """
    Safely extract the label from the data sample using various strategies.

    Search order:
        1. Top-level key == label_key
        2. example['extra_info']['label']
        3. example['reward_model']['ground_truth']

    If no label is found, returns an empty string.

    :param example: The data sample
    :type example: dict[str, Any]
    :param label_key: The user-specified label key, if any
    :type label_key: str | None
    :return: The extracted label, or an empty string if not found
    :rtype: str
    """
    if label_key is None:
        return ""  # In Reinforced Fine-tuning, label can be empty.

    # Top-level field.
    if label_key in example:
        return example[label_key]

    # Nested under 'extra_info'.
    extra = example.get("extra_info", {})
    if isinstance(extra, dict) and "label" in extra:
        return extra["label"]

    # Nested under 'reward_model'.
    rm = example.get("reward_model", {})
    if isinstance(rm, dict) and "ground_truth" in rm:
        return rm["ground_truth"]

    # If not found, return an empty string.
    return ""


def preprocess_data(
    example: dict[str, Any],
    input_template: str | None = None,
    input_key: str | None = None,
    label_key: str | None = None,
    apply_chat_template: Callable | None = None,
) -> Tuple[str, str]:
    """
    Process a single example into a (prompt, label) tuple.

    It supports the following input formats:
        - Plain text: example["input"] or example["prompt"]
        - Chat list: a list of dictionaries with roles and content.
        - Chat string: the string is treated as a user message.

    :param example: The data sample
    :type example: dict[str, Any]
    :param input_template: Template to format the prompt
    :type input_template: str | None
    :param input_key: User-specified key for input extraction
    :type input_key: str | None
    :param label_key: User-specified key for label extraction
    :type label_key: str | None
    :param apply_chat_template: Function to apply a chat template
    :type apply_chat_template: Callable | None
    :return: The processed (prompt, label) tuple
    :rtype: Tuple[str, str]
    """
    # --- Extract prompt ---
    real_input_key = _auto_pick_input_key(example, input_key)
    raw_content = example.get(real_input_key, "")

    # Handle chat mode if apply_chat_template is provided.
    if apply_chat_template:
        # Ensure raw_content is a list of dictionaries.
        if isinstance(raw_content, str):
            raw_content = [{"role": "user", "content": raw_content}]

        # Standardize possible alternate fields such as ('from', 'value').
        map_role = {"human": "user", "gpt": "assistant", "system": "system"}
        chat: list[dict[str, str]] = []
        for m in raw_content:
            role = map_role.get(m.get("from", ""), m.get("role", ""))
            chat.append({
                "role": role,
                "content": m.get("value", m.get("content", "")),
            })

        prompt = apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = raw_content
        if input_template:
            prompt = input_template.format(prompt)

    # --- Extract label ---
    label = _fetch_label(example, label_key)

    return prompt, label


class PromptDataset(Dataset):
    """
    Dataset for PPO (Proximal Policy Optimization) model training.

    This dataset processes and stores prompts and labels by applying templates
    and tokenization as specified by the strategy and tokenizer.

    :param dataset: The raw dataset used for training
    :type dataset: Any
    :param tokenizer: The tokenizer used for processing the prompts
    :type tokenizer: Any
    :param strategy: A strategy object containing configuration in its args
    :type strategy: Any
    :param input_template: Template for formatting input text
    :type input_template: str | None
    """
    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template: str | None = None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        self.input_template = input_template

        # Retrieve configuration arguments.
        input_key = getattr(self.strategy.args, "input_key", None)
        label_key = getattr(self.strategy.args, "label_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        self.labels = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt, label = preprocess_data(data, input_template, input_key, label_key, apply_chat_template)
            self.prompts.append(prompt)
            self.labels.append(label)

    def __len__(self) -> int:
        """
        Retrieve the number of processed examples.

        :return: The number of examples in the dataset
        :rtype: int
        """
        return len(self.prompts)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        """
        Retrieve the prompt and label at the specified index.

        :param idx: The index of the desired sample
        :type idx: int
        :return: A tuple containing the prompt and corresponding label
        :rtype: tuple[str, str]
        """
        return self.prompts[idx], self.labels[idx]

    def collate_fn(self, item_list: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
        """
        Collate a list of samples into separate lists for prompts and labels.

        :param item_list: A list of (prompt, label) tuples
        :type item_list: list[tuple[str, str]]
        :return: Two lists containing prompts and labels, respectively
        :rtype: tuple[list[str], list[str]]
        """
        prompts_list = []
        labels_list = []
        for prompt, label in item_list:
            prompts_list.append(prompt)
            labels_list.append(label)
        return prompts_list, labels_list
