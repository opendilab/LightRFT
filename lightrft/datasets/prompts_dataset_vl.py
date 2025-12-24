from __future__ import annotations

import ast
from typing import Any, Dict, List, Tuple, Union

from torch.utils.data import Dataset


# -------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------
def _strip_image_tokens(text: str) -> str:
    """
    Removes image placeholders like `<image>`, `<image_1>`, etc., from a string.

    :param text: The input string, which may contain image tokens
    :type text: str
    :return: The text with image tokens removed. Returns the original input if it's not a string
    :rtype: str
    """
    if not isinstance(text, str):
        return text
    return text.replace("<image>", "")


def _extract_user_text(conversation: Union[str, List, Dict]) -> str:
    """
    Extracts the user's utterance from a conversation object.

    This function supports various common conversation formats.

    :param conversation: The conversation object. Supported formats include:
        1) A raw string containing the user's message.
        2) A list/tuple of message dictionaries, from which the user's message is found.
        3) An OpenAI-style dictionary: `{"role": "user", "content": "..."}`.
        4) A HuggingFace MMChat-style dictionary with multimodal content.
    :type conversation: Union[str, List, Dict]
    :return: The extracted and cleaned user text
    :rtype: str
    :raises ValueError: If the user message cannot be found or the format is unsupported
    :raises TypeError: If the conversation object has an unsupported type
    """
    # Case 1: Raw string
    if isinstance(conversation, str):
        return _strip_image_tokens(conversation)

    # Case 2: List or tuple of messages
    if isinstance(conversation, (list, tuple)):
        for msg in conversation:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return _extract_user_text(msg)
        raise ValueError("Cannot find user message in the conversation list.")

    # Case 3: Dictionary
    if isinstance(conversation, dict):
        # Legacy format with "value" key
        if "value" in conversation:
            return _strip_image_tokens(conversation["value"])

        if "content" in conversation:
            content = conversation["content"]

            # OpenAI / Llama style with string content
            if isinstance(content, str):
                return _strip_image_tokens(content)

            # HuggingFace MMChat style with a list of content segments
            if isinstance(content, (list, tuple)):
                texts = [seg.get("text", "") for seg in content if seg.get("type") == "text"]
                return _strip_image_tokens(" ".join(texts))

        raise ValueError(f"Unsupported conversation dict format: {conversation}")

    raise TypeError(f"Unsupported conversation type: {type(conversation)}")


# -------------------------------------------------------------
# Core Pre-processing
# -------------------------------------------------------------
def _normalize_reference(val: Any) -> Any:
    """
    Standardizes various reference/ground_truth formats into a clean annotation.

    This function processes the input value to produce a consistent format by:
    - Extracting the value from a dictionary (e.g., `{"value": ...}`).
    - Unwrapping single-element lists or tuples.
    - Parsing string-literal representations of lists (e.g., "['9']") and unwrapping them.

    :param val: The raw reference value
    :type val: Any
    :return: The normalized reference value
    :rtype: Any
    """
    if val is None:
        return None

    # If it's a dict, extract the value from common keys.
    if isinstance(val, dict):
        val = val.get("value", val.get("ground_truth", val))

    # If it's a string that looks like a list/tuple, parse it.
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                parsed = ast.literal_eval(s)
                # If the parsed list/tuple has only one element, unwrap it.
                if isinstance(parsed, (list, tuple)) and len(parsed) == 1:
                    return parsed[0]
                return parsed
            except (ValueError, SyntaxError):
                # If parsing fails, return the original string.
                pass
        return s

    # If it's a list/tuple with only one element, unwrap it.
    if isinstance(val, (list, tuple)) and len(val) == 1:
        return val[0]

    return val


def _extract_label(data: Dict[str, Any], label_key: str | None) -> Any:
    """
    Extracts a label from the data dictionary using a fallback mechanism.

    It searches for the label in the following order:
    1. Top-level `label_key` in the `data` dictionary.
    2. The `label_key` within the `extra_info` dictionary, if it exists.

    :param data: The data record
    :type data: Dict[str, Any]
    :param label_key: The key for the label to be extracted
    :type label_key: str | None
    :return: The extracted label, or None if not found
    :rtype: Any
    """
    if label_key is None:
        return None

    # 1. Check top-level keys
    if label_key in data and data[label_key] is not None:
        return data[label_key]

    # 2. Check within `extra_info`
    extra = data.get("extra_info")
    if isinstance(extra, dict):
        return extra.get(label_key)

    return None


# -------------------------------------------------------------
# Global cache for chat template style
# -------------------------------------------------------------
_CHAT_TPL_STYLE: dict[str, str | None] = {"style": None}


def _render_chat(
    apply_chat_template,
    prompt_struct_segment: list,
    prompt_struct_string: list,
) -> str:
    """
    Renders a chat template by automatically detecting the required input style.

    Some tokenizers expect a list of message segments (HuggingFace MMChat standard),
    while others expect a list of messages with string content. This function
    tries the segment-based style first and, upon a specific `TypeError`, falls
    back to the string-based style. The successful style is cached globally to
    avoid repeated trial-and-error for subsequent calls.

    :param apply_chat_template: The tokenizer's `apply_chat_template` method
    :type apply_chat_template: Callable
    :param prompt_struct_segment: The prompt structured as a list of message
        segments (for MMChat-style templates)
    :type prompt_struct_segment: list
    :param prompt_struct_string: The prompt structured with simple string
        content (for Llama/Qwen-style templates)
    :type prompt_struct_string: list
    :return: The rendered prompt string
    :rtype: str
    """
    # 1. If style is already detected, use the cached style.
    if _CHAT_TPL_STYLE["style"] is not None:
        if _CHAT_TPL_STYLE["style"] == "segment":
            return apply_chat_template(prompt_struct_segment, tokenize=False, add_generation_prompt=True)
        else:  # style == "string"
            return apply_chat_template(prompt_struct_string, tokenize=False, add_generation_prompt=True)

    # 2. First attempt: Try the segment-list style.
    try:
        rendered = apply_chat_template(prompt_struct_segment, tokenize=False, add_generation_prompt=True)
        _CHAT_TPL_STYLE["style"] = "segment"  # Cache success
        return rendered
    except TypeError as e:
        # Fallback only on the typical error message, raise others.
        if "list" not in str(e):
            raise

    # 3. Fallback: Use the string-content style.
    rendered = apply_chat_template(prompt_struct_string, tokenize=False, add_generation_prompt=True)
    _CHAT_TPL_STYLE["style"] = "string"  # Cache success
    return rendered


def preprocess_data(
    data: Dict[str, Any],
    input_template: str | None = None,
    prompt_key: str | None = None,
    images_key: str = "images",
    reference_key: str | None = None,
    label_key: str | None = None,
    apply_chat_template=None,
    processor=None,
    system_prompt: str | None = None,
) -> Tuple[Any, Any, Any, Any]:
    """
    Extracts and formats prompt, images, reference, and label from a data record.

    This function serves as the core pre-processing logic for preparing a single
    data point for a vision-language model.

    :param data: A single data record as a dictionary
    :type data: Dict[str, Any]
    :param input_template: A template to format the user's prompt
    :type input_template: str | None
    :param prompt_key: The key to access the prompt/conversation
    :type prompt_key: str | None
    :param images_key: The key to access images
    :type images_key: str
    :param reference_key: The primary key for the reference/answer
    :type reference_key: str | None
    :param label_key: The key for the label
    :type label_key: str | None
    :param apply_chat_template: The tokenizer's `apply_chat_template` method
    :type apply_chat_template: Callable, optional
    :param processor: The model's processor (used for chat template)
    :type processor: Any, optional
    :param system_prompt: An optional system prompt to prepend
    :type system_prompt: str | None
    :return: A tuple containing the processed (prompt, images, reference, label)
    :rtype: Tuple[Any, Any, Any, Any]
    """

    # ---------- 1. Process Prompt ----------
    if apply_chat_template:
        system_msgs = []
        if system_prompt:
            system_msgs.append({"role": "system", "content": system_prompt})

        conversation = data.get(prompt_key)
        user_text = _extract_user_text(conversation)

        if input_template:
            user_text = input_template.format(user_text)

        # a) Segment-based structure (for HuggingFace MMChat)
        user_content_seg = []
        if data.get(images_key):
            user_content_seg.append({"type": "image", "image": ""})
        user_content_seg.append({"type": "text", "text": user_text})
        prompt_struct_seg = system_msgs + [{"role": "user", "content": user_content_seg}]

        # b) String-based structure (for Llama, Qwen, etc.)
        user_prompt_str = f"<image> {user_text}" if data.get(images_key) else user_text
        prompt_struct_str = system_msgs + [{"role": "user", "content": user_prompt_str}]

        # Render the prompt using the auto-detecting function
        prompt = _render_chat(apply_chat_template, prompt_struct_seg, prompt_struct_str)
    else:
        prompt = data.get(prompt_key, "")
        if input_template:
            prompt = input_template.format(prompt)

    # ---------- 2. Extract Images ----------
    images = data.get(images_key)

    # ---------- 3. Extract Reference with Fallbacks ----------
    reference = None
    if reference_key:
        reference = _normalize_reference(data.get(reference_key))

    # Fallback 1: Check inside a `reward_model` dictionary.
    if reference is None and isinstance(data.get("reward_model"), dict):
        reference = _normalize_reference(data["reward_model"].get("ground_truth"))

    # Fallback 2: Check for a top-level `ground_truth` key.
    if reference is None and "ground_truth" in data:
        reference = _normalize_reference(data["ground_truth"])

    # Fallback 3: Check for `constraints` inside `extra_info`.
    if reference is None:
        extra = data.get("extra_info")
        if isinstance(extra, dict):
            reference = _normalize_reference(extra.get("constraints"))

    # ---------- 4. Extract Label ----------
    label = _extract_label(data, label_key)

    return prompt, images, reference, label


# -------------------------------------------------------------
# Dataset Wrapper
# -------------------------------------------------------------
class PromptDatasetVL(Dataset):
    """
    A PyTorch Dataset for Vision-Language (VL) prompting tasks.

    This class wraps a raw dataset (e.g., a HuggingFace Dataset, list of dicts)
    and preprocesses each item on-the-fly using the `preprocess_data` function.
    It prepares the data in a `(prompt, images, reference, label)` format suitable
    for training or evaluation.

    :param dataset: The underlying raw dataset (can be a HuggingFace Dataset, list, or pandas-like object)
    :type dataset: Any
    :param tokenizer: The tokenizer for text processing
    :type tokenizer: Any
    :param processor: The processor, which may include the tokenizer and image processor
    :type processor: Any
    :param max_length: The maximum sequence length for the tokenizer
    :type max_length: int
    :param strategy: A configuration object containing keys and flags for data extraction
    :type strategy: Any
    :param input_template: Template for formatting input text
    :type input_template: str | None
    """
    def __init__(
        self,
        dataset,  # Can be a HuggingFace Dataset, list, or pandas-like object
        tokenizer,
        processor,
        max_length: int,
        strategy,
        input_template: str | None = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.strategy = strategy

        # Read field names and settings from the strategy object, with defaults.
        self.prompt_key = getattr(strategy.args, "input_key", "prompt")
        self.images_key = getattr(strategy.args, "images_key", "images")
        self.reference_key = getattr(strategy.args, "reference_key", None)
        self.label_key = getattr(strategy.args, "label_key", "label")
        self.apply_chat_template_flag = getattr(strategy.args, "apply_chat_template", False)
        self.system_prompt = getattr(strategy.args, "system_prompt", None)
        self.input_template = input_template
        # self.min_size = int(processor.image_processor.min_pixels ** 0.5)
        self.apply_chat_template = (processor.apply_chat_template if self.apply_chat_template_flag else None)

    def __len__(self) -> int:
        """
        Returns the total number of items in the dataset.

        :return: Number of items in the dataset
        :rtype: int
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any, Any]:
        """
        Retrieves and preprocesses one item from the dataset.

        :param idx: The index of the item to retrieve
        :type idx: int
        :return: A tuple containing the processed (prompt, images, reference, label)
        :rtype: Tuple[Any, Any, Any, Any]
        """
        data = self.dataset[idx]

        prompt, images, reference, label = preprocess_data(
            data=data,
            input_template=self.input_template,
            prompt_key=self.prompt_key,
            images_key=self.images_key,
            reference_key=self.reference_key,
            label_key=self.label_key,
            apply_chat_template=self.apply_chat_template,
            processor=self.processor,
            system_prompt=self.system_prompt,
        )
        return prompt, images, reference, label

    def collate_fn(self, batch: List[Tuple]) -> Tuple[List, List, List, List]:
        """
        Collates a batch of preprocessed data items.

        :param batch: A list of tuples, where each tuple is the output of `__getitem__`
        :type batch: List[Tuple]
        :return: A tuple of lists, containing (prompts, images, references, labels)
        :rtype: Tuple[List, List, List, List]
        """
        prompts, images, refs, labels = zip(*batch)
        return list(prompts), list(images), list(refs), list(labels)
