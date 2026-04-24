"""
General reward model helpers for the ORM RL Geo3K demo.

This example keeps only the general outcome reward-model path that is exercised by
`examples/orm_rl_demo/train_colocate.py` and `examples/orm_rl_demo/test_reward_models.py`.
The shared helper functions below are used by that general reward model for both
HuggingFace and engine-based inference.
"""
from __future__ import annotations

from typing import Optional, List, Tuple
import re
import copy
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import LogitsProcessor
from itertools import zip_longest

from lightrft.utils import get_current_device
from lightrft.strategy.utils.distributed_util import gather_inputs_object_for_inference
from lightrft.strategy import is_engine


# ============================================================================
# Utility Functions
# ============================================================================

def is_chinese(text):
    """
    Detect whether text contains Chinese characters.

    :param text: Text string to detect
    :type text: str
    :return: True if text contains Chinese characters, False otherwise
    :rtype: bool
    """
    if not isinstance(text, str):
        return False
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(chinese_pattern.search(text))


def _pack_engine_inputs(
    prompts: list[str],
    image_data: list[list] | None,
) -> tuple[list[str], list[list] | None]:
    """
    Pack engine inputs ensuring prompts and image_data have consistent lengths.

    Returns None for image_data when all images are empty to skip redundant parameters.

    :param prompts: List of text prompts
    :type prompts: list[str]
    :param image_data: List of image data, each element is a list of images
    :type image_data: list[list] or None
    :return: Processed (prompts, image_data or None)
    :rtype: tuple
    """
    if image_data is None:
        return prompts, None

    fixed_prompts, fixed_images = [], []
    for p, imgs in zip(prompts, image_data):
        if "<|image_pad|>" in p:
            fixed_prompts.append(p)
            fixed_images.append(imgs[:1] or [None])  # at least one placeholder
        else:
            fixed_prompts.append(p)
            fixed_images.append([])

    assert len(fixed_prompts) == len(fixed_images)

    if all(len(imgs) == 0 for imgs in fixed_images):
        fixed_images = None

    return fixed_prompts, fixed_images


def _align_prompts_images(
    prompts: list[str],
    image_data: list[list] | None,
) -> tuple[list[str], list[list] | None]:
    """
    Align prompts and images, separating text-only and multimodal data.

    Prompts containing ``<|image_pad|>`` must have at least one placeholder image;
    prompts without placeholders must have no images.

    :param prompts: List of text prompts
    :type prompts: list[str]
    :param image_data: List of image data (None if no images)
    :type image_data: list[list] or None
    :return: (text_prompts, text_indices, mm_prompts, mm_images)
    :rtype: tuple
    """
    if image_data is None:                    # No images passed at all
        return prompts, None
    text_prompts = []
    mm_prompts, mm_images = [], []
    text_inds = []

    ind = 0
    for p, imgs in zip_longest(prompts, image_data, fillvalue=None):
        if p is None:                         # Extra images → discard
            continue

        imgs = [] if imgs is None else imgs   # Ensure imgs is a list
        if "<|image_pad|>" in p:              # Must keep 1 placeholder
            imgs = imgs[:1] or [None]
            if isinstance(imgs[0], list):
                imgs = imgs[0]
            mm_images.append(imgs)
            mm_prompts.append(p)
        else:                                 # Pure text prompt cannot have images
            text_prompts.append(p)
            text_inds.append(ind)

        ind += 1

    return text_prompts, text_inds, mm_prompts, mm_images


def _hf_or_engine_generate(
    model,
    *,
    input_ids       : torch.Tensor | None = None,
    attention_mask  : torch.Tensor | None = None,
    pixel_values    : torch.Tensor | None = None,
    image_grid_thw  : torch.Tensor | None = None,
    prompts         : List[str]  | None = None,
    image_data      : List[List] | None = None,
    **gen_kwargs,
) -> Tuple[List[str], torch.Tensor | None]:
    """
    Unified generation interface supporting both HuggingFace models and SGLang engines.

    Automatically detects model type. Engine mode uses string prompts and image_data;
    HF mode uses tensor inputs (input_ids, pixel_values, etc.).

    :param model: HF model or SGLang engine instance
    :param input_ids: Input token IDs for HF mode
    :type input_ids: torch.Tensor or None
    :param attention_mask: Attention mask for HF mode
    :type attention_mask: torch.Tensor or None
    :param pixel_values: Image pixel values for HF mode
    :type pixel_values: torch.Tensor or None
    :param image_grid_thw: Image grid size for HF mode
    :type image_grid_thw: torch.Tensor or None
    :param prompts: Text prompts for Engine mode
    :type prompts: list[str] or None
    :param image_data: Image data for Engine mode
    :type image_data: list[list] or None
    :param gen_kwargs: Generation parameters (max_new_tokens, temperature, etc.)
    :return: (list of generated texts, generated token IDs or None)
    :rtype: tuple
    """
    if is_engine(model):
        assert input_ids is None, "Cannot pass input_ids in engine mode"
        enable_sleep_mode = True
        if hasattr(model, "llm_engine"):
            enable_sleep_mode = model.llm_engine.vllm_config.model_config.enable_sleep_mode

        if enable_sleep_mode:
            model.wake_up()

        if hasattr(model, "tp_group_cpu"):
            sampling_params = {
                **{k: v for k, v in gen_kwargs.items() if k not in ("do_sample")}
            }

            prompt_and_output = gather_inputs_object_for_inference(prompts, model.tp_group_cpu)
            image_data = gather_inputs_object_for_inference(image_data, model.tp_group_cpu)

            text_prompts, text_inds, mm_prompts, mm_images = _align_prompts_images(prompt_and_output, image_data)
            text_output = []
            mm_output = []

            if len(text_prompts) > 0:
                sgl_outputs = model.generate(prompt=text_prompts, sampling_params=sampling_params, gather_inputs=False)
                text_output = [sgl_out["text"] for sgl_out in sgl_outputs]

            if len(mm_prompts) > 0:
                sgl_outputs = model.generate(
                    prompt=mm_prompts,
                    image_data=mm_images,
                    sampling_params=sampling_params,
                    gather_inputs=False,
                )
                mm_output = [sgl_out["text"] for sgl_out in sgl_outputs]

            texts = []
            text_output_iter = iter(text_output)
            mm_output_iter = iter(mm_output)
            # merge results in original order
            if len(text_inds) > 0:
                for i in range(len(prompt_and_output)):
                    if i in text_inds:
                        texts.append(next(text_output_iter))
                    else:
                        texts.append(next(mm_output_iter))
            else:
                texts = mm_output

            if model._tp_size > 1:
                num_per_rank = len(texts) // model._tp_size
                texts = texts[model._tp_rank * num_per_rank : (model._tp_rank + 1) * num_per_rank]
        else:
            from vllm import SamplingParams

            sampling_kwargs = dict(gen_kwargs)
            max_tokens = sampling_kwargs.pop("max_new_tokens", None)
            if max_tokens is not None:
                sampling_kwargs["max_tokens"] = max_tokens
            sampling_kwargs.pop("do_sample", None)
            sampling_params = SamplingParams(**sampling_kwargs)

            prompt_and_output = prompts or []
            prompt_and_output, image_data = _pack_engine_inputs(
                prompt_and_output,
                image_data,
            )
            if image_data is None:
                image_data = [None] * len(prompt_and_output)

            vllm_prompts = []
            for prompt, imgs in zip(prompt_and_output, image_data):
                prompt_item = {"prompt": prompt}
                if imgs:
                    prompt_item["multi_modal_data"] = {
                        "image": imgs[0] if len(imgs) == 1 else imgs
                    }
                vllm_prompts.append(prompt_item)

            vllm_outputs = model.generate(
                vllm_prompts,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            texts = [
                out.outputs[0].text if getattr(out, "outputs", None) else ""
                for out in vllm_outputs
            ]

        if dist.is_initialized() and dist.get_rank() == 0:
            if not texts or all(not t for t in texts):
                print("WARNING: _hf_or_engine_generate produced empty output for all prompts.")

        if enable_sleep_mode:
            model.sleep()
        torch.cuda.empty_cache()
        return texts, None

    else:
        gen_ids = model.generate(
            input_ids        = input_ids,
            attention_mask   = attention_mask,
            pixel_values     = pixel_values,
            image_grid_thw   = image_grid_thw,
            **gen_kwargs,
        )
        trim = [o[len(i):] for i, o in zip(input_ids, gen_ids)]
        return trim, trim


# ============================================================================
# Vision Token Processing
# ============================================================================

_VISION_RE = re.compile(r"<\|vision_start\|>.*?<\|vision_end\|>", re.S)

def _strip_vision_tokens(text: str) -> str:
    """Remove vision token markers from text."""
    return re.sub(_VISION_RE, "", text).replace("<image>", "").strip()


def _clean_vision_token(text: str) -> str:
    """
    Clean vision tokens from text, supporting multiple formats.

    Supported formats:
        - <|vision_start|><|image_pad|>...<|vision_end|>
        - <img><IMG_CONTEXT>...</img>
        - <image>
    """
    patterns = [
        r"<\|vision_start\|>(<\|image_pad\|>)+<\|vision_end\|>",
        r"<img>(<IMG_CONTEXT>)+</img>",
        r"<image>"
    ]
    for p in patterns:
        text = re.sub(p, "", text)
    return text


def _replace_vision_token(text: str) -> str:
    """
    Replace vision tokens with standard <image> markers.

    Conversion rules:
        - <|vision_start|>...<|vision_end|> -> <image>
        - <img>...<IMG_CONTEXT>...</img> -> <image> (internvl format)
    """
    text = re.sub(r"<\|vision_start\|>(<\|image_pad\|>)+<\|vision_end\|>", "<image>", text)
    text = re.sub(r"<img>(<IMG_CONTEXT>)+</img>", "<image>", text) # internvl

    return text


def _strip_pad_eos(text: str, pad: str, eos: str) -> str:
    """
    Remove leading and trailing pad and eos tokens from text.

    :param text: Text to process
    :type text: str
    :param pad: Pad token string
    :type pad: str
    :param eos: EOS token string
    :type eos: str
    :return: Cleaned text
    :rtype: str
    """
    pad, eos = map(re.escape, (pad, eos))
    text = re.sub(f"^({eos}|{pad})+", "", text)
    text = re.sub(f"({eos}|{pad})+$", "", text)
    return text

# ============================================================================
# Dialog Parsing Constants and Functions
# ============================================================================

# Define constants for vertical bars used in role tags for better readability
FULL_BAR = "｜"  # U+FF5C Full-width vertical bar
HALF_BAR = "|"  # U+007C ASCII vertical bar

def _parse_dialog(text: str) -> dict:
    """
    Parse a full conversation string into a dictionary mapping roles to their content.

    Identifies role tags like ``<| role_name |>`` and extracts the text that follows
    each tag. If a role appears multiple times, only the last occurrence is kept.

    :param text: Conversation string with role tags
    :type text: str
    :return: Dict mapping role names to their message content
    :rtype: dict
    """
    # 1. Define the regex pattern to find all possible role tags.
    # The pattern is written in verbose mode (re.X) for clarity.
    tag_pattern = re.compile(
        rf"""
        <                       # Match the opening '<'
        [{HALF_BAR}{FULL_BAR}]  # Match either a half-width or full-width vertical bar
        \s*?                    # Match any whitespace characters (non-greedy)
        (.*?)                   # Capture the role name (non-greedy)
        \s*?                    # Match any whitespace characters (non-greedy)
        [{HALF_BAR}{FULL_BAR}]  # Match either a half-width or full-width vertical bar
        >                       # Match the closing '>'
        """, re.X | re.S
    )

    # Find all occurrences of role tags in the text.
    tags = list(tag_pattern.finditer(text))
    dialog = {}

    # 2. Iterate through the found tags to extract roles and content.
    for idx, tag in enumerate(tags):
        # Extract the role name and normalize it by stripping whitespace and converting to lowercase.
        raw_role = tag.group(1).strip()
        role = raw_role.lower()

        # Skip special meta-tags that define structure but are not roles.
        if role in {"im_start", "im_end", "begin of sentence", "end of sentence"}:
            continue

        # Determine the start and end positions of the content for the current role.
        # The content starts right after the current tag.
        start_pos = tag.end()
        # The content ends right before the next tag starts, or at the end of the text.
        end_pos = tags[idx + 1].start() if idx + 1 < len(tags) else len(text)
        content = text[start_pos:end_pos].strip()

        # 3. Special handling for the 'assistant' role to remove the chain-of-thought block.
        # If the content contains <think>...</think>, we extract only the final response
        # that appears after the last </think> tag.
        if role == "assistant" and "<think>" in content and "</think>" in content:
            think_end = content.rfind("</think>")
            if think_end != -1:
                content = content[think_end + len("</think>"):].strip()

        # Store the role and its content in the dictionary.
        # If the role already exists, its value will be updated with the new content.
        dialog[role] = content

    return dialog

def preprocess_inputs_sglang(
    prompt_and_outputs: list,
    references: list,
    question_response_format_zh: list or str,
    question_response_format_en: str,
    system_prompt_zh: str = None,
    system_prompt_en: str = None,
    system_prompt: bool = False,
):
    """
    Preprocess batch conversation inputs for SGLang engine.

    Parses conversation text, selects a format template based on detected language,
    and optionally prepends a system prompt.

    :param prompt_and_outputs: List of conversation texts
    :type prompt_and_outputs: list
    :param references: List of reference answers
    :type references: list
    :param question_response_format_zh: Chinese format template (string or per-sample list)
    :type question_response_format_zh: str or list
    :param question_response_format_en: English format template
    :type question_response_format_en: str
    :param system_prompt_zh: Chinese system prompt
    :type system_prompt_zh: str or None
    :param system_prompt_en: English system prompt
    :type system_prompt_en: str or None
    :param system_prompt: Whether to prepend a system prompt
    :type system_prompt: bool
    :return: List of formatted texts ready for model input
    :rtype: list
    """
    raw_texts = []
    # Process each conversation in the batch.
    for i, po in enumerate(prompt_and_outputs):
        # Parse the conversation string into a role-content dictionary.
        dialog = _parse_dialog(po)

        # --- Step 1: Extract the question ---
        if "user" in dialog:
            question_raw = dialog["user"]
        else:
            # Fallback logic: if 'user' role is not found, use the content from the
            # first role that is not 'assistant'. If no such role exists,
            # use the entire original string as the question.
            question_raw = next(
                (txt for role, txt in dialog.items() if role != "assistant"), po
            )
        # Clean the extracted question (e.g., remove special vision tokens).
        # Note: _clean_vision_token function is assumed to be defined elsewhere.
        question = _clean_vision_token(question_raw)

        # --- Step 2: Extract the response ---
        if "assistant" in dialog:
            response = dialog["assistant"]
        else:
            # Fallback logic: if 'assistant' role is not found, assume the response
            # is the text following the last </think> tag.
            response = po.split("</think>")[-1].strip()

        reference = references[i]

        # --- Step 3: Select the appropriate formatting template ---
        # Note: is_chinese function is assumed to be defined elsewhere.
        is_zh = is_chinese(question)
        if isinstance(question_response_format_zh, list):
            # New feature: Use a custom template for each item in the batch.
            fmt = question_response_format_zh[i]
        else:
            # Old logic: Choose the template based on the detected language.
            fmt = question_response_format_zh if is_zh else question_response_format_en

        # --- Step 4: Format the final input string ---
        # The template may or may not include a placeholder for the reference text.
        if "{reference}" in fmt:
            raw_text = fmt.format(
                question=question,
                reference=reference,
                response=response
            )
        else:
            raw_text = fmt.format(question=question, response=response)

        # --- Step 5: Prepend a system prompt if enabled ---
        if system_prompt:
            # Select the system prompt based on the language.
            system_prompt_text = system_prompt_zh if is_zh else system_prompt_en
            # Using deepcopy to avoid modifying the original system prompt object.
            final_text = copy.deepcopy(system_prompt_text) + "\n" + raw_text
            raw_texts.append(final_text)
        else:
            raw_texts.append(raw_text)

    return raw_texts


def build_general_engine_queries(
    processor,
    prompt_and_outputs: list,
    references: list,
    raw_images: list | None,
    question_response_format_zh: str,
    question_response_format_en: str,
    system_prompt_zh: str,
    system_prompt_en: str,
):
    """
    Build general-RM engine prompts using the model's chat template.

    The vLLM engine path for Qwen2.5-VL is much more stable when we append the
    assistant generation prompt explicitly. Without this, the engine often
    returns empty strings or prompt-continuation fragments instead of verdicts.
    """
    test_data = []
    expected_image_counts = []
    normalized_image_data = []

    if raw_images is None:
        raw_images = [None] * len(prompt_and_outputs)

    for i, prompt_and_output in enumerate(prompt_and_outputs):
        dialog = _parse_dialog(prompt_and_output)

        if "user" in dialog:
            question_raw = dialog["user"]
        else:
            question_raw = next(
                (txt for role, txt in dialog.items() if role != "assistant"),
                prompt_and_output,
            )

        if "assistant" in dialog:
            response = dialog["assistant"]
        else:
            response = prompt_and_output.split("</think>")[-1].strip()

        question = _clean_vision_token(question_raw)
        reference = references[i] if references is not None and i < len(references) else ""
        is_zh = is_chinese(question)
        fmt = question_response_format_zh if is_zh else question_response_format_en
        system_prompt = system_prompt_zh if is_zh else system_prompt_en
        user_text = fmt.format(question=question, response=response, reference=reference)

        raw_image = raw_images[i] if i < len(raw_images) else None
        has_image = raw_image is not None
        expected_image_counts.append(1 if has_image else 0)
        normalized_image_data.append([raw_image] if has_image else [])

        user_content = [{"type": "text", "text": user_text}]
        if has_image:
            user_content = [
                {
                    "type": "image",
                    "image": [],
                    "min_pixels": 224 * 224,
                    "max_pixels": 1280 * 1280,
                },
                {"type": "text", "text": user_text},
            ]

        test_data.append(
            [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": user_content},
            ]
        )

    queries = processor.apply_chat_template(
        test_data,
        tokenize=False,
        add_generation_prompt=True,
    )
    if isinstance(queries, str):
        queries = [queries]

    fixed_queries = []
    for query, expected_image_count in zip(queries, expected_image_counts):
        query_image_token_count = query.count("<|image_pad|>")
        if query_image_token_count > expected_image_count:
            excess_tokens = query_image_token_count - expected_image_count
            query = query.replace("<|image_pad|>", "", excess_tokens)
        fixed_queries.append(query)

    return fixed_queries, normalized_image_data


def preprocess_inputs(
    tokenizer = None,
    processor = None,
    device = get_current_device(),
    system_prompt: Optional[str] = None,
    question_response_format: str = "",
    input_ids: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pad_token: str = "<pad>",
    eos_token: str = "<|endoftext|>",
    clean_or_replace_vision_token: bool = False,
    vision_token_process_type: str = 'clean',
    padding_side: str = "left",
    return_think_content: bool = False,
    debug: bool = False,
    queries: Optional[list] = None,
    return_raw_texts: bool = False,
):
    """
    Preprocess inputs for HuggingFace models.

    Supports building inputs from ``input_ids`` or ``queries``, optional vision-token
    processing, and chain-of-thought content separation.

    :param tokenizer: HF tokenizer instance
    :param processor: HF processor instance
    :param device: Target device
    :param system_prompt: System prompt (optional; use to distinguish value/knowledge from safety/normal data)
    :type system_prompt: str or None
    :param question_response_format: Q&A format template
    :type question_response_format: str
    :param input_ids: Input token IDs
    :type input_ids: torch.Tensor or None
    :param pixel_values: Image pixel values
    :type pixel_values: torch.Tensor or None
    :param pad_token: Padding token
    :type pad_token: str
    :param eos_token: End-of-sequence token
    :type eos_token: str
    :param clean_or_replace_vision_token: Whether to process vision tokens
    :type clean_or_replace_vision_token: bool
    :param vision_token_process_type: Processing method (``'clean'`` or ``'replace'``)
    :type vision_token_process_type: str
    :param padding_side: Padding direction
    :type padding_side: str
    :param return_think_content: Whether to separate chain-of-thought content
    :type return_think_content: bool
    :param debug: Debug mode
    :type debug: bool
    :param queries: List of query texts
    :type queries: list or None
    :param return_raw_texts: Whether to return raw texts instead of tensors
    :type return_raw_texts: bool
    :return: Standard mode returns ``(input_ids, attention_mask, response_empty)``;
             CoT mode returns ``(answer_input_ids, answer_mask, think_input_ids, think_mask, valid_think, response_empty)``;
             raw text mode returns ``(raw_texts, ...)``.
    :rtype: tuple
    """
    if input_ids is not None:
        processor.tokenizer.padding_side = padding_side
        queries = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
    else:
        assert queries is not None

    for i, query in enumerate(queries):
        if clean_or_replace_vision_token:
            if vision_token_process_type == 'clean':  # value, knowledge
                queries[i] = _clean_vision_token(query)
            elif vision_token_process_type == 'replace':  # safety, normal
                queries[i] = _replace_vision_token(query)
            else:
                raise KeyError(f"Invalid vision token process type: {vision_token_process_type}")
        queries[i] = _strip_pad_eos(queries[i], pad_token, eos_token) + eos_token

    # Extract question and response from query using regex
    pattern = r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>"
    # NOTE: parse dialog logic haven't adapt to deepseek model now
    def _prepare_message(dialog, test_data, image_token_count_list):
        question = dialog.get('user', '')
        response = dialog.get('assistant', '')
        image_token_count_list.append(question.count('<|image_pad|>'))
        if system_prompt is not None:
            test_data.append(
                [
                    {"role": "system", "content":[{"type": "text", "text": system_prompt}]},
                    {"role": "user", "content": [{"type": "image", "image": [], "min_pixels": 224 * 224, "max_pixels": 1280 * 1280}, {"type": "text", "text": question_response_format.format(question=question, response=response)}]}
                ]
            )
        else:
            test_data.append(
                [
                    {"role": "user", "content": [{"type": "text", "text": question_response_format.format(question=question, response=response)}]}
                ]
            )
            if debug and dist.is_initialized() and dist.get_rank() == 0:
                print(f"test_data:\n {test_data[0]}\n")

    # Process all queries in the batch at once
    test_data, image_token_count_list = [], []
    think_test_data, think_image_token_count_list, valid_think = [], [], []
    response_empty = []
    for query in queries:
        matches = re.findall(pattern, query, re.DOTALL)
        dialog = {}
        if return_think_content:
            think_dialog = {}
            valid_think_flag = False
        for role, content in matches:
            dialog[role] = content.strip()
            if return_think_content:
                think_dialog[role] = content.strip()
            # If assistant's reply contains thinking chain content wrapped in <think> and </think>, extract only the content after </think>
            if role == "assistant" and "<think>" in content and "</think>" in content:
                # Find the position of the last </think>
                think_end_pos = content.rfind("</think>")
                if think_end_pos != -1:
                    # Extract content after </think> and remove leading/trailing whitespace
                    dialog[role] = content[think_end_pos + len("</think>"):].strip()
                    if return_think_content:
                        think_dialog[role] = content[:think_end_pos + len("</think>") + 1].strip()
                        valid_think_flag = True

        _prepare_message(dialog, test_data, image_token_count_list)
        response_empty.append(dialog.get('assistant', '') == '')
        if return_think_content:
            valid_think.append(valid_think_flag)
            _prepare_message(think_dialog, think_test_data, think_image_token_count_list)

    def _get_batch_input(test_data, image_token_count_list, return_raw_texts):
        # Process the entire batch at once
        if system_prompt is not None:
            # Only apply chat template when system prompt is provided
            queries = processor.apply_chat_template(test_data, tokenize=False, add_generation_prompt=False)
        else:
            # For data without system prompt, format directly without applying chat template
            queries = [item[0]["content"][0]["text"] for item in test_data]

        # TODO: `apply_chat_template` will add a extra image token in the query, so we need to remove it now, we need more elegant way
        for i, query in enumerate(queries):
            query_image_token_count = query.count('<|image_pad|>')
            if query_image_token_count > image_token_count_list[i]:
                # Replace all excess image tokens to match the expected count
                excess_tokens = query_image_token_count - image_token_count_list[i]
                queries[i] = query.replace('<|image_pad|>', '', excess_tokens)

        if not return_raw_texts:
            with torch.no_grad():
                batch_inputs = processor(
                    text=queries,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
            return batch_inputs
        else:
            return queries

    answer_batch_input = _get_batch_input(test_data, image_token_count_list, return_raw_texts)
    if return_think_content:
        think_batch_input = _get_batch_input(think_test_data, think_image_token_count_list, return_raw_texts)
        if not return_raw_texts:
            return answer_batch_input['input_ids'], answer_batch_input['attention_mask'], think_batch_input['input_ids'], think_batch_input['attention_mask'], valid_think, response_empty
        else:
            return answer_batch_input, think_batch_input, valid_think
    else:
        if not return_raw_texts:
            return answer_batch_input['input_ids'], answer_batch_input['attention_mask'], response_empty
        else:
            return answer_batch_input


    if engine._tp_size > 1:
        num_per_rank = len(texts) // engine._tp_size
        texts = texts[engine._tp_rank * num_per_rank : (engine._tp_rank+1) * num_per_rank]

    return texts


# ============================================================================
# General Reward Model
# ============================================================================

class AllowedTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids, scores):
        # Set all non-allowed tokens to very negative values
        mask = torch.ones_like(scores) * float('-inf')
        for token_id in self.allowed_token_ids:
            mask[:, token_id] = 0
        return scores + mask


class Qwen2VLRewardModelGeneral(nn.Module):
    """
    General quality reward model that evaluates answer correctness based on reference answers.

    Scoring rules:

    - ``1.0``: Completely correct (all sub-questions correct)
    - ``0.5``: Partially correct (at least one sub-question correct, but not all)
    - ``0.0``: Incorrect (all sub-questions wrong or answer irrelevant)

    :param base_model: HF model or Engine instance
    :param tokenizer: Tokenizer instance
    :param processor: Processor instance
    :param text_only: Whether to use text-only mode (no image inputs)
    :type text_only: bool
    """

    general_scores = [0.0, 0.5, 1.0]
    general_system_prompt_zh = """你是一个评分专家，负责根据参考答案reference评估assistant对user的回复是否正确且合理。
    **你将收到包含以下XML标签的内容：`<user>`表示用户的问题，`<assistant>`表示助手的回答，`<reference>`表示参考答案。**
    请严格按以下规则输出固定稀疏奖励：

    评估规则：
    1. 答案等价性：
    - 简洁答案和带解题步骤的答案都接受，只要包含正确答案
    - 答案可能出现在回答的开头、中间或结尾
    - 只比较核心答案，忽略解释部分

    2. 数值等价性：
    - 不同格式的数字视为等价(如2,"2",['2'],"答案是2")
    - 百分比可以用小数或%表示(如28%=0.28)
    - 带/不带逗号的数字视为等价(如123,456.7=123456.7)

    3. 格式灵活性：
    - 列表、引号、表格或纯文本中的正确答案都接受
    - 正确答案周围的额外解释或格式不影响评分
    - 大小写不敏感

    4. 多参考答案情况：
    - 参考答案有多个可接受答案时，匹配一个即可视为该部分正确。

    5. 多子问题情况：
    - 如果问题包含多个子问题，需要逐一评估assistant对每个子问题的回答。
    - 只有当所有子问题都回答正确时，总分才为 1.0。
    - 如果至少有一个子问题回答正确，但并非所有子问题都正确，则总分为 0.5。
    - 如果所有子问题都回答错误或回答与问题无关，则总分为 0.0。

    6. 容错性：
    - 轻微拼写错误或措辞差异不影响评分
    - 等价数学表达式视为正确

    输出要求：
    1. **仅允许输出以下三个数值之一：0.0、0.5、1.0**
    2. 根据参考答案与回答的匹配程度选择：
    - 完全正确 (所有子问题均正确) → 1.0
    - 部分正确 (至少答对一个子问题，但非全部) → 0.5
    - 错误 (所有子问题均错误或回答与问题无关) → 0.0
    3. 直接输出数值，不需要任何解释"""

    question_response_format_zh = """请根据以下内容进行评估：

    <user>
    {question}
    </user>


    <assistant>
    {response}
    </assistant>

    <reference>
    {reference}
    </reference>"""

    general_system_prompt_en = """You are a scoring expert responsible for evaluating whether the assistant's response to the user is correct and reasonable based on the reference answer.
    **You will receive content with the following XML tags: `<user>` represents the user's question, `<assistant>` represents the assistant's answer, and `<reference>` represents the reference answer.**
    Please strictly output fixed sparse rewards according to the following rules:

    Evaluation Rules:
    1. Answer Equivalence:
    - Both concise answers and answers with solution steps are accepted, as long as they contain the correct answer
    - The answer may appear at the beginning, middle, or end of the response
    - Only compare core answers, ignore explanation parts

    2. Numerical Equivalence:
    - Numbers in different formats are considered equivalent (e.g., 2, "2", ['2'], "the answer is 2")
    - Percentages can be expressed as decimals or % (e.g., 28% = 0.28)
    - Numbers with/without commas are equivalent (e.g., 123,456.7 = 123456.7)

    3. Format Flexibility:
    - Correct answers in lists, quotes, tables, or plain text are all accepted
    - Additional explanations or formatting around the correct answer do not affect scoring
    - Case insensitive

    4. Multiple Reference Answers:
    - When there are multiple acceptable reference answers, matching any one is considered correct for that part.

    5. Multiple Sub-questions:
    - If the question contains multiple sub-questions, evaluate the assistant's answer for each sub-question.
    - Only when all sub-questions are answered correctly will the total score be 1.0.
    - If at least one sub-question is answered correctly, but not all sub-questions are correct, the total score is 0.5.
    - If all sub-questions are answered incorrectly or the answer is irrelevant to the question, the total score is 0.0.

    6. Error Tolerance:
    - Minor spelling errors or wording differences do not affect scoring
    - Equivalent mathematical expressions are considered correct

    Output Requirements:
    1. **Only the following three values are allowed: 0.0, 0.5, 1.0**
    2. Choose based on the degree of match between the reference answer and the response:
    - Completely correct (all sub-questions correct) → 1.0
    - Partially correct (at least one sub-question correct, but not all) → 0.5
    - Incorrect (all sub-questions incorrect or answer irrelevant to question) → 0.0
    3. Output the value (0.0, 0.5, 1.0) directly, no explanation needed"""

    question_response_format_en = """Please evaluate based on the following content:

    <user>
    {question}
    </user>


    <assistant>
    {response}
    </assistant>

    <reference>
    {reference}
    </reference>"""

    ALLOWED_STR_TOKENS = ["0", "1", "0.0", "0.5", "1.0"]

    def __init__(self, base_model, tokenizer, processor, text_only: bool = False):
        super().__init__()
        self.base_model: nn.Module = base_model
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = torch.cuda.current_device()
        self.text_only = text_only

        self._allowed_token_seqs: list[list[int]] = []
        for s in self.ALLOWED_STR_TOKENS:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            self._allowed_token_seqs.append(ids)

        self._verdict_log_enabled = os.environ.get("ORM_RL_DEMO_RM_VERDICT_LOG", "0") == "1"
        self._verdict_log_max = int(os.environ.get("ORM_RL_DEMO_RM_VERDICT_LOG_MAX", "128"))
        self._verdict_log_count = 0

        first_ids = {seq[0] for seq in self._allowed_token_seqs}
        self._logits_proc = [AllowedTokensLogitsProcessor(first_ids)]
        self._max_answer_len = max(len(x) for x in self._allowed_token_seqs)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        references: List[str] | None = None,
        prompt_and_outputs=None,
        prompt_and_output=None,
        raw_images=None,
        **kwargs,  # for compatibility
    ):
        """
        Returns: {'score':  FloatTensor[B]}, only in 0/0.5/1
        """
        if prompt_and_outputs is None:
            prompt_and_outputs = prompt_and_output
        if prompt_and_outputs is None:
            prompt_and_outputs = kwargs.get("prompt_and_output")
        if prompt_and_outputs is None:
            raise ValueError("`prompt_and_outputs` or `prompt_and_output` is required")

        raw_texts = preprocess_inputs_sglang(
            prompt_and_outputs,
            references,
            self.question_response_format_zh,
            self.question_response_format_en,
            self.general_system_prompt_zh,
            self.general_system_prompt_en,
            system_prompt=True,
        )

        if is_engine(self.base_model):
            raw_texts, raw_images = build_general_engine_queries(
                self.processor,
                prompt_and_outputs,
                references,
                raw_images,
                self.question_response_format_zh,
                self.question_response_format_en,
                self.general_system_prompt_zh,
                self.general_system_prompt_en,
            )
            gen_texts, _ = _hf_or_engine_generate(
                self.base_model,
                prompts=raw_texts,
                image_data=raw_images,
                max_new_tokens=4,
                temperature=0.0,
            )
        else:
            model_in = self.processor(
                text=raw_texts, padding=True, return_tensors="pt"
            ).to(self.device)
            _, gen_ids = _hf_or_engine_generate(
                self.base_model,
                input_ids=model_in["input_ids"],
                attention_mask=model_in["attention_mask"],
                pixel_values=None,
                image_grid_thw=None,
                max_new_tokens=self._max_answer_len,
                temperature=0.0,
                do_sample=False,
                logits_processor=self._logits_proc,
            )
            gen_texts = self.tokenizer.batch_decode(
                gen_ids, skip_special_tokens=True
            )

        log_on_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0

        def _log_verdict_detail(tag: str, sample_idx: int, raw_text: str, **fields) -> None:
            if not (self._verdict_log_enabled and log_on_rank0):
                return
            if self._verdict_log_count >= self._verdict_log_max:
                return
            raw_text = raw_text if isinstance(raw_text, str) else str(raw_text)
            preview = " ".join(raw_text.split())
            if len(preview) > 200:
                preview = preview[:200] + "..."
            extras = " ".join(f"{key}={value}" for key, value in fields.items())
            print(
                f"[ORM_RM_GENERAL_VERDICT_{tag}] "
                f"sample_idx={sample_idx} text_len={len(raw_text)} raw={preview!r} {extras}".rstrip(),
                flush=True,
            )
            self._verdict_log_count += 1

        verdict_summary = {
            "total": len(gen_texts),
            "empty": 0,
            "no_numeric": 0,
            "value_error": 0,
            "parsed": 0,
            "parsed_0": 0,
            "parsed_0_5": 0,
            "parsed_1": 0,
        }

        scores = []
        for sample_idx, txt in enumerate(gen_texts):
            txt = txt if isinstance(txt, str) else str(txt)
            if txt == "":
                verdict_summary["empty"] += 1
                scores.append(0.0)
                _log_verdict_detail("EMPTY", sample_idx, txt, fallback="0.0")
                continue

            m = re.search(r"[-+]?\d*\.?\d+", txt)
            if not m:
                verdict_summary["no_numeric"] += 1
                scores.append(0.0)
                _log_verdict_detail("NO_NUMERIC", sample_idx, txt, fallback="0.0")
                continue

            matched_token = m.group()
            try:
                val = float(matched_token)
            except ValueError as exc:
                verdict_summary["value_error"] += 1
                scores.append(0.0)
                _log_verdict_detail(
                    "VALUE_ERROR",
                    sample_idx,
                    txt,
                    token=repr(matched_token),
                    error=repr(exc),
                    fallback="0.0",
                )
                continue

            nearest = min(self.general_scores, key=lambda x: abs(x - val))
            verdict_summary["parsed"] += 1
            if nearest == 0.0:
                verdict_summary["parsed_0"] += 1
            elif nearest == 0.5:
                verdict_summary["parsed_0_5"] += 1
            elif nearest == 1.0:
                verdict_summary["parsed_1"] += 1
            _log_verdict_detail(
                "PARSED",
                sample_idx,
                txt,
                token=repr(matched_token),
                parsed=repr(val),
                snapped=repr(nearest),
            )
            scores.append(nearest)

        if self._verdict_log_enabled and log_on_rank0:
            print(
                "[ORM_RM_GENERAL_VERDICT_SUMMARY] "
                + " ".join(f"{key}={value}" for key, value in verdict_summary.items()),
                flush=True,
            )

        return {"score": torch.tensor(scores, device=self.device)}
