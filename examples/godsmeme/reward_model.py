import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from lightrft.utils import get_current_device

from meme_utils import (
    MemeRenderConfig,
    PairwisePreference,
    aggregate_pairwise_preferences,
    compute_meme_format_reward,
    extract_box_texts,
    get_first_image,
    get_user_request,
    load_text_file,
    normalize_detections,
    render_meme_image,
    resolve_expected_box_count,
    sample_group_pairs,
)

_PAIRWISE_LABEL_KEY = "pairwise"
_REWARD_STATE = {
    "model_reward_weight": 1.0,
    "format_reward_weight": 0.1,
}


def _as_torch_device(device_like: Any) -> torch.device:
    if isinstance(device_like, torch.device):
        return device_like
    if isinstance(device_like, int):
        return torch.device(f"cuda:{device_like}")
    return torch.device(device_like)


def _default_reward_prompt_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "reward_compare.txt")


def _load_reward_prompt(path: Optional[str] = None) -> str:
    prompt_path = path or _default_reward_prompt_path()
    if os.path.exists(prompt_path):
        return load_text_file(prompt_path)
    return (
        "You are a strict meme reward model.\n"
        "Compare the two meme images and return exactly:\n"
        "Image 1 score: <0-10>\n"
        "Image 2 score: <0-10>\n"
        "Winner: <1 or 2 or tie>\n"
        "Reason: <short sentence>"
    )


def _parse_reward_config(raw_reward_pretrain: str) -> Dict[str, Any]:
    if raw_reward_pretrain is None or not str(raw_reward_pretrain).strip():
        raise ValueError("`reward_pretrain` must point to a meme judge model or a JSON config")

    text = str(raw_reward_pretrain).strip()
    try:
        cfg = json.loads(text)
    except json.JSONDecodeError:
        cfg = {"pairwise": {"path": text}}

    if isinstance(cfg, str):
        cfg = {"pairwise": {"path": cfg}}
    if not isinstance(cfg, dict):
        raise ValueError("Unsupported meme reward config format")

    if "pairwise" in cfg:
        pairwise_cfg = cfg["pairwise"]
    elif "outcome" in cfg:
        pairwise_cfg = cfg["outcome"]
    else:
        first_key = next(iter(cfg.keys()))
        pairwise_cfg = cfg[first_key]

    if isinstance(pairwise_cfg, str):
        pairwise_cfg = {"path": pairwise_cfg}
    if not isinstance(pairwise_cfg, dict) or not pairwise_cfg.get("path"):
        raise ValueError("Meme reward config must contain a model path")
    return pairwise_cfg


def build_pairwise_judge_message(
    reward_prompt: str,
    image_a,
    image_b,
    text_a: str,
    text_b: str,
    user_request: str,
) -> List[Dict[str, Any]]:
    prompt = reward_prompt.format(
        user_request=user_request or "N/A",
        text_a=text_a or "(empty)",
        text_b=text_b or "(empty)",
    )
    return [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"{prompt}\n\nCandidate 1 image:"
            },
            {
                "type": "image",
                "image": image_a
            },
            {
                "type": "text",
                "text": "Candidate 2 image:"
            },
            {
                "type": "image",
                "image": image_b
            },
        ],
    }]


def parse_pair_judge_response(response: str) -> Tuple[float, float]:
    response = (response or "").strip()
    if not response:
        return 0.5, 0.5

    def _extract_score(image_idx: int) -> Optional[float]:
        pattern = rf"Image\s*{image_idx}\s*score\s*[:：]\s*([0-9]+(?:\.[0-9]+)?)"
        match = re.search(pattern, response, re.IGNORECASE)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    score_a = _extract_score(1)
    score_b = _extract_score(2)
    if score_a is not None and score_b is not None:
        return score_a, score_b

    winner_match = re.search(r"Winner\s*[:：]\s*(1|2|tie)", response, re.IGNORECASE)
    if winner_match:
        winner = winner_match.group(1).lower()
        if winner == "1":
            return 1.0, 0.0
        if winner == "2":
            return 0.0, 1.0
        return 0.5, 0.5

    numeric_scores = re.findall(r"([0-9]+(?:\.[0-9]+)?)", response)
    if len(numeric_scores) >= 2:
        try:
            return float(numeric_scores[0]), float(numeric_scores[1])
        except ValueError:
            pass
    return 0.5, 0.5


def group_meme_rollout_indices(
    refs: Optional[Sequence[Any]],
    batch_size: int,
    n_samples_per_prompt: int = 1,
) -> List[List[int]]:
    if refs and len(refs) >= batch_size:
        groups: List[List[int]] = []
        current_group: List[int] = []
        current_group_id: Optional[str] = None
        usable = True

        for idx in range(batch_size):
            ref = refs[idx]
            group_id = None
            if isinstance(ref, dict):
                for key in ("group_id", "sample_id", "id"):
                    value = ref.get(key)
                    if value is not None:
                        group_id = str(value)
                        break
            if group_id is None:
                usable = False
                break

            if current_group and group_id != current_group_id:
                groups.append(current_group)
                current_group = [idx]
            else:
                current_group.append(idx)
            current_group_id = group_id

        if usable and current_group:
            groups.append(current_group)
        if usable and sum(len(group) for group in groups) == batch_size and any(len(group) > 1 for group in groups):
            return groups

    chunk_size = max(1, int(n_samples_per_prompt))
    return [list(range(start, min(start + chunk_size, batch_size))) for start in range(0, batch_size, chunk_size)]


class MemePairwiseJudge(nn.Module):
    def __init__(
        self,
        base_model: Qwen2_5_VLForConditionalGeneration,
        processor,
        reward_prompt: str,
        max_new_tokens: int = 96,
        pair_batch_size: int = 4,
        n_samples_per_prompt: int = 1,
        max_pairs_per_group: int = 0,
        render_config: Optional[MemeRenderConfig] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.processor = processor
        self.reward_prompt = reward_prompt
        self.max_new_tokens = int(max_new_tokens)
        self.pair_batch_size = int(pair_batch_size)
        self.n_samples_per_prompt = int(n_samples_per_prompt)
        self.max_pairs_per_group = int(max_pairs_per_group)
        self.render_config = render_config or MemeRenderConfig()

    def _render_candidates(
        self,
        raw_images: Sequence[Any],
        references: Sequence[Any],
        prompt_and_outputs: Sequence[str],
    ) -> Tuple[List[Any], List[str]]:
        rendered_images: List[Any] = []
        extracted_texts: List[str] = []

        for raw_image, reference, prompt_and_output in zip(raw_images, references, prompt_and_outputs):
            extracted_boxes = extract_box_texts(prompt_and_output or "")
            extracted_texts.append("\\n".join(extracted_boxes) if extracted_boxes else "")

            image = get_first_image(raw_image)
            if image is None:
                rendered_images.append(None)
                continue

            detections = normalize_detections(reference if isinstance(reference, dict) else None)
            rendered_images.append(
                render_meme_image(
                    image=image,
                    texts=extracted_boxes,
                    detections=detections,
                    reference=reference if isinstance(reference, dict) else None,
                    config=self.render_config,
                )
            )

        return rendered_images, extracted_texts

    def _generate_pair_scores(self, pair_jobs: List[Dict[str, Any]], device: torch.device) -> List[Tuple[float, float]]:
        scores: List[Tuple[float, float]] = []
        if not pair_jobs:
            return scores

        step = max(1, self.pair_batch_size)
        for start in range(0, len(pair_jobs), step):
            batch_jobs = pair_jobs[start:start + step]
            messages = [job["message"] for job in batch_jobs]
            texts = [
                self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                for message in messages
            ]
            image_inputs = []
            for message in messages:
                processed = process_vision_info(message)
                if isinstance(processed, tuple):
                    image_inputs.append(processed[0])
                else:
                    image_inputs.append(processed)

            inputs = self.processor(
                text=texts,
                images=image_inputs,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            for key, value in list(inputs.items()):
                if torch.is_tensor(value):
                    inputs[key] = value.to(device)

            generation_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
            }
            if "pixel_values" in inputs:
                generation_kwargs["pixel_values"] = inputs["pixel_values"]
            if "image_grid_thw" in inputs:
                generation_kwargs["image_grid_thw"] = inputs["image_grid_thw"]

            generated = self.base_model.generate(**generation_kwargs)
            trimmed = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs["input_ids"], generated)]
            decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            scores.extend(parse_pair_judge_response(text) for text in decoded)

        return scores

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        del attention_mask, pixel_values, image_grid_thw, return_dict, output_attentions, output_hidden_states

        prompt_and_output = kwargs.get("prompt_and_output") or []
        raw_images = kwargs.get("raw_images") or []
        references = kwargs.get("references") or []

        batch_size = len(prompt_and_output)
        if input_ids is not None:
            device = input_ids.device
        else:
            device = _as_torch_device(get_current_device()) if torch.cuda.is_available() else torch.device("cpu")
        if batch_size == 0:
            return {"score": torch.zeros(0, dtype=torch.float32, device=device)}

        rendered_images, extracted_texts = self._render_candidates(raw_images, references, prompt_and_output)
        groups = group_meme_rollout_indices(
            references, batch_size=batch_size, n_samples_per_prompt=self.n_samples_per_prompt
        )

        pair_jobs: List[Dict[str, Any]] = []
        preferences: List[PairwisePreference] = []
        comparison_counts = [0 for _ in range(batch_size)]

        for group in groups:
            if len(group) < 2:
                continue
            for local_a, local_b in sample_group_pairs(len(group), max_pairs=self.max_pairs_per_group):
                global_a = group[local_a]
                global_b = group[local_b]
                if rendered_images[global_a] is None or rendered_images[global_b] is None:
                    continue

                reference = references[global_a] if global_a < len(references) else None
                fallback_prompt = prompt_and_output[global_a] if global_a < len(prompt_and_output) else None
                user_request = get_user_request(reference if isinstance(reference, dict) else None, fallback_prompt)
                pair_jobs.append({
                    "index_a": global_a,
                    "index_b": global_b,
                    "message": build_pairwise_judge_message(
                        reward_prompt=self.reward_prompt,
                        image_a=rendered_images[global_a],
                        image_b=rendered_images[global_b],
                        text_a=extracted_texts[global_a],
                        text_b=extracted_texts[global_b],
                        user_request=user_request,
                    ),
                })

        pair_scores = self._generate_pair_scores(pair_jobs, device=device)
        for job, (score_a, score_b) in zip(pair_jobs, pair_scores):
            index_a = job["index_a"]
            index_b = job["index_b"]
            preferences.append(
                PairwisePreference(
                    index_a=index_a,
                    index_b=index_b,
                    score_a=float(score_a),
                    score_b=float(score_b),
                )
            )
            comparison_counts[index_a] += 1
            comparison_counts[index_b] += 1

        pairwise_rewards = aggregate_pairwise_preferences(batch_size, preferences)
        for idx, count in enumerate(comparison_counts):
            if count == 0:
                pairwise_rewards[idx] = 0.5

        return {"score": torch.tensor(pairwise_rewards, dtype=torch.float32, device=device)}


def load_reward_models(
    reward_pretrain: str,
    strategy,
    use_engine: bool = False,
):
    if use_engine:
        raise NotImplementedError("Engine is not supported for the meme reward model")

    cfg = _parse_reward_config(reward_pretrain)
    _REWARD_STATE["model_reward_weight"] = float(cfg.get("model_reward_weight", 1.0))
    _REWARD_STATE["format_reward_weight"] = float(cfg.get("format_reward_weight", 0.1))

    pretrain_path = cfg["path"]
    reward_prompt = _load_reward_prompt(cfg.get("reward_prompt_path"))

    with strategy.init_model_context() as _:
        model_config = AutoConfig.from_pretrained(pretrain_path, trust_remote_code=True)
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrain_path,
            config=model_config,
            torch_dtype=torch.bfloat16,
            attn_implementation=cfg.get("attn_implementation", "flash_attention_2"),
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            pretrain_path,
            min_pixels=cfg.get("min_pixels", 256 * 28 * 28),
            max_pixels=cfg.get("max_pixels", 1280 * 28 * 28),
            trust_remote_code=True,
        )
        processor.tokenizer.padding_side = "left"

        model = MemePairwiseJudge(
            base_model=base_model,
            processor=processor,
            reward_prompt=reward_prompt,
            max_new_tokens=cfg.get("max_new_tokens", 96),
            pair_batch_size=cfg.get("pair_batch_size", 4),
            n_samples_per_prompt=cfg.get("n_samples_per_prompt", 1),
            max_pairs_per_group=cfg.get("max_pairs_per_group", 0),
            render_config=MemeRenderConfig(
                font_name=cfg.get("font_name", "DejaVuSans.ttf"),
                min_font_size=cfg.get("min_font_size", 14),
                max_font_size=cfg.get("max_font_size", 72),
                line_spacing=cfg.get("line_spacing", 4),
                outline_width=cfg.get("outline_width", 2),
                margin=cfg.get("margin", 6),
                default_padding_ratio=cfg.get("default_padding_ratio", 0.06),
            ),
        )
        model.eval()

    return [model], [processor.tokenizer], {_PAIRWISE_LABEL_KEY: 0}


def reward_fn(
    model_reward_list: List[torch.Tensor],
    labels: Sequence[str],
    queries: Sequence[str],
    refs: Sequence[Any],
    label_map: Optional[Dict[str, int]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    del labels, kwargs

    if model_reward_list:
        device = model_reward_list[0].device
        dtype = model_reward_list[0].dtype
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32

    batch_size = len(queries)
    model_reward = torch.zeros(batch_size, dtype=dtype, device=device)
    if model_reward_list:
        pairwise_idx = 0
        if label_map:
            pairwise_idx = label_map.get(_PAIRWISE_LABEL_KEY, 0)
        pairwise_idx = min(pairwise_idx, len(model_reward_list) - 1)
        model_reward = torch.as_tensor(model_reward_list[pairwise_idx], dtype=dtype, device=device)

    format_values = []
    for idx, query in enumerate(queries):
        reference = refs[idx] if refs is not None and idx < len(refs) else None
        expected_boxes = resolve_expected_box_count(reference if isinstance(reference, dict) else None)
        format_values.append(compute_meme_format_reward(query, expected_boxes=expected_boxes))

    format_reward = torch.tensor(format_values, dtype=dtype, device=device)
    final_reward = (
        _REWARD_STATE["model_reward_weight"] * model_reward + _REWARD_STATE["format_reward_weight"] * format_reward
    )
    metrics = {
        "model_reward": model_reward,
        "format_reward": format_reward,
        "rule_reward": final_reward,
    }
    return final_reward, metrics
