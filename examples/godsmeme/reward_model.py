import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor
from transformers.utils import cached_file

from keye_vl_utils import process_vision_info

from lightrft.utils import get_current_device

from meme_utils import (
    MemeRenderConfig,
    PairwisePreference,
    aggregate_pairwise_preferences,
    compute_meme_format_reward,
    extract_box_texts,
    get_first_image,
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
_DTYPE_MAP = {
    "auto": None,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


class _FSDPSafeEmbedding(nn.Module):
    """Keep embedding weights in the parent FSDP unit.

    Keye's vision tower reads ``position_embedding.weight`` directly in its
    interpolation path. When FSDP2 individually wraps those embedding modules,
    the weight can become a DTensor while sibling activations stay regular
    tensors, which triggers mixed Tensor/DTensor errors on addition.
    """
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.weight = embedding.weight
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.padding_idx = embedding.padding_idx

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return nn.functional.embedding(input_ids, self.weight, padding_idx=self.padding_idx)


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
    return "Which meme is funnier?"


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


def _resolve_torch_dtype(dtype_like: Any) -> Optional[torch.dtype]:
    if dtype_like is None or isinstance(dtype_like, torch.dtype):
        return dtype_like
    key = str(dtype_like).strip().lower()
    if key not in _DTYPE_MAP:
        raise ValueError(f"Unsupported torch dtype: {dtype_like}")
    return _DTYPE_MAP[key]


def _resolve_model_file(path_or_repo: str, filename: str) -> str:
    if os.path.isdir(path_or_repo):
        local_path = os.path.join(path_or_repo, filename)
        if os.path.exists(local_path):
            return local_path
    try:
        return cached_file(path_or_repo, filename)
    except Exception as exc:  # pragma: no cover - exercised only when weights are resolved remotely.
        raise FileNotFoundError(f"Could not find {filename} under {path_or_repo}") from exc


def _replace_module_if_embedding(parent: nn.Module, attr_name: str) -> bool:
    module = getattr(parent, attr_name, None)
    if not isinstance(module, nn.Embedding):
        return False

    setattr(parent, attr_name, _FSDPSafeEmbedding(module))
    return True


def _find_keye_visual_module(model: nn.Module) -> Optional[nn.Module]:
    for attr_name in ("visual", "vision_tower", "vision_model"):
        module = getattr(model, attr_name, None)
        if isinstance(module, nn.Module):
            return module

    inner_model = getattr(model, "model", None)
    if isinstance(inner_model, nn.Module):
        return _find_keye_visual_module(inner_model)

    return None


def _patch_keye_fsdp_compat(model: nn.Module) -> None:
    """Patch Keye vision modules that are fragile under FSDP2 DTensors."""
    visual_model = _find_keye_visual_module(model)
    if visual_model is None:
        return

    replaced_names: List[str] = []
    for module_name, module in visual_model.named_modules():
        if _replace_module_if_embedding(module, "position_embedding"):
            replaced_names.append(f"{module_name}.position_embedding" if module_name else "position_embedding")
        if _replace_module_if_embedding(module, "packing_position_embedding"):
            replaced_names.append(
                f"{module_name}.packing_position_embedding" if module_name else "packing_position_embedding"
            )
        if hasattr(module, "_attn_implementation"):
            module._attn_implementation = "eager"

    if hasattr(getattr(visual_model, "config", None), "_attn_implementation"):
        visual_model.config._attn_implementation = "eager"

    vision_config = getattr(getattr(model, "config", None), "vision_config", None)
    if hasattr(vision_config, "_attn_implementation"):
        vision_config._attn_implementation = "eager"

    if replaced_names:
        print("[MemePairwiseJudge] FSDP2-safe Keye vision patch:", ", ".join(replaced_names))
    print("[MemePairwiseJudge] Set Keye vision attention to 'eager' for FSDP2 compat")


def _unwrap_head_state_dict(state_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    nested_keys = ("state_dict", "model", "module", "classification_head", "classifier")
    current = state_dict
    while isinstance(current, dict):
        tensor_values = [value for value in current.values() if torch.is_tensor(value)]
        if tensor_values:
            break
        next_state = None
        for key in nested_keys:
            candidate = current.get(key)
            if isinstance(candidate, dict):
                next_state = candidate
                break
        if next_state is None:
            break
        current = next_state

    if not isinstance(current, dict):
        raise ValueError("classification_head.pt does not contain a supported state dict")

    tensor_state = {str(key): value for key, value in current.items() if torch.is_tensor(value)}
    if not tensor_state:
        raise ValueError("classification_head.pt does not contain tensor parameters")
    return tensor_state


def _strip_common_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not all(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {key[len(prefix):]: value for key, value in state_dict.items()}


def _normalize_head_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized = dict(state_dict)
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "model.", "classification_head.", "classifier.", "head."):
            stripped = _strip_common_prefix(normalized, prefix)
            if stripped is not normalized:
                normalized = stripped
                changed = True
                break
    return normalized


class _TwoLayerTanhHead(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.dense = nn.Linear(in_features, hidden_features)
        self.out_proj = nn.Linear(hidden_features, out_features)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.activation(self.dense(hidden_states)))


def _build_classification_head(state_dict: Dict[str, torch.Tensor]) -> nn.Module:
    state_dict = _normalize_head_state_dict(_unwrap_head_state_dict(state_dict))
    keys = set(state_dict.keys())

    if {"dense.weight", "dense.bias", "out_proj.weight", "out_proj.bias"}.issubset(keys):
        dense_weight = state_dict["dense.weight"]
        out_proj_weight = state_dict["out_proj.weight"]
        head = _TwoLayerTanhHead(
            in_features=dense_weight.shape[1],
            hidden_features=dense_weight.shape[0],
            out_features=out_proj_weight.shape[0],
        )
        head.load_state_dict({
            "dense.weight": dense_weight,
            "dense.bias": state_dict["dense.bias"],
            "out_proj.weight": out_proj_weight,
            "out_proj.bias": state_dict["out_proj.bias"],
        })
        return head

    if "weight" in state_dict and state_dict["weight"].ndim == 2:
        bias = state_dict.get("bias")
        head = nn.Linear(state_dict["weight"].shape[1], state_dict["weight"].shape[0], bias=bias is not None)
        payload = {"weight": state_dict["weight"]}
        if bias is not None:
            payload["bias"] = bias
        head.load_state_dict(payload, strict=False)
        return head

    for prefix in ("score", "out_proj", "summary"):
        weight_key = f"{prefix}.weight"
        if weight_key not in state_dict:
            continue
        bias_key = f"{prefix}.bias"
        bias = state_dict.get(bias_key)
        head = nn.Linear(state_dict[weight_key].shape[1], state_dict[weight_key].shape[0], bias=bias is not None)
        payload = {"weight": state_dict[weight_key]}
        if bias is not None:
            payload["bias"] = bias
        head.load_state_dict(payload, strict=False)
        return head

    raise ValueError(
        "Unsupported classification head format. Expected a simple linear head or a dense/out_proj head, "
        f"but found keys: {sorted(state_dict.keys())}"
    )


def _load_classification_head(path_or_file: str, dtype: Optional[torch.dtype] = None) -> nn.Module:
    head_state = torch.load(path_or_file, map_location="cpu")
    head = _build_classification_head(head_state)
    if dtype is not None:
        head = head.to(dtype=dtype)
    return head


def _pool_last_non_padding_token(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if attention_mask is None:
        return hidden_states[:, -1, :]

    attention_mask = attention_mask.to(device=hidden_states.device, dtype=torch.long)
    reverse_mask = torch.flip(attention_mask, dims=[-1])
    last_offsets = torch.argmax(reverse_mask, dim=-1)
    last_indices = attention_mask.size(-1) - 1 - last_offsets
    gather_index = last_indices.view(-1, 1, 1).expand(-1, 1, hidden_states.size(-1))
    return hidden_states.gather(dim=1, index=gather_index).squeeze(1)


def _pair_scores_from_logits(logits: torch.Tensor) -> List[Tuple[float, float]]:
    if logits.ndim == 1:
        logits = logits.unsqueeze(-1)

    if logits.size(-1) == 1:
        probs_a = torch.sigmoid(logits[:, 0].float())
        probs_b = 1.0 - probs_a
    else:
        probs = torch.softmax(logits[:, :2].float(), dim=-1)
        probs_a = probs[:, 0]
        probs_b = probs[:, 1]

    return list(zip(probs_a.detach().cpu().tolist(), probs_b.detach().cpu().tolist()))


def build_pairwise_judge_message(
    reward_prompt: str,
    image_a,
    image_b,
) -> List[Dict[str, Any]]:
    return [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": reward_prompt
            },
            {
                "type": "image",
                "image": image_a
            },
            {
                "type": "image",
                "image": image_b
            },
        ],
    }]


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
        base_model: nn.Module,
        processor,
        reward_head: nn.Module,
        reward_prompt: str,
        pair_batch_size: int = 4,
        n_samples_per_prompt: int = 1,
        max_pairs_per_group: int = 0,
        max_length: int = 4096,
        render_config: Optional[MemeRenderConfig] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.processor = processor
        self.reward_head = reward_head
        self.reward_prompt = reward_prompt
        self.pair_batch_size = int(pair_batch_size)
        self.n_samples_per_prompt = int(n_samples_per_prompt)
        self.max_pairs_per_group = int(max_pairs_per_group)
        self.max_length = int(max_length)
        self.render_config = render_config or MemeRenderConfig()

    def _render_candidates(
        self,
        raw_images: Sequence[Any],
        references: Sequence[Any],
        prompt_and_outputs: Sequence[str],
    ) -> List[Any]:
        rendered_images: List[Any] = []

        for raw_image, reference, prompt_and_output in zip(raw_images, references, prompt_and_outputs):
            extracted_boxes = extract_box_texts(prompt_and_output or "")

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

        return rendered_images

    def _score_pair_jobs(self, pair_jobs: List[Dict[str, Any]], device: torch.device) -> List[Tuple[float, float]]:
        scores: List[Tuple[float, float]] = []
        if not pair_jobs:
            return scores

        @torch.no_grad()
        def _run_batch(batch_jobs: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
            if process_vision_info is None:
                raise ImportError("keye-vl-utils is required for the GodsMeme reward model")
            messages = [job["message"] for job in batch_jobs]
            texts = [
                self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                for message in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages)

            processor_kwargs = {
                "text": texts,
                "padding": True,
                "truncation": True,
                "max_length": self.max_length,
                "return_tensors": "pt",
            }
            if image_inputs is not None:
                processor_kwargs["images"] = image_inputs
            if video_inputs is not None:
                processor_kwargs["videos"] = video_inputs

            inputs = self.processor(**processor_kwargs)
            for key, value in list(inputs.items()):
                if torch.is_tensor(value):
                    inputs[key] = value.to(device)

            outputs = self.base_model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states:
                last_hidden = hidden_states[-1]
            else:
                last_hidden = getattr(outputs, "last_hidden_state", None)
            if last_hidden is None:
                raise ValueError("Reward model must return hidden states for pairwise scoring.")

            pooled = _pool_last_non_padding_token(last_hidden, inputs.get("attention_mask"))
            logits = self.reward_head(pooled)
            return _pair_scores_from_logits(logits)

        step = max(1, self.pair_batch_size)
        for start in range(0, len(pair_jobs), step):
            batch_jobs = pair_jobs[start:start + step]
            try:
                scores.extend(_run_batch(batch_jobs))
            except IndexError as exc:
                # Some multimodal processors do not align multiple two-image prompts
                # correctly in one batch. Fall back to one pair per forward pass.
                if len(batch_jobs) == 1 or "image_grid_thw" not in str(exc):
                    raise
                for job in batch_jobs:
                    scores.extend(_run_batch([job]))

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

        rendered_images = self._render_candidates(raw_images, references, prompt_and_output)
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

                pair_jobs.append({
                    "index_a": global_a,
                    "index_b": global_b,
                    "message": build_pairwise_judge_message(
                        reward_prompt=self.reward_prompt,
                        image_a=rendered_images[global_a],
                        image_b=rendered_images[global_b],
                    ),
                })

        pair_scores = self._score_pair_jobs(pair_jobs, device=device)
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

    reward_model_path = cfg["path"]
    reward_prompt = _load_reward_prompt(cfg.get("reward_prompt_path"))
    torch_dtype = _resolve_torch_dtype(cfg.get("torch_dtype", "float16"))
    classification_head_path = cfg.get("classification_head_path"
                                       ) or _resolve_model_file(reward_model_path, "classification_head.pt")

    with strategy.init_model_context() as _:
        base_model = AutoModel.from_pretrained(
            reward_model_path,
            torch_dtype=torch_dtype,
            attn_implementation=cfg.get("attn_implementation", "flash_attention_2"),
            trust_remote_code=cfg.get("trust_remote_code", True),
        )
        if getattr(getattr(strategy, "args", None), "fsdp", False):
            _patch_keye_fsdp_compat(base_model)

        processor = AutoProcessor.from_pretrained(
            reward_model_path,
            min_pixels=cfg.get("min_pixels", 256 * 28 * 28),
            max_pixels=cfg.get("max_pixels", 1280 * 28 * 28),
            trust_remote_code=cfg.get("trust_remote_code", True),
        )
        if hasattr(processor, "tokenizer"):
            processor.tokenizer.padding_side = "left"

        reward_head = _load_classification_head(classification_head_path, dtype=torch_dtype)
        model = MemePairwiseJudge(
            base_model=base_model,
            processor=processor,
            reward_head=reward_head,
            reward_prompt=reward_prompt,
            pair_batch_size=cfg.get("pair_batch_size", 1),
            n_samples_per_prompt=cfg.get("n_samples_per_prompt", 1),
            max_pairs_per_group=cfg.get("max_pairs_per_group", 0),
            max_length=cfg.get("max_length", cfg.get("cutoff_len", 4096)),
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
