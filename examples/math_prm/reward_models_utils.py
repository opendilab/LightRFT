"""Math-only reward loading and aggregation utilities for URSA-MATH Stage 3."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

from lightrft.models.monkey_patch.hf_generate_patch import apply_monkey_patch_to_generation_mixin
from lightrft.utils import get_current_device

from reward_models import MathPRMReward


class RewardModelType(str, Enum):
    """Supported reward model types for the math_prm example."""

    MATH_PRM = "math_prm"


@dataclass
class RewardModelConfig:
    """Configuration for one reward model instance."""

    rtype: RewardModelType
    path: str
    use_engine: bool = False


RawRewardInput = Union[str, Dict[str, str], List[Dict[str, str]], None]
_BUILDERS: Dict[RewardModelType, Callable[..., Tuple[Any, Any]]] = {}


def register_builder(rtype: RewardModelType) -> Callable:
    def deco(fn: Callable) -> Callable:
        _BUILDERS[rtype] = fn
        return fn

    return deco


def _guess_rtype_from_path(path: str) -> RewardModelType:
    lowered = path.lower()
    if any(keyword in lowered for keyword in ("ursa", "prm", "math-rm", "step-reward", "process-reward")):
        return RewardModelType.MATH_PRM
    return RewardModelType.MATH_PRM


def parse_reward_pretrain(
    raw: RawRewardInput,
    *,
    global_use_engine: bool,
) -> Tuple[List[RewardModelConfig], Dict[str, int]]:
    """Parse reward model config while keeping the old flexible input shapes."""

    if raw is None:
        return [], {}

    pair_list: List[Tuple[str, str, Optional[bool]]] = []
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return [], {}
        if text.startswith("{") and text.endswith("}"):
            obj = json.loads(text)
            pair_list = [(key, value, None) for key, value in obj.items()]
        else:
            for segment in re.split(r"\s*,\s*", text):
                if not segment:
                    continue
                if ":" in segment:
                    key, value = segment.split(":", 1)
                    pair_list.append((key.strip(), value.strip(), None))
                else:
                    pair_list.append(("?", segment.strip(), None))
    elif isinstance(raw, dict):
        pair_list = [(key, value, None) for key, value in raw.items()]
    elif isinstance(raw, list):
        for item in raw:
            pair_list.append((item["type"], item["path"], item.get("engine")))
    else:
        raise TypeError("Unsupported --reward_pretrain format")

    cfgs: List[RewardModelConfig] = []
    for key, path, flag in pair_list:
        use_engine = global_use_engine
        if "?engine=" in path:
            path, qs = path.split("?engine=", 1)
            use_engine = qs.lower() in {"1", "true", "yes"}
        if flag is not None:
            use_engine = bool(flag)
        rtype = _guess_rtype_from_path(path) if key == "?" else RewardModelType(key)
        cfgs.append(RewardModelConfig(rtype=rtype, path=path, use_engine=use_engine))

    label_map = {cfg.rtype.value: index for index, cfg in enumerate(cfgs)}
    return cfgs, label_map


def _load_ursa_prm_model(pretrain_path: str, device: torch.device | int) -> Tuple[Any, Any]:
    from ursa_model import UrsaForTokenClassification, UrsaProcessor

    processor = UrsaProcessor.from_pretrained(pretrain_path)
    model = UrsaForTokenClassification.from_pretrained(
        pretrain_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()
    return model, processor


def _load_engine(pretrain_path: str, device: torch.device | int) -> Tuple[Any, Any]:
    raise RuntimeError(
        "The math_prm example no longer supports external reward-model engines. "
        "URSA-RM is loaded through the local HF path instead."
    )


def _shared_base_key(cfg: RewardModelConfig) -> Optional[Tuple[str, str]]:
    if cfg.rtype != RewardModelType.MATH_PRM:
        return None
    return (cfg.path, cfg.rtype.value)


def _load_shared_base(cfg: RewardModelConfig) -> Tuple[Any, Any]:
    return _load_ursa_prm_model(cfg.path, get_current_device())


@register_builder(RewardModelType.MATH_PRM)
def build_math_prm(
    cfg: RewardModelConfig,
    strategy: Any,
    base: Optional[Tuple[Any, Any]] = None,
) -> Tuple[MathPRMReward, Any]:
    if cfg.use_engine:
        strategy.print(
            "[build_math_prm] Engine mode is not supported for URSA-RM. "
            "Falling back to direct HF loading."
        )

    if base is None:
        base_model, processor = _load_ursa_prm_model(cfg.path, get_current_device())
    else:
        base_model, processor = base

    reward_model = MathPRMReward(
        base_model=base_model,
        processor=processor,
        aggregation="min",
    )
    reward_model.eval()
    return reward_model, processor.tokenizer


def load_reward_models(
    raw_reward_pretrain: RawRewardInput,
    strategy: Any,
    use_engine: bool = False,
) -> Tuple[List[Any], List[Any], Dict[str, int]]:
    apply_monkey_patch_to_generation_mixin()
    cfgs, label_map = parse_reward_pretrain(raw_reward_pretrain, global_use_engine=use_engine)

    reward_models: List[Any] = []
    reward_tokenizers: List[Any] = []
    shared_bases: Dict[Tuple[str, str], Tuple[Any, Any]] = {}

    for cfg in cfgs:
        cache_key = _shared_base_key(cfg)
        if cache_key is not None and cache_key not in shared_bases:
            shared_bases[cache_key] = _load_shared_base(cfg)
            strategy.print(f"Init reward model base {cfg.path} (engine={cfg.use_engine}, type={cfg.rtype})")

    for cfg in cfgs:
        if cfg.rtype not in _BUILDERS:
            raise RuntimeError(f"No builder registered for {cfg.rtype}")
        strategy.print(f"Loading {cfg.rtype} from {cfg.path} (engine={cfg.use_engine})")
        with strategy.init_model_context() as _:
            reward_model, tokenizer = _BUILDERS[cfg.rtype](
                cfg,
                strategy,
                base=shared_bases.get(_shared_base_key(cfg)),
            )
        reward_models.append(reward_model)
        reward_tokenizers.append(tokenizer)
        strategy.print(f"Loaded {cfg.rtype}")

    return reward_models, reward_tokenizers, label_map


def math_prm_format_reward_fn(sol: str) -> float:
    """Diagnostic-only check for the required Stage 3 ``Step N`` / ``†Answer`` format."""
    if not isinstance(sol, str):
        return 0.0
    step_matches = re.findall(r"(?m)^Step\s+\d+\s*:\s*\S", sol)
    answer_matches = re.findall(r"(?m)^†Answer:\s*\S", sol)
    non_empty_lines = [line.strip() for line in sol.splitlines() if line.strip()]
    if not step_matches or len(answer_matches) != 1 or not non_empty_lines:
        return 0.0
    return 1.0 if non_empty_lines[-1].startswith("†Answer:") else 0.0


def format_reward_fn(sol: str) -> float:
    """Compatibility alias kept for older callers inside this example directory."""
    return math_prm_format_reward_fn(sol)


def rule_reward_fn(sol: str, gt: str) -> float:
    """Rule-only baseline using the same controlled final-answer extraction as PS-GRPO."""
    if not gt:
        return 0.0
    answer_eval = MathPRMReward._evaluate_answer_alignment(sol, gt)
    return 1.0 if answer_eval["outcome_correct"] else 0.0


RECIPE: Dict[str, List[Tuple[str, Optional[str], float]]] = {
    "math_prm": [("model", "math_prm", 1.0)],
    "math_psgrpo": [("model", "math_prm", 1.0)],
    "math_prm_combined": [("model", "math_prm", 1.0), ("rule", None, 0.5)],
    "math_rule": [("rule", None, 1.0)],
}


NO_GLOBAL_FORMAT_REWARD_LABELS = {
    "math_prm",
    "math_psgrpo",
    "math_prm_combined",
    "math_rule",
}


def mix_rewards(
    labels: Sequence[str],
    model_scores: torch.Tensor,
    label_map: Dict[str, int],
    solution_strs: Sequence[str],
    refs: Sequence[str],
    model_reward_metrics_list: Optional[List[Optional[Dict[str, torch.Tensor]]]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if model_scores.numel() > 0:
        device = model_scores.device
    elif model_reward_metrics_list:
        first_metric_tensor = next(
            (
                tensor
                for metrics in model_reward_metrics_list
                if metrics
                for tensor in metrics.values()
                if isinstance(tensor, torch.Tensor)
            ),
            None,
        )
        device = first_metric_tensor.device if first_metric_tensor is not None else torch.device("cpu")
    else:
        device = torch.device("cpu")

    n_model = int(model_scores.shape[0])
    batch_size = len(labels)
    if model_scores.ndim != 2:
        raise ValueError(f"model_scores must have shape (n_model, B), got {tuple(model_scores.shape)!r}")
    if model_scores.shape[1] != batch_size:
        raise AssertionError("model_scores second dimension must equal batch size")

    final_reward = torch.zeros(batch_size, dtype=torch.float32, device=device)
    metrics_dict: Dict[str, torch.Tensor] = {
        "format_reward": torch.zeros(batch_size, dtype=torch.float32, device=device),
        "accuracy_reward": torch.zeros(batch_size, dtype=torch.float32, device=device),
        "model_reward": torch.zeros(batch_size, dtype=torch.float32, device=device),
        "rule_reward": torch.zeros(batch_size, dtype=torch.float32, device=device),
        "outcome_correct": torch.zeros(batch_size, dtype=torch.float32, device=device),
        "max_relative_drop": torch.zeros(batch_size, dtype=torch.float32, device=device),
        "has_drop_moment": torch.zeros(batch_size, dtype=torch.float32, device=device),
        "final_reward": torch.zeros(batch_size, dtype=torch.float32, device=device),
    }

    def ensure_metric_key(metric_name: str) -> None:
        if metric_name not in metrics_dict:
            metrics_dict[metric_name] = torch.zeros(batch_size, dtype=torch.float32, device=device)

    def get_model_reward(key: str, index: int) -> float:
        if key not in label_map:
            print(f"Model reward <{key}> not loaded, using 1.0 as fallback")
            return 1.0
        model_index = label_map[key]
        if model_index >= n_model:
            print(f"Model reward <{key}> index {model_index} out of bounds, using 1.0 as fallback")
            return 1.0
        return float(model_scores[model_index, index].item())

    def get_model_metrics(key: str, index: int) -> Dict[str, float]:
        if not model_reward_metrics_list or key not in label_map:
            return {}
        model_index = label_map[key]
        if model_index >= len(model_reward_metrics_list):
            return {}
        metrics = model_reward_metrics_list[model_index]
        if not metrics:
            return {}

        sample_metrics: Dict[str, float] = {}
        for metric_name, tensor_value in metrics.items():
            if not isinstance(tensor_value, torch.Tensor):
                continue
            flat_tensor = tensor_value.reshape(-1)
            if flat_tensor.numel() <= index:
                continue
            sample_metrics[metric_name] = float(flat_tensor[index].item())
        return sample_metrics

    for index, label in enumerate(labels):
        solution = solution_strs[index]
        reference = refs[index] if index < len(refs) else ""
        format_metric = math_prm_format_reward_fn(solution)
        metrics_dict["format_reward"][index] = format_metric
        reward_value = 0.0 if label in NO_GLOBAL_FORMAT_REWARD_LABELS else format_metric

        recipe = RECIPE.get(label)
        if recipe is None:
            print(f"label <{label}> not registered in RECIPE, returning 0.0 reward")
            recipe = []

        for reward_type, key, weight in recipe:
            if reward_type == "model":
                model_reward = weight * get_model_reward(key, index)
                reward_value += model_reward
                metrics_dict["model_reward"][index] += model_reward
                for metric_name, metric_value in get_model_metrics(key, index).items():
                    ensure_metric_key(metric_name)
                    if metric_name == "final_reward":
                        continue
                    metrics_dict[metric_name][index] = metric_value
            elif reward_type == "rule":
                rule_reward = weight * rule_reward_fn(solution, reference)
                reward_value += rule_reward
                metrics_dict["rule_reward"][index] += rule_reward
                metrics_dict["accuracy_reward"][index] = rule_reward
            else:
                print(f"Unknown component type {reward_type}, ignoring")

        final_reward[index] = reward_value
        metrics_dict["final_reward"][index] = reward_value

    return final_reward, metrics_dict


def reward_fn(
    model_reward_list: List[torch.Tensor],
    model_reward_metrics_list: Optional[List[Optional[Dict[str, torch.Tensor]]]],
    labels: Sequence[str],
    queries: Sequence[str],
    refs: Sequence[str],
    label_map: Dict[str, int],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if model_reward_list:
        model_scores = torch.stack(model_reward_list)
    else:
        model_scores = torch.zeros((0, len(labels)), dtype=torch.float32, device="cpu")

    return mix_rewards(
        labels=labels,
        model_scores=model_scores,
        label_map=label_map,
        solution_strs=queries,
        refs=refs,
        model_reward_metrics_list=model_reward_metrics_list,
    )
