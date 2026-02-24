"""
Reward Models Utility Module

This module provides utility functions for loading, configuring, and managing reward models.
Supports multiple reward model types and flexible configuration parsing.

Main Features:
    - Reward model configuration parsing from various formats (JSON, CSV, dict, list)
    - Model loading for HuggingFace and SGLang engine backends
    - Builder pattern for different reward model types
    - Reward score mixing and computation
    - Rule-based reward functions

Supported Reward Types:
    - Knowledge: Factual accuracy evaluation
    - Safety: Safety and risk assessment
    - Value: Value alignment evaluation
    - General: General quality scoring
    - Normal: Normal conversation quality

Dependencies:
    - reward_models: Core reward model implementations
    - lightrft: Model loading and inference utilities
    - transformers: HuggingFace model support
"""
from __future__ import annotations

import re
import os
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence

import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from lightrft.models.monkey_patch.hf_generate_patch import (
    apply_monkey_patch_to_generation_mixin,
)
from lightrft.strategy.sglang_utils import get_sglang_engine
from lightrft.utils import get_current_device

# ============================================================================
# Optional Dependencies
# ============================================================================

try:
    # Attempt to import instruction following reward function
    # This function is part of the 'if_reward' library for deepseek model training
    from if_reward_fn import if_reward_fn
except ImportError:
    # If import fails, warn user and set to None
    print(
        "Error: The 'if_reward' library is not installed. "
        "This is required to support the instruction following reward function."
    )
    print(
        "Please install it directly from GitHub using: "
        "pip install git+https://github.com/opendilab/if_reward.git"
    )
    print(
        "Note: This reward function is currently utilized for the training of deepseek models."
    )
    if_reward_fn = None

from reward_models import (
    Qwen2VLRewardModelVauAI,
    Qwen2VLRewardModelSafety,
    Qwen2VLRewardModelKnowledge,
    Qwen2VLRewardModelGeneral,
    Qwen2VLRewardModelNormal,
)

# ============================================================================
# Configuration Classes
# ============================================================================

class RewardModelType(str, Enum):
    """Enumeration of supported reward model types."""
    KNOWLEDGE = "knowledge"
    SAFETY    = "safety"
    VALUE     = "value"
    GENERAL   = "general"
    NORMAL    = "normal"


@dataclass
class RewardModelConfig:
    """
    Configuration for a single reward model.

    :param rtype: Reward model type (e.g., RewardModelType.VALUE)
    :type rtype: RewardModelType
    :param path: Model directory path or HuggingFace model name
    :type path: str
    :param use_engine: Whether to use SGLang engine instead of HuggingFace. Default to False
    :type use_engine: bool
    """
    rtype: RewardModelType
    path : str
    use_engine: bool = False


# ============================================================================
# Model Builder Registry
# ============================================================================

_BUILDERS: Dict[RewardModelType, Callable] = {}

def register_builder(rtype: RewardModelType) -> Callable:
    """
    Decorator to register a builder function for a specific reward model type.

    Usage:
        @register_builder(RewardModelType.VALUE)
        def build_value(cfg, strategy):
            ...

    :param rtype: Reward model type to register builder for
    :type rtype: RewardModelType
    :return: Decorator function
    :rtype: Callable
    """
    def deco(fn: Callable) -> Callable:
        _BUILDERS[rtype] = fn
        return fn
    return deco


RawRewardInput = Union[str, Dict[str, str], List[Dict[str, str]], None]


# ============================================================================
# Configuration Parsing
# ============================================================================

def _guess_rtype_from_path(path: str) -> RewardModelType:
    """
    Infer reward model type from path string.

    :param path: Model path or name
    :type path: str
    :return: Inferred reward type
    :rtype: RewardModelType
    """
    p = path.lower()
    if "safety"   in p: return RewardModelType.SAFETY
    if "value"    in p or "vauai" in p: return RewardModelType.VALUE
    if "knowledge" in p or "qwen2.5-vl-72b" in p: return RewardModelType.KNOWLEDGE
    if "normal"   in p: return RewardModelType.NORMAL
    return RewardModelType.GENERAL

def parse_reward_pretrain(
    raw: RawRewardInput,
    *,
    global_use_engine: bool
) -> Tuple[List[RewardModelConfig], Dict[str, int]]:
    """
    Parse reward model configuration from various input formats.

    Supported formats:
        1. JSON: '{"knowledge":"/k", "value":"/v"}'
        2. CSV: 'knowledge:/k,value:/v'
        3. Path list: '/k,/v' (rtype auto-guessed)
        4. Dict/List: {'type':'value','path':'/v'} or [{'type':'value','path':'/v'}]

    Extra feature: Append ?engine=true to path to override global engine setting
    Example: 'knowledge:/path/to/model?engine=true'

    :param raw: Raw configuration input (string, dict, list, or None)
    :type raw: RawRewardInput
    :param global_use_engine: Global flag for whether to use engine mode
    :type global_use_engine: bool
    :return: Tuple of (cfgs, label_map) where cfgs is a list of RewardModelConfig objects
             and label_map is a dict mapping reward type to index {str: int}
    :rtype: Tuple[List[RewardModelConfig], Dict[str, int]]
    :raises TypeError: If raw input format is not supported

    Note:
        If RewardModelType.GENERAL is not present, it will be automatically added to label_map
    """
    if raw is None: raw = ""

    # ---------- 1. Convert string to unified list[(key,path,flag)] ----------
    pair_list: List[Tuple[str, str, Optional[bool]]] = []
    if isinstance(raw, str):
        s = raw.strip().lstrip("{").rstrip("}")
        # ① JSON
        if raw.strip().startswith("{") and raw.strip().endswith("}"):
            try:
                obj = json.loads(raw)
                pair_list = [(k, v, None) for k, v in obj.items()]
            except json.JSONDecodeError:
                pass
        if not pair_list:
            # ② kv/comma-separated string
            for seg in re.split(r"\s*,\s*", s):
                if not seg: continue
                if ":" in seg:
                    k, v = seg.split(":", 1)
                    pair_list.append((k.strip(), v.strip(), None))
                else:       # pure path
                    pair_list.append(("?", seg.strip(), None))
    elif isinstance(raw, dict):
        pair_list = [(k, v, None) for k, v in raw.items()]
    elif isinstance(raw, list):
        for d in raw:
            pair_list.append((d["type"], d["path"], d.get("engine")))
    else:
        raise TypeError("Unsupported --reward_pretrain format")

    # ---------- 2. Generate cfg list ----------
    cfgs: List[RewardModelConfig] = []
    for key, path, flag in pair_list:
        # Parse path?engine=true/false
        use_engine = global_use_engine
        if "?engine=" in path:
            path, qs = path.split("?engine=", 1)
            use_engine = qs.lower() in ("1", "true", "yes")
        if flag is not None:
            use_engine = flag
        rtype = _guess_rtype_from_path(path) if key == "?" else RewardModelType(key)
        cfgs.append(RewardModelConfig(rtype, path, use_engine))

    # Ensure label_map order is stable and contains general
    uniq: List[RewardModelType] = []
    for c in cfgs:
        if c.rtype not in uniq: uniq.append(c.rtype)
    if RewardModelType.GENERAL not in uniq:
        uniq.append(RewardModelType.GENERAL)
    label_map = {rt.value: i for i, rt in enumerate(uniq)}
    return cfgs, label_map


# ============================================================================
# Model Loading Functions
# ============================================================================

def _load_hf_model(
    pretrain_path: str,
    device: torch.device
) -> Tuple[Qwen2_5_VLForConditionalGeneration, Any]:
    """
    Load HuggingFace model and processor.

    :param pretrain_path: Model path or HuggingFace model name
    :type pretrain_path: str
    :param device: Target device
    :type device: torch.device
    :return: Tuple of (base_model, processor)
    :rtype: Tuple[Qwen2_5_VLForConditionalGeneration, Any]
    """
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        pretrain_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(
        pretrain_path, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28
    )
    processor.tokenizer.padding_side = "left"
    return base, processor


def _load_engine(
    pretrain_path: str,
    device: torch.device
) -> Tuple[Any, Any]:
    """
    Load SGLang engine and processor.

    Automatically determines tensor parallelism size based on reward model type:
        - value: 7B model → tp_size = 2
        - safety/safe: 72B model → tp_size = 8
        - knowledge/normal/general: 72B models → tp_size = 8

    :param pretrain_path: Model path or HuggingFace model name
    :type pretrain_path: str
    :param device: Target device
    :type device: torch.device
    :return: Tuple of (engine, processor)
    :rtype: Tuple[Any, Any]

    Note:
        Engine is set to sleep mode after loading to save memory
    """
    # TODO: more adaptive implementation
    # Determine tp_size based on model name in path
    if "value" in pretrain_path:
        # value-orm is 7B
        tp_size = 2
    elif ("safety" in pretrain_path) or ("safe" in pretrain_path):
        # safety-orm is 72B
        tp_size = 8
    else:
        # knowledge-orm, normal, general are all 72B
        tp_size = 8

    print(f"[reward_models_utils] Loading engine from {pretrain_path} with tp_size={tp_size}")

    engine = get_sglang_engine(
        pretrain_path,
        engine_mem_util=0.4,  # Increased from 0.2 to avoid CUDA graph buffer allocation failure
        # engine_mem_util=0.3,  # Increased from 0.2 to avoid CUDA graph buffer allocation failure
        tp_size=tp_size,
        skip_tokenizer_init=False,
        disable_cuda_graph=True, # only for deepseek, TODO: why deepseek pipeline (examples/safework_t1/run_grpo_svki_fsdp_deepseek.sh) need this?
    )

    print(f"[reward_models_utils] Loaded engine from {pretrain_path} with tp_size={tp_size}")


    engine.sleep()  # Sleep to save memory

    processor = AutoProcessor.from_pretrained(
        pretrain_path, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28
    )
    processor.tokenizer.padding_side = "left"
    return engine, processor


# ============================================================================
# Model Builders for Each Reward Type
# ============================================================================

@register_builder(RewardModelType.VALUE)
def build_value(
    cfg: RewardModelConfig,
    strategy: Any,
    base: Optional[Tuple[Any, Any]] = None
) -> Tuple[Qwen2VLRewardModelVauAI, Any]:
    """
    Build Value Alignment reward model.

    :param cfg: Reward model configuration
    :type cfg: RewardModelConfig
    :param strategy: Training strategy instance
    :type strategy: Any
    :param base: Optional pre-loaded (engine, processor) tuple for sharing
    :type base: Optional[Tuple[Any, Any]]
    :return: Tuple of (model, tokenizer)
    :rtype: Tuple[Qwen2VLRewardModelVauAI, Any]
    """
    if cfg.use_engine:
        if base:
            engine, proc = base
        else:
            engine, proc = _load_engine(cfg.path, get_current_device())
        model = Qwen2VLRewardModelVauAI(
            base_model=engine,
            tokenizer=proc.tokenizer,
            processor=proc,
            text_only=strategy.args.text_only,
            output_mode="hard",
        )
        return model, proc.tokenizer
    else:
        base, proc = _load_hf_model(cfg.path, get_current_device())
        model = Qwen2VLRewardModelVauAI(
            base_model=base,
            tokenizer=proc.tokenizer,
            processor=proc,
            text_only=strategy.args.text_only,
            output_mode="hard",
        )
        model.eval()
        return model, proc.tokenizer


@register_builder(RewardModelType.SAFETY)
def build_safety(
    cfg: RewardModelConfig,
    strategy: Any,
    base: Optional[Tuple[Any, Any]] = None
) -> Tuple[Qwen2VLRewardModelSafety, Any]:
    """
    Build Safety reward model.

    :param cfg: Reward model configuration
    :type cfg: RewardModelConfig
    :param strategy: Training strategy instance
    :type strategy: Any
    :param base: Optional pre-loaded (engine, processor) tuple for sharing
    :type base: Optional[Tuple[Any, Any]]
    :return: Tuple of (model, tokenizer)
    :rtype: Tuple[Qwen2VLRewardModelSafety, Any]
    """
    if cfg.use_engine:
        if base:
            engine, proc = base
        else:
            engine, proc = _load_engine(cfg.path, get_current_device())
        model = Qwen2VLRewardModelSafety(engine, proc.tokenizer, proc, text_only=strategy.args.text_only)
        return model, proc.tokenizer
    else:
        base, proc = _load_hf_model(cfg.path, get_current_device())
        model = Qwen2VLRewardModelSafety(base, proc.tokenizer, proc, text_only=strategy.args.text_only)
        model.eval()
        return model, proc.tokenizer


@register_builder(RewardModelType.KNOWLEDGE)
def build_knowledge(
    cfg: RewardModelConfig,
    strategy: Any,
    base: Optional[Tuple[Any, Any]] = None
) -> Tuple[Qwen2VLRewardModelKnowledge, Any]:
    """
    Build Knowledge reward model.

    :param cfg: Reward model configuration
    :type cfg: RewardModelConfig
    :param strategy: Training strategy instance
    :type strategy: Any
    :param base: Optional shared base model (engine, processor) tuple. Default to None
    :type base: Optional[Tuple[Any, Any]]
    :return: Tuple of (model, tokenizer)
    :rtype: Tuple[Qwen2VLRewardModelKnowledge, Any]
    """
    if cfg.use_engine:
        if base:
            engine, proc = base
        else:
            engine, proc = _load_engine(cfg.path, get_current_device())
        model = Qwen2VLRewardModelKnowledge(engine, proc.tokenizer, proc, text_only=strategy.args.text_only)
        return model, proc.tokenizer
    else:
        base_model, proc = _load_hf_model(cfg.path, get_current_device())
        model = Qwen2VLRewardModelKnowledge(base_model, proc.tokenizer, proc, text_only=strategy.args.text_only)
        model.eval()
        return model, proc.tokenizer


@register_builder(RewardModelType.GENERAL)
def build_general(
    cfg: RewardModelConfig,
    strategy: Any,
    base: Optional[Tuple[Any, Any]] = None
) -> Tuple[Qwen2VLRewardModelGeneral, Any]:
    """
    Build General quality reward model.

    :param cfg: Reward model configuration
    :type cfg: RewardModelConfig
    :param strategy: Training strategy instance
    :type strategy: Any
    :param base: Optional shared base model (engine, processor) tuple. Default to None
    :type base: Optional[Tuple[Any, Any]]
    :return: Tuple of (model, tokenizer)
    :rtype: Tuple[Qwen2VLRewardModelGeneral, Any]
    """
    if cfg.use_engine:
        if base:
            engine, proc = base
        else:
            engine, proc = _load_engine(cfg.path, get_current_device())
        model = Qwen2VLRewardModelGeneral(engine, proc.tokenizer, proc, text_only=strategy.args.text_only)
        return model, proc.tokenizer
    else:
        base_model, proc = _load_hf_model(cfg.path, get_current_device())
        model = Qwen2VLRewardModelGeneral(base_model, proc.tokenizer, proc, text_only=strategy.args.text_only)
        model.eval()
        return model, proc.tokenizer


@register_builder(RewardModelType.NORMAL)
def build_normal(
    cfg: RewardModelConfig,
    strategy: Any,
    base: Optional[Tuple[Any, Any]] = None
) -> Tuple[Qwen2VLRewardModelNormal, Any]:
    """
    Build Normal conversation quality reward model.

    :param cfg: Reward model configuration
    :type cfg: RewardModelConfig
    :param strategy: Training strategy instance
    :type strategy: Any
    :param base: Optional shared base model (engine, processor) tuple. Default to None
    :type base: Optional[Tuple[Any, Any]]
    :return: Tuple of (model, tokenizer)
    :rtype: Tuple[Qwen2VLRewardModelNormal, Any]
    """
    if cfg.use_engine:
        if base:
            engine, proc = base
        else:
            engine, proc = _load_engine(cfg.path, get_current_device())
        model = Qwen2VLRewardModelNormal(engine, proc.tokenizer, proc, text_only=strategy.args.text_only)
        return model, proc.tokenizer
    else:
        base_model, proc = _load_hf_model(cfg.path, get_current_device())
        model = Qwen2VLRewardModelNormal(base_model, proc.tokenizer, proc, text_only=strategy.args.text_only)
        model.eval()
        return model, proc.tokenizer

# ============================================================================
# Main Initialization Entry Point
# ============================================================================

def load_reward_models(
    raw_reward_pretrain: RawRewardInput,
    strategy: Any,
    use_engine: bool = False,
) -> Tuple[List[Any], List[Any], Dict[str, int]]:
    """
    Load and initialize all reward models from configuration.

    This is the main entry point for loading reward models. It handles:
        - Configuration parsing
        - Base model sharing (to save memory)
        - Model initialization with proper context
        - Monkey patching for HuggingFace generation

    :param raw_reward_pretrain: Raw configuration (see parse_reward_pretrain)
    :type raw_reward_pretrain: RawRewardInput
    :param strategy: Training strategy instance
    :type strategy: Any
    :param use_engine: Global flag for using SGLang engine. Default to False
    :type use_engine: bool
    :return: Tuple of (reward_models, reward_tokenizers, label_map) where
             reward_models is a list of initialized reward model instances,
             reward_tokenizers is a list of corresponding tokenizers,
             and label_map is a dict mapping reward type to index
    :rtype: Tuple[List[Any], List[Any], Dict[str, int]]

    Note:
        Models sharing the same base path will reuse the same loaded base model
        to reduce memory footprint.
    """
    apply_monkey_patch_to_generation_mixin()

    cfgs, label_map = parse_reward_pretrain(
        raw_reward_pretrain, global_use_engine=use_engine
    )

    rms: List[Any] = []
    toks: List[Any] = []

    # Share base models across reward models to save memory
    # Since some reward models can share the same base model, we only load it once
    shared_bases: Dict[str, Tuple[Any, Any]] = {}
    shared_count: Dict[str, int] = {}
    for cfg in cfgs:
        if cfg.path not in shared_count:
            shared_count[cfg.path] = 1
        else:
            shared_count[cfg.path] += 1

        if shared_count[cfg.path] == 1:
            shared_bases[cfg.path] = _load_engine(cfg.path, get_current_device())
            strategy.print(f"Init reward model {cfg.path} (engine={cfg.use_engine})")
        else:
            strategy.print(f"Use shared base model {cfg.path}")

    for cfg in cfgs:
        if cfg.rtype not in _BUILDERS:
            raise RuntimeError(f"No builder for {cfg.rtype}")
        strategy.print(f"Loading {cfg.rtype} from {cfg.path} (engine={cfg.use_engine})")

        # Initialize model with proper context (supports FSDP/meta device init)
        with strategy.init_model_context() as _:
            # All reward types now support shared base models
            rm, tok = _BUILDERS[cfg.rtype](cfg, strategy, base=shared_bases.get(cfg.path))
        
        rms.append(rm)
        toks.append(tok)
        strategy.print(f"Loaded {cfg.rtype}")

    return rms, toks, label_map



# ============================================================================
# Reward Functions
# ============================================================================

def format_reward_fn(sol: str) -> float:
    """
    Check if solution matches format: <think> ... </think> + non-empty content.

    :param sol: Solution string to check
    :type sol: str
    :return: 1.0 if format is valid, 0.0 otherwise
    :rtype: float
    """
    return 1.0 if re.match(r".*<think>.+?</think>\s*\S+", sol, re.DOTALL) else 0.0


def rule_reward_fn(sol: str, gt: str) -> float:
    """
    Extract content after </think> and verify against ground truth using mathruler.

    :param sol: Solution string (may contain <think>...</think>)
    :type sol: str
    :param gt: Ground truth answer
    :type gt: str
    :return: 1.0 if correct, 0.0 otherwise
    :rtype: float
    """
    from mathruler.grader import extract_boxed_content, grade_answer
    ans = sol.split("</think>")[-1]
    pred = extract_boxed_content(ans)
    if pred == gt or grade_answer(pred, gt):
        return 1.0
    return 0.0

# ============================================================================
# Reward Recipe Configuration
# ============================================================================

# Original reward recipe for SVKG dataset training (after KG dataset training)

def geo3k_accuracy_reward_fn(sol: str, gt: str) -> float:
    """
    Geo3K accuracy reward function.

    Extract answer from \boxed{} notation and use mathruler to verify correctness.
    This is based on the verl implementation for geo3k dataset.

    :param sol: Solution string from model (should contain \boxed{answer})
    :type sol: str
    :param gt: Ground truth answer
    :type gt: str
    :return: 1.0 if answer is correct, 0.0 otherwise
    :rtype: float
    """
    from mathruler.grader import extract_boxed_content, grade_answer
    pred = extract_boxed_content(sol)
    return 1.0 if grade_answer(pred, gt) else 0.0


def geo3k_format_reward_fn(sol: str) -> float:
    """
    Geo3K format reward function.

    Check if the solution follows the required format:
    - Contains <think>...</think> tags for reasoning
    - Contains \boxed{} for final answer
    - The think tags must appear BEFORE the boxed answer

    This is based on the verl implementation for geo3k dataset.

    :param sol: Solution string from model
    :type sol: str
    :return: 1.0 if format is correct, 0.0 otherwise
    :rtype: float
    """
    # Strip leading/trailing whitespace for robust matching
    sol_stripped = sol.strip()

    # Check if solution contains both <think>...</think> and \boxed{...}
    # Use re.search to find positions
    think_match = re.search(r'<think>.*?</think>', sol_stripped, re.DOTALL)
    boxed_match = re.search(r'\\boxed\{.*?\}', sol_stripped, re.DOTALL)

    # Both components must be present AND think must come before boxed
    if think_match and boxed_match:
        # Check that </think> comes before \boxed
        think_end = think_match.end()
        boxed_start = boxed_match.start()
        return 1.0 if think_end <= boxed_start else 0.0
    else:
        return 0.0


def geo3k_combined_reward_fn(
    sol: str,
    gt: str,
    format_weight: float = 0.1
) -> float:
    """
    Geo3K combined reward function.

    Combines format reward and accuracy reward with specified weights.
    Default: 90% accuracy + 10% format (matching verl implementation)

    :param sol: Solution string from model
    :type sol: str
    :param gt: Ground truth answer
    :type gt: str
    :param format_weight: Weight for format reward. Default to 0.1
    :type format_weight: float
    :return: Weighted combination of format and accuracy rewards
    :rtype: float
    """
    acc_reward = geo3k_accuracy_reward_fn(sol, gt)
    fmt_reward = geo3k_format_reward_fn(sol)
    return (1.0 - format_weight) * acc_reward + format_weight * fmt_reward


def gsm8k_accuracy_reward_fn(sol: str, gt: str) -> float:
    """
    GSM8K accuracy reward function.

    Extract answer from \boxed{} notation and use mathruler to verify correctness.
    This follows the same pattern as geo3k but for GSM8K dataset.

    :param sol: Solution string from model (should contain \boxed{answer})
    :type sol: str
    :param gt: Ground truth answer
    :type gt: str
    :return: 1.0 if answer is correct, 0.0 otherwise
    :rtype: float
    """
    from mathruler.grader import extract_boxed_content, grade_answer
    pred = extract_boxed_content(sol)
    return 1.0 if grade_answer(pred, gt) else 0.0


def gsm8k_format_reward_fn(sol: str) -> float:
    """
    GSM8K format reward function.

    Check if the solution follows the required format:
    - Contains <think>...</think> tags for reasoning
    - Contains \boxed{} for final answer
    - The think tags must appear BEFORE the boxed answer

    This follows the same pattern as geo3k format checking.

    :param sol: Solution string from model
    :type sol: str
    :return: 1.0 if format is correct, 0.0 otherwise
    :rtype: float
    """
    # Strip leading/trailing whitespace for robust matching
    sol_stripped = sol.strip()

    # Check if solution contains both <think>...</think> and \boxed{...}
    # Use re.search to find positions
    think_match = re.search(r'<think>.*?</think>', sol_stripped, re.DOTALL)
    boxed_match = re.search(r'\\boxed\{.*?\}', sol_stripped, re.DOTALL)

    # Both components must be present AND think must come before boxed
    if think_match and boxed_match:
        # Check that </think> comes before \boxed
        think_end = think_match.end()
        boxed_start = boxed_match.start()
        return 1.0 if think_end <= boxed_start else 0.0
    else:
        return 0.0


def gsm8k_combined_reward_fn(
    sol: str,
    gt: str,
    format_weight: float = 0.1
) -> float:
    """
    GSM8K combined reward function.

    Combines format reward and accuracy reward with specified weights.
    Default: 90% accuracy + 10% format (matching verl and geo3k implementation)

    :param sol: Solution string from model
    :type sol: str
    :param gt: Ground truth answer
    :type gt: str
    :param format_weight: Weight for format reward. Default to 0.1
    :type format_weight: float
    :return: Weighted combination of format and accuracy rewards
    :rtype: float
    """
    acc_reward = gsm8k_accuracy_reward_fn(sol, gt)
    fmt_reward = gsm8k_format_reward_fn(sol)
    return (1.0 - format_weight) * acc_reward + format_weight * fmt_reward

# orig reward recipe for svkg dataset training after trained on kg dataset
# RECIPE: Dict[str, List[Tuple[str, Optional[str], float]]] = {
#     "safety":          [("model", "safety", 1.0)],
#     "knowledge":       [("model", "knowledge", 1.0),
#                         ("model", "normal",    0.5)],
#     "knowledge_rule":  [("rule",  None,        1.0),
#                         ("model", "normal",    0.5)],
#     "value":           [("model", "value",     1.0)],
#     "normal":          [("model", "normal",    1.0)],
#     "general":         [("model", "general",   1.0)],
#     "general_rule":    [("rule",  None,        1.0)],
#     # Geo3K dataset: pure rule-based reward (no reward model needed)
#     "geo3k_rule":      [("geo3k_rule", None,  1.0)],
#     # GSM8K dataset: pure rule-based reward (no reward model needed)
#     "gsm8k_rule":      [("gsm8k_rule", None,  1.0)],
# }

# Current reward recipe for SVKI dataset training (DeepSeek model)
RECIPE: Dict[str, List[Tuple[str, Optional[str], float]]] = {
    "safety":          [("model", "safety", 1.0)],
    "knowledge":       [("model", "knowledge", 1.0),
                        ("model", "normal",    1.0)],
    "knowledge_rule":  [("rule",  None,        1.0),
                        ("model", "normal",    1.0)],
    "value":           [("model", "value",     1.0)],
    "normal":          [("model", "normal",    1.0)],
    "general":         [("model", "general",   1.0)],
    "general_rule":    [("rule",  None,        1.0)],
    "muldimif": [
        ("if_rule", None, 1.0),
         ("model", "normal",    1.0)
    ],
    # Geo3K dataset: pure rule-based reward (no reward model needed)
    "geo3k_rule":      [("geo3k_rule", None,  1.0)],
    # GSM8K dataset: pure rule-based reward (no reward model needed)
    "gsm8k_rule":      [("gsm8k_rule", None,  1.0)],
}


def mix_rewards(
    labels: Sequence[str],
    model_scores: torch.Tensor,
    label_map: Dict[str, int],
    solution_strs: Sequence[str],
    refs: Sequence[str],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Mix rewards from multiple sources according to recipe configuration.

    This function combines:
        1. Format reward (always applied)
        2. Model-based rewards (from neural reward models)
        3. Rule-based rewards (from heuristic functions)

    :param labels: List of data labels (length B)
    :type labels: Sequence[str]
    :param model_scores: Tensor of model scores, shape (n_model, B)
    :type model_scores: torch.Tensor
    :param label_map: Mapping from reward type to model index
    :type label_map: Dict[str, int]
    :param solution_strs: List of solution strings (length B)
    :type solution_strs: Sequence[str]
    :param refs: List of reference answers (length B)
    :type refs: Sequence[str]
    :return: Tuple of (final_reward, metrics_dict) where final_reward is tensor of shape (B,)
             containing combined rewards and metrics_dict contains detailed reward metrics
    :rtype: Tuple[torch.Tensor, Dict[str, torch.Tensor]]

    Error handling:
        - If a model is not loaded or index out of bounds, returns 1.0 with warning
        - If label not in RECIPE, returns 1.0 with warning
        - Never raises IndexError, always returns valid reward

    Note:
        Format reward is always computed first, then rewards from recipe are added
    """
    if torch.distributed.get_rank() == 0:
        print(f"labels:{labels}, model_scores:{model_scores.tolist()}")
    device = model_scores.device
    n_model, B = model_scores.shape[0], len(labels)
    assert model_scores.shape[1] == B, "model_scores second dimension must equal batch size"

    final_reward = torch.zeros(B, dtype=torch.float32, device=device)

    # Initialize metrics dict to track individual reward components
    metrics_dict: Dict[str, torch.Tensor] = {
        'format_reward': torch.zeros(B, dtype=torch.float32, device=device),
        'accuracy_reward': torch.zeros(B, dtype=torch.float32, device=device),
        'model_reward': torch.zeros(B, dtype=torch.float32, device=device),
        'rule_reward': torch.zeros(B, dtype=torch.float32, device=device),
    }

    # ---------- Fallback scoring function ----------
    def get_model_reward(key: str, i: int) -> float:
        """
        Try to return model score for <key, sample_i>, return 1.0 on failure.

        :param key: Reward model type key
        :type key: str
        :param i: Sample index
        :type i: int
        :return: Model score or 1.0 if not available
        :rtype: float
        """
        if key not in label_map:
            print(f"Model reward <{key}> not loaded, using 1 as default reward")
            return 1.0

        idx = label_map[key]
        if idx >= n_model:
            print(f"Model reward <{key}> index {idx} out of bounds "
                        f"(n_model={n_model}), using 1 as default reward")
            return 1.0

        return float(model_scores[idx, i].item())

    # ---------- Main loop ----------
    for i, lab in enumerate(labels):
        sol = solution_strs[i]
        gt  = refs[i] if i < len(refs) else ""

        # 1) format reward (always present)
        r = format_reward_fn(sol)
        # Track separately
        metrics_dict['format_reward'][i] = r

        # 2) accumulate according to recipe
        recipe = RECIPE.get(lab)
        if recipe is None:
            print(f"label <{lab}> not registered in RECIPE, giving 1 reward directly")
            recipe = []                    # or raise

        for typ, key, w in recipe:
            if typ == "model":
                model_r = w * get_model_reward(key, i)
                r += model_r
                metrics_dict['model_reward'][i] += model_r

            elif typ == "rule":
                rule_r = w * rule_reward_fn(sol, gt)
                r += rule_r
                metrics_dict['rule_reward'][i] += rule_r
                metrics_dict['accuracy_reward'][i] = rule_r

            elif typ == "if_rule":
                # refs is actually constraints for instruction_following data
                if_r = w * if_reward_fn(solution_str=sol, ground_truth=None, constraints=gt)
                r += if_r
                metrics_dict['rule_reward'][i] += if_r
            elif typ == "geo3k_rule":
                r = 0 # TODO: geo3k have own format reward
                # Track separately
                metrics_dict['accuracy_reward'][i] = 0
                metrics_dict['format_reward'][i] = 0
                # Geo3K pure rule-based reward (format + accuracy)
                # Get individual components
                acc_r = geo3k_accuracy_reward_fn(sol, gt)
                fmt_r = geo3k_format_reward_fn(sol)
                combined_r = (1.0 - 0.1) * acc_r + 0.1 * fmt_r
                r += w * combined_r
                # Track separately
                metrics_dict['accuracy_reward'][i] = acc_r
                metrics_dict['format_reward'][i] = fmt_r
            elif typ == "gsm8k_rule":
                r = 0 # TODO: gsm8k have own format reward
                # Track separately
                metrics_dict['accuracy_reward'][i] = 0
                metrics_dict['format_reward'][i] = 0
                # GSM8K pure rule-based reward (format + accuracy)
                # Get individual components
                acc_r = gsm8k_accuracy_reward_fn(sol, gt)
                fmt_r = gsm8k_format_reward_fn(sol)
                combined_r = (1.0 - 0.1) * acc_r + 0.1 * fmt_r
                r += w * combined_r
                # Track separately
                metrics_dict['accuracy_reward'][i] = acc_r
                metrics_dict['format_reward'][i] = fmt_r
            else:
                print(f"Unknown component type {typ}, ignoring")

        final_reward[i] = r

    return final_reward, metrics_dict


def reward_fn(
    model_reward_list: List[torch.Tensor],
    labels: Sequence[str],
    queries: Sequence[str],
    refs: Sequence[str],
    label_map: Dict[str, int],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    External unified interface for computing final rewards.

    This is the main entry point called by the trainer. It:
        1. Stacks individual model rewards into a single tensor
        2. Calls mix_rewards to combine all reward sources
        3. Returns final reward tensor

    :param model_reward_list: List of reward tensors from each model, each shape (B,)
    :type model_reward_list: List[torch.Tensor]
    :param labels: List of data labels indicating reward type (length B)
    :type labels: Sequence[str]
    :param queries: List of query/solution strings (length B)
    :type queries: Sequence[str]
    :param refs: List of reference answers (length B)
    :type refs: Sequence[str]
    :param label_map: Mapping from reward type to model index
    :type label_map: Dict[str, int]
    :return: Tuple of (final_reward, metrics_dict) where final_reward is combined reward tensor
             of shape (B,) and metrics_dict contains detailed reward metrics
    :rtype: Tuple[torch.Tensor, Dict[str, torch.Tensor]]

    Note:
        If model_reward_list is empty (no NN models), a placeholder zero tensor is created
    """
    # print(f"model_reward_list:{model_reward_list}, labels:{labels}, queries:{queries}, refs:{refs}, label_map:{label_map}")
    # print(f"label_map:{label_map}")

    # ------ stack to (n_model, B) ------
    if model_reward_list:
        model_scores = torch.stack(model_reward_list)  # (n_model, B)
    else:
        # When no torch.nn model RM is available, give placeholder zero score
        B = len(labels)
        model_scores = torch.zeros(0, B, dtype=torch.float32, device="cuda")

    # ------ call combination logic ------
    return mix_rewards(labels, model_scores, label_map, queries, refs)
