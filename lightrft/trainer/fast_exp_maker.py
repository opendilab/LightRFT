"""
FastExperienceMaker Module

This module provides an optimized experience maker for RLHF (Reinforcement Learning from Human Feedback)
that supports high-performance inference backends like VLLM and SGLang. It extends the base
NaiveExperienceMaker with enhanced features for multimodal data processing, reward computation,
and advantage estimation.

Key Features:
    - VLLM/SGLang backend support for efficient text generation
    - Multimodal (vision-language) data processing
    - Multiple advantage estimation methods (GAE, RLOO, REINFORCE, Group Norm)
    - Flexible reward model composition with custom reward functions
    - Sample packing support for improved training efficiency
    - Running reward normalization and advantage whitening

Classes:
    MultimodalDataProcessor: Handles preprocessing of mixed text/image data
    RewardComputationEngine: Manages reward model inference and aggregation
    FastExperienceMaker: Main experience generation class

"""

import os
import time
import pathlib
import warnings
import inspect
from contextlib import contextmanager
from typing import Callable, Dict, List, Tuple, Union, Optional
from dataclasses import dataclass
from copy import deepcopy

import torch
import numpy as np
from PIL import Image
from easydict import EasyDict

from lightrft.models.utils import (
    compute_approx_kl,
    compute_reward,
    masked_mean,
    unpacking_samples,
)
from lightrft.models.actor_modality import ActorModality, get_supported_parameters
from lightrft.trainer.experience_maker import (
    Experience,
    NaiveExperienceMaker,
    Samples,
)
from lightrft.trainer.experience_maker_vl import (
    ExperienceVL,
    SamplesVL,
)

from lightrft.utils.remote_rm_utils import remote_rm_fn
from lightrft.utils import Timer, get_current_device
from .utils import RunningMoments, compute_clip_fraction, get_cpgd_advantages_returns, fire_sampling, vllm_ge_0130
from .advantage_calculator import get_advantage_calculator, normalize_advantages_cross_batch
from .image_utils import normalize_images, get_images_num
from .video_utils import normalize_videos, get_videos_num
from lightrft.trainer.opd_utils import (
    get_teacher_logprobs_for_experiences,
    get_teacher_logprobs_by_ids,
)
import asyncio

# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class _SamplesOutput:
    """
    Lightweight dataclass for caching intermediate computation results during experience creation.

    This structure serves as a unified container for all data flowing through the parallel
    experience generation pipeline, including sequences, attention masks, multimodal inputs,
    and model outputs (log probabilities, values, rewards).

    Attributes:
        sequences: Token ID sequences [batch_size, seq_len]
        attention_mask: Attention mask for sequences
        action_mask: Mask indicating which tokens are part of the generated response
        num_actions: Number of action tokens per sequence
        packed_seq_lens: Sequence lengths for packed samples (if packing enabled)
        response_length: Length of generated responses
        total_length: Total sequence length (prompt + response)
        prompts: Original text prompts
        labels: Optional labels for the samples

        # Vision-Language Model (VLM) specific fields
        pixel_values: Processed pixel values for images (Qwen-VL format)
        pixel_values_videos: Processed pixel values for videos (Qwen-VL format)
        image_grid_thw: Image grid dimensions [temporal, height, width]
        video_grid_thw: Video grid dimensions [temporal, height, width]
        raw_images: Original PIL images
        references: Reference texts for evaluation
        image_num: Number of images per sample

        # Model inference outputs
        action_log_probs: Log probabilities from actor model
        base_action_log_probs: Log probabilities from initial/reference model
        value: Value estimates from critic model
        rewards: Reward scores from reward model(s)
        kl: KL divergence between actor and reference policy
        inputs_extra_kwargs: Additional model-specific inputs
        prompt_and_output: Concatenated prompt+output text for reward models
    """
    # Core sequence data
    sequences: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    action_mask: Optional[torch.Tensor]
    num_actions: Union[list, torch.Tensor]
    packed_seq_lens: Optional[Union[list, torch.Tensor]]
    response_length: torch.Tensor
    total_length: torch.Tensor
    prompts: List[str]
    labels: Optional[list]

    # Vision-Language Model fields
    pixel_values: Optional[torch.Tensor] = None
    image_grid_thw: Optional[torch.Tensor] = None
    pixel_values_videos: Optional[torch.Tensor] = None
    video_grid_thw: Optional[torch.Tensor] = None
    raw_images: Optional[list] = None
    references: Optional[list] = None
    image_num: Optional[List[int]] = None
    video_num: Optional[List[int]] = None

    # Model outputs
    action_log_probs: Optional[torch.Tensor] = None
    base_action_log_probs: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None
    rewards: Optional[torch.Tensor] = None
    reward_metrics: Optional[Dict[str, torch.Tensor]] = None  # Detailed reward metrics
    kl: Optional[torch.Tensor] = None
    action_entropy: Optional[torch.Tensor] = None  # Entropy for high-entropy token filtering
    inputs_extra_kwargs: Optional[dict] = None
    prompt_and_output: Optional[List[str]] = None

    # Per-step PRM rewards (variant 2). When the reward model returns
    # ``step_rewards`` / ``step_token_indices`` arrays alongside the
    # trajectory-scalar score, they're stored here so that downstream
    # advantage computation can scatter per-step credit to specific token
    # positions (instead of only the EOS token). Both fields are
    # micro-batch-sized lists; index i holds the per-step data for the i-th
    # trajectory in the micro-batch as 1-D CPU tensors of equal length.
    # Empty tensors mean "this trajectory uses trajectory-scalar rewards"
    # and the legacy EOS-scatter path is taken in compute_reward.
    step_rewards: Optional[List[torch.Tensor]] = None
    step_token_indices: Optional[List[torch.Tensor]] = None


@dataclass
class _RewardBatchResult:
    """Reward scores and optional auxiliary metrics for one micro-batch."""

    scores: torch.Tensor
    metrics: Optional[Dict[str, torch.Tensor]] = None
    # Variant 2 (per-step PRM): when present, list[i] is a 1-D CPU tensor of
    # step rewards / step boundary token indices for trajectory i in the
    # micro-batch. When None or all-empty, the trajectory-scalar (EOS-scatter)
    # path is used downstream.
    step_rewards: Optional[List[torch.Tensor]] = None
    step_token_indices: Optional[List[torch.Tensor]] = None


class _NullProfiler:
    @contextmanager
    def section(self, _name: str):
        yield


def _call_reward_fn_compat(reward_fn, *, include_metrics: bool, **kwargs):
    """Call reward_fn while remaining compatible with older example signatures."""
    if include_metrics:
        try:
            parameters = inspect.signature(reward_fn).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())
        if accepts_kwargs or "model_reward_metrics_list" in parameters:
            return reward_fn(**kwargs)

    kwargs.pop("model_reward_metrics_list", None)
    return reward_fn(**kwargs)


# ============================================================================
# Helper Classes
# ============================================================================


class MultimodalDataProcessor:
    """
    Handles preprocessing of mixed text-only and image-text multimodal data.

    This processor separates text-only and multimodal samples, processes them through
    appropriate pipelines (tokenizer vs. multimodal processor), then merges results
    back in original order to maintain batch consistency.

    Key responsibilities:
        - Normalize image inputs (file paths, PIL images, bytes)
        - Separate text-only and image-text samples
        - Process each modality through appropriate pipeline
        - Expand samples by n_samples_per_prompt factor
        - Reconstruct original batch ordering

    Args:
        tokenizer: Tokenizer for text-only samples
        processor: Multimodal processor for image-text samples
        prompt_max_len: Maximum prompt length for truncation
    """
    def __init__(self, tokenizer, processor, prompt_max_len: int):
        """
        Initialize the multimodal data processor.

        :param tokenizer: HuggingFace tokenizer for text processing
        :type tokenizer: transformers.PreTrainedTokenizer
        :param processor: Multimodal processor (e.g., Qwen-VL processor)
        :type processor: Union[transformers.ProcessorMixin, Any]
        :param prompt_max_len: Maximum allowed prompt length
        :type prompt_max_len: int
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_max_len = prompt_max_len

    def process_multimodal_batch(
        self,
        all_prompts: List[str],
        all_images: List,
        all_references: Optional[List[str]],
        images_num: List[int],
        n_samples_per_prompt: int,
        all_videos: Optional[List],
        videos_num: Optional[List[int]],
    ) -> EasyDict:
        """
        Process multimodal batch - following original implementation exactly.

        This method is a direct port of the original process_multimodal_data to ensure
        functional equivalence.

        :param all_prompts: List of text prompts
        :type all_prompts: List[str]
        :param all_images: List of images (PIL.Image or None)
        :type all_images: List[Union[List[PIL.Image.Image], None]]
        :param all_references: Optional reference texts
        :type all_references: Optional[List[str]]
        :param images_num: Number of images per sample
        :type images_num: List[int]
        :param n_samples_per_prompt: Number of samples to generate per prompt
        :type n_samples_per_prompt: int
        :param all_videos: List of videos (List[str] or None)
        :type all_videos: Optional[List[Union[List[str], None]]]
        :param videos_num: Number of videos per sample
        :type videos_num: Optional[List[int]]
        :return: Dictionary containing processed data
        :rtype: EasyDict
        """
        N = n_samples_per_prompt
        L = len(all_prompts)

        # Ensure all_images and all_videos are iterable even if None
        if all_images is None:
            all_images = [None] * L
        if all_videos is None:
            all_videos = [None] * L

        # ===== Stage 1: Separation =====
        all_prompts_text, all_prompts_multimodal = [], []
        all_images_valid = []
        all_videos_valid = []
        text_idx = []

        for idx, (prompt, image, video) in enumerate(zip(all_prompts, all_images, all_videos)):
            if image is None and video is None:
                all_prompts_text.append(prompt)
                text_idx.append(idx)
            else:
                all_prompts_multimodal.append(prompt)
                all_images_valid.append(image)
                all_videos_valid.append(video)

        # ===== Stage 2: Expansion =====
        all_prompts_text = sum([[p] * N for p in all_prompts_text], [])
        all_prompts_multimodal = sum([[p] * N for p in all_prompts_multimodal], [])
        all_images_valid = [img for img in all_images_valid for _ in range(N)]
        all_videos_valid = [vid for vid in all_videos_valid for _ in range(N)]
        all_images_num = sum([[num] * N for num in images_num], []) if images_num is not None else [0] * (L * N)
        all_videos_num = sum([[num] * N for num in videos_num], []) if videos_num is not None else [0] * (L * N)

        # ===== Stage 3-A: Text-only processing =====
        if len(all_prompts_text) > 0:
            inputs_text = self.tokenizer(
                all_prompts_text,
                max_length=self.prompt_max_len,
                truncation=True,
                add_special_tokens=False,
            )
            all_prompt_token_ids_text = inputs_text["input_ids"]
        else:
            all_prompt_token_ids_text = []

        # Initialize multimodal variables for text-only compatibility
        all_prompt_token_ids_multimodal = []
        all_images_pixel_values_multimodal = None
        all_videos_pixel_values_multimodal = None
        all_images_grid_thw_multimodal = None
        all_videos_grid_thw_multimodal = None

        # ===== Stage 3-B: Multimodal processing =====
        if len(all_prompts_multimodal) > 0:
            assert self.processor is not None, "Processor required for multimodal data"

            flat_images = []
            for img_item in all_images_valid:
                if isinstance(img_item, list):
                    flat_images.extend(img_item)
                elif img_item is not None:
                    flat_images.append(img_item)

            flat_videos = []
            for vid_item in all_videos_valid:
                if isinstance(vid_item, list):
                    flat_videos.extend(vid_item)
                elif vid_item is not None:
                    flat_videos.append(vid_item)

            processor_kwargs = {
                "text": all_prompts_multimodal.copy(),
                "add_special_tokens": False,
                "padding": True,
                "max_length": self.prompt_max_len,
                "truncation": True,
                "return_tensors": "pt",
            }
            if flat_images:
                processor_kwargs["images"] = flat_images
            if flat_videos:
                processor_kwargs["videos"] = flat_videos

            inputs_multimodal = self.processor(**processor_kwargs)

            all_prompt_token_ids_multimodal = inputs_multimodal["input_ids"]
            all_images_pixel_values_multimodal = inputs_multimodal.get("pixel_values", None)
            all_videos_pixel_values_multimodal = inputs_multimodal.get("pixel_values_videos", None)

            all_images_grid_thw_multimodal = inputs_multimodal.get("image_grid_thw", None)
            all_videos_grid_thw_multimodal = inputs_multimodal.get("video_grid_thw", None)

            # Some VLM processors (for example URSA) return batched image tensors directly
            # and do not expose Qwen2-VL style grid metadata. In that case we synthesize a
            # minimal per-image grid so the existing sample/replay slicing logic can still
            # split one tensor slice per image while the model simply ignores the grid input.
            if flat_images and all_images_grid_thw_multimodal is None:
                all_images_grid_thw_multimodal = torch.ones((len(flat_images), 3), dtype=torch.long)
            if flat_videos and all_videos_grid_thw_multimodal is None:
                all_videos_grid_thw_multimodal = torch.ones((len(flat_videos), 3), dtype=torch.long)

        # ===== Stage 4: Merge back in original order =====
        total_samples = L * N
        all_prompts_out = [None] * total_samples
        all_images_out = [None] * total_samples
        all_videos_out = [None] * total_samples
        all_prompt_token_ids_out = [None] * total_samples
        all_images_grid_thw_list = [None] * total_samples
        all_videos_grid_thw_list = [None] * total_samples

        # 4-A: Fill text-only
        text_ptr = 0
        for orig_idx in text_idx:
            for n in range(N):
                gid = orig_idx * N + n
                all_prompts_out[gid] = all_prompts_text[text_ptr]
                all_prompt_token_ids_out[gid] = all_prompt_token_ids_text[text_ptr]
                # Ensure (0, 3) shape for cat
                all_images_grid_thw_list[gid] = torch.empty((0, 3), dtype=torch.long)
                all_videos_grid_thw_list[gid] = torch.empty((0, 3), dtype=torch.long)
                text_ptr += 1

        # 4-B: Fill multimodal
        multi_ptr = 0
        image_grid_ptr = 0
        video_grid_ptr = 0
        for orig_idx in range(L):
            if orig_idx in text_idx:
                continue
            for n in range(N):
                gid = orig_idx * N + n
                all_prompts_out[gid] = all_prompts_multimodal[multi_ptr]
                all_images_out[gid] = all_images_valid[multi_ptr]
                all_videos_out[gid] = all_videos_valid[multi_ptr]
                all_prompt_token_ids_out[gid] = all_prompt_token_ids_multimodal[multi_ptr]

                # Handle image_grid_thw: extract rows based on all_images_num
                num_images = all_images_num[gid]
                if num_images > 0 and all_images_grid_thw_multimodal is not None:
                    all_images_grid_thw_list[gid] = all_images_grid_thw_multimodal[image_grid_ptr:image_grid_ptr +
                                                                                   num_images]
                    image_grid_ptr += num_images
                else:
                    all_images_grid_thw_list[gid] = torch.empty((0, 3), dtype=torch.long)

                # Handle video_grid_thw: extract rows based on all_videos_num
                num_videos = all_videos_num[gid]
                if num_videos > 0 and all_videos_grid_thw_multimodal is not None:
                    all_videos_grid_thw_list[gid] = all_videos_grid_thw_multimodal[video_grid_ptr:video_grid_ptr +
                                                                                   num_videos]
                    video_grid_ptr += num_videos
                else:
                    all_videos_grid_thw_list[gid] = torch.empty((0, 3), dtype=torch.long)

                multi_ptr += 1

        # Concatenate grid_thw (using cat instead of stack to support multi-image/video)
        all_images_grid_thw = (
            torch.cat(all_images_grid_thw_list, dim=0)
            if len(all_images_grid_thw_list) > 0 else torch.empty((0, 3), dtype=torch.long)
        )
        all_videos_grid_thw = (
            torch.cat(all_videos_grid_thw_list, dim=0)
            if len(all_videos_grid_thw_list) > 0 else torch.empty((0, 3), dtype=torch.long)
        )

        # Expand references
        if all_references is not None:
            all_references = sum([[ref] * N for ref in all_references], [])

        return EasyDict(
            all_prompt_token_ids=all_prompt_token_ids_out,
            all_prompts=all_prompts_out,
            all_images=all_images_out,
            all_videos=all_videos_out,
            all_images_num=all_images_num,
            all_videos_num=all_videos_num,
            all_images_pixel_values=all_images_pixel_values_multimodal,
            all_videos_pixel_values=all_videos_pixel_values_multimodal,
            all_images_grid_thw=all_images_grid_thw,
            all_videos_grid_thw=all_videos_grid_thw,
            all_references=all_references,
        )


class RewardComputationEngine:
    """
    Manages reward model inference and score aggregation.

    This engine handles both local and remote reward models, supporting:
        - Remote HTTP/gRPC reward models
        - Local PyTorch reward models
        - Custom reward functions and rules
        - Multi-model ensemble with custom aggregation
        - Optimized batch processing with sample filtering

    The engine uses a three-stage pipeline:
        1. Gather: Collect or filter samples based on reward recipe
        2. Process: Run forward pass through reward model(s)
        3. Aggregate: Combine scores using reward_fn

    Args:
        reward_model: Single reward model or list of models
        remote_rm_url: List of remote reward model URLs
        custom_reward_func: Custom Python function for reward computation
        reward_fn: Aggregation function for multiple reward models
        reward_fn_label_map: Mapping from reward model names to indices
        tokenizer: Tokenizer for decoding sequences
        strategy: Training strategy (for model loading/offloading)
        packing_samples: Whether samples are packed
    """
    def __init__(
        self,
        reward_model,
        remote_rm_url: Optional[List[str]],
        custom_reward_func: Optional[Callable],
        reward_fn: Optional[Callable],
        reward_fn_label_map: Optional[Dict],
        reward_recipe: Optional[Dict],
        tokenizer,
        strategy,
        packing_samples: bool,
    ):
        """
        Initialize the reward computation engine.

        :param reward_model: Single reward model or list of models
        :type reward_model: Union[torch.nn.Module, List[torch.nn.Module]]
        :param remote_rm_url: List of remote reward model URLs
        :type remote_rm_url: Optional[List[str]]
        :param custom_reward_func: Custom Python function for reward computation
        :type custom_reward_func: Optional[Callable]
        :param reward_fn: Aggregation function for multiple reward models
        :type reward_fn: Optional[Callable]
        :param reward_fn_label_map: Mapping from reward model names to indices
        :type reward_fn_label_map: Optional[Dict[str, int]]
        :param reward_recipe: Recipe configuration for reward computation
        :type reward_recipe: Optional[Dict]
        :param tokenizer: Tokenizer for decoding sequences
        :type tokenizer: transformers.PreTrainedTokenizer
        :param strategy: Training strategy (for model loading/offloading)
        :type strategy: Any
        :param packing_samples: Whether samples are packed
        :type packing_samples: bool
        """
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.custom_reward_func = custom_reward_func
        self.reward_fn = reward_fn
        self.reward_fn_label_map = reward_fn_label_map or {}
        self.reward_recipe = reward_recipe or {}
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.packing_samples = packing_samples

        # Build inverse label map for quick lookup
        self.inv_label_map = {idx: key for key, idx in self.reward_fn_label_map.items()}

        # Configuration flag for optimized filtering engine
        self.use_filtering_engine = False  # TODO: Enable after testing

    def compute_rewards(
        self,
        outputs: List[_SamplesOutput],
        vlm_mode: bool,
        device: torch.device,
    ) -> None:
        """
        Compute rewards for all samples and store in outputs[i].rewards.

        This method dispatches to the appropriate computation path based on
        whether remote or local reward models are used.

        :param outputs: List of sample outputs to compute rewards for
        :type outputs: List[_SamplesOutput]
        :param vlm_mode: Whether in vision-language mode
        :type vlm_mode: bool
        :param device: Device to place reward tensors on
        :type device: torch.device
        """
        if self.remote_rm_url:
            self._compute_remote_rewards(outputs, vlm_mode, device)
        else:
            self._compute_local_rewards(outputs, vlm_mode, device)

    def _compute_remote_rewards(
        self,
        outputs: List[_SamplesOutput],
        vlm_mode: bool,
        device: torch.device,
    ) -> None:
        """
        Compute rewards using remote reward models.

        This path maintains serial processing for compatibility with HTTP/gRPC APIs.

        :param outputs: Sample outputs to compute rewards for
        :type outputs: List[_SamplesOutput]
        :param vlm_mode: Vision-language mode flag
        :type vlm_mode: bool
        :param device: Target device for tensors
        :type device: torch.device
        """
        for output in outputs:
            # Decode sequences to text
            sequences = (
                output.sequences
                if not self.packing_samples else unpacking_samples(output.sequences, output.packed_seq_lens)
            )
            queries = self.tokenizer.batch_decode(sequences, skip_special_tokens=False)

            reward_tensors = []

            # Custom reward function
            if self.custom_reward_func:
                if vlm_mode:
                    scores = self.custom_reward_func(queries, output.prompts, output.references)
                else:
                    scores = self.custom_reward_func(queries, output.prompts, output.labels)
                reward_tensors.append(torch.as_tensor(scores, dtype=torch.float32, device=device))

            # Remote reward models
            for rm_url in self.remote_rm_url[len(reward_tensors):]:
                if vlm_mode:
                    scores = remote_rm_fn(
                        rm_url,
                        queries=queries,
                        prompts=output.prompts,
                        references=output.references,
                        raw_images=output.raw_images,
                    )
                else:
                    scores = remote_rm_fn(
                        rm_url,
                        queries=queries,
                        prompts=output.prompts,
                        labels=output.labels,
                    )
                reward_tensors.append(torch.as_tensor(scores, dtype=torch.float32, device=device))

            # Aggregate rewards
            output.rewards = (self.reward_fn(reward_tensors) if len(reward_tensors) > 1 else reward_tensors[0])

    def _compute_local_rewards(
        self,
        outputs: List[_SamplesOutput],
        vlm_mode: bool,
        device: torch.device,
    ) -> None:
        """
        Compute rewards using local reward models.

        Implements batched processing for efficiency. Supports both standard
        PyTorch models and custom engine models with optional sample filtering.

        :param outputs: Sample outputs to compute rewards for
        :type outputs: List[_SamplesOutput]
        :param vlm_mode: Vision-language mode flag
        :type vlm_mode: bool
        :param device: Target device for tensors
        :type device: torch.device
        """
        # Ensure reward_model is a list
        is_multi_rm = isinstance(self.reward_model, (list, tuple))
        rm_list = list(self.reward_model) if is_multi_rm else [self.reward_model]

        # Load all PyTorch models to GPU
        for rm in rm_list:
            if isinstance(rm, torch.nn.Module):
                self.strategy.reload_model(rm)

        # Compute rewards for each RM
        # all_rewards_list[rm_idx][micro_batch_idx] = _RewardBatchResult(batch_size,)
        all_rewards_list = []

        for rm_idx, rm in enumerate(rm_list):
            micro_batch_rewards = self._compute_single_rm_rewards(rm, rm_idx, outputs, vlm_mode, device)
            all_rewards_list.append(micro_batch_rewards)

            # Offload model immediately after use
            if isinstance(rm, torch.nn.Module):
                self.strategy.offload_model(rm)

        # Aggregate rewards across RMs for each micro-batch
        self._aggregate_rewards(outputs, all_rewards_list, is_multi_rm)

    def _compute_single_rm_rewards(
        self,
        rm,
        rm_idx: int,
        outputs: List[_SamplesOutput],
        vlm_mode: bool,
        device: torch.device,
    ) -> List[_RewardBatchResult]:
        """
        Compute rewards for a single reward model across all micro-batches.

        :param rm: Reward model instance
        :type rm: Union[torch.nn.Module, Any]
        :param rm_idx: Index of this RM in the RM list
        :type rm_idx: int
        :param outputs: Sample outputs
        :type outputs: List[_SamplesOutput]
        :param vlm_mode: Vision-language mode flag
        :type vlm_mode: bool
        :param device: Target device
        :type device: torch.device
        :return: List of reward results, one per micro-batch
        :rtype: List[_RewardBatchResult]
        """
        # Check if this is a custom engine model (non-torch base_model)
        is_custom_engine = (
            isinstance(rm, torch.nn.Module) and hasattr(rm, "base_model")
            and not isinstance(rm.base_model, torch.nn.Module)
        )

        if is_custom_engine and self.use_filtering_engine:
            return self._compute_filtered_rewards(rm, rm_idx, outputs, device)
        elif is_custom_engine:
            return self._compute_batched_custom_engine_rewards(rm, outputs, device)
        elif isinstance(rm, torch.nn.Module):
            return self._compute_standard_torch_rewards(rm, outputs, vlm_mode, device)
        else:
            raise ValueError(f"Unsupported reward model type: {type(rm)}")

    def _compute_filtered_rewards(
        self,
        rm,
        rm_idx: int,
        outputs: List[_SamplesOutput],
        device: torch.device,
    ) -> List[_RewardBatchResult]:
        """
        Compute rewards using optimized filtering (only process relevant samples).

        This optimization filters samples based on the reward recipe, only running
        the forward pass on samples that actually need this specific RM.

        :param rm: Custom engine reward model
        :type rm: Any
        :param rm_idx: RM index for label lookup
        :type rm_idx: int
        :param outputs: Sample outputs
        :type outputs: List[_SamplesOutput]
        :param device: Target device
        :type device: torch.device
        :return: List of reward results per micro-batch
        :rtype: List[_RewardBatchResult]
        """
        # Get RM key from inverse label map
        rm_key = self.inv_label_map.get(rm_idx)
        if rm_key is None:
            raise ValueError(
                f"Filtering engine requires a label map key for RM at index {rm_idx}, "
                f"but none was found. Check your reward_fn_label_map configuration."
            )

        # ========== Gather Stage: Filter samples that need this RM ==========
        flat_data = {
            "prompt_and_output": [],
            "raw_images": [],
            "image_num": [],
            "references": [],
            "labels": [],
        }
        needed_positions = []  # [(micro_batch_idx, sample_idx), ...]

        for mb_idx, output in enumerate(outputs):
            for samp_idx, label in enumerate(output.labels):
                # Check if this sample's recipe requires this RM
                needs_rm = any(
                    typ == "model" and key == rm_key and float(weight) != 0.0
                    for typ, key, weight in self.reward_recipe.get(label, [])
                )

                if needs_rm:
                    needed_positions.append((mb_idx, samp_idx))
                    flat_data["prompt_and_output"].append(output.prompt_and_output[samp_idx])
                    flat_data["raw_images"].append(output.raw_images[samp_idx])
                    flat_data["image_num"].append(output.image_num[samp_idx])
                    flat_data["references"].append(output.references[samp_idx])
                    flat_data["labels"].append(output.labels[samp_idx])

        # ========== Process Stage: Compute or skip ==========
        if not needed_positions:
            # No samples need this RM, return zeros for all micro-batches
            return [
                _RewardBatchResult(
                    scores=torch.zeros(len(output.labels), dtype=torch.float32, device=device),
                    metrics=None,
                ) for output in outputs
            ]

        # Run single forward pass on filtered samples
        rm_output = rm(
            None,
            None,
            prompt_and_outputs=flat_data["prompt_and_output"],
            raw_images=flat_data["raw_images"],
            img_num=flat_data["image_num"],
            references=flat_data["references"],
            labels=flat_data["labels"],
        )
        filtered_scores = (rm_output["score"] if isinstance(rm_output, dict) else rm_output).to(device)

        # ========== Scatter Stage: Reconstruct micro-batch structure ==========
        micro_batch_rewards = [
            torch.zeros(len(output.labels), dtype=torch.float32, device=device) for output in outputs
        ]

        for (mb_idx, samp_idx), score in zip(needed_positions, filtered_scores):
            micro_batch_rewards[mb_idx][samp_idx] = score

        return [_RewardBatchResult(scores=rewards, metrics=None) for rewards in micro_batch_rewards]

    def _compute_batched_custom_engine_rewards(
        self,
        rm,
        outputs: List[_SamplesOutput],
        device: torch.device,  # noqa: ARG002 (unused but kept for API consistency)
    ) -> List[_RewardBatchResult]:
        """
        Compute rewards using custom engine with full batch processing (legacy path).

        :param rm: Custom engine reward model
        :type rm: Any
        :param outputs: Sample outputs
        :type outputs: List[_SamplesOutput]
        :param device: Target device (unused but kept for API consistency)
        :type device: torch.device
        :return: List of reward results per micro-batch
        :rtype: List[_RewardBatchResult]
        """
        # Flatten all micro-batches into single batch
        flat_data = {
            "prompt_and_output": [],
            "raw_images": [],
            "image_num": [],
            "references": [],
            "labels": [],
        }

        for output in outputs:
            flat_data["prompt_and_output"].extend(output.prompt_and_output)
            flat_data["raw_images"].extend(output.raw_images)
            flat_data["image_num"].extend(output.image_num)
            flat_data["references"].extend(output.references)
            flat_data["labels"].extend(output.labels)

        # Single forward pass
        rm_output = rm(
            None,
            None,
            prompt_and_outputs=flat_data["prompt_and_output"],
            raw_images=flat_data["raw_images"],
            img_num=flat_data["image_num"],
            references=flat_data["references"],
            labels=flat_data["labels"],
        )
        all_scores = rm_output["score"] if isinstance(rm_output, dict) else rm_output

        # Split back into micro-batches
        batch_sizes = [len(output.prompt_and_output) for output in outputs]
        return [
            _RewardBatchResult(scores=scores.to(device=device, dtype=torch.float32), metrics=None)
            for scores in all_scores.split(batch_sizes)
        ]

    @staticmethod
    def _normalize_reward_metrics(
        rm_output: Dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Optional[Dict[str, torch.Tensor]]:
        metrics: Dict[str, torch.Tensor] = {}
        for key, value in rm_output.items():
            if key == "score":
                continue
            if not isinstance(value, torch.Tensor):
                if isinstance(value, (int, float, bool)):
                    value = torch.tensor(value, dtype=torch.float32)
                else:
                    continue
            metric = value.to(device=device)
            if metric.ndim == 0:
                metric = metric.repeat(batch_size)
            metric = metric.reshape(-1).float()
            if metric.numel() != batch_size:
                continue
            metrics[key] = metric
        return metrics or None

    def _compute_standard_torch_rewards(
        self,
        rm,
        outputs: List[_SamplesOutput],
        vlm_mode: bool,  # noqa: ARG002 (kept for future VLM-specific logic)
        device: torch.device,
    ) -> List[_RewardBatchResult]:
        """
        Compute rewards using standard PyTorch reward model.

        Processes each micro-batch sequentially.

        :param rm: PyTorch reward model
        :type rm: torch.nn.Module
        :param outputs: Sample outputs
        :type outputs: List[_SamplesOutput]
        :param vlm_mode: Vision-language mode flag (reserved for future use)
        :type vlm_mode: bool
        :param device: Target device
        :type device: torch.device
        :return: List of reward results per micro-batch
        :rtype: List[_RewardBatchResult]
        """
        micro_batch_rewards: List[_RewardBatchResult] = []

        for output in outputs:
            # Unpack sequences if needed
            sequences = (
                output.sequences
                if not self.packing_samples else unpacking_samples(output.sequences, output.packed_seq_lens)
            )

            # Forward pass
            rm_output = rm(
                sequences,
                output.attention_mask,
                prompt_and_output=output.prompt_and_output,
                raw_images=output.raw_images,
                img_num=output.image_num,
                references=output.references,
                labels=output.labels,
                **output.inputs_extra_kwargs,
            )

            if isinstance(rm_output, dict):
                score = torch.as_tensor(rm_output["score"], dtype=torch.float32, device=device)
                metrics = self._normalize_reward_metrics(rm_output, score.numel(), device)
                # Variant 2 (per-step PRM reward): the reward model may emit
                # per-trajectory step_rewards / step_token_indices lists. They
                # come back as Python lists of CPU tensors (variable length per
                # trajectory) — pass them through unchanged so compute_reward
                # downstream can scatter per-step credit.
                step_rewards = rm_output.get("step_rewards")
                step_token_indices = rm_output.get("step_token_indices")
            else:
                score = torch.as_tensor(rm_output, dtype=torch.float32, device=device)
                metrics = None
                step_rewards = None
                step_token_indices = None
            micro_batch_rewards.append(
                _RewardBatchResult(
                    scores=score,
                    metrics=metrics,
                    step_rewards=step_rewards,
                    step_token_indices=step_token_indices,
                )
            )

        return micro_batch_rewards

    def _aggregate_rewards(
        self,
        outputs: List[_SamplesOutput],
        all_rewards_list: List[List[_RewardBatchResult]],
        is_multi_rm: bool,
    ) -> None:
        """
        Aggregate rewards from multiple RMs and store in outputs.

        :param outputs: Sample outputs (modified in-place)
        :type outputs: List[_SamplesOutput]
        :param all_rewards_list: Nested list [rm_idx][micro_batch_idx] -> reward result
        :type all_rewards_list: List[List[_RewardBatchResult]]
        :param is_multi_rm: Whether using multiple reward models
        :type is_multi_rm: bool
        """
        num_micro_batches = len(outputs)
        num_rms = len(all_rewards_list)

        for mb_idx in range(num_micro_batches):
            # Collect rewards from all RMs for this micro-batch
            same_batch_results = [all_rewards_list[rm_idx][mb_idx] for rm_idx in range(num_rms)]
            same_batch_rewards = [result.scores for result in same_batch_results]

            if is_multi_rm:
                # Use custom aggregation function
                sequences = (
                    outputs[mb_idx].sequences if not self.packing_samples else
                    unpacking_samples(outputs[mb_idx].sequences, outputs[mb_idx].packed_seq_lens)
                )
                queries = self.tokenizer.batch_decode(sequences, skip_special_tokens=False)

                rewards, reward_metrics = _call_reward_fn_compat(
                    self.reward_fn,
                    include_metrics=True,
                    model_reward_list=same_batch_rewards,
                    model_reward_metrics_list=[result.metrics for result in same_batch_results],
                    labels=outputs[mb_idx].labels,
                    queries=queries,
                    refs=outputs[mb_idx].references,
                    label_map=self.reward_fn_label_map,
                )
                outputs[mb_idx].rewards = rewards
                outputs[mb_idx].reward_metrics = reward_metrics
            else:
                # Single RM, use score directly
                outputs[mb_idx].rewards = same_batch_results[0].scores
                outputs[mb_idx].reward_metrics = same_batch_results[0].metrics
                # Variant 2 (per-step PRM): forward step_rewards / token
                # indices from the single reward model into outputs so
                # _process_experiences / compute_reward can see them. Multi-RM
                # mode is intentionally NOT supported here — per-step credit
                # assignment with multiple reward models would require a
                # bespoke aggregator (open question for follow-up); we keep
                # the multi-RM path on trajectory-scalar rewards only.
                outputs[mb_idx].step_rewards = same_batch_results[0].step_rewards
                outputs[mb_idx].step_token_indices = same_batch_results[0].step_token_indices


# ============================================================================
# Main Experience Maker
# ============================================================================


class FastExperienceMaker(NaiveExperienceMaker):
    """
    Optimized experience maker with VLLM/SGLang support and advanced RL features.

    This class extends NaiveExperienceMaker to provide:
        - High-performance inference via VLLM or SGLang backends
        - Multimodal (vision-language) data processing
        - Multiple advantage estimation algorithms (GAE, RLOO, REINFORCE, Group Norm)
        - Flexible reward model composition with custom aggregation
        - Sample packing for improved training efficiency
        - Running reward normalization and advantage whitening/clipping

    The experience generation pipeline:
        1. Sample Generation: Use inference engine to generate responses
        2. Shard-Parallel Preprocessing: Distribute samples across shards
        3. Model Inference: Batch forward through actor, critic, initial, and reward models
        4. Shard-Parallel Postprocessing: Gather results back
        5. Reward Processing: Apply transformations (normalization, shaping, filtering)
        6. Advantage Estimation: Compute advantages and returns

    Args:
        packing_samples: Whether to pack multiple sequences into single batch
        processor: Multimodal processor for vision-language models
        *args, **kwargs: Arguments passed to parent NaiveExperienceMaker
    """
    def __init__(self, *args, packing_samples: bool = False, processor=None, profiler=None, **kwargs):
        """
        Initialize FastExperienceMaker.

        :param args: Positional arguments for NaiveExperienceMaker
        :type args: tuple
        :param packing_samples: Enable sample packing for efficiency
        :type packing_samples: bool
        :param processor: Multimodal processor (required for VLM models)
        :type processor: Optional[Any]
        :param kwargs: Keyword arguments for NaiveExperienceMaker
        :type kwargs: dict
        """
        super().__init__(*args, **kwargs)

        # Core configuration
        self.backend_mp_group = self.strategy.engine_mp_group
        self.backend = self.strategy.args.engine_type
        self.packing_samples = packing_samples
        self.processor = processor
        self.profiler = profiler if profiler is not None else _NullProfiler()

        # Initialize tokenizer (extract from processor if needed)
        if self.processor is not None:
            self.tokenizer = getattr(self.processor, "tokenizer", self.processor)

        # Initialize running reward normalization
        if self.strategy.args.reward_running_norm:
            self.reward_running_moments = RunningMoments()
        else:
            self.reward_running_moments = None

        # Initialize advantage calculator
        advantage_estimator = self.strategy.config.advantage_estimator
        self.advantage_calculator = get_advantage_calculator(advantage_estimator, self.strategy.config)

        # Initialize helper modules
        if self.processor is not None:
            self.multimodal_processor = MultimodalDataProcessor(
                tokenizer=self.tokenizer,
                processor=self.processor,
                prompt_max_len=self.prompt_max_len,
            )
        else:
            self.multimodal_processor = None

        # For On-Policy Distillation (OPD), prefer dedicated teacher_model_url.
        # Fall back to remote_rm_url with deprecation warning for backwards compatibility.
        if advantage_estimator == "on_policy_distillation":
            teacher_url = getattr(self.strategy.args, 'teacher_model_url', None)
            if teacher_url is not None:
                self.teacher_model_url = teacher_url
            elif self.remote_rm_url is not None:
                warnings.warn(
                    "Using --remote_rm_url as teacher URL is deprecated. "
                    "Use --teacher_model_url instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                self.teacher_model_url = self.remote_rm_url
            else:
                self.teacher_model_url = None
            rm_url_for_reward_engine = None  # Don't use remote_rm_url for rewards in OPD mode
        else:
            self.teacher_model_url = None
            rm_url_for_reward_engine = self.remote_rm_url

        self.reward_engine = RewardComputationEngine(
            reward_model=self.reward_model,
            remote_rm_url=rm_url_for_reward_engine,
            custom_reward_func=getattr(self, "custom_reward_func", None),
            reward_fn=self.reward_fn,
            reward_fn_label_map=getattr(self, "reward_fn_label_map", None),
            reward_recipe=getattr(self, "reward_recipe", None),
            tokenizer=self.tokenizer,
            strategy=self.strategy,
            packing_samples=self.packing_samples,
        )

        # Cache actor's supported parameters based on its modality
        # Default to VISION_LANGUAGE for backward compatibility with models without modality attribute
        actor_modality = self.actor.modality
        self._actor_supported_params = get_supported_parameters(actor_modality)

    # ========================================================================
    # Public API Methods
    # ========================================================================

    @torch.no_grad()
    def make_experience_list(
        self,
        all_prompts: List[str],
        all_images: Optional[List] = None,
        all_videos: Optional[List] = None,
        all_references: Optional[List[str]] = None,
        all_labels: Optional[List] = None,
        **generate_kwargs,
    ) -> List[ExperienceVL]:
        """
        Generate a list of experiences from prompts and optional multimodal inputs.

        This is the main entry point for experience generation. It orchestrates the
        entire pipeline from sampling to advantage computation.

        :param all_prompts: List of text prompts
        :type all_prompts: List[str]
        :param all_images: Optional images for multimodal generation
        :type all_images: Optional[List]
        :param all_references: Optional reference texts for evaluation
        :type all_references: Optional[List[str]]
        :param all_labels: Optional labels for samples
        :type all_labels: Optional[List]
        :param all_videos: Optional videos for multimodal generation
        :type all_videos: Optional[List]
        :param generate_kwargs: Generation parameters (temperature, max_new_tokens, etc.)
        :type generate_kwargs: dict
        :return: List of Experience or ExperienceVL objects with computed advantages and returns
        :rtype: List[Union[Experience, ExperienceVL]]
        """
        config = self.strategy.config

        with self.profiler.section("collect/total"):
            if all_images is not None:
                if self.multimodal_processor is None:
                    raise ValueError(
                        "Multimodal data (images) provided but processor was not initialized. "
                        "Please provide a processor when initializing FastExperienceMaker for VLM support."
                    )
                all_images = normalize_images(all_images)

            if all_videos is not None:
                if self.multimodal_processor is None:
                    raise ValueError(
                        "Multimodal data (videos) provided but processor was not initialized. "
                        "Please provide a processor when initializing FastExperienceMaker for VLM support."
                    )
                all_videos = normalize_videos(all_videos)

            images_num = (get_images_num(all_images) if self.multimodal_processor and all_images is not None else None)
            videos_num = (get_videos_num(all_videos) if self.multimodal_processor and all_videos is not None else None)

            Timer.start('  generate_samples')
            with self.profiler.section("collect/generate"):
                samples_list = self.generate_samples(
                    all_prompts,
                    all_images=all_images,
                    images_num=images_num,
                    all_videos=all_videos,
                    videos_num=videos_num,
                    all_references=all_references,
                    all_labels=all_labels,
                    **generate_kwargs,
                )
            Timer.stop('  generate_samples')

            torch.distributed.barrier()
            torch.cuda.synchronize()

            with self.profiler.section("collect/sp_preprocess"):
                all_samples = self.strategy.sp_data_processor.preprocess(samples_list)

            Timer.start('  make_experience')
            with self.profiler.section("collect/model_total"):
                experiences = self._make_experience_list_by_model(all_samples)
            Timer.stop('  make_experience')

            with self.profiler.section("collect/sp_postprocess"):
                experiences = self.strategy.sp_data_processor.postprocess(experiences)

            with self.profiler.section("collect/process_rewards"):
                experiences, rewards = self._process_experiences(
                    experiences, generate_kwargs.get("max_new_tokens", 1024)
                )

            if (images_num is not None and not all(num == 1 for num in images_num)) or \
               (videos_num is not None and not all(num == 1 for num in videos_num)):
                expanded_images_num = sum([[num] * config.n_samples_per_prompt
                                           for num in images_num], []) if images_num is not None else None

                expanded_videos_num = sum([[num] * config.n_samples_per_prompt
                                           for num in videos_num], []) if videos_num is not None else None

                self._process_multi_image_video_thws(experiences, expanded_images_num, expanded_videos_num)

            if config.advantage_estimator == "on_policy_distillation":
                self._fetch_teacher_logprobs(experiences)

            with self.profiler.section("collect/advantages"):
                experiences = self._compute_advantages_and_returns(experiences, rewards, generate_kwargs)

            return experiences

    @torch.no_grad()
    def generate_samples(
        self,
        all_prompts: List[str],
        all_images: Optional[List] = None,
        all_videos: Optional[List] = None,
        images_num: Optional[List[int]] = None,
        videos_num: Optional[List[int]] = None,
        all_references: Optional[List[str]] = None,
        all_labels: Optional[List] = None,
        **generate_kwargs,
    ) -> List[Samples]:
        """
        Generate samples using the inference engine (VLLM or SGLang).

        This method handles:
            - Sampling parameter configuration
            - Multimodal data processing
            - Inference engine invocation
            - Output processing into Samples format

        :param all_prompts: List of text prompts
        :type all_prompts: List[str]
        :param all_images: Optional images for VLM
        :type all_images: Optional[List]
        :param images_num: Number of images per prompt
        :type images_num: Optional[List[int]]
        :param all_references: Reference texts
        :type all_references: Optional[List[str]]
        :param all_labels: Sample labels
        :type all_labels: Optional[List]
        :param all_videos: Optional videos for VLM
        :type all_videos: Optional[List]
        :param videos_num: Number of videos per prompt
        :type videos_num: Optional[List[int]]
        :param generate_kwargs: Generation parameters (temperature, max_new_tokens, etc.)
        :type generate_kwargs: dict
        :return: List of Samples or SamplesVL objects
        :rtype: List[Union[Samples, SamplesVL]]
        """
        assert self.strategy.inference_engine is not None, "Inference engine required"

        torch.cuda.synchronize()
        start_time = time.time()

        config = self.strategy.config
        is_multimodal = all_images is not None or all_videos is not None
        n_samples = config.n_samples_per_prompt

        all_images_num = None
        all_videos_num = None
        all_images_pixel_values = None
        all_videos_pixel_values = None
        all_images_grid_thw = None
        all_videos_grid_thw = None

        with self.profiler.section("collect/generate_prepare"):
            if config.engine_type == "vllm":
                from vllm import SamplingParams

                if vllm_ge_0130():
                    max_model_len = self.strategy.inference_engine.llm_engine.model_config.max_model_len
                    truncate_tokens = min(8192, max_model_len)
                else:
                    truncate_tokens = 8192

                sampling_params = SamplingParams(
                    temperature=generate_kwargs.get("temperature", 1.0),
                    top_p=generate_kwargs.get("top_p", 1.0),
                    top_k=generate_kwargs.get("top_k", -1),
                    repetition_penalty=generate_kwargs.get("repetition_penalty", 1.0),
                    max_tokens=generate_kwargs.get("max_new_tokens", 1024),
                    min_tokens=generate_kwargs.get("min_new_tokens", 1),
                    skip_special_tokens=generate_kwargs.get("skip_special_tokens", False),
                    include_stop_str_in_output=True,
                    ignore_eos=os.environ.get("IGNORE_EOS", "0") == "1",
                    truncate_prompt_tokens=truncate_tokens,
                )
            elif config.engine_type == "sglang":
                sampling_params = dict(
                    n=1,
                    temperature=generate_kwargs.get("temperature", 1.0),
                    top_p=generate_kwargs.get("top_p", 1.0),
                    top_k=generate_kwargs.get("top_k", -1),
                    max_new_tokens=generate_kwargs.get("max_new_tokens", 1024),
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    repetition_penalty=generate_kwargs.get("repetition_penalty", 1.0),
                    skip_special_tokens=generate_kwargs.get("skip_special_tokens", False),
                    spaces_between_special_tokens=True,
                    ignore_eos=os.environ.get("IGNORE_EOS", "0") == "1",
                )
            elif config.engine_type == "hf":
                max_new_tokens = generate_kwargs.get("max_new_tokens", 1024)
                local_hf_max_new_tokens = int(getattr(config, "local_hf_max_new_tokens", 0) or 0)
                if local_hf_max_new_tokens > 0:
                    max_new_tokens = min(max_new_tokens, local_hf_max_new_tokens)
                sampling_params = dict(
                    temperature=generate_kwargs.get("temperature", 1.0),
                    top_p=generate_kwargs.get("top_p", 1.0),
                    top_k=generate_kwargs.get("top_k", -1),
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=generate_kwargs.get("min_new_tokens", 1),
                    do_sample=generate_kwargs.get("do_sample", True),
                    repetition_penalty=generate_kwargs.get("repetition_penalty", 1.0),
                    no_repeat_ngram_size=generate_kwargs.get("no_repeat_ngram_size", 0),
                    skip_special_tokens=generate_kwargs.get("skip_special_tokens", False),
                    ignore_eos=os.environ.get("IGNORE_EOS", "0") == "1",
                )
            else:
                raise ValueError(f"Unsupported engine type: {config.engine_type}")

            if all_labels is not None:
                all_labels = sum([[label] * n_samples for label in all_labels], [])

            if is_multimodal:
                processed_data = self.multimodal_processor.process_multimodal_batch(
                    all_prompts=all_prompts,
                    all_images=all_images,
                    all_references=all_references,
                    images_num=images_num,
                    n_samples_per_prompt=n_samples,
                    all_videos=all_videos,
                    videos_num=videos_num,
                )
                all_prompt_token_ids = processed_data["all_prompt_token_ids"]
                all_prompts = processed_data["all_prompts"]
                all_images = processed_data["all_images"]
                all_videos = processed_data["all_videos"]
                all_images_num = processed_data["all_images_num"]
                all_videos_num = processed_data["all_videos_num"]
                all_images_grid_thw = processed_data["all_images_grid_thw"]
                all_videos_grid_thw = processed_data["all_videos_grid_thw"]
                all_images_pixel_values = processed_data["all_images_pixel_values"]
                all_videos_pixel_values = processed_data["all_videos_pixel_values"]
                all_references = processed_data.get("all_references", None)
            else:
                tokenized = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)
                all_prompt_token_ids = sum([[token_ids] * n_samples for token_ids in tokenized["input_ids"]], [])

        try:
            with self.profiler.section("collect/generate_engine"):
                if hasattr(self.strategy.args, 'use_fire') and self.strategy.args.use_fire:
                    sleep_engine = getattr(self.strategy.args, 'enable_engine_sleep', False)

                    def generate_fn(
                        sampling_params,
                        all_prompt_token_ids,
                        all_prompts=None,
                        all_images=None,
                        all_videos=None,
                        images_num=None,
                        videos_num=None,
                    ):
                        return self.strategy.gather_and_generate(
                            sampling_params=sampling_params,
                            all_prompt_token_ids=all_prompt_token_ids,
                            all_prompts=all_prompts,
                            all_images=all_images,
                            all_videos=all_videos,
                            images_num=images_num,
                            videos_num=videos_num,
                            sleep_engine=sleep_engine,
                        )

                    all_outputs = fire_sampling(
                        all_prompt_token_ids=all_prompt_token_ids,
                        generate_fn=generate_fn,
                        engine_type=config.engine_type,
                        first_token_temperature=generate_kwargs.get("first_token_temperature", 10.0),
                        temperature=generate_kwargs.get("temperature", 1.0),
                        is_multimodal=is_multimodal,
                        all_prompts=all_prompts,
                        all_images=all_images,
                        all_videos=all_videos,
                        all_images_num=all_images_num,
                        all_videos_num=all_videos_num,
                        sampling_params=sampling_params,
                    )
                else:
                    all_outputs = self.strategy.gather_and_generate(
                        sampling_params=sampling_params,
                        all_prompt_token_ids=all_prompt_token_ids,
                        all_prompts=all_prompts if is_multimodal else None,
                        sleep_engine=self.strategy.args.enable_engine_sleep,
                        all_images=all_images if is_multimodal else None,
                        all_videos=all_videos if is_multimodal else None,
                        images_num=all_images_num if is_multimodal else None,
                        videos_num=all_videos_num if is_multimodal else None,
                        all_images_pixel_values=all_images_pixel_values if is_multimodal else None,
                        all_videos_pixel_values=all_videos_pixel_values if is_multimodal else None,
                        all_images_grid_thw=all_images_grid_thw if is_multimodal else None,
                        all_videos_grid_thw=all_videos_grid_thw if is_multimodal else None,
                    )
        except ValueError as e:
            if "prompt" in str(e) and "too long" in str(e):
                self.strategy.print(f"[Skip] {e}")
                return None  # Return None, subsequent experience_maker will ignore
            else:
                raise

        with self.profiler.section("collect/generate_build_samples"):
            samples_list = []
            image_patch_idx = 0
            video_patch_idx = 0
            image_start_idx = 0
            video_start_idx = 0

            for i in range(0, len(all_outputs), config.micro_rollout_batch_size):
                micro_batch_outputs = all_outputs[i:i + config.micro_rollout_batch_size]
                micro_batch_prompts = all_prompts[i:i + config.micro_rollout_batch_size]

                micro_batch_grid_thw = None
                micro_batch_video_grid_thw = None
                micro_batch_raw_images = None

                if is_multimodal:
                    rollout_image_count = sum(all_images_num[i:i + config.micro_rollout_batch_size])
                    micro_batch_grid_thw = all_images_grid_thw[image_start_idx:image_start_idx + rollout_image_count]
                    micro_batch_raw_images = all_images[i:i + config.micro_rollout_batch_size]
                    image_start_idx += rollout_image_count

                    rollout_video_count = sum(all_videos_num[i:i + config.micro_rollout_batch_size])
                    micro_batch_video_grid_thw = all_videos_grid_thw[video_start_idx:video_start_idx +
                                                                     rollout_video_count]
                    video_start_idx += rollout_video_count

                micro_batch_references = (
                    all_references[i:i + config.micro_rollout_batch_size] if all_references else None
                )
                micro_batch_labels = (all_labels[i:i + config.micro_rollout_batch_size] if all_labels else None)

                if not self.packing_samples:
                    sample, updated_patch_idx, updated_video_patch_idx = self._build_unpacked_sample(
                        outputs=micro_batch_outputs,
                        prompts=micro_batch_prompts,
                        labels=micro_batch_labels,
                        references=micro_batch_references,
                        is_multimodal=is_multimodal,
                        grid_thw=micro_batch_grid_thw,
                        video_grid_thw=micro_batch_video_grid_thw,
                        raw_images=micro_batch_raw_images,
                        pixel_values=all_images_pixel_values if is_multimodal else None,
                        pixel_values_videos=all_videos_pixel_values if is_multimodal else None,
                        images_num=all_images_num[i:i + config.micro_rollout_batch_size] if is_multimodal else None,
                        videos_num=all_videos_num[i:i + config.micro_rollout_batch_size] if is_multimodal else None,
                        image_patch_idx=image_patch_idx,
                        video_patch_idx=video_patch_idx,
                    )
                    if updated_patch_idx is not None:
                        image_patch_idx = updated_patch_idx
                    if updated_video_patch_idx is not None:
                        video_patch_idx = updated_video_patch_idx
                    samples_list.append(sample)
                else:
                    sample = self._build_packed_sample(
                        outputs=micro_batch_outputs,
                        prompts=micro_batch_prompts,
                        labels=micro_batch_labels,
                        references=micro_batch_references,
                    )
                    samples_list.append(sample)

        # Report timing
        torch.cuda.synchronize()
        gen_time = torch.tensor(time.time() - start_time, device=get_current_device())
        torch.distributed.all_reduce(gen_time, op=torch.distributed.ReduceOp.MAX)
        self.strategy.print(f"***Rollout engine generation time (global max): {gen_time.item():.4f}s")
        self.strategy.report_memory("after rollout engine generation")

        return samples_list

    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Compute advantages and returns using Generalized Advantage Estimation (GAE).

        Extends parent method with advantage whitening and clipping.

        :param values: Value estimates from critic
        :type values: torch.Tensor
        :param rewards: Reward signals
        :type rewards: torch.Tensor
        :param action_mask: Mask for valid action positions
        :type action_mask: torch.Tensor
        :param gamma: Discount factor
        :type gamma: float
        :param lambd: GAE lambda parameter
        :type lambd: float
        :return: Tuple of (advantages, returns, advantage_clip_fraction)
        :rtype: Tuple[torch.Tensor, torch.Tensor, float]
        """
        # Call parent GAE implementation
        advantages, returns = super().get_advantages_and_returns(values, rewards, action_mask, gamma, lambd)

        config = self.strategy.config

        # Advantage whitening (normalization)
        if config.advantages_norm:
            masked_adv = torch.masked_select(advantages, action_mask)
            adv_mean = masked_adv.mean()
            adv_std = masked_adv.std()
            advantages = (advantages - adv_mean) / (adv_std + 1e-9)

        # Advantage clipping
        advantage_clip_frac = 0.0
        if config.advantage_clip > 0:
            advantages = torch.clamp(advantages, -config.advantage_clip, config.advantage_clip)
            advantage_clip_frac = compute_clip_fraction(advantages, config.advantage_clip, -config.advantage_clip)

        return advantages, returns, advantage_clip_frac

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    def _process_multi_image_video_thws(
        self,
        experiences: List[ExperienceVL],
        images_num: Optional[List[int]] = None,
        videos_num: Optional[List[int]] = None,
    ) -> None:
        """
        Process image_grid_thws and video_grid_thws for multi-image/video scenarios.

        Ensures len(experience.sequences) == len(experience.image_grid_thws) by
        converting the stacked tensor into a list of per-sequence tensors.

        :param experiences: List of experiences to modify in-place
        :type experiences: List[ExperienceVL]
        :param images_num: Number of images per sample (expanded by n_samples_per_prompt)
        :type images_num: Optional[List[int]]
        :param videos_num: Number of videos per sample (expanded by n_samples_per_prompt)
        :type videos_num: Optional[List[int]]
        """
        config = self.strategy.config

        for i, experience in enumerate(experiences):
            # Get image and video counts for this micro-batch
            start_idx = i * config.micro_rollout_batch_size
            end_idx = (i + 1) * config.micro_rollout_batch_size

            if images_num is not None:
                micro_images_num = images_num[start_idx:end_idx]
                if sum(micro_images_num) > 0 and experience.image_grid_thws is not None:
                    image_grid_thw_list = []
                    image_grid_thws = experience.image_grid_thws
                    image_grid_thws_unbind = torch.unbind(image_grid_thws)

                    thw_idx = 0
                    for num in micro_images_num:
                        if num > 0:
                            stacked_thw = torch.stack(image_grid_thws_unbind[thw_idx:thw_idx + num], dim=0).to("cuda")
                            image_grid_thw_list.append(stacked_thw)
                            thw_idx += num
                        else:
                            image_grid_thw_list.append(None)
                    experience.image_grid_thws = image_grid_thw_list
                else:
                    experience.image_grid_thws = [None] * len(micro_images_num)

            if videos_num is not None:
                micro_videos_num = videos_num[start_idx:end_idx]
                if sum(micro_videos_num) > 0 and experience.video_grid_thws is not None:
                    video_grid_thw_list = []
                    video_grid_thws = experience.video_grid_thws
                    video_grid_thws_unbind = torch.unbind(video_grid_thws)

                    v_thw_idx = 0
                    for num in micro_videos_num:
                        if num > 0:
                            v_stacked_thw = torch.stack(video_grid_thws_unbind[v_thw_idx:v_thw_idx + num],
                                                        dim=0).to("cuda")
                            video_grid_thw_list.append(v_stacked_thw)
                            v_thw_idx += num
                        else:
                            video_grid_thw_list.append(None)
                    experience.video_grid_thws = video_grid_thw_list
                else:
                    experience.video_grid_thws = [None] * len(micro_videos_num)

    def _fetch_teacher_logprobs(
        self,
        experiences: List[ExperienceVL],
    ) -> None:
        """
        Fetch teacher log probabilities for on-policy distillation.

        Uses input_ids (not text) to query teacher model, ensuring token-level
        alignment between teacher and student log probabilities.

        Teacher log probs are aligned to the same shape as action_log_probs
        [batch_size, seq_len], with zeros for prompt positions.

        :param experiences: List of experiences to add teacher log probs to
        :type experiences: List[Union[Experience, ExperienceVL]]
        """

        # Get teacher URL from config
        teacher_url = self.teacher_model_url
        if isinstance(teacher_url, list):
            teacher_url = teacher_url[0] if teacher_url else None
        if teacher_url is None:
            raise ValueError(
                "Teacher model URL not specified. "
                "Please set --teacher_model_url to the teacher model server URL."
            )

        Timer.start('  fetch_teacher_logprobs')

        for exp in experiences:
            sequences = exp.sequences  # [batch_size, seq_len]
            attention_mask = exp.attention_mask  # [batch_size, seq_len]
            action_mask = exp.action_mask  # [batch_size, num_tokens]

            # response_lengths must be int for slicing
            response_lengths = action_mask.sum(dim=-1).int().tolist()
            num_tokens = action_mask.shape[1]

            # Strip padding tokens before sending to SGLang.
            # sequences is [prompt, response, eos, pad, pad, ...]; the padding
            # tokens would cause SGLang to return logprobs for pad positions,
            # making the [-resp_len:] slice grab wrong tokens.
            input_ids_list = []
            for j in range(sequences.shape[0]):
                valid_len = int(attention_mask[j].sum().item())
                input_ids_list.append(sequences[j, :valid_len].cpu().tolist())

            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    teacher_lp_list = loop.run_until_complete(
                        get_teacher_logprobs_by_ids(
                            url=teacher_url,
                            input_ids_list=input_ids_list,
                            response_lengths=response_lengths,
                        )
                    )
                finally:
                    loop.close()

                # Align teacher log probs to action_log_probs shape [batch_size, num_tokens].
                # Use action_mask indices directly; works regardless of left/right padding.
                #
                # Correctness: teacher_lp_list[i] contains teacher logprobs for response
                # tokens in first-to-last order (from teacher_lp[-resp_len:]).
                # valid_indices = ascending positions where action_mask==1 (real response tokens).
                # So aligned_teacher_lp[i, valid_indices[k]] = tlp[k] correctly maps the k-th
                # teacher logprob to the k-th response token position, regardless of padding direction.
                batch_size = sequences.shape[0]
                aligned_teacher_lp = torch.zeros(batch_size, num_tokens, dtype=torch.float32)
                for i, (tlp, resp_len) in enumerate(zip(teacher_lp_list, response_lengths)):
                    if resp_len == 0:
                        continue
                    valid_indices = torch.where(action_mask[i] == 1)[0]
                    actual_len = min(len(tlp), len(valid_indices))
                    aligned_teacher_lp[i, valid_indices[:actual_len]] = tlp[:actual_len]

                exp.info["teacher_log_probs"] = aligned_teacher_lp

            except Exception as e:
                raise RuntimeError(f"Failed to fetch teacher log probs from {teacher_url}: {e}") from e

        Timer.stop('  fetch_teacher_logprobs')

    def _process_experiences(
        self,
        experiences: List[ExperienceVL],
        max_new_tokens: int,
    ) -> Tuple[List[ExperienceVL], List[torch.Tensor]]:
        """
        Apply reward transformations and filtering to experiences.

        Handles:
            - Overlong sequence penalty
            - Dynamic sampling filtering
            - Advantage estimation-specific reward shaping (RLOO, REINFORCE, Group Norm)

        :param experiences: List of experiences to process
        :type experiences: List[Union[Experience, ExperienceVL]]
        :param max_new_tokens: Maximum generation length
        :type max_new_tokens: int
        :return: Tuple of (processed_experiences, shaped_rewards)
        :rtype: Tuple[List[Union[Experience, ExperienceVL]], List[torch.Tensor]]
        """
        config = self.strategy.config
        rewards = torch.cat([exp.info["reward"] for exp in experiences])

        # ========== Overlong Sequence Penalty ==========
        if config.overlong_buffer:
            expected_len = max_new_tokens - config.overlong_buffer_len
            actual_lens = torch.cat([exp.action_mask.sum(dim=1) for exp in experiences])
            exceed_len = actual_lens - expected_len

            # Penalty: clamp(-exceed_len / buffer_len * penalty_factor, max=0)
            penalty = torch.clamp(
                -exceed_len / config.overlong_buffer_len * config.overlong_buffer_penalty_factor, max=0.0
            )
            rewards += penalty

        # ========== Dynamic Sampling Warning ==========
        if config.dynamic_sampling and config.advantage_estimator in ["rloo", "reinforce_baseline"]:
            warnings.warn(f"dynamic_sampling not implemented for {config.advantage_estimator}, ignoring", UserWarning)

        # ========== Advantage Estimator-Specific Shaping ==========
        # Use calculator's preprocess_rewards method
        return self.advantage_calculator.preprocess_rewards(rewards, experiences, max_new_tokens)

    def _apply_step_reward_group_norm(self, experiences: List) -> None:
        """Variant 2 (per-step PRM) GRPO-style step-level baseline subtraction.

        Active only when ``self.strategy.config.per_step_reward_mode == "group_norm"``.
        For each step k, computes mean/std across the K trajectories that share
        the same prompt (group), then replaces step_rewards with
        ``(s - mean) / std`` so that scattered token-level rewards are zero-mean
        signed advantages — analogous to GRPO trajectory-level baseline but at
        step granularity.

        This is the difference between paper variant 2's two interpretations:
          - "raw":        scatter raw sigmoid step_score (paper Figure ablation)
          - "group_norm": scatter group-normalized step_score (GRPO convention)

        Padding (step_token_indices < 0) is masked so it doesn't pollute mean/std.
        Operates in-place on ``experience.info["step_rewards"]``.
        """
        config = self.strategy.config
        K = config.n_samples_per_prompt
        if K is None or K < 2:
            return  # need >= 2 samples per group for std

        all_sr: List[torch.Tensor] = []
        all_sti: List[torch.Tensor] = []
        for exp in experiences:
            sr_list = exp.info.get("step_rewards")
            sti_list = exp.info.get("step_token_indices")
            if sr_list is None or sti_list is None:
                return  # no per-step data present
            all_sr.extend(sr_list)
            all_sti.extend(sti_list)

        if not all_sr:
            return
        B = len(all_sr)
        if B % K != 0:
            warnings.warn(
                f"per_step_reward_mode=group_norm: trajectory count {B} not divisible "
                f"by n_samples_per_prompt {K}; skipping step-level group norm."
            )
            return

        max_steps = max((t.numel() for t in all_sr), default=0)
        if max_steps == 0:
            return

        sr_padded = torch.zeros(B, max_steps, dtype=torch.float32)
        valid = torch.zeros(B, max_steps, dtype=torch.bool)
        for i, (sr, sti) in enumerate(zip(all_sr, all_sti)):
            n = sr.numel()
            if n > 0:
                sr_padded[i, :n] = sr.float()
                valid[i, :n] = (sti >= 0)

        G = B // K
        sr_g = sr_padded.reshape(G, K, max_steps)
        valid_g = valid.reshape(G, K, max_steps).float()

        n_valid = valid_g.sum(dim=1, keepdim=True).clamp(min=1.0)
        sr_masked = sr_g * valid_g
        mean = sr_masked.sum(dim=1, keepdim=True) / n_valid
        sq = ((sr_g - mean) * valid_g) ** 2
        var = sq.sum(dim=1, keepdim=True) / n_valid
        std = (var + 1e-9).sqrt()
        sr_normed = (sr_g - mean) / std
        # Zero out invalid entries so they don't show up if accidentally scattered.
        sr_normed = sr_normed * valid_g
        sr_normed_flat = sr_normed.reshape(B, max_steps)

        # Write back, preserving each trajectory's actual step count (n).
        idx = 0
        for exp in experiences:
            sr_list = exp.info["step_rewards"]
            new_sr_list: List[torch.Tensor] = []
            for sr in sr_list:
                n = sr.numel()
                if n > 0:
                    new_sr_list.append(sr_normed_flat[idx, :n].cpu().to(sr.dtype))
                else:
                    new_sr_list.append(sr)
                idx += 1
            exp.info["step_rewards"] = new_sr_list

    def _compute_advantages_and_returns(
        self,
        experiences: List[ExperienceVL],
        rewards: List[torch.Tensor],
        generate_kwargs: Dict,
    ) -> List[ExperienceVL]:
        """
        Compute advantages and returns for each experience.

        Applies reward normalization/clipping, KL penalty, and advantage estimation
        based on the configured method (GAE, CPGD, REINFORCE, etc.).

        :param experiences: List of experiences to process
        :type experiences: List[Union[Experience, ExperienceVL]]
        :param rewards: List of reward tensors
        :type rewards: List[torch.Tensor]
        :param generate_kwargs: Generation parameters (contains gamma, lambd)
        :type generate_kwargs: Dict
        :return: List of experiences with advantages and returns filled in
        :rtype: List[Union[Experience, ExperienceVL]]
        """
        config = self.strategy.config

        # Variant 2 step-level baseline subtraction (cross-experience, group-aware).
        # Only active when label produced step_rewards AND user set
        # --per_step_reward_mode=group_norm.
        if getattr(config, "per_step_reward_mode", "raw") == "group_norm":
            self._apply_step_reward_group_norm(experiences)

        for experience, reward in zip(experiences, rewards):
            reward = reward.to("cuda")
            processed_reward = reward.clone()  # TODO：check

            # ========== Reward Normalization ==========
            if self.reward_running_moments:
                self.reward_running_moments.update(processed_reward)
                if config.reward_running_norm_minus_mean:
                    processed_reward = ((processed_reward - self.reward_running_moments.mean) /
                                        self.reward_running_moments.std)
                else:
                    processed_reward /= self.reward_running_moments.std

            # ========== Reward Clipping ==========
            if config.reward_clip > 0:
                experience.info["reward_clip_frac"] = compute_clip_fraction(
                    processed_reward, config.reward_clip, -config.reward_clip
                )
                processed_reward = torch.clamp(processed_reward, -config.reward_clip, config.reward_clip)

            # ========== Final Reward (with KL penalty) ==========
            # Variant 2 (per-step PRM): if experience carries
            # step_rewards/step_token_indices (placed by _pack_experience
            # when the reward model emitted them), build a padded
            # (batch, max_steps) tensor for compute_reward to scatter into
            # per-step boundary tokens. Trajectories with empty step lists get
            # all-(-1) indices; compute_reward falls back to scalar EOS reward
            # for those rows only, so mixed step/scalar batches retain signal.
            step_rewards_padded = None
            step_indices_padded = None
            step_rewards_list = experience.info.get("step_rewards")
            step_indices_list = experience.info.get("step_token_indices")
            if (
                step_rewards_list is not None and step_indices_list is not None
                and any(t.numel() > 0 for t in step_rewards_list)
            ):
                max_steps = max(t.numel() for t in step_rewards_list)
                if max_steps > 0:
                    B = len(step_rewards_list)
                    step_rewards_padded = torch.zeros(B, max_steps, dtype=torch.float32, device="cuda")
                    step_indices_padded = torch.full((B, max_steps), -1, dtype=torch.long, device="cuda")
                    for i, (sr, sti) in enumerate(zip(step_rewards_list, step_indices_list)):
                        n = sr.numel()
                        if n > 0:
                            step_rewards_padded[
                                i, :n] = sr.to(step_rewards_padded.device, dtype=step_rewards_padded.dtype)
                            step_indices_padded[
                                i, :n] = sti.to(step_indices_padded.device, dtype=step_indices_padded.dtype)

            final_reward = compute_reward(
                processed_reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                num_actions=experience.info["num_actions"],
                step_rewards=step_rewards_padded,
                step_token_indices=step_indices_padded,
            )

            # ========== Advantage Estimation ==========
            # Compute advantages and returns using calculator
            gamma = generate_kwargs.pop("gamma", 1.0)
            experience.advantages, experience.returns, info_dict = self.advantage_calculator.compute(
                experience,
                final_reward,
                gamma=gamma,
                generate_kwargs=generate_kwargs,
            )

            # Update experience info with calculator's info dict
            experience.info.update(info_dict)

            # ========== Store Episode Return ==========
            if not self.packing_samples:
                experience.info["return"] = final_reward.sum(dim=-1)
            else:
                experience.info["return"] = torch.tensor([r.sum() for r in final_reward], device=final_reward.device)

            # Cleanup
            experience.kl = None
            del experience.info["num_actions"]

        # ========== Cross-batch Advantage Normalization ==========
        # Use the utility function for cross-batch normalization
        experiences = normalize_advantages_cross_batch(experiences, self.advantage_estimator, self.strategy.args)

        return experiences

    @torch.no_grad()
    def _make_experience_list_by_model(
        self,
        all_samples: List[Union[Samples, SamplesVL]],
    ) -> List[Union[Experience, ExperienceVL]]:
        """
        Batch forward pass through all models to create experiences.

        This method implements role-based batching to avoid frequent model switching.
        Processing order:
            1. Actor (log probabilities)
            2. Initial model (reference log probabilities)
            3. Critic (value estimates)
            4. Reward model(s) (rewards)
            5. Assemble Experience objects

        :param all_samples: List of Samples/SamplesVL from generate_samples
        :type all_samples: List[Union[Samples, SamplesVL]]
        :return: List of Experience/ExperienceVL objects with model outputs filled in
        :rtype: List[Union[Experience, ExperienceVL]]
        """
        device = get_current_device()
        vlm_mode = isinstance(all_samples[0], SamplesVL)

        outputs = [self._preprocess_sample(sample, vlm_mode, device) for sample in all_samples]

        Timer.start('    actor_logprob')
        with self.profiler.section("collect/model/actor_logprob"):
            need_entropy = hasattr(self.actor, 'high_entropy_token_ratio') and self.actor.high_entropy_token_ratio > 0.0
            for output in outputs:
                if need_entropy:
                    action_log_probs, model_output = self.actor(
                        output.sequences,
                        output.num_actions,
                        output.attention_mask,
                        packed_seq_lens=output.packed_seq_lens,
                        return_output=True,
                        **output.inputs_extra_kwargs
                    )
                    output.action_log_probs = action_log_probs
                    if "action_entropy" in model_output:
                        output.action_entropy = model_output["action_entropy"]
                else:
                    output.action_log_probs = self.actor(
                        output.sequences,
                        output.num_actions,
                        output.attention_mask,
                        packed_seq_lens=output.packed_seq_lens,
                        **output.inputs_extra_kwargs
                    )
        Timer.stop('    actor_logprob')

        if self.initial_model is not None:
            with self.profiler.section("collect/model/reference_logprob"):
                self.strategy.reload_model(self.initial_model)
                for output in outputs:
                    output.base_action_log_probs = self.initial_model(
                        output.sequences,
                        output.num_actions,
                        output.attention_mask,
                        packed_seq_lens=output.packed_seq_lens,
                        **output.inputs_extra_kwargs
                    )
                self.strategy.offload_model(self.initial_model)

        if self.critic is not None:
            with self.profiler.section("collect/model/critic_forward"):
                self.strategy.reload_model(self.critic)
                for output in outputs:
                    output.value = self.critic(
                        output.sequences, output.num_actions, output.attention_mask, **output.inputs_extra_kwargs
                    )
                self.strategy.offload_model(self.critic)

        with self.profiler.section("collect/model/reward_forward"):
            self.reward_engine.compute_rewards(outputs, vlm_mode, device)

        return [self._pack_experience(output, vlm_mode) for output in outputs]

    def _preprocess_sample(
        self,
        sample: Union[Samples, SamplesVL],
        vlm: bool,
        device: torch.device,
    ) -> _SamplesOutput:
        """
        Convert a Samples object to _SamplesOutput for processing.

        :param sample: Input sample
        :type sample: Union[Samples, SamplesVL]
        :param vlm: Vision-language mode flag
        :type vlm: bool
        :param device: Target device
        :type device: torch.device
        :return: _SamplesOutput with data ready for model inference
        :rtype: _SamplesOutput
        """
        # Extract common fields
        sequences = sample.sequences.to(device)
        attention_mask = sample.attention_mask.to(device)
        action_mask = sample.action_mask
        num_actions = sample.num_actions
        packed_seq_lens = sample.packed_seq_lens
        response_length = sample.response_length
        total_length = sample.total_length
        prompts = sample.prompts
        labels = getattr(sample, "labels", None)
        references = sample.references
        output_texts = getattr(sample, "output_texts", None)

        # Build extra kwargs for VLM based on actor's modality
        # Only include parameters that the actor's modality supports
        extra_kwargs = {}
        if vlm:
            # Candidate parameters to pass
            candidate_params = {
                "pixel_values": sample.pixel_values,
                "image_grid_thw": sample.image_grid_thws,
                "pixel_values_videos": sample.pixel_values_videos,
                "video_grid_thw": sample.video_grid_thws,
            }

            # Filter to only include supported parameters
            extra_kwargs = {
                key: value
                for key, value in candidate_params.items()
                if key in self._actor_supported_params
            }

        # Fix Qwen-VL image token count bug
        self._fix_qwen_vl_image_tokens(sequences, sample, vlm)

        return _SamplesOutput(
            sequences=sequences,
            attention_mask=attention_mask,
            action_mask=action_mask,
            num_actions=num_actions,
            packed_seq_lens=packed_seq_lens,
            response_length=response_length,
            total_length=total_length,
            prompts=prompts,
            labels=labels,
            pixel_values=getattr(sample, "pixel_values", None),
            image_grid_thw=getattr(sample, "image_grid_thws", None),
            pixel_values_videos=getattr(sample, "pixel_values_videos", None),
            video_grid_thw=getattr(sample, "video_grid_thws", None),
            raw_images=getattr(sample, "raw_images", None),
            image_num=getattr(sample, "image_num", None),
            video_num=getattr(sample, "video_num", None),
            references=references,
            inputs_extra_kwargs=extra_kwargs,
            prompt_and_output=([p + (o or "") for p, o in zip(prompts, output_texts)] if output_texts else None),
        )

    def _fix_qwen_vl_image_tokens(
        self,
        sequences: torch.Tensor,
        sample: SamplesVL,
        vlm: bool,
    ) -> None:
        """
        Fix Qwen-VL image token count mismatch.

        In some cases, the number of image tokens in sequences doesn't match
        the number of pixel value patches. This fixes the discrepancy by replacing
        extra image tokens with padding tokens.

        :param sequences: Token sequence (modified in-place)
        :type sequences: torch.Tensor
        :param sample: Original sample
        :type sample: SamplesVL
        :param vlm: Vision-language mode flag
        :type vlm: bool
        """
        if not vlm or sample.pixel_values is None:
            return

        config = self.strategy.unwrap_model(self.actor.model).config
        image_token_id = getattr(config, "image_token_id", getattr(config, "image_token_index", None))
        if image_token_id is None:
            return
        num_tokens = (sequences == image_token_id).sum()
        vision_config = getattr(config, "vision_config", None)
        spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2)
        merge_length = spatial_merge_size ** 2

        # Qwen2-VL style processors flatten visual patches and provide real
        # image_grid_thw metadata. URSA keeps one 4-D image tensor per image and
        # does not use Qwen grid metadata, so its image-token count is per image.
        has_real_grid = sample.image_grid_thws is not None and sample.image_grid_thws.numel() > 0
        if sample.pixel_values.dim() >= 4 and not has_real_grid:
            num_patches = sample.pixel_values.shape[0]
        elif has_real_grid:
            num_patches = int(torch.prod(sample.image_grid_thws, dim=-1).sum().item() // merge_length)
        else:
            num_patches = sample.pixel_values.shape[0] // merge_length

        if num_tokens != num_patches:
            self.strategy.print(
                f"[Warning] Mismatch found during rollout step. Fixing sequences. "
                f"Tokens: {num_tokens}, Patches: {num_patches}"
            )

            pad_token_id = self.tokenizer.pad_token_id
            diff = num_tokens - num_patches
            token_positions = (sequences == image_token_id).nonzero()

            # Replace extra tokens from the end
            for k in range(diff):
                pos = token_positions[-(k + 1)]
                sequences[pos[0], pos[1]] = pad_token_id

    def _pack_experience(
        self,
        output: _SamplesOutput,
        vlm: bool,
    ) -> Union[Experience, ExperienceVL]:
        """
        Pack model outputs into an Experience object.

        :param output: Processed sample output
        :type output: _SamplesOutput
        :param vlm: Vision-language mode flag
        :type vlm: bool
        :return: Experience or ExperienceVL object
        :rtype: Union[Experience, ExperienceVL]
        """
        # Compute KL divergence
        if self.initial_model is not None and not self.strategy.args.use_kl_loss:
            # Note: When use_kl_loss is True, KL is used as a loss term;
            # when False, KL is added to reward as augmentation
            kl = compute_approx_kl(
                output.action_log_probs,
                output.base_action_log_probs,
                action_mask=output.action_mask,
                kl_estimator=self.strategy.args.kl_estimator,
            )
        else:
            kl = torch.zeros_like(output.action_log_probs)

        # Compute mean KL
        if not self.packing_samples:
            kl_mean = masked_mean(kl, output.action_mask, dim=-1)
        else:
            kl_mean = torch.tensor(
                [each.mean() for each in unpacking_samples(kl, output.num_actions)],
                device=kl.device,
            )

        # Clear base log probs if not needed
        if not self.strategy.args.use_kl_loss:
            output.base_action_log_probs = None

        # Build info dict
        info = dict(
            kl=kl_mean,
            reward=output.rewards,
            response_length=output.response_length,
            total_length=output.total_length,
            num_actions=output.num_actions,
        )

        # Add reward_metrics if available
        if output.reward_metrics is not None:
            info['reward_metrics'] = output.reward_metrics

        # Variant 2 (per-step PRM rewards): forward to experience.info so
        # downstream advantage computation can build per-token reward signals.
        # Stored as Python lists of CPU tensors (variable length per traj).
        # _process_experiences chunks experiences along micro_batch_size,
        # so we keep the *micro-batch local* indexing here — i.e. info
        # carries the lists scoped to this single _SamplesOutput.
        if output.step_rewards is not None:
            info['step_rewards'] = output.step_rewards
        if output.step_token_indices is not None:
            info['step_token_indices'] = output.step_token_indices

        # Create Experience object
        if vlm:
            return ExperienceVL(
                sequences=output.sequences,
                pixel_values=output.pixel_values,
                image_grid_thws=output.image_grid_thw,
                raw_images=output.raw_images,
                pixel_values_videos=output.pixel_values_videos,
                video_grid_thws=output.video_grid_thw,
                action_log_probs=output.action_log_probs,
                base_action_log_probs=output.base_action_log_probs,
                values=output.value,
                returns=None,  # returns (filled later)
                advantages=None,  # advantages (filled later)
                attention_mask=output.attention_mask,
                action_mask=output.action_mask,
                info=info,
                kl=kl,
                action_entropy=output.action_entropy,
            )
        else:
            return Experience(
                sequences=output.sequences,
                action_log_probs=output.action_log_probs,
                base_action_log_probs=output.base_action_log_probs,
                values=output.value,
                returns=None,  # returns (filled later)
                advantages=None,  # advantages (filled later)
                attention_mask=output.attention_mask,
                action_mask=output.action_mask,
                info=info,
                kl=kl,
                action_entropy=output.action_entropy,
            )

    def _build_unpacked_sample(
        self,
        outputs: List,
        prompts: List[str],
        labels: Optional[List],
        references: Optional[List],
        is_multimodal: bool,
        **kwargs,
    ) -> Tuple[Union[Samples, SamplesVL], Optional[int], Optional[int]]:
        """
        Build unpacked sample (one sequence per row with padding).

        Sample format:
        | [PAD] [PAD] prompt_token ... | response_token ... [EOS] [PAD] |

        :param outputs: Engine outputs
        :type outputs: List
        :param prompts: Text prompts
        :type prompts: List[str]
        :param labels: Sample labels
        :type labels: Optional[List]
        :param references: Reference texts
        :type references: Optional[List]
        :param is_multimodal: Whether in VLM mode
        :type is_multimodal: bool
        :param kwargs: Additional VLM-specific arguments
        :type kwargs: dict
        :return: Tuple of (Samples/SamplesVL object, updated image_patch_idx, updated video_patch_idx)
        :rtype: Tuple[Union[Samples, SamplesVL], Optional[int], Optional[int]]
        """
        # Find max lengths
        max_input_len = max(len(out.prompt_token_ids) for out in outputs)
        max_output_len = max(len(out.output_token_ids) for out in outputs)

        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id

        sequences = []
        all_output_ids = []

        # VLM data structures
        if is_multimodal:
            pixel_values = []
            image_grid_thw_list = []
            all_img_num = []

            pixel_values_videos = []
            video_grid_thw_list = []
            all_vid_num = []

            grid_thw = kwargs["grid_thw"]
            raw_images = kwargs["raw_images"]
            pixel_values_tensor = kwargs["pixel_values"]
            images_num = kwargs["images_num"]
            image_patch_idx = kwargs["image_patch_idx"]

            video_grid_thw = kwargs["video_grid_thw"]
            pixel_values_videos_tensor = kwargs["pixel_values_videos"]
            videos_num = kwargs["videos_num"]
            video_patch_idx = kwargs["video_patch_idx"]

            local_grid_idx = 0
            local_video_grid_idx = 0

        # Process each output
        for j, output in enumerate(outputs):
            # Left-pad input
            input_len = len(output.prompt_token_ids)
            input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

            # Right-pad output
            output_len = len(output.output_token_ids)
            output_ids = list(output.output_token_ids) + [pad_token_id] * (max_output_len - output_len)
            all_output_ids.append(output.output_token_ids)

            # Process images/videos for this sample
            if is_multimodal:
                if images_num is not None:
                    image_num = images_num[j]
                    all_img_num.append(image_num)

                    for img_idx in range(image_num):
                        grid = grid_thw[local_grid_idx + img_idx]
                        num_patch = grid[0] * grid[1] * grid[2]
                        image_grid_thw_list.append(grid.clone().unsqueeze(0))

                        if num_patch > 0:
                            pixel_slice = pixel_values_tensor[image_patch_idx:image_patch_idx + num_patch]
                            pixel_values.append(pixel_slice.clone())
                        image_patch_idx += num_patch

                    local_grid_idx += image_num

                if videos_num is not None:
                    video_num = videos_num[j]
                    all_vid_num.append(video_num)

                    for vid_idx in range(video_num):
                        grid = video_grid_thw[local_video_grid_idx + vid_idx]
                        num_patch = grid[0] * grid[1] * grid[2]
                        video_grid_thw_list.append(grid.clone().unsqueeze(0))

                        if num_patch > 0:
                            pixel_slice = pixel_values_videos_tensor[video_patch_idx:video_patch_idx + num_patch]
                            pixel_values_videos.append(pixel_slice.clone())
                        video_patch_idx += num_patch

                    local_video_grid_idx += video_num  # Concatenate input and output
            sequences.append(input_ids + output_ids)

        # Decode output texts
        output_texts = self.tokenizer.batch_decode(all_output_ids)

        # Process sequences
        sequences = torch.tensor(sequences)
        sequences, attention_mask, action_mask = self.actor.process_sequences(
            sequences, max_input_len, eos_token_id, pad_token_id
        )
        sequences = sequences.to("cuda")
        attention_mask = attention_mask.to("cuda")
        action_mask = action_mask.to("cuda")

        if not is_multimodal:
            return Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
                prompts=prompts,
                labels=labels,
                references=references,
                pad_len=None,
            ), None, None  # Return None for patch indices
        else:
            # Process VLM pixel values
            pixel_values = (
                torch.cat(pixel_values, dim=0).cuda() if pixel_values and pixel_values[0].shape[0] > 0 else None
            )
            pixel_values_videos = (
                torch.cat(pixel_values_videos, dim=0).cuda()
                if pixel_values_videos and pixel_values_videos[0].shape[0] > 0 else None
            )

            return SamplesVL(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                image_grid_thws=(torch.cat(image_grid_thw_list, dim=0).to("cuda") if image_grid_thw_list else None),
                video_grid_thws=(torch.cat(video_grid_thw_list, dim=0).to("cuda") if video_grid_thw_list else None),
                raw_images=raw_images,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
                references=references,
                labels=labels,
                prompts=prompts,
                output_texts=output_texts,
                image_num=all_img_num,
                video_num=all_vid_num,
            ), image_patch_idx, video_patch_idx

    def _build_packed_sample(
        self,
        outputs: List,
        prompts: List[str],
        labels: Optional[List],
        references: Optional[List],
    ) -> Samples:
        """
        Build packed sample (multiple sequences concatenated without padding).

        Sample format:
        | prompt1 response1 [EOS] | prompt2 response2 [EOS] | prompt3 ... |

        :param outputs: Engine outputs
        :type outputs: List
        :param prompts: Text prompts
        :type prompts: List[str]
        :param labels: Sample labels
        :type labels: Optional[List]
        :param references: Reference texts
        :type references: Optional[List]
        :return: Samples object with packed sequences
        :rtype: Samples
        """
        sequences = []
        packed_seq_lens = []
        attention_mask = []
        num_actions = []

        for idx, output in enumerate(outputs):
            input_len = len(output.prompt_token_ids)
            output_len = len(output.output_token_ids)
            packed_seq_lens.append(input_len + output_len)

            sequences.extend(output.prompt_token_ids + list(output.output_token_ids))
            attention_mask.extend([idx + 1] * (input_len + output_len))
            num_actions.append(max(1, output_len))

        sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
        attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
        response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
        total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)

        return Samples(
            sequences=sequences,
            attention_mask=attention_mask,
            action_mask=None,
            num_actions=num_actions,
            packed_seq_lens=packed_seq_lens,
            response_length=response_length,
            total_length=total_length,
            prompts=prompts,
            labels=labels,
            references=references,
            pad_len=None,
        )
