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
from typing import Callable, Dict, List, Tuple, Union, Optional
from dataclasses import dataclass
from copy import deepcopy

import torch
from PIL import Image
from easydict import EasyDict
from vllm import SamplingParams

from lightrft.models.utils import (
    compute_approx_kl,
    compute_reward,
    masked_mean,
    unpacking_samples,
)
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
from .utils import RunningMoments, compute_clip_fraction, get_cpgd_advantages_returns, fire_sampling

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
        pixel_values_intern: Processed pixel values (InternVL format)
        image_grid_thw: Image grid dimensions [temporal, height, width]
        image_flags: Flags indicating image presence (InternVL)
        raw_images: Original PIL images
        references: Reference texts for evaluation
        img_num: Number of images per sample

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
    pixel_values_intern: Optional[torch.Tensor] = None
    image_grid_thw: Optional[torch.Tensor] = None
    image_flags: Optional[torch.Tensor] = None
    raw_images: Optional[list] = None
    references: Optional[list] = None
    img_num: Optional[List[int]] = None

    # Model outputs
    action_log_probs: Optional[torch.Tensor] = None
    base_action_log_probs: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None
    rewards: Optional[torch.Tensor] = None
    reward_metrics: Optional[Dict[str, torch.Tensor]] = None  # Detailed reward metrics
    kl: Optional[torch.Tensor] = None
    inputs_extra_kwargs: Optional[dict] = None
    prompt_and_output: Optional[List[str]] = None


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
        :param processor: Multimodal processor (e.g., Qwen-VL, InternVL processor)
        :type processor: Union[transformers.ProcessorMixin, Any]
        :param prompt_max_len: Maximum allowed prompt length
        :type prompt_max_len: int
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_max_len = prompt_max_len

    def normalize_images(self, raw_images: List) -> List:
        """
        Recursively normalize image inputs to PIL.Image format.

        Handles various input formats:
            - PIL.Image: returned unchanged
            - dict with 'image' key: extracts image value
            - str/pathlib.Path: loads from file path (supports file:// prefix)
            - bytes/bytearray: loads from binary data
            - list: recursively normalizes each element

        :param raw_images: List of image inputs in various formats
        :type raw_images: List[Union[PIL.Image.Image, dict, str, pathlib.Path, bytes, bytearray, List]]
        :return: List of normalized PIL.Image objects (or lists of PIL.Images for multi-image samples)
        :rtype: List[Union[PIL.Image.Image, List[PIL.Image.Image]]]
        """
        normalized = []
        for item in raw_images:
            if isinstance(item, list):
                # Multi-image case: recursively normalize each image
                normalized.append([self._to_pil(img) for img in item])
            else:
                # Single image case
                normalized.append(self._to_pil(item))
        return normalized

    def _to_pil(self, img) -> Image.Image:
        """
        Convert a single image input to PIL.Image format.

        :param img: Image input (PIL.Image, dict, str, Path, bytes, or bytearray)
        :type img: Union[PIL.Image.Image, dict, str, pathlib.Path, bytes, bytearray]
        :return: PIL.Image in RGB mode
        :rtype: PIL.Image.Image
        :raises ValueError: If input type is not supported
        """
        # Extract from dict if needed
        if isinstance(img, dict):
            img = img.get("image", img)

        # Load from file path
        if isinstance(img, (str, pathlib.Path)):
            img = str(img)
            if img.startswith("file://"):
                img = img[7:]
            img = os.path.expanduser(img)
            img = Image.open(img).convert("RGB")

        # Load from binary data
        if isinstance(img, (bytes, bytearray)):
            from io import BytesIO
            img = Image.open(BytesIO(img)).convert("RGB")

        return img

    def get_images_num(self, all_images: Optional[List]) -> Optional[List[int]]:
        """
        Extract the number of images for each sample.

        IMPORTANT: Text-only samples (None) return 1 as placeholder, not 0.
        This maintains compatibility with the grid_thw slicing logic where
        text-only samples have [0,0,0] grids.

        :param all_images: List of images (can be None, single images, or lists of images)
        :type all_images: Optional[List[Union[None, PIL.Image.Image, List[PIL.Image.Image]]]]
        :return: List of image counts per sample, or None if no images
        :rtype: Optional[List[int]]
        """
        if all_images is None:
            return None

        if isinstance(all_images[0], list):
            # Multi-image case: use 1 for None (text-only) as placeholder
            return [len(img) if img is not None else 1 for img in all_images]
        else:
            # Single image case
            return [1 for _ in all_images]

    def process_multimodal_batch(
        self,
        all_prompts: List[str],
        all_images: List,
        all_references: Optional[List[str]],
        images_num: List[int],
        n_samples_per_prompt: int,
        is_internvl: bool,
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
        :param is_internvl: Whether using InternVL processor
        :type is_internvl: bool
        :return: Dictionary containing processed data
        :rtype: EasyDict
        """
        N = n_samples_per_prompt
        L = len(all_prompts)

        # ===== Stage 1: Separation =====
        all_prompts_text, all_prompts_image_text = [], []
        all_images_valid = []
        text_idx = []

        for idx, (prompt, image) in enumerate(zip(all_prompts, all_images)):
            if image is None:
                all_prompts_text.append(prompt)
                text_idx.append(idx)
            elif isinstance(image, list):
                all_prompts_image_text.append(prompt)
                all_images_valid.append(image)
            else:
                raise ValueError(f"Invalid image type: {type(image)}")

        # ===== Stage 2: Expansion =====
        all_prompts_text = sum([[p] * N for p in all_prompts_text], [])
        all_prompts_image_text = sum([[p] * N for p in all_prompts_image_text], [])
        all_images_valid = [img for img in all_images_valid for _ in range(N)]
        all_images_num = sum([[num] * N for num in images_num], [])

        # ===== Stage 3-A: Text-only processing =====
        if len(all_prompts_text) > 0:
            inputs_text = self.tokenizer(
                all_prompts_text,
                max_length=self.prompt_max_len,
                truncation=True,
                add_special_tokens=False,
            )
            all_prompt_token_ids_text = inputs_text["input_ids"]
            all_images_grid_thw_text = torch.cat(
                [torch.tensor([[0, 0, 0]]) for _ in range(len(all_prompt_token_ids_text))],
                dim=0,
            )
        else:
            all_prompt_token_ids_text = []
            all_images_grid_thw_text = torch.empty((0, 3), dtype=torch.long)

        all_image_flags_text = [None] * len(all_prompt_token_ids_text)

        # ===== Stage 3-B: Image-text processing =====
        if len(all_prompts_image_text) > 0:
            assert self.processor is not None, "Processor required for multimodal data"
            inputs_image_text = self.processor(
                text=all_prompts_image_text.copy(),
                images=all_images_valid,
                add_special_tokens=False,
                max_length=self.prompt_max_len,
                truncation=True,
            )

            all_prompt_token_ids_image_text = inputs_image_text["input_ids"]
            all_images_pixel_values_image_txt = inputs_image_text["pixel_values"]

            if is_internvl:
                all_images_grid_thw_image_text = inputs_image_text["num_tiles"]
                all_image_flags_image_text = inputs_image_text["image_flags"]
            else:
                all_images_grid_thw_image_text = inputs_image_text["image_grid_thw"]
                all_image_flags_image_text = [None] * len(all_images_grid_thw_image_text)
        else:
            all_prompt_token_ids_image_text = []
            all_images_pixel_values_image_txt = torch.empty(0, 1176)
            all_images_grid_thw_image_text = torch.empty((0, 3), dtype=torch.long)
            all_image_flags_image_text = []

        # ===== Stage 4: Merge back in original order =====
        total_samples = L * N
        all_prompts_out = [None] * total_samples
        all_images_out = [None] * total_samples
        all_prompt_token_ids_out = [None] * total_samples
        all_images_grid_thw_list = [None] * total_samples
        all_image_flags_out = [None] * total_samples

        # 4-A: Fill text-only
        text_ptr = 0
        for orig_idx in text_idx:
            for n in range(N):
                gid = orig_idx * N + n
                all_prompts_out[gid] = all_prompts_text[text_ptr]
                all_prompt_token_ids_out[gid] = all_prompt_token_ids_text[text_ptr]
                all_images_grid_thw_list[gid] = all_images_grid_thw_text[text_ptr]
                all_image_flags_out[gid] = all_image_flags_text[text_ptr]
                text_ptr += 1

        # 4-B: Fill image-text
        img_ptr = 0
        for orig_idx in range(L):
            if orig_idx in text_idx:
                continue
            for n in range(N):
                gid = orig_idx * N + n
                all_prompts_out[gid] = all_prompts_image_text[img_ptr]
                all_images_out[gid] = all_images_valid[img_ptr]
                all_prompt_token_ids_out[gid] = all_prompt_token_ids_image_text[img_ptr]
                all_images_grid_thw_list[gid] = all_images_grid_thw_image_text[img_ptr]
                all_image_flags_out[gid] = all_image_flags_image_text[img_ptr]
                img_ptr += 1

        # Stack grid_thw
        all_images_grid_thw = (
            torch.stack(all_images_grid_thw_list, dim=0)
            if len(all_images_grid_thw_list) > 0 else torch.empty((0, 3), dtype=torch.long)
        )

        # Expand references
        if all_references is not None:
            all_references = sum([[ref] * N for ref in all_references], [])

        return EasyDict(
            all_prompt_token_ids=all_prompt_token_ids_out,
            all_prompts=all_prompts_out,
            all_images=all_images_out,
            all_images_num=all_images_num,
            all_images_pixel_values=all_images_pixel_values_image_txt,
            all_images_grid_thw=all_images_grid_thw,
            all_image_flags=all_image_flags_out,
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
        # all_rewards_list[rm_idx][micro_batch_idx] = Tensor(batch_size,)
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
    ) -> List[torch.Tensor]:
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
        :return: List of reward tensors, one per micro-batch
        :rtype: List[torch.Tensor]
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
    ) -> List[torch.Tensor]:
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
        :return: List of reward tensors per micro-batch
        :rtype: List[torch.Tensor]
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
            "img_num": [],
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
                    flat_data["img_num"].append(output.img_num[samp_idx])
                    flat_data["references"].append(output.references[samp_idx])
                    flat_data["labels"].append(output.labels[samp_idx])

        # ========== Process Stage: Compute or skip ==========
        if not needed_positions:
            # No samples need this RM, return zeros for all micro-batches
            return [torch.zeros(len(output.labels), dtype=torch.float32, device=device) for output in outputs]

        # Run single forward pass on filtered samples
        rm_output = rm(
            None,
            None,
            prompt_and_outputs=flat_data["prompt_and_output"],
            raw_images=flat_data["raw_images"],
            img_num=flat_data["img_num"],
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

        return micro_batch_rewards

    def _compute_batched_custom_engine_rewards(
        self,
        rm,
        outputs: List[_SamplesOutput],
        device: torch.device,  # noqa: ARG002 (unused but kept for API consistency)
    ) -> List[torch.Tensor]:
        """
        Compute rewards using custom engine with full batch processing (legacy path).

        :param rm: Custom engine reward model
        :type rm: Any
        :param outputs: Sample outputs
        :type outputs: List[_SamplesOutput]
        :param device: Target device (unused but kept for API consistency)
        :type device: torch.device
        :return: List of reward tensors per micro-batch
        :rtype: List[torch.Tensor]
        """
        # Flatten all micro-batches into single batch
        flat_data = {
            "prompt_and_output": [],
            "raw_images": [],
            "img_num": [],
            "references": [],
            "labels": [],
        }

        for output in outputs:
            flat_data["prompt_and_output"].extend(output.prompt_and_output)
            flat_data["raw_images"].extend(output.raw_images)
            flat_data["img_num"].extend(output.img_num)
            flat_data["references"].extend(output.references)
            flat_data["labels"].extend(output.labels)

        # Single forward pass
        rm_output = rm(
            None,
            None,
            prompt_and_outputs=flat_data["prompt_and_output"],
            raw_images=flat_data["raw_images"],
            img_num=flat_data["img_num"],
            references=flat_data["references"],
            labels=flat_data["labels"],
        )
        all_scores = rm_output["score"] if isinstance(rm_output, dict) else rm_output

        # Split back into micro-batches
        batch_sizes = [len(output.prompt_and_output) for output in outputs]
        return list(all_scores.split(batch_sizes))

    def _compute_standard_torch_rewards(
        self,
        rm,
        outputs: List[_SamplesOutput],
        vlm_mode: bool,  # noqa: ARG002 (kept for future VLM-specific logic)
        device: torch.device,
    ) -> List[torch.Tensor]:
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
        :return: List of reward tensors per micro-batch
        :rtype: List[torch.Tensor]
        """
        micro_batch_rewards = []

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
                img_num=output.img_num,
                **output.inputs_extra_kwargs,
            )

            score = rm_output["score"] if isinstance(rm_output, dict) else rm_output
            micro_batch_rewards.append(torch.as_tensor(score, dtype=torch.float32, device=device))

        return micro_batch_rewards

    def _aggregate_rewards(
        self,
        outputs: List[_SamplesOutput],
        all_rewards_list: List[List[torch.Tensor]],
        is_multi_rm: bool,
    ) -> None:
        """
        Aggregate rewards from multiple RMs and store in outputs.

        :param outputs: Sample outputs (modified in-place)
        :type outputs: List[_SamplesOutput]
        :param all_rewards_list: Nested list [rm_idx][micro_batch_idx] -> Tensor
        :type all_rewards_list: List[List[torch.Tensor]]
        :param is_multi_rm: Whether using multiple reward models
        :type is_multi_rm: bool
        """
        num_micro_batches = len(outputs)
        num_rms = len(all_rewards_list)

        for mb_idx in range(num_micro_batches):
            # Collect rewards from all RMs for this micro-batch
            same_batch_rewards = [all_rewards_list[rm_idx][mb_idx] for rm_idx in range(num_rms)]

            if is_multi_rm:
                # Use custom aggregation function
                sequences = (
                    outputs[mb_idx].sequences if not self.packing_samples else
                    unpacking_samples(outputs[mb_idx].sequences, outputs[mb_idx].packed_seq_lens)
                )
                queries = self.tokenizer.batch_decode(sequences, skip_special_tokens=False)

                rewards, reward_metrics = self.reward_fn(
                    model_reward_list=same_batch_rewards,
                    labels=outputs[mb_idx].labels,
                    queries=queries,
                    refs=outputs[mb_idx].references,
                    label_map=self.reward_fn_label_map,
                )
                outputs[mb_idx].rewards = rewards
                outputs[mb_idx].reward_metrics = reward_metrics
            else:
                # Single RM, use score directly
                outputs[mb_idx].rewards = same_batch_rewards[0]
                outputs[mb_idx].reward_metrics = None


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
    def __init__(self, *args, packing_samples: bool = False, processor=None, **kwargs):
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

        # Initialize tokenizer (extract from processor if needed)
        if self.processor is not None:
            self.tokenizer = getattr(self.processor, "tokenizer", self.processor)

        # Initialize running reward normalization
        if self.strategy.args.reward_running_norm:
            self.reward_running_moments = RunningMoments()
        else:
            self.reward_running_moments = None

        # Initialize helper modules
        if self.processor is not None:
            self.multimodal_processor = MultimodalDataProcessor(
                tokenizer=self.tokenizer,
                processor=self.processor,
                prompt_max_len=self.prompt_max_len,
            )
        else:
            self.multimodal_processor = None

        self.reward_engine = RewardComputationEngine(
            reward_model=self.reward_model,
            remote_rm_url=self.remote_rm_url,
            custom_reward_func=getattr(self, "custom_reward_func", None),
            reward_fn=self.reward_fn,
            reward_fn_label_map=getattr(self, "reward_fn_label_map", None),
            reward_recipe=getattr(self, "reward_recipe", None),
            tokenizer=self.tokenizer,
            strategy=self.strategy,
            packing_samples=self.packing_samples,
        )

    # ========================================================================
    # Public API Methods
    # ========================================================================

    @torch.no_grad()
    def make_experience_list(
        self,
        all_prompts: List[str],
        all_images: Optional[List] = None,
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
        :param generate_kwargs: Generation parameters (temperature, max_new_tokens, etc.)
        :type generate_kwargs: dict
        :return: List of Experience or ExperienceVL objects with computed advantages and returns
        :rtype: List[Union[Experience, ExperienceVL]]
        """
        config = self.strategy.config

        # Normalize images if provided
        if all_images is not None:
            if self.multimodal_processor is None:
                raise ValueError(
                    "Multimodal data (images) provided but processor was not initialized. "
                    "Please provide a processor when initializing FastExperienceMaker for VLM support."
                )
            all_images = self.multimodal_processor.normalize_images(all_images)

        # Get image counts
        images_num = (
            self.multimodal_processor.get_images_num(all_images)
            if self.multimodal_processor and all_images is not None else None
        )

        # ========== Stage 1: Sample Generation ==========
        Timer.start('  generate_samples')
        samples_list = self.generate_samples(
            all_prompts,
            all_images=all_images,
            images_num=images_num,
            all_references=all_references,
            all_labels=all_labels,
            **generate_kwargs,
        )
        Timer.stop('  generate_samples')

        torch.distributed.barrier()
        torch.cuda.synchronize()

        # ========== Stage 2: Shard-Parallel Preprocessing ==========
        all_samples = self.strategy.sp_data_processor.preprocess(samples_list)

        # ========== Stage 3: Model Inference ==========
        Timer.start('  make_experience')
        experiences = self._make_experience_list_by_model(all_samples)
        Timer.stop('  make_experience')

        # ========== Stage 4: Shard-Parallel Postprocessing ==========
        experiences = self.strategy.sp_data_processor.postprocess(experiences)

        # ========== Stage 5: Reward Processing ==========
        experiences, rewards = self._process_experiences(  # GRPO's -mean / std operation is performed in this method
            experiences, generate_kwargs.get("max_new_tokens", 1024)
        )

        # ========== Stage 6: Multi-Image Handling ==========
        if images_num is not None and not all(num == 1 for num in images_num):
            # Expand image_num by n_samples_per_prompt
            expanded_images_num = sum([[num] * config.n_samples_per_prompt for num in images_num], [])
            self._process_multi_image_thws(experiences, expanded_images_num)

        # ========== Stage 7: Advantage Computation ==========
        experiences = self._compute_advantages_and_returns(experiences, rewards, generate_kwargs)

        return experiences

    @torch.no_grad()
    def generate_samples(
        self,
        all_prompts: List[str],
        all_images: Optional[List] = None,
        images_num: Optional[List[int]] = None,
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
        :param generate_kwargs: Generation parameters (temperature, max_new_tokens, etc.)
        :type generate_kwargs: dict
        :return: List of Samples or SamplesVL objects
        :rtype: List[Union[Samples, SamplesVL]]
        """
        assert self.strategy.inference_engine is not None, "Inference engine required"

        torch.cuda.synchronize()
        start_time = time.time()

        config = self.strategy.config
        is_multimodal = all_images is not None
        n_samples = config.n_samples_per_prompt

        if is_multimodal:
            is_internvl = "internvl" in self.actor.pretrain_or_model.lower()

        # ========== Configure Sampling Parameters ==========
        if config.engine_type == "vllm":
            sampling_params = SamplingParams(
                temperature=generate_kwargs.get("temperature", 1.0),
                top_p=generate_kwargs.get("top_p", 1.0),
                top_k=generate_kwargs.get("top_k", -1),
                max_tokens=generate_kwargs.get("max_new_tokens", 1024),
                min_tokens=generate_kwargs.get("min_new_tokens", 1),
                skip_special_tokens=generate_kwargs.get("skip_special_tokens", False),
                include_stop_str_in_output=True,
                ignore_eos=os.environ.get("IGNORE_EOS", "0") == "1",
                truncate_prompt_tokens=8192,
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
                repetition_penalty=1.0,
                skip_special_tokens=generate_kwargs.get("skip_special_tokens", False),
                spaces_between_special_tokens=True,
                ignore_eos=os.environ.get("IGNORE_EOS", "0") == "1",
            )
        else:
            raise ValueError(f"Unsupported engine type: {config.engine_type}")

        # ========== Expand Labels ==========
        if all_labels is not None:
            all_labels = sum([[label] * n_samples for label in all_labels], [])

        # ========== Process Multimodal Data ==========
        if is_multimodal:
            processed_data = self.multimodal_processor.process_multimodal_batch(
                all_prompts=all_prompts,
                all_images=all_images,
                all_references=all_references,
                images_num=images_num,
                n_samples_per_prompt=n_samples,
                is_internvl=is_internvl,
            )
            all_prompt_token_ids = processed_data["all_prompt_token_ids"]
            all_prompts = processed_data["all_prompts"]
            all_images = processed_data["all_images"]
            all_images_num = processed_data["all_images_num"]
            all_images_pixel_values = processed_data["all_images_pixel_values"]
            all_images_grid_thw = processed_data["all_images_grid_thw"]
            all_image_flags = processed_data["all_image_flags"]
            all_references = processed_data["all_references"]
        else:
            # Text-only processing
            tokenized = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)
            all_prompt_token_ids = sum([[token_ids] * n_samples for token_ids in tokenized["input_ids"]], [])

        # ========== Generate via Inference Engine ==========
        # Call fire_sampling function or direct generation
        try:
            if hasattr(self.strategy.args, 'use_fire') and self.strategy.args.use_fire:
                # Use FIRE sampling (Flaming-hot Initiation with Regular Execution)
                all_outputs = fire_sampling(
                    all_prompt_token_ids=all_prompt_token_ids,
                    generate_fn=generate_fn,  # noqa: TODO
                    engine_type=config.engine_type,
                    first_token_temperature=generate_kwargs.get("first_token_temperature", 10.0),
                    temperature=generate_kwargs.get("temperature", 1.0),
                    first_token_top_k=generate_kwargs.get(
                        "first_token_top_k", sampling_params.top_k if hasattr(sampling_params, 'top_k') else -1
                    ),
                    first_token_top_p=generate_kwargs.get(
                        "first_token_top_p", sampling_params.top_p if hasattr(sampling_params, 'top_p') else 1.0
                    ),
                    is_multimodal=is_multimodal,
                    all_prompts=all_prompts,
                    all_images=all_images,
                    all_images_num=all_images_num,
                    sampling_params=sampling_params,
                )
            else:
                # maybe this can be called in if and else respectively? or like this?
                # Use original single-shot generation
                all_outputs = self.strategy.gather_and_generate(
                    sampling_params=sampling_params,
                    all_prompt_token_ids=all_prompt_token_ids,
                    all_prompts=all_prompts if is_multimodal else None,
                    all_images=all_images,
                    sleep_engine=True,
                    images_num=all_images_num if is_multimodal else None,
                )
        except ValueError as e:
            if "prompt" in str(e) and "too long" in str(e):
                self.strategy.print(f"[Skip] {e}")
                return None  # Return None, subsequent experience_maker will ignore
            else:
                raise

        # ========== Process Outputs into Samples ==========
        samples_list = []
        image_patch_idx = 0
        image_start_idx = 0

        for i in range(0, len(all_outputs), config.micro_rollout_batch_size):
            micro_batch_outputs = all_outputs[i:i + config.micro_rollout_batch_size]
            micro_batch_prompts = all_prompts[i:i + config.micro_rollout_batch_size]

            # Extract micro-batch data
            if is_multimodal:
                rollout_image_count = sum(all_images_num[i:i + config.micro_rollout_batch_size])
                micro_batch_grid_thw = all_images_grid_thw[image_start_idx:image_start_idx + rollout_image_count]
                micro_batch_raw_images = all_images[image_start_idx:image_start_idx + rollout_image_count]
                image_start_idx += rollout_image_count

            micro_batch_references = (all_references[i:i + config.micro_rollout_batch_size] if all_references else None)
            micro_batch_labels = (all_labels[i:i + config.micro_rollout_batch_size] if all_labels else None)

            # Build samples
            if not self.packing_samples:
                sample, updated_patch_idx = self._build_unpacked_sample(
                    outputs=micro_batch_outputs,
                    prompts=micro_batch_prompts,
                    labels=micro_batch_labels,
                    references=micro_batch_references,
                    is_multimodal=is_multimodal,
                    grid_thw=micro_batch_grid_thw if is_multimodal else None,
                    raw_images=micro_batch_raw_images if is_multimodal else None,
                    pixel_values=all_images_pixel_values if is_multimodal else None,
                    image_flags=all_image_flags if is_multimodal else None,
                    images_num=all_images_num[i:i + config.micro_rollout_batch_size] if is_multimodal else None,
                    is_internvl=is_internvl if is_multimodal else False,
                    image_patch_idx=image_patch_idx,
                )
                # Update image_patch_idx from the returned value
                if updated_patch_idx is not None:
                    image_patch_idx = updated_patch_idx
                samples_list.append(sample)
            else:
                # Packed samples
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

    def _process_multi_image_thws(
        self,
        experiences: List[ExperienceVL],
        images_num: List[int],
    ) -> None:
        """
        Process image_grid_thws for multi-image scenarios.

        Ensures len(experience.sequences) == len(experience.image_grid_thws) by
        converting the stacked tensor into a list of per-sequence tensors.

        :param experiences: List of experiences to modify in-place
        :type experiences: List[ExperienceVL]
        :param images_num: Number of images per sample (expanded by n_samples_per_prompt)
        :type images_num: List[int]
        """
        config = self.strategy.config

        for i, experience in enumerate(experiences):
            # Get image counts for this micro-batch
            start_idx = i * config.micro_rollout_batch_size
            end_idx = (i + 1) * config.micro_rollout_batch_size
            micro_images_num = images_num[start_idx:end_idx]

            # Split image_grid_thws tensor
            image_grid_thw_list = []
            image_grid_thws = experience.image_grid_thws
            assert isinstance(image_grid_thws, torch.Tensor)
            image_grid_thws = torch.unbind(image_grid_thws)

            # Verify total image count matches
            assert sum(micro_images_num) == len(image_grid_thws)

            # Group by sequence
            thw_idx = 0
            for num in micro_images_num:
                stacked_thw = torch.stack(image_grid_thws[thw_idx:thw_idx + num], dim=0).to("cuda")
                image_grid_thw_list.append(stacked_thw)
                thw_idx += num

            experience.image_grid_thws = image_grid_thw_list

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
        if config.advantage_estimator == "rloo":
            # RLOO: Leave-one-out baseline
            rewards = rewards.reshape(-1, config.n_samples_per_prompt).to("cuda")
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (config.n_samples_per_prompt - 1)
            rewards = rewards - baseline
            rewards = rewards.flatten().to("cpu").chunk(len(experiences))
            return experiences, rewards

        elif config.advantage_estimator == "reinforce_baseline":
            # REINFORCE with baseline (mean subtraction)
            rewards = rewards.reshape(-1, config.n_samples_per_prompt).to("cuda")
            rewards = rewards - rewards.mean(-1, keepdim=True)
            rewards = rewards.flatten().to("cpu").chunk(len(experiences))
            return experiences, rewards

        elif config.advantage_estimator in ["group_norm", "grpo"]:
            # Group normalization with optional dynamic filtering
            if config.dynamic_sampling:
                step_size = config.n_samples_per_prompt // config.micro_train_batch_size
                for i in range(0, len(experiences), step_size):
                    chunk = experiences[i:i + step_size]
                    chunk_rewards = torch.cat([exp.info["reward"] for exp in chunk])

                    # Filter out degenerate cases (all 0s or all 1s)
                    if torch.all(chunk_rewards == 0) or torch.all(chunk_rewards == 1):
                        for exp in chunk:
                            exp.action_mask = torch.zeros_like(exp.action_mask, dtype=torch.bool)

            # # Normalize within groups
            # rewards = rewards.reshape(-1, config.n_samples_per_prompt).to("cuda")
            # rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)
            # # rewards = rewards.flatten().to("cpu").chunk(len(experiences))
            # rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))

            # import torch.distributed as dist
            # if dist.get_rank() == 0 and DEBUG_ENABLED:
            #     print(f"rank {dist.get_rank()} Entering debug mode, input 'interact' to enter full Python debugging. Set DEBUG_ENABLED = False to skip debug mode")  # noqa
            #     import ipdb; ipdb.set_trace()
            # # Synchronization point to prevent other processes from running ahead
            # dist.barrier()

            # Ensure rewards are float32. If rewards are float16 or bfloat16, 1e-9 may
            # underflow to 0, causing division by zero (NaN).
            # Normalize within groups
            rewards = rewards.reshape(-1, config.n_samples_per_prompt).to("cuda")
            baseline = rewards.mean(-1, keepdim=True)
            rewards = (rewards - baseline) / (rewards.std(1, keepdim=True) + 1e-9)
            rewards = rewards.flatten().to("cpu").chunk(len(experiences))

            return experiences, rewards

        elif config.advantage_estimator == "cpgd":
            return experiences, [experience.info["reward"] for experience in experiences]

        else:
            raise ValueError(f"Unknown advantage_estimator: {config.advantage_estimator}")

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
            final_reward = compute_reward(
                processed_reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                num_actions=experience.info["num_actions"],
            )

            # ========== Advantage Estimation ==========
            if self.advantage_estimator == "cpgd":
                experience.advantages, experience.returns = get_cpgd_advantages_returns(
                    final_reward, experience.action_mask
                )

            elif self.advantage_estimator == "gae":
                experience.advantages, experience.returns, experience.info["advantage_clip_frac"] = (
                    self.get_advantages_and_returns(
                        experience.values,
                        final_reward,
                        experience.action_mask,
                        generate_kwargs["gamma"],
                        generate_kwargs["lambd"],
                    )
                )

            elif self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm"]:
                # Compute cumulative returns
                experience.returns = self.get_cumulative_returns(
                    final_reward, experience.action_mask, generate_kwargs["gamma"]
                )
                experience.advantages = deepcopy(experience.returns)

                # Advantage whitening
                # Whether to apply normalization over the entire Batch in the Trainer's train_step
                if config.advantages_norm:
                    masked_adv = torch.masked_select(experience.advantages, experience.action_mask)
                    adv_mean = masked_adv.mean()
                    adv_std = masked_adv.std()
                    experience.advantages = (experience.advantages - adv_mean) / (adv_std + 1e-9)

                # Advantage clipping
                if config.advantage_clip > 0:
                    experience.info["advantage_clip_frac"] = compute_clip_fraction(
                        experience.advantages, config.advantage_clip, -config.advantage_clip
                    )
                    experience.advantages = torch.clamp(
                        experience.advantages, -config.advantage_clip, config.advantage_clip
                    )

            else:
                raise ValueError(f"Unknown advantage_estimator: {self.advantage_estimator}")

            # ========== Store Episode Return ==========
            if not self.packing_samples:
                experience.info["return"] = final_reward.sum(dim=-1)
            else:
                experience.info["return"] = torch.tensor([r.sum() for r in final_reward], device=final_reward.device)

            # Cleanup
            experience.kl = None
            del experience.info["num_actions"]

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
        is_internvl = "internvl" in self.actor.pretrain_or_model.lower() if vlm_mode else False

        # ========== Stage 0: Preprocessing ==========
        outputs = [self._preprocess_sample(sample, vlm_mode, is_internvl, device) for sample in all_samples]

        # ========== Stage 1: Actor Forward ==========
        Timer.start('    actor_logprob')
        for output in outputs:
            output.action_log_probs = self.actor(
                output.sequences,
                output.num_actions,
                output.attention_mask,
                packed_seq_lens=output.packed_seq_lens,
                **output.inputs_extra_kwargs
            )
        Timer.stop('    actor_logprob')

        # ========== Stage 2: Initial Model ==========
        if self.initial_model is not None:
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

        # ========== Stage 3: Critic ==========
        if self.critic is not None:
            self.strategy.reload_model(self.critic)
            for output in outputs:
                output.value = self.critic(
                    output.sequences, output.num_actions, output.attention_mask, **output.inputs_extra_kwargs
                )
            self.strategy.offload_model(self.critic)

        # ========== Stage 4: Reward Models ==========
        self.reward_engine.compute_rewards(outputs, vlm_mode, device)

        # ========== Stage 5: Assemble Experiences ==========
        return [self._pack_experience(output, vlm_mode) for output in outputs]

    def _preprocess_sample(
        self,
        sample: Union[Samples, SamplesVL],
        vlm: bool,
        internvl: bool,
        device: torch.device,
    ) -> _SamplesOutput:
        """
        Convert a Samples object to _SamplesOutput for processing.

        :param sample: Input sample
        :type sample: Union[Samples, SamplesVL]
        :param vlm: Vision-language mode flag
        :type vlm: bool
        :param internvl: InternVL model flag
        :type internvl: bool
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

        # Build extra kwargs for VLM
        extra_kwargs = {}
        if vlm:
            if internvl:
                extra_kwargs = dict(
                    pixel_values_intern=sample.pixel_values_intern,
                    image_flags=sample.image_flags,
                )
            else:
                extra_kwargs = dict(
                    pixel_values=sample.pixel_values,
                    image_grid_thw=sample.image_grid_thws,
                )

        # Fix Qwen-VL image token count bug
        self._fix_qwen_vl_image_tokens(sequences, sample, vlm, internvl)

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
            pixel_values_intern=getattr(sample, "pixel_values_intern", None),
            image_grid_thw=getattr(sample, "image_grid_thws", None),
            raw_images=getattr(sample, "raw_images", None),
            img_num=getattr(sample, "img_num", None),
            references=references,
            inputs_extra_kwargs=extra_kwargs,
            prompt_and_output=([p + (o or "") for p, o in zip(prompts, output_texts)] if output_texts else None),
        )

    def _fix_qwen_vl_image_tokens(
        self,
        sequences: torch.Tensor,
        sample: SamplesVL,
        vlm: bool,
        internvl: bool,
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
        :param internvl: InternVL flag
        :type internvl: bool
        """
        if not vlm or internvl or sample.pixel_values is None:
            return

        config = self.strategy.unwrap_model(self.actor.model).config
        image_token_id = config.image_token_id
        num_tokens = (sequences == image_token_id).sum()
        num_patches = sample.pixel_values.shape[0] // 4

        if num_tokens != num_patches:
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

        # Create Experience object
        if vlm:
            return ExperienceVL(
                output.sequences,
                output.pixel_values,
                output.image_grid_thw,
                output.raw_images,
                output.pixel_values_intern,
                output.image_flags,
                output.action_log_probs,
                output.base_action_log_probs,
                output.value,
                None,  # advantages (filled later)
                None,  # returns (filled later)
                output.attention_mask,
                output.action_mask,
                info,
                kl,
            )
        else:
            return Experience(
                output.sequences,
                output.action_log_probs,
                output.base_action_log_probs,
                output.value,
                None,  # advantages
                None,  # returns
                output.attention_mask,
                output.action_mask,
                info,
                kl,
            )

    def _build_unpacked_sample(
        self,
        outputs: List,
        prompts: List[str],
        labels: Optional[List],
        references: Optional[List],
        is_multimodal: bool,
        **kwargs,
    ) -> Tuple[Union[Samples, SamplesVL], Optional[int]]:
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
        :return: Tuple of (Samples/SamplesVL object, updated image_patch_idx or None)
        :rtype: Tuple[Union[Samples, SamplesVL], Optional[int]]
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
            image_flags = []
            all_img_num = []

            grid_thw = kwargs["grid_thw"]
            raw_images = kwargs["raw_images"]
            pixel_values_tensor = kwargs["pixel_values"]
            image_flags_tensor = kwargs["image_flags"]
            images_num = kwargs["images_num"]
            is_internvl = kwargs["is_internvl"]
            image_patch_idx = kwargs["image_patch_idx"]

            local_grid_idx = 0

        # Process each output
        for j, output in enumerate(outputs):
            # Left-pad input
            input_len = len(output.prompt_token_ids)
            input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

            # Right-pad output
            output_len = len(output.output_token_ids)
            output_ids = list(output.output_token_ids) + [pad_token_id] * (max_output_len - output_len)
            all_output_ids.append(output.output_token_ids)

            # Process images for this sample
            if is_multimodal:
                image_num = images_num[j]
                all_img_num.append(image_num)

                for img_idx in range(image_num):
                    grid = grid_thw[local_grid_idx + img_idx]

                    if is_internvl:
                        num_patch = grid if isinstance(grid, int) else grid.sum().item()
                        flags_slice = image_flags_tensor[image_patch_idx:image_patch_idx + num_patch]
                        image_flags.append(flags_slice)
                        image_grid_thw_list.append(torch.tensor([1, 1, num_patch]).unsqueeze(0))
                    else:
                        num_patch = grid[0] * grid[1] * grid[2]
                        image_grid_thw_list.append(grid.clone().unsqueeze(0))

                    pixel_slice = pixel_values_tensor[image_patch_idx:image_patch_idx + num_patch]
                    pixel_values.append(pixel_slice.clone())
                    image_patch_idx += num_patch

                local_grid_idx += image_num

            # Concatenate input and output
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

        # Build Samples object
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
            ), None  # Return None for image_patch_idx when no images
        else:
            # Process VLM pixel values
            if is_internvl:
                pixel_values_intern = (torch.cat(pixel_values, dim=0).cuda() if pixel_values[0].shape[0] > 0 else None)
                pixel_values = None
            else:
                pixel_values = (torch.cat(pixel_values, dim=0).cuda() if pixel_values[0].shape[0] > 0 else None)
                pixel_values_intern = None

            return SamplesVL(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                image_grid_thws=(torch.cat(image_grid_thw_list, dim=0).to("cuda") if not is_internvl else None),
                raw_images=raw_images,
                pixel_values=pixel_values,
                pixel_values_intern=pixel_values_intern,
                image_flags=(torch.cat(image_flags, dim=0).to("cuda") if is_internvl else None),
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
                references=references,
                labels=labels,
                prompts=prompts,
                output_texts=output_texts,
                img_num=all_img_num,
            ), image_patch_idx  # Return updated image_patch_idx

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
