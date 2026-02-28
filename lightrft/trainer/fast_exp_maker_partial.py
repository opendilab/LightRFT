"""
PartialFastExperienceMaker Module – FastExperienceMaker with Partial Rollout and Token‑Budget Regeneration.

This module extends FastExperienceMaker to support incremental rollout and controlled generation
for long‑form or high‑cost tasks. It introduces two core mechanisms:
   1. Partial rollout: only a fraction (partial_percent) of the total rollout batch is generated
      in each call; the remaining samples are kept in buffers for subsequent steps, reducing
      per‑iteration latency and enabling smoother pipeline scheduling.
   2. Token‑budget regeneration: samples that reach a predefined token budget (max_token_budget)
      are flagged and can be regenerated later, allowing continuation of long‑form generation
      without discarding already‑produced content.

The class reuses the parent's infrastructure (MultimodalDataProcessor, RewardComputationEngine,
etc.) and only overrides the methods that implement the partial‑rollout logic.

Implementation Overview:
    - The rollout batch is split into "regeneration" and "non‑regeneration" buffers based on
      whether a sample has exhausted its token budget.
    - The method `need_new_prompts` determines whether fresh prompts are required to fill the
      partial batch.
    - Generation is performed via the parent's inference engine (VLLM/SGLang), but outputs are
      post‑processed to respect the token budget and partial fraction.
    - Advantage estimation methods (RLOO, Group Norm, REINFORCE) are adjusted to account for the
      reduced group size introduced by partial rollout.

Key Features:
    - Partial rollout with configurable fraction (partial_percent) for reduced iteration latency
    - Token‑budget regeneration for long‑form continuation (max_token_budget)
    - Buffered sample management (regen_buffer, noregen_buffer) for stateful rollout
    - Seamless integration with VLLM/SGLang backends and multimodal processing
    - Adaptive advantage estimation that respects the partial batch size
    - Support for both packed and unpacked sample formats

Parameters:
    partial_percent (float): Fraction of the total rollout batch to generate in one call.
        Values between 0.0 and 1.0. Default: 0.7.
    max_token_budget (int): Maximum allowed generation length before a sample is flagged for
        regeneration. Samples that reach this length are stored in the regeneration buffer
        and can be continued in a later step. Default: 1024.

References:
    - Kimi1.5: "Kimi k1.5: Scaling Reinforcement Learning with LLMs" (https://arxiv.org/abs/2501.12599)
    - MiMo: "MiMo: Unlocking the Reasoning Potential of Language Model 
      -- From Pretraining to Posttraining" (https://arxiv.org/abs/2505.07608)

Classes:
    PartialFastExperienceMaker: Main experience generation class with partial‑rollout support.
"""

from typing import List, Optional, Union, Tuple, Dict, Any
import os
import time
from copy import deepcopy

import torch
from vllm import SamplingParams
from easydict import EasyDict

from lightrft.trainer.experience_maker import Experience, Samples
from lightrft.trainer.experience_maker_vl import SamplesVL
from lightrft.trainer.fast_exp_maker import FastExperienceMaker
from lightrft.utils import Timer, get_current_device


class PartialFastExperienceMaker(FastExperienceMaker):
    """
    FastExperienceMaker with partial rollout and token‑budget regeneration.

    This class extends FastExperienceMaker to support incremental rollout and controlled generation
    for long‑form or high‑cost tasks. It introduces two core mechanisms:
        1. Partial rollout: only a fraction (partial_percent) of the total rollout batch is generated
           in each call; the remaining samples are kept in buffers for subsequent steps, reducing
           per‑iteration latency and enabling smoother pipeline scheduling.
        2. Token‑budget regeneration: samples that reach a predefined token budget (max_token_budget)
           are flagged and can be regenerated later, allowing continuation of long‑form generation
           without discarding already‑produced content.

    The class reuses the parent's infrastructure (MultimodalDataProcessor, RewardComputationEngine,
    etc.) and only overrides the methods that implement the partial‑rollout logic.

    The partial‑rollout pipeline:
        1. Buffer Management: Maintain regeneration (regen) and non‑regeneration (noregen) buffers
        2. Need‑Prompts Check: Determine if fresh prompts are required to fill the partial batch
        3. Generation: Use parent's inference engine (VLLM/SGLang) but respect token budget and partial fraction
        4. Regeneration: For samples that exceed token budget, regenerate with continuation
        5. Advantage Adaptation: Adjust advantage estimators (RLOO, Group Norm, REINFORCE) for partial group size

    Args:
        partial_percent: Fraction of the total rollout batch to generate in one call (0.0‑1.0)
        max_token_budget: Maximum allowed generation length before a sample is flagged for regeneration
        packing_samples: Whether to pack multiple sequences into single batch (inherited from parent)
        processor: Multimodal processor for vision‑language models (inherited from parent)
        *args, **kwargs: Arguments passed to parent FastExperienceMaker
    """

    def __init__(
        self,
        *args,
        partial_percent: float = 0.7,
        max_token_budget: int = 1024,
        packing_samples: bool = False,
        processor=None,
        **kwargs
    ):
        """
        Initialize PartialFastExperienceMaker.

        :param args: Positional arguments for parent FastExperienceMaker
        :type args: tuple
        :param partial_percent: Fraction of total rollout batch to generate per call (0.0‑1.0)
        :type partial_percent: float
        :param max_token_budget: Maximum generation length before a sample is flagged for regeneration
        :type max_token_budget: int
        :param packing_samples: Enable sample packing for efficiency (inherited)
        :type packing_samples: bool
        :param processor: Multimodal processor for vision‑language models (inherited)
        :type processor: Optional[Any]
        :param kwargs: Keyword arguments for parent FastExperienceMaker
        :type kwargs: dict
        """
        super().__init__(*args, packing_samples=packing_samples, processor=processor, **kwargs)
        self.partial_percent = partial_percent
        self.max_token_budget = max_token_budget

        # Buffers for regeneration (regen) and non‑regeneration (noregen) samples.
        # Each buffer is a dict mapping field names to lists of data.
        self.regen_buffer: Dict[str, List] = {}
        self.noregen_buffer: Dict[str, List] = {}
        fields = [
            'output', 'labels', 'prompts', 'references',
            'images', 'images_num', 'images_grid_thw', 'images_pixel_values',
            'videos', 'videos_num', 'videos_grid_thw', 'videos_pixel_values'
        ]
        for field in fields:
            self.regen_buffer[field] = []
            self.noregen_buffer[field] = []

        # Placeholders for batch‑size parameters (set by need_new_prompts)
        self.rollout_batch_size = None
        self.micro_rollout_batch_size = None

    def need_new_prompts(self, rollout_batch_size: int, micro_rollout_batch_size: int) -> bool:
        """
        Determine whether new prompts are required to fill the partial rollout batch.

        This method checks the current regeneration and non‑regeneration buffers
        and compares the total stored samples against the number needed for the
        current partial rollout (partial_percent × total rollout batch size).

        If the buffers contain insufficient samples, the caller should fetch fresh
        prompts and call generate_samples with those prompts.

        :param rollout_batch_size: Total number of samples in a full rollout batch
        :type rollout_batch_size: int
        :param micro_rollout_batch_size: Size of each micro‑batch used in generation
        :type micro_rollout_batch_size: int
        :return: True if new prompts are needed (buffers below partial threshold), else False
        :rtype: bool
        """
        self.rollout_batch_size = rollout_batch_size
        self.micro_rollout_batch_size = micro_rollout_batch_size

        # Total micro‑batches needed for a full rollout
        total_micro = rollout_batch_size // self.strategy.world_size
        # Micro‑batches we want to generate in one call
        target_micro = int(self.partial_percent * total_micro)
        required_samples = target_micro * micro_rollout_batch_size

        # Count how many samples are already stored in both buffers
        total_samples = len(self.noregen_buffer.get('output', [])) + len(self.regen_buffer.get('output', []))
        return total_samples < required_samples

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
        **generate_kwargs
    ) -> List[Samples]:
        """
        Generate samples using the parent's pipeline, but only a partial fraction.

        This method implements the partial‑rollout logic:
          1. If new prompts are provided, generate them via the parent's inference engine
             (VLLM/SGLang) using token‑budget‑limited generation.
          2. Split the generated outputs into regeneration (regen) and non‑regeneration (noregen)
             buffers based on whether they have reached the token budget.
          3. Draw from the buffers to produce the requested number of samples
             (partial_percent × total rollout batch size).
          4. If the noregen buffer is insufficient, regenerate some samples from the regen buffer
             by continuing generation from the partially‑produced output.

        The method returns a list of Samples (or SamplesVL) ready for experience making.
        When new prompts are provided, it also returns the image counts for multimodal data.

        :param all_prompts: List of text prompts (or None to only draw from buffers)
        :type all_prompts: List[str]
        :param all_images: Optional images for vision‑language models
        :type all_images: Optional[List]
        :param all_videos: Optional videos for vision‑language models
        :type all_videos: Optional[List]
        :param images_num: Number of images per prompt
        :type images_num: Optional[List[int]]
        :param videos_num: Number of videos per prompt
        :type videos_num: Optional[List[int]]
        :param all_references: Reference texts for evaluation
        :type all_references: Optional[List[str]]
        :param all_labels: Sample labels for reward shaping
        :type all_labels: Optional[List]
        :param generate_kwargs: Generation parameters (temperature, max_new_tokens, etc.)
        :type generate_kwargs: dict
        :return: List of Samples (or SamplesVL) when all_prompts is None,
                 otherwise tuple (samples_list, images_num_list)
        :rtype: Union[List[Samples], Tuple[List[Samples], Optional[List[int]]]]
        """
        assert self.strategy.inference_engine is not None, "Inference engine required"

        torch.cuda.synchronize()
        start_time = time.time()

        config = self.strategy.config
        if all_prompts is not None:
            is_multimodal = all_images is not None or all_videos is not None
        else:
            is_multimodal = (len(self.noregen_buffer.get('images', [])) + len(self.regen_buffer.get('images', []))) != 0 or \
            (len(self.noregen_buffer.get('videos', [])) + len(self.regen_buffer.get('videos', []))) != 0
        n_samples = config.n_samples_per_prompt

        # --------------------------------------------------------------------
        # Step 1: Generate new samples if inputs are provided
        # --------------------------------------------------------------------
        if all_prompts is not None:
            # Replicate the generation logic from fast_exp_maker_partial.py
            # Prepare sampling parameters
            if config.engine_type == "vllm":
                sampling_params = SamplingParams(
                    temperature=generate_kwargs.get("temperature", 1.0),
                    top_p=generate_kwargs.get("top_p", 1.0),
                    top_k=generate_kwargs.get("top_k", -1),
                    max_tokens=self.max_token_budget,  # use token budget
                    min_tokens=generate_kwargs.get("min_new_tokens", 1),
                    skip_special_tokens=generate_kwargs.get("skip_special_tokens", False),
                    include_stop_str_in_output=True,
                    ignore_eos=os.environ.get("IGNORE_EOS", "0") == "1",
                )
            elif config.engine_type == "sglang":
                sampling_params = dict(
                    n=1,
                    temperature=generate_kwargs.get("temperature", 1.0),
                    top_p=generate_kwargs.get("top_p", 1.0),
                    top_k=generate_kwargs.get("top_k", -1),
                    max_new_tokens=self.max_token_budget,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    repetition_penalty=1.0,
                    skip_special_tokens=generate_kwargs.get("skip_special_tokens", False),
                    spaces_between_special_tokens=True,
                    ignore_eos=os.environ.get("IGNORE_EOS", "0") == "1",
                )
            else:
                raise ValueError(f"Unsupported backend: {config.engine_type}")

            # Expand labels
            expanded_labels = sum([[label] * n_samples for label in all_labels], []) if all_labels else []

            # Process multimodal data
            if is_multimodal:
                processed = self.multimodal_processor.process_multimodal_batch(
                    all_prompts=all_prompts,
                    all_images=all_images,
                    all_videos=all_videos,
                    all_references=all_references,
                    images_num=images_num,
                    videos_num=videos_num,
                    n_samples_per_prompt=n_samples,
                )
                prompt_token_ids = processed["all_prompt_token_ids"]
                prompts = processed["all_prompts"]
                images = processed["all_images"]
                images_num = processed["all_images_num"]
                images_pixel_values = processed["all_images_pixel_values"]
                images_grid_thw = processed["all_images_grid_thw"]
                references = processed["all_references"]
                videos = processed.get("all_videos")
                videos_num = processed.get("all_videos_num")
                videos_pixel_values = processed.get("all_videos_pixel_values")
                videos_grid_thw = processed.get("all_videos_grid_thw")
            else:
                # Text-only processing
                tokenized = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)
                prompt_token_ids = sum([[token_ids] * n_samples for token_ids in tokenized["input_ids"]], [])

            # ========== Generate via Inference Engine ==========
            # Call fire_sampling function or direct generation
            try:
                if hasattr(self.strategy.args, 'use_fire') and self.strategy.args.use_fire:
                    # Use FIRE sampling (Flaming-hot Initiation with Regular Execution)
                    outputs = fire_sampling(
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
                        all_prompts=prompts,
                        all_images=images,
                        all_videos=videos,
                        all_images_num=images_num,
                        all_videos_num=videos_num,
                        sampling_params=sampling_params,
                    )
                else:
                    # maybe this can be called in if and else respectively? or like this?
                    # Use original single-shot generation
                    outputs = self.strategy.gather_and_generate(
                        sampling_params=sampling_params,
                        all_prompt_token_ids=prompt_token_ids,
                        all_prompts=prompts if is_multimodal else None,
                        sleep_engine=self.strategy.args.enable_engine_sleep,
                        all_images=images if is_multimodal else None,
                        all_videos=videos if is_multimodal else None,
                        images_num=images_num if is_multimodal else None,
                        videos_num=videos_num if is_multimodal else None,
                    )
            except ValueError as e:
                if "prompt" in str(e) and "too long" in str(e):
                    self.strategy.print(f"[Skip] {e}")
                    return None  # Return None, subsequent experience_maker will ignore
                else:
                    raise

            # Process outputs in micro-batches and store in buffers
            for i in range(0, len(outputs), n_samples):
                batch_slice = slice(i, i + n_samples)
                output_batch = outputs[batch_slice]
                labels_batch = expanded_labels[batch_slice] if expanded_labels else []
                prompts_batch = prompts[batch_slice]
                images_batch = images[batch_slice] if images else None
                images_num_batch = images_num[batch_slice] if images_num else None
                videos_batch = videos[batch_slice] if videos else None
                videos_num_batch = videos_num[batch_slice] if videos_num else None
                references_batch = references[batch_slice] if references else None

                # Check if regeneration is needed
                needs_regen = any(len(out.output_token_ids) >= self.max_token_budget for out in output_batch)
                buffer_type = "regen" if needs_regen else "noregen"

                # Add to appropriate buffer
                self._add_to_buffer(buffer_type, "output", output_batch)
                self._add_to_buffer(buffer_type, "labels", labels_batch)
                self._add_to_buffer(buffer_type, "prompts", prompts_batch)
                if images_batch is not None:
                    self._add_to_buffer(buffer_type, "images", images_batch)
                if images_num_batch is not None:
                    self._add_to_buffer(buffer_type, "images_num", images_num_batch)
                if videos_batch is not None:
                    self._add_to_buffer(buffer_type, "videos", videos_batch)
                if videos_num_batch is not None:
                    self._add_to_buffer(buffer_type, "videos_num", videos_num_batch)
                if references_batch is not None:
                    self._add_to_buffer(buffer_type, "references", references_batch)

                if is_multimodal:
                    # Handle image tensors
                    grid_batch = images_grid_thw[batch_slice]
                    self._add_to_buffer(buffer_type, "images_grid_thw", grid_batch)
                    # Calculate pixel values slice
                    patch_start = sum(g[0] * g[1] * g[2] for g in images_grid_thw[:i])
                    patch_end = patch_start + sum(g[0] * g[1] * g[2] for g in grid_batch)
                    self._add_to_buffer(buffer_type, "images_pixel_values", images_pixel_values[patch_start:patch_end])
                    # Handle video tensors
                    if videos_grid_thw is not None:
                        videos_grid_batch = videos_grid_thw[batch_slice]
                        self._add_to_buffer(buffer_type, "videos_grid_thw", videos_grid_batch)
                    

        # --------------------------------------------------------------------
        # Step 2: Determine how many micro‑batches we need to return
        # --------------------------------------------------------------------
        total_micro = self.rollout_batch_size // self.strategy.world_size
        target_micro = int(self.partial_percent * total_micro)

        # How many micro‑batches are already available in the noregen buffer?
        noregen_micro = len(self.noregen_buffer['output']) // self.micro_rollout_batch_size
        if noregen_micro >= target_micro:
            # Enough noregen samples – just take them
            samples_data = self._get_from_buffer('noregen', target_micro * self.micro_rollout_batch_size)
        else:
            # Take all noregen samples and supplement with regenerated ones
            samples_needed = target_micro - noregen_micro
            noregen_data = self._get_from_buffer('noregen', noregen_micro * self.micro_rollout_batch_size)
            regen_data = self._regenerate_from_buffer(samples_needed * self.micro_rollout_batch_size, is_multimodal, **generate_kwargs)
            samples_data = self._merge_data(noregen_data, regen_data)

        # --------------------------------------------------------------------
        # Step 3: Convert the collected data back to Samples objects
        # --------------------------------------------------------------------
        samples_list = []
        image_patch_idx = 0
        video_patch_idx = 0
        image_start_idx = 0
        video_start_idx = 0

        all_outputs = samples_data.get("output", [])
        all_labels = samples_data.get("labels", [])
        all_prompts = samples_data.get("prompts", [])
        all_images = samples_data.get("images", [])
        all_images_num = samples_data.get("images_num", None)
        all_images_grid_thw = samples_data.get("images_grid_thw", None)
        all_images_pixel_values = samples_data.get("images_pixel_values", None)
        all_videos_num = samples_data.get("videos_num", None)
        all_videos_grid_thw = samples_data.get("videos_grid_thw", None)
        all_videos_pixel_values = samples_data.get("videos_pixel_values", None)
        all_references = samples_data.get("references", [])

        for i in range(0, len(all_outputs), config.micro_rollout_batch_size):
            micro_batch_outputs = all_outputs[i:i + config.micro_rollout_batch_size]
            micro_batch_prompts = all_prompts[i:i + config.micro_rollout_batch_size]

            # Extract micro-batch data
            micro_batch_grid_thw = None
            micro_batch_video_grid_thw = None
            micro_batch_raw_images = None

            if is_multimodal:
                rollout_image_count = sum(all_images_num[i:i + config.micro_rollout_batch_size])
                micro_batch_grid_thw = all_images_grid_thw[image_start_idx:image_start_idx + rollout_image_count]
                micro_batch_raw_images = all_images[i:i + config.micro_rollout_batch_size]
                image_start_idx += rollout_image_count

                rollout_video_count = sum(all_videos_num[i:i + config.micro_rollout_batch_size])
                micro_batch_video_grid_thw = all_videos_grid_thw[video_start_idx:video_start_idx + rollout_video_count]
                video_start_idx += rollout_video_count

            micro_batch_references = (all_references[i:i + config.micro_rollout_batch_size] if all_references else None)
            micro_batch_labels = (all_labels[i:i + config.micro_rollout_batch_size] if all_labels else None)
            # Build samples
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
                # Update patch indices from the returned values
                if updated_patch_idx is not None:
                    image_patch_idx = updated_patch_idx
                if updated_video_patch_idx is not None:
                    video_patch_idx = updated_video_patch_idx
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

    def _add_to_buffer(self, buffer_type: str, data_name: str, data):
        """Add data to specified buffer.
        
        Args:
            buffer_type: 'regen' or 'noregen'
            data_name: Key name for storing data
            data: Data to add (can be tensor, list, or other)
        
        Special handling:
        - Keys with 'grid_thw': split 2D tensors by rows
        - Keys with 'pixel_values': keep 2D tensors as-is
        - Other 2D tensors: split by rows
        """
        buffer = self.regen_buffer if buffer_type == 'regen' else self.noregen_buffer
        if data_name not in buffer:
            buffer[data_name] = []
        
        if isinstance(data, torch.Tensor):
            is_grid_thw = 'grid_thw' in data_name
            is_pixel_values = 'pixel_values' in data_name
            
            if data.dim() == 2:
                if is_grid_thw:
                    # Split grid_thw 2D tensors by rows
                    buffer[data_name].extend(torch.unbind(data, dim=0))
                elif is_pixel_values:
                    # Keep pixel_values 2D tensors intact
                    buffer[data_name].append(data)
                else:
                    # Split other 2D tensors by rows
                    buffer[data_name].extend(torch.unbind(data, dim=0))
            else:
                # Add 1D or higher-dim tensors as-is
                buffer[data_name].append(data)
        else:
            buffer[data_name].extend(data if isinstance(data, list) else [data])


    def _get_from_buffer(self, buffer_type: str, count: Optional[int] = None):
        """Retrieve data from buffer, optionally limiting the amount.
        
        Args:
            buffer_type: 'regen' or 'noregen'
            count: Number of items to retrieve. If None, retrieve all.
        
        Returns:
            Dictionary with retrieved data.
            Special handling:
            - grid_thw keys: stack 1D tensors to 2D
            - pixel_values keys: concatenate 2D tensors
        """
        buffer = self.regen_buffer if buffer_type == 'regen' else self.noregen_buffer
        result = {}
        
        for key, lst in buffer.items():
            if not lst:
                # Return empty tensor with proper shape
                if 'grid_thw' in key:
                    result[key] = torch.tensor([]).reshape(0, 3)
                elif 'pixel_values' in key:
                    result[key] = torch.tensor([])
                else:
                    result[key] = torch.tensor([])
                
                if count is None:
                    buffer[key] = []
                continue
            
            all_tensors = all(isinstance(item, torch.Tensor) for item in lst)
            
            if count is None:
                # Retrieve all data
                if all_tensors:
                    if 'pixel_values' in key and lst[0].dim() >= 2:
                        # Concatenate pixel_values 2D tensors
                        result[key] = torch.cat(lst, dim=0) if lst else torch.tensor([])
                    elif 'grid_thw' in key and lst[0].dim() == 1:
                        # Stack grid_thw 1D tensors to 2D
                        result[key] = torch.stack(lst, dim=0) if lst else torch.tensor([]).reshape(0, 3)
                    elif lst[0].dim() == 1:
                        # Stack 1D tensors to 2D
                        result[key] = torch.stack(lst, dim=0) if lst else torch.tensor([])
                    else:
                        # Concatenate other tensors
                        result[key] = torch.cat(lst, dim=0) if lst else torch.tensor([])
                else:
                    result[key] = lst.copy()
                buffer[key] = []
            else:
                # Retrieve specified number of items
                items_to_take = lst[:count]
                
                if all_tensors and items_to_take:
                    if 'pixel_values' in key and items_to_take[0].dim() >= 2:
                        result[key] = torch.cat(items_to_take, dim=0)
                    elif 'grid_thw' in key and items_to_take[0].dim() == 1:
                        result[key] = torch.stack(items_to_take, dim=0)
                    elif items_to_take[0].dim() == 1:
                        result[key] = torch.stack(items_to_take, dim=0)
                    else:
                        result[key] = torch.cat(items_to_take, dim=0)
                else:
                    result[key] = items_to_take
                
                # Update buffer
                buffer[key] = lst[count:]
        
        return result

    @torch.no_grad()
    def _regenerate_from_buffer(self, num_needed: int, is_multimodal: bool, **kwargs) -> dict:
        """Regenerate outputs for samples that reached token budget."""
        config = self.strategy.config

        # Get data from regeneration buffer
        regen_data = self._get_from_buffer("regen", num_needed)
        if not regen_data.get("output"):
            return {}

        # Identify indices needing regeneration
        regen_indices = [
            i for i, output in enumerate(regen_data["output"])
            if len(output.output_token_ids) >= self.max_token_budget
        ]

        if not regen_indices:
            return regen_data

        # Prepare regeneration inputs
        regen_outputs = [regen_data["output"][i] for i in regen_indices]
        regen_tokens = [output.output_token_ids for output in regen_outputs]
        decoded_outputs = self.tokenizer.batch_decode(regen_tokens, skip_special_tokens=False)

        # Create new inputs by combining original prompts and partial outputs
        new_inputs = [
            prompt + output
            for prompt, output in zip(
                [regen_data["prompts"][i] for i in regen_indices],
                decoded_outputs
            )
        ]

        # Prepare sampling parameters
        if config.engine_type == "vllm":
            sampling_params = SamplingParams(
                temperature=kwargs.get("temperature", 1.0),
                top_p=kwargs.get("top_p", 1.0),
                top_k=kwargs.get("top_k", -1),
                max_tokens=kwargs.get("max_new_tokens", 1024),
                min_tokens=kwargs.get("min_new_tokens", 1),
                skip_special_tokens=kwargs.get("skip_special_tokens", False),
                include_stop_str_in_output=True,
                ignore_eos=os.environ.get("IGNORE_EOS", "0") == "1",
            )
        elif config.engine_type == "sglang":
            sampling_params = dict(
                n=1,
                temperature=kwargs.get("temperature", 1.0),
                top_p=kwargs.get("top_p", 1.0),
                top_k=kwargs.get("top_k", -1),
                max_new_tokens=kwargs.get("max_new_tokens", 1024),
                presence_penalty=0.0,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
                skip_special_tokens=kwargs.get("skip_special_tokens", False),
                spaces_between_special_tokens=True,
                ignore_eos=os.environ.get("IGNORE_EOS", "0") == "1",
            )
        else:
            raise ValueError(f"Unsupported backend: {config.engine_type}")

        # Build inputs and regenerate using the same pattern
        if is_multimodal:
            # Use strategy._build_multimodal_inputs
            inputs = self.strategy._build_multimodal_inputs(
                all_prompts=new_inputs,
                all_images=[regen_data["images"][i] for i in regen_indices],
                images_num=[regen_data["images_num"][i] for i in regen_indices]
            )
            # Use engine_generate_local for multimodal regeneration
            regenerated = self.strategy.engine_generate_local(
                sampling_params=sampling_params,
                prompt_token_ids=None,
                multi_modal_inputs=inputs,
            )
        else:
            # For text-only, we can reuse parent's generate_samples but need raw outputs.
            # Instead, we can directly call strategy.gather_and_generate with tokenized inputs.
            # Tokenize new prompts
            tokenized = self.tokenize_fn(new_inputs, self.prompt_max_len, padding=False)
            prompt_token_ids = tokenized["input_ids"]
            # Expand by n_samples_per_prompt (should be 1 for regeneration?)
            # In partial rollout, each sample is already expanded, so we assume n_samples_per_prompt=1.
            # Use strategy.gather_and_generate
            regenerated = self.strategy.gather_and_generate(
                sampling_params=sampling_params,
                all_prompt_token_ids=prompt_token_ids,
                all_prompts=None,
                all_images=None,
                sleep_engine=False,
                images_num=None,
            )

        # Update regenerated outputs in regen_data
        for idx, new_output in zip(regen_indices, regenerated):
            regen_data["output"][idx] = new_output

        return regen_data

    def _merge_data(self, data1: Dict[str, List], data2: Dict[str, List]) -> Dict[str, List]:
        """Merge two data dictionaries, concatenating lists or tensors."""
        merged = {}
        for key in set(data1.keys()) | set(data2.keys()):
            val1 = data1.get(key, [])
            val2 = data2.get(key, [])
            if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
                merged[key] = torch.cat([val1, val2])
            elif isinstance(val1, list) and isinstance(val2, list):
                merged[key] = val1 + val2
            else:
                merged[key] = val1 if val1 else val2
        return merged
