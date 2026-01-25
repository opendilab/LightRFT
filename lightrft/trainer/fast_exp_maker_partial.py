"""
PartialFastExperienceMaker – FastExperienceMaker with partial rollout and token‑budget regeneration.

This subclass adds two key features:
  1. Partial rollout: only a fraction (partial_percent) of the total rollout batch is generated
     in each call; the rest is kept in buffers.
  2. Token‑budget regeneration: samples whose generation reaches max_token_budget are flagged
     and can be regenerated later (e.g., for continuing long‑form tasks).

The class reuses the parent's infrastructure (MultimodalDataProcessor, RewardComputationEngine,
etc.) and only overrides the methods that implement the partial‑rollout logic.
"""

from typing import List, Optional, Union, Tuple, Dict, Any
import os
import time
from copy import deepcopy

import torch
import torch.distributed as dist
from vllm import SamplingParams
from easydict import EasyDict

from openrlhf.trainer.ppo_utils.experience_maker import Experience, Samples
from openrlhf.trainer.ppo_utils.experience_maker_vl import SamplesVL
from lightrft.trainer.fast_exp_maker import FastExperienceMaker


class PartialFastExperienceMaker(FastExperienceMaker):
    """
    FastExperienceMaker with partial rollout and token‑budget regeneration.

    Args:
        partial_percent (float): fraction of the rollout batch to generate in one call.
        max_token_budget (int): maximum allowed generation length before regeneration.
        packing_samples (bool): whether to pack samples (inherited).
        processor: multimodal processor (inherited).
        *args, **kwargs: passed to parent.
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
        super().__init__(*args, packing_samples=packing_samples, processor=processor, **kwargs)
        self.partial_percent = partial_percent
        self.max_token_budget = max_token_budget

        # Buffers for regeneration (regen) and non‑regeneration (noregen) samples.
        # Each buffer is a dict mapping field names to lists of data.
        self.regen_buffer: Dict[str, List] = {}
        self.noregen_buffer: Dict[str, List] = {}
        fields = [
            'output', 'labels', 'prompts', 'images', 'images_num',
            'images_pixel_values', 'images_grid_thw', 'image_flags', 'references'
        ]
        for field in fields:
            self.regen_buffer[field] = []
            self.noregen_buffer[field] = []

        # Placeholders for batch‑size parameters (set by need_new_prompts)
        self.rollout_batch_size = None
        self.micro_rollout_batch_size = None

    def need_new_prompts(self, rollout_batch_size: int, micro_rollout_batch_size: int) -> bool:
        """
        Check whether the buffers contain enough data to make a full experience batch.

        Returns:
            True if new prompts need to be fetched (i.e., buffers are below the partial threshold).
        """
        self.rollout_batch_size = rollout_batch_size
        self.micro_rollout_batch_size = micro_rollout_batch_size

        # Total micro‑batches needed for a full rollout
        total_micro = rollout_batch_size // micro_rollout_batch_size
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
        images_num: Optional[List[int]] = None,
        all_references: Optional[List[str]] = None,
        all_labels: Optional[List] = None,
        **generate_kwargs
    ) -> List[Samples]:
        """
        Generate samples using the parent's pipeline, but only a partial fraction.

        The method:
          1. If new inputs are provided, generate them with the parent's generate_samples.
          2. Split the generated outputs into regeneration and non‑regeneration buffers.
          3. Draw from the buffers to produce the requested number of samples (partial_percent).
          4. If the noregen buffer is insufficient, regenerate some samples from the regen buffer.

        Returns:
            List of Samples (or SamplesVL) ready for experience making.
        """
        args = self.strategy.args
        is_multimodal = all_images is not None
        internvl = "internvl" in self.actor.pretrain_or_model.lower() if is_multimodal else False

        # --------------------------------------------------------------------
        # Step 1: Generate new samples if inputs are provided
        # --------------------------------------------------------------------
        if all_prompts is not None:
            # Replicate the generation logic from fast_exp_maker_partial.py
            # Prepare sampling parameters
            if args.engine_type == "vllm":
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
            elif args.engine_type == "sglang":
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
                raise ValueError(f"Unsupported backend: {args.engine_type}")

            # Expand labels
            expanded_labels = sum([[label] * args.n_samples_per_prompt for label in all_labels], []) if all_labels else []

            # Process multimodal data
            if is_multimodal:
                processed = self._process_multimodal_data(
                    all_prompts=all_prompts,
                    all_images=all_images,
                    is_internvl=internvl,
                    all_references=all_references,
                    images_num=images_num
                )
                prompt_token_ids = processed["all_prompt_token_ids"]
                prompts = processed["all_prompts"]
                images = processed["all_images"]
                images_num = processed["all_images_num"]
                pixel_values = processed["all_images_pixel_values"]
                grid_thw = processed["all_images_grid_thw"]
                image_flags = processed["all_image_flags"]
                references = processed["all_references"]
            else:
                tokenized = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)
                prompt_token_ids = tokenized["input_ids"]
                prompt_token_ids = sum([[token_ids] * args.n_samples_per_prompt for token_ids in prompt_token_ids], [])
                prompts = all_prompts * args.n_samples_per_prompt
                images = None
                references = all_references * args.n_samples_per_prompt if all_references else None

            # Generate outputs via inference engine
            outputs = self.strategy.gather_and_generate(
                sampling_params=sampling_params,
                all_prompt_token_ids=prompt_token_ids,
                all_prompts=prompts if is_multimodal else None,
                all_images=images if is_multimodal else None,
                sleep_engine=False,
                images_num=images_num if is_multimodal else None,
            )

            # Process outputs in micro-batches and store in buffers
            for i in range(0, len(outputs), args.micro_rollout_batch_size):
                batch_slice = slice(i, i + args.micro_rollout_batch_size)
                output_batch = outputs[batch_slice]
                labels_batch = expanded_labels[batch_slice] if expanded_labels else []
                prompts_batch = prompts[batch_slice]
                images_batch = images[batch_slice] if images else None
                images_num_batch = images_num[batch_slice] if images_num else None
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
                if references_batch is not None:
                    self._add_to_buffer(buffer_type, "references", references_batch)

                if is_multimodal:
                    self._add_to_buffer(buffer_type, "image_flags", image_flags[batch_slice])
                    # Handle image tensors
                    grid_batch = grid_thw[batch_slice]
                    self._add_to_buffer(buffer_type, "images_grid_thw", grid_batch)
                    # Calculate pixel values slice
                    patch_start = sum(g[0] * g[1] * g[2] for g in grid_thw[:i])
                    patch_end = patch_start + sum(g[0] * g[1] * g[2] for g in grid_batch)
                    self._add_to_buffer(buffer_type, "images_pixel_values", pixel_values[patch_start:patch_end])

        # --------------------------------------------------------------------
        # Step 2: Determine how many micro‑batches we need to return
        # --------------------------------------------------------------------
        total_micro = self.rollout_batch_size // self.micro_rollout_batch_size
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
            regen_data = self._regenerate_from_buffer(samples_needed * self.micro_rollout_batch_size, **generate_kwargs)
            samples_data = self._merge_data(noregen_data, regen_data)

        # --------------------------------------------------------------------
        # Step 3: Convert the collected data back to Samples objects
        # --------------------------------------------------------------------
        samples_list = self._generate_sample_list(
            samples_data,
            is_multimodal,
            internvl,
            **generate_kwargs
        )
        self.strategy.maybe_sleep_inference_engine()
        if all_prompts is None:
            return samples_list
        else:
            # Return tuple with samples_list and images_num, consistent with fast_exp_maker_partial.py
            images_num_list = samples_data.get("images_num")
            return samples_list, images_num_list

    def _process_multimodal_data(self, all_prompts, all_images, is_internvl, all_references, images_num):
        """Wrapper around parent's multimodal_processor.process_multimodal_batch."""
        if self.multimodal_processor is None:
            raise ValueError("Multimodal processor not initialized.")
        return self.multimodal_processor.process_multimodal_batch(
            all_prompts=all_prompts,
            all_images=all_images,
            all_references=all_references,
            images_num=images_num,
            n_samples_per_prompt=self.strategy.config.n_samples_per_prompt,
            is_internvl=is_internvl,
        )

    def _add_to_buffer(self, buffer_type: str, data_name: str, data):
        """Add data to specified buffer."""
        buffer = self.regen_buffer if buffer_type == 'regen' else self.noregen_buffer
        if data_name not in buffer:
            buffer[data_name] = []
        if isinstance(data, torch.Tensor):
            buffer[data_name].append(data)
        else:
            buffer[data_name].extend(data if isinstance(data, list) else [data])

    def _get_from_buffer(self, buffer_type: str, count: Optional[int]):
        """Retrieve data from buffer, optionally limiting the amount."""
        buffer = self.regen_buffer if buffer_type == 'regen' else self.noregen_buffer
        result = {}
        for key, lst in buffer.items():
            if count is None:
                result[key] = lst.copy()
                buffer[key] = []
            else:
                result[key] = lst[:count]
                buffer[key] = lst[count:]
        return result

    @torch.no_grad()
    def _regenerate_from_buffer(self, num_needed: int, **kwargs) -> dict:
        """Regenerate outputs for samples that reached token budget."""
        args = self.strategy.args

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
        if args.engine_type == "vllm":
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
        elif args.engine_type == "sglang":
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
            raise ValueError(f"Unsupported backend: {args.engine_type}")

        # Build inputs and regenerate using the same pattern as fast_exp_maker_partial.py
        # First, check if multimodal
        is_multimodal = regen_data.get("images") is not None and len(regen_data["images"]) > 0
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

    def _generate_sample_list(
        self,
        samples_data: Dict[str, List],
        is_multimodal: bool,
        internvl: bool,
        **kwargs
    ) -> List[Samples]:
        """Convert buffered data into a list of Samples."""
        args = self.strategy.args
        samples_list = []
        gen_max_len, gen_min_len = 0, 102400000
        index_pixel_patch = 0
        image_start_idx = 0

        all_outputs = samples_data.get("output", [])
        all_labels = samples_data.get("labels", [])
        all_prompts = samples_data.get("prompts", [])
        all_images = samples_data.get("images", [])
        all_images_num = samples_data.get("images_num", [])
        all_images_pixel_values = samples_data.get("images_pixel_values", [])
        all_images_grid_thw = samples_data.get("images_grid_thw", [])
        all_image_flags = samples_data.get("image_flags", [])
        all_references = samples_data.get("references", [])

        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            outputs = all_outputs[i: i + args.micro_rollout_batch_size]
            prompts = all_prompts[i: i + args.micro_rollout_batch_size]
            if all_images:
                assert all_images_num is not None
                rollout_image_num = sum(all_images_num[i: i + args.micro_rollout_batch_size])
                images_grid_thw = all_images_grid_thw[image_start_idx: image_start_idx + rollout_image_num]
                raw_images = all_images[image_start_idx: image_start_idx + rollout_image_num]
                image_start_idx += rollout_image_num
            if all_references:
                references = all_references[i: i + args.micro_rollout_batch_size]
            labels = all_labels[i: i + args.micro_rollout_batch_size]

            if not self.packing_samples:
                # Build unpacked samples
                max_input_len, max_output_len = 0, 0
                for output in outputs:
                    max_input_len = max(max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(output.output_token_ids))

                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                pixel_values = []
                image_grid_thw_list = []
                image_flags = []
                images_grid_id = 0
                for j in range(len(outputs)):
                    output = outputs[j]
                    input_len = len(output.prompt_token_ids)
                    input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)
                    output_len = len(output.output_token_ids)
                    output_ids = list(output.output_token_ids) + [pad_token_id] * (max_output_len - output_len)
                    # split pixel_patch
                    if all_images:
                        image_num = all_images_num[i + j]
                        for image_id in range(0, image_num):
                            images_grid = images_grid_thw[images_grid_id + image_id]
                            if internvl:
                                num_patch = images_grid if isinstance(images_grid, int) else images_grid.sum().item()
                                _image_flags = all_image_flags[index_pixel_patch: index_pixel_patch + num_patch]
                                image_flags.append(_image_flags)
                                image_grid_thw_list.append(torch.tensor([1, 1, num_patch]).unsqueeze(0))
                            else:
                                num_patch = images_grid[0] * images_grid[1] * images_grid[2]
                                image_grid_thw_list.append(images_grid.clone().unsqueeze(0))
                            images_pixel_value = all_images_pixel_values[index_pixel_patch: index_pixel_patch + num_patch]
                            pixel_values.append(images_pixel_value.clone())
                            index_pixel_patch += num_patch
                        images_grid_id += image_num
                    sequences.append(input_ids + output_ids)

                sequences = torch.tensor(sequences)
                sequences, attention_mask, action_mask = self.actor.process_sequences(
                    sequences, max_input_len, eos_token_id, pad_token_id
                )
                sequences = sequences.to("cuda")
                attention_mask = attention_mask.to("cuda")
                action_mask = action_mask.to("cuda")
                if not all_images:
                    samples_list.append(
                        Samples(
                            sequences=sequences,
                            attention_mask=attention_mask,
                            action_mask=action_mask,
                            num_actions=action_mask.size(1),
                            packed_seq_lens=None,
                            response_length=action_mask.float().sum(dim=-1),
                            total_length=attention_mask.float().sum(dim=-1),
                            prompts=prompts,
                            labels=labels,
                            pad_len=None,
                        )
                    )
                else:
                    if internvl:
                        pixel_values_intern = torch.cat(pixel_values, dim=0).to("cuda") if pixel_values else None
                        pixel_values = None
                    else:
                        pixel_values = torch.cat(pixel_values, dim=0).to("cuda") if pixel_values else None
                        pixel_values_intern = None
                    samples_list.append(
                        SamplesVL(
                            sequences=sequences,
                            attention_mask=attention_mask,
                            action_mask=action_mask,
                            image_grid_thws=torch.cat(image_grid_thw_list, dim=0).to("cuda") if not internvl else None,
                            raw_images=raw_images,
                            pixel_values=pixel_values,
                            pixel_values_intern=pixel_values_intern,
                            image_flags=torch.cat(image_flags, dim=0).to("cuda") if internvl else None,
                            num_actions=action_mask.size(1),
                            packed_seq_lens=None,
                            response_length=action_mask.float().sum(dim=-1),
                            total_length=attention_mask.float().sum(dim=-1),
                            references=references,
                            labels=labels,
                            prompts=prompts,
                        )
                    )
            else:
                # Packed samples (not supporting VLM yet)
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
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
                    gen_max_len = max(gen_max_len, output_len)
                    gen_min_len = min(gen_min_len, output_len)

                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                        prompts=prompts,
                        labels=labels,
                        pad_len=None,
                    )
                )

        if dist.get_rank(self.backend_mp_group) == 0:
            print(f"*** response_length {gen_max_len=}, {gen_min_len=}")

        return samples_list

    def process_experiences(self, experiences: List[Experience]) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences (reward shaping for partial rollout).

        This method overrides the parent's _process_experiences to handle
        advantage estimators that expect a different group size (partial_percent).
        """
        args = self.strategy.args
        if args.advantage_estimator == "rloo":
            rewards = torch.cat([exp.info["reward"] for exp in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt)
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
            rewards = rewards.flatten().chunk(len(experiences))
            return experiences, rewards
        elif args.advantage_estimator in ["grpo", "group_norm"]:
            # Adjust group size according to partial rollout
            group_size = int(self.partial_percent * self.rollout_batch_size // args.micro_rollout_batch_size)
            rewards = torch.cat([exp.info["reward"] for exp in experiences])
            rewards = rewards.reshape(-1, group_size)
            baseline = rewards.mean(-1, keepdim=True)
            rewards = (rewards - baseline) / (rewards.std(1, keepdim=True) + 1e-8)
            rewards = rewards.flatten().chunk(len(experiences))
            return experiences, rewards
        elif args.advantage_estimator == "reinforce_baseline":
            rewards = torch.cat([exp.info["reward"] for exp in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            rewards = rewards - rewards.mean(-1, keepdim=True)
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        else:
            raise ValueError(f"Unhandled advantage_estimator: {args.advantage_estimator}")
