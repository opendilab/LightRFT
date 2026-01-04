"""
Trajectory Saver Utility for debugging and analysis.

This module provides utilities to save experience trajectories to JSON files
for debugging and analysis purposes.
"""

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
import io


class TrajectorySaver:
    """
    Utility class to save experience trajectories to JSON files.

    Features:
        - Saves experience sequences, rewards, and metadata for individual samples.
        - Supports both text-only and vision-language models.
        - Efficiently handles image data by saving them to a separate directory with clear linkage.
        - Only saves on rank 0 to avoid duplication.
        - Produces human-readable JSON output for easy debugging.

    :param save_dir: Directory to save trajectory files
    :type save_dir: str
    :param tokenizer: Tokenizer for decoding sequences
    :type tokenizer: Any
    :param save_images_separately: If True, save images as separate files. Default to True
    :type save_images_separately: bool
    :param max_image_size: Maximum dimension for saved images (to reduce file size). Default to 512
    :type max_image_size: int
    """
    def __init__(
        self,
        save_dir: str,
        tokenizer: Any,
        save_images_separately: bool = True,
        max_image_size: int = 512,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.tokenizer = tokenizer
        self.save_images_separately = save_images_separately
        self.max_image_size = max_image_size

        # Create directory structure only on rank 0
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            if save_images_separately:
                (self.save_dir / "images").mkdir(exist_ok=True)

    def save_trajectories(
        self,
        experiences: List[Any],
        step: int,
        num_samples: int = 10,
        prefix: str = "trajectories",
    ) -> Optional[str]:
        """
        Save a subset of experiences to a JSON file.

        Each Experience object is a micro-batch. This function unpacks them
        into individual sample trajectories before saving.

        :param experiences: List of Experience or ExperienceVL objects from the replay buffer
        :type experiences: List[Any]
        :param step: Current training step (used in filename)
        :type step: int
        :param num_samples: Target number of individual sample trajectories to save. Default to 10
        :type num_samples: int
        :param prefix: Prefix for the output filename. Default to "trajectories"
        :type prefix: str
        :return: Path to the saved JSON file (None if not rank 0 or no experiences)
        :rtype: Optional[str]
        """
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() != 0:
            return None

        if not experiences:
            return None

        all_trajectories = []
        # Iterate through experience objects (micro-batches) until we have enough samples.
        for exp_idx, exp in enumerate(experiences):
            if len(all_trajectories) >= num_samples:
                break

            #  Unpack the micro-batch into individual trajectories.
            unpacked_trajs = self._unpack_experience_to_dicts(exp, step, exp_idx)
            all_trajectories.extend(unpacked_trajs)

        # Ensure we don't save more than requested.
        sampled_trajectories = all_trajectories[:num_samples]

        # Save to JSON
        output_path = self.save_dir / f"{prefix}_step_{step}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sampled_trajectories, f, indent=2, ensure_ascii=False)

        print(f"[TrajectorySaver] Saved {len(sampled_trajectories)} trajectories to {output_path}")
        return str(output_path)

    def _unpack_experience_to_dicts(self, exp: Any, step: int, exp_idx: int) -> List[Dict[str, Any]]:
        """
        Unpacks a single Experience object (a micro-batch) into a list of
        dictionaries, where each dictionary represents a single sample.

        :param exp: Experience object containing micro-batch data
        :type exp: Any
        :param step: Current training step
        :type step: int
        :param exp_idx: Index of the experience object in the list
        :type exp_idx: int
        :return: List of dictionaries, each representing a single sample trajectory
        :rtype: List[Dict[str, Any]]
        """
        # Extract tensors and move to CPU
        sequences = exp.sequences.cpu()

        # Validate sequences shape before processing
        if len(sequences.shape) == 0:
            # Scalar tensor - skip this experience
            print(
                f"[TrajectorySaver] Warning: sequences is a scalar tensor at step {step}, exp_idx {exp_idx}. Skipping."
            )
            return []
        elif len(sequences.shape) == 1:
            # 1D tensor - reshape to (1, seq_len)
            print(
                f"[TrajectorySaver] Warning: sequences is 1D tensor with shape {sequences.shape} at step {step}, exp_idx {exp_idx}. Reshaping to 2D."  # noqa: E501
            )
            sequences = sequences.unsqueeze(0)
        elif len(sequences.shape) != 2:
            # Unexpected shape
            print(
                f"[TrajectorySaver] Error: sequences has unexpected shape {sequences.shape} at step {step}, exp_idx {exp_idx}. Expected 2D tensor (B, S). Skipping."  # noqa: E501
            )
            return []

        batch_size = sequences.shape[0]

        # Handle action_mask with same shape validation
        if exp.action_mask is not None:
            action_mask = exp.action_mask.cpu()
            if len(action_mask.shape) == 1:
                action_mask = action_mask.unsqueeze(0)
            elif len(action_mask.shape) != 2:
                print(
                    f"[TrajectorySaver] Warning: action_mask has unexpected shape {action_mask.shape}. Creating default mask."  # noqa: E501
                )
                action_mask = torch.zeros_like(sequences, dtype=torch.bool)
        else:
            action_mask = torch.zeros_like(sequences, dtype=torch.bool)

        # Decode all sequences in the micro-batch at once
        decoded_sequences = self.tokenizer.batch_decode(sequences, skip_special_tokens=False)

        # Handle optional tensors with shape validation
        advantages = self._safe_extract_tensor(exp, 'advantages', batch_size)
        returns = self._safe_extract_tensor(exp, 'returns', batch_size)
        action_log_probs = self._safe_extract_tensor(exp, 'action_log_probs', batch_size)
        values = self._safe_extract_tensor(exp, 'values', batch_size)
        raw_images = exp.raw_images if hasattr(exp,
                                               'raw_images') and exp.raw_images is not None else [None] * batch_size

        unpacked_list = []
        # Iterate over each sample in the micro-batch
        for i in range(batch_size):
            # Get generated text for this specific sample
            # action_mask indices are relative to action_mask, not sequences!
            # action_mask is created from sequences[:, input_len - 1 : -1]
            # So action_mask[j] corresponds to sequences[input_len - 1 + j]
            try:
                gen_indices = action_mask[i].nonzero(as_tuple=True)[0]
                if len(gen_indices) > 0:
                    # Verify sequences[i] is indexable
                    if len(sequences[i].shape) == 0:
                        print(
                            f"[TrajectorySaver] Warning: sequences[{i}] is scalar at step {step}, exp_idx {exp_idx}. Skipping generation."  # noqa: E501
                        )
                        generated_text = ""
                        pure_generated_text = ""
                    else:
                        # Calculate offset to adjust indices from action_mask space to sequences space
                        # action_mask length = seq_length - input_len
                        # Therefore: input_len = seq_length - action_mask_len
                        # Offset = input_len - 1 (because action_mask starts from input_len - 1)
                        input_len = sequences.size(1) - action_mask.size(1)
                        offset = input_len - 1

                        # Adjust indices to sequences space
                        adjusted_indices = gen_indices + offset
                        gen_tokens = sequences[i][adjusted_indices]

                        # generated_text includes the last prompt token (for RL state-action pairing)
                        generated_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

                        # pure_generated_text excludes the last prompt token (only model's output)
                        if len(adjusted_indices) > 1:
                            pure_gen_tokens = sequences[i][adjusted_indices[1:]]
                            pure_generated_text = self.tokenizer.decode(pure_gen_tokens, skip_special_tokens=True)
                        else:
                            pure_generated_text = ""
                else:
                    generated_text = ""
                    pure_generated_text = ""
            except (IndexError, RuntimeError) as e:
                print(
                    f"[TrajectorySaver] Error extracting generated text for sample {i} at step {step}, exp_idx {exp_idx}: {e}"  # noqa: E501
                )
                generated_text = ""
                pure_generated_text = ""

            # Build the dictionary for this single sample
            traj_dict = {
                "global_step": step,
                "experience_index": exp_idx,  # which micro-batch it came from
                "sample_in_exp": i,  # which sample within the micro-batch
                "full_sequence": decoded_sequences[i],
                "generated_text": generated_text,  # Includes last prompt token (for RL state-action)
                "pure_generated_text": pure_generated_text,  # Only model's output
            }

            # Add optional fields for this sample
            if advantages[i] is not None:
                traj_dict["advantages"] = self._tensor_to_list(advantages[i])
            if returns[i] is not None:
                traj_dict["return"] = self._tensor_to_list(returns[i])
            if action_log_probs[i] is not None:
                traj_dict["action_log_probs"] = self._tensor_to_list(action_log_probs[i])
            if values[i] is not None:
                traj_dict["values"] = self._tensor_to_list(values[i])

            # Add info dict fields, slicing if they are tensors
            if hasattr(exp, 'info') and exp.info is not None:
                info_dict = {}
                for key, value in exp.info.items():
                    if isinstance(value, torch.Tensor) and len(value.shape) > 0 and len(value) == batch_size:
                        info_dict[key] = self._tensor_to_list(value[i])
                    elif key == 'reward_metrics':
                        metrics = {}
                        for metric_name, metric_tensor in value.items():
                            if isinstance(metric_tensor,
                                          torch.Tensor) and len(metric_tensor.shape
                                                                ) > 0 and len(metric_tensor) == batch_size:
                                metrics[metric_name] = self._tensor_to_list(metric_tensor[i])
                            else:  # scalar metric, applies to all
                                metrics[metric_name] = self._tensor_to_list(metric_tensor) if isinstance(
                                    metric_tensor, torch.Tensor
                                ) else metric_tensor
                        info_dict[key] = metrics
                    else:  # scalar value, applies to all samples in micro-batch
                        info_dict[key] = self._tensor_to_list(value) if isinstance(value, torch.Tensor) else value
                traj_dict["info"] = info_dict

            # Handle images for this specific sample
            sample_images = raw_images[i]

            # Normalize sample_images to always be a list or None
            if sample_images is not None:
                # Check if it's a single image object (PIL.Image.Image or similar)
                # Images have certain attributes like 'size', 'mode', etc.
                if hasattr(sample_images, 'size') and hasattr(sample_images, 'mode'):
                    # Single image - wrap in list
                    sample_images = [sample_images]
                elif not isinstance(sample_images, list):
                    # Unknown type - try to convert to list
                    try:
                        sample_images = list(sample_images)
                    except (TypeError, ValueError):
                        print(
                            f"[TrajectorySaver] Warning: Unexpected image type {type(sample_images)} at step {step}, exp_idx {exp_idx}, sample {i}. Skipping images."  # noqa: E501
                        )
                        sample_images = None

            if sample_images:
                traj_dict["has_images"] = True
                traj_dict["num_images"] = len(sample_images)

                #  Logic now correctly handles a single list of images per sample
                if self.save_images_separately:
                    image_paths = self._save_images(sample_images, step, exp_idx, i)
                    traj_dict["image_paths"] = image_paths
                else:
                    traj_dict["images_base64"] = self._encode_images_base64(sample_images)
            else:
                traj_dict["has_images"] = False

            unpacked_list.append(traj_dict)

        return unpacked_list

    def _tensor_to_list(self, tensor: Optional[torch.Tensor]) -> Union[List[Any], float, int, None]:
        """
        Convert tensor to list or scalar.

        :param tensor: Input tensor to convert
        :type tensor: Optional[torch.Tensor]
        :return: Converted value as list, scalar, or None
        :rtype: Union[List[Any], float, int, None]
        """
        if tensor is None:
            return None
        tensor = tensor.cpu().detach()
        if tensor.numel() == 1:
            return tensor.item()
        else:
            return tensor.tolist()

    def _safe_extract_tensor(self, exp: Any, attr_name: str,
                             expected_batch_size: int) -> Union[torch.Tensor, List[Optional[torch.Tensor]]]:
        """
        Safely extract a tensor attribute from an experience object.

        :param exp: Experience object
        :type exp: Any
        :param attr_name: Name of the attribute to extract
        :type attr_name: str
        :param expected_batch_size: Expected batch size for validation
        :type expected_batch_size: int
        :return: List with one element per sample, or [None] * batch_size if extraction fails
        :rtype: Union[torch.Tensor, List[Optional[torch.Tensor]]]
        """
        if not hasattr(exp, attr_name) or getattr(exp, attr_name) is None:
            return [None] * expected_batch_size

        tensor = getattr(exp, attr_name).cpu()

        # Handle scalar tensors
        if len(tensor.shape) == 0:
            # Scalar - apply to all samples
            return [tensor] * expected_batch_size

        # Handle 1D tensors
        if len(tensor.shape) == 1:
            if tensor.shape[0] == expected_batch_size:
                return tensor
            else:
                print(
                    f"[TrajectorySaver] Warning: {attr_name} has mismatched batch size {tensor.shape[0]}, expected {expected_batch_size}. Padding/truncating."  # noqa: E501
                )
                # Pad or truncate
                if tensor.shape[0] < expected_batch_size:
                    padding = [None] * (expected_batch_size - tensor.shape[0])
                    return list(tensor) + padding
                else:
                    return tensor[:expected_batch_size]

        # Handle 2D+ tensors
        if tensor.shape[0] == expected_batch_size:
            return tensor
        else:
            print(
                f"[TrajectorySaver] Warning: {attr_name} has mismatched batch size {tensor.shape[0]}, expected {expected_batch_size}. Using defaults."  # noqa: E501
            )
            return [None] * expected_batch_size

    def _save_images(self, imgs: List[Image.Image], step: int, exp_idx: int, sample_idx: int) -> List[Optional[str]]:
        """
        Save a list of images for a single sample.

        :param imgs: List of PIL Image objects to save
        :type imgs: List[Image.Image]
        :param step: Current training step
        :type step: int
        :param exp_idx: Index of the experience object
        :type exp_idx: int
        :param sample_idx: Index of the sample within the micro-batch
        :type sample_idx: int
        :return: List of relative image paths (or None for invalid images)
        :rtype: List[Optional[str]]
        """
        image_paths = []
        for img_idx, img in enumerate(imgs):
            if img is not None:
                # Resize if needed
                if max(img.size) > self.max_image_size:
                    img.thumbnail((self.max_image_size, self.max_image_size), Image.Resampling.LANCZOS)

                #  Filename is now much more specific and easier to trace
                img_filename = f"step{step}_exp{exp_idx}_sample{sample_idx}_img{img_idx}.png"
                img_path = self.save_dir / "images" / img_filename
                img.save(img_path)
                # Store a relative path for portability
                image_paths.append(f"images/{img_filename}")
            else:
                image_paths.append(None)
        return image_paths

    def _encode_images_base64(
        self,
        imgs: List[Image.Image],
    ) -> List[Optional[str]]:
        """
        Encode a list of images for a single sample as base64 strings.

        :param imgs: List of PIL Image objects to encode
        :type imgs: List[Image.Image]
        :return: List of base64-encoded image strings (or None for invalid images)
        :rtype: List[Optional[str]]
        """
        base64_images = []
        for img in imgs:
            if img is not None:
                # Resize if needed
                if max(img.size) > self.max_image_size:
                    img.thumbnail((self.max_image_size, self.max_image_size), Image.Resampling.LANCZOS)

                # Convert to base64
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                img_bytes = buffer.getvalue()
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                base64_images.append(img_base64)
            else:
                base64_images.append(None)
        return base64_images


def create_trajectory_saver(args: Any, tokenizer: Any) -> Optional[TrajectorySaver]:
    """
    Factory function to create TrajectorySaver if enabled.

    :param args: Training arguments containing save_trajectories flag and save_path
    :type args: Any
    :param tokenizer: Tokenizer for decoding sequences
    :type tokenizer: Any
    :return: TrajectorySaver instance or None if not enabled
    :rtype: Optional[TrajectorySaver]
    """
    if not getattr(args, 'save_trajectories', False):
        return None

    save_dir = os.path.join(args.save_path, "trajectories")

    return TrajectorySaver(
        save_dir=save_dir,
        tokenizer=tokenizer,
        save_images_separately=True,
        max_image_size=512,
    )
