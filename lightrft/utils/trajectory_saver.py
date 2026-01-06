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
    :param mark_high_entropy_tokens: If True, mark high-entropy tokens in saved trajectories with special markers. Default to False
    :type mark_high_entropy_tokens: bool
    :param high_entropy_token_ratio: Ratio of high-entropy tokens to mark (e.g., 0.2 means top 20%). Only used if mark_high_entropy_tokens is True. Default to 0.2
    :type high_entropy_token_ratio: float
    :param high_entropy_marker_start: Special token/marker to indicate the start of a high-entropy token. Default to "<HIGH_ENTROPY>"
    :type high_entropy_marker_start: str
    :param high_entropy_marker_end: Special token/marker to indicate the end of a high-entropy token. Default to "</HIGH_ENTROPY>"
    :type high_entropy_marker_end: str
    """
    def __init__(
        self,
        save_dir: str,
        tokenizer: Any,
        save_images_separately: bool = True,
        max_image_size: int = 512,
        mark_high_entropy_tokens: bool = False,
        high_entropy_token_ratio: float = 0.2,
        high_entropy_marker_start: str = "<HIGH_ENTROPY>",
        high_entropy_marker_end: str = "</HIGH_ENTROPY>",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.tokenizer = tokenizer
        self.save_images_separately = save_images_separately
        self.max_image_size = max_image_size
        self.mark_high_entropy_tokens = mark_high_entropy_tokens
        self.high_entropy_token_ratio = high_entropy_token_ratio
        self.high_entropy_marker_start = high_entropy_marker_start
        self.high_entropy_marker_end = high_entropy_marker_end

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

        # Check if any experience has action_entropy (silently)
        if self.mark_high_entropy_tokens:
            has_entropy_count = sum(1 for exp in experiences if hasattr(exp, 'action_entropy') and exp.action_entropy is not None)

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
            return []
        elif len(sequences.shape) == 1:
            # 1D tensor - reshape to (1, seq_len)
            sequences = sequences.unsqueeze(0)
        elif len(sequences.shape) != 2:
            # Unexpected shape
            return []

        batch_size = sequences.shape[0]

        # Handle action_mask with same shape validation
        if exp.action_mask is not None:
            action_mask = exp.action_mask.cpu()
            if len(action_mask.shape) == 1:
                action_mask = action_mask.unsqueeze(0)
            elif len(action_mask.shape) != 2:
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

                        # Check if we should mark high-entropy tokens
                        high_entropy_mask = None
                        if self.mark_high_entropy_tokens:
                            # Extract single sample's action_mask (1D)
                            sample_action_mask = action_mask[i] if len(action_mask.shape) == 2 else action_mask
                            high_entropy_mask = self._get_high_entropy_mask(exp, i, sample_action_mask)

                        # generated_text includes the last prompt token (for RL state-action pairing)
                        generated_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                        if high_entropy_mask is not None:
                            generated_text = self._mark_high_entropy_tokens_in_text(
                                generated_text, gen_tokens, gen_indices, high_entropy_mask
                            )

                        # pure_generated_text excludes the last prompt token (only model's output)
                        if len(adjusted_indices) > 1:
                            pure_gen_tokens = sequences[i][adjusted_indices[1:]]
                            pure_gen_indices = gen_indices[1:]
                            pure_generated_text = self.tokenizer.decode(pure_gen_tokens, skip_special_tokens=True)
                            if high_entropy_mask is not None:
                                pure_generated_text = self._mark_high_entropy_tokens_in_text(
                                    pure_generated_text, pure_gen_tokens, pure_gen_indices, high_entropy_mask
                                )
                        else:
                            pure_generated_text = ""
                else:
                    generated_text = ""
                    pure_generated_text = ""
            except (IndexError, RuntimeError) as e:
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

    def _get_high_entropy_mask(self, exp: Any, sample_idx: int, action_mask: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get high-entropy token mask for a specific sample.

        :param exp: Experience object
        :type exp: Any
        :param sample_idx: Index of the sample in the batch
        :type sample_idx: int
        :param action_mask: Action mask tensor for the sample
        :type action_mask: torch.Tensor
        :return: High-entropy mask for the sample, or None if not available
        :rtype: Optional[torch.Tensor]
        """
        # Check if action_entropy exists (without using getattr)
        if not hasattr(exp, 'action_entropy'):
            return None
        
        action_entropy = exp.action_entropy
        if action_entropy is None:
            return None

        # Move to CPU and extract sample
        action_entropy = action_entropy.cpu()
        
        # Get action_mask length for this sample
        action_mask_len = action_mask.shape[0] if len(action_mask.shape) > 0 else 0
        
        # Handle different tensor shapes
        if len(action_entropy.shape) == 0:
            return None
        elif len(action_entropy.shape) == 1:
            # 1D tensor - could be:
            # 1. Single sample's entropy (matches action_mask length)
            # 2. Packed batch's entropy (concatenated from multiple samples)
            if action_entropy.shape[0] == action_mask_len:
                # Case 1: Direct match - use as is
                sample_entropy = action_entropy
            elif action_entropy.shape[0] > action_mask_len:
                # Case 2: Packed batch - need to extract the correct slice
                # Try to find the correct slice by checking if action_mask lengths can be inferred
                # from the full action_mask in the experience
                if hasattr(exp, 'action_mask') and exp.action_mask is not None:
                    full_action_mask = exp.action_mask.cpu()
                    if len(full_action_mask.shape) == 2:
                        # 2D action_mask: (batch_size, max_actions)
                        # Calculate cumulative lengths to find the slice
                        batch_size = full_action_mask.shape[0]
                        cumulative_lengths = []
                        current_length = 0
                        for i in range(batch_size):
                            # Count non-padding tokens (assuming padding is False/0)
                            mask_row = full_action_mask[i]
                            num_actions = mask_row.sum().item() if mask_row.dtype == torch.bool else (mask_row != 0).sum().item()
                            cumulative_lengths.append((current_length, current_length + num_actions))
                            current_length += num_actions
                        
                        # Check if the total matches action_entropy length
                        if current_length == action_entropy.shape[0] and sample_idx < len(cumulative_lengths):
                            start_idx, end_idx = cumulative_lengths[sample_idx]
                            sample_entropy = action_entropy[start_idx:end_idx]
                        else:
                            return None
                    elif len(full_action_mask.shape) == 1:
                        # 1D action_mask: single sample, but action_entropy is longer
                        # Use the passed action_mask length (which is for this specific sample)
                        if action_mask_len > 0:
                            sample_entropy = action_entropy[:action_mask_len]
                        else:
                            return None
                    else:
                        return None
                else:
                    # No action_mask to infer from, try to use the first action_mask_len elements
                    if action_mask_len > 0 and action_entropy.shape[0] >= action_mask_len:
                        sample_entropy = action_entropy[:action_mask_len]
                    else:
                        return None
            else:
                # action_entropy is shorter than action_mask - this shouldn't happen
                return None
        elif len(action_entropy.shape) == 2:
            # 2D tensor (batch, num_actions)
            if sample_idx >= action_entropy.shape[0]:
                return None
            sample_entropy = action_entropy[sample_idx]
            # If the extracted entropy is longer than action_mask, truncate it
            if len(sample_entropy.shape) > 0 and sample_entropy.shape[0] > action_mask_len:
                sample_entropy = sample_entropy[:action_mask_len]
        else:
            return None

        # Verify that sample_entropy length matches action_mask length
        if len(sample_entropy.shape) > 0 and sample_entropy.shape[0] != action_mask_len:
            return None

        # Create high-entropy mask using the utility function
        from lightrft.models.utils import create_high_entropy_mask
        
        # Reshape to (1, num_actions) for create_high_entropy_mask
        # action_mask is already for a single sample (1D), so we need to add batch dimension
        sample_entropy_2d = sample_entropy.unsqueeze(0)
        
        # Ensure action_mask is 1D and add batch dimension
        if len(action_mask.shape) == 1:
            sample_action_mask = action_mask.unsqueeze(0)
        elif len(action_mask.shape) == 2:
            # If it's 2D, take the first row (should be the sample we're processing)
            sample_action_mask = action_mask[:1]
        else:
            return None
        
        high_entropy_mask = create_high_entropy_mask(
            sample_entropy_2d,
            sample_action_mask,
            self.high_entropy_token_ratio
        )
        
        # Return the mask for this sample (remove batch dimension)
        result_mask = high_entropy_mask[0]
        return result_mask

    def _mark_high_entropy_tokens_in_text(
        self,
        text: str,
        tokens: torch.Tensor,
        action_indices: torch.Tensor,
        high_entropy_mask: torch.Tensor,
    ) -> str:
        """
        Mark high-entropy tokens in the decoded text with special markers.

        This method decodes tokens one by one and wraps high-entropy tokens with
        special markers. Since tokenizer.decode may merge tokens, we decode each
        token individually and reconstruct the text with markers.

        :param text: Decoded text string (for reference, but we reconstruct it)
        :type text: str
        :param tokens: Token IDs tensor
        :type tokens: torch.Tensor
        :param action_indices: Indices in action_mask space corresponding to tokens
        :type action_indices: torch.Tensor
        :param high_entropy_mask: Binary mask indicating high-entropy tokens (1 for high-entropy)
        :type high_entropy_mask: torch.Tensor
        :return: Text with high-entropy tokens marked
        :rtype: str
        """
        if high_entropy_mask is None or len(tokens) == 0:
            return text

        # Convert to lists for easier processing
        tokens_list = tokens.tolist()
        action_indices_list = action_indices.tolist()
        
        # Decode each token individually and mark high-entropy ones
        marked_parts = []
        
        # Debug: log mask info (only print once to avoid spam)
        # Note: This method is called multiple times, so we'll skip detailed logging here
        
        for token_id, action_idx in zip(tokens_list, action_indices_list):
            # Decode this token
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            
            # Check if this token is high-entropy
            # action_idx should be within [0, len(high_entropy_mask))
            if action_idx < len(high_entropy_mask) and action_idx >= 0:
                is_high_entropy = high_entropy_mask[action_idx].item() > 0.5
                if is_high_entropy:
                    # Mark as high-entropy
                    marked_parts.append(self.high_entropy_marker_start)
                    marked_parts.append(token_text)
                    marked_parts.append(self.high_entropy_marker_end)
                else:
                    # Regular token
                    marked_parts.append(token_text)
            else:
                # Index out of range - just add token without marking
                marked_parts.append(token_text)
        
        result = ''.join(marked_parts)
        # Debug: check if markers were added (only log once to avoid spam)
        # Note: We'll check this in the main loop instead
        return result


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
    # Check if save_trajectories is enabled (without using getattr)
    if not hasattr(args, 'save_trajectories') or not args.save_trajectories:
        return None

    save_dir = os.path.join(args.save_path, "trajectories")

    # Extract configuration options (without using getattr)
    mark_high_entropy = False
    high_entropy_ratio = 0.2
    marker_start = "<HIGH_ENTROPY>"
    marker_end = "</HIGH_ENTROPY>"
    
    if hasattr(args, 'mark_high_entropy_tokens'):
        mark_high_entropy = args.mark_high_entropy_tokens
    if hasattr(args, 'high_entropy_token_ratio'):
        high_entropy_ratio = args.high_entropy_token_ratio

    return TrajectorySaver(
        save_dir=save_dir,
        tokenizer=tokenizer,
        save_images_separately=True,
        max_image_size=512,
        mark_high_entropy_tokens=mark_high_entropy,
        high_entropy_token_ratio=high_entropy_ratio,
        high_entropy_marker_start=marker_start,
        high_entropy_marker_end=marker_end,
    )
