#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script converts a PyTorch Distributed Checkpoint (DCP) to a standard Hugging Face checkpoint.

The script is adaptive, capable of automatically detecting and handling both
Causal Language Models (LLMs) and Vision Language Models (VLMs).
"""

import argparse
import shutil
from pathlib import Path
from typing import Type

import torch
import torch.distributed.checkpoint as dcp
# Import necessary Hugging Face classes
from transformers import (AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq, PreTrainedModel)


class ActorWrapper(torch.nn.Module):
    """A simple wrapper class for a model.

    This class is sometimes used in training frameworks to wrap a model.
    This conversion script can handle models that are wrapped in this class.
    """
    def __init__(self, model: torch.nn.Module):
        """
        :param model: The PyTorch model to be wrapped.
        :type model: torch.nn.Module
        """
        super().__init__()
        self.model = model


@torch.inference_mode()
def convert_hf_checkpoint(
    hf_base_path: Path,
    fsdp_ckpt_path: Path,
    output_dir: Path,
) -> None:
    """Converts a PyTorch Distributed Checkpoint (DCP) to a standard Hugging Face checkpoint.

    This script is adaptive and works for both Causal Language Models (LLMs) and
    Vision Language Models (VLMs).

    :param hf_base_path: Path to the directory containing the original Hugging Face model files
                         (especially the config.json).
    :type hf_base_path: Path
    :param fsdp_ckpt_path: Path to the directory containing the DCP checkpoint.
    :type fsdp_ckpt_path: Path
    :param output_dir: Path to the directory where the converted Hugging Face checkpoint will be saved.
    :type output_dir: Path
    """
    # --- 1. Load model configuration ---
    # Load the model's configuration file from the base Hugging Face model directory.
    config = AutoConfig.from_pretrained(hf_base_path)

    # --- 2. Determine the appropriate AutoModel class (LLM vs VLM) ---
    # We inspect the 'architectures' field in the config to reliably determine the model type.
    model_class: Type[PreTrainedModel] = AutoModelForCausalLM  # Default to CausalLM
    if getattr(config, "architectures", None):
        arch = config.architectures[0]
        # Use a heuristic to detect if the model is a Vision Language Model.
        if ("Vision" in arch or "Vlm" in arch or "Llava" in arch or "VL" in arch or "Qwen2_5_VL" in arch):
            print(f"Detected Vision Language Model architecture: {arch}", flush=True)
            model_class = AutoModelForVision2Seq
        else:
            print(f"Detected Language Model architecture: {arch}", flush=True)
            model_class = AutoModelForCausalLM
    else:
        # Fallback if 'architectures' is not found, with a warning.
        print(
            "Warning: 'architectures' field not found in config.json. Defaulting to AutoModelForCausalLM.", flush=True
        )

    # --- 3. Instantiate a shell model on the 'meta' device ---
    # Using torch.device('meta') allows us to instantiate the model structure
    # without allocating any memory for its weights. This is highly efficient for large models.
    with torch.device('meta'):
        try:
            hf_model = model_class.from_config(config)
        except ValueError as e:
            # Fallback mechanism: sometimes VLM configs are tricky, try AutoModel if Vision2Seq fails
            print(f"Error initializing with {model_class.__name__}: {e}")
            print("Attempting fallback to AutoModelForCausalLM (just in case) or checking config...", flush=True)
            raise e

    # Move the model to CPU to create empty tensors for the state_dict.
    # This prepares the model to receive the actual weights.
    hf_model = hf_model.to_empty(device='cpu')

    # --- 4. Load the DCP checkpoint into the model's state dictionary ---
    # Get the state_dict from the empty model, which contains all parameter names.
    model_state_dict = hf_model.state_dict()

    # Create a reader to access the distributed checkpoint from the filesystem.
    fs_storage_reader = torch.distributed.checkpoint.FileSystemReader(fsdp_ckpt_path)

    # Load the checkpoint data directly into our model's state_dict.
    # This is the core conversion step.
    dcp.load(state_dict=model_state_dict, storage_reader=fs_storage_reader)
    print("Successfully loaded state_dict from DCP checkpoint.", flush=True)

    # Apply the now-filled state_dict to the model instance.
    hf_model.load_state_dict(model_state_dict)

    # If the saved model was wrapped (e.g., in an ActorWrapper), unwrap it to get the base model.
    if isinstance(hf_model, ActorWrapper):
        hf_model = hf_model.model

    # --- 5. Save the converted model in Hugging Face format ---
    # Create the output directory if it doesn't exist.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the model using the standard Hugging Face `save_pretrained` method.
    # `safe_serialization=True` saves the model using the safer and more modern .safetensors format.
    print(f"Saving Hugging Face model to {output_path}...", flush=True)
    hf_model.save_pretrained(output_path, safe_serialization=True)

    # --- 6. Copy necessary configuration and tokenizer files ---
    # The saved model needs its config, tokenizer, and other metadata files to be complete.
    hf_base_path_obj = Path(hf_base_path)
    for item in hf_base_path_obj.iterdir():
        # Copy all files that are not model weights or checkpoint metadata.
        # This ensures the new model directory is self-contained and usable.
        if item.is_file() and not item.name.endswith((".safetensors", ".bin", ".pth", ".index.json")):
            shutil.copy(item, output_path / item.name)

    print(f"Hugging Face checkpoint successfully saved to {output_path}", flush=True)


if __name__ == '__main__':
    """
    Usage Example:
        python fsdp2hf.py --hf_base path/to/hf_model --checkpoint_dir path/to/fsdp_ckpt --output_dir path/to/save_dir

    Arguments:
        --hf_base: Path to the original Hugging Face model directory. This directory should contain
                   at least the `config.json` and other necessary json files (e.g., for the tokenizer).
        --checkpoint_dir: Path to the directory where the DCP checkpoint is stored.
        --output_dir: Path to the directory where the final, converted Hugging Face model will be saved.
    """
    parser = argparse.ArgumentParser(description='Convert DCP to HuggingFace checkpoint (adaptive for LLM/VLM).')
    parser.add_argument(
        '--hf_base', type=Path, required=True, help='Path to the base Hugging Face model config directory.'
    )
    parser.add_argument('--checkpoint_dir', type=Path, required=True, help='Path to the DCP checkpoint directory.')
    parser.add_argument(
        '--output_dir', type=Path, required=True, help='Path to save the converted Hugging Face checkpoint.'
    )

    args = parser.parse_args()
    convert_hf_checkpoint(
        args.hf_base,
        args.checkpoint_dir,
        args.output_dir,
    )
