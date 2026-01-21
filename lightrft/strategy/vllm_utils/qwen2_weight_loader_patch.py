"""
Patch for vLLM 0.13.0 Qwen2 model weight loading issue.

This module provides a monkey patch to fix the weight loading bug in vLLM 0.13.0
where the weight loader cannot find 'embed_tokens' in Qwen2ForCausalLM.

Issue: vLLM tries to load 'embed_tokens' at the top level, but it's actually at 'model.embed_tokens'
Solution: Patch the weight name mapping to add the 'model.' prefix where needed
"""

import logging
from typing import Iterable, Tuple
import torch

logger = logging.getLogger(__name__)


def apply_qwen2_weight_loader_patch():
    """
    Apply monkey patch to fix Qwen2 weight loading in vLLM 0.13.0.

    This function patches the Qwen2ForCausalLM.load_weights method to correctly
    handle weight names that need the 'model.' prefix.
    """
    try:
        # Try to import from vLLM 0.13.0 path
        from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM

        # Store the original load_weights method
        original_load_weights = Qwen2ForCausalLM.load_weights

        def patched_load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
            """
            Patched load_weights method that fixes weight name mapping.

            This method intercepts weight loading and adds the 'model.' prefix
            to weight names that need it (embed_tokens, layers, norm).
            """
            # Convert weights to a list so we can modify it
            weights_list = list(weights)
            fixed_weights = []

            for name, tensor in weights_list:
                # Fix weight names that are missing the 'model.' prefix
                # These are the main components of Qwen2Model that are inside the 'model' attribute
                if name.startswith('embed_tokens.') or name == 'embed_tokens':
                    fixed_name = f'model.{name}'
                    logger.debug(f"Fixing weight name: {name} -> {fixed_name}")
                    fixed_weights.append((fixed_name, tensor))
                elif name.startswith('layers.'):
                    fixed_name = f'model.{name}'
                    logger.debug(f"Fixing weight name: {name} -> {fixed_name}")
                    fixed_weights.append((fixed_name, tensor))
                elif name.startswith('norm.') or name == 'norm':
                    fixed_name = f'model.{name}'
                    logger.debug(f"Fixing weight name: {name} -> {fixed_name}")
                    fixed_weights.append((fixed_name, tensor))
                else:
                    # Keep the original name for other weights (like lm_head)
                    fixed_weights.append((name, tensor))

            # Call the original load_weights with fixed names
            return original_load_weights(self, fixed_weights)

        # Apply the patch
        Qwen2ForCausalLM.load_weights = patched_load_weights
        logger.info("Successfully applied Qwen2 weight loader patch for vLLM 0.13.0")
        return True

    except ImportError as e:
        logger.warning(f"Could not apply Qwen2 weight loader patch: {e}")
        logger.warning("This patch is only needed for vLLM 0.13.0 with Qwen2 models")
        return False
    except Exception as e:
        logger.error(f"Error applying Qwen2 weight loader patch: {e}")
        return False


def check_if_patch_needed():
    """
    Check if the patch is needed by testing if we can import vLLM 0.13.0 components.

    Returns:
        bool: True if patch is needed (vLLM 0.13.0 detected), False otherwise
    """
    try:
        from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
        # If we can import from the v1 path, we're likely on vLLM 0.13.0
        try:
            from vllm.v1.worker.gpu_worker import Worker
            return True
        except ImportError:
            return False
    except ImportError:
        return False
