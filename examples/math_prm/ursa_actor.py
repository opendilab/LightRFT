"""
URSA-8B Actor Model Loader

This module provides a custom actor loader for URSA-8B models, which use
UrsaForConditionalGeneration instead of the standard AutoModelForVision2Seq.

URSA-8B architecture:
- Hybrid vision tower: SAM-B (1024x1024) + SigLIP-L (384x384)
- MLP projector: Maps vision features to LLM embedding space
- Language model: Qwen2.5-Math-Instruct (8B params)
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
from transformers.integrations.deepspeed import HfDeepSpeedConfig

# Add current directory to path for ursa_model imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from ursa_model import UrsaForConditionalGeneration
from lightrft.models.actor_vl import ActorVL
from lightrft.models.utils import apply_lora_configuration, reset_position_ids, entropy_from_logits


class UrsaActor(ActorVL):
    """
    Actor wrapper for URSA-8B models.

    This class extends ActorVL to support loading URSA-8B models using
    UrsaForConditionalGeneration instead of AutoModelForVision2Seq.

    Usage:
        actor = UrsaActor(
            pretrain_or_model="/path/to/URSA-8B",
            use_flash_attention_2=True,
            bf16=True,
            lora_rank=0,
        )
    """

    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        packing_samples=False,
        high_entropy_token_ratio=0.0,
        **kwargs,
    ) -> None:
        """
        Initialize URSA-8B actor model.

        Args:
            pretrain_or_model: Path to URSA-8B checkpoint or model instance
            use_flash_attention_2: Enable Flash Attention 2.0
            bf16: Use bfloat16 precision
            lora_rank: LoRA rank (0 disables LoRA)
            lora_alpha: LoRA alpha scaling parameter
            lora_dropout: LoRA dropout rate
            target_modules: Target modules for LoRA (auto-detected if None)
            ds_config: DeepSpeed configuration
            device_map: Device mapping for model placement
            packing_samples: Enable sample packing
            high_entropy_token_ratio: High entropy token filtering ratio
        """
        # Initialize parent class without calling its __init__
        # We'll handle model loading ourselves
        nn.Module.__init__(self)
        self.high_entropy_token_ratio = high_entropy_token_ratio

        if isinstance(pretrain_or_model, str):
            self.pretrain_or_model = pretrain_or_model
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

            # DeepSpeed ZeRO-3 integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None  # noqa: F841

            # Prepare loading kwargs
            from_pretrained_kwargs = {
                "trust_remote_code": True,
                "attn_implementation": attn_implementation,
                "torch_dtype": torch.bfloat16 if bf16 else "auto",
            }

            # Check if we're in meta device context (FSDP)
            try:
                test_tensor = torch.empty(1)
                is_meta_context = test_tensor.is_meta
            except:  # noqa
                is_meta_context = False

            if not is_meta_context and device_map is not None:
                from_pretrained_kwargs["device_map"] = device_map

            print(f"[UrsaActor] Loading URSA-8B model from {pretrain_or_model}")

            # Load URSA model using UrsaForConditionalGeneration
            self.model = UrsaForConditionalGeneration.from_pretrained(
                pretrain_or_model,
                **from_pretrained_kwargs
            )

            print(f"[UrsaActor] Successfully loaded URSA-8B model")

            # Apply LoRA if requested
            if lora_rank > 0:
                print(f"[UrsaActor] Applying LoRA with rank={lora_rank}, alpha={lora_alpha}")
                self.model = apply_lora_configuration(
                    model=self.model,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules,
                    freeze_vision_tower=True,
                )

            # Disable cache for training
            self.model.config.use_cache = False

            # Enable sample packing if requested
            self.packing_samples = packing_samples
        else:
            # Model instance provided directly
            self.model = pretrain_or_model
            self.pretrain_or_model = "ursa"

        print(f"[UrsaActor] Model type: {self.pretrain_or_model}")

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions=None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        return_output: bool = False,
        packed_seq_lens: Optional[list] = None,
    ) -> torch.Tensor:
        """
        VLM-aligned forward.

        URSA's vision tower expands every <|image|> placeholder into 576 vision
        tokens during the LM forward, so ``output["logits"]`` is longer than the
        input ``sequences`` along the seq dim. The default ``ActorVL.forward``
        feeds ``output["logits"][:, :-1, :]`` (length E-1) and ``sequences[:, 1:]``
        (length T-1) into ``log_probs_from_logits``, which then hits PyTorch's
        ``gather(dim=-1, index=...)`` — that op silently TRUNCATES the rows of
        ``logits`` to ``len(labels)`` instead of erroring. The result: log-probs
        are read from the wrong (vision-token / early-prompt) positions, never
        from the actual generation positions. KL/PPO/ratio all become noise.

        We sidestep the bug entirely by slicing the logits to the action range
        on the seq dim first (where alignment is unambiguous because generation
        always lives at the tail of the expanded sequence), then using a single
        ``F.log_softmax + gather`` over the action labels. fp32 throughout so
        the precision matches the rest of the PPO loss path.
        """
        if self.packing_samples:
            position_ids = reset_position_ids(attention_mask)
            attention_mask_for_model = None
        else:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            attention_mask_for_model = attention_mask

        # Sanitize actor-leaked image tokens before forward. During GRPO rollout
        # an actor can generate literal `<|image|>` / `<image>` strings inside
        # its response (rare but observed; freq goes up when KL spikes hard
        # late in training). The tokenizer then maps those to image_token_index,
        # so the sequence ends up with MORE image-token slots than the prompt
        # actually had images. URSA's vision merge requires
        # image_token_count == n_image_features and aborts with
        #   "The input provided to the model are wrong.
        #    The number of image tokens is N while the number of image given is M.
        #    This prevents correct indexing and breaks batch generation."
        # which crashes the whole PPO step. cce5ae5 already fixed this on the
        # PRM forward path; the actor forward needs the same protection because
        # the same actor-generated sequences are replayed here every PPO inner
        # epoch. Align both directions:
        #   * token_count > image_count : extras are leaked, replace with pad.
        #   * token_count < image_count : truncate pixel_values/image_grid_thw.
        sequences, pixel_values, image_grid_thw = self._align_image_tokens_to_images(
            sequences, pixel_values, image_grid_thw
        )

        forward_kwargs = dict(
            attention_mask=attention_mask_for_model,
            position_ids=position_ids,
            pixel_values=self._cast_multimodal_tensor(pixel_values),
            image_grid_thw=image_grid_thw,
            pixel_values_videos=self._cast_multimodal_tensor(pixel_values_videos),
            video_grid_thw=video_grid_thw,
        )
        for k in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            if not self._supports_model_kwarg(k):
                forward_kwargs.pop(k, None)

        output = self.model(sequences, **forward_kwargs)

        if num_actions is None:
            assert return_output
            return output

        logits = output["logits"]
        seq_T = sequences.size(1)
        logit_T = logits.size(1)
        if self.packing_samples:
            raise NotImplementedError(
                "UrsaActor.forward does not yet support packed_seq_lens. The "
                "default ActorVL packing path is silently miscomputed for VLMs "
                "that expand image placeholders; we don't want to bake the same "
                "bug in here. Add explicit packed-aware alignment when needed."
            )

        # Generation tokens always sit at the tail of the expanded sequence,
        # so logits at expanded positions [E - num_actions - 1 .. E - 2]
        # predict tokens at expanded positions [E - num_actions .. E - 1] —
        # which are the same generation tokens as ``sequences[:, -num_actions:]``
        # in the unexpanded view (the unexpanded vs expanded offset only affects
        # positions BEFORE the image placeholders, all in the prompt).
        action_logits = logits[:, -(num_actions + 1):-1, :]
        action_labels = sequences[:, -num_actions:]
        if action_logits.size(1) != action_labels.size(1):
            raise RuntimeError(
                f"action_logits seq len {action_logits.size(1)} does not match "
                f"action_labels seq len {action_labels.size(1)} "
                f"(num_actions={num_actions}, seq_T={seq_T}, logit_T={logit_T})"
            )

        action_logp_full = F.log_softmax(action_logits.float(), dim=-1)
        action_log_probs = action_logp_full.gather(
            -1, action_labels.unsqueeze(-1)
        ).squeeze(-1)

        if self.high_entropy_token_ratio > 0.0:
            # Entropy of the action-position distribution, in the same fp32 used above.
            probs = action_logp_full.exp()
            action_entropy = -(probs * action_logp_full).sum(dim=-1)
        else:
            action_entropy = None

        if return_output:
            if action_entropy is not None:
                output_dict = dict(output)
                output_dict["action_entropy"] = action_entropy
                return (action_log_probs, output_dict)
            return (action_log_probs, output)
        return action_log_probs

    def _align_image_tokens_to_images(self, sequences, pixel_values, image_grid_thw):
        """Make ``sequences``'s image-token count match the actual image count.

        URSA's vision merge crashes with::

            ValueError: The input provided to the model are wrong.
            The number of image tokens is N while the number of image given is M.
            This prevents correct indexing and breaks batch generation.

        whenever the count of ``image_token_index`` markers inside the input
        sequence is unequal to the number of image features supplied via
        ``pixel_values`` (one row of ``image_grid_thw`` per image). During GRPO
        rollout this can happen because the actor occasionally generates a
        literal ``<|image|>`` / ``<image>`` token inside its response — usually
        rare, but observed to fire when KL drifts very high mid-training and
        the actor's output distribution gets unstable.

        Strategy (no-op fast path when counts already agree):

          * count_tok > count_img : leaked extras — replace the trailing
            (count_tok - count_img) image-token slots in each row with a benign
            text token (pad / eos). Keep the first count_img in their original
            positions so the vision features still merge into the prompt.

          * count_tok < count_img : sequence is missing some image-token slots
            relative to the image features. Truncate ``image_grid_thw`` and the
            corresponding rows of ``pixel_values`` to the first count_tok
            images. (Information loss is unavoidable, but the PPO step still
            makes progress on the other tokens.)

        Returns the (possibly sanitized) ``(sequences, pixel_values, image_grid_thw)``.
        Original tensors are returned unchanged when no mismatch exists.
        """
        if sequences is None:
            return sequences, pixel_values, image_grid_thw
        image_token_id = getattr(self.model.config, "image_token_index", None)
        if image_token_id is None:
            return sequences, pixel_values, image_grid_thw
        if pixel_values is None and image_grid_thw is None:
            # Pure text micro-batch — nothing to align; also strip any leaked
            # image tokens so the LM head doesn't see them as content.
            n_tok = int((sequences == image_token_id).sum().item())
            if n_tok == 0:
                return sequences, pixel_values, image_grid_thw

        # Per-row image-token positions (flat indices).
        # We sanitize in-place on a clone to avoid mutating shared rollout buffers.
        seq = sequences.clone()
        flat = seq.view(-1)
        tok_positions = torch.nonzero(flat == image_token_id, as_tuple=False).squeeze(-1)
        n_tok = int(tok_positions.numel())

        # Number of images supplied. image_grid_thw has one row per image and
        # is the more reliable source than pixel_values (which may be packed).
        if image_grid_thw is not None:
            n_img = int(image_grid_thw.size(0))
        elif pixel_values is not None:
            # Fallback: assume one image per row of pixel_values.
            n_img = int(pixel_values.size(0)) if pixel_values.dim() >= 1 else 0
        else:
            n_img = 0

        if n_tok == n_img:
            return sequences, pixel_values, image_grid_thw

        replacement = None
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is not None:
            replacement = tokenizer.pad_token_id
            if replacement is None:
                replacement = tokenizer.eos_token_id
        if replacement is None:
            # Last-resort: pick a known safe id (eos is usually safe across HF tokenizers).
            replacement = int(getattr(self.model.config, "eos_token_id", 0) or 0)

        if n_tok > n_img:
            # Leaked extras — replace tail extras with pad/eos so token_count == n_img.
            extras = tok_positions[n_img:]
            flat[extras] = replacement
            seq = flat.view_as(sequences)
            return seq, pixel_values, image_grid_thw

        # n_tok < n_img: truncate image features to match token slots.
        new_grid = image_grid_thw[:n_tok] if image_grid_thw is not None else None
        new_pixel = pixel_values
        if pixel_values is not None and image_grid_thw is not None and n_tok > 0:
            # pixel_values is the concat of per-image patches; the per-image
            # row counts come from image_grid_thw[i, 0] * thw[i, 1] * thw[i, 2].
            # Keep the first n_tok image's patches.
            patch_counts = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).long()
            keep = int(patch_counts[:n_tok].sum().item())
            new_pixel = pixel_values[:keep]
        elif pixel_values is not None and n_tok == 0:
            new_pixel = None
            new_grid = None
        return sequences, new_pixel, new_grid


def create_ursa_actor(args, ds_config=None):
    """
    Factory function to create URSA-8B actor from training args.

    Args:
        args: Training arguments (argparse.Namespace)
        ds_config: DeepSpeed configuration dict

    Returns:
        UrsaActor instance
    """
    return UrsaActor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=getattr(args, 'load_in_4bit', False),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=getattr(args, 'target_modules', None),
        lora_dropout=args.lora_dropout,
        ds_config=ds_config,
        packing_samples=args.packing_samples,
        disable_logprobs_flashattn=getattr(args, 'disable_logprobs_flashattn', False),
        fused_linear_logprob=getattr(args, 'fused_linear_logprob', False),
    )
