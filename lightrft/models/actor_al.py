"""
Audio-language actor for reinforcement learning.

Provides the ActorAL (Audio-language) class: an actor that generates text (actions) from audio and text inputs. Supports LoRA, Flash Attention 2, DeepSpeed,
sample packing, gradient checkpointing, and MoE.

"""

from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import Qwen2AudioForConditionalGeneration
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from .actor_modality import ActorModality
from .utils import apply_lora_configuration, log_probs_from_logits, reset_position_ids


class _AudioEmbedPositions(nn.Module):
    """FSDP2-safe replacement for ``nn.Embedding`` used in Whisper's audio tower.

    FSDP2 wraps each ``nn.Embedding`` separately. Whisper uses
    ``embed_positions.weight`` directly instead of calling ``forward()``, so
    the weight can be a sharded DTensor while conv outputs are full tensors,
    causing a mixed Tensor/DTensor error on ``inputs_embeds + embed_pos``.

    This plain ``nn.Module`` is not wrapped by FSDP2; its weight stays in the
    parent (root) FSDP unit and is all-gathered with the conv weights.
    """

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.weight = embedding.weight  # same Parameter, no copy
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return nn.functional.embedding(input_ids, self.weight)


class ActorAL(nn.Module):
    """
    Audio-language actor for RL: generates text (actions) from audio and text inputs.

    Supports LoRA, quantization, and distributed training. Can be initialized from
    a pretrained path or an existing model instance.

    :param pretrain_or_model: Path to a pretrained model or an existing model instance.
    :type pretrain_or_model: Union[str, nn.Module]
    :param use_flash_attention_2: Whether to utilize Flash Attention 2.0 for improved performance
    :type use_flash_attention_2: bool
    :param bf16: Enable bfloat16 precision for model computations
    :type bf16: bool
    :param lora_rank: Rank for LoRA adaptation (0 disables LoRA)
    :type lora_rank: int
    :param lora_alpha: Alpha parameter for LoRA scaling
    :type lora_alpha: int
    :param lora_dropout: Dropout rate for LoRA layers
    :type lora_dropout: float
    :param target_modules: List of target modules for applying LoRA (auto-detected if None)
    :type target_modules: Optional[list]
    :param ds_config: Configuration for DeepSpeed distributed training
    :type ds_config: Optional[dict]
    :param device_map: Device mapping for loading the model onto specific devices
    :type device_map: Optional[dict]
    :param packing_samples: Whether to pack samples during training for efficiency
    :type packing_samples: bool

    Example::

        # Initialize with a pretrained model path
        actor = ActorAL(
            pretrain_or_model="Qwen/Qwen2-Audio-7B-Instruct",
            use_flash_attention_2=True,
            lora_rank=16,
            lora_alpha=32
        )

        # Generate responses
        sequences, attention_mask, action_mask = actor.generate(
            input_ids=input_tensor,
            audio_values=audio_features_tensor,
            max_new_tokens=100
        )
    """
    # Model modality declaration - defines what types of inputs this model accepts
    modality = ActorModality.AUDIO_LANGUAGE

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
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(pretrain_or_model, str):
            self.pretrain_or_model = pretrain_or_model
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)  # noqa: F841
            else:
                dschf = None  # noqa: F841

            # Load Qwen2Audio model
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
            )

            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                self.model = apply_lora_configuration(
                    model=self.model,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules,
                    freeze_vision_tower=False,  # No vision tower for audio models
                )

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False

            # packing samples using Flash Attention 2
            self.packing_samples = packing_samples
        else:
            self.model = pretrain_or_model
            self.pretrain_or_model = pretrain_or_model.config.model_type

        # ------------------------------------------------------------------
        # FSDP2 compatibility fixes for the Whisper-based audio tower.
        #
        # 1. Replace embed_positions (nn.Embedding → _AudioEmbedPositions)
        #    so FSDP2 does not individually wrap it.  Whisper accesses
        #    embed_positions.weight directly, bypassing the module forward
        #    and thus FSDP's all-gather hook.
        #
        # 2. Force the audio tower to use **eager** (non-flash) attention.
        #    FSDP2 turns parameters into DTensors; computations with those
        #    parameters also produce DTensors.  Flash Attention's CUDA
        #    kernels (flash_attn_gpu.varlen_fwd) receive a mix of DTensor
        #    activations and regular-tensor cu_seqlens, leading to shape
        #    mismatches ("cu_seqlens_q must have shape (batch_size + 1)").
        #    The Whisper encoder is small (~12 layers), so using eager
        #    attention has negligible impact on overall training throughput.
        # ------------------------------------------------------------------
        audio_tower = getattr(self.model, "audio_tower", None) or getattr(self.model, "audio_encoder", None)
        if audio_tower is not None:
            # Fix 1: embed_positions
            if hasattr(audio_tower, "embed_positions") and isinstance(audio_tower.embed_positions, nn.Embedding):
                audio_tower.embed_positions = _AudioEmbedPositions(audio_tower.embed_positions)
                print(
                    "[ActorAL] Replaced audio_tower.embed_positions "
                    "(nn.Embedding → _AudioEmbedPositions) for FSDP2 compat"
                )

            # Fix 2: force eager attention in the audio encoder
            for module in audio_tower.modules():
                if hasattr(module, "_attn_implementation"):
                    module._attn_implementation = "eager"
            # Also patch the config so any lazily-constructed layers use eager
            audio_cfg = getattr(self.model.config, "audio_config", None)
            if audio_cfg is not None:
                audio_cfg._attn_implementation = "eager"
            print("[ActorAL] Set audio_tower attention to 'eager' for FSDP2 compat")

        print("pretrain_or_model: ", self.pretrain_or_model)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        audio_values: torch.Tensor = None,
        **kwargs
    ) -> Union[
        Tuple[torch.LongTensor, torch.LongTensor],
        Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor], ]:
        """
        Generate text sequences based on input text and audio information.

        This method performs text generation conditioned on both textual prompts and audio inputs.
        It handles the generation process with various sampling strategies and returns the generated
        sequences along with attention masks and action masks for RL training.

        :param input_ids: Input token IDs representing the text prompt
        :type input_ids: torch.Tensor
        :param audio_values: Preprocessed audio features (mel-spectrogram) for Qwen2-Audio
        :type audio_values: torch.Tensor
        :param kwargs: Additional generation parameters (top_k, top_p, temperature, etc.)
        :type kwargs: dict

        :return: Tuple containing generated sequences, attention mask, and action mask
        :rtype: Union[Tuple[torch.LongTensor, torch.LongTensor], Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]]  # noqa

        Example::

            sequences, attention_mask, action_mask = actor.generate(
                input_ids=torch.tensor([[1, 2, 3]]),
                audio_values=audio_features_tensor,
                max_new_tokens=50,
                temperature=0.8,
                do_sample=True
            )
        """
        # Pipeline may pass audio as pixel_values; use audio_values consistently.
        if audio_values is None:
            audio_values = kwargs.pop("input_features", None)
        if audio_values is None:
            audio_values = kwargs.get("pixel_values")

        generate_args = {
            "input_ids": input_ids,
            "top_k": kwargs.get("top_k", None),
            "top_p": kwargs.get("top_p", None),
            "do_sample": kwargs.get("do_sample", True),
            "early_stopping": kwargs.get("num_beams", 1) > 1,
            "temperature": kwargs.get("temperature", 1),
            "use_cache": True,
            "num_beams": kwargs.get("num_beams", 1),
            "attention_mask": kwargs.get("attention_mask"),
            "eos_token_id": kwargs.get("eos_token_id"),
            "pad_token_id": kwargs.get("pad_token_id"),
            "min_new_tokens": kwargs.get("min_new_tokens", 1),
        }

        if audio_values is not None:
            # Pad mel features to 3000 if shorter (see forward() for rationale)
            input_features, feature_attention_mask = self._prepare_audio_features(audio_values)
            generate_args["input_features"] = input_features
            generate_args["feature_attention_mask"] = feature_attention_mask

        if kwargs.get("max_new_tokens", None):
            generate_args["max_new_tokens"] = kwargs.get("max_new_tokens")
        if kwargs.get("max_length", None):
            generate_args["max_length"] = kwargs.get("max_length")

        # Call generate
        sequences = self.model.generate(**generate_args)

        # Prepare mask tensor
        eos_token_id = generate_args["eos_token_id"]
        pad_token_id = generate_args["pad_token_id"]

        # Process generated sequences to create proper attention and action masks
        input_len = input_ids.size(1)
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)
        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)

        # For Llama3 and Qwen2 models, there are some eos_tokens in the middle of the prompt.
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
        attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)

        # in RL, state_i (current token) + action_i (next token) -> state_i+1 (next token)
        state_seq = sequences[:, input_len - 1:-1]
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        action_mask[:, 0] = 1

        return sequences, attention_mask, action_mask

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        return_output=False,
        packed_seq_lens: Optional[list[int]] = None,
        audio_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass to compute action log probabilities for reinforcement learning.

        This method processes input sequences and audio information to compute log probabilities
        of actions (tokens) for RL training. It supports both standard and packed sequence formats
        and can return either just the action log probabilities or the full model output.

        Callers pass preprocessed audio as ``audio_values``; the pipeline maps from the
        VL ``pixel_values`` slot to ``audio_values`` for this actor.

        :param sequences: Input token sequences
        :type sequences: torch.LongTensor
        :param num_actions: Number of action tokens to extract log probs for
        :type num_actions: Optional[Union[int, list[int]]]
        :param attention_mask: Attention mask for the sequences
        :type attention_mask: Optional[torch.Tensor]
        :param pixel_values: Unused (VL compatibility; audio pipeline passes audio_values)
        :type pixel_values: Optional[torch.Tensor]
        :param image_grid_thw: Unused (accepted for pipeline compatibility)
        :type image_grid_thw: Optional[torch.Tensor]
        :param pixel_values_videos: Unused (accepted for VL pipeline compatibility)
        :type pixel_values_videos: Optional[torch.Tensor]
        :param video_grid_thw: Unused (accepted for VL pipeline compatibility)
        :type video_grid_thw: Optional[torch.Tensor]
        :param return_output: Whether to return the full model output along with log probs
        :type return_output: bool
        :param packed_seq_lens: Sequence lengths for packed samples
        :type packed_seq_lens: Optional[list[int]]
        :param audio_values: Preprocessed audio features (mel-spectrogram from pipeline)
        :type audio_values: Optional[torch.Tensor]

        :return: Action log probabilities or tuple of (action_log_probs, output) if return_output=True
        :rtype: torch.Tensor

        Example::

            # Compute action log probabilities for RL training
            log_probs = actor(
                sequences=token_sequences,
                num_actions=10,
                audio_values=input_features_tensor,
            )

            # Get both log probs and full output
            log_probs, output = actor(
                sequences=token_sequences,
                num_actions=10,
                audio_values=input_features_tensor,
                return_output=True,
            )
        """
        if not self.packing_samples:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        else:
            # convert attention_mask to position_ids
            position_ids = reset_position_ids(attention_mask)
            # explicitly ignore attention_mask for packing_samples
            attention_mask = None

        # Pipeline passes audio as audio_values; Qwen2Audio expects input_features.
        input_features = audio_values

        model_kwargs = {
            "input_ids": sequences,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

        if input_features is not None:
            # ----------------------------------------------------------
            # Guard: only pass input_features when the sequences still
            # contain ``audio_token_id`` placeholders that the model's
            # ``_merge_input_ids_with_audio_features`` can locate.
            #
            # During GRPO/PPO training the sequences come from the
            # inference engine (vLLM/SGLang) which may *expand* the
            # single ``<|AUDIO|>`` placeholder into many tokens.  After
            # expansion the original ``audio_token_id`` no longer
            # appears in the token sequence, so the model's merge step
            # would fail with a shape-mismatch error.
            #
            # When the placeholder is absent we fall back to a text-only
            # forward.  Both the actor AND the reference model see the
            # same expanded sequences, so the log-prob *ratio* used for
            # the policy gradient is still consistent.
            # ----------------------------------------------------------
            audio_token_id = getattr(self.model.config, "audio_token_id", None)
            has_audio_placeholder = (audio_token_id is not None and (sequences == audio_token_id).any().item())

            if has_audio_placeholder:
                input_features, feature_attention_mask = self._prepare_audio_features(input_features)
                model_kwargs["input_features"] = input_features
                model_kwargs["feature_attention_mask"] = feature_attention_mask
            # else: audio_token_id absent → text-only forward (see comment above)

        output = self.model(**model_kwargs)

        if num_actions is None:  # default
            assert return_output
            return output

        log_probs = log_probs_from_logits(output["logits"][:, :-1, :], sequences[:, 1:])

        if not self.packing_samples:
            action_log_probs = log_probs[:, -num_actions:]
        else:
            assert isinstance(num_actions, list) and len(num_actions) == len(packed_seq_lens)
            action_log_probs = []
            offset = 0
            for num_action, seq_len in zip(num_actions, packed_seq_lens):
                start, end = max(0, offset + seq_len - num_action - 1), offset + seq_len - 1
                action_log_probs.append(log_probs[:, start:end])
                offset += seq_len
            action_log_probs = torch.cat(action_log_probs, dim=1)

        if return_output:
            return (action_log_probs, output)
        else:
            return action_log_probs

    @staticmethod
    def _prepare_audio_features(
        input_features: torch.Tensor,
        expected_mel_len: int = 3000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize audio features to the expected mel length and build a feature mask.

        :param input_features: Raw mel-spectrogram features (e.g. (B, C, T) or (B, T)).
        :param expected_mel_len: Target temporal length for Whisper encoder.
        :return: Tuple of (padded_or_truncated_features, feature_attention_mask).
        """
        actual_len = input_features.shape[-1]
        if actual_len < expected_mel_len:
            pad_len = expected_mel_len - actual_len
            input_features = torch.nn.functional.pad(input_features, (0, pad_len), value=0.0)
        elif actual_len > expected_mel_len:
            input_features = input_features[..., :expected_mel_len]
            actual_len = expected_mel_len

        feature_attention_mask = torch.zeros(
            input_features.shape[0],
            expected_mel_len,
            dtype=torch.long,
            device=input_features.device,
        )
        feature_attention_mask[:, :actual_len] = 1
        return input_features, feature_attention_mask

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        """
        Enable gradient checkpointing to reduce memory usage during training.

        Gradient checkpointing trades compute for memory by recomputing intermediate
        activations during the backward pass instead of storing them. This is particularly
        useful for training large audio-language models with limited GPU memory.

        :param gradient_checkpointing_kwargs: Additional arguments for gradient checkpointing
        :type gradient_checkpointing_kwargs: dict

        Example::

            # Enable gradient checkpointing with default settings
            actor.gradient_checkpointing_enable()

            # Enable with custom settings
            actor.gradient_checkpointing_enable({"use_reentrant": True})
        """
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """
        Disable gradient checkpointing to use normal forward/backward computation.

        This method restores the default behavior where all intermediate activations
        are stored during the forward pass for use in the backward pass. This increases
        memory usage but reduces computation time.

        Example::

            # Disable gradient checkpointing
            actor.gradient_checkpointing_disable()
        """
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        """
        Print information about trainable parameters in the model.

        This method displays the number and percentage of trainable parameters,
        which is particularly useful when using parameter-efficient methods like LoRA.
        It helps monitor the efficiency of the fine-tuning approach.

        Example::

            # Print trainable parameter statistics
            actor.print_trainable_parameters()
            # Output: trainable params: 4,194,304 || all params: 7,241,732,096 || trainable%: 0.058
        """
        self.model.print_trainable_parameters()

    def process_sequences(self, sequences: torch.Tensor, input_len: int, eos_token_id: int,
                          pad_token_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Called by `trainer/fast_exp_maker.py`.

        Process generated sequences to create proper attention and action masks.

        This method post-processes the generated sequences to ensure proper handling of
        end-of-sequence tokens and creates masks needed for reinforcement learning training.
        It handles edge cases like multiple EOS tokens and ensures consistent sequence formatting.

        :param sequences: Generated token sequences
        :type sequences: torch.Tensor
        :param input_len: Length of the input prompt
        :type input_len: int
        :param eos_token_id: End-of-sequence token ID
        :type eos_token_id: int
        :param pad_token_id: Padding token ID
        :type pad_token_id: int

        :return: Tuple of processed sequences, attention mask, and action mask
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """

        # Process generated sequences to create proper attention and action masks
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)

        # For Llama3 and Qwen2 models, there are some eos_tokens in the middle of the prompt.
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
        attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)

        # in RL, state_i (current token) + action_i (next token) -> state_i+1 (next token)
        state_seq = sequences[:, input_len - 1:-1]
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        action_mask[:, 0] = 1

        return sequences, attention_mask, action_mask
