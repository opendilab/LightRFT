"""
Audio Pipeline Extensions for LightRFT

This module provides audio-specific adaptations for running Qwen2-Audio
within LightRFT's VL (Vision-Language) training pipeline. Since LightRFT's
pipeline is built around image/video processing, we repurpose the image
data slots to carry audio data through the pipeline.

Architecture:
    1. AudioPromptDataset: Returns (prompt_text, audio_data, reference, label)
       where audio_data flows through the 'images' slot.
    2. AudioMultimodalProcessor: Replaces the image processor to handle audio.
       - Calls processor(text=..., audios=...) instead of processor(text=..., images=...)
       - Stores input_features in the pixel_values slot
       - Stores feature_attention_mask in the image_grid_thw slot
    3. Monkey patches: Adapt normalize/count/build functions for audio data.

The Actor model is provided by lightrft.models.actor_al.ActorAL, which
natively handles Qwen2-Audio's audio_values parameter.

Key Mapping (image pipeline → audio pipeline):
    pixel_values       → input_features
    image_grid_thw     → feature_attention_mask
    raw_images (PIL)   → raw_audios (numpy_array, sr) tuples
    multi_modal_data["image"] → multi_modal_data["audio"]
"""

from __future__ import annotations

import inspect
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ============================================================================
# Audio Loading
# ============================================================================

def load_audio(audio_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio file with librosa; returns (waveform, sample_rate)."""
    return librosa.load(audio_path, sr=sr)


# ============================================================================
# Audio Prompt Dataset
# ============================================================================

class AudioPromptDataset(Dataset):
    """
    PyTorch Dataset for audio question-answering prompts.

    This dataset reads R1-AQA formatted data (with audio content type in
    the prompt) and returns (prompt_text, audio_data, reference, label).

    The audio data flows through LightRFT's 'images' slot. The prompt
    is rendered using the Qwen2-Audio processor's chat template.

    :param dataset: Underlying HuggingFace dataset.
    :param tokenizer: HuggingFace tokenizer.
    :param processor: Qwen2-Audio processor (for chat template + feature extraction).
    :param max_length: Maximum prompt length.
    :param strategy: LightRFT strategy object.
    :param input_template: Optional template for formatting input text.
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        processor,
        max_length: int,
        strategy,
        input_template: Optional[str] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.strategy = strategy
        self.input_template = input_template

        # Field keys from strategy args
        self.prompt_key = getattr(strategy.args, "input_key", "prompt")
        self.reference_key = getattr(strategy.args, "reference_key", "reference")
        self.label_key = getattr(strategy.args, "label_key", "label")
        self.audio_path_key = "audio_path"

        # Audio loading configuration
        self.target_sr = 16000
        if hasattr(processor, "feature_extractor") and processor.feature_extractor is not None:
            self.target_sr = getattr(
                processor.feature_extractor, "sampling_rate", 16000
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[str, Any, str, str]:
        """
        Return (prompt_text, audio_data, reference, label).

        - prompt_text: Rendered chat template string with audio tokens.
        - audio_data: Tuple (audio_array, sample_rate) for the inference engine.
        - reference: Ground truth answer string.
        - label: Reward recipe label (e.g., "avqa_rule").
        """
        data = self.dataset[idx]

        # ---- 1. Extract prompt (chat messages with audio content) ----
        prompt_messages = data.get(self.prompt_key, [])
        if isinstance(prompt_messages, str):
            # If stored as string, try to parse as JSON
            import json
            try:
                prompt_messages = json.loads(prompt_messages)
            except (json.JSONDecodeError, TypeError):
                prompt_messages = [{"role": "user", "content": prompt_messages}]

        # ---- 2. Render via processor's chat template ----
        try:
            prompt_text = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            # Fallback: extract text and format manually
            self.strategy.print(f"[WARNING] Chat template failed for idx {idx}: {e}")
            user_text = self._extract_text_from_messages(prompt_messages)
            prompt_text = user_text

        # ---- 3. Load audio ----
        audio_path = data.get(self.audio_path_key, "")
        audio_data = None
        if audio_path and os.path.exists(audio_path):
            try:
                audio_data = load_audio(audio_path, sr=self.target_sr)
            except Exception as e:
                self.strategy.print(f"[WARNING] Failed to load audio {audio_path}: {e}")
                audio_data = None

        # ---- 4. Reference and label (defaults if missing) ----
        reference = str(data.get(self.reference_key) or "")
        label = data.get(self.label_key) or "avqa_rule"
        return prompt_text, audio_data, reference, label

    def collate_fn(self, batch: List[Tuple]) -> Tuple[List, List, List, List]:
        """Collate a batch of (prompt, audio, reference, label) tuples."""
        prompts, audios, refs, labels = zip(*batch)
        return list(prompts), list(audios), list(refs), list(labels)

    @staticmethod
    def _extract_text_from_messages(messages) -> str:
        """Extract text content from chat messages."""
        texts = []
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    texts.append(content)
                elif isinstance(content, list):
                    for seg in content:
                        if isinstance(seg, dict) and seg.get("type") == "text":
                            texts.append(seg.get("text", ""))
        return " ".join(texts)


# ============================================================================
# Audio Multimodal Processor
# ============================================================================

class AudioMultimodalProcessor:
    """
    Multimodal data processor adapted for audio (Qwen2-Audio).

    Replaces LightRFT's ``MultimodalDataProcessor`` for audio models.
    Instead of processing images, it processes audio data through
    the Qwen2-Audio processor.

    Key differences from image processor:
    - Calls ``processor(text=..., audios=...)`` instead of ``processor(text=..., images=...)``
    - Returns ``input_features`` (stored as pixel_values in pipeline)
    - Returns ``feature_attention_mask`` (stored as image_grid_thw in pipeline)
    """

    def __init__(self, tokenizer, processor, prompt_max_len: int):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_max_len = prompt_max_len

    def process_multimodal_batch(
        self,
        all_prompts: List[str],
        all_images: List,  # Actually audio data: List[Optional[Tuple[np.ndarray, int]]]
        all_references: Optional[List[str]],
        images_num: List[int],
        n_samples_per_prompt: int,
        all_videos: Optional[List] = None,
        videos_num: Optional[List[int]] = None,
    ) -> EasyDict:
        """
        Process a batch of audio-multimodal data.

        :param all_prompts: List of prompt strings (with audio tokens).
        :param all_images: List of audio data tuples (repurposed images slot).
        :param all_references: Reference answers.
        :param images_num: Audio count per sample (always [1, 1, ...]).
        :param n_samples_per_prompt: Number of GRPO samples per prompt.
        :param all_videos: Unused (None for audio).
        :param videos_num: Unused (None for audio).
        :return: EasyDict with processed data matching LightRFT's expected format.
        """
        N = n_samples_per_prompt
        L = len(all_prompts)

        if all_images is None:
            all_images = [None] * L

        # ===== Stage 1: Separation (text-only vs audio) =====
        all_prompts_text, all_prompts_audio = [], []
        all_audios_valid = []
        text_idx = []

        for idx, (prompt, audio) in enumerate(zip(all_prompts, all_images)):
            if audio is None:
                all_prompts_text.append(prompt)
                text_idx.append(idx)
            else:
                all_prompts_audio.append(prompt)
                all_audios_valid.append(audio)

        # ===== Stage 2: Expansion for n_samples_per_prompt =====
        all_prompts_text = sum([[p] * N for p in all_prompts_text], [])
        all_prompts_audio = sum([[p] * N for p in all_prompts_audio], [])
        all_audios_valid = [a for a in all_audios_valid for _ in range(N)]
        all_images_num = sum([[num] * N for num in images_num], []) if images_num else [0] * (L * N)
        all_videos_num = [0] * (L * N)

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

        # ===== Stage 3-B: Audio processing =====
        all_prompt_token_ids_audio = []
        all_input_features = None
        all_feature_attention_mask = None

        if len(all_prompts_audio) > 0:
            # Extract audio arrays for the processor
            flat_audios = []
            for audio_tuple in all_audios_valid:
                if isinstance(audio_tuple, tuple) and len(audio_tuple) == 2:
                    flat_audios.append(audio_tuple[0])  # numpy array
                elif isinstance(audio_tuple, np.ndarray):
                    flat_audios.append(audio_tuple)
                else:
                    # Fallback: create silence
                    flat_audios.append(np.zeros(16000, dtype=np.float32))

            # Process through Qwen2-Audio processor
            # --- Sanitize <|AUDIO|> token count ---
            # Each audio prompt must contain exactly one <|AUDIO|> token.
            # Mismatches can occur when the chat template or data serialization
            # inserts duplicate tokens (e.g. template + processor both add one).
            audio_token_str = getattr(self.processor, "audio_token", "<|AUDIO|>")
            audio_bos_str = getattr(self.processor, "audio_bos_token", "<|audio_bos|>")
            audio_eos_str = getattr(self.processor, "audio_eos_token", "<|audio_eos|>")
            sanitized_prompts = []
            n_fixed = 0
            for prompt_str in all_prompts_audio:
                count = prompt_str.count(audio_token_str)
                if count == 1:
                    sanitized_prompts.append(prompt_str)
                elif count == 0:
                    # No audio token found – insert one at the beginning of the
                    # user turn (after system preamble) so the processor can
                    # expand it later.
                    insert_marker = f"{audio_bos_str}{audio_token_str}{audio_eos_str}"
                    # Try to place it after <|im_start|>user\n
                    user_tag = "<|im_start|>user\n"
                    idx_user = prompt_str.find(user_tag)
                    if idx_user >= 0:
                        ins = idx_user + len(user_tag)
                        prompt_str = prompt_str[:ins] + insert_marker + "\n" + prompt_str[ins:]
                    else:
                        prompt_str = insert_marker + "\n" + prompt_str
                    sanitized_prompts.append(prompt_str)
                    n_fixed += 1
                else:
                    # More than one <|AUDIO|> token – keep only the first one.
                    # Remove the full <|audio_bos|><|AUDIO|><|audio_eos|> group
                    # for duplicates, or bare <|AUDIO|> tokens.
                    full_group = f"{audio_bos_str}{audio_token_str}{audio_eos_str}"
                    if full_group in prompt_str:
                        # Keep first occurrence of the full group, remove the rest
                        first_end = prompt_str.find(full_group) + len(full_group)
                        before = prompt_str[:first_end]
                        after = prompt_str[first_end:]
                        after = after.replace(full_group, "")
                        # Also remove any bare <|AUDIO|> that remain
                        after = after.replace(audio_token_str, "")
                        prompt_str = before + after
                    else:
                        # No full group – just keep first bare <|AUDIO|>
                        first_end = prompt_str.find(audio_token_str) + len(audio_token_str)
                        before = prompt_str[:first_end]
                        after = prompt_str[first_end:]
                        after = after.replace(audio_token_str, "")
                        prompt_str = before + after
                    sanitized_prompts.append(prompt_str)
                    n_fixed += 1

            if n_fixed > 0:
                print(
                    f"[AudioPipeline] Fixed <|AUDIO|> token count in {n_fixed}/{len(all_prompts_audio)} prompts"
                )
            all_prompts_audio = sanitized_prompts

            # Determine the correct kwarg name for the processor (older
            # transformers versions use "audios", newer use "audio").
            proc_sig = inspect.signature(self.processor.__call__)
            audio_kwarg = "audio" if "audio" in proc_sig.parameters else "audios"

            processor_kwargs = {
                "text": all_prompts_audio,
                audio_kwarg: flat_audios,
                "add_special_tokens": False,
                "max_length": self.prompt_max_len,
                "truncation": True,
                "padding": True,
                "return_tensors": "pt",
            }

            print(f"[AudioPipeline] Processor type: {type(self.processor).__name__}")
            print(f"[AudioPipeline] Processing {len(flat_audios)} audio samples")
            total_audio_tokens = sum(p.count(audio_token_str) for p in all_prompts_audio)
            print(f"[AudioPipeline] Total <|AUDIO|> tokens in text: {total_audio_tokens}, audios: {len(flat_audios)}")

            inputs_audio = self.processor(**processor_kwargs)
            print(f"[AudioPipeline] Processor output keys: {list(inputs_audio.keys())}")

            all_prompt_token_ids_audio = inputs_audio["input_ids"].tolist()
            all_input_features = inputs_audio.get("input_features", None)
            all_feature_attention_mask = inputs_audio.get(
                "feature_attention_mask", None
            )

            # ------------------------------------------------------------------
            # Qwen2Audio's Whisper encoder requires mel features of exactly 3000
            # frames.  When ``padding=True`` is forwarded to the feature
            # extractor it pads to the batch-max instead of 3000.  Fix here.
            # ------------------------------------------------------------------
            if all_input_features is not None:
                EXPECTED_MEL_LEN = 3000
                actual_len = all_input_features.shape[-1]
                if actual_len < EXPECTED_MEL_LEN:
                    pad_len = EXPECTED_MEL_LEN - actual_len
                    all_input_features = torch.nn.functional.pad(
                        all_input_features, (0, pad_len), value=0.0,
                    )
                    if all_feature_attention_mask is not None:
                        all_feature_attention_mask = torch.nn.functional.pad(
                            all_feature_attention_mask, (0, pad_len), value=0,
                        )
                    print(
                        f"[AudioPipeline] Padded input_features from "
                        f"{actual_len} → {EXPECTED_MEL_LEN} frames"
                    )

            if all_input_features is None:
                raise RuntimeError(
                    f"Processor {type(self.processor).__name__} returned no "
                    f"'input_features'. Available keys: {list(inputs_audio.keys())}. "
                    "Ensure the processor is Qwen2AudioProcessor (not a generic text processor)."
                )

        # ===== Stage 4: Merge back in original order =====
        total_samples = L * N
        all_prompts_out = [None] * total_samples
        all_images_out = [None] * total_samples  # Raw audio data for engine
        all_prompt_token_ids_out = [None] * total_samples

        # 4-A: Fill text-only slots
        text_ptr = 0
        for orig_idx in text_idx:
            for n in range(N):
                gid = orig_idx * N + n
                all_prompts_out[gid] = all_prompts_text[text_ptr]
                all_prompt_token_ids_out[gid] = all_prompt_token_ids_text[text_ptr]
                text_ptr += 1

        # 4-B: Fill audio slots
        audio_ptr = 0
        for orig_idx in range(L):
            if orig_idx in text_idx:
                continue
            for n in range(N):
                gid = orig_idx * N + n
                all_prompts_out[gid] = all_prompts_audio[audio_ptr]
                all_images_out[gid] = all_audios_valid[audio_ptr]  # Raw audio for engine
                all_prompt_token_ids_out[gid] = all_prompt_token_ids_audio[audio_ptr]
                audio_ptr += 1

        # Expand references
        if all_references is not None:
            all_references = sum([[ref] * N for ref in all_references], [])

        # Build grid_thw entries for audio.
        # Each audio sample needs a (1, 1, 1) grid entry so that the VL pipeline's
        # per-sample slicing of pixel_values (input_features) works correctly:
        #   num_patch = 1*1*1 = 1 → slices exactly 1 row from input_features per audio.
        total_audio_count = sum(all_images_num)
        if total_audio_count > 0:
            all_images_grid_thw = torch.ones((total_audio_count, 3), dtype=torch.long)
        else:
            all_images_grid_thw = torch.empty((0, 3), dtype=torch.long)
        all_videos_grid_thw = torch.empty((0, 3), dtype=torch.long)

        # Store input_features as pixel_values for pipeline compatibility
        return EasyDict(
            all_prompt_token_ids=all_prompt_token_ids_out,
            all_prompts=all_prompts_out,
            all_images=all_images_out,  # Raw audio data for engine
            all_videos=[None] * total_samples,
            all_images_num=all_images_num,
            all_videos_num=all_videos_num,
            all_images_pixel_values=all_input_features,  # input_features stored here
            all_videos_pixel_values=None,
            all_images_grid_thw=all_images_grid_thw,
            all_videos_grid_thw=all_videos_grid_thw,
            all_references=all_references,
            # Audio-specific: store feature mask separately
            _audio_feature_attention_mask=all_feature_attention_mask,
        )


# ============================================================================
# Audio-aware image utilities (monkey-patch replacements)
# ============================================================================

def normalize_audios(raw_items: List) -> List:
    """
    Replacement for ``normalize_images`` that handles audio tuples.

    Audio data is already in the correct format (numpy array, sr) so
    we just pass it through unchanged.
    """
    return raw_items


def get_audios_num(all_items: Optional[List]) -> Optional[List[int]]:
    """
    Replacement for ``get_images_num`` that handles audio tuples.

    Returns 1 for each non-None audio item, 0 for None.
    """
    if all_items is None:
        return None
    counts = []
    for item in all_items:
        if item is None:
            counts.append(0)
        else:
            counts.append(1)
    return counts


# ============================================================================
# Strategy patching for audio
# ============================================================================

def build_audio_multimodal_inputs(
    strategy,
    all_prompts: List[str],
    all_images: List,  # Actually raw audio data
    images_num: Optional[List[int]],
    all_videos: Optional[List] = None,
    videos_num: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Replacement for ``strategy._build_multimodal_inputs`` that maps
    audio data to ``multi_modal_data["audio"]`` instead of ``["image"]``.

    :param all_prompts: List of prompt strings.
    :param all_images: List of raw audio data (tuples or numpy arrays).
    :param images_num: Audio count per sample.
    :return: List of dicts with 'prompt' and optional 'multi_modal_data'.
    """
    inputs = []
    audio_start_idx = 0

    if images_num is None:
        images_num = [0] * len(all_prompts)

    for i, prompt in enumerate(all_prompts):
        audio_num = images_num[i] if i < len(images_num) else 0

        audio_list = []
        if audio_num > 0 and all_images is not None:
            for j in range(audio_num):
                idx = audio_start_idx + j
                if idx < len(all_images) and all_images[idx] is not None:
                    audio_item = all_images[idx]
                    if isinstance(audio_item, tuple):
                        audio_list.append(audio_item)  # (array, sr)
                    elif isinstance(audio_item, np.ndarray):
                        audio_list.append((audio_item, 16000))
                    # else skip

        multi_modal_data = {}
        if audio_list:
            multi_modal_data["audio"] = audio_list

        if not multi_modal_data:
            # Remove audio placeholder tokens if no audio data
            prompt = re.sub(
                r'<\|audio_bos\|>.*?<\|audio_eos\|>', '', prompt
            )
            inputs.append({"prompt": prompt})
        else:
            inputs.append({
                "prompt": prompt,
                "multi_modal_data": multi_modal_data,
            })

        audio_start_idx += audio_num

    return inputs


def patch_strategy_for_audio(strategy):
    """
    Monkey-patch the strategy object for audio multimodal support.

    Replaces ``_build_multimodal_inputs`` with audio-aware version.
    """
    original_build = strategy._build_multimodal_inputs

    def patched_build(all_prompts, all_images=None, images_num=None,
                      all_videos=None, videos_num=None):
        return build_audio_multimodal_inputs(
            strategy, all_prompts, all_images, images_num, all_videos, videos_num
        )

    strategy._build_multimodal_inputs = patched_build
    strategy.print("[PATCH] Strategy._build_multimodal_inputs patched for audio")
    return strategy


# ============================================================================
# Experience Maker patching for audio
# ============================================================================

def patch_experience_maker_for_audio(exp_maker, processor, tokenizer, prompt_max_len):
    """
    Monkey-patch the FastExperienceMaker for audio support.

    Replaces the multimodal_processor with AudioMultimodalProcessor
    and patches normalize/count functions.
    """
    # Replace multimodal processor
    exp_maker.multimodal_processor = AudioMultimodalProcessor(
        tokenizer=tokenizer,
        processor=processor,
        prompt_max_len=prompt_max_len,
    )

    # Store original functions for potential restoration
    from lightrft.trainer.image_utils import normalize_images as _orig_normalize
    from lightrft.trainer.image_utils import get_images_num as _orig_get_num
    import lightrft.trainer.fast_exp_maker as fem_module

    # Patch module-level functions used by make_experience_list
    fem_module.normalize_images = normalize_audios
    fem_module.get_images_num = get_audios_num

    print("[PATCH] FastExperienceMaker patched for audio")
    return exp_maker



