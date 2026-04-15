"""
Utilities for audio-language rollout processing.

This module keeps audio-specific preprocessing out of the vision-language path.
"""

from __future__ import annotations

import inspect
import io
import numbers
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from easydict import EasyDict


def normalize_audios(raw_audios: List[Any]) -> List[Any]:
    """Audio payloads are already normalized by the dataset layer."""
    return raw_audios


def is_single_audio_payload(audio_item: Any) -> bool:
    """
    Return True when ``audio_item`` represents exactly one audio payload.

    Accepted forms include:
    - ``(waveform, sampling_rate)``
    - ``[waveform, sampling_rate]``
    - raw waveform arrays
    - engine-ready ``str`` / ``bytes`` / ``dict`` payloads
    - one-element wrappers such as ``[(waveform, sr)]``
    """
    if audio_item is None:
        return False
    if isinstance(audio_item, (str, bytes, dict, np.ndarray)):
        return True
    if isinstance(audio_item, tuple) and len(audio_item) == 2:
        return isinstance(audio_item[1], numbers.Number)
    if isinstance(audio_item, list):
        if len(audio_item) == 1:
            return is_single_audio_payload(audio_item[0])
        if len(audio_item) == 2 and isinstance(audio_item[1], numbers.Number):
            return True
    return False


def canonicalize_audio_payload(audio_item: Any) -> Any:
    """
    Normalize single-audio wrappers to a stable payload shape.

    This keeps the audio rollout path permissive about harmless container differences
    while still rejecting true multi-audio inputs at a higher level.
    """
    if isinstance(audio_item, list) and len(audio_item) == 1 and is_single_audio_payload(audio_item[0]):
        return canonicalize_audio_payload(audio_item[0])
    return audio_item


def get_audios_num(all_audios: Optional[List[Any]]) -> Optional[List[int]]:
    """
    Count audio items per sample.

    Audio RL currently supports zero or one audio payload per prompt in rollout.
    """
    if all_audios is None:
        return None
    counts = []
    for audio in all_audios:
        if audio is None:
            counts.append(0)
        elif is_single_audio_payload(audio):
            counts.append(1)
        elif isinstance(audio, list):
            counts.append(len(audio))
        else:
            counts.append(1)
    return counts


def extract_audio_array(audio_item: Any, default_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Normalize supported audio payloads to ``(waveform, sampling_rate)``."""
    audio_item = canonicalize_audio_payload(audio_item)
    if isinstance(audio_item, tuple) and len(audio_item) == 2:
        audio_array, sr = audio_item
        return np.asarray(audio_array, dtype=np.float32), int(sr)
    if isinstance(audio_item, list) and len(audio_item) == 2 and isinstance(audio_item[1], numbers.Number):
        audio_array, sr = audio_item
        return np.asarray(audio_array, dtype=np.float32), int(sr)
    if isinstance(audio_item, np.ndarray):
        return np.asarray(audio_item, dtype=np.float32), default_sr
    raise TypeError(f"Unsupported audio payload type: {type(audio_item).__name__}")


def serialize_audio_for_sglang(audio_item: Any, default_sr: int = 16000) -> Union[str, bytes, dict, None]:
    """
    Convert a local audio payload into a SGLang-compatible object.

    SGLang accepts file paths / URLs / bytes, but not ``(waveform, sr)`` tuples directly.
    """
    if audio_item is None:
        return None
    if isinstance(audio_item, (str, bytes, dict)):
        return audio_item

    audio_array, sr = extract_audio_array(audio_item, default_sr=default_sr)
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sr, format="WAV")
    return buffer.getvalue()


def normalize_audio_features(
    input_features: torch.Tensor,
    feature_attention_mask: Optional[torch.Tensor],
    expected_mel_len: int = 3000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize Qwen2-Audio features to ``(B, mel_bins, expected_mel_len)``.
    """
    if not isinstance(input_features, torch.Tensor):
        input_features = torch.as_tensor(input_features)
    if feature_attention_mask is not None and not isinstance(feature_attention_mask, torch.Tensor):
        feature_attention_mask = torch.as_tensor(feature_attention_mask)

    if input_features.dim() != 3:
        raise RuntimeError(
            f"Expected 3D audio features, but got shape {tuple(input_features.shape)}. "
            "Qwen2-Audio should return (batch, mel_bins, time)."
        )

    if input_features.shape[1] in (80, 128):
        normalized = input_features
    elif input_features.shape[-1] in (80, 128):
        normalized = input_features.transpose(1, 2).contiguous()
    else:
        raise RuntimeError(
            f"Unexpected Qwen2-Audio feature shape {tuple(input_features.shape)}: "
            "unable to identify mel-bin dimension."
        )

    current_len = normalized.shape[-1]
    target_len = min(current_len, expected_mel_len)
    if current_len < expected_mel_len:
        normalized = torch.nn.functional.pad(normalized, (0, expected_mel_len - current_len), value=0.0)
    elif current_len > expected_mel_len:
        normalized = normalized[..., :expected_mel_len]

    if feature_attention_mask is None:
        feature_attention_mask = torch.zeros(
            normalized.shape[0], expected_mel_len, dtype=torch.long, device=normalized.device
        )
        feature_attention_mask[:, :target_len] = 1
    else:
        feature_attention_mask = feature_attention_mask.to(device=normalized.device, dtype=torch.long)
        if feature_attention_mask.shape[-1] < expected_mel_len:
            feature_attention_mask = torch.nn.functional.pad(
                feature_attention_mask, (0, expected_mel_len - feature_attention_mask.shape[-1]), value=0
            )
        elif feature_attention_mask.shape[-1] > expected_mel_len:
            feature_attention_mask = feature_attention_mask[..., :expected_mel_len]

    return normalized, feature_attention_mask


class AudioDataProcessor:
    """
    Audio-language rollout preprocessor.

    Unlike the VL processor, audio inputs stay on an explicit audio path:
    raw audio payloads are kept for the inference engine, while ``audio_values``
    and ``feature_attention_mask`` are prepared for actor/reference forward.
    """
    def __init__(self, tokenizer, processor, prompt_max_len: int):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_max_len = prompt_max_len

    def process_audio_batch(
        self,
        all_prompts: List[str],
        all_audios: List[Any],
        all_references: Optional[List[str]],
        n_samples_per_prompt: int,
    ) -> EasyDict:
        N = n_samples_per_prompt
        L = len(all_prompts)
        if all_audios is None:
            all_audios = [None] * L

        all_prompts_text, all_prompts_audio = [], []
        all_audios_valid = []
        text_idx = []

        for idx, (prompt, audio) in enumerate(zip(all_prompts, all_audios)):
            if audio is None:
                all_prompts_text.append(prompt)
                text_idx.append(idx)
            else:
                audio = canonicalize_audio_payload(audio)
                if isinstance(audio, list) and not is_single_audio_payload(audio):
                    raise RuntimeError(
                        "Audio RL rollout currently expects at most one audio payload per prompt. "
                        f"Received list input for sample {idx}."
                    )
                all_prompts_audio.append(prompt)
                all_audios_valid.append(audio)

        all_prompts_text = sum([[prompt] * N for prompt in all_prompts_text], [])
        all_prompts_audio = sum([[prompt] * N for prompt in all_prompts_audio], [])
        all_audios_valid = [audio for audio in all_audios_valid for _ in range(N)]

        if all_prompts_text:
            inputs_text = self.tokenizer(
                all_prompts_text,
                max_length=self.prompt_max_len,
                truncation=True,
                add_special_tokens=False,
            )
            all_prompt_token_ids_text = inputs_text["input_ids"]
        else:
            all_prompt_token_ids_text = []

        all_prompt_token_ids_audio = []
        all_audio_values = None
        all_feature_attention_mask = None
        if all_prompts_audio:
            proc_sig = inspect.signature(self.processor.__call__)
            audio_kwarg = "audio" if "audio" in proc_sig.parameters else "audios"
            flat_audios = [extract_audio_array(audio, default_sr=16000)[0] for audio in all_audios_valid]
            inputs_audio = self.processor(
                text=all_prompts_audio,
                **{audio_kwarg: flat_audios},
                add_special_tokens=False,
                max_length=self.prompt_max_len,
                truncation=True,
                padding=True,
                return_tensors="pt",
                sampling_rate=getattr(self.processor.feature_extractor, "sampling_rate", 16000),
            )
            all_prompt_token_ids_audio = inputs_audio["input_ids"].tolist()
            all_audio_values = inputs_audio.get("input_features", None)
            if all_audio_values is None:
                raise RuntimeError(
                    f"Processor {type(self.processor).__name__} returned no 'input_features'. "
                    f"Available keys: {list(inputs_audio.keys())}"
                )
            all_feature_attention_mask = inputs_audio.get("feature_attention_mask", None)
            all_audio_values, all_feature_attention_mask = normalize_audio_features(
                all_audio_values, all_feature_attention_mask, expected_mel_len=3000
            )

        total_samples = L * N
        all_prompts_out = [None] * total_samples
        all_audios_out = [None] * total_samples
        all_prompt_token_ids_out = [None] * total_samples
        all_audio_values_out = None
        all_feature_attention_mask_out = None
        if all_audio_values is not None:
            all_audio_values_out = all_audio_values.new_zeros((total_samples, ) + tuple(all_audio_values.shape[1:]))
            all_feature_attention_mask_out = all_feature_attention_mask.new_zeros(
                (total_samples, ) + tuple(all_feature_attention_mask.shape[1:])
            )

        text_ptr = 0
        for orig_idx in text_idx:
            for n in range(N):
                gid = orig_idx * N + n
                all_prompts_out[gid] = all_prompts_text[text_ptr]
                all_prompt_token_ids_out[gid] = all_prompt_token_ids_text[text_ptr]
                text_ptr += 1

        audio_ptr = 0
        for orig_idx in range(L):
            if orig_idx in text_idx:
                continue
            for n in range(N):
                gid = orig_idx * N + n
                all_prompts_out[gid] = all_prompts_audio[audio_ptr]
                all_audios_out[gid] = all_audios_valid[audio_ptr]
                all_prompt_token_ids_out[gid] = all_prompt_token_ids_audio[audio_ptr]
                if all_audio_values_out is not None:
                    all_audio_values_out[gid] = all_audio_values[audio_ptr]
                    all_feature_attention_mask_out[gid] = all_feature_attention_mask[audio_ptr]
                audio_ptr += 1

        if all_references is not None:
            all_references = sum([[ref] * N for ref in all_references], [])

        return EasyDict(
            all_prompt_token_ids=all_prompt_token_ids_out,
            all_prompts=all_prompts_out,
            all_audios=all_audios_out,
            all_audio_num=get_audios_num(all_audios_out),
            all_audio_values=all_audio_values_out,
            all_feature_attention_mask=all_feature_attention_mask_out,
            all_references=all_references,
        )
