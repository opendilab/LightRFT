"""
Audio dataset helpers for the R1-AQA LightRFT example.

Historically this example carried audio through the old VL pipeline with example-side
patches. The rollout path is now native in core LightRFT, so this module is reduced
to the example-specific data layer:

Architecture:
    1. AudioPromptDataset returns ``(prompt_text, audio_data, reference, label)``.
    2. ``prompt_text`` is rendered with the Qwen2-Audio chat template.
    3. ``audio_data`` stays as raw waveform + sampling rate for core rollout code.

The actor is still ``lightrft.models.actor_al.ActorAL``, which expects audio through
the explicit audio-language interface in the trainer/model stack.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import librosa
from torch.utils.data import Dataset

# ============================================================================
# Audio Loading
# ============================================================================


def load_audio(audio_path: str, sr: int = 16000) -> Tuple[Any, int]:
    """Load an audio file as ``(waveform, sampling_rate)``."""
    return librosa.load(audio_path, sr=sr)


class AudioPromptDataset(Dataset):
    """
    PyTorch dataset for the R1-AQA audio prompt format.

    Each item returns ``(prompt_text, audio_payload, reference, label)`` where:
    - ``prompt_text`` is rendered through the Qwen2-Audio chat template
    - ``audio_payload`` is kept as raw waveform + sampling rate for rollout-side processing
    - ``reference`` and ``label`` are passed through to reward computation
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

        # Field keys from strategy args.
        self.prompt_key = getattr(strategy.args, "input_key", "prompt")
        self.reference_key = getattr(strategy.args, "reference_key", "reference")
        self.label_key = getattr(strategy.args, "label_key", "label")
        self.audio_path_key = "audio_path"

        # Audio loading configuration.
        self.target_sr = 16000
        if hasattr(processor, "feature_extractor") and processor.feature_extractor is not None:
            self.target_sr = getattr(processor.feature_extractor, "sampling_rate", 16000)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[str, Any, str, str]:
        data = self.dataset[idx]

        # ---- 1. Extract prompt (chat messages with audio content) ----
        prompt_messages = data.get(self.prompt_key, [])
        if isinstance(prompt_messages, str):
            try:
                prompt_messages = json.loads(prompt_messages)
            except (json.JSONDecodeError, TypeError):
                prompt_messages = [{"role": "user", "content": prompt_messages}]
        prompt_messages = self._drop_none_fields(prompt_messages)

        # ---- 2. Render via processor's chat template ----
        try:
            prompt_text = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            self.strategy.print(f"[WARNING] Chat template failed for idx {idx}: {exc}")
            prompt_text = self._extract_text_from_messages(prompt_messages)

        # ---- 3. Load audio ----
        audio_path = data.get(self.audio_path_key, "")
        audio_data = None
        if audio_path and os.path.exists(audio_path):
            try:
                audio_data = load_audio(audio_path, sr=self.target_sr)
            except Exception as exc:
                self.strategy.print(f"[WARNING] Failed to load audio {audio_path}: {exc}")
                audio_data = None

        # ---- 4. Reference and label (defaults if missing) ----
        reference = str(data.get(self.reference_key) or "")
        label = data.get(self.label_key) or "avqa_rule"
        return prompt_text, audio_data, reference, label

    def collate_fn(self, batch: List[Tuple]) -> Tuple[List, List, List, List]:
        """Keep prompts/audios/references/labels as plain Python lists for the rollout stack."""
        prompts, audios, refs, labels = zip(*batch)
        return list(prompts), list(audios), list(refs), list(labels)

    @staticmethod
    def _extract_text_from_messages(messages) -> str:
        """Fallback text extraction used when the upstream chat template fails."""
        texts = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                texts.append(content)
                continue
            if not isinstance(content, list):
                continue
            for segment in content:
                if isinstance(segment, dict) and segment.get("type") == "text":
                    texts.append(segment.get("text", ""))
        return " ".join(texts)

    @staticmethod
    def _drop_none_fields(obj):
        """
        Remove ``None`` values from nested prompt content before chat templating.

        The parquet loader materializes a union of nested content keys, so a text
        block may arrive as ``{"type": "text", "text": "...", "audio_url": None}``.
        Qwen2-Audio's default chat template checks key existence instead of value,
        which would misclassify that text block as a second audio placeholder.
        """
        if isinstance(obj, list):
            return [AudioPromptDataset._drop_none_fields(item) for item in obj]
        if isinstance(obj, dict):
            return {key: AudioPromptDataset._drop_none_fields(value) for key, value in obj.items() if value is not None}
        return obj
