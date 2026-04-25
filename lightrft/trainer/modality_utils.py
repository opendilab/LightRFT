"""
Shared helpers for modality-specific trainer/model wiring.
"""

from typing import Any, Dict, Set


def build_supported_model_kwargs(source: Any, supported_params: Set[str]) -> Dict[str, Any]:
    """
    Extract only the multimodal kwargs that the current actor supports.

    This helper is the bridge between the trainer's generic replay objects and the
    modality-specific model signatures.

    Example::

        # Vision-language model
        supported_params = {"pixel_values", "image_grid_thw"}
        kwargs = build_supported_model_kwargs(experience, supported_params)
        # -> {"pixel_values": experience.pixel_values, "image_grid_thw": experience.image_grid_thws}

        # Audio-language model
        supported_params = {"audio_values", "feature_attention_mask"}
        kwargs = build_supported_model_kwargs(experience, supported_params)
        # -> {"audio_values": experience.audio_values, "feature_attention_mask": experience.feature_attention_mask}

    Keeping this mapping in one place avoids trainer call sites accidentally mixing
    image fields and audio fields during future refactors.
    """
    candidate_params = {
        "pixel_values": getattr(source, "pixel_values", None),
        "image_grid_thw": getattr(source, "image_grid_thw", getattr(source, "image_grid_thws", None)),
        "pixel_values_videos": getattr(source, "pixel_values_videos", None),
        "video_grid_thw": getattr(source, "video_grid_thw", getattr(source, "video_grid_thws", None)),
        "audio_values": getattr(source, "audio_values", None),
        "feature_attention_mask": getattr(source, "feature_attention_mask", None),
    }
    return {key: value for key, value in candidate_params.items() if key in supported_params}
