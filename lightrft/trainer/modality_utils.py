"""
Shared helpers for modality-specific trainer/model wiring.
"""

from typing import Any, Dict, Set


def build_supported_model_kwargs(source: Any, supported_params: Set[str]) -> Dict[str, Any]:
    """
    Extract only the multimodal kwargs that the current actor supports.
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
