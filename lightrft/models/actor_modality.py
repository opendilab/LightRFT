"""
Actor Model Modality Definitions.

This module defines the modality types for actor models and provides
a centralized mapping of what parameters each modality supports.
"""

from enum import Enum
from typing import Set


class ActorModality(Enum):
    """
    Enumeration of actor model modality types.

    This enum defines the different types of modalities that actor models
    can support, making it explicit what kind of inputs each model accepts.
    """
    LANGUAGE_ONLY = "text"  # Pure text model (e.g., ActorLanguage)
    VISION_LANGUAGE = "vision"  # Vision-language model supporting images and videos (e.g., ActorVL)
    AUDIO_LANGUAGE = "audio"  # Audio-language model (future extension)
    OMNI = "multimodal"  # Full multimodal model (future extension)


# Mapping from modality to supported parameter names
MODALITY_PARAMETERS = {
    ActorModality.LANGUAGE_ONLY: set(),  # No multimodal parameters
    ActorModality.VISION_LANGUAGE: {
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
    },
    ActorModality.AUDIO_LANGUAGE: {
        "audio_values",
    },
    ActorModality.OMNI: {
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
        "audio_values",
    },
}


def get_supported_parameters(modality: ActorModality) -> Set[str]:
    """
    Get the set of multimodal parameters supported by a given modality.

    :param modality: The actor modality type
    :type modality: ActorModality
    :return: Set of parameter names that this modality supports
    :rtype: Set[str]

    Example::

        params = get_supported_parameters(ActorModality.VISION_LANGUAGE)
        # Returns: {"pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"}
    """
    return MODALITY_PARAMETERS.get(modality, set())


def supports_parameter(modality: ActorModality, param_name: str) -> bool:
    """
    Check if a modality supports a specific parameter.

    :param modality: The actor modality type
    :type modality: ActorModality
    :param param_name: The parameter name to check
    :type param_name: str
    :return: True if the parameter is supported, False otherwise
    :rtype: bool

    Example::

        if supports_parameter(ActorModality.LANGUAGE_ONLY, "pixel_values"):
            # This will be False
            pass
    """
    return param_name in get_supported_parameters(modality)
