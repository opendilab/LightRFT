"""
This module provides functionality for modifying the generation behavior of Hugging Face Transformers models.

It contains utilities to modify the stopping criteria during text generation, specifically by removing
the EOS token criteria to allow models to generate up to the maximum number of tokens specified.

This module is useful when you want models to generate the full number of tokens requested rather
than stopping early at EOS tokens.
"""

from transformers.generation.utils import *


# This function if modified from Transformers(https://github.com/huggingface/transformers) 4.48.3
def _get_stopping_criteria_hacked(
    self,
    generation_config: GenerationConfig,
    stopping_criteria: Optional[StoppingCriteriaList],
    tokenizer: Optional["PreTrainedTokenizerBase"] = None,
    **kwargs,
) -> StoppingCriteriaList:
    """
    Get the stopping criteria for text generation with EOS token criteria removed.

    This is a modified version of the Transformers _get_stopping_criteria method that excludes
    the EOS token stopping criterion, allowing generation to continue to max_new_tokens.

    :param generation_config: The configuration for generation containing parameters like
                            max_length, max_time etc.
    :type generation_config: GenerationConfig
    :param stopping_criteria: Optional additional stopping criteria to be merged
    :type stopping_criteria: Optional[StoppingCriteriaList]
    :param tokenizer: The tokenizer used for handling stop strings
    :type tokenizer: Optional["PreTrainedTokenizerBase"]
    :param kwargs: Additional keyword arguments
    :type kwargs: dict

    :return: A list of stopping criteria to be applied during generation
    :rtype: StoppingCriteriaList
    :raises ValueError: If stop strings are specified but no tokenizer is provided
    """
    criteria = StoppingCriteriaList()
    if generation_config.max_length is not None:
        max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
        criteria.append(
            MaxLengthCriteria(
                max_length=generation_config.max_length,
                max_position_embeddings=max_position_embeddings,
            )
        )
    if generation_config.max_time is not None:
        criteria.append(MaxTimeCriteria(max_time=generation_config.max_time))
    if generation_config.stop_strings is not None:
        if tokenizer is None:
            raise ValueError(
                "There are one or more stop strings, either in the arguments to `generate` or in the "
                "model's generation config, but we could not locate a tokenizer. When generating with "
                "stop strings, you must pass the model's tokenizer to the `tokenizer` argument of `generate`."
            )
        criteria.append(StopStringCriteria(stop_strings=generation_config.stop_strings, tokenizer=tokenizer))

    # EosTokenCriteria is removed to allow model generate to max_new_tokens
    # if generation_config._eos_token_tensor is not None:
    #     criteria.append(EosTokenCriteria(eos_token_id=generation_config._eos_token_tensor))
    if (
        generation_config.is_assistant and generation_config.assistant_confidence_threshold is not None
        and generation_config.assistant_confidence_threshold > 0
    ):
        criteria.append(
            ConfidenceCriteria(assistant_confidence_threshold=generation_config.assistant_confidence_threshold)
        )
    criteria = self._merge_criteria_processor_list(criteria, stopping_criteria)
    return criteria


def apply_monkey_patch_to_generation_mixin():
    """
    Apply the monkey patch to replace the original _get_stopping_criteria method.

    This function replaces the default _get_stopping_criteria implementation in GenerationMixin
    with our modified version that excludes EOS token stopping.

    :return: None
    """
    GenerationMixin._get_stopping_criteria = _get_stopping_criteria_hacked
