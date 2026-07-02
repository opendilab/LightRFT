"""URSA-MATH Stage 3 reward model helpers."""

from __future__ import annotations

import re
from itertools import zip_longest
from typing import Any, Dict

import torch
import torch.nn as nn

try:
    from math_prm_output import MATH_PRM_ANSWER_MARKER, find_math_prm_tail_cutoff
except ImportError:
    from .math_prm_output import MATH_PRM_ANSWER_MARKER, find_math_prm_tail_cutoff

from lightrft.evaluation.math_eval_utils import (
    compare_answers,
    extract_answer,
    extract_answer_from_tags,
    extract_boxed_answer,
    extract_multiple_choice_answer,
    extract_numeric_answer,
    normalize_answer,
)

try:
    from mathruler.grader import grade_answer as mathruler_grade_answer
except ImportError:
    mathruler_grade_answer = None


_VISION_PATTERNS = [
    r"<\|vision_start\|>(<\|image_pad\|>)+<\|vision_end\|>",
    r"<img>(<IMG_CONTEXT>)+</img>",
    r"<image>",
]


def _clean_vision_token(text: str) -> str:
    """Remove vision placeholders from a user question before PRM scoring."""
    for pattern in _VISION_PATTERNS:
        text = re.sub(pattern, "", text)
    return text


# Pattern matching "Step N:" / "†Answer:" markers — kept only for diagnostic
# `matched_patterns` output during alignment. The actual alignment uses the
# URSA-native path (PRM's own ``replace_specific_plus_minus_with_ki`` + PRM's
# tokenizer offset_mapping) instead of re-implementing char-level simulation.
_STEP_OR_ANSWER_PATTERN = re.compile(r"(Step \d+\s*:|†Answer\s*:)")


def find_step_boundaries_in_response_tokens(prm_module, response_text: str, question_text: str = ""):
    """URSA-native step-boundary alignment.

    Algorithm (no analytic re-implementation — every step uses PRM's own
    code):
      1. Build prefix exactly like ``MathPRMReward._prepare_prm_input``:
           prefix = _PRM_PROMPT + question + "\\n"   (or _PRM_PROMPT + "\\n")
      2. Form the same string PRM scores on:
           prm_input_str = prm_module.replace_specific_plus_minus_with_ki(
               prefix + response_text)
      3. Tokenize with prm_module.tokenizer (the EXACT tokenizer PRM uses) and
         locate every ` и` token (id == prm_module.tag_id).
      4. Each ` и` token's offset_mapping char_start lies inside prm_input_str.
         Subtract len(prefix) and ``2 * k_tag`` (each prior ` и` adds 2 chars)
         to recover the position in the ORIGINAL ``response_text`` where the
         step-end occurs.
      5. Re-tokenize ``response_text`` (without ` и`) and find the response
         token whose char_end <= that position. That is the per-step
         boundary token index.

    Returned indices are relative to response start (0 = first response
    token). Caller (compute_reward via fast_exp_maker._compute_advantages)
    scatters per-step rewards onto these indices.

    Why native? It avoids any divergence between an analytic char-level
    model of ` и` insertion and PRM's actual tokenizer behavior. If the
    tokenizer ever merges ` и` with adjacent chars, this path stays correct
    because we read offsets from the actual tokenization PRM uses.

    Parameters
    ----------
    prm_module : MathPRMReward
        Provides ``_PRM_PROMPT``, ``tokenizer``, ``tag_id``, and
        ``replace_specific_plus_minus_with_ki``.
    response_text : str
        The actor-generated response (assistant content only, no chat tags).
    question_text : str, optional
        Prompt question — passed through ``_prepare_prm_input`` so the prefix
        length matches the PRM-side string exactly.

    Returns
    -------
    boundaries : list[int]
        Per-step boundary token indices in the response token sequence.
        ``len(boundaries) == number of step_scores PRM emits``.
    matched_patterns : list[str]
        ``Step N:`` / ``†Answer:`` patterns found in the response (debug aid).
    """
    matched_patterns = [m.group() for m in _STEP_OR_ANSWER_PATTERN.finditer(response_text)]

    if question_text and not isinstance(question_text, float):
        prefix_str = prm_module._PRM_PROMPT + question_text + "\n"
    else:
        prefix_str = prm_module._PRM_PROMPT + "\n"
    prefix_len = len(prefix_str)
    prm_input_str = prm_module.replace_specific_plus_minus_with_ki(prefix_str + response_text)

    tok = prm_module.tokenizer
    enc_prm = tok(prm_input_str, return_offsets_mapping=True, add_special_tokens=False)
    prm_offsets = enc_prm["offset_mapping"]
    prm_ids = enc_prm["input_ids"]
    tag_id = prm_module.tag_id

    char_in_response: list[int] = []
    k_tag = 0
    for tid, off in zip(prm_ids, prm_offsets):
        if tid == tag_id:
            char_in_response.append(off[0] - prefix_len - 2 * k_tag)
            k_tag += 1

    enc_resp = tok(response_text, return_offsets_mapping=True, add_special_tokens=False)
    resp_offsets = enc_resp["offset_mapping"]

    boundaries: list[int] = []
    for cp in char_in_response:
        last_idx = -1
        for tok_idx, (_, ce) in enumerate(resp_offsets):
            if 0 < ce <= cp:
                last_idx = tok_idx
        boundaries.append(last_idx)
    return boundaries, matched_patterns


class MathPRMReward(nn.Module):
    """Wrap URSA-RM with the original URSA-MATH step-level scoring protocol."""

    _SYSTEM_PROMPT = "You are a helpful assistant."
    _PRM_PROMPT = (
        "You are given a problem and a step-by-step solution. "
        "You need to check the correctness of each step.\nQuestion:"
    )
    _IMAGE_PAD = 575
    # PS-GRPO step-score drop hyperparameters (URSA-MATH paper):
    #   _DROP_THRESHOLD - relative drop fraction that counts as a "drop moment"
    #   _DROP_GAMMA     - reward penalty when a drop moment is observed for a
    #                     correct answer; final_reward = 1 - _DROP_GAMMA = 0.5
    _DROP_THRESHOLD = 0.3
    _DROP_GAMMA = 0.5

    def __init__(self, base_model: nn.Module, processor, aggregation: str = "min") -> None:
        super().__init__()
        self.model = base_model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.aggregation = aggregation

        tag_ids = self.tokenizer.encode(" и", add_special_tokens=False)
        assert len(tag_ids) == 1, (
            "The step tag ' и' must map to exactly one token. "
            f"Got {tag_ids!r} instead."
        )
        self.tag_id = int(tag_ids[0])

    @staticmethod
    def replace_specific_plus_minus_with_ki(text: str) -> str:
        """Insert the URSA step-boundary marker `` и`` before each next step."""
        pattern = r"Step \d+"
        matches = list(re.finditer(pattern, text))
        positions = [(match.start(), match.end()) for match in matches]
        if not positions:
            return text + " и"

        text_list = list(text)
        insert_positions = []
        try:
            for i in range(1, len(positions)):
                for j in range(positions[i][0] - 1, positions[i - 1][1], -1):
                    if text_list[j] not in {" ", "\n"}:
                        insert_positions.append(j + 1)
                        break

            answer_start = text.find("†Answer:")
            if answer_start != -1:
                for j in range(answer_start - 1, positions[-1][1], -1):
                    if text_list[j] not in {" ", "\n"}:
                        insert_positions.append(j + 1)
                        break

            for index in sorted(insert_positions, reverse=True):
                text = text[:index] + " и" + text[index:]
            return text
        except Exception:
            return text + " и"

    def _prepare_prm_input(self, question: str, response: str) -> str:
        if not question or isinstance(question, float):
            instruction = self._PRM_PROMPT + "\n" + response
        else:
            instruction = self._PRM_PROMPT + question + "\n" + response
        return self.replace_specific_plus_minus_with_ki(instruction)

    def _split_conversation(self, prompt_and_output: str) -> tuple[str, str]:
        question = ""
        response = ""

        for sep in ("<|im_start|>user\n", "User:", "USER:"):
            if sep not in prompt_and_output:
                continue
            user_block = prompt_and_output.split(sep)[-1]
            for end in ("<|im_end|>", "<|im_start|>"):
                if end in user_block:
                    user_block = user_block.split(end)[0]
            question = self._clean_question_text(user_block)
            break

        for sep in ("<|im_start|>assistant\n", "Assistant:", "ASSISTANT:"):
            if sep not in prompt_and_output:
                continue
            response_block = prompt_and_output.split(sep)[-1]
            for end in ("<|im_end|>", "<|endoftext|>"):
                if end in response_block:
                    response_block = response_block.split(end)[0]
            response = response_block.strip()
            break

        if not response:
            response = prompt_and_output
        return question, response

    @staticmethod
    def _clean_question_text(question: str) -> str:
        question = _clean_vision_token(question)
        question = question.replace("<|image|>", "").replace("<image>", "")
        return question.strip()

    @staticmethod
    def _select_prm_image(raw_image: Any) -> list[Any]:
        if isinstance(raw_image, (list, tuple)):
            for item in raw_image:
                if item is not None:
                    return [item]
            return [None]
        return [raw_image] if raw_image is not None else [None]

    @staticmethod
    def _safe_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _is_multiple_choice_reference(reference: str) -> bool:
        ref = normalize_answer(reference).strip().upper()
        return len(ref) == 1 and ref in {"A", "B", "C", "D"}

    @classmethod
    def _infer_reference_type(cls, reference: Any) -> tuple[str, bool]:
        reference_text = cls._safe_text(reference)
        if not reference_text:
            return "missing", False

        reference_norm = normalize_answer(reference_text).strip()
        if cls._is_multiple_choice_reference(reference_norm):
            return "multiple_choice", True

        if reference_norm.lower() in {"yes", "no", "true", "false"}:
            return "text", True

        numeric_candidate = reference_norm.replace(",", "")
        if re.fullmatch(r"-?\d+(?:\.\d+)?", numeric_candidate):
            return "numeric", True
        if re.fullmatch(r"-?\d+/\d+", numeric_candidate):
            return "numeric", True

        if any(token in reference_norm for token in ("\\", "=", "^", "_", "{", "}", "sqrt", "frac")):
            return "formula", True
        if re.search(r"[a-zA-Z]", reference_norm) and re.search(r"[\d=+\-*/()]", reference_norm):
            return "formula", True

        return "text", True

    @classmethod
    def _extract_answer_from_candidate(cls, candidate: str, reference_type: str) -> str:
        candidate = cls._safe_text(candidate)
        if not candidate:
            return ""

        boxed = extract_boxed_answer(candidate)
        if boxed:
            return boxed

        tagged = extract_answer_from_tags(candidate, "answer")
        if tagged:
            return tagged

        candidate = re.sub(
            r"^(?:†\s*)?(?:final answer|correct answer(?: is)?|the answer is|answer)\s*[:：]?\s*",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip()
        candidate = candidate.rstrip(" .")
        if not candidate:
            return ""

        if reference_type == "multiple_choice":
            extracted = extract_multiple_choice_answer(candidate)
            return extracted or normalize_answer(candidate).strip().upper()

        if reference_type == "numeric":
            if any(token in candidate for token in ("\\", "/", "=", "^", "{", "}", "sqrt", "frac")):
                return normalize_answer(candidate)
            extracted = extract_numeric_answer(candidate)
            return extracted or normalize_answer(candidate)

        if reference_type in {"formula", "text"}:
            return normalize_answer(candidate)

        return extract_answer(candidate)

    @classmethod
    def _extract_final_answer_details(cls, response: str, reference_type: str) -> Dict[str, Any]:
        response = cls._safe_text(response)
        details: Dict[str, Any] = {
            "predicted_answer": "",
            "answer_tag_present": False,
            "answer_extraction_failed": True,
            "used_answer_fallback": False,
            "extraction_source": "missing",
            "answer_marker_count": 0,
            "extra_answer_marker_count": 0,
            "post_answer_continuation_present": False,
            "post_answer_step_present": False,
            "answer_tail_cutoff_present": False,
            "clean_answer_protocol": False,
        }
        if not response:
            return details

        if MATH_PRM_ANSWER_MARKER in response:
            details["answer_tag_present"] = True
            details["answer_marker_count"] = response.count(MATH_PRM_ANSWER_MARKER)
            details["extra_answer_marker_count"] = max(0, details["answer_marker_count"] - 1)

            answer_block = response.split(MATH_PRM_ANSWER_MARKER, 1)[-1]
            answer_lines = answer_block.splitlines()
            candidate = answer_block.strip()
            candidate_line_index = None
            for line_index, line in enumerate(answer_lines):
                stripped_line = line.strip()
                if stripped_line:
                    candidate = stripped_line
                    candidate_line_index = line_index
                    break

            trailing_text = ""
            if candidate_line_index is not None:
                trailing_text = "\n".join(answer_lines[candidate_line_index + 1:]).strip()
            tail_cutoff = find_math_prm_tail_cutoff(candidate)
            if tail_cutoff is not None:
                details["answer_tail_cutoff_present"] = True
                candidate = candidate[:tail_cutoff]

            details["post_answer_step_present"] = bool(
                re.search(r"(?m)^\s*Step\s+\d+\s*:", trailing_text)
            )
            details["post_answer_continuation_present"] = bool(
                trailing_text
                or details["extra_answer_marker_count"] > 0
                or details["post_answer_step_present"]
                or details["answer_tail_cutoff_present"]
            )
            details["clean_answer_protocol"] = not details["post_answer_continuation_present"]
            predicted_answer = cls._extract_answer_from_candidate(candidate, reference_type)
            details["predicted_answer"] = predicted_answer
            details["answer_extraction_failed"] = predicted_answer == ""
            details["extraction_source"] = "dagger_answer"
            return details

        explicit_fallbacks = [
            ("boxed", extract_boxed_answer(response)),
            ("tagged_answer", extract_answer_from_tags(response, "answer")),
        ]
        for source, match in explicit_fallbacks:
            if match:
                details["predicted_answer"] = normalize_answer(match)
                details["answer_extraction_failed"] = False
                details["used_answer_fallback"] = True
                details["extraction_source"] = source
                return details

        lines = [line.strip() for line in response.splitlines() if line.strip()]
        if lines:
            last_line = lines[-1]
            explicit_line = re.match(
                r"^(?:†\s*)?(?:final answer|correct answer(?: is)?|the answer is|answer)\b",
                last_line,
                flags=re.IGNORECASE,
            )
            if explicit_line:
                predicted_answer = cls._extract_answer_from_candidate(last_line, reference_type)
                details["predicted_answer"] = predicted_answer
                details["answer_extraction_failed"] = predicted_answer == ""
                details["used_answer_fallback"] = True
                details["extraction_source"] = "explicit_last_line"
                return details

        return details

    @classmethod
    def _compare_final_answer(
        cls,
        predicted_answer: str,
        reference: Any,
        reference_type: str,
        reference_supported: bool,
    ) -> tuple[bool, str]:
        reference_text = cls._safe_text(reference)
        if not reference_supported:
            return False, "unsupported_reference"
        if not reference_text:
            return False, "missing_reference"
        if not predicted_answer:
            return False, "missing_prediction"

        if reference_type == "multiple_choice":
            pred_norm = normalize_answer(predicted_answer).strip().upper()
            ref_norm = normalize_answer(reference_text).strip().upper()
            return pred_norm == ref_norm, "multiple_choice_exact"

        if reference_type in {"numeric", "formula"}:
            if mathruler_grade_answer is not None:
                try:
                    if mathruler_grade_answer(predicted_answer, reference_text):
                        return True, "mathruler"
                except Exception:
                    pass
            return compare_answers(predicted_answer, reference_text, is_multiple_choice=False), "math_eval"

        return compare_answers(predicted_answer, reference_text, is_multiple_choice=False), "text_compare"

    @classmethod
    def _evaluate_answer_alignment(cls, response: str, reference: Any) -> Dict[str, Any]:
        reference_type, reference_supported = cls._infer_reference_type(reference)
        extraction = cls._extract_final_answer_details(response, reference_type)
        answer_content_correct, comparison_method = cls._compare_final_answer(
            extraction["predicted_answer"],
            reference,
            reference_type,
            reference_supported,
        )
        outcome_correct = answer_content_correct and not extraction["post_answer_continuation_present"]
        return {
            "reference_type": reference_type,
            "reference_supported": reference_supported,
            "comparison_method": comparison_method,
            **extraction,
            "answer_content_correct": answer_content_correct,
            "outcome_correct": outcome_correct,
        }

    @classmethod
    def _compute_relative_drop(cls, step_scores: torch.Tensor) -> tuple[float, bool]:
        if step_scores.numel() < 2:
            return 0.0, False

        scores = step_scores.detach().float()
        prev_scores = scores[:-1]
        next_scores = scores[1:]
        denom = torch.clamp(prev_scores, min=1e-6)
        relative_drops = torch.clamp((prev_scores - next_scores) / denom, min=0.0)
        max_relative_drop = float(relative_drops.max().item()) if relative_drops.numel() else 0.0
        return max_relative_drop, max_relative_drop >= cls._DROP_THRESHOLD

    @classmethod
    def _compute_psgrpo_metrics(
        cls,
        response: str,
        reference: Any,
        step_scores: torch.Tensor,
    ) -> Dict[str, float]:
        answer_eval = cls._evaluate_answer_alignment(response, reference)
        outcome_correct = float(answer_eval["outcome_correct"])
        max_relative_drop, has_drop_moment = cls._compute_relative_drop(step_scores)

        final_reward = 0.0
        if outcome_correct > 0.0:
            final_reward = 1.0 - cls._DROP_GAMMA if has_drop_moment else 1.0

        return {
            "outcome_correct": outcome_correct,
            "max_relative_drop": max_relative_drop,
            "has_drop_moment": float(has_drop_moment),
            "final_reward": final_reward,
            "answer_tag_present": float(answer_eval["answer_tag_present"]),
            "answer_extraction_failed": float(answer_eval["answer_extraction_failed"]),
            "used_answer_fallback": float(answer_eval["used_answer_fallback"]),
            "reference_supported": float(answer_eval["reference_supported"]),
            "used_mathruler": float(answer_eval["comparison_method"] == "mathruler"),
            "answer_content_correct": float(answer_eval["answer_content_correct"]),
            "answer_marker_count": float(answer_eval["answer_marker_count"]),
            "extra_answer_marker_count": float(answer_eval["extra_answer_marker_count"]),
            "post_answer_continuation": float(answer_eval["post_answer_continuation_present"]),
            "post_answer_step_present": float(answer_eval["post_answer_step_present"]),
            "answer_tail_cutoff_present": float(answer_eval["answer_tail_cutoff_present"]),
            "clean_answer_protocol": float(answer_eval["clean_answer_protocol"]),
        }

    @torch.no_grad()
    def forward(
        self,
        sequences,
        attention_mask,
        prompt_and_output=None,
        raw_images=None,
        references=None,
        labels=None,
        **kwargs,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        device = next(self.model.parameters()).device

        if prompt_and_output is None and sequences is not None:
            prompt_and_output = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        elif prompt_and_output is None:
            raise ValueError("Either sequences or prompt_and_output must be provided")

        return_dict = bool(kwargs.get("return_dict", False))

        batch_rewards = []
        # Per-sample reward metrics emitted alongside the scalar reward.
        # They are grouped into three buckets:
        #
        # 1. PRM step-score statistics (continuous, distribution shape):
        #      model_reward     - aggregated step score (min/avg/last per agg setting)
        #      step_score_min   - lowest step score in the response
        #      step_score_mean  - mean step score
        #      step_score_last  - score of the final step
        #      step_count       - number of "Step N:" blocks scored
        #
        # 2. Outcome / correctness signals (mostly binary):
        #      outcome_correct    - 1 if extracted answer matches ground truth, else 0
        #      has_drop_moment    - 1 if any consecutive step pair dropped > _DROP_THRESHOLD
        #      max_relative_drop  - magnitude of the largest relative drop
        #      final_reward       - PS-GRPO reward {0, 1-_DROP_GAMMA, 1} fed into GRPO
        #
        # 3. Diagnostics on answer extraction / grading path (low-volume but useful
        #    when debugging dataset / format / mathruler issues):
        #      answer_tag_present       - 1 if the "†Answer:" marker appeared
        #      answer_extraction_failed - 1 if no answer string could be extracted
        #      used_answer_fallback     - 1 if the heuristic last-line fallback fired
        #      reference_supported      - 1 if the ground-truth schema is recognized
        #      used_mathruler           - 1 if mathruler grading was the deciding step
        #      answer_content_correct   - 1 if the first extracted answer matches
        #      clean_answer_protocol    - 1 if there is no output after the answer line
        #      post_answer_continuation - 1 if the model continued after the answer line
        #      extra_answer_marker_count - number of extra "†Answer:" markers
        #
        # NOTE: ``accuracy_reward`` used to live here, but for math_psgrpo it is
        # exactly equal to ``outcome_correct`` (see _compute_psgrpo_metrics).
        # It now lives only in reward_models_utils.mix_rewards where it is set
        # by the rule branch for the math_rule / math_prm_combined recipes.
        batch_metrics: Dict[str, list[float]] = {
            "model_reward": [],
            "step_score_min": [],
            "step_score_mean": [],
            "step_score_last": [],
            "step_count": [],
            "outcome_correct": [],
            "max_relative_drop": [],
            "has_drop_moment": [],
            "final_reward": [],
            "answer_tag_present": [],
            "answer_extraction_failed": [],
            "used_answer_fallback": [],
            "reference_supported": [],
            "used_mathruler": [],
            "answer_content_correct": [],
            "answer_marker_count": [],
            "extra_answer_marker_count": [],
            "post_answer_continuation": [],
            "post_answer_step_present": [],
            "answer_tail_cutoff_present": [],
            "clean_answer_protocol": [],
            # math_per_step_prm diagnostics
            "alignment_failed": [],
            "n_aligned_steps": [],
        }
        # Per-step PRM data (variant 2). Lists of per-trajectory tensors,
        # only populated for label == "math_per_step_prm" trajectories.
        # When all trajectories in a batch have label == "math_psgrpo" the
        # collected lists stay empty and the dict keys are dropped at the end
        # so legacy callers that don't know about per-step rewards see no
        # change.
        batch_step_rewards: list[torch.Tensor] = []
        batch_step_token_indices: list[torch.Tensor] = []

        image_inputs = raw_images or [None] * len(prompt_and_output)
        ref_inputs = references or [None] * len(prompt_and_output)
        label_inputs = labels or ["math_prm"] * len(prompt_and_output)

        for text, sample_image, reference, label in zip_longest(
            prompt_and_output, image_inputs, ref_inputs, label_inputs, fillvalue=None
        ):
            if text is None:
                continue

            question, response = self._split_conversation(text)
            input_prompt = self._prepare_prm_input(question, response)
            conversation = [
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": "<|image|>" + input_prompt},
            ]
            formatted_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(
                formatted_prompt,
                self._select_prm_image(sample_image),
                return_tensors="pt",
            ).to(device, torch.bfloat16)

            # Sanity check: response (RL-generated) is not vision-cleaned, so it can
            # contain literal `<|image|>` / `<image>` strings that the tokenizer maps
            # to image_token_index. The PRM only ever receives one image, so any
            # extras would crash _merge_input_ids_with_image_features. Keep the first
            # image token (intended placeholder) and replace the rest with a benign
            # text token so PRM scoring continues instead of aborting the rollout.
            image_token_id = getattr(self.model.config, "image_token_index", None)
            if image_token_id is not None:
                input_ids_view = inputs["input_ids"]
                image_mask_flat = (input_ids_view == image_token_id).view(-1)
                extras = torch.nonzero(image_mask_flat, as_tuple=False).squeeze(-1)
                if extras.numel() > 1:
                    replacement = self.tokenizer.pad_token_id
                    if replacement is None:
                        replacement = self.tokenizer.eos_token_id
                    flat = input_ids_view.view(-1)
                    flat[extras[1:]] = replacement
                    inputs["input_ids"] = flat.view(input_ids_view.shape)

            reward = self.model(**inputs).logits
            input_ids = inputs["input_ids"].view(-1)
            padding = torch.full((self._IMAGE_PAD,), -1, device=device)
            input_ids_aligned = torch.cat((input_ids[:1], padding, input_ids[1:]))

            reward_flat = reward.view(-1)
            step_logits = reward_flat[input_ids_aligned == self.tag_id]
            step_scores = torch.sigmoid(step_logits).view(-1)
            psgrpo_metrics = self._compute_psgrpo_metrics(response, reference, step_scores)

            if step_scores.numel() == 0:
                aggregated_score = 0.0
            elif self.aggregation == "min":
                aggregated_score = float(torch.min(step_scores).item())
            elif self.aggregation in {"avg", "mean"}:
                aggregated_score = float(torch.mean(step_scores).item())
            elif self.aggregation == "last":
                aggregated_score = float(step_scores[-1].item())
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation!r}")

            # ---- variant 2 (per-step PRM reward) alignment -------------------
            # For label == "math_per_step_prm" we additionally locate the
            # boundary token of each "Step N:" inside the response so the
            # step_scores tensor can be scattered to per-token positions
            # downstream (instead of being collapsed to one scalar).
            #
            # Alignment is *self-contained*: we re-tokenize ``response`` with
            # the actor's (== PRM's, in URSA family) tokenizer and use the
            # offset mapping to reverse-find token indices for each "Step N:"
            # / "†Answer:" pattern. Indices are relative to the response
            # start so they line up with the action_mask axis of final_reward
            # in compute_reward.
            #
            # If the alignment fails (n_steps_prm != n_boundaries) we
            # *bypass* per-step mode for that trajectory: emit empty tensors
            # so compute_reward falls back to the trajectory-scalar path,
            # and bump the alignment_failed metric for monitoring.
            if label == "math_per_step_prm" and step_scores.numel() > 0:
                boundaries, matched_patterns = find_step_boundaries_in_response_tokens(
                    self, response, question_text=question
                )
                aligned = (len(boundaries) == int(step_scores.numel()))
                if aligned:
                    traj_step_rewards = step_scores.detach().to(torch.float32).cpu()
                    traj_step_tokens = torch.tensor(boundaries, dtype=torch.long)
                    n_aligned = len(boundaries)
                else:
                    # Alignment failed: emit empties; downstream falls back to
                    # trajectory-scalar mode for this row.
                    traj_step_rewards = torch.empty(0, dtype=torch.float32)
                    traj_step_tokens = torch.empty(0, dtype=torch.long)
                    n_aligned = 0
                batch_step_rewards.append(traj_step_rewards)
                batch_step_token_indices.append(traj_step_tokens)
                batch_metrics["alignment_failed"].append(0.0 if aligned else 1.0)
                batch_metrics["n_aligned_steps"].append(float(n_aligned))
            else:
                # No per-step request for this row; emit empty placeholders to
                # keep the per-traj list aligned with batch_rewards.
                batch_step_rewards.append(torch.empty(0, dtype=torch.float32))
                batch_step_token_indices.append(torch.empty(0, dtype=torch.long))
                batch_metrics["alignment_failed"].append(0.0)
                batch_metrics["n_aligned_steps"].append(0.0)

            # ---- trajectory-scalar reward (PSGRPO / aggregate path) ----------
            if label == "math_psgrpo":
                sequence_reward = psgrpo_metrics["final_reward"]
            elif label == "math_per_step_prm":
                # In per-step mode the trajectory-scalar field is still used
                # by GroupNorm baseline — use outcome (clean signal) instead
                # of aggregated_score (which would double-count step rewards).
                sequence_reward = float(psgrpo_metrics["outcome_correct"])
            else:
                sequence_reward = aggregated_score
            batch_rewards.append(sequence_reward)
            batch_metrics["model_reward"].append(aggregated_score)
            batch_metrics["step_score_min"].append(float(torch.min(step_scores).item()) if step_scores.numel() else 0.0)
            batch_metrics["step_score_mean"].append(float(torch.mean(step_scores).item()) if step_scores.numel() else 0.0)
            batch_metrics["step_score_last"].append(float(step_scores[-1].item()) if step_scores.numel() else 0.0)
            batch_metrics["step_count"].append(float(step_scores.numel()))
            # Diagnostics: outcome_correct and answer-extraction signals are
            # always meaningful (they're computed by _evaluate_answer_alignment
            # which is independent of PSGRPO drop-moment); only the
            # drop-moment-specific fields (max_relative_drop, has_drop_moment,
            # final_reward) zero out for non-PSGRPO labels.
            _UNIVERSAL_METRICS = {
                "outcome_correct",
                "answer_tag_present",
                "answer_extraction_failed",
                "used_answer_fallback",
                "reference_supported",
                "used_mathruler",
                "answer_content_correct",
                "answer_marker_count",
                "extra_answer_marker_count",
                "post_answer_continuation",
                "post_answer_step_present",
                "answer_tail_cutoff_present",
                "clean_answer_protocol",
            }
            for key in (
                "outcome_correct",
                "max_relative_drop",
                "has_drop_moment",
                "final_reward",
                "answer_tag_present",
                "answer_extraction_failed",
                "used_answer_fallback",
                "reference_supported",
                "used_mathruler",
                "answer_content_correct",
                "answer_marker_count",
                "extra_answer_marker_count",
                "post_answer_continuation",
                "post_answer_step_present",
                "answer_tail_cutoff_present",
                "clean_answer_protocol",
            ):
                if label == "math_psgrpo" or key in _UNIVERSAL_METRICS:
                    batch_metrics[key].append(psgrpo_metrics[key])
                else:
                    # PSGRPO-specific (max_relative_drop, has_drop_moment,
                    # final_reward) — only meaningful when label is
                    # "math_psgrpo"; zero out for other labels to preserve
                    # historical metric tensor shape & semantics.
                    batch_metrics[key].append(0.0)

        score_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
        if references is None and labels is None and not return_dict:
            return score_tensor

        metrics_tensor = {
            key: torch.tensor(values, dtype=torch.float32, device=device)
            for key, values in batch_metrics.items()
        }
        out = {"score": score_tensor, **metrics_tensor}

        # Only attach per-step fields if any trajectory had non-empty step data.
        # Stored as Python lists of CPU tensors (variable length per traj) to
        # avoid forcing every caller to handle padded tensors.
        any_per_step = any(t.numel() > 0 for t in batch_step_rewards)
        if any_per_step:
            out["step_rewards"] = batch_step_rewards
            out["step_token_indices"] = batch_step_token_indices

        return out
