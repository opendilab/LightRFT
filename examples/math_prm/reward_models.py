"""URSA-MATH Stage 3 reward model helpers."""

from __future__ import annotations

import re
from itertools import zip_longest
from typing import Any, Dict

import torch
import torch.nn as nn

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


class MathPRMReward(nn.Module):
    """Wrap URSA-RM with the original URSA-MATH step-level scoring protocol."""

    _SYSTEM_PROMPT = "You are a helpful assistant."
    _PRM_PROMPT = (
        "You are given a problem and a step-by-step solution. "
        "You need to check the correctness of each step.\nQuestion:"
    )
    _IMAGE_PAD = 575
    _DROP_THRESHOLD = 0.3
    _DROP_GAMMA = 0.5
    _REFERENCE_TYPE_TO_ID = {
        "missing": 0.0,
        "multiple_choice": 1.0,
        "numeric": 2.0,
        "formula": 3.0,
        "text": 4.0,
    }

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
        }
        if not response:
            return details

        if "†Answer:" in response:
            details["answer_tag_present"] = True
            answer_block = response.split("†Answer:", 1)[-1]
            answer_block = re.split(r"\n\s*Step\s+\d+\s*:", answer_block, maxsplit=1)[0]
            candidate_lines = [line.strip() for line in answer_block.splitlines() if line.strip()]
            candidate = candidate_lines[0] if candidate_lines else answer_block.strip()
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
        outcome_correct, comparison_method = cls._compare_final_answer(
            extraction["predicted_answer"],
            reference,
            reference_type,
            reference_supported,
        )
        return {
            "reference_type": reference_type,
            "reference_supported": reference_supported,
            "comparison_method": comparison_method,
            **extraction,
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
            "accuracy_reward": outcome_correct,
            "max_relative_drop": max_relative_drop,
            "has_drop_moment": float(has_drop_moment),
            "final_reward": final_reward,
            "answer_tag_present": float(answer_eval["answer_tag_present"]),
            "answer_extraction_failed": float(answer_eval["answer_extraction_failed"]),
            "used_answer_fallback": float(answer_eval["used_answer_fallback"]),
            "reference_supported": float(answer_eval["reference_supported"]),
            "used_mathruler": float(answer_eval["comparison_method"] == "mathruler"),
            "reference_type_id": cls._REFERENCE_TYPE_TO_ID[answer_eval["reference_type"]],
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
        batch_metrics: Dict[str, list[float]] = {
            "model_reward": [],
            "step_score_min": [],
            "step_score_mean": [],
            "step_score_last": [],
            "step_count": [],
            "accuracy_reward": [],
            "outcome_correct": [],
            "max_relative_drop": [],
            "has_drop_moment": [],
            "final_reward": [],
            "answer_tag_present": [],
            "answer_extraction_failed": [],
            "used_answer_fallback": [],
            "reference_supported": [],
            "used_mathruler": [],
            "reference_type_id": [],
        }
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

            sequence_reward = psgrpo_metrics["final_reward"] if label == "math_psgrpo" else aggregated_score
            batch_rewards.append(sequence_reward)
            batch_metrics["model_reward"].append(aggregated_score)
            batch_metrics["step_score_min"].append(float(torch.min(step_scores).item()) if step_scores.numel() else 0.0)
            batch_metrics["step_score_mean"].append(float(torch.mean(step_scores).item()) if step_scores.numel() else 0.0)
            batch_metrics["step_score_last"].append(float(step_scores[-1].item()) if step_scores.numel() else 0.0)
            batch_metrics["step_count"].append(float(step_scores.numel()))
            for key in (
                "accuracy_reward",
                "outcome_correct",
                "max_relative_drop",
                "has_drop_moment",
                "final_reward",
                "answer_tag_present",
                "answer_extraction_failed",
                "used_answer_fallback",
                "reference_supported",
                "used_mathruler",
                "reference_type_id",
            ):
                batch_metrics[key].append(psgrpo_metrics[key] if label == "math_psgrpo" else 0.0)

        score_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
        if references is None and labels is None and not return_dict:
            return score_tensor

        metrics_tensor = {
            key: torch.tensor(values, dtype=torch.float32, device=device)
            for key, values in batch_metrics.items()
        }
        return {"score": score_tensor, **metrics_tensor}
