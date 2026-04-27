"""
Helpers for URSA Math PRM structured outputs.

These helpers centralize the heuristics used to keep Phase 3 generation inside the
expected `Step N:` / `†Answer:` format without introducing Phase 4 reward logic.
"""

import re
from typing import Optional


MATH_PRM_STRUCTURED_LABELS = frozenset({"math_prm", "math_prm_combined", "math_psgrpo"})
MATH_PRM_ANSWER_MARKER = "†Answer:"
_MAX_MATH_PRM_ANSWER_WORDS = 24
_MAX_MATH_PRM_ANSWER_CHARS = 160
_EARLY_STOP_ANSWER_WORDS = 12
_EARLY_STOP_ANSWER_CHARS = 80
_BOOLEAN_ANSWERS = {"yes", "no", "true", "false"}
_ALGEBRAIC_ANSWER_PATTERN = re.compile(
    r"[A-Za-z][A-Za-z0-9_]*(?:\s*[,;]\s*[A-Za-z][A-Za-z0-9_]*)*\s*=\s*[-+A-Za-z0-9$\\][A-Za-z0-9\s,./%()=\-+*$\\^{}]*"
)


def is_math_prm_structured_label(label: Optional[str]) -> bool:
    return isinstance(label, str) and label.lower() in MATH_PRM_STRUCTURED_LABELS


def find_math_prm_tail_cutoff(text: str) -> Optional[int]:
    cut_positions = []
    for pattern in (
        r"(?<!^)(?:†Answer:|Step\s+\d+:)",
        r"([0-9])\1{15,}",
        r"([A-Za-z])\1{15,}",
        r"(\b\S+\b)(?:\s+\1){3,}",
        r"(\b\S+\s+\S+\b)(?:\s+\1){2,}",
    ):
        match = re.search(pattern, text)
        if match:
            cut_positions.append(match.start())
    return min(cut_positions) if cut_positions else None


def _normalize_math_prm_response(response_text: str) -> str:
    if not response_text:
        return response_text
    return re.sub(r"(?m)^StepStep\s+(\d+:)", r"Step \1", response_text)


def _extract_answer_line(response_text: str) -> tuple[str, str, bool]:
    normalized_text = _normalize_math_prm_response(response_text)
    marker_index = normalized_text.find(MATH_PRM_ANSWER_MARKER)
    if marker_index < 0:
        return normalized_text, "", False

    answer_tail = normalized_text[marker_index + len(MATH_PRM_ANSWER_MARKER):].lstrip()
    answer_lines = answer_tail.splitlines()
    answer_line = " ".join(answer_lines[0].split()) if answer_lines else ""
    has_more_lines = len(answer_lines) > 1
    return normalized_text, answer_line, has_more_lines


def should_stop_math_prm_response_text(response_text: str) -> bool:
    normalized_text, answer_line, has_more_lines = _extract_answer_line(response_text)
    if normalized_text.find(MATH_PRM_ANSWER_MARKER) < 0 or not answer_line:
        return False
    if has_more_lines:
        return True
    if find_math_prm_tail_cutoff(answer_line) is not None:
        return True

    lower_answer = answer_line.lower()
    if lower_answer in _BOOLEAN_ANSWERS:
        return True
    if re.fullmatch(r"[-+]?[$]?\d[\d\s,./%()=-]*", answer_line):
        return True
    if _ALGEBRAIC_ANSWER_PATTERN.fullmatch(answer_line):
        return True
    if re.fullmatch(r"[A-E]", answer_line):
        return True
    if answer_line.endswith((".", "!", "?", "%", ")", "]")):
        return True
    if len(answer_line.split()) >= _EARLY_STOP_ANSWER_WORDS:
        return True
    if len(answer_line) >= _EARLY_STOP_ANSWER_CHARS:
        return True
    return False


def sanitize_math_prm_response_text(response_text: str) -> str:
    normalized_text, answer_line, _ = _extract_answer_line(response_text)
    marker_index = normalized_text.find(MATH_PRM_ANSWER_MARKER)
    if marker_index < 0:
        return normalized_text

    prefix = normalized_text[: marker_index + len(MATH_PRM_ANSWER_MARKER)]

    cutoff = find_math_prm_tail_cutoff(answer_line)
    if cutoff is not None:
        answer_line = answer_line[:cutoff]

    answer_words = answer_line.split()
    if len(answer_words) > _MAX_MATH_PRM_ANSWER_WORDS:
        answer_line = " ".join(answer_words[:_MAX_MATH_PRM_ANSWER_WORDS])
    if len(answer_line) > _MAX_MATH_PRM_ANSWER_CHARS:
        truncated = answer_line[:_MAX_MATH_PRM_ANSWER_CHARS]
        answer_line = truncated.rsplit(" ", 1)[0] or truncated

    answer_line = answer_line.rstrip(" ,;:")
    return prefix.rstrip() if not answer_line else f"{prefix} {answer_line}".rstrip()
