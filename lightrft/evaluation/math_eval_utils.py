"""
Mathematical Evaluation Utilities

This module provides utilities for extracting and evaluating mathematical answers
from model responses, supporting various formats including:
- LaTeX boxed answers: \\boxed{answer}
- Plain numeric answers
- Multiple choice answers (A, B, C, D)
- Text-based answers

These utilities are designed for evaluating performance on mathematical
reasoning benchmarks like Math500, AIME, and GPQA Diamond.
"""

import re
from typing import Optional, List, Tuple, Dict, Any
import sympy
from sympy.parsing.latex import parse_latex


def normalize_answer(answer: str) -> str:
    """
    Normalize a mathematical answer for comparison.

    This function:
    - Removes extra whitespace
    - Converts to lowercase (for text answers)
    - Removes LaTeX formatting
    - Normalizes common mathematical expressions

    :param answer: Raw answer string
    :type answer: str
    :return: Normalized answer string
    :rtype: str
    """
    if not answer:
        return ""

    # Remove extra whitespace
    answer = answer.strip()

    # Remove dollar signs (LaTeX math mode)
    answer = answer.replace("$", "")

    # Remove \\text{} wrapper
    answer = re.sub(r"\\text\{([^}]*)\}", r"\1", answer)

    # Remove \\mathrm{}, \\mathbf{}, etc.
    answer = re.sub(r"\\math[a-z]{2}\{([^}]*)\}", r"\1", answer)

    # Normalize spaces
    answer = " ".join(answer.split())

    return answer


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract answer from \\boxed{} LaTeX command.

    Handles nested braces and returns the innermost boxed content.

    :param text: Text containing boxed answer
    :type text: str
    :return: Extracted answer or None if not found
    :rtype: Optional[str]
    """
    # Pattern to match \boxed{...} with proper brace matching
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text)

    if matches:
        # Return the last boxed answer (usually the final answer)
        return normalize_answer(matches[-1])

    return None


def extract_answer_from_tags(text: str, tag: str = "answer") -> Optional[str]:
    """
    Extract answer from custom tags like <answer>...</answer>.

    :param text: Text containing tagged answer
    :type text: str
    :param tag: Tag name to search for
    :type tag: str
    :return: Extracted answer or None if not found
    :rtype: Optional[str]
    """
    pattern = rf"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

    if matches:
        return normalize_answer(matches[-1])

    return None


def extract_multiple_choice_answer(text: str) -> Optional[str]:
    """
    Extract multiple choice answer (A, B, C, D) from text.

    Looks for patterns like:
    - "The answer is A"
    - "Answer: B"
    - "选择 C"
    - Just "D" at the end

    :param text: Text containing multiple choice answer
    :type text: str
    :return: Extracted choice letter or None if not found
    :rtype: Optional[str]
    """
    # Remove think tags if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Pattern 1: "answer is X" or "答案是X"
    pattern1 = r"(?:answer|答案|选择)(?:\s+is)?[\s:：]*([A-D])"
    match = re.search(pattern1, text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 2: Look for standalone letter at the end
    pattern2 = r"\b([A-D])\s*$"
    match = re.search(pattern2, text, re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).upper()

    # Pattern 3: Look for "选项X" or "Option X"
    pattern3 = r"(?:option|选项)[\s:：]*([A-D])"
    match = re.search(pattern3, text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None


def extract_numeric_answer(text: str) -> Optional[str]:
    """
    Extract numeric answer from text.

    Handles:
    - Integers: 42
    - Decimals: 3.14
    - Fractions: 1/2
    - Scientific notation: 1.5e10

    :param text: Text containing numeric answer
    :type text: str
    :return: Extracted number as string or None if not found
    :rtype: Optional[str]
    """
    # Remove think tags if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Pattern for numbers (with optional comma separators)
    pattern = r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?"
    matches = re.findall(pattern, text)

    if matches:
        # Return the last number found
        number = matches[-1].replace(",", "")
        return number

    return None


def extract_answer(text: str, is_multiple_choice: bool = False) -> str:
    """
    Extract answer from model response using multiple strategies.

    Tries in order:
    1. Boxed answer (\\boxed{})
    2. Tagged answer (<answer>...</answer>)
    3. Multiple choice answer (if is_multiple_choice=True)
    4. Numeric answer
    5. Last line of text (fallback)

    :param text: Model response text
    :type text: str
    :param is_multiple_choice: Whether this is a multiple choice question
    :type is_multiple_choice: bool
    :return: Extracted answer
    :rtype: str
    """
    if not text:
        return ""

    # Try boxed answer
    answer = extract_boxed_answer(text)
    if answer:
        return answer

    # Try tagged answer
    answer = extract_answer_from_tags(text, "answer")
    if answer:
        return answer

    # For multiple choice, try to extract letter
    if is_multiple_choice:
        answer = extract_multiple_choice_answer(text)
        if answer:
            return answer

    # Try numeric answer
    answer = extract_numeric_answer(text)
    if answer:
        return answer

    # Fallback: return last non-empty line
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        return normalize_answer(lines[-1])

    return ""


def compare_answers(predicted: str, ground_truth: str, is_multiple_choice: bool = False) -> bool:
    """
    Compare predicted answer with ground truth.

    For multiple choice: exact match (case-insensitive)
    For math: tries symbolic comparison with SymPy
    For text: normalized string comparison

    :param predicted: Predicted answer
    :type predicted: str
    :param ground_truth: Ground truth answer
    :type ground_truth: str
    :param is_multiple_choice: Whether this is a multiple choice question
    :type is_multiple_choice: bool
    :return: True if answers match
    :rtype: bool
    """
    if not predicted or not ground_truth:
        return predicted == ground_truth

    # Normalize both answers
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    # Exact match (case-insensitive)
    if pred_norm.lower() == gt_norm.lower():
        return True

    # For multiple choice, only exact match counts
    if is_multiple_choice:
        return False

    # Try symbolic comparison with SymPy
    try:
        # Remove LaTeX commands for SymPy parsing
        pred_clean = pred_norm.replace("\\", "")
        gt_clean = gt_norm.replace("\\", "")

        # Parse as SymPy expressions
        pred_expr = sympy.sympify(pred_clean)
        gt_expr = sympy.sympify(gt_clean)

        # Simplify and compare
        diff = sympy.simplify(pred_expr - gt_expr)
        return diff == 0
    except Exception:
        pass

    # Try numeric comparison
    try:
        pred_float = float(pred_norm)
        gt_float = float(gt_norm)
        return abs(pred_float - gt_float) < 1e-6
    except Exception:
        pass

    # Fallback: string comparison with some flexibility
    # Remove common variations
    pred_simple = pred_norm.replace(" ", "").replace(",", "").lower()
    gt_simple = gt_norm.replace(" ", "").replace(",", "").lower()

    return pred_simple == gt_simple


def evaluate_predictions(
    predictions: List[str],
    ground_truths: List[str],
    is_multiple_choice: List[bool],
) -> Dict[str, Any]:
    """
    Evaluate a list of predictions against ground truths.

    :param predictions: List of predicted answers
    :type predictions: List[str]
    :param ground_truths: List of ground truth answers
    :type ground_truths: List[str]
    :param is_multiple_choice: List indicating if each question is multiple choice
    :type is_multiple_choice: List[bool]
    :return: Dictionary with evaluation metrics
    :rtype: Dict[str, Any]
    """
    if len(predictions) != len(ground_truths) or len(predictions) != len(is_multiple_choice):
        raise ValueError("Length mismatch between predictions, ground_truths, and is_multiple_choice")

    total = len(predictions)
    correct = 0
    results = []

    for pred, gt, is_mc in zip(predictions, ground_truths, is_multiple_choice):
        is_correct = compare_answers(pred, gt, is_mc)
        correct += int(is_correct)
        results.append({
            "predicted": pred,
            "ground_truth": gt,
            "is_correct": is_correct,
            "is_multiple_choice": is_mc,
        })

    accuracy = correct / total if total > 0 else 0.0

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "results": results,
    }


__all__ = [
    "normalize_answer",
    "extract_boxed_answer",
    "extract_answer_from_tags",
    "extract_multiple_choice_answer",
    "extract_numeric_answer",
    "extract_answer",
    "compare_answers",
    "evaluate_predictions",
]
