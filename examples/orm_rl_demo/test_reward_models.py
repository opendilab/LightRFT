"""
Smoke test for the general reward model used by the ORM RL demo.

This script intentionally uses public text-only examples so it can validate the
general reward model without depending on private datasets or absolute paths.
"""

import argparse
import os
import sys

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.append(os.path.dirname(__file__))
from reward_models import Qwen2VLRewardModelGeneral


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke test the general reward model used by orm_rl_demo."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path or HuggingFace id for the general reward model.",
    )
    return parser.parse_args()


def build_dialog(question: str, response: str) -> str:
    return (
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>\n"
    )


def load_reward_model(model_path: str) -> Qwen2VLRewardModelGeneral:
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )
    reward_model = Qwen2VLRewardModelGeneral(
        base_model,
        processor.tokenizer,
        processor,
        text_only=True,
    )
    reward_model.eval()
    return reward_model


def run_case(reward_model: Qwen2VLRewardModelGeneral, case: dict) -> None:
    outputs = reward_model(
        input_ids=None,
        attention_mask=None,
        references=[case["reference"]],
        prompt_and_output=[build_dialog(case["question"], case["response"])],
        raw_images=[None],
    )
    score = float(outputs["score"].item())
    print(f"{case['name']}: score={score:.1f}, expected={case['expected']:.1f}")
    if abs(score - case["expected"]) > 1e-6:
        raise AssertionError(
            f"{case['name']} expected {case['expected']}, got {score}"
        )


def main() -> None:
    args = parse_args()
    reward_model = load_reward_model(args.model)

    test_cases = [
        {
            "name": "correct_answer",
            "question": "What is 2 + 2?",
            "response": "The answer is 4.",
            "reference": "4",
            "expected": 1.0,
        },
        {
            "name": "incorrect_answer",
            "question": "What is 2 + 2?",
            "response": "The answer is 5.",
            "reference": "4",
            "expected": 0.0,
        },
    ]

    for case in test_cases:
        run_case(reward_model, case)

    print("general reward model smoke test passed")


if __name__ == "__main__":
    with torch.no_grad():
        main()
