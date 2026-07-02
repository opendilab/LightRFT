"""Tests for Math PRM final-answer protocol enforcement."""

from __future__ import annotations

import os
import sys
import unittest

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from reward_models import MathPRMReward  # noqa: E402
from reward_models_utils import mix_rewards  # noqa: E402


class TestAnswerProtocol(unittest.TestCase):
    def test_clean_answer_remains_correct(self):
        response = "Step 1: Compute directly.\n†Answer: 42"

        details = MathPRMReward._evaluate_answer_alignment(response, "42")

        self.assertTrue(details["answer_content_correct"])
        self.assertTrue(details["outcome_correct"])
        self.assertTrue(details["clean_answer_protocol"])
        self.assertFalse(details["post_answer_continuation_present"])

    def test_correct_first_answer_with_extra_marker_is_not_clean_outcome(self):
        response = "Step 1: Compute directly.\n†Answer: 42\n†Answer: 43"

        details = MathPRMReward._evaluate_answer_alignment(response, "42")

        self.assertTrue(details["answer_content_correct"])
        self.assertFalse(details["outcome_correct"])
        self.assertFalse(details["clean_answer_protocol"])
        self.assertTrue(details["post_answer_continuation_present"])
        self.assertEqual(details["extra_answer_marker_count"], 1)

    def test_correct_first_answer_with_extra_step_is_not_clean_outcome(self):
        response = "Step 1: Compute directly.\n†Answer: B\nStep 2: Keep talking."

        details = MathPRMReward._evaluate_answer_alignment(response, "B")

        self.assertTrue(details["answer_content_correct"])
        self.assertFalse(details["outcome_correct"])
        self.assertTrue(details["post_answer_step_present"])
        self.assertTrue(details["post_answer_continuation_present"])

    def test_mix_rewards_uses_clean_outcome_for_per_step_prm(self):
        labels = ["math_per_step_prm"]
        model_scores = torch.tensor([[0.0]], dtype=torch.float32)
        model_metrics = [{
            "outcome_correct": torch.tensor([0.0]),
            "answer_content_correct": torch.tensor([1.0]),
            "answer_tag_present": torch.tensor([1.0]),
            "answer_extraction_failed": torch.tensor([0.0]),
            "reference_supported": torch.tensor([1.0]),
            "post_answer_continuation": torch.tensor([1.0]),
            "clean_answer_protocol": torch.tensor([0.0]),
        }]

        reward, metrics = mix_rewards(
            labels=labels,
            model_scores=model_scores,
            label_map={"math_prm": 0},
            solution_strs=["Step 1: Compute directly.\n†Answer: 42\n†Answer: 43"],
            refs=["42"],
            model_reward_metrics_list=model_metrics,
        )

        self.assertEqual(float(reward[0].item()), 0.0)
        self.assertEqual(float(metrics["outcome_correct"][0].item()), 0.0)
        self.assertEqual(float(metrics["answer_content_correct"][0].item()), 1.0)
        self.assertEqual(float(metrics["post_answer_continuation"][0].item()), 1.0)

    def test_psgrpo_metrics_zero_reward_for_dirty_correct_answer(self):
        response = "Step 1: Compute directly.\n†Answer: 42\n†Answer: 43"

        metrics = MathPRMReward._compute_psgrpo_metrics(response, "42", torch.tensor([0.9]))

        self.assertEqual(metrics["answer_content_correct"], 1.0)
        self.assertEqual(metrics["outcome_correct"], 0.0)
        self.assertEqual(metrics["final_reward"], 0.0)
        self.assertEqual(metrics["post_answer_continuation"], 1.0)
        self.assertEqual(metrics["clean_answer_protocol"], 0.0)


if __name__ == "__main__":
    unittest.main()
