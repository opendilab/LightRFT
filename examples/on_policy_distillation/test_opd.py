"""
Test script for On-Policy Distillation implementation in LightRFT.

This script validates the core components of the on-policy distillation mechanism.
"""

import torch
import sys
from pathlib import Path

# Add LightRFT to path
lightrft_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(lightrft_path))

def test_advantage_calculator():
    """Test OnPolicyDistillationCalculator."""
    print("Testing OnPolicyDistillationCalculator...")

    from lightrft.trainer.advantage_calculator import get_advantage_calculator

    # Create mock config
    class MockConfig:
        advantages_norm = True
        advantage_clip = 10.0

    config = MockConfig()

    # Create calculator
    calculator = get_advantage_calculator("on_policy_distillation", config)
    print(f"✓ Created calculator: {calculator.__class__.__name__}")

    # Create mock experience
    class MockExperience:
        def __init__(self):
            self.action_log_probs = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
            self.action_mask = torch.tensor([[True, True, True], [True, True, False]])
            self.info = {
                "teacher_log_probs": torch.tensor([[0.3, 0.4, 0.5], [0.4, 0.5, 0.6]])
            }

    experience = MockExperience()
    final_reward = torch.zeros_like(experience.action_log_probs)
    generate_kwargs = {}

    # Compute advantages
    advantages, returns, info = calculator.compute(
        experience, final_reward, gamma=1.0, generate_kwargs=generate_kwargs
    )

    print(f"✓ Computed advantages: {advantages.shape}")
    print(f"  Advantages sample: {advantages[0]}")
    print(f"  Expected positive (teacher > student): {(advantages > 0).float().mean():.2f}")

    # Test error case (missing teacher_log_probs)
    experience_no_teacher = MockExperience()
    del experience_no_teacher.info["teacher_log_probs"]

    try:
        calculator.compute(experience_no_teacher, final_reward, gamma=1.0, generate_kwargs=generate_kwargs)
        print("✗ Should have raised ValueError for missing teacher_log_probs")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {str(e)[:50]}...")

    print("✓ OnPolicyDistillationCalculator tests passed\n")
    return True


def test_reward_function():
    """Test teacher logprob extraction functions."""
    print("Testing teacher logprob functions...")

    from examples.on_policy_distillation.on_policy_distillation_reward import (
        extract_teacher_logprobs
    )

    # Test SGLang format
    sglang_response = {
        "meta_info": {
            "input_token_logprobs": [
                None,  # First token has no logprob
                [-0.1, 1, "hello"],
                [-0.2, 2, "world"],
                [-0.15, 1, "!"],
            ]
        }
    }

    teacher_log_probs = extract_teacher_logprobs(
        [sglang_response],
        response_lengths=[3],
        device="cpu"
    )

    print(f"✓ Extracted teacher log probs (SGLang format): {teacher_log_probs[0]}")
    assert len(teacher_log_probs[0]) == 3, "Should extract exactly 3 response tokens"
    assert torch.allclose(teacher_log_probs[0], torch.tensor([-0.1, -0.2, -0.15])), "Values mismatch"

    # Test vLLM format
    vllm_response = {
        "token_logprobs": [None, -0.1, -0.2, -0.15, -0.3]
    }

    teacher_log_probs = extract_teacher_logprobs(
        [vllm_response],
        response_lengths=[3],
        device="cpu"
    )

    print(f"✓ Extracted teacher log probs (vLLM format): {teacher_log_probs[0]}")
    assert len(teacher_log_probs[0]) == 3, "Should extract exactly 3 response tokens"

    print("✓ Teacher logprob extraction tests passed\n")
    return True


def test_factory_function():
    """Test that on_policy_distillation is registered in factory."""
    print("Testing factory function registration...")

    from lightrft.trainer.advantage_calculator import get_advantage_calculator

    class MockConfig:
        advantages_norm = False
        advantage_clip = 0.0

    config = MockConfig()

    # Test valid estimators
    estimators = [
        "gae",
        "reinforce",
        "rloo",
        "reinforce_baseline",
        "group_norm",
        "cpgd",
        "on_policy_distillation"
    ]

    for estimator in estimators:
        try:
            calc = get_advantage_calculator(estimator, config)
            print(f"✓ Created {estimator}: {calc.__class__.__name__}")
        except Exception as e:
            print(f"✗ Failed to create {estimator}: {e}")
            return False

    # Test invalid estimator
    try:
        get_advantage_calculator("invalid_estimator", config)
        print("✗ Should have raised ValueError for invalid estimator")
        return False
    except ValueError:
        print("✓ Correctly raised ValueError for invalid estimator")

    print("✓ Factory function tests passed\n")
    return True


def test_integration():
    """Test basic integration flow."""
    print("Testing integration flow...")

    # This is a simplified test to ensure components work together
    from lightrft.trainer.advantage_calculator import OnPolicyDistillationCalculator

    class MockConfig:
        advantages_norm = True
        advantage_clip = 5.0

    calculator = OnPolicyDistillationCalculator(MockConfig())

    # Simulate a batch of experiences
    batch_size = 4
    seq_len = 10

    class MockExperience:
        def __init__(self):
            # Student generated these log probs
            self.action_log_probs = torch.randn(batch_size, seq_len) * 0.5 - 1.0

            # Teacher evaluated and got these log probs
            # Teacher is better, so generally higher log probs
            self.info = {
                "teacher_log_probs": torch.randn(batch_size, seq_len) * 0.3 - 0.5
            }

            # Action mask (last 2 tokens are padding)
            self.action_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
            self.action_mask[:, -2:] = False

    experience = MockExperience()
    final_reward = torch.zeros_like(experience.action_log_probs)

    advantages, returns, info = calculator.compute(
        experience, final_reward, gamma=1.0, generate_kwargs={}
    )

    print(f"✓ Computed advantages for batch: {advantages.shape}")
    print(f"  Mean advantage: {advantages.mean():.4f}")
    print(f"  Std advantage: {advantages.std():.4f}")
    print(f"  Masked correctly: {advantages[:, -2:].sum() == 0}")

    # Check that advantages are normalized (approximately)
    masked_adv = advantages[experience.action_mask]
    assert abs(masked_adv.mean()) < 0.1, "Advantages should be normalized (mean ≈ 0)"
    print(f"✓ Advantages are normalized (mean={masked_adv.mean():.4f})")

    print("✓ Integration tests passed\n")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("On-Policy Distillation Test Suite")
    print("=" * 70)
    print()

    tests = [
        ("Factory Function", test_factory_function),
        ("Advantage Calculator", test_advantage_calculator),
        ("Reward Function", test_reward_function),
        ("Integration", test_integration),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("=" * 70)
    print("Test Summary")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
