"""
Test script for On-Policy Distillation implementation in LightRFT.

Tests both pure and hybrid OPD modes, advantage whitening,
teacher logprob extraction, and dimension alignment.
"""

import torch
import sys
from pathlib import Path

lightrft_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(lightrft_path))


class MockConfig:
    """Reusable mock config for advantage calculators."""
    def __init__(self, **kwargs):
        self.advantages_norm = False
        self.advantage_clip = 0.0
        self.opd_kl_coef = 1.0
        self.n_samples_per_prompt = 4
        self.dynamic_sampling = False
        self.micro_train_batch_size = 4
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockExperience:
    """Reusable mock experience object."""
    def __init__(self, batch_size=4, num_actions=10, teacher_offset=-0.5):
        self.action_log_probs = torch.randn(batch_size, num_actions) * 0.5 - 1.0
        self.action_mask = torch.ones(batch_size, num_actions, dtype=torch.bool)
        self.action_mask[:, -2:] = False  # last 2 tokens are padding
        self.info = {
            "teacher_log_probs": self.action_log_probs + teacher_offset,
            "reward": torch.rand(batch_size),
            "response_length": torch.full((batch_size,), num_actions),
        }


# ============================================================================
# Test: Factory registration
# ============================================================================

def test_factory():
    """All estimators including both OPD modes are registered."""
    print("Test: Factory registration")

    from lightrft.trainer.advantage_calculator import get_advantage_calculator

    config = MockConfig()
    estimators = [
        "gae", "reinforce", "rloo", "reinforce_baseline",
        "group_norm", "grpo", "cpgd",
        "on_policy_distillation",
        "on_policy_distillation_hybrid",
    ]

    for name in estimators:
        calc = get_advantage_calculator(name, config)
        print(f"  {name} -> {calc.__class__.__name__}")

    try:
        get_advantage_calculator("nonexistent", config)
        assert False, "Should raise ValueError"
    except ValueError:
        pass

    print("  PASS\n")
    return True


# ============================================================================
# Test: Pure OPD calculator
# ============================================================================

def test_pure_opd():
    """Pure distillation: rewards zeroed, advantages from KL only."""
    print("Test: Pure OPD calculator")

    from lightrft.trainer.advantage_calculator import OnPolicyDistillationCalculator

    config = MockConfig(opd_kl_coef=1.0)
    calc = OnPolicyDistillationCalculator(config)

    # preprocess_rewards should zero out rewards
    rewards = torch.tensor([0.5, 0.8, 0.3, 0.9, 0.1, 0.7, 0.2, 0.4])
    experiences = [MockExperience(batch_size=4), MockExperience(batch_size=4)]
    exps, reward_chunks = calc.preprocess_rewards(rewards, experiences, max_new_tokens=100)
    for chunk in reward_chunks:
        assert (chunk == 0).all(), f"Pure OPD should zero rewards, got {chunk}"
    print("  preprocess_rewards zeros out rewards: OK")

    # compute should produce whitened advantages from KL
    exp = MockExperience(batch_size=4, num_actions=10, teacher_offset=-0.5)
    final_reward = torch.zeros(4, 10)
    adv, ret, info = calc.compute(exp, final_reward, gamma=1.0, generate_kwargs={})

    assert adv.shape == (4, 10), f"Wrong shape: {adv.shape}"
    assert (adv[:, -2:] == 0).all(), "Padding positions should be 0"

    # Whitened: mean should be ~0
    masked = adv[exp.action_mask]
    assert abs(masked.mean()) < 0.1, f"Whitened mean should be ~0, got {masked.mean():.4f}"
    print(f"  advantages whitened (mean={masked.mean():.4f}, std={masked.std():.4f}): OK")

    # opd_reverse_kl metric should be present
    assert "opd_reverse_kl" in info, "Missing opd_reverse_kl metric"
    print(f"  opd_reverse_kl metric present: OK")

    # Missing teacher_log_probs should raise
    exp_bad = MockExperience()
    del exp_bad.info["teacher_log_probs"]
    try:
        calc.compute(exp_bad, final_reward, gamma=1.0, generate_kwargs={})
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    print("  missing teacher_log_probs raises ValueError: OK")

    print("  PASS\n")
    return True


# ============================================================================
# Test: Hybrid OPD calculator
# ============================================================================

def test_hybrid_opd():
    """Hybrid: GRPO base advantages + OPD KL penalty, then whitened."""
    print("Test: Hybrid OPD calculator")

    from lightrft.trainer.advantage_calculator import OnPolicyDistillationHybridCalculator

    config = MockConfig(opd_kl_coef=1.0, n_samples_per_prompt=4)
    calc = OnPolicyDistillationHybridCalculator(config)

    # preprocess_rewards should apply GRPO normalization (not zero)
    rewards = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    experiences = [MockExperience(batch_size=4), MockExperience(batch_size=4)]
    exps, reward_chunks = calc.preprocess_rewards(rewards, experiences, max_new_tokens=100)
    combined = torch.cat(reward_chunks)
    assert not (combined == 0).all(), "Hybrid should NOT zero rewards"
    print(f"  preprocess_rewards applies GRPO normalization: OK")

    # compute should combine GRPO + OPD and whiten
    exp = MockExperience(batch_size=4, num_actions=10, teacher_offset=-0.3)
    # Simulate GRPO-normalized reward broadcast to tokens
    final_reward = torch.randn(4, 10) * 0.5
    adv, ret, info = calc.compute(exp, final_reward, gamma=1.0, generate_kwargs={})

    assert adv.shape == (4, 10), f"Wrong shape: {adv.shape}"
    masked = adv[exp.action_mask]
    assert abs(masked.mean()) < 0.1, f"Whitened mean should be ~0, got {masked.mean():.4f}"
    print(f"  advantages whitened (mean={masked.mean():.4f}, std={masked.std():.4f}): OK")
    assert "opd_reverse_kl" in info
    print("  PASS\n")
    return True


# ============================================================================
# Test: Advantage whitening
# ============================================================================

def test_whiten_advantages():
    """Whitening normalizes to ~zero mean, ~unit std."""
    print("Test: Advantage whitening")

    from lightrft.trainer.advantage_calculator import _whiten_advantages

    # Large-scale advantages (simulating raw OPD KL)
    adv = torch.randn(4, 20) * 10 + 5  # mean=5, std=10
    mask = torch.ones(4, 20, dtype=torch.bool)
    mask[:, -3:] = False

    whitened = _whiten_advantages(adv, mask)

    masked_vals = whitened[mask]
    assert abs(masked_vals.mean()) < 0.01, f"Mean should be ~0, got {masked_vals.mean():.4f}"
    assert abs(masked_vals.std() - 1.0) < 0.1, f"Std should be ~1, got {masked_vals.std():.4f}"
    print(f"  mean={masked_vals.mean():.4f}, std={masked_vals.std():.4f}: OK")

    # Edge case: very small batch
    adv_small = torch.tensor([[1.0, 2.0]])
    mask_small = torch.ones(1, 2, dtype=torch.bool)
    whitened_small = _whiten_advantages(adv_small, mask_small)
    assert whitened_small.shape == (1, 2)
    print("  small batch handled: OK")

    print("  PASS\n")
    return True


# ============================================================================
# Test: OPD KL penalty helper
# ============================================================================

def test_opd_kl_penalty():
    """_apply_opd_kl_penalty computes correct penalty direction."""
    print("Test: OPD KL penalty")

    from lightrft.trainer.advantage_calculator import _apply_opd_kl_penalty

    # Teacher is better (higher log probs) → student should get positive advantage
    student_lp = torch.tensor([[-2.0, -3.0, -1.5]])
    teacher_lp = torch.tensor([[-1.0, -1.5, -0.5]])
    mask = torch.ones(1, 3, dtype=torch.bool)

    opd_adv, info = _apply_opd_kl_penalty(student_lp, teacher_lp, mask, opd_kl_coef=1.0)

    # reverse_kl = student - teacher = negative (student worse)
    # opd_adv = -1.0 * reverse_kl = positive (encourage matching teacher)
    assert (opd_adv > 0).all(), f"Should be positive when teacher > student, got {opd_adv}"
    print(f"  teacher > student → positive advantage: OK ({opd_adv.tolist()})")

    # Student is overconfident → negative advantage
    student_lp2 = torch.tensor([[-0.5, -0.3, -0.2]])
    teacher_lp2 = torch.tensor([[-2.0, -2.5, -3.0]])
    opd_adv2, _ = _apply_opd_kl_penalty(student_lp2, teacher_lp2, mask, opd_kl_coef=1.0)
    assert (opd_adv2 < 0).all(), f"Should be negative when student > teacher, got {opd_adv2}"
    print(f"  student > teacher → negative advantage: OK ({opd_adv2.tolist()})")

    # opd_kl_coef scales the penalty
    opd_adv_scaled, _ = _apply_opd_kl_penalty(student_lp, teacher_lp, mask, opd_kl_coef=0.5)
    assert torch.allclose(opd_adv_scaled, opd_adv * 0.5)
    print("  opd_kl_coef scaling: OK")

    # Mask is respected
    mask_partial = torch.tensor([[True, True, False]])
    opd_adv_masked, _ = _apply_opd_kl_penalty(student_lp, teacher_lp, mask_partial, opd_kl_coef=1.0)
    assert opd_adv_masked[0, 2] == 0, "Masked position should be 0"
    print("  action mask respected: OK")

    assert "opd_reverse_kl" in info
    print("  PASS\n")
    return True


# ============================================================================
# Test: Teacher logprob extraction (get_teacher_logprobs_by_ids mock)
# ============================================================================

def test_teacher_logprob_extraction():
    """extract_teacher_logprobs handles SGLang format correctly."""
    print("Test: Teacher logprob extraction")

    from examples.on_policy_distillation.on_policy_distillation_reward import (
        extract_teacher_logprobs
    )

    # SGLang format: [logprob, rank, token_str] tuples
    response = {
        "meta_info": {
            "input_token_logprobs": [
                None,  # BOS token
                [-0.1, 1, "hello"],
                [-0.2, 2, "world"],
                [-0.15, 1, "!"],
                [-0.3, 3, "."],
                [-0.25, 2, "end"],
            ]
        }
    }

    # Extract last 3 tokens as response
    lp_list = extract_teacher_logprobs([response], response_lengths=[3], device="cpu")
    assert len(lp_list) == 1
    assert len(lp_list[0]) == 3
    expected = torch.tensor([-0.3, -0.25, -0.15])  # Wait, let me check...

    # logprob_values = [-0.1, -0.2, -0.15, -0.3, -0.25] (skip None, take [0] from each)
    # teacher_log_probs[-3:] = [-0.15, -0.3, -0.25]
    # Hmm, actually the last 3 of [-0.1, -0.2, -0.15, -0.3, -0.25] = [-0.15, -0.3, -0.25]
    expected = torch.tensor([-0.15, -0.3, -0.25])
    assert torch.allclose(lp_list[0], expected), f"Got {lp_list[0]}, expected {expected}"
    print(f"  SGLang format extraction: OK ({lp_list[0].tolist()})")

    # Test padding when response_length > available logprobs
    lp_list2 = extract_teacher_logprobs([response], response_lengths=[10], device="cpu")
    assert len(lp_list2[0]) == 10, f"Should pad to 10, got {len(lp_list2[0])}"
    print(f"  padding for short sequences: OK (len={len(lp_list2[0])})")

    print("  PASS\n")
    return True


# ============================================================================
# Test: Dimension alignment (teacher_log_probs vs action_log_probs)
# ============================================================================

def test_dimension_alignment():
    """teacher_log_probs must match action_log_probs shape [batch, num_actions]."""
    print("Test: Dimension alignment")

    from lightrft.trainer.advantage_calculator import OnPolicyDistillationCalculator

    config = MockConfig(opd_kl_coef=1.0)
    calc = OnPolicyDistillationCalculator(config)

    # Simulate different response lengths within a batch
    # action_log_probs: [batch=2, num_actions=8] (padded to max response length)
    # action_mask: first sample has 6 real tokens, second has 8
    batch_size, num_actions = 2, 8
    exp = MockExperience.__new__(MockExperience)
    exp.action_log_probs = torch.randn(batch_size, num_actions)
    exp.action_mask = torch.ones(batch_size, num_actions, dtype=torch.bool)
    exp.action_mask[0, :2] = False  # first 2 positions are prompt padding for sample 0
    exp.info = {
        "teacher_log_probs": torch.randn(batch_size, num_actions),  # same shape
    }

    final_reward = torch.zeros(batch_size, num_actions)
    adv, _, _ = calc.compute(exp, final_reward, gamma=1.0, generate_kwargs={})
    assert adv.shape == (batch_size, num_actions)
    assert adv[0, 0] == 0 and adv[0, 1] == 0, "Masked positions should be 0"
    print(f"  shape match [batch={batch_size}, num_actions={num_actions}]: OK")

    # Mismatched shapes should fail
    exp_bad = MockExperience.__new__(MockExperience)
    exp_bad.action_log_probs = torch.randn(2, 8)
    exp_bad.action_mask = torch.ones(2, 8, dtype=torch.bool)
    exp_bad.info = {"teacher_log_probs": torch.randn(2, 12)}  # wrong dim!

    try:
        calc.compute(exp_bad, final_reward, gamma=1.0, generate_kwargs={})
        assert False, "Should fail on shape mismatch"
    except RuntimeError:
        pass
    print("  shape mismatch correctly raises RuntimeError: OK")

    print("  PASS\n")
    return True


# ============================================================================
# Test: Pure vs Hybrid produce different results
# ============================================================================

def test_pure_vs_hybrid():
    """Pure and hybrid modes produce meaningfully different advantages."""
    print("Test: Pure vs Hybrid comparison")

    from lightrft.trainer.advantage_calculator import (
        OnPolicyDistillationCalculator,
        OnPolicyDistillationHybridCalculator,
    )

    config = MockConfig(opd_kl_coef=1.0, n_samples_per_prompt=4)
    pure_calc = OnPolicyDistillationCalculator(config)
    hybrid_calc = OnPolicyDistillationHybridCalculator(config)

    torch.manual_seed(42)
    exp = MockExperience(batch_size=4, num_actions=10, teacher_offset=-0.5)

    # Pure: rewards are zeroed
    rewards = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.5, 0.5, 0.8, 0.2])
    _, pure_rewards = pure_calc.preprocess_rewards(rewards.clone(), [exp, exp], 100)
    _, hybrid_rewards = hybrid_calc.preprocess_rewards(rewards.clone(), [exp, exp], 100)

    pure_r = torch.cat(pure_rewards)
    hybrid_r = torch.cat(hybrid_rewards)
    assert (pure_r == 0).all(), "Pure should zero rewards"
    assert not (hybrid_r == 0).all(), "Hybrid should keep rewards"
    print(f"  reward preprocessing differs: OK (pure=0, hybrid has signal)")

    # Both should produce valid advantages
    final_reward_pure = torch.zeros(4, 10)
    final_reward_hybrid = torch.randn(4, 10) * 0.5

    adv_pure, _, _ = pure_calc.compute(exp, final_reward_pure, 1.0, {})
    adv_hybrid, _, _ = hybrid_calc.compute(exp, final_reward_hybrid, 1.0, {})

    assert adv_pure.shape == adv_hybrid.shape
    # They should differ (different reward signals)
    assert not torch.allclose(adv_pure, adv_hybrid, atol=0.01)
    print(f"  advantages differ between modes: OK")

    print("  PASS\n")
    return True


# ============================================================================
# Test: reward_func returns zeros
# ============================================================================

def test_reward_func():
    """Placeholder reward_func returns zeros."""
    print("Test: reward_func placeholder")

    from examples.on_policy_distillation.on_policy_distillation_reward import reward_func

    result = reward_func(
        queries=["q1", "q2", "q3"],
        prompts=["p1", "p2", "p3"],
    )
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3,)
    assert (result == 0).all()
    print("  returns zeros: OK")

    print("  PASS\n")
    return True


# ============================================================================
# Runner
# ============================================================================

def run_all_tests():
    print("=" * 60)
    print("On-Policy Distillation Test Suite (v3)")
    print("=" * 60)
    print()

    tests = [
        ("Factory registration", test_factory),
        ("OPD KL penalty", test_opd_kl_penalty),
        ("Advantage whitening", test_whiten_advantages),
        ("Pure OPD calculator", test_pure_opd),
        ("Hybrid OPD calculator", test_hybrid_opd),
        ("Dimension alignment", test_dimension_alignment),
        ("Pure vs Hybrid", test_pure_vs_hybrid),
        ("Teacher logprob extraction", test_teacher_logprob_extraction),
        ("Reward func placeholder", test_reward_func),
    ]

    results = []
    for name, fn in tests:
        try:
            ok = fn()
            results.append((name, ok))
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
            print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print(f"\n{passed}/{len(results)} passed")
    print("=" * 60)

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
