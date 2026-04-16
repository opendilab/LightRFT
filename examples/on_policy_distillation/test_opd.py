"""
Pytest suite for On-Policy Distillation implementation in LightRFT.

Tests both pure and hybrid OPD modes, KL penalty computation,
teacher logprob extraction, dimension alignment, and reward engine validation.
"""

import pytest
import torch

from lightrft.trainer.advantage_calculator import (
    OnPolicyDistillationCalculator,
    OnPolicyDistillationHybridCalculator,
    _apply_opd_kl_penalty,
    get_advantage_calculator,
    normalize_advantages_cross_batch,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    """Factory fixture for mock config objects."""
    def _make(**kwargs):
        defaults = dict(
            advantages_norm=False,
            advantage_clip=0.0,
            opd_kl_coef=1.0,
            n_samples_per_prompt=4,
            dynamic_sampling=False,
            micro_train_batch_size=4,
        )
        defaults.update(kwargs)

        class _Cfg:
            pass

        cfg = _Cfg()
        for k, v in defaults.items():
            setattr(cfg, k, v)
        return cfg

    return _make


@pytest.fixture
def mock_experience():
    """Factory fixture for mock experience objects (num_tokens naming)."""
    def _make(batch_size=4, num_tokens=10, teacher_offset=-0.5):
        class _Exp:
            pass

        exp = _Exp()
        exp.action_log_probs = torch.randn(batch_size, num_tokens) * 0.5 - 1.0
        exp.action_mask = torch.ones(batch_size, num_tokens, dtype=torch.bool)
        exp.action_mask[:, -2:] = False  # last 2 tokens are padding
        exp.info = {
            "teacher_log_probs": exp.action_log_probs + teacher_offset,
            "reward": torch.rand(batch_size),
            "response_length": torch.full((batch_size,), num_tokens),
        }
        return exp

    return _make


# ---------------------------------------------------------------------------
# Test: Factory registration
# ---------------------------------------------------------------------------

class TestFactory:
    def test_all_estimators_registered(self, mock_config):
        """All estimators including both OPD modes are registered."""
        config = mock_config()
        estimators = [
            "gae", "reinforce", "rloo", "reinforce_baseline",
            "group_norm", "grpo", "cpgd",
            "on_policy_distillation",
            "on_policy_distillation_hybrid",
        ]
        for name in estimators:
            calc = get_advantage_calculator(name, config)
            assert calc is not None

    def test_unknown_estimator_raises(self, mock_config):
        """Unknown estimator name raises ValueError."""
        with pytest.raises(ValueError):
            get_advantage_calculator("nonexistent", mock_config())


# ---------------------------------------------------------------------------
# Test: Pure OPD calculator
# ---------------------------------------------------------------------------

class TestPureOPD:
    def test_preprocess_rewards_passthrough(self, mock_config, mock_experience):
        """Pure OPD preprocess_rewards passes through rewards (zeroing done upstream)."""
        calc = OnPolicyDistillationCalculator(mock_config(opd_kl_coef=1.0))
        rewards = torch.tensor([0.5, 0.8, 0.3, 0.9, 0.1, 0.7, 0.2, 0.4])
        experiences = [mock_experience(batch_size=4), mock_experience(batch_size=4)]
        _, reward_chunks = calc.preprocess_rewards(rewards, experiences, max_new_tokens=100)
        # Rewards are passed through; upstream --no_task_reward zeroes them
        combined = torch.cat(reward_chunks)
        assert combined.shape == rewards.shape

    def test_compute_advantages_shape_and_masking(self, mock_config, mock_experience):
        """Advantages have correct shape and padding positions are zero."""
        calc = OnPolicyDistillationCalculator(mock_config(opd_kl_coef=1.0))
        exp = mock_experience(batch_size=4, num_tokens=10, teacher_offset=-0.5)
        final_reward = torch.zeros(4, 10)
        adv, _ret, info = calc.compute(exp, final_reward, gamma=1.0, generate_kwargs={})

        assert adv.shape == (4, 10)
        assert (adv[:, -2:] == 0).all(), "Padding positions should be 0"
        assert "opd_reverse_kl" in info

    def test_missing_teacher_logprobs_raises(self, mock_config, mock_experience):
        """Missing teacher_log_probs in experience.info raises ValueError."""
        calc = OnPolicyDistillationCalculator(mock_config(opd_kl_coef=1.0))
        exp = mock_experience()
        del exp.info["teacher_log_probs"]
        final_reward = torch.zeros(4, 10)
        with pytest.raises(ValueError):
            calc.compute(exp, final_reward, gamma=1.0, generate_kwargs={})


# ---------------------------------------------------------------------------
# Test: Hybrid OPD calculator
# ---------------------------------------------------------------------------

class TestHybridOPD:
    def test_preprocess_rewards_grpo_normalization(self, mock_config, mock_experience):
        """Hybrid mode applies GRPO normalization (rewards not zeroed)."""
        calc = OnPolicyDistillationHybridCalculator(
            mock_config(opd_kl_coef=1.0, n_samples_per_prompt=4)
        )
        rewards = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
        experiences = [mock_experience(batch_size=4), mock_experience(batch_size=4)]
        _, reward_chunks = calc.preprocess_rewards(rewards, experiences, max_new_tokens=100)
        combined = torch.cat(reward_chunks)
        assert not (combined == 0).all(), "Hybrid should NOT zero rewards"

    def test_compute_advantages(self, mock_config, mock_experience):
        """Hybrid compute produces correct shape and includes KL metric."""
        calc = OnPolicyDistillationHybridCalculator(
            mock_config(opd_kl_coef=1.0, n_samples_per_prompt=4)
        )
        exp = mock_experience(batch_size=4, num_tokens=10, teacher_offset=-0.3)
        final_reward = torch.randn(4, 10) * 0.5
        adv, _ret, info = calc.compute(exp, final_reward, gamma=1.0, generate_kwargs={})

        assert adv.shape == (4, 10)
        assert "opd_reverse_kl" in info


# ---------------------------------------------------------------------------
# Test: OPD KL penalty helper
# ---------------------------------------------------------------------------

class TestOPDKLPenalty:
    def test_teacher_better_positive_advantage(self):
        """When teacher > student (higher logprobs), advantage should be positive."""
        student_lp = torch.tensor([[-2.0, -3.0, -1.5]])
        teacher_lp = torch.tensor([[-1.0, -1.5, -0.5]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        opd_adv, info = _apply_opd_kl_penalty(student_lp, teacher_lp, mask, opd_kl_coef=1.0)
        assert (opd_adv > 0).all()
        assert "opd_reverse_kl" in info

    def test_student_overconfident_negative_advantage(self):
        """When student > teacher, advantage should be negative."""
        student_lp = torch.tensor([[-0.5, -0.3, -0.2]])
        teacher_lp = torch.tensor([[-2.0, -2.5, -3.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        opd_adv, _ = _apply_opd_kl_penalty(student_lp, teacher_lp, mask, opd_kl_coef=1.0)
        assert (opd_adv < 0).all()

    def test_coef_scaling(self):
        """opd_kl_coef scales the penalty linearly."""
        student_lp = torch.tensor([[-2.0, -3.0, -1.5]])
        teacher_lp = torch.tensor([[-1.0, -1.5, -0.5]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        adv_1, _ = _apply_opd_kl_penalty(student_lp, teacher_lp, mask, opd_kl_coef=1.0)
        adv_half, _ = _apply_opd_kl_penalty(student_lp, teacher_lp, mask, opd_kl_coef=0.5)
        assert torch.allclose(adv_half, adv_1 * 0.5)

    def test_mask_respected(self):
        """Masked positions should have zero advantage."""
        student_lp = torch.tensor([[-2.0, -3.0, -1.5]])
        teacher_lp = torch.tensor([[-1.0, -1.5, -0.5]])
        mask = torch.tensor([[True, True, False]])

        opd_adv, _ = _apply_opd_kl_penalty(student_lp, teacher_lp, mask, opd_kl_coef=1.0)
        assert opd_adv[0, 2] == 0, "Masked position should be 0"


# ---------------------------------------------------------------------------
# Test: Advantage normalization (cross-batch)
# ---------------------------------------------------------------------------

class TestNormalizeAdvantages:
    def test_normalize_advantages_cross_batch_shape(self, mock_experience):
        """normalize_advantages_cross_batch preserves experience structure."""
        exp1 = mock_experience(batch_size=4, num_tokens=10)
        exp2 = mock_experience(batch_size=4, num_tokens=10)
        # Add advantages attribute (normally set by compute)
        exp1.advantages = torch.randn(4, 10) * 5 + 2
        exp2.advantages = torch.randn(4, 10) * 5 + 2

        class _Args:
            pass

        args = _Args()
        # on_policy_distillation_hybrid triggers normalization
        result = normalize_advantages_cross_batch(
            [exp1, exp2], "on_policy_distillation_hybrid", args
        )
        assert len(result) == 2
        assert result[0].advantages.shape == (4, 10)

    def test_pure_opd_skips_normalization(self, mock_experience):
        """Pure OPD mode skips cross-batch normalization."""
        exp = mock_experience(batch_size=4, num_tokens=10)
        exp.advantages = torch.randn(4, 10) * 5 + 2
        original = exp.advantages.clone()

        class _Args:
            pass

        result = normalize_advantages_cross_batch(
            [exp], "on_policy_distillation", _Args()
        )
        # Should return unchanged (not in whitening list)
        assert torch.equal(result[0].advantages, original)


# ---------------------------------------------------------------------------
# Test: Teacher logprob extraction
# ---------------------------------------------------------------------------

class TestTeacherLogprobExtraction:
    def test_sglang_format(self):
        """extract_teacher_logprobs handles SGLang format correctly."""
        from examples.on_policy_distillation.on_policy_distillation_reward import (
            extract_teacher_logprobs,
        )

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

        lp_list = extract_teacher_logprobs([response], response_lengths=[3], device="cpu")
        assert len(lp_list) == 1
        assert len(lp_list[0]) == 3
        # logprob_values = [-0.1, -0.2, -0.15, -0.3, -0.25]; last 3 = [-0.15, -0.3, -0.25]
        expected = torch.tensor([-0.15, -0.3, -0.25])
        assert torch.allclose(lp_list[0], expected)

    def test_padding_for_long_response(self):
        """Pads to response_length when requested length > available logprobs."""
        from examples.on_policy_distillation.on_policy_distillation_reward import (
            extract_teacher_logprobs,
        )

        response = {
            "meta_info": {
                "input_token_logprobs": [
                    None,
                    [-0.1, 1, "a"],
                    [-0.2, 2, "b"],
                ]
            }
        }

        lp_list = extract_teacher_logprobs([response], response_lengths=[10], device="cpu")
        assert len(lp_list[0]) == 10


# ---------------------------------------------------------------------------
# Test: Dimension alignment
# ---------------------------------------------------------------------------

class TestDimensionAlignment:
    def test_matching_shapes(self, mock_config):
        """teacher_log_probs matching action_log_probs shape [batch, num_tokens] works."""
        calc = OnPolicyDistillationCalculator(mock_config(opd_kl_coef=1.0))
        batch_size, num_tokens = 2, 8

        class _Exp:
            pass

        exp = _Exp()
        exp.action_log_probs = torch.randn(batch_size, num_tokens)
        exp.action_mask = torch.ones(batch_size, num_tokens, dtype=torch.bool)
        exp.action_mask[0, :2] = False  # prompt padding for sample 0
        exp.info = {"teacher_log_probs": torch.randn(batch_size, num_tokens)}

        final_reward = torch.zeros(batch_size, num_tokens)
        adv, _, _ = calc.compute(exp, final_reward, gamma=1.0, generate_kwargs={})
        assert adv.shape == (batch_size, num_tokens)
        assert adv[0, 0] == 0 and adv[0, 1] == 0, "Masked positions should be 0"

    def test_mismatched_shapes_raises(self, mock_config):
        """Mismatched teacher_log_probs shape raises RuntimeError."""
        calc = OnPolicyDistillationCalculator(mock_config(opd_kl_coef=1.0))

        class _Exp:
            pass

        exp = _Exp()
        exp.action_log_probs = torch.randn(2, 8)
        exp.action_mask = torch.ones(2, 8, dtype=torch.bool)
        exp.info = {"teacher_log_probs": torch.randn(2, 12)}  # wrong dim

        final_reward = torch.zeros(2, 8)
        with pytest.raises(RuntimeError):
            calc.compute(exp, final_reward, gamma=1.0, generate_kwargs={})


# ---------------------------------------------------------------------------
# Test: Pure vs Hybrid produce different results
# ---------------------------------------------------------------------------

class TestPureVsHybrid:
    def test_advantages_differ(self, mock_config, mock_experience):
        """Pure and hybrid modes produce meaningfully different advantages."""
        config = mock_config(opd_kl_coef=1.0, n_samples_per_prompt=4)
        pure_calc = OnPolicyDistillationCalculator(config)
        hybrid_calc = OnPolicyDistillationHybridCalculator(config)

        torch.manual_seed(42)
        exp = mock_experience(batch_size=4, num_tokens=10, teacher_offset=-0.5)

        rewards = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.5, 0.5, 0.8, 0.2])
        _, pure_rewards = pure_calc.preprocess_rewards(rewards.clone(), [exp, exp], 100)
        _, hybrid_rewards = hybrid_calc.preprocess_rewards(rewards.clone(), [exp, exp], 100)

        hybrid_r = torch.cat(hybrid_rewards)
        assert not (hybrid_r == 0).all(), "Hybrid should keep rewards"

        final_reward_pure = torch.zeros(4, 10)
        final_reward_hybrid = torch.randn(4, 10) * 0.5

        adv_pure, _, _ = pure_calc.compute(exp, final_reward_pure, 1.0, {})
        adv_hybrid, _, _ = hybrid_calc.compute(exp, final_reward_hybrid, 1.0, {})

        assert adv_pure.shape == adv_hybrid.shape
        assert not torch.allclose(adv_pure, adv_hybrid, atol=0.01)


# ---------------------------------------------------------------------------
# Test: Reward func placeholder
# ---------------------------------------------------------------------------

class TestRewardFunc:
    def test_reward_func_returns_zeros(self):
        """Placeholder reward_func returns zeros."""
        from examples.on_policy_distillation.on_policy_distillation_reward import reward_func

        result = reward_func(queries=["q1", "q2", "q3"], prompts=["p1", "p2", "p3"])
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3,)
        assert (result == 0).all()


# ---------------------------------------------------------------------------
# Test: RewardComputationEngine TypeError
# ---------------------------------------------------------------------------

class TestRewardEngineTypeError:
    def test_invalid_remote_rm_url_type_raises(self):
        """Passing an invalid type for remote_rm_url raises TypeError."""
        from lightrft.trainer.fast_exp_maker import RewardComputationEngine

        with pytest.raises(TypeError, match="remote_rm_url must be str, list, tuple, or None"):
            RewardComputationEngine(
                reward_model=None,
                remote_rm_url=12345,  # int is invalid
                custom_reward_func=None,
                reward_fn=None,
                reward_fn_label_map=None,
                reward_recipe=None,
                tokenizer=None,
                strategy=None,
                packing_samples=False,
            )
