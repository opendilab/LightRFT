"""
Pytest suite for On-Policy Distillation implementation in LightRFT.

Tests the unified OPD calculator, KL penalty computation,
teacher logprob extraction, dimension alignment, and reward engine validation.
"""

import pytest
import torch

from lightrft.trainer.advantage_calculator import (
    OnPolicyDistillationCalculator,
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
        """All estimators are registered."""
        config = mock_config()
        estimators = [
            "gae", "reinforce", "rloo", "reinforce_baseline",
            "group_norm", "grpo", "cpgd",
            "on_policy_distillation",
        ]
        for name in estimators:
            calc = get_advantage_calculator(name, config)
            assert calc is not None

    def test_unknown_estimator_raises(self, mock_config):
        """Unknown estimator name raises ValueError."""
        with pytest.raises(ValueError):
            get_advantage_calculator("nonexistent", mock_config())

    def test_hybrid_removed(self, mock_config):
        """on_policy_distillation_hybrid is no longer registered."""
        with pytest.raises(ValueError):
            get_advantage_calculator("on_policy_distillation_hybrid", mock_config())


# ---------------------------------------------------------------------------
# Test: Unified OPD calculator
# ---------------------------------------------------------------------------

class TestOPDCalculator:
    def test_preprocess_rewards_grpo_normalization(self, mock_config, mock_experience):
        """OPD applies GRPO normalization to rewards."""
        calc = OnPolicyDistillationCalculator(
            mock_config(opd_kl_coef=1.0, n_samples_per_prompt=4)
        )
        rewards = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
        experiences = [mock_experience(batch_size=4), mock_experience(batch_size=4)]
        _, reward_chunks = calc.preprocess_rewards(rewards, experiences, max_new_tokens=100)
        combined = torch.cat(reward_chunks)
        # Non-uniform rewards should produce non-zero normalized values
        assert not (combined == 0).all(), "Should apply GRPO normalization"

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
        # on_policy_distillation triggers normalization
        result = normalize_advantages_cross_batch(
            [exp1, exp2], "on_policy_distillation", args
        )
        assert len(result) == 2
        assert result[0].advantages.shape == (4, 10)

    def test_group_norm_skips_normalization(self, mock_experience):
        """group_norm mode skips cross-batch normalization."""
        exp = mock_experience(batch_size=4, num_tokens=10)
        exp.advantages = torch.randn(4, 10) * 5 + 2
        original = exp.advantages.clone()

        class _Args:
            pass

        result = normalize_advantages_cross_batch(
            [exp], "group_norm", _Args()
        )
        # Should return unchanged (not in whitening list)
        assert torch.equal(result[0].advantages, original)


# ---------------------------------------------------------------------------
# Test: Zero vs non-zero rewards (replaces TestPureVsHybrid)
# ---------------------------------------------------------------------------

class TestZeroVsNonZeroRewards:
    def test_advantages_differ_with_rewards(self, mock_config, mock_experience):
        """OPD with zero rewards vs non-zero rewards produces different advantages."""
        config = mock_config(opd_kl_coef=1.0, n_samples_per_prompt=4)
        calc = OnPolicyDistillationCalculator(config)

        torch.manual_seed(42)
        exp = mock_experience(batch_size=4, num_tokens=10, teacher_offset=-0.5)

        # Zero rewards (pure distillation mode)
        final_reward_zero = torch.zeros(4, 10)
        adv_zero, _, _ = calc.compute(exp, final_reward_zero, 1.0, {})

        # Non-zero rewards (hybrid mode with task rewards)
        final_reward_nonzero = torch.randn(4, 10) * 0.5
        adv_nonzero, _, _ = calc.compute(exp, final_reward_nonzero, 1.0, {})

        assert adv_zero.shape == adv_nonzero.shape
        # With non-zero task rewards, advantages should differ
        assert not torch.allclose(adv_zero, adv_nonzero, atol=0.01)


# ---------------------------------------------------------------------------
# Test: Teacher logprob extraction
# ---------------------------------------------------------------------------

class TestTeacherLogprobExtraction:
    def test_sglang_format(self):
        """extract_teacher_logprobs handles SGLang format correctly."""
        from lightrft.trainer.opd_utils import extract_teacher_logprobs

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
        from lightrft.trainer.opd_utils import extract_teacher_logprobs

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
# Test: Reward func placeholder
# ---------------------------------------------------------------------------

class TestRewardFunc:
    def test_reward_func_returns_zeros(self):
        """Placeholder reward_func returns zeros."""
        from lightrft.trainer.opd_utils import reward_func

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
