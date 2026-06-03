"""Strict-alignment tests for the URSA paper Eq.9 advantage estimator.

Tests cover the four acceptance criteria AC1–AC4 from the PR plan:

  AC1  numerical equivalence with hand-computed paper Eq.9 (max|Δ|<1e-5)
  AC2  outcome reward is NOT bypassed (changing r_o changes advantages)
  AC3  group normalization is correct over K=n_samples_per_prompt
  AC4  per-step advantage broadcast to the *full* step span (not just the
       boundary token); advantage jumps at step boundaries

Plus a regression test for the legacy ``per_step_reward_mode=raw`` failure
mode (advantages all-positive) which the new path must NOT exhibit.

Run from repo root:
    PYTHONPATH=examples/math_prm python3 -m pytest examples/math_prm/tests/ -v
Or directly:
    python3 examples/math_prm/tests/test_ursa_variant2.py
"""

from __future__ import annotations

import math
import os
import sys
import unittest
from types import SimpleNamespace
from typing import List

# Allow `import ursa_variant2` whether run from repo root (CI) or from
# examples/math_prm/ (developer convenience).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import torch

import ursa_variant2 as ursa_v2  # registers the monkey-patch at import time


def _make_cfg(n_samples: int = 2, advantage_clip: float = 0) -> SimpleNamespace:
    """Minimal config namespace UrsaVariant2Calculator reads from."""
    return SimpleNamespace(
        n_samples_per_prompt=n_samples,
        advantage_clip=advantage_clip,
    )


def _make_exp(
    action_mask: torch.Tensor,
    reward: torch.Tensor,
    step_rewards: List[torch.Tensor],
    step_token_indices: List[torch.Tensor],
):
    """Minimal experience stub: just info dict + action_mask."""
    return SimpleNamespace(
        action_mask=action_mask,
        info={
            "reward": reward,
            "step_rewards": step_rewards,
            "step_token_indices": step_token_indices,
        },
    )


# Hand-computed Eq.9 reference, written out explicitly so reviewers can
# verify the test logic itself matches paper Appendix B.1 Eq.9.
def _hand_compute_eq9(
    step_rewards: List[torch.Tensor],
    step_token_indices: List[torch.Tensor],
    outcome: torch.Tensor,
    K: int,
    T: int,
) -> torch.Tensor:
    """Brute-force reference: build per-token advantages following paper Eq.9.

    A_t^i = r_{s,t}^i * GroupNorm_G(r̄_s^i) + GroupNorm_G(r_o^i)
    where t indexes steps and the value is broadcast to every token within
    the span [start_k, end_k] (start_0=0, start_k = end_{k-1}+1).
    """
    B = outcome.numel()
    assert B % K == 0
    G = B // K

    r_bar = torch.stack(
        [sr.float().mean() if sr.numel() > 0 else torch.tensor(0.0) for sr in step_rewards]
    )

    def gn(x: torch.Tensor) -> torch.Tensor:
        g = x.float().reshape(G, K)
        return ((g - g.mean(dim=-1, keepdim=True))
                / (g.std(dim=-1, unbiased=False, keepdim=True) + 1e-9)).flatten()

    oc_norm = gn(outcome)
    msp_norm = gn(r_bar)

    adv = torch.zeros(B, T, dtype=torch.float32)
    for i in range(B):
        sr = step_rewards[i].float()
        si = step_token_indices[i].long()
        n = sr.numel()
        if n == 0:
            adv[i] = oc_norm[i]
            continue
        starts = torch.cat([torch.zeros(1, dtype=torch.long), si[:-1] + 1])
        ends = si
        for k in range(n):
            sk = max(0, int(starts[k]))
            ek = min(T - 1, int(ends[k]))
            adv[i, sk:ek + 1] = sr[k] * msp_norm[i] + oc_norm[i]
        last_end = int(ends[-1])
        if last_end + 1 < T:
            adv[i, last_end + 1:] = oc_norm[i]
    return adv


class _Base(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)


class TestAC1NumericalEquivalence(_Base):
    """AC1: implementation matches hand-computed Eq.9 within tolerance."""

    def test_basic_k2_three_steps(self):
        K = 2
        B = 4
        T = 30
        step_rewards = [
            torch.tensor([0.80, 0.70, 0.30]),  # traj 0
            torch.tensor([0.85, 0.75, 0.90]),  # traj 1
            torch.tensor([0.50, 0.55, 0.60]),  # traj 2
            torch.tensor([0.60, 0.65, 0.70]),  # traj 3
        ]
        step_token_indices = [torch.tensor([5, 12, 20])] * 4
        outcome = torch.tensor([1.0, 1.0, 0.0, 1.0])
        action_mask = torch.ones(B, T, dtype=torch.long)

        expected = _hand_compute_eq9(step_rewards, step_token_indices, outcome, K, T)

        calc = ursa_v2.UrsaVariant2Calculator(_make_cfg(n_samples=K))
        # preprocess_rewards takes one experience holding all B trajectories
        exp = _make_exp(action_mask, outcome, step_rewards, step_token_indices)
        calc.preprocess_rewards(outcome, [exp], max_new_tokens=T)
        adv, ret, info = calc.compute(exp, final_reward=None, gamma=None, generate_kwargs={})

        max_abs = (adv - expected).abs().max().item()
        self.assertLess(max_abs, 1e-5, f"AC1 violated: max|Δ|={max_abs}")
        # returns mirror advantages (no value function)
        self.assertLess((ret - adv).abs().max().item(), 1e-9)

    def test_k4_variable_step_count(self):
        K = 4
        T = 40
        step_rewards = [
            torch.tensor([0.9, 0.8]),                  # 2 steps
            torch.tensor([0.5, 0.4, 0.6]),             # 3 steps
            torch.tensor([0.7]),                       # 1 step
            torch.tensor([0.6, 0.65, 0.55, 0.50]),     # 4 steps
        ]
        step_token_indices = [
            torch.tensor([10, 25]),
            torch.tensor([8, 18, 30]),
            torch.tensor([22]),
            torch.tensor([7, 15, 22, 33]),
        ]
        outcome = torch.tensor([1.0, 0.0, 1.0, 0.0])
        action_mask = torch.ones(K, T, dtype=torch.long)

        expected = _hand_compute_eq9(step_rewards, step_token_indices, outcome, K, T)
        calc = ursa_v2.UrsaVariant2Calculator(_make_cfg(n_samples=K))
        exp = _make_exp(action_mask, outcome, step_rewards, step_token_indices)
        calc.preprocess_rewards(outcome, [exp], max_new_tokens=T)
        adv, _, _ = calc.compute(exp, None, None, {})
        max_abs = (adv - expected).abs().max().item()
        self.assertLess(max_abs, 1e-5, f"AC1 (K=4 variable steps) violated: max|Δ|={max_abs}")


class TestAC2OutcomeNotBypassed(_Base):
    """AC2: changing outcome reward must change advantages.

    Regression for the Mode B bypass bug: under the old per-step path,
    feeding outcome through ``compute_reward`` r had no effect because
    Mode B threw it away. UrsaVariant2Calculator must NOT have this bug.
    """

    def _run(self, outcome):
        K = 2
        B = 4
        T = 20
        step_rewards = [torch.tensor([0.5, 0.6, 0.7])] * B
        step_token_indices = [torch.tensor([5, 10, 15])] * B
        action_mask = torch.ones(B, T, dtype=torch.long)
        calc = ursa_v2.UrsaVariant2Calculator(_make_cfg(n_samples=K))
        exp = _make_exp(action_mask, outcome, step_rewards, step_token_indices)
        calc.preprocess_rewards(outcome, [exp], max_new_tokens=T)
        adv, _, _ = calc.compute(exp, None, None, {})
        return adv

    def test_all_correct_vs_all_wrong_differs(self):
        # If outcome is constant across the group, the GroupNorm of outcome
        # is exactly zero, so the additive term vanishes — that's expected
        # and not a bug; instead we compare a per-prompt mixed case.
        # (One sample correct, one wrong in each of two prompts.)
        oc_a = torch.tensor([1.0, 0.0, 1.0, 0.0])
        oc_b = torch.tensor([0.0, 1.0, 0.0, 1.0])
        adv_a = self._run(oc_a)
        adv_b = self._run(oc_b)
        diff = (adv_a - adv_b).abs().max().item()
        self.assertGreater(diff, 0.5, f"AC2 violated: outcome flip should "
                                      f"flip the sign of the outcome term "
                                      f"(max|Δ|={diff})")

    def test_outcome_anchor_extends_past_last_step(self):
        # Pad tail past the last step should carry the outcome anchor only.
        K = 2
        T = 30
        step_rewards = [
            torch.tensor([0.6, 0.7]),
            torch.tensor([0.6, 0.7]),
        ]
        step_token_indices = [torch.tensor([5, 10])] * 2
        # Trajectory 0 wins outcome, trajectory 1 loses — group_norm gives
        # ±1 for the outcome term.
        outcome = torch.tensor([1.0, 0.0])
        action_mask = torch.ones(K, T, dtype=torch.long)
        calc = ursa_v2.UrsaVariant2Calculator(_make_cfg(n_samples=K))
        exp = _make_exp(action_mask, outcome, step_rewards, step_token_indices)
        calc.preprocess_rewards(outcome, [exp], max_new_tokens=T)
        adv, _, _ = calc.compute(exp, None, None, {})
        # tail tokens (idx 11..29) should equal oc_normed (= ±1 within tol)
        traj0_tail = adv[0, 11:]
        traj1_tail = adv[1, 11:]
        # Expect tail = oc_normed[i] (no process-reward there)
        self.assertGreater(traj0_tail.mean().item(), 0.5,
                           f"traj0 tail must carry positive outcome anchor "
                           f"(got mean={traj0_tail.mean().item():.3f})")
        self.assertLess(traj1_tail.mean().item(), -0.5,
                        f"traj1 tail must carry negative outcome anchor "
                        f"(got mean={traj1_tail.mean().item():.3f})")


class TestAC3GroupNormCorrect(_Base):
    """AC3: GroupNorm zero-mean / unit-std across K siblings for both terms."""

    def test_k2_msp_normed_zero_mean(self):
        K = 2
        B = 4
        T = 10
        step_rewards = [
            torch.tensor([0.9, 0.8, 0.7]),
            torch.tensor([0.3, 0.4, 0.2]),
            torch.tensor([0.8, 0.7, 0.6]),
            torch.tensor([0.5, 0.5, 0.4]),
        ]
        sti = [torch.tensor([2, 5, 8])] * B
        outcome = torch.tensor([1.0, 0.0, 1.0, 0.0])
        action_mask = torch.ones(B, T, dtype=torch.long)
        calc = ursa_v2.UrsaVariant2Calculator(_make_cfg(n_samples=K))
        exp = _make_exp(action_mask, outcome, step_rewards, sti)
        calc.preprocess_rewards(outcome, [exp], max_new_tokens=T)

        # Stored normed values should sum to 0 per group (within tol)
        oc_n = exp.info["_ursa_oc_normed"].view(-1, K)
        msp_n = exp.info["_ursa_msp_normed"].view(-1, K)
        self.assertLess(oc_n.sum(dim=-1).abs().max().item(), 1e-5)
        self.assertLess(msp_n.sum(dim=-1).abs().max().item(), 1e-5)

    def test_k4_msp_normed_unit_std(self):
        K = 4
        T = 10
        step_rewards = [
            torch.tensor([0.9, 0.8]),
            torch.tensor([0.5, 0.4]),
            torch.tensor([0.7, 0.7]),
            torch.tensor([0.2, 0.3]),
        ]
        sti = [torch.tensor([3, 7])] * K
        outcome = torch.tensor([1.0, 0.0, 1.0, 0.0])
        action_mask = torch.ones(K, T, dtype=torch.long)
        calc = ursa_v2.UrsaVariant2Calculator(_make_cfg(n_samples=K))
        exp = _make_exp(action_mask, outcome, step_rewards, sti)
        calc.preprocess_rewards(outcome, [exp], max_new_tokens=T)
        msp = exp.info["_ursa_msp_normed"]
        # std (unbiased=False) across the single group of K=4 should be ~1
        std = msp.std(unbiased=False).item()
        self.assertAlmostEqual(std, 1.0, places=3,
                               msg=f"AC3 violated: msp_normed std={std}")


class TestAC4SpanBroadcast(_Base):
    """AC4: advantage is constant within each step span and changes at the boundary."""

    def test_advantage_constant_within_span(self):
        K = 2
        B = 2
        T = 25
        step_rewards = [
            torch.tensor([0.9, 0.5, 0.7]),
            torch.tensor([0.4, 0.6, 0.8]),
        ]
        sti = [torch.tensor([4, 12, 20])] * B
        outcome = torch.tensor([1.0, 0.0])
        action_mask = torch.ones(B, T, dtype=torch.long)
        calc = ursa_v2.UrsaVariant2Calculator(_make_cfg(n_samples=K))
        exp = _make_exp(action_mask, outcome, step_rewards, sti)
        calc.preprocess_rewards(outcome, [exp], max_new_tokens=T)
        adv, _, _ = calc.compute(exp, None, None, {})

        # span 0 of traj 0: tokens 0..4 should all be equal
        span0 = adv[0, 0:5]
        self.assertLess((span0 - span0[0]).abs().max().item(), 1e-6)
        # span 1: tokens 5..12 equal
        span1 = adv[0, 5:13]
        self.assertLess((span1 - span1[0]).abs().max().item(), 1e-6)
        # span 2: tokens 13..20 equal
        span2 = adv[0, 13:21]
        self.assertLess((span2 - span2[0]).abs().max().item(), 1e-6)
        # adjacent spans differ (otherwise per-step credit is degenerate)
        self.assertNotAlmostEqual(span0[0].item(), span1[0].item(), places=4)
        self.assertNotAlmostEqual(span1[0].item(), span2[0].item(), places=4)


class TestAC5SignedAdvantages(_Base):
    """AC5: typical inputs produce both positive and negative advantages."""

    def test_signed_advantages_on_synthetic_batch(self):
        K = 2
        B = 4
        T = 25
        step_rewards = [
            torch.tensor([0.8, 0.7, 0.3]),
            torch.tensor([0.85, 0.75, 0.9]),
            torch.tensor([0.5, 0.55, 0.6]),
            torch.tensor([0.6, 0.65, 0.7]),
        ]
        sti = [torch.tensor([5, 12, 20])] * B
        outcome = torch.tensor([1.0, 1.0, 0.0, 1.0])
        action_mask = torch.ones(B, T, dtype=torch.long)
        calc = ursa_v2.UrsaVariant2Calculator(_make_cfg(n_samples=K))
        exp = _make_exp(action_mask, outcome, step_rewards, sti)
        calc.preprocess_rewards(outcome, [exp], max_new_tokens=T)
        adv, _, info = calc.compute(exp, None, None, {})
        # advantages must contain both signs (paper Eq.9 → zero-mean per group)
        self.assertGreater(info["ursa_v2_adv_pos_frac"], 0.05,
                           "AC5: advantages should contain positive entries")
        self.assertGreater(info["ursa_v2_adv_neg_frac"], 0.05,
                           "AC5: advantages should contain negative entries")


class TestK1Fallback(_Base):
    """When K=1, group norm is degenerate; calculator must not crash."""

    def test_k1_returns_zero_advantage(self):
        K = 1
        T = 20
        step_rewards = [torch.tensor([0.5, 0.6])]
        sti = [torch.tensor([5, 10])]
        outcome = torch.tensor([1.0])
        action_mask = torch.ones(1, T, dtype=torch.long)
        calc = ursa_v2.UrsaVariant2Calculator(_make_cfg(n_samples=K))
        exp = _make_exp(action_mask, outcome, step_rewards, sti)
        calc.preprocess_rewards(outcome, [exp], max_new_tokens=T)
        adv, _, info = calc.compute(exp, None, None, {})
        self.assertEqual(adv.abs().sum().item(), 0.0)
        self.assertEqual(info.get("ursa_v2_fallback_used"), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
